import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


@triton.jit
def flash_decode_kernel_split(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    kv_len,
    head_dim: tl.constexpr,
    gqa_size: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    kv_num_heads = tl.num_programs(1)
    q_num_heads = kv_num_heads * gqa_size

    q_offset = batch_id * q_num_heads * head_dim
    q_desc = tl.make_tensor_descriptor(
        base=q_ptr + q_offset,
        shape=[q_num_heads, head_dim],
        strides=[head_dim, 1],
        block_shape=[1, head_dim],
    )

    kv_offset = (batch_id * kv_num_heads + kv_head_id) * kv_len * head_dim
    k_desc = tl.make_tensor_descriptor(
        base=k_ptr + kv_offset,
        shape=[kv_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_KV, head_dim],
    )

    v_desc = tl.make_tensor_descriptor(
        base=v_ptr + kv_offset,
        shape=[kv_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_KV, head_dim],
    )

    # iterate over GQA heads
    for q_gqa_head_id in range(0, gqa_size):
        # Load Q tile
        q_global_head_id = kv_head_id * gqa_size + q_gqa_head_id
        q_tile = q_desc.load([q_global_head_id, 0])

        # initialize tmp variables
        max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
        exp_sum = tl.zeros((1,), dtype=tl.float32)
        o_tile = tl.zeros((1, head_dim), dtype=tl.float32)

        # iterate over KV in blocks
        for kv_block_start in range(0, kv_len, BLOCK_KV):
            # Load k and v tiles
            k_tile = k_desc.load([kv_block_start, 0])
            v_tile = v_desc.load([kv_block_start, 0])

            # attn = Q @ K.T * scale
            scale = 1.0 / (head_dim**0.5)
            attn = (tl.dot(q_tile, k_tile.T) * scale)

            # compute softmax and output
            tile_max = tl.max(attn)
            max_score_new = tl.maximum(max_score, tile_max)
            exp_attn = tl.exp(attn - max_score_new)

            exp_max_diff = tl.exp(max_score - max_score_new)
            exp_sum = exp_max_diff * exp_sum + tl.sum(exp_attn)
            o_tile = exp_max_diff * o_tile + tl.dot(exp_attn.to(v_tile.dtype), v_tile)

            max_score = max_score_new

        # finalize output and store
        split_id = tl.program_id(2)
        split_num = tl.num_programs(2)

        # o in [B, H_q, split_num, head_dim]
        o_offset = (batch_id * q_num_heads + q_global_head_id) * split_num * head_dim
        o_desc = tl.make_tensor_descriptor(
            base=o_ptr + o_offset,
            shape=[split_num, head_dim],
            strides=[head_dim, 1],
            block_shape=[1, head_dim],
        )
        o_tile = o_tile / exp_sum
        o_desc.store([split_id, 0], o_tile.to(o_desc.dtype))

        # lse in [B, H_q, split_num]
        lse = tl.log(exp_sum) + max_score
        lse_offset = (batch_id * q_num_heads + q_global_head_id) * split_num + split_id + tl.arange(0, 1)
        tl.store(lse_ptr + lse_offset, lse)


@triton.jit
def flash_decode_kernel_reduce(
    per_split_o_ptr,
    per_split_lse_ptr,
    o_ptr,
    split_num: int,
    head_dim: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    q_num_heads = tl.num_programs(1)

    seq_idx = batch_id * q_num_heads + q_head_id
    per_split_o_base = per_split_o_ptr + seq_idx * split_num * head_dim
    per_split_o_desc = tl.make_tensor_descriptor(
        base=per_split_o_base,
        shape=[split_num, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SPLIT, head_dim],
    )

    per_split_lse_base = per_split_lse_ptr + seq_idx * split_num
    per_split_lse_desc = tl.make_tensor_descriptor(
        base=per_split_lse_base,
        shape=[split_num],
        strides=[1],
        block_shape=[BLOCK_SPLIT],
    )

    max_lse = tl.full((1,), float("-inf"), dtype=tl.float32)
    exp_sum = tl.zeros((1,), dtype=tl.float32)
    o_acc = tl.zeros((1, head_dim), dtype=tl.float32)

    for split_start in range(0, split_num, BLOCK_SPLIT):

        # Load lse tile
        lse = per_split_lse_desc.load([split_start])

        split_offsets = split_start + tl.arange(0, BLOCK_SPLIT)
        split_mask = split_offsets < split_num
        lse = tl.where(split_mask, lse, float("-inf"))

        tile_max = tl.max(lse)
        max_new = tl.maximum(max_lse, tile_max)

        exp_lse = tl.exp(lse - max_new)
        exp_prev = tl.exp(max_lse - max_new)

        # Load o tile
        o_tile = per_split_o_desc.load([split_start, 0])

        exp_sum = exp_prev * exp_sum + tl.sum(exp_lse, axis=0)
        o_acc = exp_prev * o_acc + tl.dot(exp_lse[None, :].to(o_tile.dtype), o_tile)

        max_lse = max_new

    o_acc = o_acc / exp_sum
    o_base = o_ptr + batch_id * q_num_heads * head_dim
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[q_num_heads, head_dim],
        strides=[head_dim, 1],
        block_shape=[1, head_dim],
    )
    o_desc.store([q_head_id, 0], o_acc.to(o_desc.dtype))


def triton_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split_len=512,
    block_kv=32,
):
    batch_size, q_num_heads, _, head_dim = q.shape
    _, kv_num_heads, kv_len, _ = k.shape

    gqa_size = q_num_heads // kv_num_heads
    split_num = triton.cdiv(kv_len, split_len)
    grid = (
        batch_size,
        kv_num_heads,
        split_num,
    )

    per_split_o = torch.empty((batch_size, q_num_heads, split_num, head_dim), device=q.device, dtype=q.dtype)
    per_split_lse = torch.empty((batch_size, q_num_heads, split_num), device=q.device, dtype=q.dtype)
    flash_decode_kernel_split[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=per_split_o,
        lse_ptr=per_split_lse,
        kv_len=kv_len,
        head_dim=head_dim,
        gqa_size=gqa_size,
        BLOCK_KV=block_kv,
    )

    # # reduce refer to https://github.com/Dao-AILab/flash-attention/issues/1248
    # lse_final = torch.logsumexp(per_split_lse, dim=2, keepdim=True)  # [B, H, 1]
    # exp_final = torch.exp(per_split_lse - lse_final).unsqueeze(2)  # [B, H, 1, l_chunk_num]
    # output = torch.matmul(exp_final, per_split_o)  # [B, H, 1, D]

    # equivalent triton reduce
    output = torch.empty_like(q)
    grid = (
        batch_size,
        q_num_heads,
    )
    flash_decode_kernel_reduce[grid](
        per_split_o_ptr=per_split_o,
        per_split_lse_ptr=per_split_lse,
        o_ptr=output,
        split_num=split_num,
        head_dim=head_dim,
        BLOCK_SPLIT=block_kv,
    )

    return output


def flash_decode(
    batch_size: int,
    kv_len: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
):
    torch.manual_seed(0)

    device = get_device()
    dtype = torch.float16
    q_tensor = torch.randn(batch_size, q_num_heads, 1, head_dim, device=device, dtype=dtype)
    k_tensor = torch.randn(batch_size, kv_num_heads, kv_len, head_dim, device=device, dtype=dtype)
    v_tensor = torch.randn(batch_size, kv_num_heads, kv_len, head_dim, device=device, dtype=dtype)

    excepted = torch.nn.functional.scaled_dot_product_attention(
        q_tensor,
        k_tensor,
        v_tensor,
        is_causal=False,
        enable_gqa=True,
    )

    o_tensor = triton_flash_decode(
        q_tensor,
        k_tensor,
        v_tensor,
    )

    acc_check(excepted, o_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    flash_decode(
        batch_size=1,
        kv_len=1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
