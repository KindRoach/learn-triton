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
        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        o_tile = tl.zeros((1, head_dim), dtype=tl.float32)

        # iterate over KV in blocks
        for kv_block_start in range(0, kv_len, BLOCK_KV):
            # Load k and v tiles
            k_tile = k_desc.load([kv_block_start, 0])
            v_tile = v_desc.load([kv_block_start, 0])

            # attn = Q @ K.T * scale
            inv_sqrt_d = 1.0 / (head_dim**0.5)
            scores = tl.dot(q_tile, k_tile.T) * inv_sqrt_d

            # compute softmax and output
            tile_max = tl.max(scores)
            new_m_i = tl.maximum(m_i, tile_max)
            exp_scores = tl.exp(scores - new_m_i)

            alpha = tl.exp(m_i - new_m_i)
            l_i = alpha * l_i + tl.sum(exp_scores)
            o_tile = alpha * o_tile + tl.dot(exp_scores.to(v_tile.dtype), v_tile)

            m_i = new_m_i

        # finalize output and store
        o_tile = o_tile / l_i
        lse = tl.log(l_i) + m_i

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
        o_desc.store([split_id, 0], o_tile.to(o_desc.dtype))

        # lse in [B, H_q, split_num]
        lse_offset = (batch_id * q_num_heads + q_global_head_id) * split_num + split_id + tl.arange(0, 1)
        tl.store(lse_ptr + lse_offset, lse)


@triton.jit
def flash_decode_kernel_reduce(
    split_o_ptr,
    split_lse_ptr,
    o_ptr,
    split_num: int,
    head_dim: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    q_num_heads = tl.num_programs(1)

    seq_idx = batch_id * q_num_heads + q_head_id
    split_o_base = split_o_ptr + seq_idx * split_num * head_dim
    split_o_desc = tl.make_tensor_descriptor(
        base=split_o_base,
        shape=[split_num, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SPLIT, head_dim],
    )

    split_lse_base = split_lse_ptr + seq_idx * split_num
    split_lse_desc = tl.make_tensor_descriptor(
        base=split_lse_base,
        shape=[split_num],
        strides=[1],
        block_shape=[BLOCK_SPLIT],
    )

    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    o_tile = tl.zeros((1, head_dim), dtype=tl.float32)

    for split_start in range(0, split_num, BLOCK_SPLIT):

        # Load lse tile (mask out-of-bounds)
        lse = split_lse_desc.load([split_start])
        split_offsets = split_start + tl.arange(0, BLOCK_SPLIT)
        split_mask = split_offsets < split_num
        lse = tl.where(split_mask, lse, float("-inf"))

        tile_max = tl.max(lse)
        new_m_i = tl.maximum(m_i, tile_max)
        exp_scores = tl.exp(lse - new_m_i)

        # Load v_tile from split_o
        v_tile = split_o_desc.load([split_start, 0])

        alpha = tl.exp(m_i - new_m_i)
        l_i = alpha * l_i + tl.sum(exp_scores, axis=0)
        o_tile = alpha * o_tile + tl.dot(exp_scores[None, :].to(v_tile.dtype), v_tile)

        m_i = new_m_i

    # finalize output and store
    o_tile = o_tile / l_i
    o_base = o_ptr + batch_id * q_num_heads * head_dim
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[q_num_heads, head_dim],
        strides=[head_dim, 1],
        block_shape=[1, head_dim],
    )
    o_desc.store([q_head_id, 0], o_tile.to(o_desc.dtype))


def triton_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    split_len=512,
    block_kv=32,
):
    batch_size, q_num_heads, q_len, head_dim = q.shape
    _, kv_num_heads, kv_len, _ = k.shape

    assert q_len == 1, "Only decoding with q_len=1 is supported."

    # split stage
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

    # reduce stage
    grid = (
        batch_size,
        q_num_heads,
    )
    flash_decode_kernel_reduce[grid](
        split_o_ptr=per_split_o,
        split_lse_ptr=per_split_lse,
        o_ptr=o,
        split_num=split_num,
        head_dim=head_dim,
        BLOCK_SPLIT=block_kv,
    )


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
    o_tensor = torch.empty_like(q_tensor)

    excepted = torch.nn.functional.scaled_dot_product_attention(
        q_tensor,
        k_tensor,
        v_tensor,
        is_causal=False,
        enable_gqa=True,
    )

    bench_by_secs(
        10,
        lambda: triton_flash_decode(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
        ),
    )

    acc_check(excepted, o_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    flash_decode(
        batch_size=1,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
