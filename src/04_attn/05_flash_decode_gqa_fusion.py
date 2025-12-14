import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device
from .utils import attn_matmul_flops, attn_mem_access_bytes


@triton.jit
def flash_decode_kernel_split(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    MAX_GQA_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):

    # define Q/K/V descriptors
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    kv_num_heads = tl.num_programs(1)
    q_num_heads = kv_num_heads * gqa_size

    q_head_offset = batch_id * q_num_heads + kv_head_id * gqa_size
    q_offset = q_head_offset * HEAD_DIM
    q_block_desc = tl.make_tensor_descriptor(
        base=q_ptr + q_offset,
        shape=[gqa_size, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[MAX_GQA_SIZE, HEAD_DIM],
    )

    kv_offset = (batch_id * kv_num_heads + kv_head_id) * kv_len * HEAD_DIM
    k_desc = tl.make_tensor_descriptor(
        base=k_ptr + kv_offset,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_KV, HEAD_DIM],
    )

    v_desc = tl.make_tensor_descriptor(
        base=v_ptr + kv_offset,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_KV, HEAD_DIM],
    )

    # determine split range
    split_id = tl.program_id(2)
    split_num = tl.num_programs(2)
    split_kv_start = split_id * (kv_len // split_num)
    split_kv_end = tl.minimum(kv_len, (split_id + 1) * (kv_len // split_num))

    # Load all Q heads in the current GQA group as one block
    q_block = q_block_desc.load([0, 0])

    # Initialize streaming softmax accumulators per Q head
    m_i = tl.full((MAX_GQA_SIZE, 1), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((MAX_GQA_SIZE, 1), dtype=tl.float32)
    o_block = tl.zeros((MAX_GQA_SIZE, HEAD_DIM), dtype=tl.float32)

    # Iterate over K/V cache in blocks in split range
    for kv_block_start in range(split_kv_start, split_kv_end, BLOCK_KV):
        k_tile = k_desc.load([kv_block_start, 0])
        v_tile = v_desc.load([kv_block_start, 0])

        inv_sqrt_d = 1.0 / (HEAD_DIM**0.5)
        scores = tl.dot(q_block, k_tile.T) * inv_sqrt_d

        tile_max = tl.max(scores, axis=1, keep_dims=True)
        new_m_i = tl.maximum(m_i, tile_max)
        exp_scores = tl.exp(scores - new_m_i)

        alpha = tl.exp(m_i - new_m_i)
        l_i = alpha * l_i + tl.sum(exp_scores, axis=1, keep_dims=True)
        o_block = alpha * o_block + tl.dot(exp_scores.to(v_tile.dtype), v_tile)

        m_i = new_m_i

    # Finalize outputs for the entire GQA block
    o_block = o_block / l_i
    lse_block = tl.log(l_i) + m_i

    split_id = tl.program_id(2)
    split_num = tl.num_programs(2)

    o_base = o_ptr + ((q_head_offset) * split_num + split_id) * HEAD_DIM
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[gqa_size, HEAD_DIM],
        strides=[split_num * HEAD_DIM, 1],
        block_shape=[MAX_GQA_SIZE, HEAD_DIM],
    )
    o_desc.store([0, 0], o_block.to(o_desc.dtype))

    lse_heads = tl.arange(0, MAX_GQA_SIZE)
    gqa_mask = lse_heads < gqa_size
    lse_offsets = (q_head_offset + lse_heads) * split_num + split_id
    tl.store(
        lse_ptr + lse_offsets,
        lse_block.reshape(
            MAX_GQA_SIZE,
        ),
        mask=gqa_mask,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_KV": block_kv,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_kv in [16, 32, 64]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
    ],
    key=["HEAD_DIM", "MAX_GQA_SIZE"],
    cache_results=True,
)
@triton.jit
def flash_decode_kernel_split_autotune(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    MAX_GQA_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    flash_decode_kernel_split(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        kv_len,
        gqa_size,
        HEAD_DIM,
        MAX_GQA_SIZE,
        BLOCK_KV,
    )


@triton.jit
def flash_decode_kernel_reduce(
    split_o_ptr,
    split_lse_ptr,
    o_ptr,
    split_num: int,
    HEAD_DIM: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    q_num_heads = tl.num_programs(1)

    seq_idx = batch_id * q_num_heads + q_head_id
    split_o_base = split_o_ptr + seq_idx * split_num * HEAD_DIM
    split_o_desc = tl.make_tensor_descriptor(
        base=split_o_base,
        shape=[split_num, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_SPLIT, HEAD_DIM],
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
    o_tile = tl.zeros((1, HEAD_DIM), dtype=tl.float32)

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
    o_base = o_ptr + batch_id * q_num_heads * HEAD_DIM
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[q_num_heads, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[1, HEAD_DIM],
    )
    o_desc.store([q_head_id, 0], o_tile.to(o_desc.dtype))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SPLIT": block_split,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_split in [16, 32, 64]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
    ],
    key=["gqa_size", "head_dim"],
    cache_results=True,
)
@triton.jit
def flash_decode_kernel_reduce_autotune(
    split_o_ptr,
    split_lse_ptr,
    o_ptr,
    split_num: int,
    HEAD_DIM: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
):
    flash_decode_kernel_reduce(
        split_o_ptr,
        split_lse_ptr,
        o_ptr,
        split_num,
        HEAD_DIM,
        BLOCK_SPLIT,
    )


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    split_len=512,
    block_kv=32,
    max_gqa_size=8,
):
    batch_size, q_num_heads, q_len, head_dim = q.shape
    _, kv_num_heads, kv_len, _ = k.shape

    assert q_len == 1, "Only decoding with q_len=1 is supported."

    # split stage
    gqa_size = q_num_heads // kv_num_heads
    assert gqa_size <= max_gqa_size, "flash_decode_kernel_split_gqa_block assumes gqa_size <= 8."

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
        HEAD_DIM=head_dim,
        gqa_size=gqa_size,
        MAX_GQA_SIZE=max_gqa_size,
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
        HEAD_DIM=head_dim,
        BLOCK_SPLIT=block_kv,
    )


def flash_decode_autotune(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    split_len=512,
    max_gqa_size=8,
):
    batch_size, q_num_heads, q_len, head_dim = q.shape
    _, kv_num_heads, kv_len, _ = k.shape

    assert q_len == 1, "Only decoding with q_len=1 is supported."

    gqa_size = q_num_heads // kv_num_heads
    split_num = int(triton.cdiv(kv_len, split_len))
    grid_split = (
        batch_size,
        kv_num_heads,
        split_num,
    )

    per_split_o = torch.empty((batch_size, q_num_heads, split_num, head_dim), device=q.device, dtype=q.dtype)
    per_split_lse = torch.empty((batch_size, q_num_heads, split_num), device=q.device, dtype=q.dtype)
    flash_decode_kernel_split_autotune[grid_split](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=per_split_o,
        lse_ptr=per_split_lse,
        kv_len=kv_len,
        gqa_size=gqa_size,
        HEAD_DIM=head_dim,
        MAX_GQA_SIZE=max_gqa_size,
    )

    grid_reduce = (
        batch_size,
        q_num_heads,
    )
    flash_decode_kernel_reduce_autotune[grid_reduce](
        split_o_ptr=per_split_o,
        split_lse_ptr=per_split_lse,
        o_ptr=o,
        split_num=split_num,
        HEAD_DIM=head_dim,
    )


def flash_decode_exp(
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

    mem_access_bytes = attn_mem_access_bytes(
        batch_size=batch_size,
        q_len=1,
        kv_len=kv_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    total_flops = attn_matmul_flops(
        batch_size=batch_size,
        q_len=1,
        kv_len=kv_len,
        q_num_heads=q_num_heads,
        head_dim=head_dim,
    )

    excepted = torch.nn.functional.scaled_dot_product_attention(
        q_tensor,
        k_tensor,
        v_tensor,
        is_causal=False,
        enable_gqa=True,
    )

    funcs_to_bench = {
        flash_decode.__name__: flash_decode,
        flash_decode_autotune.__name__: flash_decode_autotune,
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(q_tensor, k_tensor, v_tensor, o_tensor),
            mem_access_bytes=mem_access_bytes,
            total_flops=total_flops,
        )
        acc_check(excepted, o_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    flash_decode_exp(
        batch_size=1,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
