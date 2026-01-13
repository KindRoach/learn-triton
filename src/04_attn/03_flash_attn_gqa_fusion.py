import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device
from .utils import attn_matmul_flops, attn_mem_access_bytes, ref_attn_prefill


@triton.jit
def flash_attn_gqa_fusion_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_len: int,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    MAX_GQA_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    kv_num_heads = tl.num_programs(1)
    q_num_heads = kv_num_heads * gqa_size

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

    # Load all Q heads in the current GQA group as one block
    q_head_base = (batch_id * q_num_heads + kv_head_id * gqa_size) * q_len * HEAD_DIM
    q_desc = tl.make_tensor_descriptor(
        base=q_ptr + q_head_base,
        shape=[gqa_size, q_len, HEAD_DIM],
        strides=[q_len * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[MAX_GQA_SIZE, BLOCK_Q, HEAD_DIM],
    )

    q_block_start = q_block_id * BLOCK_Q
    q_block = q_desc.load([0, q_block_start, 0])
    q_block = tl.reshape(q_block, (MAX_GQA_SIZE * BLOCK_Q, HEAD_DIM))

    # streaming softmax accumulators per (head, q_row)
    m_i = tl.full((MAX_GQA_SIZE, BLOCK_Q, 1), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((MAX_GQA_SIZE, BLOCK_Q, 1), dtype=tl.float32)
    o_acc = tl.zeros((MAX_GQA_SIZE, BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    inv_sqrt_d = 1.0 / (HEAD_DIM**0.5)

    # one pass over the KV history; reuse each kv tile for all gqa heads
    for kv_block_start in range(0, kv_len, BLOCK_KV):
        k_tile = k_desc.load([kv_block_start, 0])
        v_tile = v_desc.load([kv_block_start, 0])

        # QK^T
        scores2d = tl.dot(q_block, k_tile.T) * inv_sqrt_d
        scores = tl.reshape(scores2d, (MAX_GQA_SIZE, BLOCK_Q, BLOCK_KV))

        # apply causal mask
        kv_pos = (kv_block_start + tl.arange(0, BLOCK_KV))[None, None, :]
        q_pos = (q_block_start + kv_len - q_len + tl.arange(0, BLOCK_Q))[None, :, None]
        causal_mask = kv_pos <= q_pos
        scores = tl.where(causal_mask, scores, float("-inf"))

        # apply gqa mask
        gqa_heads = tl.arange(0, MAX_GQA_SIZE)
        gqa_mask = gqa_heads < gqa_size
        scores = tl.where(gqa_mask[:, None, None], scores, float("-inf"))

        # compute softmax and output
        tile_max = tl.max(scores, axis=2, keep_dims=True)
        new_m_i = tl.maximum(m_i, tile_max)
        exp_scores = tl.exp(scores - new_m_i)

        alpha = tl.exp(m_i - new_m_i)
        l_i = alpha * l_i + tl.sum(exp_scores, axis=2, keep_dims=True)

        exp2d = tl.reshape(exp_scores, (MAX_GQA_SIZE * BLOCK_Q, BLOCK_KV))
        o2d = tl.dot(exp2d.to(v_tile.dtype), v_tile)
        o_update = tl.reshape(o2d, (MAX_GQA_SIZE, BLOCK_Q, HEAD_DIM)).to(tl.float32)
        o_acc = alpha * o_acc + o_update

        m_i = new_m_i

    # finalize output and store
    o_out = o_acc / l_i
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr + q_head_base,
        shape=[gqa_size, q_len, HEAD_DIM],
        strides=[q_len * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[MAX_GQA_SIZE, BLOCK_Q, HEAD_DIM],
    )
    o_desc.store([0, q_block_start, 0], o_out.to(o_desc.dtype))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_Q": block_q,
                "BLOCK_KV": block_kv,
            },
        )
        for block_q in [16, 32, 64]
        for block_kv in [16, 32, 64]
    ],
    key=["HEAD_DIM", "MAX_GQA_SIZE"],
    cache_results=True,
)
@triton.jit
def flash_attn_gqa_fusion_autotune_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_len: int,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    MAX_GQA_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    flash_attn_gqa_fusion_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        q_len,
        kv_len,
        gqa_size,
        HEAD_DIM,
        MAX_GQA_SIZE,
        BLOCK_Q,
        BLOCK_KV,
    )


def flash_attn_gqa_fusion(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    block_q: int = 32,
    block_kv: int = 32,
    max_gqa_size: int = 8,
) -> None:
    batch_size, q_num_heads, q_len, head_dim = q_tensor.shape
    _, kv_num_heads, kv_len, _ = k_tensor.shape
    gqa_size = q_num_heads // kv_num_heads

    assert q_num_heads % kv_num_heads == 0, "q_num_heads must be a multiple of kv_num_heads for GQA."
    assert gqa_size <= max_gqa_size, "gqa_size must be <= max_gqa_size."

    grid = (
        batch_size,
        kv_num_heads,
        triton.cdiv(q_len, block_q),
    )

    flash_attn_gqa_fusion_kernel[grid](
        q_ptr=q_tensor,
        k_ptr=k_tensor,
        v_ptr=v_tensor,
        o_ptr=o_tensor,
        q_len=q_len,
        kv_len=kv_len,
        gqa_size=gqa_size,
        HEAD_DIM=head_dim,
        MAX_GQA_SIZE=max_gqa_size,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
    )


def flash_attn_gqa_fusion_autotune(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    max_gqa_size: int = 8,
) -> None:
    batch_size, q_num_heads, q_len, head_dim = q_tensor.shape
    _, kv_num_heads, kv_len, _ = k_tensor.shape
    gqa_size = q_num_heads // kv_num_heads

    assert q_num_heads % kv_num_heads == 0, "q_num_heads must be a multiple of kv_num_heads for GQA."
    assert gqa_size <= max_gqa_size, "gqa_size must be <= max_gqa_size."

    grid = lambda meta: (
        batch_size,
        kv_num_heads,
        triton.cdiv(q_len, meta["BLOCK_Q"]),
    )

    flash_attn_gqa_fusion_autotune_kernel[grid](
        q_ptr=q_tensor,
        k_ptr=k_tensor,
        v_ptr=v_tensor,
        o_ptr=o_tensor,
        q_len=q_len,
        kv_len=kv_len,
        gqa_size=gqa_size,
        HEAD_DIM=head_dim,
        MAX_GQA_SIZE=max_gqa_size,
    )


def flash_attn_gqa_fusion_exp(
    batch_size: int,
    q_len: int,
    kv_len: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
):
    torch.manual_seed(0)

    device = get_device()
    dtype = torch.float16
    q_tensor = torch.randn(batch_size, q_num_heads, q_len, head_dim, device=device, dtype=dtype)
    k_tensor = torch.randn(batch_size, kv_num_heads, kv_len, head_dim, device=device, dtype=dtype)
    v_tensor = torch.randn(batch_size, kv_num_heads, kv_len, head_dim, device=device, dtype=dtype)
    o_tensor = torch.empty_like(q_tensor)
    excepted = torch.empty_like(q_tensor)

    mem_access_bytes = attn_mem_access_bytes(
        batch_size=batch_size,
        q_len=q_len,
        kv_len=kv_len,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    total_flops = attn_matmul_flops(
        batch_size=batch_size,
        q_len=q_len,
        kv_len=kv_len,
        q_num_heads=q_num_heads,
        head_dim=head_dim,
    )

    # reference implementation
    ref_attn_prefill(q_tensor, k_tensor, v_tensor, excepted)

    funcs_to_bench = {
        flash_attn_gqa_fusion.__name__: flash_attn_gqa_fusion,
        flash_attn_gqa_fusion_autotune.__name__: flash_attn_gqa_fusion_autotune,
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
    flash_attn_gqa_fusion_exp(
        batch_size=1,
        q_len=1 * 1024,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
