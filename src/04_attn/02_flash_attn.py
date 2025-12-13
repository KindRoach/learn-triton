import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


@triton.jit
def flash_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_len: int,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    kv_num_heads = tl.num_programs(1)
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

    # iterate over GQA heads
    for q_gqa_head_id in range(0, gqa_size):
        q_num_heads = kv_num_heads * gqa_size
        q_global_head_id = kv_head_id * gqa_size + q_gqa_head_id
        q_head_offset = (batch_id * q_num_heads + q_global_head_id) * q_len * HEAD_DIM
        q_desc = tl.make_tensor_descriptor(
            base=q_ptr + q_head_offset,
            shape=[q_len, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_Q, HEAD_DIM],
        )

        # Load Q tile
        q_block_start = tl.program_id(2) * BLOCK_Q
        q_tile = q_desc.load([q_block_start, 0])

        # initialize tmp variables
        m_i = tl.full((BLOCK_Q, 1), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)
        o_tile = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

        # iterate over KV in blocks
        for kv_block_start in range(0, kv_len, BLOCK_KV):
            # Load k and v tiles
            k_tile = k_desc.load([kv_block_start, 0])
            v_tile = v_desc.load([kv_block_start, 0])

            # attn = Q @ K.T * scale
            inv_sqrt_d = 1.0 / (HEAD_DIM**0.5)
            scores = tl.dot(q_tile, k_tile.T) * inv_sqrt_d

            # apply causal mask
            kv_offsets = (kv_block_start + tl.arange(0, BLOCK_KV))[None, :]
            q_offsets = (q_block_start + (kv_len - q_len) + tl.arange(0, BLOCK_Q))[:, None]
            casual_mask = kv_offsets <= q_offsets
            scores = tl.where(casual_mask, scores, float("-inf"))

            # compute softmax and output
            tile_max = tl.max(scores, axis=1, keep_dims=True)
            new_m_i = tl.maximum(m_i, tile_max)
            exp_scores = tl.exp(scores - new_m_i)

            alpha = tl.exp(m_i - new_m_i)
            l_i = alpha * l_i + tl.sum(exp_scores, axis=1, keep_dims=True)
            o_tile = alpha * o_tile + tl.dot(exp_scores.to(v_tile.dtype), v_tile)

            m_i = new_m_i

        # finalize output and store
        o_tile = o_tile / l_i
        o_desc = tl.make_tensor_descriptor(
            base=o_ptr + q_head_offset,
            shape=[q_len, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_Q, HEAD_DIM],
        )
        o_desc.store([q_block_start, 0], o_tile.to(o_desc.dtype))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_Q": block_q,
                "BLOCK_KV": block_kv,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_q in [16, 32, 64]
        for block_kv in [16, 32, 64]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
    ],
    key=["HEAD_DIM"],
    cache_results=True,
)
@triton.jit
def flash_attn_autotune_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_len: int,
    kv_len: int,
    gqa_size: int,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    flash_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        q_len,
        kv_len,
        gqa_size,
        HEAD_DIM,
        BLOCK_Q,
        BLOCK_KV,
    )


def flash_attn(
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
) -> None:
    batch_size, q_num_heads, q_len, head_dim = q_tensor.shape
    _, kv_num_heads, kv_len, _ = k_tensor.shape
    gqa_size = q_num_heads // kv_num_heads

    BLOCK_Q = 32
    BLOCK_KV = 32

    grid = (
        batch_size,
        kv_num_heads,
        triton.cdiv(q_len, BLOCK_Q),
    )

    flash_attn_kernel[grid](
        q_ptr=q_tensor,
        k_ptr=k_tensor,
        v_ptr=v_tensor,
        o_ptr=o_tensor,
        q_len=q_len,
        kv_len=kv_len,
        gqa_size=gqa_size,
        HEAD_DIM=head_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )


def flash_attn_autotune(
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
) -> None:
    batch_size, q_num_heads, q_len, head_dim = q_tensor.shape
    _, kv_num_heads, kv_len, _ = k_tensor.shape
    gqa_size = q_num_heads // kv_num_heads

    grid = lambda meta: (
        batch_size,
        kv_num_heads,
        triton.cdiv(q_len, meta["BLOCK_Q"]),
    )

    flash_attn_autotune_kernel[grid](
        q_ptr=q_tensor,
        k_ptr=k_tensor,
        v_ptr=v_tensor,
        o_ptr=o_tensor,
        q_len=q_len,
        kv_len=kv_len,
        gqa_size=gqa_size,
        HEAD_DIM=head_dim,
    )


def flash_attn_exp(
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

    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device)[:, None]  # [q_len, 1]
    k_idx = torch.arange(kv_len, device=device)[None, :]  # [1, kv_len]
    causal = k_idx <= (q_idx + offset)  # [q_len, kv_len]
    attn_mask = causal.view(1, 1, q_len, kv_len)  # [1,1,q_len,kv_len]
    attn_mask = attn_mask.expand(batch_size, q_num_heads, -1, -1)  # [B,H,q_len,kv_len]

    excepted = torch.nn.functional.scaled_dot_product_attention(
        q_tensor,
        k_tensor,
        v_tensor,
        attn_mask=attn_mask,
        is_causal=False,  # we provide the mask explicitly
        enable_gqa=True,  # keep if q/k heads differ by an integer factor
    )

    funcs_to_bench = {
        flash_attn.__name__: flash_attn,
        flash_attn_autotune.__name__: flash_attn_autotune,
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(q_tensor, k_tensor, v_tensor, o_tensor),
        )
        acc_check(excepted, o_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    flash_attn_exp(
        batch_size=1,
        q_len=4 * 1024,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
