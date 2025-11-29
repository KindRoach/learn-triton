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
    q_len,
    kv_len,
    head_dim: tl.constexpr,
    gqa_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    kv_num_heads = tl.num_programs(1)
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
        q_num_heads = kv_num_heads * gqa_size
        q_global_head_id = kv_head_id * gqa_size + q_gqa_head_id
        q_head_offset = (batch_id * q_num_heads + q_global_head_id) * q_len * head_dim
        q_desc = tl.make_tensor_descriptor(
            base=q_ptr + q_head_offset,
            shape=[q_len, head_dim],
            strides=[head_dim, 1],
            block_shape=[BLOCK_Q, head_dim],
        )

        # Load Q tile
        q_block_start = tl.program_id(2) * BLOCK_Q
        q_tile = q_desc.load([q_block_start, 0])

        # initialize tmp variables
        max_score = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        exp_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        o_tile = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

        # iterate over KV in blocks
        for kv_block_start in range(0, kv_len, BLOCK_KV):
            # Load k and v tiles
            k_tile = k_desc.load([kv_block_start, 0])
            v_tile = v_desc.load([kv_block_start, 0])

            # attn = Q @ K.T * scale
            scale = 1.0 / (head_dim**0.5)
            attn = tl.dot(q_tile, k_tile.T) * scale

            # apply causal mask
            kv_offsets = (kv_block_start + tl.arange(0, BLOCK_KV))[None, :]
            q_offsets = (q_block_start + (kv_len - q_len) + tl.arange(0, BLOCK_Q))[:, None]
            casual_mask = kv_offsets <= q_offsets
            attn = tl.where(casual_mask, attn, float("-inf"))

            # compute softmax and output
            tile_max = tl.max(attn, axis=1)
            max_score_new = tl.maximum(max_score, tile_max)
            exp_attn = tl.exp(attn - max_score_new[:, None])

            exp_max_diff = tl.exp(max_score - max_score_new)
            exp_sum = exp_max_diff * exp_sum + tl.sum(exp_attn, axis=1)
            o_tile = exp_max_diff[:, None] * o_tile + tl.dot(exp_attn.to(v_tile.dtype), v_tile)

            max_score = max_score_new

        # finalize output and store
        o_tile = o_tile / exp_sum[:, None]
        o_desc = tl.make_tensor_descriptor(
            base=o_ptr + q_head_offset,
            shape=[q_len, head_dim],
            strides=[head_dim, 1],
            block_shape=[BLOCK_Q, head_dim],
        )
        o_desc.store([q_block_start, 0], o_tile.to(o_desc.dtype))


def flash_attn(
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

    bench_by_secs(
        10,
        lambda: torch.nn.functional.scaled_dot_product_attention(
            q_tensor,
            k_tensor,
            v_tensor,
            attn_mask=attn_mask,
            is_causal=False,  # we provide the mask explicitly
            enable_gqa=True,  # keep if q/k heads differ by an integer factor
        ),
    )

    gqa_size = q_num_heads // kv_num_heads
    BLOCK_Q = 32
    BLOCK_KV = 32
    grid = (
        batch_size,
        kv_num_heads,
        triton.cdiv(q_len, BLOCK_Q),
    )

    bench_by_secs(
        10,
        lambda: flash_attn_kernel[grid](
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            q_len,
            kv_len,
            head_dim,
            gqa_size,
            BLOCK_Q,
            BLOCK_KV,
        ),
    )

    acc_check(excepted, o_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    flash_attn(
        batch_size=1,
        q_len=4 * 1024,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=4,
        head_dim=128,
    )
