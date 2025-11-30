import torch

from ..utils import acc_check, get_device


def flash_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_block_size=32,
    kv_block_size=32,
):
    B, Hq, Lq, D = q.shape
    _, Hkv, Lkv, _ = k.shape

    # GQA: repeat kv-heads to match q-heads
    if Hq != Hkv:
        assert Hq % Hkv == 0, f"Query heads ({Hq}) must divide KV heads ({Hkv})"
        gqa_size = Hq // Hkv
        k = k.repeat_interleave(gqa_size, dim=1)
        v = v.repeat_interleave(gqa_size, dim=1)

    device = q.device
    inv_sqrt_d = 1.0 / (D**0.5)

    # Iterate over Q blocks
    for qs in range(0, Lq, q_block_size):
        qe = min(qs + q_block_size, Lq)
        q_block = q[:, :, qs:qe]  # [B, H, M, D]
        M = qe - qs

        # Running max and exp-sum for softmax
        m_i = torch.full((B, Hq, M), float("-inf"), device=device, dtype=torch.float32)
        l_i = torch.zeros((B, Hq, M), device=device, dtype=torch.float32)
        out_block = torch.zeros((B, Hq, M, D), device=device, dtype=torch.float32)

        # Iterate over KV blocks
        for ks in range(0, Lkv, kv_block_size):
            ke = min(ks + kv_block_size, Lkv)
            N = ke - ks

            # Compute attention scores
            k_block = k[:, :, ks:ke]  # [B, H, N, D]
            v_block = v[:, :, ks:ke]  # [B, H, N, D]
            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * inv_sqrt_d  # [B, H, M, N]

            # Causal masking
            kv_offsets = torch.arange(ks, ke, device=device)[None, None, :]
            q_offsets = torch.arange(qs, qe, device=device)[:, None] + (Lkv - Lq)
            causal_mask = kv_offsets <= q_offsets
            scores = torch.where(causal_mask.unsqueeze(0), scores, float("-inf"))

            # Online softmax computation
            block_m = scores.max(dim=-1).values  # [B, H, M]
            new_m_i = torch.maximum(m_i, block_m)  # [B, H, M]
            exp_scores = torch.exp(scores - new_m_i.unsqueeze(-1))  # [B, H, M, N]

            # rescale old l_i and out_block
            alpha = torch.exp(m_i - new_m_i)  # [B, H, M]
            l_i = alpha * l_i + exp_scores.sum(dim=-1)  # [B, H, M]
            out_block = alpha.unsqueeze(-1) * out_block + torch.matmul(exp_scores.to(v_block.dtype), v_block)

            m_i = new_m_i

        # Normalize
        out_block = out_block / l_i.unsqueeze(-1)
        o[:, :, qs:qe] = out_block.to(o.dtype)

    return o


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

    flash_attention_torch(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
    )

    acc_check(excepted, o_tensor)


if __name__ == "__main__":
    flash_attn(
        batch_size=1,
        q_len=4 * 1024,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=7,  # 28/7 = 4, so each kv head will be repeated 4 times
        head_dim=128,
    )
