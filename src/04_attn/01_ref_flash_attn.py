import torch

from ..utils import acc_check, get_device


def flash_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_chunk_size=32,
    kv_chunk_size=32,
):
    B, H_q, L, D = q.shape
    _, H_kv, L_kv, _ = k.shape

    # Support for GQA: repeat k,v heads to match q heads if needed
    if H_q != H_kv:
        assert H_q % H_kv == 0, f"Query heads ({H_q}) must be divisible by key/value heads ({H_kv}) for GQA"
        repeat_factor = H_q // H_kv
        k = k.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L_kv, D]
        v = v.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L_kv, D]

    scale = 1.0 / (D**0.5)
    device = q.device

    for qs in range(0, L, q_chunk_size):
        qe = min(qs + q_chunk_size, L)
        q_chunk = q[:, :, qs:qe]  # [B, H, Cq, D]

        # Accumulate in float32 for stability
        max_score = torch.full((B, H_q, qe - qs), float("-inf"), device=device, dtype=torch.float32)  # [B, H_q, Cq]
        exp_sum = torch.zeros((B, H_q, qe - qs), device=device, dtype=torch.float32)  # [B, H_q, Cq]
        out_chunk = torch.zeros((B, H_q, qe - qs, D), device=device, dtype=torch.float32)  # [B, H_q, Cq, D]

        for ks in range(0, L_kv, kv_chunk_size):
            ke = min(ks + kv_chunk_size, L_kv)
            k_chunk = k[:, :, ks:ke]  # [B, H_q, Ck, D]
            v_chunk = v[:, :, ks:ke]  # [B, H_q, Ck, D]

            # Compute scores in float32
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale  # [B, H_q, Cq, Ck]
            casual_mask = (torch.arange(ks, ke, device=device)[None, None, :]) <= (
                torch.arange(qs, qe, device=device)[:, None] + (L_kv - L)
            )
            attn_scores = torch.where(casual_mask.unsqueeze(0).unsqueeze(0), attn_scores, float("-inf"))

            block_max = attn_scores.max(dim=-1).values  # [B, H_q, Cq]
            max_score_new = torch.maximum(max_score, block_max)  # [B, H_q, Cq]
            exp_scores = torch.exp(attn_scores - max_score_new.unsqueeze(-1))  # [B, H_q, Cq, Ck]

            exp_max_diff = torch.exp(max_score - max_score_new)  # [B, H_q, Cq]
            exp_sum = exp_max_diff * exp_sum + exp_scores.sum(dim=-1)  # [B, H_q, Cq]
            out_chunk = exp_max_diff.unsqueeze(-1) * out_chunk + torch.matmul(
                exp_scores, v_chunk.to(torch.float32)
            )  # [B, H_q, Cq, D]

            max_score = max_score_new

        out_chunk = out_chunk / exp_sum.unsqueeze(-1)
        o[:, :, qs:qe] = out_chunk.to(o.dtype)

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
