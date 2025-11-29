import torch

from ..utils import acc_check, get_device


def flash_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    split_len=512,
    kv_chunk_size=32,
):
    B, H_q, L_q, D = q.shape
    _, H_kv, L, _ = k.shape

    assert L_q == 1, "This function only supports decoding with q_len=1"

    # Support for GQA: repeat k,v heads to match q heads if needed
    if H_q != H_kv:
        assert H_q % H_kv == 0, f"Query heads ({H_q}) must be divisible by key/value heads ({H_kv}) for GQA"
        gqa_size = H_q // H_kv
        k = k.repeat_interleave(gqa_size, dim=1)  # [B, H_q, L_kv, D]
        v = v.repeat_interleave(gqa_size, dim=1)  # [B, H_q, L_kv, D]

    scale = 1.0 / (D**0.5)
    device = q.device

    num_split = L // split_len + int(L % split_len != 0)
    per_split_lse = torch.zeros((B, H_q, num_split), device=device)  # [B, H_q, l_chunk_num]
    per_split_output = torch.zeros((B, H_q, num_split, D), device=device)  # [B, H_q, l_chunk_num, D]

    for ls in range(0, L, split_len):
        le = min(ls + split_len, L)

        # for each l chunk, perform flash attention
        max_score = torch.full((B, H_q, 1), float("-inf"), device=device)  # [B, H_q, 1]
        exp_sum = torch.zeros((B, H_q, 1), device=device)  # [B, H_q, 1]
        out_chunk = torch.zeros((B, H_q, 1, D), device=device)  # [B, H_q, 1, D]

        for ks in range(ls, le, kv_chunk_size):
            ke = min(ks + kv_chunk_size, le)
            k_chunk = k[:, :, ks:ke]  # [B, H_q, Ck, D]
            v_chunk = v[:, :, ks:ke]  # [B, H_q, Ck, D]

            attn_scores = torch.matmul(q, k_chunk.transpose(-1, -2)) * scale  # [B, H_q, 1, Ck]

            block_max = attn_scores.max(dim=-1).values  # [B, H_q, 1]
            max_score_new = torch.maximum(max_score, block_max)  # [B, H_q, 1]
            exp_scores = torch.exp(attn_scores - max_score_new.unsqueeze(-1))  # [B, H_q, 1, Ck]

            exp_max_diff = torch.exp(max_score - max_score_new)  # [B, H_q, 1]
            exp_sum = exp_max_diff * exp_sum + exp_scores.sum(dim=-1)  # [B, H_q, 1]
            out_chunk = exp_max_diff.unsqueeze(-1) * out_chunk + torch.matmul(
                exp_scores.to(v_chunk.dtype), v_chunk
            )  # [B, H_q, 1, D]

            max_score = max_score_new

        out_chunk = out_chunk / exp_sum.unsqueeze(-1)

        # record output and lse for each l chunk
        split_i = ls // split_len
        per_split_lse[:, :, split_i : split_i + 1] = torch.log(exp_sum) + max_score  # [B, H_q, 1]
        per_split_output[:, :, split_i : split_i + 1] = out_chunk  # [B, H_q, 1, D]

    # reduce refer to https://github.com/Dao-AILab/flash-attention/issues/1248
    lse_final = torch.logsumexp(per_split_lse, dim=2, keepdim=True)  # [B, H_q, 1]
    exp_final = torch.exp(per_split_lse - lse_final).unsqueeze(2)  # [B, H_q, 1, l_chunk_num]
    o[:] = torch.matmul(exp_final, per_split_output)  # [B, H_q, 1, D]

    return o


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

    flash_decode_torch(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
    )

    acc_check(excepted, o_tensor)


if __name__ == "__main__":
    flash_decode(
        batch_size=1,
        kv_len=16 * 1024,
        q_num_heads=28,
        kv_num_heads=7,
        head_dim=128,
    )
