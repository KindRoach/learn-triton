import torch

from ..utils import acc_check, get_device


def flash_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    split_len=512,
    kv_block_size=32,
):
    B, Hq, Lq, D = q.shape
    _, Hkv, Lkv, _ = k.shape

    assert Lq == 1, "This function only supports decoding with q_len=1"

    # GQA: repeat KV heads to match Q heads
    if Hq != Hkv:
        assert Hq % Hkv == 0, f"Query heads ({Hq}) must divide KV heads ({Hkv})"
        gqa_size = Hq // Hkv
        k = k.repeat_interleave(gqa_size, dim=1)
        v = v.repeat_interleave(gqa_size, dim=1)

    device = q.device
    inv_sqrt_d = 1.0 / (D**0.5)

    num_splits = Lkv // split_len + int(Lkv % split_len != 0)
    split_lse = torch.zeros((B, Hq, num_splits), device=device)  # running max per split
    split_out = torch.zeros((B, Hq, num_splits, D), device=device)  # output per split

    for ls in range(0, Lkv, split_len):
        le = min(ls + split_len, Lkv)

        # per-split running max and exp-sum
        m_i = torch.full((B, Hq, 1), float("-inf"), device=device, dtype=torch.float32)
        l_i = torch.zeros((B, Hq, 1), device=device, dtype=torch.float32)
        out_block = torch.zeros((B, Hq, 1, D), device=device, dtype=torch.float32)

        # iterate over KV blocks
        for ks in range(ls, le, kv_block_size):
            ke = min(ks + kv_block_size, le)
            k_block = k[:, :, ks:ke]  # [B, H, N, D]
            v_block = v[:, :, ks:ke]  # [B, H, N, D]
            N = ke - ks

            scores = torch.matmul(q, k_block.transpose(-1, -2)) * inv_sqrt_d  # [B, H, 1, N]

            # running max and exp sum
            block_m = scores.max(dim=-1).values  # [B, H, 1]
            new_m_i = torch.maximum(m_i, block_m)  # [B, H, 1]
            exp_scores = torch.exp(scores - new_m_i.unsqueeze(-1))  # [B, H, 1, N]

            alpha = torch.exp(m_i - new_m_i)  # [B, H, 1]
            l_i = alpha * l_i + exp_scores.sum(dim=-1)  # [B, H, 1]
            out_block = alpha.unsqueeze(-1) * out_block + torch.matmul(
                exp_scores.to(v_block.dtype), v_block
            )  # [B, H, 1, D]

            m_i = new_m_i

        out_block = out_block / l_i.unsqueeze(-1)

        # store per-split outputs
        split_idx = ls // split_len
        split_lse[:, :, split_idx : split_idx + 1] = torch.log(l_i) + m_i
        split_out[:, :, split_idx : split_idx + 1] = out_block

    # final reduction over splits
    lse_final = torch.logsumexp(split_lse, dim=2, keepdim=True)  # [B, H, 1]
    exp_final = torch.exp(split_lse - lse_final).unsqueeze(2)  # [B, H, 1, num_splits]
    o[:] = torch.matmul(exp_final, split_out)  # [B, H, 1, D]

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
