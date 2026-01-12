import torch
from ..utils import get_device


def ref_flash_attn_paged(
    q: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    k: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    v: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    o: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    seqused_kv: torch.Tensor,  # [num_seqs]
    block_table: torch.Tensor,  # [num_seqs, num_used_blocks]
):
    # This is a *reference* implementation:
    # - reconstruct each sequence's KV from a paged KV cache (block_table)
    # - run attention in pure PyTorch (not optimized)
    # - uses the same causal mask convention as other ref code in this repo:
    #   key_idx <= query_idx + (kv_len - q_len)
    assert q.ndim == 3 and k.ndim == 4 and v.ndim == 4 and o.ndim == 3
    assert q.shape == o.shape
    assert k.shape == v.shape
    assert cu_seqlens_q.ndim == 1
    assert seqused_kv.ndim == 1
    assert block_table.ndim == 2

    device = q.device
    total_q_len, q_num_heads, head_dim = q.shape
    block_num, block_size, kv_num_heads, head_dim_kv = k.shape
    assert head_dim_kv == head_dim
    assert cu_seqlens_q.shape[0] == seqused_kv.shape[0] + 1

    inv_sqrt_d = 1.0 / (head_dim**0.5)
    o.zero_()

    num_seqs = seqused_kv.shape[0]
    for seq_id in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_id].item())
        q_end = int(cu_seqlens_q[seq_id + 1].item())
        q_len = q_end - q_start
        if q_len == 0:
            continue

        kv_len = int(seqused_kv[seq_id].item())
        if kv_len == 0:
            continue

        assert kv_len >= q_len, "kv_len must be greater than or equal to q_len"

        # Gather KV blocks for this sequence.
        num_used_blocks = (kv_len + block_size - 1) // block_size
        assert num_used_blocks <= block_table.shape[1]
        block_ids = block_table[seq_id, :num_used_blocks].to(torch.long)
        assert (block_ids >= 0).all() and (block_ids < block_num).all()

        # k_blocks/v_blocks: [num_used_blocks, block_size, Hkv, D]
        k_blocks = k.index_select(0, block_ids)
        v_blocks = v.index_select(0, block_ids)

        # Flatten blocks into a contiguous [kv_len, Hkv, D].
        k_seq = k_blocks.reshape(num_used_blocks * block_size, kv_num_heads, head_dim)[:kv_len]
        v_seq = v_blocks.reshape(num_used_blocks * block_size, kv_num_heads, head_dim)[:kv_len]

        # Slice Q for this sequence: [q_len, Hq, D].
        q_seq = q[q_start:q_end]

        # Convert to [H, L, D] layout for matmuls.
        q_hld = q_seq.transpose(0, 1)  # [Hq, Lq, D]
        k_hld = k_seq.transpose(0, 1)  # [Hkv, Lkv, D]
        v_hld = v_seq.transpose(0, 1)  # [Hkv, Lkv, D]

        # GQA: repeat KV heads to match Q heads.
        if q_num_heads != kv_num_heads:
            assert (
                q_num_heads % kv_num_heads == 0
            ), f"q_num_heads ({q_num_heads}) must be a multiple of kv_num_heads ({kv_num_heads})"
            gqa_size = q_num_heads // kv_num_heads
            k_hld = k_hld.repeat_interleave(gqa_size, dim=0)
            v_hld = v_hld.repeat_interleave(gqa_size, dim=0)

        # scores: [Hq, Lq, Lkv]
        # Compute in fp32 for numerical stability.
        scores = torch.matmul(q_hld.float(), k_hld.float().transpose(-1, -2)) * inv_sqrt_d

        # Causal mask (same convention as ref prefill):
        # allow attending up to the query position aligned at the end of KV.
        # i.e., query index i corresponds to KV index (i + kv_len - q_len).
        offset = kv_len - q_len
        q_idx = torch.arange(q_len, device=device)[:, None]  # [Lq, 1]
        k_idx = torch.arange(kv_len, device=device)[None, :]  # [1, Lkv]
        causal = k_idx <= (q_idx + offset)  # [Lq, Lkv]
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

        # probs: [Hq, Lq, Lkv]
        probs = torch.softmax(scores, dim=-1)

        # out: [Hq, Lq, D]
        out = torch.matmul(probs, v_hld.float()).to(o.dtype)

        # Write back to packed output: [Lq, Hq, D]
        o[q_start:q_end] = out.transpose(0, 1)


def generate_random_tensors():
    device = get_device()

    torch.manual_seed(0)

    dtype = torch.bfloat16

    # Each sequence has the same lengths.
    num_seqs = 4
    q_len = 1024
    kv_len = 8192
    q_num_heads = 28
    kv_num_heads = 4
    block_num = 5000
    block_size = 16
    head_dim = 128

    total_q_len = num_seqs * q_len
    q = torch.randn((total_q_len, q_num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((block_num, block_size, kv_num_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((block_num, block_size, kv_num_heads, head_dim), device=device, dtype=dtype)

    cu_seqlens_q = torch.arange(num_seqs + 1, device=device, dtype=torch.int32) * q_len
    max_seqlen_q = q_len

    num_used_blocks = (kv_len + block_size - 1) // block_size
    seqused_kv = torch.full((num_seqs,), kv_len, device=device, dtype=torch.int32)
    max_seqlen_kv = kv_len

    block_table = torch.empty((num_seqs, num_used_blocks), device=device, dtype=torch.int32)
    for seq_id in range(num_seqs):
        block_table[seq_id] = torch.randperm(block_num, device=device, dtype=torch.int32)[:num_used_blocks]

    return (
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_kv,
        max_seqlen_kv,
        block_table,
    )


@torch.no_grad()
def main():
    (
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_kv,
        max_seqlen_kv,
        block_table,
    ) = generate_random_tensors()

    # reference implementation
    out_ref = torch.empty_like(q)
    ref_flash_attn_paged(
        q,
        k,
        v,
        out_ref,
        cu_seqlens_q,
        seqused_kv,
        block_table,
    )


if __name__ == "__main__":
    main()
