import torch


def attn_mem_access_bytes(
    batch_size: int,
    q_len: int,
    kv_len: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Approximate HBM traffic (bytes) per forward call.

    Counts reading Q/K/V and writing O once each.
    """
    bytes_per_elem = torch.empty((), dtype=dtype).element_size()

    q_elems = batch_size * q_num_heads * q_len * head_dim
    o_elems = q_elems
    k_elems = batch_size * kv_num_heads * kv_len * head_dim
    v_elems = k_elems

    return int((q_elems + k_elems + v_elems + o_elems) * bytes_per_elem)


def attn_matmul_flops(
    batch_size: int,
    q_len: int,
    kv_len: int,
    q_num_heads: int,
    head_dim: int,
) -> int:
    """Approximate matmul FLOPs per forward call.

    Assumes two GEMMs per head:
    - QK^T: [q_len, d] x [d, kv_len]
    - P V : [q_len, kv_len] x [kv_len, d]

    Using mul+add = 2 FLOPs, total is 4 * B * Hq * q_len * kv_len * d.
    """
    return int(4 * batch_size * q_num_heads * q_len * kv_len * head_dim)


def ref_attn_prefill(
    query: torch.Tensor,  # [batch_size, num_query_heads, query_length, head_dim]
    key: torch.Tensor,  # [batch_size, num_kv_heads, kv_length, head_dim]
    value: torch.Tensor,  # [batch_size, num_kv_heads, kv_length, head_dim]
    output: torch.Tensor,  # [batch_size, num_query_heads, query_length, head_dim]
):
    # Reference (readable) attention forward for “prefill” / prompt attention.
    # Uses the same causal mask convention as other ref code in this repo:
    #   key_idx <= query_idx + (kv_len - q_len)
    # i.e., query positions align to the *end* of the KV history.
    assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4 and output.ndim == 4
    batch_size, q_num_heads, q_len, head_dim = query.shape
    b_k, kv_num_heads, kv_len, head_dim_k = key.shape
    b_v, kv_num_heads_v, kv_len_v, head_dim_v = value.shape

    assert output.shape == (batch_size, q_num_heads, q_len, head_dim)
    assert b_k == batch_size and b_v == batch_size
    assert kv_num_heads_v == kv_num_heads
    assert kv_len_v == kv_len
    assert head_dim_k == head_dim and head_dim_v == head_dim
    assert kv_len >= q_len, "kv_len must be greater than or equal to q_len"
    assert key.device == query.device == value.device == output.device

    # GQA/MQA: each KV head is shared by a group of Q heads.
    if q_num_heads != kv_num_heads:
        assert (
            q_num_heads % kv_num_heads == 0
        ), f"q_num_heads ({q_num_heads}) must be a multiple of kv_num_heads ({kv_num_heads})"
        gqa_size = q_num_heads // kv_num_heads
        key = key.repeat_interleave(gqa_size, dim=1)
        value = value.repeat_interleave(gqa_size, dim=1)

    inv_sqrt_d = 1.0 / (head_dim**0.5)

    # scores: [B, Hq, Lq, Lkv]
    # Compute in fp32 for numerical stability.
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * inv_sqrt_d

    # Causal mask with end-alignment between Q and KV.
    # query index i corresponds to KV index (i + kv_len - q_len).
    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=query.device)[:, None]  # [Lq, 1]
    k_idx = torch.arange(kv_len, device=query.device)[None, :]  # [1, Lkv]
    causal = k_idx <= (q_idx + offset)  # [Lq, Lkv]
    scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))

    # probs: [B, Hq, Lq, Lkv]
    probs = torch.softmax(scores, dim=-1)

    # out: [B, Hq, Lq, D]
    out = torch.matmul(probs, value.float()).to(dtype=output.dtype)
    output.copy_(out)
