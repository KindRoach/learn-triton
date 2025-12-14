import torch


def attn_mem_access_bytes(
    *,
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
    *,
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
