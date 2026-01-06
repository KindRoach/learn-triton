import torch
from ..utils import get_device


def generate_random_tensors():
    device = get_device()

    torch.manual_seed(0)

    dtype = torch.bfloat16

    # Each sequence has the same lengths.
    num_seqs = 4
    q_len = 1024
    kv_len = 8192
    q_num_heads = 32
    kv_num_heads = 8
    block_num = 5000
    block_size = 16
    head_dim = 128

    total_q_len = num_seqs * q_len
    q = torch.randn((total_q_len, q_num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((block_num, block_size, kv_num_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((block_num, block_size, kv_num_heads, head_dim), device=device, dtype=dtype)

    cu_seqlens_q = (torch.arange(num_seqs + 1, device=device, dtype=torch.int32) * q_len)
    max_seqlen_q = q_len

    num_used_blocks = (kv_len + block_size - 1) // block_size
    seqused_k = torch.full((num_seqs,), kv_len, device=device, dtype=torch.int32)
    max_seqlen_k = kv_len

    block_table = torch.empty((num_seqs, num_used_blocks), device=device, dtype=torch.int32)
    for seq_id in range(num_seqs):
        block_table[seq_id] = torch.randperm(block_num, device=device, dtype=torch.int32)[:num_used_blocks]

    softmax_scale = float(head_dim**-0.5)

    return (
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        block_table,
        softmax_scale,
    )

def main():
    (
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        cu_seqlens_kv,
        max_seqlen_kv,
        block_table,
        softmax_scale,
    ) = generate_random_tensors()

    print(f"q = {q.shape}, {q.dtype}")
    print(f"k = {k.shape}, {k.dtype}")
    print(f"v = {v.shape}, {v.dtype}")
    print(f"cu_seqlens_q = {cu_seqlens_q.shape}, {cu_seqlens_q.dtype}")
    print(f"max_seqlen_q = {max_seqlen_q}")
    print(f"cu_seqlens_kv = {cu_seqlens_kv.shape}, {cu_seqlens_kv.dtype}")
    print(f"max_seqlen_kv = {max_seqlen_kv}")
    print(f"block_table = {block_table.shape}, {block_table.dtype}")
    print(f"softmax_scale = {softmax_scale}")


if __name__ == "__main__":
    main()
