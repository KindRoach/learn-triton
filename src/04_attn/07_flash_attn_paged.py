import torch
import triton
from triton import language as tl


from ..utils import get_device, acc_check, bench_by_secs


@triton.jit
def flash_attn_paged_gqa_kernel(
    q_ptr,  # pointer to Q: [total_q_len, q_num_heads, head_dim] -> [total_q_len, kv_num_heads, gqa_size, head_dim]
    k_ptr,  # pointer to K: [block_num, block_size, kv_num_heads, head_dim]
    v_ptr,  # pointer to V: [block_num, block_size, kv_num_heads, head_dim]
    o_ptr,  # pointer to O: [total_q_len, q_num_heads, head_dim]
    cu_seqlens_q_ptr,  # pointer to cu_seqlens_q: [num_seqs + 1]
    seqused_kv_ptr,  # pointer to seqused_kv: [num_seqs]
    block_table_ptr,  # pointer to block_table: [num_seqs, max_num_blocks]
    block_num: int,
    q_num_heads: int,
    kv_num_heads: int,
    max_num_blocks: int,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_GQA: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    seq_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    # load query
    q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    q_len = q_end - q_start
    if q_block_id * BLOCK_Q >= q_len:
        return

    gqa_size = q_num_heads // kv_num_heads
    q_offset = kv_head_id * gqa_size * HEAD_DIM  # gqa offset
    q_offset += q_start * q_num_heads * HEAD_DIM  # q_start offset
    q_decs = tl.make_tensor_descriptor(
        base=q_ptr + q_offset,
        shape=[q_len, gqa_size, HEAD_DIM],
        strides=[q_num_heads * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[BLOCK_Q, MAX_GQA, HEAD_DIM],
    )

    q_block = q_decs.load([q_block_id * BLOCK_Q, 0, 0])
    q_block = q_block.reshape(BLOCK_Q * MAX_GQA, HEAD_DIM)

    # define kv descriptors
    kv_head_offset = kv_head_id * HEAD_DIM
    k_desc = tl.make_tensor_descriptor(
        base=k_ptr + kv_head_offset,
        shape=[block_num * BLOCK_SIZE, HEAD_DIM],
        strides=[kv_num_heads * HEAD_DIM, 1],
        block_shape=[BLOCK_SIZE, HEAD_DIM],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v_ptr + kv_head_offset,
        shape=[block_num * BLOCK_SIZE, HEAD_DIM],
        strides=[kv_num_heads * HEAD_DIM, 1],
        block_shape=[BLOCK_SIZE, HEAD_DIM],
    )

    # define accumulators
    m_i = tl.full((BLOCK_Q, MAX_GQA, 1), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q, MAX_GQA, 1), dtype=tl.float32)
    o_block = tl.zeros((BLOCK_Q, MAX_GQA, HEAD_DIM), dtype=tl.float32)

    # loop over KV blocks
    kv_len = tl.load(seqused_kv_ptr + seq_id)
    num_kv_blocks = tl.cdiv(kv_len, BLOCK_SIZE)

    for kv_block_id in range(num_kv_blocks):
        slot_id = tl.load(block_table_ptr + seq_id * max_num_blocks + kv_block_id)
        k_block = k_desc.load([slot_id * BLOCK_SIZE, 0])
        v_block = v_desc.load([slot_id * BLOCK_SIZE, 0])

        # compute attention scores
        scores = tl.dot(q_block, k_block.T) * (1.0 / (HEAD_DIM**0.5))
        scores = scores.reshape(BLOCK_Q, MAX_GQA, BLOCK_SIZE)

        # kv_len mask
        kv_block_len = tl.minimum(BLOCK_SIZE, kv_len - kv_block_id * BLOCK_SIZE)
        kv_len_mask = (tl.arange(0, BLOCK_SIZE) < kv_block_len)[None, None, :]
        scores = tl.where(kv_len_mask, scores, float("-inf"))

        # causal mask
        k_pos = kv_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, None, :]
        q_pos = q_block_id * BLOCK_Q + (kv_len - q_len) + tl.arange(0, BLOCK_Q)[:, None, None]
        causal_mask = k_pos <= q_pos
        scores = tl.where(causal_mask, scores, float("-inf"))

        # gqa mask
        gqa_mask = (tl.arange(0, MAX_GQA) < gqa_size)[None, :, None]
        scores = tl.where(gqa_mask, scores, float("-inf"))

        # softmax
        tile_max = tl.max(scores, axis=2, keep_dims=True)
        new_m_i = tl.maximum(m_i, tile_max)
        exp_scores = tl.exp(scores - new_m_i)

        alpha = tl.exp(m_i - new_m_i)
        l_i = alpha * l_i + tl.sum(exp_scores, axis=2, keep_dims=True)

        exp_scores = exp_scores.reshape(BLOCK_Q * MAX_GQA, BLOCK_SIZE)
        o_update = tl.dot(exp_scores.to(v_block.dtype), v_block)
        o_update = o_update.reshape(BLOCK_Q, MAX_GQA, HEAD_DIM)
        o_block = alpha * o_block + o_update

        m_i = new_m_i

    # finalize output and store
    o_out = o_block / l_i
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr + q_offset,
        shape=[q_len, gqa_size, HEAD_DIM],
        strides=[q_num_heads * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[BLOCK_Q, MAX_GQA, HEAD_DIM],
    )
    o_desc.store([q_block_id * BLOCK_Q, 0, 0], o_out.to(o_desc.dtype))


def triton_flash_attn_paged(
    q: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    k: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    v: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    o: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    max_seqlen_q: int,
    seqused_kv: torch.Tensor,  # [num_seqs]
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks]
    max_gqa: int = 8,
    block_q: int = 32,
):
    _, q_num_heads, head_dim = q.shape
    block_num, block_size, kv_num_heads, _ = k.shape
    num_seqs, max_num_blocks = block_table.shape

    grid = lambda meta: (
        num_seqs,
        kv_num_heads,
        triton.cdiv(max_seqlen_q, meta["BLOCK_Q"]),
    )

    flash_attn_paged_gqa_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=o,
        cu_seqlens_q_ptr=cu_seqlens_q,
        seqused_kv_ptr=seqused_kv,
        block_table_ptr=block_table,
        block_num=block_num,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        max_num_blocks=max_num_blocks,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        MAX_GQA=max_gqa,
        BLOCK_Q=block_q,
    )


def ref_flash_attn_paged(
    q: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    k: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    v: torch.Tensor,  # [block_num, block_size, kv_num_heads, head_dim]
    o: torch.Tensor,  # [total_q_len, q_num_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    seqused_kv: torch.Tensor,  # [num_seqs]
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks]
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

    dtype = torch.float16

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
    o = torch.empty_like(q)

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

    funcs_to_bench = {
        triton_flash_attn_paged.__name__: lambda: triton_flash_attn_paged(
            q,
            k,
            v,
            o,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_kv,
            block_table,
        ),
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(10, func)
        acc_check(out_ref, o)


if __name__ == "__main__":
    main()
