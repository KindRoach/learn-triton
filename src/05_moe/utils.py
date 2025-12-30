import torch

from ..utils import get_device


def generate_moe_inputs(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    internal_dim: int,
    dtype: torch.dtype,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # uniformly random logits with some expert popularity bias
    expert_popularity = torch.rand(num_experts)
    noise = torch.randn(num_tokens, num_experts)
    logits = noise + expert_popularity.log()

    hiddens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype) * scale
    w_gate = torch.randn(num_experts, hidden_dim, internal_dim, device=device, dtype=dtype) * scale
    w_up = torch.randn(num_experts, hidden_dim, internal_dim, device=device, dtype=dtype) * scale
    w_down = torch.randn(num_experts, internal_dim, hidden_dim, device=device, dtype=dtype) * scale

    return hiddens, logits, w_gate, w_up, w_down


def ref_topk_routing(
    logits: torch.Tensor,  # [T, E]
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_logits, topk_expert_ids = logits.topk(top_k, dim=-1)  # both [T, K]
    topk_weights = torch.softmax(topk_logits, dim=-1)  # [T, K]
    return topk_expert_ids, topk_weights


def ref_moe_scatter(
    hiddens: torch.Tensor,  # [T, H]
    topk_expert_ids: torch.Tensor,  # [T, K]
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    T, H = hiddens.shape
    _, K = topk_expert_ids.shape

    if topk_expert_ids.ndim != 2 or topk_expert_ids.shape[0] != T:
        raise ValueError(f"topk_expert_ids must be rank-2 [T, K] with T={T}, got shape={tuple(topk_expert_ids.shape)}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")

    device = hiddens.device

    # Flatten routing assignments.
    # - flatten_expert_ids: [T*K]
    # - original_token_ids: [T*K], maps each assignment back to its original token id
    flatten_expert_ids = topk_expert_ids.reshape(-1).to(dtype=torch.int64)
    original_token_ids = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(K)

    # Stable sort by expert id so tokens for each expert are contiguous.
    _, sort_idx = torch.sort(flatten_expert_ids, stable=True)
    reordered_token_ids = original_token_ids[sort_idx]
    reordered_hiddens = hiddens[reordered_token_ids]

    # Invert the permutation so we can map each original (t, k) assignment
    # to its position in the reordered buffer.
    # - sort_idx: [T*K], maps reordered position -> original flattened position
    # - index_map: [T, K], maps original (t, k) -> reordered position
    inv_sort_idx = torch.empty_like(sort_idx, device=device)
    inv_sort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=device, dtype=sort_idx.dtype)
    reordered_index = inv_sort_idx.reshape(T, K)

    # Prefix-sum offsets into the reordered buffer per expert.
    counts = torch.bincount(flatten_expert_ids, minlength=num_experts).to(dtype=torch.int64)
    expert_offsets = torch.zeros((num_experts + 1,), device=device, dtype=torch.int64)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)

    return reordered_hiddens, reordered_index, expert_offsets


def test_ref_moe_scatter():
    num_tokens = 2048
    num_experts = 128
    hidden_dim = 1024
    internal_dim = 512
    top_k = 8

    hiddens, logits, _, _, _ = generate_moe_inputs(
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        internal_dim=internal_dim,
        dtype=torch.float16,
        scale=0.1,
        device=get_device(),
    )

    topk_expert_ids, _ = ref_topk_routing(logits, top_k=top_k)

    reordered_hiddens, reordered_index, expert_offsets = ref_moe_scatter(
        hiddens=hiddens,
        topk_expert_ids=topk_expert_ids,
        num_experts=num_experts,
    )

    original_hiddens = reordered_hiddens[reordered_index]

    for t in range(num_tokens):
        for k in range(top_k):
            assert torch.allclose(original_hiddens[t, k], hiddens[t]), f"mismatch at token {t}, top-k {k}"

    print(f"{test_ref_moe_scatter.__name__}: OK")


def moe_align_block_size(
    topk_ids: torch.Tensor,  # [T, K]
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad and sort token assignments to experts for block matrix operations.
    Returns sorted_token_ids, expert_ids, and number of tokens after padding.

    Example:
    Given topk_ids = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3]],
    block_size = 4, and num_experts = 4:

    - First, get flatten token_ids and expert_ids as
        flatten_expert_ids = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,  2,  1,  2,  3]
        flatten_token_ids  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        original_token_ids = [0 ,0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,  4,  4]
        Note: original_token_ids = flatten_token_ids // K

    - Then sorting by expert index, we obtain
        sorted_flatten_expert_ids = [0, 0, 0, 0, 1, 1, 1, 1,  1,  2, 2, 2, 2,  2,  3]
        sorted_flatten_token_ids =  [0, 3, 6, 9, 1, 4, 7, 10, 12, 2, 5, 8, 11, 13, 14]

    - Now pad each exprt tokens to block size:
        padded_expert_ids = [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, <pad>, <pad>, <pad>,
            2, 2, 2, 2,
            2, <pad>, <pad>, <pad>,
            3, <pad>, <pad>, <pad>
        ],
        padded_token_ids = [
            0,  3,  6,  9,
            1,  4,  7,  10,
            12, <pad>, <pad>, <pad>,
            2,  5,  8,  11,
            13, <pad>, <pad>, <pad>,
            14, <pad>, <pad>, <pad>
        ]

    - Finally, for each block, replace <pad> with non-existing token id (i.e., T * K = 15)
        block_expert_ids = [0, 1, 1, 2, 2, 3]
        block_token_ids = [
            0,  3,  6,  9,
            1,  4,  7,  10,
            12, 15, 15, 15,
            2,  5,  8,  11,
            13, 15, 15, 15,
            14, 15, 15, 15
        ]

        Note: We create block_expert_ids and block_token_ids with max possible size
        then store the actual padded results. The actual number of tokens after padding
        is also returned as num_tokens_post_padded.
    """

    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2 [T, K], got shape={tuple(topk_ids.shape)}")

    device = topk_ids.device
    T, K = topk_ids.shape
    total_assignments = int(T * K)

    # sort flatten expert ids in [T * K]
    flatten_expert_ids = topk_ids.reshape(-1)
    sorted_flatten_expert_ids, sorted_flatten_token_ids = torch.sort(flatten_expert_ids, stable=True)

    # create block token and expert ids with max possible padding size
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    padding_token_idx = total_assignments  # non-existing token id
    padding_expert_idx = num_experts  # non-existing expert id

    block_token_ids = torch.full((max_num_tokens_padded,), padding_token_idx, dtype=torch.int64, device=device)
    block_expert_ids = torch.full((max_num_blocks,), padding_expert_idx, dtype=torch.int64, device=device)

    # actually pad to block size
    num_blocks = 0
    num_tokens_post_padded = 0
    last_expert_id = -1
    for expert_id, token_id in zip(sorted_flatten_expert_ids, sorted_flatten_token_ids):
        if expert_id.item() != last_expert_id or num_tokens_post_padded % block_size == 0:
            # new block
            block_expert_ids[num_blocks] = expert_id
            num_blocks += 1

            if last_expert_id != -1:
                # new expert
                while num_tokens_post_padded % block_size != 0:
                    # pad previous expert to block size
                    num_tokens_post_padded += 1

            last_expert_id = expert_id.item()

        block_token_ids[num_tokens_post_padded] = token_id
        num_tokens_post_padded += 1

    # pad the last block
    while num_tokens_post_padded % block_size != 0:
        num_tokens_post_padded += 1

    assert num_tokens_post_padded == num_blocks * block_size

    return block_token_ids, block_expert_ids, torch.tensor(num_tokens_post_padded, device=device)


def test_moe_align_block_size():
    topk_ids = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3]],
        dtype=torch.int64,
    )
    block_size = 4
    num_experts = 4

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )

    gt_sort_token_ids = [0, 3, 6, 9, 1, 4, 7, 10, 12, 15, 15, 15, 2, 5, 8, 11, 13, 15, 15, 15, 14, 15, 15, 15]
    assert num_tokens_post_padded.item() == 24
    assert sorted_token_ids[:num_tokens_post_padded].tolist() == gt_sort_token_ids
    assert expert_ids[: num_tokens_post_padded // block_size].tolist() == [0, 1, 1, 2, 2, 3]

    print(f"{moe_align_block_size.__name__}: OK")


if __name__ == "__main__":
    test_moe_align_block_size()
    test_ref_moe_scatter()
