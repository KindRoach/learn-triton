import torch

import triton
from triton import language as tl

from .utils import generate_moe_inputs, ref_sort_token_ids_by_expert, ref_topk_routing, ref_moe_scatter
from ..utils import acc_check, bench_by_secs, get_device


@triton.jit
def moe_gemm_implicit_group_kernel(
    x_ptr,  # [T, K]
    weight_ptr,  # [E, K, N]
    out_ptr,  # [T, top_k, N]
    expert_token_num_ptr,  # [E]
    sorted_token_ids_ptr,  # [M = T * top_k, N]
    E: int,
    T: int,
    M: int,
    N: int,
    K: int,
    top_k: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m_id = tl.program_id(0)
    block_n_id = tl.program_id(1)

    # find expert id and row range for this block_m_id
    expert_id = 0
    row_offset = 0
    row_count = 0
    row_sum = 0
    current_block_m_id = 0
    for e in range(E):
        rows = tl.load(expert_token_num_ptr + e)
        num_block_m = (rows + BLOCK_M - 1) // BLOCK_M
        if current_block_m_id <= block_m_id and block_m_id < current_block_m_id + num_block_m:
            expert_id = e
            row_offset = row_sum + (block_m_id - current_block_m_id) * BLOCK_M
            row_count = min(BLOCK_M, rows - (block_m_id - current_block_m_id) * BLOCK_M)
        current_block_m_id += num_block_m
        row_sum += rows

    # load token ids
    token_mask = tl.arange(0, BLOCK_M) < row_count
    token_ids = tl.load(  # [BLOCK_M,]
        sorted_token_ids_ptr + row_offset + tl.arange(0, BLOCK_M),
        mask=token_mask,
        other=0,
    )

    # compute offsets for m and n
    expert_weight_ptr = weight_ptr + expert_id * N * K
    offset_m = token_ids // top_k  # [BLOCK_M,]
    offset_n = block_n_id * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N,]

    # accumulate c_tile
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offset_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K,]

        # load a_tile
        a_tile = tl.load(  # [BLOCK_M, BLOCK_K]
            x_ptr + offset_m[:, None] * K + offset_k[None, :],
            mask=token_mask[:, None] & (offset_k[None, :] < K),
            other=0.0,
        )

        # load b_tile
        b_tile = tl.load(  # [BLOCK_K, BLOCK_N]
            expert_weight_ptr + offset_k[:, None] * N + offset_n[None, :],
            mask=(offset_k[:, None] < K) & (offset_n[None, :] < N),
            other=0.0,
        )

        # dot product and accumulate
        c_tile += tl.dot(a_tile, b_tile)

    # store c_tile
    tl.store(  # [BLOCK_M, BLOCK_N]
        out_ptr + token_ids[:, None] * N + offset_n[None, :],
        c_tile.to(tl.float16),
        mask=token_mask[:, None] & (offset_n[None, :] < N),
    )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": block_mn,
                "BLOCK_N": block_mn,
                "BLOCK_K": block_k,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_mn in [16, 32, 64, 128]
        for block_k in [16, 32, 64, 128]
        for num_stages in [1, 2, 3, 4]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def moe_gemm_implicit_group_autotune_kernel(
    x_ptr,  # [T, K]
    weight_ptr,  # [E, K, N]
    out_ptr,  # [T, top_k, N]
    expert_token_num_ptr,  # [E]
    sorted_token_ids_ptr,  # [M = T * top_k, N]
    E: int,
    T: int,
    M: int,
    N: int,
    K: int,
    top_k: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    moe_gemm_implicit_group_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        expert_token_num_ptr,
        sorted_token_ids_ptr,
        E,
        T,
        M,
        N,
        K,
        top_k,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def triton_moe_gemm_implicit_group(
    x: torch.Tensor,  # [T, K]
    weight: torch.Tensor,  # [E, K, N]
    out: torch.Tensor,  # [T, top_k, N]
    expert_token_num: torch.Tensor,  # [E]
    sorted_token_ids: torch.Tensor,  # [M = T * top_k, N]
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
    auto_tune: bool = False,
):

    T, K = x.shape
    E, _, N = weight.shape
    _, top_k, _ = out.shape

    def grid(meta):
        # count number of blocks per expert and record expert ids
        num_block_m = ((expert_token_num + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]).sum().item()
        grid = (num_block_m, triton.cdiv(N, meta["BLOCK_N"]))
        return grid

    if auto_tune:
        moe_gemm_implicit_group_autotune_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            out_ptr=out,
            expert_token_num_ptr=expert_token_num,
            sorted_token_ids_ptr=sorted_token_ids,
            E=E,
            T=T,
            M=T * top_k,
            N=N,
            K=K,
            top_k=top_k,
        )
    else:
        moe_gemm_implicit_group_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            out_ptr=out,
            expert_token_num_ptr=expert_token_num,
            sorted_token_ids_ptr=sorted_token_ids,
            E=E,
            T=T,
            M=T * top_k,
            N=N,
            K=K,
            top_k=top_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )


def ref_moe_gemm_implicit_group(
    x: torch.Tensor,  # [T, K]
    weight: torch.Tensor,  # [E, K, N]
    out: torch.Tensor,  # [T, top_k, N]
    expert_ids: torch.Tensor,  # [T, top_k]
):
    E, _, _ = weight.shape

    for e in range(E):
        mask = expert_ids == e  # [T, top_k]
        if not mask.any():
            continue

        token_idx = mask.nonzero(as_tuple=False)  # [N_e, 2]
        t_idx = token_idx[:, 0]  # [N_e]
        k_idx = token_idx[:, 1]  # [N_e]

        out[t_idx, k_idx] = x[t_idx] @ weight[e]  # [N_e, H]


@torch.inference_mode()
def main():
    T = 8192
    H = 2048
    I = 768
    E = 128
    top_k = 8

    device = get_device()
    dtype = torch.float16

    # random number of tokens per expert with a logit-normal distribution
    torch.manual_seed(0)
    hiddens, logits, w_gate_up, _ = generate_moe_inputs(
        num_tokens=T,
        num_experts=E,
        hidden_dim=H,
        internal_dim=I,
        scale=0.1,
        dtype=dtype,
        device=device,
    )

    topk_expert_ids, _ = ref_topk_routing(logits, top_k=top_k)
    sorted_token_ids = ref_sort_token_ids_by_expert(topk_expert_ids)
    _, _, expert_token_num, _ = ref_moe_scatter(
        hiddens=hiddens,
        topk_expert_ids=topk_expert_ids,
        num_experts=E,
    )

    _, _, N = w_gate_up.shape
    # moe gemm [T, K] x [E, K, N] -> [T, top_k, N]
    out_tensor = torch.empty(T, top_k, N, device=hiddens.device, dtype=hiddens.dtype)
    ref_out = torch.empty_like(out_tensor)

    ref_moe_gemm_implicit_group(hiddens, w_gate_up, ref_out, topk_expert_ids)

    # perform benchmark
    funcs_to_bench = {
        ref_moe_gemm_implicit_group.__name__: lambda: ref_moe_gemm_implicit_group(
            hiddens, w_gate_up, out_tensor, topk_expert_ids
        ),
        triton_moe_gemm_implicit_group.__name__: lambda: triton_moe_gemm_implicit_group(
            hiddens, w_gate_up, out_tensor, expert_token_num, sorted_token_ids, block_m=64, block_n=64, block_k=32
        ),
        triton_moe_gemm_implicit_group.__name__
        + "_autotune": lambda: triton_moe_gemm_implicit_group(
            hiddens, w_gate_up, out_tensor, expert_token_num, sorted_token_ids, auto_tune=True
        ),
    }

    sec = 10
    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            sec,
            func,
        )
        acc_check(ref_out, out_tensor)


if __name__ == "__main__":
    main()
