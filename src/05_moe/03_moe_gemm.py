import torch

import triton
from triton import language as tl

from .utils import generate_moe_inputs, ref_topk_routing, ref_moe_scatter
from ..utils import acc_check, bench_by_secs, get_device


@triton.jit
def moe_gemm_explicit_group_kernel(
    x_ptr,  # [M, K] (regrouped by expert)
    weight_ptr,  # [E, K, N]
    out_ptr,  # [M, N]
    expert_token_num_ptr,  # [E]
    E: int,
    M: int,
    N: int,
    K: int,
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

    a_desc = tl.make_tensor_descriptor(
        base=x_ptr + row_offset * K,
        shape=[row_count, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    b_desc = tl.make_tensor_descriptor(
        base=weight_ptr + expert_id * N * K,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    tile_n = block_n_id * BLOCK_N
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_tile = a_desc.load([0, k])
        b_tile = b_desc.load([k, tile_n])
        c_tile += tl.dot(a_tile, b_tile)

    c_desc = tl.make_tensor_descriptor(
        base=out_ptr + row_offset * N,
        shape=[row_count, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    c_desc.store([0, tile_n], c_tile.to(c_desc.dtype))


def triton_moe_gemm_explicit_group(
    x: torch.Tensor,  # [T * top_k, H] (regrouped by expert)
    expert_token_num: torch.Tensor,  # [E]
    weight: torch.Tensor,  # [E, H, I*2]
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
) -> torch.Tensor:  # [T * top_k, H]

    M, K = x.shape
    E, _, N = weight.shape

    # count number of blocks per expert and record expert ids
    num_block_m = ((expert_token_num + block_m - 1) // block_m).sum().item()

    # moe gemm [M, K] x [E, K, N] -> [M, N]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = (num_block_m, triton.cdiv(N, block_n))
    moe_gemm_explicit_group_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        expert_token_num_ptr=expert_token_num,
        E=E,
        M=M,
        N=N,
        K=K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return out


def ref_moe_gemm_explicit_group(
    x: torch.Tensor,  # [T * top_k, H] (regrouped by expert)
    expert_offsets: torch.Tensor,  # [E + 1]
    weight: torch.Tensor,  # [E, H, I*2]
) -> torch.Tensor:  # [T * top_k, H]

    M, _ = x.shape
    E, _, N = weight.shape

    # moe gemm [M, K] x [E, K, N] -> [M, N]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    for e in range(E):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if start == end:
            continue

        out[start:end] = x[start:end] @ weight[e]

    return out


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
    hiddens, logits, w_gate_up, w_down = generate_moe_inputs(
        num_tokens=T,
        num_experts=E,
        hidden_dim=H,
        internal_dim=I,
        scale=0.1,
        dtype=dtype,
        device=device,
    )

    topk_expert_ids, _ = ref_topk_routing(logits, top_k=top_k)

    reordered_hiddens, _, expert_token_num, expert_token_offsets = ref_moe_scatter(
        hiddens=hiddens,
        topk_expert_ids=topk_expert_ids,
        num_experts=E,
    )

    # accuracy check
    ref_out = ref_moe_gemm_explicit_group(reordered_hiddens, expert_token_offsets, w_gate_up)
    out = triton_moe_gemm_explicit_group(reordered_hiddens, expert_token_num, w_gate_up)
    acc_check(ref_out, out)

    # perform benchmark
    funcs_to_bench = {
        ref_moe_gemm_explicit_group.__name__: lambda: ref_moe_gemm_explicit_group(
            reordered_hiddens, expert_token_offsets, w_gate_up
        ),
        triton_moe_gemm_explicit_group.__name__: lambda: triton_moe_gemm_explicit_group(
            reordered_hiddens, expert_token_num, w_gate_up
        ),
    }

    sec = 10
    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            sec,
            func,
        )


if __name__ == "__main__":
    main()
