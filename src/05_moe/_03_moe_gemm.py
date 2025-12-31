import torch
import torch.nn.functional as F

import triton
from triton import language as tl

from .utils import generate_moe_inputs, ref_topk_routing, ref_moe_scatter
from ..utils import acc_check, bench_by_secs, get_device
from ._02_ref_moe_gemm import ref_moe_gemm_explicit_group


@triton.jit
def moe_gemm_explicit_group_kernel(
    x_ptr,  # [M, K] (regrouped by expert)
    weight_ptr,  # [E, K, N]
    out_ptr,  # [M, N]
    expert_offsets_ptr,  # [E + 1]
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
    row_start = 0
    num_rows = 0
    current_block_m_id = 0
    local_block_m_id = 0
    for e in range(E):
        start = tl.load(expert_offsets_ptr + e)
        end = tl.load(expert_offsets_ptr + e + 1)
        rows = end - start
        num_block_m = (rows + BLOCK_M - 1) // BLOCK_M
        if current_block_m_id <= block_m_id and block_m_id < current_block_m_id + num_block_m:
            expert_id = e
            row_start = start
            num_rows = rows
            local_block_m_id = block_m_id - current_block_m_id

        current_block_m_id += num_block_m

    a_desc = tl.make_tensor_descriptor(
        base=x_ptr + row_start * K,
        shape=[num_rows, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    b_desc = tl.make_tensor_descriptor(
        base=weight_ptr + expert_id * N * K,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    tile_m = local_block_m_id * BLOCK_M
    tile_n = block_n_id * BLOCK_N
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_tile = a_desc.load([tile_m, k])
        b_tile = b_desc.load([k, tile_n])
        c_tile += tl.dot(a_tile, b_tile)

    c_desc = tl.make_tensor_descriptor(
        base=out_ptr + row_start * N,
        shape=[num_rows, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    c_desc.store([tile_m, tile_n], c_tile.to(c_desc.dtype))


def triton_moe_gemm_explicit_group(
    x: torch.Tensor,  # [T * top_k, H] (regrouped by expert)
    expert_offsets: torch.Tensor,  # [E + 1]
    w_gate_up: torch.Tensor,  # [E, H, I*2]
    w_down: torch.Tensor,  # [E, I, H]
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
) -> torch.Tensor:  # [T * top_k, H]

    M, K1 = x.shape
    E, _, N1 = w_gate_up.shape
    _, K2, N2 = w_down.shape

    assert N1 // 2 == K2

    # count number of blocks per expert and record expert ids
    num_block_m = 0
    for e in range(E):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        num_rows = end - start
        num_block_m += (num_rows + block_m - 1) // block_m

    # gaet_up gemm [M, K1] x [E, K1, N1] -> [M, N1]
    gate_up = torch.empty(M, N1, device=x.device, dtype=x.dtype)
    grid = (num_block_m, triton.cdiv(N1, block_n))
    moe_gemm_explicit_group_kernel[grid](
        x_ptr=x,
        weight_ptr=w_gate_up,
        out_ptr=gate_up,
        expert_offsets_ptr=expert_offsets,
        E=E,
        M=M,
        N=N1,
        K=K1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    # apply silu activation and element-wise multiplication
    I = N1 // 2
    gate_up = F.silu(gate_up[:, :I]) * gate_up[:, I:]

    # down gemm [M, I] x [E, I, N2] -> [M, N2]
    out = torch.empty(M, N2, device=x.device, dtype=x.dtype)
    grid = (num_block_m, triton.cdiv(N2, block_n))
    moe_gemm_explicit_group_kernel[grid](
        x_ptr=gate_up,
        weight_ptr=w_down,
        out_ptr=out,
        expert_offsets_ptr=expert_offsets,
        E=E,
        M=M,
        N=N2,
        K=I,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return out


@torch.inference_mode()
def main():
    T = 1024
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

    reordered_hiddens, reordered_index, expert_offsets = ref_moe_scatter(
        hiddens=hiddens,
        topk_expert_ids=topk_expert_ids,
        num_experts=E,
    )

    # accuracy check
    ref_out = ref_moe_gemm_explicit_group(reordered_hiddens, expert_offsets, w_gate_up, w_down)
    out = triton_moe_gemm_explicit_group(reordered_hiddens, expert_offsets, w_gate_up, w_down)
    acc_check(ref_out, out)

    # perform benchmark
    funcs_to_bench = {
        ref_moe_gemm_explicit_group.__name__: lambda: ref_moe_gemm_explicit_group(
            reordered_hiddens, expert_offsets, w_gate_up, w_down
        ),
        triton_moe_gemm_explicit_group.__name__: lambda: triton_moe_gemm_explicit_group(
            reordered_hiddens, expert_offsets, w_gate_up, w_down
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
