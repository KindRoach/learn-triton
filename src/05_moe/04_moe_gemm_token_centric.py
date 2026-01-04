import torch
import torch.nn.functional as F

import triton
from triton import language as tl

from .utils import generate_moe_inputs, ref_topk_routing
from ..utils import acc_check, bench_by_secs, get_device


@triton.jit
def moe_gemm_token_centric_kernel(
    out_ptr,  # [T*top_k, N]
    x_ptr,  # [M, K] (M = T for gate_up and M = T*top_k for down)
    weight_ptr,  # [E, K, N]
    expert_ids_ptr,  # [T, top_k]
    E: int,
    M: int,
    N: int,
    K: int,
    top_k: int,
    is_gate_up: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)

    x_id = block_m // top_k if is_gate_up == 1 else block_m
    expert_id = tl.load(expert_ids_ptr + block_m)

    a_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[1, BLOCK_K],
    )

    b_desc = tl.make_tensor_descriptor(
        base=weight_ptr + expert_id * N * K,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    tile_n = block_n * BLOCK_N
    c_tile = tl.zeros([1, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_tile = a_desc.load([x_id, k])
        b_tile = b_desc.load([k, tile_n])
        c_tile += tl.dot(a_tile, b_tile)

    c_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=[M * top_k, N] if is_gate_up == 1 else [M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_N],
    )
    c_desc.store([block_m, tile_n], c_tile.to(c_desc.dtype))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_N": block_mn,
                "BLOCK_K": block_k,
            },
        )
        for block_mn in [16, 32, 64, 128]
        for block_k in [16, 32, 64, 128]
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def moe_gemm_token_centric_autotune_kernel(
    out_ptr,  # [T*top_k, N]
    x_ptr,  # [M, K] (M = T for gate_up and M = T*top_k for down)
    weight_ptr,  # [E, K, N]
    expert_ids_ptr,  # [T, top_k]
    E: int,
    M: int,
    N: int,
    K: int,
    top_k: int,
    is_gate_up: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    moe_gemm_token_centric_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        expert_ids_ptr,
        E,
        M,
        N,
        K,
        top_k,
        is_gate_up,
        BLOCK_K,
        BLOCK_N,
    )


def triton_moe_gemm_token_centric(
    out: torch.Tensor,  # [T*top_k, N]
    x: torch.Tensor,  # [M, K] (M = T for gate_up and M = T*top_k for down)
    weight: torch.Tensor,  # [E, K, N]
    expert_ids: torch.Tensor,  # [T, top_k]
    is_gate_up: bool,  # True for gate_up, False for down
    block_n: int = 32,
    block_k: int = 32,
    auto_tune: bool = False,
):
    M, K = x.shape
    E, _, N = weight.shape
    T, top_k = expert_ids.shape
    grid = lambda meta: (T * top_k, triton.cdiv(N, meta["BLOCK_N"]))

    if auto_tune:
        moe_gemm_token_centric_autotune_kernel[grid](
            out,
            x,
            weight,
            expert_ids,
            E=E,
            M=M,
            N=N,
            K=K,
            top_k=top_k,
            is_gate_up=is_gate_up,
        )
    else:
        moe_gemm_token_centric_kernel[grid](
            out,
            x,
            weight,
            expert_ids,
            E=E,
            M=M,
            N=N,
            K=K,
            top_k=top_k,
            is_gate_up=is_gate_up,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )


def ref_moe_gemm_token_centric(
    out: torch.Tensor,  # [T*top_k, N]
    x: torch.Tensor,  # [M, K] (M = T for gate_up and M = T*top_k for down)
    weight: torch.Tensor,  # [E, K, N]
    expert_ids: torch.Tensor,  # [T, top_k]
    is_gate_up: bool,  # True for gate_up, False for down
):
    _, _, N = weight.shape
    T, top_k = expert_ids.shape
    out = out.view(T * top_k, N)
    for t in range(T):
        selected_weights = weight[expert_ids[t]]  # [top_k, K, N]
        range_idx = torch.arange(t * top_k, (t + 1) * top_k, device=x.device)  # [top_k]
        x_t = x[t][None, :] if is_gate_up else x[range_idx][:, None, :]  # [1, K] for gate_up, [top_k, 1, K] for down
        out_t = x_t @ selected_weights  # [top_k, 1, N]
        out[range_idx] = out_t.squeeze(1)


@torch.inference_mode()
def main():
    T = 4
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

    _, _, N1 = w_gate_up.shape
    _, _, N2 = w_down.shape
    gate_up_out = torch.empty(T, top_k, N1, device=hiddens.device, dtype=hiddens.dtype)
    down_out = torch.empty(T, top_k, N2, device=hiddens.device, dtype=hiddens.dtype)

    # ref gate_up gemm
    ref_gate_up_out = torch.empty_like(gate_up_out)
    ref_moe_gemm_token_centric(ref_gate_up_out, hiddens, w_gate_up, topk_expert_ids, is_gate_up=True)

    # ref down gemm
    ref_down_out = torch.empty_like(down_out)
    down_hiddens = (F.silu(ref_gate_up_out)[:, :, :I] * ref_gate_up_out[:, :, I:]).reshape(T * top_k, I)
    ref_moe_gemm_token_centric(ref_down_out, down_hiddens, w_down, topk_expert_ids, is_gate_up=False)

    # perform benchmark
    sec = 10

    print("============ Gate Up GEMM ============")
    funcs_to_bench = {
        ref_moe_gemm_token_centric.__name__: lambda: ref_moe_gemm_token_centric(
            gate_up_out, hiddens, w_gate_up, topk_expert_ids, is_gate_up=True
        ),
        triton_moe_gemm_token_centric.__name__: lambda: triton_moe_gemm_token_centric(
            gate_up_out, hiddens, w_gate_up, topk_expert_ids, is_gate_up=True
        ),
        triton_moe_gemm_token_centric.__name__
        + "_autotune": lambda: triton_moe_gemm_token_centric(
            gate_up_out, hiddens, w_gate_up, topk_expert_ids, is_gate_up=True, auto_tune=True
        ),
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            sec,
            func,
        )
        acc_check(ref_gate_up_out, gate_up_out)

    print("\n============ Down GEMM ============")
    funcs_to_bench = {
        ref_moe_gemm_token_centric.__name__: lambda: ref_moe_gemm_token_centric(
            down_out, down_hiddens, w_down, topk_expert_ids, is_gate_up=False
        ),
        triton_moe_gemm_token_centric.__name__: lambda: triton_moe_gemm_token_centric(
            down_out, down_hiddens, w_down, topk_expert_ids, is_gate_up=False
        ),
        triton_moe_gemm_token_centric.__name__
        + "_autotune": lambda: triton_moe_gemm_token_centric(
            down_out, down_hiddens, w_down, topk_expert_ids, is_gate_up=False, auto_tune=True
        ),
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            sec,
            func,
        )
        acc_check(ref_down_out, down_out)


if __name__ == "__main__":
    main()
