import torch

import triton
from triton import language as tl

from .utils import generate_moe_inputs, ref_sort_token_ids_by_expert, ref_topk_routing, ref_moe_scatter
from ..utils import acc_check, bench_by_secs, get_device


def ref_moe_gemm_implicit_group(
    x: torch.Tensor,  # [T, K] (regrouped by expert)
    expert_ids: torch.Tensor,  # [T, top_k]
    weight: torch.Tensor,  # [E, K, N]
    out: torch.Tensor,  # [T, top_k, N]
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

    ref_moe_gemm_implicit_group(hiddens, topk_expert_ids, w_gate_up, ref_out)

    # perform benchmark
    funcs_to_bench = {
        "ref_moe_gemm_implicit_group": lambda: ref_moe_gemm_implicit_group(
            hiddens, topk_expert_ids, w_gate_up, out_tensor
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
