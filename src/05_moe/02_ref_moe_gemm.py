import torch
import torch.nn.functional as F

from .utils import generate_moe_inputs, ref_topk_routing, ref_moe_scatter
from ..utils import acc_check, bench_by_secs, get_device


def ref_moe_gemm_explicit_group(
    x: torch.Tensor,  # [T * K, H] (regrouped by expert)
    expert_offsets: torch.Tensor,  # [E + 1]
    w_gate: torch.Tensor,  # [E, H, I]
    w_up: torch.Tensor,  # [E, H, I]
    w_down: torch.Tensor,  # [E, I, H]
) -> torch.Tensor:  # [T * K, H]

    E = w_gate.shape[0]
    out = torch.empty_like(x)

    for e in range(E):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if start == end:
            continue

        x_e = x[start:end]  # [N_e, H]

        gate = x_e @ w_gate[e]  # [N_e, I]
        up = x_e @ w_up[e]  # [N_e, I]
        act = F.silu(gate) * up  # [N_e, I]
        y_e = act @ w_down[e]  # [N_e, H]

        out[start:end] = y_e

    return out


def ref_moe_gemm_implicit_group(
    x: torch.Tensor,  # [T, H]
    expert_ids: torch.Tensor,  # [T, K] (any order)
    w_gate: torch.Tensor,  # [E, H, I]
    w_up: torch.Tensor,  # [E, H, I]
    w_down: torch.Tensor,  # [E, I, H]
) -> torch.Tensor:  # [T, K, H]
    T, H = x.shape
    _, K = expert_ids.shape

    out = torch.empty(T, K, H, device=x.device, dtype=x.dtype)

    for e in range(w_gate.shape[0]):
        mask = expert_ids == e  # [T, K]
        if not mask.any():
            continue

        token_indices = mask.nonzero(as_tuple=False)  # [N_e, 2]
        t_indices = token_indices[:, 0]  # [N_e]
        k_indices = token_indices[:, 1]  # [N_e]

        x_e = x[t_indices]  # [N_e, H]

        gate = x_e @ w_gate[e]  # [N_e, I]
        up = x_e @ w_up[e]  # [N_e, I]
        act = F.silu(gate) * up  # [N_e, I]
        y_e = act @ w_down[e]  # [N_e, H]

        out[t_indices, k_indices] = y_e
    
    return out


@torch.inference_mode()
def main():
    # Keep sizes modest by default so this runs on most GPUs.
    T = 1024
    H = 2048
    I = 768
    E = 128
    top_k = 8

    device = get_device()
    dtype = torch.float16

    # random number of tokens per expert with a logit-normal distribution
    torch.manual_seed(0)
    hiddens, logits, w_gate, w_up, w_down = generate_moe_inputs(
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
    out_implicit = ref_moe_gemm_implicit_group(hiddens, topk_expert_ids.squeeze(1), w_gate, w_up, w_down)
    out_explicit = ref_moe_gemm_explicit_group(reordered_hiddens, expert_offsets, w_gate, w_up, w_down)
    out_explicit = out_explicit[reordered_index]
    acc_check(out_implicit, out_explicit)

    # perform benchmark
    funcs_to_bench = {
        ref_moe_gemm_implicit_group.__name__: lambda: ref_moe_gemm_implicit_group(
            hiddens, topk_expert_ids.squeeze(1), w_gate, w_up, w_down
        ),
        ref_moe_gemm_explicit_group.__name__: lambda: ref_moe_gemm_explicit_group(
            reordered_hiddens, expert_offsets, w_gate, w_up, w_down
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
