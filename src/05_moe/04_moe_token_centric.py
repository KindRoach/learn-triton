import torch
import torch.nn.functional as F

from .utils import generate_moe_inputs, ref_topk_routing
from ..utils import acc_check, bench_by_secs, get_device


def ref_moe_token_centric(
    hiddens: torch.Tensor,  # [T, H]
    out: torch.Tensor,  # [T, H]
    expert_ids: torch.Tensor,  # [T, K]
    expert_weights: torch.Tensor,  # [T, K]
    w_gate_up: torch.Tensor,  # [E, H, 2*I]
    w_down: torch.Tensor,  # [2*I, H]
):
    T, H = hiddens.shape
    _, K = expert_ids.shape
    E, _, I2 = w_gate_up.shape
    I = I2 // 2

    for t in range(T):
        # gate up: [K, 1, H] @ [K, H, 2*I] = [K, 1, 2*I]
        w_gate_up_selected = w_gate_up[expert_ids[t]]
        gate_up = hiddens[t, None, :] @ w_gate_up_selected

        # silu activation
        gate_up = F.silu(gate_up[:, :, :I]) * gate_up[:, :, I:]  # [K, 1, I]

        # down: [K, 1, I] @ [K, I, H] = [K, 1, H]
        w_down_selected = w_down[expert_ids[t]]  # [K, I, H]
        down = gate_up @ w_down_selected  # [K, 1, H]

        # gather: [1, K] @ [K, H] = [1, H]
        out[t, :] = expert_weights[None, t] @ down.squeeze(1)


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

    topk_expert_ids, topk_expert_weights = ref_topk_routing(logits, top_k=top_k)

    out_tensor = torch.empty(T, H, device=hiddens.device, dtype=hiddens.dtype)
    ref_out = torch.empty_like(out_tensor)

    ref_moe_token_centric(
        hiddens,
        ref_out,
        topk_expert_ids,
        topk_expert_weights,
        w_gate_up,
        w_down,
    )

    # perform benchmark
    funcs_to_bench = {
        ref_moe_token_centric.__name__: lambda: ref_moe_token_centric(
            hiddens,
            out_tensor,
            topk_expert_ids,
            topk_expert_weights,
            w_gate_up,
            w_down,
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
