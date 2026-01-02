import torch
import torch.nn.functional as F

from .utils import generate_moe_inputs, ref_topk_routing
from ..utils import acc_check, bench_by_secs, get_device


def ref_moe_single_token(
    hiddens: torch.Tensor,  # [1, H]
    out: torch.Tensor,  # [1, H]
    expert_ids: torch.Tensor,  # [1, K]
    expert_weights: torch.Tensor,  # [1, K]
    w_gate_up: torch.Tensor,  # [E, H, 2*I]
    w_down: torch.Tensor,  # [2*I, H]
):
    T, H = hiddens.shape
    _, K = expert_ids.shape
    E, _, I2 = w_gate_up.shape
    I = I2 // 2

    assert T == 1, "This reference implementation only supports single token (T=1)."

    # gate up
    w_gate_up_selected = w_gate_up[expert_ids[0]]  # [K, H, 2*I]
    gate_up = hiddens @ w_gate_up_selected  # [K, 1, 2*I]

    # silu activation
    gate_up = F.silu(gate_up[:, :, :I]) * gate_up[:, :, I:]  # [K, 1, I]

    # down
    w_down_selected = w_down[expert_ids[0]]  # [2*I, H]
    down = gate_up @ w_down_selected  # [K, 1, H]

    # gather
    out[:] = expert_weights @ down.squeeze(1)  # [1, H]


@torch.inference_mode()
def main():
    T = 1
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

    out_tensor = torch.empty(1, H, device=hiddens.device, dtype=hiddens.dtype)
    ref_out = torch.empty_like(out_tensor)

    ref_moe_single_token(
        hiddens,
        ref_out,
        topk_expert_ids,
        topk_expert_weights,
        w_gate_up,
        w_down,
    )

    # perform benchmark
    funcs_to_bench = {
        ref_moe_single_token.__name__: lambda: ref_moe_single_token(
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
