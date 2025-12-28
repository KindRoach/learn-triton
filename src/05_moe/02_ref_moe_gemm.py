import torch
import torch.nn.functional as F

from ..utils import acc_check, bench_by_secs, get_device


def ref_batched_gemm(
    x: torch.Tensor,  # [T, H] (regrouped by expert)
    expert_offsets: torch.Tensor,  # [E + 1]
    w_gate: torch.Tensor,  # [E, H, I]
    w_up: torch.Tensor,  # [E, H, I]
    w_down: torch.Tensor,  # [E, I, H]
) -> torch.Tensor:
    """Operator-level reference aligned with vLLM Expert FFN.

    This assumes `x` is already regrouped so that tokens for expert 0 come first,
    then expert 1, ..., expert E-1.
    `expert_offsets[e]:expert_offsets[e+1]` defines the token range for expert e.
    """

    T, H = x.shape
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


def ref_expert_ffn(
    x: torch.Tensor,  # [T, H] (any order)
    expert_ids: torch.Tensor,  # [T]
    w_gate: torch.Tensor,  # [E, H, I]
    w_up: torch.Tensor,  # [E, H, I]
    w_down: torch.Tensor,  # [E, I, H]
) -> torch.Tensor:
    """Reference implementation for arbitrary token order.

    Mirrors the "expert batching" style from 01_ref_moe.py (mask by expert, run dense GEMMs).
    """

    if expert_ids.ndim != 1 or expert_ids.numel() != x.shape[0]:
        raise ValueError("expert_ids must be 1D and match x.shape[0]")

    T, H = x.shape
    E = w_gate.shape[0]
    out = torch.empty_like(x)

    for e in range(E):
        mask = expert_ids == e
        if not mask.any():
            continue

        x_e = x[mask]
        gate = x_e @ w_gate[e]
        up = x_e @ w_up[e]
        act = F.silu(gate) * up
        out[mask] = act @ w_down[e]

    return out


@torch.inference_mode()
def main():
    # Keep sizes modest by default so this runs on most GPUs.
    T = 4096
    H = 2048
    I = 768
    E = 128

    device = get_device()
    dtype = torch.float16

    # random number of tokens per expert with a logit-normal distribution
    torch.manual_seed(0)
    logits = torch.randn((E,), device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)
    num_tokens = torch.distributions.Multinomial(total_count=T, probs=probs).sample().to(torch.int64)
    num_tokens[-1] += T - int(num_tokens.sum().item())

    expert_offsets = torch.zeros((E + 1,), device=device, dtype=torch.int64)
    expert_offsets[1:] = torch.cumsum(num_tokens, dim=0)

    expert_ids = torch.repeat_interleave(
        torch.arange(E, device=device, dtype=torch.int64),
        num_tokens,
    )

    # Use a smaller initialization scale so FP16 matmuls don't overflow to inf.
    init_scale = 0.1
    x = torch.randn((T, H), device=device, dtype=dtype) * init_scale
    w_gate = torch.randn((E, H, I), device=device, dtype=dtype) * init_scale
    w_up = torch.randn((E, H, I), device=device, dtype=dtype) * init_scale
    w_down = torch.randn((E, I, H), device=device, dtype=dtype) * init_scale

    # accuracy check
    out_ref = ref_expert_ffn(x, expert_ids, w_gate, w_up, w_down)
    out_batched = ref_batched_gemm(x, expert_offsets, w_gate, w_up, w_down)
    acc_check(out_ref, out_batched)

    # perform benchmark
    funcs_to_bench = {
        ref_expert_ffn.__name__: lambda: ref_expert_ffn(x, expert_ids, w_gate, w_up, w_down),
        ref_batched_gemm.__name__: lambda: ref_batched_gemm(x, expert_offsets, w_gate, w_up, w_down),
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
