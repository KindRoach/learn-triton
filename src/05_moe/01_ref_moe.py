import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import acc_check, bench_by_secs, get_device


class SwiGLU(nn.Module):
    """
    Qwen-style FFN:
        y = (x W1) * silu(x W2) -> W3
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class MoELayer(nn.Module):
    """
    Single-device inference-only MoE layer.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        assert top_k <= num_experts
        self.d_model = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([SwiGLU(hidden_size, moe_intermediate_size) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mathematical reference implementation of MoE.

        x: [batch, seq, d_model]
        """
        B, S, D = x.shape
        T = B * S

        # ---- Flatten tokens ----
        tokens = x.view(T, D)  # {x_t}

        # ---- Router: p_t = softmax(W_r x_t) ----
        router_logits = self.router(tokens)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)

        # ---- Top-K selection ----
        topk_weights, topk_experts = torch.topk(router_probs, self.top_k, dim=-1)  # both [T, K]

        # ---- MoE output initialization ----
        outputs = torch.zeros_like(tokens)  # y_t

        # ---- Explicit mathematical summation ----
        for t in range(T):
            x_t = tokens[t]  # x_t
            y_t = torch.zeros_like(x_t)

            for k in range(self.top_k):
                e = topk_experts[t, k].item()  # expert index
                w = topk_weights[t, k]  # scalar gate weight

                y_t = y_t + w * self.experts[e](x_t.unsqueeze(0)).squeeze(0)

            outputs[t] = y_t

        return outputs.view(B, S, D)

    def forward_expert_batching(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, d_model]
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [T, D], T = B*S

        # ---- Routing ----
        logits = self.router(x_flat)  # [T, E]
        probs = F.softmax(logits, dim=-1)  # [T, E]
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        # shapes: [T, K], [T, K]

        # ---- Dispatch ----
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]  # [T]
            weights = topk_probs[:, k]  # [T]

            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if not mask.any():
                    continue

                tokens = x_flat[mask]  # [N, D]
                expert_out = self.experts[expert_id](tokens)
                output[mask] += expert_out * weights[mask].unsqueeze(-1)

        return output.view(B, S, D)

@torch.inference_mode()
def main():
    batch_size = 1
    token_num = 8192
    hidden_size = 2048
    moe_intermediate_size = 768
    num_experts = 128
    top_k = 8

    device = get_device()
    moe_layer = MoELayer(hidden_size, moe_intermediate_size, num_experts, top_k).to(device).eval()
    x = torch.randn(batch_size, token_num, hidden_size, device=device)

    # accuracy check
    out_ref = moe_layer.forward(x)
    out_opt = moe_layer.forward_expert_batching(x)
    acc_check(out_ref, out_opt)

    # perform benchmark
    funcs_to_bench = {
        moe_layer.forward.__name__: moe_layer.forward,
        moe_layer.forward_expert_batching.__name__: moe_layer.forward_expert_batching,
    }

    sec = 10
    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            sec,
            lambda: func(x),
        )


if __name__ == "__main__":
    main()
