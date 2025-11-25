import torch


def acc_check(ground_truth: torch.Tensor, predict: torch.Tensor) -> None:
    if not ground_truth.shape == predict.shape:
        print(f"Shape mismatch: {ground_truth.shape} vs {predict.shape}")
        return

    relative_error = torch.abs(ground_truth - predict) / (torch.abs(ground_truth) + 1e-8)
    absolute_error = torch.abs(ground_truth - predict)

    print(f"Absolute error: max={absolute_error.max().item()}, mean={absolute_error.mean().item()}")
    print(f"Relative error: max={relative_error.max().item()}, mean={relative_error.mean().item()}")
