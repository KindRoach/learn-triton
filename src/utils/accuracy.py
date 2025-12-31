import torch


def acc_check(
    ground_truth: torch.Tensor,
    predict: torch.Tensor,
) -> None:
    if not ground_truth.shape == predict.shape:
        print(f"Shape mismatch: {ground_truth.shape} vs {predict.shape}")
        return

    absolute_error = torch.abs(ground_truth - predict)
    relative_error = absolute_error / (torch.abs(ground_truth) + 1e-6)

    print(
        f"Absolute error: max = {absolute_error.max().item():.3e}, mean = {absolute_error.mean().item():.3e}"
    )
    print(
        f"Relative error: max = {relative_error.max().item():.3e}, mean = {relative_error.mean().item():.3e}"
    )
