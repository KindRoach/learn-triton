import torch


def acc_check(
    ground_truth: torch.Tensor,
    predict: torch.Tensor,
) -> None:
    if not ground_truth.shape == predict.shape:
        print(f"Shape mismatch: {ground_truth.shape} vs {predict.shape}")
        return

    absolute_error = torch.abs(ground_truth - predict)
    relative_error = absolute_error / (torch.abs(ground_truth) + 1e-8)

    print(f"Absolute error: max = {absolute_error.max().item():.3e}, mean = {absolute_error.mean().item():.3e}")
    print(f"Relative error: max = {relative_error.max().item():.3e}, mean = {relative_error.mean().item():.3e}")


def is_tma_supported() -> bool:
    try:
        cuda_ok = torch.cuda.is_available()
        sm90_ok = torch.cuda.get_device_capability()[0] >= 9
        cuda_12_4_ok = torch.version.cuda >= "12.4"
        return cuda_ok and sm90_ok and cuda_12_4_ok
    except:
        return False


def enable_tma_allocator() -> None:
    import triton

    from typing import Optional

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.float32)

    triton.set_allocator(alloc_fn)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.xpu.is_available():
        return torch.device("xpu")

    raise RuntimeError("No supported device found (CUDA or XPU).")
