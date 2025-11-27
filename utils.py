import sys
import time
import torch


def bench_by_secs(secs: float, func) -> None:
    if secs <= 0:
        print("Run func once as bench secs <= 0", file=sys.stderr)
        func()
        return

    # warm-up
    start = time.perf_counter()
    warm_up_secs = 0.1 * secs
    while time.perf_counter() - start < warm_up_secs:
        func()
        torch.accelerator.synchronize()

    # benchmark
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < secs:
        func()
        torch.accelerator.synchronize()
        count += 1

    end = time.perf_counter()
    secs = end - start

    avg_duration = secs / count

    # Choose appropriate time unit based on duration
    if avg_duration >= 1.0:
        duration_str = f"{avg_duration:.3f} s"
    elif avg_duration >= 0.001:
        duration_str = f"{avg_duration * 1000:.3f} ms"
    else:
        duration_str = f"{avg_duration * 1_000_000:.3f} µs"

    print(
        f"Executed {count} iterations in {secs:.2f} seconds."
        f" Throughput: {count/secs:.2f} iters/sec."
        f" Avg duration: {duration_str}/iter."
    )


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
        if not torch.cuda.is_available():
            return False

        if torch.cuda.get_device_capability()[0] < 9:
            return False
            
        if torch.version.cuda is None or torch.version.cuda < "12.4":
            return False
        
        return True
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
