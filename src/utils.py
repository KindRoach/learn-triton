import sys
import time
import torch
from typing import Optional

import triton


def bench_by_secs(
    secs: float,
    func,
    mem_access_bytes: Optional[float] = None,
    total_flops: Optional[float] = None,
) -> None:
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
        f" Avg duration: {duration_str}."
    )

    # Calculate and display memory bandwidth and FLOPs if provided
    if mem_access_bytes is not None:
        mem_bandwidth_bytes_per_sec = (mem_access_bytes * count) / secs

        # Choose appropriate unit for memory bandwidth
        if mem_bandwidth_bytes_per_sec >= 1e12:
            bandwidth_str = f"{mem_bandwidth_bytes_per_sec / 1e12:.2f} TB/s"
        elif mem_bandwidth_bytes_per_sec >= 1e9:
            bandwidth_str = f"{mem_bandwidth_bytes_per_sec / 1e9:.2f} GB/s"
        elif mem_bandwidth_bytes_per_sec >= 1e6:
            bandwidth_str = f"{mem_bandwidth_bytes_per_sec / 1e6:.2f} MB/s"
        elif mem_bandwidth_bytes_per_sec >= 1e3:
            bandwidth_str = f"{mem_bandwidth_bytes_per_sec / 1e3:.2f} KB/s"
        else:
            bandwidth_str = f"{mem_bandwidth_bytes_per_sec:.2f} B/s"

        print(f" Memory bandwidth: {bandwidth_str}")

    if total_flops is not None:
        flops_per_sec = (total_flops * count) / secs

        # Choose appropriate unit for FLOPs
        if flops_per_sec >= 1e12:
            flops_str = f"{flops_per_sec / 1e12:.2f} TFLOP/s"
        elif flops_per_sec >= 1e9:
            flops_str = f"{flops_per_sec / 1e9:.2f} GFLOP/s"
        elif flops_per_sec >= 1e6:
            flops_str = f"{flops_per_sec / 1e6:.2f} MFLOP/s"
        else:
            flops_str = f"{flops_per_sec:.2f} FLOP/s"

        print(f" Compute throughput: {flops_str}")


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
    if not is_tma_supported():
        return

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
