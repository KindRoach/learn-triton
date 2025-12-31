import os

# Print autotuning logs
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

from .accuracy import acc_check
from .bench import bench_by_secs
from .device import get_device
from .profiler import TorchProfiler
from .tma import enable_tma_allocator, is_tma_supported

__all__ = [
    "bench_by_secs",
    "acc_check",
    "get_device",
    "TorchProfiler",
    "is_tma_supported",
    "enable_tma_allocator",
]
