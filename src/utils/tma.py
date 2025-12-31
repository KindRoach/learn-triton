from typing import Optional

import torch
import triton


def is_tma_supported() -> bool:
    try:
        if not torch.cuda.is_available():
            return False

        if torch.cuda.get_device_capability()[0] < 9:
            return False

        if torch.version.cuda is None or torch.version.cuda < "12.4":
            return False

        return True
    except Exception:
        return False


def enable_tma_allocator() -> None:
    if not is_tma_supported():
        return

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.float32)

    triton.set_allocator(alloc_fn)
