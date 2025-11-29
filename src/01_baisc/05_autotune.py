import os
import torch
import triton
import triton.language as tl

from ..utils import get_device

# Show how autotuning works, not necessarily for normal use cases
os.environ["TRITON_CACHE_DISABLE"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": bs},
            num_warps=w,
        )
        for bs in [128, 256, 1024]
        for w in [4, 8, 16]
    ],
    key=["N"],
)
@triton.jit
def add_one_kernel(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )

    x = desc.load([tl.program_id(0) * BLOCK_SIZE])
    y = x + 1
    desc.store([tl.program_id(0) * BLOCK_SIZE], y)


def add_one(N: int):
    print(f"{'='*20} add_one with N={N} {'='*20}")
    device = get_device()
    dtype = torch.float32
    x = torch.randn(N, dtype=dtype, device=device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    add_one_kernel[grid](x, N)
    torch.accelerator.synchronize()


if __name__ == "__main__":
    add_one(1024)
    add_one(1024 * 1024)
