import torch
import triton
import triton.language as tl

from ..utils import get_device


@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: 1024 if args["N"] >= 1024 * 1024 else 256,
        "num_warps": lambda args: 2 if args["N"] >= 1024 * 1024 else 1,
    }
)
@triton.jit
def add_one_kernel(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    if tl.program_id(0) == 0:
        print(f"Using BLOCK_SIZE = {BLOCK_SIZE}")

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
