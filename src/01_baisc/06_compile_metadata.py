from curses import meta
import torch
import triton
import triton.language as tl

from ..utils import get_device


@triton.jit
def add_one_kernel(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
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


def print_metadata(func):
    compiled = func()
    metadata = compiled.metadata
    print("Kernel Metadata:")
    for key, value in metadata._asdict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    device = get_device()

    N = 1024
    BLOCK_SIZE = 256

    x = torch.randn(N, device=device)
    print_metadata(lambda: add_one_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x, N, BLOCK_SIZE))
