import torch
import triton
import triton.language as tl

from utils import acc_check, enable_tma_allocator, get_device


@triton.jit
def vector_add_kernel(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    N: int,
    BLOCK: tl.constexpr,
):
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK],
    )
    y_desc = tl.make_tensor_descriptor(
        base=y_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK],
    )

    x_tile = x_desc.load([tl.program_id(0) * BLOCK])
    y_tile = y_desc.load([tl.program_id(0) * BLOCK])
    out_tile = x_tile + y_tile
    out_desc.store([tl.program_id(0) * BLOCK], out_tile)


def main():
    N = (100 * 1024 * 1024) - 3
    BLOCK = 1024

    # Initialize input tensors
    device = get_device()
    dtype = torch.float32
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.randn(N, device=device, dtype=dtype)
    z = torch.empty_like(x)

    grid = (triton.cdiv(N, BLOCK),)
    vector_add_kernel[grid](x, y, z, N, tl.constexpr(BLOCK))

    # Validate correctness
    expected = x + y

    acc_check(expected, z)


if __name__ == "__main__":
    enable_tma_allocator()
    main()
