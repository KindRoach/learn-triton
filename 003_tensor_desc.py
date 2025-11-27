import torch
import triton
import triton.language as tl

from utils import acc_check, enable_tma_allocator, get_device


@triton.jit
def copy_1D_kernel(
    input_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    N: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    input_desc = tl.make_tensor_descriptor(
        base=input_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK],
    )

    output_desc = tl.make_tensor_descriptor(
        base=output_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK],
    )

    x = input_desc.load([pid * BLOCK])
    output_desc.store([pid * BLOCK], x)


def copy_1D():
    print(f"{'='*20} 1D copy {'='*20}")
    N = (100 * 1024 * 1024) - 3
    BLOCK = 1024
    num_blocks = (N + BLOCK - 1) // BLOCK

    device = get_device()
    dtype = torch.float32
    input_tensor = torch.randn(N, dtype=dtype, device=device)
    output_tensor = torch.empty_like(input_tensor)

    copy_1D_kernel[(num_blocks,)](
        input_tensor,
        output_tensor,
        N,
        BLOCK,
    )

    acc_check(input_tensor, output_tensor)


@triton.jit
def copy_2D_kernel(
    input_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    input_desc = tl.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    output_desc = tl.make_tensor_descriptor(
        base=output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    x = input_desc.load([offset_m, offset_n])
    output_desc.store([offset_m, offset_n], x)


def copy_2D():
    print(f"{'='*20} 2D copy {'='*20}")

    # N is ok not be multiple of 16/sizeof(dtype)
    # Here are an example rows not divisible by BLOCK_M
    M = 20 * 1024 + 1

    # triton tensor descs require stride 16 bytes alignment,
    # so N should be multiple of 16/sizeof(dtype)
    N = 5 * 1024

    BLOCK_M = 32
    BLOCK_N = 32

    device = get_device()
    dtype = torch.float32
    input_tensor = torch.randn(M, N, dtype=dtype, device=device)
    output_tensor = torch.empty_like(input_tensor)

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    copy_2D_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
    )

    acc_check(input_tensor, output_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    copy_1D()
    copy_2D()
