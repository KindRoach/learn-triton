import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, get_device


@triton.jit
def copy_1D_kernel(
    input_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    N: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < N
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


def copy_1D():
    print(f"{'='*20} 1D copy {'='*20}")
    N = (100 * 1024 * 1024) - 3
    BLOCK = 1024

    device = get_device()
    dtype = torch.float32
    input_tensor = torch.randn(N, dtype=dtype, device=device)
    output_tensor = torch.empty_like(input_tensor)

    grid = (triton.cdiv(N, BLOCK),)

    bench_by_secs(
        10,
        lambda: copy_1D_kernel[grid](
            input_tensor,
            output_tensor,
            N,
            BLOCK,
        ),
        mem_access_bytes=input_tensor.element_size() * input_tensor.nelement() * 2,  # 1 read + 1 write
    )

    acc_check(input_tensor, output_tensor)


@triton.jit
def copy_2D_kernel(
    input_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    N: int,
    M: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)

    # The common pattern of creating a 2D grid of offsets
    offsets = offsets_m[:, None] * M + offsets_n[None, :]
    mask = (offsets_m[:, None] < N) & (offsets_n[None, :] < M)

    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


def copy_2D():
    print(f"{'='*20} 2D copy {'='*20}")

    M = 20 * 1024 + 1
    N = 5 * 1024 - 1

    BLOCK_M = 32
    BLOCK_N = 32

    device = get_device()
    dtype = torch.float32
    input_tensor = torch.randn(N, M, dtype=dtype, device=device)
    output_tensor = torch.empty_like(input_tensor)

    grid = (
        triton.cdiv(N, BLOCK_M),
        triton.cdiv(M, BLOCK_N),
    )

    bench_by_secs(
        10,
        lambda: copy_2D_kernel[grid](
            input_tensor,
            output_tensor,
            N,
            M,
            BLOCK_M,
            BLOCK_N,
        ),
        mem_access_bytes=input_tensor.element_size() * input_tensor.nelement() * 2,  # 1 read + 1 write
    )

    acc_check(input_tensor, output_tensor)


if __name__ == "__main__":
    copy_1D()
    copy_2D()
