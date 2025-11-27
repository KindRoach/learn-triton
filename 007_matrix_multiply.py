import torch
import triton
import triton.language as tl

from utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


@triton.jit
def matrix_multiply_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: int,
    N: int,
    K: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    tile_m = tl.program_id(0) * BLOCK_M
    tile_n = tl.program_id(1) * BLOCK_N
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=c_desc.dtype)

    for k in range(0, K, BLOCK_K):
        a_tile = a_desc.load([tile_m, k])
        b_tile = b_desc.load([k, tile_n])
        c_tile += tl.dot(a_tile, b_tile)

    c_desc.store([tile_m, tile_n], c_tile)


def matrix_multiply():
    M = 2 * 1024
    N = 512
    K = 1024

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 16

    device = get_device()
    dtype = torch.float32

    a_tensor = torch.randn((M, K), dtype=dtype, device=device)
    b_tensor = torch.randn((K, N), dtype=dtype, device=device)
    c_tensor = torch.empty((M, N), dtype=dtype, device=device)

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    bench_by_secs(
        10,
        lambda: matrix_multiply_kernel[grid](
            a_tensor,
            b_tensor,
            c_tensor,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        ),
        mem_access_bytes=(
            a_tensor.element_size() * a_tensor.nelement()
            + b_tensor.element_size() * b_tensor.nelement()
            + c_tensor.element_size() * c_tensor.nelement()
        ),
        total_flops=2 * M * N * K,
    )

    expected = torch.matmul(a_tensor, b_tensor)
    acc_check(expected, c_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    matrix_multiply()
