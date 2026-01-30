import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


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
        c_tile += tl.dot(a_tile, b_tile).to(c_tile.dtype)

    c_desc.store([tile_m, tile_n], c_tile)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": block_mn,
                "BLOCK_N": block_mn,
                "BLOCK_K": block_k,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_mn in [16, 32, 64, 128]
        for block_k in [16, 32, 64, 128]
        for num_stages in [1, 2, 3, 4]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def matrix_multiply_autotune_kernel(
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
    matrix_multiply_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def matrix_multiply(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
) -> None:

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 16

    M, K = a_tensor.shape
    K_b, N = b_tensor.shape
    assert K == K_b, "Input shapes must align for matrix multiplication"

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matrix_multiply_kernel[grid](
        a_tensor,
        b_tensor,
        c_tensor,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def matrix_multiply_autotune(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
) -> None:
    M, K = a_tensor.shape
    K_b, N = b_tensor.shape
    assert K == K_b, "Input shapes must align for matrix multiplication"

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matrix_multiply_autotune_kernel[grid](
        a_tensor,
        b_tensor,
        c_tensor,
        M,
        N,
        K,
    )


def main():
    M = 4096
    N = 4096
    K = 4096

    device = get_device()
    dtype = torch.float16

    a_tensor = torch.randn((M, K), dtype=dtype, device=device)
    b_tensor = torch.randn((K, N), dtype=dtype, device=device)
    c_tensor = torch.empty((M, N), dtype=dtype, device=device)

    funcs_to_bench = {
        "matrix_multiply": matrix_multiply,
        "matrix_multiply_autotune": matrix_multiply_autotune,
    }

    mem_access_bytes = (
        a_tensor.element_size() * a_tensor.nelement()
        + b_tensor.element_size() * b_tensor.nelement()
        + c_tensor.element_size() * c_tensor.nelement()
    )
    total_flops = 2 * M * N * K

    expected = torch.matmul(a_tensor, b_tensor)

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(a_tensor, b_tensor, c_tensor),
            mem_access_bytes=mem_access_bytes,
            total_flops=total_flops,
        )
        acc_check(expected, c_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    main()
