import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        base=in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=[N, M],
        strides=[M, 1],
        block_shape=[BLOCK_N, BLOCK_M],
    )

    tile_m = tl.program_id(0) * BLOCK_M
    tile_n = tl.program_id(1) * BLOCK_N
    in_tile = in_desc.load([tile_m, tile_n])
    out_desc.store([tile_n, tile_m], tl.trans(in_tile))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m in [32, 64, 128]
        for block_n in [32, 64, 128]
        for num_stages in [1, 2, 3, 4]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["M", "N"],
    cache_results=True,
)
@triton.jit
def transpose_autotune_kernel(
    in_ptr,
    out_ptr,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    transpose_kernel(
        in_ptr,
        out_ptr,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def transpose(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    M, N = input_tensor.shape
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    transpose_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
    )


def transpose_autotune(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    M, N = input_tensor.shape

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    transpose_autotune_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
    )


def transpose_2D():
    M = 20 * 1024
    N = 5 * 1024

    device = get_device()
    dtype = torch.float32

    input_tensor = torch.randn((M, N), dtype=dtype, device=device)
    output_tensor = torch.empty((N, M), dtype=dtype, device=device)

    funcs_to_bench = {transpose.__name__: transpose, transpose_autotune.__name__: transpose_autotune}

    mem_access_bytes = input_tensor.element_size() * input_tensor.nelement() * 2

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(input_tensor, output_tensor),
            mem_access_bytes=mem_access_bytes,  # 1 read + 1 write
        )

        # Verification
        expected = input_tensor.transpose(0, 1)
        acc_check(expected, output_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    transpose_2D()
