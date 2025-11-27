import torch
import triton
import triton.language as tl

from utils import acc_check, enable_tma_allocator


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


def transpose_2D():
    M = 20 * 1024
    N = 5 * 1024
    BLOCK_M = 64
    BLOCK_N = 64

    device = "cuda"
    dtype = torch.float32

    input_tensor = torch.randn((M, N), dtype=dtype, device=device)
    output_tensor = torch.empty((N, M), dtype=dtype, device=device)

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

    # Verification
    expected = input_tensor.transpose(0, 1)
    acc_check(expected, output_tensor)


if __name__ == "__main__":
    enable_tma_allocator()
    transpose_2D()
