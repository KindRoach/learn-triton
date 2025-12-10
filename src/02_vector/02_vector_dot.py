import torch
import triton
import triton.language as tl

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


@triton.jit
def vector_dot_kernel(
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

    x_tile = x_desc.load([tl.program_id(0) * BLOCK])
    y_tile = y_desc.load([tl.program_id(0) * BLOCK])
    block_sum = tl.sum(x_tile * y_tile)
    tl.atomic_add(out_ptr, block_sum)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK": block_size,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size in [256, 512, 1024]
        for num_stages in [1, 2, 3, 4]
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["N"],
    cache_results=True,
)
@triton.jit
def vector_dot_autotune_kernel(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    N: int,
    BLOCK: tl.constexpr,
):
    vector_dot_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        N,
        BLOCK=BLOCK,
    )


def vector_dot(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
) -> None:
    BLOCK = 1024
    N = x.shape[0]
    grid = (triton.cdiv(N, BLOCK),)
    out.fill_(0)
    vector_dot_kernel[grid](x, y, out, N, BLOCK)  # pyright: ignore[reportGeneralTypeIssues]


def vector_dot_autotune(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
) -> None:
    N = x.shape[0]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    out.fill_(0)
    vector_dot_autotune_kernel[grid](x, y, out, N)  # pyright: ignore[reportGeneralTypeIssues]


def main():
    N = (100 * 1024 * 1024) - 3

    # Initialize input tensors
    device = get_device()
    dtype = torch.float32
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.randn(N, device=device, dtype=dtype)
    z = torch.empty(1, device=device, dtype=dtype)

    funcs_to_bench = {
        vector_dot.__name__: vector_dot,
        vector_dot_autotune.__name__: vector_dot_autotune,
    }

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(x, y, z),
            mem_access_bytes=x.element_size() * x.nelement() * 2 + z.element_size() * z.nelement(),  # 2 reads + 1 write
            total_flops=x.nelement() * 2,  # 1 multiplication + 1 addition per element
        )

        # Validate correctness
        expected = torch.dot(x, y).unsqueeze(0)
        acc_check(expected, z)


if __name__ == "__main__":
    enable_tma_allocator()
    main()
