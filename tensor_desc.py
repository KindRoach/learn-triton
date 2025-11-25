import torch
import triton
import triton.language as tl

from utils import acc_check


@triton.jit
def batched_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    batch_idx = tl.program_id(0)

    # Base pointers for this batch
    A_base = A_ptr + batch_idx * BLOCK_M * BLOCK_K
    B_base = B_ptr + batch_idx * BLOCK_K * BLOCK_N
    C_base = C_ptr + batch_idx * BLOCK_M * BLOCK_N

    # Create tensor descriptors
    A_desc = tl.make_tensor_descriptor(
        base=A_base,
        shape=[BLOCK_M, BLOCK_K],
        strides=[BLOCK_K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    B_desc = tl.make_tensor_descriptor(
        base=B_base,
        shape=[BLOCK_K, BLOCK_N],
        strides=[BLOCK_N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    C_desc = tl.make_tensor_descriptor(
        base=C_base,
        shape=[BLOCK_M, BLOCK_N],
        strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Load blocks
    A_block = tl.load_tensor_descriptor(A_desc, [0, 0])
    B_block = tl.load_tensor_descriptor(B_desc, [0, 0])

    # Compute GEMM
    C_block = tl.dot(A_block, B_block)

    # Store result
    tl.store_tensor_descriptor(C_desc, [0, 0], C_block)


def main():
    # Example usage
    BATCH, BLOCK_M, BLOCK_N, BLOCK_K = 4, 16, 16, 16

    device = "cuda"
    dtype = torch.float16
    A = torch.randn(BATCH, BLOCK_M, BLOCK_K, device=device, dtype=dtype)
    B = torch.randn(BATCH, BLOCK_K, BLOCK_N, device=device, dtype=dtype)
    C = torch.empty(BATCH, BLOCK_M, BLOCK_N, device=device, dtype=dtype)

    grid = (BATCH,)
    batched_gemm_kernel[grid](A, B, C, BLOCK_M, BLOCK_N, BLOCK_K)

    expected = torch.bmm(A, B)
    acc_check(expected, C)


if __name__ == "__main__":
    main()
