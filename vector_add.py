import triton
import triton.language as tl
import torch


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_ele, BLOCK_SIZE: tl.constexpr):
    # calculate offset and mask
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele

    # read
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # add
    out = x + y

    # write
    tl.store(out_ptr + offsets, out, mask=mask)


def main():
    # Problem size
    N = 1 << 20  # 1 million elements
    BLOCK_SIZE = 1024

    # Initialize input tensors
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, z, N, BLOCK_SIZE)

    # Validate correctness
    expected = x + y
    if torch.allclose(z, expected, atol=1e-5):
        print("Vector addition successful and correct.")
    else:
        print("Mismatch found in vector addition result.")


if __name__ == "__main__":
    main()
