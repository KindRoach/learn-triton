import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import acc_check, bench_by_secs, enable_tma_allocator, get_device


def conv2d_with_builtin(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Reference implementation of 2D convolution using PyTorch's built-in Conv2D.
    This is not optimized for performance and serves as a correctness reference.

    Args:
        input: [N, C_in, H_in, W_in]
        weight: [C_out, C_in, K_h, K_w]
        bias: [C_out]
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
    Returns:
        output: [N, C_out, H_out, W_out]
    """

    return F.conv2d(input, weight, bias, stride=stride, padding=padding)


def conv2d_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
):
    """
    Reference implementation of 2D convolution using explicit loops.
    This is not optimized for performance and serves as a correctness reference.

    Args:
        input: [N, C_in, H_in, W_in]
        weight: [C_out, C_in, K_h, K_w]
        bias: [C_out]
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
    Returns:
        output: [N, C_out, H_out, W_out]
    """
    N, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weight.shape

    # Compute output dimensions
    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1

    # Pad the input
    if padding > 0:
        input_padded = F.pad(input, (padding, padding, padding, padding))
    else:
        input_padded = input

    # Perform convolution using explicit loops
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + K_h
                    w_end = w_start + K_w

                    # Extract the relevant patch from the input
                    input_patch = input_padded[n, :, h_start:h_end, w_start:w_end]

                    # Perform element-wise multiplication and sum
                    output[n, c_out, h, w] = torch.sum(input_patch * weight[c_out]) + bias[c_out]

    return output


def main():
    # Example usage
    N, C_in, H_in, W_in = 2, 3, 192, 256
    C_out, K_h, K_w = 4, 5, 5
    stride = 1
    padding = 1

    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1

    device = get_device()
    dtype = torch.float16

    input = torch.randn(N, C_in, H_in, W_in, device=device, dtype=dtype)
    weight = torch.randn(C_out, C_in, K_h, K_w, device=device, dtype=dtype)
    bias = torch.randn(C_out, device=device, dtype=dtype)
    output = torch.empty(N, C_out, H_out, W_out, device=device, dtype=dtype)

    expected = conv2d_with_builtin(input, weight, bias, stride=stride, padding=padding)

    funcs_to_check = {
        "conv2d_reference": conv2d_reference,
    }

    mem_access_bytes = input.element_size() * (input.numel() + weight.numel() + bias.numel() + output.numel())
    total_flops = 2 * N * C_out * H_out * W_out * C_in * K_h * K_w

    for name, func in funcs_to_check.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(input, weight, bias, output, stride, padding),
            mem_access_bytes=mem_access_bytes,
            total_flops=total_flops,
        )
        acc_check(expected, output)


if __name__ == "__main__":
    main()
