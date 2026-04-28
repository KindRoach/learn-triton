import torch
import triton
from triton import language as tl


from .utils import quantize_per_tensor, ref_per_tensor_quant_gemm
from ..utils import acc_check, bench_by_secs, get_device


@triton.jit
def per_tensor_quant_gemm_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    scale_x_ptr,
    scale_w_ptr,
    M: int,
    N: int,
    K: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Tensor descriptors for input and output tensors
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    w_desc = tl.make_tensor_descriptor(
        base=w_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    # Get current block position
    tile_m = tl.program_id(0) * BLOCK_M
    tile_n = tl.program_id(1) * BLOCK_N
    
    # Initialize accumulator in int32
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    
    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load quantized tiles
        x_tile = x_desc.load([tile_m, k])  # (BLOCK_M, BLOCK_K) in int8
        w_tile = w_desc.load([tile_n, k])  # (BLOCK_N, BLOCK_K) in int8
        
        # Matrix multiply: x @ w.T
        c_tile += tl.dot(x_tile, tl.trans(w_tile)).to(tl.int32)

    # Dequantize c to fp32
    scale = tl.load(scale_x_ptr) * tl.load(scale_w_ptr)
    c_tile = c_tile.to(tl.float32) * scale

    # Store result as float32
    o_desc.store([tile_m, tile_n], c_tile)


def per_tensor_quant_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    o: torch.Tensor,
    w_scale: float,
    x_scale: float,
) -> torch.Tensor:

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, "Input shapes must align for matrix multiplication"

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    per_tensor_quant_gemm_kernel[grid](
        x,
        w,
        o,
        x_scale,
        w_scale,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return o


def main():

    device = get_device()
    dtype = torch.float32

    M = 4096
    N = 4096
    K = 4096

    W = torch.randn(N, K, dtype=dtype, device=device)
    X = torch.randn(M, K, dtype=dtype, device=device)

    # Quantize inputs to int8
    X_q, x_scale = quantize_per_tensor(X)
    W_q, w_scale = quantize_per_tensor(W)

    # Compute ref output
    y_ref = ref_per_tensor_quant_gemm(
        X_q.cpu(),
        W_q.cpu(),
        w_scale.cpu(),
        x_scale.cpu(),
    ).to(device)

    # output Tensor
    out = torch.empty_like(y_ref)

    funcs_to_bench = {
        "per_tensor_quant_gemm": per_tensor_quant_gemm,
    }

    mem_access_bytes = (
        X_q.element_size() * X_q.nelement()
        + W_q.element_size() * W_q.nelement()
        + out.element_size() * out.nelement()
        + w_scale.element_size() * w_scale.nelement()
        + x_scale.element_size() * x_scale.nelement()
    )

    total_flops = 2 * M * N * K

    for name, func in funcs_to_bench.items():
        print(f"\nBenchmarking {name}...")
        bench_by_secs(
            10,
            lambda: func(X_q, W_q, out, w_scale, x_scale),
            mem_access_bytes=mem_access_bytes,
            total_flops=total_flops,
        )
        acc_check(y_ref, out)


if __name__ == "__main__":
    main()
