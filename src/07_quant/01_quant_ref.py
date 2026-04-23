import torch

from ..utils import acc_check


# ============================================================================
# Quantization Functions (FP32 -> INT8)
# ============================================================================

def quantize_per_channel(T: torch.Tensor) -> tuple:
    """
    Quantize tensor from FP32 to INT8 with per-channel scaling.
    Scales are calculated as max(abs(T)) / 128 per channel (INT8 range: [-128, 127]).
    For weights (N, K): one scale per output channel (per row).
    
    Args:
        T: Input tensor (N, K) in FP32
    
    Returns:
        T_q: Quantized tensor (N, K) in INT8
        scales: Per-channel scales (N,)
    """
    # Calculate per-channel scales: max(abs()) per row / 128 (INT8 range)
    scales = torch.max(torch.abs(T), dim=1)[0] / 128.0
    T_q = torch.round(T / scales.view(-1, 1)).to(torch.int8)
    return T_q, scales


def quantize_per_tensor(T: torch.Tensor) -> tuple:
    """
    Quantize tensor from FP32 to INT8 with global scaling.
    Scale is calculated as max(abs(T)) / 128 (INT8 range: [-128, 127]).
    
    Args:
        T: Input tensor (M, K) in FP32
    
    Returns:
        T_q: Quantized tensor (M, K) in INT8
        scale: Global scale (scalar)
    """
    # Calculate global scale: max(abs()) / 128 (INT8 range)
    scale = torch.max(torch.abs(T)).item() / 128.0
    T_q = torch.round(T / scale).to(torch.int8)
    return T_q, scale


# ============================================================================
# GEMM Functions
# ============================================================================

def fp32_gemm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Baseline FP32 GEMM: Y = X @ W.T"""
    return X @ W.t()


def per_channel_quant_gemm(X_q: torch.Tensor, W_q: torch.Tensor, w_scales: torch.Tensor, x_scale: float) -> torch.Tensor:
    """
    Per-channel quantized GEMM: Y = X_q @ W_q.T with int8 inputs.

    Args:
        X_q: Quantized input activations (M, K) in INT8
        W_q: Quantized weight matrix (N, K) in INT8
        w_scales: Per-channel weight scales (N,)
        x_scale: Activation scale (scalar)
    """
    # Integer matmul (int32 compute)
    Y_int = X_q.to(torch.int32) @ W_q.t().to(torch.int32)

    # Dequantize
    scale_factor = x_scale * w_scales
    Y = Y_int.to(torch.float32) * scale_factor

    return Y


def per_tensor_quant_gemm(X_q: torch.Tensor, W_q: torch.Tensor, w_scale: float, x_scale: float) -> torch.Tensor:
    """
    Per-tensor quantized GEMM: Y = X_q @ W_q.T with int8 inputs.

    Args:
        X_q: Quantized input activations (M, K) in INT8
        W_q: Quantized weight matrix (N, K) in INT8
        w_scale: Global weight scale (scalar)
        x_scale: Global activation scale (scalar)
    """
    # Integer matmul (int32 compute)
    Y_int = X_q.to(torch.int32) @ W_q.t().to(torch.int32)

    # Dequantize
    scale_factor = x_scale * w_scale
    Y = Y_int.to(torch.float32) * scale_factor

    return Y


def per_channel_quant_gemm_scalar(X_q: torch.Tensor, W_q: torch.Tensor, w_scales: torch.Tensor, x_scale: float) -> torch.Tensor:
    """
    Per-channel quantized GEMM using scalar operations and loops.
    Shows how each output element Y[m, n] is calculated with int8 inputs.
    
    Args:
        X_q: Quantized input activations (M, K) in INT8
        W_q: Quantized weight matrix (N, K) in INT8
        w_scales: Per-channel weight scales (N,)
        x_scale: Activation scale (scalar)
    
    Returns:
        Y: Output (M, N) with dequantization applied
    """
    M, K = X_q.shape
    N = W_q.shape[0]
    
    # Convert to int32 for accumulation
    X_q_int32 = X_q.to(torch.int32)
    W_q_int32 = W_q.to(torch.int32)
    
    # Initialize output tensor
    Y = torch.zeros((M, N), dtype=torch.float32)
    
    # Compute each output element Y[m, n] using loops
    for m in range(M):
        for n in range(N):
            # Compute dot product: Y[m, n] = sum_k (X_q[m, k] * W_q[n, k])
            acc_int32 = 0
            
            for k in range(K):
                # Load quantized scalar values
                x_q_val = X_q_int32[m, k].item()
                w_q_val = W_q_int32[n, k].item()
                
                # Accumulate in int32
                acc_int32 += x_q_val * w_q_val
            
            # Dequantize: Y[m, n] = acc_int32 * x_scale * w_scales[n]
            Y[m, n] = float(acc_int32) * x_scale * w_scales[n].item()
    
    return Y


def main():
    # --- Create test data ---
    M, N, K = 2, 4, 3
    W = torch.randn(N, K)  # weight matrix (N output channels, K input channels)
    X = torch.randn(M, K)  # input activations (M samples, K input channels)

    # --- Compute reference FP32 output ---
    Y_fp32 = fp32_gemm(X, W)

    # --- Test per-tensor quantized GEMM ---
    print("\n--- Per-Tensor Quantized GEMM ---")

    # Quantize inputs (scale is calculated as max(abs(x))/128)
    X_q_pt, x_scale = quantize_per_tensor(X)
    W_q_pt, w_scale = quantize_per_tensor(W)

    # Compute quantized output
    Y_quant_pt = per_tensor_quant_gemm(X_q_pt, W_q_pt, w_scale, x_scale)
    acc_check(Y_fp32, Y_quant_pt)

    # --- Test per-channel quantized GEMM (vectorized) ---
    print("\n--- Per-Channel Quantized GEMM (vectorized) ---")

    # Quantize inputs (scales are calculated as max(abs()) per channel / 128)
    X_q_pc, x_scale = quantize_per_tensor(X)
    W_q_pc, w_scales = quantize_per_channel(W)
    
    # Compute quantized output
    Y_quant_pc = per_channel_quant_gemm(X_q_pc, W_q_pc, w_scales, x_scale)
    acc_check(Y_fp32, Y_quant_pc)

    # --- Test per-channel quantized GEMM (scalar/loop) ---
    print("\n--- Per-Channel Quantized GEMM (scalar loops) ---")
    Y_quant_pc_scalar = per_channel_quant_gemm_scalar(X_q_pc, W_q_pc, w_scales, x_scale)
    acc_check(Y_fp32, Y_quant_pc_scalar)


if __name__ == "__main__":
    main()
