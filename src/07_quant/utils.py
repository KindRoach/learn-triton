import torch


def quantize_per_channel(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def quantize_per_tensor(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    scale = torch.max(torch.abs(T)) / 128.0
    T_q = torch.round(T / scale).to(torch.int8)
    return T_q, scale


def ref_per_channel_quant_gemm(
    X_q: torch.Tensor, W_q: torch.Tensor, w_scales: torch.Tensor, x_scale: float
) -> torch.Tensor:
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


def ref_per_tensor_quant_gemm(X_q: torch.Tensor, W_q: torch.Tensor, w_scale: float, x_scale: float) -> torch.Tensor:
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
