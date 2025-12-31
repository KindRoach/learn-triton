import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.xpu.is_available():
        return torch.device("xpu")

    raise RuntimeError("No supported device found (CUDA or XPU).")
