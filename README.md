# learn-triton

A small, example-driven repo for learning Triton by implementing and benchmarking kernels.

## What’s inside

Examples live under `src/`:

- `src/01_baisc/` — Triton basics: hello world, memory access, autotune, compile metadata, etc.
- `src/02_vector/` — vector add / dot.
- `src/03_matrix/` — matrix transpose / matmul.
- `src/04_attn/` — attention kernerls.
- `src/05_moe/` — moe kernels.
- `src/utils/` — helpers for benchmarking, profiling, etc.

## Requirements

- Python 3
- PyTorch
- Triton
- A supported accelerator:
  - NVIDIA CUDA GPU (`torch.cuda.is_available()`), or
  - Intel XPU (`torch.xpu.is_available()`)

## Setup (recommended): VS Code Dev Container

This repo includes devcontainer configs:

- CUDA: `.devcontainer/cuda/`
- XPU: `.devcontainer/xpu/`

In VS Code: **Command Palette → “Dev Containers: Reopen in Container”** and pick the appropriate config.

Notes:

- The CUDA container is based on `nvcr.io/nvidia/pytorch:25.12-py3` and installs Triton.
- The XPU container installs Intel GPU runtime libs, PyTorch XPU wheels, and tooling.

## Setup (local)

If you prefer local installs, you’ll need a working PyTorch + accelerator stack first.

Then install Triton:

```bash
pip install triton
```

(Exact PyTorch installation commands vary by platform/driver; use the official PyTorch installer for CUDA or XPU.)

## Running the examples

Run from the repo root using module mode so relative imports work:

```bash
python3 -m src.01_baisc.01_hello_world
```

## Benchmarking and correctness

Many scripts:

- benchmark with `bench_by_secs()` (`src/utils/bench.py`), and
- validate outputs with `acc_check()` (`src/utils/accuracy.py`).

Autotuning logs are enabled by default because `src/utils/__init__.py` sets:

- `TRITON_PRINT_AUTOTUNING=1`

## Debugging

This repo includes VS Code launch configurations in `.vscode/launch.json`:

- **debug triton: Current File** — runs with `TRITON_INTERPRET=1` (useful for stepping through Triton logic)
- **debug py: Current File** — normal Python debugging
