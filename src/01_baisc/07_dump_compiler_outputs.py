import os
from pathlib import Path

# Triton reads these variables when it is imported, so set them first.
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_DUMP_DIR"] = str(Path.cwd() / "triton-dump" / "automatic")

import torch
import triton
import triton.language as tl

from ..utils import get_device


@triton.jit
def add_one_kernel(
    x_ptr,
    output_ptr,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + 1, mask=mask)


def dump_compiler_outputs(compiled_kernel, output_dir: Path) -> None:
    """Write the compiler stages held by a Triton CompiledKernel to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = dict(compiled_kernel.asm)

    # SASS is generated lazily from a cubin and is therefore not initially in
    # compiled_kernel.asm. It is NVIDIA-specific and requires nvdisasm.
    if "cubin" in outputs:
        try:
            outputs["sass"] = compiled_kernel.asm["sass"]
        except (KeyError, RuntimeError):
            pass

    for extension, contents in outputs.items():
        path = output_dir / f"{compiled_kernel.name}.{extension}"
        if isinstance(contents, bytes):
            path.write_bytes(contents)
        else:
            path.write_text(str(contents))
        print(path)


def main():
    device = get_device()
    N = 1024
    BLOCK_SIZE = 256

    x = torch.randn(N, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # A Triton launch returns the CompiledKernel. Its `asm` dictionary contains
    # the compiler stages for this specialization of the kernel.
    compiled = add_one_kernel[grid](x, output, N, BLOCK_SIZE)
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, x + 1)

    dump_compiler_outputs(compiled, Path("triton-dump/programmatic"))


if __name__ == "__main__":
    main()
