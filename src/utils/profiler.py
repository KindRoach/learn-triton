import os

import torch
from torch.profiler import profile, ProfilerActivity


class TorchProfiler:
    def __init__(
        self,
        profile_name: str,
        trace_dir: str = "./torch_profiler_traces",
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        profile_memory: bool = True,
        print_table: bool = False,
    ):
        self.profile_name = profile_name
        self.trace_dir = trace_dir
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.print_table = print_table

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            if torch.xpu.is_available():
                activities.append(ProfilerActivity.XPU)

            with profile(
                activities=activities,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                on_trace_ready=self._save_trace,
            ) as prof:
                result = func(*args, **kwargs)

            if self.print_table:
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=20,
                    )
                )

            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__qualname__ = func.__qualname__

        return wrapper

    def _save_trace(self, prof: profile) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        trace_path = os.path.join(self.trace_dir, f"{self.profile_name}.gz")
        prof.export_chrome_trace(trace_path)
        print(f"Profiler trace saved to: {trace_path}")
