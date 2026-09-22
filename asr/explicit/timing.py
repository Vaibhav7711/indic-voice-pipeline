"""Device-agnostic timing and memory helpers for the explicit runtime.

On CUDA these are event timings and allocator peaks — the numbers the
project reports. On CPU (and MPS) they fall back to ``perf_counter`` and
zero memory, so the same loop runs on a laptop for tests and demos; those
timings are wall clock and are never quoted as GPU results.
"""

from __future__ import annotations

from time import perf_counter_ns

import torch

__all__ = ["Timer", "reset_peak", "peak_allocated", "peak_reserved", "is_cuda"]


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


class Timer:
    """``with Timer(device) as t: ...`` then ``t.ms``."""

    def __init__(self, device: torch.device):
        self.device = device
        self.ms = 0.0

    def __enter__(self):
        if is_cuda(self.device):
            torch.cuda.synchronize(self.device)
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = perf_counter_ns()
        return self

    def __exit__(self, *exc):
        if is_cuda(self.device):
            self._end.record()
            self._end.synchronize()
            self.ms = self._start.elapsed_time(self._end)
        else:
            self.ms = (perf_counter_ns() - self._t0) / 1_000_000
        return False


def reset_peak(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_allocated(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if is_cuda(device) else 0


def peak_reserved(device: torch.device) -> int:
    return torch.cuda.max_memory_reserved(device) if is_cuda(device) else 0
