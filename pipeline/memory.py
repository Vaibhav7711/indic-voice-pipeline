"""GPU memory management for multi-model pipeline on a single device.

Whisper-small (488 MB) + Qwen3-0.6B (1.2 GiB) = ~1.7 GiB → concurrent on T4.
Larger Indic LLMs → sequential (offload Whisper to CPU after transcription).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MemoryStrategy(Enum):
    CONCURRENT = "concurrent"
    SEQUENTIAL = "sequential"


@dataclass
class VRAMSnapshot:
    allocated_bytes: int
    reserved_bytes: int
    total_bytes: int

    @property
    def free_gib(self) -> float:
        return (self.total_bytes - self.allocated_bytes) / (1024**3)

    @property
    def allocated_gib(self) -> float:
        return self.allocated_bytes / (1024**3)


def snapshot_vram(device: torch.device | None = None) -> VRAMSnapshot:
    device = device or torch.device("cuda")
    return VRAMSnapshot(
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
        torch.cuda.get_device_properties(device).total_mem,
    )


def estimate_model_bytes(model: torch.nn.Module) -> int:
    return sum(p.nelement() * p.element_size() for p in model.parameters())


def offload_to_cpu(model: torch.nn.Module) -> None:
    model.to("cpu")
    torch.cuda.empty_cache()


def reload_to_gpu(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    model.to(device=device, dtype=dtype)


def choose_strategy(
    whisper_bytes: int,
    llm_bytes: int,
    device: torch.device | None = None,
    headroom_ratio: float = 0.25,
) -> MemoryStrategy:
    snap = snapshot_vram(device)
    headroom = int(snap.total_bytes * headroom_ratio)
    if whisper_bytes + llm_bytes + headroom <= snap.total_bytes:
        return MemoryStrategy.CONCURRENT
    return MemoryStrategy.SEQUENTIAL
