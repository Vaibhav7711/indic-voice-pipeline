"""GPU memory management for multi-model pipeline on a single device.

Whisper-medium (1.5 GiB fp16) + Qwen3-0.6B (1.2 GiB) ≈ 2.7 GiB → concurrent on
T4. A 7B-class LLM → sequential (offload Whisper to CPU after transcription).

Qwen3-4B fp16 is ~7.5 GiB of weights plus a paged KV pool at 144 KiB per
cached token, so it is concurrent with Whisper on a 15 GiB T4 and does not
fit an 8 GiB card at all. When it is served by the external engine it is in
another process and invisible to this module's snapshot: two CUDA contexts,
no shared allocator, and the pool sized by `scripts/llm_server_app.py`
instead. Use `--llm-model Qwen/Qwen3-0.6B` on a small card.
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
    if device.type != "cuda":
        # No allocator stats off-GPU; report "plenty" so the concurrent
        # strategy is chosen and nothing is offloaded.
        return VRAMSnapshot(0, 0, 1 << 40)
    return VRAMSnapshot(
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
        torch.cuda.get_device_properties(device).total_memory,
    )


def estimate_model_bytes(model: torch.nn.Module) -> int:
    """Resident weight bytes: parameters plus persistent buffers.

    Deliberately excludes KV cache and activations — those are workload
    dependent and covered by ``choose_strategy``'s headroom ratio.
    """
    params = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return params + buffers


def offload_to_cpu(model: torch.nn.Module) -> None:
    model.to("cpu")
    if torch.cuda.is_available():
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
