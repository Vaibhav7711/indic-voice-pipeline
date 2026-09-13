"""Load a Whisper checkpoint for explicit encoder/decoder inference.

Returns an immutable bundle the runner consumes. FP16 by default because
Whisper was trained in FP16 and Whisper-small (244M, ~488 MB) fits on T4
alongside a 0.6B LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass(frozen=True)
class LoadedWhisper:
    model_name: str
    model: WhisperForConditionalGeneration
    processor: WhisperProcessor
    dtype: torch.dtype
    device: torch.device


def load_whisper(
    model_name: str = "openai/whisper-small",
    *,
    dtype: torch.dtype = torch.float16,
) -> LoadedWhisper:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Use a GPU Colab runtime.")

    device = torch.device("cuda")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return LoadedWhisper(model_name, model, processor, dtype, device)
