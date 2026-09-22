"""Load a Whisper checkpoint for explicit encoder/decoder inference.

Returns an immutable bundle the runner consumes. FP16 by default because
Whisper was trained in FP16 and Whisper-small (244M, ~488 MB) fits on T4
alongside a 0.6B LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass(frozen=True)
class LoadedWhisper:
    model_name: str
    adapter_path: str | None
    model: WhisperForConditionalGeneration
    processor: WhisperProcessor
    dtype: torch.dtype
    device: torch.device


def pick_device(device: str | torch.device | None = None) -> torch.device:
    """CUDA when available, else CPU. ``device`` overrides ("cpu", "cuda:1")."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_whisper(
    model_name: str = "openai/whisper-small",
    *,
    adapter_path: str | Path | None = None,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> LoadedWhisper:
    """Load Whisper, optionally merging a PEFT LoRA adapter for inference.

    ``adapter_path`` is a local directory containing PEFT's
    ``adapter_config.json`` and ``adapter_model.safetensors``, or a Hub repo id
    such as ``Hugme6969/whisper-medium-hindi-lora`` (downloaded to the HF
    cache). The adapter is merged before the explicit runner receives the
    model, so its encoder/decoder and KV-cache path remain exactly the same as
    base-model inference.
    """
    device = pick_device(device)
    if dtype is None:
        # fp16 on GPU (the reported configuration); fp32 on CPU, where fp16
        # matmuls are unsupported or slow.
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    adapter = None
    if adapter_path:
        from benchmarks.checkpoints import resolve_hub_adapter

        adapter = resolve_hub_adapter(adapter_path).expanduser().resolve()
    if adapter is not None:
        required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Invalid LoRA adapter directory; missing: " + ", ".join(missing),
            )

    # Trainer checkpoints contain the adapter and feature-extractor config but
    # not necessarily the tokenizer files. Final exported adapters do contain
    # them, so prefer those only when the complete processor is present.
    processor_source = model_name
    if adapter is not None and (adapter / "tokenizer_config.json").is_file():
        processor_source = str(adapter)
    processor = WhisperProcessor.from_pretrained(processor_source)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    model = model.to(device)
    model.eval()
    return LoadedWhisper(
        model_name=model_name,
        adapter_path=str(adapter) if adapter is not None else None,
        model=model,
        processor=processor,
        dtype=dtype,
        device=device,
    )
