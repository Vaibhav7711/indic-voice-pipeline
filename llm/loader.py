from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedLLM:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    dtype: torch.dtype
    device: torch.device


def pick_dtype(requested: torch.dtype | None = None) -> torch.dtype:
    """bf16 where the GPU has it natively, else fp16.

    T4 (sm_75) has no bf16 tensor cores: PyTorch runs bf16 matmuls through
    a slow path, and a 0.6B model decoded at ~41 ms/token in the first
    sweeps for that reason. Ampere and later take bf16 as requested.
    """
    if requested is not None and requested != torch.bfloat16:
        return requested
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_llm(
    model_name: str = "Qwen/Qwen3-0.6B",
    *,
    dtype: torch.dtype | None = None,
) -> LoadedLLM:
    """Load a causal LM on the GPU. ``dtype=None`` picks per GPU (see pick_dtype)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    dtype = pick_dtype(dtype)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return LoadedLLM(model_name, model, tokenizer, dtype, device)
