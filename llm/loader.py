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


def load_llm(
    model_name: str = "Qwen/Qwen3-0.6B",
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedLLM:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return LoadedLLM(model_name, model, tokenizer, dtype, device)
