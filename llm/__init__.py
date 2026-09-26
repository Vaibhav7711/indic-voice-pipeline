"""Lightweight explicit LLM runner for the voice pipeline.

Owns prefill/decode split with CUDA timing. Not a full inference engine —
just enough to chain after ASR with measured latency.

The runner and loader import torch/transformers, so they are resolved lazily:
``llm.prompting`` is pure Python and must stay importable on CPU-only test
machines (``agent.turn`` depends on it).
"""

from __future__ import annotations

from llm.prompting import (
    SYSTEM_PROMPTS,
    SYSTEM_VARIANTS,
    build_chat_prompt,
    system_prompt_for,
)

__all__ = [
    "LoadedLLM",
    "load_llm",
    "LLMRunner",
    "LLMResult",
    "SYSTEM_PROMPTS",
    "build_chat_prompt",
    "system_prompt_for",
    "SYSTEM_VARIANTS",
]

_LAZY = {
    "LoadedLLM": ("llm.loader", "LoadedLLM"),
    "load_llm": ("llm.loader", "load_llm"),
    "LLMRunner": ("llm.runner", "LLMRunner"),
    "LLMResult": ("llm.runner", "LLMResult"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'llm' has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
