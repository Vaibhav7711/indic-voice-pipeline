"""Explicit LLM prefill/decode runner with CUDA timing.

Lightweight version for the voice pipeline. Owns the autoregressive loop —
does not call model.generate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMMetrics:
    prefill_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_allocated_bytes: int = 0

    @property
    def mean_decode_ms(self) -> float:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0

    def as_dict(self) -> dict:
        return {
            "prefill_ms": self.prefill_ms,
            "mean_decode_ms": self.mean_decode_ms,
            "total_decode_ms": sum(self.decode_ms),
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "peak_allocated_bytes": self.peak_allocated_bytes,
        }


@dataclass
class LLMResult:
    text: str
    token_ids: list[int]
    metrics: LLMMetrics


class LLMRunner:
    """Explicit prefill/decode LLM runner."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        eos = tokenizer.eos_token_id
        self.eos_ids: set[int] = {eos} if isinstance(eos, int) else set(eos or [])

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> LLMResult:
        """Generate text from a prompt with explicit prefill/decode and timing."""
        total_start = perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)
        metrics = LLMMetrics()

        # Tokenize.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        metrics.prompt_tokens = input_ids.shape[1]

        # Prefill.
        torch.cuda.synchronize(self.device)
        pf_start = torch.cuda.Event(enable_timing=True)
        pf_end = torch.cuda.Event(enable_timing=True)

        pf_start.record()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        pf_end.record()
        pf_end.synchronize()
        metrics.prefill_ms = pf_start.elapsed_time(pf_end)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_kv = outputs.past_key_values
        generated: list[int] = []

        # Decode loop.
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in self.eos_ids:
                break
            if len(generated) >= 8 and generated[-4:] == generated[-8:-4]:
                generated = generated[:-4]
                break

            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device),
            ], dim=1)

            d_start = torch.cuda.Event(enable_timing=True)
            d_end = torch.cuda.Event(enable_timing=True)
            d_start.record()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
            d_end.record()
            d_end.synchronize()
            metrics.decode_ms.append(d_start.elapsed_time(d_end))

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = outputs.past_key_values

        metrics.generated_tokens = len(generated)
        metrics.total_ms = (perf_counter_ns() - total_start) / 1_000_000
        metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return LLMResult(text.strip(), generated, metrics)
