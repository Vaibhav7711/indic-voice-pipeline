"""Voice pipeline: audio → ASR → LLM answer.

Chains ASR and LLM on a single GPU with per-stage timing and automatic
memory strategy selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np
import torch

from asr.explicit.loader import LoadedWhisper
from asr.explicit.runner import ASRMetrics, ASRRunner
from llm.loader import LoadedLLM
from llm.runner import LLMRunner
from pipeline.memory import (
    MemoryStrategy, choose_strategy, estimate_model_bytes,
    offload_to_cpu, reload_to_gpu, snapshot_vram,
)


SYSTEM_PROMPTS = {
    "hi": (
        "You are a helpful voice assistant. The user spoke in Hindi "
        "(possibly code-switched with English). Respond entirely in natural "
        "Hindi; keep unavoidable proper nouns and technical terms as-is. "
        "Keep answers brief — this will be spoken aloud."
    ),
    "te": (
        "You are a helpful voice assistant. The user spoke in Telugu "
        "(possibly code-switched with English). Respond concisely."
    ),
    "en": "You are a helpful voice assistant. Respond concisely.",
    None: "You are a helpful voice assistant. Respond concisely.",
}


@dataclass
class PipelineMetrics:
    asr: ASRMetrics | None = None
    llm_prefill_ms: float = 0.0
    llm_decode_ms: list[float] = field(default_factory=list)
    llm_total_ms: float = 0.0
    llm_tokens_generated: int = 0
    memory_strategy: str = ""
    model_swap_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    vram_after_asr_gib: float = 0.0
    vram_after_llm_gib: float = 0.0
    peak_allocated_bytes: int = 0

    @property
    def audio_to_first_llm_token_ms(self) -> float:
        asr_total = self.asr.total_ms if self.asr else 0.0
        return asr_total + self.model_swap_ms + self.llm_prefill_ms

    def as_dict(self) -> dict:
        return {
            "asr": self.asr.as_dict() if self.asr else None,
            "llm_prefill_ms": self.llm_prefill_ms,
            "llm_mean_decode_ms": (
                sum(self.llm_decode_ms) / len(self.llm_decode_ms)
                if self.llm_decode_ms else 0.0
            ),
            "llm_total_ms": self.llm_total_ms,
            "llm_tokens_generated": self.llm_tokens_generated,
            "memory_strategy": self.memory_strategy,
            "model_swap_ms": self.model_swap_ms,
            "audio_to_first_llm_token_ms": self.audio_to_first_llm_token_ms,
            "total_pipeline_ms": self.total_pipeline_ms,
            "peak_allocated_bytes": self.peak_allocated_bytes,
        }


@dataclass
class PipelineResult:
    transcript: str
    answer: str
    answer_token_ids: list[int]
    language: str | None
    metrics: PipelineMetrics


class VoicePipeline:
    def __init__(
        self,
        whisper: LoadedWhisper,
        llm: LoadedLLM,
        *,
        system_prompt: str | None = None,
    ):
        self.whisper = whisper
        self.llm = llm
        self.device = whisper.device
        self.custom_system_prompt = system_prompt

        self.asr_runner = ASRRunner(
            whisper.model, whisper.processor, whisper.device, whisper.dtype,
        )
        self.llm_runner = LLMRunner(llm.model, llm.tokenizer, llm.device)

        self.strategy = choose_strategy(
            estimate_model_bytes(whisper.model),
            estimate_model_bytes(llm.model),
            self.device,
        )

    def _build_prompt(self, transcript: str, language: str | None) -> str:
        system = self.custom_system_prompt or SYSTEM_PROMPTS.get(
            language, SYSTEM_PROMPTS[None],
        )
        if hasattr(self.llm.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript},
                ]
                return self.llm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System: {system}\n\nUser: {transcript}\n\nAssistant:"

    def run(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        asr_max_tokens: int = 225,
        llm_max_tokens: int = 128,
    ) -> PipelineResult:
        pipe_start = perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)
        metrics = PipelineMetrics(memory_strategy=self.strategy.value)

        # ASR.
        if self.strategy == MemoryStrategy.SEQUENTIAL:
            reload_to_gpu(self.whisper.model, self.device, self.whisper.dtype)

        asr_result = self.asr_runner.transcribe_file(
            audio_path, language=language, max_new_tokens=asr_max_tokens,
        )
        metrics.asr = asr_result.metrics
        metrics.vram_after_asr_gib = snapshot_vram(self.device).allocated_gib

        # Model swap (sequential only).
        if self.strategy == MemoryStrategy.SEQUENTIAL:
            swap_start = perf_counter_ns()
            offload_to_cpu(self.whisper.model)
            reload_to_gpu(self.llm.model, self.device, self.llm.dtype)
            metrics.model_swap_ms = (perf_counter_ns() - swap_start) / 1_000_000

        # LLM.
        prompt = self._build_prompt(asr_result.text, language or asr_result.language)
        llm_result = self.llm_runner.generate(prompt, max_new_tokens=llm_max_tokens)

        metrics.llm_prefill_ms = llm_result.metrics.prefill_ms
        metrics.llm_decode_ms = llm_result.metrics.decode_ms
        metrics.llm_total_ms = llm_result.metrics.total_ms
        metrics.llm_tokens_generated = llm_result.metrics.generated_tokens
        metrics.vram_after_llm_gib = snapshot_vram(self.device).allocated_gib

        if self.strategy == MemoryStrategy.SEQUENTIAL:
            offload_to_cpu(self.llm.model)
            reload_to_gpu(self.whisper.model, self.device, self.whisper.dtype)

        metrics.total_pipeline_ms = (perf_counter_ns() - pipe_start) / 1_000_000
        metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)

        return PipelineResult(
            transcript=asr_result.text,
            answer=llm_result.text,
            answer_token_ids=llm_result.token_ids,
            language=language or asr_result.language,
            metrics=metrics,
        )

    def run_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        language: str | None = None,
        llm_max_tokens: int = 128,
    ) -> PipelineResult:
        """Run pipeline on in-memory waveform (Gradio/API)."""
        pipe_start = perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)
        metrics = PipelineMetrics(memory_strategy=self.strategy.value)

        asr_result = self.asr_runner.transcribe_array(
            waveform, sample_rate, language=language,
        )
        metrics.asr = asr_result.metrics

        prompt = self._build_prompt(asr_result.text, language or asr_result.language)
        llm_result = self.llm_runner.generate(prompt, max_new_tokens=llm_max_tokens)

        metrics.llm_prefill_ms = llm_result.metrics.prefill_ms
        metrics.llm_decode_ms = llm_result.metrics.decode_ms
        metrics.llm_total_ms = llm_result.metrics.total_ms
        metrics.llm_tokens_generated = llm_result.metrics.generated_tokens
        metrics.total_pipeline_ms = (perf_counter_ns() - pipe_start) / 1_000_000
        metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)

        return PipelineResult(
            transcript=asr_result.text,
            answer=llm_result.text,
            answer_token_ids=llm_result.token_ids,
            language=language or asr_result.language,
            metrics=metrics,
        )
