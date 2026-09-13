"""End-to-end ASR runner: mel → encoder → autoregressive decoder.

Owns the full inference loop with per-stage CUDA timing. model.generate()
and pipeline("asr") are never called — correctness references only in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter_ns

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr.explicit.decoder import WhisperDecoder
from asr.explicit.encoder import WhisperEncoder
from asr.explicit.mel import extract_mel, load_audio, load_audio_from_array


@dataclass
class ASRMetrics:
    """Per-stage latency breakdown — the whole point of owning the loop."""

    audio_load_ms: float = 0.0
    mel_extraction_ms: float = 0.0
    encoder_ms: float = 0.0
    decoder_prefill_ms: float = 0.0
    decoder_steps: int = 0
    decode_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0
    audio_duration_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    encoder_hidden_size: int = 0
    encoder_seq_length: int = 0

    @property
    def mean_decode_ms(self) -> float:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0

    @property
    def total_decode_ms(self) -> float:
        return sum(self.decode_ms)

    @property
    def real_time_factor(self) -> float:
        """RTF = processing_time / audio_duration. < 1.0 = faster than real-time."""
        if self.audio_duration_seconds <= 0:
            return 0.0
        return (self.total_ms / 1000.0) / self.audio_duration_seconds

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mean_decode_ms"] = self.mean_decode_ms
        d["total_decode_ms"] = self.total_decode_ms
        d["real_time_factor"] = self.real_time_factor
        return d


@dataclass
class ASRResult:
    text: str
    token_ids: list[int]
    language: str | None
    metrics: ASRMetrics


class ASRRunner:
    """Explicit Whisper ASR runtime.

    Owns: mel extraction, encoder forward, autoregressive decoder, timing.
    Does NOT call: model.generate(), pipeline("asr").
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.encoder = WhisperEncoder(model, device)
        self.decoder = WhisperDecoder(model, device)
        self.eos_token_id = model.generation_config.eos_token_id

    def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
        max_new_tokens: int = 225,
    ) -> ASRResult:
        """Transcribe an audio file with full latency breakdown."""
        total_start = perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)

        load_start = perf_counter_ns()
        waveform, duration = load_audio(path)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000

        return self._transcribe(waveform, duration, load_ms, language,
                                max_new_tokens, total_start)

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        language: str | None = None,
        max_new_tokens: int = 225,
    ) -> ASRResult:
        """Transcribe an in-memory waveform (for Gradio/pipeline use)."""
        total_start = perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)

        load_start = perf_counter_ns()
        waveform, duration = load_audio_from_array(waveform, sample_rate)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000

        return self._transcribe(waveform, duration, load_ms, language,
                                max_new_tokens, total_start)

    def _transcribe(
        self,
        waveform: np.ndarray,
        duration: float,
        load_ms: float,
        language: str | None,
        max_new_tokens: int,
        total_start: int,
    ) -> ASRResult:
        metrics = ASRMetrics(audio_load_ms=load_ms, audio_duration_seconds=duration)

        # 1. Mel spectrogram.
        mel = extract_mel(waveform, duration, self.processor, self.device, self.dtype)
        metrics.mel_extraction_ms = mel.mel_ms

        # 2. Encoder forward.
        enc = self.encoder.forward(mel.input_features)
        metrics.encoder_ms = enc.encoder_ms
        metrics.encoder_hidden_size = enc.hidden_size
        metrics.encoder_seq_length = enc.sequence_length

        # 3. Decoder prefill.
        state, prefill_ms = self.decoder.prefill(
            enc.encoder_outputs, language=language,
        )
        metrics.decoder_prefill_ms = prefill_ms

        # 4. Autoregressive decode loop.
        eos_ids = (
            {self.eos_token_id}
            if isinstance(self.eos_token_id, int)
            else set(self.eos_token_id or [])
        )

        for step in range(max_new_tokens):
            token_id = int(state.next_token.item())
            state.decoded_tokens.append(token_id)
            if token_id in eos_ids:
                break
            if step == max_new_tokens - 1:
                break
            state, step_ms = self.decoder.decode_one(state)
            metrics.decode_ms.append(step_ms)

        metrics.decoder_steps = len(state.decoded_tokens)
        metrics.total_ms = (perf_counter_ns() - total_start) / 1_000_000
        metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        metrics.peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)

        text = self.processor.tokenizer.decode(
            state.decoded_tokens, skip_special_tokens=True,
        )

        return ASRResult(text.strip(), state.decoded_tokens, language, metrics)
