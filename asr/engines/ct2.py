"""CTranslate2 / faster-whisper as a serving tier behind the explicit runner.

The explicit runner (``asr.explicit``) owns the loop and is the correctness
reference: it is what found the truncated-onset, eager-synthesis and
token-suppression bugs. An inference engine is a *faster implementation of
the same model*, and it earns its place by matching the reference on the
same audio — the check lives in ``scripts/gpu_validation.py`` — not by
replacing it.

CTranslate2 runs Whisper with fused kernels and int8/float16 weights; on a
T4 it is typically 3–4× faster than eager PyTorch decode at the same greedy
settings. It needs the model converted once (``scripts/convert_ct2.py``,
which merges the LoRA first so the adapter is baked into the weights).

``CT2Transcriber`` satisfies ``asr.streaming.StreamingTranscriber`` and
returns objects with the same ``.text`` / ``.language`` / ``.metrics`` shape
the session and the benchmarks read, so it drops into ``StreamingSession``,
``benchmarks.streaming_eval`` and the live agent unchanged.

Two things it does not give you that the explicit runner does: per-token
timings and the split between encoder and decoder time. ``metrics.total_ms``
is wall clock around the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns

import numpy as np

__all__ = ["CT2Metrics", "CT2Result", "CT2Transcriber"]


@dataclass
class CT2Metrics:
    total_ms: float = 0.0
    audio_duration_seconds: float = 0.0
    decoder_steps: int = 0
    segments: int = 0
    language_probability: float | None = None
    compute_type: str = ""
    # Present so callers reading the explicit runner's fields get None, not
    # AttributeError; the engine does not expose them.
    encoder_ms: float | None = None
    decode_ms: list[float] = field(default_factory=list)
    language_detection_ms: float | None = None

    @property
    def real_time_factor(self) -> float:
        if self.audio_duration_seconds <= 0:
            return 0.0
        return self.total_ms / 1000.0 / self.audio_duration_seconds

    def as_dict(self) -> dict:
        return {
            "total_ms": self.total_ms, "audio_duration_seconds": self.audio_duration_seconds,
            "decoder_steps": self.decoder_steps, "segments": self.segments,
            "language_probability": self.language_probability,
            "compute_type": self.compute_type, "real_time_factor": self.real_time_factor,
        }


@dataclass
class CT2Result:
    text: str
    token_ids: list[int]
    language: str | None
    metrics: CT2Metrics


class CT2Transcriber:
    """faster-whisper with greedy decoding, matching the explicit runner's
    settings (no beam, no temperature fallback, no VAD, no timestamps)."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        compute_type: str = "default",
        language_candidates: list[str] | None = None,
        cpu_threads: int = 0,
    ):
        from faster_whisper import WhisperModel

        self.model_dir = str(model_dir)
        self.compute_type = compute_type
        self.language_candidates = language_candidates
        self.model = WhisperModel(self.model_dir, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads)

    def _run(self, waveform: np.ndarray, sample_rate: int, *, language: str | None,
             max_new_tokens: int) -> CT2Result:
        from asr.explicit.mel import load_audio_from_array

        waveform, seconds = load_audio_from_array(np.asarray(waveform, dtype=np.float32), sample_rate)
        start = perf_counter_ns()
        # Greedy, deterministic, no fallbacks: the same decode the explicit
        # runner performs, so outputs are comparable.
        segments, info = self.model.transcribe(
            waveform, language=language, task="transcribe", beam_size=1, best_of=1,
            temperature=0.0, vad_filter=False, without_timestamps=True,
            condition_on_previous_text=False, max_new_tokens=max_new_tokens,
            # Restrict auto-detect like ASRRunner.language_candidates does.
            language_detection_segments=1,
        )
        texts, tokens = [], []
        for seg in segments:               # generator: decoding happens here
            texts.append(seg.text)
            tokens.extend(seg.tokens)
        total_ms = (perf_counter_ns() - start) / 1_000_000
        detected = info.language if language is None else language
        if language is None and self.language_candidates and detected not in self.language_candidates:
            # faster-whisper cannot restrict detection; fall back to the
            # first candidate rather than decoding in an unwanted language.
            detected = self.language_candidates[0]
        metrics = CT2Metrics(
            total_ms=total_ms, audio_duration_seconds=seconds,
            decoder_steps=len(tokens), segments=len(texts),
            language_probability=(info.language_probability if language is None else None),
            compute_type=self.compute_type,
        )
        return CT2Result(" ".join(t.strip() for t in texts).strip(), tokens, detected, metrics)

    # -- StreamingTranscriber protocol ---------------------------------------

    def transcribe_array(self, waveform, sample_rate, *, language=None,
                         max_new_tokens: int = 225, **kw) -> CT2Result:
        return self._run(waveform, sample_rate, language=language, max_new_tokens=max_new_tokens)

    def transcribe_long_array(self, waveform, sample_rate, *, language=None,
                              max_new_tokens: int = 225, **kw) -> CT2Result:
        # faster-whisper windows long audio itself (30 s, with context
        # carried by the tokens of the previous window disabled above).
        return self._run(waveform, sample_rate, language=language, max_new_tokens=max_new_tokens)

    def transcribe_file(self, path: str, *, language=None, max_new_tokens: int = 225) -> CT2Result:
        from asr.explicit.mel import load_audio

        waveform, _ = load_audio(path)
        return self._run(waveform, 16_000, language=language, max_new_tokens=max_new_tokens)
