"""A transparent energy VAD for endpointing speech before ASR.

This is intentionally a baseline, not a learned speech detector.  Its value is
that every decision is inspectable: frame RMS → dBFS threshold → minimum speech
duration → silence merge → boundary padding.  It is a useful control when we
later evaluate a neural VAD on Hindi, code-switching, and noisy audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 30
    hop_ms: int = 10
    threshold_dbfs: float = -40.0
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    padding_ms: int = 200


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of true frame indexes."""
    starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_speech(
    waveform: np.ndarray,
    sample_rate: int,
    config: VADConfig = VADConfig(),
) -> list[SpeechSegment]:
    """Return padded speech regions using RMS energy and endpointing rules."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.frame_ms <= 0 or config.hop_ms <= 0:
        raise ValueError("frame_ms and hop_ms must be positive")
    if not (0 <= config.padding_ms and 0 <= config.min_speech_ms and 0 <= config.min_silence_ms):
        raise ValueError("VAD durations must be non-negative")

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return []

    frame = max(1, round(sample_rate * config.frame_ms / 1000))
    hop = max(1, round(sample_rate * config.hop_ms / 1000))
    starts = np.arange(0, samples.size, hop)
    dbfs = np.empty(len(starts), dtype=np.float32)
    for index, start in enumerate(starts):
        window = samples[start:min(start + frame, samples.size)]
        rms = np.sqrt(np.mean(np.square(window), dtype=np.float64))
        dbfs[index] = 20.0 * np.log10(max(rms, 1e-10))

    voiced = dbfs >= config.threshold_dbfs
    min_speech_frames = max(1, int(np.ceil(config.min_speech_ms / config.hop_ms)))
    speech_runs = [(start, end) for start, end in _runs(voiced) if end - start >= min_speech_frames]
    if not speech_runs:
        return []

    # A short silence is an intra-utterance pause, not an endpoint.
    max_gap_frames = int(np.floor(config.min_silence_ms / config.hop_ms))
    merged: list[tuple[int, int]] = []
    for start, end in speech_runs:
        if merged and start - merged[-1][1] <= max_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    padding = round(sample_rate * config.padding_ms / 1000)
    return [
        SpeechSegment(
            start_sample=max(0, int(starts[start]) - padding),
            end_sample=min(samples.size, int(starts[end - 1] + frame) + padding),
            sample_rate=sample_rate,
        )
        for start, end in merged
    ]
