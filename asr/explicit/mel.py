"""Explicit mel-spectrogram extraction for Whisper.

Whisper expects 80-channel log-mel spectrograms from 16 kHz audio (25 ms
window, 10 ms hop). The HF WhisperFeatureExtractor computes this; we call
it directly and time the CPU work.

Audio loading is separated from feature extraction so each stage is
independently measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import torch
from transformers import WhisperProcessor

WHISPER_SAMPLE_RATE = 16_000


@dataclass
class MelResult:
    input_features: torch.Tensor  # (1, 80, 3000)
    duration_seconds: float
    load_ms: float
    mel_ms: float


def load_audio(path: str) -> tuple[np.ndarray, float]:
    """Load and resample audio to 16 kHz mono float32."""
    import librosa

    waveform, _ = librosa.load(path, sr=WHISPER_SAMPLE_RATE, mono=True)
    return waveform, len(waveform) / WHISPER_SAMPLE_RATE


def load_audio_from_array(
    waveform: np.ndarray, sample_rate: int,
) -> tuple[np.ndarray, float]:
    """Resample an in-memory waveform to 16 kHz if needed."""
    if sample_rate != WHISPER_SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=WHISPER_SAMPLE_RATE,
        )
    return waveform.astype(np.float32), len(waveform) / WHISPER_SAMPLE_RATE


def extract_mel(
    waveform: np.ndarray,
    duration_seconds: float,
    processor: WhisperProcessor,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    load_ms: float = 0.0,
) -> MelResult:
    """Compute log-mel spectrogram and move to GPU.

    The feature extractor pads/truncates to 30 s (3000 mel frames).
    For audio > 30 s the caller must chunk.
    """
    start = perf_counter_ns()
    features = processor.feature_extractor(
        waveform, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt",
    )
    input_features = features.input_features.to(device=device, dtype=dtype)
    mel_ms = (perf_counter_ns() - start) / 1_000_000

    return MelResult(input_features, duration_seconds, load_ms, mel_ms)
