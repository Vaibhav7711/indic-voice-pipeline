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
    """Energy-VAD rules shared by the offline detector and the online endpointer.

    Threshold
    ---------
    A fixed dBFS threshold cannot serve real audio: in the first GPU
    validation sweep two FLEURS clips differed by 13 dB in level, and a
    ``-40`` threshold kept 69% of one and 17% of the other, deleting whole
    phrases before Whisper saw them. With ``adaptive_threshold`` (default) the
    threshold tracks the signal: the minimum frame level over the last
    ``noise_window_ms`` (inter-word gaps expose the noise floor even in
    continuous speech) plus ``noise_margin_db``, never below
    ``threshold_floor_dbfs`` so digital silence does not pull it to -80, and
    never above ``threshold_dbfs``: a frame at or above that level is speech
    regardless of what the tracker thinks, which keeps loud audio (and audio
    that starts mid-speech, where no gap has revealed the floor yet) working
    exactly as the fixed rule did. So the adaptive rule can only ever *lower*
    the bar. With ``adaptive_threshold=False`` the fixed value is used alone.
    The tracker is causal so offline and online produce identical decisions.
    """

    frame_ms: int = 30
    hop_ms: int = 10
    threshold_dbfs: float = -40.0
    adaptive_threshold: bool = True
    noise_window_ms: int = 3000
    noise_margin_db: float = 10.0
    threshold_floor_dbfs: float = -60.0
    min_speech_ms: int = 250
    #: 600 rather than 400: read Hindi has comma-length pauses near 400 ms and
    #: a cut there costs Whisper the context for both halves. +200 ms latency.
    min_silence_ms: int = 600
    padding_ms: int = 200

    def as_dict(self) -> dict:
        return {
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "threshold_dbfs": self.threshold_dbfs,
            "adaptive_threshold": self.adaptive_threshold,
            "noise_window_ms": self.noise_window_ms,
            "noise_margin_db": self.noise_margin_db,
            "threshold_floor_dbfs": self.threshold_floor_dbfs,
            "min_speech_ms": self.min_speech_ms,
            "min_silence_ms": self.min_silence_ms,
            "padding_ms": self.padding_ms,
        }


def frame_dbfs(window: np.ndarray) -> float:
    """RMS level of one frame in dBFS. The one arithmetic both modules share."""
    rms = np.sqrt(np.mean(np.square(window), dtype=np.float64))
    return 20.0 * float(np.log10(max(float(rms), 1e-10)))


class AdaptiveThreshold:
    """Causal per-frame voicing threshold: sliding-window minimum plus margin.

    ``update(dbfs)`` consumes one frame level and returns the threshold to
    apply to *that* frame. Uses a monotonic deque, so each frame is O(1)
    amortised. With ``config.adaptive_threshold=False`` it returns the fixed
    ``threshold_dbfs`` and keeps no state.
    """

    def __init__(self, config: VADConfig):
        self.config = config
        self.window = max(1, int(round(config.noise_window_ms / config.hop_ms)))
        self._index = 0
        self._deque: list[tuple[int, float]] = []   # (frame index, dbfs), increasing dbfs
        self.noise_floor_dbfs: float | None = None

    def update(self, dbfs: float) -> float:
        if not self.config.adaptive_threshold:
            return self.config.threshold_dbfs
        i = self._index
        self._index += 1
        while self._deque and self._deque[-1][1] >= dbfs:
            self._deque.pop()
        self._deque.append((i, dbfs))
        while self._deque and self._deque[0][0] <= i - self.window:
            self._deque.pop(0)
        self.noise_floor_dbfs = self._deque[0][1]
        adaptive = max(self.config.threshold_floor_dbfs,
                       self.noise_floor_dbfs + self.config.noise_margin_db)
        return min(self.config.threshold_dbfs, adaptive)


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
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


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
    voiced = np.empty(len(starts), dtype=bool)
    threshold = AdaptiveThreshold(config)
    for index, start in enumerate(starts):
        window = samples[start:min(start + frame, samples.size)]
        level = frame_dbfs(window)
        voiced[index] = level >= threshold.update(level)

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
