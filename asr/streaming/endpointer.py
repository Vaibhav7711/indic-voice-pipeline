"""Online endpointing: the streaming counterpart to ``asr.vad.detect_speech``.

Why a second implementation
---------------------------
``asr.vad.detect_speech`` is *offline*. It sees the whole waveform, finds every
voiced run, then merges short gaps and pads boundaries. That is the right shape
for scoring a file, and the wrong shape for a microphone: it cannot answer
"has the user stopped talking?" until the recording is already over.

This module answers exactly that question, frame by frame, having seen only the
past. It deliberately reuses :class:`asr.vad.VADConfig` rather than defining its
own thresholds, so tuning one tunes both and the offline module stays the
reference implementation the online one is checked against.

For an LLM engineer, the analogy is prefill versus decode. Offline VAD is a
single batched pass over a known-length sequence. Online endpointing is an
autoregressive loop with carried state: you commit to a decision at step *t*
knowing nothing about *t+1*, and you cannot revise it later.

The decision rules, matched to the offline module
-------------------------------------------------
* **Frame energy** — RMS over a ``frame_ms`` window every ``hop_ms``, converted
  to dBFS. Identical arithmetic to ``asr.vad``.
* **Speech onset** — ``min_speech_ms`` of *consecutive* voiced frames. This is
  what rejects a cough or a door slam: brief energy never accumulates enough
  consecutive frames to start an utterance.
* **Endpoint** — a silence gap **longer than** ``min_silence_ms`` ends the
  utterance. At most ``min_silence_ms`` is treated as an intra-utterance pause,
  which mirrors the offline merge rule exactly (offline merges gaps of
  ``floor(min_silence_ms / hop_ms)`` frames or fewer, so online endpoints at one
  frame more).
* **Padding** — ``padding_ms`` is added either side of the detected boundaries.
  Whisper needs the leading consonant and trailing vowel that a tight cut
  removes; a hard boundary at the energy threshold clips both.

One deliberate difference from offline
--------------------------------------
Offline VAD computes a final partial frame from whatever samples remain. Online,
a frame is only scored once ``frame`` samples have actually arrived — you cannot
score audio you have not received. Frame alignment is therefore identical while
the stream is running, and the two can differ by at most one frame at the very
end of a stream. :meth:`StreamEndpointer.flush` closes an open utterance at the
stream boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from asr.vad import VADConfig

__all__ = [
    "EndpointerState",
    "EndpointEventKind",
    "EndpointEvent",
    "StreamEndpointer",
]


class EndpointerState(str, Enum):
    """Serializable state, mirrored into every streaming update."""

    SILENCE = "silence"
    SPEECH = "speech"


class EndpointEventKind(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class EndpointEvent:
    """A boundary, in absolute stream-sample coordinates.

    Absolute rather than buffer-relative because the session trims its buffer
    as utterances finalize; buffer-relative indices would silently shift.
    """

    kind: EndpointEventKind
    sample: int
    frame_index: int

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "sample": self.sample,
            "frame_index": self.frame_index,
        }


class StreamEndpointer:
    """Frame-synchronous speech/silence state machine over a live stream."""

    def __init__(self, sample_rate: int, config: VADConfig | None = None):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        config = config or VADConfig()
        if config.frame_ms <= 0 or config.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if min(config.padding_ms, config.min_speech_ms, config.min_silence_ms) < 0:
            raise ValueError("VAD durations must be non-negative")

        self.sample_rate = sample_rate
        self.config = config

        self.frame_samples = max(1, round(sample_rate * config.frame_ms / 1000))
        self.hop_samples = max(1, round(sample_rate * config.hop_ms / 1000))
        self.padding_samples = round(sample_rate * config.padding_ms / 1000)

        self.min_speech_frames = max(
            1, math.ceil(config.min_speech_ms / config.hop_ms)
        )
        # Offline merges gaps of floor(min_silence_ms / hop_ms) frames or fewer
        # as intra-utterance pauses, so an endpoint needs one frame more.
        self.endpoint_silence_frames = (
            math.floor(config.min_silence_ms / config.hop_ms) + 1
        )

        self.state = EndpointerState.SILENCE
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_origin = 0      # absolute sample index of _pending[0]
        self._frame_index = 0
        self._total_samples = 0

        self._voiced_run = 0
        self._silence_run = 0
        self._run_start_frame = 0     # first frame of the current voiced run
        self._last_voiced_frame = 0

    # -- geometry ---------------------------------------------------------

    def _frame_start_sample(self, frame_index: int) -> int:
        return frame_index * self.hop_samples

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def stream_seconds(self) -> float:
        return self._total_samples / self.sample_rate

    # -- main entry point -------------------------------------------------

    def push(self, samples: np.ndarray) -> list[EndpointEvent]:
        """Feed audio; return boundaries crossed by this chunk.

        Chunk sizes are arbitrary and need not align to frames: leftover
        samples are carried in ``_pending`` until a whole frame exists. A
        microphone delivering 10 ms buffers and a file replayed in 5 s blocks
        therefore produce identical frame decisions.
        """
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size:
            self._pending = (
                block if self._pending.size == 0
                else np.concatenate([self._pending, block])
            )
            self._total_samples += block.size

        events: list[EndpointEvent] = []
        while True:
            offset = self._frame_start_sample(self._frame_index) - self._pending_origin
            if offset < 0:  # pragma: no cover - defensive
                raise RuntimeError("endpointer frame cursor fell behind its buffer")
            if offset + self.frame_samples > self._pending.size:
                break

            window = self._pending[offset : offset + self.frame_samples]
            events.extend(self._consume_frame(window))
            self._frame_index += 1

            # Drop audio no future frame can reach.
            keep_from = self._frame_start_sample(self._frame_index) - self._pending_origin
            if keep_from > 0:
                self._pending = self._pending[keep_from:]
                self._pending_origin += keep_from

        return events

    def _consume_frame(self, window: np.ndarray) -> list[EndpointEvent]:
        rms = np.sqrt(np.mean(np.square(window), dtype=np.float64))
        dbfs = 20.0 * math.log10(max(float(rms), 1e-10))
        voiced = dbfs >= self.config.threshold_dbfs
        index = self._frame_index

        if self.state is EndpointerState.SILENCE:
            if voiced:
                if self._voiced_run == 0:
                    self._run_start_frame = index
                self._voiced_run += 1
                if self._voiced_run >= self.min_speech_frames:
                    self.state = EndpointerState.SPEECH
                    self._silence_run = 0
                    self._last_voiced_frame = index
                    start = max(
                        0,
                        self._frame_start_sample(self._run_start_frame)
                        - self.padding_samples,
                    )
                    return [
                        EndpointEvent(
                            EndpointEventKind.SPEECH_START,
                            start,
                            self._run_start_frame,
                        )
                    ]
            else:
                # Consecutive, not cumulative: a cough separated by silence
                # must not accumulate toward the speech-onset threshold.
                self._voiced_run = 0
            return []

        # state is SPEECH
        if voiced:
            self._silence_run = 0
            self._last_voiced_frame = index
            return []

        self._silence_run += 1
        if self._silence_run >= self.endpoint_silence_frames:
            return [self._close_utterance()]
        return []

    def _close_utterance(self) -> EndpointEvent:
        end = min(
            self._total_samples,
            self._frame_start_sample(self._last_voiced_frame)
            + self.frame_samples
            + self.padding_samples,
        )
        self.state = EndpointerState.SILENCE
        self._voiced_run = 0
        self._silence_run = 0
        return EndpointEvent(
            EndpointEventKind.SPEECH_END, end, self._last_voiced_frame
        )

    def flush(self) -> list[EndpointEvent]:
        """Close an utterance still open at end of stream.

        A speaker who stops talking and immediately disconnects never produces
        ``min_silence_ms`` of trailing silence. Without this the final
        utterance would be dropped.
        """
        if self.state is EndpointerState.SPEECH:
            return [self._close_utterance()]
        return []

    def reset(self) -> None:
        """Forget speech state but keep stream-position bookkeeping."""
        self.state = EndpointerState.SILENCE
        self._voiced_run = 0
        self._silence_run = 0

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "stream_seconds": round(self.stream_seconds, 3),
            "frames_processed": self._frame_index,
            "min_speech_frames": self.min_speech_frames,
            "endpoint_silence_frames": self.endpoint_silence_frames,
        }
