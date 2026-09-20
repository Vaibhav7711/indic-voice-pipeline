"""Stateful streaming ASR: feed PCM chunks, receive structured updates.

The shape of the problem
------------------------
A microphone hands you 10-40 ms of audio at a time, forever. Whisper is an
offline encoder-decoder: it wants a bounded utterance and produces one answer.
Bridging the two needs three things this module provides.

**Endpointing.** Something must decide where an utterance ends, because Whisper
will not. That is :mod:`asr.streaming.endpointer`, which reuses the project's
VAD thresholds.

**Rate limiting.** Running ASR on every arriving buffer would re-transcribe the
same audio hundreds of times per utterance for almost no new information. Two
independent limits apply, and both must clear:

* ``min_partial_audio_ms`` — how much *new audio* justifies another pass. This
  is the one that stops 20 ms microphone buffers from triggering 50 decodes a
  second.
* ``partial_interval_ms`` — how much *wall-clock* must pass. This is the one
  that matters when audio arrives faster than real time, replaying a file or
  catching up after a stall, where the audio test alone would fire constantly.

They are separate because they fail in different situations, and a single limit
cannot cover both.

**Buffering.** Audio is retained from the padded utterance start and trimmed
once an utterance finalizes, so memory is bounded by ``max_utterance_seconds``
rather than by session length.

Partials are approximate by construction
----------------------------------------
A partial is decoded from incomplete audio, so the model lacks the right context
that fixes word boundaries and case endings. Partials **will** change as more
audio arrives. They exist for perceived latency — showing the user something at
300 ms rather than nothing until 2 s — and must never be treated as the result.
Only ``is_final=True`` carries the committed transcript.

The LLM-inference analogy: a partial is a mid-generation preview of a beam that
has not converged. Useful to display, wrong to act on.

Cost note: with the default 25 s partial window, every partial re-encodes the
whole window from scratch. Whisper's encoder is non-causal, so there is no
incremental encode to reuse — the price of a partial is a full forward pass,
which is precisely why the rate limits matter.

Example
-------
::

    session = StreamingSession(runner, StreamingConfig(language="hi"))
    for block in microphone:                 # float32, 16 kHz, any length
        for update in session.push(block):
            if update.is_final:
                handle(update.text)
    for update in session.flush():           # close an open utterance
        handle(update.text)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

import numpy as np

from asr.streaming.endpointer import (
    EndpointEventKind,
    StreamEndpointer,
)
from asr.vad import VADConfig

__all__ = [
    "SessionState",
    "UpdateKind",
    "EndpointReason",
    "StreamUpdate",
    "StreamingConfig",
    "StreamingSession",
    "StreamingTranscriber",
]


class SessionState(str, Enum):
    """Observable session state. Serialized into every update."""

    IDLE = "idle"              # no speech detected yet
    SPEAKING = "speaking"      # utterance in progress
    CLOSED = "closed"          # flush() called; no further audio accepted


class UpdateKind(str, Enum):
    STATE = "state"
    PARTIAL = "partial"
    FINAL = "final"


class EndpointReason(str, Enum):
    SILENCE = "silence"                  # min_silence_ms of trailing silence
    MAX_DURATION = "max_duration"        # utterance hit max_utterance_seconds
    STREAM_FLUSH = "stream_flush"        # flush() closed an open utterance


class StreamingTranscriber(Protocol):
    """What the session needs from an ASR backend.

    ``asr.explicit.ASRRunner`` satisfies this structurally. Typed as a Protocol
    so this module never imports torch, which keeps the session unit-testable
    on CPU against a fake.
    """

    def transcribe_array(
        self, waveform: np.ndarray, sample_rate: int, **kwargs: Any
    ) -> Any: ...

    def transcribe_long_array(
        self, waveform: np.ndarray, sample_rate: int, **kwargs: Any
    ) -> Any: ...


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate: int = 16_000
    vad: VADConfig = field(default_factory=VADConfig)

    emit_partials: bool = True
    #: Wall-clock gap between partial decodes.
    partial_interval_ms: int = 700
    #: New audio required since the last partial.
    min_partial_audio_ms: int = 500
    #: Tail of the utterance a partial decodes. Kept at or under Whisper's 30 s
    #: receptive field so a partial is always a single encoder pass.
    partial_window_seconds: float = 25.0

    #: A speaker who never pauses would otherwise grow the buffer without
    #: bound. Forcing a cut is worse than an unbounded buffer is not — this
    #: bounds memory and guarantees the session keeps emitting.
    max_utterance_seconds: float = 60.0
    #: Above this, finals route through the long-form chunk-and-stitch path.
    long_form_threshold_seconds: float = 25.0

    language: str | None = "hi"
    max_new_tokens: int = 225
    chunk_seconds: float = 25.0
    overlap_seconds: float = 5.0

    def as_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "emit_partials": self.emit_partials,
            "partial_interval_ms": self.partial_interval_ms,
            "min_partial_audio_ms": self.min_partial_audio_ms,
            "partial_window_seconds": self.partial_window_seconds,
            "max_utterance_seconds": self.max_utterance_seconds,
            "long_form_threshold_seconds": self.long_form_threshold_seconds,
            "language": self.language,
            "vad": self.vad.as_dict(),
        }


@dataclass(frozen=True)
class StreamUpdate:
    """One structured, serializable event. The session's entire output."""

    kind: UpdateKind
    sequence: int
    session_id: str
    state: SessionState
    utterance_index: int

    text: str = ""
    is_final: bool = False
    endpoint_reason: EndpointReason | None = None

    #: Audio actually transcribed for this update.
    audio_seconds: float = 0.0
    #: Total audio pushed into the session so far.
    stream_seconds: float = 0.0
    #: Utterance start, in stream time. Lets a caller align updates to audio.
    utterance_start_seconds: float = 0.0

    asr_ms: float = 0.0
    real_time_factor: float = 0.0
    long_form: bool = False
    #: True when a partial covered only the tail of a longer utterance.
    partial_is_tail: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "state": self.state.value,
            "utterance_index": self.utterance_index,
            "text": self.text,
            "is_final": self.is_final,
            "endpoint_reason": (
                self.endpoint_reason.value if self.endpoint_reason else None
            ),
            "audio_seconds": round(self.audio_seconds, 3),
            "stream_seconds": round(self.stream_seconds, 3),
            "utterance_start_seconds": round(self.utterance_start_seconds, 3),
            "asr_ms": round(self.asr_ms, 3),
            "real_time_factor": round(self.real_time_factor, 4),
            "long_form": self.long_form,
            "partial_is_tail": self.partial_is_tail,
        }


class StreamingSession:
    """Feed audio with :meth:`push`; close with :meth:`flush`."""

    def __init__(
        self,
        transcriber: StreamingTranscriber,
        config: StreamingConfig | None = None,
        *,
        session_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.transcriber = transcriber
        self.config = config or StreamingConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        # Injectable so tests are deterministic rather than sleeping.
        self._clock = clock or time.monotonic

        self.state = SessionState.IDLE
        self.utterance_index = 0

        self._endpointer = StreamEndpointer(self.config.sample_rate, self.config.vad)

        # Samples the idle trim must keep behind the cursor; see push().
        vad = self.config.vad
        self._onset_reserve = int(
            self.config.sample_rate
            * (vad.min_speech_ms + vad.padding_ms + vad.frame_ms + vad.hop_ms)
            / 1000
        )
        self._sequence = 0

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_origin = 0        # absolute stream index of _buffer[0]
        self._utterance_start: int | None = None

        self._last_partial_samples = 0
        self._last_partial_time = 0.0

    # -- introspection ----------------------------------------------------

    @property
    def stream_seconds(self) -> float:
        return self._endpointer.stream_seconds

    def snapshot(self) -> dict:
        """Serializable state dump for debugging a live session."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "utterance_index": self.utterance_index,
            "sequence": self._sequence,
            "buffered_seconds": round(
                self._buffer.size / self.config.sample_rate, 3
            ),
            "utterance_seconds": round(self._utterance_seconds(), 3),
            "endpointer": self._endpointer.as_dict(),
        }

    # -- helpers ----------------------------------------------------------

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _utterance_seconds(self) -> float:
        if self._utterance_start is None:
            return 0.0
        total = self._endpointer.total_samples - self._utterance_start
        return max(0.0, total / self.config.sample_rate)

    def _utterance_audio(self, end_sample: int | None = None) -> np.ndarray:
        if self._utterance_start is None:
            return np.zeros(0, dtype=np.float32)
        start = self._utterance_start - self._buffer_origin
        stop = (
            self._buffer.size
            if end_sample is None
            else min(self._buffer.size, end_sample - self._buffer_origin)
        )
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        return self._buffer[max(0, start) : stop]

    def _trim_buffer(self, keep_from_absolute: int) -> None:
        offset = keep_from_absolute - self._buffer_origin
        if offset > 0:
            self._buffer = self._buffer[offset:]
            self._buffer_origin += offset

    def _state_update(self, state: SessionState) -> StreamUpdate:
        return StreamUpdate(
            kind=UpdateKind.STATE,
            sequence=self._next_sequence(),
            session_id=self.session_id,
            state=state,
            utterance_index=self.utterance_index,
            stream_seconds=self.stream_seconds,
            utterance_start_seconds=(
                0.0
                if self._utterance_start is None
                else self._utterance_start / self.config.sample_rate
            ),
        )

    # -- ASR --------------------------------------------------------------

    def _run_asr(self, audio: np.ndarray, *, long_form: bool):
        rate = self.config.sample_rate
        if long_form:
            return self.transcriber.transcribe_long_array(
                audio,
                rate,
                language=self.config.language,
                chunk_seconds=self.config.chunk_seconds,
                overlap_seconds=self.config.overlap_seconds,
            )
        return self.transcriber.transcribe_array(
            audio,
            rate,
            language=self.config.language,
            max_new_tokens=self.config.max_new_tokens,
        )

    @staticmethod
    def _asr_ms(result: Any) -> float:
        metrics = getattr(result, "metrics", None)
        return float(getattr(metrics, "total_ms", 0.0) or 0.0)

    def _should_emit_partial(self) -> bool:
        if not self.config.emit_partials or self.state is not SessionState.SPEAKING:
            return False
        if self.config.partial_interval_ms <= 0:
            return False

        rate = self.config.sample_rate
        new_samples = self._endpointer.total_samples - self._last_partial_samples
        if new_samples < self.config.min_partial_audio_ms * rate / 1000:
            return False
        elapsed_ms = (self._clock() - self._last_partial_time) * 1000.0
        return elapsed_ms >= self.config.partial_interval_ms

    def _emit_partial(self) -> list[StreamUpdate]:
        audio = self._utterance_audio()
        if audio.size == 0:
            return []

        rate = self.config.sample_rate
        window = int(self.config.partial_window_seconds * rate)
        is_tail = audio.size > window
        if is_tail:
            audio = audio[-window:]

        result = self._run_asr(audio, long_form=False)
        self._last_partial_samples = self._endpointer.total_samples
        self._last_partial_time = self._clock()

        seconds = audio.size / rate
        asr_ms = self._asr_ms(result)
        return [
            StreamUpdate(
                kind=UpdateKind.PARTIAL,
                sequence=self._next_sequence(),
                session_id=self.session_id,
                state=self.state,
                utterance_index=self.utterance_index,
                text=getattr(result, "text", "") or "",
                is_final=False,
                audio_seconds=seconds,
                stream_seconds=self.stream_seconds,
                utterance_start_seconds=(
                    self._utterance_start or 0
                ) / rate,
                asr_ms=asr_ms,
                real_time_factor=(asr_ms / 1000.0 / seconds) if seconds else 0.0,
                partial_is_tail=is_tail,
            )
        ]

    def _finalize(
        self, end_sample: int, reason: EndpointReason
    ) -> list[StreamUpdate]:
        rate = self.config.sample_rate
        audio = self._utterance_audio(end_sample)
        start_seconds = (self._utterance_start or 0) / rate
        seconds = audio.size / rate
        long_form = seconds > self.config.long_form_threshold_seconds

        text, asr_ms = "", 0.0
        if audio.size:
            result = self._run_asr(audio, long_form=long_form)
            text = getattr(result, "text", "") or ""
            asr_ms = self._asr_ms(result)

        update = StreamUpdate(
            kind=UpdateKind.FINAL,
            sequence=self._next_sequence(),
            session_id=self.session_id,
            state=SessionState.IDLE,
            utterance_index=self.utterance_index,
            text=text,
            is_final=True,
            endpoint_reason=reason,
            audio_seconds=seconds,
            stream_seconds=self.stream_seconds,
            utterance_start_seconds=start_seconds,
            asr_ms=asr_ms,
            real_time_factor=(asr_ms / 1000.0 / seconds) if seconds else 0.0,
            long_form=long_form,
        )

        self.utterance_index += 1
        self._utterance_start = None
        self._last_partial_samples = self._endpointer.total_samples
        self._last_partial_time = self._clock()
        self._trim_buffer(end_sample)
        if self.state is not SessionState.CLOSED:
            self.state = SessionState.IDLE
        return [update]

    # -- public API -------------------------------------------------------

    def push(self, samples: np.ndarray) -> list[StreamUpdate]:
        """Feed one chunk of 16 kHz float32 PCM; return updates it produced.

        Chunk size is arbitrary. Returning a list rather than a generator keeps
        the call synchronous and the side effects ordered: a caller that stops
        iterating early must not leave the session half-advanced.
        """
        if self.state is SessionState.CLOSED:
            raise RuntimeError("session is closed; create a new StreamingSession")

        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        updates: list[StreamUpdate] = []

        if block.size:
            self._buffer = (
                block if self._buffer.size == 0
                else np.concatenate([self._buffer, block])
            )

        for event in self._endpointer.push(block):
            if event.kind is EndpointEventKind.SPEECH_START:
                self._utterance_start = event.sample
                self.state = SessionState.SPEAKING
                self._last_partial_samples = self._endpointer.total_samples
                self._last_partial_time = self._clock()
                # Audio before the padded start is unreachable; free it.
                self._trim_buffer(event.sample)
                updates.append(self._state_update(SessionState.SPEAKING))
            else:
                updates.extend(self._finalize(event.sample, EndpointReason.SILENCE))
                updates.append(self._state_update(SessionState.IDLE))

        # Forced cut for a speaker who never pauses.
        if (
            self.state is SessionState.SPEAKING
            and self._utterance_seconds() >= self.config.max_utterance_seconds
        ):
            end = self._endpointer.total_samples
            updates.extend(self._finalize(end, EndpointReason.MAX_DURATION))
            self._endpointer.reset()
            updates.append(self._state_update(SessionState.IDLE))
            return updates

        if self._should_emit_partial():
            updates.extend(self._emit_partial())

        if self.state is SessionState.IDLE and not updates:
            # Idle memory is bounded, but never trim what a future SPEECH_START
            # can point back to: the endpointer confirms an onset only after
            # min_speech_ms of voiced audio and then reports a start padded a
            # further padding_ms earlier. Trimming to "now" deleted the first
            # ~450 ms of every utterance and Whisper hallucinated on the
            # clipped onset (the GPU sweep's "जी जी जी").
            self._trim_buffer(self._endpointer.total_samples - self._onset_reserve)

        return updates

    def flush(self) -> list[StreamUpdate]:
        """Close the session, finalizing any utterance still in progress."""
        if self.state is SessionState.CLOSED:
            return []

        updates: list[StreamUpdate] = []
        for event in self._endpointer.flush():
            updates.extend(
                self._finalize(event.sample, EndpointReason.STREAM_FLUSH)
            )
        self.state = SessionState.CLOSED
        updates.append(self._state_update(SessionState.CLOSED))
        return updates

    def reset(self) -> StreamingSession:
        """Return a fresh session with the same transcriber and config."""
        return StreamingSession(
            self.transcriber,
            replace(self.config),
            clock=self._clock,
        )
