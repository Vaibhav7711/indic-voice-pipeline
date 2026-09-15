"""Response playback lifecycle: start, stream, cancel, complete, fail.

Why barge-in needs its own object
---------------------------------
In a voice agent, the user can start talking while the agent is still speaking.
Handling that ("barge-in") is not a TTS concern and not an ASR concern — it is a
*lifecycle* concern, and it is concurrent: the cancel signal arrives on the
audio-input thread while the playback loop is running on another.

Hence :class:`PlaybackSession`: one object that owns the state machine, records
every transition as a serializable event, and exposes a thread-safe
:meth:`PlaybackSession.cancel`.

What cancellation can and cannot do
-----------------------------------
This is cooperative cancellation, and the honest boundary is worth stating.

Cancelling **stops feeding new audio** to the sink, which stops synthesis work
and stops queueing. It does **not** silence audio already accepted by the
operating system's audio buffer — that keeps playing until it drains, typically
tens to low hundreds of milliseconds depending on device buffer size.

Truly instant barge-in requires the sink to drop its own buffered audio, which
is why :class:`AudioSink` has a ``stop()`` method distinct from ``close()``.
:meth:`PlaybackSession.cancel` calls ``stop()``; whether that is actually
instantaneous is a property of the sink implementation, not of this module. The
smaller the chunks fed, the tighter the cancellation granularity.

State machine
-------------
``IDLE → STARTING → PLAYING → {COMPLETED, CANCELLED, FAILED}``

Terminal states are terminal: a cancel arriving after completion is recorded and
ignored rather than rewriting history, because in a race the question "did the
user interrupt, or did we finish first?" has one correct answer.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Protocol

__all__ = [
    "PlaybackState",
    "PlaybackEventKind",
    "PlaybackEvent",
    "PlaybackResult",
    "AudioSink",
    "BufferSink",
    "PlaybackSession",
]


class PlaybackState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    PLAYING = "playing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            PlaybackState.COMPLETED,
            PlaybackState.CANCELLED,
            PlaybackState.FAILED,
        }


class PlaybackEventKind(str, Enum):
    STARTED = "started"
    FIRST_AUDIO = "first_audio"
    CHUNK = "chunk"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED_CANCEL = "ignored_cancel"


@dataclass(frozen=True)
class PlaybackEvent:
    kind: PlaybackEventKind
    sequence: int
    playback_id: str
    state: PlaybackState
    elapsed_ms: float
    chunk_index: int | None = None
    bytes_written: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "sequence": self.sequence,
            "playback_id": self.playback_id,
            "state": self.state.value,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "chunk_index": self.chunk_index,
            "bytes_written": self.bytes_written,
            "detail": self.detail,
        }


@dataclass
class PlaybackResult:
    playback_id: str
    state: PlaybackState
    chunks_written: int = 0
    bytes_written: int = 0
    first_audio_ms: float | None = None
    total_ms: float = 0.0
    cancel_reason: str = ""
    error: str | None = None
    events: list[PlaybackEvent] = field(default_factory=list)

    @property
    def interrupted(self) -> bool:
        return self.state is PlaybackState.CANCELLED

    def as_dict(self) -> dict:
        return {
            "playback_id": self.playback_id,
            "state": self.state.value,
            "chunks_written": self.chunks_written,
            "bytes_written": self.bytes_written,
            "first_audio_ms": (
                None if self.first_audio_ms is None else round(self.first_audio_ms, 3)
            ),
            "total_ms": round(self.total_ms, 3),
            "cancel_reason": self.cancel_reason,
            "error": self.error,
            "events": [event.as_dict() for event in self.events],
        }


class AudioSink(Protocol):
    """Where playback audio goes.

    ``stop()`` and ``close()`` are separate on purpose: ``stop()`` means
    "abandon what is queued, we were interrupted", ``close()`` means "we are
    finished normally". A sink that treats them identically cannot implement
    responsive barge-in.
    """

    def write(self, chunk: bytes) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class BufferSink:
    """In-memory sink. Used by tests and by file-output demos."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.stopped = False
        self.closed = False

    def write(self, chunk: bytes) -> None:
        if self.stopped:
            raise RuntimeError("write after stop")
        self.chunks.append(chunk)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    @property
    def data(self) -> bytes:
        return b"".join(self.chunks)


class PlaybackSession:
    """Drives one response's audio to a sink, interruptibly."""

    def __init__(
        self,
        sink: AudioSink,
        *,
        playback_id: str | None = None,
        clock=None,
    ):
        self.sink = sink
        self.playback_id = playback_id or uuid.uuid4().hex[:12]
        self._clock = clock or (lambda: perf_counter_ns() / 1_000_000)

        self.state = PlaybackState.IDLE
        self.events: list[PlaybackEvent] = []

        # threading.Event, not a bool: cancel() is called from the audio-input
        # thread while play() runs on another. A plain flag would be a data
        # race, and the memory barrier is what makes the cancel visible
        # promptly to the playback loop.
        self._cancel = threading.Event()
        self._cancel_reason = ""
        self._lock = threading.Lock()
        self._sequence = 0
        self._started_at: float | None = None

    # -- events -----------------------------------------------------------

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    def _emit(
        self,
        kind: PlaybackEventKind,
        *,
        chunk_index: int | None = None,
        bytes_written: int = 0,
        detail: str = "",
    ) -> PlaybackEvent:
        self._sequence += 1
        event = PlaybackEvent(
            kind=kind,
            sequence=self._sequence,
            playback_id=self.playback_id,
            state=self.state,
            elapsed_ms=self._elapsed(),
            chunk_index=chunk_index,
            bytes_written=bytes_written,
            detail=detail,
        )
        self.events.append(event)
        return event

    # -- control ----------------------------------------------------------

    def cancel(self, reason: str = "barge_in") -> bool:
        """Request interruption. Thread-safe; returns whether it took effect.

        Returns ``False`` when playback already finished — the caller can then
        distinguish "the user interrupted us" from "we had already stopped",
        which matters for turn accounting.
        """
        with self._lock:
            if self.state.is_terminal:
                self._emit(
                    PlaybackEventKind.IGNORED_CANCEL,
                    detail=f"{reason} after {self.state.value}",
                )
                return False
            self._cancel_reason = reason
        self._cancel.set()
        return True

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def should_stop(self) -> bool:
        """Poll predicate for producers, so synthesis stops too.

        Passed into the TTS stream so a barge-in stops *generating* audio, not
        merely discards audio already paid for.
        """
        return self._cancel.is_set()

    # -- main loop --------------------------------------------------------

    def play(self, chunks: Iterable[bytes]) -> PlaybackResult:
        """Consume an audio chunk iterable, honouring cancellation."""
        if self.state is not PlaybackState.IDLE:
            raise RuntimeError(
                f"playback {self.playback_id} already ran (state={self.state.value})"
            )

        self._started_at = self._clock()
        self.state = PlaybackState.STARTING
        self._emit(PlaybackEventKind.STARTED)

        result = PlaybackResult(playback_id=self.playback_id, state=self.state)
        error: str | None = None

        try:
            for index, chunk in enumerate(chunks):
                if self._cancel.is_set():
                    break
                if not chunk:
                    continue

                self.sink.write(chunk)

                if self.state is PlaybackState.STARTING:
                    self.state = PlaybackState.PLAYING
                    result.first_audio_ms = self._elapsed()
                    self._emit(
                        PlaybackEventKind.FIRST_AUDIO,
                        chunk_index=index,
                        bytes_written=len(chunk),
                    )
                else:
                    self._emit(
                        PlaybackEventKind.CHUNK,
                        chunk_index=index,
                        bytes_written=len(chunk),
                    )

                result.chunks_written += 1
                result.bytes_written += len(chunk)

                # Checked again after the write so a cancel arriving mid-chunk
                # is honoured before the next one is fetched from the producer.
                if self._cancel.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 - reported as state
            error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            if error is not None:
                self.state = PlaybackState.FAILED
            elif self._cancel.is_set():
                self.state = PlaybackState.CANCELLED
            else:
                self.state = PlaybackState.COMPLETED

        if self.state is PlaybackState.CANCELLED:
            self.sink.stop()
            result.cancel_reason = self._cancel_reason
            self._emit(PlaybackEventKind.CANCELLED, detail=self._cancel_reason)
        elif self.state is PlaybackState.FAILED:
            self.sink.stop()
            result.error = error
            self._emit(PlaybackEventKind.FAILED, detail=error or "")
        else:
            self.sink.close()
            self._emit(PlaybackEventKind.COMPLETED)

        result.state = self.state
        result.total_ms = self._elapsed()
        result.events = list(self.events)
        return result

    def snapshot(self) -> dict:
        return {
            "playback_id": self.playback_id,
            "state": self.state.value,
            "cancelled": self.cancelled,
            "cancel_reason": self._cancel_reason,
            "events": len(self.events),
        }
