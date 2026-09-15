"""Streaming speech synthesis: sentence splitting plus incremental audio.

What is actually streaming here, and what is not
------------------------------------------------
Be precise about this, because "streaming TTS" is used to mean three different
things and only two of them are true here.

1. **Chunk streaming from the provider — real.** ``edge_tts.Communicate.stream()``
   is an async generator that yields MP3 frames over a websocket as the service
   produces them. :class:`EdgeStreamingSynthesizer` forwards those frames as
   they arrive instead of buffering the whole response. First-audio latency is
   therefore genuinely lower than waiting for a complete file.

2. **Sentence-level streaming — real, and the bigger win.** Synthesising a
   whole LLM response before playing anything means the user waits for the
   longest part of the pipeline. :func:`split_sentences` cuts the response at
   sentence boundaries so the first sentence can be synthesised and played
   while later ones are still being generated.

3. **Low-latency local synthesis — not this.** edge-tts is a network service. A
   round trip to Microsoft's endpoint dominates first-chunk latency, and it
   varies with connectivity. It is not a local neural vocoder and this module
   does not pretend otherwise. Measure ``first_chunk_ms`` on your own network
   before quoting any number; a local TTS engine is the fix, not a wrapper.

The existing :class:`tts.TTSSynthesizer` is untouched — ``synthesize()`` keeps
its exact behaviour and return type. This module adds a streaming path beside
it.
"""

from __future__ import annotations

import queue
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Protocol

__all__ = [
    "split_sentences",
    "AudioChunk",
    "SpeechStream",
    "StreamingSynthesizer",
    "EdgeStreamingSynthesizer",
]


# Devanagari danda and double danda terminate sentences in Hindi; ASCII
# punctuation covers code-switched English. Newlines are treated as hard
# breaks because LLM responses often use them as list separators.
_SENTENCE_END = re.compile(r"(?<=[।॥.!?…])\s+|\n+")

#: Short fragments are merged forward rather than synthesised alone: a
#: one-word utterance costs a full network round trip for almost no audio, and
#: prosody across a too-short clip sounds clipped.
#:
#: Tuned for Devanagari, which is far denser than Latin per character — a
#: complete Hindi sentence such as "मैं ठीक हूँ।" is only 12 characters. A
#: threshold set for English (16+) merges an entire short reply into one unit
#: and silently disables sentence-level streaming.
MIN_SENTENCE_CHARS = 8


def split_sentences(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Split a response into synthesis units.

    Deliberately simple: no abbreviation model, no learned segmenter. A wrong
    split costs slightly odd prosody at one boundary; an over-engineered
    splitter costs latency on every turn. Fragments shorter than ``min_chars``
    are merged into the following sentence.
    """
    if not text or not text.strip():
        return []

    raw = [piece.strip() for piece in _SENTENCE_END.split(text) if piece.strip()]
    if not raw:
        return []

    merged: list[str] = []
    pending = ""
    for piece in raw:
        candidate = f"{pending} {piece}".strip() if pending else piece
        if len(candidate) < min_chars:
            pending = candidate
            continue
        merged.append(candidate)
        pending = ""
    if pending:
        # Trailing fragment: append to the previous unit rather than emitting
        # a stub, unless it is all we have.
        if merged:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending)
    return merged


@dataclass(frozen=True)
class AudioChunk:
    """One piece of encoded audio, with the timing needed to report latency."""

    data: bytes
    index: int
    sentence_index: int
    elapsed_ms: float

    @property
    def size(self) -> int:
        return len(self.data)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "sentence_index": self.sentence_index,
            "bytes": self.size,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


@dataclass
class SpeechStream:
    """Result of consuming a synthesis stream."""

    sentences: list[str] = field(default_factory=list)
    chunks: list[AudioChunk] = field(default_factory=list)
    first_chunk_ms: float | None = None
    total_ms: float = 0.0
    streaming: bool = False
    error: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(chunk.size for chunk in self.chunks)

    def as_dict(self) -> dict:
        return {
            "sentences": len(self.sentences),
            "chunks": len(self.chunks),
            "total_bytes": self.total_bytes,
            "first_chunk_ms": (
                None if self.first_chunk_ms is None else round(self.first_chunk_ms, 3)
            ),
            "total_ms": round(self.total_ms, 3),
            "streaming": self.streaming,
            "error": self.error,
        }


class StreamingSynthesizer(Protocol):
    """Minimal contract the agent needs from a TTS backend."""

    #: False means the backend buffers internally, so ``first_chunk_ms`` is the
    #: time to the *whole* clip and must not be reported as streaming latency.
    streaming: bool

    def stream(self, text: str) -> Iterator[bytes]: ...


class EdgeStreamingSynthesizer:
    """edge-tts with real incremental chunk delivery.

    ``edge_tts`` is async; the agent loop is synchronous. Rather than force
    async through the whole call path, the async generator is drained on a
    worker thread into a bounded queue. The bound matters: an unbounded queue
    would let synthesis race ahead of playback and buffer the entire response
    in memory, which defeats the point of streaming.
    """

    streaming = True

    def __init__(
        self,
        voice: str | None = None,
        language: str | None = "hi",
        *,
        max_buffered_chunks: int = 32,
    ):
        from tts.synthesis import DEFAULT_VOICES

        self.language = language
        self.voice = voice or DEFAULT_VOICES.get(language, DEFAULT_VOICES[None])
        self.max_buffered_chunks = max_buffered_chunks

    def stream(self, text: str) -> Iterator[bytes]:
        if not text or not text.strip():
            return

        sentinel = object()
        buffer: queue.Queue = queue.Queue(maxsize=self.max_buffered_chunks)

        def worker() -> None:
            import asyncio

            async def pump() -> None:
                import edge_tts

                comm = edge_tts.Communicate(text, self.voice)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        buffer.put(chunk["data"])

            try:
                asyncio.run(pump())
            except BaseException as exc:  # noqa: BLE001 - re-raised on consumer
                buffer.put(exc)
            finally:
                buffer.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = buffer.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # A consumer that stops early (barge-in) must not leak the worker.
            # The queue bound means the producer blocks rather than spinning,
            # and the daemon thread dies with the process.
            thread.join(timeout=0.1)


def synthesize_stream(
    synthesizer: StreamingSynthesizer,
    text: str,
    *,
    split: bool = True,
    on_chunk=None,
    should_stop=None,
) -> SpeechStream:
    """Drive a synthesizer over a response, sentence by sentence.

    ``should_stop`` is polled between chunks so barge-in stops synthesis rather
    than merely discarding audio that was already paid for.
    """
    start = perf_counter_ns()
    sentences = split_sentences(text) if split else ([text] if text.strip() else [])
    stream = SpeechStream(
        sentences=sentences,
        streaming=bool(getattr(synthesizer, "streaming", False)),
    )

    index = 0
    try:
        for sentence_index, sentence in enumerate(sentences):
            if should_stop is not None and should_stop():
                break
            for data in synthesizer.stream(sentence):
                if should_stop is not None and should_stop():
                    break
                elapsed = (perf_counter_ns() - start) / 1_000_000
                chunk = AudioChunk(data, index, sentence_index, elapsed)
                index += 1
                if stream.first_chunk_ms is None:
                    stream.first_chunk_ms = elapsed
                stream.chunks.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
    except Exception as exc:  # noqa: BLE001 - surfaced as state, not a crash
        stream.error = f"{type(exc).__name__}: {exc}"

    stream.total_ms = (perf_counter_ns() - start) / 1_000_000
    return stream
