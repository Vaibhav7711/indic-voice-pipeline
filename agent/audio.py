"""Real audio output: decode what the synthesizer produces, play it on a device.

`PlaybackSession` only needs an ``AudioSink`` with ``write``/``stop``/``close``.
``BufferSink`` satisfies that in memory; this module provides sinks that
reach an actual sound device, which is what turns the barge-in and latency
numbers from "the state machine did the right thing" into "the speaker went
quiet".

Two things a device sink has to solve that a buffer does not:

**Format.** edge-tts delivers MP3 frames; a local VITS model delivers raw
PCM. The sink is told the format up front (``AudioFormat``) and decodes MP3
incrementally with PyAV — packets are parsed and decoded as bytes arrive, so
the first audio plays before the sentence has finished downloading.

**Instant stop.** ``stop()`` (barge-in) must drop audio the OS has already
accepted, not just stop feeding it. ``sounddevice``'s ``abort()`` does that;
the tens-to-hundreds of milliseconds of buffered audio that ``stop()`` would
otherwise let play are the difference between an interruption and a stumble.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AudioFormat",
    "MP3_24K",
    "PCM16_16K",
    "Mp3Decoder",
    "DecodingBufferSink",
    "PacedBufferSink",
    "SoundDeviceSink",
]


@dataclass(frozen=True)
class AudioFormat:
    """What a synthesizer emits. ``codec`` is ``"mp3"`` or ``"pcm_s16le"``."""

    codec: str
    sample_rate: int
    channels: int = 1

    @property
    def is_pcm(self) -> bool:
        return self.codec == "pcm_s16le"

    def as_dict(self) -> dict:
        return {"codec": self.codec, "sample_rate": self.sample_rate, "channels": self.channels}


MP3_24K = AudioFormat("mp3", 24_000, 1)        # edge-tts default
PCM16_16K = AudioFormat("pcm_s16le", 16_000, 1)  # MMS-TTS / most VITS models


class Mp3Decoder:
    """Incremental MP3 → int16 mono PCM using PyAV's parser + decoder.

    Bytes can arrive at any granularity (edge-tts chunks are not frame
    aligned); the parser buffers until whole frames exist. Output is resampled
    to ``target_rate`` so a device stream opened once can play everything.
    """

    def __init__(self, target_rate: int = 24_000):
        import av

        self.target_rate = target_rate
        self._codec = av.CodecContext.create("mp3", "r")
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        self.frames = 0
        self.samples = 0
        self.dropped_packets = 0
        self._head = b""            # bytes held back until an ID3 tag is ruled out
        self._started = False

    def _strip_id3(self, data: bytes) -> bytes:
        """Drop an ID3v2 tag at the start of the stream (files have one,
        edge-tts frames do not); the raw MP3 decoder rejects it."""
        if self._started:
            return data
        self._head += data
        if len(self._head) < 10:
            return b""
        if self._head[:3] == b"ID3":
            size = 0
            for b in self._head[6:10]:
                size = (size << 7) | (b & 0x7F)
            total = 10 + size
            if len(self._head) < total:
                return b""
            out, self._head = self._head[total:], b""
        else:
            out, self._head = self._head, b""
        self._started = True
        return out

    def _decode_packets(self, packets) -> list:
        frames = []
        for packet in packets:
            try:
                frames.extend(self._codec.decode(packet))
            except Exception:  # noqa: BLE001 - a bad packet costs one frame, not the turn
                self.dropped_packets += 1
        return frames

    def _frames_to_pcm(self, frames) -> np.ndarray:
        out = []
        for frame in frames:
            for resampled in self._resampler.resample(frame):
                out.append(resampled.to_ndarray().reshape(-1))
                self.frames += 1
        if not out:
            return np.zeros(0, dtype=np.int16)
        pcm = np.concatenate(out).astype(np.int16)
        self.samples += pcm.size
        return pcm

    def feed(self, data: bytes) -> np.ndarray:
        """Decode whatever complete frames ``data`` completes; may be empty."""
        data = self._strip_id3(data)
        if not data:
            return np.zeros(0, dtype=np.int16)
        return self._frames_to_pcm(self._decode_packets(self._codec.parse(data)))

    def flush(self) -> np.ndarray:
        """Drain the parser, the decoder and the resampler at end of stream."""
        frames = []
        if self._head and not self._started:            # tiny stream, no tag
            frames.extend(self._decode_packets(self._codec.parse(self._head)))
            self._head, self._started = b"", True
        frames.extend(self._decode_packets(self._codec.parse(None)))
        frames.extend(self._decode_packets([None]))
        pcm = [self._frames_to_pcm(frames)]
        tail = [r.to_ndarray().reshape(-1) for r in self._resampler.resample(None)]
        if tail:
            t = np.concatenate(tail).astype(np.int16)
            self.samples += t.size
            pcm.append(t)
        return np.concatenate(pcm)


class DecodingBufferSink:
    """In-memory sink that decodes to PCM. Lets tests and file demos check
    *audio*, not bytes; also the reference for what a device sink plays."""

    def __init__(self, fmt: AudioFormat = MP3_24K, target_rate: int | None = None):
        self.format = fmt
        self.target_rate = target_rate or fmt.sample_rate
        self._decoder = None if fmt.is_pcm else Mp3Decoder(self.target_rate)
        self.pcm: list[np.ndarray] = []
        self.stopped = False
        self.closed = False

    def write(self, chunk: bytes) -> None:
        if self.stopped:
            raise RuntimeError("write after stop")
        if self._decoder is None:
            pcm = np.frombuffer(chunk, dtype=np.int16)
        else:
            pcm = self._decoder.feed(chunk)
        if pcm.size:
            self.pcm.append(pcm)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        if self._decoder is not None and not self.stopped:
            tail = self._decoder.flush()
            if tail.size:
                self.pcm.append(tail)
        self.closed = True

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.pcm) if self.pcm else np.zeros(0, dtype=np.int16)

    @property
    def seconds(self) -> float:
        return self.audio.size / self.target_rate


class PacedBufferSink(DecodingBufferSink):
    """A buffer sink that takes as long to "play" as the audio actually lasts.

    `DecodingBufferSink` consumes every chunk instantly, which is right for
    checking *what* was synthesised and wrong for anything that depends on the
    agent still speaking. Barge-in is exactly that: a headless turn finishes in
    milliseconds, so there is nothing left to interrupt, and barge-in could only
    ever be tested on a machine with a sound card. This sink closes that gap --
    it decodes like its parent and then waits out the audio's real duration, so
    a GPU box with no audio hardware can run the same test a laptop with
    speakers would.

    The wait is on an `Event`, not `sleep`: `stop()` has to cut playback
    promptly or a barge-in would be recorded while the sink kept "playing" the
    sentence the user interrupted, and the measured cancel latency would be the
    length of the audio rather than the length of the cancel.

    It tracks wall-clock rather than trusting the decoder: `elapsed` and
    `played_seconds` together say whether pacing actually happened, which is
    what a test should assert rather than assuming.
    """

    def __init__(self, fmt: AudioFormat = MP3_24K, target_rate: int | None = None,
                 *, speed: float = 1.0, clock: Callable[[], float] | None = None,
                 waiter: Callable[[float], bool] | None = None):
        super().__init__(fmt, target_rate)
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        #: >1 plays faster than real time. For a test that wants ordering
        #: without the wall-clock cost; 1.0 is the honest default.
        self.speed = speed
        self._clock = clock or time.monotonic
        self._cancel = threading.Event()
        self._waiter = waiter or (lambda seconds: self._cancel.wait(seconds))
        self.paced_seconds = 0.0
        self.started_at: float | None = None

    def write(self, chunk: bytes) -> None:
        before = self.audio.size
        super().write(chunk)
        if self.started_at is None:
            self.started_at = self._clock()
        gained = (self.audio.size - before) / self.target_rate
        if gained <= 0:
            return
        self.paced_seconds += gained
        self._waiter(gained / self.speed)

    def stop(self) -> None:
        # Released before the flag is set, so a writer blocked in the wait
        # returns immediately rather than after the current chunk's duration.
        self._cancel.set()
        super().stop()

    @property
    def elapsed(self) -> float:
        """Wall-clock since the first chunk, or 0.0 before one arrived."""
        return 0.0 if self.started_at is None else self._clock() - self.started_at


class SoundDeviceSink:
    """Play through the default (or named) output device via ``sounddevice``.

    ``stream_factory`` is injectable so the sink's lifecycle is unit-testable
    without a sound card; by default it opens a ``sounddevice.RawOutputStream``.
    """

    def __init__(
        self,
        fmt: AudioFormat = MP3_24K,
        *,
        device: int | str | None = None,
        blocksize: int = 1024,
        stream_factory: Callable[..., object] | None = None,
    ):
        self.format = fmt
        self.rate = fmt.sample_rate
        self._decoder = None if fmt.is_pcm else Mp3Decoder(self.rate)
        self._stream = None
        self._factory = stream_factory or self._default_factory(device, blocksize)
        self.samples_written = 0
        self.stopped = False
        self.closed = False

    @staticmethod
    def _default_factory(device, blocksize):
        def make(rate: int, channels: int):
            import sounddevice as sd

            stream = sd.RawOutputStream(
                samplerate=rate, channels=channels, dtype="int16",
                device=device, blocksize=blocksize,
            )
            stream.start()
            return stream

        return make

    def _ensure_stream(self):
        if self._stream is None:
            self._stream = self._factory(self.rate, self.format.channels)
        return self._stream

    def open(self) -> None:
        """Open the device now (~100-300 ms) rather than on the first write,
        so it is off the first-audio path. Safe to call more than once."""
        self._ensure_stream()

    def write(self, chunk: bytes) -> None:
        if self.stopped:
            raise RuntimeError("write after stop")
        pcm = np.frombuffer(chunk, dtype=np.int16) if self._decoder is None else self._decoder.feed(chunk)
        if pcm.size:
            self._ensure_stream().write(pcm.tobytes())
            self.samples_written += pcm.size

    def stop(self) -> None:
        """Barge-in: drop everything queued in the device, then close."""
        self.stopped = True
        if self._stream is not None:
            abort = getattr(self._stream, "abort", None)
            if callable(abort):
                abort()
            self._stream.close()
            self._stream = None

    def close(self) -> None:
        """Normal end: let queued audio drain, then release the device."""
        if self.closed:
            return
        if self._decoder is not None and not self.stopped:
            tail = self._decoder.flush()
            if tail.size:
                self._ensure_stream().write(tail.tobytes())
                self.samples_written += tail.size
        if self._stream is not None:
            self._stream.stop()      # blocks until the buffer has played
            self._stream.close()
            self._stream = None
        self.closed = True
