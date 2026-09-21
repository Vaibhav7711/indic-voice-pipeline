"""Audio output path: MP3 decoding and the device sink lifecycle."""

from __future__ import annotations

import io
import random

import numpy as np
import pytest

av = pytest.importorskip("av")

from agent.audio import (  # noqa: E402
    MP3_24K,
    PCM16_16K,
    DecodingBufferSink,
    Mp3Decoder,
    SoundDeviceSink,
)


def encode_mp3(freq: float = 440.0, seconds: float = 1.0, rate: int = 24_000) -> bytes:
    """A sine tone as MP3 bytes, via PyAV, so the test needs no fixture file."""
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp3") as out:
        stream = out.add_stream("mp3", rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def dominant_hz(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64)))
    return float(np.fft.rfftfreq(pcm.size, 1 / rate)[int(spectrum.argmax())])


class TestMp3Decoder:
    def test_decodes_arbitrary_chunking_to_the_same_audio(self):
        data = encode_mp3()
        whole = Mp3Decoder(24_000)
        ref = np.concatenate([whole.feed(data), whole.flush()])

        rng = random.Random(0)
        dec = Mp3Decoder(24_000)
        pieces, i = [], 0
        while i < len(data):
            n = rng.randint(1, 700)
            pieces.append(dec.feed(data[i:i + n]))
            i += n
        pieces.append(dec.flush())
        got = np.concatenate(pieces)

        assert got.size == ref.size
        assert np.array_equal(got, ref)
        # ~1 s of audio, allowing for encoder delay/padding.
        assert abs(ref.size / 24_000 - 1.0) < 0.1
        assert abs(dominant_hz(ref, 24_000) - 440.0) < 5.0

    def test_first_audio_arrives_before_the_stream_ends(self):
        data = encode_mp3(seconds=2.0)
        dec = Mp3Decoder(24_000)
        first_at = None
        for i in range(0, len(data), 512):
            if dec.feed(data[i:i + 512]).size:
                first_at = i + 512
                break
        assert first_at is not None and first_at < len(data) / 4, "decoding must be incremental"

    def test_resamples_to_target_rate(self):
        data = encode_mp3(seconds=1.0, rate=24_000)
        dec = Mp3Decoder(16_000)
        pcm = np.concatenate([dec.feed(data), dec.flush()])
        assert abs(pcm.size / 16_000 - 1.0) < 0.1
        assert abs(dominant_hz(pcm, 16_000) - 440.0) < 5.0


class TestDecodingBufferSink:
    def test_mp3_sink_yields_pcm_seconds(self):
        sink = DecodingBufferSink(MP3_24K)
        data = encode_mp3(seconds=1.5)
        for i in range(0, len(data), 300):
            sink.write(data[i:i + 300])
        sink.close()
        assert abs(sink.seconds - 1.5) < 0.1

    def test_pcm_sink_passes_through(self):
        sink = DecodingBufferSink(PCM16_16K)
        pcm = np.zeros(16_000, dtype=np.int16)
        sink.write(pcm.tobytes())
        sink.close()
        assert sink.seconds == 1.0


class _FakeStream:
    def __init__(self, log):
        self.log = log
        self.bytes = 0

    def write(self, data):
        self.bytes += len(data)
        self.log.append(("write", len(data)))

    def abort(self):
        self.log.append(("abort", None))

    def stop(self):
        self.log.append(("stop", None))

    def close(self):
        self.log.append(("close", None))


class TestSoundDeviceSink:
    def test_pcm_writes_go_to_the_stream_and_close_drains(self):
        log = []
        sink = SoundDeviceSink(PCM16_16K, stream_factory=lambda rate, ch: _FakeStream(log))
        sink.write(np.zeros(800, dtype=np.int16).tobytes())
        sink.write(np.zeros(800, dtype=np.int16).tobytes())
        sink.close()
        assert [e[0] for e in log] == ["write", "write", "stop", "close"]
        assert sink.samples_written == 1600

    def test_stop_aborts_queued_audio_instead_of_draining(self):
        log = []
        sink = SoundDeviceSink(PCM16_16K, stream_factory=lambda rate, ch: _FakeStream(log))
        sink.write(np.zeros(800, dtype=np.int16).tobytes())
        sink.stop()
        assert [e[0] for e in log] == ["write", "abort", "close"]
        with pytest.raises(RuntimeError):
            sink.write(b"\x00\x00")
        sink.close()                                # idempotent after stop
        assert log[-1][0] == "close"

    def test_mp3_sink_decodes_before_writing(self):
        log = []
        sink = SoundDeviceSink(MP3_24K, stream_factory=lambda rate, ch: _FakeStream(log))
        data = encode_mp3(seconds=0.5)
        for i in range(0, len(data), 400):
            sink.write(data[i:i + 400])
        sink.close()
        assert sink.samples_written > 0
        assert abs(sink.samples_written / 24_000 - 0.5) < 0.1
        assert log[-2:] == [("stop", None), ("close", None)]

    def test_stream_opened_lazily_on_first_audio(self):
        opened = []
        sink = SoundDeviceSink(PCM16_16K, stream_factory=lambda r, c: opened.append(r) or _FakeStream([]))
        assert opened == []
        sink.write(np.zeros(10, dtype=np.int16).tobytes())
        assert opened == [16_000]
