"""Headless barge-in and multi-turn: the two paths never exercised.

Testing so far has been request/response — one clip in, one answer out. Barge-in
and follow-up turns were untested because a buffer sink consumes audio
instantly: the turn finishes in milliseconds, so there is nothing still playing
to interrupt, and barge-in could only be reached on a machine with a sound card.

`PacedBufferSink` closes that: it decodes like its parent and waits out the
audio's real duration, so a GPU box with no audio hardware behaves like one with
speakers in the time domain. These tests use a `waiter` stub where the wait
itself is the thing under test, and real waits (small, 50 ms) where the point is
that wall-clock actually elapsed.

No model, no sound card, no network.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from agent.audio import PCM16_16K, DecodingBufferSink, PacedBufferSink
from scripts.make_conversation_wav import (
    PRESETS,
    build,
    silence,
    to_mono_16k,
    write_wav,
)


def pcm_chunk(seconds: float) -> bytes:
    """`seconds` of silence as PCM16 at 16 kHz."""
    return np.zeros(int(seconds * 16_000), dtype=np.int16).tobytes()


class TestPacingIsReal:
    def test_it_waits_for_the_audio_duration(self):
        sink = PacedBufferSink(PCM16_16K)
        started = time.monotonic()
        sink.write(pcm_chunk(0.05))
        assert time.monotonic() - started >= 0.045
        assert sink.paced_seconds == pytest.approx(0.05, abs=1e-6)

    def test_an_instant_sink_does_not(self):
        """The contrast that makes barge-in untestable without this."""
        sink = DecodingBufferSink(PCM16_16K)
        started = time.monotonic()
        sink.write(pcm_chunk(0.5))
        assert time.monotonic() - started < 0.05

    def test_speed_scales_the_wait(self):
        waits: list[float] = []
        sink = PacedBufferSink(PCM16_16K, speed=10.0,
                               waiter=lambda s: waits.append(s) or False)
        sink.write(pcm_chunk(1.0))
        assert waits == [pytest.approx(0.1)]

    def test_a_non_positive_speed_is_refused(self):
        with pytest.raises(ValueError, match="speed must be positive"):
            PacedBufferSink(PCM16_16K, speed=0)

    def test_elapsed_is_zero_before_any_audio(self):
        """Not a fabricated 0.0 duration: nothing has played, and `elapsed`
        being zero says exactly that."""
        assert PacedBufferSink(PCM16_16K).elapsed == 0.0

    def test_the_audio_is_still_captured(self):
        sink = PacedBufferSink(PCM16_16K, speed=1000.0)
        sink.write(pcm_chunk(0.1))
        sink.close()
        assert sink.audio.size == 1600
        assert sink.seconds == pytest.approx(0.1)


class TestStopCutsPlaybackImmediately:
    def test_stop_releases_a_blocked_writer(self):
        """Barge-in has to cut audio, not wait out the chunk it is interrupting.
        Otherwise the measured cancel latency is the length of the sentence."""
        sink = PacedBufferSink(PCM16_16K)
        released = threading.Event()

        def writer():
            try:
                sink.write(pcm_chunk(5.0))
            except RuntimeError:
                pass
            released.set()

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        time.sleep(0.05)
        started = time.monotonic()
        sink.stop()
        assert released.wait(timeout=2.0), "the writer stayed blocked"
        assert time.monotonic() - started < 1.0, "stop did not cut the wait"

    def test_a_write_after_stop_is_refused(self):
        sink = PacedBufferSink(PCM16_16K)
        sink.stop()
        with pytest.raises(RuntimeError, match="write after stop"):
            sink.write(pcm_chunk(0.01))

    def test_stop_before_any_write_does_not_hang(self):
        sink = PacedBufferSink(PCM16_16K)
        sink.stop()
        assert sink.elapsed == 0.0


class TestConversationTimeline:
    @staticmethod
    def fake_synth(text: str) -> np.ndarray:
        """One second of tone per utterance; content is irrelevant here."""
        return np.full(16_000, 0.1, dtype=np.float32)

    def test_gaps_land_where_the_manifest_says(self):
        audio, manifest = build([("एक", 2.0), ("दो", 3.0)], self.fake_synth,
                                lead_in=0.5)
        assert manifest[0]["starts_at"] == 0.5
        assert manifest[0]["ends_at"] == 1.5
        # first utterance + its gap, after the lead-in
        assert manifest[1]["starts_at"] == pytest.approx(3.5)
        expected = 0.5 + 1.0 + 2.0 + 1.0 + 3.0
        assert audio.size / 16_000 == pytest.approx(expected)

    def test_the_followup_preset_leaves_room_for_a_reply(self):
        """Its gap must exceed the endpointer's silence window, or the two
        utterances merge into one turn and there is no follow-up to test."""
        gaps = [gap for _text, gap in PRESETS["followup"]]
        assert min(gaps) >= 2.0

    def test_the_bargein_preset_deliberately_does_not(self):
        """Its first gap has to be shorter than a reply takes to speak, which
        is the whole mechanism: the second utterance arrives mid-answer."""
        first_gap = PRESETS["bargein"][0][1]
        assert first_gap < 2.0
        assert first_gap > 0.5, "still long enough to close the first utterance"

    def test_the_memory_preset_chains_references(self):
        texts = [text for text, _gap in PRESETS["memory"]]
        assert len(texts) >= 3

    def test_silent_synthesis_is_refused(self):
        """A preset that produced no audio would run as pure silence and the
        agent would simply never trigger -- reported as a failed run rather
        than diagnosed."""
        with pytest.raises(ValueError, match="no audio"):
            build([("x", 1.0)], lambda _text: np.zeros(0, dtype=np.float32))

    def test_every_preset_has_a_gap_after_its_last_utterance(self):
        """Trailing silence is what closes the final turn; without it the run
        ends mid-utterance and the last answer never happens."""
        for name, timeline in PRESETS.items():
            assert timeline[-1][1] >= 2.0, name


class TestWavOutput:
    def test_it_writes_16k_mono_16bit(self, tmp_path):
        import wave

        path = tmp_path / "out.wav"
        write_wav(path, np.concatenate([silence(0.1),
                                        np.full(1600, 0.5, dtype=np.float32)]))
        with wave.open(str(path)) as handle:
            assert handle.getframerate() == 16_000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == 3200

    def test_it_normalises_the_peak(self, tmp_path):
        """TTS levels vary by voice and the VAD's noise floor is in dBFS, so an
        un-normalised take can sit below the speech threshold entirely."""
        import wave

        path = tmp_path / "quiet.wav"
        write_wav(path, np.full(1600, 0.01, dtype=np.float32))
        with wave.open(str(path)) as handle:
            audio = np.frombuffer(handle.readframes(1600), dtype=np.int16)
        assert np.max(np.abs(audio)) > 20_000

    def test_all_silence_does_not_divide_by_zero(self, tmp_path):
        write_wav(tmp_path / "silent.wav", silence(0.1))
        assert (tmp_path / "silent.wav").exists()


class TestResampling:
    def test_a_matching_rate_passes_through(self):
        pcm = np.linspace(-1, 1, 16_000, dtype=np.float32)
        assert np.array_equal(to_mono_16k(pcm, 16_000), pcm)

    def test_downsampling_preserves_duration(self):
        pcm = np.zeros(48_000, dtype=np.float32)          # 2 s at 24 kHz
        assert to_mono_16k(pcm, 24_000).size == 32_000

    def test_stereo_is_mixed_to_mono(self):
        stereo = np.ones((16_000, 2), dtype=np.float32)
        assert to_mono_16k(stereo, 16_000).shape == (16_000,)
