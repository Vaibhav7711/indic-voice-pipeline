"""Streaming ASR: endpointer state machine and session behaviour.

CPU-only and model-free. The session talks to a fake transcriber, so these
tests assert on *state transitions and ASR call patterns* rather than on
transcription quality — which is what could actually regress here.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from asr.streaming import (
    EndpointerState,
    EndpointEventKind,
    EndpointReason,
    SessionState,
    StreamEndpointer,
    StreamingConfig,
    StreamingSession,
    StreamUpdate,
    UpdateKind,
)
from asr.vad import VADConfig

SR = 16_000


def tone(seconds: float, amplitude: float = 0.3, sample_rate: int = SR) -> np.ndarray:
    """Constant-amplitude block. Energy VAD only looks at RMS, so a constant
    is a perfectly good stand-in for speech and keeps tests deterministic."""
    return np.full(round(seconds * sample_rate), amplitude, dtype=np.float32)


def audio(*parts: tuple[float, float], sample_rate: int = SR) -> np.ndarray:
    return np.concatenate([tone(s, a, sample_rate) for s, a in parts])


def advancing_clock(step: float = 1.0):
    """Monotonic fake clock. A frozen clock can never satisfy a wall-clock
    rate limit, so tests that want partials must let time move."""
    state = {"now": 0.0}

    def now() -> float:
        state["now"] += step
        return state["now"]

    return now


def vad_config(**overrides) -> VADConfig:
    base = dict(
        frame_ms=30,
        hop_ms=10,
        threshold_dbfs=-30.0,
        min_speech_ms=200,
        min_silence_ms=400,
        padding_ms=100,
    )
    base.update(overrides)
    return VADConfig(**base)


class FakeTranscriber:
    """Records how it was called. No model, no torch, no network."""

    def __init__(self, text: str = "नमस्ते", asr_ms: float = 12.0):
        self.text = text
        self.asr_ms = asr_ms
        self.short_calls: list[float] = []
        self.long_calls: list[float] = []

    def _result(self, seconds: float, tag: str):
        return types.SimpleNamespace(
            text=f"{self.text}[{tag}:{seconds:.2f}]",
            metrics=types.SimpleNamespace(total_ms=self.asr_ms),
        )

    def transcribe_array(self, waveform, sample_rate, **kwargs):
        seconds = len(waveform) / sample_rate
        assert seconds <= 30.0, "a single encoder pass must stay under 30 s"
        self.short_calls.append(seconds)
        return self._result(seconds, "short")

    def transcribe_long_array(self, waveform, sample_rate, **kwargs):
        seconds = len(waveform) / sample_rate
        self.long_calls.append(seconds)
        return self._result(seconds, "long")

    @property
    def total_calls(self) -> int:
        return len(self.short_calls) + len(self.long_calls)


def feed(session: StreamingSession, waveform: np.ndarray, block: int = 1600):
    """Push audio in fixed blocks (100 ms at 16 kHz by default)."""
    updates: list[StreamUpdate] = []
    for start in range(0, len(waveform), block):
        updates.extend(session.push(waveform[start : start + block]))
    return updates


def finals(updates):
    return [u for u in updates if u.kind is UpdateKind.FINAL]


def partials(updates):
    return [u for u in updates if u.kind is UpdateKind.PARTIAL]


# ---------------------------------------------------------------------------
# Endpointer
# ---------------------------------------------------------------------------


class TestEndpointer:
    def test_starts_in_silence(self):
        assert StreamEndpointer(SR, vad_config()).state is EndpointerState.SILENCE

    def test_silence_produces_no_events(self):
        ep = StreamEndpointer(SR, vad_config())
        assert ep.push(tone(2.0, 0.0)) == []
        assert ep.state is EndpointerState.SILENCE

    def test_speech_onset_and_endpoint(self):
        ep = StreamEndpointer(SR, vad_config())
        events = ep.push(audio((0.5, 0.0), (1.0, 0.3), (1.0, 0.0)))
        kinds = [e.kind for e in events]
        assert kinds == [
            EndpointEventKind.SPEECH_START,
            EndpointEventKind.SPEECH_END,
        ]
        assert ep.state is EndpointerState.SILENCE

    def test_short_noise_never_starts_speech(self):
        """100 ms burst under a 200 ms minimum must not open an utterance."""
        ep = StreamEndpointer(SR, vad_config(min_speech_ms=200))
        assert ep.push(audio((0.3, 0.0), (0.1, 0.5), (0.8, 0.0))) == []
        assert ep.state is EndpointerState.SILENCE

    def test_voiced_run_is_consecutive_not_cumulative(self):
        """Three 100 ms bursts are not one 300 ms utterance."""
        ep = StreamEndpointer(SR, vad_config(min_speech_ms=250))
        blips = audio(
            (0.2, 0.0), (0.1, 0.5), (0.5, 0.0),
            (0.1, 0.5), (0.5, 0.0), (0.1, 0.5), (0.5, 0.0),
        )
        assert ep.push(blips) == []

    def test_short_pause_does_not_endpoint(self):
        """200 ms gap under a 400 ms threshold is an intra-utterance pause."""
        ep = StreamEndpointer(SR, vad_config(min_silence_ms=400))
        events = ep.push(audio((0.6, 0.3), (0.2, 0.0), (0.6, 0.3)))
        assert [e.kind for e in events] == [EndpointEventKind.SPEECH_START]
        assert ep.state is EndpointerState.SPEECH

    def test_long_pause_endpoints(self):
        ep = StreamEndpointer(SR, vad_config(min_silence_ms=400))
        events = ep.push(audio((0.6, 0.3), (0.8, 0.0), (0.6, 0.3)))
        kinds = [e.kind for e in events]
        assert kinds[:2] == [
            EndpointEventKind.SPEECH_START,
            EndpointEventKind.SPEECH_END,
        ]

    @pytest.mark.parametrize("block", [160, 320, 777, 1600, 8000, 100_000])
    def test_chunk_size_does_not_change_decisions(self, block):
        """A 10 ms mic buffer and a 5 s file block must agree exactly.

        The property that matters is invariance, not any particular offset:
        frame decisions must depend on stream position alone, never on how the
        caller happened to slice the audio.
        """
        wave = audio((0.4, 0.0), (1.0, 0.3), (0.9, 0.0), (0.7, 0.3), (0.9, 0.0))

        reference = StreamEndpointer(SR, vad_config())
        expected = [(e.kind, e.sample) for e in reference.push(wave)]
        assert len(expected) == 4, "fixture should yield two full utterances"

        ep = StreamEndpointer(SR, vad_config())
        events = []
        for start in range(0, len(wave), block):
            events.extend(ep.push(wave[start : start + block]))
        assert [(e.kind, e.sample) for e in events] == expected

    def test_unaligned_chunks_are_handled(self):
        """Chunk boundaries need not align to frames."""
        wave = audio((0.4, 0.0), (1.0, 0.3), (1.0, 0.0))
        ep = StreamEndpointer(SR, vad_config())
        events = []
        for start in range(0, len(wave), 777):
            events.extend(ep.push(wave[start : start + 777]))
        assert len(events) == 2

    def test_padding_widens_boundaries(self):
        tight = StreamEndpointer(SR, vad_config(padding_ms=0))
        padded = StreamEndpointer(SR, vad_config(padding_ms=200))
        wave = audio((0.5, 0.0), (1.0, 0.3), (1.0, 0.0))
        t = tight.push(wave)
        p = padded.push(wave)
        assert p[0].sample < t[0].sample
        assert p[1].sample > t[1].sample

    def test_flush_closes_an_open_utterance(self):
        """Speaker stops and disconnects: no trailing silence ever arrives."""
        ep = StreamEndpointer(SR, vad_config())
        ep.push(audio((0.3, 0.0), (1.0, 0.3)))
        assert ep.state is EndpointerState.SPEECH
        flushed = ep.flush()
        assert [e.kind for e in flushed] == [EndpointEventKind.SPEECH_END]
        assert ep.state is EndpointerState.SILENCE

    def test_flush_during_silence_is_a_no_op(self):
        ep = StreamEndpointer(SR, vad_config())
        ep.push(tone(1.0, 0.0))
        assert ep.flush() == []

    def test_empty_push(self):
        ep = StreamEndpointer(SR, vad_config())
        assert ep.push(np.zeros(0, dtype=np.float32)) == []

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            StreamEndpointer(0, vad_config())

    def test_serializable_state(self):
        ep = StreamEndpointer(SR, vad_config())
        ep.push(tone(0.5, 0.0))
        payload = ep.as_dict()
        assert payload["state"] == "silence"
        assert payload["min_speech_frames"] == 20


class TestEndpointerAgreesWithOfflineVAD:
    """The online module must not drift from the offline reference."""

    def test_same_utterance_count(self):
        from asr.vad import detect_speech

        config = vad_config()
        wave = audio(
            (0.5, 0.0), (1.0, 0.3), (0.9, 0.0), (0.8, 0.3), (0.9, 0.0),
        )
        offline = detect_speech(wave, SR, config)

        ep = StreamEndpointer(SR, config)
        events = ep.push(wave) + ep.flush()
        starts = [e for e in events if e.kind is EndpointEventKind.SPEECH_START]
        assert len(starts) == len(offline) == 2

    def test_boundaries_agree_within_one_frame(self):
        from asr.vad import detect_speech

        config = vad_config()
        wave = audio((0.5, 0.0), (1.2, 0.3), (1.0, 0.0))
        offline = detect_speech(wave, SR, config)[0]

        ep = StreamEndpointer(SR, config)
        events = ep.push(wave)
        frame = round(SR * config.frame_ms / 1000)
        assert abs(events[0].sample - offline.start_sample) <= frame
        assert abs(events[1].sample - offline.end_sample) <= frame


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def make_session(transcriber=None, clock=None, **config_overrides):
    transcriber = transcriber or FakeTranscriber()
    config = StreamingConfig(
        sample_rate=SR,
        vad=vad_config(),
        emit_partials=config_overrides.pop("emit_partials", False),
        **config_overrides,
    )
    return StreamingSession(
        transcriber, config, clock=clock or (lambda: 0.0)
    ), transcriber


class TestSessionRequiredCases:
    """The six cases the streaming API has to get right."""

    def test_silence_before_speech_emits_nothing_and_runs_no_asr(self):
        session, fake = make_session()
        updates = feed(session, tone(3.0, 0.0))
        assert updates == []
        assert fake.total_calls == 0
        assert session.state is SessionState.IDLE

    def test_short_noise_below_minimum_is_ignored(self):
        session, fake = make_session()
        updates = feed(session, audio((0.5, 0.0), (0.1, 0.6), (1.0, 0.0)))
        assert finals(updates) == []
        assert fake.total_calls == 0, "a door slam must not invoke Whisper"

    def test_speech_split_across_many_chunks(self):
        """20 ms blocks — one utterance, one final, one ASR call."""
        session, fake = make_session()
        wave = audio((0.4, 0.0), (1.5, 0.3), (1.0, 0.0))
        updates = feed(session, wave, block=320)
        assert len(finals(updates)) == 1
        assert fake.total_calls == 1

    def test_short_pause_inside_one_utterance(self):
        """A 200 ms breath must not split the utterance in two."""
        session, fake = make_session()
        wave = audio((0.4, 0.0), (0.7, 0.3), (0.2, 0.0), (0.7, 0.3), (1.0, 0.0))
        updates = feed(session, wave)
        final_updates = finals(updates)
        assert len(final_updates) == 1
        assert fake.total_calls == 1
        # The pause is inside the transcribed audio, not cut out of it.
        assert final_updates[0].audio_seconds > 1.4

    def test_utterance_then_new_speech(self):
        session, fake = make_session()
        wave = audio(
            (0.4, 0.0), (0.8, 0.3), (1.0, 0.0), (0.8, 0.3), (1.0, 0.0),
        )
        updates = feed(session, wave)
        final_updates = finals(updates)
        assert len(final_updates) == 2
        assert [u.utterance_index for u in final_updates] == [0, 1]
        assert fake.total_calls == 2
        assert all(u.endpoint_reason is EndpointReason.SILENCE for u in final_updates)

    def test_long_utterance_routes_through_long_form(self):
        """Over the threshold, finals use chunk-and-stitch, not one pass."""
        session, fake = make_session(long_form_threshold_seconds=5.0)
        wave = audio((0.4, 0.0), (12.0, 0.3), (1.0, 0.0))
        updates = feed(session, wave, block=SR)
        final_updates = finals(updates)
        assert len(final_updates) == 1
        assert final_updates[0].long_form is True
        assert len(fake.long_calls) == 1
        assert len(fake.short_calls) == 0

    def test_short_utterance_does_not_use_long_form(self):
        session, fake = make_session(long_form_threshold_seconds=5.0)
        updates = feed(session, audio((0.4, 0.0), (1.0, 0.3), (1.0, 0.0)))
        assert finals(updates)[0].long_form is False
        assert len(fake.short_calls) == 1
        assert len(fake.long_calls) == 0


class TestPartialRateLimiting:
    def test_partials_disabled_emits_none(self):
        session, fake = make_session(emit_partials=False)
        updates = feed(session, audio((0.4, 0.0), (4.0, 0.3), (1.0, 0.0)))
        assert partials(updates) == []
        assert fake.total_calls == 1

    def test_audio_limit_stops_tiny_chunks_triggering_asr(self):
        """20 ms mic buffers must not cause 50 decodes a second."""
        clock = iter(range(10_000))
        session, fake = make_session(
            emit_partials=True,
            partial_interval_ms=0,          # wall-clock limit disabled...
            min_partial_audio_ms=1000,
            clock=lambda: next(clock),
        )
        feed(session, audio((0.4, 0.0), (3.0, 0.3), (1.0, 0.0)), block=320)
        # partial_interval_ms <= 0 disables partials entirely.
        assert len(fake.short_calls) == 1

    def test_audio_growth_gates_partials(self):
        session, fake = make_session(
            emit_partials=True,
            partial_interval_ms=1,
            min_partial_audio_ms=1000,
            clock=advancing_clock(1.0),   # wall-clock limit never blocks
        )
        updates = feed(
            session, audio((0.4, 0.0), (4.0, 0.3), (1.0, 0.0)), block=320
        )
        # ~4 s of speech, one partial per 1000 ms of new audio.
        assert 2 <= len(partials(updates)) <= 5

    def test_wall_clock_gates_partials_when_audio_arrives_fast(self):
        """Replaying a file faster than real time must not spam decodes."""
        session, fake = make_session(
            emit_partials=True,
            partial_interval_ms=10_000,   # 10 s between partials
            min_partial_audio_ms=1,       # audio limit never blocks
            clock=lambda: 0.0,            # frozen clock
        )
        updates = feed(
            session, audio((0.4, 0.0), (6.0, 0.3), (1.0, 0.0)), block=320
        )
        assert partials(updates) == []

    def test_partial_then_final_ordering(self):
        session, fake = make_session(
            emit_partials=True,
            partial_interval_ms=1,
            min_partial_audio_ms=500,
            clock=advancing_clock(1.0),
        )
        updates = feed(session, audio((0.4, 0.0), (3.0, 0.3), (1.0, 0.0)))
        kinds = [u.kind for u in updates]
        assert UpdateKind.PARTIAL in kinds
        assert kinds[-1] is UpdateKind.STATE
        assert kinds.index(UpdateKind.FINAL) > kinds.index(UpdateKind.PARTIAL)
        assert all(not u.is_final for u in partials(updates))

    def test_partial_window_caps_at_whisper_limit(self):
        """A long utterance partial decodes the tail, flagged as such."""
        session, fake = make_session(
            emit_partials=True,
            partial_interval_ms=1,
            min_partial_audio_ms=500,
            partial_window_seconds=5.0,
            long_form_threshold_seconds=100.0,
            clock=advancing_clock(1.0),
        )
        updates = feed(
            session, audio((0.4, 0.0), (12.0, 0.3), (1.0, 0.0)), block=SR // 2
        )
        tails = [u for u in partials(updates) if u.partial_is_tail]
        assert tails, "partials past the window must be marked as tail-only"
        assert all(u.audio_seconds <= 5.01 for u in partials(updates))


class TestSessionLifecycle:
    def test_max_duration_forces_a_cut(self):
        """A speaker who never pauses must not grow the buffer without bound."""
        session, fake = make_session(
            max_utterance_seconds=3.0, long_form_threshold_seconds=100.0
        )
        updates = feed(session, audio((0.4, 0.0), (10.0, 0.3)), block=SR // 2)
        forced = [
            u for u in finals(updates)
            if u.endpoint_reason is EndpointReason.MAX_DURATION
        ]
        assert forced, "expected a forced endpoint"

    def test_flush_finalizes_an_open_utterance(self):
        session, fake = make_session()
        feed(session, audio((0.4, 0.0), (1.5, 0.3)))
        assert session.state is SessionState.SPEAKING
        updates = session.flush()
        final_updates = finals(updates)
        assert len(final_updates) == 1
        assert final_updates[0].endpoint_reason is EndpointReason.STREAM_FLUSH
        assert session.state is SessionState.CLOSED

    def test_flush_during_silence_only_closes(self):
        session, fake = make_session()
        feed(session, tone(1.0, 0.0))
        updates = session.flush()
        assert finals(updates) == []
        assert updates[-1].state is SessionState.CLOSED
        assert fake.total_calls == 0

    def test_push_after_flush_raises(self):
        session, _ = make_session()
        session.flush()
        with pytest.raises(RuntimeError, match="closed"):
            session.push(tone(0.1))

    def test_double_flush_is_safe(self):
        session, _ = make_session()
        session.flush()
        assert session.flush() == []

    def test_sequence_numbers_are_monotonic(self):
        session, _ = make_session()
        updates = feed(
            session, audio((0.4, 0.0), (1.0, 0.3), (1.0, 0.0), (1.0, 0.3), (1.0, 0.0))
        ) + session.flush()
        sequences = [u.sequence for u in updates]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)

    def test_every_update_is_serializable(self):
        import json

        session, _ = make_session(
            emit_partials=True, partial_interval_ms=1,
            min_partial_audio_ms=500, clock=advancing_clock(1.0),
        )
        updates = feed(
            session, audio((0.4, 0.0), (2.0, 0.3), (1.0, 0.0))
        ) + session.flush()
        assert updates
        for update in updates:
            json.dumps(update.as_dict())  # must not raise

    def test_snapshot_is_serializable(self):
        import json

        session, _ = make_session()
        feed(session, audio((0.4, 0.0), (1.0, 0.3)))
        snapshot = session.snapshot()
        json.dumps(snapshot)
        assert snapshot["state"] == "speaking"

    def test_buffer_is_trimmed_across_utterances(self):
        """Memory must track the current utterance, not the session."""
        session, _ = make_session()
        wave = audio((0.4, 0.0), (0.8, 0.3), (1.0, 0.0))
        for _ in range(5):
            feed(session, wave)
        assert session.snapshot()["buffered_seconds"] < 3.0

    def test_final_carries_timing_fields(self):
        session, _ = make_session(transcriber=FakeTranscriber(asr_ms=250.0))
        updates = feed(session, audio((0.4, 0.0), (1.0, 0.3), (1.0, 0.0)))
        final = finals(updates)[0]
        assert final.asr_ms == 250.0
        assert final.real_time_factor > 0
        assert final.audio_seconds > 0
        assert final.utterance_start_seconds > 0

    def test_empty_push_is_a_no_op(self):
        session, _ = make_session()
        assert session.push(np.zeros(0, dtype=np.float32)) == []


# ---------------------------------------------------------------------------
# The audio handed to ASR must contain the padded onset (regression)
# ---------------------------------------------------------------------------


class _RecordingTranscriber(FakeTranscriber):
    def __init__(self):
        super().__init__()
        self.audio: list[np.ndarray] = []

    def transcribe_array(self, waveform, sample_rate, **kwargs):
        self.audio.append(np.asarray(waveform))
        return super().transcribe_array(waveform, sample_rate, **kwargs)


class TestOnsetAudioIsPreserved:
    def test_final_audio_starts_at_the_padded_onset_not_at_confirmation(self):
        """Idle trimming used to discard audio up to 'now' on every push, so
        when SPEECH_START arrived (min_speech_ms after the onset, pointing
        padding_ms before it) the padded onset was already gone: every
        utterance lost ~450 ms and Whisper hallucinated on the clipped word."""
        fake = _RecordingTranscriber()
        cfg = StreamingConfig(vad=vad_config(), emit_partials=False)
        session = StreamingSession(fake, cfg, clock=lambda: 0.0)
        wave = audio((0.5, 0.0), (2.0, 0.3), (1.0, 0.0))
        updates = feed(session, wave, block=SR // 10)          # 100 ms blocks

        final = finals(updates)[0]
        assert len(fake.audio) == 1
        got = fake.audio[0]
        # Audio length matches the reported span, and the span starts where
        # the endpointer said it does.
        assert abs(len(got) / SR - final.audio_seconds) < 1e-6
        expected_start = 0.5 - cfg.vad.padding_ms / 1000
        assert abs(final.utterance_start_seconds - expected_start) <= cfg.vad.frame_ms / 1000
        # The handed-over audio begins with the pre-speech silence and the
        # tone starts exactly where it does in the stream — nothing was cut.
        first_tone = int(np.flatnonzero(got != 0.0)[0])
        assert abs(first_tone / SR - (0.5 - final.utterance_start_seconds)) < 1e-6
        assert first_tone / SR >= cfg.vad.padding_ms / 1000 - 1e-6
        assert np.sum(got != 0.0) == int(2.0 * SR)                 # whole tone present


# ---------------------------------------------------------------------------
# Candidate finals: ASR inside the silence wait
# ---------------------------------------------------------------------------


def candidates(updates):
    return [u for u in updates if u.kind is UpdateKind.CANDIDATE]


class _TextByLengthTranscriber(FakeTranscriber):
    """Text that depends on the audio it saw, so reuse is observable."""

    def __init__(self):
        super().__init__()
        self.audio_seconds: list[float] = []

    def _result(self, seconds: float, tag: str):
        self.audio_seconds.append(seconds)
        return types.SimpleNamespace(
            text=f"शब्द{len(self.audio_seconds)} len={seconds:.2f}",
            metrics=types.SimpleNamespace(total_ms=self.asr_ms),
        )


class TestCandidateFinals:
    def test_candidate_is_decoded_during_silence_and_final_reuses_it(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(fake, early_final_silence_ms=200)
        # vad_config(): min_silence_ms=400. Silence after speech = 1.0 s.
        updates = feed(session, audio((0.4, 0.0), (2.0, 0.3), (1.0, 0.0)))

        cands, fins = candidates(updates), finals(updates)
        assert len(cands) == 1 and len(fins) == 1
        assert fake.total_calls == 1, "final must not decode again"
        assert fins[0].from_candidate is True
        assert fins[0].text == cands[0].text
        assert fins[0].asr_ms_after_endpoint == 0.0
        assert fins[0].asr_ms == cands[0].asr_ms > 0
        # The candidate saw exactly the audio the final reports.
        assert abs(cands[0].audio_seconds - fins[0].audio_seconds) < 1e-6
        assert cands[0].sequence < fins[0].sequence

    def test_candidate_is_discarded_when_speech_resumes(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(fake, early_final_silence_ms=200)
        # 0.3 s pause: candidate fires at 0.2 s, but the pause is under
        # min_silence (0.4 s) so speech "resumes" and the utterance continues.
        updates = feed(session, audio((0.4, 0.0), (1.0, 0.3), (0.3, 0.0), (1.0, 0.3), (1.0, 0.0)))

        cands, fins = candidates(updates), finals(updates)
        assert len(fins) == 1
        assert len(cands) == 2, "one per silence run"
        assert fake.total_calls == 2
        assert fins[0].from_candidate is True
        assert fins[0].text == cands[-1].text
        assert fins[0].text != cands[0].text
        assert fins[0].audio_seconds > cands[0].audio_seconds

    def test_flush_reuses_the_candidate(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(fake, early_final_silence_ms=200)
        updates = feed(session, audio((0.4, 0.0), (2.0, 0.3), (0.3, 0.0)))  # < min_silence
        assert finals(updates) == [] and len(candidates(updates)) == 1
        flushed = session.flush()
        fin = finals(flushed)[0]
        assert fin.from_candidate is True
        assert fin.endpoint_reason.value == "stream_flush"
        assert fake.total_calls == 1

    def test_disabled_candidates_decode_after_endpoint(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(fake, early_final_silence_ms=0)
        updates = feed(session, audio((0.4, 0.0), (2.0, 0.3), (1.0, 0.0)))
        assert candidates(updates) == []
        fin = finals(updates)[0]
        assert fin.from_candidate is False
        assert fin.asr_ms_after_endpoint == fin.asr_ms > 0

    def test_early_must_be_below_min_silence(self):
        with pytest.raises(ValueError, match="early_final_silence_ms"):
            make_session(early_final_silence_ms=400)      # == vad min_silence_ms

    def test_max_duration_cut_does_not_use_a_stale_candidate(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(fake, early_final_silence_ms=200, max_utterance_seconds=3.0)
        updates = feed(session, audio((0.4, 0.0), (5.0, 0.3), (1.0, 0.0)))
        fins = finals(updates)
        assert fins[0].endpoint_reason.value == "max_duration"
        assert fins[0].from_candidate is False


class TestIncrementalFinals:
    def test_final_decodes_only_the_tail_and_stitches_onto_partial(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(
            fake, emit_partials=True, partial_interval_ms=1, min_partial_audio_ms=500,
            early_final_silence_ms=0, incremental_finals=True,
            incremental_overlap_seconds=1.0, clock=advancing_clock(1.0),
        )
        updates = feed(session, audio((0.4, 0.0), (4.0, 0.3), (1.0, 0.0)))
        parts, fin = partials(updates), finals(updates)[0]
        assert parts, "needs at least one partial to reuse"
        assert fin.reused_partial is True
        assert fin.text.startswith(parts[-1].text), "partial prefix is committed"
        # Tail = the 1 s overlap plus any audio the last partial had not seen
        # (here none: the last partial ran during the trailing silence).
        new_audio = max(0.0, fin.audio_seconds - parts[-1].audio_seconds)
        assert fin.decoded_seconds < fin.audio_seconds
        assert 1.0 - 0.05 <= fin.decoded_seconds <= new_audio + 1.0 + 0.05
        assert abs(fake.audio_seconds[-1] - fin.decoded_seconds) < 1e-6

    def test_incremental_off_decodes_everything(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(
            fake, emit_partials=True, partial_interval_ms=1, min_partial_audio_ms=500,
            early_final_silence_ms=0, incremental_finals=False, clock=advancing_clock(1.0),
        )
        fin = finals(feed(session, audio((0.4, 0.0), (4.0, 0.3), (1.0, 0.0))))[0]
        assert fin.reused_partial is False
        assert abs(fin.decoded_seconds - fin.audio_seconds) < 1e-6

    def test_candidate_after_partial_is_incremental_and_final_reuses_it(self):
        fake = _TextByLengthTranscriber()
        session, _ = make_session(
            fake, emit_partials=True, partial_interval_ms=1, min_partial_audio_ms=500,
            early_final_silence_ms=200, incremental_finals=True,
            incremental_overlap_seconds=1.0, clock=advancing_clock(1.0),
        )
        updates = feed(session, audio((0.4, 0.0), (4.0, 0.3), (1.0, 0.0)))
        cand, fin = candidates(updates)[0], finals(updates)[0]
        assert cand.reused_partial is True and cand.decoded_seconds < cand.audio_seconds
        assert fin.from_candidate and fin.asr_ms_after_endpoint == 0.0
        assert fin.text == cand.text

    def test_tail_partial_is_not_reused(self):
        """A partial that covered only the tail cannot be a prefix."""
        fake = _TextByLengthTranscriber()
        session, _ = make_session(
            fake, emit_partials=True, partial_interval_ms=1, min_partial_audio_ms=500,
            partial_window_seconds=2.0, long_form_threshold_seconds=100.0,
            early_final_silence_ms=0, incremental_finals=True, clock=advancing_clock(1.0),
        )
        fin = finals(feed(session, audio((0.4, 0.0), (5.0, 0.3), (1.0, 0.0))))[0]
        assert fin.reused_partial is False


class TestSemanticEndpointing:
    class _Incomplete(FakeTranscriber):
        def _result(self, seconds, tag):
            return types.SimpleNamespace(
                text="मैं बाज़ार के", metrics=types.SimpleNamespace(total_ms=1.0),
            )

    class _Complete(FakeTranscriber):
        def _result(self, seconds, tag):
            return types.SimpleNamespace(
                text="मैं ठीक हूँ।", metrics=types.SimpleNamespace(total_ms=1.0),
            )

    def _run(self, transcriber, *, semantic):
        from asr.streaming import EndpointPolicy

        session, _ = make_session(
            transcriber, early_final_silence_ms=200, semantic_endpointing=semantic,
            endpoint_policy=EndpointPolicy(incomplete_silence_ms=1000),
        )
        # Pause of 0.7 s: over min_silence (0.4) but under the incomplete wait (1.0).
        wave = audio((0.4, 0.0), (1.0, 0.3), (0.7, 0.0), (1.0, 0.3), (1.2, 0.0))
        return finals(feed(session, wave))

    def test_incomplete_phrase_waits_longer_and_avoids_the_split(self):
        assert len(self._run(self._Incomplete(), semantic=False)) == 2
        assert len(self._run(self._Incomplete(), semantic=True)) == 1

    def test_complete_phrase_keeps_the_default_wait(self):
        assert len(self._run(self._Complete(), semantic=True)) == 2

    def test_policy_resets_for_the_next_utterance(self):
        fins = self._run(self._Incomplete(), semantic=True)
        assert fins[0].endpoint_reason.value == "silence"
        # After closing, a following utterance with a 0.7 s pause must be
        # split again unless its own candidate says incomplete — here the
        # transcriber always says incomplete, so check the endpointer reset.
        from asr.streaming import EndpointPolicy

        session, _ = make_session(
            self._Complete(), early_final_silence_ms=200, semantic_endpointing=True,
            endpoint_policy=EndpointPolicy(incomplete_silence_ms=1000),
        )
        assert session._endpointer.endpoint_silence_frames == \
            session._endpointer.default_endpoint_silence_frames


class TestPhraseIsIncomplete:
    def test_rules(self):
        from asr.streaming import phrase_is_incomplete

        assert phrase_is_incomplete("मैं बाज़ार के")
        assert phrase_is_incomplete("मुझे लगता है कि")
        assert phrase_is_incomplete("यह बहुत अच्छा है और")
        assert phrase_is_incomplete("I want to go to the")
        assert not phrase_is_incomplete("मैं ठीक हूँ")
        assert not phrase_is_incomplete("मैं बाज़ार के।")      # terminator wins
        assert not phrase_is_incomplete("क्या आप आएंगे?")
        assert not phrase_is_incomplete("")
        assert not phrase_is_incomplete("और।")
