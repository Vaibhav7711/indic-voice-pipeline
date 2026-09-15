"""Voice-agent output: playback lifecycle, sentence streaming, turn latency.

No network, no GPU, no model. Every backend is a fake, so these tests assert on
state transitions, cancellation semantics and latency provenance — the things
that actually break.
"""

from __future__ import annotations

import json
import threading
import types

import pytest

from agent import (
    BufferSink,
    PlaybackEventKind,
    PlaybackSession,
    PlaybackState,
    TurnState,
    VoiceTurn,
)
from tts.streaming import SpeechStream, split_sentences, synthesize_stream


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Non-streaming backend, shaped like llm.runner.LLMRunner."""

    def __init__(self, text: str = "नमस्ते। मैं ठीक हूँ।", prefill_ms: float = 40.0):
        self.text = text
        self.prefill_ms = prefill_ms
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        return types.SimpleNamespace(
            text=self.text,
            metrics=types.SimpleNamespace(prefill_ms=self.prefill_ms, total_ms=120.0),
        )


class StreamingLLM(FakeLLM):
    """Backend that exposes stream(), making first-token time observable."""

    def stream(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        for piece in self.text.split():
            yield piece + " "


class FakeTTS:
    streaming = True

    def __init__(self, chunks_per_sentence: int = 2):
        self.chunks_per_sentence = chunks_per_sentence
        self.sentences: list[str] = []

    def stream(self, text: str):
        self.sentences.append(text)
        for index in range(self.chunks_per_sentence):
            yield f"{text[:3]}:{index}".encode()


class BufferingTTS(FakeTTS):
    """Backend with no real streaming — must not be reported as streaming."""

    streaming = False

    def stream(self, text: str):
        self.sentences.append(text)
        yield text.encode()


class ExplodingTTS:
    streaming = True

    def stream(self, text: str):
        yield b"first"
        raise RuntimeError("synthesis backend died")


class ExplodingSink(BufferSink):
    def write(self, chunk: bytes) -> None:
        raise OSError("audio device unavailable")


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_danda_splits_hindi(self):
        assert split_sentences("नमस्ते। मैं ठीक हूँ। आप कैसे हैं?") == [
            "नमस्ते। मैं ठीक हूँ।",
            "आप कैसे हैं?",
        ]

    def test_ascii_punctuation_splits_english(self):
        assert len(split_sentences("Hello there. How are you? I am fine.")) == 3

    def test_code_switched_text(self):
        assert split_sentences("मैंने laptop खरीदा। यह अच्छा है।") == [
            "मैंने laptop खरीदा।",
            "यह अच्छा है।",
        ]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_single_short_fragment_survives(self):
        assert split_sentences("ok.") == ["ok."]

    def test_no_terminator_is_one_unit(self):
        assert split_sentences("बिना विराम के वाक्य") == ["बिना विराम के वाक्य"]

    def test_newlines_are_hard_breaks(self):
        assert len(split_sentences("पहली पंक्ति\nदूसरी पंक्ति यहाँ")) == 2

    def test_tiny_fragments_merge_forward(self):
        """A stray token must not cost its own synthesis round trip."""
        assert split_sentences("हाँ। यह एक लंबा वाक्य है।") == [
            "हाँ। यह एक लंबा वाक्य है।"
        ]


# ---------------------------------------------------------------------------
# synthesize_stream
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    def test_yields_chunks_per_sentence(self):
        tts = FakeTTS(chunks_per_sentence=3)
        stream = synthesize_stream(tts, "नमस्ते। मैं ठीक हूँ। आप कैसे हैं?")
        assert len(stream.sentences) == 2
        assert len(stream.chunks) == 6
        assert stream.first_chunk_ms is not None
        assert stream.streaming is True

    def test_non_streaming_backend_is_reported_honestly(self):
        stream = synthesize_stream(BufferingTTS(), "एक वाक्य यहाँ है।")
        assert stream.streaming is False

    def test_should_stop_halts_synthesis(self):
        """Barge-in must stop generating, not just discard already-paid work."""
        tts = FakeTTS()
        calls = {"n": 0}

        def should_stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 2        # allow the first sentence, then stop

        stream = synthesize_stream(
            tts,
            "पहला वाक्य यहाँ। दूसरा वाक्य यहाँ। तीसरा वाक्य यहाँ।",
            should_stop=should_stop,
        )
        assert len(stream.sentences) == 3, "fixture should split into three"
        assert len(tts.sentences) == 1, "later sentences must never be synthesised"
        assert stream.chunks, "audio produced before the stop is kept"

    def test_backend_error_is_captured_not_raised(self):
        stream = synthesize_stream(ExplodingTTS(), "एक वाक्य यहाँ है।")
        assert stream.error is not None
        assert "synthesis backend died" in stream.error
        assert len(stream.chunks) == 1, "audio produced before the error is kept"

    def test_empty_text(self):
        stream = synthesize_stream(FakeTTS(), "")
        assert stream.chunks == []
        assert stream.sentences == []

    def test_serializable(self):
        stream = synthesize_stream(FakeTTS(), "नमस्ते। मैं ठीक हूँ।")
        json.dumps(stream.as_dict())



# ---------------------------------------------------------------------------
# Playback lifecycle
# ---------------------------------------------------------------------------


class TestPlaybackLifecycle:
    def test_completes_normally(self):
        sink = BufferSink()
        result = PlaybackSession(sink).play([b"a", b"b", b"c"])
        assert result.state is PlaybackState.COMPLETED
        assert result.chunks_written == 3
        assert result.bytes_written == 3
        assert sink.data == b"abc"
        assert sink.closed and not sink.stopped

    def test_first_audio_recorded_once(self):
        result = PlaybackSession(BufferSink()).play([b"a", b"b", b"c"])
        kinds = [e.kind for e in result.events]
        assert kinds.count(PlaybackEventKind.FIRST_AUDIO) == 1
        assert result.first_audio_ms is not None

    def test_empty_stream_completes_with_no_audio(self):
        result = PlaybackSession(BufferSink()).play([])
        assert result.state is PlaybackState.COMPLETED
        assert result.first_audio_ms is None
        assert result.chunks_written == 0

    def test_empty_chunks_are_skipped(self):
        result = PlaybackSession(BufferSink()).play([b"", b"a", b""])
        assert result.chunks_written == 1

    def test_state_progression_is_recorded(self):
        result = PlaybackSession(BufferSink()).play([b"a"])
        kinds = [e.kind for e in result.events]
        assert kinds[0] is PlaybackEventKind.STARTED
        assert kinds[-1] is PlaybackEventKind.COMPLETED

    def test_sink_failure_becomes_failed_state(self):
        sink = ExplodingSink()
        result = PlaybackSession(sink).play([b"a"])
        assert result.state is PlaybackState.FAILED
        assert "audio device unavailable" in result.error
        assert sink.stopped, "a failed playback must stop the sink"

    def test_cannot_replay_a_session(self):
        session = PlaybackSession(BufferSink())
        session.play([b"a"])
        with pytest.raises(RuntimeError, match="already ran"):
            session.play([b"b"])

    def test_events_are_serializable(self):
        result = PlaybackSession(BufferSink()).play([b"a", b"b"])
        json.dumps(result.as_dict())


class TestBargeIn:
    def test_cancel_before_play_produces_no_audio(self):
        sink = BufferSink()
        session = PlaybackSession(sink)
        assert session.cancel("user_spoke") is True
        result = session.play([b"a", b"b", b"c"])
        assert result.state is PlaybackState.CANCELLED
        assert result.chunks_written == 0
        assert sink.stopped

    def test_cancel_midway_stops_writing(self):
        sink = BufferSink()
        session = PlaybackSession(sink)

        def chunks():
            yield b"a"
            session.cancel("user_spoke")   # arrives while playing
            yield b"b"
            yield b"c"

        result = session.play(chunks())
        assert result.state is PlaybackState.CANCELLED
        assert result.cancel_reason == "user_spoke"
        assert result.chunks_written == 1, "only the pre-cancel chunk is played"
        assert sink.stopped

    def test_cancel_stops_the_producer(self):
        """The generator must not be fully drained after a cancel."""
        session = PlaybackSession(BufferSink())
        produced = []

        def chunks():
            for index in range(10):
                produced.append(index)
                if index == 1:
                    session.cancel()
                yield f"{index}".encode()

        session.play(chunks())
        assert len(produced) < 10

    def test_cancel_after_completion_is_ignored(self):
        """In a race, 'we already finished' must not be rewritten."""
        session = PlaybackSession(BufferSink())
        result = session.play([b"a"])
        assert result.state is PlaybackState.COMPLETED
        assert session.cancel("too_late") is False
        assert session.state is PlaybackState.COMPLETED
        assert session.events[-1].kind is PlaybackEventKind.IGNORED_CANCEL

    def test_cancel_is_thread_safe(self):
        """Cancel arrives on the audio-input thread while playback runs.

        Synchronised with Events rather than sleeps: the playback loop can
        drain 200 in-memory chunks in well under a millisecond, so any
        timing-based version of this test races and passes by luck.
        """
        session = PlaybackSession(BufferSink())
        reached = threading.Event()
        cancelled = threading.Event()

        def chunks():
            for index in range(200):
                if index == 5:
                    reached.set()
                    cancelled.wait(timeout=2)   # hold until cancel has landed
                yield b"x"

        def canceller():
            reached.wait(timeout=2)
            session.cancel("from_thread")
            cancelled.set()

        thread = threading.Thread(target=canceller)
        thread.start()
        result = session.play(chunks())
        thread.join(timeout=2)

        assert result.state is PlaybackState.CANCELLED
        assert result.cancel_reason == "from_thread"
        assert result.chunks_written < 200

    def test_should_stop_reflects_cancellation(self):
        session = PlaybackSession(BufferSink())
        assert session.should_stop() is False
        session.cancel()
        assert session.should_stop() is True

    def test_snapshot_is_serializable(self):
        session = PlaybackSession(BufferSink())
        session.play([b"a"])
        json.dumps(session.snapshot())


# ---------------------------------------------------------------------------
# Turn orchestration
# ---------------------------------------------------------------------------


class TestVoiceTurn:
    def test_full_turn_completes(self):
        turn = VoiceTurn(FakeLLM(), FakeTTS())
        result = turn.run("आप कैसे हैं", speech_end_to_transcript_ms=300.0)
        assert result.state is TurnState.COMPLETED
        assert result.response
        assert result.playback.state is PlaybackState.COMPLETED

    def test_empty_transcript_short_circuits(self):
        llm = FakeLLM()
        result = VoiceTurn(llm, FakeTTS()).run("   ")
        assert result.state is TurnState.COMPLETED
        assert llm.calls == [], "no LLM call for an empty transcript"

    def test_all_four_latency_fields_present(self):
        turn = VoiceTurn(FakeLLM(), FakeTTS())
        metrics = turn.run("नमस्ते", speech_end_to_transcript_ms=250.0).metrics
        assert metrics.speech_end_to_final_transcript_ms == 250.0
        assert metrics.final_transcript_to_first_llm_token_ms is not None
        assert metrics.first_llm_token_to_playback_start_ms is not None
        assert metrics.total_turn_ms > 0

    def test_unmeasured_latency_is_none_not_zero(self):
        """A zero would silently pollute an average; None forces the question."""
        metrics = VoiceTurn(FakeLLM(), FakeTTS()).run("नमस्ते").metrics
        assert metrics.speech_end_to_final_transcript_ms is None
        assert metrics.response_latency_ms is None

    def test_non_streaming_llm_is_flagged_as_a_proxy(self):
        metrics = VoiceTurn(FakeLLM(prefill_ms=55.0), FakeTTS()).run("नमस्ते").metrics
        assert metrics.llm_streaming is False
        assert metrics.first_token_is_prefill_proxy is True
        assert metrics.final_transcript_to_first_llm_token_ms == 55.0

    def test_streaming_llm_measures_first_token_directly(self):
        metrics = VoiceTurn(StreamingLLM(), FakeTTS()).run("नमस्ते").metrics
        assert metrics.llm_streaming is True
        assert metrics.first_token_is_prefill_proxy is False
        assert metrics.final_transcript_to_first_llm_token_ms is not None

    def test_non_streaming_tts_is_not_claimed_as_streaming(self):
        metrics = VoiceTurn(FakeLLM(), BufferingTTS()).run("नमस्ते").metrics
        assert metrics.tts_streaming is False

    def test_response_latency_sums_the_three_segments(self):
        metrics = VoiceTurn(FakeLLM(prefill_ms=40.0), FakeTTS()).run(
            "नमस्ते", speech_end_to_transcript_ms=300.0
        ).metrics
        expected = (
            metrics.speech_end_to_final_transcript_ms
            + metrics.final_transcript_to_first_llm_token_ms
            + metrics.first_llm_token_to_playback_start_ms
        )
        assert metrics.response_latency_ms == pytest.approx(expected)

    def test_sentence_level_streaming_is_used(self):
        tts = FakeTTS()
        VoiceTurn(FakeLLM("पहला वाक्य यहाँ। दूसरा वाक्य यहाँ।"), tts).run("नमस्ते")
        assert len(tts.sentences) == 2, "response must be synthesised per sentence"

    def test_sentence_splitting_can_be_disabled(self):
        tts = FakeTTS()
        VoiceTurn(
            FakeLLM("पहला वाक्य यहाँ। दूसरा वाक्य यहाँ।"),
            tts,
            split_into_sentences=False,
        ).run("नमस्ते")
        assert len(tts.sentences) == 1

    def test_llm_failure_becomes_failed_state(self):
        class BrokenLLM:
            def generate(self, prompt, **kwargs):
                raise RuntimeError("model OOM")

        result = VoiceTurn(BrokenLLM(), FakeTTS()).run("नमस्ते")
        assert result.state is TurnState.FAILED
        assert "model OOM" in result.error

    def test_tts_failure_is_surfaced(self):
        result = VoiceTurn(FakeLLM(), ExplodingTTS()).run("नमस्ते")
        assert result.speech.error is not None

    def test_barge_in_interrupts_the_turn(self):
        turn = VoiceTurn(FakeLLM("पहला वाक्य यहाँ। दूसरा वाक्य यहाँ।"), FakeTTS())

        class InterruptingSink(BufferSink):
            def write(self, chunk: bytes) -> None:
                super().write(chunk)
                turn.interrupt("user_started_speaking")

        result = turn.run("नमस्ते", sink=InterruptingSink())
        assert result.state is TurnState.INTERRUPTED
        assert result.metrics.barge_in is True
        assert result.playback.cancel_reason == "user_started_speaking"

    def test_interrupt_without_playback_returns_false(self):
        assert VoiceTurn(FakeLLM(), FakeTTS()).interrupt() is False

    def test_result_is_serializable(self):
        result = VoiceTurn(FakeLLM(), FakeTTS()).run(
            "नमस्ते", speech_end_to_transcript_ms=300.0
        )
        json.dumps(result.as_dict())

    def test_snapshot_is_serializable(self):
        turn = VoiceTurn(FakeLLM(), FakeTTS())
        turn.run("नमस्ते")
        json.dumps(turn.snapshot())

    def test_prompt_includes_transcript(self):
        llm = FakeLLM()
        VoiceTurn(llm, FakeTTS(), system_prompt="Be brief.").run("मेरा नाम क्या है")
        assert "मेरा नाम क्या है" in llm.calls[0]
        assert "Be brief." in llm.calls[0]
