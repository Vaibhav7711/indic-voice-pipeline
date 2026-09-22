"""Notebook front end: WAV packing, session summary, and the no-model paths."""

from __future__ import annotations

import io
import wave

import numpy as np

from demo.notebook import NotebookAgent, pcm_to_wav_bytes


def test_pcm_to_wav_bytes_is_a_real_wav():
    pcm = (np.sin(np.arange(16_000) * 0.1) * 10_000).astype(np.int16)
    data = pcm_to_wav_bytes(pcm, 16_000)
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 16_000
        back = np.frombuffer(handle.readframes(16_000), dtype=np.int16)
    assert np.array_equal(back, pcm)


def test_empty_pcm_still_produces_a_valid_container():
    with wave.open(io.BytesIO(pcm_to_wav_bytes(np.zeros(0, np.int16), 16_000))) as h:
        assert h.getnframes() == 0


class TestNotebookAgent:
    def test_construction_loads_nothing(self):
        agent = NotebookAgent(adapter="/nowhere")
        assert agent.asr is None and agent.conversation is None
        assert agent.config["adapter"] == "/nowhere"
        assert agent.config["tts"] == "mms"

    def test_summary_is_empty_before_any_turn(self):
        assert NotebookAgent().summary() == {}

    def test_summary_averages_recorded_turns_and_skips_unmeasured(self):
        agent = NotebookAgent()
        agent.turns = [
            {"asr_ms": 700.0, "asr_rtf": 0.07, "first_token_ms": 90.0,
             "to_audio_ms": 300.0, "response_latency_ms": 1090.0, "turn_total_ms": 2000.0},
            {"asr_ms": 900.0, "asr_rtf": 0.09, "first_token_ms": 110.0,
             "to_audio_ms": None, "response_latency_ms": None, "turn_total_ms": 2400.0},
        ]
        s = agent.summary()
        assert s["turns"] == 2
        assert s["asr_ms_mean"] == 800.0
        assert s["first_token_ms_mean"] == 100.0
        assert s["to_audio_ms_mean"] == 300.0, "unmeasured turns must not count as zero"
        assert s["response_latency_ms_mean"] == 1090.0

    def test_reset_is_safe_without_a_conversation(self, capsys):
        NotebookAgent(history_turns=0).reset()
        assert "cleared" in capsys.readouterr().out

    def test_history_turns_zero_means_no_memory(self):
        assert NotebookAgent(history_turns=0).config["history_turns"] == 0


class _FakeAsr:
    """Text that depends on the audio length, so reuse is observable."""

    def __init__(self):
        self.calls = []

    def _result(self, seconds):
        from types import SimpleNamespace

        self.calls.append(round(seconds, 2))
        return SimpleNamespace(
            text=f"बात {len(self.calls)}", token_ids=[], language="hi",
            metrics=SimpleNamespace(total_ms=20.0, real_time_factor=0.1,
                                    audio_duration_seconds=seconds,
                                    hit_token_budget=False),
        )

    def transcribe_array(self, waveform, sample_rate, **kw):
        return self._result(len(waveform) / sample_rate)

    transcribe_long_array = transcribe_array


class _FakeLlm:
    def __init__(self):
        self.prompts = []

    def stream(self, prompt, **kw):
        self.prompts.append(prompt)
        yield "ठीक है।"


class _FakeTts:
    streaming = True

    def __init__(self):
        from agent.audio import PCM16_16K

        self.format = PCM16_16K

    def stream(self, text):
        yield np.zeros(1600, dtype=np.int16).tobytes()


def _wired_agent(tmp_path, seconds=2.0, amplitude=0.3, gap=0.0):
    """A NotebookAgent with fakes in place of models, plus a WAV on disk."""
    import soundfile as sf

    from agent import Conversation
    from demo.notebook import NotebookAgent

    parts = [np.full(int(seconds * 16_000), amplitude, np.float32)]
    if gap:
        parts += [np.zeros(int(gap * 16_000), np.float32),
                  np.full(int(seconds * 16_000), amplitude, np.float32)]
    path = tmp_path / "clip.wav"
    sf.write(path, np.concatenate(parts), 16_000)

    agent = NotebookAgent(history_turns=4)
    agent.asr, agent.llm_runner, agent.synth = _FakeAsr(), _FakeLlm(), _FakeTts()
    agent.conversation = Conversation(max_turns=4)
    agent.load = lambda: agent            # models already in place
    return agent, str(path)


class TestStreamTurn:
    def test_one_utterance_is_endpointed_and_answered(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=2.0)
        out = agent.stream_turn(path, speak=True, quiet=True)

        assert out["utterances"] == 1
        answer = out["answers"][0]
        assert answer["state"] == "completed" and answer["response"] == "ठीक है।"
        assert answer["endpoint_reason"] == "silence"
        # The silence wait is min_silence_ms by construction; it is the dead
        # time the user feels and must appear in the latency.
        assert answer["silence_wait_ms"] >= out["vad"]["min_silence_ms"]
        assert answer["response_latency_ms"] >= answer["endpoint_to_final_ms"]

    def test_candidate_final_removes_asr_from_the_wait(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=2.0)
        out = agent.stream_turn(path, quiet=True, speak=False)
        answer = out["answers"][0]
        assert out["candidates"] >= 1
        assert answer["from_candidate"] is True
        assert answer["asr_after_endpoint_ms"] == 0.0
        assert answer["asr_ms"] > 0, "the compute still happened, just earlier"

    def test_disabling_candidates_puts_asr_back_after_the_endpoint(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=2.0)
        out = agent.stream_turn(path, quiet=True, speak=False, early_final_silence_ms=0)
        answer = out["answers"][0]
        assert out["candidates"] == 0
        assert answer["from_candidate"] is False
        assert answer["asr_after_endpoint_ms"] > 0
        assert answer["endpoint_to_final_ms"] > answer["silence_wait_ms"]

    def test_a_long_pause_produces_two_utterances_and_two_answers(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=1.5, gap=1.2)
        out = agent.stream_turn(path, quiet=True, speak=False)
        assert out["utterances"] == 2
        first, second = (a["transcript"] for a in out["answers"])
        assert first != second
        # Each answer is a turn, so memory grows and the second prompt carries
        # the first exchange.
        assert agent.conversation.snapshot()["turns"] == 2
        assert first in agent.llm_runner.prompts[1]
        assert "ठीक है।" in agent.llm_runner.prompts[1]
        assert second in agent.llm_runner.prompts[1]

    def test_trace_records_partials_and_finals_in_time_order(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=4.0)
        out = agent.stream_turn(path, quiet=True, speak=False)
        kinds = [e["kind"] for e in out["trace"]]
        times = [e["at"] for e in out["trace"]]
        assert "final" in kinds
        assert times == sorted(times)
        assert out["partials"] >= 1, "partials must appear while audio arrives"

    def test_silence_only_clip_answers_nothing(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=2.0, amplitude=0.0)
        out = agent.stream_turn(path, quiet=True, speak=False)
        assert out["utterances"] == 0 and out["answers"] == []
        assert agent.asr.calls == [], "no ASR should run on silence"

    def test_session_summary_includes_streaming_turns(self, tmp_path):
        agent, path = _wired_agent(tmp_path, seconds=2.0)
        agent.stream_turn(path, quiet=True, speak=False)
        summary = agent.summary()
        assert summary["turns"] == 1
        assert summary["response_latency_ms_mean"] is not None
