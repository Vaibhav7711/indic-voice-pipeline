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
