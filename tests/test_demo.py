"""Demo formatting and audio handling — the parts that need no models."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agent.conversation import Conversation
from demo.app import (
    DemoAgent,
    history_markdown,
    latency_table,
    stream_status,
    to_float_mono,
)


class TestToFloatMono:
    def test_int16_stereo_becomes_float_mono_in_range(self):
        stereo = np.stack([np.full(100, 16384, np.int16), np.full(100, -16384, np.int16)], axis=1)
        wave, rate = to_float_mono((44100, stereo))
        assert rate == 44100 and wave.shape == (100,) and wave.dtype == np.float32
        assert abs(float(wave[0])) < 1e-6          # +/- cancel to silence

    def test_float_input_is_untouched(self):
        wave, rate = to_float_mono((16000, np.full(10, 0.5, np.float32)))
        assert rate == 16000 and np.allclose(wave, 0.5)

    def test_int16_scales_to_unit_range(self):
        wave, _ = to_float_mono((16000, np.full(4, np.iinfo(np.int16).max, np.int16)))
        assert np.allclose(wave, 1.0, atol=1e-4)


class TestLatencyTable:
    def _metrics(self, **kw):
        base = dict(mel_extraction_ms=9.0, encoder_ms=128.0, total_decode_ms=380.0,
                    total_ms=520.0, hit_token_budget=False, language_probability=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def _turn(self, **kw):
        base = dict(final_transcript_to_first_llm_token_ms=92.0,
                    first_llm_token_to_playback_start_ms=610.0,
                    response_latency_ms=1222.0, llm_total_ms=1900.0,
                    tts_first_chunk_ms=240.0, total_turn_ms=2600.0,
                    first_token_is_prefill_proxy=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_renders_measured_values(self):
        out = latency_table(self._metrics(), self._turn())
        assert "| Whisper encoder | 128 |" in out
        assert "**Response latency** (excl. endpoint silence)" in out
        assert "**1222**" in out

    def test_unmeasured_fields_show_a_dash_not_zero(self):
        out = latency_table(self._metrics(), self._turn(response_latency_ms=None,
                                                        tts_first_chunk_ms=None))
        assert "**—**" in out and "| TTS first chunk | — |" in out
        assert "| 0 |" not in out

    def test_caveats_are_surfaced(self):
        out = latency_table(self._metrics(hit_token_budget=True, language_probability=0.99),
                            self._turn(first_token_is_prefill_proxy=True))
        assert "cut off at the token budget" in out
        assert "prefill proxy" in out
        assert "p=0.99" in out


class TestHistoryMarkdown:
    def test_no_conversation_and_empty_conversation(self):
        assert "memory disabled" in history_markdown(None)
        assert "no turns yet" in history_markdown(Conversation("SYS"))

    def test_exchanges_and_interruption_marker(self):
        c = Conversation("SYS")
        c.record("दिल्ली का मौसम?", "धूप है।")
        c.record("और मुंबई?", "बारिश", interrupted=True)
        out = history_markdown(c)
        assert "2 turn(s)" in out and "tokens in prompt" in out
        assert "दिल्ली का मौसम?" in out and "_(interrupted)_" in out

    def test_dropped_turns_are_reported(self):
        c = Conversation("SYS", max_turns=1)
        c.record("a", "b")
        c.record("c", "d")
        assert "1 dropped" in history_markdown(c)


def test_stream_status_counts_update_kinds():
    from asr.streaming import UpdateKind

    updates = [SimpleNamespace(kind=UpdateKind.PARTIAL), SimpleNamespace(kind=UpdateKind.PARTIAL),
               SimpleNamespace(kind=UpdateKind.FINAL)]
    out = stream_status(updates, "speaking")
    assert "state: speaking" in out and "partial: 2" in out and "final: 1" in out
    assert stream_status([], "idle") == "state: idle"


class TestDemoAgentWithoutModels:
    def test_construction_loads_nothing(self):
        agent = DemoAgent()
        assert agent.asr is None and agent.conversation is None
        assert "not loaded" in agent.status()

    def test_no_audio_short_circuits_before_loading(self):
        agent = DemoAgent()
        transcript, answer, table, audio, history = agent.run_turn(None, "hi", 96, True)
        assert (transcript, answer, audio) == ("", "", None)
        assert "record something" in table
        assert agent.asr is None, "must not load models just to reject empty input"

    def test_reset_is_safe_before_loading(self):
        assert DemoAgent().reset()[4] == "_memory disabled_"

    def test_push_chunk_without_audio_does_not_load(self):
        agent = DemoAgent()
        session, text, status = agent.push_chunk(None, None)
        assert session is None and text == "" and "waiting" in status
        assert agent.asr is None

    def test_config_reads_environment(self, monkeypatch):
        from demo.app import Config

        monkeypatch.setenv("WHISPER_MODEL", "openai/whisper-small")
        monkeypatch.setenv("TTS_BACKEND", "mms")
        monkeypatch.delenv("WHISPER_ADAPTER_PATH", raising=False)
        c = Config()
        assert c.whisper_model == "openai/whisper-small"
        assert c.tts_backend == "mms" and c.adapter is None


class TestFailureReachesTheBrowser:
    def test_model_load_error_is_shown_not_swallowed(self, monkeypatch):
        agent = DemoAgent()
        monkeypatch.setattr(type(agent), "load",
                            lambda self: (_ for _ in ()).throw(RuntimeError("no CUDA")))
        transcript, answer, table, audio, _ = agent.run_turn(
            (16000, np.zeros(1600, np.float32)), "hi", 96, False,
        )
        assert "model load failed" in table and "no CUDA" in table
        assert (transcript, answer, audio) == ("", "", None)

    def test_streaming_load_error_is_shown(self, monkeypatch):
        agent = DemoAgent()
        monkeypatch.setattr(type(agent), "new_session",
                            lambda self: (_ for _ in ()).throw(RuntimeError("no CUDA")))
        session, text, status = agent.push_chunk((16000, np.zeros(160, np.float32)), None)
        assert session is None and text == "" and "model load failed" in status


def test_launch_kwargs_disable_ssr_by_default():
    """Gradio 5+ SSR starts a Node subprocess; where that is blocked the page
    is unreachable even though Python is listening."""
    from demo.app import launch_kwargs

    kw = launch_kwargs()
    assert kw["ssr_mode"] is False
    assert kw["server_name"] == "0.0.0.0" and kw["server_port"] == 7860
    assert launch_kwargs(share=True, server_port=7000)["share"] is True
    assert launch_kwargs(server_port=7000)["server_port"] == 7000
    assert launch_kwargs()["ssr_mode"] is False
