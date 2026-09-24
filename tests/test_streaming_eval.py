"""Streaming benchmark plumbing, on CPU with a fake transcriber."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from benchmarks.streaming_eval import (
    VAD_GRID,
    aggregate,
    onset_hallucination,
    stream_clip,
    vad_from_name,
)

SR = 16_000


class _Transcriber:
    def __init__(self, text="नमस्ते दुनिया"):
        self.text = text
        self.calls = 0

    def _result(self, wav):
        self.calls += 1
        return SimpleNamespace(text=self.text, token_ids=[], language="hi",
                               metrics=SimpleNamespace(total_ms=5.0, real_time_factor=0.1))

    def transcribe_array(self, waveform, sample_rate, **kw):
        return self._result(waveform)

    transcribe_long_array = transcribe_array


def _speech(seconds, dbfs=-30.0):
    return np.full(int(seconds * SR), 10 ** (dbfs / 20), dtype=np.float32)


def test_grid_names_resolve_and_default_is_untouched():
    assert vad_from_name("default").as_dict() == vad_from_name("default").as_dict()
    assert vad_from_name("pad500").padding_ms == 500
    assert vad_from_name("fixed40").adaptive_threshold is False
    assert set(VAD_GRID) >= {"default", "fixed40", "v1-adaptive", "pad500", "floor60"}


def test_stream_clip_reports_one_final_and_full_vad_agreement():
    t = _Transcriber()
    out = stream_clip(t, _speech(2.0), vad_from_name("default"), language="hi",
                      lead=0.5, trail=1.0)
    assert len(out["finals"]) == 1
    assert out["text"] == "नमस्ते दुनिया"
    assert out["vad_agreement"] is not None and out["vad_agreement"] > 0.95
    vad = vad_from_name("default")
    pre_roll = (vad.padding_ms + vad.frame_ms) / 1000          # padded onset, one frame slack
    assert out["first_final_start"] is not None
    assert -pre_roll <= out["first_final_start"] <= 0.0
    assert t.calls == 1                       # partials disabled: one decode per final


def test_stream_clip_counts_a_split_as_two_finals():
    wav = np.concatenate([_speech(1.0), np.zeros(SR, dtype=np.float32), _speech(1.0)])
    out = stream_clip(_Transcriber(), wav, vad_from_name("default"), language="hi",
                      lead=0.5, trail=1.0)
    assert len(out["finals"]) == 2
    assert out["offline_vad_segments"] == 2


def test_onset_hallucination_flags_repeated_head_only():
    assert onset_hallucination("जी जी जी कैमिकल का पीएज कहा जाता है")
    assert not onset_hallucination("इसे कैमिकल का पीएज कहा जाता है")
    assert not onset_hallucination("इसे कैमिकल का पीएज कहा जाता है जी जी जी")  # not at onset


def test_aggregate_separates_model_error_from_streaming_penalty():
    rows = [
        {"reference": "एक दो तीन चार", "offline_text": "एक दो तीन चार",
         "streamed_text": "एक दो तीन चार", "n_finals": 1, "vad_agreement": 1.0,
         "onset_hallucination": False, "endpoint_reasons": ["silence"],
         "streamed_asr_ms": 10.0, "offline_asr_ms": 10.0},
        {"reference": "एक दो तीन चार", "offline_text": "एक दो तीन पांच",
         "streamed_text": "दो तीन पांच", "n_finals": 2, "vad_agreement": 0.8,
         "onset_hallucination": False, "endpoint_reasons": ["silence", "silence"],
         "streamed_asr_ms": 12.0, "offline_asr_ms": 10.0},
    ]
    m = aggregate(rows, "standard")
    assert m["offline_wer_vs_reference"]["micro_percent"] == 12.5     # 1 / 8
    assert m["wer_vs_reference"]["micro_percent"] == 25.0             # 2 / 8
    assert m["wer_vs_offline"]["micro_percent"] == 12.5               # 1 deletion / 8
    assert m["streaming_penalty_points"] == 12.5
    assert m["structure"]["clips_split"] == 1
    assert m["structure"]["clips_with_one_final"] == 1
    assert m["structure"]["endpoint_reasons"] == {"silence": 3}
    assert m["structure"]["vad_agreement_mean"] == 0.9


def test_session_grid_early_reuses_candidate_and_reports_zero_asr_after_endpoint():
    from benchmarks.streaming_eval import session_kwargs

    t = _Transcriber()
    out = stream_clip(t, _speech(2.0), vad_from_name("default"), language="hi",
                      lead=0.5, trail=1.0, session=session_kwargs("early"))
    assert out["candidates"] == 1 and len(out["finals"]) == 1
    assert out["finals_from_candidate"] == 1
    assert out["asr_ms_after_endpoint"] == 0.0
    assert out["asr_ms_total"] == 5.0                       # the candidate's decode
    # end→final = the 600 ms silence wait only (no ASR after the endpoint).
    assert 600.0 <= out["endpoint_to_final_ms_last"] <= 600.0 + 130.0   # + frame + one 100 ms block


def test_session_grid_baseline_pays_asr_after_endpoint():
    from benchmarks.streaming_eval import session_kwargs

    out = stream_clip(_Transcriber(), _speech(2.0), vad_from_name("default"), language="hi",
                      lead=0.5, trail=1.0, session=session_kwargs("baseline"))
    assert out["candidates"] == 0
    assert out["asr_ms_after_endpoint"] == 5.0
    assert 605.0 <= out["endpoint_to_final_ms_last"] <= 605.0 + 130.0


def test_session_grid_names():
    from benchmarks.streaming_eval import SESSION_GRID, session_kwargs

    assert {"baseline", "early", "early-incr", "early-sem", "full"} <= set(SESSION_GRID)
    assert session_kwargs("full")["incremental_finals"] and session_kwargs("full")["semantic_endpointing"]


class TestGridActuallyMeasuresWhatItClaims:
    def test_incremental_configs_use_a_positive_partial_interval(self):
        """The session treats partial_interval_ms <= 0 as 'partials off', so a
        zero here made early-incr silently identical to early."""
        from benchmarks.streaming_eval import INCREMENTAL_CONFIGS, session_kwargs

        for name in INCREMENTAL_CONFIGS:
            kwargs = session_kwargs(name)
            assert kwargs["emit_partials"] is True, name
            assert kwargs["partial_interval_ms"] > 0, name
            assert kwargs["incremental_finals"] is True, name

    def test_sanity_check_flags_a_config_that_produced_no_partials(self):
        from benchmarks.streaming_eval import grid_sanity

        rows = [{"partials": 0, "finals_reused_partial": 0} for _ in range(5)]
        warning = grid_sanity("early-incr", rows)
        assert warning and "ZERO partials" in warning

    def test_sanity_check_flags_partials_that_were_never_reused(self):
        from benchmarks.streaming_eval import grid_sanity

        rows = [{"partials": 4, "finals_reused_partial": 0} for _ in range(5)]
        warning = grid_sanity("full", rows)
        assert warning and "never engaged" in warning

    def test_healthy_incremental_run_and_non_incremental_configs_pass(self):
        from benchmarks.streaming_eval import grid_sanity

        healthy = [{"partials": 4, "finals_reused_partial": 1} for _ in range(5)]
        assert grid_sanity("early-incr", healthy) is None
        assert grid_sanity("baseline", [{"partials": 0}]) is None
        assert grid_sanity("default+early-incr", healthy) is None


class TestIncrementalTailIsActuallyATail:
    def test_short_utterance_falls_back_to_a_full_decode(self, tmp_path):
        """When the partial covers less than the overlap, there is no tail to
        decode; the old code full-decoded and then concatenated onto the
        partial's text, duplicating the opening."""
        from benchmarks.streaming_eval import session_kwargs

        agent_kwargs = session_kwargs("early-incr")
        t = _Transcriber()
        out = stream_clip(t, _speech(1.2), vad_from_name("default"), language="hi",
                          lead=0.5, trail=1.0, session=agent_kwargs)
        finals = out["finals"]
        assert finals
        for final in finals:
            if final["reused_partial"]:
                # If it claims reuse, the decode must be strictly shorter than
                # the utterance — otherwise it was not incremental at all.
                assert final["decoded_seconds"] < final["seconds"] - 0.05, final

    def test_long_utterance_can_reuse_a_partial(self, tmp_path):
        from benchmarks.streaming_eval import session_kwargs

        out = stream_clip(_Transcriber(), _speech(8.0), vad_from_name("default"),
                          language="hi", lead=0.5, trail=1.0,
                          session=session_kwargs("early-incr"))
        assert out["partials"] > 0, "a positive interval must let partials run"
