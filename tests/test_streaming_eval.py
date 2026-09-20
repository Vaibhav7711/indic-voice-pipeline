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
    assert vad_from_name("pad300").padding_ms == 300
    assert vad_from_name("fixed40").adaptive_threshold is False
    assert set(VAD_GRID) >= {"default", "fixed40", "pad300", "floor70"}


def test_stream_clip_reports_one_final_and_full_vad_agreement():
    t = _Transcriber()
    out = stream_clip(t, _speech(2.0), vad_from_name("default"), language="hi",
                      lead=0.5, trail=1.0)
    assert len(out["finals"]) == 1
    assert out["text"] == "नमस्ते दुनिया"
    assert out["vad_agreement"] is not None and out["vad_agreement"] > 0.95
    assert out["first_final_start"] is not None and abs(out["first_final_start"]) <= 0.25
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
