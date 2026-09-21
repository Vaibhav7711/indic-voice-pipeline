"""Bake-off scoring helpers (CPU)."""

from __future__ import annotations

from benchmarks.llm_bakeoff import DEFAULT_PROMPTS, devanagari_ratio, summarize


def test_devanagari_ratio():
    assert devanagari_ratio("नमस्ते दुनिया") == 1.0
    assert devanagari_ratio("Hello world") == 0.0
    assert devanagari_ratio("मेरा laptop slow है") < 0.8
    assert devanagari_ratio("123 !!") is None


def test_prompts_are_hindi_and_spoken_length():
    assert len(DEFAULT_PROMPTS) >= 10
    assert all(devanagari_ratio(p) and devanagari_ratio(p) > 0.5 for p in DEFAULT_PROMPTS)
    assert all(len(p) < 120 for p in DEFAULT_PROMPTS)


def test_summarize_counts_quality_flags_and_averages():
    rows = [
        {"prefill_ms": 100.0, "ms_per_token": 40.0, "first_sentence_ms": 700.0,
         "first_sentence_tokens": 15, "total_ms": 2000.0, "generated_tokens": 47,
         "devanagari_ratio": 0.98, "think_leak": False, "repetition_stop": False, "empty": False},
        {"prefill_ms": 120.0, "ms_per_token": 42.0, "first_sentence_ms": 500.0,
         "first_sentence_tokens": 10, "total_ms": 1500.0, "generated_tokens": 33,
         "devanagari_ratio": 0.6, "think_leak": True, "repetition_stop": True, "empty": False},
        {"prompt": "x", "error": "OOM"},
    ]
    s = summarize(rows)
    assert s["prompts"] == 3 and s["errors"] == 1
    assert s["prefill_ms_mean"] == 110.0
    assert s["first_sentence_ms_p50"] == 600.0
    assert s["devanagari_below_0_8"] == 1
    assert s["think_leaks"] == 1 and s["repetition_stops"] == 1 and s["empty"] == 0
