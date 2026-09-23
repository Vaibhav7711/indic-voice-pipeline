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


class TestModelSpec:
    def test_quantization_suffix_is_parsed(self):
        from benchmarks.llm_bakeoff import parse_model_spec

        assert parse_model_spec("Qwen/Qwen3-1.7B") == ("Qwen/Qwen3-1.7B", None)
        assert parse_model_spec("Qwen/Qwen3-1.7B:4bit") == ("Qwen/Qwen3-1.7B", "4bit")
        assert parse_model_spec("Qwen/Qwen3-4B:8bit") == ("Qwen/Qwen3-4B", "8bit")

    def test_other_colons_are_left_alone(self):
        from benchmarks.llm_bakeoff import parse_model_spec

        assert parse_model_spec("C:/models/local") == ("C:/models/local", None)
        assert parse_model_spec("org/name:v2") == ("org/name:v2", None)


class TestVramArithmetic:
    def test_footprints_match_the_planning_table(self):
        from llm.loader import estimate_vram_gib

        turbo, mms = estimate_vram_gib(809), estimate_vram_gib(36)
        fp16 = turbo + estimate_vram_gib(1700) + mms
        nf4 = turbo + estimate_vram_gib(1700, "4bit") + mms
        # The 6 GB decision: 1.7B in fp16 alongside Whisper leaves too little
        # for KV cache and activations; at 4-bit it does not.
        assert 4.5 < fp16 < 5.0
        assert 2.2 < nf4 < 2.7
        assert estimate_vram_gib(4000) > 7.0, "Qwen3-4B fp16 is out of reach on 6 GB"

    def test_quantization_is_rejected_without_cuda(self):
        import pytest as _pytest

        from llm.loader import load_llm

        with _pytest.raises(ValueError, match="requires a CUDA device"):
            load_llm("Qwen/Qwen3-0.6B", device="cpu", quantization="4bit")

    def test_unknown_quantization_is_rejected(self):
        import pytest as _pytest

        from llm.loader import load_llm

        with _pytest.raises(ValueError, match="must be None"):
            load_llm("Qwen/Qwen3-0.6B", device="cpu", quantization="3bit")
