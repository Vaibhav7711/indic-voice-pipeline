"""Pipeline integration tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def pipe():
    from asr.explicit import load_whisper
    from llm import load_llm
    from pipeline import VoicePipeline

    w = load_whisper("openai/whisper-small")
    l = load_llm("Qwen/Qwen3-0.6B")
    return VoicePipeline(w, l)


@pytest.fixture(scope="module")
def tone():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


class TestPipeline:
    def test_produces_output(self, pipe, tone):
        result = pipe.run_array(tone, 16000, language="en", llm_max_tokens=16)
        assert isinstance(result.answer, str)
        assert len(result.answer_token_ids) > 0

    def test_metrics_complete(self, pipe, tone):
        result = pipe.run_array(tone, 16000, language="en", llm_max_tokens=16)
        m = result.metrics
        assert m.asr is not None
        assert m.asr.encoder_ms > 0
        assert m.llm_prefill_ms > 0
        assert m.total_pipeline_ms > 0
        assert m.memory_strategy in ("concurrent", "sequential")

    def test_concurrent_strategy(self, pipe):
        assert pipe.strategy.value == "concurrent"
