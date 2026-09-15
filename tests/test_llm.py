"""LLM runner correctness: explicit loop vs model.generate()."""

from __future__ import annotations

import pytest

# importorskip so the module skips cleanly at collection time when torch is
# absent. Without it, `pytest tests/` fails outright on a CPU-only machine
# and the CPU-runnable scoring tests never get a chance to run.
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def llm():
    from llm import load_llm
    return load_llm("Qwen/Qwen3-0.6B")


class TestLLMRunner:
    def test_generates_text(self, llm):
        from llm.runner import LLMRunner

        runner = LLMRunner(llm.model, llm.tokenizer, llm.device)
        result = runner.generate("What is 2+2?", max_new_tokens=16)
        assert len(result.token_ids) > 0
        assert len(result.text) > 0

    def test_metrics_populated(self, llm):
        from llm.runner import LLMRunner

        runner = LLMRunner(llm.model, llm.tokenizer, llm.device)
        result = runner.generate("Hello", max_new_tokens=8)
        assert result.metrics.prefill_ms > 0
        assert result.metrics.total_ms > 0
        assert result.metrics.prompt_tokens > 0
