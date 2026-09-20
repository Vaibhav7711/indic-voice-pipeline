"""Repetition penalty and loop guard in LLMRunner, without a model (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from llm.runner import LLMRunner  # noqa: E402


def _runner(**kw):
    tok = SimpleNamespace(eos_token_id=0)
    return LLMRunner(model=None, tokenizer=tok, device=torch.device("cpu"), **kw)


def test_penalty_demotes_already_seen_tokens():
    logits = torch.tensor([[0.0, 2.0, 1.9]])
    assert _runner(repetition_penalty=1.0)._select(logits, [1]).item() == 1
    assert _runner(repetition_penalty=1.2)._select(logits, [1]).item() == 2


def test_penalty_multiplies_negative_scores():
    logits = torch.tensor([[-1.0, -0.5, -3.0]])
    # token 1 seen: -0.5 * 1.5 = -0.75, still the best -> stays.
    assert _runner(repetition_penalty=1.5)._select(logits, [1]).item() == 1
    # token 1 seen with a harsher penalty: -0.5 * 3 = -1.5 < -1.0 -> token 0.
    assert _runner(repetition_penalty=3.0)._select(logits, [1]).item() == 0


def test_no_seen_tokens_is_plain_argmax():
    logits = torch.tensor([[0.1, 0.3, 0.2]])
    assert _runner()._select(logits, []).item() == 1
