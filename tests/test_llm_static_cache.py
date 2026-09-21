"""Static-cache decode path produces the same tokens as the dynamic path.

Uses a tiny random Qwen3 built from config: no download, runs on CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llm.runner import LLMRunner  # noqa: E402


class _Tok:
    """Minimal tokenizer: prompt string → fixed ids; decode → ids as text."""

    eos_token_id = 255

    def __call__(self, prompt, return_tensors=None):
        from types import SimpleNamespace

        ids = torch.tensor([[int(c) for c in prompt.split()]])
        return SimpleNamespace(to=lambda d: {"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=256)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def test_static_cache_matches_dynamic_tokens(tiny_model):
    prompt = "3 14 15 92 65 35 89"
    dyn = LLMRunner(tiny_model, _Tok(), torch.device("cpu"), repetition_penalty=1.0)
    sta = LLMRunner(tiny_model, _Tok(), torch.device("cpu"), repetition_penalty=1.0,
                    static_cache=True, max_cache_len=64)
    a = dyn.generate(prompt, max_new_tokens=12)
    b = sta.generate(prompt, max_new_tokens=12)
    assert a.token_ids == b.token_ids and len(a.token_ids) > 0
    assert b.metrics.static_cache is True and a.metrics.static_cache is False
    assert b.metrics.prefill_ms > 0 and len(b.metrics.decode_ms) == len(a.metrics.decode_ms)


def test_static_cache_is_reset_between_calls(tiny_model):
    sta = LLMRunner(tiny_model, _Tok(), torch.device("cpu"), repetition_penalty=1.0,
                    static_cache=True, max_cache_len=64)
    first = sta.generate("3 14 15 92", max_new_tokens=6).token_ids
    sta.generate("7 7 7 7 7 7 7 7 7", max_new_tokens=6)          # pollute
    again = sta.generate("3 14 15 92", max_new_tokens=6).token_ids
    assert first == again


def test_static_cache_stream_matches_generate(tiny_model):
    sta = LLMRunner(tiny_model, _Tok(), torch.device("cpu"), repetition_penalty=1.0,
                    static_cache=True, max_cache_len=64)
    text = sta.generate("3 14 15 92 65", max_new_tokens=8).text
    streamed = "".join(sta.stream("3 14 15 92 65", max_new_tokens=8))
    assert streamed == text


def test_cache_length_is_enforced(tiny_model):
    sta = LLMRunner(tiny_model, _Tok(), torch.device("cpu"), static_cache=True, max_cache_len=10)
    with pytest.raises(ValueError, match="max_cache_len"):
        sta.generate("1 2 3 4 5 6", max_new_tokens=8)


def test_compile_requires_static_cache(tiny_model):
    with pytest.raises(ValueError, match="requires static_cache"):
        LLMRunner(tiny_model, _Tok(), torch.device("cpu"), compile_decode=True)
