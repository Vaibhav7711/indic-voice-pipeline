"""Whisper decoder prompt grammar and language detection, on CPU.

The decoder is driven with a fake model so no weights are needed. What is
pinned down: the prompt is always <|sot|><|lang|><|task|><|notimestamps|>,
a missing language is an error rather than a silently malformed prompt, and
detection is an argmax restricted to language tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from asr.explicit.decoder import WhisperDecoder  # noqa: E402

SOT, HI, EN, TE, TRANSCRIBE, NOTS = 50258, 50276, 50259, 50299, 50359, 50363
VOCAB = 51865


def _gen_config():
    return SimpleNamespace(
        decoder_start_token_id=SOT,
        lang_to_id={"<|hi|>": HI, "<|en|>": EN, "<|te|>": TE},
        task_to_id={"transcribe": TRANSCRIBE, "translate": 50358},
        no_timestamps_token_id=NOTS,
    )


class _FakeModel:
    """Returns logits that favour one language token, and a huge score on an
    unrelated token so a naive full-vocab argmax would pick the wrong thing."""

    def __init__(self, favoured: int):
        self.generation_config = _gen_config()
        self.favoured = favoured
        self.seen_inputs = []

    def __call__(self, *, encoder_outputs, decoder_input_ids, **kw):
        self.seen_inputs.append(decoder_input_ids.tolist())
        logits = torch.zeros(1, decoder_input_ids.shape[1], VOCAB)
        logits[0, -1, 1234] = 50.0          # a non-language token
        logits[0, -1, self.favoured] = 5.0
        logits[0, -1, EN] = 4.0
        return SimpleNamespace(logits=logits)


def _decoder(favoured=HI):
    return WhisperDecoder(_FakeModel(favoured), torch.device("cpu"))


def test_prompt_is_sot_lang_task_notimestamps():
    dec = _decoder()
    assert dec._build_prompt_ids("hi") == [SOT, HI, TRANSCRIBE, NOTS]
    assert dec._build_prompt_ids("te", timestamps=True) == [SOT, TE, TRANSCRIBE]


def test_missing_language_is_an_error_not_a_malformed_prompt():
    dec = _decoder()
    with pytest.raises(ValueError, match="language is required"):
        dec._build_prompt_ids(None)
    with pytest.raises(ValueError, match="unknown Whisper language"):
        dec._build_prompt_ids("xx")


def test_bare_language_codes_are_accepted_in_lang_to_id():
    dec = _decoder()
    dec.gen_config.lang_to_id = {"hi": HI}
    assert dec._build_prompt_ids("hi")[1] == HI


def test_detect_language_uses_sot_only_and_restricts_to_language_tokens():
    dec = _decoder(favoured=TE)
    code, prob, ms = dec.detect_language(encoder_outputs=None)
    assert code == "te"
    assert 0.5 < prob < 1.0          # softmax over {hi:0, en:4, te:5}
    assert ms >= 0.0
    assert dec.model.seen_inputs == [[[SOT]]]


def test_strip_generate_output_handles_both_transformers_shapes():
    dec = _decoder()
    dec.gen_config.eos_token_id = 50257
    content = [100, 200, 300]
    # transformers < 5: prompt included, EOS at the end.
    assert dec.strip_generate_output([SOT, HI, TRANSCRIBE, NOTS] + content + [50257]) == content
    # transformers >= 5: prompt stripped already.
    assert dec.strip_generate_output(content + [50257]) == content
    # Explicit-loop output (content + EOS) reduces to the same thing.
    assert dec.strip_generate_output(content + [50257]) == dec.strip_generate_output(content)
    assert dec.strip_generate_output([]) == []


def test_detect_language_candidates_restrict_the_argmax():
    dec = _decoder(favoured=TE)              # te=5, en=4, hi=0
    code, prob, _ = dec.detect_language(None, candidates=["hi", "en"])
    assert code == "en"
    assert prob > 0.9                        # softmax over {hi:0, en:4}
    with pytest.raises(ValueError, match="unknown Whisper language codes"):
        dec.detect_language(None, candidates=["hi", "xx"])


def test_pick_applies_suppress_and_begin_suppress_like_generate():
    dec = _decoder()
    dec.gen_config.suppress_tokens = [7]
    dec.gen_config.begin_suppress_tokens = [9]
    dec = WhisperDecoder(dec.model, torch.device("cpu"))     # re-read config

    logits = torch.zeros(1, VOCAB)
    logits[0, 7] = 3.0      # always suppressed
    logits[0, 9] = 2.0      # suppressed only at begin
    logits[0, 11] = 1.0
    assert dec._pick(logits, at_begin=True).item() == 11
    assert dec._pick(logits, at_begin=False).item() == 9


def test_pick_without_suppression_is_plain_argmax():
    dec = _decoder()
    logits = torch.zeros(1, VOCAB)
    logits[0, 5] = 1.0
    assert dec._pick(logits, at_begin=True).item() == 5


# ---------------------------------------------------------------------------
# Token budget: Devanagari needs far more tokens than the old 225 default
# ---------------------------------------------------------------------------


class _BudgetModel(_FakeModel):
    """Never emits EOS, so the decode always runs to the budget."""

    def __init__(self, positions=448):
        super().__init__(favoured=HI)
        self.generation_config.eos_token_id = 50257
        self.config = SimpleNamespace(max_target_positions=positions)

    def __call__(self, *, encoder_outputs=None, decoder_input_ids=None, **kw):
        logits = torch.zeros(1, decoder_input_ids.shape[1], VOCAB)
        logits[0, -1, 1234] = 10.0                 # a normal token, never EOS
        return SimpleNamespace(logits=logits, past_key_values=None)

    def get_encoder(self):
        return lambda *a, **k: SimpleNamespace(last_hidden_state=torch.zeros(1, 1500, 1024))


def _budget_runner(positions=448):
    from asr.explicit.runner import ASRRunner

    model = _BudgetModel(positions)
    processor = SimpleNamespace(
        feature_extractor=lambda *a, **k: SimpleNamespace(
            input_features=torch.zeros(1, 80, 3000),
        ),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "x" * len(ids)),
    )
    return ASRRunner(model, processor, torch.device("cpu"), torch.float32)


def test_budget_defaults_to_the_models_limit_minus_the_prompt():
    r = _budget_runner()
    # prompt is <|sot|><|hi|><|transcribe|><|notimestamps|> = 4 tokens
    assert r.token_budget("hi", None) == 444
    assert r.token_budget("hi", 100) == 100          # caller may ask for less
    assert r.token_budget("hi", 10_000) == 444       # never more than the model
    assert _budget_runner(positions=200).token_budget("hi", None) == 196


def test_runner_default_can_be_set_at_construction():
    from asr.explicit.runner import ASRRunner

    r = _budget_runner()
    r2 = ASRRunner(r.model, r.processor, torch.device("cpu"), torch.float32, max_new_tokens=64)
    assert r2.token_budget("hi", None) == 64
    assert r2.token_budget("hi", 32) == 32


def test_hitting_the_budget_is_recorded_not_silent():
    r = _budget_runner()
    result = r.transcribe_array(np.zeros(16_000, dtype=np.float32), 16_000,
                                language="hi", max_new_tokens=12)
    assert result.metrics.token_budget == 12
    assert result.metrics.decoder_steps == 12
    assert result.metrics.hit_token_budget is True, "a cut-off transcript must be flagged"
