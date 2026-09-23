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
    # Loop guard off: the fake model emits one token forever, which the guard
    # would (correctly) stop, and this test is about the budget.
    r = _budget_runner()
    r.loop_guard_ngram = 0
    result = r.transcribe_array(np.zeros(16_000, dtype=np.float32), 16_000,
                                language="hi", max_new_tokens=12)
    assert result.metrics.token_budget == 12
    assert result.metrics.decoder_steps == 12
    assert result.metrics.hit_token_budget is True, "a cut-off transcript must be flagged"


# ---------------------------------------------------------------------------
# Safeguards generate() has and a bare greedy loop does not
# ---------------------------------------------------------------------------

NO_SPEECH = 50363


class _ScriptedModel(_BudgetModel):
    """Emits a token script, and a settable P(<|nospeech|>) at the <|sot|>
    position — position 0 of the prefill forward, which is where Whisper's
    no-speech probability lives."""

    def __init__(self, script, no_speech_logit=-60.0, positions=448):
        super().__init__(positions)
        self.script = list(script)
        self.no_speech_logit = no_speech_logit
        self.steps = 0

    def __call__(self, *, encoder_outputs=None, decoder_input_ids=None, **kw):
        prefill = decoder_input_ids.shape[1] > 1
        logits = torch.full((1, decoder_input_ids.shape[1], VOCAB), -30.0)
        token = self.script[min(self.steps, len(self.script) - 1)]
        self.steps += 1
        logits[0, -1, token] = 10.0
        if prefill:
            logits[0, 0, NO_SPEECH] = self.no_speech_logit
        return SimpleNamespace(logits=logits, past_key_values=None)


def _runner(model, **kw):
    from asr.explicit.runner import ASRRunner

    # The tokenizer must answer the conversion calls, or the no-speech token
    # cannot be resolved and the check is (correctly) disabled.
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: " ".join("क" for _ in ids),
        convert_tokens_to_ids=lambda name: NO_SPEECH if name == "<|nospeech|>" else 50257,
        convert_ids_to_tokens=lambda i: "<|nospeech|>" if i == NO_SPEECH else "<|endoftext|>",
    )
    processor = SimpleNamespace(
        feature_extractor=lambda *a, **k: SimpleNamespace(
            input_features=torch.zeros(1, 80, 3000)),
        tokenizer=tokenizer,
    )
    return ASRRunner(model, processor, torch.device("cpu"), torch.float32, **kw)


def _transcribe(runner):
    return runner.transcribe_array(np.zeros(16_000, dtype=np.float32), 16_000, language="hi")


def test_no_speech_window_is_not_transcribed():
    """A high P(<|nospeech|>) must return empty, not a hallucinated phrase."""
    model = _ScriptedModel([1234], no_speech_logit=20.0)     # dominates the softmax
    result = _transcribe(_runner(model, no_speech_threshold=0.6))
    assert result.text == "" and result.token_ids == []
    assert result.metrics.no_speech is True
    assert result.metrics.no_speech_prob > 0.6
    assert result.metrics.decoder_steps == 0, "no decode loop should run"


def test_no_speech_check_can_be_disabled():
    model = _ScriptedModel([1234], no_speech_logit=20.0)
    result = _transcribe(_runner(model, no_speech_threshold=None, loop_guard_ngram=0))
    assert result.metrics.no_speech is False
    assert result.metrics.decoder_steps > 0


def test_speech_window_is_transcribed_and_probability_recorded():
    model = _ScriptedModel([1234, 1235, 0], no_speech_logit=-60.0)
    model.generation_config.eos_token_id = 0
    result = _transcribe(_runner(model))
    assert result.metrics.no_speech is False
    assert result.metrics.no_speech_prob < 0.01
    assert result.metrics.decoder_steps > 0


def test_repetition_loop_stops_early_and_is_flagged():
    """The live failure: one token repeated to the token budget, ~2 s of GPU
    time for garbage that then goes to the LLM as a question."""
    model = _ScriptedModel([7] * 200)
    model.generation_config.eos_token_id = 0
    guarded = _transcribe(_runner(model, loop_guard_ngram=1, loop_guard_repeats=4))
    assert guarded.metrics.stopped_on_repetition is True
    assert guarded.metrics.decoder_steps == 1, "keep one copy, drop the repeats"

    loose = _transcribe(_runner(_ScriptedModel([7] * 500), loop_guard_ngram=0))
    assert loose.metrics.stopped_on_repetition is False
    assert loose.metrics.decoder_steps == loose.metrics.token_budget
    assert guarded.metrics.decoder_steps < loose.metrics.decoder_steps / 10


def test_loop_guard_detects_multi_token_cycles():
    model = _ScriptedModel([11, 12, 13] * 50)
    model.generation_config.eos_token_id = 0
    result = _transcribe(_runner(model, loop_guard_ngram=3, loop_guard_repeats=3))
    assert result.metrics.stopped_on_repetition is True
    assert result.metrics.decoder_steps == 3


def test_loop_guard_does_not_fire_on_ordinary_text():
    script = list(range(100, 160)) + [0]
    model = _ScriptedModel(script)
    model.generation_config.eos_token_id = 0
    result = _transcribe(_runner(model, loop_guard_ngram=3, loop_guard_repeats=4))
    assert result.metrics.stopped_on_repetition is False
    assert result.metrics.decoder_steps == len(script)


def test_compression_ratio_separates_repetition_from_language():
    from asr.explicit.runner import compression_ratio

    repeated = "जी जैए " * 70
    natural = ("इसे केमिकल का पीएच कहा जाता है आप लाल गोभी के जूस को "
               "इस्तेमाल करके एक संकेतक बना सकते हैं")
    assert compression_ratio(repeated) > 2.4
    assert compression_ratio(natural) < 2.4
    assert compression_ratio("") == 0.0


# ---------------------------------------------------------------------------
# The no-speech token id is checkpoint-specific and must never be guessed
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Mimics a real Whisper tokenizer: an unknown special token resolves to
    the unk/eos id rather than failing, which is how a guessed id silently
    reads the wrong distribution."""

    def __init__(self, table):
        self.table = dict(table)
        self.unk = 50257

    def convert_tokens_to_ids(self, name):
        return self.table.get(name, self.unk)

    def convert_ids_to_tokens(self, token_id):
        for name, value in self.table.items():
            if value == token_id:
                return name
        return "<|endoftext|>"


def _processor(table):
    return SimpleNamespace(tokenizer=_Tokenizer(table))


def _model(**generation):
    return SimpleNamespace(generation_config=SimpleNamespace(**generation))


class TestNoSpeechTokenResolution:
    def test_large_v3_family_uses_nospeech(self):
        from asr.explicit.runner import resolve_no_speech_token

        proc = _processor({"<|nospeech|>": 50363, "<|notimestamps|>": 50364})
        assert resolve_no_speech_token(proc, _model()) == 50363

    def test_small_and_medium_use_nocaptions_and_50363_is_notimestamps(self):
        """The bug: 50363 was hardcoded, but on these checkpoints it is
        <|notimestamps|> — the guard was reading an unrelated token."""
        from asr.explicit.runner import resolve_no_speech_token

        proc = _processor({"<|nocaptions|>": 50362, "<|notimestamps|>": 50363})
        resolved = resolve_no_speech_token(proc, _model())
        assert resolved == 50362
        assert resolved != 50363

    def test_a_tokenizer_without_the_token_disables_the_check(self):
        """convert_tokens_to_ids returns the unk id for an absent token; that
        must not be accepted, because it is a real token id."""
        from asr.explicit.runner import resolve_no_speech_token

        proc = _processor({"<|notimestamps|>": 50363})
        assert resolve_no_speech_token(proc, _model()) is None

    def test_generation_config_wins_when_it_has_the_id(self):
        from asr.explicit.runner import resolve_no_speech_token

        proc = _processor({"<|nospeech|>": 50363})
        assert resolve_no_speech_token(proc, _model(no_speech_token_id=99)) == 99

    def test_no_tokenizer_disables_the_check_rather_than_guessing(self):
        from asr.explicit.runner import resolve_no_speech_token

        assert resolve_no_speech_token(SimpleNamespace(), _model()) is None

    def test_decoder_without_a_resolved_id_reports_no_probability(self):
        dec = _decoder()
        dec.gen_config.no_speech_token_id = None
        from asr.explicit.decoder import WhisperDecoder

        fresh = WhisperDecoder(dec.model, torch.device("cpu"))
        assert fresh.no_speech_token_id is None


def test_no_speech_probability_is_read_at_the_sot_position():
    """Whisper's no-speech probability belongs to the distribution after
    <|sot|> alone. Reading the last prompt position (after <|notimestamps|>)
    samples a distribution where the token cannot appear."""
    from asr.explicit.decoder import WhisperDecoder

    NO_SPEECH = 50363

    class _PositionalModel:
        """Puts <|nospeech|> mass at position 0 only, and a decoy elsewhere."""

        def __init__(self):
            self.generation_config = _gen_config()
            self.config = SimpleNamespace(max_target_positions=448)

        def __call__(self, *, encoder_outputs, decoder_input_ids, **kw):
            n = decoder_input_ids.shape[1]
            logits = torch.full((1, n, VOCAB), -30.0)
            logits[0, 0, NO_SPEECH] = 20.0          # SOT position: no speech
            if n > 1:
                logits[0, -1, 1234] = 20.0          # last position: a word
            return SimpleNamespace(logits=logits, past_key_values=None)

    dec = WhisperDecoder(_PositionalModel(), torch.device("cpu"),
                         no_speech_token_id=NO_SPEECH)
    state, _ = dec.prefill(None, language="hi")
    assert state.no_speech_prob > 0.9, "must read position 0, not the last"
