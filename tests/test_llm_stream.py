"""LLMRunner.generate()/stream() on CPU with a scripted model.

The fake model emits a fixed token script regardless of input; the fake
tokenizer maps ids to string pieces, including pieces that are *fragments*
of a character, to exercise the delta logic on Devanagari.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from llm.runner import LLMRunner  # noqa: E402

EOS = 0
VOCAB = 16


class ScriptedModel:
    """Returns one-hot logits following ``script`` one token per call."""

    def __init__(self, script: list[int]):
        self.script = list(script)
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask, past_key_values=None, **kw):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        logits = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
        logits[0, -1, self.script[idx]] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class PieceTokenizer:
    """id -> piece. Piece 9 is a lone combining matra: decoded after a
    consonant it forms one grapheme; decoded alone it is a fragment."""

    pieces = {
        1: "नम", 2: "स्ते", 3: " ", 4: "दुनिया", 5: "।", 6: " x", 7: "य",
        8: "ा",   # matra
        9: "�",  # simulates a half-emitted multibyte piece
    }
    eos_token_id = EOS

    def __call__(self, prompt, return_tensors=None):
        ids = torch.tensor([[11, 12, 13]])
        return SimpleNamespace(
            to=lambda device: {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        )

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces.get(i, "") for i in ids if i != EOS)


def runner(script, **kw):
    return LLMRunner(ScriptedModel(script), PieceTokenizer(), torch.device("cpu"),
                     repetition_penalty=1.0, **kw)


def test_generate_matches_stream_text():
    script = [1, 2, 3, 4, 5, EOS]
    text = runner(script).generate("p", max_new_tokens=32).text
    pieces = list(runner(script).stream("p", max_new_tokens=32))
    assert text == "नमस्ते दुनिया।"
    assert "".join(pieces) == text
    assert pieces == ["नम", "स्ते", " ", "दुनिया", "।"]


def test_stream_metrics_are_recorded_and_first_token_is_after_prefill():
    r = runner([1, 2, EOS])
    pieces = list(r.stream("p", max_new_tokens=8))
    m = r.last_metrics
    assert pieces == ["नम", "स्ते"]
    assert m.generated_tokens == 2
    assert m.prompt_tokens == 3
    assert m.prefill_ms > 0 and len(m.decode_ms) == 2   # one decode step per further token
    assert not m.stopped_on_repetition and not m.stopped_by_caller


def test_should_stop_ends_decoding_and_is_flagged():
    r = runner([1, 2, 3, 4, 5, EOS])
    seen = []
    pieces = []
    for piece in r.stream("p", max_new_tokens=32, should_stop=lambda: len(seen) >= 2):
        pieces.append(piece)
        seen.append(piece)
    assert pieces == ["नम", "स्ते"]
    assert r.last_metrics.stopped_by_caller is True
    assert r.model.calls == 2                            # prefill + one decode step


def test_loop_guard_stops_repetition_in_generate_and_stream():
    script = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    g = runner(script, loop_guard_ngram=2).generate("p", max_new_tokens=32)
    assert g.metrics.stopped_on_repetition
    assert g.text == "x x"                               # 4 seen, last 2 retracted
    r = runner(script, loop_guard_ngram=2)
    out = "".join(r.stream("p", max_new_tokens=32))
    assert r.last_metrics.stopped_on_repetition
    assert out == " x x x"          # 3rd was yielded before the guard fired; 4th never was


def test_matra_is_joined_to_its_consonant_in_the_delta():
    # य then ा: the second piece must arrive as its own delta and combine.
    pieces = list(runner([7, 8, EOS]).stream("p"))
    assert pieces == ["य", "ा"]
    assert "".join(pieces) == "या"


def test_fragment_ending_in_replacement_char_is_held_back():
    # piece 9 decodes to U+FFFD; the delta is held until a later token
    # completes it (here the script continues with a real piece).
    pieces = list(runner([9, 4, EOS]).stream("p"))
    assert pieces == ["�दुनिया"]


def test_token_budget_respected():
    r = runner([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, EOS])
    out = list(r.stream("p", max_new_tokens=3))
    assert out == ["नम", "स्ते", " "]
    assert r.last_metrics.generated_tokens == 3
