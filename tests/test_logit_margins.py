"""Locating a divergence and judging it: the parts that decide the verdict.

The parity gate says two greedy decoders over the same weights produced
different text and cannot say whether that matters. Two readings fit: a near-tie
that fp16 rounding decided, or the engine computing something else. The
top-two logit margin at the step where they part is what separates them.

Running it needs the model. Locating the step and classifying the margin do
not, and those are where a wrong answer would be confidently wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.logit_margins import (
    BYTE_DECODER,
    NEAR_TIE_FRACTION,
    classify,
    divergence_step,
    locate_divergence,
    served_token_at,
    summarise,
    token_bytes,
)

REPO = Path(__file__).resolve().parent.parent


def _cut(text: str, offsets: list[int]) -> list[bytes]:
    """Cumulative byte steps of `text`, cut at the given byte offsets."""
    data = text.encode("utf-8")
    return [data[:offset] for offset in offsets]


class TestDivergenceStep:
    def test_it_finds_the_first_step_that_leaves_the_prefix(self):
        steps = ["नम", "नमस्", "नमस्ते", "नमस्ते मैं"]
        assert divergence_step(steps, "नमस्ते आप") == 3

    def test_agreement_throughout_is_none(self):
        """Not step 0, and not the last step: there is no divergence to
        explain, and returning a number would invent one."""
        steps = ["नम", "नमस्", "नमस्ते"]
        assert divergence_step(steps, "नमस्ते और आगे") is None

    def test_divergence_on_the_very_first_token_is_step_zero(self):
        assert divergence_step(["क", "कुछ"], "मुझे नहीं") == 0

    def test_it_works_on_tokens_not_character_offsets(self):
        """A character offset does not identify a token. Byte-level BPE splits
        a 3-byte Devanagari character across two tokens, so the character where
        two strings differ can sit inside a token rather than at its boundary.
        Walking the incremental decode sidesteps that entirely.
        """
        # Step 1's text is a prefix; step 2's is not, though the strings first
        # differ in the middle of the character step 2 completed.
        steps = ["नमस्", "नमस्त", "नमस्ते"]
        assert divergence_step(steps, "नमस्क") == 1

    def test_a_served_text_shorter_than_the_reference_diverges_when_passed(self):
        steps = ["ab", "abc", "abcd"]
        assert divergence_step(steps, "abc") == 2

    def test_an_empty_step_list_is_none(self):
        assert divergence_step([], "anything") is None

    def test_an_empty_served_text_diverges_immediately(self):
        """A server that returned nothing is not an agreement."""
        assert divergence_step(["a"], "") == 0


class TestDivergenceOnBytes:
    """What the tokenizer actually produces. Qwen3 cuts a Devanagari character
    between tokens within the first four tokens of every Hindi answer in the
    parity set, and a step cut that way decodes to U+FFFD."""

    def test_a_step_ending_inside_a_character_is_not_a_divergence(self):
        served = "भारत की राजधानी"
        # "भा" is 6 bytes; the second step ends one byte into "र".
        steps = _cut(served, [6, 7, 9, len(served.encode())])
        assert steps[1].decode("utf-8", "replace").endswith("�")
        assert divergence_step(steps, served) is None

    def test_the_same_steps_as_decoded_text_invent_a_divergence(self):
        """Why the steps are bytes: as decoded text, an answer identical to the
        served one diverges at the first character a token boundary cuts."""
        served = "भारत की राजधानी"
        steps = [s.decode("utf-8", "replace") for s in _cut(served, [6, 7, 9])]
        assert divergence_step(steps, served) == 1

    def test_characters_that_differ_only_in_their_last_byte(self):
        """त and क share their first two bytes. A step holding only those is
        still on the served text; the step completing the character parts."""
        reference = "नमस्त".encode()
        steps = [reference[:12], reference[:14], reference]
        assert divergence_step(steps, "नमस्क") == 2

    def test_whitespace_the_engines_strip_is_not_a_divergence(self):
        """Both engines strip their output; the raw decode here is not."""
        assert divergence_step([b"\n", b"\nab", b"\nab\n"], "ab") is None

    def test_text_past_the_served_end_still_diverges(self):
        assert divergence_step([b"ab", b"ab c"], "ab") == 1

    def test_a_served_text_cut_inside_a_character(self):
        """Cut by the token cap mid-character, the served decode ends in U+FFFD.
        Reference bytes stopping inside a character there agree with it."""
        full = "रहित".encode()
        served = full[:-2].decode("utf-8", "replace")      # ends in U+FFFD
        assert served.endswith("�")
        assert divergence_step([full[:6], full[:-2]], served) is None

    def test_after_a_cut_a_different_complete_text_still_diverges(self):
        served = "रहित".encode()[:-2].decode("utf-8", "replace")
        assert divergence_step([b"x"], served) == 0


class TestLocateDivergence:
    def test_stopping_where_the_served_text_went_on_is_a_divergence(self):
        """End-of-sequence was the reference's choice at step 2; the margin
        that decided it is the one at that step."""
        assert locate_divergence([b"a", b"ab"], "abc", stopped_on_eos=True) == 2

    def test_stopping_with_the_served_text_is_agreement(self):
        assert locate_divergence([b"a", b"ab"], "ab", stopped_on_eos=True) is None

    def test_running_out_of_tokens_is_not_a_divergence(self):
        """Cut by max_new_tokens, not by a choice the model made."""
        assert locate_divergence([b"a", b"ab"], "abc", stopped_on_eos=False) is None

    def test_an_earlier_divergence_wins(self):
        assert locate_divergence([b"a", b"x"], "abc", stopped_on_eos=True) == 1

    def test_stopping_before_any_token_is_step_zero(self):
        assert locate_divergence([], "abc", stopped_on_eos=True) == 0


class _Tokenizer:
    """A byte-level vocabulary: pieces by id, and a fixed canonical
    tokenization standing in for encoding the served text."""

    def __init__(self, pieces: dict[int, bytes], canonical: tuple[int, ...] = (),
                 plain: dict[int, str] | None = None):
        to_char = {byte: char for char, byte in BYTE_DECODER.items()}
        self.pieces = dict(pieces)
        self.vocab = {i: "".join(to_char[b] for b in p) for i, p in pieces.items()}
        for i, text in (plain or {}).items():      # added tokens, stored as text
            self.vocab[i] = text
            self.pieces[i] = text.encode("utf-8")
        self.canonical = list(canonical)

    def convert_ids_to_tokens(self, token_id):
        return self.vocab[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", "replace")

    def __call__(self, text, add_special_tokens=True):
        return types.SimpleNamespace(input_ids=self.canonical)


class TestTokenBytes:
    def test_the_table_maps_every_byte_once(self):
        assert sorted(BYTE_DECODER.values()) == list(range(256))

    def test_a_token_holding_part_of_a_character(self):
        tokenizer = _Tokenizer({0: "भ".encode()[:2]})
        assert token_bytes(tokenizer, 0) == b"\xe0\xa4"

    def test_special_tokens_contribute_nothing(self):
        """As under decode(skip_special_tokens=True)."""
        tokenizer = _Tokenizer({7: b"<|im_end|>"})
        assert token_bytes(tokenizer, 7, special_ids={7}) == b""

    def test_an_added_token_outside_the_table_falls_back_to_decode(self):
        tokenizer = _Tokenizer({}, plain={9: "a b"})
        assert token_bytes(tokenizer, 9) == b"a b"


class TestServedTokenAt:
    def _tokenizer(self):
        return _Tokenizer({1: "नम".encode(), 2: "स्".encode(), 3: "ते".encode()},
                          canonical=(1, 2, 3))

    def test_the_token_starting_at_a_boundary(self):
        assert served_token_at(self._tokenizer(), "नमस्ते", 0) == 1
        assert served_token_at(self._tokenizer(), "नमस्ते", 6) == 2

    def test_no_guess_inside_a_token(self):
        assert served_token_at(self._tokenizer(), "नमस्ते", 7) is None

    def test_no_guess_past_the_end(self):
        assert served_token_at(self._tokenizer(), "नमस्ते", 18) is None


def test_the_documented_command_gets_past_its_imports(tmp_path):
    """`python scripts/logit_margins.py` puts scripts/ on the path, not the
    repo root, and `scripts` is not an installed package: the first version
    failed at `from scripts.engine_parity import ...` before loading a model.
    The model path here does not exist, so this stops right after the imports.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"reference_dtype_requested": "float16", "results": []}))
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "logit_margins.py"),
         "--parity", str(parity), "--llm-model", str(tmp_path / "no-model"),
         "--out", str(tmp_path / "out.json")],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert "No module named" not in run.stderr
    assert "loading the reference" in run.stdout


class TestClassify:
    def test_a_margin_far_below_the_median_is_a_near_tie(self):
        assert classify(0.01, median=4.0) == "near_tie"

    def test_a_margin_at_the_median_is_clear(self):
        assert classify(4.0, median=4.0) == "clear"

    def test_the_boundary_is_the_stated_fraction(self):
        median = 4.0
        assert classify(NEAR_TIE_FRACTION * median - 1e-9, median) == "near_tie"
        assert classify(NEAR_TIE_FRACTION * median, median) == "clear"

    def test_a_missing_margin_is_unknown_not_a_pass(self):
        """No divergence found, or it fell outside the recorded steps. Calling
        that a near-tie would exonerate the engine on absent evidence."""
        assert classify(None, median=4.0) == "unknown"

    def test_a_missing_median_is_unknown(self):
        assert classify(0.01, median=None) == "unknown"

    def test_a_zero_median_is_unknown_not_a_division(self):
        assert classify(0.01, median=0.0) == "unknown"

    def test_the_comparison_is_relative_not_absolute(self):
        """Logit scale varies by model and by position, so the same absolute
        margin is a near-tie in one run and clear in another."""
        assert classify(0.5, median=10.0) == "near_tie"
        assert classify(0.5, median=1.0) == "clear"


class TestSummary:
    def _finding(self, verdict, at=3):
        return {"verdict": verdict, "diverged_at_step": at}

    def test_all_near_ties_is_consistent_with_rounding(self):
        summary = summarise([self._finding("near_tie")] * 3)
        assert summary["near_tie"] == 3
        assert summary["clear"] == 0
        assert summary["consistent_with_fp16_noise"] is True
        assert summary["near_tie_rate"] == 1.0

    def test_one_clear_divergence_breaks_that(self):
        """The conclusion has to be unanimous. A single divergence the
        reference was confident about is a defect, whatever the others were."""
        findings = [self._finding("near_tie")] * 4 + [self._finding("clear")]
        summary = summarise(findings)
        assert summary["consistent_with_fp16_noise"] is False
        assert summary["clear"] == 1

    def test_unknowns_are_excluded_from_the_rate_and_block_the_conclusion(self):
        summary = summarise([self._finding("unknown")] * 2)
        assert summary["judged"] == 0
        assert summary["near_tie_rate"] is None
        assert summary["consistent_with_fp16_noise"] is False

    def test_agreement_throughout_counts_as_undiverged(self):
        findings = [self._finding("unknown", at=None),
                    self._finding("near_tie", at=2)]
        summary = summarise(findings)
        assert summary["prompts"] == 2
        assert summary["diverged"] == 1

    def test_an_empty_run_does_not_conclude(self):
        summary = summarise([])
        assert summary["consistent_with_fp16_noise"] is False
        assert summary["near_tie_rate"] is None
