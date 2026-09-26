"""Locating a divergence and judging it: the parts that decide the verdict.

The parity gate says two greedy decoders over the same weights produced
different text and cannot say whether that matters. Two readings fit: a near-tie
that fp16 rounding decided, or the engine computing something else. The
top-two logit margin at the step where they part is what separates them.

Running it needs the model. Locating the step and classifying the margin do
not, and those are where a wrong answer would be confidently wrong.
"""

from __future__ import annotations

from scripts.logit_margins import (
    NEAR_TIE_FRACTION,
    classify,
    divergence_step,
    summarise,
)


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
