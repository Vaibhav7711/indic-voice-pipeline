"""The parity gate's comparison logic, including what it must refuse to pass.

The gate itself needs two engines and a GPU. Its judgement does not, and the
judgement is the part that can quietly be wrong: a gate that passes an engine
producing different text is worse than no gate, because it launders the
difference into a recorded result.
"""

from __future__ import annotations

from scripts.engine_parity import common_prefix_length, compare, summarize


class TestCommonPrefix:
    def test_identical_strings_share_everything(self):
        assert common_prefix_length("नमस्ते", "नमस्ते") == len("नमस्ते")

    def test_it_reports_where_they_stop_agreeing(self):
        assert common_prefix_length("abcdef", "abcxyz") == 3

    def test_nothing_in_common_is_zero(self):
        assert common_prefix_length("abc", "xyz") == 0

    def test_a_truncated_response_shares_its_whole_length(self):
        assert common_prefix_length("abcdef", "abc") == 3

    def test_empty_against_anything_is_zero(self):
        assert common_prefix_length("", "abc") == 0


class TestComparison:
    def test_identical_text_passes_and_is_marked_identical(self):
        verdict = compare("भारत की राजधानी नई दिल्ली है।",
                          "भारत की राजधानी नई दिल्ली है।", prefix_chars=24)
        assert verdict["agreed"] and verdict["identical"]
        assert verdict["divergence"] is None

    def test_late_divergence_passes_the_prefix_gate(self):
        """fp16 attention is not associative: two implementations can rank the
        top two candidates differently at some step and then follow different
        but equally valid continuations. The user has already heard the first
        unit by then."""
        reference = "नई दिल्ली भारत की राजधानी है और यह एक बड़ा शहर है।"
        served = "नई दिल्ली भारत की राजधानी है और वहाँ बहुत लोग रहते हैं।"
        verdict = compare(reference, served, prefix_chars=24)
        assert verdict["agreed"] is True
        assert verdict["identical"] is False
        assert verdict["divergence"]["at"] >= 24

    def test_early_divergence_fails(self):
        """Differing in the first characters means prefill differs, not that
        decode drifted -- a real defect, and the one this gate exists for."""
        verdict = compare("भारत की राजधानी नई दिल्ली है।",
                          "मुझे यह नहीं पता।", prefix_chars=24)
        assert verdict["agreed"] is False
        assert verdict["divergence"]["at"] == 0

    def test_the_divergence_carries_both_sides(self):
        verdict = compare("abcdefghij", "abcXXXXXXX", prefix_chars=24)
        assert verdict["divergence"]["reference"].startswith("defg")
        assert verdict["divergence"]["served"].startswith("XXXX")

    def test_an_empty_served_response_cannot_pass(self):
        """A server returning nothing shares a zero-length prefix with a real
        answer. Requiring min(prefix, len) characters must not let that
        through on the grounds that zero characters matched out of zero."""
        verdict = compare("भारत की राजधानी नई दिल्ली है।", "", prefix_chars=24)
        assert verdict["agreed"] is False
        assert verdict["served_chars"] == 0

    def test_both_empty_is_reported_rather_than_silently_passing(self):
        verdict = compare("", "", prefix_chars=24)
        assert verdict["reference_chars"] == 0 and verdict["served_chars"] == 0

    def test_a_truncated_but_agreeing_response_passes_the_prefix_gate(self):
        """The served engine stopping early is a `max_tokens` difference, not
        a decode difference; it is reported through the char counts."""
        verdict = compare("नई दिल्ली भारत की राजधानी है और बड़ा शहर है।",
                          "नई दिल्ली भारत की राजधानी है", prefix_chars=24)
        assert verdict["agreed"] is True
        assert verdict["served_chars"] < verdict["reference_chars"]


class TestSummary:
    def _result(self, *, agreed, identical=False, shared=30, reference=40):
        # Mirrors what `compare()` returns, fields included: a stub that omits
        # them lets a change to the summary pass its own tests while failing
        # on real input.
        return {"comparison": {"agreed": agreed, "identical": identical,
                               "shared_prefix_chars": shared,
                               "reference_chars": reference}}

    def test_all_agreeing_passes(self):
        summary = summarize([self._result(agreed=True, identical=True)] * 5)
        assert summary["passed"] is True
        assert summary["agreed"] == 5 and summary["identical"] == 5
        assert summary["identity_rate"] == 1.0

    def test_one_disagreement_fails_the_whole_gate(self):
        results = [self._result(agreed=True)] * 9 + [self._result(agreed=False)]
        summary = summarize(results)
        assert summary["passed"] is False
        assert summary["agreement_rate"] == 0.9

    def test_agreement_and_identity_are_reported_separately(self):
        """Agreement gates; identity is information. Reporting only agreement
        would hide an engine that diverges on every prompt but late."""
        summary = summarize([self._result(agreed=True, identical=False)] * 4)
        assert summary["passed"] is True
        assert summary["identity_rate"] == 0.0

    def test_an_empty_run_does_not_pass(self):
        """Zero prompts compared is not a green gate: it is a gate that did
        not run, and `0/0 == 100%` is exactly how that gets reported as one."""
        summary = summarize([])
        assert summary["passed"] is False
        assert summary["agreed"] is None
        assert summary["identical"] is None


class TestAgreementStatistics:
    """Agreement as a number, not just a verdict.

    A greedy decoder diverges at the first step where two implementations rank
    the top two candidates differently, so agreement tracks per-step
    confidence. A small model has flatter logits and smaller top-two margins,
    so an fp16 rounding difference flips a tie readily. That makes the shared
    fraction the figure to compare across checkpoints, and a bare "they
    disagreed" close to useless.
    """

    def _result(self, shared, reference, *, agreed=True):
        return {"comparison": {"agreed": agreed, "identical": shared == reference,
                               "shared_prefix_chars": shared,
                               "reference_chars": reference}}

    def test_the_shared_fraction_is_reported(self):
        summary = summarize([self._result(20, 40), self._result(40, 40)])
        assert summary["mean_shared_prefix_chars"] == 30.0
        assert summary["mean_shared_fraction"] == 0.75

    def test_a_confident_model_agrees_for_longer(self):
        """The comparison this figure exists to make."""
        small = summarize([self._result(8, 40)] * 4)
        large = summarize([self._result(36, 40)] * 4)
        assert large["mean_shared_fraction"] > small["mean_shared_fraction"]

    def test_an_empty_reference_is_skipped_rather_than_dividing_by_zero(self):
        summary = summarize([self._result(0, 0, agreed=False), self._result(20, 40)])
        assert summary["mean_shared_fraction"] == 0.5
        assert summary["prompts"] == 2

    def test_all_references_empty_reports_none(self):
        summary = summarize([self._result(0, 0, agreed=False)])
        assert summary["mean_shared_fraction"] is None
        assert summary["passed"] is False

    def test_an_empty_run_reports_none_for_both(self):
        summary = summarize([])
        assert summary["mean_shared_prefix_chars"] is None
        assert summary["mean_shared_fraction"] is None
