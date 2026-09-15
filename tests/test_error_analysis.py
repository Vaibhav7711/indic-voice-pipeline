"""Error categorization behaviour."""

from __future__ import annotations

from collections import Counter

import pytest

from benchmarks.error_analysis import (
    AnalysisConfig,
    ErrorCategory,
    aggregate,
    analyze_example,
    build_reference_vocabulary,
)


def analyze(ref, hyp, vocab=None, **kwargs):
    vocab = vocab if vocab is not None else Counter()
    return analyze_example("x", ref, hyp, vocab, AnalysisConfig(**kwargs))


def categories(analysis):
    return [e.category for e in analysis.errors]


class TestVocabulary:
    def test_counts_normalized_tokens(self):
        vocab = build_reference_vocabulary(["यह एक बात है।", "एक और बात"])
        assert vocab["एक"] == 2
        assert vocab["बात"] == 2
        assert vocab["और"] == 1
        assert "है।" not in vocab  # danda stripped by normalization


class TestOrthographic:
    def test_conjunct_variant_is_orthographic(self):
        a = analyze("हिन्दी भाषा है", "हिंदी भाषा है")
        assert categories(a) == [ErrorCategory.ORTHOGRAPHIC]

    def test_nukta_variant_is_orthographic(self):
        a = analyze("ज़रूरी काम था", "जरूरी काम था")
        assert categories(a) == [ErrorCategory.ORTHOGRAPHIC]

    def test_real_word_error_is_not_orthographic(self):
        a = analyze("किताब मेज़ पर है", "कुर्सी मेज़ पर है")
        assert ErrorCategory.ORTHOGRAPHIC not in categories(a)


class TestNumeric:
    def test_digit_substitution(self):
        a = analyze("मैंने 25 रुपये दिए", "मैंने 35 रुपये दिए")
        assert categories(a) == [ErrorCategory.NUMERIC]

    def test_number_word_vs_digit(self):
        a = analyze("मैंने पाँच रुपये दिए", "मैंने 5 रुपये दिए")
        assert categories(a) == [ErrorCategory.NUMERIC]

    def test_english_number_word(self):
        a = analyze("give me five rupees", "give me 5 rupees")
        assert categories(a) == [ErrorCategory.NUMERIC]


class TestCodeSwitch:
    def test_latin_token_error(self):
        a = analyze("मैंने laptop खरीदा", "मैंने लैपटॉप खरीदा")
        assert categories(a) == [ErrorCategory.CODE_SWITCH]

    def test_latin_to_latin_error(self):
        a = analyze("this is machine learning", "this is machine learming")
        assert categories(a) == [ErrorCategory.CODE_SWITCH]


class TestRareWord:
    def test_hapax_reference_token_is_rare(self):
        refs = ["वैभव ने कहा", "उसने कहा", "फिर कहा"]
        vocab = build_reference_vocabulary(refs)
        assert vocab["वैभव"] == 1
        a = analyze("वैभव ने कहा", "विभव ने कहा", vocab=vocab)
        assert categories(a) == [ErrorCategory.RARE_WORD]

    def test_frequent_token_is_not_rare(self):
        vocab = Counter({"किताब": 9, "कुर्सी": 9})
        a = analyze("किताब यहाँ", "कुर्सी यहाँ", vocab=vocab)
        assert ErrorCategory.RARE_WORD not in categories(a)

    def test_short_token_not_treated_as_name(self):
        vocab = Counter({"कम": 1})
        a = analyze("कम है", "गम है", vocab=vocab)
        assert ErrorCategory.RARE_WORD not in categories(a)


class TestFunctionWord:
    def test_short_token_error(self):
        vocab = Counter({"है": 5, "था": 5})
        a = analyze("वह यहाँ है", "वह यहाँ था", vocab=vocab)
        assert categories(a) == [ErrorCategory.FUNCTION_WORD]


class TestStructuralRuns:
    def test_trailing_deletions_are_truncation(self):
        ref = "एक दो तीन चार पाँच छह सात आठ"
        hyp = "एक दो तीन चार"
        a = analyze(ref, hyp)
        assert all(c is ErrorCategory.TRUNCATION for c in categories(a))
        assert a.flags["truncated"] is True

    def test_mid_deletions_are_deletion_run_not_truncation(self):
        ref = "अ ब स द इ फ ग ह ट"
        hyp = "अ ब ट"
        a = analyze(ref, hyp)
        assert ErrorCategory.DELETION_RUN in categories(a)
        assert ErrorCategory.TRUNCATION not in categories(a)
        assert a.flags["truncated"] is False

    def test_insertion_run_is_hallucination(self):
        a = analyze("नमस्ते", "नमस्ते और और और और और")
        assert ErrorCategory.HALLUCINATION in categories(a)
        assert a.flags["hallucination_run"] is True

    def test_short_deletion_is_not_structural(self):
        a = analyze("अ ब स द इ", "अ ब द इ")
        assert ErrorCategory.DELETION_RUN not in categories(a)
        assert ErrorCategory.TRUNCATION not in categories(a)


class TestFlags:
    def test_empty_hypothesis(self):
        a = analyze("कुछ शब्द यहाँ", "")
        assert a.flags["empty_hypothesis"] is True
        assert a.wer == 1.0

    def test_repetition_loop_detected(self):
        a = analyze("नमस्ते", "हाँ जी हाँ जी हाँ जी हाँ जी")
        assert a.flags["repetition_loop"] is True

    def test_no_false_repetition(self):
        a = analyze("यह एक सामान्य वाक्य है", "यह एक सामान्य वाक्य है")
        assert a.flags["repetition_loop"] is False
        assert a.errors == []

    def test_length_ratio(self):
        a = analyze("अ ब", "अ ब स द")
        assert a.flags["length_ratio"] == 2.0


class TestPrecedence:
    def test_structural_beats_token_category(self):
        """A long numeric deletion run is reported as truncation, not numeric."""
        a = analyze("मुझे 1 2 3 4 5 6 चाहिए", "मुझे")
        assert ErrorCategory.NUMERIC not in categories(a)
        assert ErrorCategory.TRUNCATION in categories(a)

    def test_orthographic_beats_rare_word(self):
        vocab = Counter({"हिन्दी": 1})
        a = analyze("हिन्दी भाषा", "हिंदी भाषा", vocab=vocab)
        assert categories(a) == [ErrorCategory.ORTHOGRAPHIC]


class TestAggregate:
    def test_shares_sum_to_one_hundred(self):
        vocab = Counter()
        analyses = [
            analyze_example("a", "मैंने 25 रुपये दिए", "मैंने 35 रुपये दिए", vocab),
            analyze_example("b", "हिन्दी भाषा है", "हिंदी भाषा है", vocab),
        ]
        report = aggregate(analyses)
        assert report["total_errors"] == 2
        total_share = sum(
            v["share_of_errors_percent"] for v in report["by_category"].values()
        )
        assert total_share == pytest.approx(100.0)

    def test_worst_examples_sorted_descending(self):
        vocab = Counter()
        analyses = [
            analyze_example("good", "अ ब स", "अ ब स", vocab),
            analyze_example("bad", "अ ब स", "x y z", vocab),
        ]
        report = aggregate(analyses)
        assert report["worst_examples"][0]["id"] == "bad"

    def test_confusion_pairs_recorded(self):
        vocab = Counter()
        analyses = [
            analyze_example("a", "मैंने 25 रुपये दिए", "मैंने 35 रुपये दिए", vocab)
        ]
        report = aggregate(analyses)
        pairs = report["by_category"]["numeric"]["top_confusions"]
        assert pairs[0][0] == "25 -> 35"

    def test_empty_input(self):
        report = aggregate([])
        assert report["total_errors"] == 0
        assert report["worst_examples"] == []
