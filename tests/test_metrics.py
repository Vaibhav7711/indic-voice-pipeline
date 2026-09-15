"""Metric correctness: alignment invariants plus an independent cross-check."""

from __future__ import annotations

import random

import pytest

from benchmarks.metrics import (
    CorpusScore,
    OpKind,
    align,
    score_text,
)


def naive_levenshtein(a: list[str], b: list[str]) -> int:
    """Independent reference implementation — no backtrace, no shared code."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


class TestAlignmentBasics:
    def test_identical(self):
        a = align(["a", "b", "c"], ["a", "b", "c"])
        assert a.counts.errors == 0
        assert a.counts.hits == 3
        assert a.error_rate == 0.0
        assert all(op.kind is OpKind.EQUAL for op in a.ops)

    def test_single_substitution(self):
        a = align(["a", "b", "c"], ["a", "x", "c"])
        assert a.counts.substitutions == 1
        assert a.counts.deletions == 0
        assert a.counts.insertions == 0
        assert pytest.approx(a.error_rate) == 1 / 3

    def test_single_deletion(self):
        a = align(["a", "b", "c"], ["a", "c"])
        assert a.counts.deletions == 1
        assert a.counts.errors == 1

    def test_single_insertion(self):
        a = align(["a", "c"], ["a", "b", "c"])
        assert a.counts.insertions == 1
        assert a.counts.errors == 1

    def test_wer_can_exceed_one(self):
        """Insertions are not bounded by reference length."""
        a = align(["a"], ["x", "y", "z", "w"])
        assert a.error_rate > 1.0

    def test_empty_reference_rate_is_none(self):
        a = align([], ["x"])
        assert a.error_rate is None
        assert a.counts.insertions == 1

    def test_both_empty(self):
        a = align([], [])
        assert a.counts.errors == 0
        assert a.error_rate is None

    def test_empty_hypothesis_is_all_deletions(self):
        a = align(["a", "b"], [])
        assert a.counts.deletions == 2
        assert a.error_rate == 1.0


class TestAlignmentInvariants:
    """Properties that must hold for every alignment, on random input."""

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_independent_implementation(self, seed):
        rng = random.Random(seed)
        vocab = "abcde"
        ref = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        a = align(ref, hyp)
        assert a.counts.errors == naive_levenshtein(ref, hyp)

    @pytest.mark.parametrize("seed", range(40))
    def test_ops_reconstruct_both_sequences(self, seed):
        rng = random.Random(seed + 500)
        vocab = "abcde"
        ref = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        a = align(ref, hyp)

        rebuilt_ref = [op.ref_token for op in a.ops if op.ref_token is not None]
        rebuilt_hyp = [op.hyp_token for op in a.ops if op.hyp_token is not None]
        assert rebuilt_ref == ref
        assert rebuilt_hyp == hyp

    @pytest.mark.parametrize("seed", range(40))
    def test_counts_agree_with_ops(self, seed):
        rng = random.Random(seed + 900)
        ref = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        hyp = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        a = align(ref, hyp)
        kinds = [op.kind for op in a.ops]
        assert kinds.count(OpKind.EQUAL) == a.counts.hits
        assert kinds.count(OpKind.SUBSTITUTION) == a.counts.substitutions
        assert kinds.count(OpKind.DELETION) == a.counts.deletions
        assert kinds.count(OpKind.INSERTION) == a.counts.insertions
        assert a.counts.reference_length == len(ref)

    def test_indices_are_monotonic(self):
        a = align(list("kitten"), list("sitting"))
        ref_ids = [op.ref_index for op in a.ops if op.ref_index is not None]
        hyp_ids = [op.hyp_index for op in a.ops if op.hyp_index is not None]
        assert ref_ids == sorted(ref_ids)
        assert hyp_ids == sorted(hyp_ids)

    def test_classic_kitten_sitting(self):
        assert align(list("kitten"), list("sitting")).counts.errors == 3


class TestNormalizationCoupling:
    """Normalization level must actually change the measured error."""

    def test_danda_is_free_at_basic_but_costly_at_none(self):
        ref, hyp = "यह सही है।", "यह सही है"
        assert score_text(ref, hyp, level="none").counts.errors == 1
        assert score_text(ref, hyp, level="basic").counts.errors == 0

    def test_nukta_encoding_costs_nothing_at_standard(self):
        ref = "\u095c"          # single codepoint ड़
        hyp = "\u0921\u093c"    # ड + nukta
        assert score_text(ref, hyp, level="none").counts.errors == 1
        assert score_text(ref, hyp, level="standard").counts.errors == 0

    def test_conjunct_variant_only_free_at_aggressive(self):
        ref, hyp = "हिन्दी भाषा", "हिंदी भाषा"
        assert score_text(ref, hyp, level="standard").counts.errors == 1
        assert score_text(ref, hyp, level="aggressive").counts.errors == 0

    def test_char_unit_is_gentler_than_word_unit(self):
        """One wrong matra: 100% WER but small CER."""
        ref, hyp = "किताब", "कितिब"
        wer = score_text(ref, hyp, unit="word").error_rate
        cer = score_text(ref, hyp, unit="char").error_rate
        assert wer == 1.0
        assert cer < wer

    def test_char_unit_without_spaces(self):
        a = score_text("अ ब", "अब", unit="char", keep_spaces=False)
        assert a.counts.errors == 0


class TestCorpusScore:
    def test_micro_weights_by_length(self):
        score = CorpusScore()
        # 10-token utterance, 1 error -> 10%
        score.update(align(["w"] * 10, ["w"] * 9 + ["x"]))
        # 1-token utterance, 1 error -> 100%
        score.update(align(["a"], ["b"]))

        assert pytest.approx(score.micro) == 2 / 11
        assert pytest.approx(score.macro) == (0.1 + 1.0) / 2
        assert score.micro < score.macro  # the whole reason to report both

    def test_empty_references_tracked_not_averaged(self):
        score = CorpusScore()
        score.update(align(["a"], ["a"]))
        score.update(align([], ["x"]))
        assert score.utterances == 2
        assert score.empty_references == 1
        assert len(score.per_utterance) == 1

    def test_as_dict_shape(self):
        score = CorpusScore()
        score.update(align(["a", "b"], ["a", "c"]))
        d = score.as_dict()
        assert d["micro_percent"] == 50.0
        assert d["substitutions"] == 1
        assert d["reference_length"] == 2

    def test_empty_corpus(self):
        score = CorpusScore()
        assert score.micro is None
        assert score.macro is None
