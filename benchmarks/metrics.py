"""WER / CER with explicit edit alignments.

Why not just call ``evaluate.load("wer")``
------------------------------------------
Two reasons, both practical:

1. A single aggregate WER tells you nothing actionable. The interesting object
   is the *alignment* — which reference token was substituted by which
   hypothesis token — because that is what error categorization consumes. The
   HF metric throws the alignment away.
2. Scoring must not depend on torch/transformers/datasets. Keeping this module
   pure-Python means saved predictions can be re-scored on any machine, in
   milliseconds, without a GPU. Decoupling expensive inference from cheap
   scoring is the same discipline as caching logits instead of re-running a
   model every time you want a different metric.

Corpus aggregation
------------------
Corpus WER is ``sum(S + D + I) / sum(N_ref)`` over the whole set — *not* the
mean of per-utterance WERs. The two differ whenever utterance lengths vary,
and the macro average is dominated by short utterances where a single error is
100% WER. Both are reported here, with the micro average as the headline, so
the difference is visible rather than accidental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from text.normalize import NormalizationLevel, get_normalizer

__all__ = [
    "OpKind",
    "EditOp",
    "ErrorCounts",
    "Alignment",
    "align",
    "score_tokens",
    "score_text",
    "character_tokens",
    "CorpusScore",
]


class OpKind(str, Enum):
    EQUAL = "equal"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"      # reference token missing from hypothesis
    INSERTION = "insertion"    # hypothesis token absent from reference


@dataclass(frozen=True)
class EditOp:
    kind: OpKind
    ref_index: int | None
    hyp_index: int | None
    ref_token: str | None
    hyp_token: str | None


@dataclass(frozen=True)
class ErrorCounts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_length": self.reference_length,
        }


@dataclass
class Alignment:
    ops: list[EditOp]
    counts: ErrorCounts
    ref_tokens: list[str]
    hyp_tokens: list[str]

    @property
    def error_rate(self) -> float | None:
        """Per-utterance rate. ``None`` when the reference is empty.

        An empty reference makes the rate undefined (division by zero), not
        zero and not one. Returning ``None`` forces the caller to decide,
        instead of silently poisoning an average with a sentinel value.
        """
        n = self.counts.reference_length
        if n == 0:
            return None
        return self.counts.errors / n


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

# Backpointer codes.
_DIAG, _UP, _LEFT = 0, 1, 2

# Guard against accidentally aligning two enormous strings character-wise.
# 16M cells ≈ 16 MB as a bytearray, which is a reasonable ceiling for a
# sentence-level metric. Anything larger is a bug in the caller.
MAX_DP_CELLS = 16_000_000


def align(ref_tokens: list[str], hyp_tokens: list[str]) -> Alignment:
    """Levenshtein alignment with backtrace.

    Unit costs for substitution, deletion and insertion — the standard WER
    definition. Ties are broken toward the diagonal so that a same-length
    mismatch reads as a substitution rather than a deletion/insertion pair.
    """
    n, m = len(ref_tokens), len(hyp_tokens)

    if n == 0 and m == 0:
        return Alignment([], ErrorCounts(), ref_tokens, hyp_tokens)
    if n == 0:
        ops = [
            EditOp(OpKind.INSERTION, None, j, None, hyp_tokens[j]) for j in range(m)
        ]
        return Alignment(ops, ErrorCounts(insertions=m), ref_tokens, hyp_tokens)
    if m == 0:
        ops = [
            EditOp(OpKind.DELETION, i, None, ref_tokens[i], None) for i in range(n)
        ]
        return Alignment(ops, ErrorCounts(deletions=n), ref_tokens, hyp_tokens)

    if (n + 1) * (m + 1) > MAX_DP_CELLS:
        raise ValueError(
            f"Alignment matrix too large: {n + 1} x {m + 1} cells. "
            "Split the input or score at token level instead of character level."
        )

    width = m + 1
    back = bytearray(width * (n + 1))

    previous = list(range(width))
    for j in range(width):
        back[j] = _LEFT
    back[0] = _DIAG

    current = [0] * width
    for i in range(1, n + 1):
        current[0] = i
        back[i * width] = _UP
        ref_tok = ref_tokens[i - 1]
        for j in range(1, width):
            sub_cost = previous[j - 1] + (0 if ref_tok == hyp_tokens[j - 1] else 1)
            del_cost = previous[j] + 1
            ins_cost = current[j - 1] + 1

            best = sub_cost
            code = _DIAG
            if del_cost < best:
                best, code = del_cost, _UP
            if ins_cost < best:
                best, code = ins_cost, _LEFT

            current[j] = best
            back[i * width + j] = code
        previous, current = current, previous

    # Backtrace from (n, m) to (0, 0).
    ops: list[EditOp] = []
    hits = subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        code = back[i * width + j]
        if code == _DIAG and i > 0 and j > 0:
            r, h = ref_tokens[i - 1], hyp_tokens[j - 1]
            if r == h:
                ops.append(EditOp(OpKind.EQUAL, i - 1, j - 1, r, h))
                hits += 1
            else:
                ops.append(EditOp(OpKind.SUBSTITUTION, i - 1, j - 1, r, h))
                subs += 1
            i -= 1
            j -= 1
        elif code == _UP and i > 0:
            ops.append(EditOp(OpKind.DELETION, i - 1, None, ref_tokens[i - 1], None))
            dels += 1
            i -= 1
        else:
            ops.append(EditOp(OpKind.INSERTION, None, j - 1, None, hyp_tokens[j - 1]))
            ins += 1
            j -= 1

    ops.reverse()
    counts = ErrorCounts(hits=hits, substitutions=subs, deletions=dels, insertions=ins)
    return Alignment(ops, counts, ref_tokens, hyp_tokens)


# --------------------------------------------------------------------------
# Scoring entry points
# --------------------------------------------------------------------------


def score_tokens(ref_tokens: list[str], hyp_tokens: list[str]) -> Alignment:
    return align(ref_tokens, hyp_tokens)


def character_tokens(text: str, keep_spaces: bool = True) -> list[str]:
    """Character sequence for CER.

    Spaces are kept by default (the jiwer/ESPnet convention) so that word-
    boundary errors are still penalised. CER matters for Devanagari because a
    single wrong matra costs one character instead of one whole word — it
    separates "the model misheard the word" from "the model got the word
    nearly right".
    """
    if keep_spaces:
        return list(text)
    return list(text.replace(" ", ""))


def score_text(
    reference: str,
    hypothesis: str,
    *,
    level: NormalizationLevel | str = NormalizationLevel.STANDARD,
    unit: str = "word",
    keep_spaces: bool = True,
) -> Alignment:
    """Normalize both sides identically, then align."""
    normalizer = get_normalizer(level)
    if unit == "word":
        return align(normalizer.tokenize(reference), normalizer.tokenize(hypothesis))
    if unit == "char":
        return align(
            character_tokens(normalizer.normalize(reference), keep_spaces),
            character_tokens(normalizer.normalize(hypothesis), keep_spaces),
        )
    raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")


# --------------------------------------------------------------------------
# Corpus aggregation
# --------------------------------------------------------------------------


@dataclass
class CorpusScore:
    """Accumulates alignments into micro and macro rates."""

    counts: ErrorCounts = field(default_factory=ErrorCounts)
    per_utterance: list[float] = field(default_factory=list)
    empty_references: int = 0
    utterances: int = 0

    def update(self, alignment: Alignment) -> None:
        self.counts = self.counts + alignment.counts
        self.utterances += 1
        rate = alignment.error_rate
        if rate is None:
            self.empty_references += 1
        else:
            self.per_utterance.append(rate)

    @property
    def micro(self) -> float | None:
        """sum(errors) / sum(reference tokens) — the headline number."""
        n = self.counts.reference_length
        if n == 0:
            return None
        return self.counts.errors / n

    @property
    def macro(self) -> float | None:
        """Mean of per-utterance rates. Short utterances dominate this."""
        if not self.per_utterance:
            return None
        return sum(self.per_utterance) / len(self.per_utterance)

    def as_dict(self) -> dict:
        micro, macro = self.micro, self.macro
        return {
            "micro_percent": None if micro is None else round(micro * 100, 4),
            "macro_percent": None if macro is None else round(macro * 100, 4),
            "utterances": self.utterances,
            "empty_references": self.empty_references,
            **self.counts.as_dict(),
        }
