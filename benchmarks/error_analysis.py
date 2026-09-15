"""Turn an alignment into failure modes.

A single WER number tells you the model is wrong; it does not tell you what to
fix. This module classifies every individual edit operation, so a run answers
questions like:

- How much of our WER is *spelling convention* rather than misrecognition?
- Are we losing numbers, or names, or Latin-script code-switches specifically?
- Is the decoder truncating (early EOS) or looping (repetition)?

The LLM-engineering analogue is moving from "eval loss went down" to a
per-slice breakdown: the aggregate is the thing you report, the slices are the
thing you act on.

Category precedence
-------------------
Each error is assigned exactly one category, first match wins, in this order:

1. ``truncation`` / ``deletion_run`` / ``hallucination`` — structural failures
   detected as *runs* of consecutive ops. These are decoder-behaviour bugs and
   dominate any token-level reading, so they are claimed first.
2. ``orthographic`` — a substitution that disappears under ``aggressive``
   normalization. The two tokens are the same word spelled differently.
3. ``numeric`` — a digit string or a spelled-out numeral on either side.
4. ``code_switch`` — the two sides disagree on script, or a Latin token is
   involved. Hinglish is where Whisper-style models are weakest.
5. ``rare_word`` — the reference token occurs at most ``rare_threshold`` times
   in the whole evaluation set. A cheap, lexicon-free proxy for names and
   domain vocabulary.
6. ``function_word`` — very short tokens (है, का, की, में). High frequency, low
   information; a bad ratio here usually means audio quality, not vocabulary.
7. ``other``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from benchmarks.metrics import Alignment, EditOp, OpKind, score_text
from text.lexicon import is_number_word
from text.normalize import NormalizationLevel, Script, get_normalizer, script_of

__all__ = [
    "ErrorCategory",
    "CategorizedError",
    "ExampleAnalysis",
    "AnalysisConfig",
    "analyze_example",
    "aggregate",
    "build_reference_vocabulary",
]


class ErrorCategory(str, Enum):
    ORTHOGRAPHIC = "orthographic"
    NUMERIC = "numeric"
    CODE_SWITCH = "code_switch"
    RARE_WORD = "rare_word"
    FUNCTION_WORD = "function_word"
    TRUNCATION = "truncation"
    DELETION_RUN = "deletion_run"
    HALLUCINATION = "hallucination"
    OTHER = "other"


@dataclass(frozen=True)
class AnalysisConfig:
    #: Consecutive deletions/insertions needed to call something structural.
    min_run: int = 4
    #: Reference-corpus count at or below which a token counts as rare.
    rare_threshold: int = 1
    #: Rare-word proxy only applies to tokens this long or longer, so that
    #: short rare tokens are not mistaken for named entities.
    rare_min_chars: int = 3
    #: Tokens this short are treated as function words.
    function_word_max_chars: int = 2
    #: Consecutive repeats of an n-gram that indicate a decode loop.
    repetition_threshold: int = 4
    #: Normalization level used for the primary metric.
    level: NormalizationLevel | str = NormalizationLevel.STANDARD


@dataclass(frozen=True)
class CategorizedError:
    kind: OpKind
    category: ErrorCategory
    ref_token: str | None
    hyp_token: str | None
    ref_index: int | None
    hyp_index: int | None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "category": self.category.value,
            "ref_token": self.ref_token,
            "hyp_token": self.hyp_token,
            "ref_index": self.ref_index,
            "hyp_index": self.hyp_index,
        }


@dataclass
class ExampleAnalysis:
    example_id: str
    reference: str
    hypothesis: str
    word: Alignment
    char: Alignment
    errors: list[CategorizedError] = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    @property
    def wer(self) -> float | None:
        return self.word.error_rate

    @property
    def cer(self) -> float | None:
        return self.char.error_rate

    def as_dict(self, include_errors: bool = True) -> dict:
        out = {
            "id": self.example_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": self.wer,
            "cer": self.cer,
            "word_counts": self.word.counts.as_dict(),
            "char_counts": self.char.counts.as_dict(),
            "flags": self.flags,
        }
        if include_errors:
            out["errors"] = [e.as_dict() for e in self.errors]
        return out


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def build_reference_vocabulary(
    references: list[str],
    level: NormalizationLevel | str = NormalizationLevel.STANDARD,
) -> Counter:
    """Token frequencies over the whole reference set.

    Built once per run and shared across examples — rarity is a *corpus*
    property, so computing it per example would be meaningless.
    """
    normalizer = get_normalizer(level)
    counter: Counter = Counter()
    for ref in references:
        counter.update(normalizer.tokenize(ref))
    return counter


# --------------------------------------------------------------------------
# Structural detection
# --------------------------------------------------------------------------


def _find_runs(ops: list[EditOp], kind: OpKind, min_run: int) -> list[list[int]]:
    """Indices of consecutive ops of a single kind, length >= min_run."""
    runs: list[list[int]] = []
    current: list[int] = []
    for idx, op in enumerate(ops):
        if op.kind is kind:
            current.append(idx)
        else:
            if len(current) >= min_run:
                runs.append(current)
            current = []
    if len(current) >= min_run:
        runs.append(current)
    return runs


def _has_repetition_loop(tokens: list[str], threshold: int) -> bool:
    """Detect an n-gram repeated back-to-back `threshold` times.

    This is the classic autoregressive failure: the decoder falls into a cycle
    and emits the same phrase until it hits max_new_tokens. Identical in shape
    to degenerate LLM sampling, and it inflates insertions dramatically.
    """
    n_tokens = len(tokens)
    for n in range(1, 5):
        if n * threshold > n_tokens:
            break
        for start in range(n_tokens - n * threshold + 1):
            gram = tokens[start : start + n]
            if all(
                tokens[start + k * n : start + (k + 1) * n] == gram
                for k in range(1, threshold)
            ):
                return True
    return False


# --------------------------------------------------------------------------
# Token-level categorization
# --------------------------------------------------------------------------


def _is_numeric(token: str | None) -> bool:
    if not token:
        return False
    return any(ch.isdigit() for ch in token) or is_number_word(token)


def _is_code_switch(ref: str | None, hyp: str | None) -> bool:
    ref_script = script_of(ref) if ref else None
    hyp_script = script_of(hyp) if hyp else None
    scripts = {s for s in (ref_script, hyp_script) if s is not None}
    if Script.LATIN in scripts or Script.MIXED in scripts:
        return True
    # Substitution across scripts, e.g. Devanagari reference vs Latin output.
    return (
        ref_script is not None
        and hyp_script is not None
        and ref_script != hyp_script
        and Script.OTHER not in scripts
    )


def _categorize_token_error(
    op: EditOp,
    vocabulary: Counter,
    config: AnalysisConfig,
    aggressive,
) -> ErrorCategory:
    ref, hyp = op.ref_token, op.hyp_token

    if op.kind is OpKind.SUBSTITUTION and ref and hyp:
        if aggressive.normalize(ref) == aggressive.normalize(hyp):
            return ErrorCategory.ORTHOGRAPHIC

    if _is_numeric(ref) or _is_numeric(hyp):
        return ErrorCategory.NUMERIC

    if _is_code_switch(ref, hyp):
        return ErrorCategory.CODE_SWITCH

    if (
        ref
        and len(ref) >= config.rare_min_chars
        and vocabulary.get(ref, 0) <= config.rare_threshold
    ):
        return ErrorCategory.RARE_WORD

    probe = ref or hyp or ""
    if len(probe) <= config.function_word_max_chars:
        return ErrorCategory.FUNCTION_WORD

    return ErrorCategory.OTHER


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def analyze_example(
    example_id: str,
    reference: str,
    hypothesis: str,
    vocabulary: Counter,
    config: AnalysisConfig | None = None,
) -> ExampleAnalysis:
    """Score one utterance and classify every error in it."""
    config = config or AnalysisConfig()
    aggressive = get_normalizer(NormalizationLevel.AGGRESSIVE)

    word = score_text(reference, hypothesis, level=config.level, unit="word")
    char = score_text(reference, hypothesis, level=config.level, unit="char")

    ops = word.ops
    n_ref = len(word.ref_tokens)

    # Structural runs claim their ops first.
    structural: dict[int, ErrorCategory] = {}
    for run in _find_runs(ops, OpKind.DELETION, config.min_run):
        last_ref_index = ops[run[-1]].ref_index
        ends_at_reference_end = last_ref_index == n_ref - 1
        category = (
            ErrorCategory.TRUNCATION
            if ends_at_reference_end
            else ErrorCategory.DELETION_RUN
        )
        for idx in run:
            structural[idx] = category
    for run in _find_runs(ops, OpKind.INSERTION, config.min_run):
        for idx in run:
            structural[idx] = ErrorCategory.HALLUCINATION

    errors: list[CategorizedError] = []
    for idx, op in enumerate(ops):
        if op.kind is OpKind.EQUAL:
            continue
        category = structural.get(idx) or _categorize_token_error(
            op, vocabulary, config, aggressive
        )
        errors.append(
            CategorizedError(
                kind=op.kind,
                category=category,
                ref_token=op.ref_token,
                hyp_token=op.hyp_token,
                ref_index=op.ref_index,
                hyp_index=op.hyp_index,
            )
        )

    hyp_tokens = word.hyp_tokens
    length_ratio = (len(hyp_tokens) / n_ref) if n_ref else None
    flags = {
        "empty_hypothesis": len(hyp_tokens) == 0,
        "empty_reference": n_ref == 0,
        "truncated": any(c is ErrorCategory.TRUNCATION for c in structural.values()),
        "hallucination_run": any(
            c is ErrorCategory.HALLUCINATION for c in structural.values()
        ),
        "repetition_loop": _has_repetition_loop(
            hyp_tokens, config.repetition_threshold
        ),
        "length_ratio": None if length_ratio is None else round(length_ratio, 3),
    }

    return ExampleAnalysis(
        example_id=example_id,
        reference=reference,
        hypothesis=hypothesis,
        word=word,
        char=char,
        errors=errors,
        flags=flags,
    )


def aggregate(
    analyses: list[ExampleAnalysis],
    top_confusions: int = 10,
    worst_examples: int = 10,
) -> dict:
    """Roll per-example analyses into a reportable breakdown."""
    category_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    confusions: dict[str, Counter] = {}
    flag_counts: Counter = Counter()

    total_errors = 0
    for analysis in analyses:
        for flag, value in analysis.flags.items():
            if value is True:
                flag_counts[flag] += 1
        for error in analysis.errors:
            total_errors += 1
            category_counts[error.category.value] += 1
            kind_counts[error.kind.value] += 1
            if error.kind is OpKind.SUBSTITUTION:
                bucket = confusions.setdefault(error.category.value, Counter())
                bucket[f"{error.ref_token} -> {error.hyp_token}"] += 1

    breakdown = {}
    for category, count in category_counts.most_common():
        breakdown[category] = {
            "errors": count,
            "share_of_errors_percent": (
                round(100 * count / total_errors, 2) if total_errors else 0.0
            ),
            "top_confusions": confusions.get(category, Counter()).most_common(
                top_confusions
            ),
        }

    scored = [a for a in analyses if a.wer is not None]
    scored.sort(key=lambda a: (-a.wer, a.example_id))

    return {
        "total_errors": total_errors,
        "by_category": breakdown,
        "by_operation": dict(kind_counts),
        "flag_counts": dict(flag_counts),
        "worst_examples": [
            {
                "id": a.example_id,
                "wer": round(a.wer, 4),
                "cer": None if a.cer is None else round(a.cer, 4),
                "reference": a.reference,
                "hypothesis": a.hypothesis,
            }
            for a in scored[:worst_examples]
        ],
    }
