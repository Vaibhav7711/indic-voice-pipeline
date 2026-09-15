"""Hindi / Hinglish text normalization for ASR scoring and serving.

Why this module exists
----------------------
WER is an edit distance between two token sequences. If the reference writes
``हिन्दी`` and the model writes ``हिंदी``, that is one substitution — even though
the two strings are the same word, pronounced identically, differing only in
orthographic convention. The same applies to ``।`` vs ``.``, to Devanagari vs
ASCII digits, and to the two Unicode encodings of ``ज़``.

For an LLM engineer the analogy is exact: this is the detokenizer/normalizer
contract you fix *before* you trust an eval number. Skipping it means you are
partly measuring formatting, not recognition — and the error is not small. On
Devanagari it routinely moves WER by double-digit relative amounts.

Normalization ladder
--------------------
Each level is a strict superset of the one above it, and each answers a
different question:

``none``
    The literal strings. Nothing is touched. Report this so a reader can see
    exactly how much normalization is buying you.
``basic``
    Formatting-blind: case, whitespace, punctuation (including danda),
    zero-width joiners, Devanagari digits → ASCII digits.
``standard``
    Encoding-blind: ``basic`` plus a single canonical encoding for nukta
    consonants. This is the **primary reporting level**. It removes Unicode
    representation differences without erasing any linguistic distinction.
``aggressive``
    Orthography-blind: ``standard`` plus nukta removal, anusvara/conjunct-nasal
    unification, and chandrabindu folding. This level **is not a headline
    metric**. It exists so that error analysis can ask "would this error
    disappear if we ignored spelling convention entirely?" — the gap between
    ``standard`` and ``aggressive`` is the share of your WER that is
    transcription style rather than acoustic modelling.

Deliberate non-goal: number words
---------------------------------
This module does **not** rewrite ``पाँच`` to ``5`` or vice versa. Choosing a
direction silently decides which spelling is "correct" and moves WER without
the model changing. Numeric mismatches are instead *detected and reported* as
their own error category (see ``benchmarks.error_analysis``), so the cost is
visible rather than normalized away.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum

__all__ = [
    "NormalizationLevel",
    "NormalizationConfig",
    "IndicNormalizer",
    "normalize",
    "tokenize",
    "script_of",
    "Script",
]


# --------------------------------------------------------------------------
# Unicode constants
# --------------------------------------------------------------------------

ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"

NUKTA = "\u093c"
ANUSVARA = "\u0902"
CHANDRABINDU = "\u0901"
VIRAMA = "\u094d"
AVAGRAHA = "\u093d"

DEVANAGARI_DIGITS = "\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f"
ASCII_DIGITS = "0123456789"

# U+0958..U+095F are canonical-decomposition characters that Unicode places on
# the composition-exclusion list, so NFC *decomposes* them to base + nukta.
# U+0929, U+0931 and U+0934 are NOT excluded, so NFC leaves them precomposed.
# Verified empirically — this asymmetry is the whole reason the map below
# exists. Without it, nukta encoding is inconsistent within a single string.
PRECOMPOSED_NUKTA = {
    "\u0929": "\u0928" + NUKTA,  # ऩ  NNNA  -> न + nukta
    "\u0931": "\u0930" + NUKTA,  # ऱ  RRA   -> र + nukta
    "\u0934": "\u0933" + NUKTA,  # ऴ  LLLA  -> ळ + nukta
}

# Consonant range used for conjunct detection. After NFC every nukta consonant
# is represented as a base consonant from this range plus U+093C.
DEVANAGARI_CONSONANTS = "\u0915-\u0939\u0958-\u095f\u0979-\u097f"

# Homorganic nasal → anusvara. Restricted to the correct varga in each case:
# a blanket "nasal + virama → anusvara" rule would wrongly rewrite अन्य to अंय,
# because न् before a semivowel is not an anusvara context.
_NASAL_VARGA = [
    ("\u0919", "\u0915-\u0918"),  # ङ् before क ख ग घ
    ("\u091e", "\u091a-\u091d"),  # ञ् before च छ ज झ
    ("\u0923", "\u091f-\u0922"),  # ण् before ट ठ ड ढ
    ("\u0928", "\u0924-\u0927"),  # न् before त थ द ध
    ("\u092e", "\u092a-\u092d"),  # म् before प फ ब भ
]

# Punctuation removed outright rather than replaced by a space, because it sits
# *inside* tokens. Everything else becomes a space so that "एक,दो" tokenizes as
# two words instead of gluing into one.
INTRA_WORD_PUNCT = "'\u2019\u2018\u00b4\u02bc"


def _build_punctuation_class() -> str:
    """Every Unicode punctuation codepoint, minus the intra-word set.

    Built from Unicode categories (``P*``) rather than a hand-listed string so
    that danda (``।``), double danda (``॥``), the Devanagari abbreviation sign
    (``॰``), Urdu comma/question mark, and typographic quotes are all covered
    without having to remember them. Combining marks are category ``M*`` and
    letters ``L*``, so nukta and avagraha are never swept up here.
    """
    chars = []
    for cp in range(0x20, 0x3000):
        ch = chr(cp)
        if ch in INTRA_WORD_PUNCT:
            continue
        if unicodedata.category(ch).startswith("P"):
            chars.append(ch)
    return "".join(chars)


_PUNCT_CHARS = _build_punctuation_class()


class Script(str, Enum):
    """Coarse script label for a token. Used by error categorization."""

    DEVANAGARI = "devanagari"
    LATIN = "latin"
    DIGIT = "digit"
    MIXED = "mixed"
    OTHER = "other"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class NormalizationLevel(str, Enum):
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class NormalizationConfig:
    """Explicit switches so a run config records exactly what was applied."""

    unicode_nfc: bool = False
    strip_zero_width: bool = False
    devanagari_digits_to_ascii: bool = False
    strip_punctuation: bool = False
    lowercase_latin: bool = False
    collapse_whitespace: bool = False
    canonical_nukta: bool = False
    strip_nukta: bool = False
    unify_homorganic_nasals: bool = False
    chandrabindu_to_anusvara: bool = False
    strip_avagraha: bool = False

    @classmethod
    def for_level(cls, level: NormalizationLevel | str) -> NormalizationConfig:
        level = NormalizationLevel(level)
        if level is NormalizationLevel.NONE:
            return cls()

        basic = cls(
            unicode_nfc=True,
            strip_zero_width=True,
            devanagari_digits_to_ascii=True,
            strip_punctuation=True,
            lowercase_latin=True,
            collapse_whitespace=True,
        )
        if level is NormalizationLevel.BASIC:
            return basic

        standard = replace(basic, canonical_nukta=True)
        if level is NormalizationLevel.STANDARD:
            return standard

        return replace(
            standard,
            strip_nukta=True,
            unify_homorganic_nasals=True,
            chandrabindu_to_anusvara=True,
            strip_avagraha=True,
        )

    def describe(self) -> dict[str, bool]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------
# Normalizer
# --------------------------------------------------------------------------


class IndicNormalizer:
    """Compiled, reusable normalizer. Construct once, call many times."""

    def __init__(self, level: NormalizationLevel | str = NormalizationLevel.STANDARD):
        self.level = NormalizationLevel(level)
        self.config = NormalizationConfig.for_level(self.level)

        self._zero_width_re = re.compile(f"[{ZERO_WIDTH}]")
        self._digit_table = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)
        self._intra_word_re = re.compile(f"[{re.escape(INTRA_WORD_PUNCT)}]")
        self._punct_re = re.compile(f"[{re.escape(_PUNCT_CHARS)}]")
        self._whitespace_re = re.compile(r"\s+")
        self._nasal_res = [
            (re.compile(f"{nasal}{VIRAMA}(?=[{varga}])"), ANUSVARA)
            for nasal, varga in _NASAL_VARGA
        ]

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        cfg = self.config

        # 1. Canonical Unicode form first, so every later rule sees one
        #    encoding. This also decomposes U+0958..U+095F to base + nukta.
        if cfg.unicode_nfc:
            text = unicodedata.normalize("NFC", text)

        # 2. Patch the three precomposed nukta letters NFC leaves alone.
        if cfg.canonical_nukta:
            for src, dst in PRECOMPOSED_NUKTA.items():
                if src in text:
                    text = text.replace(src, dst)

        if cfg.strip_zero_width:
            text = self._zero_width_re.sub("", text)

        if cfg.devanagari_digits_to_ascii:
            text = text.translate(self._digit_table)

        if cfg.strip_punctuation:
            text = self._intra_word_re.sub("", text)
            text = self._punct_re.sub(" ", text)

        # 3. Orthography folding (aggressive only). Nasal unification must run
        #    while viramas are still present.
        if cfg.unify_homorganic_nasals:
            for pattern, repl in self._nasal_res:
                text = pattern.sub(repl, text)

        if cfg.chandrabindu_to_anusvara:
            text = text.replace(CHANDRABINDU, ANUSVARA)

        if cfg.strip_nukta:
            text = text.replace(NUKTA, "")

        if cfg.strip_avagraha:
            text = text.replace(AVAGRAHA, "")

        if cfg.lowercase_latin:
            text = text.lower()

        if cfg.collapse_whitespace:
            text = self._whitespace_re.sub(" ", text).strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """Whitespace tokenization of normalized text — the unit WER counts."""
        normalized = self.normalize(text)
        return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------

_CACHE: dict[NormalizationLevel, IndicNormalizer] = {}


def get_normalizer(
    level: NormalizationLevel | str = NormalizationLevel.STANDARD,
) -> IndicNormalizer:
    level = NormalizationLevel(level)
    if level not in _CACHE:
        _CACHE[level] = IndicNormalizer(level)
    return _CACHE[level]


def normalize(
    text: str, level: NormalizationLevel | str = NormalizationLevel.STANDARD
) -> str:
    return get_normalizer(level).normalize(text)


def tokenize(
    text: str, level: NormalizationLevel | str = NormalizationLevel.STANDARD
) -> list[str]:
    return get_normalizer(level).tokenize(text)


_DEVA_RE = re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9\u0966-\u096f]")


def script_of(token: str) -> Script:
    """Coarse script of a token — the signal behind code-switch detection."""
    if not token:
        return Script.OTHER
    has_deva = bool(_DEVA_RE.search(token))
    has_latin = bool(_LATIN_RE.search(token))
    has_digit = bool(_DIGIT_RE.search(token))

    if has_deva and has_latin:
        return Script.MIXED
    if has_deva:
        return Script.DEVANAGARI
    if has_latin:
        return Script.LATIN
    if has_digit:
        return Script.DIGIT
    return Script.OTHER
