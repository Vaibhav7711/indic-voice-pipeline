"""Text normalization and lexicons for Hindi / Hinglish ASR."""

from text.lexicon import NUMBER_WORDS, is_number_word
from text.normalize import (
    IndicNormalizer,
    NormalizationConfig,
    NormalizationLevel,
    Script,
    get_normalizer,
    normalize,
    script_of,
    tokenize,
)

__all__ = [
    "IndicNormalizer",
    "NormalizationConfig",
    "NormalizationLevel",
    "Script",
    "get_normalizer",
    "normalize",
    "script_of",
    "tokenize",
    "NUMBER_WORDS",
    "is_number_word",
]
