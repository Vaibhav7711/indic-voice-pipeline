"""Small lexicons used to *classify* tokens during error analysis.

Scope note: these tables are deliberately used for **detection only**, never
for rewriting text. A missing entry therefore degrades gracefully — a numeric
error is filed under ``other`` instead of ``numeric``. It can never corrupt a
transcript or silently change a WER number.

That is why partial coverage is acceptable here: 0–20, the tens, the Indian
numbering scales, and common ordinals cover the overwhelming majority of
spoken numerals in FLEURS/IndicVoices-style read and conversational speech.
Extend freely — ``NUMBER_WORDS_HI`` is just a set.
"""

from __future__ import annotations

__all__ = ["NUMBER_WORDS_HI", "NUMBER_WORDS_EN", "NUMBER_WORDS", "is_number_word"]


# Spelling variants are listed explicitly rather than generated, because Hindi
# transcripts vary on nasalisation (पाँच / पांच) and conjuncts (पंद्रह / पन्द्रह).
NUMBER_WORDS_HI: set[str] = {
    # 0-10
    "शून्य", "सिफर", "सिफ़र",
    "एक", "दो", "तीन", "चार",
    "पाँच", "पांच", "पंच",
    "छह", "छै", "छ",
    "सात", "आठ", "नौ", "दस",
    # 11-20
    "ग्यारह", "बारह", "तेरह", "चौदह",
    "पंद्रह", "पन्द्रह",
    "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    # tens
    "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे",
    # scales
    "सौ", "हज़ार", "हजार", "लाख", "करोड़", "करोड", "अरब", "खरब",
    # ordinals / fractions in common use
    "पहला", "पहली", "पहले", "प्रथम",
    "दूसरा", "दूसरी", "दूसरे", "द्वितीय",
    "तीसरा", "तीसरी", "तीसरे", "तृतीय",
    "चौथा", "चौथी", "चौथे",
    "आधा", "आधी", "डेढ़", "ढाई", "पौने", "सवा", "साढ़े",
}

NUMBER_WORDS_EN: set[str] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "lakh", "lakhs", "crore", "crores",
    "million", "billion", "trillion",
    "first", "second", "third", "fourth", "fifth", "half", "quarter",
}

NUMBER_WORDS: set[str] = NUMBER_WORDS_HI | NUMBER_WORDS_EN


def is_number_word(token: str) -> bool:
    """True if the token is a spelled-out numeral in Hindi or English."""
    if not token:
        return False
    return token.lower() in NUMBER_WORDS
