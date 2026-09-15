"""Normalization behaviour, especially the Devanagari-specific cases.

These are CPU-only and must stay dependency-free: the scoring layer is
deliberately decoupled from torch/transformers so predictions can be re-scored
anywhere.
"""

from __future__ import annotations

import pytest

from text.normalize import (
    IndicNormalizer,
    NormalizationLevel,
    Script,
    normalize,
    script_of,
    tokenize,
)


class TestLevelNone:
    def test_identity(self):
        raw = "  हिन्दी,  Test।  "
        assert normalize(raw, NormalizationLevel.NONE) == raw


class TestBasic:
    def test_danda_removed(self):
        assert normalize("यह एक वाक्य है।", "basic") == "यह एक वाक्य है"

    def test_double_danda_removed(self):
        assert normalize("श्लोक॥", "basic") == "श्लोक"

    def test_ascii_punctuation_becomes_space(self):
        # Must not glue tokens together.
        assert tokenize("एक,दो;तीन", "basic") == ["एक", "दो", "तीन"]

    def test_apostrophe_removed_in_word(self):
        assert normalize("don't", "basic") == "dont"

    def test_latin_lowercased(self):
        assert normalize("Sarvam AI", "basic") == "sarvam ai"

    def test_devanagari_digits_to_ascii(self):
        assert normalize("२०२४", "basic") == "2024"

    def test_zero_width_stripped(self):
        assert normalize("क\u200dम", "basic") == "कम"

    def test_whitespace_collapsed(self):
        assert normalize("  a\t\n b  ", "basic") == "a b"

    def test_nukta_not_stripped_at_basic(self):
        from text.normalize import NUKTA

        assert NUKTA in normalize("ज़रूरी", "basic")


class TestStandardNukta:
    """The reason `standard` exists: one encoding for nukta consonants."""

    def test_precomposed_and_decomposed_agree(self):
        precomposed = "\u095c"            # ड़ as a single codepoint
        decomposed = "\u0921\u093c"       # ड + nukta
        assert precomposed != decomposed
        n = IndicNormalizer("standard")
        assert n.normalize(precomposed) == n.normalize(decomposed)

    @pytest.mark.parametrize(
        "precomposed,base",
        [("\u0929", "\u0928"), ("\u0931", "\u0930"), ("\u0934", "\u0933")],
    )
    def test_nfc_exempt_letters_are_canonicalised(self, precomposed, base):
        """NFC leaves ऩ/ऱ/ऴ composed; `standard` must split them anyway."""
        import unicodedata

        assert len(unicodedata.normalize("NFC", precomposed)) == 1  # NFC alone fails
        out = normalize(precomposed, "standard")
        assert out == base + "\u093c"

    def test_standard_preserves_linguistic_distinctions(self):
        """ज vs ज़ are different words — `standard` must keep them apart."""
        assert normalize("जरूरी", "standard") != normalize("ज़रूरी", "standard")


class TestAggressive:
    def test_nukta_folded(self):
        assert normalize("जरूरी", "aggressive") == normalize("ज़रूरी", "aggressive")

    def test_homorganic_nasal_unified(self):
        assert normalize("हिन्दी", "aggressive") == normalize("हिंदी", "aggressive")
        assert normalize("सम्भव", "aggressive") == normalize("संभव", "aggressive")

    def test_non_homorganic_nasal_preserved(self):
        """अन्य must NOT become अंय — a blanket nasal rule would break this."""
        assert normalize("अन्य", "aggressive") == "अन्य"

    def test_chandrabindu_folded(self):
        assert normalize("हँसी", "aggressive") == normalize("हंसी", "aggressive")


class TestLadderIsMonotonic:
    """Each level must be at least as collapsing as the one before it."""

    SAMPLES = ["हिन्दी।", "ज़रूरी", "२५ Rupees", "सम्भव", "don't stop"]

    def test_equality_is_preserved_down_the_ladder(self):
        levels = ["none", "basic", "standard", "aggressive"]
        for a in self.SAMPLES:
            for b in self.SAMPLES:
                for i in range(len(levels) - 1):
                    if normalize(a, levels[i]) == normalize(b, levels[i]):
                        assert normalize(a, levels[i + 1]) == normalize(b, levels[i + 1])


class TestScriptDetection:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("हिंदी", Script.DEVANAGARI),
            ("hello", Script.LATIN),
            ("2024", Script.DIGIT),
            ("hindiमें", Script.MIXED),
            ("!!", Script.OTHER),
            ("", Script.OTHER),
        ],
    )
    def test_script_of(self, token, expected):
        assert script_of(token) == expected


class TestTokenize:
    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_mixed_script_sentence(self):
        assert tokenize("मैंने AI use किया।") == ["मैंने", "ai", "use", "किया"]
