"""Streamed multi-byte characters: the defect a real Colab run surfaced.

Observed on a T4 against `full-inference-engine`, Qwen3-0.6B: 1/1 English
prompt identical, 0/4 Hindi prompts agreeing, and every served Hindi response
studded with U+FFFD. The gate called it a decode divergence, which sent the
reader looking for a numerics bug in attention kernels. It was neither.

Qwen's tokenizer is byte-level BPE, so a 3-byte Devanagari character is
routinely split across two tokens. Decoding the tokens received so far yields
U+FFFD while a character is half-arrived. Reproduced with the real tokenizer:

    n= 3  U+FFFD=False  sent_is_prefix=True   'नमस'
    n= 4  U+FFFD=True   sent_is_prefix=True   'नमस्<U+FFFD>'
    n= 5  U+FFFD=False  sent_is_prefix=False  'नमस्त'

At n=4 the partial character is U+FFFD. At n=5 the real character replaces it
and the text already sent **stops being a prefix**. A server computing its
delta as `decoded[len(already_sent):]` -- a slice by length -- then emits the
replacement character permanently and drops the real one: at the next poll
the lengths are equal so nothing is sent, and at the one after that the slice
starts past the character that arrived.

This is a serving defect in the engine and cannot be repaired downstream --
the bytes never arrive. What this repo can do is refuse to mistake it for
something else, and count it, because corrupted text is handed to TTS and
pronounced: a voice agent mangling every fourth character would otherwise
pass every test here. `docs/ENGINE_BUG_UTF8_STREAMING.md` has the report and
the patch.
"""

from __future__ import annotations

import pytest

from llm.engines import HttpLLMEngine
from scripts.engine_parity import compare, corruption, summarize

# The `engine_server` fixture is re-exported by tests/conftest.py, so it
# arrives as a test argument rather than an import.
from tests.test_serving_end_to_end import FakeTTS, base_url

#: Written as a code point rather than a literal so a reader can tell this
#: apart from an encoding accident in the file itself.
FFFD = chr(0xFFFD)

#: Exactly what the engine returned for "नमस्ते, आज मौसम कैसा है?" on the T4.
OBSERVED_REFERENCE = "नमस्ते, मौसम कैसा है?"
OBSERVED_SERVED = f"नमस्{FFFD}े, म{FFFD}सम क{FFFD}सा ह{FFFD}?"


class TestCorruptionIsDetected:
    def test_the_observed_output_is_flagged(self):
        found = corruption(OBSERVED_REFERENCE, OBSERVED_SERVED)
        assert found is not None
        assert found["served_replacement_chars"] == 4
        assert found["reference_replacement_chars"] == 0

    def test_clean_output_is_not_flagged(self):
        assert corruption(OBSERVED_REFERENCE, OBSERVED_REFERENCE) is None

    def test_a_merely_divergent_response_is_not_called_corrupt(self):
        """The distinction the gate exists to draw: different text is a
        divergence, mangled text is a serving defect, and they need different
        fixes."""
        assert corruption("भारत की राजधानी नई दिल्ली है।",
                          "मुझे यह नहीं पता।") is None

    def test_the_diagnosis_says_a_larger_model_will_not_help(self):
        """Because the obvious response to "the small model disagrees" is to
        reach for a bigger one, and here that changes nothing: any byte-level
        BPE over any multi-byte script hits it."""
        found = corruption(OBSERVED_REFERENCE, OBSERVED_SERVED)
        assert "larger model will not fix it" in found["diagnosis"]

    def test_corruption_in_the_reference_too_is_still_reported(self):
        """If both sides are mangled the engine is not exonerated; the
        reference path would be corrupt as well, and both counts are given."""
        found = corruption(f"न{FFFD}स्ते", f"न{FFFD}स्ते")
        assert found["served_replacement_chars"] == 1
        assert found["reference_replacement_chars"] == 1


class TestTheGateFailsOnCorruption:
    def test_a_corrupt_run_cannot_pass_even_if_prefixes_agree(self):
        """The dangerous case: corruption starting after the gated prefix. The
        first 24 characters match, so the agreement check is satisfied, and
        without a separate corruption check the engine would be certified."""
        reference = "नई दिल्ली भारत की राजधानी है और वहाँ लोग रहते हैं।"
        served = reference[:30] + f"वहाँ ल{FFFD}ग रहते ह{FFFD}ं।"
        verdict = compare(reference, served, prefix_chars=24)
        assert verdict["agreed"] is True, "the prefix does match"
        assert verdict["corruption"] is not None
        assert summarize([{"comparison": verdict}])["passed"] is False

    def test_the_summary_counts_corrupt_prompts_and_characters(self):
        verdicts = [{"comparison": compare(OBSERVED_REFERENCE, OBSERVED_SERVED,
                                           prefix_chars=24)} for _ in range(3)]
        summary = summarize(verdicts)
        assert summary["corrupted_prompts"] == 3
        assert summary["served_replacement_chars"] == 12
        assert summary["passed"] is False

    def test_a_clean_run_still_passes(self):
        verdict = compare(OBSERVED_REFERENCE, OBSERVED_REFERENCE, prefix_chars=24)
        summary = summarize([{"comparison": verdict}])
        assert summary["corrupted_prompts"] == 0
        assert summary["served_replacement_chars"] == 0
        assert summary["passed"] is True

    def test_english_alone_would_have_passed_the_gate(self):
        """Why the default prompts must include Devanagari. One byte per
        character means this defect is invisible in ASCII, and the observed run
        shows exactly that: the English prompt was the one that agreed."""
        verdict = compare("The capital of India is New Delhi.",
                          "The capital of India is New Delhi.", prefix_chars=24)
        assert verdict["corruption"] is None
        assert summarize([{"comparison": verdict}])["passed"] is True


class TestTheAdapterCountsIt:
    """Counted at the adapter too, not only in the gate: the gate is run once,
    and corrupted text reaching TTS is a production defect."""

    def test_replacement_characters_are_counted_over_a_real_socket(self,
                                                                   engine_server):
        engine_server.script = [f"नमस्{FFFD}", f"े, म{FFFD}सम"]
        engine_server.usage = None
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        list(engine.stream("p"))
        assert engine.last_metrics.replacement_chars == 2
        assert engine.last_metrics.as_dict()["replacement_chars"] == 2

    def test_clean_devanagari_counts_zero(self, engine_server):
        engine_server.script = ["नमस्ते", "। मैं ठीक हूँ।"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        list(engine.stream("p"))
        assert engine.last_metrics.replacement_chars == 0

    def test_the_probe_reports_it_at_startup(self, engine_server):
        """The probe prompt is Devanagari on purpose, so a server with this
        defect is caught before the first turn rather than in a transcript."""
        engine_server.script = [f"{FFFD}पन"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        assert engine.probe()["replacement_chars"] == 1

    def test_a_turn_records_it_rather_than_failing_silently(self, engine_server):
        """The turn still runs -- refusing to speak would be worse than
        speaking imperfectly -- but the count is on the record."""
        from agent import VoiceTurn
        from agent.playback import BufferSink

        engine_server.script = [f"ठ{FFFD}क है।"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        synth = FakeTTS()
        result = VoiceTurn(engine, synth, sink_factory=BufferSink).run("बोलो")
        assert result.error is None
        assert synth.sentences, "it still spoke"
        assert engine.last_metrics.replacement_chars == 1


class TestStreamingMechanism:
    """The cause, with no tokenizer and no network: pure UTF-8 arithmetic.

    Kept separate from the tokenizer-backed reproduction so it always runs. It
    pins the invariant a correct streaming server must hold -- that what has
    already been sent stays a prefix of what is decoded next -- and shows a
    length-based delta breaking it.
    """

    @staticmethod
    def decode_prefix(data: bytes, upto: int) -> str:
        return data[:upto].decode("utf-8", "replace")

    @staticmethod
    def stream_by_length(data: bytes) -> str:
        """The engine's current bookkeeping, byte by byte."""
        sent_text, received = "", []
        for upto in range(1, len(data) + 1):
            decoded = data[:upto].decode("utf-8", "replace")
            if len(decoded) > len(sent_text):
                received.append(decoded[len(sent_text):])
                sent_text = decoded
        return "".join(received)

    @staticmethod
    def stream_with_holdback(data: bytes) -> str:
        """The patch proposed to the engine: never emit a trailing U+FFFD, and
        emit only what genuinely extends what was already sent."""
        sent_text, received = "", []
        for upto in range(1, len(data) + 1):
            decoded = data[:upto].decode("utf-8", "replace")
            stable = decoded[:-1] if decoded.endswith(FFFD) else decoded
            if stable.startswith(sent_text) and len(stable) > len(sent_text):
                received.append(stable[len(sent_text):])
                sent_text = stable
        return "".join(received)

    def test_a_split_devanagari_character_decodes_to_u_fffd(self):
        data = "नमस्ते".encode()
        assert len(data) % 3 == 0, "three bytes per character here"
        assert self.decode_prefix(data, len(data) - 1).endswith(FFFD)

    def test_the_already_sent_text_stops_being_a_prefix(self):
        """The invariant a length-based delta assumes, and this breaks."""
        data = "नमस्ते".encode()
        sent = self.decode_prefix(data, len(data) - 1)      # ends in U+FFFD
        assert not data.decode().startswith(sent)

    def test_a_length_based_delta_loses_the_character(self):
        corrupted = self.stream_by_length("नमस्ते".encode())
        assert FFFD in corrupted
        assert corrupted != "नमस्ते"

    def test_the_holdback_fix_reproduces_the_text_exactly(self):
        assert self.stream_with_holdback("नमस्ते".encode()) == "नमस्ते"

    def test_the_fix_holds_for_a_longer_mixed_script_string(self):
        text = "नमस्ते! The capital is नई दिल्ली, ठीक है? 🌟"
        assert self.stream_with_holdback(text.encode()) == text
        assert FFFD not in self.stream_with_holdback(text.encode())

    def test_the_fix_drops_a_character_truncated_mid_stream(self):
        """A stream cut mid-character loses that character, which is correct:
        the alternative is speaking a replacement character. The complete
        characters before it survive."""
        data = "नमस्ते".encode()[:-1]           # the final vowel sign lost a byte
        assert self.stream_with_holdback(data) == "नमस्त"
        assert FFFD not in self.stream_with_holdback(data)

    def test_ascii_is_unaffected_which_is_why_this_hid(self):
        assert self.stream_by_length(b"The capital of India") == "The capital of India"


def test_the_real_tokenizer_splits_devanagari_across_tokens():
    """The same thing against the actual checkpoint's tokenizer.

    Skipped when the tokenizer is not cached, because it downloads.
    `TestStreamingMechanism` covers the mechanism offline; this confirms Qwen3
    really does split these characters, rather than that UTF-8 can be split in
    principle.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as error:  # noqa: BLE001 - not cached, nothing to test
        pytest.skip(f"tokenizer not cached locally: {error}")

    text = "नमस्ते, मौसम कैसा है?"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    partials, broken_prefixes, sent = 0, 0, ""
    for count in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:count], skip_special_tokens=True)
        partials += FFFD in decoded
        broken_prefixes += not decoded.startswith(sent)
        if len(decoded) > len(sent):
            sent = decoded
    assert partials > 0, "no character was split; the premise would not hold"
    assert broken_prefixes > 0, "the prefix invariant was never broken"
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
