"""The A/B driver's pure parts: arm parsing, absent-vs-zero, arm construction.

The measurement itself needs a GPU. What can be tested here is everything
that decides *what* gets measured, and the two ways an A/B lies: a treatment
that silently did not apply, and an absent metric averaged in as zero.
"""

from __future__ import annotations

import pytest

from scripts.latency_ab import (
    KNOBS,
    build_arm,
    metrics_of,
    parse_arm,
    percentile,
    summarize,
)


class TestArmParsing:
    def test_a_bare_name_is_the_control(self):
        assert parse_arm("baseline") == ("baseline", {})

    def test_knobs_are_parsed_as_integers(self):
        assert parse_arm("both:history_tokens=200,unit_chars=30") == (
            "both", {"history_tokens": 200, "unit_chars": 30})

    def test_the_engine_knob_stays_a_string(self):
        assert parse_arm("served:llm_engine=http") == ("served", {"llm_engine": "http"})

    def test_an_unknown_knob_is_refused(self):
        """A typo that silently does nothing produces an arm identical to the
        control, and 'no significant difference' looks like a finding."""
        with pytest.raises(ValueError, match="unknown knob"):
            parse_arm("typo:history_token=200")

    def test_the_refusal_lists_what_is_available(self):
        with pytest.raises(ValueError, match="history_tokens"):
            parse_arm("typo:hisory_tokens=200")

    def test_an_unnamed_arm_is_refused(self):
        with pytest.raises(ValueError, match="no name"):
            parse_arm(":history_tokens=200")

    def test_every_documented_knob_parses(self):
        for knob in KNOBS:
            value = "http" if knob == "llm_engine" else "1"
            name, overrides = parse_arm(f"a:{knob}={value}")
            assert knob in overrides


class TestPercentile:
    def test_an_empty_sample_is_none_not_zero(self):
        assert percentile([], 0.5) is None

    def test_a_single_value_is_its_own_percentile(self):
        assert percentile([42.0], 0.9) == 42.0

    def test_nearest_rank(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9) == 5.0


class TestSummaryHonesty:
    def test_a_field_absent_from_every_turn_stays_none(self):
        """An arm that never produced audio must not report 0 ms to first
        audio, which is what averaging `None` as zero would say."""
        summary = summarize([{"response_latency_ms": 100.0},
                             {"response_latency_ms": 200.0}])
        assert summary["first_token_to_first_unit_ms"]["p50"] is None
        assert summary["first_token_to_first_unit_ms"]["n"] == 0

    def test_partially_present_fields_report_their_own_n(self):
        """Six turns, four of which reached audio: the mean is over four, and
        the count says so rather than the reader assuming six."""
        records = [{"response_latency_ms": float(i), "tts_synthesis_ms": 10.0 * i}
                   for i in range(1, 5)]
        records += [{"response_latency_ms": 9.0}, {"response_latency_ms": 9.0}]
        summary = summarize(records)
        assert summary["turns"] == 6
        assert summary["response_latency_ms"]["n"] == 6
        assert summary["tts_synthesis_ms"]["n"] == 4

    def test_none_valued_fields_are_not_counted(self):
        summary = summarize([{"response_latency_ms": None},
                             {"response_latency_ms": 50.0}])
        assert summary["response_latency_ms"]["n"] == 1
        assert summary["response_latency_ms"]["mean"] == 50.0

    def test_generated_tokens_are_reported_when_present(self):
        summary = summarize([{"llm_generated_tokens": 20}, {"llm_generated_tokens": 40}])
        assert summary["mean_generated_tokens"] == 30.0

    def test_generated_tokens_absent_is_none(self):
        assert summarize([{"response_latency_ms": 1.0}])["mean_generated_tokens"] is None


class FakeGenerator:
    tokenizer = None

    def stream(self, prompt, **kwargs):
        yield "ठीक है।"


class TestArmConstruction:
    """Every arm must share one set of weights, and differ only in the knob."""

    def _generators(self):
        return {"explicit": (FakeGenerator(), None)}

    def _base(self):
        return {"history_tokens": 800, "history_turns": 6, "unit_chars": 60,
                "llm_engine": "explicit"}

    def test_the_control_takes_the_base_configuration(self):
        turn = build_arm({}, base=self._base(), generators=self._generators(),
                         synth=object(), sink_factory=lambda: None)
        assert turn.max_unit_chars == 60
        assert turn.conversation.max_history_tokens == 800

    def test_an_override_reaches_the_turn(self):
        turn = build_arm({"unit_chars": 30, "history_tokens": 200}, base=self._base(),
                         generators=self._generators(), synth=object(),
                         sink_factory=lambda: None)
        assert turn.max_unit_chars == 30
        assert turn.conversation.max_history_tokens == 200

    def test_arms_share_the_loaded_generator(self):
        """Loading a second copy of the weights would change device state and
        make the second arm's numbers not comparable to the first's."""
        generators = self._generators()
        first = build_arm({}, base=self._base(), generators=generators,
                          synth=object(), sink_factory=lambda: None)
        second = build_arm({"unit_chars": 30}, base=self._base(), generators=generators,
                           synth=object(), sink_factory=lambda: None)
        assert first.generator is second.generator

    def test_zero_history_turns_disables_memory(self):
        turn = build_arm({"history_turns": 0}, base=self._base(),
                         generators=self._generators(), synth=object(),
                         sink_factory=lambda: None)
        assert turn.conversation is None

    def test_an_engine_that_was_not_loaded_is_refused(self):
        """Building it lazily mid-run would put a model load inside a
        measured turn."""
        with pytest.raises(ValueError, match="only \\['explicit'\\] were loaded"):
            build_arm({"llm_engine": "http"}, base=self._base(),
                      generators=self._generators(), synth=object(),
                      sink_factory=lambda: None)


class TestRecordShape:
    """`TurnResult.as_dict()` nests the metrics. Reading them off the top
    level finds nothing, and an arm that measured fine reports every latency
    as `None` -- which looks like a pipeline that produced no audio rather
    than like a bug in the reader."""

    def _record(self):
        # The shape the driver writes, abbreviated: what TurnResult.as_dict()
        # actually returns.
        return {
            "turn_id": "abc123", "state": "done", "transcript": "नमस्ते",
            "response": "नमस्ते! मैं ठीक हूँ।",
            "metrics": {"response_latency_ms": 1234.5, "total_turn_ms": 1300.0,
                        "llm_generated_tokens": 21},
            "speech": None, "playback": None, "error": None,
            "arm": "baseline", "round": 0,
        }

    def test_metrics_are_found_inside_the_nested_block(self):
        assert metrics_of(self._record())["response_latency_ms"] == 1234.5

    def test_an_already_flat_metrics_dict_passes_through(self):
        assert metrics_of({"response_latency_ms": 9.0})["response_latency_ms"] == 9.0

    def test_summarize_reads_real_turn_records(self):
        summary = summarize([self._record(), self._record()])
        assert summary["turns"] == 2
        assert summary["response_latency_ms"]["p50"] == 1234.5
        assert summary["mean_generated_tokens"] == 21.0

    def test_a_record_with_a_non_dict_metrics_field_does_not_crash(self):
        record = self._record() | {"metrics": None}
        assert summarize([record])["response_latency_ms"]["n"] == 0
