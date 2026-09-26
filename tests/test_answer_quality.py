"""The answer-quality harness, scored against answers a real run produced.

`benchmarks/llm_bakeoff.py` measured `devanagari_ratio` and `think_leaks`, so a
model scoring 1.000 was one whose script was right — never one whose answers
were. Four answers from a measured Colab run all scored 1.000 on that metric
while being, respectively: correct, a thirteen-fold loop, the opposite of what
was asked, and an invented fact. Those four are the fixtures here.

The harness needs a model; its scoring does not, which is the part that decides
what a future run concludes.
"""

from __future__ import annotations

from benchmarks.answer_quality import (
    CASES,
    Case,
    count_sentences,
    declined,
    devanagari_ratio,
    looks_looping,
    normalise,
    score,
    summarise,
)

#: Verbatim from results/latency_ab/turns.jsonl and results/engine_parity.
REAL_CORRECT = "भारत की राजधानी नई दिल्ली है।"
REAL_LOOP = "एक छोटी कहानी आपके साथ सुनूंगी। " + "एक बर्फ के " * 13
REAL_CONTRADICTION = "चलिए तेज बोलकर बात करते हैं।"
REAL_INVENTED = "आज मौसम बराबर है।"


def case_for(prefix: str) -> Case:
    return next(c for c in CASES if c.prompt.startswith(prefix))


class TestTheCaseSet:
    def test_every_factual_case_states_what_it_expects(self):
        """A factual case with nothing expected can never be marked wrong, so
        it would inflate accuracy by being unfalsifiable."""
        for case in CASES:
            if case.kind == "factual":
                assert case.expected, case.prompt

    def test_every_case_has_a_known_kind(self):
        for case in CASES:
            assert case.kind in {"factual", "instruction", "unanswerable"}

    def test_it_covers_all_three_kinds(self):
        kinds = {case.kind for case in CASES}
        assert kinds == {"factual", "instruction", "unanswerable"}

    def test_unanswerable_cases_have_no_expected_answer(self):
        """The point of them is that there is no right answer to contain."""
        for case in CASES:
            if case.kind == "unanswerable":
                assert not case.expected, case.prompt


class TestRealAnswers:
    """The four the measured run produced. All scored 1.000 Devanagari."""

    def test_the_correct_fact_scores_correct(self):
        result = score(case_for("भारत की राजधानी"), REAL_CORRECT)
        assert result.correct is True
        assert result.looping is False

    def test_the_loop_is_caught(self):
        result = score(case_for("मुझे एक छोटी कहानी"), REAL_LOOP)
        assert result.looping is True
        assert result.obeyed is False

    def test_the_contradiction_is_caught_by_a_forbidden_word(self):
        """Asked to speak slower, it offered to speak faster. No judge model
        needed: तेज is simply not allowed in that answer."""
        result = score(case_for("थोड़ा धीरे बोलो"), REAL_CONTRADICTION)
        assert "तेज" in result.forbidden_hit
        assert result.obeyed is False

    def test_the_invented_weather_fails_the_unanswerable_case(self):
        result = score(case_for("आज मौसम कैसा है"), REAL_INVENTED)
        assert result.declined is False

    def test_a_refusal_passes_the_unanswerable_case(self):
        result = score(case_for("आज मौसम कैसा है"),
                       "मुझे मौसम की जानकारी नहीं है।")
        assert result.declined is True

    def test_all_four_score_a_perfect_devanagari_ratio(self):
        """The whole reason this harness exists: the old metric sees nothing
        wrong with any of them."""
        summary = summarise([
            score(case_for("भारत की राजधानी"), REAL_CORRECT),
            score(case_for("मुझे एक छोटी कहानी"), REAL_LOOP),
            score(case_for("थोड़ा धीरे बोलो"), REAL_CONTRADICTION),
            score(case_for("आज मौसम कैसा है"), REAL_INVENTED),
        ])
        assert summary["devanagari_ratio_mean"] == 1.0
        assert summary["instruction_obeyed"] == 0.0
        assert summary["declined_rate"] == 0.0
        assert summary["looping"] == 1


class TestScoringSeparatesTheKinds:
    def test_an_instruction_case_has_no_factual_verdict(self):
        """Recording it as incorrect would drag factual accuracy down with
        cases it was never asked about."""
        result = score(case_for("मुझे एक छोटी कहानी"), "एक अच्छी कहानी।")
        assert result.correct is None

    def test_a_factual_case_has_no_instruction_verdict(self):
        result = score(case_for("भारत की राजधानी"), REAL_CORRECT)
        assert result.obeyed is None
        assert result.declined is None

    def test_summary_counts_each_kind_separately(self):
        summary = summarise([score(c, "x") for c in CASES])
        assert summary["factual_n"] == sum(1 for c in CASES if c.kind == "factual")
        assert summary["instruction_n"] == sum(
            1 for c in CASES if c.kind == "instruction")
        assert summary["unanswerable_n"] == sum(
            1 for c in CASES if c.kind == "unanswerable")

    def test_an_empty_score_list_reports_none_not_zero(self):
        summary = summarise([])
        assert summary["factual_accuracy"] is None
        assert summary["instruction_obeyed"] is None
        assert summary["devanagari_ratio_mean"] is None


class TestForbiddenWords:
    def test_a_wrong_fact_in_the_forbidden_list_fails_even_if_the_right_one_appears(self):
        """The served answer `चीनी राजधानी बेंगलुरु है।` names a city; naming
        the right one alongside a wrong one is not a correct answer."""
        result = score(case_for("भारत की राजधानी"),
                       "भारत की राजधानी नई दिल्ली नहीं, बेंगलुरु है।")
        assert "बेंगलुरु" in result.forbidden_hit
        assert result.correct is False


class TestHelpers:
    def test_devanagari_ratio_of_mixed_script(self):
        assert devanagari_ratio("नमस्ते hello") is not None
        assert 0 < devanagari_ratio("नमस्ते hello") < 1

    def test_devanagari_ratio_of_pure_hindi_is_one(self):
        assert devanagari_ratio("नमस्ते") == 1.0

    def test_a_letterless_answer_is_none_not_a_perfect_score(self):
        """An empty or punctuation-only answer is a failure; 1.0 would read as
        perfect script."""
        assert devanagari_ratio("") is None
        assert devanagari_ratio("... !!") is None
        assert summarise([score(case_for("भारत की"), "")])[
            "empty_or_scriptless"] == 1

    def test_sentences_are_counted_on_terminators(self):
        assert count_sentences("नमस्ते। मैं ठीक हूँ।") == 2
        assert count_sentences("एक वाक्य।") == 1
        assert count_sentences("आप कैसे हैं?") == 1

    def test_a_decimal_does_not_end_a_sentence(self):
        assert count_sentences("यह 3.5 है।") == 1

    def test_normalise_strips_punctuation_and_case(self):
        assert normalise("New Delhi!") == "new delhi"
        assert normalise("नई दिल्ली।") == "नई दिल्ली"

    def test_loop_detection_matches_the_engine_guard(self):
        assert looks_looping(REAL_LOOP) is True
        assert looks_looping(REAL_CORRECT) is False

    def test_refusal_phrases_are_matched_loosely(self):
        assert declined("मुझे नहीं पता।") is True
        assert declined("यह जानकारी नहीं है।") is True
        assert declined("आज मौसम बराबर है।") is False


class TestBrevity:
    def test_an_over_long_answer_is_flagged(self):
        case = case_for("एक वाक्य में")
        assert case.max_sentences == 1
        result = score(case, "योग एक अभ्यास है। यह शरीर के लिए अच्छा है।")
        assert result.too_long is True
        assert result.obeyed is False

    def test_a_one_sentence_answer_passes(self):
        result = score(case_for("एक वाक्य में"), "योग एक प्राचीन भारतीय अभ्यास है।")
        assert result.too_long is False
        assert result.obeyed is True


class TestSystemPromptVariants:
    """The one lever left untested after sampling was measured and refuted.

    Instruction-following measured 80% while 0 of 3 unanswerable cases were
    declined: the model obeys the prompt, and the prompt never asked it to
    admit ignorance. It answered "3:45 बजे" to a question about the time with
    no clock.
    """

    def test_grounded_extends_the_default_rather_than_replacing_it(self):
        """The brevity and script instructions are load-bearing -- 80% obeyed
        and zero over-long answers -- so the variant must add to them."""
        from llm.prompting import system_prompt_for

        default = system_prompt_for("hi")
        grounded = system_prompt_for("hi", "grounded")
        assert grounded.startswith(default)
        assert len(grounded) > len(default)

    def test_it_tells_the_model_not_to_invent(self):
        from llm.prompting import system_prompt_for

        grounded = system_prompt_for("hi", "grounded")
        assert "नहीं पता" in grounded
        assert "मत गढ़िए" in grounded

    def test_the_instruction_generalises_past_the_cases_it_was_written_for(self):
        """Time, weather and private data appear as examples, not as the rule.
        A variant that only named them would be an overfit to three cases."""
        from llm.prompting import GROUNDED_SUFFIX

        english = GROUNDED_SUFFIX[None]
        assert "do not know" in english
        assert "Never invent a fact" in english

    def test_every_language_has_a_grounded_form(self):
        from llm.prompting import SYSTEM_PROMPTS, system_prompt_for

        for language in SYSTEM_PROMPTS:
            grounded = system_prompt_for(language, "grounded")
            assert grounded != system_prompt_for(language)

    def test_the_default_variant_is_byte_identical_to_before(self):
        """Adding a variant must not change what the pipeline already serves,
        or every recorded number becomes incomparable."""
        from llm.prompting import SYSTEM_PROMPTS, system_prompt_for

        for language, expected in SYSTEM_PROMPTS.items():
            assert system_prompt_for(language) == expected
            assert system_prompt_for(language, "default") == expected

    def test_an_unknown_variant_is_refused(self):
        import pytest

        from llm.prompting import system_prompt_for

        with pytest.raises(ValueError, match="variant"):
            system_prompt_for("hi", "grouded")

    def test_the_refusal_names_the_options(self):
        import pytest

        from llm.prompting import system_prompt_for

        with pytest.raises(ValueError, match="grounded"):
            system_prompt_for("hi", "nope")
