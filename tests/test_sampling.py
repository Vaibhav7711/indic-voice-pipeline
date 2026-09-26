"""Sampling in the explicit runner: filters, order, and greedy staying default.

Greedy is load-bearing here. Every correctness gate in this project — explicit
against `generate()`, static cache against eager, CTranslate2 against explicit,
the served engine against explicit — compares two greedy decoders, and would be
meaningless against a sampler. So greedy has to remain the default and
`temperature=0` has to mean exactly `argmax`.

Greedy is also, on the model authors' own guidance, the wrong choice for
answers: it repeats. A measured run produced the same 11-character phrase
thirteen times until it hit the token cap. Hence sampling exists and is off.

No model is loaded. Fixed logits and a seeded generator are enough to pin the
filtering arithmetic, which is the part that goes subtly wrong.
"""

from __future__ import annotations

import pytest
import torch

from llm.runner import LLMRunner


def selector(**settings) -> LLMRunner:
    """An `LLMRunner` with only the fields `_select` touches.

    Built without `__init__` on purpose: constructing one needs a model and a
    tokenizer, and none of that is involved in choosing a token.
    """
    runner = LLMRunner.__new__(LLMRunner)
    runner.repetition_penalty = settings.get("repetition_penalty", 1.0)
    runner.temperature = settings.get("temperature", 0.0)
    runner.top_p = settings.get("top_p", 1.0)
    runner.top_k = settings.get("top_k", 0)
    runner.min_p = settings.get("min_p", 0.0)
    seed = settings.get("seed")
    runner._generator = (torch.Generator().manual_seed(seed)
                         if seed is not None else None)
    return runner


LOGITS = torch.tensor([[1.0, 5.0, 3.0, 0.5, 2.0]])


class TestGreedyIsUntouched:
    def test_temperature_zero_is_argmax(self):
        assert int(selector()._select(LOGITS, [])) == 1

    def test_greedy_is_deterministic_over_many_calls(self):
        runner = selector()
        assert {int(runner._select(LOGITS, [])) for _ in range(50)} == {1}

    def test_greedy_does_not_touch_the_generator(self):
        """A greedy decode must not consume RNG, or two greedy runs in one
        process would stop being reproducible in different ways."""
        runner = selector(seed=0)
        before = runner._generator.get_state()
        runner._select(LOGITS, [])
        assert torch.equal(runner._generator.get_state(), before)

    def test_the_runner_defaults_to_greedy(self):
        """Read off the signature, so a change to the default fails here rather
        than quietly invalidating every parity gate."""
        import inspect

        defaults = inspect.signature(LLMRunner.__init__).parameters
        assert defaults["temperature"].default == 0.0
        assert defaults["top_p"].default == 1.0
        assert defaults["top_k"].default == 0
        assert defaults["seed"].default is None


class TestFilters:
    def test_top_k_one_is_argmax(self):
        runner = selector(temperature=1.0, top_k=1, seed=0)
        assert {int(runner._select(LOGITS, [])) for _ in range(20)} == {1}

    def test_top_k_restricts_to_the_k_best(self):
        """k=2 leaves indices 1 (5.0) and 2 (3.0) reachable and nothing else."""
        runner = selector(temperature=1.0, top_k=2, seed=0)
        drawn = {int(runner._select(LOGITS, [])) for _ in range(200)}
        assert drawn <= {1, 2}
        assert len(drawn) == 2, "both survivors should appear over 200 draws"

    def test_top_k_larger_than_the_vocabulary_is_harmless(self):
        runner = selector(temperature=1.0, top_k=999, seed=0)
        assert int(runner._select(LOGITS, [])) in range(LOGITS.shape[-1])

    def test_a_tiny_top_p_still_leaves_one_token(self):
        """The nucleus must be inclusive. A naive mask empties it whenever one
        token already holds more than top_p of the mass, and multinomial then
        raises on an all-zero distribution."""
        runner = selector(temperature=1.0, top_p=0.01, seed=0)
        assert {int(runner._select(LOGITS, [])) for _ in range(20)} == {1}

    def test_top_p_one_keeps_everything_reachable(self):
        runner = selector(temperature=1.0, top_p=1.0, seed=0)
        drawn = {int(runner._select(LOGITS, [])) for _ in range(500)}
        assert len(drawn) == LOGITS.shape[-1]

    def test_min_p_drops_the_unlikely_tail(self):
        runner = selector(temperature=1.0, min_p=0.5, seed=0)
        drawn = {int(runner._select(LOGITS, [])) for _ in range(200)}
        assert drawn == {1}, "only the top token is within half of the maximum"

    def test_low_temperature_concentrates_on_the_top_token(self):
        runner = selector(temperature=0.1, seed=0)
        drawn = [int(runner._select(LOGITS, [])) for _ in range(200)]
        assert drawn.count(1) > 190

    def test_high_temperature_spreads(self):
        hot = selector(temperature=2.0, seed=0)
        drawn = {int(hot._select(LOGITS, [])) for _ in range(300)}
        assert len(drawn) >= 4


class TestReproducibility:
    def test_the_same_seed_gives_the_same_draws(self):
        first = [int(selector(temperature=1.0, seed=7)._select(LOGITS, []))
                 for _ in range(1)]
        second = [int(selector(temperature=1.0, seed=7)._select(LOGITS, []))
                  for _ in range(1)]
        assert first == second

    def test_a_sequence_of_draws_is_reproducible(self):
        def run():
            runner = selector(temperature=1.5, seed=42)
            return [int(runner._select(LOGITS, [])) for _ in range(20)]

        assert run() == run()

    def test_different_seeds_differ(self):
        def run(seed):
            runner = selector(temperature=1.5, seed=seed)
            return [int(runner._select(LOGITS, [])) for _ in range(30)]

        assert run(1) != run(2)


class TestRepetitionPenaltyOrder:
    def test_the_penalty_applies_before_filtering(self):
        """Penalising after top-k would let a repeated token win selection and
        then be penalised into a rank nothing can reach."""
        # Index 1 is the argmax until it is penalised for having been seen.
        penalised = selector(repetition_penalty=4.0)
        assert int(penalised._select(LOGITS, [1])) == 2

    def test_the_penalty_also_applies_under_sampling(self):
        runner = selector(repetition_penalty=8.0, temperature=1.0, top_k=1,
                          seed=0)
        assert int(runner._select(LOGITS, [1])) == 2

    def test_no_penalty_without_history(self):
        assert int(selector(repetition_penalty=4.0)._select(LOGITS, [])) == 1


class TestValidation:
    @pytest.mark.parametrize(
        ("setting", "value"),
        [("temperature", -0.1), ("temperature", 2.1), ("top_p", 0.0),
         ("top_p", 1.1), ("top_k", -1), ("min_p", -0.1), ("min_p", 1.1)])
    def test_out_of_range_settings_are_refused_at_construction(self, setting, value,
                                                               monkeypatch):
        """Refused where it is cheap to fix, not at the first decode step in the
        middle of a benchmark."""
        import llm.runner as module

        class FakeTokenizer:
            eos_token_id = 0

        monkeypatch.setattr(module, "AutoModelForCausalLM", object, raising=False)
        with pytest.raises(ValueError, match=setting):
            LLMRunner(object(), FakeTokenizer(), torch.device("cpu"),
                      **{setting: value})
