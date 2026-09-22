"""Dialogue history: budget, barge-in honesty, and what reaches the prompt."""

from __future__ import annotations

from types import SimpleNamespace

from agent.conversation import INTERRUPTED_MARKER, Conversation, Exchange, spoken_text


def _result(state, transcript, response, sentences=None):
    return SimpleNamespace(
        state=SimpleNamespace(value=state), transcript=transcript, response=response,
        speech=SimpleNamespace(sentences=sentences or []),
    )


class TestMessages:
    def test_history_reaches_the_prompt_in_turn_order(self):
        c = Conversation("SYS")
        c.record("दिल्ली का मौसम कैसा है?", "दिल्ली में आज धूप है।")
        msgs = c.messages("और मुंबई का?")
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[0]["content"] == "SYS"
        assert msgs[1]["content"] == "दिल्ली का मौसम कैसा है?"
        assert msgs[-1]["content"] == "और मुंबई का?"

    def test_no_system_prompt_means_no_system_message(self):
        c = Conversation()
        assert [m["role"] for m in c.messages("x")] == ["user"]

    def test_empty_sides_are_never_recorded(self):
        c = Conversation("SYS")
        assert c.record("", "उत्तर") is False
        assert c.record("प्रश्न", "   ") is False
        assert c.history == []


class TestBargeInHonesty:
    def test_only_the_spoken_part_of_an_interrupted_turn_is_kept(self):
        """The user never heard sentence two; the model must not assume they did."""
        r = _result("interrupted", "एक सवाल", "पहला वाक्य। दूसरा वाक्य।",
                    sentences=["पहला वाक्य।"])
        spoken, interrupted = spoken_text(r)
        assert spoken == "पहला वाक्य।" and interrupted is True

        c = Conversation("SYS")
        assert c.record_turn(r) is True
        assistant = c.messages("अगला")[2]
        assert assistant["content"] == "पहला वाक्य।" + INTERRUPTED_MARKER
        assert "दूसरा" not in assistant["content"]

    def test_completed_turn_keeps_the_whole_response(self):
        r = _result("completed", "सवाल", "पूरा उत्तर है।", sentences=["पूरा उत्तर है।"])
        assert spoken_text(r) == ("पूरा उत्तर है।", False)
        c = Conversation()
        c.record_turn(r)
        assert c.history[0] == Exchange("सवाल", "पूरा उत्तर है।", False)

    def test_failed_turn_is_not_recorded(self):
        c = Conversation()
        assert c.record_turn(_result("failed", "सवाल", "")) is False
        assert c.history == []

    def test_interrupted_before_any_audio_records_nothing(self):
        c = Conversation()
        assert c.record_turn(_result("interrupted", "सवाल", "कुछ", sentences=[])) is False
        assert c.history == []


class TestBudget:
    def test_oldest_turns_drop_at_max_turns(self):
        c = Conversation("SYS", max_turns=2, max_history_tokens=10_000)
        for i in range(5):
            c.record(f"प्रश्न {i}", f"उत्तर {i}")
        assert len(c.history) == 2
        assert c.history[0].user == "प्रश्न 3"
        assert c.dropped == 3

    def test_token_budget_drops_old_turns_but_never_the_last(self):
        c = Conversation("SYS", max_turns=99, max_history_tokens=40, tokenizer=None)
        for i in range(8):
            c.record(f"यह एक लंबा प्रश्न है संख्या {i}", f"यह एक लंबा उत्तर है संख्या {i}")
        assert 1 <= len(c.history) < 8
        assert c.history_tokens() <= 40 or len(c.history) == 1
        assert c.history[-1].user.endswith("7")

        huge = Conversation("SYS", max_history_tokens=1)
        huge.record("बहुत लंबा प्रश्न " * 20, "बहुत लंबा उत्तर " * 20)
        assert len(huge.history) == 1, "the current exchange is never dropped"

    def test_devanagari_is_not_counted_as_four_chars_per_token(self):
        """A Latin-tuned estimate underestimates Hindi by several times; the
        same error produced the 225-token truncation bug."""
        c = Conversation()
        hindi = "नमस्ते दुनिया"
        assert c._count(hindi) > len(hindi) / 4
        assert c._count(hindi) >= len(hindi) / 2

    def test_real_tokenizer_is_used_when_supplied(self):
        calls = []

        def tok(text, add_special_tokens=False):
            calls.append(text)
            return {"input_ids": [0] * 7}

        c = Conversation(tokenizer=tok)
        c.record("प्रश्न", "उत्तर")
        assert c.history_tokens() == 14        # two messages x 7
        assert calls

    def test_broken_tokenizer_falls_back_instead_of_failing_a_turn(self):
        def tok(text, add_special_tokens=False):
            raise RuntimeError("boom")

        c = Conversation(tokenizer=tok)
        c.record("प्रश्न", "उत्तर")
        assert c.history_tokens() > 0

    def test_reset_clears_everything(self):
        c = Conversation("SYS", max_turns=1)
        c.record("a", "b")
        c.record("c", "d")
        c.reset()
        assert c.history == [] and c.dropped == 0
        assert [m["role"] for m in c.messages("x")] == ["system", "user"]


class TestTurnIntegration:
    def test_turn_uses_history_in_its_prompt_and_records_itself(self):
        from agent import BufferSink, VoiceTurn
        from agent import Conversation as Conv

        prompts = []

        class Recording:
            def __init__(self):
                self.tokenizer = None          # no chat template → text transcript

            def stream(self, prompt, **kw):
                prompts.append(prompt)
                yield "मुंबई में भी धूप है।"

        class Tts:
            streaming = True

            def stream(self, text):
                yield b"\x00" * 16

        conv = Conv("SYS", max_turns=4)
        turn = VoiceTurn(Recording(), Tts(), conversation=conv)

        first = turn.run("दिल्ली का मौसम?", sink=BufferSink())
        assert first.state.value == "completed"
        assert first.metrics.recorded_in_history is True
        assert "दिल्ली का मौसम?" in prompts[0]

        VoiceTurn(Recording(), Tts(), conversation=conv).run("और मुंबई का?", sink=BufferSink())
        # The second prompt carries the first exchange.
        assert "दिल्ली का मौसम?" in prompts[1]
        assert "मुंबई में भी धूप है।" in prompts[1]
        assert conv.snapshot()["turns"] == 2

    def test_turn_without_a_conversation_is_unchanged(self):
        from agent import BufferSink, VoiceTurn

        class Gen:
            def generate(self, prompt, **kw):
                return SimpleNamespace(text="ठीक है।",
                                       metrics=SimpleNamespace(prefill_ms=1.0))

        class Tts:
            streaming = True

            def stream(self, text):
                yield b"\x00"

        r = VoiceTurn(Gen(), Tts()).run("सवाल", sink=BufferSink())
        assert r.state.value == "completed"
        assert r.metrics.recorded_in_history is None

    def test_turn_adopts_its_own_system_prompt_when_the_conversation_has_none(self):
        from agent import Conversation as Conv
        from agent import VoiceTurn

        conv = Conv()
        turn = VoiceTurn(SimpleNamespace(generate=lambda *a, **k: None), None,
                         conversation=conv, response_language="Hindi")
        turn.build_prompt("सवाल")
        assert conv.system and "Hindi" in conv.system
