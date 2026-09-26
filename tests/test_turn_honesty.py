"""A turn that produced no audio must not enter the dialogue history as spoken.

`agent/conversation.py` states the contract: "Nothing invented. An empty or
failed turn is not added to history at all", and "a turn records what was
*spoken*". A synthesis failure broke it. `synth()` swallows the backend
exception into `errors["tts"]`, playback then consumes an empty iterable and
reports COMPLETED and not interrupted, so the turn was classified COMPLETED --
and `spoken_text` returns the whole response for any state that is not
"interrupted". The full assistant sentence entered the history and every later
prompt claimed the agent had answered.

Fixed in two places on purpose. The state machine now calls it FAILED, and
`record_turn` independently refuses a result that wrote no bytes, because the
failure that matters is silence whatever caused it.
"""

from __future__ import annotations

from agent import Conversation, VoiceTurn
from agent.conversation import heard_nothing
from agent.playback import BufferSink
from agent.turn import TurnState


class ExplodingTTS:
    streaming = True
    format = "mp3"

    def stream(self, text: str):
        raise RuntimeError("synthesis backend died")
        yield b""            # pragma: no cover - generator marker


class SilentTTS:
    """Succeeds but emits nothing, so there is no error to notice."""

    streaming = True
    format = "mp3"

    def stream(self, text: str):
        return iter(())


class WorkingTTS:
    streaming = True
    format = "mp3"

    def stream(self, text: str):
        yield f"audio:{text[:6]}".encode()


class FakeLLM:
    def __init__(self, text="नमस्ते। मैं ठीक हूँ।"):
        self.text = text

    def stream(self, prompt, **kwargs):
        for piece in self.text.split():
            yield piece + " "


class TestSynthesisFailure:
    def test_the_turn_is_not_completed(self):
        turn = VoiceTurn(FakeLLM(), ExplodingTTS(), sink_factory=BufferSink)
        result = turn.run("आप कैसे हैं?")
        assert result.state is TurnState.FAILED
        assert "tts" in (result.error or "")

    def test_nothing_enters_the_history(self):
        conversation = Conversation(max_turns=6)
        turn = VoiceTurn(FakeLLM(), ExplodingTTS(), sink_factory=BufferSink,
                         conversation=conversation)
        result = turn.run("आप कैसे हैं?")
        assert result.metrics.recorded_in_history is False
        assert conversation.history == []

    def test_the_next_prompt_does_not_claim_the_agent_answered(self):
        """The consequence the contract exists to prevent."""
        conversation = Conversation(max_turns=6)
        turn = VoiceTurn(FakeLLM(), ExplodingTTS(), sink_factory=BufferSink,
                         conversation=conversation)
        turn.run("पहला सवाल")
        messages = conversation.messages("दूसरा सवाल")
        assert all(m["role"] != "assistant" for m in messages)

    def test_a_working_turn_still_records(self):
        conversation = Conversation(max_turns=6)
        turn = VoiceTurn(FakeLLM(), WorkingTTS(), sink_factory=BufferSink,
                         conversation=conversation)
        result = turn.run("आप कैसे हैं?")
        assert result.state is TurnState.COMPLETED
        assert result.metrics.recorded_in_history is True
        assert any(m["role"] == "assistant"
                   for m in conversation.messages("अगला"))


class TestSilentSynthesisWithoutAnError:
    """The case the state machine alone would miss: no exception, no audio."""

    def test_record_turn_refuses_it(self):
        conversation = Conversation(max_turns=6)
        turn = VoiceTurn(FakeLLM(), SilentTTS(), sink_factory=BufferSink,
                         conversation=conversation)
        result = turn.run("आप कैसे हैं?")
        assert result.playback.bytes_written == 0
        assert result.metrics.recorded_in_history is False
        assert conversation.history == []


class TestHeardNothing:
    def test_text_with_no_audio_is_caught(self):
        class R:
            response = "कुछ कहा"
            playback = type("P", (), {"bytes_written": 0})()

        assert heard_nothing(R()) is True

    def test_text_with_audio_is_fine(self):
        class R:
            response = "कुछ कहा"
            playback = type("P", (), {"bytes_written": 128})()

        assert heard_nothing(R()) is False

    def test_an_empty_response_is_not_this_case(self):
        """Nothing to say is handled by the empty-turn path, not here."""
        class R:
            response = ""
            playback = type("P", (), {"bytes_written": 0})()

        assert heard_nothing(R()) is False

    def test_a_result_with_no_playback_at_all_is_not_flagged(self):
        """A turn that failed before playback: its state already says so, and
        guessing from a missing field would be inventing a verdict."""
        class R:
            response = "कुछ कहा"
            playback = None

        assert heard_nothing(R()) is False
