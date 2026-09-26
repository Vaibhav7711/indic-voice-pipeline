"""Dialogue state: what was said, what was actually heard, what fits in the prompt.

Without this, every utterance is answered as if it were the first — ask
"दिल्ली का मौसम कैसा है?" then "और मुंबई का?" and the second question has no
referent. A voice assistant needs the previous turns in its prompt.

Three things this has to get right, and the third is specific to voice:

**A budget.** History grows without bound and prefill cost grows with it
(and the agent's time-to-first-token with it). Turns are dropped oldest-first
to stay under ``max_turns`` and a token budget. The budget is measured with
the model's own tokenizer when one is supplied, because a character estimate
is wrong by a factor of ~6 for Devanagari — the same mistake that produced
the 225-token truncation bug.

**Barge-in honesty.** If the user interrupted, the agent's later sentences
were never heard. Recording the full generated text would leave the model
believing the user has information they do not have; the next turn then
refers back to something unsaid. So a turn records what was *spoken* — the
sentences synthesis actually reached — and marks it truncated. This is the
voice-specific part: in a text chat, generated and delivered are the same
thing.

**Nothing invented.** An empty or failed turn is not added to history at
all, rather than stored as an empty assistant message that teaches the model
that silence is a valid reply.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Exchange", "Conversation", "spoken_text", "heard_nothing"]

#: Appended to an assistant message the user interrupted, so the model can
#: see the difference between "I said this" and "I started saying this".
INTERRUPTED_MARKER = "…"


@dataclass(frozen=True)
class Exchange:
    """One completed user→assistant round trip, as the user experienced it."""

    user: str
    assistant: str
    interrupted: bool = False

    def as_messages(self) -> list[dict]:
        content = self.assistant
        if self.interrupted and content and not content.endswith(INTERRUPTED_MARKER):
            content = f"{content}{INTERRUPTED_MARKER}"
        return [{"role": "user", "content": self.user},
                {"role": "assistant", "content": content}]

    def as_dict(self) -> dict:
        return {"user": self.user, "assistant": self.assistant,
                "interrupted": self.interrupted}


def heard_nothing(result) -> bool:
    """True when the turn had something to say and no audio reached the sink.

    A second line of defence behind the turn's own state machine. `spoken_text`
    returns the *whole* response for any state that is not "interrupted", so a
    turn that produced text and no audio would enter the history as spoken. The
    check is on bytes actually written rather than on an error field, because
    the failure that matters is silence, whatever caused it.

    An empty response is not this case -- there was nothing to say, and the
    empty-turn path already refuses it.
    """
    response = (getattr(result, "response", "") or "").strip()
    if not response:
        return False
    written = getattr(getattr(result, "playback", None), "bytes_written", None)
    return written == 0


def spoken_text(result) -> tuple[str, bool]:
    """What the user actually heard from a :class:`agent.turn.TurnResult`.

    On a completed turn that is the whole response. On a barge-in it is the
    sentences that were synthesised — ``speech.sentences`` only contains
    sentences that reached the synthesizer, because the turn feeds them
    lazily as the LLM produces them.
    """
    state = getattr(getattr(result, "state", None), "value", "")
    response = (getattr(result, "response", "") or "").strip()
    if state != "interrupted":
        return response, False
    speech = getattr(result, "speech", None)
    spoken = " ".join(getattr(speech, "sentences", []) or []).strip()
    return spoken, True


class Conversation:
    """Rolling dialogue history with a prompt budget.

    ``messages(user)`` returns the message list for the next prompt:
    system, then the surviving history, then this user turn.
    """

    def __init__(
        self,
        system: str | None = None,
        *,
        max_turns: int = 6,
        max_history_tokens: int = 800,
        tokenizer=None,
    ):
        self.system = system
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens
        self.tokenizer = tokenizer
        self.history: list[Exchange] = []
        #: Exchanges dropped to stay inside the budget, for observability.
        self.dropped = 0

    # -- recording --------------------------------------------------------

    def record(self, user: str, assistant: str, *, interrupted: bool = False) -> bool:
        """Add an exchange. Returns False if it was not worth recording."""
        if not user.strip() or not assistant.strip():
            return False
        self.history.append(Exchange(user.strip(), assistant.strip(), interrupted))
        self._trim()
        return True

    def record_turn(self, result) -> bool:
        """Record a :class:`agent.turn.TurnResult`, keeping only what was heard."""
        if getattr(getattr(result, "state", None), "value", "") == "failed":
            return False
        if heard_nothing(result):
            return False
        spoken, interrupted = spoken_text(result)
        return self.record(getattr(result, "transcript", "") or "", spoken,
                           interrupted=interrupted)

    def reset(self) -> None:
        self.history.clear()
        self.dropped = 0

    # -- prompting --------------------------------------------------------

    def messages(self, user: str) -> list[dict]:
        out: list[dict] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        for exchange in self.history:
            out.extend(exchange.as_messages())
        out.append({"role": "user", "content": user})
        return out

    # -- budget -----------------------------------------------------------

    def _count(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:  # noqa: BLE001 - fall back rather than fail a turn
                pass
        # Fallback: ~4 characters per token for Latin, ~1.5 for Devanagari.
        # Deliberately pessimistic for Devanagari so the budget is not
        # overshot on Hindi, where a character estimate tuned for English is
        # wrong by a factor of several.
        deva = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
        return int(deva / 1.5 + (len(text) - deva) / 4) + 1

    def history_tokens(self) -> int:
        return sum(self._count(m["content"]) for e in self.history for m in e.as_messages())

    def _trim(self) -> None:
        while len(self.history) > self.max_turns:
            self.history.pop(0)
            self.dropped += 1
        # Always keep the most recent exchange, even if it alone exceeds the
        # budget: dropping it would answer the current question with no
        # context at all, which is worse than a long prompt.
        while len(self.history) > 1 and self.history_tokens() > self.max_history_tokens:
            self.history.pop(0)
            self.dropped += 1

    # -- observability ----------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "turns": len(self.history),
            "dropped": self.dropped,
            "history_tokens": self.history_tokens(),
            "max_turns": self.max_turns,
            "max_history_tokens": self.max_history_tokens,
            "exchanges": [e.as_dict() for e in self.history],
        }
