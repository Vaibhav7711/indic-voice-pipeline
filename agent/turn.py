"""One conversational turn: final transcript → LLM → TTS → playback.

The four latencies, and what each one honestly measures
-------------------------------------------------------
A voice agent's perceived responsiveness is one number — silence between the
user finishing and the agent starting — but it is made of four segments, and
only by splitting them can you tell which component to fix.

``speech_end_to_final_transcript_ms``
    From the VAD endpoint to the committed transcript. Note this **includes
    the endpointer's own silence threshold**: ``min_silence_ms`` is dead time by
    construction, because you cannot know the user stopped until they have been
    quiet for a while. Halving ``min_silence_ms`` halves that component and
    doubles the rate of cutting people off mid-sentence. It is a tuning
    decision, not a bug, and it should be visible in the number.

``final_transcript_to_first_llm_token_ms``
    Prompt construction plus prefill. Prefill is compute-bound in the prompt
    length — the same time-to-first-token you would optimise in any LLM serving
    stack.

``first_llm_token_to_playback_start_ms``
    Where sentence-level TTS earns its keep: synthesise the first sentence
    rather than the whole response and this segment shrinks by most of the
    response length.

``total_turn_ms``
    End to end. Not the sum of the above when stages overlap.

Pipelining
----------
LLM deltas feed a :class:`tts.streaming.SentenceBuffer`; each complete
sentence is synthesised and played while the LLM is still producing the next
one. So ``first_llm_token_to_playback_start_ms`` covers the first *sentence*
of decode plus the first TTS round trip, not the whole response. A barge-in
cancels playback, which stops synthesis, which stops the LLM decode loop
(``should_stop`` is threaded all the way back) — tokens that would never be
heard are never generated.

Measurement honesty
-------------------
Two kinds of backend, and the turn records which applied:

* Backend exposes ``stream()`` → first-token time is measured at the first
  yielded piece. ``llm_streaming=True``. ``llm.runner.LLMRunner`` does.
* Backend only has ``generate()`` → the whole response arrives at once. The
  first-token instant is taken as ``llm_start + metrics.prefill_ms`` (prefill
  completion is when the first token exists) and
  ``first_token_is_prefill_proxy=True`` is set. The decode time then lands in
  ``first_llm_token_to_playback_start_ms``, where it belongs: the user is
  waiting through it. (An earlier version stamped "first token" *after*
  ``generate()`` returned, which hid the entire decode from
  ``response_latency_ms``.)

Any field that could not be measured is ``None``, never zero. A zero would be
averaged into a benchmark; a ``None`` forces the question.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Any, Protocol

from agent.playback import AudioSink, BufferSink, PlaybackResult, PlaybackSession
from llm.prompting import build_chat_prompt, render_messages
from tts.streaming import SentenceBuffer, SpeechStream, iter_sentence

__all__ = [
    "TurnState",
    "TurnMetrics",
    "TurnResult",
    "ResponseGenerator",
    "VoiceTurn",
]


class TurnState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"        # LLM running
    SPEAKING = "speaking"        # TTS + playback running
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"  # barge-in
    FAILED = "failed"


class ResponseGenerator(Protocol):
    """What the turn needs from an LLM backend.

    ``llm.runner.LLMRunner`` satisfies this structurally. A backend may also
    expose ``stream(prompt, **kwargs) -> Iterator[str]``; when present it is
    preferred, because it makes first-token latency directly observable.
    """

    def generate(self, prompt: str, **kwargs: Any) -> Any: ...


@dataclass
class TurnMetrics:
    """All fields optional — ``None`` means "not measurable", never zero."""

    speech_end_to_final_transcript_ms: float | None = None
    final_transcript_to_first_llm_token_ms: float | None = None
    first_llm_token_to_playback_start_ms: float | None = None
    total_turn_ms: float = 0.0

    llm_total_ms: float | None = None
    tts_first_chunk_ms: float | None = None
    #: ``first_llm_token_to_playback_start_ms`` split in two: how long the
    #: LLM kept generating before the first speakable unit existed, and how
    #: long the synthesizer then took. These do sum to that segment.
    first_token_to_first_unit_ms: float | None = None
    tts_synthesis_ms: float | None = None
    playback_first_audio_ms: float | None = None

    # Provenance, so a reader can tell a measurement from an approximation.
    llm_streaming: bool = False
    first_token_is_prefill_proxy: bool = False
    tts_streaming: bool = False
    tts_sentence_level: bool = False
    barge_in: bool = False
    #: LLM decode was cut short by the barge-in (streaming backends only).
    llm_stopped_by_barge_in: bool = False
    #: Whether this turn was added to the conversation history.
    recorded_in_history: bool | None = None
    llm_generated_tokens: int | None = None

    @property
    def response_latency_ms(self) -> float | None:
        """User stops talking → agent starts talking. The perceived number."""
        parts = [
            self.speech_end_to_final_transcript_ms,
            self.final_transcript_to_first_llm_token_ms,
            self.first_llm_token_to_playback_start_ms,
        ]
        if any(part is None for part in parts):
            return None
        return sum(parts)  # type: ignore[arg-type]

    def as_dict(self) -> dict:
        def ms(value):
            return None if value is None else round(value, 3)

        return {
            "speech_end_to_final_transcript_ms": ms(
                self.speech_end_to_final_transcript_ms
            ),
            "final_transcript_to_first_llm_token_ms": ms(
                self.final_transcript_to_first_llm_token_ms
            ),
            "first_llm_token_to_playback_start_ms": ms(
                self.first_llm_token_to_playback_start_ms
            ),
            "response_latency_ms": ms(self.response_latency_ms),
            "total_turn_ms": ms(self.total_turn_ms),
            "llm_total_ms": ms(self.llm_total_ms),
            "tts_first_chunk_ms": ms(self.tts_first_chunk_ms),
            "first_token_to_first_unit_ms": ms(self.first_token_to_first_unit_ms),
            "tts_synthesis_ms": ms(self.tts_synthesis_ms),
            "playback_first_audio_ms": ms(self.playback_first_audio_ms),
            "llm_streaming": self.llm_streaming,
            "first_token_is_prefill_proxy": self.first_token_is_prefill_proxy,
            "tts_streaming": self.tts_streaming,
            "tts_sentence_level": self.tts_sentence_level,
            "barge_in": self.barge_in,
            "llm_stopped_by_barge_in": self.llm_stopped_by_barge_in,
            "recorded_in_history": self.recorded_in_history,
            "llm_generated_tokens": self.llm_generated_tokens,
        }


@dataclass
class TurnResult:
    turn_id: str
    state: TurnState
    transcript: str = ""
    response: str = ""
    metrics: TurnMetrics = field(default_factory=TurnMetrics)
    speech: SpeechStream | None = None
    playback: PlaybackResult | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "state": self.state.value,
            "transcript": self.transcript,
            "response": self.response,
            "metrics": self.metrics.as_dict(),
            "speech": self.speech.as_dict() if self.speech else None,
            "playback": self.playback.as_dict() if self.playback else None,
            "error": self.error,
        }


class VoiceTurn:
    """Runs one turn and reports where the time went.

    Composition, not inheritance: the LLM, synthesizer and sink are injected,
    so unit tests substitute fakes and never touch a network or a GPU.
    """

    def __init__(
        self,
        generator: ResponseGenerator,
        synthesizer,
        *,
        sink_factory=None,
        system_prompt: str | None = None,
        response_language: str = "hi",
        split_into_sentences: bool = True,
        max_unit_chars: int = 60,
        llm_max_tokens: int = 128,
        conversation=None,
        clock=None,
    ):
        self.generator = generator
        self.synthesizer = synthesizer
        self.sink_factory = sink_factory or BufferSink
        self.system_prompt = system_prompt
        self.response_language = response_language
        self.split_into_sentences = split_into_sentences
        #: Longest unit sent to synthesis before it is cut at a clause
        #: boundary. It sets how long the user waits for the *first* audio:
        #: the measured first unit was 41.2 characters at ~61 ms each, so the
        #: cap and the silence before speech are the same number. Injectable
        #: because it is the cheapest latency knob in the pipeline and so has
        #: to be A/B-able without editing this file (`scripts/latency_ab.py`).
        self.max_unit_chars = max_unit_chars
        self.llm_max_tokens = llm_max_tokens
        #: Optional :class:`agent.conversation.Conversation`. When present the
        #: prompt carries the dialogue history and each completed turn is
        #: recorded — with only the audio the user actually heard, so a
        #: barge-in does not leave the model assuming an unheard sentence
        #: was delivered.
        self.conversation = conversation
        self._clock = clock or (lambda: perf_counter_ns() / 1_000_000)

        self.state = TurnState.IDLE
        self.playback: PlaybackSession | None = None

    # -- prompt -----------------------------------------------------------

    def build_prompt(self, transcript: str) -> str:
        system = self.system_prompt or (
            f"You are a helpful voice assistant. Reply entirely in natural "
            f"{self.response_language}, in one or two short sentences. Your "
            "reply is spoken aloud, so a long first sentence leaves the user "
            "waiting in silence. Never repeat the question back."
        )
        tokenizer = getattr(self.generator, "tokenizer", None)
        if self.conversation is not None:
            if self.conversation.system is None:
                self.conversation.system = system
            return render_messages(tokenizer, self.conversation.messages(transcript))
        return build_chat_prompt(tokenizer, system, transcript)

    # -- barge-in ---------------------------------------------------------

    def interrupt(self, reason: str = "barge_in") -> bool:
        """Cancel the in-flight response. Safe from another thread."""
        if self.playback is None:
            return False
        return self.playback.cancel(reason)

    # -- LLM --------------------------------------------------------------

    def _llm_deltas(self, prompt: str, metrics: TurnMetrics, playback: PlaybackSession,
                    marks: dict):
        """Yield response text pieces, stamping ``marks["first_token_at"]``."""
        start = self._clock()
        marks["llm_start"] = start

        streamer = getattr(self.generator, "stream", None)
        if callable(streamer):
            metrics.llm_streaming = True
            try:
                pieces = streamer(prompt, max_new_tokens=self.llm_max_tokens,
                                  should_stop=playback.should_stop)
            except TypeError:  # backend without should_stop support
                pieces = streamer(prompt, max_new_tokens=self.llm_max_tokens)
            finished = False
            try:
                for piece in pieces:
                    if marks.get("first_token_at") is None:
                        marks["first_token_at"] = self._clock()
                        metrics.final_transcript_to_first_llm_token_ms = (
                            marks["first_token_at"] - start
                        )
                    yield piece
                finished = True
            finally:
                # Runs on natural completion *and* when the consumer stops
                # early (barge-in closes this generator). Close the backend's
                # generator now so its own bookkeeping runs before we read it.
                close = getattr(pieces, "close", None)
                if callable(close):
                    close()
                metrics.llm_total_ms = self._clock() - start
                last = getattr(self.generator, "last_metrics", None)
                metrics.llm_stopped_by_barge_in = (
                    bool(getattr(last, "stopped_by_caller", False))
                    or (not finished and playback.should_stop())
                )
                metrics.llm_generated_tokens = getattr(last, "generated_tokens", None)
            return

        result = self.generator.generate(prompt, max_new_tokens=self.llm_max_tokens)
        metrics.llm_total_ms = self._clock() - start
        prefill = getattr(getattr(result, "metrics", None), "prefill_ms", None)
        metrics.first_token_is_prefill_proxy = True
        if prefill is not None:
            metrics.final_transcript_to_first_llm_token_ms = float(prefill)
            marks["first_token_at"] = start + float(prefill)
        else:
            metrics.final_transcript_to_first_llm_token_ms = metrics.llm_total_ms
            marks["first_token_at"] = self._clock()
        metrics.llm_generated_tokens = getattr(
            getattr(result, "metrics", None), "generated_tokens", None,
        )
        text = getattr(result, "text", "") or ""
        if text:
            yield text

    # -- turn -------------------------------------------------------------

    def run(
        self,
        transcript: str,
        *,
        speech_end_to_transcript_ms: float | None = None,
        sink: AudioSink | None = None,
    ) -> TurnResult:
        """Run a full turn from a committed (final) transcript."""
        turn_id = uuid.uuid4().hex[:12]
        turn_start = self._clock()
        metrics = TurnMetrics(
            speech_end_to_final_transcript_ms=speech_end_to_transcript_ms,
            tts_streaming=bool(getattr(self.synthesizer, "streaming", False)),
            tts_sentence_level=self.split_into_sentences,
        )
        result = TurnResult(turn_id=turn_id, state=TurnState.THINKING, transcript=transcript)

        if not transcript.strip():
            self.state = TurnState.COMPLETED
            result.state = TurnState.COMPLETED
            metrics.total_turn_ms = self._clock() - turn_start
            result.metrics = metrics
            return result

        self.state = TurnState.THINKING
        playback = PlaybackSession(
            sink or self.sink_factory(), playback_id=turn_id, clock=self._clock
        )
        self.playback = playback
        speech = SpeechStream(streaming=metrics.tts_streaming)
        marks: dict = {}
        response_parts: list[str] = []
        errors: dict = {}
        synth_start_ns = perf_counter_ns()

        def produce():
            """Audio chunks, driven by LLM deltas through the sentence buffer."""
            buffer = SentenceBuffer(enabled=self.split_into_sentences,
                                    max_unit_chars=self.max_unit_chars)
            prompt = self.build_prompt(transcript)
            try:
                for piece in self._llm_deltas(prompt, metrics, playback, marks):
                    response_parts.append(piece)
                    for sentence in buffer.feed(piece):
                        yield from synth(sentence)
                    if playback.should_stop():
                        return
                for sentence in buffer.flush():
                    yield from synth(sentence)
            except Exception as exc:  # noqa: BLE001 - recorded on the result
                errors.setdefault("llm", f"{type(exc).__name__}: {exc}")
            finally:
                speech.sentences = list(buffer.emitted)
                speech.total_ms = (perf_counter_ns() - synth_start_ns) / 1_000_000

        synthesised = {"count": 0}

        def synth(sentence: str):
            self.state = TurnState.SPEAKING
            index = synthesised["count"]
            synthesised["count"] += 1
            # On the turn's own clock, so it is comparable with the first-token
            # and first-audio marks. SpeechStream keeps its own perf_counter
            # timeline for standalone use of iter_synthesis.
            if marks.get("first_unit_at") is None:
                marks["first_unit_at"] = self._clock()
            try:
                for chunk in iter_sentence(
                    self.synthesizer, sentence, index, speech,
                    start_ns=synth_start_ns, should_stop=playback.should_stop,
                ):
                    yield chunk.data
            except Exception as exc:  # noqa: BLE001 - surfaced as state, not a crash
                speech.error = f"{type(exc).__name__}: {exc}"
                errors.setdefault("tts", speech.error)

        playback_start = self._clock()
        playback_result = playback.play(produce())

        result.response = "".join(response_parts).strip()
        result.speech = speech
        result.playback = playback_result
        metrics.tts_first_chunk_ms = speech.first_chunk_ms
        metrics.playback_first_audio_ms = playback_result.first_audio_ms

        first_token_at = marks.get("first_token_at")
        first_unit_at = marks.get("first_unit_at")
        if playback_result.first_audio_ms is not None and first_token_at is not None:
            first_audio_at = playback_start + playback_result.first_audio_ms
            metrics.first_llm_token_to_playback_start_ms = max(0.0, first_audio_at - first_token_at)
            if first_unit_at is not None:
                # The segment split in two, on one clock, so they sum: the LLM
                # still generating until a unit was speakable, then synthesis.
                metrics.first_token_to_first_unit_ms = max(0.0, first_unit_at - first_token_at)
                metrics.tts_synthesis_ms = max(0.0, first_audio_at - first_unit_at)
        metrics.barge_in = playback_result.interrupted

        if "llm" in errors and not response_parts:
            self.state = TurnState.FAILED
            result.error = errors["llm"]
        elif playback_result.state.value == "failed":
            self.state = TurnState.FAILED
            result.error = playback_result.error
        elif playback_result.interrupted:
            self.state = TurnState.INTERRUPTED
        elif errors.get("tts") and playback_result.chunks_written == 0:
            # Synthesis raised and nothing reached the sink. Playback consumed
            # an empty iterable, so it reports COMPLETED and not interrupted --
            # and a completed turn hands its whole response to the dialogue
            # history as though it had been spoken. Every later prompt would
            # then claim the agent said something the user never heard, which
            # is the one thing `agent.conversation` promises it never does.
            self.state = TurnState.FAILED
            result.error = "; ".join(f"{k}: {v}" for k, v in errors.items())
        else:
            self.state = TurnState.COMPLETED
            if errors:
                result.error = "; ".join(f"{k}: {v}" for k, v in errors.items())

        result.state = self.state
        metrics.total_turn_ms = self._clock() - turn_start
        result.metrics = metrics
        if self.conversation is not None:
            metrics.recorded_in_history = self.conversation.record_turn(result)
        return result

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "playback": self.playback.snapshot() if self.playback else None,
            "conversation": self.conversation.snapshot() if self.conversation else None,
        }
