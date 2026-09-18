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

Measurement honesty
-------------------
``LLMRunner.generate()` is **not** a streaming generator — it returns the whole
response. From outside, there is no way to observe when the first token
appeared. Two cases, and the turn records which one applied:

* Backend exposes ``stream()`` → first-token time is measured directly at the
  first yielded piece. ``llm_streaming=True``.
* Backend does not → the turn falls back to the runner's internally measured
  ``metrics.prefill_ms``, since prefill completion is when the first token
  exists. ``first_token_is_prefill_proxy=True`` is set so no one later reads it
  as a wall-clock measurement. It excludes the Python-side overhead between
  ``generate()`` returning and the caller seeing it.

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
from tts.streaming import SpeechStream, split_sentences, synthesize_stream

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
    playback_first_audio_ms: float | None = None

    # Provenance, so a reader can tell a measurement from an approximation.
    llm_streaming: bool = False
    first_token_is_prefill_proxy: bool = False
    tts_streaming: bool = False
    tts_sentence_level: bool = False
    barge_in: bool = False

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
            "playback_first_audio_ms": ms(self.playback_first_audio_ms),
            "llm_streaming": self.llm_streaming,
            "first_token_is_prefill_proxy": self.first_token_is_prefill_proxy,
            "tts_streaming": self.tts_streaming,
            "tts_sentence_level": self.tts_sentence_level,
            "barge_in": self.barge_in,
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
        llm_max_tokens: int = 128,
        clock=None,
    ):
        self.generator = generator
        self.synthesizer = synthesizer
        self.sink_factory = sink_factory or BufferSink
        self.system_prompt = system_prompt
        self.response_language = response_language
        self.split_into_sentences = split_into_sentences
        self.llm_max_tokens = llm_max_tokens
        self._clock = clock or (lambda: perf_counter_ns() / 1_000_000)

        self.state = TurnState.IDLE
        self.playback: PlaybackSession | None = None

    # -- prompt -----------------------------------------------------------

    def build_prompt(self, transcript: str) -> str:
        system = self.system_prompt or (
            f"You are a helpful voice assistant. Reply entirely in natural "
            f"{self.response_language} and keep the answer brief — this will "
            "be spoken aloud."
        )
        tokenizer = getattr(self.generator, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except (TypeError, ValueError):
                try:
                    return tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                except (TypeError, ValueError):
                    pass
        return f"System: {system}\n\nUser: {transcript}\n\nAssistant:"

    # -- barge-in ---------------------------------------------------------

    def interrupt(self, reason: str = "barge_in") -> bool:
        """Cancel the in-flight response. Safe from another thread."""
        if self.playback is None:
            return False
        return self.playback.cancel(reason)

    # -- LLM --------------------------------------------------------------

    def _run_llm(self, prompt: str, metrics: TurnMetrics) -> str:
        start = self._clock()

        streamer = getattr(self.generator, "stream", None)
        if callable(streamer):
            metrics.llm_streaming = True
            pieces: list[str] = []
            for piece in streamer(prompt, max_new_tokens=self.llm_max_tokens):
                if not pieces:
                    metrics.final_transcript_to_first_llm_token_ms = (
                        self._clock() - start
                    )
                pieces.append(piece)
            metrics.llm_total_ms = self._clock() - start
            return "".join(pieces)

        result = self.generator.generate(prompt, max_new_tokens=self.llm_max_tokens)
        metrics.llm_total_ms = self._clock() - start

        # Non-streaming backend: prefill completion is when the first token
        # exists, and only the runner can see it. Flagged as a proxy.
        prefill = getattr(getattr(result, "metrics", None), "prefill_ms", None)
        if prefill is not None:
            metrics.final_transcript_to_first_llm_token_ms = float(prefill)
            metrics.first_token_is_prefill_proxy = True
        else:
            metrics.final_transcript_to_first_llm_token_ms = metrics.llm_total_ms
            metrics.first_token_is_prefill_proxy = True

        return getattr(result, "text", "") or ""

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
        try:
            response = self._run_llm(self.build_prompt(transcript), metrics)
        except Exception as exc:  # noqa: BLE001
            self.state = TurnState.FAILED
            result.state = TurnState.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            metrics.total_turn_ms = self._clock() - turn_start
            result.metrics = metrics
            return result

        result.response = response
        first_token_at = self._clock()

        self.state = TurnState.SPEAKING
        playback = PlaybackSession(
            sink or self.sink_factory(), playback_id=turn_id, clock=self._clock
        )
        self.playback = playback

        # Synthesis is driven lazily by playback, so a barge-in during the
        # first sentence stops the remaining sentences from being synthesised
        # at all rather than being generated and thrown away.
        chunks: list[bytes] = []
        speech = SpeechStream(
            sentences=split_sentences(response) if self.split_into_sentences else [response],
            streaming=metrics.tts_streaming,
        )

        def produce():
            nonlocal speech
            speech = synthesize_stream(
                self.synthesizer,
                response,
                split=self.split_into_sentences,
                should_stop=playback.should_stop,
            )
            for chunk in speech.chunks:
                chunks.append(chunk.data)
                yield chunk.data

        playback_start = self._clock()
        playback_result = playback.play(produce())

        result.speech = speech
        result.playback = playback_result
        metrics.tts_first_chunk_ms = speech.first_chunk_ms
        metrics.playback_first_audio_ms = playback_result.first_audio_ms

        if playback_result.first_audio_ms is not None:
            # first_audio_ms is relative to when play() began, so the segment
            # from first token is the gap before play() plus that offset.
            metrics.first_llm_token_to_playback_start_ms = max(
                0.0,
                (playback_start - first_token_at) + playback_result.first_audio_ms,
            )
        metrics.barge_in = playback_result.interrupted

        if playback_result.state.value == "failed":
            self.state = TurnState.FAILED
            result.error = playback_result.error
        elif playback_result.interrupted:
            self.state = TurnState.INTERRUPTED
        else:
            self.state = TurnState.COMPLETED

        result.state = self.state
        metrics.total_turn_ms = self._clock() - turn_start
        result.metrics = metrics
        return result

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "playback": self.playback.snapshot() if self.playback else None,
        }
