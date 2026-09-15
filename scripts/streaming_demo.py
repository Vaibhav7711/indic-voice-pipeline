#!/usr/bin/env python3
"""Simulated end-to-end voice turn: microphone → streaming ASR → LLM → speech.

Everything is faked — no GPU, no model download, no network. The point is to
show the *shape* of the integration an application writes, and to make the
state transitions and latency breakdown visible.

    python scripts/streaming_demo.py
    python scripts/streaming_demo.py --seconds 20 --json
    python scripts/streaming_demo.py --barge-in

To drive real audio, swap ``FakeTranscriber`` for a real ``ASRRunner``; the
session only needs ``transcribe_array`` and ``transcribe_long_array``.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import BufferSink, TurnState, VoiceTurn  # noqa: E402
from asr.streaming import (  # noqa: E402
    SessionState,
    StreamingConfig,
    StreamingSession,
    UpdateKind,
)
from asr.vad import VADConfig  # noqa: E402

SAMPLE_RATE = 16_000

TRANSCRIPTS = [
    "नमस्ते मैं दिल्ली से बोल रहा हूँ",
    "कल का मौसम कैसा रहेगा",
    "मैंने laptop पर meeting schedule कर दी है",
]


class SimulatedClock:
    """Wall clock tied to stream position, not to how fast the loop runs.

    Without this the demo replays 16 s of audio in a few milliseconds, so
    ``partial_interval_ms`` never elapses and no partials appear. That is the
    wall-clock rate limiter working exactly as designed — but it hides the
    feature. Advancing the clock with the audio simulates real-time capture.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTranscriber:
    """Stands in for ASRRunner.

    Reveals words in proportion to how much audio it was given, so a partial
    (less audio) naturally looks like a prefix of the final (all the audio).
    That mirrors the real behaviour worth demonstrating: partials are
    incomplete and change as more audio arrives.
    """

    #: Audio length at which the canned transcript is considered complete.
    REFERENCE_SECONDS = 6.0

    def __init__(self) -> None:
        self.index = 0
        self.previous_seconds = 0.0

    def _text(self, seconds: float) -> str:
        # A *much* shorter clip than the last one means a new utterance began.
        # The 0.5 factor matters: a final is slightly shorter than the partial
        # before it, because finals are trimmed to the padded speech boundary
        # while partials include the trailing silence accumulated so far. A
        # strict "shorter than last" test would read that as a new utterance.
        if seconds < self.previous_seconds * 0.5:
            self.index += 1
        self.previous_seconds = seconds

        words = TRANSCRIPTS[self.index % len(TRANSCRIPTS)].split()
        shown = round(len(words) * min(1.0, seconds / self.REFERENCE_SECONDS))
        return " ".join(words[: max(1, shown)])

    def transcribe_array(self, waveform, sample_rate, **kwargs):
        seconds = len(waveform) / sample_rate
        return types.SimpleNamespace(
            text=self._text(seconds),
            metrics=types.SimpleNamespace(total_ms=seconds * 120.0),
        )

    def transcribe_long_array(self, waveform, sample_rate, **kwargs):
        seconds = len(waveform) / sample_rate
        return types.SimpleNamespace(
            text=self._text(seconds),
            metrics=types.SimpleNamespace(total_ms=seconds * 140.0),
        )


class FakeLLM:
    """Non-streaming, like the real LLMRunner — exercises the prefill proxy."""

    def generate(self, prompt: str, **kwargs):
        return types.SimpleNamespace(
            text="मौसम साफ़ रहेगा। तापमान लगभग 28 डिग्री रहेगा। छाता ज़रूरी नहीं है।",
            metrics=types.SimpleNamespace(prefill_ms=85.0, total_ms=340.0),
        )


class FakeTTS:
    streaming = True

    def stream(self, text: str):
        # One "frame" per few characters, roughly what a real stream looks like.
        encoded = text.encode()
        for start in range(0, len(encoded), 64):
            yield encoded[start : start + 64]


def build_audio(seconds: float) -> np.ndarray:
    """Silence, speech, pause, speech, silence — the interesting cases."""
    rng = np.random.default_rng(0)

    def block(duration: float, amplitude: float) -> np.ndarray:
        samples = round(duration * SAMPLE_RATE)
        if amplitude == 0.0:
            return (rng.normal(0, 0.0005, samples)).astype(np.float32)
        return (rng.normal(0, amplitude, samples)).astype(np.float32)

    speech = max(1.0, (seconds - 4.0) / 2)
    return np.concatenate([
        block(1.0, 0.0),        # silence before speech
        block(0.08, 0.4),       # a cough: below min_speech_ms, must be ignored
        block(0.6, 0.0),
        block(speech, 0.25),    # utterance 1
        block(0.25, 0.0),       # short pause: must NOT endpoint
        block(speech, 0.25),    # ...same utterance continues
        block(1.0, 0.0),        # real endpoint
        block(speech, 0.25),    # utterance 2
        block(1.2, 0.0),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--block-ms", type=int, default=20,
                        help="simulated microphone buffer size")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument("--barge-in", action="store_true",
                        help="interrupt the agent during playback")
    args = parser.parse_args()

    transcriber = FakeTranscriber()
    clock = SimulatedClock()
    session = StreamingSession(
        transcriber,
        StreamingConfig(
            sample_rate=SAMPLE_RATE,
            vad=VADConfig(threshold_dbfs=-45.0, min_speech_ms=250,
                          min_silence_ms=500, padding_ms=200),
            emit_partials=True,
            partial_interval_ms=400,
            min_partial_audio_ms=600,
            language="hi",
        ),
        clock=clock,
    )

    audio = build_audio(args.seconds)
    block = max(1, round(args.block_ms * SAMPLE_RATE / 1000))
    records: list[dict] = []
    finals: list[dict] = []

    if not args.json:
        print(f"Feeding {len(audio) / SAMPLE_RATE:.1f}s of audio "
              f"in {args.block_ms}ms blocks ({len(audio) // block} chunks)\n")

    updates = []
    for start in range(0, len(audio), block):
        chunk = audio[start : start + block]
        clock.advance(len(chunk) / SAMPLE_RATE)   # simulate real-time capture
        updates.extend(session.push(chunk))
    updates.extend(session.flush())

    for update in updates:
        payload = update.as_dict()
        records.append(payload)
        if update.kind is UpdateKind.FINAL:
            finals.append(payload)
        if args.json:
            continue
        if update.kind is UpdateKind.STATE:
            print(f"  [{update.sequence:>2}] state -> {update.state.value}")
        elif update.kind is UpdateKind.PARTIAL:
            tail = " (tail)" if update.partial_is_tail else ""
            print(f"  [{update.sequence:>2}] partial{tail}: {update.text}")
        else:
            print(f"  [{update.sequence:>2}] FINAL ({update.endpoint_reason.value}, "
                  f"{update.audio_seconds:.2f}s, RTF {update.real_time_factor:.2f}): "
                  f"{update.text}")

    if not finals:
        print("\nNo utterance detected — try --seconds 12 or a lower threshold.")
        return 1

    # Take the last final transcript through a full agent turn.
    turn = VoiceTurn(FakeLLM(), FakeTTS())
    sink = BufferSink()

    if args.barge_in:
        class InterruptingSink(BufferSink):
            def write(self, chunk: bytes) -> None:
                super().write(chunk)
                if len(self.chunks) == 2:
                    turn.interrupt("user_started_speaking")

        sink = InterruptingSink()

    result = turn.run(
        finals[-1]["text"],
        speech_end_to_transcript_ms=finals[-1]["asr_ms"],
        sink=sink,
    )

    if args.json:
        print(json.dumps(
            {"updates": records, "turn": result.as_dict()},
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"\n--- agent turn ({result.state.value}) ---")
    print(f"  user : {result.transcript}")
    print(f"  agent: {result.response}")
    print(f"  sentences synthesised: {len(result.speech.sentences)}")
    print(f"  audio: {result.playback.bytes_written} bytes in "
          f"{result.playback.chunks_written} chunks")
    if result.state is TurnState.INTERRUPTED:
        print(f"  INTERRUPTED: {result.playback.cancel_reason}")

    print("\n--- latency ---")
    for key, value in result.metrics.as_dict().items():
        if isinstance(value, bool) or value is None:
            print(f"  {key:<42} {value}")
        else:
            print(f"  {key:<42} {value}")
    print("\nNote: all components are fakes; timings show the accounting, "
          "not real performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
