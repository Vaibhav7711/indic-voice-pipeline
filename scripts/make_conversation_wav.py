"""Build a scripted conversation as one WAV, to drive the live agent headlessly.

`scripts/live_agent.py --input-wav` replays a file as the microphone at
real-time pacing, which is the full duplex path -- streaming VAD, endpointing,
turn pipelining, barge-in -- with the only difference being where the samples
come from. What it lacked was audio worth replaying: a single clip exercises one
turn and nothing else.

This writes a timeline. Each utterance is synthesised, then a gap of chosen
length follows, so the two behaviours that have never been tested become
reproducible rather than a matter of luck:

**Follow-up questions.** A gap longer than the endpointer's silence window
(default 700 ms padding) closes the first utterance, the agent answers, and the
second utterance arrives as a separate turn carrying dialogue history. Ask
something that only resolves against the previous turn and the transcript says
whether memory worked.

**Barge-in.** A gap *shorter* than the agent's reply takes to speak means the
second utterance starts while the agent is still talking, which is what
barge-in is. This needs `--sink paced`: a sink that consumes instantly leaves
nothing to interrupt, which is why this could previously only be tested on a
machine with a sound card.

    python scripts/make_conversation_wav.py --preset followup --out conv.wav
    python scripts/make_conversation_wav.py --preset bargein  --out barge.wav
    python scripts/live_agent.py --input-wav conv.wav --sink paced \
        --max-turns 2 --log results/live/followup.jsonl

**The input is synthetic speech, and that bounds what this measures.** TTS
output is cleaner than a microphone: no room, no clipping, no breath. These runs
test the *plumbing* -- does a second turn happen, does history carry, does
barge-in fire and cancel -- not recognition accuracy, which is what
`benchmarks/streaming_eval.py` measures on real recorded speech. A clean WER
here would not mean the ASR is good.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

#: Utterance texts and the silence that follows each, in seconds.
#:
#: The gaps are the whole design. `followup` leaves time for the agent to
#: finish; `bargein` deliberately does not.
PRESETS: dict[str, list[tuple[str, float]]] = {
    # Two turns, the second resolvable only against the first.
    "followup": [
        ("भारत की राजधानी क्या है?", 6.0),
        ("वहाँ की आबादी कितनी है?", 4.0),
    ],
    # The second utterance lands ~1.2 s in, while the reply is still playing.
    "bargein": [
        ("मुझे भारत के बारे में विस्तार से बताओ।", 1.2),
        ("रुको, बस इतना ही।", 4.0),
    ],
    # Three turns with pronoun chains, for dialogue memory across a window.
    "memory": [
        ("मेरा नाम वैभव है।", 5.0),
        ("मेरा नाम क्या है?", 5.0),
        ("मैंने पहले क्या पूछा था?", 4.0),
    ],
    # One turn, as a control: if this fails the problem is not conversational.
    "single": [
        ("नमस्ते, आप कैसे हैं?", 4.0),
    ],
}


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def to_mono_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """Resample by linear interpolation, which is enough for a test signal.

    Not a resampler worth using on evaluation audio; the point here is that the
    VAD and the mel front end see the sample rate they expect.
    """
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if rate == SAMPLE_RATE:
        return pcm.astype(np.float32)
    duration = pcm.size / rate
    target = np.linspace(0.0, duration, int(duration * SAMPLE_RATE),
                         endpoint=False)
    source = np.linspace(0.0, duration, pcm.size, endpoint=False)
    return np.interp(target, source, pcm).astype(np.float32)


def build(timeline: list[tuple[str, float]], synthesise, *,
          lead_in: float = 0.5) -> tuple[np.ndarray, list[dict]]:
    """Concatenate synthesised utterances and gaps into one waveform.

    Returns the audio and a manifest of where each utterance starts and ends,
    so a test can say which turn a transcript should correspond to instead of
    guessing from order alone.
    """
    parts = [silence(lead_in)]
    manifest = []
    cursor = lead_in
    for text, gap in timeline:
        audio = synthesise(text)
        if audio.size == 0:
            raise ValueError(f"synthesis returned no audio for {text!r}")
        manifest.append({
            "text": text,
            "starts_at": round(cursor, 3),
            "ends_at": round(cursor + audio.size / SAMPLE_RATE, 3),
            "gap_after": gap,
        })
        cursor += audio.size / SAMPLE_RATE + gap
        parts.extend([audio, silence(gap)])
    return np.concatenate(parts), manifest


def write_wav(path: Path, audio: np.ndarray) -> None:
    """16-bit mono at 16 kHz, which is what the pipeline's front end expects."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        # Normalise to -3 dBFS: TTS levels vary between voices, and the VAD's
        # adaptive noise floor is measured in dBFS.
        audio = audio / peak * 0.707
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((audio * 32767).astype(np.int16).tobytes())


def edge_synthesiser(language: str = "hi"):
    """Synthesise with the project's own TTS, decoded to 16 kHz mono float."""
    from agent.audio import MP3_24K, DecodingBufferSink
    from tts import EdgeStreamingSynthesizer

    synth = EdgeStreamingSynthesizer(language=language)

    def synthesise(text: str) -> np.ndarray:
        sink = DecodingBufferSink(MP3_24K)
        for chunk in synth.stream(text):
            sink.write(chunk)
        sink.close()
        return to_mono_16k(sink.audio.astype(np.float32) / 32768.0,
                           sink.target_rate)

    return synthesise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", default="followup", choices=sorted(PRESETS),
                        help="which conversation timeline to build")
    parser.add_argument("--out", default="results/live/conversation.wav")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--lead-in", type=float, default=0.5,
                        help="silence before the first utterance, so the VAD has "
                             "noise floor to measure before speech starts")
    parser.add_argument("--say", action="append", default=[], dest="says",
                        metavar="TEXT:GAP",
                        help="override the preset: repeatable 'text:gap_seconds'")
    args = parser.parse_args(argv)

    if args.says:
        timeline = []
        for item in args.says:
            text, _, gap = item.rpartition(":")
            if not text:
                parser.error(f"--say {item!r} needs the form 'text:gap_seconds'")
            try:
                timeline.append((text, float(gap)))
            except ValueError:
                parser.error(f"--say {item!r}: {gap!r} is not a number of seconds")
    else:
        timeline = PRESETS[args.preset]

    print(f"synthesising {len(timeline)} utterance(s)…", flush=True)
    audio, manifest = build(timeline, edge_synthesiser(args.language),
                            lead_in=args.lead_in)
    out = Path(args.out)
    write_wav(out, audio)

    print(f"wrote {out} ({audio.size / SAMPLE_RATE:.1f} s)")
    for entry in manifest:
        print(f"  {entry['starts_at']:6.2f}-{entry['ends_at']:6.2f} s  "
              f"gap {entry['gap_after']:.1f} s  {entry['text']}")
    manifest_path = out.with_suffix(".manifest.json")
    import json

    manifest_path.write_text(
        json.dumps({"preset": args.preset if not args.says else "custom",
                    "sample_rate": SAMPLE_RATE, "lead_in": args.lead_in,
                    "utterances": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print("\nNext:  python scripts/live_agent.py --input-wav "
          f"{out} --sink paced --max-turns {len(timeline)}")
    if args.preset == "bargein" or args.says:
        print("A barge-in run needs --sink paced. An instant sink finishes the "
              "turn in milliseconds and there is nothing left to interrupt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
