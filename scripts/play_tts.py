"""Speak a sentence through the real audio path: synthesizer → decode → device.

    python scripts/play_tts.py "नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?"
    python scripts/play_tts.py --backend mms --interrupt-after 0.6 "..."

Prints first-audio latency and, with ``--interrupt-after``, cancels playback
from another thread to show the sink actually goes quiet (``abort``), not
just stops being fed.
"""

from __future__ import annotations

import argparse
import json
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="नमस्ते, मैं आपकी क्या मदद कर सकती हूँ? आज का दिन अच्छा है।")
    parser.add_argument("--backend", default="edge", choices=["edge", "mms"])
    parser.add_argument("--language", default="hi")
    parser.add_argument("--device", default=None, help="sounddevice output device")
    parser.add_argument("--interrupt-after", type=float, default=None,
                        help="seconds after first audio to barge in")
    parser.add_argument("--no-split", action="store_true")
    args = parser.parse_args()

    from agent.audio import SoundDeviceSink
    from agent.playback import PlaybackSession
    from tts.streaming import SpeechStream, iter_synthesis

    if args.backend == "edge":
        from tts import EdgeStreamingSynthesizer

        synth = EdgeStreamingSynthesizer(language=args.language)
    else:
        from tts.local import MmsTtsSynthesizer

        synth = MmsTtsSynthesizer(language=args.language)

    warm = getattr(synth, "warm_up", None)
    if callable(warm):
        print(f"warm-up: {warm():.0f} ms (model load excluded from the numbers below)")

    sink = SoundDeviceSink(synth.format, device=args.device)
    sink.open()
    playback = PlaybackSession(sink, playback_id="play_tts")
    speech = SpeechStream()

    def produce():
        for chunk in iter_synthesis(
            synth, args.text, speech, split=not args.no_split, should_stop=playback.should_stop,
        ):
            yield chunk.data

    if args.interrupt_after is not None:
        def interrupter():
            while playback.state.value not in ("playing", "completed", "failed") \
                    and not playback.cancelled:
                time.sleep(0.01)
            time.sleep(args.interrupt_after)
            playback.cancel("barge_in_demo")

        threading.Thread(target=interrupter, daemon=True).start()

    t0 = time.perf_counter()
    result = playback.play(produce())
    wall = (time.perf_counter() - t0) * 1000
    print(json.dumps({
        "backend": args.backend, "format": synth.format.as_dict(),
        "sentences": len(speech.sentences), "chunks": len(speech.chunks),
        "tts_first_chunk_ms": speech.first_chunk_ms,
        "playback_first_audio_ms": result.first_audio_ms,
        "state": result.state.value, "interrupted": result.interrupted,
        "samples_played": sink.samples_written,
        "seconds_played": round(sink.samples_written / synth.format.sample_rate, 2),
        "wall_ms": round(wall, 1), "error": result.error or speech.error,
    }, ensure_ascii=False, indent=2))
    return 0 if result.state.value in ("completed", "cancelled") else 1


if __name__ == "__main__":
    raise SystemExit(main())
