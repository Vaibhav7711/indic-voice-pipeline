"""TTS backend bake-off: first-audio latency, throughput, and clips to listen to.

For each backend and each sentence: ``first_chunk_ms`` (what the turn's
``first_llm_token_to_playback_start_ms`` inherits), ``total_ms``, audio
seconds, and real-time factor. Every clip is written as WAV so a Hindi
speaker can rank voice quality — the numbers rank latency only.

    python -m benchmarks.tts_bakeoff --backends edge,mms --out-dir results/tts_bakeoff/t4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SENTENCES = [
    "नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?",
    "दिल्ली से मुंबई जाने का सबसे तेज़ तरीका हवाई जहाज़ है, जिसमें लगभग दो घंटे लगते हैं।",
    "आपका laptop slow होने की सबसे आम वजह पुराना software और भरी हुई disk होती है।",
    "एक किलो चावल में लगभग तीन हज़ार छह सौ कैलोरी होती है।",
    "ठीक है, मैंने कल सुबह छह बजे का alarm लगा दिया है।",
]


def make_backend(name: str, language: str):
    if name == "edge":
        from tts import EdgeStreamingSynthesizer

        return EdgeStreamingSynthesizer(language=language)
    if name == "mms":
        from tts.local import MmsTtsSynthesizer

        return MmsTtsSynthesizer(language)
    raise SystemExit(f"unknown backend {name!r}")


def run_backend(name: str, synth, sentences: list[str], out_dir: Path) -> dict:
    import soundfile as sf

    from agent.audio import DecodingBufferSink

    warm = getattr(synth, "warm_up", None)
    warm_ms = warm() if callable(warm) else None

    rows = []
    for i, sentence in enumerate(sentences):
        sink = DecodingBufferSink(synth.format)
        t0 = time.perf_counter()
        first = None
        for chunk in synth.stream(sentence):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            sink.write(chunk)
        sink.close()
        total = (time.perf_counter() - t0) * 1000
        seconds = sink.seconds
        path = out_dir / f"{name}_{i}.wav"
        sf.write(path, sink.audio, synth.format.sample_rate)
        rows.append({"sentence": sentence, "first_chunk_ms": round(first or total, 1),
                     "total_ms": round(total, 1), "audio_seconds": round(seconds, 2),
                     "rtf": round(total / 1000 / seconds, 3) if seconds else None,
                     "wav": str(path)})
        print(f"  [{name}] first {rows[-1]['first_chunk_ms']:6.0f} ms | total {total:6.0f} ms | "
              f"{seconds:4.1f} s audio | RTF {rows[-1]['rtf']}")
    return {
        "backend": name, "format": synth.format.as_dict(), "streaming": bool(synth.streaming),
        "warm_up_ms": warm_ms,
        "first_chunk_ms_p50": float(np.median([r["first_chunk_ms"] for r in rows])),
        "total_ms_mean": float(np.mean([r["total_ms"] for r in rows])),
        "rtf_mean": float(np.mean([r["rtf"] for r in rows if r["rtf"]])),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="edge,mms")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    from benchmarks.asr_eval import provenance, write_json

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"\n=== {name} ===")
        try:
            results[name] = run_backend(name, make_backend(name, args.language), SENTENCES, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        write_json(out_dir / "summary.json", {"backends": results, "provenance": provenance()})

    print(f"\n{'backend':8s} {'stream':>6s} {'first p50':>9s} {'total':>7s} {'RTF':>6s}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:8s} ERROR {r['error'][:60]}")
        else:
            print(f"{name:8s} {str(r['streaming']):>6s} {r['first_chunk_ms_p50']:9.0f} "
                  f"{r['total_ms_mean']:7.0f} {r['rtf_mean']:6.2f}")
    print(f"\nclips: {out_dir}/*.wav — listen before choosing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
