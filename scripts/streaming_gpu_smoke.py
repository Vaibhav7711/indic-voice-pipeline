"""Exercise the real Whisper runner behind the streaming state machine.

This is deliberately a file replay, not microphone capture: it makes the
first GPU integration reproducible.  It feeds real 16 kHz PCM in microphone-
sized blocks, obtains partial/final events from ``StreamingSession``, and
writes a reviewable JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class AudioClock:
    """A deterministic clock that advances with captured audio, not GPU time."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, samples: int, sample_rate: int) -> None:
        self.now += samples / sample_rate

    def __call__(self) -> float:
        return self.now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Path to a local audio file.")
    source.add_argument("--fleurs-index", type=int, help="Hindi FLEURS test index.")
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--adapter", required=True, help="Local PEFT LoRA directory.")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--block-ms", type=int, default=100)
    parser.add_argument("--partial-interval-ms", type=int, default=1000)
    parser.add_argument("--min-partial-audio-ms", type=int, default=1200)
    parser.add_argument("--output", default="results/streaming/real_gpu_smoke.json")
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[np.ndarray, int, str]:
    if args.audio:
        from asr.explicit.mel import load_audio

        waveform, _ = load_audio(args.audio)
        return waveform, 16_000, str(Path(args.audio).resolve())

    from datasets import load_dataset

    dataset = load_dataset("google/fleurs", "hi_in", split="test")
    row = dataset[args.fleurs_index]
    audio = row["audio"]
    return np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"], (
        f"google/fleurs:hi_in:test:{args.fleurs_index}"
    )


def main() -> int:
    args = parse_args()
    if args.block_ms <= 0:
        raise ValueError("--block-ms must be positive")

    from asr.explicit import ASRRunner, load_whisper
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind

    waveform, sample_rate, source = load_input(args)
    loaded = load_whisper(args.model, adapter_path=args.adapter)
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
    config = StreamingConfig(
        sample_rate=16_000,
        language=args.language,
        partial_interval_ms=args.partial_interval_ms,
        min_partial_audio_ms=args.min_partial_audio_ms,
    )
    clock = AudioClock()
    session = StreamingSession(runner, config, clock=clock)

    # ASRRunner also normalizes input, but pre-normalizing here makes the
    # controller's 16 kHz contract explicit and its timestamps trustworthy.
    from asr.explicit.mel import load_audio_from_array

    waveform, _ = load_audio_from_array(waveform, sample_rate)
    block = round(args.block_ms * config.sample_rate / 1000)
    updates = []
    for start in range(0, len(waveform), block):
        chunk = waveform[start : start + block]
        clock.advance(len(chunk), config.sample_rate)
        updates.extend(session.push(chunk))
    updates.extend(session.flush())

    records = [update.as_dict() for update in updates]
    finals = [record for record in records if record["kind"] == UpdateKind.FINAL.value]
    report = {
        "source": source,
        "model": args.model,
        "adapter": str(Path(args.adapter).resolve()),
        "audio_seconds": round(len(waveform) / config.sample_rate, 3),
        "config": config.as_dict(),
        "updates": records,
        "finals": finals,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"source: {source}")
    print(f"audio: {report['audio_seconds']:.2f}s | updates: {len(records)} | finals: {len(finals)}")
    for final in finals:
        print(f"FINAL [{final['endpoint_reason']}] {final['text']}")
        print(f"  ASR {final['asr_ms']:.1f}ms | RTF {final['real_time_factor']:.3f}")
    print(f"saved: {output}")
    return 0 if finals else 1


if __name__ == "__main__":
    raise SystemExit(main())
