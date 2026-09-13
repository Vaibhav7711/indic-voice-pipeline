"""Full pipeline waterfall benchmark.

Usage:
    python -m benchmarks.pipeline_e2e --audio test.wav --language hi --runs 3 --output results/pipeline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--llm-max-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from asr.explicit import load_whisper
    from llm import load_llm
    from pipeline import VoicePipeline

    whisper = load_whisper(args.whisper_model)
    llm = load_llm(args.llm_model)
    pipe = VoicePipeline(whisper, llm)
    print(f"Strategy: {pipe.strategy.value}")

    for _ in range(args.warmup_runs):
        pipe.run(args.audio, language=args.language, llm_max_tokens=args.llm_max_tokens)

    all_results = []
    for i in range(args.runs):
        r = pipe.run(args.audio, language=args.language, llm_max_tokens=args.llm_max_tokens)
        all_results.append(r)
        print(f"  Run {i+1}: {r.metrics.total_pipeline_ms:.0f} ms")

    def avg(fn):
        return float(np.mean([fn(r.metrics) for r in all_results]))

    summary = {
        "whisper_model": args.whisper_model,
        "llm_model": args.llm_model,
        "language": args.language,
        "runs": args.runs,
        "sample_transcript": all_results[0].transcript,
        "waterfall_ms": {
            "mel": avg(lambda m: m.asr.mel_extraction_ms),
            "encoder": avg(lambda m: m.asr.encoder_ms),
            "asr_decode": avg(lambda m: m.asr.total_decode_ms),
            "model_swap": avg(lambda m: m.model_swap_ms),
            "llm_prefill": avg(lambda m: m.llm_prefill_ms),
            "llm_decode": avg(lambda m: sum(m.llm_decode_ms)),
        },
        "totals": {
            "pipeline_ms": avg(lambda m: m.total_pipeline_ms),
            "audio_to_first_llm_token_ms": avg(lambda m: m.audio_to_first_llm_token_ms),
            "asr_rtf": avg(lambda m: m.asr.real_time_factor),
        },
        "memory": {
            "strategy": pipe.strategy.value,
            "peak_gib": max(r.metrics.peak_allocated_bytes for r in all_results) / (1024**3),
        },
        "gpu": torch.cuda.get_device_name(0),
    }

    print("\n" + json.dumps(summary, indent=2, default=str))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
