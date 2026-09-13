"""WER evaluation on FLEURS Hindi test set.

Usage:
    python -m benchmarks.asr_wer --language hi --samples 50 --output results/wer_baseline.json
    python -m benchmarks.asr_wer --language hi --checkpoint results/lora-hi/best --output results/wer_finetuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import evaluate
    from datasets import load_dataset

    # Load model.
    if args.checkpoint:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel

        processor = WhisperProcessor.from_pretrained(args.checkpoint)
        base = WhisperForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base, args.checkpoint)
        model = model.merge_and_unload().to("cuda").eval()
        print(f"Loaded fine-tuned model from {args.checkpoint}")
    else:
        from asr.explicit import load_whisper
        loaded = load_whisper(args.model)
        model, processor = loaded.model, loaded.processor
        print(f"Loaded base model: {args.model}")

    from asr.explicit.runner import ASRRunner
    runner = ASRRunner(model, processor, torch.device("cuda"), torch.float16)

    # Load dataset.
    lang_code = f"{args.language}_in"
    ds = load_dataset("google/fleurs", lang_code, split="test")
    n = min(args.samples, len(ds))
    print(f"Evaluating on {n} FLEURS {args.language} samples...")

    predictions, references = [], []
    for i in range(n):
        audio = ds[i]["audio"]["array"].astype(np.float32)
        sr = ds[i]["audio"]["sampling_rate"]
        result = runner.transcribe_array(audio, sr, language=args.language)
        predictions.append(result.text)
        references.append(ds[i]["transcription"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)

    results = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "language": args.language,
        "samples": n,
        "wer_percent": round(wer * 100, 2),
    }

    print(f"\nWER: {wer*100:.2f}%")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
