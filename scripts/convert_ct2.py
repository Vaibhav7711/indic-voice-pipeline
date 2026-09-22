"""Merge a LoRA adapter into Whisper and convert it to CTranslate2.

    python scripts/convert_ct2.py --model openai/whisper-medium \\
        --adapter Hugme6969/whisper-medium-hindi-lora \\
        --out models/ct2/whisper-medium-hindi-lora --quantization int8_float16

Two steps, both deterministic:

1. Load the base model, merge the adapter (``merge_and_unload``), save the
   merged HF checkpoint with its processor to ``<out>-hf-merged`` (a sibling).
2. ``ct2-transformers-converter`` on that directory → ``<out>``.

``int8_float16`` is the T4 sweet spot (int8 weights, fp16 compute);
``int8`` for CPU; ``float16`` keeps weights exact for a like-for-like
comparison with the explicit runner.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def merge_and_save(model_name: str, adapter: str | None, out: Path) -> Path:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from benchmarks.checkpoints import resolve_hub_adapter

    # Sibling of the output, not inside it: the converter's --force wipes
    # the output directory before writing.
    merged = out.parent / f"{out.name}-hf-merged"
    if (merged / "config.json").is_file():
        print(f"merged checkpoint exists: {merged}")
        return merged
    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=torch.float32)
    processor_source = model_name
    if adapter:
        from peft import PeftModel

        adapter_dir = resolve_hub_adapter(adapter)
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        if (Path(adapter_dir) / "tokenizer_config.json").is_file():
            processor_source = str(adapter_dir)
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True)
    processor = WhisperProcessor.from_pretrained(processor_source)
    processor.save_pretrained(merged)
    # transformers >= 5 writes processor_config.json; the CTranslate2 converter
    # reads preprocessor_config.json. Write the feature extractor under both.
    processor.feature_extractor.save_pretrained(merged)
    if not (merged / "preprocessor_config.json").is_file():
        import json

        (merged / "preprocessor_config.json").write_text(
            json.dumps(processor.feature_extractor.to_dict(), indent=2)
        )
    print(f"saved merged model to {merged}")
    return merged


def convert(merged: Path, out: Path, quantization: str) -> None:
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(merged), "--output_dir", str(out),
        "--quantization", quantization, "--force",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    # faster-whisper wants the tokenizer and feature-extractor config beside model.bin.
    for name in ("tokenizer.json", "preprocessor_config.json", "tokenizer_config.json"):
        if (merged / name).is_file():
            shutil.copy(merged / name, out / name)
    print(f"CTranslate2 model at {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quantization", default="int8_float16",
                        choices=["float32", "float16", "int8", "int8_float16", "int8_float32"])
    parser.add_argument("--keep-merged", action="store_true",
                        help="Keep <out>-hf-merged (several GB) after conversion")
    args = parser.parse_args()

    out = Path(args.out)
    merged = merge_and_save(args.model, args.adapter, out)
    convert(merged, out, args.quantization)
    if not args.keep_merged:
        shutil.rmtree(merged, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
