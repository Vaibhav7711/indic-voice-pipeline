"""Quick pipeline demo — load models, run on FLEURS Hindi sample, print waterfall."""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", default="openai/whisper-medium")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", default=None, help="Path to a PEFT LoRA adapter")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB\n")

    # Load models.
    from asr.explicit import load_whisper
    from llm import load_llm
    from pipeline import VoicePipeline

    print(f"Loading {args.whisper_model}...")
    whisper = load_whisper(args.whisper_model, adapter_path=args.adapter)

    print(f"Loading {args.llm_model}...")
    llm = load_llm(args.llm_model)

    pipe = VoicePipeline(whisper, llm)
    print(f"Memory strategy: {pipe.strategy.value}")
    print(f"VRAM used: {torch.cuda.memory_allocated() / (1024**3):.2f} GiB\n")

    # Get a Hindi audio sample.
    print("Downloading FLEURS Hindi sample...")
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", "hi_in", split="test", streaming=True)
    sample = next(iter(ds))
    audio = sample["audio"]["array"].astype(np.float32)
    sr = sample["audio"]["sampling_rate"]
    ref = sample["transcription"]
    sf.write("demo_hindi.wav", audio, sr)

    print(f"Reference: {ref}")
    print(f"Duration: {len(audio)/sr:.1f}s\n")

    # Run pipeline.
    print("Running pipeline...")
    result = pipe.run("demo_hindi.wav", language="hi", llm_max_tokens=64)

    # Print results.
    print(f"\n{'='*60}")
    print(f"Transcript: {result.transcript}")
    print(f"Answer:     {result.answer[:200]}")
    print(f"{'='*60}\n")

    # Waterfall.
    m = result.metrics
    print("PIPELINE LATENCY WATERFALL")
    print("-" * 50)
    print(f"  Mel extraction:         {m.asr.mel_extraction_ms:8.1f} ms")
    print(f"  Whisper encoder:        {m.asr.encoder_ms:8.1f} ms")
    print(f"  ASR decoder prefill:    {m.asr.decoder_prefill_ms:8.1f} ms")
    print(f"  ASR decode ({m.asr.decoder_steps:3d} tok):   {m.asr.total_decode_ms:8.1f} ms")
    print(f"  Model swap:             {m.model_swap_ms:8.1f} ms")
    print(f"  LLM prefill:            {m.llm_prefill_ms:8.1f} ms")
    decode_total = sum(m.llm_decode_ms)
    print(f"  LLM decode ({m.llm_tokens_generated:3d} tok):   {decode_total:8.1f} ms")
    print("-" * 50)
    print(f"  TOTAL PIPELINE:         {m.total_pipeline_ms:8.0f} ms")
    print(f"  Audio → 1st LLM tok:    {m.audio_to_first_llm_token_ms:8.0f} ms")
    rtf = m.asr.real_time_factor
    print(f"  ASR RTF:                {'undefined' if rtf is None else f'{rtf:8.3f}'}")
    print(f"  Peak VRAM:              {m.peak_allocated_bytes / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
