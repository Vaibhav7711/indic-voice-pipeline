"""LLM bake-off for the voice turn: Hindi quality proxies and turn latency.

What is measured, per model, over a fixed set of spoken-style Hindi prompts
rendered through the same ``build_chat_prompt`` the agent uses:

Latency (the turn's critical path, through ``LLMRunner.stream()``):
  ``prefill_ms``              time to first token
  ``ms_per_token``            steady-state decode
  ``first_sentence_ms``       prefill + decode up to the first complete
                              sentence — what the user waits for before TTS
                              can start, now that the turn is pipelined
  ``first_sentence_tokens``   how much of that is the model being verbose

Quality proxies (cheap, automatic; human judging still decides):
  ``devanagari_ratio``        share of letters that are Devanagari — did it
                              answer in Hindi at all
  ``think_leak``              ``<think>`` reached the output
  ``repetition_stop``         the n-gram guard fired
  ``empty``                   nothing usable came back
  ``mean_tokens``             verbosity; spoken answers should be short

Every response is saved to ``outputs.jsonl`` so a Hindi speaker can rank
them; the numeric table alone must not pick the model.

Usage::

    python -m benchmarks.llm_bakeoff \\
        --models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,google/gemma-3-1b-it,Qwen/Qwen3-4B \\
        --out-dir results/llm_bakeoff/t4

Gated models (Gemma, Llama) need ``huggingface-cli login`` after accepting
the licence on the Hub. Models are loaded one at a time and freed between.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: Spoken-style Hindi user turns: the kind of thing the ASR hands the agent.
#: Mixed: questions, a request, code-switched, a numeric one, an instruction.
DEFAULT_PROMPTS = [
    "नमस्ते, आप कैसे हैं?",
    "मुझे दिल्ली से मुंबई जाने का सबसे तेज़ तरीका बताओ।",
    "आज मौसम कैसा रहेगा, बारिश होगी क्या?",
    "मेरा laptop बहुत slow हो गया है, क्या करूँ?",
    "एक किलो चावल में कितनी कैलोरी होती है?",
    "मुझे कल सुबह छह बजे उठना है, याद दिला देना।",
    "हिंदी में एक छोटी सी कविता सुनाओ।",
    "ब्याज दर सात प्रतिशत हो तो दस हज़ार पर एक साल में कितना ब्याज बनेगा?",
    "मुझे थोड़ा सिर दर्द है, कोई घरेलू उपाय बताओ।",
    "भारत की राजधानी क्या है और वहाँ क्या देखने लायक है?",
    "मैं English सीखना चाहता हूँ, कहाँ से शुरू करूँ?",
    "एक मिनट में समझाओ कि इंटरनेट कैसे काम करता है।",
]

DEVANAGARI = range(0x0900, 0x0980)


def devanagari_ratio(text: str) -> float | None:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    return sum(1 for ch in letters if ord(ch) in DEVANAGARI) / len(letters)


def summarize(rows: list[dict]) -> dict:
    """Aggregate one model's per-prompt rows."""
    ok = [r for r in rows if not r.get("error")]

    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    def p50(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(float(np.percentile(vals, 50)), 2) if vals else None

    ratios = [r["devanagari_ratio"] for r in ok if r.get("devanagari_ratio") is not None]
    return {
        "prompts": len(rows),
        "errors": len(rows) - len(ok),
        "prefill_ms_mean": mean("prefill_ms"),
        "ms_per_token_mean": mean("ms_per_token"),
        "first_sentence_ms_p50": p50("first_sentence_ms"),
        "first_sentence_ms_mean": mean("first_sentence_ms"),
        "first_sentence_tokens_mean": mean("first_sentence_tokens"),
        "total_ms_mean": mean("total_ms"),
        "mean_tokens": mean("generated_tokens"),
        "devanagari_ratio_mean": round(float(np.mean(ratios)), 3) if ratios else None,
        "devanagari_below_0_8": sum(1 for x in ratios if x < 0.8),
        "think_leaks": sum(1 for r in ok if r.get("think_leak")),
        "repetition_stops": sum(1 for r in ok if r.get("repetition_stop")),
        "empty": sum(1 for r in ok if r.get("empty")),
    }


def run_model(model_name: str, prompts: list[str], *, max_new_tokens: int,
              dtype: str, language: str) -> tuple[list[dict], dict]:
    import torch

    from llm import LLMRunner, build_chat_prompt, load_llm, system_prompt_for
    from tts.streaming import SentenceBuffer

    t0 = time.perf_counter()
    loaded = load_llm(model_name, dtype=getattr(torch, dtype))
    load_s = time.perf_counter() - t0
    runner = LLMRunner(loaded.model, loaded.tokenizer, loaded.device)
    params = sum(p.numel() for p in loaded.model.parameters())
    torch.cuda.reset_peak_memory_stats(loaded.device)

    # Warm-up: first call pays kernel/cache setup that is not per-turn cost.
    runner.generate(build_chat_prompt(loaded.tokenizer, system_prompt_for(language), "नमस्ते"),
                    max_new_tokens=8)

    rows = []
    for prompt_text in prompts:
        prompt = build_chat_prompt(loaded.tokenizer, system_prompt_for(language), prompt_text)
        buffer = SentenceBuffer()
        pieces: list[str] = []
        first_sentence_at = None
        first_sentence_tokens = None
        start = time.perf_counter()
        try:
            for piece in runner.stream(prompt, max_new_tokens=max_new_tokens):
                pieces.append(piece)
                if first_sentence_at is None and buffer.feed(piece):
                    first_sentence_at = (time.perf_counter() - start) * 1000
                    first_sentence_tokens = len(runner.last_metrics.decode_ms) + 1 \
                        if runner.last_metrics else None
            m = runner.last_metrics
            text = "".join(pieces).strip()
            if first_sentence_at is None and text:
                # Response ended without a sentence terminator: the whole
                # thing is the first sentence.
                first_sentence_at = m.total_ms
                first_sentence_tokens = m.generated_tokens
            rows.append({
                "prompt": prompt_text, "response": text,
                "prefill_ms": round(m.prefill_ms, 1),
                "ms_per_token": round(m.mean_decode_ms, 2) if m.decode_ms else None,
                "first_sentence_ms": round(first_sentence_at, 1) if first_sentence_at else None,
                "first_sentence_tokens": first_sentence_tokens,
                "total_ms": round(m.total_ms, 1),
                "generated_tokens": m.generated_tokens,
                "devanagari_ratio": devanagari_ratio(text),
                "think_leak": "<think>" in text,
                "repetition_stop": m.stopped_on_repetition,
                "empty": not text,
            })
        except Exception as exc:  # noqa: BLE001 - recorded per prompt
            rows.append({"prompt": prompt_text, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()})

    info = {
        "model": model_name, "dtype": dtype, "params_millions": round(params / 1e6),
        "load_seconds": round(load_s, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(loaded.device) / 2**30, 2),
    }
    del runner, loaded
    torch.cuda.empty_cache()
    return rows, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B",
                        help="Comma-separated Hub ids, evaluated in order")
    parser.add_argument("--prompts", default=None, help="JSON list file; default: built-in 12")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--language", default="hi")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    from benchmarks.asr_eval import provenance, write_json, write_jsonl

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for model_name in models:
        print(f"\n=== {model_name} ===")
        slug = model_name.replace("/", "__")
        try:
            rows, info = run_model(model_name, prompts, max_new_tokens=args.max_new_tokens,
                                   dtype=args.dtype, language=args.language)
        except Exception as exc:  # noqa: BLE001 - one model failing must not end the run
            print(f"  FAILED to run: {type(exc).__name__}: {exc}")
            summary[model_name] = {"error": f"{type(exc).__name__}: {exc}"}
            write_json(out_dir / "summary.json", {"prompts": prompts, "models": summary,
                                                  "provenance": provenance()})
            continue
        write_jsonl(out_dir / f"{slug}.outputs.jsonl", rows)
        stats = summarize(rows)
        stats.update(info)
        summary[model_name] = stats
        write_json(out_dir / "summary.json", {"prompts": prompts, "models": summary,
                                              "provenance": provenance()})
        print(f"  params {info['params_millions']}M | VRAM {info['peak_vram_gib']} GiB | "
              f"prefill {stats['prefill_ms_mean']} ms | {stats['ms_per_token_mean']} ms/tok | "
              f"first sentence p50 {stats['first_sentence_ms_p50']} ms "
              f"({stats['first_sentence_tokens_mean']} tok) | "
              f"devanagari {stats['devanagari_ratio_mean']} | "
              f"think {stats['think_leaks']} rep {stats['repetition_stops']} empty {stats['empty']}")
        for r in rows[:3]:
            print(f"    Q: {r['prompt']}\n    A: {r.get('response', r.get('error'))!s:.160}")

    print(f"\n{'model':32s} {'M':>6s} {'VRAM':>5s} {'prefill':>7s} {'ms/tok':>6s} "
          f"{'1st-sent':>8s} {'tok':>4s} {'deva':>5s} {'think':>5s} {'rep':>3s}")
    for name, st in summary.items():
        if "error" in st:
            print(f"{name:32s} ERROR {st['error'][:60]}")
            continue
        print(f"{name:32s} {st['params_millions']:6d} {st['peak_vram_gib']:5.1f} "
              f"{st['prefill_ms_mean']:7.0f} {st['ms_per_token_mean']:6.1f} "
              f"{st['first_sentence_ms_p50']:8.0f} {st['first_sentence_tokens_mean']:4.0f} "
              f"{st['devanagari_ratio_mean']:5.2f} {st['think_leaks']:5d} {st['repetition_stops']:3d}")
    print(f"\nwritten: {out_dir}  (read the *.outputs.jsonl before choosing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
