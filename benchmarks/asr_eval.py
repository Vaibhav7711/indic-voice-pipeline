"""Reproducible ASR evaluation harness.

Two subcommands, deliberately split:

``run``
    Inference. Needs a GPU, the explicit ``ASRRunner``, and the dataset. Writes
    ``predictions.jsonl`` plus per-example latency, then scores.
``score``
    Pure Python. Re-scores an existing ``predictions.jsonl`` with different
    normalization or analysis settings. No torch, no GPU, milliseconds.

The split matters. Inference over FLEURS test is minutes of GPU time; deciding
whether to report WER at ``standard`` or ``aggressive`` normalization should
not cost another run. This is the same discipline as caching model outputs
before iterating on an LLM eval rubric — separate the expensive, deterministic
step from the cheap, opinionated one.

What gets written to ``--out-dir``
----------------------------------
``run_config.json``   Everything needed to reproduce: git SHA + dirty flag,
                      model, adapter, dataset, split, seed, the exact selected
                      indices, normalization level, package versions.
``predictions.jsonl`` One row per utterance: reference, hypothesis, duration,
                      latency breakdown, audio reference.
``metrics.json``      WER and CER at every normalization level, micro and
                      macro, error-category breakdown, latency percentiles,
                      and per-category results for a hard set.
``errors.jsonl``      Per-example categorized edit operations.

Examples
--------
    # Final test-set number for the fine-tuned adapter.
    python -m benchmarks.asr_eval run \
        --model openai/whisper-medium \
        --adapter results/whisper-lora-hi-full/best \
        --split test --limit 300 --seed 0 \
        --out-dir results/eval/medium-lora-test

    # Same audio, base model, for a like-for-like comparison.
    python -m benchmarks.asr_eval run \
        --model openai/whisper-medium \
        --split test --limit 300 --seed 0 \
        --out-dir results/eval/medium-base-test

    # Re-score without touching the GPU.
    python -m benchmarks.asr_eval score \
        --predictions results/eval/medium-lora-test/predictions.jsonl \
        --level aggressive \
        --out-dir results/eval/medium-lora-test-aggressive
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.error_analysis import (
    AnalysisConfig,
    aggregate,
    analyze_example,
    build_reference_vocabulary,
)
from benchmarks.metrics import CorpusScore, score_text
from text.normalize import NormalizationLevel

__all__ = ["score_predictions", "main"]

ALL_LEVELS = [
    NormalizationLevel.NONE,
    NormalizationLevel.BASIC,
    NormalizationLevel.STANDARD,
    NormalizationLevel.AGGRESSIVE,
]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "datasets", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def provenance() -> dict:
    """Capture enough to reproduce a number six months from now."""
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv,
        "packages": _package_versions(),
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile: ``rank = ceil(p/100 * N)``.

    ``ceil``, not ``round`` — Python's ``round`` is banker's rounding, so
    ``round(2.5) == 2`` and the median of a 5-element list would come back as
    the 2nd element instead of the 3rd. No numpy dependency here, because the
    scoring path must stay importable without the model stack.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _corpus_rates(rows: list[dict], level, unit: str) -> dict:
    score = CorpusScore()
    for row in rows:
        score.update(
            score_text(
                row.get("reference", ""),
                row.get("hypothesis", ""),
                level=level,
                unit=unit,
            )
        )
    return score.as_dict()


def _latency_summary(rows: list[dict]) -> dict:
    totals = [r["latency"]["total_ms"] for r in rows if r.get("latency")]
    rtfs = [r["latency"]["real_time_factor"] for r in rows if r.get("latency")]
    if not totals:
        return {}

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None

    def stage(key):
        vals = [r["latency"].get(key) for r in rows if r.get("latency")]
        vals = [v for v in vals if v is not None]
        return mean(vals)

    return {
        "utterances": len(totals),
        "total_ms_mean": mean(totals),
        "total_ms_p50": round(_percentile(totals, 50), 3),
        "total_ms_p90": round(_percentile(totals, 90), 3),
        "total_ms_p95": round(_percentile(totals, 95), 3),
        "real_time_factor_mean": mean(rtfs),
        "real_time_factor_p90": round(_percentile(rtfs, 90), 4),
        "mel_extraction_ms_mean": stage("mel_extraction_ms"),
        "encoder_ms_mean": stage("encoder_ms"),
        "decoder_prefill_ms_mean": stage("decoder_prefill_ms"),
        "mean_decode_ms_mean": stage("mean_decode_ms"),
        "decoder_steps_mean": stage("decoder_steps"),
    }


def _category_breakdown(rows: list[dict], level) -> dict:
    """Per-category WER/CER for a hard set. Items may carry many categories."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        for category in row.get("categories") or []:
            buckets.setdefault(category, []).append(row)

    out = {}
    for category, subset in sorted(buckets.items()):
        word = _corpus_rates(subset, level, "word")
        char = _corpus_rates(subset, level, "char")
        out[category] = {
            "items": len(subset),
            "wer_percent": word["micro_percent"],
            "cer_percent": char["micro_percent"],
        }
    return out


def score_predictions(
    rows: list[dict],
    *,
    level: NormalizationLevel | str = NormalizationLevel.STANDARD,
    analysis_config: AnalysisConfig | None = None,
    top_confusions: int = 10,
    worst_examples: int = 15,
) -> tuple[dict, list[dict]]:
    """Score saved predictions. Returns ``(metrics, per_example_rows)``.

    Pure Python — this is what makes ``score`` runnable anywhere.
    """
    level = NormalizationLevel(level)
    analysis_config = analysis_config or AnalysisConfig(level=level)

    # Primary metric at the reporting level, plus every other level so the
    # cost of normalization is visible instead of hidden.
    by_level = {}
    for candidate in ALL_LEVELS:
        by_level[candidate.value] = {
            "wer": _corpus_rates(rows, candidate, "word"),
            "cer": _corpus_rates(rows, candidate, "char"),
        }

    primary_wer = by_level[level.value]["wer"]["micro_percent"]
    raw_wer = by_level[NormalizationLevel.NONE.value]["wer"]["micro_percent"]
    aggressive_wer = by_level[NormalizationLevel.AGGRESSIVE.value]["wer"][
        "micro_percent"
    ]

    vocabulary = build_reference_vocabulary(
        [r.get("reference", "") for r in rows], level=level
    )
    analyses = [
        analyze_example(
            str(row.get("id", index)),
            row.get("reference", ""),
            row.get("hypothesis", ""),
            vocabulary,
            analysis_config,
        )
        for index, row in enumerate(rows)
    ]

    metrics = {
        "reporting_level": level.value,
        "examples": len(rows),
        "headline": {
            "wer_percent": primary_wer,
            "cer_percent": by_level[level.value]["cer"]["micro_percent"],
        },
        "normalization_sensitivity": {
            "raw_wer_percent": raw_wer,
            "reported_wer_percent": primary_wer,
            "orthography_blind_wer_percent": aggressive_wer,
            # How much WER is pure formatting, and how much is spelling
            # convention. Both are reported so neither can be claimed as a
            # modelling win.
            "formatting_delta_points": (
                None
                if raw_wer is None or primary_wer is None
                else round(raw_wer - primary_wer, 4)
            ),
            "orthography_delta_points": (
                None
                if aggressive_wer is None or primary_wer is None
                else round(primary_wer - aggressive_wer, 4)
            ),
        },
        "by_normalization_level": by_level,
        "error_analysis": aggregate(
            analyses, top_confusions=top_confusions, worst_examples=worst_examples
        ),
        "latency": _latency_summary(rows),
    }

    category_metrics = _category_breakdown(rows, level)
    if category_metrics:
        metrics["by_category"] = category_metrics

    per_example = [a.as_dict() for a in analyses]
    return metrics, per_example


# --------------------------------------------------------------------------
# Dataset loading (run mode only)
# --------------------------------------------------------------------------


def _select_indices(total: int, limit: int | None, strategy: str, seed: int) -> list[int]:
    """Choose which examples to evaluate.

    Default is a seeded random sample, not the first N. FLEURS is ordered, so
    ``[:50]`` is a biased slice — it can share speakers and topics, and the
    resulting WER is not an estimate of test-set performance. The chosen
    indices are written to ``run_config.json`` so the subset is reproducible.
    """
    if limit is None or limit >= total:
        return list(range(total))
    if strategy == "first":
        return list(range(limit))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), limit))


def _iter_fleurs(dataset, config: str, split: str, indices: list[int]):
    """Yield selected FLEURS rows from an already-loaded dataset.

    Takes the dataset object rather than a name so the split is loaded exactly
    once. Loading it twice is not a re-download thanks to the local cache, but
    on Colab it still re-scans and re-prepares the split, which is minutes of
    wall clock for no benefit.
    """
    from benchmarks.fleurs import extract_audio

    for index in indices:
        row = dataset[index]
        waveform, sample_rate = extract_audio(row)
        yield {
            "id": f"fleurs-{config}-{split}-{index}",
            "reference": row["transcription"],
            "audio_array": waveform,
            "sampling_rate": sample_rate,
            "categories": [],
            "audio_ref": {
                "type": "hf",
                "dataset": "google/fleurs",
                "config": config,
                "split": split,
                "index": index,
            },
        }


def _iter_hard_set(manifest: str, root: str):
    import numpy as np

    from benchmarks.hard_set import curated_items, load_manifest, validate_items

    items = load_manifest(manifest)
    problems = validate_items(items, root=root)
    if problems:
        raise SystemExit(
            "Hard set manifest is invalid:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    eligible = curated_items(items)
    if not eligible:
        raise SystemExit(
            f"No curated items in {manifest}. "
            "Candidate items are excluded from reporting until reviewed."
        )

    for item in eligible:
        audio = item.audio
        if audio.get("type") == "local":
            import soundfile as sf

            waveform, sample_rate = sf.read(str(Path(root) / audio["path"]))
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
        else:
            from datasets import load_dataset

            dataset = load_dataset(
                audio["dataset"], audio.get("config"), split=audio["split"]
            )
            row = dataset[int(audio["index"])]
            waveform = np.asarray(row["audio"]["array"], dtype=np.float32)
            sample_rate = row["audio"]["sampling_rate"]

        yield {
            "id": item.id,
            "reference": item.transcript,
            "audio_array": waveform,
            "sampling_rate": sample_rate,
            "categories": item.categories,
            "audio_ref": audio,
        }


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from asr.explicit import load_whisper
    from asr.explicit.runner import ASRRunner

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the example list first so run_config records exactly what was run.
    if args.hard_set:
        examples = list(_iter_hard_set(args.hard_set, args.root))
        selection = {"source": "hard_set", "manifest": args.hard_set}
    else:
        from benchmarks.fleurs import load_fleurs

        config = f"{args.language}_in"
        dataset = load_fleurs(config, args.split)
        indices = _select_indices(len(dataset), args.limit, args.sample, args.seed)
        examples = list(_iter_fleurs(dataset, config, args.split, indices))
        selection = {
            "source": "fleurs",
            "config": config,
            "split": args.split,
            "strategy": args.sample,
            "seed": args.seed,
            "limit": args.limit,
            "available": len(dataset),
            "indices": indices,
        }

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Resolve the adapter by policy where asked, then verify it before the
    # model loads. A hardcoded checkpoint number goes stale the moment
    # save_total_limit rotates it away, and an adapter read while the Trainer
    # is mid-save surfaces as a confusing deserialisation error deep inside
    # safetensors. Both fail here instead, with an actionable message.
    adapter = args.adapter
    if args.adapter_dir:
        from benchmarks.checkpoints import resolve_adapter

        adapter = str(resolve_adapter(args.adapter_dir, args.adapter_policy))
        print(f"Resolved adapter ({args.adapter_policy}): {adapter}")

    adapter_info = None
    if adapter:
        from benchmarks.checkpoints import stage_checkpoint, verify_adapter

        if args.stage_adapter:
            adapter = str(stage_checkpoint(adapter, args.stage_adapter))
            print(f"Staged adapter to local disk: {adapter}")
        adapter_info = verify_adapter(adapter)

    loaded = load_whisper(args.model, adapter_path=adapter, dtype=dtype)
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    print(
        f"Model: {args.model}"
        + (f" + adapter {adapter}" if adapter else " (base, no adapter)")
    )
    print(f"Device: {device_name}   dtype: {args.dtype}")

    # Warmup. The first GPU forward pass pays for cuDNN autotuning, kernel
    # loading and allocator growth, and on a T4 it can be several times slower
    # than steady state. Without this the first utterance poisons the mean RTF
    # and, on a small run, the p90 as well. Results are discarded.
    if examples and args.warmup > 0:
        print(f"Warmup: {args.warmup} pass(es), discarded...")
        warm = np.asarray(examples[0]["audio_array"], dtype=np.float32)
        for _ in range(args.warmup):
            runner.transcribe_array(
                warm,
                examples[0]["sampling_rate"],
                language=args.language,
                max_new_tokens=args.max_new_tokens,
            )
        torch.cuda.synchronize()

    print(f"Evaluating {len(examples)} example(s)...")

    rows = []
    for position, example in enumerate(examples, 1):
        waveform = np.asarray(example["audio_array"], dtype=np.float32)
        result = runner.transcribe_array(
            waveform,
            example["sampling_rate"],
            language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
        latency = result.metrics.as_dict()
        latency.pop("decode_ms", None)  # per-step list is large and not needed
        rows.append(
            {
                "id": example["id"],
                "reference": example["reference"],
                "hypothesis": result.text,
                "audio_duration_s": round(result.metrics.audio_duration_seconds, 3),
                "categories": example["categories"],
                "audio_ref": example["audio_ref"],
                "latency": latency,
            }
        )
        if position % 25 == 0 or position == len(examples):
            print(f"  {position}/{len(examples)}")

    write_jsonl(out_dir / "predictions.jsonl", rows)

    run_config = {
        "model": args.model,
        "adapter": adapter,
        "adapter_info": adapter_info,
        "note": args.note,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "reporting_level": args.level,
        "selection": selection,
        "runtime": "asr.explicit.ASRRunner",
        "cuda_device": device_name,
        "provenance": provenance(),
    }
    write_json(out_dir / "run_config.json", run_config)

    _finalize(rows, args, out_dir)
    return 0


def command_score(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.predictions)
    out_dir = Path(args.out_dir or Path(args.predictions).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    _finalize(rows, args, out_dir)
    return 0


def _finalize(rows: list[dict], args: argparse.Namespace, out_dir: Path) -> None:
    metrics, per_example = score_predictions(
        rows,
        level=args.level,
        analysis_config=AnalysisConfig(
            level=args.level,
            min_run=args.min_run,
            rare_threshold=args.rare_threshold,
        ),
    )
    metrics["provenance"] = provenance()
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "errors.jsonl", per_example)
    _print_report(metrics)
    print(f"\nWrote metrics.json and errors.jsonl to {out_dir}")


def _print_report(metrics: dict) -> None:
    head = metrics["headline"]
    sens = metrics["normalization_sensitivity"]
    print(f"\n{'=' * 62}")
    print(f"Examples: {metrics['examples']}   level: {metrics['reporting_level']}")
    print(f"WER: {head['wer_percent']}%    CER: {head['cer_percent']}%")
    print(f"{'-' * 62}")
    print("Normalization sensitivity (WER %):")
    print(f"  raw (none)            {sens['raw_wer_percent']}")
    print(f"  reported              {sens['reported_wer_percent']}")
    print(f"  orthography-blind     {sens['orthography_blind_wer_percent']}")
    print(
        f"  formatting cost {sens['formatting_delta_points']} pts | "
        f"orthography cost {sens['orthography_delta_points']} pts"
    )

    analysis = metrics["error_analysis"]
    if analysis["total_errors"]:
        print(f"{'-' * 62}")
        print(f"Error categories ({analysis['total_errors']} errors):")
        for category, payload in analysis["by_category"].items():
            print(
                f"  {category:<16} {payload['errors']:>6}  "
                f"{payload['share_of_errors_percent']:>6}%"
            )
        if analysis["flag_counts"]:
            print(f"  flags: {analysis['flag_counts']}")

    if metrics.get("by_category"):
        print(f"{'-' * 62}")
        print("Hard set by category:")
        for category, payload in metrics["by_category"].items():
            print(
                f"  {category:<16} n={payload['items']:<4} "
                f"WER {payload['wer_percent']}%  CER {payload['cer_percent']}%"
            )

    latency = metrics.get("latency") or {}
    if latency:
        print(f"{'-' * 62}")
        print(
            f"Latency: p50 {latency['total_ms_p50']} ms | "
            f"p90 {latency['total_ms_p90']} ms | "
            f"RTF mean {latency['real_time_factor_mean']}"
        )
    print("=" * 62)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _add_scoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--level",
        default=NormalizationLevel.STANDARD.value,
        choices=[level.value for level in ALL_LEVELS],
        help="Normalization level for the reported metric (default: standard).",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=4,
        help="Consecutive deletions/insertions treated as structural.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=1,
        help="Corpus count at or below which a reference token counts as rare.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible Hindi ASR evaluation harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="transcribe and score (needs GPU)")
    run.add_argument("--model", default="openai/whisper-medium")
    run.add_argument("--adapter", default=None, help="PEFT LoRA adapter directory")
    run.add_argument(
        "--adapter-dir",
        default=None,
        help="Trainer output directory; resolves a checkpoint by policy "
        "instead of a hardcoded step number, which goes stale as "
        "save_total_limit rotates checkpoints away.",
    )
    run.add_argument(
        "--adapter-policy",
        default="latest",
        choices=["latest", "best", "final"],
        help="With --adapter-dir: newest checkpoint, the Trainer's recorded "
        "best, or the exported best/ directory. Default: latest.",
    )
    run.add_argument(
        "--stage-adapter",
        default=None,
        metavar="DIR",
        help="Copy the adapter here (e.g. /content/adapters) before loading. "
        "Avoids repeated slow reads over the Drive mount and avoids adding "
        "read load to a mount a training job is writing to.",
    )
    run.add_argument(
        "--note",
        default="",
        help="Free-text note recorded in run_config.json. Use it for caveats, "
        "e.g. 'GPU shared with a training run - latency invalid'.",
    )
    run.add_argument("--language", default="hi")
    run.add_argument(
        "--split",
        default="test",
        help="FLEURS split. Use 'validation' for model selection; 'test' is "
        "for final reporting only.",
    )
    run.add_argument("--limit", type=int, default=None, help="None means full split")
    run.add_argument("--sample", default="random", choices=["random", "first"])
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-new-tokens", type=int, default=225)
    run.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="float16 uses T4 tensor cores and is the serving default; "
        "float32 is the reference for checking quantisation loss.",
    )
    run.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Discarded warmup passes before timing. The first GPU forward "
        "pays for cuDNN autotuning and allocator growth; without warmup it "
        "poisons mean RTF. Set 0 to measure cold-start explicitly.",
    )
    run.add_argument("--hard-set", default=None, help="Hard set manifest path")
    run.add_argument("--root", default=".", help="Root for relative audio paths")
    run.add_argument("--out-dir", required=True)
    _add_scoring_args(run)
    run.set_defaults(func=command_run)

    score = sub.add_parser("score", help="re-score saved predictions (no GPU)")
    score.add_argument("--predictions", required=True)
    score.add_argument("--out-dir", default=None)
    _add_scoring_args(score)
    score.set_defaults(func=command_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
