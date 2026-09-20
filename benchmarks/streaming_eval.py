"""Streaming ASR evaluation: the endpointer + session against offline decoding.

Why this exists
---------------
The GPU validation sweep streams two clips. That is enough to *find* a
segmentation failure (it found a 13 dB level difference deleting whole
phrases) and not enough to *tune* one: any threshold chosen on two clips is
fitted to them. This benchmark runs the streaming session over a seeded FLEURS
subset — the same selection discipline as ``benchmarks.asr_eval`` — and
separates the three things "streaming WER" conflates:

* ``wer_vs_reference``  — what the user would experience (model + segmentation).
* ``wer_vs_offline``    — the streaming penalty alone: the same audio decoded
  once, offline, is the reference. A perfect endpointer scores ~0 here.
* ``vad_agreement``     — how much offline-VAD speech the online finals cover.

plus per-clip structure: finals per clip (splits), empty clips, onset
hallucinations (a repeated n-gram in the first words of a final, the
signature of a clipped soft onset), and endpoint reasons.

Several VAD configurations can be evaluated in one run; the offline decode of
each clip is done once and shared, so a grid of *k* configs costs ``1 + k``
decodes per clip rather than ``2k``.

Each clip is streamed as ``lead`` seconds of silence + clip + ``trail`` seconds
of silence in 100 ms blocks with an audio-driven clock, so the endpointer sees
the same shape a microphone gives it: quiet, speech, quiet.

Usage::

    python -m benchmarks.streaming_eval \\
        --adapter Hugme6969/whisper-medium-hindi-lora \\
        --split test --limit 100 --seed 0 \\
        --grid default,pad300,floor70,pad300-floor70 \\
        --out-dir results/streaming_eval/medium-lora-test-100

Outputs, per config, under ``<out-dir>/<config>/``: ``clips.jsonl`` (one row
per clip with finals, texts and flags) and ``metrics.json``; plus a top-level
``run_config.json`` and ``summary.json`` comparing configs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from asr.vad import VADConfig

SR = 16_000

#: Named VAD variants. Keys are what ``--grid`` accepts. Each is a dict of
#: VADConfig field overrides applied on top of the defaults.
VAD_GRID: dict[str, dict] = {
    "default": {},
    "fixed40": {"adaptive_threshold": False, "threshold_dbfs": -40.0},
    "pad300": {"padding_ms": 300},
    "pad500": {"padding_ms": 500},
    "floor65": {"threshold_floor_dbfs": -65.0},
    "floor70": {"threshold_floor_dbfs": -70.0},
    "pad300-floor70": {"padding_ms": 300, "threshold_floor_dbfs": -70.0},
    "pad500-floor70": {"padding_ms": 500, "threshold_floor_dbfs": -70.0},
    "sil400": {"min_silence_ms": 400},
    "sil800": {"min_silence_ms": 800},
    "margin8": {"noise_margin_db": 8.0},
    "margin12": {"noise_margin_db": 12.0},
}


def vad_from_name(name: str) -> VADConfig:
    if name not in VAD_GRID:
        raise SystemExit(f"unknown VAD config {name!r}; choose from {sorted(VAD_GRID)}")
    return replace(VADConfig(), **VAD_GRID[name])


# --------------------------------------------------------------------------
# Per-clip streaming
# --------------------------------------------------------------------------


class _AudioClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def stream_clip(runner, waveform: np.ndarray, vad: VADConfig, *, language: str,
                lead: float, trail: float, block_ms: int = 100) -> dict:
    """Stream one clip through a fresh session; return finals and geometry."""
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind
    from asr.vad import detect_speech

    audio = np.concatenate([
        np.zeros(int(lead * SR), dtype=np.float32),
        waveform.astype(np.float32),
        np.zeros(int(trail * SR), dtype=np.float32),
    ])
    config = StreamingConfig(language=language, vad=vad, emit_partials=False)
    clock = _AudioClock()
    session = StreamingSession(runner, config, clock=clock)
    block = SR * block_ms // 1000
    updates = []
    for start in range(0, len(audio), block):
        chunk = audio[start:start + block]
        clock.now += len(chunk) / SR
        updates.extend(session.push(chunk))
    updates.extend(session.flush())
    finals = [u for u in updates if u.kind == UpdateKind.FINAL]

    spans = [(f.utterance_start_seconds, f.utterance_start_seconds + f.audio_seconds)
             for f in finals]
    offline_segments = detect_speech(audio, SR, vad)
    offline_speech = sum(s.duration_seconds for s in offline_segments)
    agreed = sum(
        max(0.0, min(s.end_seconds, f1) - max(s.start_seconds, f0))
        for s in offline_segments for f0, f1 in spans
    )
    clip_start = lead
    return {
        "finals": [
            {"start": round(f.utterance_start_seconds - clip_start, 3),
             "seconds": round(f.audio_seconds, 3),
             "reason": f.endpoint_reason.value if f.endpoint_reason else None,
             "asr_ms": round(f.asr_ms, 1), "text": f.text}
            for f in finals
        ],
        "text": " ".join(f.text for f in finals).strip(),
        "vad_agreement": round(agreed / offline_speech, 4) if offline_speech else None,
        "offline_vad_segments": len(offline_segments),
        "first_final_start": round(spans[0][0] - clip_start, 3) if spans else None,
        "streamed_asr_ms": round(sum(f.asr_ms for f in finals), 1),
    }


def onset_hallucination(text: str, level: str = "standard") -> bool:
    """A token repeated 3+ times back-to-back within the first 6 tokens: the
    signature of Whisper decoding a clipped soft onset (``जी जी जी कैमिकल``)."""
    from benchmarks.error_analysis import _has_repetition_loop
    from text.normalize import tokenize

    head = tokenize(text, level)[:6]
    return _has_repetition_loop(head, 3)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(rows: list[dict], level: str) -> dict:
    from benchmarks.metrics import CorpusScore, score_text

    vs_ref, vs_off = CorpusScore(), CorpusScore()
    for row in rows:
        vs_ref.update(score_text(row["reference"], row["streamed_text"], level=level))
        vs_off.update(score_text(row["offline_text"], row["streamed_text"], level=level))
    offline = CorpusScore()
    for row in rows:
        offline.update(score_text(row["reference"], row["offline_text"], level=level))

    n = len(rows)
    finals = [row["n_finals"] for row in rows]
    agreements = [row["vad_agreement"] for row in rows if row["vad_agreement"] is not None]
    return {
        "clips": n,
        "wer_vs_reference": vs_ref.as_dict(),
        "wer_vs_offline": vs_off.as_dict(),
        "offline_wer_vs_reference": offline.as_dict(),
        "streaming_penalty_points": (
            None if vs_ref.micro is None or offline.micro is None
            else round((vs_ref.micro - offline.micro) * 100, 4)
        ),
        "structure": {
            "clips_with_one_final": sum(1 for f in finals if f == 1),
            "clips_split": sum(1 for f in finals if f > 1),
            "clips_empty": sum(1 for f in finals if f == 0),
            "mean_finals_per_clip": round(float(np.mean(finals)), 3) if n else None,
            "onset_hallucinations": sum(1 for row in rows if row["onset_hallucination"]),
            "vad_agreement_mean": round(float(np.mean(agreements)), 4) if agreements else None,
            "vad_agreement_p10": (
                round(float(np.percentile(agreements, 10)), 4) if agreements else None
            ),
            "endpoint_reasons": _count(r for row in rows for r in row["endpoint_reasons"]),
        },
        "latency": {
            "streamed_asr_ms_mean": round(float(np.mean([r["streamed_asr_ms"] for r in rows])), 1),
            "offline_asr_ms_mean": round(float(np.mean([r["offline_asr_ms"] for r in rows])), 1),
        },
    }


def _count(items) -> dict:
    out: dict = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory or Hub id")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--config", default="hi_in", help="FLEURS config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", default="random", choices=["random", "first"])
    parser.add_argument("--grid", default="default",
                        help="Comma-separated VAD configs; see VAD_GRID")
    parser.add_argument("--lead", type=float, default=0.5, help="Leading silence (s)")
    parser.add_argument("--trail", type=float, default=1.0, help="Trailing silence (s)")
    parser.add_argument("--level", default="standard")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    import torch

    from asr.explicit import ASRRunner, load_whisper
    from benchmarks.asr_eval import (
        _iter_fleurs,
        _select_indices,
        provenance,
        write_json,
        write_jsonl,
    )
    from benchmarks.fleurs import load_fleurs

    names = [n.strip() for n in args.grid.split(",") if n.strip()]
    grid = {name: vad_from_name(name) for name in names}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_fleurs(args.config, args.split)
    indices = _select_indices(len(dataset), args.limit, args.sample, args.seed)
    write_json(out_dir / "run_config.json", {
        "model": args.model, "adapter": args.adapter, "language": args.language,
        "selection": {"source": "fleurs", "config": args.config, "split": args.split,
                      "strategy": args.sample, "seed": args.seed, "limit": args.limit,
                      "available": len(dataset), "indices": indices},
        "grid": {name: cfg.as_dict() for name, cfg in grid.items()},
        "lead_seconds": args.lead, "trail_seconds": args.trail, "level": args.level,
        "dtype": args.dtype, "note": args.note, "provenance": provenance(),
    })

    dtype = getattr(torch, args.dtype)
    loaded = load_whisper(args.model, adapter_path=args.adapter, dtype=dtype)
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
    print(f"{args.model}" + (f" + {args.adapter}" if args.adapter else "")
          + f" | {len(indices)} clips | configs: {', '.join(names)}")

    per_config: dict[str, list[dict]] = {name: [] for name in names}
    for i, ex in enumerate(_iter_fleurs(dataset, args.config, args.split, indices)):
        from asr.explicit.mel import load_audio_from_array

        waveform, _ = load_audio_from_array(np.asarray(ex["audio_array"], dtype=np.float32),
                                            ex["sampling_rate"])
        # One offline decode per clip, shared by every config.
        offline = runner.transcribe_array(waveform, SR, language=args.language)
        for name, vad in grid.items():
            streamed = stream_clip(runner, waveform, vad, language=args.language,
                                   lead=args.lead, trail=args.trail)
            per_config[name].append({
                "id": ex["id"], "audio_ref": ex["audio_ref"],
                "audio_seconds": round(len(waveform) / SR, 3),
                "reference": ex["reference"],
                "offline_text": offline.text,
                "offline_asr_ms": round(offline.metrics.total_ms, 1),
                "streamed_text": streamed["text"],
                "streamed_asr_ms": streamed["streamed_asr_ms"],
                "n_finals": len(streamed["finals"]),
                "finals": streamed["finals"],
                "endpoint_reasons": [f["reason"] for f in streamed["finals"]],
                "vad_agreement": streamed["vad_agreement"],
                "offline_vad_segments": streamed["offline_vad_segments"],
                "first_final_start": streamed["first_final_start"],
                "onset_hallucination": onset_hallucination(streamed["text"], args.level),
            })
        if (i + 1) % 10 == 0 or i + 1 == len(indices):
            print(f"  {i + 1}/{len(indices)}")

    summary = {}
    for name, rows in per_config.items():
        cfg_dir = out_dir / name
        cfg_dir.mkdir(exist_ok=True)
        write_jsonl(cfg_dir / "clips.jsonl", rows)
        metrics = aggregate(rows, args.level)
        metrics["vad"] = grid[name].as_dict()
        write_json(cfg_dir / "metrics.json", metrics)
        summary[name] = {
            "wer_vs_reference": metrics["wer_vs_reference"]["micro_percent"],
            "wer_vs_offline": metrics["wer_vs_offline"]["micro_percent"],
            "streaming_penalty_points": metrics["streaming_penalty_points"],
            "clips_split": metrics["structure"]["clips_split"],
            "clips_empty": metrics["structure"]["clips_empty"],
            "onset_hallucinations": metrics["structure"]["onset_hallucinations"],
            "vad_agreement_mean": metrics["structure"]["vad_agreement_mean"],
        }
    offline_wer = aggregate(per_config[names[0]], args.level)["offline_wer_vs_reference"]
    write_json(out_dir / "summary.json", {
        "offline_wer_vs_reference": offline_wer["micro_percent"], "configs": summary,
    })

    print(f"\noffline WER vs reference: {offline_wer['micro_percent']:.2f}%\n")
    header = f"{'config':16s} {'WER ref':>8s} {'WER off':>8s} {'penalty':>8s} " \
             f"{'split':>5s} {'empty':>5s} {'onset':>5s} {'agree':>6s}"
    print(header)
    for name, row in summary.items():
        print(f"{name:16s} {row['wer_vs_reference']:8.2f} {row['wer_vs_offline']:8.2f} "
              f"{row['streaming_penalty_points']:+8.2f} {row['clips_split']:5d} "
              f"{row['clips_empty']:5d} {row['onset_hallucinations']:5d} "
              f"{row['vad_agreement_mean']:6.3f}")
    print(f"\nwritten: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
