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
    # The pre-2026-09-20 behaviour: fixed -40 dBFS, 200 ms padding, -60 floor.
    "fixed40": {"adaptive_threshold": False, "threshold_dbfs": -40.0, "padding_ms": 200},
    # The defaults as first benchmarked (results/streaming_eval/medium-lora-test-100).
    "v1-adaptive": {"padding_ms": 200, "threshold_floor_dbfs": -60.0},
    "pad200": {"padding_ms": 200},
    "pad500": {"padding_ms": 500},
    "floor60": {"threshold_floor_dbfs": -60.0},
    "floor80": {"threshold_floor_dbfs": -80.0},
    "sil400": {"min_silence_ms": 400},
    "sil800": {"min_silence_ms": 800},
    "margin8": {"noise_margin_db": 8.0},
    "margin12": {"noise_margin_db": 12.0},
}


#: Session-level variants: how the final transcript is produced.
#:   baseline    decode the whole utterance after the endpoint (pre-5.3)
#:   early       candidate decode at 300 ms of silence, committed at 600 ms
#:   early-incr  + partials every ~1.2 s and only the tail re-decoded
#:   early-sem   early + semantic endpoint policy (wait longer mid-phrase)
#:   full        early-incr + semantic
SESSION_GRID: dict[str, dict] = {
    "baseline": {"early_final_silence_ms": 0, "incremental_finals": False,
                 "semantic_endpointing": False, "emit_partials": False},
    "early": {"early_final_silence_ms": 300, "incremental_finals": False,
              "semantic_endpointing": False, "emit_partials": False},
    # partial_interval_ms must be POSITIVE: the session treats <= 0 as
    # "partials off", so the two incremental modes previously emitted none,
    # incremental_finals never found a partial to reuse, and early-incr was
    # identical to early. The benchmark drives an audio clock, so 1 ms is
    # effectively "no wall-clock limit" and min_partial_audio_ms is the real
    # gate. assert_grid_produced_partials() below fails loudly if this
    # regresses.
    "early-incr": {"early_final_silence_ms": 300, "incremental_finals": True,
                   "semantic_endpointing": False, "emit_partials": True,
                   "partial_interval_ms": 1, "min_partial_audio_ms": 1200},
    "early-sem": {"early_final_silence_ms": 300, "incremental_finals": False,
                  "semantic_endpointing": True, "emit_partials": False},
    "full": {"early_final_silence_ms": 300, "incremental_finals": True,
             "semantic_endpointing": True, "emit_partials": True,
             "partial_interval_ms": 1, "min_partial_audio_ms": 1200},
}


#: Grid configs that only mean anything if partials actually happened.
INCREMENTAL_CONFIGS = ("early-incr", "full")


def grid_sanity(name: str, rows: list[dict]) -> str | None:
    """Did this config measure what its name claims? Returns a warning or None.

    A silent no-op is the failure mode that matters here: ``early-incr``
    with partials disabled runs happily and reports numbers identical to
    ``early``, which reads as "incremental decoding buys nothing" rather
    than "incremental decoding never ran".
    """
    base = name.split("+")[-1]
    if base not in INCREMENTAL_CONFIGS:
        return None
    partials = sum(r.get("partials", 0) for r in rows)
    reused = sum(r.get("finals_reused_partial", 0) for r in rows)
    if partials == 0:
        return (f"{name}: ZERO partials — incremental decoding never ran, so these "
                f"numbers say nothing about it (check partial_interval_ms > 0)")
    if reused == 0:
        return (f"{name}: {partials} partials but no final reused one — the "
                f"incremental path never engaged")
    return None


def session_kwargs(name: str) -> dict:
    if name not in SESSION_GRID:
        raise SystemExit(f"unknown session config {name!r}; choose from {sorted(SESSION_GRID)}")
    return dict(SESSION_GRID[name])


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
                lead: float, trail: float, block_ms: int = 100,
                session: dict | None = None) -> dict:
    """Stream one clip through a fresh session; return finals and geometry.

    With the audio-driven clock, ASR compute does not advance stream time, so
    "speech end → final" is reconstructed as the silence the endpointer
    waited plus the ASR that ran *after* the endpoint. Compute that ran
    during the wait (a candidate) is not on that path.
    """
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind
    from asr.vad import detect_speech

    audio = np.concatenate([
        np.zeros(int(lead * SR), dtype=np.float32),
        waveform.astype(np.float32),
        np.zeros(int(trail * SR), dtype=np.float32),
    ])
    session_opts = {"emit_partials": False, **(session or {})}
    config = StreamingConfig(language=language, vad=vad, **session_opts)
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
    n_partials = sum(1 for u in updates if u.kind == UpdateKind.PARTIAL)
    n_candidates = sum(1 for u in updates if u.kind == UpdateKind.CANDIDATE)

    spans = [(f.utterance_start_seconds, f.utterance_start_seconds + f.audio_seconds)
             for f in finals]
    # Per final: silence the endpointer waited + ASR after the endpoint.
    # audio_seconds includes the trailing padding, so speech ended
    # padding_ms before the utterance's audio does.
    endpoint_to_final_ms = [
        max(0.0, (f.stream_seconds - (f.utterance_start_seconds + f.audio_seconds)) * 1000
            + vad.padding_ms) + f.asr_ms_after_endpoint
        for f in finals
    ]
    # Total ASR compute for the clip: finals (a final from a candidate carries
    # the candidate's time), plus partials, plus candidates that were wasted.
    candidate_ms = sum(u.asr_ms for u in updates if u.kind == UpdateKind.CANDIDATE)
    committed_candidate_ms = sum(f.asr_ms for f in finals if f.from_candidate)
    asr_ms_total = (
        sum(f.asr_ms for f in finals)
        + sum(u.asr_ms for u in updates if u.kind == UpdateKind.PARTIAL)
        + max(0.0, candidate_ms - committed_candidate_ms)
    )
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
             "asr_ms": round(f.asr_ms, 1),
             "asr_ms_after_endpoint": round(f.asr_ms_after_endpoint, 1),
             "from_candidate": f.from_candidate, "reused_partial": f.reused_partial,
             "decoded_seconds": round(f.decoded_seconds, 3), "text": f.text}
            for f in finals
        ],
        "text": " ".join(f.text for f in finals).strip(),
        "partials": n_partials, "candidates": n_candidates,
        "asr_ms_after_endpoint": round(sum(f.asr_ms_after_endpoint for f in finals), 1),
        "asr_ms_total": round(asr_ms_total, 1),
        "endpoint_to_final_ms_last": round(endpoint_to_final_ms[-1], 1) if finals else None,
        "finals_from_candidate": sum(1 for f in finals if f.from_candidate),
        "finals_reused_partial": sum(1 for f in finals if f.reused_partial),
        "decoded_seconds": round(sum(f.decoded_seconds for f in finals), 3),
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
            # What the user waits for after they stop talking, for the last
            # final of each clip: silence wait + ASR after the endpoint.
            "endpoint_to_final_ms_mean": _mean_of(rows, "endpoint_to_final_ms_last"),
            "asr_after_endpoint_ms_mean": _mean_of(rows, "asr_ms_after_endpoint"),
            "asr_total_ms_mean": _mean_of(rows, "asr_ms_total"),
            "decoded_seconds_mean": _mean_of(rows, "decoded_seconds"),
            "finals_from_candidate": sum(r.get("finals_from_candidate", 0) for r in rows),
            "finals_reused_partial": sum(r.get("finals_reused_partial", 0) for r in rows),
            "partials_total": sum(r.get("partials", 0) for r in rows),
        },
    }


def _mean_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 1) if vals else None


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
    parser.add_argument("--session-grid", default="baseline",
                        help="Comma-separated session configs; see SESSION_GRID")
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

    vad_names = [n.strip() for n in args.grid.split(",") if n.strip()]
    sess_names = [n.strip() for n in args.session_grid.split(",") if n.strip()]
    # Cartesian product; a run name is "<vad>+<session>" unless one side is
    # a single default, in which case the other side's name is used alone.
    grid: dict[str, tuple[VADConfig, dict]] = {}
    for v in vad_names:
        for sname in sess_names:
            if len(sess_names) == 1 and sname == "baseline":
                key = v
            elif len(vad_names) == 1 and v == "default":
                key = sname
            else:
                key = f"{v}+{sname}"
            grid[key] = (vad_from_name(v), session_kwargs(sname))
    names = list(grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_fleurs(args.config, args.split)
    indices = _select_indices(len(dataset), args.limit, args.sample, args.seed)
    write_json(out_dir / "run_config.json", {
        "model": args.model, "adapter": args.adapter, "language": args.language,
        "selection": {"source": "fleurs", "config": args.config, "split": args.split,
                      "strategy": args.sample, "seed": args.seed, "limit": args.limit,
                      "available": len(dataset), "indices": indices},
        "grid": {name: {"vad": vad.as_dict(), "session": sess} for name, (vad, sess) in grid.items()},
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
        for name, (vad, sess) in grid.items():
            streamed = stream_clip(runner, waveform, vad, language=args.language,
                                   lead=args.lead, trail=args.trail, session=sess)
            per_config[name].append({
                "id": ex["id"], "audio_ref": ex["audio_ref"],
                "audio_seconds": round(len(waveform) / SR, 3),
                "reference": ex["reference"],
                "offline_text": offline.text,
                "offline_asr_ms": round(offline.metrics.total_ms, 1),
                "streamed_text": streamed["text"],
                "streamed_asr_ms": streamed["streamed_asr_ms"],
                "partials": streamed["partials"], "candidates": streamed["candidates"],
                "asr_ms_after_endpoint": streamed["asr_ms_after_endpoint"],
                "asr_ms_total": streamed["asr_ms_total"],
                "endpoint_to_final_ms_last": streamed["endpoint_to_final_ms_last"],
                "finals_from_candidate": streamed["finals_from_candidate"],
                "finals_reused_partial": streamed["finals_reused_partial"],
                "decoded_seconds": streamed["decoded_seconds"],
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
        metrics["vad"] = grid[name][0].as_dict()
        metrics["session"] = grid[name][1]
        warning = grid_sanity(name, rows)
        metrics["sanity_warning"] = warning
        if warning:
            print(f"  !! {warning}")
        write_json(cfg_dir / "metrics.json", metrics)
        summary[name] = {
            "wer_vs_reference": metrics["wer_vs_reference"]["micro_percent"],
            "wer_vs_offline": metrics["wer_vs_offline"]["micro_percent"],
            "streaming_penalty_points": metrics["streaming_penalty_points"],
            "clips_split": metrics["structure"]["clips_split"],
            "clips_empty": metrics["structure"]["clips_empty"],
            "onset_hallucinations": metrics["structure"]["onset_hallucinations"],
            "vad_agreement_mean": metrics["structure"]["vad_agreement_mean"],
            "endpoint_to_final_ms": metrics["latency"]["endpoint_to_final_ms_mean"],
            "asr_after_endpoint_ms": metrics["latency"]["asr_after_endpoint_ms_mean"],
            "asr_total_ms": metrics["latency"]["asr_total_ms_mean"],
        }
    offline_wer = aggregate(per_config[names[0]], args.level)["offline_wer_vs_reference"]
    write_json(out_dir / "summary.json", {
        "offline_wer_vs_reference": offline_wer["micro_percent"], "configs": summary,
    })

    print(f"\noffline WER vs reference: {offline_wer['micro_percent']:.2f}%\n")
    header = f"{'config':22s} {'WER ref':>8s} {'WER off':>8s} {'penalty':>8s} " \
             f"{'split':>5s} {'empty':>5s} {'onset':>5s} {'end→final':>10s} {'asr after':>9s} {'asr total':>9s}"
    print(header)
    for name, row in summary.items():
        print(f"{name:22s} {row['wer_vs_reference']:8.2f} {row['wer_vs_offline']:8.2f} "
              f"{row['streaming_penalty_points']:+8.2f} {row['clips_split']:5d} "
              f"{row['clips_empty']:5d} {row['onset_hallucinations']:5d} "
              f"{row['endpoint_to_final_ms'] or 0:10.0f} {row['asr_after_endpoint_ms'] or 0:9.0f} "
              f"{row['asr_total_ms'] or 0:9.0f}")
    print(f"\nwritten: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
