"""Run every batch benchmark unattended and write one manifest.

Each open decision in the project has a benchmark that settles it, and none
of them need a browser, a microphone or a person watching. This driver runs
them in order of what they decide, tolerates failures (a later suite still
runs), streams each suite's log to disk, and writes ``manifest.json`` with
the command, exit code, duration and output directory for every step.

    python scripts/bench_all.py --adapter /content/v2-final/best
    python scripts/bench_all.py --adapter … --only guards,llm

The order is deliberate: the regression check first, because today's decode
guards changed serving behaviour and an unmeasured regression would make
every later number meaningless.

Nothing here prints a conclusion. Read the JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ADAPTER = "/content/v2-final/best"
TURBO = "openai/whisper-large-v3-turbo"


def steps(args) -> list[dict]:
    """(name, why it exists, command). ``out`` is where its evidence lands."""
    root = Path(args.out_root)
    adapter = args.adapter
    eval_common = ["--split", "test", "--limit", str(args.limit), "--seed", "0",
                   "--dtype", "float16"]
    return [
        {
            "name": "unit_tests",
            "decides": "nothing regressed in the 545 CPU tests",
            "cmd": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            "out": None,
        },
        {
            # First, because the no-speech threshold added today can suppress
            # a clip that does contain speech, which is a whole-utterance
            # deletion. Two runs one flag apart.
            "name": "guards_on",
            "decides": "WER with today's decode guards (the shipped default)",
            "cmd": [sys.executable, "-m", "benchmarks.asr_eval", "run",
                    "--model", TURBO, "--adapter", adapter, *eval_common,
                    "--out-dir", str(root / "eval/v2-guards-on"),
                    "--note", "decode guards at defaults"],
            "out": root / "eval/v2-guards-on",
        },
        {
            "name": "guards_off",
            "decides": "WER without them — the cost or benefit of the guards",
            "cmd": [sys.executable, "-m", "benchmarks.asr_eval", "run",
                    "--model", TURBO, "--adapter", adapter, *eval_common,
                    "--no-speech-threshold", "-1", "--loop-guard-ngram", "0",
                    "--out-dir", str(root / "eval/v2-guards-off"),
                    "--note", "no-speech check and loop guard disabled"],
            "out": root / "eval/v2-guards-off",
        },
        {
            "name": "compare_guards",
            "decides": "whether the guard change is a regression",
            "cmd": [sys.executable, "-m", "benchmarks.compare",
                    "--baseline", str(root / "eval/v2-guards-off"),
                    "--candidate", str(root / "eval/v2-guards-on"),
                    "--out", str(root / "eval/compare-guards.json")],
            "out": root / "eval/compare-guards.json",
        },
        {
            "name": "llm",
            "decides": "which LLM — the largest open quality question",
            "cmd": [sys.executable, "-m", "benchmarks.llm_bakeoff",
                    "--models", args.llm_models,
                    "--out-dir", str(root / "llm_bakeoff")],
            "out": root / "llm_bakeoff",
        },
        {
            "name": "tts",
            "decides": "edge (network) vs MMS (local) on this GPU",
            "cmd": [sys.executable, "-m", "benchmarks.tts_bakeoff",
                    "--backends", args.tts_backends,
                    "--out-dir", str(root / "tts_bakeoff")],
            "out": root / "tts_bakeoff",
        },
        {
            "name": "streaming",
            "decides": "incremental finals and semantic endpointing on/off",
            "cmd": [sys.executable, "-m", "benchmarks.streaming_eval",
                    "--model", TURBO, "--adapter", adapter,
                    "--split", "test", "--limit", str(args.streaming_limit),
                    "--seed", "0", "--dtype", "float16",
                    "--session-grid", "baseline,early,early-incr,early-sem,full",
                    "--out-dir", str(root / "streaming_eval")],
            "out": root / "streaming_eval",
        },
        {
            "name": "sweep",
            "decides": "the 16 correctness checks, incl. compiled decode on this GPU",
            "cmd": [sys.executable, "scripts/gpu_validation.py",
                    "--whisper-model", TURBO, "--adapter", adapter,
                    "--skip-unit-tests",
                    "--out-dir", str(root / "gpu_validation")],
            "out": root / "gpu_validation",
        },
        {
            "name": "ct2_convert",
            "decides": "engine weights exist (prerequisite for the next step)",
            "cmd": [sys.executable, "scripts/convert_ct2.py", "--model", TURBO,
                    "--adapter", adapter, "--quantization", "int8_float16",
                    "--out", str(Path(args.ct2_dir))],
            "out": Path(args.ct2_dir),
        },
        {
            "name": "ct2_check",
            "decides": "engine speedup, gated on matching the explicit runner",
            "cmd": [sys.executable, "scripts/gpu_validation.py",
                    "--whisper-model", TURBO, "--adapter", adapter,
                    "--skip-unit-tests", "--skip-network",
                    "--ct2-model", str(Path(args.ct2_dir)),
                    "--ct2-compute-type", "int8_float16",
                    "--out-dir", str(root / "gpu_validation-ct2")],
            "out": root / "gpu_validation-ct2",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--limit", type=int, default=300,
                        help="Clips for the ASR evaluations")
    parser.add_argument("--streaming-limit", type=int, default=100)
    parser.add_argument("--llm-models",
                        default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B",
                        help="Comma-separated; a 'name:4bit' spec loads that "
                             "candidate quantized. On an 8 GB card Qwen3-4B "
                             "only fits as ':4bit' alongside Whisper.")
    parser.add_argument("--tts-backends", default="edge,mms")
    parser.add_argument("--ct2-dir", default="models/ct2/turbo-hindi-v2")
    parser.add_argument("--only", default=None,
                        help="Comma-separated step names to run")
    parser.add_argument("--skip", default=None)
    parser.add_argument("--logs", default="results/bench_logs")
    args = parser.parse_args(argv)

    plan = steps(args)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        plan = [s for s in plan if s["name"] in wanted]
    if args.skip:
        unwanted = {n.strip() for n in args.skip.split(",")}
        plan = [s for s in plan if s["name"] not in unwanted]

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out_root) / "bench_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.time()

    print(f"{len(plan)} step(s); logs in {logs}\n")
    for index, step in enumerate(plan, 1):
        log_path = logs / f"{step['name']}.log"
        print(f"[{index}/{len(plan)}] {step['name']}: {step['decides']}", flush=True)
        begin = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(" ".join(step["cmd"]) + "\n\n")
            handle.flush()
            proc = subprocess.run(step["cmd"], stdout=handle,
                                  stderr=subprocess.STDOUT, text=True)
        seconds = time.time() - begin
        tail = ""
        try:
            lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            tail = " | ".join(lines[-3:])[:300]
        except Exception:  # noqa: BLE001
            pass
        records.append({
            "step": step["name"], "decides": step["decides"],
            "cmd": step["cmd"], "returncode": proc.returncode,
            "seconds": round(seconds, 1), "log": str(log_path),
            "out": str(step["out"]) if step["out"] else None,
            "tail": tail,
        })
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"    {status} in {seconds / 60:.1f} min — {log_path}")
        if proc.returncode != 0:
            print(f"    {tail}")
        manifest_path.write_text(
            json.dumps({"adapter": args.adapter,
                        "total_minutes": round((time.time() - started) / 60, 1),
                        "steps": records}, indent=2) + "\n", encoding="utf-8")

    failed = [r["step"] for r in records if r["returncode"]]
    print(f"\n{len(records) - len(failed)}/{len(records)} step(s) ok in "
          f"{(time.time() - started) / 60:.0f} min")
    if failed:
        print(f"failed: {failed} — see {logs}")
    print(f"manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
