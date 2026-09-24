"""Run every batch benchmark unattended and write one manifest.

Each open decision in the project has a benchmark that settles it, and none
of them need a browser, a microphone or a person watching. This driver runs
them in order of what they decide, tolerates failures (a later suite still
runs), streams each suite's log to disk, and writes ``manifest.json`` with
the command, exit code, duration and output directory for every step.

    python scripts/bench_all.py --adapter /content/v2-final/best
    python scripts/bench_all.py --adapter … --only guards_on,llm

The order is deliberate: the regression check first, because the decode
guards changed serving behaviour and an unmeasured regression would make
every later number meaningless.

Built for a hosted notebook, where the session dies before the suite
finishes more often than not:

* **Resumable by default**, and completion is an explicit claim rather than a
  guess about files. Three of the four harnesses rewrite their evidence file
  after *every* unit of work — ``gpu_validation`` saves ``report.json`` after
  each of its 16 checks, ``llm_bakeoff`` rewrites ``summary.json`` after each
  model (including a failed one) — so "the evidence file exists" says nothing
  about whether the step finished. This driver therefore records each step's
  exit code in ``.bench_state.json`` and only skips a step it saw succeed.
  ``--force`` redoes everything, ``--force-steps a,b`` some.
* **Surviving the session.** A disconnect gives you a fresh VM and an empty
  checkout, so resume only helps if the evidence outlives it. Either point
  ``--out-root`` at a mounted Drive (simplest — the state file lives beside
  the evidence and resume just works), or use ``--mirror DIR``, which copies
  path-preservingly after each step and is restored into ``--out-root`` at
  startup.

Nothing here prints a conclusion. Read the JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


#: Where the driver records which steps it saw succeed. Lives under
#: --out-root so it travels with the evidence.
STATE_FILE = ".bench_state.json"
#: Never mirrored: large, regenerable, or not evidence.
MIRROR_SKIP = ("*.wav", "*.mp3", "*.bin", "*.pt", "*.safetensors")


def load_state(out_root: Path) -> dict:
    path = out_root / STATE_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt state file just means re-run
        return {}


def save_state(out_root: Path, state: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n",
                                       encoding="utf-8")


def is_done(step: dict, state: dict) -> bool:
    """Did a previous run of this driver see this step succeed?

    Completion is an explicit claim, not an inference from files. The
    harnesses rewrite their evidence file after every unit of work —
    gpu_validation after each of 16 checks, llm_bakeoff after each model
    including a failed one — so a session that died mid-step leaves a
    perfectly well-formed report.json behind. Skipping on that produced a
    green manifest over a sweep that had run two checks.
    """
    record = state.get(step["name"])
    if not isinstance(record, dict) or record.get("returncode") != 0:
        return False
    out = step.get("out")
    if out is None:
        return False                      # cheap steps always re-run
    return Path(out).exists()             # evidence must still be there


def mirror_output(step: dict, out_root: Path, mirror_root: Path) -> str | None:
    """Copy a step's evidence somewhere that survives the session.

    Path-preserving: flattening to ``mirror/<basename>`` meant the mirror
    could not be restored into --out-root, so resume never fired across the
    disconnect it exists for.
    """
    out = step.get("out")
    if out is None:
        return None
    out = Path(out)
    if not out.exists():
        return None
    try:
        relative = out.relative_to(out_root)
    except ValueError:
        relative = Path(out.name)
    target = mirror_root / relative
    try:
        if out.is_dir():
            shutil.copytree(out, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*MIRROR_SKIP))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, target)
        return str(target)
    except Exception as exc:  # noqa: BLE001 - a failed mirror must not fail the run
        return f"mirror failed: {type(exc).__name__}: {exc}"


def restore_from_mirror(out_root: Path, mirror_root: Path) -> int:
    """Copy a surviving mirror back into --out-root before planning.

    This is what makes --mirror actually resumable: a disconnect leaves a
    fresh VM with an empty checkout, and without this the driver plans as
    though nothing had ever run.
    """
    if not mirror_root.is_dir():
        return 0
    restored = 0
    for source in mirror_root.rglob("*"):
        if not source.is_file():
            continue
        target = out_root / source.relative_to(mirror_root)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
            restored += 1
        except Exception:  # noqa: BLE001 - a partial restore just re-runs a step
            continue
    return restored


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
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps whose evidence already exists")
    parser.add_argument("--force-steps", default=None,
                        help="Comma-separated step names to re-run even if done")
    parser.add_argument("--mirror", default=None,
                        help="Copy each step's evidence here as it finishes "
                             "(e.g. a mounted Drive), so a dead session loses nothing")
    parser.add_argument("--logs", default="results/bench_logs")
    args = parser.parse_args(argv)

    plan = steps(args)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        plan = [s for s in plan if s["name"] in wanted]
    if args.skip:
        unwanted = {n.strip() for n in args.skip.split(",")}
        plan = [s for s in plan if s["name"] not in unwanted]

    force_steps = ({n.strip() for n in args.force_steps.split(",")}
                   if args.force_steps else set())
    out_root = Path(args.out_root)
    mirror_root = Path(args.mirror) if args.mirror else None
    if mirror_root:
        mirror_root.mkdir(parents=True, exist_ok=True)
        restored = restore_from_mirror(out_root, mirror_root)
        if restored:
            print(f"restored {restored} file(s) from {mirror_root}")
    state = load_state(out_root)

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out_root) / "bench_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.time()

    print(f"{len(plan)} step(s); logs in {logs}\n")
    for index, step in enumerate(plan, 1):
        log_path = logs / f"{step['name']}.log"
        if not args.force and step["name"] not in force_steps and is_done(step, state):
            print(f"[{index}/{len(plan)}] {step['name']}: already done "
                  f"({step['out']}) — skipping", flush=True)
            records.append({"step": step["name"], "decides": step["decides"],
                            "cmd": step["cmd"], "returncode": 0, "seconds": 0.0,
                            "skipped": True, "log": str(log_path),
                            "out": str(step["out"]) if step["out"] else None,
                            "tail": "skipped: evidence already present"})
            manifest_path.write_text(
                json.dumps({"adapter": args.adapter,
                            "total_minutes": round((time.time() - started) / 60, 1),
                            "steps": records}, indent=2) + "\n", encoding="utf-8")
            continue
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
        state[step["name"]] = {"returncode": proc.returncode,
                               "seconds": round(seconds, 1),
                               "out": str(step["out"]) if step["out"] else None}
        save_state(out_root, state)
        mirrored = (mirror_output(step, out_root, mirror_root) if mirror_root else None)
        if mirror_root:
            mirror_output({"name": "state", "out": out_root / STATE_FILE},
                          out_root, mirror_root)
        records.append({
            "step": step["name"], "decides": step["decides"],
            "cmd": step["cmd"], "returncode": proc.returncode,
            "seconds": round(seconds, 1), "skipped": False, "log": str(log_path),
            "out": str(step["out"]) if step["out"] else None,
            "mirrored": mirrored,
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
    skipped = [r["step"] for r in records if r.get("skipped")]
    print(f"\n{len(records) - len(failed)}/{len(records)} step(s) ok in "
          f"{(time.time() - started) / 60:.0f} min"
          + (f"; {len(skipped)} skipped as already done: {skipped}" if skipped else ""))
    if failed:
        print(f"failed: {failed} — see {logs}")
    print(f"manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
