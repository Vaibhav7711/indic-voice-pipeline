"""indic-voice-pipeline, run 3 on a Kaggle T4: the grounded prompt and the parity margins.

Clones the repository at the handoff commit, fetches the fix commit (registered
before this run) from an embedded git bundle, and runs in order:

  1. benchmarks.answer_quality --system-variant grounded           task 1
  2. benchmarks.answer_quality --system-variant default            same-session control
  3. scripts/logit_margins.py as documented, at the handoff commit expected not to start
  4. scripts.logit_margins as a module, at the handoff commit      the unchanged script
  5. scripts/logit_margins.py as documented, at the fix commit     task 2

Every command passes --dtype float16 and records GPU, dtype and commit in --note.
Results, logs and a sha256 manifest land in /kaggle/working/out.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/Vaibhav7711/indic-voice-pipeline.git"
BASE = "edfe2e77e09203032fb7494dd639c574b310468e"
FIX = "186a530e05dd45e521bc2b43dc68c3b66916576d"
BUNDLE = "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"
MODEL = "Qwen/Qwen3-1.7B"
REPO = Path("/tmp/ivp")
OUT = Path("/kaggle/working/out")
LOGS = OUT / "logs"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
       "TOKENIZERS_PARALLELISM": "false"}
steps = []


def run(name, cmd, cwd=REPO, check=True):
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    (LOGS / f"{name}.log").write_text(proc.stdout)
    print(proc.stdout[-8000:], flush=True)
    print(f"=== {name}: exit {proc.returncode} in {time.time() - started:.0f}s", flush=True)
    steps.append({"step": name, "cmd": cmd, "exit": proc.returncode,
                  "seconds": round(time.time() - started, 1)})
    if check and proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit {proc.returncode}")
    return proc


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


LOGS.mkdir(parents=True, exist_ok=True)
run("nvidia-smi", ["nvidia-smi"], cwd="/", check=False)
run("clone", ["git", "clone", "-q", REPO_URL, str(REPO)], cwd="/")
run("checkout-base", ["git", "checkout", "-q", "--detach", BASE])
assert head() == BASE, head()
Path("/tmp/fix.bundle").write_bytes(base64.b64decode(BUNDLE))
run("fetch-fix", ["git", "fetch", "-q", "/tmp/fix.bundle",
                  "refs/heads/gpu-handoff-results:refs/heads/fix"])
run("install", [sys.executable, "-m", "pip", "install", "-q", "-e", ".[audio]"])
probe = run("env", [sys.executable, "-c", (
    "import json, platform, torch, transformers, accelerate, tokenizers; "
    "print(json.dumps({'gpu': torch.cuda.get_device_name(0), "
    "'capability': list(torch.cuda.get_device_capability(0)), 'gpus': torch.cuda.device_count(), "
    "'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, "
    "'tokenizers': tokenizers.__version__, 'accelerate': accelerate.__version__, "
    "'python': platform.python_version()}))")])
env = json.loads(probe.stdout.strip().splitlines()[-1])
gpu = f"Kaggle {env['gpu']} (sm_{env['capability'][0]}{env['capability'][1]})"


def note(commit, what):
    return f"{gpu}, float16, commit {commit[:7]}, {what}"


common = ["--llm-model", MODEL, "--dtype", "float16"]
quality = [sys.executable, "-m", "benchmarks.answer_quality", "--llm-engine", "explicit", *common]
run("grounded", [*quality, "--system-variant", "grounded",
                 "--out", "results/answer_quality/grounded.json",
                 "--note", note(BASE, "system-variant=grounded, greedy")], check=False)
run("default", [*quality, "--system-variant", "default",
                "--out", "results/answer_quality/greedy_run3.json",
                "--note", note(BASE, "system-variant=default, greedy; same-session control for grounded.json")],
    check=False)
margins = ["--parity", "results/engine_parity/parity.json", *common]
run("margins-documented-at-base", [sys.executable, "scripts/logit_margins.py", *margins,
    "--out", "results/engine_parity/logit_margins_documented_edfe2e7.json",
    "--note", note(BASE, "documented command, explaining the 4/5 parity run")], check=False)
run("margins-module-at-base", [sys.executable, "-m", "scripts.logit_margins", *margins,
    "--out", "results/engine_parity/logit_margins_edfe2e7.json",
    "--note", note(BASE, "script as committed, run as a module since the documented form "
                         "cannot start, explaining the 4/5 parity run")], check=False)
run("checkout-fix", ["git", "checkout", "-q", "--detach", FIX])
assert head() == FIX, head()
run("margins", [sys.executable, "scripts/logit_margins.py", *margins,
                "--note", note(FIX, "explaining the 4/5 parity run")], check=False)

files = {}
for rel in ("results/answer_quality/grounded.json", "results/answer_quality/greedy_run3.json",
            "results/engine_parity/logit_margins.json", "results/engine_parity/logit_margins_edfe2e7.json",
            "results/engine_parity/logit_margins_documented_edfe2e7.json"):
    src = REPO / rel
    if src.is_file():
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
    else:
        files[rel] = None
manifest = {"base": BASE, "fix": FIX, "model": MODEL, "env": env, "steps": steps, "files": files}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print("RUN3_DONE", flush=True)
