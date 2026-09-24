"""Publish selected Colab benchmark evidence to a Git branch.

The branch is deliberately evidence-only: it contains machine-readable
results and logs, never audio, model weights, checkpoints, or arbitrary
working-tree edits. A Codex session can fetch this branch after a hosted
runtime pushes it, avoiding copy/paste of benchmark output.

In Colab, store a fine-grained GitHub token with repository Contents
read/write permission in the ``GITHUB_TOKEN`` secret, then run::

    import os
    from google.colab import userdata
    os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    !python scripts/publish_colab_evidence.py

The credential helper is an in-memory cache and expires with the runtime.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

EVIDENCE_DIRS = (
    "results/eval",
    "results/llm_bakeoff",
    "results/tts_bakeoff",
    "results/streaming_eval",
    "results/gpu_validation",
    "results/gpu_validation-ct2",
    "results/live",
    "results/bench_logs",
)
EVIDENCE_FILES = (
    "results/bench_manifest.json",
    "results/.bench_state.json",
    "results/SESSION.md",
)
ALLOWED_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt"}


def git(*args: str, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], text=True, input=input_text, capture_output=True, check=check,
    )


def ensure_no_tracked_code_changes() -> None:
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = git(*args, check=False)
        if result.returncode:
            raise SystemExit(
                "Refusing to switch branches with tracked edits present. Commit, stash, "
                "or discard code changes before publishing Colab evidence."
            )


def evidence_paths(root: Path) -> list[str]:
    paths: list[Path] = []
    for rel in EVIDENCE_FILES:
        candidate = root / rel
        if candidate.is_file():
            paths.append(candidate)
    for rel in EVIDENCE_DIRS:
        directory = root / rel
        if directory.is_dir():
            paths.extend(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
            )
    return sorted(str(path.relative_to(root)) for path in paths)


def configure_ephemeral_github_credential(remote: str, token_env: str) -> None:
    """Make a secret token available to Git only through its cache helper."""
    token = os.environ.get(token_env)
    if not token:
        return
    url = git("remote", "get-url", remote).stdout.strip()
    parsed = urlparse(url)
    host = parsed.hostname or ("github.com" if url.startswith("git@github.com:") else None)
    if host != "github.com":
        raise SystemExit(
            f"{token_env} credential setup only supports github.com remotes, got {url!r}."
        )
    git("config", "--local", "credential.helper", "cache --timeout=3600")
    git(
        "credential", "approve",
        input_text=(f"protocol=https\nhost={host}\nusername=x-access-token\npassword={token}\n\n"),
    )


def switch_evidence_branch(remote: str, branch: str) -> None:
    remote_has_branch = git("ls-remote", "--exit-code", "--heads", remote, branch,
                            check=False).returncode == 0
    local_has_branch = git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}",
                           check=False).returncode == 0
    if remote_has_branch:
        git("fetch", remote, branch)
        if local_has_branch:
            git("switch", branch)
            git("merge", "--ff-only", f"{remote}/{branch}")
        else:
            git("switch", "--track", "-c", branch, f"{remote}/{branch}")
    elif local_has_branch:
        git("switch", branch)
    else:
        git("switch", "-c", branch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--branch", default="evidence/colab")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--message", default="evidence(colab): publish benchmark results")
    args = parser.parse_args(argv)

    root = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    ensure_no_tracked_code_changes()
    configure_ephemeral_github_credential(args.remote, args.token_env)
    switch_evidence_branch(args.remote, args.branch)
    paths = evidence_paths(root)
    if not paths:
        raise SystemExit("No allowlisted evidence files found under results/.")
    git("add", "-f", "--", *paths)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("No evidence changes to publish.")
        return 0
    body = (
        f"{args.message}\n\n"
        "Only allowlisted JSON, JSONL, logs, Markdown, and text evidence are included.\n\n"
        "Co-Authored-By: Codex (GPT-5) <noreply@anthropic.com>\n"
    )
    git("commit", "-F", "-", input_text=body)
    git("push", args.remote, f"HEAD:{args.branch}")
    print(f"Published {len(paths)} evidence file(s) to {args.remote}/{args.branch}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
