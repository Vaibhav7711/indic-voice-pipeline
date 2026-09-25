"""Start an OpenAI-compatible LLM server and wait until it is actually ready.

The agent turn reaches a serving engine through `--llm-engine http`
(`llm/engines/http_engine.py`). Something has to start that engine, and the
"something" cannot be a bare `uvicorn ... &` for one reason: an engine that
captures CUDA graphs and JITs Triton kernels at startup answers `503` on
`/ready` until warmup finishes, which on a cold cache is tens of seconds. A
pipeline that starts talking to it immediately records a first turn several
seconds slower than the steady state and then averages that into a benchmark.
Warmup is a startup cost, not a serving cost, and this script is where the
distinction is enforced.

Built for `Vaibhav7711/full-inference-engine` -- paged KV cache, continuous
batching, CUDA-graphed decode, and a measured per-architecture backend policy.
It works with any server exposing `/v1/completions` with `stream: true`; only
`--app` and the readiness path are engine-specific, and both are flags.

    # default: this repo's factory, Qwen3-4B with a pool sized for one stream
    python scripts/serve_llm.py --engine-root ../full-inference-engine

    # a profile the engine A/B-ed on a specific architecture, as shipped
    python scripts/serve_llm.py --engine-root ../full-inference-engine \
        --app engine.server.api:create_rtx4060_flash_app

    # then, in another process
    python scripts/live_agent.py --llm-engine http \
        --llm-base-url http://127.0.0.1:8000/v1

**Why a separate process, on one GPU.** Whisper-medium fp16 (1.5 GiB) plus
Qwen3-4B fp16 (~7.5 GiB) plus the engine's paged KV blocks share one card.
Two processes mean two CUDA contexts (~300 MiB each) and no shared allocator,
which is a real cost -- but it is the only arrangement where the engine owns
its own block pool and graph memory without the ASR allocator fragmenting it
underneath. `pipeline/memory.py` has the budget.

Blocks are the thing to size: `num_blocks * block_size` is the total KV token
capacity across all concurrent requests. At Qwen3-4B's 144 KiB per cached
token the engine's 1024x16 default costs 2.25 GiB; a voice agent is one
stream with an <=800-token prompt, so `scripts/llm_server_app.py` defaults to
512x16 = 8192 tokens (1.125 GiB) and prints the arithmetic before allocating.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

#: This repo's configurable factory, so a model and a pool size can be chosen.
#: The engine's own zero-argument profiles stay available through `--app`, and
#: are the right choice on an architecture they were measured on.
DEFAULT_APP = "scripts.llm_server_app:create"
DEFAULT_PORT = 8000


def probe(url: str, timeout: float = 2.0) -> tuple[int, dict | str]:
    """GET `url`, returning (status, parsed body). A 503 body is diagnostic.

    This engine's `/ready` returns the service snapshot alongside the status,
    so a server that is still loading says so rather than just refusing.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        try:
            body = error.read().decode("utf-8", "replace")
        except OSError:
            body = ""
    except (urllib.error.URLError, OSError) as error:
        return 0, f"{type(error).__name__}: {error}"
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def wait_until_ready(base: str, *, timeout_s: float, path: str = "/ready",
                     process: subprocess.Popen | None = None,
                     interval_s: float = 1.0, log=print) -> dict:
    """Poll until the server reports ready, or the deadline passes.

    Returns the readiness body. Raises `RuntimeError` on timeout, or as soon as
    the server process exits -- polling a dead process until the deadline turns
    a crash on startup (a missing Triton wheel, an unsupported head geometry,
    which this engine refuses at load *with the reason*) into a slow, silent
    failure.
    """
    url = f"{base.rstrip('/')}{path}"
    deadline = time.monotonic() + timeout_s
    waited = 0.0
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"the server exited with code {process.returncode} during warmup; "
                f"its output above says why",
            )
        status, body = probe(url)
        if status == 200:
            log(f"ready after {waited:.1f}s: {body}")
            return body if isinstance(body, dict) else {"status": "ready"}
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{url} did not report ready within {timeout_s:.0f}s; "
                f"last status {status}, body {body!r}",
            )
        if waited and waited % 10 < interval_s:
            log(f"  waiting for warmup ({waited:.0f}s): status {status}, {body!r}")
        time.sleep(interval_s)
        waited += interval_s


def server_command(app: str, *, port: int, host: str, extra: list[str]) -> list[str]:
    """`uvicorn --factory`, because the engine's app factories take no arguments.

    The measured profiles (`create_rtx4060_flash_app`, and the T4 x2
    speculative one) are zero-argument factories precisely so a serving
    configuration that was A/B-ed on a device cannot drift from what is run.
    Pass `--app` to pick one; do not hand-assemble its arguments here.
    """
    return [sys.executable, "-m", "uvicorn", app, "--factory",
            "--host", host, "--port", str(port), *extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine-root", default=os.getenv("LLM_ENGINE_ROOT"),
                        help="checkout of the serving engine; prepended to PYTHONPATH")
    parser.add_argument("--app", default=DEFAULT_APP,
                        help=f"uvicorn --factory target (default {DEFAULT_APP}, "
                             "this repo's env-configured factory). Use "
                             "engine.server.api:create_rtx4060_flash_app to serve "
                             "a profile that was A/B-ed on that architecture; the "
                             "--model and pool flags below are then ignored, "
                             "because that is the point of those profiles")
    parser.add_argument("--model", default=None,
                        help="checkpoint to serve (default: the factory's own). "
                             "Qwen3-4B fp16 needs ~7.5 GiB of weights, so it fits "
                             "a 15 GB T4 beside Whisper but not an 8 GB card")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="paged KV blocks. num_blocks x block_size is the KV "
                             "token capacity across all concurrent requests; a "
                             "voice agent is one stream with an <=800-token prompt")
    parser.add_argument("--block-size", type=int, default=None,
                        help="tokens per page; FlashAttention-2 paged KV needs 256")
    parser.add_argument("--max-active", type=int, default=None)
    parser.add_argument("--graph-buckets", default=None,
                        help="comma-separated batch sizes to capture graphs for; "
                             "each costs memory, and buckets above --max-active "
                             "are dropped")
    parser.add_argument("--dtype", default=None, help="float16 / bfloat16 / auto")
    parser.add_argument("--decode-attention", default=None,
                        help="named backend; the engine raises with a reason "
                             "rather than falling back if it cannot run here")
    parser.add_argument("--prefill-attention", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ready-path", default="/ready",
                        help="readiness route; /health is liveness and answers sooner")
    parser.add_argument("--ready-timeout", type=float, default=600.0,
                        help="graph capture and Triton JIT on a cold cache are slow")
    parser.add_argument("--wait-only", action="store_true",
                        help="do not start a server; just wait on one already running")
    parser.add_argument("--uvicorn-arg", action="append", default=[],
                        dest="extra", help="passed through to uvicorn; repeatable")
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    if args.wait_only:
        wait_until_ready(base, timeout_s=args.ready_timeout, path=args.ready_path)
        print(f"base url: {base}/v1")
        return 0

    environment = dict(os.environ)
    overrides = {
        "LLM_SERVER_MODEL": args.model,
        "LLM_SERVER_NUM_BLOCKS": args.num_blocks,
        "LLM_SERVER_BLOCK_SIZE": args.block_size,
        "LLM_SERVER_MAX_ACTIVE": args.max_active,
        "LLM_SERVER_GRAPH_BUCKETS": args.graph_buckets,
        "LLM_SERVER_DTYPE": args.dtype,
        "LLM_SERVER_DECODE_ATTENTION": args.decode_attention,
        "LLM_SERVER_PREFILL_ATTENTION": args.prefill_attention,
    }
    chosen = {name: str(value) for name, value in overrides.items() if value is not None}
    if chosen and args.app != DEFAULT_APP:
        # Silently ignoring them would mean the recorded configuration and the
        # served one disagree, which is the failure this whole file is about.
        parser.error(
            f"--app {args.app} is a zero-argument factory and cannot receive "
            f"{sorted(chosen)}. Either drop those flags, or use the default "
            f"factory ({DEFAULT_APP}) which reads them.",
        )
    environment.update(chosen)
    if args.engine_root:
        root = os.path.abspath(os.path.expanduser(args.engine_root))
        if not os.path.isdir(root):
            parser.error(f"--engine-root {root} is not a directory")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root

    command = server_command(args.app, port=args.port, host=args.host, extra=args.extra)
    print("starting:", " ".join(command), flush=True)
    if chosen:
        print("configuration:", chosen, flush=True)
    if args.app == DEFAULT_APP:
        from scripts.llm_server_app import describe, resolve_config

        # Printed before the allocation, so an out-of-memory death is a number
        # someone chose rather than a surprise.
        print("memory:", describe(resolve_config(environment)), flush=True)
    process = subprocess.Popen(command, env=environment)
    try:
        body = wait_until_ready(base, timeout_s=args.ready_timeout, path=args.ready_path,
                                process=process)
    except (RuntimeError, KeyboardInterrupt) as error:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"base url: {base}/v1 -- point the agent at it with "
          f"--llm-engine http --llm-base-url {base}/v1", flush=True)
    print(f"readiness: {body}", flush=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Drain rather than kill: the engine finishes in-flight requests and
        # reports 503 on /ready while it does.
        process.send_signal(signal.SIGINT)
        try:
            return process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
