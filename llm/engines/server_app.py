"""A configurable app factory for the serving engine, owned by this repo.

**Why this lives under `llm/engines/` and not in `scripts/`.** It was in
`scripts/` and could not be imported. `full-inference-engine` also has a
top-level `scripts/` directory, and that one is a regular package with an
`__init__.py` while this repo's is a PEP 420 namespace package. Serving puts
both checkouts on `PYTHONPATH` with the engine first, and a regular package
found earlier on the path wins outright -- so `import scripts.server_app`
resolved to the engine's `scripts/`, which has no such module, and uvicorn
reported only "Could not import module". Five top-level names collide between
the two repos (`benchmarks`, `docs`, `results`, `scripts`, `tests`); `llm` is
unique to this one, so a module here cannot be shadowed.

The engine's own factories take no arguments on purpose: `create_rtx4060_flash_app`
and the T4 x2 speculative profile encode settings an A/B chose *on that
architecture*, and a zero-argument factory cannot drift from what was measured.
That is the right design for those profiles and the wrong one for a model they
do not cover.

Qwen3-4B needs a pool sized for it. Its KV is 36 layers x 2 x 8 KV heads x 128
head dim x 2 bytes = **144 KiB per token**, against 0.6B's much cheaper cache,
so the engine's default 1024 x 16 = 16384-token pool costs 2.25 GiB for 4B
where it is loose change for 0.6B. There is no argument to `uvicorn --factory`
to say so, hence this module: it reads the configuration from the environment,
so `scripts/serve_llm.py` can set it and the process still starts with a
zero-argument factory.

    LLM_SERVER_MODEL=Qwen/Qwen3-4B LLM_SERVER_NUM_BLOCKS=512 \
        uvicorn llm.engines.server_app:create --factory

Nothing here decides a serving policy. It passes configuration through and
refuses what it cannot pass through, so a typo in an environment variable
fails at startup rather than silently serving a different configuration than
the one recorded next to the numbers.
"""

from __future__ import annotations

import os

__all__ = ["create", "kv_bytes_per_token", "pool_bytes", "resolve_config",
           "MODEL_GEOMETRY"]

#: Published geometry for the checkpoints this pipeline serves, used only to
#: report what a pool will cost before it is allocated. Source: the models'
#: own configs. Anything absent falls back to reporting `None` rather than
#: guessing -- a made-up VRAM figure is worse than no figure, because it is
#: acted on.
MODEL_GEOMETRY: dict[str, dict] = {
    # layers, kv_heads, head_dim
    "Qwen/Qwen3-0.6B": {"layers": 28, "kv_heads": 8, "head_dim": 128},
    "Qwen/Qwen3-1.7B": {"layers": 28, "kv_heads": 8, "head_dim": 128},
    "Qwen/Qwen3-4B": {"layers": 36, "kv_heads": 8, "head_dim": 128},
}

#: Approximate fp16 checkpoint sizes, for the same reporting purpose.
MODEL_WEIGHT_GIB: dict[str, float] = {
    "Qwen/Qwen3-0.6B": 1.2,
    "Qwen/Qwen3-1.7B": 3.8,
    "Qwen/Qwen3-4B": 7.5,
}

DEFAULTS = {
    "model": "Qwen/Qwen3-4B",
    # A voice agent is one stream. 512 x 16 = 8192 KV tokens is 10x a
    # 800-token dialogue prompt and costs 1.125 GiB at 4B's 144 KiB/token,
    # against 2.25 GiB for the engine's 1024-block default.
    "num_blocks": 512,
    "block_size": 16,
    # One in-flight request, plus one so a barge-in's replacement turn does
    # not queue behind the request it cancelled.
    "max_active": 2,
    # Graph capture costs memory per bucket, and buckets above max_active are
    # dropped by the engine anyway.
    "graph_buckets": (1, 2),
    "dtype": "float16",
    "max_prompt_tokens": 4096,
}


def kv_bytes_per_token(model: str, *, dtype_bytes: int = 2) -> int | None:
    """KV bytes one cached token costs, or `None` for an unknown checkpoint."""
    geometry = MODEL_GEOMETRY.get(model)
    if geometry is None:
        return None
    return (2 * geometry["layers"] * geometry["kv_heads"]
            * geometry["head_dim"] * dtype_bytes)


def pool_bytes(model: str, *, num_blocks: int, block_size: int,
               dtype_bytes: int = 2) -> int | None:
    """What the whole paged pool will cost, before it is allocated.

    Reported at startup so an out-of-memory death is a number someone chose
    rather than a surprise. `None` when the geometry is unknown.
    """
    per_token = kv_bytes_per_token(model, dtype_bytes=dtype_bytes)
    return None if per_token is None else per_token * num_blocks * block_size


def _int(environ: dict, name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _buckets(environ: dict, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} must be a comma-separated integer list") from None
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive integers, got {raw!r}")
    return values


def resolve_config(environ: dict | None = None) -> dict:
    """The configuration this process will serve with, from the environment.

    Separated from `create` so it is testable without the engine installed,
    and so the resolved values can be printed and recorded next to the
    measurements they produced.
    """
    environ = os.environ if environ is None else environ
    config = {
        "model_name": environ.get("LLM_SERVER_MODEL") or DEFAULTS["model"],
        "num_blocks": _int(environ, "LLM_SERVER_NUM_BLOCKS", DEFAULTS["num_blocks"]),
        "block_size": _int(environ, "LLM_SERVER_BLOCK_SIZE", DEFAULTS["block_size"]),
        "max_active": _int(environ, "LLM_SERVER_MAX_ACTIVE", DEFAULTS["max_active"]),
        "graph_buckets": _buckets(environ, "LLM_SERVER_GRAPH_BUCKETS",
                                  DEFAULTS["graph_buckets"]),
        "dtype": environ.get("LLM_SERVER_DTYPE") or DEFAULTS["dtype"],
        "max_prompt_tokens": _int(environ, "LLM_SERVER_MAX_PROMPT_TOKENS",
                                  DEFAULTS["max_prompt_tokens"]),
    }
    for key, variable in (("decode_attention", "LLM_SERVER_DECODE_ATTENTION"),
                          ("prefill_attention", "LLM_SERVER_PREFILL_ATTENTION")):
        value = environ.get(variable)
        if value:
            config[key] = value
    return config


def describe(config: dict) -> dict:
    """The pool's cost and the weights' cost, for the startup log.

    `None` where the checkpoint's geometry is not recorded here, rather than
    a plausible number: this figure is used to decide whether a model fits
    beside Whisper on one card.
    """
    model = config["model_name"]
    tokens = config["num_blocks"] * config["block_size"]
    bytes_total = pool_bytes(model, num_blocks=config["num_blocks"],
                             block_size=config["block_size"])
    weights = MODEL_WEIGHT_GIB.get(model)
    return {
        "model": model,
        "kv_tokens": tokens,
        "kv_bytes_per_token": kv_bytes_per_token(model),
        "kv_pool_gib": None if bytes_total is None else round(bytes_total / 1024**3, 3),
        "weights_gib_fp16": weights,
        "resident_gib_estimate": (
            None if bytes_total is None or weights is None
            else round(weights + bytes_total / 1024**3, 3)
        ),
        "note": ("add ~0.3 GiB for this process's CUDA context, plus graph and "
                 "activation memory; Whisper runs in the other process"),
    }


def create():
    """The `uvicorn --factory` target. Imports the engine only when called."""
    from engine.server.api import create_app

    config = resolve_config()
    print(f"serving configuration: {config}", flush=True)
    print(f"memory: {describe(config)}", flush=True)
    return create_app(**config)
