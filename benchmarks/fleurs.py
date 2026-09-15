"""FLEURS loading that survives the `datasets` script-removal breakage.

Background
----------
`google/fleurs` is still a script-backed dataset on the Hub. The `datasets`
library tightened this over three releases:

* `< 2.20`  — loading scripts run by default.
* `2.20–3.x` — scripts require an explicit `trust_remote_code=True`.
* `>= 4.0`   — scripts removed entirely; `trust_remote_code` itself raises.

A fresh Colab installs the newest `datasets`, so an unpinned environment fails
on `load_dataset("google/fleurs", ...)` with a `RuntimeError` that does not
obviously point at the cause. `scripts/setup.sh` therefore pins `datasets<4`,
which is the supported path and matches what the training code already used.

This module exists so the failure is *legible* rather than mysterious: it tries
each strategy in order and, if all fail, raises an error that states exactly
what to run.

Audio column shape also changed in `datasets` 4.x (a decoder object rather than
a `{"array", "sampling_rate"}` dict). `extract_audio` handles both so a future
un-pin degrades gracefully instead of throwing an AttributeError deep in a
loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["load_fleurs", "extract_audio", "FleursLoadError"]

_PARQUET_REVISION = "refs/convert/parquet"


class FleursLoadError(RuntimeError):
    """Raised when every FLEURS loading strategy fails."""


def load_fleurs(config: str, split: str):
    """Load a FLEURS split, trying each strategy the library version allows.

    ``config`` is a FLEURS language code such as ``hi_in``.
    """
    from datasets import load_dataset

    attempts: list[tuple[str, Exception]] = []

    # 1. datasets < 2.20, or a future parquet-native version of the repo.
    try:
        return load_dataset("google/fleurs", config, split=split)
    except Exception as exc:  # noqa: BLE001 - we re-raise a combined error below
        attempts.append(("plain load_dataset", exc))

    # 2. datasets 2.20 - 3.x: explicit opt-in to the loading script.
    try:
        return load_dataset(
            "google/fleurs", config, split=split, trust_remote_code=True
        )
    except Exception as exc:  # noqa: BLE001
        attempts.append(("trust_remote_code=True", exc))

    # 3. The Hub's auto-converted parquet branch. Best effort: the branch
    #    layout is not part of any stability guarantee, so this is a
    #    convenience fallback, not the supported path.
    try:
        return load_dataset(
            "google/fleurs", config, split=split, revision=_PARQUET_REVISION
        )
    except Exception as exc:  # noqa: BLE001
        attempts.append((f"revision={_PARQUET_REVISION}", exc))

    detail = "\n".join(
        f"  - {name}: {type(exc).__name__}: {exc}" for name, exc in attempts
    )
    raise FleursLoadError(
        f"Could not load google/fleurs config={config!r} split={split!r}.\n"
        f"Tried:\n{detail}\n\n"
        "google/fleurs is still a script-backed dataset, and datasets>=4.0 "
        "removed loading-script support entirely.\n"
        "Fix (this is what scripts/setup.sh pins):\n"
        '    pip install -U "datasets>=2.19,<4"\n'
        "then restart the runtime so the new version is imported."
    )


def extract_audio(row: dict) -> tuple[np.ndarray, int]:
    """Return ``(waveform_float32_mono, sampling_rate)`` from a FLEURS row.

    Handles the `datasets` 3.x dict shape and, best effort, the 4.x decoder
    object. Downmixes to mono because the explicit mel front-end expects a
    single channel.
    """
    audio = row["audio"]

    if isinstance(audio, dict):
        waveform = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
    elif hasattr(audio, "get_all_samples"):
        # datasets >= 4 AudioDecoder. Untested against a live Hub here; the
        # pinned path above is the supported one.
        samples = audio.get_all_samples()
        waveform = np.asarray(samples.data, dtype=np.float32)
        sample_rate = int(samples.sample_rate)
    else:
        raise TypeError(
            f"Unrecognised FLEURS audio column type: {type(audio)!r}. "
            'Pin with: pip install -U "datasets>=2.19,<4"'
        )

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[-1] else 1)
    return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate
