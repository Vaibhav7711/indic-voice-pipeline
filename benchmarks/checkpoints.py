"""Find, verify and stage Trainer checkpoints.

Three problems this solves, all of which bite when evaluation runs against a
training job that is still going.

**Checkpoints move.** `save_total_limit` rotates old checkpoints away, and
`load_best_model_at_end=True` protects the best one from rotation. So the set
of directories on disk changes as training advances, and any hardcoded
`checkpoint-1400` path eventually becomes a `FileNotFoundError`. Resolve by
policy (`latest` / `best`) instead of by number.

**Checkpoints can be read mid-write.** If evaluation opens
`checkpoint-2000/adapter_model.safetensors` while the Trainer is still writing
it, the file is truncated and the failure surfaces much later as a confusing
deserialisation error. `verify_adapter` parses the safetensors header and
checks the declared tensor extents against the real file size, which catches a
torn write immediately and cheaply.

**Drive is slow and shared.** Reading model weights over the Drive FUSE mount
on every run is slow, and it puts read load on the same mount the training job
is writing checkpoints to. `stage_checkpoint` copies the adapter to local disk
once and verifies it there.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

__all__ = [
    "CheckpointError",
    "list_checkpoints",
    "latest_checkpoint",
    "best_checkpoint",
    "resolve_adapter",
    "resolve_hub_adapter",
    "is_hub_repo_id",
    "verify_adapter",
    "stage_checkpoint",
]

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

#: PEFT writes these two; everything else in a Trainer checkpoint is optional
#: as far as inference is concerned.
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, incomplete or corrupt."""


_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_hub_repo_id(value: str | Path) -> bool:
    """``owner/name`` that does not exist on disk is a Hugging Face Hub id."""
    text = str(value)
    return bool(_HUB_ID_RE.match(text)) and not Path(text).exists()


def resolve_hub_adapter(value: str | Path) -> Path:
    """Return a local adapter directory for a path or a Hub repo id.

    A local path is returned unchanged. A Hub id is downloaded (only the files
    an adapter needs, into the normal HF cache) so that every caller —
    ``load_whisper``, ``asr_eval``, the smoke scripts — can take either form.
    """
    if not is_hub_repo_id(value):
        return Path(value)
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=str(value),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md"],
    )
    return Path(local)


def list_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """Return ``(step, path)`` for every ``checkpoint-N`` directory, ascending."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise CheckpointError(f"Not a directory: {output_dir}")

    found = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match:
            found.append((int(match.group(1)), child))
    return sorted(found)


def latest_checkpoint(output_dir: str | Path) -> Path:
    """Highest-numbered checkpoint. Newest, not necessarily best."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        raise CheckpointError(f"No checkpoint-* directories under {output_dir}")
    return checkpoints[-1][1]


def best_checkpoint(output_dir: str | Path) -> Path:
    """Best checkpoint according to the Trainer's own bookkeeping.

    `trainer_state.json` records `best_model_checkpoint` as the path from the
    machine that trained, which is usually wrong locally (a Colab path, or a
    directory since rotated away). Only the basename is trusted, and it is
    re-resolved against `output_dir`.
    """
    output_dir = Path(output_dir)
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        raise CheckpointError(f"No checkpoint-* directories under {output_dir}")

    # Prefer the newest state file; older ones name older bests.
    for _, path in reversed(checkpoints):
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recorded = state.get("best_model_checkpoint")
        if not recorded:
            continue
        candidate = output_dir / Path(recorded).name
        if candidate.is_dir():
            return candidate
        raise CheckpointError(
            f"trainer_state.json names best checkpoint {Path(recorded).name!r}, "
            f"but it is not present under {output_dir}. It was most likely "
            "rotated away by save_total_limit. Available: "
            + ", ".join(p.name for _, p in checkpoints)
        )

    raise CheckpointError(
        f"No trainer_state.json records a best checkpoint under {output_dir}"
    )


def resolve_adapter(output_dir: str | Path, prefer: str = "latest") -> Path:
    """Resolve an adapter directory by policy rather than by step number.

    ``prefer`` is ``latest``, ``best``, or ``final`` (the exported ``best/``
    directory that `lora.py` writes when training completes).
    """
    output_dir = Path(output_dir)
    if prefer == "final":
        final = output_dir / "best"
        if not final.is_dir():
            raise CheckpointError(
                f"No exported {final} directory. Training may not have "
                "finished; use prefer='latest' or 'best' instead."
            )
        return final
    if prefer == "latest":
        return latest_checkpoint(output_dir)
    if prefer == "best":
        return best_checkpoint(output_dir)
    raise ValueError(f"prefer must be latest/best/final, got {prefer!r}")


def _safetensors_payload_size(path: Path) -> int:
    """Smallest file size consistent with the safetensors header.

    Layout is: 8-byte little-endian header length, then that many bytes of
    JSON, then the tensor buffer. Each tensor records ``data_offsets`` into
    that buffer, so the maximum end offset gives the expected total size. A
    file shorter than this was truncated — which is exactly what a checkpoint
    read mid-write looks like.
    """
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            raise CheckpointError(f"{path} is too short to be a safetensors file")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 200_000_000:
            raise CheckpointError(f"{path} has an implausible header length")
        header_bytes = handle.read(header_len)
        if len(header_bytes) < header_len:
            raise CheckpointError(f"{path} is truncated inside its header")

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path} has an unparseable header: {exc}") from exc

    end = 0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
            end = max(end, int(offsets[1]))
    return 8 + header_len + end


def verify_adapter(path: str | Path) -> dict:
    """Check an adapter directory is complete and not mid-write.

    Returns a small summary dict. Raises ``CheckpointError`` on any problem.
    """
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"Not a directory: {path}")

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise CheckpointError(
            f"{path} is missing {', '.join(missing)}. "
            "It is not a usable PEFT adapter directory."
        )

    weights = path / "adapter_model.safetensors"
    actual = weights.stat().st_size
    expected = _safetensors_payload_size(weights)
    if actual < expected:
        raise CheckpointError(
            f"{weights} is {actual} bytes but its header declares at least "
            f"{expected}. The file is truncated — most likely it was read "
            "while the Trainer was still writing it. Wait for the save to "
            "finish, or use the previous checkpoint."
        )

    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "weights_bytes": actual,
        "has_tokenizer": (path / "tokenizer_config.json").is_file(),
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "base_model": config.get("base_model_name_or_path"),
    }


def stage_checkpoint(
    source: str | Path, dest_root: str | Path = "/content/adapters"
) -> Path:
    """Copy an adapter out of Drive to local disk, then verify it there.

    Only inference-relevant files are copied: optimizer, scheduler, RNG and
    scaler state are training-resume artefacts and can be hundreds of
    megabytes. Nothing is written back to the source, so a live training job's
    output directory is never modified.
    """
    source = Path(source)
    verify_adapter(source)

    dest = Path(dest_root) / source.name
    dest.mkdir(parents=True, exist_ok=True)

    wanted = {
        *REQUIRED_FILES,
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "added_tokens.json",
        "README.md",
    }
    for name in wanted:
        src_file = source / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)

    verify_adapter(dest)
    return dest
