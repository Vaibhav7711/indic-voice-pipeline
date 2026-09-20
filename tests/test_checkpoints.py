"""Checkpoint resolution and torn-write detection."""

from __future__ import annotations

import json
import struct

import pytest

from benchmarks.checkpoints import (
    CheckpointError,
    best_checkpoint,
    latest_checkpoint,
    list_checkpoints,
    resolve_adapter,
    stage_checkpoint,
    verify_adapter,
)


def write_safetensors(path, n_bytes=64, truncate_to=None):
    """Minimal valid safetensors file: 8-byte length, JSON header, payload."""
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [n_bytes // 4],
            "data_offsets": [0, n_bytes],
        }
    }
    blob = json.dumps(header).encode("utf-8")
    data = struct.pack("<Q", len(blob)) + blob + (b"\x00" * n_bytes)
    if truncate_to is not None:
        data = data[:truncate_to]
    path.write_bytes(data)


def make_checkpoint(root, step, *, best=None, complete=True, tokenizer=False):
    ckpt = root / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "base_model_name_or_path": "openai/whisper-medium",
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        ckpt / "adapter_model.safetensors",
        truncate_to=None if complete else 30,
    )
    if tokenizer:
        (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    state = {"global_step": step}
    if best is not None:
        state["best_model_checkpoint"] = f"/content/run/checkpoint-{best}"
    (ckpt / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    # Training-resume artefacts that must NOT be copied when staging.
    (ckpt / "optimizer.pt").write_bytes(b"\x00" * 1024)
    (ckpt / "rng_state.pth").write_bytes(b"\x00" * 64)
    return ckpt


class TestDiscovery:
    def test_lists_sorted_numerically(self, tmp_path):
        for step in (200, 1800, 1000):
            make_checkpoint(tmp_path, step)
        assert [s for s, _ in list_checkpoints(tmp_path)] == [200, 1000, 1800]

    def test_numeric_not_lexicographic(self, tmp_path):
        """checkpoint-2000 must beat checkpoint-800, not lose to it."""
        make_checkpoint(tmp_path, 800)
        make_checkpoint(tmp_path, 2000)
        assert latest_checkpoint(tmp_path).name == "checkpoint-2000"

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        (tmp_path / "best").mkdir()
        (tmp_path / "checkpoint-notanumber").mkdir()
        assert len(list_checkpoints(tmp_path)) == 1

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="No checkpoint-"):
            latest_checkpoint(tmp_path)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="Not a directory"):
            list_checkpoints(tmp_path / "nope")


class TestBestCheckpoint:
    def test_resolves_basename_not_recorded_path(self, tmp_path):
        """trainer_state records the training machine's path, not ours."""
        make_checkpoint(tmp_path, 1600, best=1600)
        make_checkpoint(tmp_path, 1800, best=1600)
        make_checkpoint(tmp_path, 2000, best=1600)
        assert best_checkpoint(tmp_path).name == "checkpoint-1600"

    def test_rotated_away_best_gives_actionable_error(self, tmp_path):
        """The exact situation save_total_limit creates."""
        make_checkpoint(tmp_path, 1600, best=1400)
        make_checkpoint(tmp_path, 1800, best=1400)
        with pytest.raises(CheckpointError, match="rotated away by save_total_limit"):
            best_checkpoint(tmp_path)

    def test_no_best_recorded(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        with pytest.raises(CheckpointError, match="No trainer_state.json records"):
            best_checkpoint(tmp_path)

    def test_newest_state_file_wins(self, tmp_path):
        make_checkpoint(tmp_path, 1000, best=1000)
        make_checkpoint(tmp_path, 2000, best=2000)
        assert best_checkpoint(tmp_path).name == "checkpoint-2000"


class TestResolveAdapter:
    def test_latest(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        make_checkpoint(tmp_path, 200)
        assert resolve_adapter(tmp_path, "latest").name == "checkpoint-200"

    def test_final_export(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        (tmp_path / "best").mkdir()
        assert resolve_adapter(tmp_path, "final").name == "best"

    def test_final_missing_is_actionable(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        with pytest.raises(CheckpointError, match="Training may not have"):
            resolve_adapter(tmp_path, "final")

    def test_bad_policy(self, tmp_path):
        with pytest.raises(ValueError, match="latest/best/final"):
            resolve_adapter(tmp_path, "newest")


class TestVerifyAdapter:
    def test_valid(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100)
        info = verify_adapter(ckpt)
        assert info["r"] == 16
        assert info["peft_type"] == "LORA"
        assert info["has_tokenizer"] is False

    def test_tokenizer_detected(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100, tokenizer=True)
        assert verify_adapter(ckpt)["has_tokenizer"] is True

    def test_missing_weights(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100)
        (ckpt / "adapter_model.safetensors").unlink()
        with pytest.raises(CheckpointError, match="missing adapter_model"):
            verify_adapter(ckpt)

    def test_missing_config(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100)
        (ckpt / "adapter_config.json").unlink()
        with pytest.raises(CheckpointError, match="missing adapter_config"):
            verify_adapter(ckpt)

    def test_truncated_weights_detected(self, tmp_path):
        """A checkpoint read while the Trainer is still writing it."""
        ckpt = make_checkpoint(tmp_path, 2000, complete=False)
        with pytest.raises(CheckpointError, match="truncated"):
            verify_adapter(ckpt)

    def test_payload_truncation_detected(self, tmp_path):
        """Header intact, tensor buffer cut short — the common torn write."""
        ckpt = make_checkpoint(tmp_path, 2000)
        weights = ckpt / "adapter_model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-16])
        with pytest.raises(CheckpointError, match="header declares at least"):
            verify_adapter(ckpt)

    def test_empty_file(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100)
        (ckpt / "adapter_model.safetensors").write_bytes(b"")
        with pytest.raises(CheckpointError, match="too short"):
            verify_adapter(ckpt)

    def test_garbage_header(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, 100)
        blob = b"not json at all"
        (ckpt / "adapter_model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob
        )
        with pytest.raises(CheckpointError, match="unparseable header"):
            verify_adapter(ckpt)


class TestStageCheckpoint:
    def test_copies_inference_files_only(self, tmp_path):
        ckpt = make_checkpoint(tmp_path / "drive", 2000, tokenizer=True)
        staged = stage_checkpoint(ckpt, tmp_path / "local")

        names = {p.name for p in staged.iterdir()}
        assert "adapter_model.safetensors" in names
        assert "adapter_config.json" in names
        assert "tokenizer_config.json" in names
        # Training-resume artefacts are large and useless for inference.
        assert "optimizer.pt" not in names
        assert "rng_state.pth" not in names

    def test_source_untouched(self, tmp_path):
        """A live training job's output directory must never be modified."""
        source = tmp_path / "drive"
        ckpt = make_checkpoint(source, 2000)
        before = sorted(p.name for p in ckpt.iterdir())
        stage_checkpoint(ckpt, tmp_path / "local")
        assert sorted(p.name for p in ckpt.iterdir()) == before

    def test_refuses_torn_source(self, tmp_path):
        ckpt = make_checkpoint(tmp_path / "drive", 2000, complete=False)
        with pytest.raises(CheckpointError):
            stage_checkpoint(ckpt, tmp_path / "local")

    def test_staged_dir_named_after_checkpoint(self, tmp_path):
        ckpt = make_checkpoint(tmp_path / "drive", 1800)
        assert stage_checkpoint(ckpt, tmp_path / "local").name == "checkpoint-1800"


class TestHubResolution:
    def test_owner_name_that_is_not_on_disk_is_a_hub_id(self, tmp_path):
        from benchmarks.checkpoints import is_hub_repo_id

        assert is_hub_repo_id("Hugme6969/whisper-medium-hindi-lora")
        assert not is_hub_repo_id("results/whisper-lora-hi-full/best")
        assert not is_hub_repo_id("/content/adapters/best")
        local = tmp_path / "owner" / "name"
        local.mkdir(parents=True)
        assert not is_hub_repo_id(str(local))

    def test_local_path_passes_through_without_network(self, tmp_path, monkeypatch):
        import benchmarks.checkpoints as ck

        monkeypatch.setattr(ck, "is_hub_repo_id", lambda v: False)
        assert ck.resolve_hub_adapter(tmp_path) == tmp_path
