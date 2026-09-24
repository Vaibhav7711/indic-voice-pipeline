"""Training presets resolve to the ledger's recipe; flags override presets.

CPU only — parses arguments and builds a TrainingConfig, nothing else.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from asr.training import lora  # noqa: E402


def _config(*argv):
    return lora.build_config(lora.parse_args(list(argv)))


def test_v1_preset_matches_experiments_ledger():
    c = _config("--preset", "v1")
    assert c.model_name == "openai/whisper-medium"
    assert (c.dataset, c.language, c.indicvoices_samples) == ("fleurs", "hi", 5000)
    assert (c.batch_size, c.gradient_accumulation_steps) == (2, 4)
    assert (c.lora_rank, c.lora_alpha, c.lora_dropout) == (16, 32, 0.05)
    assert set(c.target_modules) == {"q_proj", "k_proj", "v_proj", "out_proj"}
    assert (c.save_steps, c.save_total_limit, c.epochs) == (200, 3, 3)


def test_v2_turbo_differs_from_v1_only_in_base_model():
    v1, v2 = _config("--preset", "v1"), _config("--preset", "v2-turbo")
    assert v2.model_name == "openai/whisper-large-v3-turbo"
    for field in ("dataset", "language", "indicvoices_samples", "epochs", "batch_size",
                  "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout",
                  "target_modules", "eval_steps", "save_steps", "save_total_limit"):
        assert getattr(v1, field) == getattr(v2, field), field


def test_flags_override_preset():
    c = _config("--preset", "v1", "--indicvoices-samples", "0", "--epochs", "1",
                "--output-dir", "/tmp/x")
    assert c.indicvoices_samples == 0
    assert c.epochs == 1
    assert c.output_dir == "/tmp/x"
    assert c.model_name == "openai/whisper-medium"


def test_small_preset_is_the_superseded_run():
    c = _config("--preset", "small-fleurs")
    assert c.model_name == "openai/whisper-small"
    assert c.indicvoices_samples == 0
    assert (c.batch_size, c.gradient_accumulation_steps, c.save_steps) == (4, 2, 100)


def test_defaults_without_preset_are_the_dataclass_defaults():
    c = _config()
    assert c == lora.TrainingConfig()


def test_fleurs_training_load_uses_compatibility_shim(monkeypatch):
    calls = []

    class Split:
        column_names = ["audio", "transcription", "speaker_id"]

        def __init__(self, rows):
            self.rows = rows
            self.removed = None

        def map(self, function, remove_columns):
            self.removed = remove_columns
            return [function(row) for row in self.rows]

    train = Split([{"audio": "train.wav", "transcription": "नमस्ते", "speaker_id": 1}])
    validation = Split([{"audio": "valid.wav", "transcription": "धन्यवाद", "speaker_id": 2}])

    def fake_load_fleurs(config, split):
        calls.append((config, split))
        return {"train": train, "validation": validation}[split]

    monkeypatch.setattr("benchmarks.fleurs.load_fleurs", fake_load_fleurs)
    loaded_train, loaded_eval = lora.load_primary(lora.TrainingConfig())

    assert calls == [("hi_in", "train"), ("hi_in", "validation")]
    assert loaded_train == [{"audio": "train.wav", "sentence": "नमस्ते"}]
    assert loaded_eval == [{"audio": "valid.wav", "sentence": "धन्यवाद"}]


def test_hub_checkpoint_steps_only_counts_complete_checkpoints():
    files = [
        "checkpoint-200/adapter_model.safetensors", "checkpoint-200/trainer_state.json",
        "checkpoint-400/adapter_model.safetensors",              # upload interrupted
        "checkpoint-600/trainer_state.json", "checkpoint-600/optimizer.pt",
        "best/adapter_model.safetensors", "train_config.json",
    ]
    assert lora.hub_checkpoint_steps(files) == [200, 600]
    assert lora.hub_checkpoint_steps([]) == []


def test_hub_flags_reach_the_config():
    c = _config("--preset", "v2-turbo", "--hub-repo", "me/ckpts", "--hub-keep-checkpoints", "3")
    assert c.hub_repo == "me/ckpts" and c.hub_keep_checkpoints == 3
    assert _config("--preset", "v2-turbo").hub_repo is None


def test_ensure_hub_repo_uses_existing_and_explains_create_failure():
    from types import SimpleNamespace

    from huggingface_hub.errors import HfHubHTTPError

    class Api:
        def __init__(self, exists):
            self.exists, self.created = exists, False

        def repo_exists(self, repo_id):
            return self.exists

        def create_repo(self, repo_id, private=True, exist_ok=True):
            self.created = True
            resp = SimpleNamespace(status_code=403, headers={}, text="", url="", request=None)
            raise HfHubHTTPError("403 Forbidden", response=resp)

    ok = Api(exists=True)
    lora.ensure_hub_repo(ok, "me/ckpt")
    assert ok.created is False

    with pytest.raises(SystemExit, match="huggingface.co/new"):
        lora.ensure_hub_repo(Api(exists=False), "me/ckpt")


def test_runtime_geometry_exposes_the_batch_actually_in_force(monkeypatch):
    import torch

    cfg = _config("--preset", "v2-turbo")          # batch 2 x accum 4 = 8 intended
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    one = lora.runtime_geometry(cfg, 7115)
    assert one["effective_batch"] == 8 and one["expected_total_steps"] == 2667

    # Kaggle "T4 x2": Trainer's DataParallel doubles the batch and halves the steps.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    two = lora.runtime_geometry(cfg, 7115)
    assert two["effective_batch"] == 16
    assert two["intended_effective_batch"] == 8
    assert two["expected_total_steps"] == 1332          # what v2 actually ran (1335)
