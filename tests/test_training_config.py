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
