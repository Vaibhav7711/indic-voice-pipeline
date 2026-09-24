"""Evidence publisher only stages the narrow, reviewable result allowlist."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "scripts" / "publish_colab_evidence.py"
_spec = importlib.util.spec_from_file_location("publish_colab_evidence", _path)
publisher = importlib.util.module_from_spec(_spec)
sys.modules["publish_colab_evidence"] = publisher
_spec.loader.exec_module(publisher)


def test_evidence_paths_allow_metrics_but_never_audio_or_weights(tmp_path):
    (tmp_path / "results/eval/run").mkdir(parents=True)
    (tmp_path / "results/eval/run/metrics.json").write_text("{}")
    (tmp_path / "results/eval/run/predictions.jsonl").write_text("{}\n")
    (tmp_path / "results/eval/run/clip.wav").write_bytes(b"audio")
    (tmp_path / "results/eval/run/adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "results/live").mkdir(parents=True)
    (tmp_path / "results/live/turns.jsonl").write_text("{}\n")
    (tmp_path / "results/bench_manifest.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not evidence")

    assert publisher.evidence_paths(tmp_path) == [
        "results/bench_manifest.json",
        "results/eval/run/metrics.json",
        "results/eval/run/predictions.jsonl",
        "results/live/turns.jsonl",
    ]
