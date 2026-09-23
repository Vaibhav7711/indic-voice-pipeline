"""Resumability of the benchmark driver.

A hosted-notebook session usually dies before the suite finishes, so
re-running must continue rather than restart, and an *interrupted* step must
not be mistaken for a finished one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_all.py"
_spec = importlib.util.spec_from_file_location("bench_all_script", _path)
bench_all = importlib.util.module_from_spec(_spec)
sys.modules["bench_all_script"] = bench_all
_spec.loader.exec_module(bench_all)


def test_step_with_no_output_always_reruns():
    assert bench_all.is_done({"out": None}) is False


def test_missing_output_is_not_done(tmp_path):
    assert bench_all.is_done({"out": tmp_path / "nope"}) is False
    assert bench_all.is_done({"out": tmp_path / "nope.json"}) is False


def test_json_output_is_done_when_the_file_exists(tmp_path):
    target = tmp_path / "compare.json"
    target.write_text("{}")
    assert bench_all.is_done({"out": target}) is True


def test_interrupted_step_leaves_a_directory_but_is_not_done(tmp_path):
    """The directory exists because the harness created it before crashing;
    treating that as done would silently skip a step that never ran."""
    out = tmp_path / "eval-run"
    out.mkdir()
    (out / "run_config.json").write_text("{}")      # written first, not last
    assert bench_all.is_done({"out": out}) is False

    (out / "metrics.json").write_text("{}")         # written last
    assert bench_all.is_done({"out": out}) is True


@pytest.mark.parametrize("name", ["metrics.json", "summary.json", "report.json", "model.bin"])
def test_each_harness_final_output_counts(tmp_path, name):
    out = tmp_path / name.replace(".", "-")
    out.mkdir()
    (out / name).write_text("x")
    assert bench_all.is_done({"out": out}) is True


def test_grid_harness_with_per_config_subdirectories_is_done(tmp_path):
    out = tmp_path / "streaming_eval"
    (out / "default").mkdir(parents=True)
    (out / "default" / "metrics.json").write_text("{}")
    assert bench_all.is_done({"out": out}) is True


def test_mirror_copies_evidence_but_not_weights_or_audio(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text("{}")
    (out / "clip.wav").write_bytes(b"\x00" * 10)
    (out / "model.bin").write_bytes(b"\x00" * 10)
    mirror = tmp_path / "drive"
    mirror.mkdir()

    where = bench_all.mirror_output({"out": out}, mirror)
    assert where and (mirror / "run" / "metrics.json").is_file()
    assert not (mirror / "run" / "clip.wav").exists(), "audio must not be mirrored"
    assert not (mirror / "run" / "model.bin").exists(), "weights must not be mirrored"


def test_mirror_failure_is_reported_not_raised(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text("{}")
    # A file where the mirror directory should be: copytree cannot proceed.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    result = bench_all.mirror_output({"out": out}, blocked)
    assert result is not None and "mirror failed" in result


def test_skipped_steps_are_recorded_in_the_manifest(tmp_path):
    """A resumed run must still describe every step, marked as skipped, so the
    manifest is a complete record rather than only what ran this time."""
    done = tmp_path / "results" / "eval" / "v2-guards-on"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")

    code = bench_all.main([
        "--adapter", "/nonexistent", "--out-root", str(tmp_path / "results"),
        "--logs", str(tmp_path / "logs"), "--only", "guards_on",
    ])
    manifest = json.loads((tmp_path / "results" / "bench_manifest.json").read_text())
    assert code == 0
    assert len(manifest["steps"]) == 1
    assert manifest["steps"][0]["skipped"] is True
    assert manifest["steps"][0]["returncode"] == 0


def test_force_reruns_a_finished_step(tmp_path):
    done = tmp_path / "results" / "eval" / "v2-guards-on"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")

    bench_all.main([
        "--adapter", "/nonexistent", "--out-root", str(tmp_path / "results"),
        "--logs", str(tmp_path / "logs"), "--only", "guards_on", "--force",
    ])
    manifest = json.loads((tmp_path / "results" / "bench_manifest.json").read_text())
    assert manifest["steps"][0]["skipped"] is False
    assert manifest["steps"][0]["returncode"] != 0, "it really tried to run"
