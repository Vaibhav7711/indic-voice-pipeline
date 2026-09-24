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
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_all.py"
_spec = importlib.util.spec_from_file_location("bench_all_script", _path)
bench_all = importlib.util.module_from_spec(_spec)
sys.modules["bench_all_script"] = bench_all
_spec.loader.exec_module(bench_all)


def test_a_step_never_run_is_not_done(tmp_path):
    assert bench_all.is_done({"name": "sweep", "out": tmp_path}, {}) is False


def test_evidence_alone_is_not_completion(tmp_path):
    """The failure this replaced: gpu_validation rewrites report.json after
    each of its 16 checks and llm_bakeoff after each model, so a session that
    died mid-step leaves a well-formed evidence file. Inferring completion
    from it produced a green manifest over a sweep that ran two checks."""
    out = tmp_path / "gpu_validation"
    out.mkdir()
    (out / "report.json").write_text('{"checks": []}')
    (out / "summary.md").write_text("partial")
    assert bench_all.is_done({"name": "sweep", "out": out}, {}) is False


def test_a_step_the_driver_saw_succeed_is_done(tmp_path):
    out = tmp_path / "gpu_validation"
    out.mkdir()
    state = {"sweep": {"returncode": 0, "out": str(out)}}
    assert bench_all.is_done({"name": "sweep", "out": out}, state) is True


def test_a_step_that_failed_is_not_done(tmp_path):
    out = tmp_path / "gpu_validation"
    out.mkdir()
    state = {"sweep": {"returncode": 1, "out": str(out)}}
    assert bench_all.is_done({"name": "sweep", "out": out}, state) is False


def test_success_without_surviving_evidence_is_not_done(tmp_path):
    """State says it ran, but the ephemeral disk lost the output."""
    state = {"sweep": {"returncode": 0, "out": str(tmp_path / "gone")}}
    assert bench_all.is_done({"name": "sweep", "out": tmp_path / "gone"}, state) is False


def test_steps_with_no_output_always_rerun(tmp_path):
    state = {"unit_tests": {"returncode": 0, "out": None}}
    assert bench_all.is_done({"name": "unit_tests", "out": None}, state) is False


def test_state_round_trips_and_survives_corruption(tmp_path):
    bench_all.save_state(tmp_path, {"llm": {"returncode": 0}})
    assert bench_all.load_state(tmp_path) == {"llm": {"returncode": 0}}
    (tmp_path / bench_all.STATE_FILE).write_text("{not json")
    assert bench_all.load_state(tmp_path) == {}, "a corrupt state file re-runs, never skips"


def test_guards_on_command_records_override_values():
    args = SimpleNamespace(
        out_root="results", adapter="/adapter", limit=300,
        no_speech_threshold=0.9, loop_guard_ngram=5, loop_guard_repeats=6,
        llm_models="x", tts_backends="edge", streaming_limit=100,
        ct2_dir="models/ct2",
    )
    command = next(step["cmd"] for step in bench_all.steps(args)
                   if step["name"] == "guards_on")
    assert command[command.index("--no-speech-threshold") + 1] == "0.9"
    assert command[command.index("--loop-guard-ngram") + 1] == "5"
    assert command[command.index("--loop-guard-repeats") + 1] == "6"


def test_mirror_preserves_the_path_so_it_can_be_restored(tmp_path):
    """Flattening to <mirror>/<basename> meant the mirror could not be copied
    back into --out-root, so resume never fired across a disconnect."""
    out_root = tmp_path / "results"
    out = out_root / "eval" / "v2-guards-on"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    mirror = tmp_path / "drive"

    bench_all.mirror_output({"name": "guards_on", "out": out}, out_root, mirror)
    assert (mirror / "eval" / "v2-guards-on" / "metrics.json").is_file()


def test_mirror_excludes_audio_and_weights(tmp_path):
    out_root = tmp_path / "results"
    out = out_root / "run"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    (out / "clip.wav").write_bytes(b"\x00")
    (out / "model.bin").write_bytes(b"\x00")
    mirror = tmp_path / "drive"

    bench_all.mirror_output({"name": "x", "out": out}, out_root, mirror)
    assert (mirror / "run" / "metrics.json").is_file()
    assert not (mirror / "run" / "clip.wav").exists()
    assert not (mirror / "run" / "model.bin").exists()


def test_mirror_failure_is_reported_not_raised(tmp_path):
    out_root = tmp_path / "results"
    out = out_root / "run"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    result = bench_all.mirror_output({"name": "x", "out": out}, out_root, blocked)
    assert result is not None and "mirror failed" in result


def test_restore_brings_a_dead_session_back(tmp_path):
    """The case --mirror exists for: a fresh VM with an empty checkout."""
    mirror = tmp_path / "drive"
    (mirror / "eval" / "v2-guards-on").mkdir(parents=True)
    (mirror / "eval" / "v2-guards-on" / "metrics.json").write_text("{}")
    (mirror / bench_all.STATE_FILE).write_text(
        json.dumps({"guards_on": {"returncode": 0,
                                  "out": str(tmp_path / "results/eval/v2-guards-on")}}))
    out_root = tmp_path / "results"

    restored = bench_all.restore_from_mirror(out_root, mirror)
    assert restored == 2
    state = bench_all.load_state(out_root)
    assert bench_all.is_done(
        {"name": "guards_on", "out": out_root / "eval/v2-guards-on"}, state) is True


def test_restore_never_overwrites_newer_local_evidence(tmp_path):
    mirror = tmp_path / "drive"
    mirror.mkdir()
    (mirror / "metrics.json").write_text("old")
    out_root = tmp_path / "results"
    out_root.mkdir()
    (out_root / "metrics.json").write_text("new")

    bench_all.restore_from_mirror(out_root, mirror)
    assert (out_root / "metrics.json").read_text() == "new"


def test_skipped_steps_are_recorded_in_the_manifest(tmp_path):
    """A resumed run must still describe every step, marked as skipped, so the
    manifest is a complete record rather than only what ran this time."""
    out_root = tmp_path / "results"
    done = out_root / "eval" / "v2-guards-on"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    bench_all.save_state(out_root, {"guards_on": {"returncode": 0, "out": str(done)}})

    code = bench_all.main([
        "--adapter", "/nonexistent", "--out-root", str(out_root),
        "--logs", str(tmp_path / "logs"), "--only", "guards_on",
    ])
    manifest = json.loads((out_root / "bench_manifest.json").read_text())
    assert code == 0
    assert len(manifest["steps"]) == 1
    assert manifest["steps"][0]["skipped"] is True


def test_force_reruns_a_step_the_state_says_succeeded(tmp_path):
    out_root = tmp_path / "results"
    done = out_root / "eval" / "v2-guards-on"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    bench_all.save_state(out_root, {"guards_on": {"returncode": 0, "out": str(done)}})

    bench_all.main([
        "--adapter", "/nonexistent", "--out-root", str(out_root),
        "--logs", str(tmp_path / "logs"), "--only", "guards_on", "--force",
    ])
    manifest = json.loads((out_root / "bench_manifest.json").read_text())
    assert manifest["steps"][0]["skipped"] is False
    assert manifest["steps"][0]["returncode"] != 0, "it really tried to run"


def test_a_failed_step_is_retried_on_the_next_run(tmp_path):
    out_root = tmp_path / "results"
    bench_all.save_state(out_root, {"guards_on": {"returncode": 1, "out": None}})
    bench_all.main([
        "--adapter", "/nonexistent", "--out-root", str(out_root),
        "--logs", str(tmp_path / "logs"), "--only", "guards_on",
    ])
    manifest = json.loads((out_root / "bench_manifest.json").read_text())
    assert manifest["steps"][0]["skipped"] is False
