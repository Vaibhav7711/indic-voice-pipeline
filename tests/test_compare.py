"""Run-comparison tests, focused on the guard against incomparable runs."""

from __future__ import annotations

import json

import pytest

from benchmarks.compare import SelectionMismatch, compare_runs, main


def make_run(
    tmp_path,
    name,
    *,
    wer=40.0,
    cer=20.0,
    indices=(1, 2, 3),
    split="test",
    level="standard",
    model="openai/whisper-medium",
    adapter=None,
    p50=400.0,
    categories=None,
):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "reporting_level": level,
        "examples": len(indices),
        "headline": {"wer_percent": wer, "cer_percent": cer},
        "latency": {
            "total_ms_p50": p50,
            "total_ms_p90": p50 * 1.5,
            "real_time_factor_mean": 0.5,
        },
        "error_analysis": {
            "by_category": categories
            or {"numeric": {"errors": 10}, "rare_word": {"errors": 5}}
        },
    }
    config = {
        "model": model,
        "adapter": adapter,
        "selection": {
            "source": "fleurs",
            "config": "hi_in",
            "split": split,
            "seed": 0,
            "limit": len(indices),
            "indices": list(indices),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_dir


class TestGuard:
    def test_identical_selection_is_comparable(self, tmp_path):
        a = make_run(tmp_path, "a", wer=50.0)
        b = make_run(tmp_path, "b", wer=40.0, adapter="ckpt")
        report = compare_runs(a, b)
        assert report["comparable"] is True
        assert report["selection_warnings"] == []

    def test_different_indices_rejected(self, tmp_path):
        a = make_run(tmp_path, "a", indices=(1, 2, 3))
        b = make_run(tmp_path, "b", indices=(4, 5, 6))
        with pytest.raises(SelectionMismatch, match="different example indices"):
            compare_runs(a, b)

    def test_different_split_rejected(self, tmp_path):
        """The classic silent error: validation vs test."""
        a = make_run(tmp_path, "a", split="validation")
        b = make_run(tmp_path, "b", split="test")
        with pytest.raises(SelectionMismatch, match="different split"):
            compare_runs(a, b)

    def test_different_normalization_level_rejected(self, tmp_path):
        a = make_run(tmp_path, "a", level="standard")
        b = make_run(tmp_path, "b", level="aggressive")
        with pytest.raises(SelectionMismatch, match="normalization level"):
            compare_runs(a, b)

    def test_missing_run_config_rejected(self, tmp_path):
        a = make_run(tmp_path, "a")
        b = make_run(tmp_path, "b")
        (b / "run_config.json").unlink()
        with pytest.raises(SelectionMismatch, match="no run_config.json"):
            compare_runs(a, b)

    def test_override_reports_with_warnings(self, tmp_path):
        a = make_run(tmp_path, "a", indices=(1, 2, 3))
        b = make_run(tmp_path, "b", indices=(4, 5, 6))
        report = compare_runs(a, b, allow_mismatch=True)
        assert report["comparable"] is False
        assert report["selection_warnings"]

    def test_missing_metrics_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            compare_runs(tmp_path / "empty", tmp_path / "empty")


class TestDeltas:
    def test_improvement_is_positive_relative(self, tmp_path):
        a = make_run(tmp_path, "a", wer=50.0)
        b = make_run(tmp_path, "b", wer=40.0)
        report = compare_runs(a, b)
        assert report["wer"]["points"] == -10.0       # WER fell 10 points
        assert report["wer"]["relative_percent"] == 20.0  # a 20% relative cut

    def test_regression_is_negative_relative(self, tmp_path):
        a = make_run(tmp_path, "a", wer=40.0)
        b = make_run(tmp_path, "b", wer=50.0)
        report = compare_runs(a, b)
        assert report["wer"]["points"] == 10.0
        assert report["wer"]["relative_percent"] == -25.0

    def test_points_and_relative_differ(self, tmp_path):
        """Why both are reported: they answer different questions."""
        a = make_run(tmp_path, "a", wer=10.0)
        b = make_run(tmp_path, "b", wer=5.0)
        report = compare_runs(a, b)
        assert report["wer"]["points"] == -5.0
        assert report["wer"]["relative_percent"] == 50.0

    def test_zero_baseline_relative_is_none(self, tmp_path):
        a = make_run(tmp_path, "a", wer=0.0)
        b = make_run(tmp_path, "b", wer=1.0)
        report = compare_runs(a, b)
        assert report["wer"]["relative_percent"] is None

    def test_latency_delta(self, tmp_path):
        a = make_run(tmp_path, "a", p50=400.0)
        b = make_run(tmp_path, "b", p50=500.0)
        report = compare_runs(a, b)
        assert report["latency_p50_ms"]["points"] == 100.0


class TestCategoryDelta:
    def test_errors_moving_between_categories_is_visible(self, tmp_path):
        a = make_run(
            tmp_path,
            "a",
            categories={"numeric": {"errors": 10}, "rare_word": {"errors": 2}},
        )
        b = make_run(
            tmp_path,
            "b",
            categories={"numeric": {"errors": 2}, "rare_word": {"errors": 10}},
        )
        report = compare_runs(a, b)
        delta = report["errors_by_category"]
        assert delta["numeric"]["change"] == -8
        assert delta["rare_word"]["change"] == 8

    def test_category_absent_from_one_run(self, tmp_path):
        a = make_run(tmp_path, "a", categories={"numeric": {"errors": 5}})
        b = make_run(tmp_path, "b", categories={"hallucination": {"errors": 3}})
        report = compare_runs(a, b)
        assert report["errors_by_category"]["numeric"]["change"] == -5
        assert report["errors_by_category"]["hallucination"]["change"] == 3


class TestCLI:
    def test_writes_comparison_json(self, tmp_path):
        a = make_run(tmp_path, "a", wer=50.0)
        b = make_run(tmp_path, "b", wer=40.0)
        out = tmp_path / "comparison.json"
        code = main(
            ["--baseline", str(a), "--candidate", str(b), "--out", str(out)]
        )
        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["wer"]["relative_percent"] == 20.0

    def test_mismatch_exits_nonzero(self, tmp_path):
        a = make_run(tmp_path, "a", indices=(1, 2))
        b = make_run(tmp_path, "b", indices=(3, 4))
        assert main(["--baseline", str(a), "--candidate", str(b)]) == 2

    def test_allow_mismatch_exits_zero(self, tmp_path):
        a = make_run(tmp_path, "a", indices=(1, 2))
        b = make_run(tmp_path, "b", indices=(3, 4))
        assert (
            main(
                [
                    "--baseline",
                    str(a),
                    "--candidate",
                    str(b),
                    "--allow-mismatch",
                ]
            )
            == 0
        )
