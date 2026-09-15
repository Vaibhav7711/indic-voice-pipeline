"""Harness and hard-set tests.

Everything here runs on CPU without torch — that is the point of splitting
`run` (inference) from `score` (scoring).
"""

from __future__ import annotations

import json

import pytest

from benchmarks.asr_eval import (
    _percentile,
    _select_indices,
    main,
    score_predictions,
)
from benchmarks.hard_set import (
    HardSetItem,
    curated_items,
    load_manifest,
    summarize,
    validate_items,
)

PREDICTIONS = [
    {
        "id": "a",
        "reference": "यह एक साधारण वाक्य है।",
        "hypothesis": "यह एक साधारण वाक्य है",
        "categories": ["noisy"],
        "latency": {"total_ms": 100.0, "real_time_factor": 0.5, "encoder_ms": 20.0},
    },
    {
        "id": "b",
        "reference": "मैंने 25 रुपये दिए",
        "hypothesis": "मैंने 35 रुपये दिए",
        "categories": ["numeric"],
        "latency": {"total_ms": 200.0, "real_time_factor": 0.7, "encoder_ms": 25.0},
    },
    {
        "id": "c",
        "reference": "हिन्दी भाषा बहुत पुरानी है",
        "hypothesis": "हिंदी भाषा बहुत पुरानी है",
        "categories": ["noisy", "accented"],
        "latency": {"total_ms": 300.0, "real_time_factor": 0.9, "encoder_ms": 30.0},
    },
]


class TestSelectIndices:
    def test_first_strategy(self):
        assert _select_indices(100, 5, "first", 0) == [0, 1, 2, 3, 4]

    def test_random_is_seeded_and_sorted(self):
        a = _select_indices(100, 10, "random", 42)
        b = _select_indices(100, 10, "random", 42)
        assert a == b == sorted(a)
        assert len(set(a)) == 10

    def test_different_seeds_differ(self):
        assert _select_indices(1000, 20, "random", 1) != _select_indices(
            1000, 20, "random", 2
        )

    def test_random_is_not_a_prefix(self):
        """The reason random is the default: [:N] is a biased slice."""
        assert _select_indices(1000, 20, "random", 0) != list(range(20))

    def test_limit_none_returns_all(self):
        assert _select_indices(7, None, "random", 0) == list(range(7))

    def test_limit_larger_than_dataset(self):
        assert _select_indices(3, 99, "random", 0) == [0, 1, 2]


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) is None

    def test_median(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p100_is_max(self):
        assert _percentile([1, 2, 3], 100) == 3

    def test_low_percentile_is_min(self):
        assert _percentile([1, 2, 3], 1) == 1


class TestScorePredictions:
    def test_headline_present(self):
        metrics, per_example = score_predictions(PREDICTIONS)
        assert metrics["examples"] == 3
        assert metrics["headline"]["wer_percent"] is not None
        assert metrics["headline"]["cer_percent"] is not None
        assert len(per_example) == 3

    def test_all_levels_reported(self):
        metrics, _ = score_predictions(PREDICTIONS)
        for level in ("none", "basic", "standard", "aggressive"):
            assert level in metrics["by_normalization_level"]

    def test_normalization_is_monotonic_across_levels(self):
        """More normalization can never increase WER."""
        metrics, _ = score_predictions(PREDICTIONS)
        levels = ["none", "basic", "standard", "aggressive"]
        wers = [
            metrics["by_normalization_level"][lv]["wer"]["micro_percent"]
            for lv in levels
        ]
        assert wers == sorted(wers, reverse=True)

    def test_sensitivity_deltas_non_negative(self):
        metrics, _ = score_predictions(PREDICTIONS)
        sens = metrics["normalization_sensitivity"]
        assert sens["formatting_delta_points"] >= 0
        assert sens["orthography_delta_points"] >= 0

    def test_danda_costs_nothing_at_standard(self):
        """Example 'a' differs only by a danda."""
        metrics, per_example = score_predictions(PREDICTIONS)
        row = next(r for r in per_example if r["id"] == "a")
        assert row["wer"] == 0.0

    def test_category_breakdown(self):
        metrics, _ = score_predictions(PREDICTIONS)
        by_cat = metrics["by_category"]
        assert by_cat["noisy"]["items"] == 2
        assert by_cat["numeric"]["items"] == 1
        assert by_cat["accented"]["items"] == 1

    def test_latency_summary(self):
        metrics, _ = score_predictions(PREDICTIONS)
        lat = metrics["latency"]
        assert lat["utterances"] == 3
        assert lat["total_ms_p50"] == 200.0
        assert lat["total_ms_mean"] == 200.0

    def test_no_latency_is_tolerated(self):
        rows = [{"id": "x", "reference": "अ ब", "hypothesis": "अ ब"}]
        metrics, _ = score_predictions(rows)
        assert metrics["latency"] == {}

    def test_empty_predictions(self):
        metrics, per_example = score_predictions([])
        assert metrics["examples"] == 0
        assert per_example == []

    def test_error_categories_surface(self):
        metrics, _ = score_predictions(PREDICTIONS)
        cats = metrics["error_analysis"]["by_category"]
        assert "numeric" in cats
        assert "orthographic" in cats


class TestScoreCLI:
    def test_end_to_end(self, tmp_path):
        predictions = tmp_path / "predictions.jsonl"
        predictions.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in PREDICTIONS),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        code = main(
            [
                "score",
                "--predictions",
                str(predictions),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert code == 0
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["examples"] == 3
        assert metrics["provenance"]["git_commit"] is not None

        errors = (out_dir / "errors.jsonl").read_text(encoding="utf-8").strip()
        assert len(errors.splitlines()) == 3

    def test_level_flag_changes_result(self, tmp_path):
        predictions = tmp_path / "p.jsonl"
        predictions.write_text(
            json.dumps(
                {"id": "x", "reference": "हिन्दी", "hypothesis": "हिंदी"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        for level, expected in (("standard", 100.0), ("aggressive", 0.0)):
            out_dir = tmp_path / level
            main(
                [
                    "score",
                    "--predictions",
                    str(predictions),
                    "--out-dir",
                    str(out_dir),
                    "--level",
                    level,
                ]
            )
            metrics = json.loads(
                (out_dir / "metrics.json").read_text(encoding="utf-8")
            )
            assert metrics["headline"]["wer_percent"] == expected


class TestHardSet:
    def _item(self, **overrides):
        payload = {
            "id": "noisy_001",
            "transcript": "कुछ शब्द",
            "categories": ["noisy"],
            "audio": {
                "type": "hf",
                "dataset": "google/fleurs",
                "split": "test",
                "index": 1,
            },
            "status": "curated",
        }
        payload.update(overrides)
        return HardSetItem.from_dict(payload)

    def test_valid_item(self):
        assert validate_items([self._item()]) == []

    def test_duplicate_id(self):
        problems = validate_items([self._item(), self._item()])
        assert any("duplicate id" in p for p in problems)

    def test_unknown_category(self):
        problems = validate_items([self._item(categories=["spicy"])])
        assert any("unknown categories" in p for p in problems)

    def test_curated_needs_category(self):
        problems = validate_items([self._item(categories=[])])
        assert any("at least one category" in p for p in problems)

    def test_candidate_may_lack_category(self):
        """Bootstrapped review queues have no categories yet, by design."""
        problems = validate_items([self._item(categories=[], status="candidate")])
        assert not any("at least one category" in p for p in problems)

    def test_curated_needs_transcript(self):
        problems = validate_items([self._item(transcript="  ")])
        assert any("non-empty transcript" in p for p in problems)

    def test_candidate_may_lack_transcript(self):
        problems = validate_items([self._item(transcript="", status="candidate")])
        assert not any("non-empty transcript" in p for p in problems)

    def test_bad_status(self):
        problems = validate_items([self._item(status="maybe")])
        assert any("status must be one of" in p for p in problems)

    def test_missing_local_file(self):
        problems = validate_items(
            [self._item(audio={"type": "local", "path": "does/not/exist.wav"})]
        )
        assert any("audio file not found" in p for p in problems)

    def test_bad_audio_type(self):
        problems = validate_items([self._item(audio={"type": "magic"})])
        assert any("audio.type must be" in p for p in problems)

    def test_candidates_excluded_from_reporting(self):
        items = [self._item(), self._item(id="b", status="candidate")]
        assert [i.id for i in curated_items(items)] == ["noisy_001"]

    def test_summary(self):
        items = [
            self._item(),
            self._item(id="b", categories=["noisy", "numeric"]),
            self._item(id="c", status="candidate"),
        ]
        summary = summarize(items)
        assert summary["total"] == 3
        assert summary["curated"] == 2
        assert summary["candidate"] == 1
        assert summary["by_category"] == {"noisy": 2, "numeric": 1}

    def test_missing_manifest_is_empty_not_error(self, tmp_path):
        assert load_manifest(tmp_path / "nope.jsonl") == []

    def test_malformed_json_raises_with_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_manifest(path)

    def test_shipped_example_manifest_is_valid(self):
        """The committed example must parse and validate as a schema demo."""
        items = load_manifest("data/hard_set/manifest.example.jsonl")
        assert len(items) == 4
        problems = validate_items(items)
        # The example uses placeholder local paths that are intentionally
        # absent; every other rule must still pass.
        assert all("audio file not found" in p for p in problems)
