"""Compare two evaluation runs.

The point of this module is the guard, not the arithmetic. The easiest way to
produce a fake improvement is to compare a fine-tuned model evaluated on one
sample against a baseline evaluated on another — different `--seed`, different
`--limit`, or one run on `validation` and the other on `test`. The numbers look
comparable and are not.

`compare_runs` refuses to report a delta when the two runs did not see the same
audio, unless explicitly overridden. Treat an override as a result that needs a
caveat written next to it.

Deltas are reported in **percentage points** (an arithmetic difference between
two percentages), plus relative reduction, because "WER fell 4 points" and "WER
fell 12% relative" answer different questions and are routinely confused.

Usage:
    python -m benchmarks.compare \
        --baseline results/eval/medium-base-test \
        --candidate results/eval/medium-lora-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["compare_runs", "SelectionMismatch", "main"]


class SelectionMismatch(RuntimeError):
    """Raised when two runs did not evaluate the same audio."""


def _load(run_dir: str | Path) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.json in {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    config_path = run_dir / "run_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    return metrics, config


def _selection_problems(baseline: dict, candidate: dict) -> list[str]:
    """Reasons the two runs are not comparable."""
    problems: list[str] = []
    a = baseline.get("selection") or {}
    b = candidate.get("selection") or {}

    if not a or not b:
        problems.append(
            "one or both runs have no run_config.json selection block; "
            "cannot verify the same audio was used"
        )
        return problems

    if a.get("source") != b.get("source"):
        problems.append(f"different source: {a.get('source')} vs {b.get('source')}")

    for key in ("config", "split", "manifest"):
        if a.get(key) != b.get(key):
            problems.append(f"different {key}: {a.get(key)!r} vs {b.get(key)!r}")

    # The strongest check: the literal example indices.
    if a.get("indices") is not None and b.get("indices") is not None:
        if a["indices"] != b["indices"]:
            shared = len(set(a["indices"]) & set(b["indices"]))
            problems.append(
                f"different example indices ({len(a['indices'])} vs "
                f"{len(b['indices'])}, {shared} shared)"
            )

    if baseline.get("reporting_level") != candidate.get("reporting_level"):
        problems.append(
            f"different normalization level: {baseline.get('reporting_level')!r} "
            f"vs {candidate.get('reporting_level')!r}"
        )

    return problems


def _delta(before: float | None, after: float | None) -> dict:
    if before is None or after is None:
        return {"before": before, "after": after, "points": None, "relative_percent": None}
    points = round(after - before, 4)
    relative = round(100.0 * (before - after) / before, 2) if before else None
    return {
        "before": before,
        "after": after,
        "points": points,
        "relative_percent": relative,  # positive means the candidate improved
    }


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    allow_mismatch: bool = False,
) -> dict:
    """Compare two run directories. Raises if they are not comparable."""
    base_metrics, base_config = _load(baseline_dir)
    cand_metrics, cand_config = _load(candidate_dir)

    problems = _selection_problems(
        {**base_config, "reporting_level": base_metrics.get("reporting_level")},
        {**cand_config, "reporting_level": cand_metrics.get("reporting_level")},
    )
    if problems and not allow_mismatch:
        raise SelectionMismatch(
            "Runs are not comparable:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nRe-run both with identical --split, --limit, --seed and "
            "--level, or pass --allow-mismatch and write the caveat into "
            "docs/EXPERIMENTS.md."
        )

    base_head = base_metrics.get("headline", {})
    cand_head = cand_metrics.get("headline", {})
    base_lat = base_metrics.get("latency") or {}
    cand_lat = cand_metrics.get("latency") or {}

    # Error-category shares, so a WER gain that merely moved errors between
    # categories is visible.
    base_cats = (base_metrics.get("error_analysis") or {}).get("by_category", {})
    cand_cats = (cand_metrics.get("error_analysis") or {}).get("by_category", {})
    category_delta = {}
    for name in sorted(set(base_cats) | set(cand_cats)):
        before = base_cats.get(name, {}).get("errors", 0)
        after = cand_cats.get(name, {}).get("errors", 0)
        category_delta[name] = {
            "errors_before": before,
            "errors_after": after,
            "change": after - before,
        }

    return {
        "baseline": {
            "dir": str(baseline_dir),
            "model": base_config.get("model"),
            "adapter": base_config.get("adapter"),
        },
        "candidate": {
            "dir": str(candidate_dir),
            "model": cand_config.get("model"),
            "adapter": cand_config.get("adapter"),
        },
        "comparable": not problems,
        "selection_warnings": problems,
        "examples": {
            "baseline": base_metrics.get("examples"),
            "candidate": cand_metrics.get("examples"),
        },
        "wer": _delta(base_head.get("wer_percent"), cand_head.get("wer_percent")),
        "cer": _delta(base_head.get("cer_percent"), cand_head.get("cer_percent")),
        "latency_p50_ms": _delta(
            base_lat.get("total_ms_p50"), cand_lat.get("total_ms_p50")
        ),
        "latency_p90_ms": _delta(
            base_lat.get("total_ms_p90"), cand_lat.get("total_ms_p90")
        ),
        "real_time_factor": _delta(
            base_lat.get("real_time_factor_mean"),
            cand_lat.get("real_time_factor_mean"),
        ),
        "errors_by_category": category_delta,
    }


def _print(report: dict) -> None:
    print("=" * 66)
    base, cand = report["baseline"], report["candidate"]
    print(f"baseline : {base['model']} adapter={base['adapter']}")
    print(f"candidate: {cand['model']} adapter={cand['adapter']}")
    if report["selection_warnings"]:
        print("\n!! NOT STRICTLY COMPARABLE:")
        for warning in report["selection_warnings"]:
            print(f"   - {warning}")
    print("-" * 66)
    print(f"{'metric':<20}{'before':>12}{'after':>12}{'delta':>11}{'rel %':>10}")
    for key in ("wer", "cer", "latency_p50_ms", "latency_p90_ms", "real_time_factor"):
        d = report[key]
        if d["before"] is None or d["after"] is None:
            continue
        print(
            f"{key:<20}{d['before']:>12}{d['after']:>12}"
            f"{d['points']:>+11}{d['relative_percent']:>+10}"
        )

    changes = {
        k: v for k, v in report["errors_by_category"].items() if v["change"] != 0
    }
    if changes:
        print("-" * 66)
        print("Errors by category (negative = fewer errors):")
        for name, payload in sorted(changes.items(), key=lambda kv: kv[1]["change"]):
            print(
                f"  {name:<16} {payload['errors_before']:>5} -> "
                f"{payload['errors_after']:<5} {payload['change']:>+5}"
            )
    print("=" * 66)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs")
    parser.add_argument("--baseline", required=True, help="run directory")
    parser.add_argument("--candidate", required=True, help="run directory")
    parser.add_argument("--out", default=None, help="write comparison.json here")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="report a delta even when the runs saw different audio",
    )
    args = parser.parse_args(argv)

    try:
        report = compare_runs(
            args.baseline, args.candidate, allow_mismatch=args.allow_mismatch
        )
    except SelectionMismatch as exc:
        print(str(exc))
        return 2

    _print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
