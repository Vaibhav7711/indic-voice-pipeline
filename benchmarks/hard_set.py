"""A fixed, manually curated hard set for Hindi ASR.

Why a hard set at all
---------------------
FLEURS is read speech recorded in quiet conditions. A model can improve on
FLEURS while getting *worse* at the things a voice agent actually hears:
background noise, regional accents, proper nouns, phone numbers, and Hinglish.
An aggregate test WER cannot show that, because those conditions are barely
present in the test set.

The hard set is the ASR equivalent of a hand-written regression suite next to
your generic benchmark: small, fixed, adversarial, and reported **per
category** so a regression in one condition cannot hide behind a gain in
another.

Curation rules
--------------
1. **Fixed.** Once an item is marked ``curated`` its audio and transcript never
   change. Editing the set invalidates comparisons across runs.
2. **Never trained on.** These clips must not enter any training mixture.
3. **Human-verified transcripts.** ``candidate`` items produced by bootstrap
   carry the *model's* output as a placeholder and are excluded from reporting
   until a human corrects the transcript and flips the status.
4. **Category-tagged.** Every item carries at least one category so results can
   be sliced.

Manifest format: JSON Lines, one object per item. See
``data/hard_set/manifest.example.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CATEGORIES",
    "HardSetItem",
    "load_manifest",
    "validate_items",
    "curated_items",
    "DEFAULT_MANIFEST",
]

DEFAULT_MANIFEST = Path("data/hard_set/manifest.jsonl")

#: Fixed category vocabulary. Adding a category is a deliberate act — it means
#: every prior run's per-category table gains a column that is empty.
CATEGORIES: frozenset[str] = frozenset(
    {
        "noisy",          # background noise, low SNR
        "accented",       # regional / L2 Hindi accent
        "named_entity",   # person, place, brand, organisation
        "numeric",        # digits, amounts, dates, phone numbers
        "code_switch",    # Hinglish, Latin-script insertions
        "spontaneous",    # disfluent, conversational, non-read
        "far_field",      # distant mic, room reverberation
        "telephony",      # 8 kHz / codec-degraded
        "fast_speech",    # unusually high speaking rate
        "long_form",      # longer than Whisper's 30 s window
    }
)

_VALID_STATUS = {"curated", "candidate"}


@dataclass
class HardSetItem:
    id: str
    transcript: str
    categories: list[str]
    audio: dict
    status: str = "curated"
    notes: str = ""
    duration_s: float | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> HardSetItem:
        known = {
            "id",
            "transcript",
            "categories",
            "audio",
            "status",
            "notes",
            "duration_s",
            "source",
        }
        return cls(
            id=str(payload.get("id", "")),
            transcript=payload.get("transcript", ""),
            categories=list(payload.get("categories", [])),
            audio=payload.get("audio", {}) or {},
            status=payload.get("status", "curated"),
            notes=payload.get("notes", ""),
            duration_s=payload.get("duration_s"),
            source=payload.get("source", ""),
            extra={k: v for k, v in payload.items() if k not in known},
        )

    def as_dict(self) -> dict:
        out = {
            "id": self.id,
            "transcript": self.transcript,
            "categories": self.categories,
            "audio": self.audio,
            "status": self.status,
            "notes": self.notes,
            "duration_s": self.duration_s,
            "source": self.source,
        }
        out.update(self.extra)
        return out


def load_manifest(path: str | Path) -> list[HardSetItem]:
    """Read a JSONL manifest. Missing file yields an empty set, not an error."""
    path = Path(path)
    if not path.is_file():
        return []
    items: list[HardSetItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
            items.append(HardSetItem.from_dict(payload))
    return items


def validate_items(
    items: list[HardSetItem], *, root: str | Path = "."
) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    problems: list[str] = []
    root = Path(root)
    seen: set[str] = set()

    for index, item in enumerate(items):
        where = f"item {index} (id={item.id or '<missing>'})"

        if not item.id:
            problems.append(f"{where}: missing 'id'")
        elif item.id in seen:
            problems.append(f"{where}: duplicate id")
        else:
            seen.add(item.id)

        if item.status not in _VALID_STATUS:
            problems.append(
                f"{where}: status must be one of {sorted(_VALID_STATUS)}, "
                f"got {item.status!r}"
            )

        # Categories are assigned during human review, so only finished
        # (curated) items are required to have them. Candidates are a queue.
        if item.status == "curated" and not item.categories:
            problems.append(f"{where}: curated item needs at least one category")
        unknown = sorted(set(item.categories) - CATEGORIES)
        if unknown:
            problems.append(f"{where}: unknown categories {unknown}")

        if item.status == "curated" and not item.transcript.strip():
            problems.append(f"{where}: curated item needs a non-empty transcript")

        audio_type = item.audio.get("type")
        if audio_type == "local":
            rel = item.audio.get("path")
            if not rel:
                problems.append(f"{where}: local audio needs 'path'")
            elif not (root / rel).is_file():
                problems.append(f"{where}: audio file not found — {rel}")
        elif audio_type == "hf":
            for key in ("dataset", "split", "index"):
                if key not in item.audio:
                    problems.append(f"{where}: hf audio needs '{key}'")
        else:
            problems.append(
                f"{where}: audio.type must be 'local' or 'hf', got {audio_type!r}"
            )

        if item.duration_s is not None and item.duration_s <= 0:
            problems.append(f"{where}: duration_s must be positive")

    return problems


def curated_items(items: list[HardSetItem]) -> list[HardSetItem]:
    """Only ``curated`` items are eligible for reporting."""
    return [item for item in items if item.status == "curated"]


def summarize(items: list[HardSetItem]) -> dict:
    counts: dict[str, int] = {}
    for item in curated_items(items):
        for category in item.categories:
            counts[category] = counts.get(category, 0) + 1
    return {
        "total": len(items),
        "curated": len(curated_items(items)),
        "candidate": sum(1 for i in items if i.status == "candidate"),
        "by_category": dict(sorted(counts.items())),
    }


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_from_predictions(
    predictions_path: str | Path,
    *,
    limit: int = 25,
    min_wer: float = 0.4,
) -> list[HardSetItem]:
    """Mine the worst examples of a finished run into a review queue.

    This does **not** create a hard set. It creates ``candidate`` items whose
    transcript is the *reference* from the source dataset, tagged for a human
    to confirm, re-tag, and promote. Bootstrapping from real failures is how
    you find genuinely hard audio instead of guessing what hard audio is.
    """
    from benchmarks.metrics import score_text

    path = Path(predictions_path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        alignment = score_text(row.get("reference", ""), row.get("hypothesis", ""))
        rate = alignment.error_rate
        if rate is not None and rate >= min_wer:
            scored.append((rate, row))
    scored.sort(key=lambda pair: -pair[0])

    items = []
    for rate, row in scored[:limit]:
        items.append(
            HardSetItem(
                id=f"candidate_{row.get('id', 'unknown')}",
                transcript=row.get("reference", ""),
                categories=[],
                audio=row.get("audio_ref", {}),
                status="candidate",
                notes=(
                    f"auto-selected: run WER {rate:.1%}. "
                    "REVIEW REQUIRED: verify transcript, assign categories, "
                    "then set status to 'curated'."
                ),
                duration_s=row.get("audio_duration_s"),
                source=f"bootstrap:{path.name}",
                extra={"bootstrap_wer": round(rate, 4)},
            )
        )
    return items


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hard set utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check a manifest")
    v.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    v.add_argument("--root", default=".")

    b = sub.add_parser("bootstrap", help="propose candidates from a finished run")
    b.add_argument("--predictions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=25)
    b.add_argument("--min-wer", type=float, default=0.4)

    args = parser.parse_args(argv)

    if args.command == "validate":
        items = load_manifest(args.manifest)
        problems = validate_items(items, root=args.root)
        print(json.dumps(summarize(items), indent=2, ensure_ascii=False))
        if problems:
            print(f"\n{len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\nmanifest OK")
        return 0

    if args.command == "bootstrap":
        items = bootstrap_from_predictions(
            args.predictions, limit=args.limit, min_wer=args.min_wer
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(
                    json.dumps(item.as_dict(), ensure_ascii=False) + "\n"
                )
        print(f"Wrote {len(items)} candidate item(s) to {out}")
        print("All are status='candidate' and excluded from reporting until reviewed.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
