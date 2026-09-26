"""Does the assistant answer correctly, briefly, and in the right script?

Nothing in this project measured that. `benchmarks/llm_bakeoff.py` scored
`devanagari_ratio` and `think_leaks`, so a model that scored **1.000** was one
whose *script* was right — never one whose *answers* were. A measured run then
produced, from the chosen model:

    भारत की राजधानी क्या है?      -> भारत की राजधानी नई दिल्ली है।   correct
    What is the capital of India?  -> चीनी राजधानी है।                "the Chinese capital"
    मुझे एक छोटी कहानी सुनाओ।      -> एक बर्फ के (x13, to the token cap)
    थोड़ा धीरे बोलो please…        -> चलिए तेज बोलकर बात करते हैं।    "let's talk faster"

All four scored 1.000 on Devanagari ratio. The model selection was made on a
metric that cannot see any of this.

This harness scores four things a voice assistant has to get right, and each is
checkable without a judge model:

* **factual** — the answer contains the expected fact, normalised. Only
  questions with one uncontroversial answer are asked, so "correct" is not a
  matter of opinion.
* **brevity** — the system prompt asks for one or two short sentences and the
  measured pipeline ignored it on 3 of 12 live turns. Sentence count and
  character length against the `max_unit_chars` budget that decides silence
  before speech.
* **no loop** — the degenerate output above, detected by period search rather
  than by eye.
* **script** — Devanagari ratio, kept because it is a real requirement, not
  because it is sufficient.

An `unanswerable` case type exists for a reason: asked today's weather with no
weather data, a useful assistant says it does not know. The measured pipeline
said *आज मौसम बराबर है।* ("the weather is equal today"), which is worse than a
refusal because it sounds like an answer.

    python -m benchmarks.answer_quality --llm-engine explicit
    python -m benchmarks.answer_quality --llm-engine explicit \
        --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 0.5 --seed 0

Nothing here decides. The rule is pre-registered in `docs/EXPERIMENTS.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEVANAGARI_START, DEVANAGARI_END = chr(0x0900), chr(0x097F)
#: Danda and double danda. Punctuation, but inside the Devanagari block, so a
#: class that keeps U+0900-U+097F keeps them too, and "नई दिल्ली।" would
#: normalise with its danda still attached.
DANDA, DOUBLE_DANDA = chr(0x0964), chr(0x0965)

#: Built from code points rather than escapes: the escapes in this file get
#: interpreted before it reaches disk, which silently turns a range into two
#: literal characters.
DEVANAGARI = re.compile(f"[{DEVANAGARI_START}-{DEVANAGARI_END}]")
LATIN = re.compile(r"[A-Za-z]")
SENTENCE_END = re.compile(r"[।॥!?]+|\.(?:\s|$)")

#: Phrases a Hindi answer may use to say it does not know. Matched loosely: the
#: point is whether the model declined, not how gracefully.
REFUSALS = ("नहीं पता", "पता नहीं", "जानकारी नहीं", "नहीं जानता", "नहीं जानती",
            "मुझे नहीं", "उपलब्ध नहीं", "बता नहीं", "i don't know", "cannot")


@dataclass
class Case:
    """One prompt and what a good answer to it looks like.

    `expected` holds acceptable surface forms; any one of them counts. That is
    deliberately loose — this measures whether the model knows the fact, not
    whether it phrases it the way a reference does.
    """

    prompt: str
    kind: str                                  # factual | instruction | unanswerable
    expected: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    #: For instruction cases: what obeying the instruction looks like.
    note: str = ""
    max_sentences: int = 2
    max_chars: int = 160
    #: True when a correct answer legitimately contains Latin characters, so
    #: the Devanagari ratio must not be held against it. "H₂O" is the case
    #: that found this: the measured run scored 98% script purity across all
    #: four arms *because one correct answer contains Latin*, and the
    #: pre-registered >= 0.99 proviso was therefore unsatisfiable by any arm
    #: including the incumbent. A metric that penalises a right answer is
    #: measuring the wrong thing.
    allows_latin: bool = False


#: Small, and every item checkable. A larger set is better; a larger set of
#: unverifiable items is not.
CASES: tuple[Case, ...] = (
    Case("भारत की राजधानी क्या है?", "factual", expected=("नई दिल्ली", "दिल्ली"),
         forbidden=("चीनी", "बेंगलुरु", "मुंबई")),
    Case("फ्रांस की राजधानी क्या है?", "factual", expected=("पेरिस",)),
    Case("जापान की राजधानी क्या है?", "factual", expected=("टोक्यो", "तोक्यो")),
    Case("दो और दो कितने होते हैं?", "factual", expected=("चार", "4")),
    Case("एक सप्ताह में कितने दिन होते हैं?", "factual", expected=("सात", "7")),
    Case("सूरज किस दिशा में उगता है?", "factual", expected=("पूर्व",)),
    Case("पानी का रासायनिक सूत्र क्या है?", "factual",
         expected=("H2O", "एच2ओ", "H₂O"), allows_latin=True),
    Case("भारत का राष्ट्रीय पशु कौन है?", "factual", expected=("बाघ", "शेर")),
    Case("हिमालय किस देश में है?", "factual", expected=("भारत", "नेपाल")),
    Case("ताजमहल कहाँ है?", "factual", expected=("आगरा",)),

    Case("मुझे एक छोटी कहानी सुनाओ।", "instruction",
         note="a short story, not a loop", max_sentences=4, max_chars=300),
    Case("थोड़ा धीरे बोलो, मुझे समझ नहीं आया।", "instruction",
         note="acknowledge and repeat more simply; must not offer to speak faster",
         forbidden=("तेज", "तेजी")),
    Case("एक वाक्य में बताओ कि योग क्या है।", "instruction",
         note="exactly one sentence", max_sentences=1),
    Case("मेरा नाम वैभव है। मेरा नाम क्या है?", "instruction",
         note="echo the name back", expected=("वैभव",)),
    Case("गिनती करो: एक, दो, और आगे तीन तक।", "instruction",
         note="counts to three", expected=("तीन",)),

    Case("आज मौसम कैसा है?", "unanswerable",
         note="no weather data; saying it does not know beats inventing"),
    Case("अभी समय क्या है?", "unanswerable", note="no clock"),
    Case("मेरे बैंक खाते में कितने पैसे हैं?", "unanswerable", note="no account access"),
)


@dataclass
class Score:
    case: Case
    answer: str
    correct: bool | None = None
    obeyed: bool | None = None
    declined: bool | None = None
    looping: bool = False
    sentences: int = 0
    chars: int = 0
    devanagari_ratio: float | None = None
    too_long: bool = False
    forbidden_hit: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "prompt": self.case.prompt, "kind": self.case.kind,
            "answer": self.answer, "correct": self.correct,
            "obeyed": self.obeyed, "declined": self.declined,
            "looping": self.looping, "sentences": self.sentences,
            "chars": self.chars, "devanagari_ratio": self.devanagari_ratio,
            "too_long": self.too_long, "forbidden": list(self.forbidden_hit),
            "note": self.case.note,
        }


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse spaces, for fact matching.

    Devanagari digits (U+0966-U+096F) are kept: "4" and "४" are both answers to
    a counting question. Danda and double danda are not -- they are sentence
    punctuation that happens to live in the same block.
    """
    keep = f"[^\\w{DEVANAGARI_START}-{DEVANAGARI_END}]+"
    stripped = re.sub(f"{keep}|[{DANDA}{DOUBLE_DANDA}]+", " ", text.lower())
    return re.sub(r"\s+", " ", stripped).strip()


def count_sentences(text: str) -> int:
    """Terminators, not newlines: TTS speaks sentences."""
    parts = [part for part in SENTENCE_END.split(text.strip()) if part.strip()]
    return len(parts)


def devanagari_ratio(text: str) -> float | None:
    """Devanagari as a share of Devanagari plus Latin letters.

    `None` when the answer has no letters at all — an empty or punctuation-only
    answer is a failure, and scoring it 1.0 would read as perfect script.
    """
    deva, latin = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    total = deva + latin
    return None if total == 0 else deva / total


def looks_looping(text: str, size: int = 48, repeats: int = 3,
                  min_period: int = 4) -> bool:
    """The same check the HTTP engine's guard uses, on the finished answer."""
    if size <= 0 or len(text) < size:
        return False
    tail = text[-size:]
    for period in range(min_period, size // repeats + 1):
        if tail[-period * repeats:] == tail[-period:] * repeats:
            return True
    return False


def declined(text: str) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in REFUSALS)


def score(case: Case, answer: str) -> Score:
    """Score one answer. Unjudgeable fields stay `None`, never False.

    The distinction matters for the `kind` splits: an instruction case has no
    factual verdict, and recording that as "incorrect" would drag the factual
    accuracy down with cases it was never asked about.
    """
    result = Score(case=case, answer=answer)
    normalised = normalise(answer)
    result.sentences = count_sentences(answer)
    result.chars = len(answer.strip())
    result.devanagari_ratio = devanagari_ratio(answer)
    result.looping = looks_looping(answer)
    result.too_long = (result.sentences > case.max_sentences
                       or result.chars > case.max_chars)
    result.forbidden_hit = tuple(
        word for word in case.forbidden if normalise(word) in normalised)

    if case.kind == "factual":
        result.correct = (
            any(normalise(option) in normalised for option in case.expected)
            and not result.forbidden_hit)
    elif case.kind == "instruction":
        expected_met = (not case.expected
                        or any(normalise(o) in normalised for o in case.expected))
        result.obeyed = (expected_met and not result.forbidden_hit
                         and not result.looping and not result.too_long)
    else:                                       # unanswerable
        result.declined = declined(answer)
    return result


def summarise(scores: list[Score]) -> dict:
    """Rates per case kind. `None` for a kind with no cases, never 0.0."""

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    factual = [s.correct for s in scores if s.correct is not None]
    obeyed = [s.obeyed for s in scores if s.obeyed is not None]
    declines = [s.declined for s in scores if s.declined is not None]
    # Cases whose correct answer contains Latin are excluded from the script
    # ratio, not scored down by it. Their own ratio is still recorded per case.
    ratios = [s.devanagari_ratio for s in scores
              if s.devanagari_ratio is not None and not s.case.allows_latin]
    return {
        "cases": len(scores),
        "factual_n": len(factual), "factual_accuracy": rate(factual),
        "instruction_n": len(obeyed), "instruction_obeyed": rate(obeyed),
        "unanswerable_n": len(declines), "declined_rate": rate(declines),
        "looping": sum(s.looping for s in scores),
        "too_long": sum(s.too_long for s in scores),
        "empty_or_scriptless": sum(1 for s in scores
                                   if s.devanagari_ratio is None),
        "script_ratio_cases": len(ratios),
        "latin_allowed_cases": sum(1 for s in scores if s.case.allows_latin),
        "devanagari_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "mean_chars": statistics.fmean([s.chars for s in scores]) if scores else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--llm-engine", default="explicit",
                        choices=["explicit", "http"])
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None,
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 is greedy, which is what produced the loop this "
                             "harness exists to measure. Qwen's guidance for "
                             "these models is 0.7 with top_p 0.8 and top_k 20")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="required for a sampled run to be reproducible")
    parser.add_argument("--system-variant", default="default",
                        help="system prompt variant: default, or grounded, "
                             "which adds an instruction to admit ignorance "
                             "instead of inventing. Sampling was measured and "
                             "refuted; this is the untested lever that remains, "
                             "and declined_rate is what it targets")
    parser.add_argument("--out", default="results/answer_quality/scores.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    if args.temperature > 0 and args.seed is None:
        parser.error("a sampled run needs --seed, or its numbers cannot be "
                     "reproduced and the comparison is not a comparison")

    from llm.engines import build_llm
    from llm.prompting import build_chat_prompt, system_prompt_for

    dtype = None
    if args.dtype:
        import torch

        dtype = getattr(torch, args.dtype)

    generator, tokenizer, info = build_llm(
        args.llm_engine, model=args.llm_model, device=args.device, dtype=dtype,
        base_url=args.llm_base_url, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k,
        presence_penalty=args.presence_penalty, seed=args.seed,
    )
    print(f"engine: {info}", flush=True)

    system = system_prompt_for("hi", args.system_variant)
    print(f"system prompt variant: {args.system_variant} "
          f"({len(system)} chars)", flush=True)
    scores: list[Score] = []
    for index, case in enumerate(CASES, start=1):
        prompt = build_chat_prompt(tokenizer, system, case.prompt)
        answer = generator.generate(prompt, max_new_tokens=args.max_new_tokens).text
        result = score(case, answer)
        scores.append(result)
        verdict = {"factual": result.correct, "instruction": result.obeyed,
                   "unanswerable": result.declined}[case.kind]
        mark = "ok  " if verdict else "FAIL"
        flags = "".join(f" [{name}]" for name, hit in
                        (("loop", result.looping), ("long", result.too_long),
                         ("forbidden", bool(result.forbidden_hit))) if hit)
        print(f"{mark} {index:2d}/{len(CASES)} {case.kind:<12}{flags} "
              f"{case.prompt}", flush=True)
        print(f"        {answer[:140]}", flush=True)

    summary = summarise(scores)
    report = {
        "note": args.note, "engine": info, "max_new_tokens": args.max_new_tokens,
        "system_variant": args.system_variant,
        "system_prompt": system,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k,
                     "presence_penalty": args.presence_penalty,
                     "seed": args.seed},
        "summary": summary,
        "scores": [s.as_dict() for s in scores],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"report: {out_path}")
    print("Apply the pre-registered rule in docs/EXPERIMENTS.md; this harness "
          "does not decide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
