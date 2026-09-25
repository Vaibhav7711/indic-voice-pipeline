"""Does the served engine answer what the reference implementation answers?

The project's rule is that a faster tier has to be shown equivalent to the
explicit reference before it serves, and the ASR side already works this way:
`scripts/gpu_validation.py` checks CTranslate2 against the explicit Whisper
loop and adopted it at 1.5873% mean WER against a 5% tolerance. The LLM side
had no such gate, which is the more dangerous gap -- a serving engine with a
subtly different attention kernel, a rounding difference in its paged KV
cache, or a sampler that is not actually greedy produces fluent, plausible,
*different* Hindi, and nothing downstream would notice.

Greedy makes this testable. `SamplingParams` in `full-inference-engine`
documents `temperature = 0` as greedy and taking precedence over every other
knob, and `llm/engines/http_engine.py` sends `temperature: 0.0`; the explicit
runner is greedy too. Two greedy decoders over the same weights and the same
prompt must produce the same text. Where they diverge, the divergence point is
the evidence.

    python scripts/engine_parity.py --llm-base-url http://127.0.0.1:8000/v1

Exit status is 0 only if every prompt matched to the configured tolerance, so
this is usable as a gate rather than something to read and nod at.

**What a mismatch does and does not prove.** Divergence late in a long
response is expected and is not a defect: fp16 attention is not associative,
so two implementations can rank the top two candidates differently at one
step and then follow different-but-equally-valid continuations. That is why
the default tolerance compares the *prefix* -- the first `--prefix-chars`
characters, which is what the user actually hears before the sentence buffer
hands the first unit to TTS -- and reports full-text agreement separately
rather than gating on it. A divergence in the first few characters is a real
defect: it means prefill differs, not that decode drifted.

**Agreement depends on model size, and the report says so rather than leaving
it to be rediscovered.** A greedy decoder diverges at the first step where two
implementations rank the top two candidates differently, and how often that
happens depends on how close those candidates are. A small model has flatter
logits and smaller top-two margins, so an fp16 rounding difference flips a
tie more readily and the shared prefix is shorter; a larger model is more
confident per step and agrees for longer. Low agreement at 0.6B is therefore
weak evidence of an engine defect and strong evidence of a near-tie, which is
why this reports mean shared-prefix *fraction* alongside the pass/fail: it is
the number to compare across checkpoints, and a bare "they disagreed" is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Fixed prompts, in the languages this pipeline serves, exercising prefill
#: lengths from a few tokens to a realistic dialogue prompt.
DEFAULT_PROMPTS = [
    "नमस्ते, आज मौसम कैसा है?",
    "भारत की राजधानी क्या है?",
    "मुझे एक छोटी कहानी सुनाओ।",
    "What is the capital of India?",
    "थोड़ा धीरे बोलो please, मुझे समझ नहीं आया।",
]


def common_prefix_length(left: str, right: str) -> int:
    """Characters both strings agree on, which is where they stop agreeing."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def compare(reference: str, served: str, *, prefix_chars: int) -> dict:
    """One prompt's verdict. `agreed` is the gate; the rest is diagnosis."""
    shared = common_prefix_length(reference, served)
    # Measured against the reference's length, never the served response's. A
    # server that returns nothing shares a zero-length prefix, and requiring
    # min(prefix, len(served)) would ask it to match zero characters and pass
    # it -- a broken engine certified as equivalent. An empty response on
    # either side means the comparison did not happen, which is not agreement.
    required = min(prefix_chars, len(reference))
    agreed = bool(reference) and bool(served) and shared >= required
    return {
        "identical": reference == served,
        "shared_prefix_chars": shared,
        "required_prefix_chars": required,
        "agreed": agreed,
        "reference_chars": len(reference),
        "served_chars": len(served),
        # The first place they differ, with a little context on each side, so
        # a failure is readable without re-running anything.
        "divergence": None if reference == served else {
            "at": shared,
            "reference": reference[shared:shared + 40],
            "served": served[shared:shared + 40],
        },
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate. Rates are `None` on an empty sample rather than 1.0 or 0.0."""
    total = len(results)
    if not total:
        return {"prompts": 0, "agreed": None, "identical": None, "passed": False,
                "mean_shared_prefix_chars": None, "mean_shared_fraction": None}
    agreed = sum(1 for item in results if item["comparison"]["agreed"])
    identical = sum(1 for item in results if item["comparison"]["identical"])
    comparisons = [item["comparison"] for item in results]
    shared = [item["shared_prefix_chars"] for item in comparisons]
    # How much of the reference each served response reproduced. This is the
    # figure to compare between checkpoints: it distinguishes "the engine is
    # wrong" from "this model's top-two logits are close enough that an fp16
    # rounding difference decides the token".
    fractions = [item["shared_prefix_chars"] / item["reference_chars"]
                 for item in comparisons if item["reference_chars"]]
    return {
        "prompts": total,
        "agreed": agreed,
        "identical": identical,
        "agreement_rate": agreed / total,
        "identity_rate": identical / total,
        "mean_shared_prefix_chars": sum(shared) / len(shared) if shared else None,
        "mean_shared_fraction": (sum(fractions) / len(fractions)
                                 if fractions else None),
        "passed": agreed == total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="matches the turn's budget, so this exercises the "
                             "lengths the agent actually generates")
    parser.add_argument("--prefix-chars", type=int, default=24,
                        help="characters that must match. fp16 attention is not "
                             "associative, so late divergence is expected; an "
                             "early one means prefill differs")
    parser.add_argument("--prompt", action="append", default=[], dest="prompts")
    parser.add_argument("--out", default="results/engine_parity/parity.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    from llm.engines import build_llm
    from llm.prompting import build_chat_prompt, system_prompt_for

    print("loading the reference (explicit) runner…", flush=True)
    reference, tokenizer, reference_info = build_llm(
        "explicit", model=args.llm_model, device=args.device)
    print(f"  {reference_info}", flush=True)

    print("connecting to the served engine…", flush=True)
    served, _, served_info = build_llm(
        "http", model=args.llm_model, base_url=args.llm_base_url,
        api_key=args.llm_api_key, tokenizer=tokenizer)
    print(f"  {served_info}", flush=True)
    probe = served.probe()
    print(f"  probe: {probe}", flush=True)

    prompts = args.prompts or DEFAULT_PROMPTS
    results = []
    for index, question in enumerate(prompts, start=1):
        # Both engines get the byte-identical prompt, rendered once here with
        # the same builder the turn uses -- including enable_thinking=False,
        # without which Qwen3 answers from inside <think> and neither engine
        # produces anything speakable.
        prompt = build_chat_prompt(tokenizer, system_prompt_for("hi"), question)
        reference_text = reference.generate(
            prompt, max_new_tokens=args.max_new_tokens).text
        served_text = served.generate(prompt, max_new_tokens=args.max_new_tokens).text
        comparison = compare(reference_text, served_text,
                             prefix_chars=args.prefix_chars)
        results.append({
            "prompt": question,
            "reference": reference_text,
            "served": served_text,
            "comparison": comparison,
            "served_metrics": (served.last_metrics.as_dict()
                               if served.last_metrics else None),
        })
        mark = "ok  " if comparison["agreed"] else "FAIL"
        extra = "identical" if comparison["identical"] else (
            f"shared {comparison['shared_prefix_chars']} chars")
        print(f"{mark} {index}/{len(prompts)} {extra}: {question}", flush=True)
        if not comparison["agreed"]:
            divergence = comparison["divergence"]
            print(f"     at char {divergence['at']}", flush=True)
            print(f"     reference: {divergence['reference']!r}", flush=True)
            print(f"     served:    {divergence['served']!r}", flush=True)

    summary = summarize(results)
    report = {
        "note": args.note,
        "model": args.llm_model,
        "max_new_tokens": args.max_new_tokens,
        "prefix_chars": args.prefix_chars,
        "reference": reference_info,
        "served": served_info,
        "probe": probe,
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    fraction = summary["mean_shared_fraction"]
    print(f"\n{summary['agreed']}/{summary['prompts']} agreed on the first "
          f"{args.prefix_chars} characters; "
          f"{summary['identical']}/{summary['prompts']} identical throughout")
    if fraction is not None:
        print(f"mean shared prefix: {summary['mean_shared_prefix_chars']:.1f} chars "
              f"= {fraction:.1%} of the reference response")
        print("Compare that fraction across checkpoints before reading a low "
              "figure as an engine defect: a small model's top-two logits are "
              "close, so an fp16 rounding difference flips the token and the "
              "two greedy decoders part company early.")
    print(f"report: {out_path}")
    if not summary["passed"]:
        print("PARITY FAILED — the served engine is not answering as the "
              "reference does. Do not record its latency as this pipeline's "
              "latency until this is explained.", file=sys.stderr)
        return 1
    print("parity holds; the served engine may be recorded as this pipeline's "
          "serving tier")
    return 0


if __name__ == "__main__":
    sys.exit(main())
