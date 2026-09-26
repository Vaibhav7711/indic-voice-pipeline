"""At each parity divergence, how close was the reference's top-two call?

`scripts/engine_parity.py` reports that two greedy decoders over the same
weights produced different text — 4 of 5 prompts agreeing on the gated prefix,
3 identical throughout — and cannot say whether that matters. Both readings fit
the same evidence:

* **A near-tie.** The engine fuses chunked prefill with decode and uses its own
  attention kernels, so its fp16 accumulation order differs. Wherever the top
  two candidates are within rounding distance, the argmax flips, and both
  continuations are equally valid greedy decodes of the same model. Nothing is
  wrong.
* **A real difference.** The engine computes something else — a kernel bug, a
  wrong mask, a rotary embedding off by one — and the divergence is the symptom.

The distinguishing measurement is the **top-two logit margin at the step where
they part**. A margin near zero is a coin flip that fp16 noise decided; a wide
margin means the engine picked a token the reference considered clearly worse,
and that is a defect.

This runs the reference greedily, keeps the top-two margin at every step, finds
the first step whose decoded text stops being a prefix of the served text, and
reports the margin there.

    python scripts/logit_margins.py --parity results/engine_parity/parity.json

Only the reference model is loaded: the served text is read from the parity
report, so the server does not need to be running and there is no second copy
of the weights in VRAM — the constraint that blocked this on a single T4.

**Reading the result.** `margin_at_divergence` is in logits, not probabilities,
so scale matters: compare it against `median_margin` from the same run rather
than against an absolute threshold. A divergence at a margin far below that
median is a near-tie. One at or above it is not, and wants investigating.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

#: A divergence whose margin is below this fraction of the run's median margin
#: is called a near-tie. Relative, because logit scale varies by model and by
#: position; 0.25 is a judgement, stated here so it can be argued with rather
#: than buried in a comparison.
NEAR_TIE_FRACTION = 0.25


def divergence_step(step_texts: list[str], served: str) -> int | None:
    """First step whose decoded text stops being a prefix of `served`.

    Works on the incremental decode rather than on character offsets, because a
    character offset does not identify a token: byte-level BPE splits a
    3-byte Devanagari character across two tokens, so the character where two
    strings differ can sit inside a token rather than at its boundary.

    `None` when the reference never leaves the served text's prefix — the two
    agreed for as long as the reference ran, and there is no divergence to
    explain.
    """
    for index, text in enumerate(step_texts):
        if not served.startswith(text):
            return index
    return None


def classify(margin: float | None, median: float | None) -> str:
    """`near_tie`, `clear`, or `unknown`. Never a guess dressed as a verdict."""
    if margin is None or median is None or median <= 0:
        return "unknown"
    return "near_tie" if margin < NEAR_TIE_FRACTION * median else "clear"


def summarise(findings: list[dict]) -> dict:
    """Aggregate. Rates are `None` on an empty sample rather than 0.0."""
    judged = [f for f in findings if f["verdict"] != "unknown"]
    near = [f for f in judged if f["verdict"] == "near_tie"]
    return {
        "prompts": len(findings),
        "diverged": sum(1 for f in findings if f["diverged_at_step"] is not None),
        "judged": len(judged),
        "near_tie": len(near),
        "clear": len(judged) - len(near),
        "near_tie_rate": len(near) / len(judged) if judged else None,
        # The reading this script exists to support: every divergence a
        # near-tie means the engine agrees with the reference wherever the
        # reference was confident, which is as much as fp16 allows.
        "consistent_with_fp16_noise": bool(judged) and len(near) == len(judged),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parity", default="results/engine_parity/parity.json",
                        help="report from scripts/engine_parity.py; its served "
                             "texts are what the reference is compared against")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="must match the dtype the parity run used, or this "
                             "measures a different model's margins")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out", default="results/engine_parity/logit_margins.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    report_path = Path(args.parity)
    if not report_path.is_file():
        print(f"error: {report_path} does not exist. Run "
              f"scripts/engine_parity.py first -- this explains its "
              f"divergences and cannot invent them.", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))

    recorded = report.get("reference_dtype_requested")
    if recorded and recorded != args.dtype:
        print(f"error: the parity run used dtype {recorded!r} and this would "
              f"use {args.dtype!r}. Margins from different numerics do not "
              f"explain that run's divergences.", file=sys.stderr)
        return 2

    import torch

    from llm.loader import load_llm
    from llm.prompting import build_chat_prompt, system_prompt_for
    from scripts.engine_parity import system_language

    print(f"loading the reference in {args.dtype}…", flush=True)
    loaded = load_llm(args.llm_model, device=args.device,
                      dtype=getattr(torch, args.dtype))
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    findings = []
    for item in report["results"]:
        question, served = item["prompt"], item["served"]
        language = item.get("system_language") or system_language(question)
        prompt = build_chat_prompt(tokenizer, system_prompt_for(language),
                                   question)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # A plain greedy loop rather than LLMRunner: this needs the logits at
        # every step, and adding a capture hook to the serving runner to
        # support a diagnostic would put a branch in the path being validated.
        margins: list[float] = []
        step_texts: list[str] = []
        generated: list[int] = []
        past = None
        with torch.inference_mode():
            for _step in range(args.max_new_tokens):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1, :].float()
                top2 = torch.topk(logits, 2)
                margins.append(float(top2.values[0] - top2.values[1]))
                token = int(top2.indices[0])
                if token in eos_ids:
                    break
                generated.append(token)
                step_texts.append(
                    tokenizer.decode(generated, skip_special_tokens=True))
                ids = torch.tensor([[token]], device=device)

        at = divergence_step(step_texts, served)
        median = statistics.median(margins) if margins else None
        margin = margins[at] if at is not None and at < len(margins) else None
        finding = {
            "prompt": question,
            "steps": len(margins),
            "diverged_at_step": at,
            "margin_at_divergence": margin,
            "median_margin": median,
            "min_margin": min(margins) if margins else None,
            "verdict": classify(margin, median),
            "reference_prefix_at_divergence": (
                step_texts[at - 1] if at else ""),
        }
        findings.append(finding)
        shown = "agreed throughout" if at is None else (
            f"step {at}, margin {margin:.3f} vs median {median:.3f} "
            f"-> {finding['verdict']}")
        print(f"  {question[:40]:<42} {shown}", flush=True)

    summary = summarise(findings)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"note": args.note, "model": args.llm_model, "dtype": args.dtype,
         "near_tie_fraction": NEAR_TIE_FRACTION,
         "parity_report": str(report_path),
         "summary": summary, "findings": findings},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"report: {out_path}")
    if summary["consistent_with_fp16_noise"]:
        print("\nEvery divergence sits at a near-tie: the engine agrees with "
              "the reference wherever the reference was confident, which is as "
              "much as fp16 allows. The parity gate's remaining failures are "
              "explained.")
    elif summary["judged"]:
        print("\nAt least one divergence is NOT a near-tie -- the engine chose "
              "a token the reference considered clearly worse. That is a "
              "defect to investigate, not rounding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
