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

**Bytes, not decoded text.** The first version compared each step's decoded
text with the served text. A step that ends inside a character decodes to
U+FFFD, which is never a prefix of real text, and Qwen3 splits a Devanagari
character across tokens within the first four tokens of every Hindi answer in
the parity set -- so it reported a divergence on answers that are
byte-identical, and measured margins at steps where nothing diverged. Steps are
now the UTF-8 bytes the tokens produced, compared with the served text's bytes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

# `python scripts/logit_margins.py` puts scripts/ on the path, not the repo
# root, and `scripts` is not an installed package: without this the
# documented command fails at `from scripts.engine_parity import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: A divergence whose margin is below this fraction of the run's median margin
#: is called a near-tie. Relative, because logit scale varies by model and by
#: position; 0.25 is a judgement, stated here so it can be argued with rather
#: than buried in a comparison.
NEAR_TIE_FRACTION = 0.25


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table byte-level BPE vocabularies
    (Qwen's included) are written in: vocabulary character -> byte."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1),
                 *range(ord("®"), ord("ÿ") + 1)]
    codes, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes, strict=True)}


BYTE_DECODER = _byte_decoder()


def token_bytes(tokenizer, token_id: int, special_ids=frozenset()) -> bytes:
    """The UTF-8 bytes one token contributes to the output.

    Its decoded text cannot say: a token holding part of a character decodes to
    U+FFFD. Special tokens contribute nothing, as under
    `decode(skip_special_tokens=True)`.
    """
    if token_id in special_ids:
        return b""
    piece = tokenizer.convert_ids_to_tokens(token_id)
    try:
        return bytes(BYTE_DECODER[char] for char in piece)
    except KeyError:                       # an added token stored as plain text
        return tokenizer.decode([token_id]).encode("utf-8")


def divergence_step(steps: Sequence[str] | Sequence[bytes], served: str) -> int | None:
    """First step whose output stops being a prefix of `served`.

    Pass the cumulative bytes of the generated tokens. Works on the incremental
    decode rather than on character offsets, because a character offset does
    not identify a token: byte-level BPE splits a 3-byte Devanagari character
    across two tokens, so the character where two strings differ can sit inside
    a token rather than at its boundary. And on bytes rather than decoded text,
    because a step that ends inside a character decodes to U+FFFD and would
    read as a divergence where there is none. Text steps are still accepted.

    Both engines strip their output, so leading whitespace is ignored, and
    whitespace the reference adds after the served text's end is not a
    divergence. A served text cut inside a character by the token cap ends in
    U+FFFD; its last character is unknowable, so reference bytes that stop
    inside a character there are not a divergence either.

    `None` when the reference never leaves the served text's prefix — the two
    agreed for as long as the reference ran, and there is no divergence to
    explain.
    """
    as_bytes = bool(steps) and isinstance(steps[0], bytes)
    cut = as_bytes and served.endswith("�")
    target = served.rstrip("�").encode("utf-8") if cut else (
        served.encode("utf-8") if as_bytes else served)
    for index, step in enumerate(steps):
        step = step.lstrip()
        rest = step[len(target):] if step.startswith(target) else None
        past_end = rest is not None and (not rest.strip() or (cut and _partial_character(rest)))
        if not target.startswith(step) and not past_end:
            return index
    return None


def _partial_character(data: bytes) -> bool:
    """True for the leading bytes of one UTF-8 character that stop short of it."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        return error.reason == "unexpected end of data" and error.start == 0 and len(data) < 4
    return False


def locate_divergence(step_bytes: Sequence[bytes], served: str,
                      stopped_on_eos: bool) -> int | None:
    """`divergence_step`, plus the case it cannot see: the reference stopped
    where the served text went on. Stopping is a choice too, made at the step
    after the last token, and the margin there is the reference's confidence in
    ending the answer."""
    at = divergence_step(step_bytes, served)
    produced = step_bytes[-1] if step_bytes else b""
    if (at is None and stopped_on_eos
            and served.encode("utf-8")[len(produced.strip()):].strip()):
        return len(step_bytes)
    return at


def served_token_at(tokenizer, served: str, offset: int, special_ids=frozenset()) -> int | None:
    """The served text's token that starts at byte `offset`.

    From the served text's canonical tokenization, because the parity report
    records the engine's text and not its token ids. `None` when no token
    boundary falls at `offset` -- a guess the tokenization cannot support is not
    made.
    """
    position = 0
    for token in tokenizer(served, add_special_tokens=False).input_ids:
        if position == offset:
            return token
        position += len(token_bytes(tokenizer, token, special_ids))
        if position > offset:
            return None
    return None


def divergence_detail(tokenizer, logits, prefix: bytes, served: str,
                      special_ids=frozenset(), top: int = 5) -> dict:
    """What the reference weighed at the divergence.

    The verdict uses the top-two margin, as registered. That margin reads as a
    near-tie only if the served engine's token was the runner-up, so this also
    reports where the served text's own next token sat in the reference's
    ranking. Recorded beside the verdict; it does not change it.
    """
    import torch

    rest = served.encode("utf-8")[len(prefix):]
    values, indices = torch.topk(logits, top)
    candidates = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        piece = token_bytes(tokenizer, index, special_ids)
        candidates.append({
            "token_id": index, "text": piece.decode("utf-8", "replace"),
            "bytes": piece.hex(), "logit": value,
            "consistent_with_served": bool(piece) and rest.startswith(piece),
        })
    guess = served_token_at(tokenizer, served, len(prefix), special_ids)
    served_token = None
    if guess is not None:
        served_token = {
            "token_id": guess,
            "text": token_bytes(tokenizer, guess, special_ids).decode("utf-8", "replace"),
            "rank": int((logits > logits[guess]).sum()),
            "gap_to_top": float(logits.max() - logits[guess]),
        }
    return {"served_continuation": rest.decode("utf-8", "replace")[:24],
            "top": candidates, "served_token": served_token}


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
    special_ids = frozenset(tokenizer.all_special_ids)

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
        step_logits = []
        step_bytes: list[bytes] = []
        generated: list[int] = []
        produced = b""
        stopped_on_eos = False
        past = None
        with torch.inference_mode():
            for _step in range(args.max_new_tokens):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1, :].float()
                step_logits.append(logits.cpu())
                top2 = torch.topk(logits, 2)
                margins.append(float(top2.values[0] - top2.values[1]))
                token = int(top2.indices[0])
                if token in eos_ids:
                    stopped_on_eos = True
                    break
                generated.append(token)
                produced += token_bytes(tokenizer, token, special_ids)
                step_bytes.append(produced)
                ids = torch.tensor([[token]], device=device)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        # The byte walk is only trusted when it reproduces the decode.
        bytes_ok = produced.decode("utf-8", "replace") == text
        at = locate_divergence(step_bytes, served, stopped_on_eos)
        median = statistics.median(margins) if margins else None
        margin = margins[at] if at is not None and at < len(margins) else None
        prefix = step_bytes[at - 1].lstrip() if at else b""
        finding = {
            "prompt": question,
            "steps": len(margins),
            "diverged_at_step": at,
            "margin_at_divergence": margin,
            "median_margin": median,
            "min_margin": min(margins) if margins else None,
            "verdict": classify(margin, median) if bytes_ok else "unknown",
            "reference_prefix_at_divergence": prefix.decode("utf-8", "replace"),
            "reference_text": text.strip(),
            "matches_recorded_reference": text.strip() == item.get("reference"),
            "bytes_reproduce_decode": bytes_ok,
            "stopped_on_eos": stopped_on_eos,
            "at_divergence": (
                divergence_detail(tokenizer, step_logits[at], prefix, served,
                                  special_ids)
                if at is not None and at < len(step_logits) else None),
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
