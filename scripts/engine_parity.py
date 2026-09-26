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
import re
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


def system_language(prompt: str) -> str:
    """Which system prompt a prompt should be answered under.

    Every prompt used to get the Hindi one, which instructs the model to answer
    entirely in Hindi. For the English prompt in the default set that produced
    `चीनी राजधानी है।` -- "it is the Chinese capital" -- and the parity report
    then read as a model that cannot name a capital city. The Hindi form of the
    same question answered correctly in both engines.

    Harmless for the comparison itself, since both engines get the identical
    prompt either way, and badly misleading for anyone reading the answers.
    Script is a sufficient signal here: these prompts are written, not
    transcribed.
    """
    return "hi" if re.search(r"[\u0900-\u097F]", prompt) else "en"


def common_prefix_length(left: str, right: str) -> int:
    """Characters both strings agree on, which is where they stop agreeing."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


REPLACEMENT = "�"


def corruption(reference: str, served: str) -> dict | None:
    """U+FFFD in the served text that is not in the reference.

    This is not a divergence and must not be reported as one. Qwen's byte-level
    BPE splits a 3-byte Devanagari character across two tokens, so a server
    that streams the diff of its decoded-so-far text emits U+FFFD for the
    partial character and then cannot retract it: the corrected text is no
    longer an extension of what it already sent. The real character is lost.

    It is invisible in English, where one byte is one character, so a gate run
    only on Latin prompts would pass. Calling this "the engine answers
    differently" sends the reader hunting a numerics bug in attention kernels
    when the defect is four lines of stream bookkeeping.
    """
    served_count = served.count(REPLACEMENT)
    if not served_count:
        return None
    return {
        "served_replacement_chars": served_count,
        "reference_replacement_chars": reference.count(REPLACEMENT),
        "diagnosis": (
            "the served stream contains U+FFFD replacement characters: its "
            "incremental decode is splitting multi-byte characters across "
            "chunks, not answering differently. This is a serving defect in "
            "the engine, it corrupts every Indic script, and a larger model "
            "will not fix it."
        ),
    }


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
        # Checked separately from agreement, because a corrupted stream and a
        # divergent decode need entirely different fixes.
        "corruption": corruption(reference, served),
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


def vram_preflight(model: str, *, free_gib: float | None) -> dict:
    """Will the reference copy fit beside the one the server already holds?

    This gate loads the weights a *second* time: the server has its own copy,
    and the reference runner needs one in this process to decode against. On a
    single card that doubles the weight cost, which the serving VRAM budget
    does not account for -- it counts the server's copy and Whisper.

    Checked before the load rather than discovered at 87% of it, because the
    failure mode is a killed process with no explanation, and the remedy
    (a smaller checkpoint, or generating the reference in a separate run) is
    not guessable from that.

    `free_gib=None` means the free VRAM could not be read, so this reports
    `unknown` rather than inventing a verdict.
    """
    # Imported here, not at module scope: every heavy import in this file is
    # lazy so `--help` works in a checkout with no editable install.
    from llm.engines.server_app import MODEL_WEIGHT_GIB

    weights = MODEL_WEIGHT_GIB.get(model)
    needed = None if weights is None else round(weights + 0.6, 2)
    if free_gib is None or needed is None:
        fits = None
    else:
        fits = free_gib >= needed
    return {
        "model": model,
        "reference_copy_gib": weights,
        # Weights plus room for activations and this process's CUDA context.
        "needed_gib": needed,
        "free_gib": None if free_gib is None else round(free_gib, 2),
        "fits": fits,
    }


def free_vram_gib() -> float | None:
    """Free VRAM on the default device, or `None` off-GPU or on error."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a missing figure must not fail the gate
        return None
    return free_bytes / 1024**3


def summarize(results: list[dict]) -> dict:
    """Aggregate. Rates are `None` on an empty sample rather than 1.0 or 0.0."""
    total = len(results)
    if not total:
        return {"prompts": 0, "agreed": None, "identical": None, "passed": False,
                "mean_shared_prefix_chars": None, "mean_shared_fraction": None}
    agreed = sum(1 for item in results if item["comparison"]["agreed"])
    identical = sum(1 for item in results if item["comparison"]["identical"])
    comparisons = [item["comparison"] for item in results]
    corrupted = [item for item in comparisons if item.get("corruption")]
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
        # Reported before the agreement figures are interpreted: while the
        # stream is corrupted, the agreement rate measures the corruption and
        # says nothing about whether the two decoders agree.
        "corrupted_prompts": len(corrupted),
        "served_replacement_chars": sum(item["corruption"]["served_replacement_chars"]
                                        for item in corrupted),
        "passed": agreed == total and not corrupted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
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
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="the REFERENCE runner's dtype. It must match what "
                             "the server loaded, or the two greedy decoders are "
                             "not over the same numerics and will diverge for "
                             "reasons that have nothing to do with the engine. "
                             "Defaults to float16 because that is what "
                             "llm/engines/server_app.py serves")
    parser.add_argument("--skip-vram-check", action="store_true",
                        help="load the reference copy even when the preflight "
                             "says it will not fit beside the server's copy")
    parser.add_argument("--prompt", action="append", default=[], dest="prompts")
    parser.add_argument("--out", default="results/engine_parity/parity.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    import torch

    from llm.engines import build_llm
    from llm.prompting import build_chat_prompt, system_prompt_for

    # The server already holds one copy of these weights; this process is about
    # to load a second. Say whether that fits before spending two minutes
    # finding out.
    preflight = vram_preflight(args.llm_model, free_gib=free_vram_gib())
    print(f"vram preflight: {preflight}", flush=True)
    if preflight["fits"] is False:
        print(
            f"\nerror: loading a reference copy of {args.llm_model} needs about "
            f"{preflight['needed_gib']} GiB and only {preflight['free_gib']} GiB "
            f"is free.\nThis gate decodes the same weights twice on one card -- "
            f"the server has a copy and\nso must this process. Either serve a "
            f"smaller checkpoint (Qwen/Qwen3-1.7B fits a\n15 GiB T4 twice over, "
            f"Qwen3-4B does not), or stop the server, run this gate\nagainst a "
            f"server on another device, and restart it.\nOverride with "
            f"--skip-vram-check if you believe the figure is wrong.",
            file=sys.stderr)
        if not args.skip_vram_check:
            return 3

    print(f"loading the reference (explicit) runner in {args.dtype}…", flush=True)
    reference, tokenizer, reference_info = build_llm(
        "explicit", model=args.llm_model, device=args.device,
        dtype=getattr(torch, args.dtype))
    print(f"  {reference_info}", flush=True)
    if args.dtype not in reference_info.get("dtype", ""):
        print(f"  WARNING: asked for {args.dtype}, loaded "
              f"{reference_info.get('dtype')}", flush=True)

    print("connecting to the served engine…", flush=True)
    served, _, served_info = build_llm(
        "http", model=args.llm_model, base_url=args.llm_base_url,
        api_key=args.llm_api_key, tokenizer=tokenizer)
    print(f"  {served_info}", flush=True)
    probe = served.probe()
    print(f"  probe: {probe}", flush=True)
    if probe.get("replacement_chars"):
        # The probe prompt is Devanagari, so this is known before a single
        # comparison runs. Say it now: everything printed below would otherwise
        # read as a decode divergence.
        print("\n  *** the probe came back with U+FFFD replacement characters.",
              flush=True)
        print("  *** The served stream is corrupting multi-byte characters; the",
              flush=True)
        print("  *** comparisons below measure that, not decoder agreement.\n",
              flush=True)

    prompts = args.prompts or DEFAULT_PROMPTS
    results = []
    for index, question in enumerate(prompts, start=1):
        # Both engines get the byte-identical prompt, rendered once here with
        # the same builder the turn uses -- including enable_thinking=False,
        # without which Qwen3 answers from inside <think> and neither engine
        # produces anything speakable.
        language = system_language(question)
        prompt = build_chat_prompt(tokenizer, system_prompt_for(language), question)
        reference_text = reference.generate(
            prompt, max_new_tokens=args.max_new_tokens).text
        served_text = served.generate(prompt, max_new_tokens=args.max_new_tokens).text
        comparison = compare(reference_text, served_text,
                             prefix_chars=args.prefix_chars)
        results.append({
            "prompt": question,
            "system_language": language,
            "reference": reference_text,
            "served": served_text,
            "comparison": comparison,
            "served_metrics": (served.last_metrics.as_dict()
                               if served.last_metrics else None),
        })
        mark = "CORRUPT" if comparison["corruption"] else (
            "ok  " if comparison["agreed"] else "FAIL")
        extra = "identical" if comparison["identical"] else (
            f"shared {comparison['shared_prefix_chars']} chars")
        print(f"{mark} {index}/{len(prompts)} {extra}: {question}", flush=True)
        if comparison["corruption"]:
            found = comparison["corruption"]["served_replacement_chars"]
            print(f"     CORRUPTED: {found} U+FFFD in the served text — "
                  f"not a divergence", flush=True)
        elif not comparison["agreed"]:
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
        "vram_preflight": preflight,
        # Recorded because a mismatch here invalidates the comparison, and a
        # report that does not state both dtypes cannot be checked for it
        # later.
        "reference_dtype_requested": args.dtype,
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

    if summary["corrupted_prompts"]:
        print(f"\n*** {summary['corrupted_prompts']}/{summary['prompts']} served "
              f"responses contain U+FFFD "
              f"({summary['served_replacement_chars']} characters total).",
              file=sys.stderr)
        print("*** This is a streaming defect, not a decode divergence, and it "
              "has to be fixed\n*** before the agreement figures below mean "
              "anything. Qwen's byte-level BPE\n*** splits a 3-byte Devanagari "
              "character across two tokens; a server that\n*** sends the diff "
              "of its decoded-so-far text emits U+FFFD for the partial\n*** "
              "character and cannot retract it, because the corrected text is "
              "no longer\n*** an extension of what it already sent. The real "
              "character is lost, and it\n*** is then spoken. A larger model "
              "does not fix this. English does not show\n*** it, because ASCII "
              "is one byte per character.", file=sys.stderr)

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
