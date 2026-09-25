"""Interleaved A/B over the turn's latency knobs, on one set of loaded weights.

`docs/EXPERIMENTS.md` names three latency changes and measures none of them.
This runs them. All three are configuration, not weights, so every arm shares
one loaded model and one device state:

  * `history_tokens` -- prefill grows with the dialogue prompt. Measured:
    `first_token_ms` went 209 -> 1474 ms over 12 turns, Pearson r = 0.99
    against turn index. Predicted worth ~1.2 s at the tail.
  * `unit_chars` -- how much text must exist before the first audio. The
    measured first unit was 41.2 characters at ~61 ms each; halving the cap
    is predicted to move first audio from ~2446 ms to ~1841 ms.
  * `llm_engine` -- `explicit` against an OpenAI-compatible server. The LLM
    was 3324 ms of a 4400 ms turn, so this is the one with seconds in it.

**Arms are interleaved, not run back to back.** A GPU warms, a notebook's
neighbour starts training, a network TTS voice has a bad minute. Running arm A
ten times and then arm B ten times attributes all of that drift to the arm.
One turn of each per round, in rotation, spreads it across both.

**The transcript is fixed, not spoken.** This measures the LLM and TTS half of
the turn, which is where 94% of the controllable latency was. ASR is exercised
by `benchmarks/streaming_eval.py` and is off the critical path anyway. Feeding
identical text to every arm is what makes the arms comparable at all -- a
microphone would vary the prompt between them.

    python scripts/latency_ab.py --rounds 8 \
        --arm baseline \
        --arm history200:history_tokens=200 \
        --arm units30:unit_chars=30 \
        --arm both:history_tokens=200,unit_chars=30

Every turn record is written to `--out` as JSONL with its arm, and the summary
prints p50/p90 per arm. Nothing here decides anything: the rules in
`docs/EXPERIMENTS.md` do, and they are pre-registered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

#: Turns whose content stays fixed across arms, so the only difference between
#: arms is the configuration. Short questions a voice assistant would get.
DEFAULT_PROMPTS = [
    "नमस्ते, आज मौसम कैसा है?",
    "मुझे एक छोटी कहानी सुनाओ।",
    "भारत की राजधानी क्या है?",
    "तुम मेरी क्या मदद कर सकते हो?",
    "थोड़ा धीरे बोलो please।",
]

KNOBS = ("history_tokens", "unit_chars", "llm_engine", "history_turns")


def parse_arm(spec: str) -> tuple[str, dict]:
    """`name:knob=value,knob=value` -> (name, overrides).

    An unknown knob is an error rather than a no-op: an A/B whose treatment
    silently did not apply reports "no difference" and looks like a result.
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"arm {spec!r} has no name")
    overrides: dict = {}
    for item in filter(None, (part.strip() for part in rest.split(","))):
        knob, _, value = item.partition("=")
        knob = knob.strip()
        if knob not in KNOBS:
            raise ValueError(f"unknown knob {knob!r} in arm {name!r}; known: {KNOBS}")
        value = value.strip()
        overrides[knob] = value if knob == "llm_engine" else int(value)
    return name, overrides


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile. `None` for an empty sample, never 0.0."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


#: The metric this sweep is judged on. `response_latency_ms` -- user stops
#: talking to agent starts talking -- is `None` here and correctly so: it
#: needs the endpoint-to-final segment, and these turns are driven by fixed
#: text with no speech in them. Supplying a plausible number for a segment
#: that did not happen would make every arm's headline figure partly
#: invented. So the sweep reports what it can actually observe: committed
#: transcript to first audio, which is the two segments the arms change.
PRIMARY = "transcript_to_first_audio_ms"


def metrics_of(record: dict) -> dict:
    """The metrics block of a turn record, wherever it lives.

    `TurnResult.as_dict()` nests them under "metrics", which is the shape
    `results/live/turns.jsonl` is already in and so the shape this script
    writes. Reading the latencies off the top level instead finds nothing,
    every field reports `None`, and the A/B looks like it ran and measured
    a pipeline that produces no audio. Hence one function, used by both the
    driver and the summary.
    """
    nested = record.get("metrics")
    block = dict(nested if isinstance(nested, dict) else record)
    first_token = block.get("final_transcript_to_first_llm_token_ms")
    to_audio = block.get("first_llm_token_to_playback_start_ms")
    if isinstance(first_token, (int, float)) and isinstance(to_audio, (int, float)):
        block[PRIMARY] = first_token + to_audio
    else:
        # One of the halves is missing, which means the turn did not reach
        # audio. That is not a fast turn; it is a turn that did not finish.
        block.setdefault(PRIMARY, None)
    return block


def summarize(records: list[dict]) -> dict:
    """Per-arm distribution of the latencies this project reports.

    A field absent from every record stays `None`. It is the difference
    between "this arm did not produce audio" and "this arm produced audio
    instantly", and averaging the second into a benchmark is how a broken run
    becomes a headline.
    """
    fields = (PRIMARY, "response_latency_ms",
              "final_transcript_to_first_llm_token_ms",
              "first_token_to_first_unit_ms", "tts_synthesis_ms", "total_turn_ms")
    blocks = [metrics_of(record) for record in records]
    summary: dict = {"turns": len(records)}
    for field in fields:
        values = [block[field] for block in blocks
                  if isinstance(block.get(field), (int, float))]
        summary[field] = {
            "n": len(values),
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "mean": statistics.fmean(values) if values else None,
        }
    tokens = [block["llm_generated_tokens"] for block in blocks
              if isinstance(block.get("llm_generated_tokens"), int)]
    summary["mean_generated_tokens"] = statistics.fmean(tokens) if tokens else None
    return summary


def build_arm(overrides: dict, *, base: dict, generators: dict, synth, sink_factory):
    """One `VoiceTurn` per arm, sharing weights with every other arm."""
    from agent import Conversation, VoiceTurn

    engine = overrides.get("llm_engine", base["llm_engine"])
    if engine not in generators:
        raise ValueError(
            f"arm asks for llm_engine={engine!r} but only {sorted(generators)} were "
            f"loaded; pass --llm-engine {engine} so it is built up front",
        )
    generator, tokenizer = generators[engine]
    turns = overrides.get("history_turns", base["history_turns"])
    conversation = None
    if turns > 0:
        conversation = Conversation(
            max_turns=turns,
            max_history_tokens=overrides.get("history_tokens", base["history_tokens"]),
            tokenizer=tokenizer,
        )
    return VoiceTurn(
        generator, synth, response_language="Hindi", sink_factory=sink_factory,
        conversation=conversation,
        max_unit_chars=overrides.get("unit_chars", base["unit_chars"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", action="append", default=[], dest="arms",
                        help="name[:knob=value,...]; repeatable. Knobs: "
                             + ", ".join(KNOBS))
    parser.add_argument("--rounds", type=int, default=6,
                        help="turns per arm; arms are interleaved within a round")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm-engine", action="append", default=[],
                        dest="engines", choices=["explicit", "http"],
                        help="engines to load; repeat to compare them as arms")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--tts", default="edge", choices=["edge", "mms"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--history-tokens", type=int, default=800)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--unit-chars", type=int, default=60)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--prompt", action="append", default=[], dest="prompts")
    parser.add_argument("--out", default="results/latency_ab/turns.jsonl")
    parser.add_argument("--summary", default=None,
                        help="write the summary JSON here (default: next to --out)")
    parser.add_argument("--note", default="", help="recorded verbatim in the summary")
    args = parser.parse_args(argv)

    specs = args.arms or ["baseline"]
    try:
        arms = [parse_arm(spec) for spec in specs]
    except ValueError as error:
        parser.error(str(error))
    if len({name for name, _ in arms}) != len(arms):
        parser.error("arm names must be unique; they label the records")

    engines = args.engines or ["explicit"]
    wanted = {overrides.get("llm_engine", engines[0]) for _, overrides in arms}
    for engine in wanted:
        if engine not in engines:
            parser.error(f"an arm asks for llm_engine={engine}; add --llm-engine {engine}")

    from llm.engines import build_llm
    from tts.streaming import SpeechStream  # noqa: F401 - import cost off the clock

    generators: dict = {}
    engine_info: dict = {}
    for engine in engines:
        print(f"loading llm ({engine})…", flush=True)
        generator, tokenizer, info = build_llm(
            engine, model=args.llm_model, device=args.device,
            base_url=args.llm_base_url, api_key=args.llm_api_key,
        )
        generators[engine] = (generator, tokenizer)
        engine_info[engine] = info
        if engine == "http":
            # A cold server answers the first request seconds late; that is a
            # startup cost and must not land in the first arm's sample.
            print("  probe:", generator.probe(), flush=True)

    if args.tts == "edge":
        from tts import EdgeStreamingSynthesizer

        synth = EdgeStreamingSynthesizer(language=args.language)
    else:
        from tts.local import MmsTtsSynthesizer

        synth = MmsTtsSynthesizer(args.language)
        print(f"tts warm-up {synth.warm_up():.0f} ms", flush=True)

    from agent.audio import DecodingBufferSink

    def sink_factory():
        # No sound device: this runs on headless GPU boxes, and playback to a
        # real speaker would add a device's buffering to every measurement.
        return DecodingBufferSink(synth.format)

    base = {"history_tokens": args.history_tokens, "history_turns": args.history_turns,
            "unit_chars": args.unit_chars, "llm_engine": engines[0]}
    built = {}
    for name, overrides in arms:
        try:
            built[name] = build_arm(overrides, base=base, generators=generators,
                                    synth=synth, sink_factory=sink_factory)
        except ValueError as error:
            parser.error(str(error))

    prompts = args.prompts or DEFAULT_PROMPTS
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict]] = {name: [] for name, _ in arms}

    started = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for index in range(args.rounds):
            transcript = prompts[index % len(prompts)]
            for name, overrides in arms:
                turn = built[name]
                result = turn.run(transcript)
                record = result.as_dict()
                record.update({"arm": name, "overrides": overrides,
                               "round": index, "transcript": transcript})
                records[name].append(record)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                latency = metrics_of(record).get(PRIMARY)
                shown = f"{latency:7.1f} ms" if isinstance(latency, (int, float)) else "     n/a"
                print(f"round {index + 1}/{args.rounds}  {name:<14} {shown}", flush=True)

    summary = {
        "note": args.note,
        "primary_metric": PRIMARY,
        "rounds": args.rounds,
        "elapsed_s": round(time.time() - started, 1),
        "prompts": prompts,
        "base": base,
        "engines": engine_info,
        "tts": args.tts,
        "arms": {name: {"overrides": overrides, **summarize(records[name])}
                 for name, overrides in arms},
    }
    summary_path = Path(args.summary) if args.summary else out_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    def show(block: dict, field: str, key: str = "p50") -> str:
        value = block[field][key]
        return f"{value:10.1f}" if isinstance(value, (int, float)) else f"{'n/a':>10}"

    print(f"\n{'arm':<16}{'turns':>6}{'p50 ms':>10}{'p90 ms':>10}{'first tok':>11}"
          f"{'to unit':>10}")
    for name, _ in arms:
        block = summary["arms"][name]
        print(f"{name:<16}{block['turns']:>6}"
              f"{show(block, PRIMARY)}"
              f"{show(block, PRIMARY, 'p90')}"
              f"{show(block, 'final_transcript_to_first_llm_token_ms'):>11}"
              f"{show(block, 'first_token_to_first_unit_ms')}")
    print("p50/p90 are transcript -> first audio. response_latency_ms is n/a "
          "by construction: these turns carry no speech, so the "
          "endpoint-to-final segment does not exist and is not invented.")
    print(f"\nrecords: {out_path}\nsummary: {summary_path}")
    print("Apply the pre-registered rules in docs/EXPERIMENTS.md; this script "
          "does not decide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
