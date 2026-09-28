# Handoff: the GPU work this project still needs

Written for an agent with GPU access (an AWS node, or Kaggle reached through
AWS) and no prior context on this repository. Everything here is executable;
where a judgement is needed the rule is already fixed and quoted, and where a
human is required it says so instead of asking you to decide.

Repository: `https://github.com/Vaibhav7711/indic-voice-pipeline`, branch
`main`, at or after commit `53ae8b7`.

## What this project is, in four sentences

A correctness-first Hindi voice agent: streaming Whisper ASR with an explicitly
implemented encoder/decoder loop, a LoRA adapter for Hindi, VAD endpointing,
and an LLM turn pipelined sentence-by-sentence into TTS with barge-in. The
explicit implementations are the **reference**; faster tiers (CTranslate2 for
ASR, an external paged-KV serving engine for the LLM) must be shown equivalent
to them before they serve. Every number lives in `docs/EXPERIMENTS.md` beside
a committed evidence file. Decision rules are written **before** runs, and
applied literally afterwards.

## The state you are inheriting

| | status | evidence |
| --- | --- | --- |
| Streaming ASR, endpointing, candidate finals | working, measured | `docs/EXPERIMENTS.md` |
| Barge-in and follow-up turns | working, measured 2026-09-26 | `results/live/bargein.jsonl` |
| Serving engine (LLM over HTTP) | **measured at 2.7365× but NOT adopted** | `results/latency_ab/summary.json` |
| Engine correctness gate | **failing, 4/5** | `results/engine_parity/parity.json` |
| Answer quality, 4 decoder arms | measured; sampling refuted | `results/answer_quality/*.json` |
| Answer quality, grounded prompt arm | **not run** | — |
| Parity logit margins | **not run** | — |

Two things are unresolved, and both have a script and a pre-registered rule
waiting. That is your job.

## Task 1 — the grounded system prompt

**Why.** The measured run split the quality problem: instruction-following
**80%** with zero over-long answers, factual accuracy **4/10**, and **0 of 3**
unanswerable cases declined. The model obeys the prompt; the prompt never asked
it to admit ignorance, so asked the time with no clock it answered
*3:45 बजे*, and asked the weather with no weather data, *खुशी से बराबर है*.
`system_prompt_for(language, "grounded")` appends an instruction to say so
instead of inventing.

```bash
python -m benchmarks.answer_quality \
    --llm-engine explicit --llm-model Qwen/Qwen3-1.7B --dtype float16 \
    --system-variant grounded \
    --out results/answer_quality/grounded.json \
    --note "<gpu name>, system-variant=grounded, greedy"
```

Compare against the committed control, `results/answer_quality/greedy.json`.

**Pre-registered rule — apply literally, do not re-derive.** Adopt `grounded`
as the default if `declined_rate` reaches **≥ 2 of 3** *and* neither
`factual_accuracy` nor `instruction_obeyed` falls by more than **10 percentage
points**. A refusal instruction that also makes the model refuse answerable
questions is a worse assistant, not a better one.

**If `declined_rate` stays at 0**, the model is not following the instruction
and prompting is exhausted as a lever. Record that; do not iterate on wording.
The next step would be a larger checkpoint, which is not your call.

**Read the splits together.** `declined_rate` rewards refusing, so a model that
refuses everything scores 3/3 there while failing all ten factual cases.

## Task 2 — the parity margins

**Why.** The correctness gate reports that two greedy decoders over the same
weights produced different text (4/5 agreeing, 3/5 identical) and cannot say
whether that matters. Two readings fit: a near-tie that fp16 accumulation order
decided, or the engine computing something else. The distinguishing measurement
is the reference's **top-two logit margin at the step where they part**.

```bash
python scripts/logit_margins.py \
    --parity results/engine_parity/parity.json \
    --llm-model Qwen/Qwen3-1.7B --dtype float16 \
    --note "<gpu name>, explaining the 4/5 parity run"
```

This loads **only** the reference model — the served text comes from the parity
report — so no server is needed and there is no second copy of the weights in
VRAM.

**Pre-registered rule.** If **every** divergence sits below 25% of that run's
median margin, the engine agrees with the reference wherever the reference was
confident, which is as much as fp16 permits: record the gate as passing on the
margin criterion and the engine as adopted at its measured 2.7365×. If **any
one** divergence is at or above that fraction, the engine chose a token the
reference considered clearly worse — that is a defect, the engine is not
adopted, and the divergent step is where to look. Unanimity is required; the
rule does not average.

## The one thing most likely to invalidate your run

**Pass `--dtype float16` on every command, and check what GPU you have.**

`llm/loader.py:pick_dtype` returns bfloat16 on compute capability ≥ 8.0 and
float16 below it. Every committed number in this repository was measured on a
**Tesla T4 (sm_75)** in **float16**. So:

- **On a T4** (Kaggle, AWS `g4dn`): float16 is what you get anyway, and your
  numbers are directly comparable to the committed ones.
- **On sm_80 or later** (A10G, L4, A100, H100 — AWS `g5`, `g6`, `p4`): the
  default becomes bfloat16, which is a **different configuration**. Passing
  `--dtype float16` keeps the comparison valid. Record the GPU name in
  `--note` either way.

This is not hypothetical. A previous run measured the explicit arm in bfloat16
on a T4, where bf16 has no tensor cores, and its prefill carried a 5.9×
emulation penalty that read as the serving engine being fast. That confound is
why `--dtype` exists on these scripts.

## Setup

```bash
git clone https://github.com/Vaibhav7711/indic-voice-pipeline.git
cd indic-voice-pipeline
pip install -e ".[audio]"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Both tasks need only this repo and ~4 GB for Qwen3-1.7B. Neither needs the
external serving engine, a running server, or audio hardware. Set `HF_TOKEN` if
you have one — the download is rate-limited without it.

Expected runtime: about a minute per task after the model is cached.

## Hard rules

These are the project's, not suggestions.

1. **Do not invent or interpolate a number.** Every figure must come from a
   file the run produced. If something did not run, say it did not run.
2. **Absent is not zero.** A metric that was not measured is `None`, never
   `0.0`. A `0.0` gets averaged into a benchmark and becomes a claim.
3. **Commit or export the evidence before the session ends.** This is the
   failure that has cost this project most: three of the shipped ASR figures
   matched no file in the repository because the run that produced them died
   with its runtime. `results/` is gitignored in places, so use
   `git add -f results/answer_quality results/engine_parity` — and name the
   paths rather than adding the whole tree, which sweeps in large regenerable
   audio.
4. **Apply the pre-registered rules as written.** If a rule turns out to be
   unsatisfiable or mis-specified, record that as a finding with its reason and
   say what it did and did not change. Do not adjust a threshold to fit the
   data you just saw.
5. **Record the GPU name, the dtype, and the commit** in every `--note`.
6. **Report a refuted prediction as a result**, not a failure. The last run
   refuted its own sampling hypothesis and that is the most useful thing it
   produced.

## What to hand back

Append a section to `docs/EXPERIMENTS.md` containing: the GPU and dtype used,
the commit, both summary tables, the rule applied to each with its verdict, and
anything that surprised you. Then commit the evidence files and the ledger
entry together.

If either rule's verdict is ambiguous, write down why and leave it undecided.
An honest "undecided, because X" is worth more here than a decision the
evidence does not support.

## Not your call

- Whether **40% factual accuracy** is acceptable for this project. If it is a
  systems demonstration it is, and the README should say so; if it is meant to
  answer questions, the model changes and that is a new phase.
- The shipped v2 ASR adapter is **not obtainable** — it lives at a local path
  `/content/v2-final/best` with a blank model card, so its headline numbers are
  not reproducible by anyone. Do not attempt to re-train or substitute it.
- `data/hard_set/` is empty; curating it is a human judgement.

## Optional, if you have a second GPU

Kaggle's T4×2 makes two things possible that a single card blocks, both
currently recorded as unmeasured rather than rejected:

- **Qwen3-4B parity.** The gate loads the weights twice — the server holds a
  copy and the reference needs one — and two 4B copies exceed a 15 GiB card. On
  two devices it fits.
- **The engine's speculative profile**, `create_kaggle_t4x2_speculative_app`,
  which needs a 4B target and a 0.6B draft on separate devices.

Neither is required. If you run them, they are new experiments and need their
own pre-registered rules written before the run, not after.
