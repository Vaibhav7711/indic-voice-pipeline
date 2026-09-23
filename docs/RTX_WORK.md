# Agent brief: benchmarking this project on the local RTX box

You are an autonomous coding agent working in this repository on a machine
with an RTX 4060 (**6 GB VRAM**), Ubuntu under WSL2 with a native Ubuntu dual
boot available. There is **no human in the loop**. Work through the plan
below, record every result as a file, commit the evidence, and stop with a
written note when you hit something that genuinely needs a person.

Your job is not to make the project look good. It is to turn a list of
unmeasured questions into recorded answers, and to say plainly when an answer
is worse than hoped.

## Read first, in this order

1. `README.md` — what the system is, the repo map, the correctness policy.
2. `docs/EXPERIMENTS.md` — the ledger. Every number the project claims, with
   its conditions. This is where your results go.
3. `docs/STREAMING.md` — the streaming session, the agent turn, and which
   latency fields are measured versus approximated.
4. `git log --oneline -40` — the commit messages record *why* things changed.

## Current state

| Component | Status | Evidence |
| --- | --- | --- |
| ASR (whisper-large-v3-turbo + Hindi LoRA v2) | **shipped**: 23.83% WER, 8.43% CER, 694 ms p50, RTF 0.066 on 300 seeded FLEURS-hi test clips | `docs/EXPERIMENTS.md`, `results/eval/` |
| Explicit encoder/decoder runtime | validated token-identical to `generate()` on real Hindi audio | `results/gpu_validation/report.json` |
| Streaming ASR + adaptive VAD | validated on 100 clips; streaming at offline parity | `results/streaming_eval/` |
| Agent turn (LLM stream → sentences → TTS → playback, barge-in) | logic validated; one live turn measured at 1.57 s response latency | `results/gpu_validation/report.json` |
| Dialogue memory | unit-tested only | — |
| ASR decode guards (no-speech, repetition loop) | **added, effect on WER never measured** | none — this is task 1 |
| LLM choice | **open.** Qwen3-0.6B answers Hindi questions by restating them | none |
| TTS backend (edge vs local MMS) | **open** | none |
| Incremental finals, semantic endpointing | **open**, both default off | none |
| CTranslate2 engine tier | validated on CPU only (fp32 token-identical, int8 2.7× faster) | commit `e52cc08` |
| Compiled decode (static cache + CUDA graphs) | token-identical, but **0.9× — slower — on a T4**; unmeasured on Ada | `results/gpu_validation/report.json` |
| Live turns with real microphone and speakers | **never done** | none |
| `data/hard_set/` | empty; the ledger requires per-category numbers for a result entry | — |
| ASR v3 re-run | decided **not** to do; recorded with reasons | `docs/EXPERIMENTS.md` |

550 automated tests pass on CPU; 8 more are GPU-gated and will run here.

## Environment bring-up

Run these and check each gate before proceeding. If a gate fails, fix it or
record it as blocked — do not proceed past a failing gate and report numbers
from a broken environment.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,demo,audio]" faster-whisper ctranslate2 bitsandbytes
python scripts/preflight.py
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch; print('bf16 native:', torch.cuda.is_bf16_supported())"
```

| Gate | Expected | If it fails |
| --- | --- | --- |
| `torch.cuda.is_available()` | True | stop; record blocked |
| `bf16 native` | True on Ada | note it; `pick_dtype` adapts either way |
| free VRAM at idle | ≥ 5 GB | a desktop session may be holding VRAM; note the baseline and subtract it from every budget below |
| `pytest` | 550 passed, 8 skipped | fix before benchmarking; a failing suite invalidates everything after it |

### The adapter

Not in git. Fetch from the Hub checkpoint repo the training run wrote to:

```bash
huggingface-cli login
python - <<'PY'
from huggingface_hub import snapshot_download, whoami
repo = f"{whoami()['name']}/whisper-turbo-hindi-lora-ckpt"
print("adapter:", snapshot_download(repo, allow_patterns=["best/*"],
                                    local_dir="models/v2-final") + "/best")
PY
export ADAPTER=$PWD/models/v2-final/best
```

Verify it contains `adapter_config.json` and `adapter_model.safetensors`.

### Audio: which machine you are on decides what you can test

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

- **Devices listed (native Ubuntu, or WSL2 with working WSLg audio):** the
  live-audio tasks are available to you. Run them.
- **No devices (typical WSL2):** the live microphone and speaker tasks are
  **blocked and must not be faked**. Do the file-replay equivalents
  (`scripts/live_agent.py --input-wav <clip> --sink buffer`), record that
  real barge-in against a device buffer remains unmeasured, and write the
  reason into `results/BLOCKED.md`. Do not report a file replay as a live
  turn.

## The 6 GB budget — this constrains what you may deploy

`llm_bakeoff` loads one model at a time, so it can benchmark models the agent
cannot run. The agent needs Whisper **and** the LLM resident together, because
the streaming session needs Whisper for the next utterance's partials;
sequential offload would add ~0.5–1 s per turn and break partials.

| Agent stack | Weights | On 6 GB |
| --- | ---: | --- |
| turbo + Qwen3-0.6B + MMS | 2.7 GB | fits; 0.6B is the known weak link |
| turbo + **Qwen3-1.7B 4-bit** + MMS | 2.4 GB | **the target if the bake-off justifies it** |
| turbo + Qwen3-1.7B fp16 + MMS | 4.7 GB | too tight with KV cache and activations |
| CT2-int8 turbo + 1.7B 4-bit + MMS | 1.7 GB | most headroom; requires `ct2_check` to pass |
| anything + Qwen3-4B | 7.5 GB+ | out of reach |

`llm.loader.estimate_vram_gib(param_millions, quantization)` gives these
numbers; recompute rather than trusting the table if a model changes.

## Plan

### Phase 1 — regression check (do this first)

The decode guards added in commit `55f5c1e` (no-speech threshold 0.6,
repetition loop guard) changed serving behaviour and their effect on WER was
never measured. A no-speech suppression on a clip that *does* contain speech
deletes a whole utterance. Everything downstream is meaningless if this is a
regression.

```bash
python scripts/bench_all.py --adapter "$ADAPTER" \
    --only guards_on,guards_off,compare_guards
```

Read `results/eval/compare-guards.json` and the `decode_guards_fired` block in
each `metrics.json`.

- **WER unchanged or better, guards fired on 0–2 clips** → keep the defaults,
  record the numbers, continue.
- **WER worse, or no-speech fired on clips whose reference is non-empty** →
  this is a regression you must fix, not report around. Raise
  `--no-speech-threshold` until it fires only on genuinely empty clips,
  re-run, and record both the old and new values in the ledger.

### Phase 2 — the open questions

```bash
python scripts/bench_all.py --adapter "$ADAPTER" \
    --llm-models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-1.7B:4bit \
    --skip guards_on,guards_off,compare_guards
```

~2 hours unattended. It writes a log per step and
`results/bench_manifest.json`. A failing step does not stop the rest; check
the manifest for non-zero exit codes and read that step's log before
concluding anything about it.

### Phase 3 — live turns

Only if audio devices exist. Twenty turns, varied utterance lengths, at least
three deliberate barge-ins:

```bash
python scripts/live_agent.py --adapter "$ADAPTER" --tts mms \
    --llm-model <winner from phase 2> --log results/live/turns.jsonl
```

Then summarise the distribution — not one turn — of
`response_latency_ms` and its four segments. One turn is an anecdote; the
project has already been misled once by quoting a single number.

## Pre-registered decision rules

These are fixed **now**, before you see the results, so that a disappointing
number cannot be reinterpreted into a success. Apply them literally.

| Question | Adopt the change if | Otherwise |
| --- | --- | --- |
| Incremental finals (`early-incr`) | `wer_vs_offline` rises < 1.0 pp versus `early` **and** mean `endpoint_to_final_ms` drops | keep off, record the measured cost |
| Semantic endpointing (`early-sem`) | `clips_split` drops from 16/100 to single digits at < 150 ms mean added `endpoint_to_final_ms` | keep off |
| LLM | highest `devanagari_ratio_mean` with **zero** `think_leaks`, `first_sentence_ms_p50` < 800 ms, and a stack that fits the 6 GB table | keep Qwen3-0.6B and record why the alternatives failed |
| 4-bit quantization | 4-bit's Hindi quality is not visibly worse in `outputs.jsonl` **and** it is the only way the chosen model fits | use fp16 if it fits, else the smaller model |
| TTS backend | lower `first_chunk_ms_p50` with acceptable audio in the saved WAVs | keep edge-tts and record the local model's cost |
| CTranslate2 engine | `ct2_matches_explicit` passes (token-identical at fp16, ≤ 5% WER apart at int8) **and** speedup > 1.3× | keep the explicit runner |
| Compiled decode | `llm_compiled_matches_eager` shows tokens match **and** speedup > 1.1× | keep it off; record the Ada number next to the T4's 0.9× |

Quality judgements that need ears (Hindi fluency of an LLM answer, TTS voice
quality) are the one place you must not decide alone: score what you can
automatically, write the candidates and their outputs into the ledger, and
mark the choice **pending human listening**.

## Recording protocol

Every result is a file in the repo. A number that exists only in a terminal
did not happen.

1. **Evidence** stays where the harness put it: `results/eval/<run>/`,
   `results/llm_bakeoff/`, `results/tts_bakeoff/`, `results/streaming_eval/`,
   `results/gpu_validation*/`, `results/live/turns.jsonl`,
   `results/bench_manifest.json`.
2. **Commit the JSON and Markdown, never the audio or the weights.**
   `.gitignore` already excludes WAVs, MP3s and checkpoints; do not force-add
   them.
3. **Append to `docs/EXPERIMENTS.md`** using the result-entry template at the
   end of that file. Include the conditions (GPU, dtype, split, seed, limit,
   normalization level) and the path to the evidence. Leave a field blank
   rather than filling it from a different run.
4. **Commit per logical result**, conventional-commit style, body explaining
   what was decided and why. End every commit message with:
   `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
5. **Write `results/BLOCKED.md`** for anything you could not do, with the
   reason and what a person would need to provide.

## Hard rules

Each of these exists because violating it cost real time on this project.

1. **Measure before attributing.** A 4062 ms latency segment was called a TTS
   problem; it was mostly the LLM still generating. If you cannot name the
   file a claim comes from, do not make the claim.
2. **`None`, never `0.0`,** for anything unmeasured. A zero is averaged into a
   benchmark; a `None` forces the question.
3. **The explicit runners are the reference.** An engine or a compiled path is
   a faster way to serve the same model and must match them token-for-token
   (fp16/fp32) or within a stated WER (int8) before it is used anywhere.
4. **Do not move a target after seeing the result.** v2 missed its
   pre-registered 22.0% WER and that is recorded as a miss, with its cause
   (multi-GPU DataParallel halved the optimizer steps). Do the same.
5. **One variable per experiment.** If you must change two, say so explicitly
   in the ledger and do not attribute the result to either.
6. **Never train on, or tune against, the test split.** Model selection uses
   the validation split; `--seed`/`--limit` must match between compared runs.
7. **`git fetch && git reset --hard origin/main`, not `git pull`** — local
   installs and result files make `pull` refuse, and `-q` hides the refusal.
8. **Restart the interpreter after changing modules.** Python caches imports;
   a fix that "did not work" was usually a stale one.
9. **Do not delete or rewrite existing evidence.** Superseded results get a
   note saying why, not a deletion.

## Requires a human — stop and write it down

- **Curating `data/hard_set/`.** Clip selection and per-category labelling are
  judgement calls. Do not synthesise a hard set.
- **Final LLM and TTS choice**, insofar as it depends on hearing Hindi
  fluency and voice quality. Narrow it to a ranked shortlist with evidence.
- **Publishing anything** (Hub uploads, pushing to a new remote, making a repo
  public).
- **Live-audio tasks if no audio device exists.** Do the file-replay version
  and mark the gap.
- **Anything destructive**: rewriting git history, deleting `results/`,
  force-pushing.

## Definition of done

- Phase 1 concluded: the decode guards are either confirmed harmless or fixed,
  with numbers in the ledger.
- Every row of the "open" table in Current State has either a recorded
  measurement and a decision applied from the rules above, or an entry in
  `results/BLOCKED.md` explaining why not.
- `docs/EXPERIMENTS.md` has an entry per result, with evidence paths.
- `pytest` still passes, `ruff check .` is clean, and everything is committed
  and pushed.
- A short `results/SESSION.md` summarising what was decided, what changed, and
  what the next session should pick up — written for a reader who was not
  here.
