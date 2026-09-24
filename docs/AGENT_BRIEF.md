# Agent brief: finishing this project on Google Colab

You are an autonomous coding agent finishing this repository. There is **no
local GPU available** — all GPU work happens in a Google Colab notebook on a
**Tesla T4 (16 GB, ~14.6 GiB usable)**. There is **no human in the loop**.
Work through the plan below, record every result as a file, commit the
evidence, and stop with a written note when you hit something that genuinely
needs a person.

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
| ASR (whisper-large-v3-turbo + Hindi LoRA v2) | **measured candidate; unpublished artifact**: guards-on 23.8265% WER / 8.4585% CER, 765.675 ms p50 on 300 seeded FLEURS-hi clips | `results/eval/v2-guards-on/`; adapter remains unavailable |
| Explicit encoder/decoder runtime | validated token-identical to `generate()` on real Hindi audio | `results/gpu_validation/report.json` |
| Streaming ASR + adaptive VAD | validated on 100 clips; streaming at offline parity | `results/streaming_eval/` |
| Agent turn (LLM stream → sentences → TTS → playback, barge-in) | logic validated; one live turn measured at 719.8 ms response latency (first-token field is a prefill proxy) | `results/gpu_validation/report.json` |
| Dialogue memory | unit-tested only | — |
| ASR decode guards (no-speech, repetition loop) | defaults kept: WER unchanged; both guards fired on 0/300 clips | `results/eval/compare-guards.json` |
| LLM choice | Qwen3-0.6B kept by fallback; no candidate met <800 ms first-sentence rule; human quality review pending | `results/llm_bakeoff/summary.json` |
| TTS backend (edge vs local MMS) | MMS latency-leading; final choice pending human listening, so edge remains default | `results/tts_bakeoff/summary.json` |
| Incremental finals, semantic endpointing | both remain off by fixed decision rules | `results/streaming_eval/summary.json` |
| CTranslate2 engine tier | **adopt int8 tier**: 1.3145×, 1.5873% WER vs explicit | `results/gpu_validation-ct2/report.json` |
| Compiled decode (static cache + CUDA graphs) | token-identical but **0.8526× — slower — on a T4**; off | `results/gpu_validation-ct2/report.json` |
| Live turns (browser mic → agent → playable audio) | 12 persisted records: 4435.5 ms p50 / 5069.6 ms p90 response latency; configuration incomplete | `results/live/turns.jsonl` |
| Device-level barge-in (`abort()` on a real output stream) | **not possible in Colab**; validated on a laptop only | commit `bf90aba` |
| `data/hard_set/` | empty; the ledger requires per-category numbers for a result entry | — |
| ASR v3 re-run | decided **not** to do; recorded with reasons | `docs/EXPERIMENTS.md` |

592 automated tests pass on CPU; 8 more are GPU-gated and will run in Colab.

## What Colab can and cannot do

Be precise about this — an earlier plan wrongly assumed live audio was
impossible here, and wrongly assumed it was all possible.

| Capability | In Colab | How |
| --- | --- | --- |
| Microphone capture | **yes** | `demo.notebook.record()` uses the browser's `MediaRecorder` through the kernel bridge; peak-normalizes the take |
| Playing the agent's reply | **yes** | `IPython.display.Audio` widget |
| A full live turn, end to end, measured | **yes** | `demo.notebook.NotebookAgent.stream_turn()` — real endpointer, real partials |
| Device-level barge-in (`sounddevice` + `abort()`) | **no** | no audio device exists; record it as blocked |
| A reachable server URL | **unreliable** | the `gradio.live` tunnel, the Colab port proxy and Gradio SSR each failed in turn. Use `demo/notebook.py`, which needs no port |
| Long unattended runs | **partly** | the session dies on idle; the driver is resumable and can mirror to Drive |

So the deliverable for live measurement is **a distribution of turns through
`stream_turn()`**, not a single anecdote, plus a written note that device-level
barge-in remains laptop-validated only.

## Environment bring-up

Run in a Colab notebook with **Runtime → T4 GPU**. Check each gate; do not
report numbers from a failing environment.

```python
%cd /content
!rm -rf indic-voice-pipeline
!git clone https://github.com/Vaibhav7711/indic-voice-pipeline.git
%cd /content/indic-voice-pipeline
!pip install -q -e ".[dev,demo,audio]" faster-whisper ctranslate2 bitsandbytes 2>&1 | tail -2
!python scripts/preflight.py
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

| Gate | Expected | If it fails |
| --- | --- | --- |
| `torch.cuda.is_available()` | True | wrong runtime type; fix before anything else |
| GPU | Tesla T4, ~15 GB | a P100/V100/A100 is fine but re-note every latency figure |
| `pytest` | 563 passed, 8 skipped | fix first; a failing suite invalidates everything after it |

Mount Drive so results survive the session, and fetch the adapter (it is not
in git):

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.makedirs('/content/drive/MyDrive/ivp-results', exist_ok=True)

from google.colab import userdata
from huggingface_hub import login, snapshot_download, whoami
login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)
repo = f"{whoami()['name']}/whisper-turbo-hindi-lora-ckpt"
ADAPTER = snapshot_download(repo, allow_patterns=['best/*'],
                            local_dir='/content/v2-final') + '/best'
print(ADAPTER)
```

## The T4 budget — everything fits

Weights plus ~0.4 GiB of KV cache and activations (measured peak was 2.9 GiB
for a 2.7 GiB-weight stack):

| Agent stack | Weights | Peak | On a T4 |
| --- | ---: | ---: | --- |
| turbo + Qwen3-0.6B fp16 + MMS | 2.69 | 3.1 | fits |
| turbo + Qwen3-1.7B fp16 + MMS | 4.74 | 5.1 | fits |
| turbo + **Qwen3-4B fp16** + MMS | 9.02 | 9.4 | **fits — so measure it unquantized** |
| turbo + Qwen3-4B 4-bit + MMS | 3.62 | 4.0 | fits, but pointless here |

On 16 GB there is no memory reason to quantize. Run the bake-off in fp16.
4-bit support exists (`load_llm(quantization="4bit")`, and a `model:4bit`
bake-off spec) for a future smaller deployment target — use it only if you
deliberately want to measure that cost, and say so in the ledger.

`llm.loader.estimate_vram_gib(param_millions, quantization)` produced this
table; recompute rather than trusting it if a candidate changes.

## Plan

### Phase 1 — regression check (first, before anything else)

The decode guards in commit `55f5c1e` (no-speech threshold 0.6, repetition
loop guard) changed serving behaviour and their effect on WER was never
measured. A no-speech suppression on a clip that *does* contain speech deletes
a whole utterance. Everything downstream is meaningless if this is a
regression.

```python
!python scripts/bench_all.py --adapter {ADAPTER} \
    --mirror /content/drive/MyDrive/ivp-results \
    --only guards_on,guards_off,compare_guards
```

Read `results/eval/compare-guards.json` and the `decode_guards_fired` block in
each `metrics.json`.

- **WER unchanged or better, guards fired on 0–2 clips** → keep the defaults,
  record the numbers, continue.
- **WER worse, or no-speech fired on a clip whose reference is non-empty** →
  a regression you must **fix**, not report around. Raise
  `--no-speech-threshold` until it fires only on genuinely empty clips,
  re-run, and record both values in the ledger.

### Phase 2 — the open questions

```python
!python scripts/bench_all.py --adapter {ADAPTER} \
    --mirror /content/drive/MyDrive/ivp-results \
    --llm-models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B \
    --skip guards_on,guards_off,compare_guards
```

~2–2.5 h. The driver is **resumable**: if the session dies, re-run the same
command and finished steps are skipped. Check `results/bench_manifest.json`
for non-zero exit codes and read that step's log before concluding anything
about it.

### Phase 3 — live turns through the notebook

Not a script — do this in the notebook, since it needs the browser mic:

```python
from demo.notebook import NotebookAgent, audio_report, record
agent = NotebookAgent(adapter=ADAPTER, tts='mms',
                      llm='<winner from Phase 2>').load()
for i in range(12):
    clip = record(6)
    audio_report(clip)
    agent.stream_turn(clip)
print(agent.summary())
```

Twelve or more turns, varied lengths, including follow-up questions that test
dialogue memory. Save `agent.turns` to `results/live/turns.jsonl` and report
the **distribution** of `response_latency_ms` and its segments. One turn is an
anecdote; this project has already been misled once by quoting a single
number.

## Pre-registered decision rules

Fixed **now**, before the results exist, so a disappointing number cannot be
reinterpreted into a success. Apply literally.

| Question | Adopt if | Otherwise |
| --- | --- | --- |
| Incremental finals (`early-incr`) | `wer_vs_offline` rises < 1.0 pp versus `early` **and** mean `endpoint_to_final_ms` drops | keep off, record the measured cost |
| Semantic endpointing (`early-sem`) | `clips_split` drops from 16/100 to single digits at < 150 ms mean added `endpoint_to_final_ms` | keep off |
| LLM | highest `devanagari_ratio_mean` with **zero** `think_leaks`, `first_sentence_ms_p50` < 800 ms, and it fits the T4 table | keep Qwen3-0.6B and record why the alternatives failed |
| TTS backend | lower `first_chunk_ms_p50` with acceptable audio in the saved WAVs | keep edge-tts and record the local model's cost |
| CTranslate2 engine | `ct2_matches_explicit` passes (token-identical at fp16, ≤ 5% WER apart at int8) **and** speedup > 1.3× | keep the explicit runner |
| Compiled decode | tokens match **and** speedup > 1.1× | keep it off; the T4 measured 0.9× |

Judgements that need ears — Hindi fluency of an LLM answer, TTS voice quality
— are the one place you must not decide alone. Score what you can
automatically, write the candidates and their outputs into the ledger, and
mark the choice **pending human listening**.

## Recording protocol

A number that exists only in a terminal did not happen.

1. **Evidence** stays where the harness put it: `results/eval/<run>/`,
   `results/llm_bakeoff/`, `results/tts_bakeoff/`, `results/streaming_eval/`,
   `results/gpu_validation*/`, `results/live/turns.jsonl`,
   `results/bench_manifest.json`. Mirror to Drive as you go.
2. **Commit the JSON and Markdown, never the audio or the weights.**
   `.gitignore` already excludes WAVs, MP3s and checkpoints; do not force-add
   them.
3. **Append to `docs/EXPERIMENTS.md`** using the result-entry template at the
   end of that file (line ~438). Include the conditions (GPU, dtype, split,
   seed, limit, normalization level) and the evidence path. Leave a field
   blank rather than filling it from a different run.
4. **Commit per logical result**, conventional-commit style, body explaining
   what was decided and why. End every commit message with:
   `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
5. **Write `results/BLOCKED.md`** for anything you could not do, with the
   reason and what a person would need to provide.

## Hard rules

Each exists because violating it cost real time on this project.

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
   validation; `--seed`/`--limit` must match between compared runs.
7. **`git fetch && git reset --hard origin/main`, not `git pull`** — local
   installs and result files make `pull` refuse, and `-q` hides the refusal.
8. **Restart the runtime after pulling.** Python caches modules; a pulled fix
   that "did not work" was usually a stale import.
9. **Do not delete or rewrite existing evidence.** Superseded results get a
   note saying why, not a deletion.

## Requires a human — stop and write it down

- **Curating `data/hard_set/`.** Clip selection and per-category labelling are
  judgement calls. Do not synthesise a hard set.
- **Final LLM and TTS choice**, insofar as it depends on hearing Hindi
  fluency and voice quality. Narrow to a ranked shortlist with evidence.
- **Publishing anything** (Hub uploads, new remotes, making a repo public).
- **Device-level barge-in** — impossible in Colab. Leave the laptop-validated
  result standing and say so.
- **Anything destructive**: rewriting git history, deleting `results/`,
  force-pushing.

## Definition of done

- Phase 1 concluded: the decode guards are confirmed harmless or fixed, with
  numbers in the ledger.
- Every row marked **open** in Current State has either a recorded measurement
  with a decision applied from the rules above, or an entry in
  `results/BLOCKED.md` explaining why not.
- `docs/EXPERIMENTS.md` has an entry per result, with evidence paths.
- Live turns: a distribution over ≥ 12 turns, not one example.
- `pytest` passes, `ruff check .` is clean, everything committed and pushed.
- `results/SESSION.md` summarising what was decided, what changed, and what a
  next session should pick up — written for a reader who was not here.

## If a local GPU becomes available later

An RTX 4060 (8 GB) was attempted and abandoned on setup trouble. What it would
add, and nothing else: device-level barge-in via `sounddevice`, a reliable
`http://localhost:7860` for `demo/app.py`, and a re-measurement of compiled
decode on Ada (the T4's 0.9× is architecture-specific). On 8 GB, Qwen3-4B
would fit only at 4-bit — see `git show f340814:docs/RTX_WORK.md` for that
budget table.
