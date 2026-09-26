# ASR experiment ledger

This file is the project-level index for every training run. Keep the full
machine-readable configuration and metrics next to each checkpoint; summarize
only decisions and comparable results here.

## Evaluation protocol

Implemented by `benchmarks/asr_eval.py`. Every number in this ledger must come
from that harness, with the emitted `run_config.json` retained next to it.

- **Model selection:** FLEURS Hindi validation WER. The test split is never
  used to decide whether to keep training.
- **Final reporting:** FLEURS Hindi test WER *and* CER, plus per-category hard
  set results and the error-category breakdown. A single aggregate WER is not
  an acceptable result entry.
- **Sampling:** seeded random selection, never `[:N]`. FLEURS is ordered, so a
  prefix slice shares speakers and topics and is not an estimate of test
  performance. The chosen indices are written to `run_config.json`, so a
  subset is exactly reproducible. Two models must be compared on the same
  `--seed` and `--limit`.
- **Text policy:** the reported metric is computed at the `standard`
  normalization level, and the harness *always* also reports `none` and
  `aggressive` alongside it. Normalization can therefore never be used to
  quietly improve a number — the cost of formatting and of orthographic
  variation is printed on every run.
- **Latency:** measured on the merged LoRA model through `ASRRunner`, not
  through Hugging Face `generate()`, so quality and runtime results describe
  the same serving implementation. Reported as p50/p90/p95 and mean RTF.
- **Provenance:** git SHA, dirty flag, package versions, device and full argv
  are captured in `run_config.json` and `metrics.json`.

### Metric definitions

- **WER** = `(S + D + I) / N_ref` over word tokens. **CER** = the same over
  characters, spaces included. CER is reported because a single wrong matra
  costs one character but a whole word of WER — the gap between the two
  separates "misheard the word" from "nearly right".
- **Micro is the headline.** Corpus WER is `sum(errors) / sum(reference
  tokens)`, not the mean of per-utterance WERs. The macro average is reported
  too; it is dominated by short utterances where one error is 100%. Quote the
  micro figure and note the macro when they diverge.
- Utterances with an empty reference have an undefined rate and are counted
  separately rather than scored as 0 or 1.

### Normalization ladder

| Level | Folds away | Use |
| --- | --- | --- |
| `none` | nothing | Shows the literal string distance |
| `basic` | case, whitespace, punctuation incl. danda, Devanagari digits | Formatting-blind |
| `standard` | + one canonical nukta encoding | **Primary reported metric** |
| `aggressive` | + nukta, homorganic nasal conjuncts, chandrabindu | Diagnostic only |

`standard` removes Unicode *representation* differences without erasing any
linguistic distinction (ज vs ज़ stay distinct). The `standard → aggressive` gap
is the share of WER that is transcription convention rather than acoustic
modelling; report it, never headline it.

Number words are deliberately **not** rewritten to digits. Choosing a direction
would silently decide which spelling is correct and move WER without the model
changing. Numeric mismatches are surfaced as an error category instead.

### Error categories

Each edit operation gets exactly one category (`benchmarks/error_analysis.py`),
structural runs first:

| Category | Meaning | Typical fix |
| --- | --- | --- |
| `truncation` | Deletion run ending at the reference end | Early EOS, `max_new_tokens`, chunking |
| `deletion_run` | Deletion run mid-utterance | Dropped segment, VAD/endpointing |
| `hallucination` | Insertion run | Decode loop, repetition penalty |
| `orthographic` | Vanishes under `aggressive` | Not a modelling error — normalization policy |
| `numeric` | Digits or spelled numerals | Output-format policy, targeted data |
| `code_switch` | Latin-script or cross-script mismatch | Hinglish training data |
| `rare_word` | Reference token is a corpus hapax | Named entities, domain vocabulary |
| `function_word` | Tokens ≤ 2 characters | Usually audio quality, not vocabulary |
| `other` | Everything else | — |

Per-example flags also record `empty_hypothesis`, `repetition_loop`,
`truncated`, and hypothesis/reference `length_ratio`.

### Reproducing a number

```bash
# Final test number for the fine-tuned adapter (Hub id or local directory).
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter Hugme6969/whisper-medium-hindi-lora \
    --split test --limit 300 --seed 0 \
    --out-dir results/eval/medium-lora-test

# Same audio, base model, for a like-for-like comparison.
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --split test --limit 300 --seed 0 \
    --out-dir results/eval/medium-base-test

# Re-score saved predictions with different settings — no GPU, milliseconds.
python -m benchmarks.asr_eval score \
    --predictions results/eval/medium-lora-test/predictions.jsonl \
    --level aggressive --out-dir results/eval/medium-lora-test-aggressive
```

`run` and `score` are split on purpose: inference over FLEURS test is minutes
of GPU time, and deciding how to report should not cost another run.

### Hard set

`data/hard_set/` holds a fixed, manually curated adversarial set reported **per
category** (noisy, accented, named_entity, numeric, code_switch, spontaneous,
far_field, telephony, fast_speech, long_form). Rules and the curation workflow
are in `data/hard_set/README.md`. Items marked `candidate` are excluded from
reporting until a human verifies the transcript and assigns categories.

A change that improves FLEURS test WER but regresses a hard-set category is not
automatically an improvement. Say so explicitly in the decision line.

### Superseded results

`results/wer_baseline.json` and `results/wer_finetuned.json` (75.80% and 37.66%)
are from an earlier **whisper-small** run scored on the first 50 test examples
with no normalization, no CER, and no saved predictions. They are kept for
history but are **not comparable** to anything produced by `asr_eval.py`: the
base model differs, the sample is a biased prefix, and the text policy differs.
Do not cite them alongside new numbers.

`results/whisper-lora-hi-full/best` is that whisper-small adapter (rank 16,
FLEURS only, `--preset small-fleurs`; validation WER 39.25%, eval loss 0.364).
It is **not** the v1 adapter and cannot be loaded onto whisper-medium. Its
intermediate `checkpoint-*` directories were removed from the repository on
2026-09-20 (they held ~330 MB of optimizer state); they remain in git history
before that date.

## Active run: Whisper-medium Hindi LoRA

| Field | Value |
| --- | --- |
| Base model | `openai/whisper-medium` |
| Data | FLEURS Hindi train + first 5,000 streamed IndicVoices Hindi examples |
| Validation | FLEURS Hindi validation (239 examples) |
| LoRA | rank 16, alpha 32, dropout 0.05; q/k/v/out projections |
| Effective batch size | 8 (2 × 4 gradient accumulation) |
| Target | 3 epochs / 2,670 optimizer steps |
| Checkpoint policy | every 200 steps; retain latest 3 in Drive |
| Checkpoint location | Drive `whisper-training/` (read-only to this repo) |
| Final adapter | [`Hugme6969/whisper-medium-hindi-lora`](https://huggingface.co/Hugme6969/whisper-medium-hindi-lora) |
| Status | **completed**: 3 epochs / 2,670 optimizer steps |
| Recipe | `python asr/training/lora.py --preset v1 --output-dir <drive>/whisper-training` |

**Provenance caveat.** The v1 run was launched from a Colab-side copy of the
trainer that was never committed; the table above was recorded by hand at the
time. `--preset v1` in `asr/training/lora.py` is a faithful reconstruction of
that configuration (same model, data mix, LoRA targets, batch geometry and
checkpoint policy) and every future run writes `train_config.json` beside its
checkpoints, so this caveat applies to v1 only. A re-run from the preset should
be recorded as a separate entry (`v1-repro`) and compared to v1 on validation
WER; it will not be bit-identical (streamed IndicVoices decode order, Colab
library versions).

### Validation history

| Step | FLEURS Hindi validation WER | Note |
| --- | --- | --- |
| 600 | 32.70% | |
| 1400 | 28.00% | interrupted for GPU availability; later resumed |
| 2000 | 26.23% | resumed checkpoint |
| 2670 / epoch 3 | **25.7437%** | final validation WER; eval loss 0.23547 |

These are *validation* figures used for model selection. The final test result
below is kept separate and was not used to choose a checkpoint.

### Final v1 held-out evaluation

The frozen v1 adapter and the base model were evaluated through the explicit
`ASRRunner` on the same seeded random **300-example** subset of the 418-example
FLEURS Hindi test split (`seed=0`, standard normalization, Tesla T4). Complete
machine-readable evidence is committed under `results/eval/`; the matching
`run_config.json` files contain the literal selected indices, package versions,
GPU, CLI arguments and git provenance.

| Model | WER | CER | WER change | Mean RTF |
| --- | ---: | ---: | ---: | ---: |
| `openai/whisper-medium` base | 40.4270% | 16.7354% | — | 0.2298 |
| Hindi LoRA v1 | **25.8246%** | **9.6055%** | **−14.6024 pp** (36.12% relative reduction) | 0.2382 |

The adapter improves WER and CER substantially with a small serving cost:
p50 latency rises from 2460.554 ms to 2539.962 ms (+79.408 ms), and p90 from
3729.869 ms to 3850.711 ms (+120.842 ms). This is an evaluation of the fixed
300-example subset, not a claim about all Hindi speech or a benchmark tuned on
the test split.

| Normalization sensitivity | Base WER | LoRA WER |
| --- | ---: | ---: |
| raw | 43.0582% | 26.6216% |
| standard (reported) | 40.4270% | 25.8246% |
| orthography-blind diagnostic | 39.0858% | 24.2644% |

For the LoRA run, the largest remaining categories are `other` (1,061 errors,
56.23%), `rare_word` (377, 19.98%), and `function_word` (185, 9.80%). It has no
truncation, repetition-loop, or hallucination-run flags on this sample; the
base run had 27 truncation and 16 hallucination errors, plus one repetition and
one truncation flag. These observations motivate later data/decoding work, not
training on the held-out test examples.

Evidence paths:

- `results/eval/medium-base-test-300-seed0/`
- `results/eval/medium-lora-test-300-seed0/`
- `results/eval/compare-medium-base-vs-lora.json`

**Never hardcode a checkpoint step.** With `save_total_limit=3`, only the most
recent checkpoints survive; `load_best_model_at_end=True` additionally protects
whichever checkpoint is best by validation WER. So the set on disk changes as
training advances and `checkpoint-1400` may already be gone. Resolve by policy:

```bash
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter-dir /content/drive/MyDrive/whisper-training \
    --adapter-policy best \
    --stage-adapter /content/adapters \
    --split validation --limit 300 --seed 0 \
    --out-dir results/eval/medium-best-validation
```

`--stage-adapter` copies the adapter off the Drive mount and verifies the
safetensors header against the real file size, which catches a checkpoint read
while the Trainer is mid-save instead of failing later inside safetensors.

Drive checkpoints are read-only: nothing in this repo writes to, deletes, or
assumes access to `whisper-training/`.

### Known v1 limitation: language detection

The v1 labels were tokenized with the processor's default prefix, which is
`<|sot|><|notimestamps|>` — no `<|hi|>`, no `<|transcribe|>`. Transcription
with a forced language prompt is unaffected (that is how every number above
was produced), but the merged model's distribution at the position after
`<|sot|>` no longer favours `<|hi|>`: unrestricted language detection on
FLEURS Hindi clips returns `ca` with p≈0.65–0.73, while the same procedure on
the base model returns `hi`. Evidence: `results/gpu_validation/report.json`,
checks `language_detection_base_model` and `language_detection_adapter`.

Consequences:

- `language=None` with the v1 adapter must restrict candidates
  (`ASRRunner(..., language_candidates=["hi", "en", "te"])`; the demo does).
- `asr/training/lora.py` now calls `set_prefix_tokens(language, task)` so
  labels match the serving prompt. The next training run should verify
  unrestricted detection returns `hi` before it is promoted.

### Base-model decision for v2 (2026-09-21)

`openai/whisper-large-v3-turbo`, no adapter, on the same seed-0 300-clip
subset as the v1 evaluation (Tesla T4, fp16, `standard` normalization).
Evidence: `results/eval/turbo-base-test-300-seed0/` (run from Colab at
commit `e0daacb`; to be committed with the next evidence drop).

| Model | WER | CER | p50 latency | RTF |
| --- | ---: | ---: | ---: | ---: |
| whisper-medium base | 40.43% | 16.74% | 2461 ms | 0.230 |
| whisper-medium + LoRA v1 | 25.82% | 9.61% | 2540 ms | 0.238 |
| **large-v3-turbo base** | 30.40% | 11.55% | **1238 ms** | **0.117** |

Turbo base: raw 34.56% / orthography-blind 28.58% (formatting cost 4.2 pts —
punctuation and Latin digits the references do not use; fine-tuning on FLEURS
references removes this). Categories: other 57.5%, rare_word 19.1%,
function_word 9.7%, orthographic 5.7%, **deletion_run 2.8% (63)**, code_switch
2.8%, numeric 2.3%; one repetition-loop flag.

**Decision: v2 trains on large-v3-turbo** (`--preset v2-turbo`). Base turbo
is 10 points better than base medium at half the latency, and the v1 recipe's
gain on medium (−14.6 pts) transferring even at a conservative 25% relative
puts v2 at ~22–23%, beating v1 while halving the largest term in the agent's
response delay. Full large-v3 was rejected: same encoder, 32 decoder layers,
~4× turbo's decode cost — the wrong trade for a voice agent.

Everything else is held at v1 (data mix, rank 16, 2×4, 200-step checkpoints)
so the comparison isolates the base model. The label prefix now includes
`<|hi|><|transcribe|>`; that is a correctness fix (see "Known v1
limitation"), not a tuning change.

**Success criteria, fixed before training:** test WER ≤ 22.0% and CER ≤ 8.5%
on the seed-0 300 subset; unrestricted language detection returns `hi` in the
GPU sweep; p50 ASR latency ≤ 1.35 s; `deletion_run` not above base turbo's 63.
Rank (32) and data-mix (10k IndicVoices) ablations follow only after v2 is
recorded against these.

## v2: Whisper-large-v3-turbo Hindi LoRA (2026-09-22)

`--preset v2-turbo`: v1's recipe with the base model swapped (decision and
rationale above). The measured adapter was at the ephemeral path
`/kaggle/working/v2-final/best` and its checkpoint repo was private, so the
weights are not obtainable or reproducible from this checkout. Evidence from
that run:
`results/eval/turbo-lora-v2-test-300-seed0-full/` (seed 0, 300 clips,
`standard`, Tesla T4, fp16, commit `55f5c1e`).

| Model | WER | CER | p50 | RTF |
| --- | ---: | ---: | ---: | ---: |
| medium base | 40.43% | 16.74% | 2461 ms | 0.230 |
| medium + v1 | 25.82% | 9.61% | 2540 ms | 0.238 |
| turbo base | 30.40% | 11.55% | 1238 ms | 0.117 |
| **turbo + v2** | **23.83%** | **8.43%** | **694 ms** | **0.066** |

Sensitivity: raw 24.83% / orthography-blind 22.54% (formatting 1.0 pt,
orthography 1.29 pts — both back in line with v1 after fine-tuning, versus
4.2 pts for base turbo). Categories: other 55.6%, rare_word 21.4%,
function_word 10.7%, orthographic 5.3%, numeric 3.2%, truncation 1.6% (27),
deletion_run 1.2% (21), code_switch 0.5%, hallucination 0.5%.

### The 225-token bug, and why the first v2 number was wrong

The first v2 evaluation reported 24.37% / 9.18% with 95 `deletion_run`
errors. Those hypotheses all ended mid-word, mid-UTF-8 character: Devanagari
costs ~6 BPE tokens per word, and `max_new_tokens=225` (hardcoded in the
runner, the eval harness, the streaming config and the pipeline) cut every
utterance over ~37 words. The affected references needed 263–399 tokens.
`flags.truncated` did not catch it because that category requires the
deletion run to reach the very last reference token, which a mangled final
word prevents.

Fixed in `55f5c1e`: the budget is derived from the model
(`max_target_positions - prompt`, so 444), `metrics.hit_token_budget` and
`metrics.json`'s `truncated_by_token_budget` make a cut hypothesis loud, and
the harness prints a warning. **Any number produced before that commit on
Hindi is an overstatement**; re-run rather than cite it.

The same 225 was also truncating *training labels*, teaching the model to
stop early on long utterances — the likely source of v2's remaining 27
truncation and 21 deletion-run errors. Labels and `generation_max_length`
now use 448, so v3 is the first run without it.

### Decision

**v2 is the measured candidate, not a reproducible shipped artifact.** It was
better on WER, CER and every
latency measure (ASR decode 1712 ms → 380 ms in the pipeline waterfall), and
it passes unrestricted language detection, which v1 cannot. It misses its
WER target of 22.0% by 1.8 points, so it is *accepted but not final*:

| Criterion | Target | v2 | |
| --- | --- | --- | --- |
| Test WER | ≤ 22.0% | 23.83% | miss |
| Test CER | ≤ 8.5% | 8.43% | pass |
| p50 ASR latency | ≤ 1.35 s | 694 ms | pass |
| `deletion_run` | ≤ 63 | 21 | pass |
| Unrestricted language detection | `hi` | `hi` | pass |

### v2 trained on half the optimizer steps v1 did

Data was as intended: `train_config.json` reports 2,120 FLEURS-hi train +
5,000 IndicVoices, 5 dropped by the 30-second filter, so 7,115 examples. At
the preset's effective batch of 8 that is 2,668 steps for 3 epochs. The run
did **1,335** — exactly half, because Kaggle's "T4 x2" makes two GPUs
visible and HF Trainer wraps the model in DataParallel, multiplying
`per_device_train_batch_size` by the device count: effective batch 16, not 8,
at the same learning rate and warmup.

So v2 is not v1's recipe on a new base model; it is that recipe with double
the batch and half the steps. That is the most likely reason its LoRA gain
over its base was 6.6 points where v1's was 14.6. Validation WER was still
falling when the best checkpoint was taken (28.91 → 27.58 → 26.32 → 25.33 at
steps 200/400/600/800), which is consistent with under-training rather than
overfitting.

`asr/training/lora.py` now computes the batch geometry at startup, records it
in `train_config.json` under `runtime`, and prints a warning with the fix
(`CUDA_VISIBLE_DEVICES=0`) when more than one GPU is visible. Recipe drift of
this kind should not need arithmetic on `trainer_state.json` to notice.

A v3 re-run with the preset's geometry restored (single GPU, plus the
label-truncation fix) is the obvious next experiment and would likely close
much of the 1.8-point gap, since v2 appears simply under-trained.

**Decision (2026-09-22): not run. v2 is the final adapter for this project.**
It is better than v1 on every quality and latency measure and is good enough
for the agent; further ASR training is not where the remaining value is. The
WER target of 22.0% is recorded as **missed**, not retroactively lowered, and
the reason (half the optimizer steps) is documented above so the number is
interpretable. Anyone resuming this work should start with that v3 re-run.

## Live turn latency: where the 4.4 s goes (2026-09-24)

Twelve live turns through `NotebookAgent.stream_turn()` on a Colab T4
(whisper-large-v3-turbo + LoRA v2, Qwen3-0.6B, edge-tts). Evidence:
`results/live/turns.jsonl`.

| Metric | Value |
| --- | ---: |
| Response latency p50 | 4435.5 ms |
| Response latency p90 | 5069.6 ms |
| Response latency mean | 4129.5 ms |
| Endpoint → final transcript (mean) | 660.8 ms |
| ASR after the endpoint | ~0 ms (candidate reused on every turn) |
| Transcript → first LLM token (mean) | 877.9 ms |
| First token → first audio (mean) | 2590.7 ms |
| ⤷ LLM until a unit was speakable | 2445.7 ms |
| ⤷ synthesis of that unit | 145.0 ms |

ASR is not the bottleneck and TTS is not the bottleneck. **94% of the
controllable latency is the LLM**, in two separable parts.

### Prefill grows with dialogue history — it is not a fixed cost

`first_token_ms` rises monotonically across the session: 209, 275, 376, 486,
704, 844, 1024, 1145, 1292, 1259, 1447, 1474 ms — Pearson r = **0.99**
against turn index, a 7× increase. Reporting the 877.9 ms mean as a constant
hides this; the cost is the conversation prompt getting longer, since
`Conversation` keeps up to 6 exchanges (800 tokens).

Holding prefill at its turn-1 value would put mean latency at **3461 ms**
instead of 4129 ms. Options, cheapest first: a smaller history budget; or
KV-cache reuse across turns, since the system prompt and the older exchanges
are a stable prefix and only the tail changes. Neither is measured yet.

### The first speakable unit costs ~61 ms per character

The first spoken unit averaged 41.2 characters and took 2445.7 ms to
produce — about 61 ms per character at ~42 ms/token. Three of twelve turns
ran to the 96-token cap, so the "answer in one or two short sentences"
instruction is not reliably obeyed by Qwen3-0.6B.

`max_unit_chars` is 60. At the measured rate, cutting it to 30 would put the
first unit out in roughly **1841 ms** instead of 2446 ms — about 600 ms — at
the cost of a breath in a slightly odder place. Not measured; worth an A/B.

### What this does not support

These records were persisted without their configuration (no LLM name, TTS
backend, GPU or commit), so they support a latency distribution and **not**
a comparison between model candidates. `demo/notebook.py` now records that
block per turn, so the next run does not have the same gap.

### Streaming evaluation (2026-09-21)

`benchmarks/streaming_eval.py`, Hindi LoRA v1, 100 seeded FLEURS-hi test clips
(`seed=0`, mean 10.7 s), each streamed as 0.5 s silence + clip + 1 s silence
in 100 ms blocks, `standard` normalization, Tesla T4, commit `21ebce7`.
Evidence: `results/streaming_eval/medium-lora-test-100/`.

Offline WER on the same 100 clips: **27.26%**.

| VAD config | Streamed WER vs ref | Penalty vs offline | WER vs offline decode | Split clips | Empty | Onset halluc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed40` (pre-fix: fixed −40 dBFS) | 34.60% | **+7.34 pp** | 21.05% | 25 | 3 | 0 |
| adaptive, pad 200, floor −60 | 27.04% | −0.22 pp | 9.90% | 16 | 0 | 1 |
| adaptive, pad 300, floor −60 | 27.04% | −0.22 pp | 9.81% | 16 | 0 | 0 |
| adaptive, pad 200, floor −70 | 26.77% | −0.49 pp | 9.85% | 14 | 0 | 1 |

Reading it:

- The fixed threshold cost **7.3 points** and lost three clips entirely; the
  adaptive threshold plus the onset-trim fix (`asr/streaming/session.py`,
  same day) bring streaming to offline quality. This is the number that
  matters: streaming is no longer a quality regression.
- "WER vs offline decode" ≈ 10% is not a loss: on the 84 single-final clips
  it is 6.8% and nets to zero against the reference (different boundaries →
  different greedy path). On the 16 split clips it is 23%: a split costs
  context. Splits are the remaining streaming cost, and they are a
  turn-taking problem (the agent would answer mid-sentence) more than a WER
  one.
- The floor and padding deltas are within noise on 100 clips (0.27 pp ≈ six
  words). Both were adopted as defaults anyway because they are cheap and
  mechanistically justified (pre-roll for soft onsets; floor irrelevant
  outside near-silent recordings); confirm on 300 clips before citing them.

**Decision:** `VADConfig` defaults are now adaptive, `padding_ms=300`,
`threshold_floor_dbfs=-70`, `min_silence_ms=600`. This is the energy-VAD
baseline a neural VAD (Silero) must beat on the same benchmark in Stage 5.

### T4 decision sweep (2026-09-24)

Imported from the Colab evidence archive. The adapter remained the unavailable
local path `/content/v2-final/best`; these measurements support serving
decisions but do not make the v2 artifact reproducible. The guard, LLM, TTS,
and streaming harnesses record commit `31a842f` with `git_dirty: true`; their
LLM/TTS/streaming summaries also omit the GPU name. Evidence is committed
under `results/eval/v2-guards-{on,off}/`, `results/llm_bakeoff/`,
`results/tts_bakeoff/`, `results/streaming_eval/`, and
`results/gpu_validation-ct2/`.

**Decode guards.** Two like-for-like 300-clip FLEURS-hi test evaluations
(`seed=0`, fp16, standard normalization) compared the defaults (no-speech
threshold 0.6; repetition guard 3-gram × 4) with both guards disabled. WER
was **23.8265%** in each run and CER **8.4585%** in each run. The enabled run
recorded zero no-speech suppressions and zero repetition stops; p50 latency
was 765.675 ms enabled versus 764.222 ms disabled. **Decision: keep the
defaults.** WER is unchanged and guards fired on 0 clips.

**LLM bake-off.** Twelve Hindi prompts, bf16:

| Model | Devanagari ratio | Think leaks | First sentence p50 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1.000 | 0 | 1744 ms | 1.17 GiB |
| Qwen3-1.7B | 1.000 | 0 | 2534 ms | 3.27 GiB |
| Qwen3-4B | 0.988 | 0 | 3553 ms | 7.61 GiB |

No candidate meets the fixed <800 ms first-sentence threshold. **Decision:
keep Qwen3-0.6B**, the prescribed fallback and fastest candidate; final Hindi
answer quality remains pending human review of the saved outputs.

**TTS bake-off and human listening.** A rerun at commit `a708857` (dirty
worktree) measured MMS first-chunk p50 at **124.6 ms** versus edge at
**607.8 ms**, and mean RTF 0.0404 versus 0.2070, over the same five Hindi
sentences. Evidence:
`results/tts_bakeoff/t4-rerun-2026-09-24/summary.json`. A human listener
reviewed the regenerated clips and reported a significant MMS quality issue.
**Decision: retain edge-tts; reject MMS despite its latency advantage.** This
is the required human-quality condition in the pre-registered rule, not an
inference from latency metrics.

**Streaming decisions.** On 100 seeded FLEURS-hi test clips, `early` had
12.3831% WER versus offline and 14 split clips at 652.8 ms mean
endpoint-to-final. `early-incr` raised WER versus offline to 35.7684% with no
endpoint-time reduction. `early-sem` reduced splits only to 13 and added 10.9
ms mean endpoint-to-final time. **Decision: keep incremental finals and
semantic endpointing off.** The first fails the <1.0 pp WER-cost rule; the
second fails the single-digit-splits rule.

**CTranslate2 and compiled decode.** The T4 int8-float16 CTranslate2 check
passed with 1.5873% mean WER versus the explicit runner (within the 5%
tolerance) and **1.3145×** speedup. **Decision: adopt CTranslate2 for the
int8 serving tier**, with its stated WER tolerance. The compiled LLM path was
token-identical but **0.8526×** eager speed, so **keep compiled decode off**
under the >1.1× rule.

**Live turns.** `results/live/turns.jsonl` contains 12 persisted streaming
turn records. Response latency is **4435.5 ms p50**, **5069.6 ms p90**, and
**4129.5 ms mean**. Mean ASR time is 342.2 ms; first-token 877.9 ms;
first-token-to-first-unit 2446.0 ms; TTS synthesis 144.7 ms; and
endpoint-to-final 660.8 ms. The main delay before first audio is generation
until a speakable unit, not TTS synthesis. The live records do not record the
selected LLM, TTS backend, GPU, or notebook commit, so this is a real
distribution but not a configuration-comparison result.

## Pre-registered: the serving-engine and turn-latency sweep (registered 2026-09-25)

Registered **before** the runs, so the rules cannot be chosen after seeing the
numbers. Harnesses: `scripts/engine_parity.py` (correctness gate),
`scripts/latency_ab.py` (interleaved arms on one set of loaded weights).
Evidence to `results/engine_parity/` and `results/latency_ab/`.

Every arm here is configuration, not weights. The 12 live turns in
`results/live/turns.jsonl` are the baseline the predictions come from, and
those records omit their configuration — so the sweep's own baseline arm, not
those 12 turns, is the comparison. The predictions below are stated in advance
precisely so that being wrong is visible.

**Superseded 2026-09-26: the model change is to Qwen3-1.7B, not 4B.** The 4B
entry below stands as written — it was registered before the run and its
prediction was not tested, so it is superseded rather than deleted. What
refuted it was not latency but VRAM: `scripts/engine_parity.py` decodes the
same weights a second time in its own process, since a reference decode has to
happen locally, and two 7.5 GiB copies do not fit a 15 GiB T4. A Colab run died
at 87% of the second load. That cost was absent from the VRAM arithmetic in the
4B entry, which counted the server's copy and Whisper only.

Qwen3-1.7B is the revised default, and it is better on the evidence already
collected rather than merely smaller: the T4 bake-off measured Devanagari ratio
**1.000 for 1.7B against 0.988 for 4B**, at first-sentence p50 2534 ms against
3553 ms. So 1.7B keeps the script purity 4B gives up, at 1.45× the 0.6B latency
instead of 2×.

- *Decision rule, revised:* the sweep's baseline and served arms now use the
  **same checkpoint**, so the comparison isolates the engine. Adopt the served
  engine if parity passes and p50 transcript → first audio improves by ≥ 1.1×.
  The 0.6B-versus-1.7B question is not re-opened here: the bake-off answered
  it, and mixing a model change into an engine A/B would make neither
  attributable.
- *What this gives up:* the 4B quality question is now unmeasured rather than
  measured-and-rejected. Recorded as such. Testing it needs either a second GPU
  or a two-phase gate that generates the reference outputs before the server
  starts; neither is built.
- *Prediction:* unchanged for the engine — above 1.1×. The model change is not
  predicted, because it is not being measured.

**Model change: Qwen3-0.6B → Qwen3-4B (registered 2026-09-25, superseded).** The default
LLM is now Qwen3-4B, served by the engine. The reasons and the counter-evidence
are both recorded here, before the run, because they point in opposite
directions.

*For:* answer quality, and a parity gate that is informative. A greedy decoder
diverges at the first step where two implementations rank the top two
candidates differently, so token agreement tracks per-step confidence. A 0.6B
model has flatter logits and smaller top-two margins, so an fp16 rounding
difference flips a tie readily and two correct implementations part company
early. Low agreement at 0.6B is weak evidence of an engine defect. A larger
model makes the gate mean something.

*Against, from this project's own measurements:* the T4 bake-off measured
first-sentence p50 at **1744 ms for 0.6B and 3553 ms for 4B** on the explicit
runner, and Devanagari ratio 1.000 versus 0.988. So on the metric this whole
sweep exists to reduce, 4B starts about 2× worse, and it was very slightly
worse on script purity too.

*What has to be true for the change to pay:* the engine must recover more than
2× on 4B relative to the explicit runner at 0.6B. On a single card it has to
do that with CUDA-graphed decode and chunked prefill alone — speculative
decoding, the mechanism that would most plausibly cover 4B's decode cost,
requires two GPUs (`create_app` refuses a shared target/draft device) and the
engine's own record notes the 0.6B drafter did not beat target-only on a T4.

- *Decision rule:* keep Qwen3-4B as the default if the served-4B arm's p50
  transcript → first audio is **at or below** the 0.6B baseline arm's. If 4B
  is slower, it is retained only on an explicit quality judgement made by a
  human listening to both, recorded as such — not on latency, which will have
  said the opposite. If neither holds, revert to 0.6B and record the measured
  cost of 4B.
- *Required arms:* the sweep must therefore run **both** checkpoints, not just
  the new default. An A/B that changes the model and the engine at once cannot
  attribute the difference to either.
- *Prediction:* 4B on the engine lands between the two explicit figures —
  faster than 3553 ms, slower than 1744 ms. If so, the rule reverts to 0.6B
  and the 4B decision becomes a quality question with a known latency price.
- *VRAM, recorded in advance:* ~7.5 GiB weights + 1.125 GiB for a 512 × 16
  pool at 144 KiB/token + 1.5 GiB Whisper in the other process + ~0.6 GiB of
  CUDA contexts ≈ 10.7 GiB before graphs and activations. Fits a 15 GB T4.
  Does **not** fit an 8 GB card in fp16, so the RTX 4060 path stays on 0.6B.

**Served LLM engine (`full-inference-engine`).** The candidate is
an OpenAI-compatible server with a paged KV cache, continuous batching and
CUDA-graphed decode, whose own T4 record is TTFT p50 ~0.3–0.4 s and ITL p50
19.8 ms at ~656-token prompts. The explicit runner measured 877.9 ms mean to
first token and ~42 ms/token.

- *Gate, before any latency is recorded:* `scripts/engine_parity.py` must exit
  0 — every prompt agreeing on the first 24 characters. Both decoders are
  greedy, so this is a real constraint and not a formality. A failure is
  reported and investigated; it is not "close enough".
- *Decision rule:* adopt the served engine as the LLM tier if parity passes
  **and** p50 **committed transcript → first audio** improves by ≥ 1.1×
  against the baseline arm. Below 1.1×, keep the explicit runner — the same
  threshold that rejected compiled decode at 0.8526× and accepted
  CTranslate2 at 1.3145×.
- *Metric correction, made before the first run and recorded rather than
  quietly applied:* this rule first named `response_latency_ms`, which the
  harness cannot produce. That metric is speech-end → agent-speaks, and it is
  `None` unless the endpoint-to-final segment exists; these arms are driven by
  fixed text with no speech in them, so it would have been `None` for every
  arm and the rule unfalsifiable. Supplying a plausible figure for a segment
  that did not happen would have made every headline number partly invented.
  Transcript → first audio is the sum of the two segments the arms actually
  change. The endpoint-to-final floor (660.8 ms mean, measured) sits under any
  perceived-latency figure and is unaffected by all three arms.
- *Prediction:* p50 improves by more than 1.1×. If it does not, the likely
  cause is that batch-1 single-stream serving does not benefit from continuous
  batching, which is a throughput optimization; the win would then have to
  come from CUDA-graphed decode and chunked prefill alone.
- *Caveat recorded in advance:* two processes on one card cost two CUDA
  contexts (~300 MiB each) and lose the shared allocator. If VRAM forces a
  smaller Whisper or a smaller KV pool, that trade is part of the result.

**Dialogue history budget (`max_history_tokens` 800 → 200).** `first_token_ms`
rose 209 → 1474 ms across 12 turns, Pearson r = 0.99 against turn index.

- *Decision rule:* adopt 200 if transcript-to-first-token p50 improves by
  ≥ 200 ms **and** the assistant still resolves a referring expression across
  turns — asked a follow-up that depends on the previous turn ("उसका मतलब
  क्या है?"), it must not answer as if the exchange had not happened. Latency
  bought by forgetting the conversation is not a win for a dialogue agent, and
  no latency threshold can substitute for checking that.
- *Prediction:* ~1.2 s at the tail of a 12-turn session, less early on. The
  gain is not uniform: at turn 1 there is no history to trim, so an A/B over
  few rounds will understate it. Rounds are therefore ≥ 8.

**First speakable unit (`max_unit_chars` 60 → 30).** The measured first unit
was 41.2 characters at ~61 ms each.

- *Decision rule:* adopt 30 if first-token-to-first-unit p50 improves by
  ≥ 300 ms **and** a human listening pass on ≥ 10 turns does not report the
  clause break as unnatural. This is the same human-quality condition that
  retained `edge` TTS; a shorter unit buys silence-to-speech by cutting the
  sentence in a slightly odder place, and only listening settles whether that
  is acceptable.
- *Prediction:* ~600 ms, from 2446 ms to ~1841 ms.

**What this sweep cannot settle.** Endpoint-to-final (660.8 ms mean) is the
VAD's silence window, not compute, and no engine changes it. It is the floor
any turn-latency figure sits on. KV-cache reuse across turns — the system
prompt and older exchanges are a stable prefix — is the other named
unmeasured option, and is not in this sweep.

### Measuring latency while training runs

Don't. RTF, p50 and p90 measured on a GPU that is simultaneously training
describe contention, not the serving path, and `--warmup` does not help — it
removes cold-start cost, not a competing process. Either use a separate
runtime, or record WER/CER only and pass
`--note "GPU shared with training run - latency invalid"` so the caveat lands
in `run_config.json` rather than being forgotten.

### Required comparisons after training

| Experiment | Primary question | Selection metric |
| --- | --- | --- |
| Whisper-medium baseline | What does model scale buy before adaptation? | validation WER |
| FLEURS-only LoRA | Is IndicVoices data helping this domain? | validation WER + hard set |
| FLEURS + IndicVoices LoRA | Active data-mixture baseline | validation WER + hard set |
| q/v-only vs q/k/v/out LoRA | Does adapting keys/output projections justify added capacity? | WER, adapter size, latency |
| Augmentation ablation | Does robustness improve without clean-speech regression? | WER by condition |

## Result entry template

Copy this block for each completed run.

```markdown
### <run name>
- Commit: `<git SHA>`   Harness output: `results/eval/<dir>/`
- Data and filtering:
- Training configuration:
- Best checkpoint / epoch:
- Selection: split / limit / seed (must match the comparison run)
- Validation WER / CER (standard):
- Test WER / CER (standard):
- Normalization sensitivity: raw / standard / aggressive WER
- Error categories: top three by share of errors
- Flags: truncated / repetition_loop / empty_hypothesis counts
- Hard-set WER by category (with n per category):
- Merged-adapter ASR latency: p50 / p90 ms, mean RTF
- Failure modes observed (cite example ids from `errors.jsonl`):
- Decision: keep / reject, and why:
```

Leave any field blank rather than filling it from a different run. A partially
completed entry is evidence; a plausible-looking one is not.
