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
rationale above). Adapter: `/kaggle/working/v2-final/best`, checkpoints in a
private Hub repo. Evidence:
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

**v2 replaces v1 as the served adapter.** Better on WER, CER and every
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

**v3 therefore repeats v2 with the preset's geometry restored** (single GPU,
or `--grad-accum 2` on two) and the label-truncation fix, before any
hyperparameter or data change is considered. Same base model, same data, same
rank: if v2 was simply under-trained, that alone should close much of the
1.8-point gap to the WER target.

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
