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
# Final test number for the fine-tuned adapter.
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter results/whisper-lora-hi-full/best \
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
