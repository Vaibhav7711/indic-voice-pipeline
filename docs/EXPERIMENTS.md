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
Evidence: `results/eval/turbo-base-test-300-seed0/` — **never committed, and
absent from this repository.** The comparison below therefore rests on numbers
that cannot be checked. It informed a decision that has already been taken and
is retained for that reason; re-run it before citing it again.

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
weights are not obtainable or reproducible from this checkout. Evidence:
`results/eval/v2-guards-on/` (seed 0, 300 clips, `standard`, Tesla T4, fp16,
commit `31a842f`, `git_dirty: true`, adapter `/content/v2-final/best`).

The directory this entry previously cited,
`results/eval/turbo-lora-v2-test-300-seed0-full/`, was never committed and does
not exist in this repository. It also reported a lower CER and latency than the
run that *is* committed (8.43% / 694 ms / RTF 0.066 against 8.46% / 766 ms /
0.073). The sensitivity ladder and every error-category share in this entry
match `v2-guards-on` exactly, so the three latency-and-CER figures were
transcribed from a run that cannot be checked. **The committed numbers are the
ones stated below**, per this ledger's own rule that every number come from the
harness with its `run_config.json` retained beside it. If the uncommitted run
is real it has to be committed to be cited.

| Model | WER | CER | p50 | RTF |
| --- | ---: | ---: | ---: | ---: |
| medium base | 40.43% | 16.74% | 2461 ms | 0.230 |
| medium + v1 | 25.82% | 9.61% | 2540 ms | 0.238 |
| turbo base † | 30.40% | 11.55% | 1238 ms | 0.117 |
| **turbo + v2** | **23.83%** | **8.46%** | **766 ms** | **0.073** |

† No committed evidence: `results/eval/turbo-base-test-300-seed0/` does not
exist in this repository. Every row whose evidence *is* committed matches it to
the decimal; this row and the previously published v2 latency are the two that
do not, and both come from Colab runs that were never persisted.

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
| Test CER | ≤ 8.5% | 8.46% | pass (by 0.04 pt) |
| p50 ASR latency | ≤ 1.35 s | 766 ms | pass |
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

## Live conversation, dtype-clean sweep, and the sampling arms (2026-09-26, run 2)

Colab T4, Qwen3-1.7B, engine patched. Evidence: `results/live/{followup,bargein}.jsonl`
with their manifests, `results/latency_ab/`, `results/answer_quality/{greedy,greedy+penalty,sampled,sampled+penalty}.json`,
`results/engine_parity/parity.json`.

### Barge-in works. First time it has been exercised at all.

Second utterance scripted to start 1.2 s after the first ends
(`bargein.manifest.json`: 0.50–4.34 s, then 5.54–8.40 s), replayed through
`--sink paced`. Turn 1:

| signal | value |
| --- | --- |
| `state` | `interrupted` |
| `barge_in` | `true` |
| `llm_stopped_by_barge_in` | `true` |
| `playback.state` | `cancelled` after 3 chunks / 19 200 bytes |
| recorded response | truncated mid-word: *…अपनी अपेक्षाकृत बड* |

All three signals agree, playback was cancelled rather than drained, and the
history holds what was spoken rather than the whole response. Turn 2 then
completed normally. This is the property `docs/AGENT_BRIEF.md` filed under
"requires a human"; what it actually required was a sink that occupies
wall-clock time.

### Follow-up turns: the plumbing works, the memory test is inconclusive

Two turns, 6 s apart, both recorded. Turn 1 answered correctly
(*भारत की राजधानी नई दिल्ली है।*). Turn 2 asked *वहाँ की आबादी कितनी है?* and
answered *वहाँ की आबाधी 100 फीट है।* — a population given in feet.

It did not ask "where?", which is weak evidence the antecedent survived, but
the answer contains no reference to Delhi or India, so **this run does not
establish that dialogue memory reached the prompt**. The model's answer is too
poor to reveal it either way. Recorded as inconclusive rather than as a pass.
ASR also returned *आबाधी* for *आबादी* and dropped *रुको* from the barge-in
utterance.

### The dtype confound was real and large

| | bf16 baseline (run 1) | fp16 baseline (run 2) |
| --- | ---: | ---: |
| baseline p50 transcript→first audio | 2411.7 ms | **1789.0 ms** |
| baseline first token | 772.4 ms | **131.1 ms** |
| served p50 | 848.6 ms | **653.7 ms** |
| served first token | 42.0 ms | **36.7 ms** |
| ratio | 2.84× | **2.7365×** |

Fixing `pick_dtype` cut the baseline's prefill by **5.9×**, so the engine's
prefill advantage falls from 18× to 3.6×. The headline ratio barely moved,
because both arms got faster. **2.7365× is the defensible figure**: same
checkpoint, same dtype, interleaved arms.

| arm | p50 | p90 | first token | to first unit | mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (explicit) | 1789.0 | 2831.8 | 131.1 | 1604.9 | 51.9 |
| **served (engine)** | **653.7** | **777.8** | **36.7** | **449.4** | 31.6 |
| history200 | 1472.2 | 2295.2 | 76.3 | 1340.0 | 41.1 |
| units30 | 1510.5 | 1661.1 | 115.7 | 1296.7 | 51.9 |

*Caveat recorded:* the served arm generated 31.6 tokens against the baseline's
51.9, so the two arms produced different amounts of text — consistent with
parity being 4/5, not 5/5. The primary metric is time to *first* audio, which a
shorter total response does not shorten, so the ratio stands; but the arms are
not producing identical content and the token counts say so.

**`history200` is now worth 317 ms, not the ~1.2 s predicted**, because the
prefill it trims is 131 ms rather than 772 ms. Most of that prediction was the
bf16 penalty. `units30` is worth 278 ms on the primary metric and 308 ms on
time-to-first-unit.

### Sampling versus greedy: prediction refuted, all four arms identical

| arm | factual | obeyed | declined | loops | too long | Devanagari |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | 40% | 80% | 0% | 0 | 0 | 98.3% |
| greedy+penalty | 40% | 80% | 0% | 0 | 0 | 98.3% |
| sampled | 40% | 80% | 0% | 0 | 0 | 98.3% |
| sampled+penalty | 40% | 80% | 0% | 0 | 0 | 98.3% |

Sampling did apply — 9 of 18 answers differ between `greedy` and `sampled`, and
both runs record their parameters. It changed the wording and **not one score**.

**Two predictions refuted, both stated in advance:**

1. *"`looping` goes to zero on both sampled arms and on greedy+penalty."* It was
   already zero on greedy, across all 18 cases. The `एक बर्फ के` ×13 loop seen
   earlier was **prompt-specific, not a general property of greedy decoding**.
   The same story prompt here produced 105 clean characters. The hypothesis that
   greedy caused that loop is not supported.
2. *"`instruction_obeyed` improves most on sampled+penalty."* Nothing moved.

**Decision, by the pre-registered rule: keep greedy.** No arm satisfies the
rule's `devanagari_ratio_mean ≥ 0.99` proviso — including the incumbent — so the
rule yields no adoption; and on the tie-break (equal scores, prefer the lower
temperature) greedy also wins. The decision is the same either way, which is
the only reason the failed proviso does not need adjudicating.

*A measurement error found in the proviso, and what it did not change.* The
threshold was set from a bake-off that measured different prompts. Here one
**correct** answer (*पानी के रासायनिक सूत्र आमतौर पर H₂O होता है।*) contains
Latin characters, so script purity was scored down for being right.
`Case.allows_latin` now excludes such cases; that lifts the figure from 0.9826
to 0.9851, still under 0.99. The remaining shortfall is genuine: the model emits
mixed-script tokens such as *बेंगalore*, which a TTS front end will mispronounce.
The fix changes no decision, and is recorded here rather than applied quietly.

### The real problem is the model's knowledge, not the decoder

Instruction-following is **80%** with zero over-long answers, so the model obeys
the prompt. Factual accuracy is **4 of 10**, and the failures are visible:

| asked | answered |
| --- | --- |
| capital of France | *लिस्टन* — Lisbon |
| capital of Japan | *तोकियोसहा* — a garbled Tokyo |
| where is the Taj Mahal | *बेंगalore* — wrong city, and mid-word script mixing |
| two plus two | *दोही* |
| India's national animal | *बिंदौर एवं देश के अधिकांश जिलों में…* |
| which direction the sun rises | *सूरज आकाश में उगता है* — "in the sky" |

And **0 of 3** unanswerable cases were declined. Asked the time with no clock it
answered *3:45 बजे*; asked the weather with no weather data, *खुशी से बराबर है*.
Inventing a plausible answer is worse than refusing, because it sounds like an
answer.

Decoding cannot fix any of this, and this run is the evidence. Two levers remain
untested, in cost order: the system prompt never tells the model to admit
ignorance, which is what `declined_rate` measures and is a one-line change; and
a larger checkpoint, which is a VRAM question for the parity gate rather than
for serving.

### Parity: 4/5, up from 3/5, still failing

`corrupted_prompts: 0`, `served_replacement_chars: 0`, both sides fp16. Agreement
4/5, identical 3/5, mean shared prefix 66.4% (from 62.6%). The gate still does
not pass, and the logit-margin measurement that would settle whether the
remaining divergence is a near-tie is still unbuilt.

## Pre-registered: sampling versus greedy, and answer quality (registered 2026-09-26)

Registered before the run. Harness: `benchmarks/answer_quality.py`, 18 cases
(10 factual with checkable answers, 5 instruction-following, 3 unanswerable).
Evidence to `results/answer_quality/`.

**Why this exists.** The model was selected on `devanagari_ratio` and
`think_leaks`. Four answers from the measured run all score **1.000 Devanagari**
while being: correct, a thirteen-fold repetition loop, the opposite of what was
asked, and an invented fact. The selection metric cannot see any of it, so
"1.000" has never meant the answers were good — only that the script was.

**The hypothesis.** The bad answers are largely a decoding artefact, not model
capacity. This pipeline decodes hard greedy (`argmax`), because every
correctness gate here compares two greedy decoders. Qwen's guidance for these
models is against greedy decoding precisely because it repeats, and the
observed loop is that failure exactly.

**Arms**, same checkpoint, same prompts, `seed=0` for every sampled arm:

| arm | settings |
| --- | --- |
| `greedy` | temperature 0 — today's serving path |
| `greedy+penalty` | temperature 0, presence_penalty 0.5 |
| `sampled` | temperature 0.7, top_p 0.8, top_k 20 |
| `sampled+penalty` | temperature 0.7, top_p 0.8, top_k 20, presence_penalty 0.5 |

- *Decision rule:* adopt the arm with the highest `instruction_obeyed` **and**
  zero `looping`, provided its `factual_accuracy` is not below `greedy`'s by
  more than 10 percentage points and its `devanagari_ratio_mean` stays ≥ 0.99.
  Ties go to the lower `temperature`, because a sampled serving path is
  reproducible only with its seed recorded and greedy needs no such caveat.
- *Greedy remains the reference regardless of the outcome.* Every gate —
  explicit against `generate()`, static cache against eager, CTranslate2
  against explicit, served against explicit — decodes greedily on both sides. A
  sampled serving path does not change that, and this rule does not license
  changing it.
- *Prediction:* `looping` goes to zero on both sampled arms and on
  `greedy+penalty`; `instruction_obeyed` improves most on `sampled+penalty`;
  `factual_accuracy` drops slightly under sampling. If sampling does **not**
  fix the loop, the cause is not the decoder and the next suspect is the
  system prompt.
- *What this cannot settle:* whether Qwen3-1.7B is good enough. It measures one
  checkpoint under four decoders. A model comparison needs the same harness run
  across checkpoints, which is the bake-off that should have been run with this
  metric in the first place.
- *Threats to validity, recorded now:* the factual set is small (10 items) and
  general-knowledge; `declined_rate` rewards refusing, so a model that refuses
  everything scores 3/3 there while failing every factual case — read the
  splits together, never `declined_rate` alone.

## Served-engine sweep, first measured run (2026-09-26)

Colab T4, Qwen3-1.7B, `full-inference-engine` patched for the UTF-8 streaming
defect, `llm.engines.server_app` with a 512 × 16 pool. Evidence:
`results/engine_parity/parity.json`, `results/latency_ab/{turns.jsonl,summary.json}`.

### The UTF-8 corruption is fixed

`corrupted_prompts: 0`, `served_replacement_chars: 0` across 5 parity prompts,
and **0 U+FFFD across all 32 sweep turns**. Before the patch, 4 of 4 Hindi
prompts were corrupted. `docs/ENGINE_BUG_UTF8_STREAMING.md` holds the report;
the patch is applied by `scripts/patch_engine_utf8.py` and belongs upstream.

### Latency: the engine is faster, by less than this run says

| arm | p50 transcript → first audio | p90 | first token p50 | to first unit p50 | mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` explicit 1.7B | 2411.7 ms | 3360.2 | 772.4 | 1670.0 | 52.0 |
| `served` engine 1.7B | **848.6 ms** | 924.8 | **42.0** | 547.2 | 50.6 |
| `history200` | 1849.2 ms | 3114.9 | 511.7 | 1341.1 | 41.1 |
| `units30` | 2394.3 ms | 2540.1 | 780.0 | 1326.4 | 52.0 |

Eight interleaved rounds per arm, 86.5 s total, MMS TTS, no errors. Generated
token counts are comparable across arms, so the gap is not a shorter-answer
artefact.

**The 2.84× is confounded and overstates the engine.** The explicit arm loaded
**bfloat16** and the server served **float16**, and this is a T4: bf16 has no
tensor cores before Ampere, so the baseline carried an emulation penalty on
top of whatever the engine saves. `llm.loader.pick_dtype` caused it — its
docstring says exactly this ("T4 (sm_75) has no bf16 tensor cores … a slow
path") while the code gated on `torch.cuda.is_bf16_supported()`, which counts
emulated support and answers True on a T4. Now gated on compute capability
≥ 8.0, and `scripts/latency_ab.py --dtype` sets the explicit arm explicitly and
records it.

**Not decided.** The pre-registered rule requires the parity gate to pass
first, and it did not. The ratio is also unusable until the arms share a dtype.
Both are re-runs, not re-interpretations.

The prefill figure is the one to watch: 772.4 → 42.0 ms is 18×, far more than
a dtype penalty plausibly explains on its own, so a real prefill win is likely
to survive the re-run. That is a prediction, not a result.

### Parity: 3/5, with matched fp16

| prompt | verdict |
| --- | --- |
| नमस्ते, आज मौसम कैसा है? | identical (17 chars) |
| भारत की राजधानी क्या है? | identical (29 chars) |
| मुझे एक छोटी कहानी सुनाओ। | agreed on 26 chars, then the served side loops |
| What is the capital of India? | diverged at char 13 |
| थोड़ा धीरे बोलो please… | diverged at char **2** |

Mean shared prefix 17.4 characters, 62.6% of the reference response. Dtypes
matched (both `torch.float16`), so this is not the dtype confound — it is
genuine numerical divergence between two greedy decoders.

Divergence at char 2 means the **first token** already differed, so prefill
differs. That is expected to some degree: the engine fuses chunked prefill with
decode, and a different accumulation order in fp16 flips an argmax wherever the
top two candidates are close. What is not yet known is whether these were close
calls. **The deciding measurement is the reference's top-2 logit margin at each
divergence point** — a small margin means numerical noise and the engine is
sound; a large one means it is computing something else. Not built, needs a GPU.

### Two findings the gate surfaced incidentally

**Serving dropped the repetition guard.** `LLMRunner` stops degenerate greedy
output with an n-gram check and reports `stopped_on_repetition`. Over HTTP the
server owns the decode, so nothing downstream could stop it: the served arm
emitted `एक बर्फ के` thirteen times and ran to its 128-token cap where the
explicit arm answered the same prompt coherently. `HttpLLMEngine` now carries a
character-level guard (48-character window, period searched, three repeats
required) which closes the stream — so the engine cancels the request and the
loop is neither spoken nor paid for. It is weaker than a token-level guard and
no substitute for one in the engine.

**Qwen3-1.7B's Hindi answers are poor, and the bake-off never measured that.**
"What is the capital of India?" produced *चीनी राजधानी है।* ("it is the Chinese
capital") from the reference and *चीनी राजधानी बेंगलुरु है।* from the server;
"नमस्ते, आज मौसम कैसा है?" produced *आज मौसम बराबर है।* ("the weather is equal
today"). The bake-off scored `devanagari_ratio` and `think_leaks`, so **1.000
Devanagari never meant correct answers** — it meant the script was right. No
harness in this project measures answer correctness, and that gap now has
evidence rather than being hypothetical.

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
