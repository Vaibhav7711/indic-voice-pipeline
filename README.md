# indic-voice-pipeline

A correctness-first, single-GPU Indic voice pipeline for Google Colab. Takes
Hindi/Telugu speech as input, transcribes it, generates an LLM answer, and
speaks it back — owning every stage of the inference loop.

```text
Audio → Mel → Whisper Encoder → Whisper Decoder (autoregressive) → LLM → TTS
```

The explicit Whisper encoder/decoder loop, the LLM runner, the pipeline
orchestration, the memory management, the LoRA fine-tuning, and every
benchmark are built here. `model.generate()` and `pipeline("asr")` are used
only as correctness references in tests, never in the execution path.

## Why this exists

Indian speech has specific challenges that off-the-shelf ASR doesn't handle
well: Hindi-English code-switching, retroflex consonants (ट vs त), schwa
deletion, and accented/noisy speech. This project fine-tunes Whisper for
Hindi and wires it into a full voice pipeline on a single T4 GPU.

## Run in Google Colab

Choose **Runtime → Change runtime type → T4 GPU**, then run:

```python
!git clone https://github.com/Vaibhav7711/indic-voice-pipeline.git
%cd indic-voice-pipeline
!bash scripts/setup.sh
!python scripts/preflight.py
```

## Repository structure

```text
asr/
  explicit/
    loader.py         Whisper model loader
    mel.py            Timed mel-spectrogram extraction
    encoder.py        Explicit encoder forward with CUDA timing
    decoder.py        Autoregressive decoder loop with dual KV cache
    runner.py         Full ASR runner: mel → encoder → decoder
  training/
    lora.py           LoRA fine-tuning on FLEURS/CommonVoice Hindi

llm/
    runner.py         Explicit LLM decode runner (prefill/decode split)
    loader.py         Model loader for Indic-capable LLMs

pipeline/
    orchestrator.py   ASR → LLM chaining with memory management
    memory.py         VRAM strategy (concurrent vs sequential)

tts/
    synthesis.py      Edge-TTS wrapper (Hindi/Telugu neural voices)

benchmarks/
    asr_eval.py       Evaluation harness: WER/CER, error categories, latency
    metrics.py        Levenshtein alignment, WER/CER, corpus aggregation
    error_analysis.py Error categorization and per-example diagnostics
    hard_set.py       Hard-set manifest schema, validator, bootstrap
    asr_latency.py    Per-stage ASR timing with multi-run statistics
    asr_wer.py        Legacy single-number WER script (superseded by asr_eval)
    pipeline_e2e.py   Full waterfall benchmark
    memory_profile.py VRAM timeline across pipeline stages

text/
    normalize.py      Hindi/Hinglish normalization ladder
    lexicon.py        Number-word lexicon for error categorization

data/
    hard_set/         Curated adversarial evaluation set + curation protocol

tests/
    test_asr.py       Explicit loop vs model.generate() token match
    test_pipeline.py  Integration tests and memory bounds
    test_llm.py       LLM runner correctness
    test_normalize.py Normalization behaviour (CPU only)
    test_metrics.py   Metric correctness vs independent implementation
    test_error_analysis.py  Error categorization (CPU only)
    test_asr_eval.py  Harness and hard-set tests (CPU only)

demo/
    app.py            Gradio: record audio → transcript → answer → speech

scripts/
    setup.sh          Colab dependency installation
    preflight.py      GPU/import verification
```

## Architecture

```text
Hindi/English audio (16 kHz WAV)
        │
        ▼
┌──────────────────────────────────┐
│  Mel extraction (CPU)            │  librosa → 80-channel log-mel
│  Timed: perf_counter_ns          │  Output: (1, 80, 3000)
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Whisper encoder (GPU)           │  12 layers, bidirectional self-attn
│  Timed: CUDA events              │  Output: (1, 1500, 768)
│  Runs ONCE per audio chunk       │  ~100-200 ms on T4
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Whisper decoder (GPU)           │  12 layers, causal self-attn +
│  Autoregressive, token-by-token  │  cross-attn to encoder output
│  Dual KV cache:                  │  Self-attn cache grows per token
│    self-attn (grows)             │  Cross-attn cache fixed (50.6 MB)
│    cross-attn (fixed)            │
│  Timed: per-step CUDA events     │  ~5-10 ms/token on T4
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  LLM (GPU)                       │  Qwen3-0.6B (or Indic model)
│  Explicit prefill/decode loop    │  Chat-templated prompt from
│  Timed: CUDA events              │  transcript + Indic system prompt
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  TTS (CPU/network)               │  edge-tts: Hindi/Telugu neural
│  Zero GPU cost                   │  voices, ~200-500 ms
└──────────────────────────────────┘
```

## Quick demo

```python
from asr.explicit import load_whisper, ASRRunner
from llm import load_llm, LLMRunner
from pipeline import VoicePipeline

whisper = load_whisper("openai/whisper-small")
llm = load_llm("Qwen/Qwen3-0.6B")
pipe = VoicePipeline(whisper, llm)

result = pipe.run("hindi_audio.wav", language="hi")
print(result.transcript)        # Hindi transcription
print(result.answer)            # LLM response
print(result.metrics.as_dict()) # Full latency waterfall
```

### Serve a fine-tuned LoRA adapter

The explicit ASR runner can merge a PEFT LoRA adapter before inference. This
means the production path still owns Whisper's encoder/decoder loop and KV
cache; it does not fall back to `model.generate()`.

```python
whisper = load_whisper(
    "openai/whisper-medium",
    adapter_path="/content/drive/MyDrive/whisper-training/checkpoint-600",
)
```

For the Gradio demo, set `WHISPER_ADAPTER_PATH` to the adapter directory. For
the command-line demo:

```bash
python scripts/demo.py --whisper-model openai/whisper-medium \
  --adapter /content/drive/MyDrive/whisper-training/checkpoint-600
```

## Key design decisions

### Why own the Whisper loop?

`pipeline("asr")` and `model.generate()` hide the encoder/decoder split,
the dual KV cache, and the per-stage latency. Owning the loop exposes:

- Encoder time vs decoder time (where is the bottleneck?)
- Per-token decode latency (is it memory-bound or compute-bound?)
- Cross-attention cache size (50.6 MB fixed for Whisper-small)
- Real-time factor broken down by stage

### Why LoRA, not QLoRA?

Whisper-small is 244M params = 488 MB in FP16. LoRA rank-16 adds ~3.5M
trainable params. Total training memory with batch 4 + gradient
checkpointing: ~6-7 GiB. T4 has 16 GiB. There is no memory problem to
solve. QLoRA would add NF4 dequantization overhead to solve a nonexistent
constraint.

### Why edge-tts instead of a GPU TTS model?

On T4, VRAM is split between Whisper (~488 MB) and the LLM (~1.2 GiB). A
GPU-based TTS model (VITS, StyleTTS2: 80-150 MB) would compete for the same
memory. edge-tts provides high-quality Hindi/Telugu neural voices at zero
GPU cost. The interface is designed so swapping in a local TTS model is a
single module change.

### Memory strategy

Whisper-small (488 MB) + Qwen3-0.6B (1.2 GiB) fit concurrently on T4
(~1.7 GiB total, well under 16 GiB). For larger Indic LLMs, the pipeline
auto-detects and switches to sequential mode: offload Whisper to CPU after
transcription, load LLM, pay the swap cost. The strategy is chosen by
measurement at construction time.

## Evaluation

```bash
# Final test number (GPU). Seeded random sample, never a prefix slice.
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter results/whisper-lora-hi-full/best \
    --split test --limit 300 --seed 0 \
    --out-dir results/eval/medium-lora-test

# Re-score saved predictions with different settings (no GPU, milliseconds).
python -m benchmarks.asr_eval score \
    --predictions results/eval/medium-lora-test/predictions.jsonl \
    --level aggressive --out-dir results/eval/medium-lora-test-aggressive
```

Each run writes `run_config.json` (git SHA, versions, the exact sampled
indices), `predictions.jsonl`, `metrics.json` and `errors.jsonl`. WER is
reported with CER, at three normalization levels simultaneously, and broken
down by error category — so normalization can never be used to quietly improve
a number. See `docs/EXPERIMENTS.md` for the full protocol.

The scoring layer has no torch/transformers dependency: `pytest tests/` runs on
CPU, and saved predictions can be re-scored anywhere.

## Measured Colab T4 results

_To be filled after running benchmarks. The numbers below predate
`benchmarks/asr_eval.py` and are not comparable to it — different base model,
biased 50-example prefix sample, no normalization policy, no CER._

### ASR baseline (Whisper-small, no fine-tuning)

| Metric | Result |
| --- | ---: |
| Mel extraction | ___ ms |
| Encoder forward | ___ ms |
| Decoder prefill | ___ ms |
| Mean decode/token | ___ ms |
| Total ASR | ___ ms |
| RTF | ___ |
| WER (FLEURS Hindi) | ___% |

### ASR fine-tuned (LoRA rank-16, FLEURS Hindi)

| Metric | Result |
| --- | ---: |
| WER (before) | ___% |
| WER (after) | ___% |
| Improvement | ___ pp |

### Full pipeline waterfall

| Stage | Time |
| --- | ---: |
| Mel extraction | ___ ms |
| Whisper encoder | ___ ms |
| ASR decode (___ tokens) | ___ ms |
| LLM prefill | ___ ms |
| LLM decode (___ tokens) | ___ ms |
| **Total pipeline** | **___ ms** |
| Audio → first LLM token | ___ ms |
| Peak VRAM | ___ GiB |

## Correctness policy

| Component | Required comparison |
| --- | --- |
| Explicit ASR decode | HF `model.generate()` greedy tokens |
| LLM decode | HF `model.generate()` greedy tokens |
| Pipeline end-to-end | Standalone ASR + standalone LLM outputs |

## License

MIT
