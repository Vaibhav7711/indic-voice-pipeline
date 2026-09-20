# indic-voice-pipeline

A correctness-first, single-GPU Indic voice pipeline for Google Colab. Takes
Hindi speech as input, transcribes it, generates an LLM answer, and speaks it
back — owning every stage of the inference loop. (Telugu has a TTS voice and a
system prompt, but no adapter or evaluation yet; treat it as untested.)

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
deletion, and accented/noisy speech. This project fine-tunes Whisper-medium for
Hindi (**40.4% → 25.8% WER** on FLEURS Hindi test, see below) and wires it into
a full voice pipeline on a single T4 GPU.

## Run in Google Colab

Choose **Runtime → Change runtime type → T4 GPU**, then run:

```python
!git clone https://github.com/Vaibhav7711/indic-voice-pipeline.git
%cd indic-voice-pipeline
!bash scripts/setup.sh
!python scripts/preflight.py
```

For the full evaluation workflow — base vs adapter on identical audio, guarded
comparison, error analysis and hard-set bootstrap — open
[`notebooks/eval_colab.ipynb`](notebooks/eval_colab.ipynb) in Colab.

To validate everything built *after* the harness — served runner vs
`generate()`, language detection, streaming session, LLM decoding fixes,
pipeline waterfall, voice turn with real TTS and barge-in — run
[`notebooks/gpu_validation_kaggle.ipynb`](notebooks/gpu_validation_kaggle.ipynb)
on Kaggle (GPU + Internet on). It drives `scripts/gpu_validation.py` and
writes `results/gpu_validation/report.json` with a pass/fail per check.

> `scripts/setup.sh` pins `datasets<4` deliberately. `google/fleurs` is still a
> script-backed dataset and `datasets>=4.0` removed loading-script support, so
> an unpinned install fails on every FLEURS load. `benchmarks/fleurs.py` also
> falls back through `trust_remote_code` and the Hub's parquet revision, and
> raises an actionable error if all three fail.

## Streaming and voice-agent behaviour

Streaming ASR (`asr/streaming/`) and agent output (`agent/`, `tts/streaming.py`)
are documented in [`docs/STREAMING.md`](docs/STREAMING.md), including what is
genuinely streaming versus buffered, and which latency fields are measured
versus approximated.

```bash
python scripts/streaming_demo.py            # simulated turn, no GPU or network
python scripts/streaming_demo.py --barge-in # interruption path
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
    chunking.py       Long-form windowing and transcript stitching
  streaming/
    endpointer.py     Online (frame-synchronous) speech endpointing
    session.py        Stateful streaming session with partial updates
  training/
    lora.py           LoRA fine-tuning; --preset v1 is the shipped recipe
  vad.py              Offline energy VAD (the endpointer's reference)

llm/
    runner.py         Explicit LLM decode runner (prefill/decode split)
    loader.py         Model loader for Indic-capable LLMs
    prompting.py      The one chat-prompt builder (thinking mode off)

pipeline/
    orchestrator.py   ASR → LLM chaining with memory management
    memory.py         VRAM strategy (concurrent vs sequential)

tts/
    synthesis.py      Edge-TTS wrapper (Hindi/Telugu neural voices)
    streaming.py      Sentence splitting and incremental synthesis

benchmarks/
    asr_eval.py       Evaluation harness: WER/CER, error categories, latency
    metrics.py        Levenshtein alignment, WER/CER, corpus aggregation
    error_analysis.py Error categorization and per-example diagnostics
    hard_set.py       Hard-set manifest schema, validator, bootstrap
    compare.py        Guarded run-vs-run comparison
    fleurs.py         FLEURS loading compatibility across datasets versions
    checkpoints.py    Checkpoint discovery, integrity checks, local staging
    asr_wer.py        Legacy single-number WER script (superseded by asr_eval)
    pipeline_e2e.py   Full waterfall benchmark
    streaming_eval.py Streaming session vs offline decode over a seeded subset; VAD grid

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
    test_compare.py   Run-comparison guard tests (CPU only)
    test_checkpoints.py  Checkpoint resolution and torn-write detection
    test_streaming.py Endpointer and streaming session (CPU only)
    test_agent.py     Playback, barge-in, turn latency (CPU only)
    test_prompting.py Shared chat-prompt builder (CPU only)
    test_decoder_prompt.py  Whisper prompt grammar + language detection (CPU)
    test_llm_select.py      Repetition penalty (CPU only)
    test_training_config.py Training presets match the ledger (CPU only)
    test_streaming_eval.py  Streaming benchmark plumbing (CPU only)

notebooks/
    eval_colab.ipynb  End-to-end GPU evaluation workflow for Colab
    gpu_validation_kaggle.ipynb  Stage 3-5 validation sweep on Kaggle GPU

agent/
    playback.py       Playback lifecycle and thread-safe barge-in
    turn.py           Turn orchestration with four-segment latency accounting

demo/
    app.py            Gradio: record audio → transcript → answer → speech

scripts/
    setup.sh          Colab dependency installation
    preflight.py      GPU/import verification
    streaming_demo.py Simulated streaming turn (no GPU)
    streaming_gpu_smoke.py   Real Whisper behind StreamingSession, file replay
    voice_turn_gpu_smoke.py  One real ASR → LLM → TTS turn
    gpu_validation.py        All post-harness checks → results/gpu_validation/

results/
    eval/             Committed asr_eval runs backing docs/EXPERIMENTS.md
    whisper-lora-hi-full/best   Superseded whisper-small adapter (history only)
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
│  Whisper-medium encoder (GPU)    │  24 layers, bidirectional self-attn
│  Timed: CUDA events              │  Output: (1, 1500, 1024)
│  Runs ONCE per audio chunk       │  ~67 ms on T4 (fp16, measured)
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Whisper-medium decoder (GPU)    │  24 layers, causal self-attn +
│  Language detect if not given    │  cross-attn to encoder output
│  Autoregressive, token-by-token  │  Self-attn cache grows per token
│  Dual KV cache:                  │  Cross-attn cache fixed
│    self-attn (grows)             │  ~29 ms prefill, ~20 ms/token on T4
│    cross-attn (fixed)            │  (measured, LoRA merged)
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  LLM (GPU)                       │  Qwen3-0.6B (thinking disabled)
│  Explicit prefill/decode loop    │  Chat-templated prompt from
│  Repetition penalty + loop guard │  transcript + Hindi system prompt
│  Timed: CUDA events              │
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

whisper = load_whisper(
    "openai/whisper-medium",
    adapter_path="Hugme6969/whisper-medium-hindi-lora",  # or a local directory
)
llm = load_llm("Qwen/Qwen3-0.6B")
pipe = VoicePipeline(whisper, llm)

result = pipe.run("hindi_audio.wav", language="hi")  # language=None auto-detects
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

For the Gradio demo, set `WHISPER_ADAPTER_PATH` to the adapter directory
(and optionally `WHISPER_MODEL`, `LLM_MODEL`). For the command-line demo:

```bash
python scripts/demo.py --adapter /content/drive/MyDrive/whisper-training/best
```

**Which adapter is which.** The v1 result below is the **whisper-medium**
adapter published at
[`Hugme6969/whisper-medium-hindi-lora`](https://huggingface.co/Hugme6969/whisper-medium-hindi-lora).
The directory `results/whisper-lora-hi-full/best` committed in this repo is an
older **whisper-small** adapter (39.3% validation WER) kept for history; it
cannot be loaded onto whisper-medium.

### Transcribe recordings longer than 30 seconds

Whisper's encoder is a fixed 30-second window. `transcribe_long_file` uses
25-second windows with a 5-second overlap, then removes only verified repeated
tokens at each boundary. It returns both the merged transcript and per-window
timing/locations so the behaviour can be measured and debugged.

```python
from asr.explicit import ASRRunner, load_whisper

loaded = load_whisper(
    "openai/whisper-medium",
    adapter_path="Hugme6969/whisper-medium-hindi-lora",
)
runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
result = runner.transcribe_long_file("meeting.wav", language="hi")

print(result.text)
print(result.metrics.chunk_count, result.metrics.real_time_factor)
for chunk in result.chunks:
    print(chunk.index, chunk.start_seconds, chunk.end_seconds)
```

This is deliberately fixed-window chunking, not VAD: a later stage will place
boundaries at actual speech/silence transitions and support streaming partials.

### Speech endpointing (VAD baseline)

`asr.vad.detect_speech()` now provides a dependency-free energy VAD. It exposes
its actual controls—RMS dBFS threshold, minimum speech, endpoint silence, and
padding—so its segment decisions can be logged and compared with a neural VAD.
This is the segmentation layer that will feed long-form ASR rather than sending
silence and arbitrary cut points to Whisper.

## Key design decisions

### Why own the Whisper loop?

`pipeline("asr")` and `model.generate()` hide the encoder/decoder split,
the dual KV cache, and the per-stage latency. Owning the loop exposes:

- Encoder time vs decoder time (where is the bottleneck?)
- Per-token decode latency (is it memory-bound or compute-bound?)
- Cross-attention cache size (50.6 MB fixed for Whisper-small)
- Real-time factor broken down by stage

### Why LoRA, not QLoRA?

Whisper-medium is 769M params = 1.5 GiB in FP16. LoRA rank-16 on q/k/v/out
adds ~9.4M trainable params. Training memory with batch 2 × 4 accumulation and
gradient checkpointing stays well inside a T4's 16 GiB. There is no memory
problem to solve; QLoRA would add NF4 dequantization overhead to solve a
nonexistent constraint.

### Why edge-tts instead of a GPU TTS model?

On T4, VRAM is split between Whisper (~1.5 GiB) and the LLM (~1.2 GiB). A
GPU-based TTS model (VITS, StyleTTS2: 80-150 MB) would compete for the same
memory. edge-tts provides high-quality Hindi/Telugu neural voices at zero
GPU cost. The interface is designed so swapping in a local TTS model is a
single module change.

### Memory strategy

Whisper-medium (1.5 GiB) + Qwen3-0.6B (1.2 GiB) fit concurrently on T4
(~2.7 GiB total, well under 16 GiB). For larger Indic LLMs, the pipeline
auto-detects and switches to sequential mode: offload Whisper to CPU after
transcription, load LLM, pay the swap cost. The strategy is chosen by
measurement at construction time.

## Evaluation

```bash
# Final test number (GPU). Seeded random sample, never a prefix slice.
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter Hugme6969/whisper-medium-hindi-lora \
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

All numbers below are from `benchmarks/asr_eval.py` on the same seeded random
300-example subset of FLEURS Hindi test (`--seed 0`, `standard` normalization,
Tesla T4, fp16, LoRA merged, explicit `ASRRunner`). Evidence:
`results/eval/medium-base-test-300-seed0/`, `results/eval/medium-lora-test-300-seed0/`.

### ASR quality: Whisper-medium base vs Hindi LoRA v1

| Metric | Base | LoRA v1 | Change |
| --- | ---: | ---: | ---: |
| WER (standard) | 40.43% | **25.82%** | −14.60 pp (−36.1% rel.) |
| CER (standard) | 16.74% | **9.61%** | −7.13 pp |
| WER (raw, no normalization) | 43.06% | 26.62% | |
| WER (orthography-blind) | 39.09% | 24.26% | |
| Truncation errors | 27 | 0 | |
| Hallucination-run errors | 16 | 0 | |

### ASR latency per utterance (mean over 300, LoRA v1)

| Stage | Base | LoRA v1 |
| --- | ---: | ---: |
| Mel extraction | 7.6 ms | 9.4 ms |
| Encoder forward | 66.0 ms | 66.9 ms |
| Decoder prefill | 28.7 ms | 29.0 ms |
| Mean decode / token | 19.9 ms | 20.4 ms |
| Decoder steps | 122.6 | 124.2 |
| Total ASR p50 / p90 | 2461 / 3730 ms | 2540 / 3851 ms |
| Mean RTF | 0.230 | 0.238 |

Decode dominates: ~124 tokens × ~20 ms ≈ 2.5 s of a 2.6 s utterance. The
merged adapter costs ~3% latency.

### Full pipeline waterfall (whisper-medium + LoRA v1 → Qwen3-0.6B)

From `scripts/gpu_validation.py` (`pipeline_waterfall`, mean of 3 runs after
warm-up, one ~10 s FLEURS clip, 48-token LLM budget, T4, commit `ee2b7c3`).
Evidence: `results/gpu_validation/report.json`.

| Stage | Time |
| --- | ---: |
| Mel extraction | 7.7 ms |
| Whisper encoder | 74.8 ms |
| ASR decode | 1711 ms |
| LLM prefill | 100 ms |
| LLM decode (~47 tokens @ ~41 ms) | 1922 ms |
| **Total pipeline** | **3885 ms** |
| Audio → first LLM token | 1938 ms |
| Peak VRAM | 2.9 GiB (concurrent strategy) |

ASR decode and LLM decode are each ~45% of the turn; both are per-token
sequential cost, which is why the agent path streams sentences to TTS rather
than waiting for the whole answer.

### Voice turn (final transcript → audio playing)

_To be re-measured._ The first sweep reported 291 ms, but that accounting
stamped "first token" after the whole response had been generated and so
omitted ~1.9 s of LLM decode; the honest figure for that turn was ~2.2 s. The
turn now streams the LLM and starts TTS after the first sentence, and the
next sweep will report `response_latency_ms` measured end to end. The
endpointer's `min_silence_ms` is added on top in a live session; see
`docs/STREAMING.md` §5.

### Streaming vs offline (100 seeded FLEURS-hi test clips)

| VAD | Streamed WER | Offline WER | Penalty |
| --- | ---: | ---: | ---: |
| fixed −40 dBFS (before) | 34.60% | 27.26% | +7.34 pp, 3 clips lost |
| adaptive (now) | **26.77%** | 27.26% | −0.49 pp |

`benchmarks/streaming_eval.py`; details and the VAD grid in
`docs/EXPERIMENTS.md`.

## Correctness policy

| Component | Required comparison |
| --- | --- |
| Explicit ASR decode | HF `model.generate()` greedy tokens |
| LLM decode | HF `model.generate()` greedy tokens |
| Pipeline end-to-end | Standalone ASR + standalone LLM outputs |

## License

MIT
