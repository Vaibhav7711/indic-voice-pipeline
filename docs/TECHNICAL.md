# indic-voice-pipeline — Technical Documentation

## Table of contents

1. Project overview
2. Voice model theory (for LLM engineers)
3. Architecture and data flow
4. File-by-file code walkthrough
5. Dependency map
6. KV cache memory math
7. LoRA fine-tuning details
8. Pipeline memory management
9. Measured results and analysis
10. Known limitations and future work

---

## 1. Project overview

This project is a single-GPU Indic voice pipeline that takes Hindi speech as
input, transcribes it, generates an LLM answer, and speaks it back. Every
stage of the inference loop is explicitly owned — `model.generate()` and
`pipeline("asr")` are never used in the execution path.

The project exists as a portfolio piece for an ML inference engineering role
at Sarvam AI, a company building Indian-language AI (speech, translation,
multilingual LLMs). It complements a separate project (full-inference-engine)
that demonstrates LLM runtime engineering. This project adds the speech
dimension.

Hardware target: Google Colab with a Tesla T4 GPU (14.6 GiB VRAM).

### What the project proves

- Understanding of encoder-decoder ASR architectures (not just decoder-only LLMs)
- Ability to own the inference loop for Whisper (explicit encoder/decoder with
  dual KV cache management)
- LoRA fine-tuning for Indic languages (Hindi WER: 75.8% → 37.7%)
- Multi-model pipeline engineering on constrained hardware
- Per-stage CUDA timing and latency breakdown
- Honest measurement and documentation of limitations

### What the project does NOT claim

- No custom model architectures — Whisper and Qwen are used as-is
- No custom attention kernels — HuggingFace provides the transformer layers
- No production-grade serving — this is a reference pipeline, not a server
- The LLM (Qwen3-0.6B) produces poor Hindi answers — documented, not hidden

---

## 2. Voice model theory (for LLM engineers)

### From audio to "tokens"

In LLMs, text goes through a tokenizer that maps subwords to integer IDs.
Speech has an analogous process, but the path is different:

**Raw audio**: A 1D array of air pressure samples. At 16 kHz (Whisper's
expected rate), one second = 16,000 floating-point numbers. A 30-second clip
= 480,000 floats.

**STFT (Short-Time Fourier Transform)**: Slide a 25 ms window across the
audio with a 10 ms hop. At each position, compute the FFT to decompose the
signal into frequency components. This converts the 1D waveform into a 2D
matrix: (frequency_bins × time_frames).

**Mel filterbank**: Human hearing is logarithmic — the difference between
100 Hz and 200 Hz sounds large, but 8000 Hz and 8100 Hz sounds negligible.
The mel scale applies 80 triangular filters that are narrow at low
frequencies and wide at high frequencies, then takes the log of each band's
energy. Output: (80 × 3000) matrix — the **log-mel spectrogram**.

This is Whisper's input. The 80 channels are like an 80-dimensional
"embedding" at each of 3000 time positions. Unlike LLM token embeddings
(looked up from a table), these are computed from the physics of the sound.

```
LLM:     text  → tokenizer → (seq_len,)  integer IDs
Whisper: audio → mel-spec   → (80, 3000)  continuous features
```

The number 3000 comes from: 30 seconds × 100 frames/second (10 ms hop).
Whisper always pads/truncates to exactly 30 seconds.

### Whisper architecture: encoder-decoder

Whisper is an encoder-decoder transformer (like the original "Attention Is
All You Need"), not a decoder-only model (like GPT/Llama/Qwen). This
distinction changes the compute and memory profile.

**Encoder**: A stack of transformer layers with **bidirectional**
self-attention (no causal mask). Every mel frame can attend to every other
frame — past and future. This is possible because the encoder processes the
entire 30-second audio chunk in a single pass. It's not generating anything
sequentially.

For Whisper-small: 12 layers, 12 heads, hidden size 768. The input (80,
3000) mel goes through two 1D convolutions that downsample by 2x, producing
(768, 1500). The encoder processes 1500 positions.

**Decoder**: Autoregressive, like an LLM. Each layer has three sublayers:

1. **Masked self-attention**: identical to an LLM. Each position attends
   only to previous positions. KV cache grows with each generated token.
2. **Cross-attention**: queries come from the decoder, but keys and values
   come from the encoder output. This is how the decoder "reads" the audio.
3. **FFN**: standard feedforward.

The cross-attention is the bridge between audio and text. Without it, the
decoder would be a language model with no knowledge of the audio.

### The two KV caches

In an LLM, you manage one KV cache per layer. In Whisper's decoder, you
manage **two** KV caches per layer, with completely different lifetimes:

**Self-attention KV cache**: Grows by one entry per generated token. This is
identical to an LLM's KV cache.

**Cross-attention KV cache**: Computed once during decoder prefill from the
encoder's output. Fixed at 1500 positions for the entire decode. Never
recomputed, never grows.

When HuggingFace returns `past_key_values` from a Whisper decoder call, each
layer contains four tensors:

```python
past_key_values[layer] = (
    self_attn_key,      # (batch, heads, decoded_so_far, head_dim) — grows
    self_attn_value,    # (batch, heads, decoded_so_far, head_dim) — grows
    cross_attn_key,     # (batch, heads, 1500, head_dim) — fixed
    cross_attn_value,   # (batch, heads, 1500, head_dim) — fixed
)
```

Four tensors per layer, not two. The first two grow; the last two are frozen
after prefill.

### Decoder prompt tokens

Whisper's decoder expects an initial prompt that controls behavior:

```
<|startoftranscript|> <|language|> <|task|> [<|notimestamps|>]
```

For Hindi transcription: `[50258, 50276, 50359, 50363]`

These are part of the decoder vocabulary — not learned prompts, but special
token IDs. Language detection is actually a generation problem: if you don't
specify the language, Whisper predicts the language token as its second
output.

### The cascade approach

This pipeline uses the cascade approach: ASR → LLM → TTS. This is how most
production voice systems work (Sarvam, Alexa, Google Assistant). Each model
does one job. The downside: errors cascade. If ASR garbles a word, the LLM
gets garbage input.

The alternative — speech LLMs (GPT-4o, Gemini) — process audio tokens
directly in the LLM, eliminating the text bottleneck. This is where the
field is heading, but open implementations for Indic languages don't exist
yet.

---

## 3. Architecture and data flow

### Pipeline stages

```
Hindi audio (16 kHz WAV, up to 30s)
     │
     ▼
[1] mel.py — load_audio() + extract_mel()
     │   CPU: librosa resample → WhisperFeatureExtractor → GPU transfer
     │   Output: (1, 80, 3000) tensor on CUDA
     │   Timed: perf_counter_ns (CPU work)
     ▼
[2] encoder.py — WhisperEncoder.forward()
     │   GPU: 2 conv layers + 12 bidirectional transformer layers
     │   Output: (1, 1500, 768) hidden states
     │   Timed: CUDA events
     │   Runs ONCE per audio chunk
     ▼
[3] decoder.py — WhisperDecoder.prefill() + decode_one() loop
     │   GPU: 12 decoder layers × (self-attn + cross-attn + FFN)
     │   Prefill: 4 prompt tokens → first output logit
     │   Decode: token-by-token until EOS or 225 max
     │   Populates both KV caches during prefill
     │   Timed: CUDA events per step
     │   Output: list of token IDs
     ▼
[4] runner.py — ASRRunner._transcribe()
     │   Chains [1]→[2]→[3], adds timing aggregation
     │   Detokenizes: token IDs → Hindi text
     │   Output: ASRResult(text, token_ids, metrics)
     │
     ├── ASRRunner.transcribe_long_file()/transcribe_long_array()
     │   Splits arbitrary-length audio into 25 s windows with 5 s overlap
     │   (all windows remain within Whisper's 30 s encoder limit)
     │   Merges only a verified token suffix/prefix at each boundary
     │   Output: LongFormASRResult(text, chunks, per-chunk metrics)
     ▼
[5] orchestrator.py — VoicePipeline._build_prompt()
     │   CPU: Applies Qwen chat template with Indic system prompt
     │   Output: formatted prompt string
     ▼
[6] llm/runner.py — LLMRunner.generate()
     │   GPU: Qwen3-0.6B explicit prefill + decode loop
     │   Same pattern as ASR decoder: token-by-token, KV cached
     │   Timed: CUDA events (prefill + per-step decode)
     │   Output: LLMResult(text, token_ids, metrics)
     ▼
[7] tts/synthesis.py — TTSSynthesizer.synthesize()
     │   Network: edge-tts API call (Microsoft neural voices)
     │   Zero GPU cost
     │   Output: MP3 audio bytes
     ▼
Spoken Hindi answer
```

### Memory layout on T4

```
T4 total:          14.6 GiB
─────────────────────────────
Whisper-small FP16:  0.49 GiB (488 MB model weights)
Qwen3-0.6B BF16:    1.12 GiB (model weights)
─────────────────────────────
Both models:         1.61 GiB
KV caches + activs:  ~0.05 GiB (small at these model sizes)
Peak measured:       1.66 GiB
Free:               ~13.0 GiB

Strategy: CONCURRENT (both models stay resident)
```

For larger Indic LLMs (2B+), the pipeline auto-switches to SEQUENTIAL
strategy: offload Whisper to CPU after transcription, load LLM, pay the
swap cost (~500-1000 ms).

---

## 4. File-by-file code walkthrough

### `asr/explicit/loader.py`

**Purpose**: Load a Whisper checkpoint onto CUDA.

**Key decisions**:
- Returns a frozen dataclass `LoadedWhisper` (immutable after load)
- Uses `WhisperProcessor` (bundles tokenizer + feature extractor) because
  the decoder needs special token IDs from the tokenizer and mel extraction
  needs the feature extractor config
- Uses `WhisperForConditionalGeneration` (not `WhisperModel`) because we
  need the `lm_head` for next-token logits
- FP16, not BF16: Whisper was trained in FP16. Using BF16 would introduce
  unnecessary numerical differences. Qwen uses BF16 separately.

### `asr/explicit/mel.py`

**Purpose**: Convert audio waveform to mel spectrogram for Whisper.

**Key decisions**:
- `load_audio()` and `extract_mel()` are separate functions. Loading is I/O
  bound (disk read + resample), mel extraction is CPU compute bound. The
  pipeline needs both entry points (file path for benchmarks, numpy array
  for Gradio)
- Uses `perf_counter_ns` for timing, not CUDA events, because mel extraction
  is CPU work
- Uses HuggingFace's `WhisperFeatureExtractor` for mel computation, not
  manual librosa. Reason: Whisper's exact filterbank weights, log offset,
  and padding behavior are baked into the checkpoint. A different mel would
  degrade accuracy.
- The `.to(device=device, dtype=dtype)` call on the features tensor is the
  CPU→GPU transfer boundary. This is included in `mel_ms`.
- `WHISPER_SAMPLE_RATE = 16_000` — Whisper only accepts 16 kHz. If the input
  audio is at a different rate, `load_audio_from_array()` resamples it.

**Constants**:
- 80 mel channels: from psychoacoustics (mel scale filterbank)
- 3000 frames: 30 seconds × 100 frames/second (10 ms hop)
- 25 ms window, 10 ms hop: Whisper's STFT parameters (not configurable)

### `asr/explicit/encoder.py`

**Purpose**: Run the Whisper encoder as a single timed GPU operation.

**Key decisions**:
- `model.get_encoder()` (public API) vs `model.model.encoder` (internal).
  Using the public method is more robust across HuggingFace versions.
- `torch.cuda.synchronize()` before timing: forces all prior GPU work to
  complete before measurement starts. Without this, overlapping work from
  previous operations could inflate the timing.
- Returns `BaseModelOutput` (not a raw tensor) because HuggingFace's decoder
  expects this wrapper type for `encoder_outputs=` parameter.
- Reports `hidden_size` and `sequence_length` for validation: should be 768
  and 1500 for Whisper-small.

**Why a separate module?** The encoder is non-autoregressive (one forward
pass). Separating it from the decoder enables independent timing, potential
CUDA graph capture of the encoder alone, and encoder output caching for
multi-decode scenarios.

### `asr/explicit/decoder.py`

**Purpose**: Autoregressive Whisper decoder with dual KV cache management.

This is the most complex module. It mirrors the LLM's `ExplicitDecodeRunner`
pattern but handles two KV caches.

**`DecoderState` dataclass**:
- `past_key_values`: tuple of (self_attn_kv, cross_attn_kv) per layer. HF
  manages the internal split; we pass the whole object through.
- `encoder_outputs`: stored because HF requires it on every `model()` call
  for shape validation, even though the cross-attention KV is in
  `past_key_values`.
- `next_token`: (1, 1) tensor — the token to feed on the next decode step.
- `decoded_tokens`: accumulated output token IDs for final detokenization.

**`_build_prompt_ids()`**: Constructs the 3-4 token decoder prompt. Uses
`getattr()` with fallbacks because HuggingFace's `generation_config` format
has changed across versions — `lang_to_id` sometimes expects `"<|hi|>"`,
sometimes `"hi"`.

**`prefill()`**: Passes decoder prompt tokens through the model with
`past_key_values=None` (signals first call). This populates both KV caches.
The cross-attention K and V are computed here from the encoder outputs and
never recomputed. Returns `DecoderState` + timing.

**`decode_one()`**: Passes one new token through the model with cached
`past_key_values`. Self-attention cache grows by 1; cross-attention cache is
reused unchanged. Returns new `DecoderState` + timing.

**Immutability**: `new_decoded = list(state.decoded_tokens)` creates a new
list each step. Each `DecoderState` is a snapshot. This prevents bugs from
shared mutable state.

### `asr/explicit/runner.py`

**Purpose**: Chain mel → encoder → decoder into a single transcription call
with full metrics.

**`ASRMetrics`**: Per-stage latency breakdown. Key computed properties:
- `mean_decode_ms`: average per-token decode time
- `total_decode_ms`: sum of all decode steps
- `real_time_factor`: total_ms / (audio_duration × 1000). RTF < 1.0 = faster
  than real-time.

**`ASRRunner._transcribe()`**: The core loop, annotated:

```python
# Stage 1: mel spectrogram (CPU)
mel = extract_mel(waveform, ...)
# Stage 2: encoder forward (GPU, runs once)
enc = self.encoder.forward(mel.input_features)
# Stage 3: decoder prefill (GPU, populates both KV caches)
state, prefill_ms = self.decoder.prefill(enc.encoder_outputs, ...)
# Stage 4: autoregressive decode (GPU, per-token)
for step in range(max_new_tokens):
    token_id = int(state.next_token.item())
    if token_id in eos_ids: break
    state, step_ms = self.decoder.decode_one(state)
```

**Two entry points**: `transcribe_file()` (loads from disk, for benchmarks)
and `transcribe_array()` (takes numpy array, for Gradio/pipeline). Both
delegate to `_transcribe()`.

**`max_new_tokens=225` default**: Whisper's 30-second chunk produces at most
~225 text tokens (empirically ~7-8 tokens per second of audio).

### `llm/loader.py`

**Purpose**: Load an LLM checkpoint for the pipeline. Mirrors the Whisper
loader pattern.

**Key decisions**:
- BF16 dtype (Qwen was trained in BF16)
- Sets `pad_token_id = eos_token_id` if unset (Qwen doesn't ship a pad token)
- Returns frozen `LoadedLLM` dataclass

### `llm/runner.py`

**Purpose**: Explicit LLM prefill/decode with CUDA timing. Self-contained,
no dependency on the inference-engine repo.

**Structural parallel with ASR decoder**:

```
Whisper decoder:                    LLM decoder:
  prefill(prompt_ids)                 model(input_ids, attention_mask)
  for step:                           for step:
    decode_one(state)                   model(next_token, past_kv)
    state.past_key_values grows         past_kv grows
    cross-attn KV unchanged             (no cross-attention)
```

The LLM loop also manages `attention_mask` explicitly — concatenating a `[1]`
each step. Whisper's decoder handles masking internally.

### `pipeline/memory.py`

**Purpose**: Decide whether both models fit on GPU simultaneously.

**`choose_strategy()`**: Measures total VRAM, estimates both models' weight
memory, reserves 25% headroom for KV caches and activations. If both fit:
CONCURRENT. Otherwise: SEQUENTIAL (offload/reload between ASR and LLM).

**`estimate_model_bytes()`**: Sums `param.nelement() * param.element_size()`
across all parameters. This gives the weight memory, not activation memory.
For small models at batch 1, this is a good enough estimate.

**`offload_to_cpu()` / `reload_to_gpu()`**: Move model parameters between
CPU and GPU. `torch.cuda.empty_cache()` after offloading reclaims the CUDA
allocator's cached blocks.

### `pipeline/orchestrator.py`

**Purpose**: Chain ASR → LLM with timing and memory management.

**System prompts**: Per-language prompts tell the LLM to respond concisely
in the same language mix. "This will be spoken aloud" encourages brevity.

**`_build_prompt()`**: Uses `tokenizer.apply_chat_template()` for models that
support it (Qwen3, Gemma). Falls back to a plain format for models without
templates.

**`run()`**: The main pipeline method:
1. If SEQUENTIAL: reload Whisper to GPU
2. Run ASR → get transcript
3. Snapshot VRAM after ASR
4. If SEQUENTIAL: offload Whisper, reload LLM
5. Build prompt from transcript
6. Run LLM → get answer
7. Snapshot VRAM after LLM
8. If SEQUENTIAL: offload LLM, reload Whisper (ready for next request)
9. Record peak memory

**`run_array()`**: Same flow but for in-memory audio (Gradio). Skips the
file-load step. Does not handle model swapping (assumes CONCURRENT for
simplicity in the demo path).

### `tts/synthesis.py`

**Purpose**: Text-to-speech using edge-tts (Microsoft's free neural TTS API).

**Why edge-tts, not a GPU model?** On T4, VRAM is split between Whisper and
the LLM. A GPU-based TTS model (VITS: ~80 MB, StyleTTS2: ~150 MB) would
compete for memory. edge-tts gives high-quality Hindi/Telugu voices at zero
GPU cost. The interface is designed so swapping in a local model is a single
module change.

**Default voices**:
- Hindi: `hi-IN-SwaraNeural`
- Telugu: `te-IN-ShrutiNeural`
- Tamil: `ta-IN-PallaviNeural`
- English: `en-IN-NeerjaNeural` (Indian accent)

**Async handling**: edge-tts is async. `synthesize()` uses
`ThreadPoolExecutor` + `asyncio.run()` to call it synchronously from any
context (notebook, script, pipeline).

### `asr/training/lora.py`

**Purpose**: LoRA fine-tuning of Whisper-small for Hindi ASR.

Covered in detail in Section 7 below.

### `benchmarks/asr_wer.py`

**Purpose**: Evaluate WER on FLEURS Hindi test set.

Supports both baseline (load from HuggingFace) and fine-tuned (load from
LoRA checkpoint with `PeftModel.from_pretrained()` + `merge_and_unload()`).
Outputs JSON with model name, WER, and sample count.

### `benchmarks/pipeline_e2e.py`

**Purpose**: Full pipeline waterfall benchmark with multi-run averaging.

Loads both models, runs warmup, then N timed runs. Outputs a JSON with
per-stage averages, total latency, audio-to-first-LLM-token, and peak VRAM.

### `tests/test_asr.py`

**Purpose**: Correctness — explicit ASR loop vs `model.generate()`.

Uses a synthetic sine tone (440 Hz, 3 seconds) to validate that the explicit
encoder/decoder loop produces the same token sequence as HuggingFace's
`model.generate()` with greedy decoding. Also validates that all metrics
fields are populated and encoder geometry is correct (768 hidden, 1500 seq
length for Whisper-small).

### `tests/test_llm.py`

**Purpose**: Validate that the LLM runner produces non-empty output with
populated metrics.

### `tests/test_pipeline.py`

**Purpose**: Integration test — pipeline produces output, metrics are
complete, memory strategy is correct, VRAM is within bounds.

### `scripts/setup.sh`

**Purpose**: Install all dependencies in Colab. Installs the project as an
editable package with `pip install -e .` so `asr`, `llm`, `pipeline`, `tts`
are importable from any script.

### `scripts/preflight.py`

**Purpose**: Verify GPU, CUDA, all dependencies, and project imports before
running anything. Catches missing packages early.

### `scripts/demo.py`

**Purpose**: Quick end-to-end demo. Downloads a FLEURS Hindi sample, runs
the full pipeline, prints the latency waterfall.

### `demo/app.py`

**Purpose**: Gradio web interface. Record audio or upload a file, see the
transcript, LLM answer, latency breakdown, and optionally hear the TTS
output. Models are lazy-loaded on first request.

### `setup.py`

**Purpose**: Makes the project installable with `pip install -e .`.
`find_packages()` discovers `asr`, `llm`, `pipeline`, `tts`, `benchmarks`
as importable Python packages. Required because running
`python scripts/demo.py` wouldn't find `from asr.explicit import ...`
without this.

---

## 5. Dependency map

### Python packages

| Package | Version | Used by | Purpose |
| --- | --- | --- | --- |
| torch | Colab-provided | everywhere | GPU compute, tensors |
| transformers | >=4.51 | asr, llm | Whisper + Qwen model code |
| accelerate | >=1.4 | model loading | HuggingFace model sharding |
| peft | >=0.11 | asr/training | LoRA adapter application |
| datasets | >=2.19 | asr/training, benchmarks | FLEURS dataset loading |
| evaluate | >=0.4 | benchmarks | WER metric computation |
| jiwer | >=3.0 | evaluate (transitive) | Word error rate algorithm |
| librosa | >=0.10 | asr/explicit/mel | Audio loading + resampling |
| soundfile | >=0.12 | scripts, benchmarks | WAV file I/O |
| edge-tts | >=6.1 | tts | Microsoft neural TTS API |
| gradio | >=4.0 | demo | Web interface |
| pytest | >=8.0 | tests | Test runner |

### What is NOT a dependency

| Package | Why not |
| --- | --- |
| torchao | Must be uninstalled — Colab's version conflicts with PEFT |
| bitsandbytes | Not needed — LoRA, not QLoRA |
| flash-attn | Not available on T4 (compute capability 7.5) |
| deepspeed | Single GPU, not needed |
| vllm / sglang | This is a reference pipeline, not a server |

### HuggingFace models

| Model | Size (FP16/BF16) | Used for |
| --- | --- | --- |
| openai/whisper-small | 488 MB (FP16) | ASR encoder-decoder |
| Qwen/Qwen3-0.6B | 1.2 GiB (BF16) | LLM for answer generation |
| google/fleurs (hi_in) | ~1.5 GB dataset | Training + evaluation data |

---

## 6. KV cache memory math

### Whisper-small decoder

Geometry: 12 layers, 12 heads, head dimension 64, FP16 (2 bytes).

**Cross-attention KV cache (fixed, computed once during prefill)**:

```
2 (K+V) × 12 (layers) × 12 (heads) × 1500 (encoder positions) × 64 (head_dim) × 2 (bytes)
= 2 × 12 × 12 × 1500 × 64 × 2
= 53,084,160 bytes
≈ 50.6 MB
```

This is allocated during decoder prefill and never changes. It's the
decoder's permanent view of the audio.

**Self-attention KV cache (grows per token)**:

```
Per token: 2 × 12 × 12 × 1 × 64 × 2 = 36,864 bytes ≈ 36 KB
At max tokens (225): 36 KB × 225 ≈ 8.1 MB
```

**Total Whisper decoder KV memory at max decode**: 50.6 + 8.1 = 58.7 MB

The cross-attention cache is ~6x larger than the self-attention cache at
maximum length. This is the opposite of LLMs, where self-attention
dominates.

### Qwen3-0.6B (for comparison)

Geometry: 28 layers, 8 KV heads (GQA 4:1), head dimension 128, BF16.

```
Per token: 2 × 28 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB
```

Qwen's per-token KV cost is 112 KB vs Whisper's 36 KB. But Whisper has the
additional 50.6 MB fixed cross-attention overhead that Qwen doesn't have.

---

## 7. LoRA fine-tuning details

### Configuration

```
Base model:       openai/whisper-small (244M params, 488 MB FP16)
LoRA rank:        16
LoRA alpha:       32
LoRA dropout:     0.05
Target modules:   q_proj, v_proj, k_proj, out_proj
Trainable:        3,538,944 / 245,273,856 = 1.44%
Adapter file:     14.2 MB
```

### Why these targets?

Whisper-small has 12 encoder layers (1 self-attention each) + 12 decoder
layers (1 self-attention + 1 cross-attention each):

```
Encoder: 12 layers × 1 attn × 4 projections = 48 LoRA matrices
Decoder self-attn: 12 × 1 × 4 = 48 LoRA matrices
Decoder cross-attn: 12 × 1 × 4 = 48 LoRA matrices
Total: 144 LoRA matrices
Each at rank 16 on 768×768 projection: 2 × 16 × 768 = 24,576 params
144 × 24,576 = 3,538,944 total trainable params
```

Attention projections are targeted because they encode language-specific
phonetic patterns. FFN layers are more generic (shared acoustic features).

**Encoder LoRA** adapts how mel spectrograms are read — Hindi retroflex
consonants (ट/ठ/ड/ढ), nasalized vowels, and specific formant patterns.

**Decoder self-attention LoRA** adapts text generation patterns — Devanagari
bigram statistics, code-switching transitions (Latin ↔ Devanagari).

**Decoder cross-attention LoRA** adapts the audio-to-text alignment — the
phoneme-to-grapheme mapping for Hindi is different from English.

### Why LoRA, not QLoRA?

Whisper-small (488 MB) + LoRA adapters (~3.5M params) + batch 4 + gradient
checkpointing = ~6-7 GiB training memory. T4 has 16 GiB. There is no
memory constraint to solve. QLoRA would save ~250 MB at the cost of NF4
dequantization overhead during every forward pass.

### Why no task_type in LoRA config?

PEFT's `task_type="SEQ_2_SEQ_LM"` wraps the model in
`PeftModelForSeq2SeqLM`, which remaps `input_features` to `input_ids` in
its `forward()` method. This breaks Whisper because Whisper's encoder
expects `input_features`, not `input_ids`. Omitting task_type creates a
base `PeftModel` that passes through to the underlying model's forward
without argument remapping.

### Training data

FLEURS (google/fleurs, hi_in split): 2,120 training samples, 239 validation
samples. Each sample is a (audio, transcription) pair with Hindi speech and
Devanagari text. Audio durations range from ~3-15 seconds.

### Results

```
Baseline WER (pre-fine-tuning):  75.80%
Fine-tuned WER:                  37.66%
Improvement:                     38.14 percentage points (50% relative)
Train loss:                      0.945
Training time:                   ~30 minutes on T4
```

---

## 8. Pipeline memory management

### Strategy selection

At pipeline construction, `choose_strategy()` runs:

```python
whisper_bytes = estimate_model_bytes(whisper.model)  # ~488 MB
llm_bytes = estimate_model_bytes(llm.model)          # ~1.2 GB
headroom = total_vram * 0.25                          # ~3.65 GB on T4
if whisper_bytes + llm_bytes + headroom <= total_vram:
    return CONCURRENT
else:
    return SEQUENTIAL
```

For Whisper-small + Qwen-0.6B: 488 MB + 1.2 GB + 3.65 GB = 5.34 GB < 14.6
GB → CONCURRENT.

For Whisper-small + a 7B model: 488 MB + 14 GB + 3.65 GB = 18.1 GB > 14.6
GB → SEQUENTIAL.

### Sequential strategy flow

```
1. Reload Whisper to GPU
2. Run ASR
3. Offload Whisper to CPU (torch.cuda.empty_cache())
4. Reload LLM to GPU
5. Run LLM
6. Offload LLM to CPU
7. Reload Whisper (ready for next request)
```

The swap cost is ~500-1000 ms per direction (measured by model_swap_ms in
the metrics).

---

## 9. Measured results and analysis

### Hardware

Tesla T4, 14.6 GiB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128.

### ASR latency breakdown

| Stage | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Mel extraction | 22.4 ms | 21.7 ms |
| Encoder | 326.9 ms | 25.9 ms |
| Decoder prefill | 79.2 ms | 16.4 ms |
| Decode (all tokens) | 1,194.1 ms | 1,283.7 ms |
| Decode tokens | 119 | ~120 |
| RTF | 0.275 | 0.282 |

Note: baseline encoder is slower on first measurement due to warmup
effects. Both achieve RTF ~0.28 (3.5x faster than real-time).

### WER comparison

| Model | WER (50 FLEURS Hindi) |
| --- | ---: |
| Whisper-small (baseline) | 75.80% |
| Whisper-small (LoRA fine-tuned) | 37.66% |

### Pipeline latency

| Stage | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Total pipeline | 5,325 ms | 4,061 ms |
| Audio → 1st LLM token | 2,739 ms | 1,476 ms |
| LLM decode (64 tokens) | 2,548.5 ms | 2,569.6 ms |
| Peak VRAM | 1.65 GiB | 1.66 GiB |
| Memory strategy | concurrent | concurrent |

LLM decode dominates both pipelines (~60% of total time).

---

## 10. Known limitations and future work

### Current limitations

1. **LLM quality**: Qwen3-0.6B produces poor Hindi answers. It
   hallucinates, doesn't understand Indic input well, and wastes tokens on
   `<think>` reasoning. The LLM is a plug-in — swapping in Sarvam-2 or
   Gemma-2-2B-IT improves quality without pipeline changes.

2. **Audio length**: Whisper processes max 30 seconds per chunk. Longer
   audio requires chunking with overlap and stitching, which this pipeline
   doesn't implement.

3. **Batch size 1 only**: The pipeline processes one request at a time. No
   concurrent batching, no request queuing.

4. **No streaming**: The LLM generates all tokens before TTS starts. A
   production system would stream TTS as LLM sentences complete.

5. **WER still 37.7%**: Roughly 1 in 3 Hindi words is wrong. More training
   data (CommonVoice, MUCS code-switched corpus) and more epochs would
   improve this. Whisper-medium with LoRA would likely reach ~20% WER.

6. **No correctness test for Hindi**: The ASR correctness test uses a
   synthetic tone, not real Hindi speech. A proper test would compare
   explicit-loop output vs model.generate() on Hindi audio.

### Potential extensions

1. Swap LLM for a stronger Indic model (Gemma-2-2B with AWQ quantization)
2. Add code-switched Hindi-English training data (MUCS corpus)
3. Implement 30-second chunking for longer audio
4. Stream TTS synthesis concurrent with LLM generation
5. Add Whisper-medium with LoRA for better WER
6. Implement the sequential memory strategy benchmark (currently untested
   because Whisper-small + Qwen-0.6B always fits concurrently)
7. Profile with torch.profiler to find actual bottlenecks in the decode loop
