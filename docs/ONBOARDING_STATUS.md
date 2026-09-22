# Engineer onboarding: architecture and evidence status

This is a correctness-first reference pipeline, not a claim that every code path is production-validated. The repository deliberately distinguishes code, fake-backed tests, and real-model GPU evidence. Paths below ground the claims.

## 1. Architecture

### Audio to speech

`asr/explicit` is the ASR reference runtime. `asr/explicit/mel.py` loads/resamples audio and extracts Whisper features; `encoder.py` runs the encoder once per bounded window; `decoder.py:WhisperDecoder` performs greedy autoregressive decode; and `runner.py:ASRRunner` composes these stages, detects/selects language, detokenizes, and records `ASRMetrics`. Long files use `ASRRunner.transcribe_long_array/transcribe_long_file` and chunk-and-stitch (`asr/explicit/chunking.py`). This is code, not a `model.generate()` serving call.

`asr/streaming` turns microphone blocks into those bounded inputs. `StreamEndpointer` (`asr/streaming/endpointer.py`) makes causal frame decisions using the shared `VADConfig`/`AdaptiveThreshold` from `asr/vad.py`. `StreamingSession.push()` (`asr/streaming/session.py`) buffers audio and emits state, partial, candidate, and final updates; finals use the same explicit runner (or a compatible transcriber protocol).

`llm/loader.py` loads a chat model and `llm/runner.py:LLMRunner` answers a chat-templated transcript. `pipeline/orchestrator.py:VoicePipeline` is the non-streaming ASR → prompt → LLM waterfall and selects concurrent/sequential residency with `pipeline/memory.py`. Interactive output uses `agent/turn.py:VoiceTurn`: LLM deltas enter `tts/streaming.py:SentenceBuffer` and `iter_synthesis`, then `agent/playback.py:PlaybackSession` writes chunks to a sink such as `agent/audio.py:SoundDeviceSink`. `tts/synthesis.py` wraps non-streaming Edge TTS; `tts/streaming.py` forwards Edge MP3 chunks; `tts/local.py` implements local MMS-TTS.

The intended path is therefore:

`microphone/audio → asr/streaming → asr/explicit → llm → agent → tts → playback`.

### The two owned loops

Whisper is explicit because it is the measurement and correctness reference. `WhisperDecoder.prefill()` and its token loop (`asr/explicit/decoder.py`) retain Whisper’s four cache tensors per layer: growing self-attention K/V and fixed encoder-derived cross-attention K/V. `ASRRunner` records mel, encoder, language-detection, and per-step decode time. The real-model T4 check `served_model_matches_generate` in `scripts/gpu_validation.py` reports three exact greedy-token matches in `results/gpu_validation/report.json`.

The LLM loop is likewise explicit. `LLMRunner._decode_tokens()` performs prompt prefill once, then greedy one-token decode with timing, repetition penalty, and loop guard. The optional fast path uses Transformers `StaticCache`; `compile_decode=True` compiles only the single-token step and requires that cache (`llm/runner.py`). `stream()` decodes the accumulated sequence before yielding its suffix, avoiding invalid Hindi combining-character fragments. The explicit eager loop is therefore the comparison point for the compiled version.

### Streaming policy

The default baseline is adaptive energy VAD. `asr/vad.py:VADConfig` specifies a causal sliding-minimum noise floor plus margin, a −40 dBFS ceiling, −70 dBFS floor, 250 ms onset, 600 ms silence, and 300 ms padding. `StreamingConfig` enables partials by default with both 500 ms new-audio and 700 ms wall-clock gates. Partials are provisional and must not drive the agent (`asr/streaming/session.py`).

Candidate finals are also **on** by default: after 300 ms silence the session decodes the projected endpoint; if silence reaches the 600 ms endpoint it commits that candidate without another ASR pass (`early_final_silence_ms=300`, `from_candidate`, `asr_ms_after_endpoint`). Resumed speech discards it. Incremental finals are implemented but **off**: they stitch a tail onto a previous partial and can change output. Semantic endpointing is also **off**: `asr/streaming/policy.py:EndpointPolicy` lengthens the silence wait when a candidate ends in an incomplete Hindi phrase. Promotion of both is gated by the `early-incr`, `early-sem`, and `full` grids in `benchmarks/streaming_eval.py`.

### Agent turn and barge-in

`VoiceTurn.run()` builds the prompt, consumes LLM deltas, buffers complete sentences, then lazily synthesizes/plays each sentence; subsequent sentences can remain unsynthesized while the first plays (`agent/turn.py`, `tts/streaming.py`). `PlaybackSession.cancel()` uses a thread-safe event. Its `should_stop` callback is threaded from playback to synthesis and then into `LLMRunner.stream(..., should_stop=...)`, so barge-in stops queued audio, future synthesis, and further decode. It cannot retract audio already given to the operating system; the sink determines that final-buffer latency (`docs/STREAMING.md` §3).

### Serving tiers and training

The tiers are explicit PyTorch reference loops; LLM static-cache/compiled decode; and CTranslate2/faster-whisper ASR (`asr/engines/ct2.py:CT2Transcriber`). README’s “Correctness policy” requires: explicit ASR/LLM greedy-token identity with HF `generate()`; compiled LLM identity with eager explicit decode; CTranslate2 token identity at fp16/fp32 or ≤5% WER difference at int8; and pipeline equality to standalone stages. `scripts/gpu_validation.py` implements `llm_compiled_matches_eager` and optional `ct2_matches_explicit`. A tier is not accepted from a speed claim alone.

`asr/training/lora.py` has `v1` (Whisper-medium) and `v2-turbo` (large-v3-turbo) presets. Both use FLEURS Hindi plus 5,000 streamed IndicVoices Hindi examples, rank 16/alpha 32/dropout .05 q/k/v/out LoRA, batch 2 × accumulation 4, three epochs, and 200-step checkpoints (`PRESETS`, `TrainingConfig`). `load_data` mixes sources and new runs set language/task prefix tokens. With `--hub-repo`, `make_hub_callback`/ `resume_from_hub` use a private Hub repository as checkpoint store. The original completed v1 has a provenance caveat: its original Colab trainer was never committed (`docs/EXPERIMENTS.md`, “Active run”).

## 2. Current stage

| Component | Classification | Evidence / boundary |
| --- | --- | --- |
| Explicit Whisper, Hub v1 loading, long form, restricted detection | **validated on GPU** | T4 checks `load_whisper_with_hub_adapter`, `served_model_matches_generate`, `asr_quality_on_clips`, `long_form_chunking`, and the restricted part of `language_detection_adapter` in `results/gpu_validation/report.json`. |
| Streaming VAD/session, partials, default candidates | **validated on GPU** | `streaming_session` passes in that report; the 100-clip grid is `results/streaming_eval/medium-lora-test-100/`. It does not validate incremental/semantic modes. |
| Explicit LLM, compiled equivalence, waterfall | **validated on GPU** | `llm_prompt_and_decoding`, `llm_compiled_matches_eager`, and `pipeline_waterfall` pass in the report. The LLM selection itself is unfinished. |
| Edge streaming TTS, sentence pipeline, barge-in | **validated on GPU** | `voice_turn_real_tts` and `barge_in_real_tts` pass. The validation sink is buffered; real-device behavior remains sink-dependent (`docs/STREAMING.md` §7). |
| Local MMS-TTS choice | **validated on CPU only** | `results/tts_bakeoff/macbook-cpu/summary.json` records macOS arm64 CPU provenance; `tests/test_local_tts.py` covers fake-backed logic. |
| Playback state, fake streaming plumbing, incremental/semantic policy | **logic-complete, unit-tested against fakes** | `tests/test_agent.py`, `tests/test_streaming.py`, and `tests/test_streaming_eval.py`; non-default modes remain disabled. |
| CTranslate2 | **pending a GPU run** | Engine code exists, but `ct2_matches_explicit` is skipped because the committed sweep had no `--ct2-model`. |
| v2-turbo adapter | **pending a GPU run** | The preset exists, but no v2 adapter/run is under `results/`; success criteria are fixed in `docs/EXPERIMENTS.md`. |

Open decisions close with: LLM selection via `python -m benchmarks.llm_bakeoff ...`; TTS selection via `python -m benchmarks.tts_bakeoff ...`; incremental/semantic endpointing via `python -m benchmarks.streaming_eval --session-grid early,early-incr,early-sem,full ...`; CT2 via `python scripts/convert_ct2.py ...` then `python scripts/gpu_validation.py --ct2-model <dir>`; and v2 via `python asr/training/lora.py --preset v2-turbo ...` plus `benchmarks.asr_eval`. V2 promotion requires WER ≤22.0%, CER ≤8.5%, unrestricted `hi` detection, ASR p50 ≤1.35 s, and deletion runs ≤63 (`docs/EXPERIMENTS.md`, “Base-model decision for v2”).

## 3. Numbers

All ASR quality below is explicit `ASRRunner`, Tesla T4, the same seeded random 300/418 FLEURS Hindi test clips (seed 0), and standard normalization, as recorded in each run’s `run_config.json` and `metrics.json`.

| Measurement | Result | Source and conditions |
| --- | ---: | --- |
| Medium base | WER 40.4270%; CER 16.7354%; mean RTF .2298 | `results/eval/medium-base-test-300-seed0/metrics.json`. |
| Medium + Hindi LoRA v1 | WER 25.8246%; CER 9.6055%; mean RTF .2382 | `results/eval/medium-lora-test-300-seed0/metrics.json`; same clips. |
| v1 versus base | −14.6024 WER points (36.12% relative); p50 +79.408 ms; p90 +120.842 ms | `results/eval/compare-medium-base-vs-lora.json`. |
| Streaming offline/fixed VAD | Offline WER 27.2605%; fixed40 streamed 34.6016%, +7.3411 points; 25 split/3 empty | `results/streaming_eval/medium-lora-test-100/{summary.json,fixed40/metrics.json}`; T4, v1, seeded 100 clips, standard normalization, 0.5 s lead/1 s trail, 100 ms blocks. |
| Streaming adaptive VAD | default 27.0367% (16 split/0 empty); pad300 27.0367% (16/0); floor70 26.7681% (14/0) | corresponding `default`, `pad300`, `floor70` `metrics.json` files. The grid predates the final combined pad300+floor70 default, so it does not measure that exact configuration. |
| GPU waterfall | mel 14.38 ms; encoder 80.38; ASR decode 1900.22; LLM prefill 105.01; LLM decode 2323.56; total 4514.52 ms | `report.json:pipeline_waterfall.detail`; one validation clip, not a distribution benchmark. |
| GPU voice turn | prefill proxy 99.96 ms; first TTS chunk 619.74; response latency 719.82; total turn 4797.12 ms | `report.json:voice_turn_real_tts.detail.turn.metrics`; Edge network and non-streaming validation LLM (`first_token_is_prefill_proxy=true`). |

`docs/EXPERIMENTS.md` records large-v3-turbo base at 30.40% WER, 11.55% CER, 1238 ms p50, and .117 RTF on that T4 subset. It says the evidence is `results/eval/turbo-base-test-300-seed0/`, but that directory is absent. Treat it as documented-but-not-committed evidence, not a repository result.

The older `results/wer_*.json` figures (75.80%/37.66%) are superseded: whisper-small, first 50 ordered examples, no normalization/CER/saved predictions (`docs/EXPERIMENTS.md`, “Superseded results”). The earlier 291 ms response-latency figure is also superseded: per `docs/STREAMING.md` “Measurement honesty”, it placed “first token” after non-streaming generation returned and excluded about 2.2 s decode. Current code marks a prefill proxy when needed; unknown fields are `None`, not zero.

## 4. Known limitations and traps

- v1 skews unrestricted language detection because labels lacked `<|hi|><|transcribe|>`. Forced Hindi produced the quality results, but two Hindi clips detect as `ca` (p≈.64–.73); restrict candidates to hi/en/te until corrected training passes (`docs/EXPERIMENTS.md`; `report.json:language_detection_adapter`).
- `results/whisper-lora-hi-full/best` is an older **whisper-small** adapter, not v1, and cannot load on medium; its historical validation WER is 39.25% (`docs/EXPERIMENTS.md`, “Superseded results”).
- Never report CPU timing as GPU: README says CPU is for correctness/demos, whereas performance claims are GPU-only. The committed TTS bake-off is MacBook CPU.
- Edge TTS streams provider chunks but remains network-bound. Local low-latency synthesis on GPU and actual queued-audio cancellation of a real `AudioSink` are not yet measured (`docs/STREAMING.md` §7).
- Incremental finals and semantic endpointing are not yet measured for promotion. The default VAD floor/padding deltas are only a 100-clip comparison and need 300-clip confirmation before robust claims (`docs/EXPERIMENTS.md`).
- No hard-set result, final LLM bake-off, CT2 speedup, or v2 adapter result is committed. Telugu has prompt/TTS support but no adapter or evaluation (`README.md`).

## Where to start if you want to contribute

1. **Record the v2-turbo decision:** `python asr/training/lora.py --preset v2-turbo --output-dir <dir> --hub-repo <user>/<repo>`, then run `python -m benchmarks.asr_eval run ...` on the fixed seed-0 subset.
2. **Measure turn-taking modes:** `python -m benchmarks.streaming_eval --adapter Hugme6969/whisper-medium-hindi-lora --split test --limit 300 --seed 0 --session-grid early,early-incr,early-sem,full --out-dir results/streaming_eval/<run>`.
3. **Prove or reject CT2:** `python scripts/convert_ct2.py --model openai/whisper-medium --adapter Hugme6969/whisper-medium-hindi-lora --out <dir>`, then `python scripts/gpu_validation.py --ct2-model <dir>`; acceptance is the explicit-runner comparison, not speed alone.

