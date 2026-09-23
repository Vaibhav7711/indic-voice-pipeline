# Working on the local RTX box

Written for whoever sits down at the RTX 4060 machine with this repo freshly
cloned. It is self-contained: nothing here depends on the Colab/Kaggle
sessions the project was built in.

## Why this machine exists

Everything in this project was developed on hosted notebooks, and four things
were simply not testable there. They are the whole reason for moving:

| Blocked in a hosted notebook | Works here |
| --- | --- |
| A reachable URL — the `gradio.live` tunnel, the Colab port proxy and SSR all failed in turn | `http://localhost:7860` |
| A real microphone and real speakers | `sounddevice` gets actual devices |
| Barge-in against a device that has already buffered audio | `stream.abort()` on a real stream |
| CUDA graphs / `torch.compile` — on a T4 the compiled decode was **0.9× (slower)**, and inductor logged "Not enough SMs" | Ada (sm_89) has native bf16; re-measure, do not assume |

What carries over unchanged: **WER and CER are device-independent**, so the
numbers in `docs/EXPERIMENTS.md` stand. **Every latency number does not** —
re-measure anything you intend to quote.

## Setup

### 1. Environment

```bash
git clone <this repo> indic-voice-pipeline && cd indic-voice-pipeline
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install a CUDA build of torch **first** (the project does not pin torch on
purpose — hosted runtimes ship their own):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; print('bf16 native:', torch.cuda.is_bf16_supported())"   # expect True on Ada
```

Then the project and its extras:

```bash
pip install -e ".[dev,demo,audio]"
pip install faster-whisper ctranslate2        # the CTranslate2 engine tier
python scripts/preflight.py
```

`bf16 native: True` matters: `llm/loader.py::pick_dtype` will now choose
bf16, where on the T4 it deliberately fell back to fp16 because sm_75 has no
bf16 tensor cores (a 0.6B model was decoding at ~41 ms/token because of it).

### 2. Audio devices

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

- **Linux** — if this errors, `sudo apt install libportaudio2`. Confirm both
  an input and an output device are listed.
- **Windows** — the `sounddevice` wheel bundles PortAudio; WASAPI devices
  should appear with no extra install.
- **WSL2** — there is normally **no audio device and no microphone**. Do not
  fight it: run the live tests from Windows-native Python instead, or use the
  file-replay paths (`--input-wav`, `--sink buffer`) and accept that real
  barge-in stays unmeasured. Check before planning a session around it.

### 3. The v2 adapter

The shipped adapter is **not in git** (weights do not belong there). It lives
in the private Hub checkpoint repo the training run wrote to, under `best/`:

```bash
huggingface-cli login                      # a token with read access
python - <<'PY'
from huggingface_hub import snapshot_download, whoami
repo = f"{whoami()['name']}/whisper-turbo-hindi-lora-ckpt"
path = snapshot_download(repo, allow_patterns=["best/*"], local_dir="models/v2-final")
print("adapter:", path + "/best")
PY
export ADAPTER=$PWD/models/v2-final/best        # Windows: set ADAPTER=...
```

It should contain `adapter_config.json` and `adapter_model.safetensors`.
Base model is `openai/whisper-large-v3-turbo` — `--adapter` also accepts a
Hub id directly if you publish it flat.

### 4. VRAM budget — this decides the LLM

fp16/bf16 resident weights, before KV cache and activations:

| Component | VRAM |
| --- | ---: |
| whisper-large-v3-turbo + merged LoRA | 1.6 GB |
| MMS-TTS Hindi (VITS, 36M) | 0.15 GB |
| Qwen3-0.6B | 1.2 GB |
| Qwen3-1.7B | 3.4 GB |
| Qwen3-4B | 8.0 GB |

- **8 GB card** — turbo + Qwen3-1.7B + MMS ≈ 5.2 GB resident, comfortable
  with KV cache. **Qwen3-4B does not fit** alongside Whisper in fp16; only
  4-bit, which measures a different model, so leave it out of the bake-off
  rather than quietly quantizing one candidate.
- **16 GB card** — Qwen3-4B fp16 fits alongside everything (≈ 9.8 GB). Include
  it; it is a genuinely different quality tier.

`pipeline/memory.py` picks concurrent vs sequential loading by measurement at
construction time, so it will adapt — but the bake-off loads one model at a
time and will OOM on a candidate that cannot fit.

## Verify before measuring

```bash
pytest                                    # 545 tests; 8 GPU-gated ones now run
python scripts/gpu_validation.py --whisper-model openai/whisper-large-v3-turbo \
    --adapter "$ADAPTER" --out-dir results/gpu_validation-rtx
```

16 checks, pass/fail JSON. Two expected results worth knowing so you do not
chase them: `language_detection_base_model` may **warn** (base Whisper flips
hi↔ur on short clips — expected, and the v2 adapter passes unrestricted), and
`llm_compiled_matches_eager` reports a speedup that was **below 1.0 on the
T4**. If it is now above 1.0, that is a real Ada finding worth recording.

## The measurement surface

Everything measured here should land in a file, not in a terminal read out to
someone. Two entry points:

**Batch — unattended, ~2 h:**

```bash
python scripts/bench_all.py --adapter "$ADAPTER" \
    --llm-models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B      # add Qwen3-4B only on 16 GB
```

Runs ten suites in the order of what they decide, tolerates a failing step,
writes a log per step and `results/bench_manifest.json`. It prints no
conclusions.

**Interactive — the URL that was the point of moving:**

```bash
python demo/app.py --no-share --preload        # http://localhost:7860
```

The Turn tab runs the same `VoiceTurn` as the agent, so its latency table is
what the agent measures; the Stream tab feeds the real `StreamingSession`.
`--preload` loads the models before serving so the first turn is not a
multi-minute download that looks like a hang.

**Live, with real audio:**

```bash
python scripts/live_agent.py --adapter "$ADAPTER" --tts mms --llm-compile
```

Microphone → streaming ASR → LLM → TTS → speakers, barge-in on a new speech
onset, every turn appended to `results/live/turns.jsonl`. **This is the row of
the test matrix that only this machine can do.** Drop `--llm-compile` if the
sweep says compilation is not a win here.

## Open questions and what settles each

| Question | Command | Recorded in |
| --- | --- | --- |
| Did today's decode guards (no-speech, loop guard) regress WER? | `bench_all.py --only guards_on,guards_off,compare_guards` | `results/eval/compare-guards.json` |
| Which LLM? **The largest open quality question** — Qwen3-0.6B answers Hindi questions by restating them | `bench_all.py --only llm` | `results/llm_bakeoff/summary.json` + every answer in `*.outputs.jsonl` |
| edge-tts or local MMS? | `--only tts` | `results/tts_bakeoff/summary.json` |
| Incremental finals / semantic endpointing on? | `--only streaming` | `results/streaming_eval/*/metrics.json` |
| CTranslate2 speedup, and does it match the reference? | `--only ct2_convert,ct2_check` | the sweep report |
| Do CUDA graphs pay off on Ada? | the sweep | `llm_compiled_matches_eager` |
| What is the real conversational latency, as a distribution? | 20 live turns | `results/live/turns.jsonl` |

Decision rules already written down, so they are not chosen after seeing the
numbers: `early-incr` becomes default if `WER off` rises < 1 pt versus `early`
and `end→final` drops; `early-sem` / `full` become default if split clips drop
from 16/100 to single digits at under ~150 ms average cost.

## House rules that earned their place

Each of these exists because ignoring it cost real time on this project:

1. **Measure before attributing.** A 4062 ms segment was called a TTS problem;
   it was mostly the LLM still generating. Instrument the split, then decide.
   Any field that cannot be measured is `None`, never `0.0` — a zero gets
   averaged into a benchmark, a `None` forces the question.
2. **The explicit runners are the reference.** An engine or a compiled path is
   a faster way to serve the same model and must match them token-for-token
   (fp16/fp32) or within a stated WER (int8) before it is used. See the
   correctness policy in the README.
3. **Commit the evidence, not the summary.** Every evaluation writes
   `run_config.json` with git SHA, package versions and the literal sampled
   indices. Quote numbers that have a file behind them.
4. **Record a missed target as missed.** v2 hit 23.83% WER against a
   pre-registered 22.0%. The miss is documented with its cause (multi-GPU
   DataParallel halved the optimizer steps), not retroactively softened.
5. **`git fetch && git reset --hard origin/main`, not `git pull`.** Local
   installs and result files make `pull` refuse, and with `-q` that refusal is
   silent. This cost two debugging rounds.
6. **Restart the interpreter after pulling.** Python caches modules; a pulled
   fix that "did not work" was usually a stale import.

## Known state

- **Shipped ASR:** turbo + LoRA v2 — 23.83% WER / 8.43% CER, 694 ms p50, RTF
  0.066 on 300 seeded FLEURS-hi test clips (T4). Full history, including a
  225-token bug that inflated an earlier number, in `docs/EXPERIMENTS.md`.
- **Not doing:** a v3 ASR re-run. Decided and recorded; the remaining value is
  elsewhere.
- **Empty:** `data/hard_set/` — the ledger requires per-category numbers for a
  result entry and none can be produced until clips are curated. Needs human
  judgement on clip selection.
- **Unmeasured:** everything in the table above.
