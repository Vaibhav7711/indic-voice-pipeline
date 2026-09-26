# Serving tiers

Correctness-first means the explicit implementations are the reference, not
the serving path. This document is how the fast tiers get in front of them,
and what has to be true before their numbers count.

## Where the time actually goes

From the 12 persisted live turns (`results/live/turns.jsonl`, 4435.5 ms p50):

| stage | mean | on the critical path? |
| --- | ---: | --- |
| ASR decode | 342.2 ms | **no** — the candidate final is decoded during endpoint silence, so `asr_after_endpoint ≈ 0` |
| endpoint → final transcript | 660.8 ms | yes, but it is the VAD's silence window, not compute |
| transcript → first LLM token | 877.9 ms | yes, and it **grows with dialogue history** (209 → 1474 ms over 12 turns, r = 0.99) |
| first token → first speakable unit | 2445.7 ms | yes |
| synthesis of that unit | 144.7 ms | yes |

**94% of the controllable latency is the LLM.** This is the single most
important fact for choosing what to optimize, and it is the opposite of where
intuition points — "the model is slow" sounds like an ASR problem in a speech
pipeline. A Whisper engine measured at 1.3145× saves ~90 ms of a 4.4 s turn,
about 2%. The LLM half is 3324 ms.

## ASR tier: CTranslate2

Adopted on measured evidence: 1.5873% mean WER against the explicit runner
(inside the pre-registered 5% tolerance) at **1.3145×** on a T4 int8-float16.
See `docs/EXPERIMENTS.md`.

```bash
python scripts/convert_ct2.py --model openai/whisper-medium \
    --adapter Hugme6969/whisper-medium-hindi-lora --out models/ct2-medium-hi
python scripts/live_agent.py --asr-engine ct2 --ct2-model models/ct2-medium-hi
```

The converted directory is required, never built on demand: conversion takes
minutes, and doing it silently inside startup would hide that cost inside a
measurement.

## LLM tier: an OpenAI-compatible server

`llm/engines/http_engine.py` speaks `/v1/completions` with `stream: true`,
which covers vLLM, SGLang, llama.cpp's server, Ollama, TGI — and
[`Vaibhav7711/full-inference-engine`](https://github.com/Vaibhav7711/full-inference-engine),
which is what this pipeline is set up for.

Its optimizations were each measured on a T4 or an RTX 4060 with interleaved
A/Bs and a token-identity gate against stock Transformers — the same standard
this repo holds itself to. Its recorded figures on a T4 with ~656-token chat
prompts are TTFT p50 ~0.3–0.4 s and ITL p50 19.8 ms, for Qwen3-0.6B.

### The model: Qwen3-1.7B

The default is **Qwen3-1.7B**, served by the engine: 28 layers, 16 query heads,
8 KV heads, head dimension 128, which clears the paged kernels' constraints
(under the 128 limit, a multiple of 8, GQA divisible, no sliding window, not
MLA).

**Why not 4B.** The parity gate decodes the same weights **twice on one card** —
the server holds a copy and the reference runner needs one in its own process.
Two 4B copies is ~15 GiB of weights on a 15 GiB T4, and a measured Colab run
died at 87% of the second load with no explanation. This cost was missing from
the VRAM budget below until that run found it. 1.7B fits twice over with room
for Whisper.

4B also loses on the evidence already collected. The T4 bake-off measured
Devanagari ratio **1.000 for both 0.6B and 1.7B against 0.988 for 4B**, and
first-sentence p50 at 1744 / 2534 / 3553 ms. So 1.7B keeps the script purity
4B gives up, at 1.45× the 0.6B latency rather than 2×.

**Speculative decoding is not available on one card** either way — the
mechanism that could most plausibly pay for a larger model's decode cost needs
two GPUs, since `create_app` raises when target and draft share a device, and
the engine's own record notes the 0.6B drafter did not beat target-only on the
measured T4. A larger model's win here has to come from CUDA-graphed decode and
chunked prefill alone.

### Known blocker: the engine corrupts streamed Indic text

Measured 2026-09-25 on a Colab T4, Qwen3-0.6B: 1/1 English prompt identical,
**0/4 Hindi prompts agreeing**, every served Hindi response studded with
U+FFFD. `nमस्ते, मौसम कैसा है?` came back as `नमस्�े, म�सम क�सा ह�?` — the
correct characters *dropped*, not merely altered.

It is not a decode divergence. Qwen's tokenizer is byte-level BPE, so a 3-byte
Devanagari character is routinely split across two tokens; the engine's
`_stream` decodes the tokens received so far and sends
`decoded[len(already_sent):]`, a slice by length. When a character is
half-arrived that decode ends in U+FFFD, which gets sent; when the rest
arrives the corrected text is no longer an extension of what was sent, so the
replacement character can never be retracted and the real character is skipped.
ASCII never triggers it, which is why it survived a suite whose streaming
tests are all English.

**This blocks the whole sweep.** Corrupted text goes to TTS and is pronounced.
A larger model does not help — any byte-level BPE over any multi-byte script
hits it.

Apply the fix before starting the server; uvicorn imports the module once:

```bash
python scripts/patch_engine_utf8.py --engine-root ../full-inference-engine
python scripts/patch_engine_utf8.py --engine-root ../full-inference-engine --check
```

It is idempotent, so a re-run of a Colab notebook is safe, and it refuses
rather than guessing if the engine's `_stream` has changed — which is what
upstream fixing this itself looks like, and the right response then is to
delete the patcher.
[`ENGINE_BUG_UTF8_STREAMING.md`](ENGINE_BUG_UTF8_STREAMING.md) is the report
to send upstream: the byte-level walkthrough, the patch and the three tests
that would have caught it.

Downstream, `HttpEngineMetrics.replacement_chars` counts U+FFFD, the startup
probe uses a Devanagari prompt so this is caught before the first turn, and
`scripts/engine_parity.py` fails on corruption with its own diagnosis rather
than reporting a divergence.

### Match the dtypes, or the gate is meaningless

The same run had a second, independent flaw: the reference loaded **bfloat16**
(`pick_dtype`'s default on that device) while the server served **float16**.
Two greedy decoders over different numerics diverge for reasons that say
nothing about either engine. `scripts/engine_parity.py --dtype` now sets the
reference explicitly, defaults to `float16` to match
`llm/engines/server_app.py`, and records both dtypes in the report so a
mismatch can be checked for afterwards.

**On token agreement.** A greedy decoder diverges at the first step where two
implementations rank the top two candidates differently, so agreement tracks
how confident the model is per step. A small model has flatter logits and
smaller top-two margins, so an fp16 rounding difference flips a tie more
readily and the shared prefix is shorter. Low agreement at 0.6B is weak
evidence of an engine defect and strong evidence of a near-tie. That is a good
reason to prefer a larger model for a parity gate to be *informative* — and
`scripts/engine_parity.py` now reports mean shared-prefix fraction so the
comparison across checkpoints is a number rather than an impression. It is not
on its own a reason to expect 4B to be faster.

### Start it, then point the agent at it

```bash
# terminal 1 — waits for warmup, then prints the base url
python scripts/serve_llm.py --engine-root ../full-inference-engine

# terminal 2
python scripts/live_agent.py --llm-engine http \
    --llm-base-url http://127.0.0.1:8000/v1
```

The default `--app` is `llm.engines.server_app:create`, this repo's factory: it
takes `--model`, `--num-blocks`, `--block-size`, `--max-active` and
`--graph-buckets`, and prints the pool's cost in GiB before allocating it, so
an out-of-memory death is a number someone chose rather than a surprise.

The engine's own profiles stay available and are the better choice on an
architecture they were measured on — `--app
engine.server.api:create_rtx4060_flash_app` on an 8 GB Ada card, whose Flash
prefill and 256-token pages were chosen by an A/B there (prefill step −7.8%,
ITL p99 −22.6%). Those are zero-argument factories on purpose, so passing
`--model` alongside one is refused rather than ignored: a recorded
configuration that disagrees with the served one is the failure this launcher
exists to prevent.

**Wait for `/ready`, not `/health`.** The engine captures CUDA graphs and JITs
Triton kernels at startup and answers 503 on `/ready` until that finishes.
`/health` is liveness and answers throughout. A pipeline that starts measuring
against a warming server records a first turn seconds slower than the steady
state and then averages it in; `scripts/serve_llm.py` exists to make that
impossible to do by accident.

### Never the chat endpoint

`build_llm` refuses `chat=True` when a tokenizer is present, and this is not
fussiness. `agent.turn` already renders the prompt through `llm.prompting`,
which passes `enable_thinking=False`; the engine's `_chat_prompt` calls
`apply_chat_template` without it. Sending a rendered prompt to the chat route
would both double-wrap it and re-enable Qwen3's thinking mode, and the model
would spend its whole budget inside `<think>` with nothing to speak. That
exact failure is why `llm/prompting.py` exists, and it would come back looking
like a model-quality problem rather than a serving bug. The engine's pydantic
request model ignores unknown fields, so `chat_template_kwargs` would not help
and would not error either.

### VRAM, on one card

`pipeline/memory.py` has the budget. Two processes mean two CUDA contexts
(~300 MiB each) plus no shared allocator. That cost is accepted deliberately:
the engine needs to own its block pool and graph memory without the ASR
allocator fragmenting it underneath.

| on a 15 GB T4 | GiB |
| --- | ---: |
| Qwen3-1.7B fp16 weights, server process | ~3.8 |
| KV pool, 512 × 16 = 8192 tokens at 112 KiB | 0.875 |
| Whisper-medium fp16, other process | 1.5 |
| two CUDA contexts | ~0.6 |
| **subtotal, before graphs and activations** | **~6.8** |
| *plus, while the parity gate or the sweep runs:* a **second** copy of the LLM weights | ~3.8 |

That last row is the one this budget originally omitted, and it is what killed
a 4B run: `scripts/engine_parity.py` and the explicit arm of
`scripts/latency_ab.py` each load the weights again in their own process,
because a reference decode has to happen locally. The gate now runs a VRAM
preflight and refuses with the arithmetic rather than dying partway through the
load.

Graph capture costs memory per bucket, so the default buckets are `(1, 2)`
with `max_active=2` — one in-flight request, plus one so a barge-in's
replacement turn does not queue behind the request it cancelled.

Size the KV pool with `num_blocks × block_size` = total KV tokens across all
concurrent requests. `llm/engines/server_app.py` prints the arithmetic, and
reports `None` rather than a plausible figure for a checkpoint whose geometry
it does not have recorded — a made-up VRAM number is worse than none, because
it gets acted on.

## Answer quality is a separate question from serving

The engine changes how fast a token arrives, not what the token is. Both
engines produced, from the same checkpoint and the same prompt, *चीनी राजधानी
है।* — "it is the Chinese capital" — for "What is the capital of India?", and
the served arm repeated one 11-character phrase thirteen times until it hit the
token cap.

Two causes, neither of them the engine:

**Greedy decoding.** This pipeline decodes `argmax`, because every correctness
gate compares two greedy decoders. Qwen's guidance for these models is against
greedy precisely because it repeats. `benchmarks/answer_quality.py` measures
four decoders against 18 checkable cases; the rule is pre-registered in
`docs/EXPERIMENTS.md`. Greedy stays the reference for the gates whatever it
concludes — sampling is a serving option, not a replacement.

**A metric that could not see the problem.** The model was chosen on
`devanagari_ratio`, and all four bad answers score 1.000 on it. Script purity
was never answer quality.

## Before a fast tier's numbers count

```bash
python scripts/engine_parity.py --llm-base-url http://127.0.0.1:8000/v1
```

Both engines are greedy — `temperature = 0` takes precedence over every other
knob in the engine's `SamplingParams`, and the explicit runner is greedy — so
two greedy decoders over the same weights and the same prompt must produce the
same text. The gate compares the first 24 characters, because that is what the
user hears before the first unit reaches TTS, and reports full-text identity
separately.

Late divergence is expected and is not a defect: fp16 attention is not
associative, so two implementations can rank the top two candidates
differently at one step and then follow equally valid continuations. Early
divergence means **prefill** differs, which is a real defect. An empty
response fails, and zero prompts compared fails — a gate that did not run is
not a gate that passed.

Exit status is 0 only if every prompt agreed, so this is usable in a script.

## Measuring the remaining knobs

```bash
python scripts/latency_ab.py --rounds 8 \
    --arm baseline \
    --arm history200:history_tokens=200 \
    --arm units30:unit_chars=30 \
    --arm both:history_tokens=200,unit_chars=30
```

Arms share one set of loaded weights and are **interleaved**, one turn of each
per round. Running arm A ten times and then arm B ten times attributes GPU
warming, a neighbour's training job and a bad minute from a network TTS voice
entirely to the arm. The transcript is fixed across arms, because a microphone
would vary the prompt between them and there would be nothing left to compare.

To include the served engine as an arm, load it too:

```bash
python scripts/latency_ab.py --rounds 8 \
    --llm-engine explicit --llm-engine http \
    --arm reference --arm served:llm_engine=http
```

The decision rules are pre-registered in `docs/EXPERIMENTS.md`. The script
prints distributions and decides nothing.

Its primary metric is **committed transcript → first audio**, not
`response_latency_ms`. The latter is speech-end → agent-speaks and is `None`
without the endpoint-to-final segment; these turns carry fixed text and no
speech, so that segment does not exist and is not invented. It appears in the
summary as `n/a`, which is the honest reading. The measured 660.8 ms
endpoint-to-final floor sits under any perceived-latency figure and none of
these arms move it.
