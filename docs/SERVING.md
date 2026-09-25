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

That engine is the right fit for a specific reason: it serves **Qwen3-0.6B**,
which is already this pipeline's LLM, and its optimizations were each measured
on a T4 or an RTX 4060 with interleaved A/Bs and a token-identity gate against
stock Transformers — the same standard this repo holds itself to. Its recorded
figures on a T4 with ~656-token chat prompts are TTFT p50 ~0.3–0.4 s and ITL
p50 19.8 ms.

### Start it, then point the agent at it

```bash
# terminal 1 — waits for warmup, then prints the base url
python scripts/serve_llm.py --engine-root ../full-inference-engine \
    --app engine.server.api:create_app

# terminal 2
python scripts/live_agent.py --llm-engine http \
    --llm-base-url http://127.0.0.1:8000/v1
```

Use `--app engine.server.api:create_rtx4060_flash_app` on an 8 GB Ada card;
that profile's Flash prefill and 256-token pages were chosen by an A/B on that
architecture (prefill step −7.8%, ITL p99 −22.6%).

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

`pipeline/memory.py` has the budget. Whisper-medium fp16 is 1.5 GiB and
Qwen3-0.6B fp16 is 1.2 GiB, and two processes mean two CUDA contexts
(~300 MiB each) plus no shared allocator. That cost is accepted deliberately:
the engine needs to own its block pool and graph memory without the ASR
allocator fragmenting it underneath.

Size the KV pool with `num_blocks × block_size` = total KV tokens across all
concurrent requests. A voice agent is **one** stream with an ≤800-token
prompt, so the 1024 × 16 = 16384-token default is ~20× more than needed; the
`create_rtx4060_flash_app` profile's 64 × 256 is the same capacity in pages
FlashAttention-2 can use.

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
