# Streaming ASR and voice-agent output

Written for an LLM inference engineer picking up voice systems. Where a speech
concept has a clean analogue in LLM serving, the analogue is stated rather than
assumed.

Covers two stages:

- **Stage 3 — streaming ASR** (`asr/streaming/`): microphone chunks in,
  structured transcript updates out.
- **Stage 4 — agent output** (`agent/`, `tts/streaming.py`): response
  generation, speech synthesis, playback, barge-in, and latency accounting.

Neither stage changes the explicit ASR runtime. There is still no
`model.generate()` and no `pipeline("asr")` anywhere in the serving path.

---

## 1. Why streaming is a different problem

Whisper is an offline encoder-decoder. It wants a bounded utterance and gives
one answer. A microphone gives you 10–40 ms of audio at a time, forever, and
never tells you when a sentence ended.

Three things have to be added.

| Need | Why | Where |
| --- | --- | --- |
| Endpointing | Whisper will not tell you the user stopped talking | `asr/streaming/endpointer.py` |
| Rate limiting | Re-transcribing on every buffer is ~50 decodes/second | `asr/streaming/session.py` |
| Buffering | Memory must track the utterance, not the session | `asr/streaming/session.py` |

### Online vs offline VAD

`asr/vad.py` already finds speech in a file. It is not reusable here, and the
reason is structural rather than a matter of tidying.

Offline VAD sees the whole waveform, finds every voiced run, then merges short
gaps and pads boundaries. It cannot answer "has the user stopped talking?"
until the recording is already over.

`StreamEndpointer` answers that question frame by frame having seen only the
past.

> **LLM analogy.** Offline VAD is prefill: one batched pass over a known-length
> sequence. Online endpointing is decode: an autoregressive loop with carried
> state, committing at step *t* without seeing *t+1*, unable to revise.

The online module imports `VADConfig` from `asr/vad.py` rather than defining its
own thresholds, so tuning one tunes both. Two tests assert the two agree on
utterance count and on boundaries to within one frame — the offline module stays
the reference implementation.

### The rules, and what each one is for

| Rule | Purpose | Failure if wrong |
| --- | --- | --- |
| `threshold_dbfs` | Frame is speech if RMS ≥ threshold | Too low: fans and hum open utterances |
| `min_speech_ms` | Consecutive voiced frames to open an utterance | Too low: a cough invokes Whisper |
| `min_silence_ms` | Silence that ends an utterance | Too low: users cut off mid-sentence |
| `padding_ms` | Widen boundaries either side | Too low: clipped first consonant, last vowel |

Two details that are easy to get wrong:

**Speech onset counts *consecutive* frames, not cumulative.** Three separate
coughs must never sum to one utterance.

**The endpoint rule mirrors the offline merge rule exactly.** Offline treats a
gap of `floor(min_silence_ms / hop_ms)` frames or fewer as an intra-utterance
pause, so online endpoints at one frame more. Keeping these consistent is what
makes the two modules comparable.

One deliberate difference: offline computes a final partial frame from whatever
samples remain; online only scores a frame once its samples have actually
arrived. Frame alignment is identical while the stream runs, and the two can
differ by at most one frame at the very end. `flush()` closes an utterance left
open when the stream ends — a speaker who stops and immediately disconnects
never produces trailing silence, and without `flush()` their last utterance
would be dropped.

---

## 2. The streaming session

```python
from asr.streaming import StreamingSession, StreamingConfig

session = StreamingSession(runner, StreamingConfig(language="hi"))

for block in microphone:                 # float32, 16 kHz, any length
    for update in session.push(block):
        if update.is_final:
            handle(update.text)
        else:
            show_provisional(update.text)

for update in session.flush():           # close an open utterance
    handle(update.text)
```

`push()` returns a list, not a generator, on purpose: the call is synchronous
and its side effects are ordered, so a caller who stops iterating early cannot
leave the session half-advanced.

### States and updates

`SessionState`: `idle` → `speaking` → `idle` … → `closed`.

Every update is a frozen dataclass with `as_dict()`, so a whole session
serializes to JSONL for debugging:

| Field | Meaning |
| --- | --- |
| `kind` | `state`, `partial`, `final` |
| `sequence` | Monotonic within the session |
| `text`, `is_final` | Only `is_final=True` is committed |
| `endpoint_reason` | `silence`, `max_duration`, `stream_flush` |
| `audio_seconds` | Audio actually transcribed for this update |
| `asr_ms`, `real_time_factor` | Timing for this update |
| `long_form` | Final routed through chunk-and-stitch |
| `partial_is_tail` | Partial covered only the tail of a longer utterance |

### Partials are approximate by construction

A partial is decoded from incomplete audio, so the model lacks the right
context that fixes word boundaries and case endings. **Partials will change as
more audio arrives.** They exist for perceived latency — something at 300 ms
instead of nothing until 2 s — and must never be acted on.

> **LLM analogy.** A partial is a mid-generation preview of a beam that has not
> converged. Useful to display, wrong to act on.

Cost note: Whisper's encoder is non-causal, so there is no incremental encode
to reuse. Every partial is a **full forward pass** over its window. That is
exactly why the rate limits exist.

### Two rate limits, because one is not enough

Both must clear before a partial runs:

- **`min_partial_audio_ms`** — how much *new audio* justifies another pass.
  This stops 20 ms microphone buffers causing 50 decodes a second.
- **`partial_interval_ms`** — how much *wall-clock* must pass. This matters
  when audio arrives faster than real time (replaying a file, catching up after
  a stall), where the audio test alone would fire constantly.

They fail in different situations and neither covers the other. Set
`partial_interval_ms=0` or `emit_partials=False` to disable partials entirely.

### Memory and long utterances

The buffer is trimmed as utterances finalize, so memory is bounded by
`max_utterance_seconds` rather than session length. A speaker who never pauses
hits `max_utterance_seconds` and gets a forced cut with
`endpoint_reason="max_duration"` — bounded memory is worth an occasional
awkward split.

Utterances over `long_form_threshold_seconds` route their **final** through the
existing `transcribe_long_array` chunk-and-stitch path. Partials always decode
only the last `partial_window_seconds` (default 25 s, under Whisper's 30 s
limit) so a partial is always a single encoder pass, and set
`partial_is_tail=True` when the utterance outgrew that window.

---

## 3. Agent output: playback and barge-in

### Barge-in is a lifecycle problem, not a TTS problem

The user can start talking while the agent is still speaking. Handling that is
concurrent: the cancel signal arrives on the audio-input thread while the
playback loop runs on another. `PlaybackSession` owns that state machine and
exposes a thread-safe `cancel()`.

```
IDLE → STARTING → PLAYING → {COMPLETED, CANCELLED, FAILED}
```

Terminal states are terminal. A cancel arriving after completion is recorded as
`ignored_cancel` and does not rewrite history — in a race, "did the user
interrupt or did we finish first?" has one correct answer, and turn accounting
depends on it.

### What cancellation can and cannot do

Cancelling **stops feeding new audio** to the sink, which stops synthesis and
stops queueing. It does **not** silence audio the operating system has already
accepted; that plays until its buffer drains, typically tens to low hundreds of
milliseconds.

Truly instant barge-in requires the sink to drop its own buffered audio, which
is why `AudioSink` separates `stop()` ("abandon what is queued, we were
interrupted") from `close()` ("finished normally"). Whether `stop()` is actually
instantaneous is a property of your sink, not of this module. Smaller chunks
give tighter cancellation granularity.

`cancel()` is a `threading.Event`, not a bool — a plain flag would be a data
race, and the memory barrier is what makes the cancel visible promptly.

---

## 4. TTS: what streams and what does not

Three things get called "streaming TTS" and only two are true here.

| Claim | Status |
| --- | --- |
| Chunk streaming from provider | **Real.** `edge_tts.Communicate.stream()` yields MP3 frames over a websocket as they are produced; `EdgeStreamingSynthesizer` forwards them instead of buffering. |
| Sentence-level streaming | **Real, and the bigger win.** The first sentence is synthesised and played while later ones are still being produced. |
| Low-latency local synthesis | **Not this.** edge-tts is a network service; a round trip to Microsoft dominates first-chunk latency and varies with connectivity. |

Measure `first_chunk_ms` on your own network before quoting a number. A local
TTS engine is the fix, not a wrapper.

`split_sentences` is deliberately simple — no abbreviation model, no learned
segmenter. A wrong split costs slightly odd prosody at one boundary; an
over-engineered splitter costs latency on every turn. `MIN_SENTENCE_CHARS` is
tuned for Devanagari, which is far denser per character than Latin: a complete
Hindi sentence like `मैं ठीक हूँ।` is only 12 characters, so an English-tuned
threshold merges a whole short reply into one unit and silently disables
sentence-level streaming.

Synthesis is driven **lazily by playback**, so a barge-in during the first
sentence stops the remaining sentences being synthesised at all, rather than
generating audio and discarding it.

---

## 5. The four latencies

Perceived responsiveness is one number — silence between the user finishing and
the agent starting — made of four segments. Splitting them is how you know
which component to fix.

| Field | Measures | Dominated by |
| --- | --- | --- |
| `speech_end_to_final_transcript_ms` | VAD endpoint → committed transcript | **`min_silence_ms`**, then ASR |
| `final_transcript_to_first_llm_token_ms` | Prompt build + prefill | Prompt length |
| `first_llm_token_to_playback_start_ms` | First token → first audio out | TTS round trip |
| `total_turn_ms` | End to end | — |

**The first one includes the endpointer's own silence threshold.** That is dead
time by construction: you cannot know the user stopped until they have been
quiet a while. Halving `min_silence_ms` halves that segment and doubles the rate
of cutting people off. It is a tuning decision, not a bug, and it should be
visible in the number.

### Measurement honesty

`LLMRunner.generate()` is **not** a streaming generator — it returns the whole
response, and from outside there is no way to observe when the first token
appeared. Two cases, and the turn records which applied:

- Backend exposes `stream()` → first-token time measured directly at the first
  yielded piece. `llm_streaming=True`.
- Backend does not → falls back to the runner's internally measured
  `metrics.prefill_ms`, since prefill completion is when the first token
  exists. `first_token_is_prefill_proxy=True` is set so nobody later reads it
  as a wall-clock measurement. It excludes Python-side overhead between
  `generate()` returning and the caller seeing it.

**Any field that could not be measured is `None`, never `0.0`.** A zero gets
averaged into a benchmark; a `None` forces the question.

---

## 6. Running the demo

```bash
python scripts/streaming_demo.py --seconds 12
```

Simulated microphone, fake ASR/LLM/TTS, no GPU and no network. Prints every
update and the full latency breakdown. Use `--json` for machine-readable output.

To drive real audio through it, replace the fake transcriber with a real
`ASRRunner` — the session only needs `transcribe_array` and
`transcribe_long_array`:

```python
from asr.explicit import load_whisper, ASRRunner
from asr.streaming import StreamingSession, StreamingConfig

loaded = load_whisper("openai/whisper-medium", adapter_path="path/to/adapter")
runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
session = StreamingSession(runner, StreamingConfig(language="hi"))
```

---

## 7. What is real and what is a stand-in

| Component | Status |
| --- | --- |
| Online endpointing | Production logic; energy VAD baseline, same caveats as `asr/vad.py` |
| Session state machine, buffering, rate limiting | Production logic |
| Long-form routing | Production; uses the existing chunk-and-stitch path |
| Playback lifecycle and barge-in | Production logic; **needs a real `AudioSink`** — `BufferSink` writes to memory |
| Sentence splitting | Production, intentionally heuristic |
| edge-tts chunk streaming | Real streaming, but a **network service** |
| LLM first-token timing | **Proxy** unless the backend exposes `stream()` |
| `scripts/streaming_demo.py` | Entirely simulated — fakes throughout |

The gaps that matter most, in order: a real `AudioSink` for actual audio output,
a streaming LLM backend to replace the prefill proxy, and a local TTS engine to
remove the network round trip from the critical path.

### Validation status

Everything in this document is exercised against real models by
`scripts/gpu_validation.py` (Kaggle notebook:
`notebooks/gpu_validation_kaggle.ipynb`). Until `results/gpu_validation/report.json`
is committed with a passing sweep, the "production logic" rows above mean
"unit-tested against fakes", not "observed working on a GPU".
