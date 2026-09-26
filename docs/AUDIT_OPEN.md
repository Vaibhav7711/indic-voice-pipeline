# Open audit findings

From a 112-agent audit run on 2026-09-24 (commit `ac33def`). Every finding
below survived three independent adversarial lenses (code-reality, impact,
already-handled); 3 further findings were rejected by that verification and
are not listed.

**Fixed already** (commits `b4adcc3`, `5fc7683`, `593e97b`, `31a842f`): the
no-speech token id and read position, long-form guard aggregation, the
`cancel()` lock race, `SentenceBuffer.flush()` duplication, the Devanagari
mid-word clause cut, the inert incremental-finals path and its benchmark
grid, `bench_all` resume/mirror, the fabricated `0.0` turn latency, and
absent-vs-zero flag reporting.

What follows is what remains. Severity is the auditor's; `location` is
`file:line` at the audit commit, so line numbers may have drifted.

## Remaining: 16 confirmed + 6 from the completeness critique

**Resolved since the audit** (this list is the audit's, not a live tracker, so
the findings below are left in place with their original text and annotated):

- *README waterfall table matches no value in its cited evidence* — **fixed.**
  The table now carries the committed figures (14.4 / 80.4 / 1900.2 / 105.0 /
  2323.6 ms, total 4514.5, audio→first token 2159.3, peak 2.6 GiB) and cites
  commit `21ebce7`.
- *Shipped v2 metrics cite evidence directories that do not exist* — **fixed**
  by correcting the numbers rather than the citation. `results/eval/v2-guards-on/`
  is committed and holds the run; three of the four published figures did not
  match it (CER 8.43 → 8.46%, p50 694 → 766 ms, RTF 0.066 → 0.073) and the
  README's "3.7× faster" became 3.2×. The turbo-base row still has no committed
  evidence and is now marked as such.
- *A TTS failure records the unspoken response as heard* — **fixed** in
  `agent/turn.py` (a synthesis error with zero chunks written is a FAILED turn)
  and independently in `Conversation.record_turn` (a result that wrote no bytes
  is refused whatever its state). `tests/test_turn_honesty.py`.
- *MIT claimed with no LICENSE file* — **LICENSE added.** The blank model card
  on the shipped weights remains open, and is part of the unobtainable-weights
  blocker.


## doc_accuracy

### BLOCKER — README waterfall table matches no value in its cited evidence

**Where:** `README.md:449`

**Claim.** The "Full pipeline waterfall" table names `results/gpu_validation/report.json` and commit `ee2b7c3` as its evidence, but not one of its eight numbers appears in that file, and the file records a different commit. The committed `pipeline_waterfall` check holds mel 14.377 ms, encoder 80.382 ms, asr_decode 1900.217 ms, llm_prefill 105.015 ms, llm_decode 2323.558 ms, total 4514.515 ms, audio_to_first_llm_token 2159.303 ms, peak_vram 2.605 GiB, under `git_commit: 21ebce752fa78f8c6dc20cb58b115d21aba449be`. Every published figure understates the measured one (total by 630 ms, peak VRAM by 0.3 GiB). This is the project's headline end-to-end latency claim, and the adjacent tables (ASR per-utterance latency, streaming WER) do match their evidence exactly, so this table is not a stale-units artifact — it is a different, unrecorded run.

**Evidence.** README.md:449-462 — "From `scripts/gpu_validation.py` (`pipeline_waterfall`, mean of 3 runs after / warm-up, one ~10 s FLEURS clip, 48-token LLM budget, T4, commit `ee2b7c3`). / Evidence: `results/gpu_validation/report.json`." then "| Mel extraction | 7.7 ms |", "| Whisper encoder | 74.8 ms |", "| ASR decode | 1711 ms |", "| LLM decode (~47 tokens @ ~41 ms) | 1922 ms |", "| **Total pipeline** | **3885 ms** |", "| Peak VRAM | 2.9 GiB (concurrent strategy) |". results/gpu_validation/report.json — {"name": "pipeline_waterfall", "status": "pass", "detail": {"strategy": "concurrent", "mel_ms": 14.377256666666668, "encoder_ms": 80.38227081298828, "asr_decode_ms": 1900.2168083190918, "llm_prefill_ms": 105.0146967569987, "llm_decode_ms": 2323.5581843058267, "total_ms": 4514.515189666666, "audio_to_first_llm_token_ms": 2159.303421423665, "peak_vram_gib": 2.60540771484375}} and "git_commit": "21ebce752fa78f8c6dc20cb58b115d21aba449be"

**Fix.** Replace the table with the values actually in `results/gpu_validation/report.json` (14.4 / 80.4 / 1900.2 / 105.0 / 2323.6 ms, total 4514.5 ms, audio→first token 2159.3 ms, peak 2.6 GiB) and correct the commit to `21ebce7`. If the 3885 ms run is real, commit its report and cite that file instead; do not leave a number whose named evidence contradicts it.

### BLOCKER — Shipped v2 metrics cite evidence directories that do not exist

**Where:** `docs/EXPERIMENTS.md:296`

**Claim.** The shipped adapter's headline numbers (23.83% WER, 8.43% CER, 694 ms p50, RTF 0.066) have no evidence file anywhere in the repo. EXPERIMENTS.md cites `results/eval/turbo-lora-v2-test-300-seed0-full/` and `results/eval/turbo-base-test-300-seed0/`; neither path exists on disk or in git. README points the whole results section at `results/eval/medium-base-test-300-seed0/` and `results/eval/medium-lora-test-300-seed0/`, which exist but contain only the medium/v1 numbers (40.427% and 25.8246%) — not the turbo rows in the same table. `results/eval/` is gitignored, so a future turbo run will not be committed without `git add -f`. The ledger's own protocol requires this: "Every number in this ledger must come from that harness, with the emitted `run_config.json` retained next to it."

**Evidence.** docs/EXPERIMENTS.md:295-296 — "private Hub repo. Evidence: / `results/eval/turbo-lora-v2-test-300-seed0-full/` (seed 0, 300 clips,"; :258 — "Evidence: `results/eval/turbo-base-test-300-seed0/` (run from Colab at". README.md:412-413 — "Tesla T4, fp16, LoRA merged, explicit `ASRRunner`). Evidence: / `results/eval/medium-base-test-300-seed0/`, `results/eval/medium-lora-test-300-seed0/`." followed by "| **turbo + LoRA v2 (shipped)** | **23.83%** | **8.43%** | **694 ms** | **0.066** |". `ls results/eval/` → compare-medium-base-vs-lora.json, medium-base-test-300-seed0, medium-lora-test-300-seed0. .gitignore — "results/eval/"

**Fix.** Either commit the two turbo run directories with `git add -f` (they are the only evidence for the shipped model), or mark both turbo rows as uncommitted-evidence the way EXPERIMENTS.md:258 already does ("to be committed with the next evidence drop") and stop citing the medium v1 directories as evidence for them in README.md:412.

### MAJOR — TECHNICAL.md lists four shipped capabilities as missing

**Where:** `docs/TECHNICAL.md:803`

**Claim.** Section 10 "Known limitations and future work" states four things the repo now does as things it does not do, and three of them reappear under "Potential extensions". Long-form chunking is implemented (`asr/explicit/chunking.py`, `ASRRunner.transcribe_long_file` at runner.py:216 and `transcribe_long_array` at :233) and is described in TECHNICAL.md's own section 4 (lines 201-205, "Splits arbitrary-length audio into 25 s windows with 5 s overlap"). Sentence-level TTS streaming concurrent with LLM decode is implemented (`agent/turn.py`, `tts/streaming.py`) and is the subject of STREAMING.md section 5 "The turn is a pipeline". Whisper-medium LoRA shipped as v1 and was superseded by turbo v2. The Hindi correctness test exists: `served_model_matches_generate` is token-identical on three real Hindi clips in results/gpu_validation/report.json, and README's correctness policy claims it ("token-identical; verified on real Hindi audio"). The "WER still 37.7%" figure is the whisper-small number EXPERIMENTS.md forbids citing.

**Evidence.** docs/TECHNICAL.md:803-819 — "2. **Audio length**: Whisper processes max 30 seconds per chunk. Longer / audio requires chunking with overlap and stitching, which this pipeline / doesn't implement." … "4. **No streaming**: The LLM generates all tokens before TTS starts." … "5. **WER still 37.7%**: Roughly 1 in 3 Hindi words is wrong." … "6. **No correctness test for Hindi**: The ASR correctness test uses a / synthetic tone, not real Hindi speech."; :825-827 — "3. Implement 30-second chunking for longer audio / 4. Stream TTS synthesis concurrent with LLM generation / 5. Add Whisper-medium with LoRA for better WER". docs/EXPERIMENTS.md:122-127 — "`results/wer_baseline.json` and `results/wer_finetuned.json` (75.80% and 37.66%) … Do not cite them alongside new numbers."

**Fix.** Rewrite section 10 against the current tree: delete limitations 2, 4 and 6 and extensions 3, 4, 5, replace "WER still 37.7%" with the shipped 23.83%, and state the limitations that are actually open (empty `data/hard_set/`, Qwen3-0.6B Hindi quality, no live-turn distribution, device barge-in laptop-only). Also fix the same superseded numbers at lines 38, 705-706 and 777-778, and drop the `setup.py` walkthrough at line 547 — the project uses pyproject.toml and has no setup.py.

### MAJOR — Documented streaming_eval command exits on unknown VAD config names

**Where:** `docs/STREAMING.md:487`

**Claim.** The command STREAMING.md gives for reproducing the streaming numbers fails immediately: `pad300` and `floor70` were removed from `VAD_GRID`, and `vad_from_name` raises `SystemExit` on an unknown key. Verified by running it: `vad_from_name('pad300')` → "unknown VAD config 'pad300'; choose from ['default', 'fixed40', 'floor60', 'floor80', 'margin12', 'margin8', 'pad200', 'pad500', 'sil400', 'sil800', 'v1-adaptive']". The same dead names are in benchmarks/streaming_eval.py's own usage docstring (`--grid default,pad300,floor70,pad300-floor70`, all three invalid), and EXPERIMENTS.md's streaming table and the committed result directories are keyed by them. Worse, `default` no longer means what the committed run recorded: run_config.json captured `default` as padding_ms 200 / threshold_floor_dbfs -60, while today's VADConfig defaults are 300 / -70, so re-running the documented command reproduces neither the old rows nor a named equivalent of the current defaults.

**Evidence.** docs/STREAMING.md:484-488 — "python -m benchmarks.streaming_eval \\ / --adapter Hugme6969/whisper-medium-hindi-lora \\ / --split test --limit 100 --seed 0 \\ / --grid default,fixed40,pad300,floor70 \\ / --out-dir results/streaming_eval/medium-lora-test-100". benchmarks/streaming_eval.py:57-70 — VAD_GRID keys are only "default", "fixed40", "v1-adaptive", "pad200", "pad500", "floor60", "floor80", "sil400", "sil800", "margin8", "margin12"; :103-104 — "if name not in VAD_GRID: raise SystemExit(f\"unknown VAD config {name!r}; choose from {sorted(VAD_GRID)}\")". results/streaming_eval/medium-lora-test-100/run_config.json — "default": {… "threshold_floor_dbfs": -60.0, "padding_ms": 200}

**Fix.** Update the command to names that exist — `--grid default,fixed40,v1-adaptive,pad200,floor60` reproduces the same four points against today's defaults — and fix the same command in benchmarks/streaming_eval.py:34. Add a note that the committed run's `default` is now `v1-adaptive`, so the old rows are reproducible under that key.

### MAJOR — TECHNICAL.md documents max_new_tokens=225, the bug that was fixed

**Where:** `docs/TECHNICAL.md:373`

**Claim.** TECHNICAL.md still documents 225 as the runner's decode budget, which is exactly the defect EXPERIMENTS.md records as fixed and warns invalidates prior Hindi numbers. The code default is `max_new_tokens: int | None = None` and `token_budget()` derives the limit from the model: `limit = self.max_target_positions - prompt_len` (448 minus the prompt, so 444). A reader trusting the doc would believe long Hindi utterances are still capped at ~37 words, and the KV-cache arithmetic at line 618 ("At max tokens (225): 36 KB × 225 ≈ 8.1 MB") understates self-attention cache by roughly half.

**Evidence.** docs/TECHNICAL.md:373-374 — "**`max_new_tokens=225` default**: Whisper's 30-second chunk produces at most / ~225 text tokens (empirically ~7-8 tokens per second of audio)."; :191 — "│   Decode: token-by-token until EOS or 225 max". asr/explicit/runner.py:142 — "max_new_tokens: int | None = None,"; :291-299 — "def token_budget(self, language: str | None, requested: int | None) -> int: … limit = self.max_target_positions - prompt_len / wanted = requested if requested is not None else self.default_max_new_tokens / return limit if wanted is None else max(1, min(int(wanted), limit))". docs/EXPERIMENTS.md:317-327 — "`max_new_tokens=225` (hardcoded in the runner, the eval harness, the streaming config and the pipeline) cut every utterance over ~37 words … Fixed in `55f5c1e`: the budget is derived from the model (`max_target_positions - prompt`, so 444)"

**Fix.** Replace the 225 default with the derived budget (`max_target_positions - prompt_len`, 444 for a 448-position decoder), update line 191 and the line 618 KV math accordingly, and cross-reference the EXPERIMENTS.md section on the 225-token bug so the doc cannot be mistaken for a description of the pre-fix runner.

### MAJOR — Agent brief cites report.json for a latency it does not contain

**Where:** `docs/AGENT_BRIEF.md:30`

**Claim.** The handoff brief's Current-state table attributes "one live turn at 1.57 s response latency" to `results/gpu_validation/report.json`, but the only voice turn in that file records `response_latency_ms: 719.8217661264353` (with `first_token_is_prefill_proxy: true`); 1.57 s appears nowhere in the file, and the only other place in the repo that mentions 1.57 is this line. README's own account of that field says the honest figure for the first sweep's turn was ~2.2 s and marks it "To be re-measured". Separately, Phase 1 attributes the decode guards to commit `55f5c1e`, which is the token-budget fix; the guards were added in `df6f5b2` ("fix(asr): guard the greedy decode against no-speech and repetition loops", the commit that introduced `no_speech_threshold` into asr/explicit/runner.py per `git log -S`). Both errors sit in the document the brief's own Hard rule 1 governs: "If you cannot name the file a claim comes from, do not make the claim."

**Evidence.** docs/AGENT_BRIEF.md:30 — "| Agent turn (LLM stream → sentences → TTS → playback, barge-in) | logic validated; one live turn at 1.57 s response latency | `results/gpu_validation/report.json` |"; :124 — "The decode guards in commit `55f5c1e` (no-speech threshold 0.6, repetition / loop guard) changed serving behaviour". results/gpu_validation/report.json, voice_turn_real_tts detail — "response_latency_ms": 719.8217661264353, "first_token_is_prefill_proxy": true. `git log --oneline -S "no_speech_threshold" -- asr/explicit/runner.py` → "df6f5b2 fix(asr): guard the greedy decode against no-speech and repetition loops"; `git log --oneline -1 55f5c1e` → "fix: derive the decode token budget from the model; 225 truncated long Hindi"

**Fix.** Change the row to the recorded figure — "one turn at 720 ms response latency (first token is a prefill proxy; no distribution)" — or drop the number and keep only "logic validated". Change `55f5c1e` to `df6f5b2` in Phase 1. While there, fix STREAMING.md:496, which says "15/15 checks pass" of the same report: it records 14 pass and 1 warn.


## asr_runtime

### MINOR — language_probability is renormalized over the candidate subset

**Where:** `asr/explicit/decoder.py:198`

**Claim.** `detect_language` softmaxes over only the selected language logits, and when `candidates` is passed the table has already been reduced to that subset (line 181), so the returned probability is P(lang | one of the candidates), not P(lang). Whisper's reference masks every *non-language* token to -inf and normalizes across all ~99 language tokens; restricting the denominator to 2-3 codes makes the number approach 1.0 for any input, including audio in none of the candidate languages. The docstring claims it is "the same procedure Whisper's reference implementation uses". Both interactive callers always restrict candidates (demo/app.py:175 `language_candidates=LANGUAGES`, demo/notebook.py:221 `["hi", "en", "te"]`), and the value is then surfaced as a confidence in demo/app.py:98 (`f"language detected at p={asr_metrics.language_probability:.2f}"`) and recorded as evidence via `metrics.language_probability` (runner.py:328).

**Evidence.**         codes = list(table)
        ids = torch.tensor([table[c] for c in codes], device=logits.device)
        probs = torch.softmax(logits[ids], dim=-1)
        best = int(probs.argmax().item())

**Fix.** Keep the argmax restricted but normalize over the full language table: build `full = self._language_token_ids()`, compute `probs` over `full`'s ids, then pick `best` among the candidate codes and report that entry's probability. The selected code stays the same while the reported number becomes comparable to the reference and to unrestricted runs.

### MINOR — Offline VAD deletes short voiced runs before the silence merge

**Where:** `asr/vad.py:164`

**Claim.** `detect_speech` applies the `min_speech_ms` filter to raw voiced runs and only merges intra-utterance gaps afterwards, so a voiced run shorter than 250 ms is discarded even when it sits inside an utterance the merge would have joined. The online endpointer requires consecutive frames only at *onset*; once in SPEECH any voiced frame extends the utterance (endpointer.py:257-260, `if voiced: self._silence_run = 0; self._last_voiced_frame = index`). With defaults (hop 10 ms, min_speech 25 frames, min_silence 60 frames, padding 300 ms), a 500 ms run, a 400 ms gap and a 100 ms trailing word give offline a segment ending at sample 13120 (0.82 s) while the word starts at 14400 (0.90 s) — offline drops it, online keeps it. That contradicts both equivalence claims (vad.py:35 "offline and online produce identical decisions", endpointer.py:42 "can differ by at most one frame"), and no test covers it: tests/test_vad.py:94 and tests/test_streaming.py:236,251 only use runs well above min_speech_ms. Short Hindi backchannels (हाँ, जी, ना) are exactly the runs this deletes.

**Evidence.**     min_speech_frames = max(1, int(np.ceil(config.min_speech_ms / config.hop_ms)))
    speech_runs = [(start, end) for start, end in _runs(voiced) if end - start >= min_speech_frames]
    if not speech_runs:
        return []

    # A short silence is an intra-utterance pause, not an endpoint.
    max_gap_frames = int(np.floor(config.min_silence_ms / config.hop_ms))

**Fix.** Reorder: merge gaps of `max_gap_frames` or fewer over all raw runs from `_runs(voiced)` first, then drop merged segments whose total voiced length is below `min_speech_frames`. That keeps the cough rejection (an isolated short run still fails the filter) while matching the online rule, and add a regression test with a 500 ms / 400 ms gap / 100 ms geometry asserting offline and online agree within one frame.


## streaming_agent

### MAJOR — A TTS failure records the unspoken response as heard

**Where:** `agent/turn.py:426`

**Claim.** `run()` classifies a turn where synthesis produced nothing as COMPLETED: `synth()` swallows the backend exception into `errors["tts"]`, so `errors` is non-empty but `playback_result.state` is COMPLETED (playback consumed an empty iterable) and `interrupted` is False. The turn then calls `conversation.record_turn(result)`, and `Conversation.record_turn` only rejects state == "failed", so `spoken_text` takes the `state != "interrupted"` branch and returns the full LLM response. Reproduced with a synthesizer that raises: `bytes_written=0, chunks_written=0`, state COMPLETED, `recorded_in_history=True`, and the history holds the complete assistant sentence the user never heard. Every later turn is prompted as if the agent had answered. This contradicts agent/conversation.py's stated contract — '**Nothing invented.** An empty or failed turn is not added to history at all' and 'a turn records what was *spoken*'. `tests/test_agent.py::test_tts_failure_is_surfaced` only asserts `result.speech.error is not None`; it checks neither the state nor the history.

**Evidence.**         elif playback_result.interrupted:
            self.state = TurnState.INTERRUPTED
        else:
            self.state = TurnState.COMPLETED
            if errors:
                result.error = "; ".join(f"{k}: {v}" for k, v in errors.items())

**Fix.** Treat 'no audio reached the sink' as a failed turn: in the classification chain add `elif "tts" in errors and playback_result.chunks_written == 0: self.state = TurnState.FAILED`. Alternatively make `Conversation.record_turn` reject a result whose `playback.bytes_written == 0`, and make `spoken_text` fall back to `speech.sentences` whenever synthesis errored, so only audio that actually played enters the history.


## metric_honesty

### MAJOR — compare.py never checks decode config, only the audio selection

**Where:** `benchmarks/compare.py:68`

**Claim.** `_selection_problems` compares only `source`, `config`, `split`, `manifest`, the literal `indices`, and `reporting_level`. It never reads `max_new_tokens`, `decode_guards`, or `dtype` — all recorded in `run_config.json` for exactly this purpose. I built two run dirs identical in selection but differing in `max_new_tokens` (225 vs 445), `no_speech_threshold` (0.6 vs 0.2), `loop_guard_ngram` (3 vs 0) and `dtype` (float16 vs float32); `compare_runs` returned `comparable: True` with `selection_warnings: []` and printed a clean -15.0 point WER win. The 225-vs-445 case is not hypothetical: both shipped runs used 225, and the project's own plan is to re-run at the model's real limit — the next comparison will silently mix a truncating budget against a non-truncating one, and truncation shows up as deletions, i.e. as WER.

**Evidence.** benchmarks/compare.py:68:
    for key in ("config", "split", "manifest"):
        if a.get(key) != b.get(key):

# asr_eval.py, where those fields are written, states the guarantee this misses:
# "Decode guards are part of the serving configuration, so they are set
#  here and recorded in run_config.json: changing one and not recording it
#  would make two runs incomparable without anyone noticing."

# observed with mismatched configs, identical indices:
#   comparable: True
#   warnings: []
#   wer: {'before': 40.0, 'after': 25.0, 'points': -15.0, 'relative_percent': 37.5}

**Fix.** In `_selection_problems`, compare the decode configuration too — `max_new_tokens`, `decode_guards` (as a whole dict), `dtype`, `language`, and `model` — appending a problem for each difference, e.g. `for key in ("max_new_tokens", "dtype", "language"): ...` plus an explicit `if a.get("decode_guards") != b.get("decode_guards")`. Note these live in run_config, which `compare_runs` already merges into the dicts it passes in, so no extra loading is needed.

### MAJOR — Empty streaming clips contribute 0.0 to latency means

**Where:** `benchmarks/streaming_eval.py:202`

**Claim.** When a config's endpointer produces no final for a clip, `stream_clip` returns `sum(...)` over an empty `finals` list for `streamed_asr_ms`, `asr_ms_after_endpoint`, `asr_ms_total` and `decoded_seconds` — i.e. 0.0, meaning "nothing was measured" — while the sibling `endpoint_to_final_ms_last` on line 195 correctly returns `None`. `aggregate` then averages them: line 257 does no filtering at all, and `_mean_of` filters only `None`. So a VAD config that drops clips entirely reports a *faster* ASR. This is visible in the shipped run: `fixed40` has `clips_empty: 3` and reports `streamed_asr_ms_mean: 2556.2`, whereas the mean over the 97 clips that actually produced output is 2635.2 — the config that loses 3% of the audio looks 79 ms faster than `default` partly because of the zeros, and `clips_empty` sits in a different block of the JSON from the latency it distorts.

**Evidence.** benchmarks/streaming_eval.py:195-202:
        "endpoint_to_final_ms_last": round(endpoint_to_final_ms[-1], 1) if finals else None,
        ...
        "decoded_seconds": round(sum(f.decoded_seconds for f in finals), 3),
        ...
        "streamed_asr_ms": round(sum(f.asr_ms for f in finals), 1),

benchmarks/streaming_eval.py:257:
            "streamed_asr_ms_mean": round(float(np.mean([r["streamed_asr_ms"] for r in rows])), 1),

benchmarks/streaming_eval.py:272:
def _mean_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]

# results/streaming_eval/medium-lora-test-100/fixed40/clips.jsonl:
#   3 rows with n_finals == 0 and 'streamed_asr_ms': 0
#   mean all 2556.2 | mean over clips with output 2635.2

**Fix.** Guard all four on `finals`, mirroring line 195: `"streamed_asr_ms": round(sum(...), 1) if finals else None`, and likewise for `asr_ms_after_endpoint`, `asr_ms_total`, `decoded_seconds`. Then route line 257 through `_mean_of(rows, "streamed_asr_ms")` so it skips the Nones, and add the count of skipped clips next to each latency mean so the reader sees the denominator.

### MINOR — tts_first_chunk_ms contains LLM generation time

**Where:** `agent/turn.py:402`

**Claim.** `TurnMetrics.tts_first_chunk_ms` is copied from `speech.first_chunk_ms`, which `iter_sentence` measures from `start_ns=synth_start_ns` — set at turn.py:354, before `playback.play(produce())` and therefore before the LLM has run. So a field named for TTS carries prompt build + prefill + first-sentence decode + synthesis, and `demo/app.py:87` renders it to the user as the row "TTS first chunk". The shipped report shows the two are the same quantity: `tts_first_chunk_ms: 619.744` against `first_llm_token_to_playback_start_ms: 619.86`. Anyone comparing this against `tts_bakeoff`'s `first_chunk_ms` — measured from a `t0` taken immediately before `synth.stream()` — will attribute the LLM's time to the synthesizer. `SpeechStream` already carries the honest split (`first_unit_queued_ms`, `synthesis_ms`); the turn just exports the unsplit number under a TTS name.

**Evidence.** agent/turn.py:354:
        synth_start_ns = perf_counter_ns()
agent/turn.py:397:
        playback_result = playback.play(produce())     # <- the LLM runs in here
agent/turn.py:402:
        metrics.tts_first_chunk_ms = speech.first_chunk_ms

tts/streaming.py:399-402:
        elapsed = (perf_counter_ns() - start_ns) / 1_000_000
        ...
        if stream.first_chunk_ms is None:
            stream.first_chunk_ms = elapsed

demo/app.py:87:
        ("TTS first chunk", _ms(getattr(turn_metrics, "tts_first_chunk_ms", None))),

**Fix.** Either rename the field to `turn_start_to_first_audio_chunk_ms`, or assign the synthesizer's own cost instead: `metrics.tts_first_chunk_ms = speech.synthesis_ms` (already defined as `first_chunk_ms - first_unit_queued_ms`). Update the `demo/app.py:87` label and `benchmarks/tts_bakeoff.py`'s docstring claim that the turn "inherits" `first_chunk_ms` so the two numbers are the same quantity.


## unfinished

### MAJOR — served_model_matches_generate passes on empty or truncated output

**Where:** `scripts/gpu_validation.py:213`

**Claim.** The check that backs README's "Explicit ASR decode | HF `model.generate()` greedy tokens (token-identical; verified on real Hindi audio)" only compares the common prefix and scores `matches / n if n else 1.0`, then gates on `worst < 0.9`. If the explicit loop emits nothing — exactly what the new no-speech guard does when it fires — `n` is 0, `prefix_match` is 1.0 and the check reports PASS. A transcript truncated by the repetition guard or the token budget also scores 1.0 because it is a prefix of generate()'s output. `exact` is recorded in the detail but never gated, so the harness cannot distinguish "token-identical" from "emitted a strict prefix" or "emitted nothing".

**Evidence.** scripts/gpu_validation.py:211-220 — `n = min(len(ours), len(ref_out))
        matches = sum(a == b for a, b in zip(ours[:n], ref_out[:n], strict=True))
        per_clip.append({
            "clip": path.name, "explicit_tokens": len(ours),
            "generate_tokens": len(ref_out), "prefix_match": matches / n if n else 1.0,
            "exact": ours == ref_out, ...
    worst = min(c["prefix_match"] for c in per_clip)
    if worst < 0.9:
        raise AssertionError(...)`

**Fix.** Fail when `ours` is empty or when `len(ours) != len(ref_out)`, and gate on `all_exact` (fall back to the prefix ratio only as a warning). Also build the comparison runner with the guards disabled (`no_speech_threshold=None, loop_guard_ngram=0`) so the identity check tests the decode loop rather than the serving guards, and check the guards separately.

### MINOR — Two divergent system prompts despite "one builder for every path"

**Where:** `agent/turn.py:240`

**Claim.** `llm/prompting.py` exists because "The agent turn and the batch pipeline used to build prompts separately", yet only the *builder* was unified. `pipeline/orchestrator.py` and `benchmarks/llm_bakeoff.py` use `system_prompt_for(language)` / `SYSTEM_PROMPTS["hi"]`, while `VoiceTurn.build_prompt` hardcodes a different, shorter prompt that no live call site overrides (`system_prompt=` appears only in tests/test_agent.py:451). The LLM bake-off that decides which model to ship therefore measures prefill and token counts against a prompt the agent never sends, and prompt fixes such as "keep unavoidable proper nouns and technical terms as-is" reach the batch pipeline only. Relatedly, `response_language` defaults to "hi", which renders "Reply entirely in natural hi", while all six real call sites pass "Hindi". tests/test_prompting.py's `test_pipeline_and_turn_use_the_same_builder` only asserts `turn.build_chat_prompt is build_chat_prompt`, so it does not guard the prompt text.

**Evidence.** agent/turn.py:239-244 — `system = self.system_prompt or (
            f"You are a helpful voice assistant. Reply entirely in natural "
            f"{self.response_language}, in one or two short sentences. Your "
            "reply is spoken aloud, so a long first sentence leaves the user "
            "waiting in silence. Never repeat the question back."
        )`. llm/prompting.py:1-6 — `"""One chat-prompt builder for every path that talks to the LLM.

The agent turn and the batch pipeline used to build prompts separately...`. benchmarks/llm_bakeoff.py:147 — `prompt = build_chat_prompt(loaded.tokenizer, system_prompt_for(language), prompt_text)`

**Fix.** Have `VoiceTurn.build_prompt` call `system_prompt_for(self.response_language)` (and change the parameter to a language code, since that is what the table is keyed by), keeping the voice-specific sentence-length instruction inside `SYSTEM_PROMPTS`. Extend the regression test to assert the rendered system prompt is identical for the turn, the pipeline and the bake-off.

### MINOR — Both demo entry points hide TurnResult.error on a completed turn

**Where:** `demo/app.py:236`

**Claim.** `VoiceTurn.run` deliberately completes a turn that produced text but whose TTS failed, or whose LLM died mid-response, recording the reason in `result.error` (tests/test_agent.py:635 `test_llm_error_after_some_output_is_reported_not_fatal` asserts state "completed" plus an error). Neither entry point reads that field when the state is COMPLETED: the Gradio app shows `result.error` only when `result.response` is empty, and the notebook warns only when `state != "completed"` and does not even keep `error` in the returned record. A user whose edge-tts call failed sees a normal answer with no audio and no explanation, and a logged notebook session loses the failure entirely.

**Evidence.** demo/app.py:236-238 — `return (asr.text, result.response or f"_{result.state.value}: {result.error or ''}_",
                latency_table(asr.metrics, result.metrics), audio_out,
                history_markdown(self.conversation))`. demo/notebook.py:300-301 — `if record_["state"] != "completed":
            lines.append(f"\n⚠️ turn ended **{record_['state']}**")`

**Fix.** Surface `result.error` (and `result.speech.error`) whenever it is set, regardless of state — append a warning row to `latency_table`/`_show` — and add `"error": result.error` to the notebook's `record_` dict so a logged session carries it.


## colab_readiness

### MAJOR — Documented --grid configs pad300/floor70 no longer exist

**Where:** `notebooks/gpu_validation_kaggle.ipynb:109`

**Claim.** The streaming benchmark command published in the notebook and in docs/STREAMING.md names two VAD configs that were deleted from `VAD_GRID`, so it exits before loading a model. `benchmarks/streaming_eval.py:57` now defines `default, fixed40, v1-adaptive, pad200, pad500, floor60, floor80, sil400, sil800, margin8, margin12` — `pad300` and `floor70` are gone, replaced by pad200/pad500 and floor60/floor80. `vad_from_name` raises `SystemExit(f"unknown VAD config {name!r}")`, so `python -m benchmarks.streaming_eval ... --grid default,fixed40,pad300,floor70 ...` dies immediately with `unknown VAD config 'pad300'`. This is not a stale example nobody ran: `results/streaming_eval/medium-lora-test-100/` contains committed `pad300/` and `floor70/` metrics, and docs/ONBOARDING_STATUS.md:66 reports their WER, so the rename silently orphaned the command that produced the project's streaming evidence — nobody can reproduce those two rows. The module's own usage docstring (line 34) is stale the same way. CI cannot catch it: `tests/test_streaming_eval.py:44` asserts only `set(VAD_GRID) >= {"default", "fixed40", "v1-adaptive", "pad500", "floor60"}`, a subset check that passes however many names are removed.

**Evidence.** notebooks/gpu_validation_kaggle.ipynb:109  "!python -m benchmarks.streaming_eval \\\n    --adapter Hugme6969/whisper-medium-hindi-lora \\\n    --split test --limit 100 --seed 0 \\\n    --grid default,fixed40,pad300,floor70 \\\n    --out-dir results/streaming_eval/medium-lora-test-100"
docs/STREAMING.md:487      --grid default,fixed40,pad300,floor70 \
benchmarks/streaming_eval.py:34      --grid default,pad300,floor70,pad300-floor70 \\
benchmarks/streaming_eval.py:104-106  def vad_from_name(name: str) -> VADConfig:\n    if name not in VAD_GRID:\n        raise SystemExit(f"unknown VAD config {name!r}; choose from {sorted(VAD_GRID)}")
Verified: sorted(VAD_GRID) == ['default','fixed40','floor60','floor80','margin12','margin8','pad200','pad500','sil400','sil800','v1-adaptive']  # pad300 MISSING, floor70 MISSING

**Fix.** Either re-add `pad300`/`floor70` as aliases (they are one-line dicts and they are what the committed evidence was produced with), or update the notebook cell, docs/STREAMING.md:487 and the docstring at line 34 to `--grid default,fixed40,pad200,pad500,floor60,floor80`. Then tighten `tests/test_streaming_eval.py:44` to assert every name that appears in a committed `results/streaming_eval/*/` subdirectory is still a `VAD_GRID` key, so the next rename fails CI.


## From the completeness critique

These are areas the six audit dimensions would not have examined.

### BLOCKER — Shipped v2 model has no obtainable weights and no reproduction command

**Where:** `docs/EXPERIMENTS.md:294`

**Claim.** Every headline number the project is about to be declared finished on (40.4% -> 23.8% WER, 8.43% CER, 694 ms p50, RTF 0.066) comes from an adapter that no reader can obtain, and no command anywhere in the repo reproduces it. The ledger locates the v2 weights at an ephemeral Kaggle path plus an unnamed 'private Hub repo'; the only place the repo names that repo is docs/AGENT_BRIEF.md:94, which computes it from the operator's own account (`whoami()['name']`), so the identifier is not even a constant. Meanwhile every runnable `--adapter` example in the repository names the *superseded v1 medium* adapter: README.md:275, :325, :389, docs/EXPERIMENTS.md:90, :148, docs/STREAMING.md:485, docs/ONBOARDING_STATUS.md:86, :87. `benchmarks/checkpoints.py::verify_adapter` records `weights_bytes`, `r`, `target_modules` and `base_model` but no content hash, so `run_config.json` cannot even identify which adapter produced a number. This is distinct from the missing `results/eval/turbo-*` directories: even with those JSONs committed, the artifact behind them would remain unfetchable and unidentifiable. A reviewer's first question — 'can I run your model?' — has no answer.

**Evidence.** docs/EXPERIMENTS.md:293-296: "`--preset v2-turbo`: v1's recipe with the base model swapped ... Adapter: `/kaggle/working/v2-final/best`, checkpoints in a private Hub repo."  docs/AGENT_BRIEF.md:94: `repo = f"{whoami()['name']}/whisper-turbo-hindi-lora-ckpt"`.  README.md:22: "(**40.4% → 23.8% WER** on FLEURS Hindi test, see below)".  README.md:389 (the Evaluation section's "Final test number"): `--adapter Hugme6969/whisper-medium-hindi-lora`.  benchmarks/checkpoints.py:232-241 returns only `{"path", "weights_bytes", "has_tokenizer", "peft_type", "r", "lora_alpha", "target_modules", "base_model"}` — no digest.

**Fix.** Either publish the v2 adapter to a public Hub repo and put its id (plus the commit revision) in EXPERIMENTS.md, README.md and AGENT_BRIEF.md in place of `/kaggle/working/v2-final/best`, or demote the turbo rows to 'measured on an unpublished artifact, not reproducible' until it is published. Add a SHA-256 of `adapter_model.safetensors` to `verify_adapter`'s summary so `run_config.json` pins the exact weights, and add one copy-pasteable command that reproduces the 23.83% row end to end.

### BLOCKER — Training entry point bypasses the FLEURS shim and cannot load data

**Where:** `asr/training/lora.py:168`

**Claim.** `load_primary` is the only FLEURS caller in the repo that does not go through `benchmarks.fleurs.load_fleurs`. It calls plain `load_dataset("google/fleurs", ...)` with no `trust_remote_code=True` and no fallback — which, by the repo's own documented compatibility matrix, fails on `datasets` 2.20 through 3.x. That is exactly the range `pyproject.toml` resolves to (`datasets>=2.19,<4`; the committed eval run_config records `datasets 3.6.0`, and evaluation only worked there because `load_fleurs` retries with `trust_remote_code=True`). So `python asr/training/lora.py --preset v2-turbo` — the documented command that produced the shipped adapter, and the only way to retrain it — dies at data loading in a correctly-installed environment, with the same opaque RuntimeError `benchmarks/fleurs.py` was written to prevent. `evaluate_checkpoint` calls `load_data` too, so `--evaluate` is equally dead. No test covers it (tests/test_training_config.py only checks preset fields), and no dimension of the audit read the training code.

**Evidence.** asr/training/lora.py:167-170: `if config.dataset == "fleurs":\n        ds = load_dataset("google/fleurs", f"{config.language}_in")\n        train_ds, eval_ds = ds["train"], ds["validation"]`.  benchmarks/fleurs.py:8-10 states the matrix: "* `< 2.20`  — loading scripts run by default. * `2.20–3.x` — scripts require an explicit `trust_remote_code=True`. * `>= 4.0`   — scripts removed entirely".  pyproject.toml: `"datasets>=2.19,<4"`.  results/eval/medium-lora-test-300-seed0/run_config.json: `"datasets": "3.6.0"`.

**Fix.** Replace the direct `load_dataset` in `load_primary` with `from benchmarks.fleurs import load_fleurs` and call it per split (`load_fleurs(f"{config.language}_in", "train")` / `"validation"`), so training uses the same three-strategy path and the same actionable error as evaluation. Add a CPU test that asserts `asr.training.lora.load_primary` routes through `benchmarks.fleurs.load_fleurs`.

### MAJOR — Documented Phase-1 fix is unrunnable: bench_all has no such flag

**Where:** `docs/AGENT_BRIEF.md:143`

**Claim.** The brief pre-registers one remediation — raise the no-speech threshold and re-run — and it cannot be executed. `scripts/bench_all.py` exposes no `--no-speech-threshold`; its `guards_on` step hardcodes the `asr_eval` command with no threshold override, so there is no path from the documented instruction to a re-measured run. I ran it: `python3 scripts/bench_all.py --no-speech-threshold 0.9 --force-steps guards_on --only guards_on` exits with `bench_all.py: error: unrecognized arguments: --no-speech-threshold 0.9`. Working around it by calling `benchmarks.asr_eval` directly needs a different `--out-dir`, which then breaks `compare_guards`, whose baseline and candidate paths are hardcoded to `results/eval/v2-guards-{off,on}`. This is the one defect the author explicitly expects Phase 1 to surface, in a plan whose whole premise is an autonomous agent with no human in the loop; the same dead instruction is repeated in the notebook the agent is told to run.

**Evidence.** docs/AGENT_BRIEF.md:141-144: "**WER worse, or no-speech fired on a clip whose reference is non-empty** → a regression you must **fix**, not report around. Raise `--no-speech-threshold` until it fires only on genuinely empty clips, re-run, and record both values in the ledger."  notebooks/finish_colab.ipynb:66 repeats it: "raise `--no-speech-threshold` and re-run with `--force-steps guards_on`".  scripts/bench_all.py:60-66 hardcodes the step: `"cmd": [sys.executable, "-m", "benchmarks.asr_eval", "run", "--model", TURBO, "--adapter", adapter, *eval_common, "--out-dir", str(root / "eval/v2-guards-on"), "--note", "decode guards at defaults"]` — and the parser at :195-217 defines only adapter/out-root/limit/streaming-limit/llm-models/tts-backends/ct2-dir/only/skip/force/force-steps/mirror/logs.

**Fix.** Add `--no-speech-threshold` and `--loop-guard-ngram` pass-through arguments to `bench_all.py` and thread them into the `guards_on` step's command (keeping them in the recorded `cmd` so the manifest shows which values were used), or change the brief and notebook to give the literal two-command `asr_eval` + `benchmarks.compare` invocation with explicit `--out-dir`/`--baseline`/`--candidate` paths.

### MAJOR — Hard-set curation commands pair whisper-medium with a 768-dim adapter

**Where:** `data/hard_set/README.md:79`

**Claim.** `data/hard_set/` is empty, the brief lists curating it as requiring a human, and the protocol a human would follow crashes on its first command. Both documented `asr_eval` invocations pass `--model openai/whisper-medium` with `--adapter results/whisper-lora-hi-full/best`, but that committed adapter is a whisper-small adapter: I parsed its safetensors header and all 288 tensors are `[16, 768]` / `[768, 16]` (whisper-small's d_model=768, 12+12 layers), against medium's 1024-dim, 24+24-layer stack. The repo states the incompatibility elsewhere — docs/ONBOARDING_STATUS.md:77 and README.md:308-309 both say it "cannot be loaded onto whisper-medium" — so this is a doc that contradicts the repo's own known fact, not a subtle bug. It also fails opaquely: `benchmarks/checkpoints.py::verify_adapter` already reads `base_model_name_or_path` and `asr_eval` records it in `adapter_info`, but nothing compares it to `--model`, so the user gets a PEFT `load_state_dict` size mismatch instead of a legible error.

**Evidence.** data/hard_set/README.md:77-81: `python -m benchmarks.asr_eval run \\\n    --model openai/whisper-medium \\\n    --adapter results/whisper-lora-hi-full/best \\\n    --split test --limit 300 \\\n    --out-dir results/eval/medium-lora-test` (repeated at :104-108 with `--hard-set`).  results/whisper-lora-hi-full/best/adapter_config.json: `"base_model_name_or_path": "openai/whisper-small"`.  Parsed header: `base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_A.weight [16, 768]`, distinct shapes `[(16, 768), (768, 16)]`, 288 tensors.  docs/ONBOARDING_STATUS.md:77: "an older **whisper-small** adapter, not v1, and cannot load on medium".  benchmarks/asr_eval.py:499 `adapter_info = verify_adapter(adapter)` — the returned `base_model` is written to run_config and never checked.

**Fix.** Rewrite both commands in data/hard_set/README.md to use a real adapter/base pair (e.g. `--model openai/whisper-large-v3-turbo --adapter <v2 adapter>`, or `--model openai/whisper-medium --adapter Hugme6969/whisper-medium-hindi-lora`). Separately, in `benchmarks/asr_eval.py` right after `verify_adapter`, compare `adapter_info["base_model"]` to `args.model` and `SystemExit` with both names when they differ; add the same guard to `asr/explicit/loader.py::load_whisper`.

### MAJOR — Phase-3 turns.jsonl drops the transcript and every latency segment

**Where:** `demo/notebook.py:417`

**Claim.** The live-turn deliverable — the one thing the brief says must be a distribution rather than an anecdote — is persisted from an object that does not contain the segments it is supposed to report. `stream_turn` builds a full per-answer record (transcript, response, state, endpoint_reason, silence_wait_ms, endpoint_to_final_ms, first_token_to_first_unit_ms, tts_synthesis_ms, llm_total_ms, spoken_units) and returns it, but `self.turns.extend(...)` keeps only eight fields: audio, streaming, asr_ms, asr_rtf(None), first_token_ms, to_audio_ms, response_latency_ms, turn_total_ms(None). The notebook cell calls `agent.stream_turn(clip)` without capturing the return value, then writes `results/live/turns.jsonl` from `agent.turns` — so the committed evidence has no transcript, no response, no endpoint term and no decomposition, and the cell's own summary loop over `first_token_to_first_unit_ms` and `tts_synthesis_ms` silently prints nothing because those keys are never present. `summary()` also averages `turn()` and `stream_turn()` rows together, mixing a no-endpoint latency with an endpoint-inclusive one under one mean. tests/test_demo_notebook.py:197-208 asserts the decomposition exists in `out["answers"][0]`, i.e. in the discarded return value, never in `agent.turns`.

**Evidence.** demo/notebook.py:417-422: `self.turns.extend({"audio": str(audio_path), "streaming": True, "asr_ms": a["asr_ms"], "asr_rtf": None, "first_token_ms": a["first_token_ms"], "to_audio_ms": a["to_audio_ms"], "response_latency_ms": a["response_latency_ms"], "turn_total_ms": None} for a in answers)`.  notebooks/finish_colab.ipynb:195: `for t in agent.turns: fh.write(json.dumps(t, ...))` then `for key in ('asr_ms', 'first_token_ms', 'to_audio_ms', 'first_token_to_first_unit_ms', 'tts_synthesis_ms')`.  docs/AGENT_BRIEF.md:176-178: "Save `agent.turns` to `results/live/turns.jsonl` and report the **distribution** of `response_latency_ms` and its segments."

**Fix.** Append the whole per-answer dict (minus the `sink` object) to `self.turns` instead of the eight-field projection, keeping `streaming: True` as a discriminator, and have `summary()` report streaming and non-streaming turns separately so the endpoint term is not averaged into a mean that excludes it. Add a test asserting that after `stream_turn`, `agent.turns[0]` contains `transcript`, `endpoint_to_final_ms`, `first_token_to_first_unit_ms` and `tts_synthesis_ms`.

### MINOR — MIT claimed with no LICENSE file; shipped weights carry a blank card

**Where:** `README.md:512`

**Claim.** The repo declares MIT in two places and contains no LICENSE file at all — `git ls-files` shows only `.gitignore`, `README.md` and `pyproject.toml` at the root — so nothing in the distribution actually grants the licence, and the terms of the two things the project redistributes are unstated. The only committed model weights ship PEFT's untouched auto-generated card: "# Model Card for Model ID", 40+ `[More Information Needed]` placeholders including `**License:** [More Information Needed]`, and a `base_model` that the repo's own docs call superseded and uncitable. Meanwhile `asr/training/lora.py` documents that the shipped recipe mixes in a gated dataset whose terms must be accepted on the Hub, and the primary corpus is FLEURS — neither is attributed anywhere, and a LoRA adapter trained on them is a derivative that cannot simply be relicensed MIT by omission. No audit dimension looked at licensing, attribution or the model card, and it is the first file a reviewer opens after the README.

**Evidence.** README.md:512-514: "## License\n\nMIT"; pyproject.toml:11: `license = { text = "MIT" }`; `ls -a | grep -i licen` → no match, `git ls-files | grep -v "/"` → `.gitignore`, `README.md`, `pyproject.toml`.  results/whisper-lora-hi-full/best/README.md:10 "# Model Card for Model ID", :29 "- **License:** [More Information Needed]", :86 "### Training Data ... [More Information Needed]".  asr/training/lora.py:20-21: "IndicVoices (``ai4bharat/IndicVoices``) is gated: accept the terms on the Hub and ``huggingface-cli login`` before using ``--indicvoices-samples``."

**Fix.** Add a top-level `LICENSE` file with the MIT text and a copyright line. Fill in `results/whisper-lora-hi-full/best/README.md` (base model, language, training data with FLEURS CC-BY-4.0 attribution and the IndicVoices terms, the 39.25% validation WER, and a 'superseded, kept for history' note), and write the same card for the v2 adapter before it is published — including a sentence stating the licence the adapter itself is offered under and the upstream dataset terms it inherits.
