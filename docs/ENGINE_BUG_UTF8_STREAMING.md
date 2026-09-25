# Bug report: streamed multi-byte characters are corrupted

**Repository:** `Vaibhav7711/full-inference-engine`
**File:** `engine/server/openai.py`, the `_stream` generator
**Affects:** every non-ASCII script, streaming only. Non-streaming
`/v1/completions` and `/v1/chat/completions` are correct.
**Severity:** silent data corruption. The output is fluent-looking and wrong.
**Found:** 2026-09-25, Colab T4, Qwen3-0.6B fp16, via
`indic-voice-pipeline/scripts/engine_parity.py`.

## Symptom

Five prompts through `/v1/completions` with `stream: true`, compared against
the same weights decoded by a reference implementation. The English prompt
was byte-identical. All four Hindi prompts came back studded with U+FFFD:

```
prompt:    नमस्ते, आज मौसम कैसा है?
reference: नमस्ते, मौसम कैसा है?
served:    नमस्�े, म�सम क�सा ह�?
```

The four replacement characters stand where `त`, `ौ`, `ै` and `ै` should be.
Note what was lost: not mangled, **dropped**. The `त` is absent from the
output entirely.

That split — 1/1 English clean, 0/4 Hindi corrupt — is the whole diagnosis.
This is not a numerics difference in an attention kernel. It is byte
arithmetic in the streaming loop.

## Cause

Qwen's tokenizer is byte-level BPE, so a 3-byte Devanagari character is
routinely split across two tokens. Decoding the tokens received so far, while
a character is half-arrived, yields U+FFFD. With the real tokenizer:

```
n= 3  U+FFFD=False  sent_is_prefix=True   'नमस'
n= 4  U+FFFD=True   sent_is_prefix=True   'नमस्�'
n= 5  U+FFFD=False  sent_is_prefix=False  'नमस्त'
n= 6  U+FFFD=False  sent_is_prefix=False  'नमस्ते'
```

The line that matters is n=5. When the second half of the character arrives,
the corrected text is **no longer an extension of what was already sent** —
U+FFFD is replaced by the real character at the same position.

`_stream` computes its delta by length:

```python
text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
trimmed, hit = truncate_at_stop(text, stops) if stops else (text, False)
if len(trimmed) > len(sent_text):
    delta = trimmed[len(sent_text):]
    sent_text = trimmed
    yield f"data: {json.dumps(chunk(delta, None))}\n\n"
```

`trimmed[len(sent_text):]` assumes `sent_text` is a prefix of `trimmed`. Once
that assumption breaks, the loop does this:

| poll | `trimmed` | `len` | `len > len(sent_text)`? | delta sent | `sent_text` after |
| ---: | --- | ---: | --- | --- | --- |
| 4 | `'नमस्�'` | 5 | yes (5 > 4) | `'�'` | `'नमस्�'` |
| 5 | `'नमस्त'` | 5 | **no** (5 = 5) | nothing | unchanged, still holds U+FFFD |
| 6 | `'नमस्ते'` | 6 | yes (6 > 5) | `'े'` | `'नमस्ते'` |

The client assembles `'नमस्�'` + `'े'`. The replacement character was
committed at poll 4 and can never be retracted; the real `'त'` is skipped at
poll 6 because the slice starts past it.

ASCII never triggers this: one byte per character means a token boundary is
always a character boundary, so `sent_text` stays a prefix.

## Fix

Two changes, both in `_stream`. Hold back a trailing replacement character,
and stop assuming the prefix.

```python
    text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
    trimmed, hit = truncate_at_stop(text, stops) if stops else (text, False)
    # A trailing U+FFFD is a multi-byte character that has only partly
    # arrived: byte-level BPE splits a 3-byte Devanagari character across two
    # tokens. Emitting it is unrecoverable -- the next decode replaces it in
    # place, so the text already sent stops being a prefix of the corrected
    # text and the real character is skipped by the slice below. Holding it
    # back costs one poll of latency and keeps the stream a pure append.
    stable = trimmed[:-1] if trimmed.endswith("�") else trimmed
    if stable.startswith(sent_text) and len(stable) > len(sent_text):
        delta = stable[len(sent_text):]
        sent_text = stable
        yield f"data: {json.dumps(chunk(delta, None))}\n\n"
```

`stable.startswith(sent_text)` is deliberately a guard rather than an
assertion: with the hold-back in place it should always hold, and if it ever
does not, skipping a poll is better than emitting a delta computed from a
wrong offset.

Three details worth keeping:

- **Only a *trailing* U+FFFD is held back.** A replacement character in the
  middle of the decoded text is genuinely undecodable output, not a
  half-arrived character, and suppressing it would hide a real problem.
- **A stream that ends mid-character loses that character**, which is correct:
  the alternative is speaking a replacement character.
- **One poll of added latency**, bounded by the 5 ms poll interval. Against
  the cost: `truncate_at_stop` and the `_collect` path have the same
  structure, so they are worth checking for the same assumption.

## Why it matters more than it looks

The corrupted text is handed to a TTS engine and **pronounced**. A voice agent
in Hindi mispronouncing or dropping every fourth character is unusable, and
nothing about the output looks broken to a monitoring system: the response is
the right length, arrives at the right rate, and reports clean usage counts.

It also cannot be repaired downstream. By the time a client sees U+FFFD the
bytes are gone. The fix has to be server-side.

## Tests that would have caught it

The engine's CPU suite is 296 tests and its GPU gates check token identity
against stock Transformers — but a token-identity gate compares **token ids**,
and these are correct. The corruption is introduced when ids are turned into
streamed text, which is the one step such a gate does not cover.

Suggested additions:

1. A streaming test whose expected output is non-ASCII, asserting the
   concatenated deltas equal `tokenizer.decode(all_ids)`. Any script with
   multi-byte characters works; Devanagari, CJK and emoji all reproduce it.
2. An assertion inside `_stream` that `sent_text` is always a prefix of what
   is about to be sent. That invariant is what broke, and it is cheap to
   check.
3. At least one non-ASCII prompt in any parity or identity suite. The reason
   this survived is that every streaming test used English.

Downstream, `indic-voice-pipeline` now counts U+FFFD in
`HttpEngineMetrics.replacement_chars`, probes with a Devanagari prompt at
startup, and fails `scripts/engine_parity.py` on corruption with a distinct
diagnosis — so it reports a serving defect rather than a decode divergence.
`tests/test_engine_corruption.py` holds the offline reproduction and the
proposed fix as a passing test.
