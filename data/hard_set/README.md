# Hindi ASR hard set

A small, fixed, adversarial evaluation set. It exists because FLEURS is clean
read speech: a model can improve on FLEURS test while getting worse at what a
voice agent actually hears. Aggregate WER cannot show that. This set can,
because it is reported **per category**.

`manifest.jsonl` is the live set. It is **not** in the repository yet — nobody
can hand you a hard set, because a hard set is only meaningful if the audio
comes from the conditions *you* care about. `manifest.example.jsonl` shows the
schema.

## Rules

1. **Fixed.** Once an item is `curated`, its audio and transcript never change.
   Editing the set silently invalidates every comparison made against it.
2. **Never trained on.** These clips must not enter any training mixture, and
   must not overlap the FLEURS train/validation splits.
3. **Human-verified transcripts.** `candidate` items carry an unverified
   transcript and are excluded from all reported metrics until a human checks
   them and sets `status` to `curated`.
4. **Every item is category-tagged.** Untagged items fail validation.
5. **Target ~15–30 items per category.** Small enough to curate honestly, large
   enough that a category WER is not one clip's coin flip. Report per-category
   counts alongside per-category WER so the reader can judge the noise floor.

## Categories

| Category | What it probes |
| --- | --- |
| `noisy` | Background noise, low SNR |
| `accented` | Regional or L2 Hindi accents |
| `named_entity` | People, places, brands, organisations |
| `numeric` | Digits, amounts, dates, phone numbers |
| `code_switch` | Hinglish, Latin-script insertions |
| `spontaneous` | Disfluent, conversational, non-read speech |
| `far_field` | Distant mic, room reverberation |
| `telephony` | 8 kHz or codec-degraded audio |
| `fast_speech` | Unusually high speaking rate |
| `long_form` | Longer than Whisper's 30 s window |

## Schema

One JSON object per line:

```json
{
  "id": "noisy_001",
  "transcript": "मुझे कल सुबह दिल्ली जाना है",
  "categories": ["noisy", "spontaneous"],
  "audio": {"type": "local", "path": "data/hard_set/audio/noisy_001.wav"},
  "status": "curated",
  "notes": "street traffic behind speaker, roughly 10 dB SNR",
  "duration_s": 4.1,
  "source": "self-recorded 2026-09"
}
```

`audio` is either:

- `{"type": "local", "path": "<repo-relative path>"}` — 16 kHz mono WAV, or
- `{"type": "hf", "dataset": "google/fleurs", "config": "hi_in", "split": "test", "index": 123}`

The `hf` form lets you pin a specific public example without committing audio.
The `local` form is for your own recordings. Keep local audio under
`data/hard_set/audio/` and commit it only if it is small and you have the right
to redistribute it; otherwise add it to `.gitignore` and document where it
lives.

## How to build the set

The efficient path is to mine your own failures rather than guess what "hard"
means:

```bash
# 1. Run a normal evaluation first.
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter results/whisper-lora-hi-full/best \
    --split test --limit 300 \
    --out-dir results/eval/medium-lora-test

# 2. Mine the worst examples into a review queue.
python -m benchmarks.hard_set bootstrap \
    --predictions results/eval/medium-lora-test/predictions.jsonl \
    --out data/hard_set/candidates.jsonl \
    --limit 30 --min-wer 0.4

# 3. Listen to each clip. Fix the transcript. Assign categories.
#    Delete anything that is a bad reference rather than hard audio —
#    FLEURS does contain transcription errors, and those are not hard audio.
#    Then move reviewed lines into manifest.jsonl with status "curated".

# 4. Validate.
python -m benchmarks.hard_set validate --manifest data/hard_set/manifest.jsonl
```

Step 3 is the part that cannot be automated and is the part that makes the set
worth anything.

## Reporting

```bash
python -m benchmarks.asr_eval run \
    --model openai/whisper-medium \
    --adapter results/whisper-lora-hi-full/best \
    --hard-set data/hard_set/manifest.jsonl \
    --out-dir results/eval/medium-lora-hardset
```

Record per-category WER in `docs/EXPERIMENTS.md`. A change that improves FLEURS
test WER but regresses a hard-set category is not automatically an improvement
— say so explicitly in the decision line rather than reporting the headline
number alone.
