# ASR experiment ledger

This file is the project-level index for every training run. Keep the full
machine-readable configuration and metrics next to each checkpoint; summarize
only decisions and comparable results here.

## Evaluation protocol

- **Model selection:** FLEURS Hindi validation WER. The test split is never
  used to decide whether to keep training.
- **Final reporting:** FLEURS Hindi test WER and CER, plus the fixed hard set
  once it is added.
- **Text policy:** retain the original reference text for the primary metric;
  record any normalization separately rather than silently improving WER.
- **Latency:** measure the merged LoRA model through `ASRRunner`, not through
  Hugging Face `generate()`, so quality and runtime results describe the same
  serving implementation.

## Active run: Whisper-medium Hindi LoRA

| Field | Value |
| --- | --- |
| Base model | `openai/whisper-medium` |
| Data | FLEURS Hindi train + first 5,000 streamed IndicVoices Hindi examples |
| Validation | FLEURS Hindi validation (239 examples) |
| LoRA | rank 16, alpha 32, dropout 0.05; q/k/v/out projections |
| Effective batch size | 8 (2 × 4 gradient accumulation) |
| Target | 3 epochs / 2,670 optimizer steps |
| Checkpoint policy | every 200 steps; retain latest 3 in Drive |
| Latest observed checkpoint | step 600, epoch 0.674 |
| Best validation WER at step 600 | 32.70% |

### Required comparisons after training

| Experiment | Primary question | Selection metric |
| --- | --- | --- |
| Whisper-medium baseline | What does model scale buy before adaptation? | validation WER |
| FLEURS-only LoRA | Is IndicVoices data helping this domain? | validation WER + hard set |
| FLEURS + IndicVoices LoRA | Active data-mixture baseline | validation WER + hard set |
| q/v-only vs q/k/v/out LoRA | Does adapting keys/output projections justify added capacity? | WER, adapter size, latency |
| Augmentation ablation | Does robustness improve without clean-speech regression? | WER by condition |

## Result entry template

Copy this block for each completed run.

```markdown
### <run name>
- Commit: `<git SHA>`
- Data and filtering:
- Training configuration:
- Best checkpoint / epoch:
- Validation WER / CER:
- Test WER / CER:
- Hard-set WER by category:
- Merged-adapter ASR RTF and latency:
- Error examples and observed failure modes:
- Decision: keep / reject, and why:
```
