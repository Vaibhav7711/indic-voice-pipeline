"""LoRA fine-tuning of Whisper-small on Indic code-switched speech.

This script fine-tunes Whisper-small using LoRA (rank 16) on attention
projections for Hindi/Hindi-English code-switched ASR. Designed to run
on a single T4 in Google Colab.

Memory budget
-------------
Whisper-small FP16:     ~488 MB (model weights)
LoRA rank-16 adapters:  ~2-4 M trainable parameters
Batch 4 + grad ckpt:    ~6-8 GiB total
T4 budget:              16 GiB — fits comfortably, QLoRA NOT needed.

Datasets
--------
- **google/fleurs** (hi_in split) — readily available, ~10 hrs Hindi.
- **mozilla-foundation/common_voice_17_0** (hi split) — larger, needs
  agreement.
- **MUCS 2021** — Hindi-English code-switched. If available locally,
  point ``--data-dir`` to the extracted folder.

Usage
-----
    python -m training.whisper_lora \\
        --dataset fleurs \\
        --language hi \\
        --epochs 3 \\
        --batch-size 4 \\
        --lora-rank 16 \\
        --output-dir results/whisper-lora-hi

    # Evaluate:
    python -m training.whisper_lora \\
        --evaluate \\
        --checkpoint results/whisper-lora-hi/best \\
        --dataset fleurs \\
        --language hi

Design rationale
----------------
- LoRA, not full fine-tune: preserves Whisper's multilingual backbone while
  adapting attention patterns for Indic phonology and code-switching.
- Not QLoRA: the base model (488 MB) fits in FP16 with room to spare.
  QLoRA would save ~250 MB and add NF4 dequantization overhead — a cost
  without a real benefit on T4.
- Attention-only targets: Whisper's attention layers are where language-
  specific patterns live. FFN layers are more generic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "openai/whisper-small"
    dataset: str = "fleurs"  # "fleurs", "common_voice", or "local"
    data_dir: str | None = None  # for local datasets
    language: str = "hi"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    output_dir: str = "results/whisper-lora-hi"
    gradient_checkpointing: bool = True
    fp16: bool = True
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    max_train_samples: int | None = None  # for quick iteration
    max_eval_samples: int | None = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target attention projections in both encoder and decoder.
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "out_proj",
            ]


def load_dataset_split(config: TrainingConfig):
    """Load train/eval splits based on dataset choice."""
    from datasets import load_dataset, Audio

    if config.dataset == "fleurs":
        lang_code = f"{config.language}_in"  # e.g. "hi_in"
        ds = load_dataset(
            "google/fleurs", lang_code, trust_remote_code=True
        )
        train_ds = ds["train"]
        eval_ds = ds["validation"]

        # FLEURS uses "audio" (dict with array+sampling_rate) and "transcription".
        def normalize(example):
            return {
                "audio": example["audio"],
                "sentence": example["transcription"],
            }

        train_ds = train_ds.map(normalize, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(normalize, remove_columns=eval_ds.column_names)

    elif config.dataset == "common_voice":
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            config.language,
            trust_remote_code=True,
        )
        train_ds = ds["train"]
        eval_ds = ds["validation"]

        def normalize_cv(example):
            return {
                "audio": example["audio"],
                "sentence": example["sentence"],
            }

        train_ds = train_ds.map(normalize_cv, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(normalize_cv, remove_columns=eval_ds.column_names)

    elif config.dataset == "local":
        raise NotImplementedError(
            "Local dataset loading (MUCS, custom) requires dataset-specific "
            "code. See README for format requirements."
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Subsample for quick iteration.
    if config.max_train_samples:
        train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
    if config.max_eval_samples:
        eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))

    # Ensure audio is 16 kHz.
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))

    return train_ds, eval_ds


def prepare_model_and_tokenizer(config: TrainingConfig):
    """Load Whisper + apply LoRA adapters."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    )

    # Freeze encoder initially — let LoRA handle adaptation.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # Apply LoRA.
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def create_data_collator(processor):
    """Build a data collator that handles audio→features + text→labels."""
    from dataclasses import dataclass as dc
    from typing import Any

    @dc
    class WhisperDataCollator:
        processor: Any

        def __call__(self, features: list[dict]) -> dict:
            # Extract audio arrays.
            audio_arrays = [f["audio"]["array"] for f in features]
            sentences = [f["sentence"] for f in features]

            # Process audio to mel features.
            input_features = self.processor.feature_extractor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features

            # Tokenize labels.
            labels = self.processor.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=225,
            )
            # Replace padding token ID with -100 for loss masking.
            label_ids = labels.input_ids.masked_fill(
                labels.attention_mask.ne(1), -100
            )
            # Remove BOS token if present at start.
            if (
                label_ids[:, 0] == self.processor.tokenizer.bos_token_id
            ).all():
                label_ids = label_ids[:, 1:]

            return {
                "input_features": input_features,
                "labels": label_ids,
            }

    return WhisperDataCollator(processor=processor)


def compute_wer_metric(processor):
    """Build a WER compute function for the Trainer."""
    import evaluate

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer * 100}  # percentage

    return compute_metrics


def train(config: TrainingConfig):
    """Run the LoRA fine-tuning loop."""
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    print(f"Training config: {config}")
    print(f"Loading dataset: {config.dataset} ({config.language})")

    train_ds, eval_ds = load_dataset_split(config)
    print(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    model, processor = prepare_model_and_tokenizer(config)
    data_collator = create_data_collator(processor)
    compute_metrics = compute_wer_metric(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",  # disable wandb in Colab
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    train_result = trainer.train()

    # Save best model.
    best_dir = Path(config.output_dir) / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    # Save training metrics.
    metrics = train_result.metrics
    metrics_path = Path(config.output_dir) / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Best model saved to {best_dir}")
    print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")

    # Run final evaluation.
    eval_metrics = trainer.evaluate()
    eval_path = Path(config.output_dir) / "eval_metrics.json"
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Final WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")

    return train_result, eval_metrics


def evaluate_checkpoint(config: TrainingConfig, checkpoint: str):
    """Evaluate a saved LoRA checkpoint."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    import evaluate

    processor = WhisperProcessor.from_pretrained(checkpoint)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model = model.merge_and_unload()  # merge LoRA weights for inference
    model = model.to("cuda")
    model.eval()

    _, eval_ds = load_dataset_split(config)
    wer_metric = evaluate.load("wer")

    predictions, references = [], []
    for sample in eval_ds:
        audio = sample["audio"]["array"]
        features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.float16)

        with torch.inference_mode():
            predicted_ids = model.generate(features, max_new_tokens=225)

        pred_text = processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )
        predictions.append(pred_text)
        references.append(sample["sentence"])

    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"WER: {wer * 100:.2f}%")
    print(f"Evaluated on {len(predictions)} samples")
    return wer


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Whisper for Indic ASR"
    )
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument(
        "--dataset", default="fleurs",
        choices=["fleurs", "common_voice", "local"],
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results/whisper-lora-hi")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        data_dir=args.data_dir,
        language=args.language,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    if args.evaluate:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluation")
        evaluate_checkpoint(config, args.checkpoint)
    else:
        train(config)


if __name__ == "__main__":
    main()
