"""LoRA fine-tuning of Whisper-small on Indic speech.

Usage (run from repo root):
    python asr/training/lora.py --dataset fleurs --language hi --epochs 1 \
        --max-train-samples 100 --max-eval-samples 50 \
        --output-dir results/whisper-lora-hi-quick
"""
from __future__ import annotations

import argparse
import json
import sys
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    model_name: str = "openai/whisper-small"
    dataset: str = "fleurs"
    language: str = "hi"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "results/whisper-lora-hi"
    fp16: bool = True
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def load_data(config):
    from datasets import load_dataset, Audio

    if config.dataset == "fleurs":
        ds = load_dataset("google/fleurs", f"{config.language}_in")
        train_ds, eval_ds = ds["train"], ds["validation"]

        def norm(ex):
            return {"audio": ex["audio"], "sentence": ex["transcription"]}

        train_ds = train_ds.map(norm, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(norm, remove_columns=eval_ds.column_names)

    elif config.dataset == "common_voice":
        ds = load_dataset("mozilla-foundation/common_voice_17_0", config.language)
        train_ds, eval_ds = ds["train"], ds["validation"]

        def norm(ex):
            return {"audio": ex["audio"], "sentence": ex["sentence"]}

        train_ds = train_ds.map(norm, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(norm, remove_columns=eval_ds.column_names)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    if config.max_train_samples:
        train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
    if config.max_eval_samples:
        eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))

    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    return train_ds, eval_ds


def prepare_model(config):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model

    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16,
    )

    # Clear forced decoding so LoRA can learn freely.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Gradient checkpointing saves memory; requires these two calls.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA on attention projections only. No task_type — avoids PEFT's
    # Seq2Seq wrapper which remaps input_features to input_ids and breaks Whisper.
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


class DataCollator:
    """Audio + text -> input_features + labels."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        audio = [f["audio"]["array"] for f in features]
        text = [f["sentence"] for f in features]

        batch = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=225,
        )
        label_ids = labels.input_ids.masked_fill(
            labels.attention_mask.ne(1), -100,
        )
        # Strip BOS if present.
        if (label_ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            label_ids = label_ids[:, 1:]

        batch["labels"] = label_ids
        return batch


def train(config):
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    import evaluate

    print(f"Loading dataset: {config.dataset} ({config.language})")
    train_ds, eval_ds = load_data(config)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    print("Loading model + applying LoRA...")
    model, processor = prepare_model(config)
    collator = DataCollator(processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        p = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        r = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=p, references=r) * 100}

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
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    result = trainer.train()

    # Save.
    best_dir = Path(config.output_dir) / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    metrics_path = Path(config.output_dir) / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)

    print(f"\nTrain loss: {result.metrics.get('train_loss', 'N/A')}")

    # Final eval.
    eval_metrics = trainer.evaluate()
    with open(Path(config.output_dir) / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Final WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")
    print(f"Saved to {best_dir}")


def evaluate_checkpoint(config, checkpoint):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    import evaluate

    processor = WhisperProcessor.from_pretrained(checkpoint)
    base = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model = model.merge_and_unload().to("cuda").eval()

    _, eval_ds = load_data(config)
    wer_metric = evaluate.load("wer")

    preds, refs = [], []
    for i, s in enumerate(eval_ds):
        feat = processor.feature_extractor(
            s["audio"]["array"], sampling_rate=16000, return_tensors="pt",
        ).input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            ids = model.generate(feat, max_new_tokens=225)
        preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
        refs.append(s["sentence"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_ds)}")

    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"WER: {wer * 100:.2f}% ({len(preds)} samples)")


def main():
    p = argparse.ArgumentParser(description="LoRA fine-tune Whisper for Indic ASR")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "common_voice"])
    p.add_argument("--language", default="hi")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--output-dir", default="results/whisper-lora-hi")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args(sys.argv[1:])

    config = TrainingConfig(
        model_name=a.model, dataset=a.dataset, language=a.language,
        epochs=a.epochs, batch_size=a.batch_size, lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha, learning_rate=a.learning_rate,
        output_dir=a.output_dir, max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
    )

    if a.evaluate:
        if not a.checkpoint:
            raise ValueError("--checkpoint required for --evaluate")
        evaluate_checkpoint(config, a.checkpoint)
    else:
        train(config)


if __name__ == "__main__":
    main()
