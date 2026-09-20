"""LoRA fine-tuning of Whisper on Hindi speech.

The v1 adapter in ``docs/EXPERIMENTS.md`` is the ``v1`` preset::

    python asr/training/lora.py --preset v1 --output-dir /content/drive/MyDrive/whisper-training

which expands to: ``openai/whisper-medium``, FLEURS Hindi train plus the first
5,000 streamed IndicVoices Hindi examples, validation on FLEURS Hindi
validation, rank 16 / alpha 32 / dropout 0.05 on q/k/v/out projections,
batch 2 x 4 accumulation, 3 epochs, checkpoint every 200 steps keeping 3.
Every run writes ``train_config.json`` next to its checkpoints so the ledger
never has to describe a configuration from memory again.

Quick smoke run (small model, few samples)::

    python asr/training/lora.py --model openai/whisper-small \\
        --max-train-samples 100 --max-eval-samples 50 --epochs 1 \\
        --output-dir results/whisper-lora-hi-quick

IndicVoices (``ai4bharat/IndicVoices``) is gated: accept the terms on the Hub
and ``huggingface-cli login`` before using ``--indicvoices-samples``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

INDICVOICES_REPO = "ai4bharat/IndicVoices"
INDICVOICES_CONFIGS = {"hi": "hindi", "te": "telugu", "ta": "tamil", "bn": "bengali"}


@dataclass
class TrainingConfig:
    model_name: str = "openai/whisper-medium"
    dataset: str = "fleurs"
    language: str = "hi"
    #: Number of IndicVoices train examples to stream in and mix with the
    #: primary dataset. 0 disables the mix.
    indicvoices_samples: int = 0
    indicvoices_text_column: str = "text"
    #: Clips longer than this are dropped — Whisper's window is 30 s and a
    #: truncated label teaches the model to stop early.
    max_audio_seconds: float = 30.0
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    output_dir: str = "results/whisper-lora-hi"
    fp16: bool = True
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    resume_from_checkpoint: str | None = None
    seed: int = 42


PRESETS: dict[str, dict] = {
    # Exactly the ledger's "Active run: Whisper-medium Hindi LoRA".
    "v1": dict(
        model_name="openai/whisper-medium",
        dataset="fleurs",
        language="hi",
        indicvoices_samples=5000,
        epochs=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
    ),
    # The superseded whisper-small run kept under results/whisper-lora-hi-full.
    "small-fleurs": dict(
        model_name="openai/whisper-small",
        dataset="fleurs",
        language="hi",
        indicvoices_samples=0,
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=2,
        eval_steps=100,
        save_steps=100,
    ),
}


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Preset first, then any explicitly passed flag overrides it."""
    values: dict = {}
    if args.preset:
        values.update(PRESETS[args.preset])
    for key, value in vars(args).items():
        if key in ("preset", "evaluate", "checkpoint") or value is None:
            continue
        values[key] = value
    return TrainingConfig(**values)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _keep(example, max_seconds: float) -> bool:
    audio = example["audio"]
    seconds = len(audio["array"]) / audio["sampling_rate"]
    return 0 < seconds <= max_seconds and bool(example["sentence"].strip())


def load_primary(config: TrainingConfig):
    from datasets import load_dataset

    if config.dataset == "fleurs":
        ds = load_dataset("google/fleurs", f"{config.language}_in")
        train_ds, eval_ds = ds["train"], ds["validation"]
        text_col = "transcription"
    elif config.dataset == "common_voice":
        ds = load_dataset("mozilla-foundation/common_voice_17_0", config.language)
        train_ds, eval_ds = ds["train"], ds["validation"]
        text_col = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    def norm(ex):
        return {"audio": ex["audio"], "sentence": ex[text_col]}

    train_ds = train_ds.map(norm, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(norm, remove_columns=eval_ds.column_names)
    return train_ds, eval_ds


def load_indicvoices(config: TrainingConfig):
    """Stream the first N IndicVoices train examples into a cached Dataset.

    Streaming avoids downloading the 445k-example split; ``from_generator``
    writes the taken rows to the datasets cache as WAV bytes, so memory does
    not scale with N. "First N" is deterministic across runs because the Hub
    shard order is fixed — that is what makes the v1 mix reproducible.
    """
    from datasets import Audio, Dataset, Features, Value, load_dataset

    name = INDICVOICES_CONFIGS.get(config.language)
    if name is None:
        raise ValueError(f"No IndicVoices config known for language {config.language!r}")

    stream = load_dataset(INDICVOICES_REPO, name, split="train", streaming=True)
    stream = stream.cast_column("audio_filepath", Audio(sampling_rate=16000))
    n = config.indicvoices_samples
    text_col = config.indicvoices_text_column

    def rows():
        for i, ex in enumerate(stream):
            if i >= n:
                break
            audio = ex["audio_filepath"]
            yield {
                "audio": {"array": np.asarray(audio["array"], dtype=np.float32),
                          "sampling_rate": audio["sampling_rate"]},
                "sentence": ex[text_col],
            }

    features = Features({"audio": Audio(sampling_rate=16000), "sentence": Value("string")})
    return Dataset.from_generator(rows, features=features)


def load_data(config: TrainingConfig):
    from datasets import Audio, concatenate_datasets

    train_ds, eval_ds = load_primary(config)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    sources = {"primary_train": len(train_ds), "eval": len(eval_ds)}

    if config.indicvoices_samples > 0:
        extra = load_indicvoices(config)
        sources["indicvoices_train"] = len(extra)
        train_ds = concatenate_datasets([train_ds, extra]).shuffle(seed=config.seed)

    if config.max_train_samples:
        train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
    if config.max_eval_samples:
        eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))

    before = len(train_ds)
    train_ds = train_ds.filter(lambda ex: _keep(ex, config.max_audio_seconds))
    sources["dropped_too_long_or_empty"] = before - len(train_ds)
    return train_ds, eval_ds, sources


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def prepare_model(config: TrainingConfig):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16 if config.fp16 else torch.float32,
    )

    # Clear forced decoding so LoRA can learn freely.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Gradient checkpointing saves memory; requires these two calls.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # No task_type — avoids PEFT's Seq2Seq wrapper, which remaps
    # input_features to input_ids and breaks Whisper.
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.generation_config.language = config.language
    model.generation_config.task = "transcribe"
    model.print_trainable_parameters()
    return model, processor


class DataCollator:
    """Audio + text -> input_features + labels."""

    def __init__(self, processor, fp16: bool = True):
        self.processor = processor
        self.fp16 = fp16

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

        if self.fp16:
            batch["input_features"] = batch["input_features"].half()
        batch["labels"] = label_ids
        return batch


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------

def write_train_config(config: TrainingConfig, sources: dict) -> None:
    import subprocess

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        sha = None
    payload = {"config": asdict(config), "data": sources, "git_commit": sha,
               "argv": sys.argv}
    (out / "train_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def train(config: TrainingConfig):
    import evaluate
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    print(f"Loading dataset: {config.dataset} ({config.language})"
          + (f" + {config.indicvoices_samples} IndicVoices" if config.indicvoices_samples else ""))
    train_ds, eval_ds, sources = load_data(config)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}  {sources}")
    write_train_config(config, sources)

    print("Loading model + applying LoRA...")
    model, processor = prepare_model(config)
    collator = DataCollator(processor, fp16=config.fp16)

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
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=config.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save the best adapter with the full processor so serving can load
    # tokenizer + feature extractor from the adapter directory alone.
    best_dir = Path(config.output_dir) / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    with open(Path(config.output_dir) / "train_metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2)
    print(f"\nTrain loss: {result.metrics.get('train_loss', 'N/A')}")

    eval_metrics = trainer.evaluate()
    with open(Path(config.output_dir) / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Final validation WER: {eval_metrics.get('eval_wer', float('nan')):.2f}%")
    print(f"Saved to {best_dir}")
    print("Final test numbers come from benchmarks.asr_eval, not from here.")


def evaluate_checkpoint(config: TrainingConfig, checkpoint: str):
    """Quick validation-WER check on a checkpoint via HF generate().

    This is a training-time sanity check only. Reportable numbers come from
    ``benchmarks.asr_eval``, which runs the explicit serving path.
    """
    import evaluate
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        checkpoint if (Path(checkpoint) / "tokenizer_config.json").is_file()
        else config.model_name,
    )
    base = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model = model.merge_and_unload().to("cuda").eval()

    _, eval_ds, _ = load_data(config)
    wer_metric = evaluate.load("wer")

    preds, refs = [], []
    for i, s in enumerate(eval_ds):
        feat = processor.feature_extractor(
            s["audio"]["array"], sampling_rate=16000, return_tensors="pt",
        ).input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            ids = model.generate(feat, max_new_tokens=225, language=config.language,
                                 task="transcribe")
        preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
        refs.append(s["sentence"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_ds)}")

    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"Validation WER: {wer * 100:.2f}% ({len(preds)} samples)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Whisper for Indic ASR")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="Named recipe from docs/EXPERIMENTS.md; flags override it.")
    p.add_argument("--model", dest="model_name", default=None)
    p.add_argument("--dataset", choices=["fleurs", "common_voice"], default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--indicvoices-samples", type=int, default=None)
    p.add_argument("--indicvoices-text-column", default=None)
    p.add_argument("--max-audio-seconds", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", dest="gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--target-modules", nargs="+", default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    config = build_config(args)
    if args.evaluate:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for --evaluate")
        evaluate_checkpoint(config, args.checkpoint)
    else:
        train(config)


if __name__ == "__main__":
    main()
