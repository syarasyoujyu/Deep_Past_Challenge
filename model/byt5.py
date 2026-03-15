from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import sacrebleu
import torch
from bert_score import score as bert_score
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from model.common import (
    DATA_DIR,
    ROOT_DIR,
    build_generation_config,
    load_tokenizer,
    read_train_frame,
    seed_everything,
)

DEFAULT_MODEL_NAME = "google/byt5-small"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "byt5-small"

SUBSCRIPT_TRANSLATION_TABLE = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "ₓ": "x",
    }
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_optional_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise argparse.ArgumentTypeError(
            f"invalid torch dtype: {value}. expected one of auto, float32, float16, bfloat16"
        )
    return mapping[normalized]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune google/byt5-* for Akkadian transliteration to English."
    )
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--torch-dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--normalize-source", type=parse_bool, default=True)
    parser.add_argument("--normalize-target", type=parse_bool, default=True)
    parser.add_argument("--normalize-h", type=parse_bool, default=True)
    parser.add_argument("--normalize-subscripts", type=parse_bool, default=True)
    parser.add_argument("--normalize-breaks", type=parse_bool, default=True)
    parser.add_argument("--remove-editorial-marks", type=parse_bool, default=True)
    parser.add_argument("--strip-word-dividers", type=parse_bool, default=False)
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--disable-wandb", type=parse_bool, default=True)
    parser.add_argument("--push-to-hub", type=parse_bool, default=False)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private-repo", type=parse_bool, default=True)
    parser.add_argument("--hub-strategy", type=str, default="end")
    return parser.parse_args()


def normalize_transliteration(text: str, args: argparse.Namespace) -> str:
    text = text.replace("\u00a0", " ").strip()

    if args.normalize_h:
        text = text.replace("Ḫ", "H").replace("ḫ", "h")

    if args.normalize_subscripts:
        text = text.translate(SUBSCRIPT_TRANSLATION_TABLE)

    if args.normalize_breaks:
        text = re.sub(r"\[\s*[xX]\s*\]", " <gap> ", text)
        text = re.sub(r"\[\s*(?:…|\.\.\.)+\s*\]", " <big_gap> ", text)
        text = text.replace("…", " <big_gap> ")
        text = re.sub(r"\[([^\[\]]+)\]", r" \1 ", text)

    if args.remove_editorial_marks:
        text = re.sub(r"[!?/]", " ", text)
        text = text.replace("˹", "").replace("˺", "")

    if args.strip_word_dividers:
        text = text.replace(":", " ").replace(".", " ")

    return " ".join(text.split())


def normalize_translation(text: str) -> str:
    text = text.replace("\u00a0", " ").strip()
    return " ".join(text.split())


def prepare_frame(args: argparse.Namespace):
    frame = read_train_frame(args.train_path)
    frame = frame.copy()

    if args.normalize_source:
        frame["transliteration"] = frame["transliteration"].map(
            lambda text: normalize_transliteration(text, args)
        )
    if args.normalize_target:
        frame["translation"] = frame["translation"].map(normalize_translation)

    frame = frame[
        (frame["transliteration"].astype(str).str.strip() != "")
        & (frame["translation"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    return frame


def split_frame(frame, val_size: float, seed: int):
    if val_size <= 0:
        return frame, None

    train_frame, val_frame = train_test_split(
        frame,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )
    return train_frame, val_frame


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to: list[str] = []
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        report_to = [value for value in args.report_to.split(",") if value]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_frame(args)
    train_frame, val_frame = split_frame(frame, args.val_size, args.seed)

    dataset_dict = {
        "train": Dataset.from_pandas(train_frame.reset_index(drop=True), preserve_index=False)
    }
    has_validation = val_frame is not None and not val_frame.empty
    if has_validation:
        dataset_dict["validation"] = Dataset.from_pandas(
            val_frame.reset_index(drop=True), preserve_index=False
        )
    dataset = DatasetDict(dataset_dict)

    tokenizer = load_tokenizer(args.model_name)
    model_kwargs = {"attn_implementation": args.attn_implementation}
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        inputs = [f"{args.source_prefix}{text}" for text in batch["transliteration"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["translation"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    generation_config = build_generation_config(args)
    pad_to_multiple_of = 8 if args.fp16 or args.bf16 else None
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        chrfpp = sacrebleu.corpus_chrf(
            decoded_predictions,
            [decoded_labels],
            word_order=2,
        ).score
        _, _, bertscore_f1 = bert_score(
            decoded_predictions,
            decoded_labels,
            lang="en",
            model_type=args.bertscore_model_type,
            batch_size=args.bertscore_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        bertscore = float(bertscore_f1.mean().item())
        geometric_mean = math.sqrt(max(bertscore, 0.0) * max(chrfpp / 100.0, 0.0)) * 100.0
        prediction_lengths = [
            np.count_nonzero(prediction != tokenizer.pad_token_id) for prediction in predictions
        ]
        return {
            "bertscore": round(bertscore, 4),
            "chrfpp": round(float(chrfpp), 4),
            "bertscore_chrfpp_geometric_mean": round(float(geometric_mean), 4),
            "gen_len": round(float(np.mean(prediction_lengths)), 4),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=has_validation,
        predict_with_generate=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if has_validation else "no",
        save_strategy="steps" if has_validation else "epoch",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation,
        metric_for_best_model="bertscore_chrfpp_geometric_mean" if has_validation else None,
        greater_is_better=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        label_smoothing_factor=args.label_smoothing_factor,
        report_to=report_to,
        run_name=args.wandb_run_name,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_validation else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if has_validation else None,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    if has_validation:
        eval_metrics = trainer.evaluate(**generation_config)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
