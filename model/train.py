from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import sacrebleu
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hugging Face seq2seq model for Akkadian translation.")
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--model-name", type=str, default="google/byt5-small")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "artifacts" / "byt5-small")
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=8.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--disable-wandb", type=bool, default=True)
    parser.add_argument("--push-to-hub", type=bool, default=True)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private-repo", type=bool, default=True)
    parser.add_argument("--hub-strategy", type=str, default="end")
    return parser.parse_args()


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

    frame = read_train_frame(args.train_path)
    train_frame, val_frame = train_test_split(
        frame,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_frame.reset_index(drop=True), preserve_index=False),
            "validation": Dataset.from_pandas(val_frame.reset_index(drop=True), preserve_index=False),
        }
    )

    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        inputs = [f"{args.source_prefix}{text}" for text in batch["transliteration"]]
        labels = batch["translation"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
        )
        label_tokens = tokenizer(
            text_target=labels,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = label_tokens["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    generation_config = build_generation_config(args)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [pred.strip() for pred in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu_score = sacrebleu.corpus_bleu(decoded_predictions, [decoded_labels]).score
        chrf_score = sacrebleu.corpus_chrf(decoded_predictions, [decoded_labels]).score

        prediction_lengths = [
            np.count_nonzero(prediction != tokenizer.pad_token_id) for prediction in predictions
        ]
        return {
            "bleu": round(float(bleu_score), 4),
            "chrf": round(float(chrf_score), 4),
            "gen_len": round(float(np.mean(prediction_lengths)), 4),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=report_to,
        run_name=args.wandb_run_name,
        bf16=args.bf16,
        fp16=args.fp16,
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
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(max_length=args.max_target_length, **generation_config)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
