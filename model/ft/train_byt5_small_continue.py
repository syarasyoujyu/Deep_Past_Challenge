from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import sacrebleu
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from model.byt5 import (
    normalize_translation,
    normalize_transliteration,
    parse_bool,
    parse_interval_strategy,
    parse_optional_torch_dtype,
    resolve_interval_strategy,
)
from model.common import (
    ROOT_DIR,
    build_generation_config,
    load_tokenizer,
    read_train_frame,
    seed_everything,
)


DEFAULT_BASE_MODEL_NAME = "google/byt5-small"
DEFAULT_LOCAL_MODEL_DIR = ROOT_DIR / "artifacts" / "byt5-small"
DEFAULT_TRAIN_PATH = (
    ROOT_DIR
    / "data"
    / "supplement"
    / "Michel_Old_Assyrian_Letters_Corpus"
    / "train_refined_v2_sentence_split_refined_refined.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "byt5-small-continued"


def resolve_default_model_path() -> str:
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_BASE_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue fine-tuning ByT5-small on a small Akkadian-English dataset with "
            "conservative defaults to reduce catastrophic forgetting."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--model-name-or-path", type=str, default=resolve_default_model_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--min-eval-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=12.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", type=parse_interval_strategy, default="epoch")
    parser.add_argument("--save-strategy", type=parse_interval_strategy, default="epoch")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.05)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--dtype", dest="dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--torch-dtype", dest="dtype", type=parse_optional_torch_dtype)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--normalize-source", type=parse_bool, default=True)
    parser.add_argument("--normalize-target", type=parse_bool, default=True)
    parser.add_argument("--normalize-h", type=parse_bool, default=True)
    parser.add_argument("--normalize-subscripts", type=parse_bool, default=True)
    parser.add_argument("--normalize-breaks", type=parse_bool, default=True)
    parser.add_argument("--remove-editorial-marks", type=parse_bool, default=True)
    parser.add_argument("--strip-word-dividers", type=parse_bool, default=False)
    parser.add_argument("--deduplicate-exact-pairs", type=parse_bool, default=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--freeze-shared-embeddings", type=parse_bool, default=True)
    parser.add_argument("--freeze-encoder", type=parse_bool, default=True)
    parser.add_argument("--unfreeze-last-n-decoder-blocks", type=int, default=2)
    parser.add_argument("--train-decoder-final-layer-norm", type=parse_bool, default=True)
    parser.add_argument("--train-lm-head", type=parse_bool, default=True)
    parser.add_argument("--metric-for-best-model", type=str, default="chrfpp")
    parser.add_argument("--greater-is-better", type=parse_bool, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--disable-wandb", type=parse_bool, default=True)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--push-to-hub", type=parse_bool, default=False)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private-repo", type=parse_bool, default=True)
    parser.add_argument("--hub-strategy", type=str, default="end")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_reporting(args: argparse.Namespace) -> list[str]:
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    return [value.strip() for value in args.report_to.split(",") if value.strip()]


def prepare_frame(args: argparse.Namespace) -> pd.DataFrame:
    frame = read_train_frame(args.train_path).copy()

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

    if args.deduplicate_exact_pairs:
        frame = frame.drop_duplicates(
            subset=["transliteration", "translation"], keep="first"
        ).reset_index(drop=True)

    if args.max_train_samples is not None and args.max_train_samples < len(frame):
        frame = frame.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)

    return frame


def build_stratify_labels(frame: pd.DataFrame) -> pd.Series | None:
    if len(frame) < 50:
        return None

    lengths = frame["translation"].astype(str).map(lambda text: len(text.split()))
    bin_count = min(5, max(2, len(frame) // 100))
    try:
        labels = pd.qcut(lengths.rank(method="first"), q=bin_count, duplicates="drop")
    except ValueError:
        return None

    value_counts = labels.value_counts()
    if value_counts.empty or int(value_counts.min()) < 2:
        return None
    return labels.astype(str)


def resolve_validation_size(args: argparse.Namespace, row_count: int) -> int:
    if row_count <= 1 or args.val_size <= 0:
        return 0

    if args.val_size < 1:
        requested = int(round(row_count * args.val_size))
    else:
        requested = int(args.val_size)

    requested = max(requested, args.min_eval_examples)
    requested = min(requested, row_count - 1)
    return max(requested, 0)


def split_frame(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    eval_size = resolve_validation_size(args, len(frame))
    if eval_size == 0:
        return frame.reset_index(drop=True), None

    stratify_labels = build_stratify_labels(frame)
    train_frame, val_frame = train_test_split(
        frame,
        test_size=eval_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def load_model(args: argparse.Namespace):
    model_kwargs: dict[str, Any] = {"attn_implementation": args.attn_implementation}
    if args.dtype is not None:
        model_kwargs["dtype"] = args.dtype

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.dropout_rate is not None:
        for attr in ("dropout_rate", "dropout", "attention_dropout", "classifier_dropout"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, args.dropout_rate)

    return model


def set_module_trainable(module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def configure_trainable_parameters(model, args: argparse.Namespace) -> dict[str, int]:
    if args.freeze_shared_embeddings and hasattr(model, "shared"):
        set_module_trainable(model.shared, False)

    if args.freeze_encoder and hasattr(model, "encoder"):
        set_module_trainable(model.encoder, False)

    if hasattr(model, "decoder"):
        set_module_trainable(model.decoder, False)
        decoder_blocks = getattr(model.decoder, "block", [])
        if args.unfreeze_last_n_decoder_blocks > 0:
            for block in decoder_blocks[-args.unfreeze_last_n_decoder_blocks :]:
                set_module_trainable(block, True)
        if args.train_decoder_final_layer_norm and hasattr(model.decoder, "final_layer_norm"):
            set_module_trainable(model.decoder.final_layer_norm, True)

    if args.train_lm_head and hasattr(model, "lm_head"):
        set_module_trainable(model.lm_head, True)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_ratio_percent": round((trainable_params / max(total_params, 1)) * 100.0, 4),
    }


def build_dataset_dict(train_frame: pd.DataFrame, val_frame: pd.DataFrame | None) -> DatasetDict:
    dataset_dict = {
        "train": Dataset.from_pandas(train_frame, preserve_index=False),
    }
    if val_frame is not None and not val_frame.empty:
        dataset_dict["validation"] = Dataset.from_pandas(val_frame, preserve_index=False)
    return DatasetDict(dataset_dict)


def count_non_padding_tokens(token_ids: np.ndarray, pad_token_id: int) -> list[int]:
    return [int(np.count_nonzero(token_row != pad_token_id)) for token_row in token_ids]


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        labels = np.where(labels != -100, labels, pad_token_id)

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu = sacrebleu.corpus_bleu(decoded_predictions, [decoded_labels]).score
        chrfpp = sacrebleu.corpus_chrf(decoded_predictions, [decoded_labels], word_order=2).score
        geometric_mean = math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))
        prediction_lengths = count_non_padding_tokens(predictions, pad_token_id)

        return {
            "bleu": round(float(bleu), 4),
            "chrfpp": round(float(chrfpp), 4),
            "bleu_chrfpp_geometric_mean": round(float(geometric_mean), 4),
            "gen_len": round(float(np.mean(prediction_lengths)), 4),
        }

    return compute_metrics


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame | None,
    parameter_summary: dict[str, int],
) -> None:
    payload = {
        "train_path": str(args.train_path),
        "model_name_or_path": args.model_name_or_path,
        "output_dir": str(output_dir),
        "row_count": int(len(frame)),
        "train_row_count": int(len(train_frame)),
        "validation_row_count": int(0 if val_frame is None else len(val_frame)),
        "deduplicate_exact_pairs": bool(args.deduplicate_exact_pairs),
        "freeze_shared_embeddings": bool(args.freeze_shared_embeddings),
        "freeze_encoder": bool(args.freeze_encoder),
        "unfreeze_last_n_decoder_blocks": int(args.unfreeze_last_n_decoder_blocks),
        "dropout_rate": float(args.dropout_rate),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_train_epochs": float(args.num_train_epochs),
        "label_smoothing_factor": float(args.label_smoothing_factor),
        "parameter_summary": parameter_summary,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    report_to = setup_reporting(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_frame(args)
    train_frame, val_frame = split_frame(frame, args)
    dataset = build_dataset_dict(train_frame, val_frame)
    has_validation = "validation" in dataset

    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_model(args)
    parameter_summary = configure_trainable_parameters(model, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print(
        "Prepared dataset: "
        f"total={len(frame)} train={len(train_frame)} validation={0 if val_frame is None else len(val_frame)}"
    )
    print(
        "Trainable parameters: "
        f"{parameter_summary['trainable_parameters']:,}/{parameter_summary['total_parameters']:,} "
        f"({parameter_summary['trainable_ratio_percent']}%)"
    )

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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    generation_config = build_generation_config(args)

    eval_strategy = resolve_interval_strategy(args.eval_strategy, has_validation, "eval")
    save_strategy = resolve_interval_strategy(args.save_strategy, has_validation, "save")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=eval_strategy != "no",
        predict_with_generate=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_strategy != "no",
        metric_for_best_model=args.metric_for_best_model if eval_strategy != "no" else None,
        greater_is_better=args.greater_is_better,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        label_smoothing_factor=args.label_smoothing_factor,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=True,
        report_to=report_to,
        run_name=args.wandb_run_name,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        seed=args.seed,
    )

    callbacks = []
    if eval_strategy != "no" and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer) if eval_strategy != "no" else None,
        callbacks=callbacks,
    )

    write_run_summary(args.output_dir, args, frame, train_frame, val_frame, parameter_summary)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    if eval_strategy != "no":
        eval_metrics = trainer.evaluate(**generation_config)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
