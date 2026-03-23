from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from model.byt5 import (
    debug_first_batch,
    normalize_translation,
    normalize_transliteration,
    parse_bool,
    parse_interval_strategy,
    parse_optional_torch_dtype,
    resolve_interval_strategy,
    sanitize_token_ids,
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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "byt5-small-expanded-distill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a ByT5 checkpoint by adding encoder/decoder layers, then continue "
            "training with a frozen teacher via sequence-to-sequence distillation."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--load-trained-model", type=parse_bool, default=False)
    parser.add_argument("--trained-model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", type=parse_interval_strategy, default="epoch")
    parser.add_argument("--save-strategy", type=parse_interval_strategy, default="epoch")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--dtype", dest="dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--torch-dtype", dest="dtype", type=parse_optional_torch_dtype)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--normalize-source", type=parse_bool, default=True)
    parser.add_argument("--normalize-target", type=parse_bool, default=True)
    parser.add_argument("--normalize-h", type=parse_bool, default=True)
    parser.add_argument("--normalize-subscripts", type=parse_bool, default=True)
    parser.add_argument("--normalize-breaks", type=parse_bool, default=True)
    parser.add_argument("--remove-editorial-marks", type=parse_bool, default=True)
    parser.add_argument("--strip-word-dividers", type=parse_bool, default=False)
    parser.add_argument("--debug-first-batch", type=parse_bool, default=True)
    parser.add_argument("--disable-wandb", type=parse_bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--push-to-hub", type=parse_bool, default=False)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private-repo", type=parse_bool, default=True)
    parser.add_argument("--hub-strategy", type=str, default="end")
    parser.add_argument("--teacher-device", type=str, default="auto")
    parser.add_argument("--extra-encoder-layers", type=int, default=2)
    parser.add_argument("--extra-decoder-layers", type=int, default=2)
    parser.add_argument("--copy-from-last-n-encoder-layers", type=int, default=1)
    parser.add_argument("--copy-from-last-n-decoder-layers", type=int, default=1)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--freeze-shared-embeddings", type=parse_bool, default=False)
    parser.add_argument("--freeze-bottom-encoder-layers", type=int, default=0)
    parser.add_argument("--freeze-bottom-decoder-layers", type=int, default=0)
    parser.add_argument("--train-lm-head", type=parse_bool, default=True)
    parser.add_argument("--train-final-layer-norms", type=parse_bool, default=True)
    return parser.parse_args()


def resolve_model_source(args: argparse.Namespace) -> str:
    if args.load_trained_model:
        if args.trained_model_path is None:
            raise ValueError(
                "--load-trained-model true requires --trained-model-path to be set."
            )
        return str(args.trained_model_path)
    return args.model_name


def setup_reporting(args: argparse.Namespace) -> list[str]:
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    return [value.strip() for value in args.report_to.split(",") if value.strip()]


def prepare_frame(args: argparse.Namespace):
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


def load_teacher_model(args: argparse.Namespace):
    model_source = resolve_model_source(args)
    model_kwargs: dict[str, Any] = {"attn_implementation": args.attn_implementation}
    if args.dtype is not None:
        model_kwargs["dtype"] = args.dtype
    return AutoModelForSeq2SeqLM.from_pretrained(model_source, **model_kwargs)


def repeat_block_sources(block_count: int, extra_layers: int, copy_from_last_n: int) -> list[int]:
    if extra_layers <= 0:
        return []
    if block_count <= 0:
        raise ValueError("teacher model has no blocks to copy from")

    copy_from_last_n = max(1, min(copy_from_last_n, block_count))
    source_indices = list(range(block_count - copy_from_last_n, block_count))
    return [source_indices[index % len(source_indices)] for index in range(extra_layers)]


def build_expanded_student(teacher_model, args: argparse.Namespace):
    student_config = copy.deepcopy(teacher_model.config)
    original_encoder_layers = int(getattr(teacher_model.config, "num_layers"))
    original_decoder_layers = int(getattr(teacher_model.config, "num_decoder_layers"))
    student_config.num_layers = original_encoder_layers + max(args.extra_encoder_layers, 0)
    student_config.num_decoder_layers = original_decoder_layers + max(args.extra_decoder_layers, 0)

    student_model = teacher_model.__class__(student_config)

    student_model.shared.load_state_dict(teacher_model.shared.state_dict())
    if hasattr(student_model, "lm_head") and hasattr(teacher_model, "lm_head"):
        student_model.lm_head.load_state_dict(teacher_model.lm_head.state_dict())

    student_model.encoder.embed_tokens.load_state_dict(teacher_model.encoder.embed_tokens.state_dict())
    student_model.decoder.embed_tokens.load_state_dict(teacher_model.decoder.embed_tokens.state_dict())
    student_model.encoder.final_layer_norm.load_state_dict(
        teacher_model.encoder.final_layer_norm.state_dict()
    )
    student_model.decoder.final_layer_norm.load_state_dict(
        teacher_model.decoder.final_layer_norm.state_dict()
    )

    for layer_index in range(original_encoder_layers):
        student_model.encoder.block[layer_index].load_state_dict(
            teacher_model.encoder.block[layer_index].state_dict()
        )
    for new_layer_index, source_layer_index in enumerate(
        repeat_block_sources(
            original_encoder_layers,
            args.extra_encoder_layers,
            args.copy_from_last_n_encoder_layers,
        ),
        start=original_encoder_layers,
    ):
        student_model.encoder.block[new_layer_index].load_state_dict(
            teacher_model.encoder.block[source_layer_index].state_dict()
        )

    for layer_index in range(original_decoder_layers):
        student_model.decoder.block[layer_index].load_state_dict(
            teacher_model.decoder.block[layer_index].state_dict()
        )
    for new_layer_index, source_layer_index in enumerate(
        repeat_block_sources(
            original_decoder_layers,
            args.extra_decoder_layers,
            args.copy_from_last_n_decoder_layers,
        ),
        start=original_decoder_layers,
    ):
        student_model.decoder.block[new_layer_index].load_state_dict(
            teacher_model.decoder.block[source_layer_index].state_dict()
        )

    if hasattr(student_model, "tie_weights"):
        student_model.tie_weights()

    expansion_summary = {
        "original_encoder_layers": original_encoder_layers,
        "original_decoder_layers": original_decoder_layers,
        "expanded_encoder_layers": int(student_config.num_layers),
        "expanded_decoder_layers": int(student_config.num_decoder_layers),
        "extra_encoder_layers": max(args.extra_encoder_layers, 0),
        "extra_decoder_layers": max(args.extra_decoder_layers, 0),
    }
    return student_model, expansion_summary


def set_module_trainable(module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def configure_student_trainability(student_model, args: argparse.Namespace) -> dict[str, float]:
    if args.freeze_shared_embeddings and hasattr(student_model, "shared"):
        set_module_trainable(student_model.shared, False)
        if hasattr(student_model.encoder, "embed_tokens"):
            set_module_trainable(student_model.encoder.embed_tokens, False)
        if hasattr(student_model.decoder, "embed_tokens"):
            set_module_trainable(student_model.decoder.embed_tokens, False)

    encoder_blocks = getattr(student_model.encoder, "block", [])
    for layer_index in range(min(args.freeze_bottom_encoder_layers, len(encoder_blocks))):
        set_module_trainable(encoder_blocks[layer_index], False)

    decoder_blocks = getattr(student_model.decoder, "block", [])
    for layer_index in range(min(args.freeze_bottom_decoder_layers, len(decoder_blocks))):
        set_module_trainable(decoder_blocks[layer_index], False)

    if not args.train_lm_head and hasattr(student_model, "lm_head"):
        set_module_trainable(student_model.lm_head, False)

    if not args.train_final_layer_norms:
        if hasattr(student_model.encoder, "final_layer_norm"):
            set_module_trainable(student_model.encoder.final_layer_norm, False)
        if hasattr(student_model.decoder, "final_layer_norm"):
            set_module_trainable(student_model.decoder.final_layer_norm, False)

    trainable_params = sum(parameter.numel() for parameter in student_model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in student_model.parameters())
    return {
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_ratio_percent": round((trainable_params / max(total_params, 1)) * 100.0, 4),
    }


def move_teacher_for_distillation(teacher_model, teacher_device: str):
    if teacher_device == "auto":
        if torch.cuda.is_available():
            teacher_device = "cuda"
        else:
            teacher_device = "cpu"
    resolved_device = torch.device(teacher_device)
    teacher_model.to(resolved_device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False
    return teacher_model, resolved_device


class DistillationSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        teacher_model,
        teacher_device: torch.device,
        distill_alpha: float,
        distill_temperature: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_device = teacher_device
        self.distill_alpha = float(distill_alpha)
        self.distill_temperature = float(distill_temperature)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.loss
        total_loss = ce_loss

        labels = inputs.get("labels")
        if (
            self.teacher_model is not None
            and self.distill_alpha > 0.0
            and labels is not None
        ):
            teacher_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    teacher_inputs[key] = value.to(self.teacher_device)
                else:
                    teacher_inputs[key] = value

            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)

            valid_mask = labels.ne(-100)
            if torch.any(valid_mask):
                student_logits = outputs.logits[valid_mask]
                teacher_logits = teacher_outputs.logits.to(outputs.logits.device)[valid_mask]
                temperature = max(self.distill_temperature, 1e-6)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                total_loss = (
                    (1.0 - self.distill_alpha) * ce_loss
                    + self.distill_alpha * distill_loss
                )

        return (total_loss, outputs) if return_outputs else total_loss


def build_compute_metrics(tokenizer, args: argparse.Namespace):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = sanitize_token_ids(predictions, tokenizer, "predictions")
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = sanitize_token_ids(labels, tokenizer, "labels")
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

    return compute_metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    report_to = setup_reporting(args)
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

    model_source = resolve_model_source(args)
    print(f"Loading teacher ByT5 from: {model_source}")
    tokenizer = load_tokenizer(model_source)
    teacher_model = load_teacher_model(args)
    student_model, expansion_summary = build_expanded_student(teacher_model, args)
    teacher_model, teacher_device = move_teacher_for_distillation(teacher_model, args.teacher_device)

    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False

    trainability_summary = configure_student_trainability(student_model, args)

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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        pad_to_multiple_of=None,
    )
    debug_metrics = debug_first_batch(tokenized["train"], data_collator, student_model, args)
    compute_metrics = build_compute_metrics(tokenizer, args)

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
        metric_for_best_model=(
            "bertscore_chrfpp_geometric_mean" if eval_strategy != "no" else None
        ),
        greater_is_better=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        label_smoothing_factor=args.label_smoothing_factor,
        report_to=report_to,
        run_name=args.wandb_run_name,
        fp16=False,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        seed=args.seed,
    )

    trainer = DistillationSeq2SeqTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_strategy != "no" else None,
        teacher_model=teacher_model,
        teacher_device=teacher_device,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
    )

    trainer.log(
        {
            **expansion_summary,
            **trainability_summary,
            "distill_alpha": float(args.distill_alpha),
            "distill_temperature": float(args.distill_temperature),
        }
    )
    if debug_metrics:
        trainer.log(debug_metrics)

    trainer.train()
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
