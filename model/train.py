from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from model.common import DATA_DIR, ROOT_DIR, read_train_frame, seed_everything

DEFAULT_MODEL_NAME = "google/gemma-2-27b-it"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "gemma-2-27b-it"
DEFAULT_SYSTEM_PROMPT = "You are an expert translator from Akkadian transliteration to English."
DEFAULT_USER_PROMPT_TEMPLATE = (
    "Translate the following Akkadian transliteration into English.\n"
    "Return only the English translation.\n\n"
    "Akkadian transliteration:\n{source}"
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


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def maybe_import_peft() -> tuple[Any, Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "LoRA/QLoRA training requires `peft`. Install it with `pip install peft`."
        ) from exc

    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def maybe_make_quantization_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if not args.use_qlora:
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "QLoRA training requires `bitsandbytes`. Install it with `pip install bitsandbytes`."
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )


class SupervisedDataCollator:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = [
            label + [-100] * (max_length - len(label))
            for label in labels
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune google/gemma-3-27b-it for Akkadian transliteration to English."
    )
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt-template", type=str, default=DEFAULT_USER_PROMPT_TEMPLATE)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--fp16", type=parse_bool, default=False)
    parser.add_argument("--tf32", type=parse_bool, default=True)
    parser.add_argument("--torch-dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--train-on-inputs", type=parse_bool, default=False)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--remove-unused-columns", type=parse_bool, default=False)
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--use-qlora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=parse_list,
        default=parse_list("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
    )
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--lora-modules-to-save", type=parse_list, default=None)
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", type=parse_bool, default=True)
    parser.add_argument("--bnb-4bit-compute-dtype", type=parse_optional_torch_dtype, default=torch.bfloat16)
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


def build_messages(source_text: str, target_text: str | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": args.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": args.user_prompt_template.format(source=source_text)}],
        },
    ]
    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    return messages


def main() -> None:
    args = parse_args()
    if args.use_qlora:
        args.use_lora = True
    seed_everything(args.seed)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

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

    train_dataset = Dataset.from_pandas(train_frame.reset_index(drop=True), preserve_index=False)
    eval_dataset = Dataset.from_pandas(val_frame.reset_index(drop=True), preserve_index=False)

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = maybe_make_quantization_config(args)
    model_kwargs: dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = Gemma3ForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora or args.use_qlora:
        LoraConfig, get_peft_model, prepare_model_for_kbit_training = maybe_import_peft()
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        elif args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
            modules_to_save=args.lora_modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def preprocess(example: dict[str, str]) -> dict[str, list[int]]:
        source_text = example["transliteration"]
        target_text = example["translation"]

        prompt_text = processor.apply_chat_template(
            build_messages(source_text, None, args),
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            build_messages(source_text, target_text, args),
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=False,
        )
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()

        if not args.train_on_inputs:
            prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
            labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset",
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
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
        report_to=report_to,
        run_name=args.wandb_run_name,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=args.remove_unused_columns,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        save_safetensors=True,
        seed=args.seed,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=SupervisedDataCollator(processor),
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    trainer.save_state()

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
