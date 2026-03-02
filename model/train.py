from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
import numpy as np
import sacrebleu
import torch
from bert_score import score as bert_score
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from transformers import EvalPrediction
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from model.common import DATA_DIR, ROOT_DIR, read_train_frame, seed_everything

DEFAULT_MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "gemma-2-9b-unsloth"
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


def parse_auto_int(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "auto":
        cpu_count = os.cpu_count() or 1
        return max(cpu_count - 1, 1)
    return int(value)


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


def resolve_model_load_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if args.torch_dtype is not None:
        return args.torch_dtype
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune instruction-tuned causal LMs for Akkadian transliteration to English with Unsloth."
    )
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt-template", type=str, default=DEFAULT_USER_PROMPT_TEMPLATE)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--preprocess-num-proc", type=parse_auto_int, default=1)
    parser.add_argument("--dataloader-num-workers", type=parse_auto_int, default=0)
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
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--fp16", type=parse_bool, default=False)
    parser.add_argument("--tf32", type=parse_bool, default=True)
    parser.add_argument("--torch-dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--train-on-inputs", type=parse_bool, default=False)
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--remove-unused-columns", type=parse_bool, default=False)
    parser.add_argument("--packing", type=parse_bool, default=False)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--use-qlora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        type=parse_list,
        default=parse_list("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
    )
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--lora-modules-to-save", type=parse_list, default=None)
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--disable-wandb", type=parse_bool, default=True)
    parser.add_argument("--push-to-hub", type=parse_bool, default=False)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private-repo", type=parse_bool, default=True)
    return parser.parse_args()


def build_prompt(source_text: str, args: argparse.Namespace) -> str:
    user_prompt = args.user_prompt_template.format(source=source_text)
    if args.system_prompt:
        return f"{args.system_prompt}\n\n{user_prompt}"
    return user_prompt

def format_prompt_completion_batch(
    examples: dict[str, list[str]],
    args: argparse.Namespace,
    eos_token: str,
) -> dict[str, list[str]]:
    texts: list[str] = []
    for source_text, target_text in zip(examples["transliteration"], examples["translation"]):
        # プロンプトと回答を結合し、最後にEOSトークンを付与
        full_text = f"{build_prompt(source_text, args)}\n\n{target_text.rstrip()}{eos_token}"
        texts.append(full_text)
    return {"text": texts}


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.lora_modules_to_save is not None:
        raise ValueError("`--lora-modules-to-save` is not supported in the current Unsloth path.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=resolve_model_load_dtype(args),
        load_in_4bit=args.use_qlora,
        load_in_8bit=False,
        load_in_16bit=args.use_lora and not args.use_qlora,
        full_finetuning=not (args.use_lora or args.use_qlora),
        token=os.environ.get("HF_TOKEN"),
    )

    if args.use_lora or args.use_qlora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            max_seq_length=args.max_seq_length,
            use_rslora=False,
            loftq_config=None,
        )
        model.print_trainable_parameters()

    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


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


def build_dataset(frame, args: argparse.Namespace, tokenizer, desc: str) -> Dataset:
    dataset = Dataset.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    
    def tokenize_function(examples):
        texts = [
            f"{build_prompt(s, args)}\n\n{t.rstrip()}{tokenizer.eos_token}"
            for s, t in zip(examples["transliteration"], examples["translation"])
        ]
        # ここでトークナイズを完結させる
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False, # Trainer側のデータコレーターに任せる
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1, # ここは必ず 1
        desc=desc,
    )

def preprocess_logits_for_metrics(logits, labels):
    """
    メモリ節約のための最重要関数。
    全語彙の確率(Logits)を保持せず、最大値のインデックス(Token ID)のみを返します。
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics_wrapper(eval_preds:EvalPrediction,tokenizer,args):
        """SFTTrainerから渡されるEvalPredictionを分解して計算関数に渡す"""
        # eval_preds.predictions: モデルの出力（通常はロジット。生成タスクではデコード済みの場合も）
        # eval_preds.label_ids: 正解ラベル
        preds, labels = eval_preds.predictions,eval_preds.label_ids
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ※SFT（生成）の場合、ここでは単純なLoss計算以外が難しいことが多いため
        # 後の generate_validation_predictions で一括計算する流れが一般的です。
        # ひとまずエラーを消すには、型を合わせる必要があります。
        return compute_generation_metrics(decoded_preds, decoded_labels, args=args)

def generate_validation_predictions(model, tokenizer, frame, args: argparse.Namespace) -> list[str]:
    FastLanguageModel.for_inference(model)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_size = args.per_device_eval_batch_size
    device = next(model.parameters()).device
    predictions: list[str] = []

    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        prompts = [build_prompt(text, args) for text in batch["transliteration"].tolist()]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.eval_max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_length = encoded["input_ids"].shape[1]
        continuations = generated[:, prompt_length:]
        decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        predictions.extend(prediction.strip() for prediction in decoded)

    tokenizer.padding_side = original_padding_side
    return predictions


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    cleaned_predictions = [prediction.strip() for prediction in predictions]
    cleaned_references = [reference.strip() for reference in references]

    bleu = float(sacrebleu.corpus_bleu(cleaned_predictions, [cleaned_references]).score)
    chrfpp = float(
        sacrebleu.corpus_chrf(cleaned_predictions, [cleaned_references], word_order=2).score
    )

    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=False)
    rouge2_precision: list[float] = []
    rouge2_recall: list[float] = []
    rouge2_f1: list[float] = []
    for prediction, reference in zip(cleaned_predictions, cleaned_references):
        rouge2 = scorer.score(reference, prediction)["rouge2"]
        rouge2_precision.append(float(rouge2.precision))
        rouge2_recall.append(float(rouge2.recall))
        rouge2_f1.append(float(rouge2.fmeasure))

    _, _, bertscore_f1 = bert_score(
        cleaned_predictions,
        cleaned_references,
        lang="en",
        model_type=args.bertscore_model_type,
        batch_size=args.bertscore_batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    geometric_mean = math.sqrt((bleu / 100.0) * (chrfpp / 100.0)) * 100.0
    return {
        "eval_bleu": round(bleu, 4),
        "eval_rouge2_precision": round(float(np.mean(rouge2_precision)), 4),
        "eval_rouge2_recall": round(float(np.mean(rouge2_recall)), 4),
        "eval_rouge2_f1": round(float(np.mean(rouge2_f1)), 4),
        "eval_bertscore": round(float(bertscore_f1.mean().item()), 4),
        "eval_chrfpp": round(chrfpp, 4),
        "eval_bleu_chrfpp_geometric_mean": round(geometric_mean, 4),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("`--hub-model-id` is required when `--push-to-hub true`.")

    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to: str | list[str] = "none"
    else:
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = os.environ.get("WANDB_ENTITY", args.wandb_entity)
        report_to = [value for value in args.report_to.split(",") if value]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_train_frame(args.train_path)
    train_frame, val_frame = split_frame(frame, args.val_size, args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = build_dataset(train_frame, args, tokenizer, desc="Formatting train dataset")
    eval_dataset = None
    if val_frame is not None:
        eval_dataset = build_dataset(val_frame, args, tokenizer, desc="Formatting eval dataset")

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=args.wandb_run_name,
        bf16=args.bf16,
        fp16=args.fp16,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=args.remove_unused_columns,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        seed=args.seed,
        completion_only_loss=not args.train_on_inputs,
        packing=args.packing,
        dataset_num_proc=1,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        load_best_model_at_end=True, # 学習終了時に最強モデルをロード
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=eval_dataset,
        compute_metrics=lambda x:compute_metrics_wrapper(x,tokenizer=tokenizer,args=args),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=sft_args,
    )
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    trainer.train()
    if val_frame is not None and not val_frame.empty:
        metrics = compute_generation_metrics(
            predictions=generate_validation_predictions(model, tokenizer, val_frame, args),
            references=val_frame["translation"].tolist(),
            args=args,
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        model.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    main()
