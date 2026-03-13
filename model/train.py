from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from unsloth import FastLanguageModel

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd
import sacrebleu
import torch
from bert_score import score as bert_score
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from transformers import EvalPrediction
from trl import SFTConfig, SFTTrainer

from model.common import DATA_DIR, ROOT_DIR, read_train_frame, seed_everything
from refine.refine_train_v2 import postprocessor as translation_postprocessor

DEFAULT_MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "gemma-2-9b-unsloth"
DEFAULT_SYSTEM_PROMPT = "You are a professional Akkadian translation expert skilled in functional equivalence translation."
DEFAULT_USER_PROMPT_TEMPLATE = (
    "As an Akkadian translation expert, please translate the following text into English "
    "following functional equivalence principles:\n\n"
    "Translation requirements:\n"
    "1. Accurately convey the semantic content of the original text\n"
    "2. Consider cultural adaptation (convert unfamiliar concepts to target-culture "
    "understandable concepts)\n"
    "3. Maintain natural and fluent language\n\n"
    "Original text: \"{source_text}\"\n\n"
    "Please only provide the translation:\n"
)

def simple_sentence_splitter(text, max_length=200):
    """
    古代テキストの特徴を考慮したシンプルな文分割関数。
    - 改行や複数スペースで分割
    - 長すぎる文はスペースで分割して複数の文にする
    """
    if pd.isna(text):
        return []

    separators = [
        '\n',
        "  ", # 半角スペース×2
    ]
    
    sentences = [text]
    for sep in separators:
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(sent.split(sep))
        sentences = new_sentences
    
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Split overly long sentences at logical break points
    final_sentences = []
    for sent in sentences:
        if len(sent) <= max_length:
            final_sentences.append(sent)
        else:
            # Split long sentences at spaces
            words = sent.split()
            current = []
            for word in words:
                current.append(word)
                if len(' '.join(current)) > max_length:
                    final_sentences.append(' '.join(current[:-1]))
                    current = [word]
            if current:
                final_sentences.append(' '.join(current))
    
    return final_sentences


def setup_wandb(args: argparse.Namespace) -> list[str]:
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    return [value.strip() for value in args.report_to.split(",") if value.strip()]


def postprocess_translations(texts: list[str]) -> list[str]:
    if not texts:
        return []
    return translation_postprocessor.postprocess_batch(texts)

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
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1)
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
    user_prompt = args.user_prompt_template.format(
        source=source_text,
        source_text=source_text,
    )
    if args.system_prompt:
        return f"{args.system_prompt}\n\n{user_prompt}"
    return user_prompt


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


def preprocess_logits_for_metrics(logits, labels):
    """VRAM節約のため、評価時に確率分布(Logits)からToken ID(argmax)へ即座に変換する"""
    if isinstance(logits, tuple):
        logits = logits[0]
    # argmaxを取り、整数型(int32)に変換して返す
    return logits.argmax(dim=-1).to(torch.int32)

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

def generate_predictions_for_texts(
    model,
    tokenizer,
    source_texts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    FastLanguageModel.for_inference(model)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_size = args.per_device_eval_batch_size
    device = next(model.parameters()).device
    predictions: list[str] = []

    for start in range(0, len(source_texts), batch_size):
        batch_texts = source_texts[start : start + batch_size]
        prompts = [build_prompt(text, args) for text in batch_texts]
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


def generate_validation_predictions(model, tokenizer, frame, args: argparse.Namespace) -> list[str]:
    source_texts = frame["transliteration"].fillna("").astype(str).tolist()
    return generate_predictions_for_texts(model, tokenizer, source_texts, args)


def build_validation_prediction_records(
    model,
    tokenizer,
    frame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    transliterations = frame["transliteration"].fillna("").astype(str).tolist()
    references = frame["translation"].fillna("").astype(str).tolist()
    raw_plain_predictions = generate_predictions_for_texts(model, tokenizer, transliterations, args)
    plain_predictions = postprocess_translations(raw_plain_predictions)

    split_inputs_per_sample = [simple_sentence_splitter(text) for text in transliterations]
    flat_split_inputs = [segment for segments in split_inputs_per_sample for segment in segments]
    raw_flat_split_predictions = (
        generate_predictions_for_texts(model, tokenizer, flat_split_inputs, args)
        if flat_split_inputs
        else []
    )
    flat_split_predictions = postprocess_translations(raw_flat_split_predictions)

    records: list[dict[str, Any]] = []
    merged_split_predictions: list[str] = []
    split_cursor = 0

    for transliteration, reference, raw_plain_prediction, plain_prediction, split_inputs in zip(
        transliterations,
        references,
        raw_plain_predictions,
        plain_predictions,
        split_inputs_per_sample,
    ):
        split_count = len(split_inputs)
        raw_split_predictions = raw_flat_split_predictions[split_cursor : split_cursor + split_count]
        split_predictions = flat_split_predictions[split_cursor : split_cursor + split_count]
        split_cursor += split_count

        raw_merged_split_prediction = " ".join(
            prediction.strip() for prediction in raw_split_predictions if prediction.strip()
        ).strip()
        merged_split_prediction = postprocess_translations([raw_merged_split_prediction])[0]
        merged_split_predictions.append(merged_split_prediction)

        records.append(
            {
                "transliteration": transliteration,
                "translation_reference": reference,
                "raw_prediction_without_split": raw_plain_prediction,
                "prediction_without_split": plain_prediction,
                "split_applied": split_count > 1,
                "split_segment_count": split_count,
                "split_transliterations": " || ".join(split_inputs),
                "raw_split_predictions": " || ".join(raw_split_predictions),
                "split_predictions": " || ".join(split_predictions),
                "raw_prediction_with_split": raw_merged_split_prediction,
                "prediction_with_split": merged_split_prediction,
            }
        )

    return records, plain_predictions, merged_split_predictions


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    args: argparse.Namespace,
    metric_prefix: str = "eval",
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
    logs={
        f"{metric_prefix}_bleu": round(bleu, 4),
        f"{metric_prefix}_rouge2_precision": round(float(np.mean(rouge2_precision)), 4),
        f"{metric_prefix}_rouge2_recall": round(float(np.mean(rouge2_recall)), 4),
        f"{metric_prefix}_rouge2_f1": round(float(np.mean(rouge2_f1)), 4),
        f"{metric_prefix}_bertscore": round(float(bertscore_f1.mean().item()), 4),
        f"{metric_prefix}_chrfpp": round(chrfpp, 4),
        f"{metric_prefix}_bleu_chrfpp_geometric_mean": round(geometric_mean, 4),
    }
    return logs


def log_validation_prediction_records(
    records: list[dict[str, Any]],
    output_dir: Path,
    disable_wandb: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "validation_prediction_logs.csv"
    log_frame = pd.DataFrame(records)
    log_frame.to_csv(log_path, index=False)
    print(f"Saved validation prediction logs to {log_path}")

    if disable_wandb:
        return

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipped validation prediction table logging.")
        return

    if wandb.run is None:
        return

    wandb.log({"eval/prediction_table": wandb.Table(dataframe=log_frame)})

def warp_metric(p, tokenizer, args):
    """SFTTrainerから渡される数値をテキストに変換し、メトリクス計算へ橋渡しする"""
    preds, labels = p.predictions, p.label_ids

    # 1. 予測値の次元調整 (安全策)
    # [batch, seq, 1] のような形であれば最後の1を消す
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = np.squeeze(preds, axis=-1)
    
    # 2. 正解ラベルの調整 (-100 を Pad ID に変換)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 3. デコード処理
    # Unsloth環境では numpy 配列がそのまま渡されるため、整数であることを確認してデコード
    decoded_preds = tokenizer.batch_decode(preds.astype(np.int32), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.astype(np.int32), skip_special_tokens=True)
    decoded_preds = postprocess_translations([pred.strip() for pred in decoded_preds])

    print(f"Evaluation: Decoded {len(decoded_preds)} samples.")
    return compute_generation_metrics(decoded_preds, decoded_labels, args)

def build_dataset(frame, args, tokenizer, desc):
    """データセットをテキスト形式で作成（SFTTrainerの標準方式）"""
    dataset = Dataset.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    eos_token = tokenizer.eos_token or ""

    def format_func(examples):
        texts = []
        for s, t in zip(examples["transliteration"], examples["translation"]):
            full_text = f"{build_prompt(s, args)}\n\n{t.rstrip()}{eos_token}"
            texts.append(full_text)
        return {"text": texts}

    return dataset.map(format_func, batched=True, desc=desc)

def load_model_and_tokenizer(args):
    """UnslothモデルのロードとLoRA設定"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None, # 自動検出
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # 最適化のため0
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )
    # 学習可能パラメータが0でないか確認
    model.print_trainable_parameters()
    return model, tokenizer

def main():
    args = parse_args() # 既存の引数パース関数
    seed_everything(args.seed)
    report_to = setup_wandb(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    
    frame = read_train_frame(args.train_path)
    train_frame, val_frame = split_frame(frame, args.val_size, args.seed)
    
    train_dataset = build_dataset(train_frame, args, tokenizer, "Formatting train dataset")
    eval_dataset = (
        build_dataset(val_frame, args, tokenizer, "Formatting eval dataset")
        if val_frame is not None
        else None
    )

    # 公式ノートブックに準拠したシンプルな設定
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",         # text列を使用することを明示
        remove_unused_columns=False,       # モデルが知らない列(text等)を消さないようにする
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps", 
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        fp16=False, # Gemma 3 のログに従い float32/bf16 を優先
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to=report_to,
        run_name=args.wandb_run_name,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
        load_best_model_at_end=True,
    )

    # ログにある警告に基づき、trainの直前に環境変数をセット
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1" 
    
    trainer.train()

    # --- 学習完了後の一括評価（VRAMを節約しつつ正確な指標を出す） ---
    if val_frame is not None:
        print("Calculating final evaluation metrics...")
        records, plain_predictions, split_predictions = build_validation_prediction_records(
            model,
            tokenizer,
            val_frame,
            args,
        )
        metrics = compute_generation_metrics(
            predictions=plain_predictions,
            references=val_frame["translation"].tolist(),
            args=args,
        )
        split_metrics = compute_generation_metrics(
            predictions=split_predictions,
            references=val_frame["translation"].tolist(),
            args=args,
            metric_prefix="eval_split",
        )
        metrics.update(split_metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        log_validation_prediction_records(records, args.output_dir, args.disable_wandb)
        print(metrics)

    # 保存
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        model.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    main()
