from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import sacrebleu
import torch
from bert_score import score as bert_score
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from model.common import DATA_DIR, ROOT_DIR, build_generation_config, load_tokenizer


DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "byt5-small"
DEFAULT_INPUT_PATH = DATA_DIR / "test.csv"
DEFAULT_SUBMISSION_PATH = ROOT_DIR / "submission.csv"
DEFAULT_EVAL_OUTPUT_PATH = ROOT_DIR / "artifacts" / "byt5-predictions.csv"
DEFAULT_METRICS_PATH = ROOT_DIR / "artifacts" / "byt5-predictions-metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ByT5 translation inference and optionally write row-level evaluation metrics."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--eval-output-path", type=Path, default=DEFAULT_EVAL_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_input_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()

    if "transliteration" not in frame.columns:
        raise ValueError("input CSV must contain a transliteration column")

    if "oare_id" in frame.columns:
        frame["oare_id"] = frame["oare_id"].fillna("").astype(str).str.strip()
    elif "id" in frame.columns:
        frame["oare_id"] = frame["id"].fillna("").astype(str).str.strip()
    else:
        frame["oare_id"] = frame.index.map(str)

    frame["transliteration"] = frame["transliteration"].fillna("").astype(str).str.strip()
    if "translation" in frame.columns:
        frame["translation"] = frame["translation"].fillna("").astype(str).str.strip()
    else:
        frame["translation"] = ""

    frame = frame[frame["transliteration"] != ""].reset_index(drop=True)
    return frame


def build_prediction_dataset(frame: pd.DataFrame, args: argparse.Namespace) -> Dataset:
    return Dataset.from_dict(
        {
            "oare_id": frame["oare_id"].tolist(),
            "input_text": [f"{args.source_prefix}{text}" for text in frame["transliteration"].tolist()],
        }
    )


def predict_translations(frame: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    dataset = build_prediction_dataset(frame, args)
    generation_config = build_generation_config(args)

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            truncation=True,
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    predict_args = Seq2SeqTrainingArguments(
        output_dir=str(args.eval_output_path.parent / ".predict_tmp"),
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        report_to=[],
        fp16=False,
        bf16=False,
        use_cpu=args.device == "cpu" or not torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=predict_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    prediction_output = trainer.predict(tokenized, **generation_config)
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    predictions = np.where(predictions != -100, predictions, pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return [prediction.strip() for prediction in decoded_predictions]


def compute_row_metrics(
    reference_texts: list[str],
    hypothesis_texts: list[str],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[float]]:
    _, _, bertscore_f1 = bert_score(
        hypothesis_texts,
        reference_texts,
        lang="en",
        model_type=args.bertscore_model_type,
        batch_size=args.bertscore_batch_size,
        device="cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    bertscores = [float(score.item()) for score in bertscore_f1]

    chrf_scores: list[float] = []
    geometric_means: list[float] = []
    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts, strict=True):
        sentence_bleu = float(sacrebleu.sentence_bleu(hypothesis_text, [reference_text]).score)
        sentence_chrf = float(sacrebleu.sentence_chrf(hypothesis_text, [reference_text], word_order=2).score)
        geometric_mean = math.sqrt((sentence_bleu / 100.0) * (sentence_chrf / 100.0)) * 100.0
        chrf_scores.append(sentence_chrf)
        geometric_means.append(geometric_mean)

    return bertscores, chrf_scores, geometric_means


def compute_corpus_metrics(reference_texts: list[str], hypothesis_texts: list[str]) -> dict[str, float]:
    bleu = float(sacrebleu.corpus_bleu(hypothesis_texts, [reference_texts]).score)
    chrfpp = float(sacrebleu.corpus_chrf(hypothesis_texts, [reference_texts], word_order=2).score)
    geometric_mean = math.sqrt((bleu / 100.0) * (chrfpp / 100.0)) * 100.0
    return {
        "_bleu": bleu,
        "chrf++": chrfpp,
        "_bleu_chrfpp_geometric_mean": geometric_mean,
    }


def write_submission(frame: pd.DataFrame, predictions: list[str], submission_path: Path) -> None:
    submission = pd.DataFrame(
        {
            "id": frame["oare_id"].astype(str),
            "translation": predictions,
        }
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)


def write_eval_csv(frame: pd.DataFrame, predictions: list[str], args: argparse.Namespace) -> None:
    reference_texts = frame["translation"].tolist()
    bertscores, chrf_scores, geometric_means = compute_row_metrics(reference_texts, predictions, args)
    eval_frame = pd.DataFrame(
        {
            "oare_id": frame["oare_id"].astype(str),
            "transliteration": frame["transliteration"].astype(str),
            "translation": predictions,
            "reference_translation": reference_texts,
            "bertscore": bertscores,
            "chrf++": chrf_scores,
            "_bleu_chrfpp_geometric_mean": geometric_means,
        }
    )
    args.eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_frame.to_csv(args.eval_output_path, index=False)

    corpus_metrics = compute_corpus_metrics(reference_texts, predictions)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(corpus_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    frame = load_input_frame(args.input_path)
    predictions = predict_translations(frame, args)

    write_submission(frame, predictions, args.submission_path)

    has_reference = bool(frame["translation"].astype(str).str.strip().ne("").any())
    if has_reference:
        write_eval_csv(frame, predictions, args)


if __name__ == "__main__":
    main()
