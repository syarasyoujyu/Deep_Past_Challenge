from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM

from model.common import (
    DATA_DIR,
    ROOT_DIR,
    build_generation_config,
    load_tokenizer,
    seed_everything,
)

DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "byt5-small"
DEFAULT_INPUT_PATH = DATA_DIR / "test.csv"
DEFAULT_SUBMISSION_PATH = ROOT_DIR / "submission.csv"
DEFAULT_EVAL_OUTPUT_PATH = ROOT_DIR / "artifacts" / "byt5-predictions.csv"
DEFAULT_METRICS_PATH = ROOT_DIR / "artifacts" / "byt5-predictions-metrics.json"


def maybe_transform_with_bettertransformer(model, enabled: bool):
    if not enabled:
        return model

    try:
        from optimum.bettertransformer import BetterTransformer
    except ImportError as error:
        raise ImportError(
            "--use-bettertransformer true was specified, but optimum is not installed."
        ) from error

    print("Applying BetterTransformer...")
    return BetterTransformer.transform(model)


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
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-bettertransformer", type=str, default="false")
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


def predict_translations(frame: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    use_bettertransformer = str(args.use_bettertransformer).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }
    model = maybe_transform_with_bettertransformer(model, use_bettertransformer)
    generation_config = build_generation_config(args)
    use_cpu = args.device == "cpu" or not torch.cuda.is_available()
    device = torch.device("cpu" if use_cpu else args.device)
    print(
        "Inference device: "
        f"requested={args.device} "
        f"resolved={device} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    model = model.to(device)
    model.eval()

    source_texts = [
        f"{args.source_prefix}{text}" for text in frame["transliteration"].tolist()
    ]
    predictions: list[str] = []

    for start in range(0, len(source_texts), args.batch_size):
        batch_texts = source_texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_config)

        decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(prediction.strip() for prediction in decoded_batch)

    return predictions


def compute_row_metrics(
    reference_texts: list[str],
    hypothesis_texts: list[str],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[float]]:
    bertscores = []
    chrf_scores: list[float] = []
    geometric_means: list[float] = []
    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts, strict=True):
        sentence_bleu = float(sacrebleu.corpus_bleu(hypothesis_text, [reference_text]).score)
        sentence_chrf = float(sacrebleu.corpus_chrf(hypothesis_text, [reference_text], word_order=2).score)
        bertscores.append(sentence_bleu)
        geometric_mean = math.sqrt(sentence_bleu*sentence_chrf)
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
    seed_everything(args.seed)
    frame = load_input_frame(args.input_path)
    predictions = predict_translations(frame, args)

    write_submission(frame, predictions, args.submission_path)

    has_reference = bool(frame["translation"].astype(str).str.strip().ne("").any())
    if has_reference:
        write_eval_csv(frame, predictions, args)


if __name__ == "__main__":
    main()
