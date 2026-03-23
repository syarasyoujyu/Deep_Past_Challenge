from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import pandas as pd
import torch

from model.byt5 import (
    normalize_translation,
    normalize_transliteration,
    parse_bool,
    parse_optional_torch_dtype,
)
from model.common import ROOT_DIR, build_generation_config, load_tokenizer, seed_everything
from model.ft.dict.train_byt5_with_dictionary import (
    DEFAULT_GLOSS_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_ONOMASTICON_PATH,
    build_augmented_source,
    find_dictionary_hints,
    fit_dictionary_hints_to_source_budget,
    load_dictionary_entries,
    load_model,
    resolve_model_source,
    serialize_dictionary_hints,
)
from model.predict import (
    compute_corpus_metrics,
    compute_row_metrics,
    load_input_frame,
    maybe_transform_with_bettertransformer,
    write_submission,
)


DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "byt5-small-dict"
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "test.csv"
DEFAULT_SUBMISSION_PATH = ROOT_DIR / "submission.csv"
DEFAULT_EVAL_OUTPUT_PATH = ROOT_DIR / "artifacts" / "byt5-dict-predictions.csv"
DEFAULT_METRICS_PATH = ROOT_DIR / "artifacts" / "byt5-dict-predictions-metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dictionary-augmented ByT5 inference and optionally write row-level "
            "evaluation metrics in the same format as model/predict.py."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--eval-output-path", type=Path, default=DEFAULT_EVAL_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-name", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--load-trained-model", type=parse_bool, default=False)
    parser.add_argument("--trained-model-path", type=Path, default=None)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--gloss-path", type=Path, default=DEFAULT_GLOSS_PATH)
    parser.add_argument("--include-onomasticon", type=parse_bool, default=False)
    parser.add_argument("--onomasticon-path", type=Path, default=DEFAULT_ONOMASTICON_PATH)
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English with dictionary: ")
    parser.add_argument("--hint-placement", choices=["prepend", "append"], default="prepend")
    parser.add_argument("--max-dictionary-hints", type=int, default=8)
    parser.add_argument("--max-entry-token-length", type=int, default=4)
    parser.add_argument("--max-hint-words", type=int, default=6)
    parser.add_argument("--max-gloss-variants", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-bettertransformer", type=parse_bool, default=False)
    parser.add_argument("--dtype", dest="dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--torch-dtype", dest="dtype", type=parse_optional_torch_dtype)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--normalize-source", type=parse_bool, default=True)
    parser.add_argument("--normalize-target", type=parse_bool, default=True)
    parser.add_argument("--normalize-h", type=parse_bool, default=True)
    parser.add_argument("--normalize-subscripts", type=parse_bool, default=True)
    parser.add_argument("--normalize-breaks", type=parse_bool, default=True)
    parser.add_argument("--remove-editorial-marks", type=parse_bool, default=True)
    parser.add_argument("--strip-word-dividers", type=parse_bool, default=False)
    return parser.parse_args()


def prepare_inference_frame(frame: pd.DataFrame, dictionary_index, tokenizer, args: argparse.Namespace) -> pd.DataFrame:
    frame = frame.copy()
    if args.normalize_source:
        frame["transliteration"] = frame["transliteration"].map(
            lambda text: normalize_transliteration(text, args)
        )
    if args.normalize_target and "translation" in frame.columns:
        frame["translation"] = frame["translation"].map(normalize_translation)

    dictionary_hints_column = []
    augmented_source_column = []
    for transliteration in frame["transliteration"].tolist():
        hints = find_dictionary_hints(transliteration, dictionary_index, args)
        fitted_hints, augmented_source = fit_dictionary_hints_to_source_budget(
            transliteration,
            hints,
            args,
            tokenizer,
        )
        dictionary_hints_column.append(fitted_hints)
        augmented_source_column.append(augmented_source)

    frame["dictionary_hints"] = dictionary_hints_column
    frame["dictionary_hint_text"] = frame["dictionary_hints"].map(serialize_dictionary_hints)
    frame["dictionary_hint_count"] = frame["dictionary_hints"].map(len)
    frame["augmented_source"] = augmented_source_column
    return frame


def predict_translations(frame: pd.DataFrame, tokenizer, args: argparse.Namespace) -> list[str]:
    model = load_model(args)
    model = maybe_transform_with_bettertransformer(model, args.use_bettertransformer)
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
    predictions: list[str] = []

    for start in range(0, len(frame), args.batch_size):
        batch_texts = frame["augmented_source"].iloc[start : start + args.batch_size].tolist()
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


def write_eval_csv(frame: pd.DataFrame, predictions: list[str], args: argparse.Namespace) -> None:
    reference_texts = frame["translation"].tolist()
    bleu_scores, chrf_scores, geometric_means = compute_row_metrics(reference_texts, predictions, args)
    eval_frame = pd.DataFrame(
        {
            "oare_id": frame["oare_id"].astype(str),
            "transliteration": frame["transliteration"].astype(str),
            "augmented_source": frame["augmented_source"].astype(str),
            "dictionary_hint_text": frame["dictionary_hint_text"].astype(str),
            "dictionary_hint_count": frame["dictionary_hint_count"].astype(int),
            "translation": predictions,
            "reference_translation": reference_texts,
            "bertscore": bleu_scores,
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
    model_source = resolve_model_source(args)
    print(f"Loading ByT5 from: {model_source}")
    tokenizer = load_tokenizer(model_source)

    frame = load_input_frame(args.input_path)
    dictionary_index = load_dictionary_entries(args)
    frame = prepare_inference_frame(frame, dictionary_index, tokenizer, args)
    predictions = predict_translations(frame, tokenizer, args)

    write_submission(frame, predictions, args.submission_path)

    has_reference = bool(frame["translation"].astype(str).str.strip().ne("").any())
    if has_reference:
        write_eval_csv(frame, predictions, args)


if __name__ == "__main__":
    main()
