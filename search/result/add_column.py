from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import sacrebleu

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.byt5 import normalize_translation


DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "infer" / "low_score_akkadian_texts.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add translation_with_fix to a CSV by applying the same translation "
            "normalization used in model/ft/dict/train_byt5_with_dictionary.py."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def compute_row_metrics(
    reference_texts: list[str],
    hypothesis_texts: list[str],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[float]]:
    del args
    bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    bleus: list[float] = []
    chrf_scores: list[float] = []
    geometric_means: list[float] = []

    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts, strict=True):
        sentence_bleu = float(bleu_metric.sentence_score(hypothesis_text, [reference_text]).score)
        sentence_chrf = float(chrf_metric.sentence_score(hypothesis_text, [reference_text]).score)
        bleus.append(sentence_bleu)
        chrf_scores.append(sentence_chrf)
        geometric_means.append(math.sqrt(max(sentence_bleu, 0.0) * max(sentence_chrf, 0.0)))

    return bleus, chrf_scores, geometric_means


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.input_path

    frame = pd.read_csv(args.input_path).copy()
    if "translation" not in frame.columns:
        raise ValueError(f"{args.input_path} is missing the 'translation' column.")
    if "reference_translation" not in frame.columns:
        raise ValueError(f"{args.input_path} is missing the 'reference_translation' column.")

    frame["translation_with_fix"] = (
        frame["translation"]
        .fillna("")
        .astype(str)
        .map(normalize_translation)
    )
    reference_texts = frame["reference_translation"].fillna("").astype(str).map(normalize_translation).tolist()
    hypothesis_texts = frame["translation_with_fix"].fillna("").astype(str).tolist()
    bleu_scores, chrf_scores, geometric_means = compute_row_metrics(
        reference_texts,
        hypothesis_texts,
        args,
    )
    frame["reference_translation_with_fix"] = reference_texts

    if "bertscore" in frame.columns and "bertscore_old" not in frame.columns:
        frame["bertscore_old"] = frame["bertscore"]
    if "chrfpp" in frame.columns and "chrfpp_old" not in frame.columns:
        frame["chrfpp_old"] = frame["chrfpp"]
    if "geometric_mean" in frame.columns and "geometric_mean_old" not in frame.columns:
        frame["geometric_mean_old"] = frame["geometric_mean"]

    frame["bertscore"] = bleu_scores
    frame["chrfpp"] = chrf_scores
    frame["geometric_mean"] = geometric_means
    frame.to_csv(output_path, index=False)

    changed = int((frame["translation"].fillna("").astype(str) != frame["translation_with_fix"]).sum())
    print(f"Saved: {output_path}")
    print(f"Rows: {len(frame)}")
    print(f"Changed translations: {changed}")


if __name__ == "__main__":
    main()
