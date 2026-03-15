from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_data(reference_path: str | Path, hypothesis_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ref = pd.read_csv(reference_path)
    df_hyp = pd.read_csv(hypothesis_path)
    return df_ref, df_hyp


def align_by_id(df_ref: pd.DataFrame, df_hyp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"id", "translation"}
    missing_ref = required_columns - set(df_ref.columns)
    missing_hyp = required_columns - set(df_hyp.columns)
    if missing_ref:
        raise ValueError(f"reference file is missing columns: {sorted(missing_ref)}")
    if missing_hyp:
        raise ValueError(f"hypothesis file is missing columns: {sorted(missing_hyp)}")

    df_ref = df_ref.sort_values("id").reset_index(drop=True)
    df_hyp = df_hyp.sort_values("id").reset_index(drop=True)

    if len(df_ref) != len(df_hyp):
        raise ValueError(
            f"row count mismatch: reference has {len(df_ref)} rows, hypothesis has {len(df_hyp)} rows"
        )
    if not df_ref["id"].equals(df_hyp["id"]):
        raise ValueError("id columns do not match after sorting")

    return df_ref, df_hyp


def calculate_scores(df_ref: pd.DataFrame, df_hyp: pd.DataFrame) -> dict[str, float]:
    df_ref, df_hyp = align_by_id(df_ref, df_hyp)

    references = df_ref["translation"].fillna("").astype(str).tolist()
    hypotheses = df_hyp["translation"].fillna("").astype(str).tolist()

    bleu = float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
    chrfpp = float(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score)
    bleu_chrfpp_geometric_mean = math.sqrt((bleu / 100.0) * (chrfpp / 100.0)) * 100.0

    return {
        "_bleu": bleu,
        "chrf++": chrfpp,
        "_bleu_chrfpp_geometric_mean": bleu_chrfpp_geometric_mean,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate BLEU, chrF++, and geometric mean.")
    parser.add_argument(
        "reference_path",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_submission.csv",
    )
    parser.add_argument(
        "hypothesis_path",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "submission_r2.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_ref, df_hyp = load_data(args.reference_path, args.hypothesis_path)
    scores = calculate_scores(df_ref, df_hyp)
    for key, value in scores.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
