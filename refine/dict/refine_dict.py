from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from refine.refine_train_v2 import preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined.csv"
DEFAULT_DICTIONARY_PATH = PROJECT_ROOT / "data" / "eBL_Dictionary.csv"
DEFAULT_JOINED_OUTPUT_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFINITION_SEPARATOR = "\n---------\n"


def strip_dictionary_suffix(word: str) -> str:
    return re.sub(r"\s+[IVXLCDM]+$", "", word.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a refined OA lexicon by normalizing the form column."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-column", type=str, default="form")
    parser.add_argument("--no-keep-original-column", dest="keep_original_column", action="store_false")
    parser.set_defaults(keep_original_column=True)
    parser.add_argument("--original-column-name", type=str, default="form_original")
    parser.add_argument("--dictionary-path", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--joined-output-path", type=Path, default=DEFAULT_JOINED_OUTPUT_PATH)
    return parser.parse_args()


def build_refined_dictionary(
    input_path: Path,
    output_path: Path,
    source_column: str,
    keep_original_column: bool,
    original_column_name: str,
) -> dict[str, int]:
    frame = pd.read_csv(input_path)
    if source_column not in frame.columns:
        raise ValueError(f"missing source column: {source_column}")

    original_values = frame[source_column].fillna("").astype(str)
    refined_values = preprocessor.preprocess_batch(original_values.tolist())

    refined_frame = frame.copy()
    if keep_original_column and original_column_name not in refined_frame.columns:
        source_index = refined_frame.columns.get_loc(source_column)
        refined_frame.insert(source_index + 1, original_column_name, original_values)

    refined_frame[source_column] = refined_values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined_frame.to_csv(output_path, index=False)

    changed_count = int((original_values != pd.Series(refined_values)).sum())
    return {
        "rows": int(len(refined_frame)),
        "changed_rows": changed_count,
    }


def _join_unique_non_empty(values: pd.Series) -> str:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values.fillna("").astype(str):
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
    return DEFINITION_SEPARATOR.join(unique_values)


def build_joined_dictionary(
    lexicon_path: Path,
    dictionary_path: Path,
    output_path: Path,
) -> dict[str, int]:
    lexicon_frame = pd.read_csv(lexicon_path)
    dictionary_frame = pd.read_csv(dictionary_path)

    required_lexicon_columns = {"form", "lexeme"}
    missing_lexicon_columns = required_lexicon_columns.difference(lexicon_frame.columns)
    if missing_lexicon_columns:
        raise ValueError(f"lexicon is missing columns: {sorted(missing_lexicon_columns)}")

    required_dictionary_columns = {"word", "definition"}
    missing_dictionary_columns = required_dictionary_columns.difference(dictionary_frame.columns)
    if missing_dictionary_columns:
        raise ValueError(f"dictionary is missing columns: {sorted(missing_dictionary_columns)}")

    dictionary_frame = dictionary_frame.copy()
    dictionary_frame["word"] = dictionary_frame["word"].fillna("").astype(str).map(strip_dictionary_suffix)
    dictionary_frame = dictionary_frame[dictionary_frame["word"].str.strip() != ""].reset_index(drop=True)

    aggregated_dictionary = (
        dictionary_frame.groupby("word", dropna=False, sort=False)
        .agg(
            definition=("definition", _join_unique_non_empty),
            derived_from=("derived_from", _join_unique_non_empty),
        )
        .reset_index()
    )

    joined_frame = lexicon_frame.merge(
        aggregated_dictionary,
        how="left",
        left_on="lexeme",
        right_on="word",
        validate="many_to_one",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined_frame.to_csv(output_path, index=False)

    matched_rows = int(joined_frame["definition"].fillna("").astype(str).str.strip().ne("").sum())
    return {
        "rows": int(len(joined_frame)),
        "matched_rows": matched_rows,
    }


def main() -> None:
    args = parse_args()
    summary = build_refined_dictionary(
        input_path=args.input_path,
        output_path=args.output_path,
        source_column=args.source_column,
        keep_original_column=args.keep_original_column,
        original_column_name=args.original_column_name,
    )
    joined_summary = build_joined_dictionary(
        lexicon_path=args.output_path,
        dictionary_path=args.dictionary_path,
        output_path=args.joined_output_path,
    )
    print(
        f"Wrote refined dictionary to {args.output_path} "
        f"(rows={summary['rows']}, changed_rows={summary['changed_rows']})"
    )
    print(
        f"Wrote joined dictionary to {args.joined_output_path} "
        f"(rows={joined_summary['rows']}, matched_rows={joined_summary['matched_rows']})"
    )


if __name__ == "__main__":
    main()
