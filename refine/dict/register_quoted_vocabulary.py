from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition_with_word_defs.csv"
DEFAULT_CHANGED_ROWS_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_word_defs_changed_rows.csv"
QUOTED_TEXT_RE = re.compile(r'"([^"]+)"')
PAREN_CONTENT_RE = re.compile(r"\([^()]*\)")
MULTI_WS_RE = re.compile(r"\s+")
NOISE_PUNCT_RE = re.compile(r"[!?]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract quoted fragments from a CSV column, split them by commas, "
            "and store the result in a word_defs column as list[str]."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--changed-rows-path", type=Path, default=DEFAULT_CHANGED_ROWS_PATH)
    parser.add_argument("--quoted-source-column", type=str, default="definition")
    parser.add_argument("--target-column", type=str, default="word_defs")
    return parser.parse_args()


def extract_quoted_vocabulary(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for quoted_text in QUOTED_TEXT_RE.findall(str(text or "")):
        for chunk in quoted_text.split(","):
            normalized = PAREN_CONTENT_RE.sub(" ", chunk)
            normalized = NOISE_PUNCT_RE.sub(" ", normalized)
            normalized = MULTI_WS_RE.sub(" ", normalized).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return values


def parse_existing_word_defs(value: str) -> list[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return []
    try:
        parsed = ast.literal_eval(normalized)
    except Exception:
        return [normalized]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def merge_word_defs(existing: list[str], extracted: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in existing + extracted:
        normalized = str(value).strip()
        dedup_key = normalized.casefold()
        if not normalized or dedup_key in seen:
            continue
        seen.add(dedup_key)
        merged.append(normalized)
    return merged


def serialize_word_defs(values: list[str]) -> str:
    escaped_values = [value.replace("\\", "\\\\").replace("'", "\\'") for value in values]
    return "[" + ", ".join(f"'{value}'" for value in escaped_values) + "]"


def build_rows(
    input_path: Path,
    quoted_source_column: str,
    target_column: str,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if quoted_source_column not in fieldnames:
            raise ValueError(f"missing quoted source column: {quoted_source_column}")
        if target_column not in fieldnames:
            fieldnames.append(target_column)

        all_rows: list[dict[str, str]] = []
        changed_rows: list[dict[str, str]] = []

        for row in reader:
            row = dict(row)
            existing_word_defs = parse_existing_word_defs(row.get(target_column, ""))
            extracted_word_defs = extract_quoted_vocabulary(row.get(quoted_source_column, ""))
            merged_word_defs = merge_word_defs(existing_word_defs, extracted_word_defs)
            row[target_column] = serialize_word_defs(merged_word_defs)
            all_rows.append(row)
            if extracted_word_defs:
                changed_rows.append(dict(row))

    return fieldnames, all_rows, changed_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    fieldnames, rows, changed_rows = build_rows(
        input_path=args.input_path,
        quoted_source_column=args.quoted_source_column,
        target_column=args.target_column,
    )
    write_rows(args.output_path, fieldnames, rows)
    write_rows(args.changed_rows_path, fieldnames, changed_rows)

    print(f"rows: {len(rows)}")
    print(f"rows_with_added_word_defs: {len(changed_rows)}")
    print(f"wrote augmented CSV: {args.output_path}")
    print(f"wrote changed-rows CSV: {args.changed_rows_path}")


if __name__ == "__main__":
    main()
