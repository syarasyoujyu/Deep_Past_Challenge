from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "onomasticon.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "onomasticon_refined.csv"

_BIG_GAP_RE = re.compile(r"<\s*big_gap\s*>", re.IGNORECASE)
_X_GAP_RE = re.compile(r"\[\s*x\s*\]", re.IGNORECASE)
_ANGLE_X_GAP_RE = re.compile(r"⌈\s*x\s*⌉", re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)", re.IGNORECASE)
_NON_GAP_ANGLE_TAG_RE = re.compile(r"<(?!\s*gap\s*>)[^>]*>", re.IGNORECASE)
_SLASH_VARIANT_RE = re.compile(r"(?P<left>[^/\s;]+?)/(?P<right>[^/\s;]+)")
_BRACKETS_RE = re.compile(r"[\[\]]")
_ANGLE_BRACKETS_RE = re.compile(r"[⌈⌉]")
_PARENS_RE = re.compile(r"[()]")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine onomasticon Spellings_semicolon_separated values."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--source-column",
        type=str,
        default="Spellings_semicolon_separated",
    )
    return parser.parse_args()


def _keep_left_slash_variant(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _SLASH_VARIANT_RE.sub(r"\g<left>", current)
    return current


def normalize_spelling(spelling: str) -> str:
    normalized = str(spelling or "").strip()
    if not normalized:
        return ""

    normalized = _BIG_GAP_RE.sub("<gap>", normalized)
    normalized = _X_GAP_RE.sub("<gap>", normalized)
    normalized = _ANGLE_X_GAP_RE.sub("<gap>", normalized)
    normalized = _ELLIPSIS_RE.sub("<gap>", normalized)
    normalized = _NON_GAP_ANGLE_TAG_RE.sub("", normalized)
    normalized = _keep_left_slash_variant(normalized)
    normalized = _BRACKETS_RE.sub("", normalized)
    normalized = _ANGLE_BRACKETS_RE.sub("", normalized)
    normalized = _PARENS_RE.sub("", normalized)
    normalized = _MULTI_GAP_RE.sub("<gap> ", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def normalize_spelling_field(value: str) -> str:
    pieces = str(value or "").split(";")
    normalized_pieces: list[str] = []
    seen: set[str] = set()

    for piece in pieces:
        normalized = normalize_spelling(piece)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_pieces.append(normalized)

    return "; ".join(normalized_pieces)


def refine_onomasticon(
    input_path: Path,
    output_path: Path,
    source_column: str,
) -> dict[str, int]:
    frame = pd.read_csv(input_path)
    if source_column not in frame.columns:
        raise ValueError(f"missing source column: {source_column}")

    original_values = frame[source_column].fillna("").astype(str)
    refined_values = original_values.map(normalize_spelling_field)

    refined_frame = frame.copy()
    if "Name" in refined_frame.columns:
        refined_frame["Name"] = refined_frame["Name"].fillna("").astype(str).map(normalize_spelling)
    refined_frame[source_column] = refined_values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined_frame.to_csv(output_path, index=False)

    changed_rows = int((original_values != refined_values).sum())
    return {
        "rows": int(len(refined_frame)),
        "changed_rows": changed_rows,
    }


def main() -> None:
    args = parse_args()
    summary = refine_onomasticon(
        input_path=args.input_path,
        output_path=args.output_path,
        source_column=args.source_column,
    )
    print(
        f"Wrote refined onomasticon to {args.output_path} "
        f"(rows={summary['rows']}, changed_rows={summary['changed_rows']})"
    )


if __name__ == "__main__":
    main()
