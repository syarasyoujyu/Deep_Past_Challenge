from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "v2" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "v2" / "train_truncated.csv"
TOP_FRACTION = 0.15
OUTER_TOKEN_STRIP_CHARS = "[](){}<>⸢⸣⌈⌉⌊⌋⟦⟧"
INNER_TOKEN_STRIP_CHARS = ".,;:!?+-*'\""
ELLIPSIS_TOKEN_RE = re.compile(r"^(?:\.{3,}|…+)$")


@dataclass(frozen=True)
class RowStats:
    index: int
    row_id: str
    word_count: int
    x_count: int
    ellipsis_count: int

    @property
    def marker_count(self) -> int:
        return self.x_count + self.ellipsis_count

    @property
    def marker_density(self) -> float:
        if self.word_count == 0:
            return 0.0
        return self.marker_count / self.word_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows whose transliterations contain too many gap markers "
            "(`x` and `...`) per token."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=TOP_FRACTION,
        help="Fraction of rows to remove after sorting by marker density.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def normalize_token(token: str) -> str:
    return token.strip(OUTER_TOKEN_STRIP_CHARS).strip(INNER_TOKEN_STRIP_CHARS)


def build_row_stats(rows: list[dict[str, str]]) -> list[RowStats]:
    stats: list[RowStats] = []

    for index, row in enumerate(rows):
        transliteration = row.get("transliteration", "")
        word_count = 0
        x_count = 0
        ellipsis_count = 0

        for raw_token in transliteration.split():
            marker_token = raw_token.strip(OUTER_TOKEN_STRIP_CHARS).strip("*'\"")
            if not marker_token:
                continue

            if ELLIPSIS_TOKEN_RE.fullmatch(marker_token):
                word_count += 1
                ellipsis_count += 1
                continue

            token = normalize_token(raw_token)
            if not token:
                continue

            word_count += 1
            if token == "x":
                x_count += 1

        stats.append(
            RowStats(
                index=index,
                row_id=row.get("id", ""),
                word_count=word_count,
                x_count=x_count,
                ellipsis_count=ellipsis_count,
            )
        )

    return stats


def select_rows_to_remove(stats: list[RowStats], top_fraction: float) -> set[int]:
    if not 0.0 <= top_fraction <= 1.0:
        raise ValueError(f"--top-fraction must be between 0 and 1, got {top_fraction}")

    if not stats or top_fraction == 0.0:
        return set()

    remove_count = math.ceil(len(stats) * top_fraction)
    ranked = sorted(
        stats,
        key=lambda row: (
            row.marker_density,
            row.marker_count,
            row.ellipsis_count,
            row.x_count,
            row.index,
        ),
        reverse=True,
    )
    return {row.index for row in ranked[:remove_count]}


def percentile_value(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    rank = math.ceil((percentile / 100) * len(sorted_values)) - 1
    rank = min(len(sorted_values) - 1, max(0, rank))
    return sorted_values[rank]


def print_x_distribution(label: str, stats: list[RowStats]) -> None:
    x_counts = sorted(row.x_count for row in stats)
    nonzero_rows = sum(1 for count in x_counts if count > 0)
    print(f"{label} x-count distribution:")
    print(f"  rows: {len(x_counts)}")
    print(f"  rows with x: {nonzero_rows}")
    for percentile in (0, 25, 50, 75, 85, 90, 95, 99, 100):
        value = percentile_value(x_counts, percentile)
        print(f"  p{percentile:>2}: {value}")


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, fieldnames = load_rows(args.input)
    stats = build_row_stats(rows)
    rows_to_remove = select_rows_to_remove(stats, args.top_fraction)

    filtered_rows = [row for index, row in enumerate(rows) if index not in rows_to_remove]
    removed_stats = [row for row in stats if row.index in rows_to_remove]

    write_rows(args.output, filtered_rows, fieldnames)

    cutoff_density = min((row.marker_density for row in removed_stats), default=0.0)
    removed_marker_count = sum(row.marker_count for row in removed_stats)
    removed_word_count = sum(row.word_count for row in removed_stats)
    removed_density = (
        removed_marker_count / removed_word_count if removed_word_count else 0.0
    )

    print(f"Loaded {len(rows)} rows from {args.input}")
    print_x_distribution("All rows", stats)
    print_x_distribution("Removed rows", removed_stats)
    print(
        f"Removed {len(removed_stats)} rows "
        f"(top {args.top_fraction:.2%} by x / ellipsis density)"
    )
    print(f"Cutoff density: {cutoff_density:.6f}")
    print(f"Removed rows combined density: {removed_density:.6f}")
    print(f"Wrote {len(filtered_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
