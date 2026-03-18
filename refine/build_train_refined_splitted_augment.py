from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted.csv"
DEFAULT_OUTPUT_V1_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted_augmented_v1.csv"
DEFAULT_OUTPUT_V2_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted_augmented_v2.csv"

DEFAULT_V2_MAX_SEGMENTS = 5
DEFAULT_V2_MAX_TRANSLITERATION_WORDS = 80
DEFAULT_V2_MAX_TRANSLATION_WORDS = 80


@dataclass(frozen=True)
class SegmentRow:
    oare_id: str
    source_oare_id: str
    segment_index: int
    transliteration: str
    translation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build v1/v2 augmented datasets from train_refined_splitted.csv. "
            "v1 merges by sentence-final '.', v2 enumerates contiguous windows."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-v1-path", type=Path, default=DEFAULT_OUTPUT_V1_PATH)
    parser.add_argument("--output-v2-path", type=Path, default=DEFAULT_OUTPUT_V2_PATH)
    parser.add_argument("--v2-max-segments", type=int, default=DEFAULT_V2_MAX_SEGMENTS)
    parser.add_argument(
        "--v2-max-transliteration-words",
        type=int,
        default=DEFAULT_V2_MAX_TRANSLITERATION_WORDS,
    )
    parser.add_argument(
        "--v2-max-translation-words",
        type=int,
        default=DEFAULT_V2_MAX_TRANSLATION_WORDS,
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[SegmentRow]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows: list[SegmentRow] = []
        for row in reader:
            rows.append(
                SegmentRow(
                    oare_id=row["oare_id"].strip(),
                    source_oare_id=row["source_oare_id"].strip(),
                    segment_index=int(row["segment_index"]),
                    transliteration=row["transliteration"].strip(),
                    translation=row["translation"].strip(),
                )
            )
    return rows


def group_rows(rows: list[SegmentRow]) -> dict[str, list[SegmentRow]]:
    grouped: dict[str, list[SegmentRow]] = {}
    for row in rows:
        grouped.setdefault(row.source_oare_id, []).append(row)
    for source_oare_id in grouped:
        grouped[source_oare_id].sort(key=lambda item: item.segment_index)
    return grouped


def count_words(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def build_span_oare_id(source_oare_id: str, start_index: int, end_index: int) -> str:
    if start_index == end_index:
        return f"{source_oare_id}--{start_index}"
    return f"{source_oare_id}--{start_index}-{end_index}"


def merge_segments(rows: list[SegmentRow]) -> dict[str, str | int]:
    start_index = rows[0].segment_index
    end_index = rows[-1].segment_index
    transliteration = " ".join(row.transliteration for row in rows).strip()
    translation = " ".join(row.translation for row in rows).strip()
    return {
        "oare_id": build_span_oare_id(rows[0].source_oare_id, start_index, end_index),
        "source_oare_id": rows[0].source_oare_id,
        "start_segment_index": start_index,
        "end_segment_index": end_index,
        "segment_span_length": len(rows),
        "transliteration": transliteration,
        "translation": translation,
        "transliteration_word_count": count_words(transliteration),
        "translation_word_count": count_words(translation),
    }


def build_v1_rows(grouped_rows: dict[str, list[SegmentRow]]) -> list[dict[str, str | int]]:
    augmented_rows: list[dict[str, str | int]] = []
    for rows in grouped_rows.values():
        sentence_buffer: list[SegmentRow] = []
        for row in rows:
            sentence_buffer.append(row)
            if row.translation.endswith("."):
                augmented_rows.append(merge_segments(sentence_buffer))
                sentence_buffer = []
        if sentence_buffer:
            augmented_rows.append(merge_segments(sentence_buffer))
    return augmented_rows


def build_v2_rows(
    grouped_rows: dict[str, list[SegmentRow]],
    max_segments: int,
    max_transliteration_words: int,
    max_translation_words: int,
) -> list[dict[str, str | int]]:
    augmented_rows: list[dict[str, str | int]] = []

    for rows in grouped_rows.values():
        for start in range(len(rows)):
            current_rows: list[SegmentRow] = []
            current_transliteration_words = 0
            current_translation_words = 0
            for end in range(start, min(len(rows), start + max_segments)):
                row = rows[end]
                current_rows.append(row)
                current_transliteration_words += count_words(row.transliteration)
                current_translation_words += count_words(row.translation)

                if current_transliteration_words > max_transliteration_words:
                    break
                if current_translation_words > max_translation_words:
                    break

                augmented_rows.append(merge_segments(current_rows))

    return augmented_rows


def write_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "oare_id",
        "source_oare_id",
        "start_segment_index",
        "end_segment_index",
        "segment_span_length",
        "transliteration",
        "translation",
        "transliteration_word_count",
        "translation_word_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_path)
    grouped_rows = group_rows(rows)

    v1_rows = build_v1_rows(grouped_rows)
    v2_rows = build_v2_rows(
        grouped_rows,
        max_segments=max(args.v2_max_segments, 1),
        max_transliteration_words=max(args.v2_max_transliteration_words, 1),
        max_translation_words=max(args.v2_max_translation_words, 1),
    )

    write_rows(args.output_v1_path, v1_rows)
    write_rows(args.output_v2_path, v2_rows)

    print(f"Wrote {len(v1_rows)} v1 row(s) to {args.output_v1_path}")
    print(f"Wrote {len(v2_rows)} v2 row(s) to {args.output_v2_path}")


if __name__ == "__main__":
    main()
