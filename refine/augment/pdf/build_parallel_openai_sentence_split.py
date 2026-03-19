from __future__ import annotations

import argparse
import csv
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
    / "akt6a_parallel_openai.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
    / "akt6a_parallel_openai_sentence_split.csv"
)

SENTENCE_END_RE = re.compile(r'[.!?](?:["\'”’)\]]+)?$')
ELLIPSIS_END_RE = re.compile(r'(?:\.\.\.|…)(?:["\'”’)\]]+)?$')


@dataclass(frozen=True)
class LineRow:
    oare_id: str
    doc_label: str
    pdf_page: int
    transliteration: str
    translation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert akt6a_parallel_openai.csv-style line rows into sentence-level CSV rows "
            "with page_start/page_end metadata."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--keep-fragments",
        action="store_true",
        help="Keep trailing non-terminal fragments instead of dropping them.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path: Path) -> list[LineRow]:
    rows: list[LineRow] = []
    current_doc_label = ""
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for raw_row in reader:
            doc_label = normalize_text(raw_row.get("doc_label", ""))
            if doc_label:
                current_doc_label = doc_label
            rows.append(
                LineRow(
                    oare_id=normalize_text(raw_row["oare_id"]),
                    doc_label=current_doc_label,
                    pdf_page=int(raw_row["pdf_page"]),
                    transliteration=normalize_text(raw_row.get("transliteration", "")),
                    translation=normalize_text(raw_row.get("translation", "")),
                )
            )
    return rows


def build_source_oare_id(doc_label: str) -> str:
    seed = f"akt_parallel_sentence_split|{normalize_text(doc_label)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def build_segment_oare_id(source_oare_id: str, segment_index: int) -> str:
    return f"{source_oare_id}--{segment_index}"


def is_sentence_terminal(translation: str) -> bool:
    if not translation:
        return False
    if ELLIPSIS_END_RE.search(translation):
        return False
    return bool(SENTENCE_END_RE.search(translation))


def count_words(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def group_rows_by_doc_label(rows: list[LineRow]) -> list[tuple[str, list[LineRow]]]:
    groups: list[tuple[str, list[LineRow]]] = []
    current_doc_label = ""
    current_rows: list[LineRow] = []

    for row in rows:
        if row.doc_label != current_doc_label:
            if current_rows:
                groups.append((current_doc_label, current_rows))
            current_doc_label = row.doc_label
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        groups.append((current_doc_label, current_rows))

    return groups


def merge_buffer(
    *,
    source_oare_id: str,
    doc_label: str,
    segment_index: int,
    rows: list[LineRow],
) -> dict[str, str | int]:
    transliteration = " ".join(row.transliteration for row in rows if row.transliteration).strip()
    translation = " ".join(row.translation for row in rows if row.translation).strip()
    page_start = min(row.pdf_page for row in rows)
    page_end = max(row.pdf_page for row in rows)
    return {
        "oare_id": build_segment_oare_id(source_oare_id, segment_index),
        "source_oare_id": source_oare_id,
        "doc_label": doc_label,
        "segment_index": segment_index,
        "page_start": page_start,
        "page_end": page_end,
        "line_start_oare_id": rows[0].oare_id,
        "line_end_oare_id": rows[-1].oare_id,
        "line_span_length": len(rows),
        "transliteration": transliteration,
        "translation": translation,
        "transliteration_word_count": count_words(transliteration),
        "translation_word_count": count_words(translation),
    }


def build_sentence_rows(
    rows: list[LineRow],
    *,
    keep_fragments: bool = False,
) -> tuple[list[dict[str, str | int]], int]:
    sentence_rows: list[dict[str, str | int]] = []
    dropped_fragments = 0
    for doc_label, group_rows in group_rows_by_doc_label(rows):
        source_oare_id = build_source_oare_id(doc_label or group_rows[0].oare_id)
        buffer: list[LineRow] = []
        segment_index = 1

        for row in group_rows:
            buffer.append(row)
            if is_sentence_terminal(row.translation):
                sentence_rows.append(
                    merge_buffer(
                        source_oare_id=source_oare_id,
                        doc_label=doc_label,
                        segment_index=segment_index,
                        rows=buffer,
                    )
                )
                buffer = []
                segment_index += 1

        if buffer:
            if keep_fragments:
                sentence_rows.append(
                    merge_buffer(
                        source_oare_id=source_oare_id,
                        doc_label=doc_label,
                        segment_index=segment_index,
                        rows=buffer,
                    )
                )
            else:
                dropped_fragments += 1

    return sentence_rows, dropped_fragments


def write_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "oare_id",
        "source_oare_id",
        "doc_label",
        "segment_index",
        "page_start",
        "page_end",
        "line_start_oare_id",
        "line_end_oare_id",
        "line_span_length",
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
    line_rows = load_rows(args.input_path)
    sentence_rows, dropped_fragments = build_sentence_rows(
        line_rows,
        keep_fragments=args.keep_fragments,
    )
    write_rows(args.output_path, sentence_rows)
    print(f"Wrote {len(sentence_rows)} sentence row(s) to {args.output_path}")
    if not args.keep_fragments:
        print(f"Dropped {dropped_fragments} trailing fragment(s) without sentence-final punctuation")


if __name__ == "__main__":
    main()
