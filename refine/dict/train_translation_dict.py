from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refine.refine_train_v2 import preprocessor

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_BUILD_OUTPUT_PATH = PROJECT_ROOT / "data" / "train_translation_dict.json"
DEFAULT_LOOKUP_OUTPUT_PATH = PROJECT_ROOT / "data" / "train_translation_lookup.json"
TRANSLATION_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
MIN_TOP_COUNT = 2
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


@dataclass
class Example:
    oare_id: str
    source_context: str
    translation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "oare_id": self.oare_id,
            "source_context": self.source_context,
            "translation": self.translation,
        }


@dataclass
class TokenEntry:
    row_count: int = 0
    occurrence_count: int = 0
    translation_word_counts: Counter[str] = field(default_factory=Counter)
    translation_bigram_counts: Counter[str] = field(default_factory=Counter)
    examples: list[Example] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple train.csv-based transliteration-to-translation concordance. "
            "Use --query for phrase lookup or --build-output-path to export a token dictionary."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--id-column", type=str, default="oare_id")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--lookup-output-path", type=Path, default=None)
    parser.add_argument("--build-output-path", type=Path, default=None)
    parser.add_argument("--min-doc-freq", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def ensure_required_columns(rows: list[dict[str, str]], id_column: str) -> None:
    if not rows:
        raise ValueError("input CSV has no rows")

    required_columns = {id_column, "transliteration", "translation"}
    available_columns = set(rows[0].keys())
    missing_columns = required_columns.difference(available_columns)
    if missing_columns:
        raise ValueError(f"input CSV is missing columns: {sorted(missing_columns)}")


def normalize_transliterations(rows: list[dict[str, str]]) -> list[str]:
    transliterations = [row.get("transliteration", "") for row in rows]
    return preprocessor.preprocess_batch(transliterations)


def normalize_query(query: str) -> str:
    return preprocessor.preprocess_batch([query])[0].strip()


def is_indexable_token(token: str) -> bool:
    if not token or token == "<gap>":
        return False
    return any(character.isalpha() for character in token)


def tokenize_translation(text: str) -> list[str]:
    tokens = [token.lower() for token in TRANSLATION_TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS]


def top_items(counter: Counter[str], top_k: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for value, count in counter.most_common():
        if count < MIN_TOP_COUNT:
            continue
        items.append({"value": value, "count": count})
        if len(items) >= top_k:
            break
    return items


def find_phrase_positions(tokens: list[str], query_tokens: list[str]) -> list[int]:
    if not query_tokens or len(query_tokens) > len(tokens):
        return []

    positions: list[int] = []
    last_start = len(tokens) - len(query_tokens) + 1
    for start in range(last_start):
        if tokens[start : start + len(query_tokens)] == query_tokens:
            positions.append(start)
    return positions


def extract_context(tokens: list[str], start: int, length: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(tokens), start + length + window)
    return " ".join(tokens[left:right])


def lookup_phrase(
    rows: list[dict[str, str]],
    normalized_transliterations: list[str],
    query: str,
    id_column: str,
    context_window: int,
    max_examples: int,
    top_k: int,
) -> dict[str, Any]:
    normalized_query = normalize_query(query)
    query_tokens = [token for token in normalized_query.split() if token]
    translation_word_counts: Counter[str] = Counter()
    translation_bigram_counts: Counter[str] = Counter()
    matched_translations: Counter[str] = Counter()
    examples: list[Example] = []
    match_rows = 0
    total_occurrences = 0

    for row, normalized_text in zip(rows, normalized_transliterations):
        tokens = [token for token in normalized_text.split() if token]
        positions = find_phrase_positions(tokens, query_tokens)
        if not positions:
            continue

        match_rows += 1
        total_occurrences += len(positions)
        translation = row.get("translation", "").strip()
        matched_translations[translation] += 1

        translation_tokens = tokenize_translation(translation)
        translation_word_counts.update(translation_tokens)
        translation_bigram_counts.update(
            " ".join(translation_tokens[index : index + 2])
            for index in range(len(translation_tokens) - 1)
        )

        if len(examples) >= max_examples:
            continue

        for start in positions:
            examples.append(
                Example(
                    oare_id=row.get(id_column, ""),
                    source_context=extract_context(tokens, start, len(query_tokens), context_window),
                    translation=translation,
                )
            )
            if len(examples) >= max_examples:
                break

    return {
        "query": query,
        "normalized_query": normalized_query,
        "query_tokens": query_tokens,
        "match_rows": match_rows,
        "total_occurrences": total_occurrences,
        "top_translation_words": top_items(translation_word_counts, top_k),
        "top_translation_bigrams": top_items(translation_bigram_counts, top_k),
        "top_full_translations": top_items(matched_translations, top_k),
        "examples": [example.to_dict() for example in examples],
    }


def build_token_dictionary(
    rows: list[dict[str, str]],
    normalized_transliterations: list[str],
    id_column: str,
    min_doc_freq: int,
    max_examples: int,
    context_window: int,
    top_k: int,
) -> list[dict[str, Any]]:
    token_entries: dict[str, TokenEntry] = {}

    for row, normalized_text in zip(rows, normalized_transliterations):
        tokens = [token for token in normalized_text.split() if is_indexable_token(token)]
        if not tokens:
            continue

        token_counts = Counter(tokens)
        translation = row.get("translation", "").strip()
        translation_tokens = tokenize_translation(translation)
        translation_bigrams = [
            " ".join(translation_tokens[index : index + 2])
            for index in range(len(translation_tokens) - 1)
        ]
        first_positions: dict[str, int] = {}
        for index, token in enumerate(tokens):
            first_positions.setdefault(token, index)

        for token, count in token_counts.items():
            entry = token_entries.setdefault(token, TokenEntry())
            entry.row_count += 1
            entry.occurrence_count += count
            entry.translation_word_counts.update(translation_tokens)
            entry.translation_bigram_counts.update(translation_bigrams)

            if len(entry.examples) < max_examples:
                start = first_positions[token]
                entry.examples.append(
                    Example(
                        oare_id=row.get(id_column, ""),
                        source_context=extract_context(tokens, start, 1, context_window),
                        translation=translation,
                    )
                )

    dictionary: list[dict[str, Any]] = []
    for token, entry in sorted(
        token_entries.items(),
        key=lambda item: (-item[1].row_count, -item[1].occurrence_count, item[0]),
    ):
        if entry.row_count < min_doc_freq:
            continue

        dictionary.append(
            {
                "source_token": token,
                "row_count": entry.row_count,
                "occurrence_count": entry.occurrence_count,
                "top_translation_words": top_items(entry.translation_word_counts, top_k),
                "top_translation_bigrams": top_items(entry.translation_bigram_counts, top_k),
                "examples": [example.to_dict() for example in entry.examples],
            }
        )

    return dictionary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def print_lookup_summary(summary: dict[str, Any]) -> None:
    print(f"query: {summary['query']}")
    print(f"normalized_query: {summary['normalized_query']}")
    print(f"match_rows: {summary['match_rows']}")
    print(f"total_occurrences: {summary['total_occurrences']}")
    print("top_translation_words:")
    for item in summary["top_translation_words"]:
        print(f"  {item['value']}: {item['count']}")
    print("top_translation_bigrams:")
    for item in summary["top_translation_bigrams"]:
        print(f"  {item['value']}: {item['count']}")
    print("examples:")
    for example in summary["examples"]:
        print(f"  [{example['oare_id']}] {example['source_context']}")
        print(f"    {example['translation']}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_path)
    ensure_required_columns(rows, args.id_column)
    normalized_transliterations = normalize_transliterations(rows)

    did_work = False

    if args.query:
        summary = lookup_phrase(
            rows=rows,
            normalized_transliterations=normalized_transliterations,
            query=args.query,
            id_column=args.id_column,
            context_window=args.context_window,
            max_examples=args.max_examples,
            top_k=args.top_k,
        )
        print_lookup_summary(summary)
        if args.lookup_output_path:
            write_json(args.lookup_output_path, summary)
            print(f"Wrote lookup summary to {args.lookup_output_path}")
        did_work = True

    if args.build_output_path:
        dictionary = build_token_dictionary(
            rows=rows,
            normalized_transliterations=normalized_transliterations,
            id_column=args.id_column,
            min_doc_freq=args.min_doc_freq,
            max_examples=args.max_examples,
            context_window=args.context_window,
            top_k=args.top_k,
        )
        payload = {
            "input_path": str(args.input_path),
            "entry_count": len(dictionary),
            "min_doc_freq": args.min_doc_freq,
            "entries": dictionary,
        }
        write_json(args.build_output_path, payload)
        print(f"Wrote {len(dictionary)} dictionary entries to {args.build_output_path}")
        did_work = True

    if not did_work:
        raise SystemExit("Specify at least one of --query or --build-output-path.")


if __name__ == "__main__":
    main()
