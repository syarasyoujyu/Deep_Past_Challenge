from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.find_missing_translation_pn_matches import (
    DEFAULT_LEXICON_PATH,
    DEFAULT_ONOMASTICON_PATH,
    DEFAULT_TRAIN_PATH,
    PNEntry,
    fold_onomasticon_spelling_for_match,
    load_pn_index,
    normalize_transliteration_for_match,
)
from search.find_translation_pn_gn_frequencies import load_gn_index


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "search" / "train_transliteration_pn_gn_frequencies.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count PN and GN occurrences directly from train transliteration."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--onomasticon-path", type=Path, default=DEFAULT_ONOMASTICON_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def build_type_sets_by_form(lexicon_path: Path) -> dict[str, set[str]]:
    type_sets_by_form: dict[str, set[str]] = defaultdict(set)
    with lexicon_path.open("r", encoding="utf-8", newline="") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        for row in reader:
            normalized_form = normalize_transliteration_for_match(row.get("form", ""))
            if not normalized_form:
                continue
            entry_type = str(row.get("type", "")).strip()
            if not entry_type:
                continue
            type_sets_by_form[normalized_form].add(entry_type)
    return type_sets_by_form


def build_spelling_index(
    entries: dict[str, PNEntry],
    type_sets_by_form: dict[str, set[str]],
    target_type: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, PNEntry]]:
    exact_index: dict[str, set[str]] = defaultdict(set)
    folded_index: dict[str, set[str]] = defaultdict(set)
    key_to_entry: dict[str, PNEntry] = {}

    for entry in entries.values():
        for spelling in entry.spellings:
            allowed_types = type_sets_by_form.get(spelling, set())
            if allowed_types and allowed_types != {target_type}:
                continue
            token_parts = spelling.split()
            if len(token_parts) != 1:
                continue
            exact_index[token_parts[0]].add(spelling)
            key_to_entry[spelling] = entry
        for spelling in entry.onomasticon_spellings:
            allowed_types = type_sets_by_form.get(spelling, set())
            if allowed_types and allowed_types != {target_type}:
                continue
            token_parts = spelling.split()
            if len(token_parts) != 1:
                continue
            folded_spelling = fold_onomasticon_spelling_for_match(spelling)
            folded_parts = folded_spelling.split()
            if len(folded_parts) != 1:
                continue
            folded_index[folded_parts[0]].add(spelling)
            key_to_entry[spelling] = entry

    return exact_index, folded_index, key_to_entry


def count_transliteration_mentions(
    train_path: Path,
    entries: dict[str, PNEntry],
    type_sets_by_form: dict[str, set[str]],
    target_type: str,
) -> tuple[Counter[str], Counter[str], dict[str, PNEntry]]:
    mention_counter: Counter[str] = Counter()
    row_counter: Counter[str] = Counter()
    exact_index, folded_index, key_to_entry = build_spelling_index(
        entries,
        type_sets_by_form,
        target_type,
    )

    with train_path.open("r", encoding="utf-8", newline="") as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            transliteration = row.get("transliteration", "") or ""
            normalized_text = normalize_transliteration_for_match(transliteration)
            exact_counts: Counter[str] = Counter()
            folded_counts: Counter[str] = Counter()

            exact_tokens = normalized_text.split()
            for token in exact_tokens:
                for dict_key in exact_index.get(token, ()):
                    exact_counts[dict_key] += 1

            if folded_index:
                folded_tokens = fold_onomasticon_spelling_for_match(normalized_text).split()
                for token in folded_tokens:
                    for dict_key in folded_index.get(token, ()):
                        folded_counts[dict_key] += 1

            seen_keys = set(exact_counts) | set(folded_counts)
            for dict_key in seen_keys:
                count = max(exact_counts.get(dict_key, 0), folded_counts.get(dict_key, 0))
                if count <= 0:
                    continue
                mention_counter[dict_key] += count
                row_counter[dict_key] += 1

    return mention_counter, row_counter, key_to_entry


def build_rows(
    entity_type: str,
    key_to_entry: dict[str, PNEntry],
    mention_counter: Counter[str],
    row_counter: Counter[str],
) -> list[dict[str, str]]:
    rows = []
    for dict_key, mention_count in mention_counter.most_common():
        entry = key_to_entry.get(dict_key)
        rows.append(
            {
                "entity_type": entity_type,
                "dict_key": dict_key,
                "mention_count": str(mention_count),
                "row_count": str(row_counter[dict_key]),
                "sources": "; ".join(sorted(entry.sources)) if entry else "",
                "canonical_names": "; ".join(sorted(entry.names)) if entry else "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    pn_entries = load_pn_index(args.onomasticon_path, args.lexicon_path)
    gn_entries = load_gn_index(args.lexicon_path)
    type_sets_by_form = build_type_sets_by_form(args.lexicon_path)

    pn_mentions, pn_rows, pn_key_to_entry = count_transliteration_mentions(
        args.train_path,
        pn_entries,
        type_sets_by_form,
        "PN",
    )
    gn_mentions, gn_rows, gn_key_to_entry = count_transliteration_mentions(
        args.train_path,
        gn_entries,
        type_sets_by_form,
        "GN",
    )

    output_rows = build_rows("PN", pn_key_to_entry, pn_mentions, pn_rows)
    output_rows.extend(build_rows("GN", gn_key_to_entry, gn_mentions, gn_rows))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "entity_type",
                "dict_key",
                "mention_count",
                "row_count",
                "sources",
                "canonical_names",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded PN entries: {len(pn_entries)}")
    print(f"Loaded GN entries: {len(gn_entries)}")
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
