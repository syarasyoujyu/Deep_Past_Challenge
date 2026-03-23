from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.find_missing_translation_pn_matches import (
    DEFAULT_LEXICON_PATH,
    DEFAULT_ONOMASTICON_PATH,
    DEFAULT_TRAIN_PATH,
    PNEntry,
    _split_name_variants,
    add_entry,
    build_name_regex,
    find_pn_mentions,
    fold_english_text,
    load_pn_index,
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "search" / "train_translation_pn_gn_frequencies.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count PN and GN mentions in train translation using the current search heuristics."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--onomasticon-path", type=Path, default=DEFAULT_ONOMASTICON_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_gn_index(lexicon_path: Path) -> dict[str, PNEntry]:
    variant_index: dict[str, PNEntry] = {}

    with lexicon_path.open("r", encoding="utf-8", newline="") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        for row in reader:
            if str(row.get("type", "")).strip() != "GN":
                continue
            names = []
            for column in ["lexeme", "Alt_lex"]:
                names.extend(_split_name_variants(row.get(column, "")))
            if not names:
                names.extend(_split_name_variants(row.get("norm", "")))
            add_entry(variant_index, names, row.get("form", ""), "lexicon")

    unique_entries: list[PNEntry] = []
    seen_entries: set[int] = set()
    for entry in variant_index.values():
        entry_id = id(entry)
        if entry_id in seen_entries:
            continue
        seen_entries.add(entry_id)
        unique_entries.append(entry)

    merged_entries: list[PNEntry] = []
    merged_name_sets: list[set[str]] = []
    for entry in unique_entries:
        folded_names = {fold_english_text(name) for name in entry.names if fold_english_text(name)}
        merged_into_existing = False
        for index, existing_folded_names in enumerate(merged_name_sets):
            if folded_names & existing_folded_names:
                merged = merged_entries[index]
                merged.names.update(entry.names)
                merged.spellings.update(entry.spellings)
                merged.sources.update(entry.sources)
                existing_folded_names.update(folded_names)
                merged_into_existing = True
                break
        if merged_into_existing:
            continue
        merged_entries.append(
            PNEntry(
                display_name=entry.display_name,
                names=set(entry.names),
                spellings=set(entry.spellings),
                onomasticon_spellings=set(entry.onomasticon_spellings),
                sources=set(entry.sources),
            )
        )
        merged_name_sets.append(set(folded_names))

    deduped_entries: dict[str, PNEntry] = {}
    for entry in merged_entries:
        deduped_entries[fold_english_text(entry.display_name)] = entry
    return deduped_entries


def count_mentions(
    train_path: Path,
    entries: dict[str, PNEntry],
) -> tuple[Counter[str], Counter[str]]:
    variant_index: dict[str, PNEntry] = {}
    for entry in entries.values():
        for name in entry.names:
            variant_index[fold_english_text(name)] = entry
    name_regex = build_name_regex(variant_index)

    mention_counter: Counter[str] = Counter()
    row_counter: Counter[str] = Counter()

    with train_path.open("r", encoding="utf-8", newline="") as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            translation = row.get("translation", "") or ""
            mentions = find_pn_mentions(translation, variant_index, name_regex)
            for entry in mentions:
                mention_counter[entry.display_name] += 1
                row_counter[entry.display_name] += 1

    return mention_counter, row_counter


def count_surface_mentions(
    train_path: Path,
    entries: dict[str, PNEntry],
) -> tuple[Counter[str], Counter[str]]:
    variant_index: dict[str, PNEntry] = {}
    for entry in entries.values():
        for name in entry.names:
            variant_index[fold_english_text(name)] = entry
    name_regex = build_name_regex(variant_index)

    mention_counter: Counter[str] = Counter()
    row_counter: Counter[str] = Counter()

    with train_path.open("r", encoding="utf-8", newline="") as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            translation = row.get("translation", "") or ""
            if name_regex is None:
                continue
            original_translation = f" {translation} "
            seen_surfaces_in_row: set[str] = set()
            for entry in find_pn_mentions(translation, variant_index, name_regex):
                matched_surface = None
                for name in sorted(entry.names, key=len, reverse=True):
                    original_pattern = re.compile(
                        rf"(?<![A-Za-zÀ-ÖØ-öø-ÿĀ-žḀ-ỿ-])({re.escape(name)})(?![A-Za-zÀ-ÖØ-öø-ÿĀ-žḀ-ỿ-])"
                    )
                    found = original_pattern.search(original_translation)
                    if found:
                        matched_surface = found.group(1)
                        break
                if not matched_surface:
                    continue
                mention_counter[matched_surface] += 1
                seen_surfaces_in_row.add(matched_surface)
            for surface in seen_surfaces_in_row:
                row_counter[surface] += 1

    return mention_counter, row_counter


def build_rows(entity_type: str, entries: dict[str, PNEntry], mention_counter: Counter[str], row_counter: Counter[str]) -> list[dict[str, str]]:
    rows = []
    for entity_name, mention_count in mention_counter.most_common():
        entry = entries.get(fold_english_text(entity_name))
        rows.append(
            {
                "entity_type": entity_type,
                "name": entity_name,
                "mention_count": str(mention_count),
                "row_count": str(row_counter[entity_name]),
                "sources": "; ".join(sorted(entry.sources)) if entry else "",
                "variants": "; ".join(sorted(entry.names)) if entry else entity_name,
                "spellings": "; ".join(sorted(entry.spellings)) if entry else "",
            }
        )
    return rows


def build_surface_rows(entity_type: str, mention_counter: Counter[str], row_counter: Counter[str]) -> list[dict[str, str]]:
    rows = []
    for surface, mention_count in mention_counter.most_common():
        rows.append(
            {
                "entity_type": entity_type,
                "name": surface,
                "mention_count": str(mention_count),
                "row_count": str(row_counter[surface]),
                "sources": "translation_surface",
                "variants": surface,
                "spellings": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    pn_entries = load_pn_index(args.onomasticon_path, args.lexicon_path)
    gn_entries = load_gn_index(args.lexicon_path)

    pn_mentions, pn_rows = count_mentions(args.train_path, pn_entries)
    gn_mentions, gn_rows = count_mentions(args.train_path, gn_entries)
    pn_surface_mentions, pn_surface_rows = count_surface_mentions(args.train_path, pn_entries)
    gn_surface_mentions, gn_surface_rows = count_surface_mentions(args.train_path, gn_entries)

    output_rows = build_rows("PN", pn_entries, pn_mentions, pn_rows)
    output_rows.extend(build_rows("GN", gn_entries, gn_mentions, gn_rows))
    output_rows.extend(build_surface_rows("PN_SURFACE", pn_surface_mentions, pn_surface_rows))
    output_rows.extend(build_surface_rows("GN_SURFACE", gn_surface_mentions, gn_surface_rows))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "entity_type",
                "name",
                "mention_count",
                "row_count",
                "sources",
                "variants",
                "spellings",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded PN entries: {len(pn_entries)}")
    print(f"Loaded GN entries: {len(gn_entries)}")
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
