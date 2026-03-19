from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted_augmented_v1.csv"
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted_augmented_v1_item_swap.csv"
DEFAULT_MAX_AUG_PER_ROW = 3

ITEM_GROUPS = {
    "metal": ["silver", "gold", "tin", "copper"],
    "textile": ["textile", "garment", "robe", "shawl"],
}
DEFINITION_TO_GROUP = {
    item_definition: group_name
    for group_name, item_definitions in ITEM_GROUPS.items()
    for item_definition in item_definitions
}
FORM_GLOSS_OVERRIDES = {
    "AN.NA": "tin",
    "KÙ.GI": "gold",
    "URUDU": "copper",
}


@dataclass(frozen=True)
class LexiconItem:
    group_name: str
    form: str
    form_original: str
    english_gloss: str


@dataclass(frozen=True)
class TranslationMatch:
    start: int
    end: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment v1 sentence data by swapping item/material words such as "
            "silver/gold/tin/copper and matching translation words."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-aug-per-row", type=int, default=DEFAULT_MAX_AUG_PER_ROW)
    parser.add_argument(
        "--include-original-rows",
        action="store_true",
        help="Include original input rows in the output CSV before augmented rows.",
    )
    return parser.parse_args()


def simplify_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_marks.lower()


def singularize_definition(definition: str) -> str:
    simplified = simplify_text(definition).strip()
    simplified = simplified.replace('"', "").replace("'", "")
    simplified = re.split(r"[,;/()]", simplified, maxsplit=1)[0].strip()
    simplified = simplified.removeprefix("a ").removeprefix("an ").removeprefix("the ")
    if simplified.endswith("s"):
        simplified = simplified[:-1]
    return simplified


def pluralize(word: str) -> str:
    if word.endswith("s"):
        return word
    return f"{word}s"


def load_input_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_lexicon_items(path: Path) -> tuple[dict[str, list[LexiconItem]], dict[str, list[LexiconItem]]]:
    by_token: dict[str, list[LexiconItem]] = {}
    by_group: dict[str, list[LexiconItem]] = {group_name: [] for group_name in ITEM_GROUPS}

    with path.open("r", encoding="utf-8", newline="") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        for row in reader:
            if row.get("type", "").strip() != "word":
                continue
            form = row.get("form", "").strip()
            definition_key = FORM_GLOSS_OVERRIDES.get(form, singularize_definition(row.get("definition", "")))
            group_name = DEFINITION_TO_GROUP.get(definition_key)
            if group_name is None:
                continue

            item = LexiconItem(
                group_name=group_name,
                form=form,
                form_original=row.get("form_original", "").strip(),
                english_gloss=definition_key,
            )
            if not item.form or not item.english_gloss:
                continue

            by_group[group_name].append(item)
            for token in {item.form, item.form_original}:
                token = token.strip()
                if token:
                    by_token.setdefault(token, []).append(item)

    deduped_by_group: dict[str, list[LexiconItem]] = {}
    for group_name, items in by_group.items():
        seen: set[tuple[str, str]] = set()
        deduped: list[LexiconItem] = []
        for item in items:
            key = (item.form, item.english_gloss)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        deduped_by_group[group_name] = sorted(deduped, key=lambda item: (item.english_gloss, item.form))

    return by_token, deduped_by_group


def find_translation_match(translation: str, english_gloss: str) -> TranslationMatch | None:
    patterns = [
        rf"\b{re.escape(pluralize(english_gloss))}\b",
        rf"\b{re.escape(english_gloss)}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, translation, flags=re.IGNORECASE)
        if match:
            return TranslationMatch(start=match.start(), end=match.end(), text=match.group(0))
    return None


def choose_replacement(
    row_oare_id: str,
    token_index: int,
    source_item: LexiconItem,
    items_by_group: dict[str, list[LexiconItem]],
) -> LexiconItem | None:
    candidates = [
        candidate
        for candidate in items_by_group[source_item.group_name]
        if candidate.form != source_item.form and candidate.english_gloss != source_item.english_gloss
    ]
    if not candidates:
        return None

    digest = hashlib.md5(
        f"{row_oare_id}|{token_index}|{source_item.form}|{source_item.english_gloss}".encode("utf-8")
    ).hexdigest()
    index = int(digest, 16) % len(candidates)
    return candidates[index]


def replace_token(tokens: list[str], token_index: int, replacement_form: str) -> str:
    replaced = list(tokens)
    replaced[token_index] = replacement_form
    return " ".join(replaced)


def replace_translation_span(
    translation: str,
    match: TranslationMatch,
    replacement_gloss: str,
) -> str:
    replacement_text = replacement_gloss
    if match.text.lower().endswith("s"):
        replacement_text = pluralize(replacement_text)
    return translation[: match.start] + replacement_text + translation[match.end :]


def build_augmented_rows(
    input_rows: list[dict[str, str]],
    items_by_token: dict[str, list[LexiconItem]],
    items_by_group: dict[str, list[LexiconItem]],
    max_aug_per_row: int,
) -> list[dict[str, str]]:
    augmented_rows: list[dict[str, str]] = []

    for row in input_rows:
        transliteration = row["transliteration"].strip()
        translation = row["translation"].strip()
        tokens = transliteration.split()
        produced = 0

        for token_index, token in enumerate(tokens):
            source_items = items_by_token.get(token, [])
            if not source_items:
                continue

            source_item = source_items[0]
            translation_match = find_translation_match(translation, source_item.english_gloss)
            if translation_match is None:
                continue

            replacement = choose_replacement(
                row_oare_id=row["oare_id"],
                token_index=token_index,
                source_item=source_item,
                items_by_group=items_by_group,
            )
            if replacement is None:
                continue

            augmented_row = dict(row)
            augmented_row["oare_id"] = f"{row['oare_id']}--item-swap-{produced + 1}"
            augmented_row["transliteration"] = replace_token(tokens, token_index, replacement.form)
            augmented_row["translation"] = replace_translation_span(
                translation=translation,
                match=translation_match,
                replacement_gloss=replacement.english_gloss,
            )
            augmented_row["augmentation_type"] = f"{source_item.group_name}_item_swap"
            augmented_row["replaced_source_form"] = source_item.form
            augmented_row["replacement_form"] = replacement.form
            augmented_row["replaced_source_name"] = translation_match.text
            augmented_row["replacement_name"] = replacement.english_gloss
            augmented_rows.append(augmented_row)

            produced += 1
            if produced >= max_aug_per_row:
                break

    return augmented_rows


def write_rows(
    path: Path,
    original_rows: list[dict[str, str]],
    augmented_rows: list[dict[str, str]],
    include_original_rows: bool,
) -> None:
    fieldnames = list(original_rows[0].keys())
    extra_fields = [
        "augmentation_type",
        "replaced_source_form",
        "replacement_form",
        "replaced_source_name",
        "replacement_name",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    rows_to_write: list[dict[str, str]] = []
    if include_original_rows:
        for row in original_rows:
            enriched = dict(row)
            for field in extra_fields:
                enriched.setdefault(field, "")
            rows_to_write.append(enriched)

    for row in augmented_rows:
        enriched = dict(row)
        for field in extra_fields:
            enriched.setdefault(field, "")
        rows_to_write.append(enriched)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)


def main() -> None:
    args = parse_args()
    input_rows = load_input_rows(args.input_path)
    items_by_token, items_by_group = load_lexicon_items(args.lexicon_path)
    augmented_rows = build_augmented_rows(
        input_rows=input_rows,
        items_by_token=items_by_token,
        items_by_group=items_by_group,
        max_aug_per_row=max(args.max_aug_per_row, 1),
    )
    write_rows(
        path=args.output_path,
        original_rows=input_rows,
        augmented_rows=augmented_rows,
        include_original_rows=args.include_original_rows,
    )
    print(f"Input rows: {len(input_rows)}")
    print(f"Augmented rows: {len(augmented_rows)}")
    print(f"Wrote output to {args.output_path}")


if __name__ == "__main__":
    main()
