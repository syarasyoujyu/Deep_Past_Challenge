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
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_splitted_augmented_v1_pn_gn_swap.csv"
DEFAULT_MAX_AUG_PER_ROW = 3


@dataclass(frozen=True)
class LexiconName:
    entity_type: str
    form: str
    form_original: str
    norm: str
    lexeme: str

    @property
    def display_name(self) -> str:
        return (self.norm or self.lexeme).strip()


@dataclass(frozen=True)
class TranslationMatch:
    start: int
    end: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment train_refined_splitted_augmented_v1.csv by swapping PN/GN "
            "tokens in transliteration and their aligned names in translation."
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


def build_simplified_with_mapping(text: str) -> tuple[str, list[int]]:
    simplified_chars: list[str] = []
    mapping: list[int] = []
    for original_index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKD", char)
        for piece in normalized:
            if unicodedata.combining(piece):
                continue
            simplified_chars.append(piece.lower())
            mapping.append(original_index)
    return "".join(simplified_chars), mapping


def find_simplified_substring(text: str, candidate: str) -> TranslationMatch | None:
    simplified_text, mapping = build_simplified_with_mapping(text)
    simplified_candidate = simplify_text(candidate)
    if not simplified_candidate:
        return None
    start = simplified_text.find(simplified_candidate)
    if start == -1:
        return None
    end = start + len(simplified_candidate)
    original_start = mapping[start]
    original_end = mapping[end - 1] + 1
    return TranslationMatch(
        start=original_start,
        end=original_end,
        text=text[original_start:original_end],
    )


def load_input_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_lexicon_names(path: Path) -> tuple[dict[str, list[LexiconName]], dict[str, list[LexiconName]]]:
    by_token: dict[str, list[LexiconName]] = {}
    by_type: dict[str, list[LexiconName]] = {"PN": [], "GN": []}

    with path.open("r", encoding="utf-8", newline="") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        for row in reader:
            entity_type = row.get("type", "").strip()
            if entity_type not in {"PN", "GN"}:
                continue
            name = LexiconName(
                entity_type=entity_type,
                form=row.get("form", "").strip(),
                form_original=row.get("form_original", "").strip(),
                norm=row.get("norm", "").strip(),
                lexeme=row.get("lexeme", "").strip(),
            )
            if not name.display_name:
                continue

            by_type[entity_type].append(name)
            for token in {name.form, name.form_original}:
                token = token.strip()
                if token:
                    by_token.setdefault(token, []).append(name)

    deduped_by_type: dict[str, list[LexiconName]] = {}
    for entity_type, names in by_type.items():
        seen: set[tuple[str, str]] = set()
        deduped: list[LexiconName] = []
        for name in names:
            key = (name.form, name.display_name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(name)
        deduped_by_type[entity_type] = sorted(deduped, key=lambda item: (item.display_name, item.form))

    return by_token, deduped_by_type


def candidate_translation_names(name: LexiconName) -> list[str]:
    candidates = [name.norm.strip(), name.lexeme.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        key = simplify_text(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def find_translation_match(translation: str, name: LexiconName) -> TranslationMatch | None:
    for candidate in candidate_translation_names(name):
        match = find_simplified_substring(translation, candidate)
        if match is not None:
            return match
    return None


def choose_replacement(
    row_oare_id: str,
    token_index: int,
    source_name: LexiconName,
    names_by_type: dict[str, list[LexiconName]],
) -> LexiconName | None:
    candidates = [
        candidate
        for candidate in names_by_type[source_name.entity_type]
        if candidate.form != source_name.form
        and simplify_text(candidate.display_name) != simplify_text(source_name.display_name)
    ]
    if not candidates:
        return None

    digest = hashlib.md5(
        f"{row_oare_id}|{token_index}|{source_name.form}|{source_name.display_name}".encode("utf-8")
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
    replacement_name: str,
) -> str:
    return translation[: match.start] + replacement_name + translation[match.end :]


def build_augmented_rows(
    input_rows: list[dict[str, str]],
    names_by_token: dict[str, list[LexiconName]],
    names_by_type: dict[str, list[LexiconName]],
    max_aug_per_row: int,
) -> list[dict[str, str]]:
    augmented_rows: list[dict[str, str]] = []

    for row in input_rows:
        transliteration = row["transliteration"].strip()
        translation = row["translation"].strip()
        tokens = transliteration.split()
        produced = 0

        for token_index, token in enumerate(tokens):
            source_names = names_by_token.get(token, [])
            if not source_names:
                continue

            source_name = source_names[0]
            translation_match = find_translation_match(translation, source_name)
            if translation_match is None:
                continue

            replacement = choose_replacement(
                row_oare_id=row["oare_id"],
                token_index=token_index,
                source_name=source_name,
                names_by_type=names_by_type,
            )
            if replacement is None:
                continue

            augmented_row = dict(row)
            augmented_row["oare_id"] = f"{row['oare_id']}--swap-{produced + 1}"
            augmented_row["transliteration"] = replace_token(tokens, token_index, replacement.form)
            augmented_row["translation"] = replace_translation_span(
                translation=translation,
                match=translation_match,
                replacement_name=replacement.display_name,
            )
            augmented_row["augmentation_type"] = f"{source_name.entity_type}_swap"
            augmented_row["replaced_source_form"] = source_name.form
            augmented_row["replacement_form"] = replacement.form
            augmented_row["replaced_source_name"] = translation_match.text
            augmented_row["replacement_name"] = replacement.display_name
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
    names_by_token, names_by_type = load_lexicon_names(args.lexicon_path)
    augmented_rows = build_augmented_rows(
        input_rows=input_rows,
        names_by_token=names_by_token,
        names_by_type=names_by_type,
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
