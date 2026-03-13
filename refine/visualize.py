from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "train_refined.csv"
DISALLOWED_TRANSLITERATION_SYLLABLES_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "disallowed_transliteration_syllables.csv"
)
ANGLE_BRACKET_PATTERN = re.compile(r"<[^<>]+>")
TARGET_FIELDS = ("transliteration", "translation")
ALLOWED_TRANSLITERATION_CHARACTERS = set(
    "!+-.0123456789:<>ABDEGHIKLMNPQRSTUWZ_abdeghiklmnpqrstuwz{}¼½ÀÁÈÉÌÍÙÚàáèéìíùúİışŠšṢṣṬṭ…⅓⅔⅙⅚"
)
ALLOWED_TRANSLATION_CHARACTERS = set(
    "!\"'()+,-.0123456789:;<>?ABCDEFGHIJKLMNOPQRSTUWYZ[]_abcdefghijklmnopqrstuvwxyz¼½àâāēğīışŠšūṢṣṬṭ–—‘’“”⅓⅔⅙⅚"
)


def load_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def extract_angle_bracket_tokens(text: str) -> list[str]:
    return ANGLE_BRACKET_PATTERN.findall(text)


def collect_unique_tokens(rows: list[dict[str, str]]) -> list[str]:
    unique_tokens: set[str] = set()

    for row in rows:
        for field in TARGET_FIELDS:
            unique_tokens.update(extract_angle_bracket_tokens(row[field]))

    return sorted(unique_tokens)


def collect_unbalanced_angle_bracket_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unbalanced_rows: list[dict[str, str]] = []

    for row in rows:
        for field in TARGET_FIELDS:
            text = row[field]
            left_count = text.count("<")
            right_count = text.count(">")
            if left_count == right_count:
                continue

            unbalanced_rows.append(
                {
                    "oare_id": row["oare_id"],
                    "field": field,
                    "left_count": str(left_count),
                    "right_count": str(right_count),
                    "text": text,
                }
            )

    return unbalanced_rows


def collect_disallowed_characters(
    rows: list[dict[str, str]], field: str, allowed_characters: set[str]
) -> tuple[Counter[str], dict[str, str]]:
    disallowed_characters: Counter[str] = Counter()
    example_oare_ids: dict[str, str] = {}

    for row in rows:
        for character in row[field]:
            if character.isspace():
                continue
            if character in allowed_characters:
                continue
            disallowed_characters[character] += 1
            example_oare_ids.setdefault(character, row["oare_id"])

    return disallowed_characters, example_oare_ids


def collect_disallowed_transliteration_syllables(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    disallowed_syllables: dict[str, dict[str, str]] = {}

    for row in rows:
        for word in row["transliteration"].split():
            for syllable in word.split("-"):
                if not syllable:
                    continue

                disallowed_characters = sorted(
                    {
                        character
                        for character in syllable
                        if not character.isspace()
                        and character not in ALLOWED_TRANSLITERATION_CHARACTERS
                    }
                )
                if not disallowed_characters:
                    continue

                disallowed_syllables.setdefault(
                    syllable,
                    {
                        "oare_id": row["oare_id"],
                        "syllable": syllable,
                        "disallowed_characters": "".join(disallowed_characters),
                    },
                )

    return sorted(disallowed_syllables.values(), key=lambda row: row["syllable"])


def write_disallowed_transliteration_syllables(rows: list[dict[str, str]]) -> None:
    with DISALLOWED_TRANSLITERATION_SYLLABLES_OUTPUT_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["oare_id", "syllable", "disallowed_characters"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    unique_tokens = collect_unique_tokens(rows)
    unbalanced_rows = collect_unbalanced_angle_bracket_rows(rows)
    disallowed_transliteration_characters, transliteration_examples = collect_disallowed_characters(
        rows, "transliteration", ALLOWED_TRANSLITERATION_CHARACTERS
    )
    disallowed_translation_characters, translation_examples = collect_disallowed_characters(
        rows, "translation", ALLOWED_TRANSLATION_CHARACTERS
    )
    disallowed_transliteration_syllables = collect_disallowed_transliteration_syllables(rows)
    write_disallowed_transliteration_syllables(disallowed_transliteration_syllables)

    print(f"Input: {INPUT_PATH}")
    print(f"Unique <...> token count: {len(unique_tokens)}")
    for token in unique_tokens:
        print(token)

    print(f"\nRows with unmatched < and > count: {len(unbalanced_rows)}")
    for row in unbalanced_rows:
        print(
            f"[{row['field']}] {row['oare_id']} "
            f"<= {row['left_count']} >= {row['right_count']}: {row['text']}"
        )

    print(
        "\nDisallowed transliteration characters "
        f"({len(disallowed_transliteration_characters)} kinds):"
    )
    for character, count in sorted(disallowed_transliteration_characters.items()):
        print(
            f"{character!r}: {count} "
            f"(example oare_id: {transliteration_examples[character]})"
        )

    print(
        "\nDisallowed translation characters "
        f"({len(disallowed_translation_characters)} kinds):"
    )
    for character, count in sorted(disallowed_translation_characters.items()):
        print(
            f"{character!r}: {count} "
            f"(example oare_id: {translation_examples[character]})"
        )

    print(
        "\nWrote disallowed transliteration syllables to "
        f"{DISALLOWED_TRANSLITERATION_SYLLABLES_OUTPUT_PATH} "
        f"({len(disallowed_transliteration_syllables)} rows)"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
