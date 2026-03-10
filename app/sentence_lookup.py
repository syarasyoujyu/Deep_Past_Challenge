from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train_truncated.csv"
SENTENCE_PATH = PROJECT_ROOT / "data" / "Sentences_Oare_FirstWord_LinNum.csv"


@lru_cache(maxsize=1)
def load_train_index() -> dict[str, dict[str, str]]:
    with TRAIN_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return {row["oare_id"]: row for row in csv.DictReader(csv_file)}


@lru_cache(maxsize=1)
def load_sentence_index() -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    with SENTENCE_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            index.setdefault(row["text_uuid"], []).append(row)
    return index


def extract_sentence_matches(oare_id: str) -> list[dict[str, str]]:
    train_row = load_train_index().get(oare_id)
    if not train_row:
        return []

    transliteration_words = [token for token in train_row["transliteration"].split(" ") if token]
    matches: list[dict[str, str]] = []

    for row in load_sentence_index().get(oare_id, []):
        try:
            word_index = int(row["first_word_number"]) - 1
        except ValueError:
            word_index = -1

        train_word = (
            transliteration_words[word_index]
            if 0 <= word_index < len(transliteration_words)
            else ""
        )
        same = "Yes" if train_word == row["first_word_spelling"] else "No"
        translation = row["translation"] if same == "Yes" else ""

        matches.append(
            {
                "sentence_obj_in_text": row["sentence_obj_in_text"],
                "first_word_number": row["first_word_number"],
                "first_word_spelling": row["first_word_spelling"],
                "train_word": train_word,
                "same": same,
                "translation": translation,
            }
        )

    return matches


@lru_cache(maxsize=1)
def count_texts_with_same_no() -> int:
    count = 0
    for oare_id in load_sentence_index():
        matches = extract_sentence_matches(oare_id)
        if any(match["same"] == "No" for match in matches):
            count += 1
    return count
