from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DICTIONARY_PATH = PROJECT_ROOT / "data" / "eBL_Dictionary.csv"


def split_space_separated_text(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]


def contains_token_sequence(tokens: list[str], candidate_tokens: list[str]) -> bool:
    candidate_length = len(candidate_tokens)
    if candidate_length == 0 or candidate_length > len(tokens):
        return False

    for index in range(len(tokens) - candidate_length + 1):
        if tokens[index : index + candidate_length] == candidate_tokens:
            return True

    return False


@lru_cache(maxsize=1)
def load_dictionary_entries() -> list[tuple[list[str], dict[str, str]]]:
    entries: list[tuple[list[str], dict[str, str]]] = []
    with DICTIONARY_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            word_tokens = split_space_separated_text(row["word"])
            if not word_tokens:
                continue
            entries.append((word_tokens, row))
    return entries


def extract_dictionary_matches(transliteration: str) -> list[dict[str, str]]:
    transliteration_tokens = split_space_separated_text(transliteration)
    matches: list[dict[str, str]] = []

    for word_tokens, row in load_dictionary_entries():
        if contains_token_sequence(transliteration_tokens, word_tokens):
            matches.append(row)

    return matches
