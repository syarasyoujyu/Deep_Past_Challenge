from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OA_LEXICON_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL.csv"
DICTIONARY_PATH = PROJECT_ROOT / "data" / "eBL_Dictionary.csv"
DEFINITION_SEPARATOR = "\n---------\n"


def split_space_separated_text(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]


def strip_dictionary_suffix(word: str) -> str:
    return re.sub(r"\s+[IVXLCDM]+$", "", word.strip())


@lru_cache(maxsize=1)
def load_dictionary_definition_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    with DICTIONARY_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            stripped_word = strip_dictionary_suffix(row["word"])
            if not stripped_word:
                continue
            index.setdefault(stripped_word, []).append(row["definition"])
    return index


@lru_cache(maxsize=1)
def load_lexicon_index() -> dict[str, list[tuple[list[str], dict[str, str]]]]:
    index: dict[str, list[tuple[list[str], dict[str, str]]]] = {}
    with OA_LEXICON_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            form_tokens = split_space_separated_text(row["form"])
            if not form_tokens:
                continue
            index.setdefault(form_tokens[0], []).append((form_tokens, row))
    return index


def extract_lexicon_matches(transliteration: str) -> list[dict[str, str]]:
    transliteration_tokens = split_space_separated_text(transliteration)
    lexicon_index = load_lexicon_index()
    definition_index = load_dictionary_definition_index()
    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for index, token in enumerate(transliteration_tokens):
        for form_tokens, row in lexicon_index.get(token, []):
            form_length = len(form_tokens)
            if transliteration_tokens[index : index + form_length] != form_tokens:
                continue

            key = (
                row["form"],
                row["norm"],
                row["type"],
                row["lexeme"],
                row["Female(f)"],
            )
            if key in seen:
                continue
            seen.add(key)

            definitions = definition_index.get(row["lexeme"], [])
            matches.append(
                {
                    "form": row["form"],
                    "norm": row["norm"],
                    "lexeme": row["lexeme"],
                    "type": row["type"],
                    "female": row["Female(f)"],
                    "definitions": DEFINITION_SEPARATOR.join(definitions),
                }
            )

    return matches
