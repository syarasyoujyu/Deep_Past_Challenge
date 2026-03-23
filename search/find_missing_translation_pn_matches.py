from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_ONOMASTICON_PATH = PROJECT_ROOT / "data" / "onomasticon_refined.csv"
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "search" / "train_translation_missing_pn_matches.csv"
DEFAULT_FREQUENCY_OUTPUT_PATH = PROJECT_ROOT / "search" / "train_translation_pn_frequencies.csv"


_SUBSCRIPT_TRANS = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "—": "-",
        "–": "-",
    }
)
_WS_RE = re.compile(r"\s+")
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})
_BIG_GAP_RE = re.compile(r"<\s*big_gap\s*>", re.IGNORECASE)
_X_GAP_RE = re.compile(r"\[\s*x\s*\]", re.IGNORECASE)
_ANGLE_X_GAP_RE = re.compile(r"⌈\s*x\s*⌉", re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)", re.IGNORECASE)
_NON_GAP_ANGLE_TAG_RE = re.compile(r"<(?!\s*gap\s*>)[^>]*>", re.IGNORECASE)
_SLASH_VARIANT_RE = re.compile(r"(?P<left>[^/\s;]+?)/(?P<right>[^/\s;]+)")
_BRACKETS_RE = re.compile(r"[\[\]]")
_ANGLE_BRACKETS_RE = re.compile(r"[⌈⌉]")
_PARENS_RE = re.compile(r"[()]")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}", re.IGNORECASE)
_DET_UPPER_RE = re.compile(r"\(([A-ZŠṬṢḪÀ-ÖØ-ÞŠḀ-ỿ0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([a-zšṭṣḫà-öø-ÿšḁ-ỿ]{1,4})\)")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}
_ALLOWED_FRACS = [
    (1/6, "0.16666"),
    (1/4, "0.25"),
    (1/3, "0.33333"),
    (1/2, "0.5"),
    (2/3, "0.66666"),
    (3/4, "0.75"),
    (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_NUMERIC_PLUS_CHARS = "0-9¼½¾⅓⅔⅙⅚⅝"
_NUMERIC_TOKEN_RE = rf"[{_NUMERIC_PLUS_CHARS}]+(?:\.[0-9]+)?"
_NUMERIC_PLUS_EXPR_RE = re.compile(
    rf"(?P<left>{_NUMERIC_TOKEN_RE})\s*\+\s*(?P<right>{_NUMERIC_TOKEN_RE})"
)
_HYPHEN_PLUS_RE = re.compile(r"-\s*\++")
_PLUS_HYPHEN_RE = re.compile(r"\++\s*-")
_NON_NUMERIC_ADJ_PLUS_RE = re.compile(
    rf"(?<=[^\s{_NUMERIC_PLUS_CHARS}])\+(?=[^\s{_NUMERIC_PLUS_CHARS}])"
)
_NON_NUMERIC_PLUS_RE = re.compile(rf"(?<![{_NUMERIC_PLUS_CHARS}])\+|\+(?![{_NUMERIC_PLUS_CHARS}])")
_NAME_BOUNDARY_CHARS = "A-Za-zÀ-ÖØ-öø-ÿĀ-žḀ-ỿ-"
_CHAR_TRANS = str.maketrans(
    {
        "ḫ": "h",
        "Ḫ": "H",
        "ʾ": "",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "—": "-",
        "–": "-",
    }
)
_SUB_X = "ₓ"


@dataclass
class PNEntry:
    display_name: str
    names: set[str] = field(default_factory=set)
    spellings: set[str] = field(default_factory=set)
    onomasticon_spellings: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find PN mentions in translation whose corresponding transliteration "
            "spellings are not present in the same training row."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--onomasticon-path", type=Path, default=DEFAULT_ONOMASTICON_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--frequency-output-path", type=Path, default=DEFAULT_FREQUENCY_OUTPUT_PATH)
    return parser.parse_args()


def fold_english_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = _WS_RE.sub(" ", normalized).strip().lower()
    return normalized


def _ascii_to_diacritics(text: str) -> str:
    normalized = str(text)
    normalized = normalized.replace("sz", "š").replace("SZ", "Š")
    normalized = normalized.replace("s,", "ṣ").replace("S,", "Ṣ")
    normalized = normalized.replace("t,", "ṭ").replace("T,", "Ṭ")
    normalized = _V2.sub(lambda m: m.group(1).translate(_ACUTE), normalized)
    normalized = _V3.sub(lambda m: m.group(1).translate(_GRAVE), normalized)
    return normalized


def _frac_repl(match: re.Match[str]) -> str:
    return _EXACT_FRAC_MAP[match.group(0)]


def _canon_decimal(value: float) -> str:
    integer_part = int(value)
    fraction = value - integer_part
    best = min(_ALLOWED_FRACS, key=lambda item: abs(fraction - item[0]))
    if abs(fraction - best[0]) <= _FRAC_TOL:
        decimal = best[1]
        if integer_part == 0:
            return decimal
        return f"{integer_part}{decimal[1:]}" if decimal.startswith("0.") else f"{integer_part}+{decimal}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _normalize_plus_usage(text: str) -> str:
    normalized = str(text)
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = _NUMERIC_PLUS_EXPR_RE.sub(r"\g<left>+\g<right>", normalized)
    normalized = _HYPHEN_PLUS_RE.sub("-", normalized)
    normalized = _PLUS_HYPHEN_RE.sub("-", normalized)
    normalized = _NON_NUMERIC_ADJ_PLUS_RE.sub("-", normalized)
    normalized = _NON_NUMERIC_PLUS_RE.sub("", normalized)
    return normalized


def _keep_left_slash_variant(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _SLASH_VARIANT_RE.sub(r"\g<left>", current)
    return current


def normalize_spelling(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    normalized = _BIG_GAP_RE.sub("<gap>", normalized)
    normalized = _X_GAP_RE.sub("<gap>", normalized)
    normalized = _ANGLE_X_GAP_RE.sub("<gap>", normalized)
    normalized = _ELLIPSIS_RE.sub("<gap>", normalized)
    normalized = _NON_GAP_ANGLE_TAG_RE.sub("", normalized)
    normalized = _keep_left_slash_variant(normalized)
    normalized = _BRACKETS_RE.sub("", normalized)
    normalized = _ANGLE_BRACKETS_RE.sub("", normalized)
    normalized = _PARENS_RE.sub("", normalized)
    normalized = _MULTI_GAP_RE.sub("<gap> ", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def normalize_transliteration_for_match(text: str) -> str:
    # Search-time normalization should first apply the same cleanup we use
    # for curated spelling fields (gap tags, brackets, slash alternatives, etc.).
    normalized = normalize_spelling(text)
    normalized = _ascii_to_diacritics(normalized)
    normalized = _DET_UPPER_RE.sub(r"\1", normalized)
    normalized = _DET_LOWER_RE.sub(r"{\1}", normalized)
    normalized = _BIG_GAP_RE.sub("<gap>", normalized)
    normalized = _X_GAP_RE.sub("<gap>", normalized)
    normalized = _ANGLE_X_GAP_RE.sub("<gap>", normalized)
    normalized = _ELLIPSIS_RE.sub("<gap>", normalized)
    normalized = normalized.translate(_CHAR_TRANS)
    normalized = normalized.replace(_SUB_X, "")
    normalized = _KUBABBAR_RE.sub("KÙ.BABBAR", normalized)
    normalized = _EXACT_FRAC_RE.sub(_frac_repl, normalized)
    normalized = _FLOAT_RE.sub(lambda m: _canon_decimal(float(m.group(1))), normalized)
    normalized = normalized.replace(",", "")
    normalized = _normalize_plus_usage(normalized)
    normalized = _WS_RE.sub(" ", normalized).strip().lower()
    return normalized


def fold_onomasticon_spelling_for_match(text: str) -> str:
    # Relaxed comparison for onomasticon-only spellings.
    # This intentionally collapses i/í/ī (and other combining-mark variants)
    # so entries like a-la-hi-im can still match a-lá-hi-im.
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.lower()
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def _split_name_variants(raw_name: str) -> list[str]:
    text = str(raw_name or "")
    variants = re.split(r"\n|;|\|", text)
    cleaned_variants = []
    for variant in variants:
        normalized = " ".join(variant.split()).strip()
        if not normalized:
            continue
        # Lexicon alt forms sometimes contain placeholder-like one-letter entries
        # such as "a"; those should not be treated as PN mentions in translation.
        if len(normalized) <= 1:
            continue
        cleaned_variants.append(normalized)
    return cleaned_variants


def add_entry(
    variant_index: dict[str, PNEntry],
    names: list[str],
    spelling: str,
    source: str,
) -> None:
    normalized_names = [
        " ".join(str(name or "").split()).strip()
        for name in names
        if " ".join(str(name or "").split()).strip()
    ]
    normalized_spelling = normalize_transliteration_for_match(spelling)
    if not normalized_names or not normalized_spelling:
        return

    folded_names = [fold_english_text(name) for name in normalized_names]
    entry = None
    for key in folded_names:
        existing = variant_index.get(key)
        if existing is not None:
            entry = existing
            break

    if entry is None:
        entry = PNEntry(display_name=normalized_names[0])

    for name, key in zip(normalized_names, folded_names):
        entry.names.add(name)
        variant_index[key] = entry
    entry.spellings.add(normalized_spelling)
    if source == "onomasticon":
        entry.onomasticon_spellings.add(normalized_spelling)
    entry.sources.add(source)


def load_pn_index(onomasticon_path: Path, lexicon_path: Path) -> dict[str, PNEntry]:
    variant_index: dict[str, PNEntry] = {}

    onomasticon_rows: list[tuple[list[str], set[str]]] = []
    with onomasticon_path.open("r", encoding="utf-8", newline="") as onomasticon_file:
        reader = csv.DictReader(onomasticon_file)
        for row in reader:
            names = _split_name_variants(row.get("Name", ""))
            normalized_spellings = set()
            for spelling in str(row.get("Spellings_semicolon_separated", "")).split(";"):
                spelling = spelling.strip()
                if spelling:
                    normalized_spelling = normalize_transliteration_for_match(spelling)
                    if normalized_spelling:
                        normalized_spellings.add(normalized_spelling)
            if names and normalized_spellings:
                onomasticon_rows.append((names, normalized_spellings))

    dominated_onomasticon_rows: set[int] = set()
    for row_index, (_, spellings) in enumerate(onomasticon_rows):
        if not spellings:
            continue
        for other_index, (_, other_spellings) in enumerate(onomasticon_rows):
            if row_index == other_index:
                continue
            if spellings < other_spellings:
                dominated_onomasticon_rows.add(row_index)
                break

    for row_index, (names, spellings) in enumerate(onomasticon_rows):
        if row_index in dominated_onomasticon_rows:
            continue
        for spelling in spellings:
            add_entry(variant_index, names, spelling, "onomasticon")

    with lexicon_path.open("r", encoding="utf-8", newline="") as lexicon_file:
        reader = csv.DictReader(lexicon_file)
        for row in reader:
            if str(row.get("type", "")).strip() != "PN":
                continue
            names = []
            # `norm` often reflects inflected / orthographic variants like
            # "Šu-Kūbim" for the canonical PN "Šu-Kūbum". For translation-side
            # PN detection we prefer canonical English names only.
            for column in ["lexeme", "Alt_lex"]:
                names.extend(_split_name_variants(row.get(column, "")))
            add_entry(
                variant_index,
                names,
                row.get("form", ""),
                "lexicon",
            )

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
                merged.onomasticon_spellings.update(entry.onomasticon_spellings)
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


def build_name_regex(variant_index: dict[str, PNEntry]) -> re.Pattern[str] | None:
    if not variant_index:
        return None
    patterns = sorted((re.escape(key) for key in variant_index), key=len, reverse=True)
    return re.compile(
        rf"(?<![{_NAME_BOUNDARY_CHARS}])(?:{'|'.join(patterns)})(?![{_NAME_BOUNDARY_CHARS}])"
    )


def find_pn_mentions(
    translation: str,
    variant_index: dict[str, PNEntry],
    name_regex: re.Pattern[str] | None,
) -> list[PNEntry]:
    if name_regex is None:
        return []
    folded_translation = f" {fold_english_text(translation)} "
    original_translation = f" {str(translation or '')} "
    matches: list[PNEntry] = []
    seen: set[int] = set()

    for match in name_regex.finditer(folded_translation):
        key = match.group(0)
        if not key:
            continue
        entry = variant_index.get(key)
        if entry is None:
            continue
        entry_id = id(entry)
        if entry_id in seen:
            continue
        # Translation-side PN lookup should prefer actual named-entity mentions.
        # This rejects lowercase common-noun matches such as "mina".
        has_original_name_match = False
        for name in sorted(entry.names, key=len, reverse=True):
            original_pattern = re.compile(
                rf"(?<![{_NAME_BOUNDARY_CHARS}]){re.escape(name)}(?![{_NAME_BOUNDARY_CHARS}])"
            )
            if original_pattern.search(original_translation):
                has_original_name_match = True
                break
        if not has_original_name_match:
            continue
        matches.append(entry)
        seen.add(entry_id)

    return matches


def transliteration_contains_any_spelling(transliteration: str, entry: PNEntry) -> bool:
    normalized_text = normalize_transliteration_for_match(transliteration)
    normalized_transliteration = f" {normalized_text} "
    for spelling in entry.spellings:
        if f" {spelling} " in normalized_transliteration:
            return True
        if spelling in normalized_transliteration:
            return True

    if entry.onomasticon_spellings:
        folded_transliteration = f" {fold_onomasticon_spelling_for_match(normalized_text)} "
        for spelling in entry.onomasticon_spellings:
            folded_spelling = fold_onomasticon_spelling_for_match(spelling)
            if f" {folded_spelling} " in folded_transliteration:
                return True
            if folded_spelling in folded_transliteration:
                return True
    return False


def main() -> None:
    args = parse_args()
    pn_entries = load_pn_index(args.onomasticon_path, args.lexicon_path)
    variant_index: dict[str, PNEntry] = {}
    for entry in pn_entries.values():
        for name in entry.names:
            variant_index[fold_english_text(name)] = entry
    name_regex = build_name_regex(variant_index)

    missing_rows: list[dict[str, str]] = []
    pn_frequency_counter: Counter[str] = Counter()
    pn_row_counter: Counter[str] = Counter()
    total_rows = 0
    rows_with_pn = 0
    total_pn_mentions = 0

    with args.train_path.open("r", encoding="utf-8", newline="") as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            total_rows += 1
            translation = row.get("translation", "") or ""
            transliteration = row.get("transliteration", "") or ""
            mentions = find_pn_mentions(translation, variant_index, name_regex)
            if not mentions:
                continue

            rows_with_pn += 1
            for entry in mentions:
                total_pn_mentions += 1
                pn_frequency_counter[entry.display_name] += 1
                pn_row_counter[entry.display_name] += 1
                if transliteration_contains_any_spelling(transliteration, entry):
                    continue

                missing_rows.append(
                    {
                        "oare_id": row.get("oare_id", ""),
                        "pn_name": entry.display_name,
                        "pn_sources": "; ".join(sorted(entry.sources)),
                        "pn_variants": "; ".join(sorted(entry.names)),
                        "spellings_checked": "; ".join(sorted(entry.spellings)),
                        "translation": translation,
                        "transliteration": transliteration,
                    }
                )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "oare_id",
        "pn_name",
        "pn_sources",
        "pn_variants",
        "spellings_checked",
        "translation",
        "transliteration",
    ]
    with args.output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    frequency_rows = []
    for pn_name, mention_count in pn_frequency_counter.most_common():
        entry = pn_entries.get(fold_english_text(pn_name))
        frequency_rows.append(
            {
                "pn_name": pn_name,
                "mention_count": mention_count,
                "row_count": pn_row_counter[pn_name],
                "pn_sources": "; ".join(sorted(entry.sources)) if entry else "",
                "pn_variants": "; ".join(sorted(entry.names)) if entry else pn_name,
                "spellings_checked": "; ".join(sorted(entry.spellings)) if entry else "",
            }
        )

    args.frequency_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.frequency_output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "pn_name",
                "mention_count",
                "row_count",
                "pn_sources",
                "pn_variants",
                "spellings_checked",
            ],
        )
        writer.writeheader()
        writer.writerows(frequency_rows)

    print(f"Loaded PN entries: {len(pn_entries)}")
    print(f"Rows scanned: {total_rows}")
    print(f"Rows with PN mention(s): {rows_with_pn}")
    print(f"Total PN mentions found in translation: {total_pn_mentions}")
    print(f"Missing transliteration matches: {len(missing_rows)}")
    print(f"Wrote: {args.output_path}")
    print(f"Wrote frequencies: {args.frequency_output_path}")


if __name__ == "__main__":
    main()
