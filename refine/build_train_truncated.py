from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_SENTENCE_PATH = PROJECT_ROOT / "data" / "Sentences_Oare_FirstWord_LinNum.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "train_truncated.csv"
EXCLUDED_TRANSLATIONS = {
    "Lullu,",
    "Obverse too broken for translation",
    "10 minas <gap>",
    "1 drink: Aššur-nādā;",
    "1 kutānu-textile for 10 shekel of silver to Šumma-libbi-Aššur;",
}
EXCLUDED_TRANSLATION_PATTERNS = (
    #r"<gap>",
)
RATIO_IQR_MULTIPLIER = 1.5


@dataclass(frozen=True)
class SentenceCheck:
    first_word_index: int
    first_word_spelling: str
    translation: str
    same: bool


def load_train_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def load_sentence_index(path: Path) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            index.setdefault(row["text_uuid"], []).append(row)
    return index


def build_sentence_checks(
    transliteration_words: list[str], sentence_rows: list[dict[str, str]]
) -> list[SentenceCheck]:
    """Sentence_Oare_FirstWord_LinNum.csvと矛盾する文章を検出"""
    checks: list[SentenceCheck] = []

    sorted_rows = sorted(
        sentence_rows,
        key=lambda row: (safe_int(row["first_word_number"]), safe_int(row["sentence_obj_in_text"])),
    )

    for row in sorted_rows:
        first_word_index = safe_int(row["first_word_number"]) - 1
        train_word = (
            transliteration_words[first_word_index]
            if 0 <= first_word_index < len(transliteration_words)
            else None
        )
        checks.append(
            SentenceCheck(
                first_word_index=first_word_index,
                first_word_spelling=row["first_word_spelling"],
                translation=row["translation"].strip(),
                same=train_word == row["first_word_spelling"],
            )
        )

    return checks


def truncate_row(row: dict[str, str], sentence_rows: list[dict[str, str]]) -> tuple[dict[str, str], bool]:
    words = [token for token in row["transliteration"].split(" ") if token]
    if not sentence_rows:
        return row, False

    checks = build_sentence_checks(words, sentence_rows)
    changed = False

    for check in checks:
        if not check.same:
            changed = True
            break

    return row, changed


def safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def compile_excluded_translation_patterns() -> list[re.Pattern[str]]:
    compiled_patterns: list[re.Pattern[str]] = []

    for pattern in EXCLUDED_TRANSLATION_PATTERNS:
        try:
            compiled_patterns.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(
                f"Invalid EXCLUDED_TRANSLATION_PATTERNS entry: {pattern!r}. "
                'Use a valid regex such as r"<gap>" or r"^To .+?:$".'
            ) from exc

    return compiled_patterns


def is_excluded_translation(
    translation: str, compiled_patterns: list[re.Pattern[str]]
) -> bool:
    normalized_translation = translation.strip()
    if normalized_translation in EXCLUDED_TRANSLATIONS:
        return True

    for pattern in compiled_patterns:
        if pattern.search(normalized_translation):
            return True

    return False


def count_words(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def compute_length_ratio_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    ratios: list[float] = []

    for row in rows:
        transliteration_word_count = count_words(row["transliteration"])
        if transliteration_word_count == 0:
            continue

        translation_word_count = count_words(row["translation"])
        ratios.append(translation_word_count / transliteration_word_count)

    if not ratios:
        return {
            "q1": 0.0,
            "q3": 0.0,
            "lower_fence": 0.0,
            "upper_fence": 0.0,
            "lower_whisker": 0.0,
            "upper_whisker": 0.0,
        }

    sorted_ratios = sorted(ratios)
    if len(sorted_ratios) == 1:
        q1 = sorted_ratios[0]
        q3 = sorted_ratios[0]
    else:
        quartiles = statistics.quantiles(sorted_ratios, n=4, method="inclusive")
        q1 = quartiles[0]
        q3 = quartiles[2]

    iqr = q3 - q1
    lower_fence = q1 - (RATIO_IQR_MULTIPLIER * iqr)
    upper_fence = q3 + (RATIO_IQR_MULTIPLIER * iqr)
    non_outliers = [ratio for ratio in sorted_ratios if lower_fence <= ratio <= upper_fence]

    return {
        "q1": q1,
        "q3": q3,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_whisker": non_outliers[0] if non_outliers else sorted_ratios[0],
        "upper_whisker": non_outliers[-1] if non_outliers else sorted_ratios[-1],
    }


def filter_ratio_outlier_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, float]]:
    ratio_stats = compute_length_ratio_stats(rows)
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []

    for row in rows:
        transliteration_word_count = count_words(row["transliteration"])
        if transliteration_word_count == 0:
            removed_rows.append(row)
            continue

        translation_word_count = count_words(row["translation"])
        ratio = translation_word_count / transliteration_word_count
        if ratio < ratio_stats["lower_fence"] or ratio > ratio_stats["upper_fence"]:
            removed_rows.append(row)
            continue
        kept_rows.append(row)

    return kept_rows, removed_rows, ratio_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter train CSV rows by dropping excluded translations, sentence alignment mismatches, "
            "and translation/transliteration length-ratio outliers."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Input train CSV. Defaults to data/train.csv.",
    )
    parser.add_argument(
        "--sentence-path",
        type=Path,
        default=DEFAULT_SENTENCE_PATH,
        help="Sentence index CSV. Defaults to data/Sentences_Oare_FirstWord_LinNum.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Filtered output CSV. Defaults to data/train_truncated.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compiled_patterns = compile_excluded_translation_patterns()
    rows, input_fieldnames = load_train_rows(args.train_path)
    sentence_index = load_sentence_index(args.sentence_path)
    truncated_rows: list[dict[str, str]] = []
    removed_rows = 0
    removed_by_same_no = 0
    removed_by_translation = 0

    for row in rows:
        if is_excluded_translation(row["translation"], compiled_patterns):
            removed_rows += 1
            removed_by_translation += 1
            continue

        truncated_row, changed = truncate_row(row, sentence_index.get(row["oare_id"], []))
        if changed:
            removed_rows += 1
            removed_by_same_no += 1
            continue
        truncated_rows.append(truncated_row)

    truncated_rows, ratio_outlier_rows, ratio_stats = filter_ratio_outlier_rows(truncated_rows)
    removed_by_ratio_outlier = len(ratio_outlier_rows)
    removed_rows += removed_by_ratio_outlier

    output_fieldnames = input_fieldnames or ["oare_id", "transliteration", "translation"]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(truncated_rows)

    print(f"Wrote {len(truncated_rows)} rows to {args.output_path}")
    print(f"Rows removed because of Same=No: {removed_by_same_no}")
    print(f"Rows removed because of excluded translation: {removed_by_translation}")
    print(
        "Rows removed because of ratio outlier: "
        f"{removed_by_ratio_outlier} "
        f"(lower_fence={ratio_stats['lower_fence']:.3f}, upper_fence={ratio_stats['upper_fence']:.3f}, "
        f"whisker_range={ratio_stats['lower_whisker']:.3f}-{ratio_stats['upper_whisker']:.3f})"
    )
    print(f"Rows removed total: {removed_rows}")
    if ratio_outlier_rows:
        sample = ratio_outlier_rows[0]
        sample_ratio = count_words(sample["translation"]) / max(count_words(sample["transliteration"]), 1)
        print(
            f"[RATIO OUTLIER] {sample['oare_id']}: "
            f"ratio={sample_ratio:.3f} "
            f"transliteration={sample['transliteration']} "
            f"translation={sample['translation']}"
        )


if __name__ == "__main__":
    main()
