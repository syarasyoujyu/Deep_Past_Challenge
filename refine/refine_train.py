from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from .mathematicals import replace_addition_with_sum, replace_fractions_in_text
except ImportError:
    from mathematicals import replace_addition_with_sum, replace_fractions_in_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / 'data' / 'train_truncated.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'train_refined.csv'
SHORT_TRANSLITERATION_WORD_LIMIT = 10
SHORT_TRANSLATION_WORD_LIMIT=10

TRANSLITERATION_REGEX_REPLACEMENTS = (
    (r"\([(\d)? ]*broken line[s]*\)", "<gap>"),
)
TRANSLITERATION_WORD_REPLACEMENTS = (
    ("-gold","pašallum gold"),
    ("-tax","šadduātum tax"),
    ("-textiles","kutānum textiles"),
    ("x","<gap>"),
    ("<gap>+","<gap>"),
    ("<gap>-","<gap>"),
    ("-",""),
    ("II","2"),
    ("III","3"),
    ("IV","4"),
    ("V","5"),
    ("VI","6"),
    ("VII","7"),
    ("VIII","8"),
    ("IX","9"),
    ("X","10"),
    ("XI","11"),
    ("XII","12"),
)
TRANSLITERATION_REPLACEMENTS = (
    #数値変換
    ('₀','0'),
    ('₁','1'), 
    ('₂','2'), 
    ('₃','3'), 
    ('₄','4'),
    ('₅','5'), 
    ('₆','6'), 
    ('₇','7'), 
    ('₈','8'), 
    ('₉','9'),
    #x自体がアッカド語にはないので、xすべてを許さない(Xは10月の10系統の可能性があるので消さない)
    ('x',''),
    ("ₓ",""),
    ("ʾ","'"),
    ("a2","á"),
    ("a3","à"),
    ("e2","é"),
    ("e3","è"),
    ("i2","í"),
    ("i3","ì"),
    ("u2","ú"),
    ("u3","ù"),
    ("sz","š"),
    ("SZ","Š"),
    ("s,","Ṣ"),
    ("S,","ṣ"),
    ("t,","ṭ"),
    ("T,","Ṭ"),
    #せめて知ってる単語で攻める
    ("Ḫ","H"),
    ("ḫ","h"),
    ("(d)",'{d}'),
    ("{tug2}","TÚG"),
    ("{tug₂}","TÚG"),
    ("(ki)",'{ki}'),
    ("(TÚG)",'TÚG'),
    (".3333300000000001", ".3333"),
    (".6666600000000003", ".6666"),
    ("KÙ.B.","KÙ.BABBAR"),
    #数値計算の後に行うので問題なし(ケース:lá+lá系・+x系)
    ("+","-"),
    #gap関連
    #("xx","<gap>"),
    ("-x","-<gap>"),
    ("+x","-<gap>"),
    ("<gap>+","<gap>-"),
    ("(large break)","<gap>"),
    ("<gap> <gap>","<gap>"),
)
TRANSLITERATION_REMOVALS = ("!","?","(",")","⌈","⌉","[","]",'"',"'","ʾ")
TRANSLATION_REGEX_REPLACEMENTS = (
    (r"<[\s]*big[\s_\-]*gap[\s]*>","<gap>"),
    (r"\([(\d)? ]*broken line[s]*\)", "<gap>"),
    (r"\[[[\s]*…]+[\s]*\]", "<gap>"),
    (r"<gap>[[\s]*<gap>]+", "<gap>"),
)
TRANSLATION_WORD_REPLACEMENTS = (
    ("x","<gap>"),
    ("<gap>+","<gap>"),
    ("<gap>-","<gap>"),
    ("-",""),
    ("II","2"),
    ("III","3"),
    ("IV","4"),
    ("V","5"),
    ("VI","6"),
    ("VII","7"),
    ("VIII","8"),
    ("IX","9"),
    ("X","10"),
    ("XI","11"),
    ("XII","12"),
)
TRANSLATION_REPLACEMENTS = (
    ('₀','0'),
    ('₁','1'), 
    ('₂','2'), 
    ('₃','3'), 
    ('₄','4'),
    ('₅','5'), 
    ('₆','6'), 
    ('₇','7'), 
    ('₈','8'), 
    ('₉','9'), 
    ('ₓ','x'),
    ("ʾ","'"),
    ('ofדsilver', 'of silver'),
    ("myself()","myself"),
    ("<lil>","-lil"),
    ("<of firewood>","of firewood"),
    (".3333300000000001", ".3333"),
    (".6666600000000003", ".6666"),
    #数値計算の後に行うので問題なし(ケース:lá+lá系・+x系)
    ("+","-"),
    #gap関連
    #("xx","<gap>"),
    ("[x]","<gap>"),
    ("(large break)","<gap>"),
    ("ê","e"),
    ("ì","ī"),
    ("û","u"),
    ("Ā","A"),
    ("Ē","E"),
    ("Ī","I"),
    ("ạ","a"),
    #文中の特殊ケースに遭遇した場合の対処
    ("andĀl-ṭāb","and Āl-ṭāb"),
)
#"!","?"自体は意味のあるものだが、それ以上に推論では余分（考えずに出力する方が良い）
#(や)も意味のあるものだが、翻訳の中で括弧は不要だと判断（括弧は不要だが、中身は文の内容をわかる範囲で補ったという意図があり、使える）
TRANSLATION_REMOVALS = ("!","?","(",")","fem.","plur.","pl.","sing.","plural",'"',"'","—","–","⌈","⌉","[","]","ʾ",
#文章の整合性ように消さん方が良かったかもだが、いったんとっとく
":",";",".",",","/","{","}","")
#<・>はstrip_angle_brackets_except_gap_tokensによって特殊なケースを考慮しながら削除


@dataclass(frozen=True)
class DetectionRule:
    label: str
    field: str
    pattern: str
    use_regex: bool = False

    def matches(self, row: dict[str, str]) -> bool:
        value = row[self.field]
        if self.use_regex:
            return re.search(self.pattern, value) is not None
        return self.pattern in value


@dataclass(frozen=True)
class ReplacementRule:
    field: str
    before: str
    after: str


@dataclass(frozen=True)
class WordReplacementRule:
    field: str
    before: str
    after: str


@dataclass(frozen=True)
class RegexReplacementRule:
    field: str
    pattern: str
    after: str
    compiled: re.Pattern[str]


def build_replacement_rules(
    field: str, replacements: tuple[tuple[str, ...], ...]
) -> tuple[ReplacementRule, ...]:
    rules: list[ReplacementRule] = []
    for replacement in replacements:
        if len(replacement) != 2:
            continue
        before, after = replacement
        rules.append(ReplacementRule(field=field, before=before, after=after))
    return tuple(rules)


def build_regex_replacement_rules(
    field: str, replacements: tuple[tuple[str, ...], ...]
) -> tuple[RegexReplacementRule, ...]:
    rules: list[RegexReplacementRule] = []
    for replacement in replacements:
        if len(replacement) != 2:
            continue
        pattern, after = replacement
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex replacement for {field}: {pattern!r}"
            ) from exc
        rules.append(
            RegexReplacementRule(field=field, pattern=pattern, after=after, compiled=compiled)
        )
    return tuple(rules)


def build_word_replacement_rules(
    field: str, replacements: tuple[tuple[str, ...], ...]
) -> tuple[WordReplacementRule, ...]:
    rules: list[WordReplacementRule] = []
    for replacement in replacements:
        if len(replacement) != 2:
            continue
        before, after = replacement
        rules.append(WordReplacementRule(field=field, before=before, after=after))
    return tuple(rules)


REPLACEMENT_RULES = (
    *build_replacement_rules('transliteration', TRANSLITERATION_REPLACEMENTS),
    *build_replacement_rules('translation', TRANSLATION_REPLACEMENTS),
)
REGEX_REPLACEMENT_RULES = (
    *build_regex_replacement_rules('transliteration', TRANSLITERATION_REGEX_REPLACEMENTS),
    *build_regex_replacement_rules('translation', TRANSLATION_REGEX_REPLACEMENTS),
)
WORD_REPLACEMENT_RULES = (
    *build_word_replacement_rules('transliteration', TRANSLITERATION_WORD_REPLACEMENTS),
    *build_word_replacement_rules('translation', TRANSLATION_WORD_REPLACEMENTS),
)


DETECTION_RULES = (
    DetectionRule(label='<gap>', field='translation', pattern='<gap>'),
)
PUNCTUATION_TOKENS = {',', ':', '.',"?","!"}
GAP_SENTINEL = "\uFFF0"
BIG_GAP_SENTINEL = "\uFFF1"


def clean_transliteration(text: str) -> str:
    cleaned=normalize_spaces(text)
    cleaned = replace_fractions_in_text(cleaned)
    cleaned = replace_addition_with_sum(cleaned)
    cleaned = apply_regex_replacements(cleaned, 'transliteration')
    cleaned = apply_word_replacements(cleaned, 'transliteration')
    cleaned = apply_replacements(cleaned, 'transliteration')
    cleaned = strip_angle_brackets_except_gap_tokens(cleaned)
    for token in TRANSLITERATION_REMOVALS:
        cleaned = cleaned.replace(token, '')
    cleaned = insert_hyphen_around_gap(cleaned)
    return normalize_spaces(cleaned)


def clean_translation(text: str) -> str:
    cleaned = normalize_spaces(text)
    cleaned = replace_fractions_in_text(cleaned)
    cleaned = replace_addition_with_sum(cleaned)
    cleaned = apply_regex_replacements(cleaned, 'translation')
    cleaned = apply_word_replacements(cleaned, 'translation')
    cleaned = apply_replacements(cleaned, 'translation')
    cleaned = strip_angle_brackets_except_gap_tokens(cleaned)
    for token in TRANSLATION_REMOVALS:
        cleaned = cleaned.replace(token, '')
    cleaned = insert_hyphen_around_gap(cleaned)
    return normalize_spaces(cleaned)


def normalize_spaces(text: str) -> str:
    return ' '.join(text.split())


def strip_angle_brackets_except_gap_tokens(text: str) -> str:
    cleaned = text.replace("<big_gap>", BIG_GAP_SENTINEL)
    cleaned = cleaned.replace("<gap>", GAP_SENTINEL)
    cleaned = cleaned.replace("<", "")
    cleaned = cleaned.replace(">", "")
    cleaned = cleaned.replace(BIG_GAP_SENTINEL, "<big_gap>")
    cleaned = cleaned.replace(GAP_SENTINEL, "<gap>")
    return cleaned


def insert_hyphen_around_gap(text: str) -> str:
    text = re.sub(r'(?<=[^\s-])<gap>', r'-<gap>', text)
    text = re.sub(r'<gap>(?=[^\s-])', r'<gap>-', text)
    return text


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in text.split():
        current = chunk

        trailing_punctuation: list[str] = []
        while current and current[-1] in PUNCTUATION_TOKENS:
            trailing_punctuation.append(current[-1])
            current = current[:-1]

        leading_punctuation: list[str] = []
        while current and current[0] in PUNCTUATION_TOKENS:
            leading_punctuation.append(current[0])
            current = current[1:]

        tokens.extend(leading_punctuation)
        if current:
            tokens.append(current)
        tokens.extend(reversed(trailing_punctuation))

    return tokens


def detokenize_text(tokens: list[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token in PUNCTUATION_TOKENS:
            if parts:
                parts[-1] = f'{parts[-1]}{token}'
            else:
                parts.append(token)
            continue
        parts.append(token)
    return ' '.join(parts)


def apply_replacements(text: str, field: str) -> str:
    cleaned = text
    for rule in REPLACEMENT_RULES:
        if rule.field == field:
            cleaned = cleaned.replace(rule.before, rule.after)
    return cleaned


def apply_regex_replacements(text: str, field: str) -> str:
    cleaned = text
    for rule in REGEX_REPLACEMENT_RULES:
        if rule.field == field:
            cleaned = rule.compiled.sub(rule.after, cleaned)
    return cleaned


def apply_word_replacements(text: str, field: str) -> str:
    word_map = {
        rule.before: rule.after
        for rule in WORD_REPLACEMENT_RULES
        if rule.field == field
    }
    if not word_map:
        return text

    tokens = tokenize_text(text)
    replaced_tokens = [word_map.get(token, token) for token in tokens]
    return detokenize_text(replaced_tokens)


def count_replacement_hits(
    rows: list[dict[str, str]], replacement_rules: tuple[ReplacementRule, ...]
) -> dict[tuple[str, str], int]:
    replacement_hits = {(rule.field, rule.before): 0 for rule in replacement_rules}

    for row in rows:
        for rule in replacement_rules:
            if rule.before in row[rule.field]:
                replacement_hits[(rule.field, rule.before)] += 1

    return replacement_hits


def count_word_replacement_hits(
    rows: list[dict[str, str]], replacement_rules: tuple[WordReplacementRule, ...]
) -> dict[tuple[str, str], int]:
    replacement_hits = {(rule.field, rule.before): 0 for rule in replacement_rules}

    for row in rows:
        for rule in replacement_rules:
            tokens = tokenize_text(row[rule.field])
            if rule.before in tokens:
                replacement_hits[(rule.field, rule.before)] += 1

    return replacement_hits


def count_regex_replacement_hits(
    rows: list[dict[str, str]], replacement_rules: tuple[RegexReplacementRule, ...]
) -> dict[tuple[str, str], int]:
    replacement_hits = {(rule.field, rule.pattern): 0 for rule in replacement_rules}

    for row in rows:
        for rule in replacement_rules:
            if rule.compiled.search(row[rule.field]):
                replacement_hits[(rule.field, rule.pattern)] += 1

    return replacement_hits


def print_replacement_summary(
    replacement_hits: dict[tuple[str, str], int], replacement_rules: tuple[ReplacementRule, ...]
) -> None:
    for rule in replacement_rules:
        print(
            f"Rows matching {rule.field} replacement "
            f"{rule.before!r} -> {rule.after!r}: {replacement_hits[(rule.field, rule.before)]}"
        )


def print_word_replacement_summary(
    replacement_hits: dict[tuple[str, str], int], replacement_rules: tuple[WordReplacementRule, ...]
) -> None:
    for rule in replacement_rules:
        print(
            f"Rows matching {rule.field} word replacement "
            f"{rule.before!r} -> {rule.after!r}: {replacement_hits[(rule.field, rule.before)]}"
        )


def print_regex_replacement_summary(
    replacement_hits: dict[tuple[str, str], int], replacement_rules: tuple[RegexReplacementRule, ...]
) -> None:
    for rule in replacement_rules:
        print(
            f"Rows matching {rule.field} regex replacement "
            f"{rule.pattern!r} -> {rule.after!r}: {replacement_hits[(rule.field, rule.pattern)]}"
        )


def collect_detection_rows(
    rows: list[dict[str, str]], detection_rules: tuple[DetectionRule, ...]
) -> dict[str, list[dict[str, str]]]:
    detected_rows = {rule.label: [] for rule in detection_rules}

    for row in rows:
        for rule in detection_rules:
            if rule.matches(row):
                detected_rows[rule.label].append(row)

    return detected_rows


def print_detection_summary(
    detected_rows: dict[str, list[dict[str, str]]], detection_rules: tuple[DetectionRule, ...]
) -> None:
    for rule in detection_rules:
        rows = detected_rows[rule.label]
        print(f'Rows containing {rule.label}: {len(rows)}')
        if rows:
            sample = rows[0]
            print(f"[FOUND {rule.label}] {sample['oare_id']}: {sample[rule.field]}")


def collect_short_rows(rows: list[dict[str, str]], field: str, word_limit: int) -> list[dict[str, str]]:
    return [row for row in rows if len(row[field].split()) <= word_limit]


def print_short_rows(rows: list[dict[str, str]], field: str, word_limit: int) -> None:
    print(f'Rows with {field} <= {word_limit} words: {len(rows)}')
    for row in rows:
        print(f"[SHORT {field}] {row['oare_id']}: {row[field]}")


def main() -> None:
    source_rows: list[dict[str, str]] = []
    refined_rows: list[dict[str, str]] = []
    changed_rows = 0

    with INPUT_PATH.open('r', encoding='utf-8', newline='') as csv_file:
        for row in csv.DictReader(csv_file):
            source_rows.append(row)
            cleaned_transliteration = clean_transliteration(row['transliteration'])
            cleaned_translation = clean_translation(row['translation'])
            if (
                cleaned_transliteration != row['transliteration']
                or cleaned_translation != row['translation']
            ):
                changed_rows += 1

            refined_row = {
                'oare_id': row['oare_id'],
                'transliteration': cleaned_transliteration,
                "translation":cleaned_translation,
            }
            refined_rows.append(refined_row)

    with OUTPUT_PATH.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['oare_id', 'transliteration', 'translation'])
        writer.writeheader()
        writer.writerows(refined_rows)

    replacement_hits = count_replacement_hits(source_rows, REPLACEMENT_RULES)
    regex_replacement_hits = count_regex_replacement_hits(source_rows, REGEX_REPLACEMENT_RULES)
    word_replacement_hits = count_word_replacement_hits(source_rows, WORD_REPLACEMENT_RULES)
    detected_rows = collect_detection_rows(refined_rows, DETECTION_RULES)
    short_transliteration_rows = collect_short_rows(
        refined_rows, 'transliteration', SHORT_TRANSLITERATION_WORD_LIMIT
    )
    short_translation_rows = collect_short_rows(
        refined_rows, 'translation', SHORT_TRANSLATION_WORD_LIMIT
    )

    print(f'Wrote {len(refined_rows)} rows to {OUTPUT_PATH}')
    print(f'Rows changed by punctuation cleanup: {changed_rows}')
    print_replacement_summary(replacement_hits, REPLACEMENT_RULES)
    print_regex_replacement_summary(regex_replacement_hits, REGEX_REPLACEMENT_RULES)
    print_word_replacement_summary(word_replacement_hits, WORD_REPLACEMENT_RULES)
    print_detection_summary(detected_rows, DETECTION_RULES)
    print_short_rows(short_transliteration_rows, 'transliteration', SHORT_TRANSLITERATION_WORD_LIMIT)
    print_short_rows(short_translation_rows, 'translation', SHORT_TRANSLATION_WORD_LIMIT)


if __name__ == '__main__':
    main()
