#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import logging
import math
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import sacrebleu
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

DEFAULT_TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
DEFAULT_OUTPUT_DIR = "/kaggle/working/"
DEFAULT_MODEL_A_PATH = "/kaggle/input/models/yokoinaba/akkad-dict-v3/transformers/default/1/checkpoint-164"
DEFAULT_MODEL_B_PATH = "/kaggle/input/models/yokoinaba/akkad-dict-v3/transformers/default/1/checkpoint-164"
DEFAULT_LEXICON_PATH = "/kaggle/input/datasets/yokoinaba/word-dict-akkad/OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_GLOSS_PATH = "/kaggle/input/datasets/yokoinaba/akkad-dict/train_openai_gloss_compact.jsonl"
DEFAULT_ONOMASTICON_PATH = "/kaggle/input/datasets/yokoinaba/onomatsticon/onomasticon_refined.csv"

SUBSCRIPT_TRANSLATION_TABLE = str.maketrans(
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
        "ₓ": "x",
    }
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability(0)
        return bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            and capability[0] >= 8
        )
    except Exception:
        return False


def _bf16_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and _cuda_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "ensemble_dict.log"),
        ],
    )
    return logging.getLogger("ensemble_dict")


def normalize_transliteration(text: str, cfg) -> str:
    text = str(text).replace("\u00a0", " ").strip()
    if cfg.normalize_h:
        text = text.replace("Ḫ", "H").replace("ḫ", "h")
    if cfg.normalize_subscripts:
        text = text.translate(SUBSCRIPT_TRANSLATION_TABLE)
    if cfg.normalize_breaks:
        text = re.sub(r"\[\s*[xX]\s*\]", " <gap> ", text)
        text = re.sub(r"\[\s*(?:…|\.\.\.)+\s*\]", " <big_gap> ", text)
        text = text.replace("…", " <big_gap> ")
        text = re.sub(r"\[([^\[\]]+)\]", r" \1 ", text)
    if cfg.remove_editorial_marks:
        text = re.sub(r"[!?/]", " ", text)
        text = text.replace("˹", "").replace("˺", "")
    if cfg.strip_word_dividers:
        text = text.replace(":", " ").replace(".", " ")
    return " ".join(text.split())


def normalize_translation(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").strip().split())


@dataclass(frozen=True)
class DictionaryEntry:
    source_form: str
    target_hint: str
    token_count: int
    source_kind: str


@dataclass
class EnsembleDictConfig:
    test_data_path: str = DEFAULT_TEST_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    model_a_path: str = DEFAULT_MODEL_A_PATH
    model_b_path: str = DEFAULT_MODEL_B_PATH
    tokenizer_path: str | None = None
    lexicon_path: str = DEFAULT_LEXICON_PATH
    gloss_path: str = DEFAULT_GLOSS_PATH
    include_onomasticon: bool = False
    onomasticon_path: str = DEFAULT_ONOMASTICON_PATH
    source_prefix: str = "translate Akkadian to English with dictionary: "
    hint_placement: str = "prepend"
    max_dictionary_hints: int = 8
    max_entry_token_length: int = 4
    max_hint_words: int = 6
    max_gloss_variants: int = 4
    normalize_source: bool = True
    normalize_target: bool = True
    normalize_h: bool = True
    normalize_subscripts: bool = True
    normalize_breaks: bool = True
    remove_editorial_marks: bool = True
    strip_word_dividers: bool = False
    gpu_a: int = 0
    gpu_b: int = 1
    seed: int = 42

    max_source_length: int = 768
    max_target_length: int = 384
    use_dynamic_max_new_tokens: bool = True
    dynamic_max_new_tokens_ratio: float = 2.5
    dynamic_max_new_tokens_cap: int = 512
    batch_size: int = 8
    num_workers: int = 2
    num_buckets: int = 6

    num_beam_cands: int = 5
    num_beams: int = 8
    length_penalty: float = 1.3
    early_stopping: bool = True
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 0

    use_diverse_beam: bool = True
    num_diverse_cands: int = 5
    num_diverse_beams: int = 8
    num_beam_groups: int = 4
    diversity_penalty: float = 0.9

    use_sampling: bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.65, 0.85, 1.05])
    num_sample_per_temp: int = 2
    mbr_top_p: float = 0.92

    mbr_pool_cap: int = 32
    mbr_w_chrf: float = 0.55
    mbr_w_bleu: float = 0.25
    mbr_w_jaccard: float = 0.20
    mbr_w_length: float = 0.04
    mbr_w_support: float = 0.00

    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_adaptive_beams: bool = True
    checkpoint_freq: int = 200

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    @property
    def max_input_length(self) -> int:
        return self.max_source_length

    @property
    def max_new_tokens(self) -> int:
        return self.max_target_length

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_a_path
        self.lexicon_path = Path(self.lexicon_path)
        self.gloss_path = Path(self.gloss_path)
        self.onomasticon_path = Path(self.onomasticon_path)
        self.cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.dual_gpu = self.cuda_device_count >= 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_a_device = (
            torch.device(f"cuda:{self.gpu_a}") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_b_device = (
            torch.device(f"cuda:{self.gpu_b}") if self.dual_gpu else self.model_a_device
        )
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False
        self.use_bf16_amp = bool(
            self.use_mixed_precision and self.device.type == "cuda" and _cuda_bf16_supported()
        )
        assert self.num_beams >= self.num_beam_cands
        if self.use_diverse_beam:
            assert self.num_diverse_beams % self.num_beam_groups == 0
            assert self.num_diverse_beams >= self.num_diverse_cands


def load_input_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    if "transliteration" not in frame.columns:
        raise ValueError("input CSV must contain a transliteration column")
    if "oare_id" in frame.columns:
        frame["oare_id"] = frame["oare_id"].fillna("").astype(str).str.strip()
    elif "id" in frame.columns:
        frame["oare_id"] = frame["id"].fillna("").astype(str).str.strip()
    else:
        frame["oare_id"] = frame.index.map(str)
    frame["transliteration"] = frame["transliteration"].fillna("").astype(str).str.strip()
    if "translation" in frame.columns:
        frame["translation"] = frame["translation"].fillna("").astype(str).str.strip()
    else:
        frame["translation"] = ""
    frame = frame[frame["transliteration"] != ""].reset_index(drop=True)
    return frame


def write_submission(frame: pd.DataFrame, predictions: list[str], submission_path: Path) -> None:
    submission = pd.DataFrame({"id": frame["oare_id"].astype(str), "translation": predictions})
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)


def clean_hint_text(text: str, max_hint_words: int) -> str:
    normalized = normalize_translation(text).strip().strip('"')
    if not normalized:
        return ""
    normalized = normalized.split(";")[0].split("(")[0].strip()
    words = normalized.split()
    if max_hint_words > 0:
        words = words[:max_hint_words]
    return " ".join(words).strip()


def normalize_dict_form(text: str, cfg: EnsembleDictConfig) -> str:
    return normalize_transliteration(text, cfg).strip()


def load_named_entity_entries(cfg: EnsembleDictConfig) -> dict[tuple[str, ...], list[DictionaryEntry]]:
    index: dict[tuple[str, ...], list[DictionaryEntry]] = {}
    with cfg.lexicon_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entity_type = str(row.get("type", "")).strip()
            if entity_type not in {"PN", "GN"}:
                continue
            source_form = normalize_dict_form(row.get("form", ""), cfg)
            if not source_form:
                continue
            source_tokens = tuple(token for token in source_form.split() if token)
            if not source_tokens or len(source_tokens) > cfg.max_entry_token_length:
                continue
            hint_text = clean_hint_text(row.get("lexeme", ""), cfg.max_hint_words)
            if not hint_text:
                hint_text = clean_hint_text(row.get("norm", ""), cfg.max_hint_words)
            if not hint_text:
                continue
            entry = DictionaryEntry(
                source_form=" ".join(source_tokens),
                target_hint=hint_text,
                token_count=len(source_tokens),
                source_kind=entity_type,
            )
            index.setdefault(source_tokens, [])
            if entry not in index[source_tokens]:
                index[source_tokens].append(entry)

    if cfg.include_onomasticon and cfg.onomasticon_path.exists():
        with cfg.onomasticon_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = clean_hint_text(row.get("Name", ""), cfg.max_hint_words)
                if not name:
                    continue
                spellings = str(row.get("Spellings_semicolon_separated", "")).split(";")
                for spelling in spellings:
                    source_form = normalize_dict_form(spelling, cfg)
                    if not source_form:
                        continue
                    source_tokens = tuple(token for token in source_form.split() if token)
                    if not source_tokens or len(source_tokens) > cfg.max_entry_token_length:
                        continue
                    entry = DictionaryEntry(
                        source_form=" ".join(source_tokens),
                        target_hint=name,
                        token_count=len(source_tokens),
                        source_kind="ONOMASTICON",
                    )
                    index.setdefault(source_tokens, [])
                    if entry not in index[source_tokens]:
                        index[source_tokens].append(entry)
    return index


def load_gloss_entries(cfg: EnsembleDictConfig) -> dict[tuple[str, ...], list[DictionaryEntry]]:
    gloss_candidates: dict[str, dict[str, int]] = {}
    with cfg.gloss_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            for token_gloss in payload.get("token_glosses", []):
                confidence = str(token_gloss.get("confidence", "")).strip().lower()
                if confidence == "low":
                    continue
                normalized_token = normalize_dict_form(token_gloss.get("normalized_token", ""), cfg)
                if not normalized_token:
                    continue
                gloss = clean_hint_text(token_gloss.get("gloss", ""), cfg.max_hint_words)
                if not gloss:
                    continue
                gloss_candidates.setdefault(normalized_token, {})
                gloss_candidates[normalized_token][gloss] = (
                    gloss_candidates[normalized_token].get(gloss, 0) + 1
                )

    index: dict[tuple[str, ...], list[DictionaryEntry]] = {}
    for normalized_token, candidate_counts in gloss_candidates.items():
        glosses = [
            gloss
            for gloss, _ in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        if not glosses or len(glosses) >= cfg.max_gloss_variants + 1:
            continue
        source_tokens = tuple(token for token in normalized_token.split() if token)
        if not source_tokens or len(source_tokens) > cfg.max_entry_token_length:
            continue
        entry = DictionaryEntry(
            source_form=" ".join(source_tokens),
            target_hint=" / ".join(glosses),
            token_count=len(source_tokens),
            source_kind="GLOSS",
        )
        index.setdefault(source_tokens, []).append(entry)
    return index


def load_dictionary_entries(cfg: EnsembleDictConfig) -> dict[tuple[str, ...], list[DictionaryEntry]]:
    combined_index = load_named_entity_entries(cfg)
    gloss_index = load_gloss_entries(cfg)
    for key, entries in gloss_index.items():
        if key in combined_index:
            continue
        combined_index[key] = entries
    return combined_index


def find_dictionary_hints(
    normalized_transliteration: str,
    dictionary_index: dict[tuple[str, ...], list[DictionaryEntry]],
    cfg: EnsembleDictConfig,
) -> list[DictionaryEntry]:
    tokens = [token for token in normalized_transliteration.split() if token]
    if not tokens:
        return []
    matches: list[DictionaryEntry] = []
    occupied_positions: set[int] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for start in range(len(tokens)):
        if start in occupied_positions:
            continue
        for length in range(cfg.max_entry_token_length, 0, -1):
            end = start + length
            if end > len(tokens):
                continue
            if any(position in occupied_positions for position in range(start, end)):
                continue
            key = tuple(tokens[start:end])
            candidates = dictionary_index.get(key)
            if not candidates:
                continue
            chosen = candidates[0]
            pair = (chosen.source_form, chosen.target_hint)
            if pair not in seen_pairs:
                matches.append(chosen)
                seen_pairs.add(pair)
                occupied_positions.update(range(start, end))
            break
        if len(matches) >= cfg.max_dictionary_hints:
            break
    return matches[: cfg.max_dictionary_hints]


def build_augmented_source(
    normalized_transliteration: str,
    dictionary_hints: list[DictionaryEntry],
    cfg: EnsembleDictConfig,
) -> str:
    if not dictionary_hints:
        return f"{cfg.source_prefix}{normalized_transliteration}".strip()
    hint_text = " ; \n".join(f"{entry.source_form} = {entry.target_hint}" for entry in dictionary_hints)
    if cfg.hint_placement == "prepend":
        return f"{cfg.source_prefix} \n\n dictionary: {hint_text} \n\n text: {normalized_transliteration}".strip()
    return f"{cfg.source_prefix}{normalized_transliteration} \n\n dictionary: {hint_text}".strip()


def fit_dictionary_hints_to_source_budget(
    normalized_transliteration: str,
    dictionary_hints: list[DictionaryEntry],
    cfg: EnsembleDictConfig,
    tokenizer,
) -> tuple[list[DictionaryEntry], str]:
    kept_hints = list(dictionary_hints)
    while True:
        augmented_source = build_augmented_source(normalized_transliteration, kept_hints, cfg)
        token_count = len(
            tokenizer(
                augmented_source,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if token_count <= cfg.max_source_length or not kept_hints:
            return kept_hints, augmented_source
        kept_hints = kept_hints[:-1]


def serialize_dictionary_hints(dictionary_hints: list[DictionaryEntry]) -> str:
    return " ;\n ".join(f"{entry.source_form} = {entry.target_hint}" for entry in dictionary_hints)


def prepare_dictionary_frame_for_inference(
    frame: pd.DataFrame,
    dictionary_index,
    tokenizer,
    cfg: EnsembleDictConfig,
) -> pd.DataFrame:
    frame = frame.copy()
    if cfg.normalize_source:
        frame["transliteration"] = frame["transliteration"].map(
            lambda text: normalize_transliteration(text, cfg)
        )
    if cfg.normalize_target and "translation" in frame.columns:
        frame["translation"] = frame["translation"].map(normalize_translation)
    dictionary_hints_column = []
    augmented_source_column = []
    for transliteration in frame["transliteration"].tolist():
        hints = find_dictionary_hints(transliteration, dictionary_index, cfg)
        fitted_hints, augmented_source = fit_dictionary_hints_to_source_budget(
            transliteration,
            hints,
            cfg,
            tokenizer,
        )
        dictionary_hints_column.append(fitted_hints)
        augmented_source_column.append(augmented_source)
    frame["dictionary_hints"] = dictionary_hints_column
    frame["dictionary_hint_text"] = frame["dictionary_hints"].map(serialize_dictionary_hints)
    frame["dictionary_hint_count"] = frame["dictionary_hints"].map(len)
    frame["augmented_source"] = augmented_source_column
    return frame


_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]ʾ;')
_COMMODITY_RE = re.compile(r"(?<=\s)-(gold|tax|textiles)\b")
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}
_SHEKEL_REPLS = [
    (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
    (re.compile(r"5\s*/\s*12\s+shekels?", re.I), "⅓ shekel 15 grains"),
    (re.compile(r"7\s*/\s*12\s+shekels?", re.I), "½ shekel 15 grains"),
    (re.compile(r"1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?", re.I), "15 grains"),
]
_SLASH_ALT_RE = re.compile(r"(?<![0-9/])\s+/\s+(?![0-9])\S+")
_STRAY_MARKS_RE = re.compile(r"<<[^>]*>>|<(?!gap\b)[^>]*>")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}")
_EXTRA_STRAY_RE = re.compile(r"(?<!\w)(?:\.\.+|xx+)(?!\w)")
_HACEK_TRANS = str.maketrans({"ḫ": "h", "Ḫ": "H"})

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})
_ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_WS_RE = re.compile(r"\s+")
_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)
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
_PN_RE = re.compile(r"\bPN\b")
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


def _ascii_to_diacritics(text: str) -> str:
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
    text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
    text = _V2.sub(lambda match: match.group(1).translate(_ACUTE), text)
    text = _V3.sub(lambda match: match.group(1).translate(_GRAVE), text)
    return text


def _normalize_gaps_vec(series: pd.Series) -> pd.Series:
    return series.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)


def _frac_repl(match: re.Match) -> str:
    return _EXACT_FRAC_MAP[match.group(0)]


def _canon_decimal(value: float) -> str:
    integer_part = int(math.floor(value + 1e-12))
    frac = value - integer_part
    best = min(_ALLOWED_FRACS, key=lambda item: abs(frac - item[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        decimal = best[1]
        if integer_part == 0:
            return decimal
        return f"{integer_part}{decimal[1:]}" if decimal.startswith("0.") else f"{integer_part}+{decimal}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _commodity_repl(match: re.Match) -> str:
    return _COMMODITY_REPL[match.group(1)]


def _month_repl(match: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(match.group(1).upper(), match.group(1))}"


class VectorizedPostprocessor:
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        series = pd.Series(translations).fillna("").astype(str)
        series = _normalize_gaps_vec(series)
        series = series.str.replace(_PN_RE, "<gap>", regex=True)
        series = series.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)
        for pattern, repl in _SHEKEL_REPLS:
            series = series.str.replace(pattern, repl, regex=True)
        series = series.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        series = series.str.replace(_FLOAT_RE, lambda match: _canon_decimal(float(match.group(1))), regex=True)
        series = series.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        series = series.str.replace(_BARE_GRAM_RE, " ", regex=True)
        series = series.str.replace(_UNCERTAIN_RE, "", regex=True)
        series = series.str.replace(_STRAY_MARKS_RE, "", regex=True)
        series = series.str.replace(_EXTRA_STRAY_RE, "", regex=True)
        series = series.str.replace(_SLASH_ALT_RE, "", regex=True)
        series = series.str.replace(_CURLY_DQ_RE, '"', regex=True)
        series = series.str.replace(_CURLY_SQ_RE, "'", regex=True)
        series = series.str.replace(_MONTH_RE, _month_repl, regex=True)
        series = series.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        series = series.str.replace("<gap>", "\x00GAP\x00", regex=False)
        series = series.str.translate(_FORBIDDEN_TRANS)
        series = series.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        series = series.str.translate(_HACEK_TRANS)
        series = series.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for ngram in range(4, 1, -1):
            pattern = r"\b((?:\w+\s+){" + str(ngram - 1) + r"}\w+)(?:\s+\1\b)+"
            series = series.str.replace(pattern, r"\1", regex=True)
        series = series.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        series = series.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        series = series.str.replace(_WS_RE, " ", regex=True).str.strip()
        return series.tolist()


class DictionaryAkkadianDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, logger: logging.Logger):
        self.sample_ids = frame["oare_id"].tolist()
        self.input_texts = frame["augmented_source"].fillna("").astype(str).tolist()
        self.dictionary_hint_counts = frame["dictionary_hint_count"].fillna(0).astype(int).tolist()
        logger.info(
            "Dictionary prompt coverage: "
            f"{sum(count > 0 for count in self.dictionary_hint_counts)}/{len(self.dictionary_hint_counts)} rows, "
            f"mean hints={sum(self.dictionary_hint_counts) / max(len(self.dictionary_hint_counts), 1):.2f}"
        )
        logger.info(f"Dataset: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets, logger, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [len(text.split()) for _, text in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda idx: lengths[idx])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i * bsize : None if i == num_buckets - 1 else (i + 1) * bsize]
            for i in range(num_buckets)
        ]
        for idx, bucket in enumerate(self.buckets):
            if bucket:
                bucket_lengths = [lengths[item] for item in bucket]
                logger.info(f"  Bucket {idx}: {len(bucket)} samples, len [{min(bucket_lengths)}, {max(bucket_lengths)}]")

    def __iter__(self):
        for bucket in self.buckets:
            items = list(bucket)
            if self.shuffle:
                random.shuffle(items)
            for start in range(0, len(items), self.batch_size):
                yield items[start : start + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.buckets)


class ModelWrapper:
    def __init__(self, model_path: str, cfg: EnsembleDictConfig, logger: logging.Logger, label: str, device: torch.device):
        self.cfg = cfg
        self.logger = logger
        self.label = label
        self.device = device
        logger.info(f"[{label}] Loading from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._load_model_on_device(model_path).eval()
        if self.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        parameter_count = sum(param.numel() for param in self.model.parameters())
        logger.info(f"[{label}] {parameter_count:,} parameters")
        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated(self.device) / 1e9
            logger.info(f"[{label}] GPU mem used on {self.device}: {used:.2f} GB")
        if cfg.use_better_transformer and self.device.type == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer

                self.model = BetterTransformer.transform(self.model)
                logger.info(f"[{label}] BetterTransformer applied")
            except Exception as error:
                logger.warning(f"[{label}] BetterTransformer skipped: {error}")

    def _load_model_on_device(self, model_path: str):
        if self.device.type == "cuda":
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    device_map={"": str(self.device)},
                    low_cpu_mem_usage=True,
                )
            except Exception as error:
                self.logger.warning(f"[{self.label}] device_map load skipped ({error}), falling back to standard load")
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=False).to(self.device)
        except NotImplementedError as error:
            if "meta tensor" not in str(error):
                raise
            self.logger.warning(
                f"[{self.label}] standard load hit meta tensors ({error}), falling back to config + sharded checkpoint load"
            )
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_config(config)
        if any(parameter.is_meta for parameter in model.parameters()):
            model = model.to_empty(device=self.device)
        else:
            model = model.to(self.device)
        model_dir = Path(model_path)
        safetensors_path = model_dir / "model.safetensors"
        pytorch_bin_path = model_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file as safe_load_file

            state_dict = safe_load_file(str(safetensors_path), device="cpu")
            model.load_state_dict(state_dict, strict=False)
        elif pytorch_bin_path.exists():
            state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            load_sharded_checkpoint(model, model_path, strict=False, prefer_safe=True)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        return model

    def _sampling_seed(self, temp_index: int) -> int:
        label_offset = 0 if self.label == "Model-A" else 1_000_000_000
        return int(self.cfg.seed) + label_offset + (int(temp_index) * 1_000)

    def collate(self, batch_samples):
        ids = [item[0] for item in batch_samples]
        texts = [item[1] for item in batch_samples]
        enc = self.tokenizer(
            texts,
            max_length=self.cfg.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return ids, enc

    @staticmethod
    def _token_jaccard(a: str, b: str) -> float:
        a_tokens = set(str(a).split())
        b_tokens = set(str(b).split())
        if not a_tokens and not b_tokens:
            return 1.0
        union = a_tokens | b_tokens
        if not union:
            return 0.0
        return len(a_tokens & b_tokens) / len(union)

    def _select_diverse_subset(self, candidates: List[str], k: int) -> List[str]:
        unique_candidates = []
        seen = set()
        for text in candidates:
            normalized = str(text).strip()
            if not normalized or normalized in seen:
                continue
            unique_candidates.append(normalized)
            seen.add(normalized)
        if len(unique_candidates) <= k:
            return unique_candidates
        selected = [unique_candidates[0]]
        remaining = unique_candidates[1:]
        while remaining and len(selected) < k:
            best_idx = 0
            best_score = float("-inf")
            for idx, candidate in enumerate(remaining):
                max_similarity = max(self._token_jaccard(candidate, chosen) for chosen in selected)
                score = -max_similarity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def generate_candidates(self, input_ids, attention_mask, beam_size: int) -> List[List[str]]:
        cfg = self.cfg
        batch_size = input_ids.shape[0]
        ctx = _bf16_ctx(self.device, cfg.use_bf16_amp)
        dynamic_max_new_tokens = cfg.max_new_tokens
        if cfg.use_dynamic_max_new_tokens:
            dynamic_max_new_tokens = min(
                cfg.dynamic_max_new_tokens_cap,
                max(1, int(input_ids.shape[1] * cfg.dynamic_max_new_tokens_ratio)),
            )

        beam_count = cfg.num_beam_cands
        diverse_count = cfg.num_diverse_cands if cfg.use_diverse_beam else 0
        sample_count = cfg.num_sample_per_temp

        with ctx:
            num_beams = max(beam_size, beam_count)
            common_generate_kwargs = {
                "max_new_tokens": dynamic_max_new_tokens,
                "repetition_penalty": cfg.repetition_penalty,
                "use_cache": True,
            }
            if cfg.no_repeat_ngram_size > 0:
                common_generate_kwargs["no_repeat_ngram_size"] = cfg.no_repeat_ngram_size
            beam_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=beam_count,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                **common_generate_kwargs,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            diverse_texts = []
            actual_diverse_count = 0
            if cfg.use_diverse_beam:
                try:
                    diverse_pool = max(cfg.num_diverse_beams, cfg.num_diverse_cands * 3)
                    diverse_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=diverse_pool,
                        num_return_sequences=diverse_pool,
                        length_penalty=cfg.length_penalty,
                        early_stopping=cfg.early_stopping,
                        **common_generate_kwargs,
                    )
                    diverse_pool_texts = self.tokenizer.batch_decode(diverse_out, skip_special_tokens=True)
                    for batch_idx in range(batch_size):
                        start = batch_idx * diverse_pool
                        stop = start + diverse_pool
                        subset = self._select_diverse_subset(
                            diverse_pool_texts[start:stop],
                            cfg.num_diverse_cands,
                        )
                        if len(subset) < cfg.num_diverse_cands:
                            subset.extend([""] * (cfg.num_diverse_cands - len(subset)))
                        diverse_texts.extend(subset)
                    actual_diverse_count = cfg.num_diverse_cands
                except Exception as error:
                    self.logger.warning(f"[{self.label}] Diverse beam failed ({error}), skipping")

            all_sample_texts = []
            num_temps = 0
            if cfg.use_sampling and cfg.sample_temperatures:
                num_temps = len(cfg.sample_temperatures)
                for temp_index, temp in enumerate(cfg.sample_temperatures):
                    try:
                        sampling_seed = self._sampling_seed(temp_index)
                        fork_devices = [self.device.index] if self.device.type == "cuda" else []
                        with torch.random.fork_rng(devices=fork_devices):
                            torch.manual_seed(sampling_seed)
                            if self.device.type == "cuda":
                                torch.cuda.manual_seed_all(sampling_seed)
                            sample_out = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                do_sample=True,
                                num_beams=1,
                                top_p=cfg.mbr_top_p,
                                temperature=temp,
                                num_return_sequences=sample_count,
                                **common_generate_kwargs,
                            )
                        sample_texts = self.tokenizer.batch_decode(sample_out, skip_special_tokens=True)
                        all_sample_texts.extend(sample_texts)
                    except Exception as error:
                        self.logger.warning(
                            f"[{self.label}] Sampling @ temp={temp:.2f} failed ({error}), padding with empty strings"
                        )
                        all_sample_texts.extend([""] * (batch_size * sample_count))

        pools = []
        for idx in range(batch_size):
            pool = []
            pool.extend(beam_texts[idx * beam_count : (idx + 1) * beam_count])
            if diverse_texts and actual_diverse_count > 0:
                pool.extend(diverse_texts[idx * actual_diverse_count : (idx + 1) * actual_diverse_count])
            if all_sample_texts and num_temps > 0:
                for temp_idx in range(num_temps):
                    start = temp_idx * batch_size * sample_count + idx * sample_count
                    pool.extend(all_sample_texts[start : start + sample_count])
            pools.append(pool)

        if pools:
            self.logger.info(
                f"[{self.label}] Pool per sample: "
                f"beam={beam_count} + diverse={actual_diverse_count} + sample={num_temps}x{sample_count}={num_temps * sample_count} "
                f"= {len(pools[0])} total"
            )
        return pools

    def unload(self):
        try:
            from optimum.bettertransformer import BetterTransformer

            self.model = BetterTransformer.reverse(self.model)
        except Exception:
            pass
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            total = torch.cuda.get_device_properties(self.device).total_memory
            free = (total - torch.cuda.memory_allocated(self.device)) / 1e9
            self.logger.info(f"[{self.label}] Unloaded from {self.device}. GPU free: {free:.2f} GB")


class MBRSelector:
    def __init__(
        self,
        pool_cap: int = 32,
        w_chrf: float = 0.55,
        w_bleu: float = 0.25,
        w_jaccard: float = 0.20,
        w_length: float = 0.10,
        w_support: float = 0.12,
    ):
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        self.pool_cap = pool_cap
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self.w_support = w_support
        self._full_total = max(w_chrf + w_bleu + w_jaccard + w_length + w_support, 1e-9)

    def _chrfpp(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(self._chrf_metric.sentence_score(a, [b]).score)

    def _bleu(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(self._bleu_metric.sentence_score(a, [b]).score)
        except Exception:
            return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 100.0
        if not ta or not tb:
            return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise_score_raw(self, a: str, b: str) -> float:
        return (
            self.w_chrf * self._chrfpp(a, b)
            + self.w_bleu * self._bleu(a, b)
            + self.w_jaccard * self._jaccard(a, b)
        )

    @staticmethod
    def _length_bonus(lengths: List[int], idx: int) -> float:
        if len(lengths) == 0:
            return 100.0
        median = float(np.median(lengths))
        sigma = max(median * 0.4, 5.0)
        z_score = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z_score * z_score)

    @staticmethod
    def _normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    def _dedup_with_support(self, xs: List[str]) -> tuple[List[str], List[int]]:
        seen = {}
        out = []
        support = []
        for text in xs:
            text = str(text).strip()
            if not text:
                continue
            key = self._normalize_key(text)
            if key in seen:
                support[seen[key]] += 1
                continue
            seen[key] = len(out)
            out.append(text)
            support.append(1)
        return out, support

    @staticmethod
    def _support_bonus(support: List[int], idx: int) -> float:
        if not support:
            return 0.0
        max_support = max(support)
        if max_support <= 1:
            return 0.0
        return 100.0 * (support[idx] - 1) / (max_support - 1)

    def pick(self, candidates: List[str]) -> str:
        cands, support = self._dedup_with_support(candidates)
        if support:
            ranked = sorted(zip(cands, support), key=lambda item: (-item[1], item[0]))
            cands = [item[0] for item in ranked]
            support = [item[1] for item in ranked]
        if self.pool_cap:
            cands = cands[: self.pool_cap]
            support = support[: self.pool_cap]
        num_candidates = len(cands)
        if num_candidates == 0:
            return ""
        if num_candidates == 1:
            return cands[0]
        lengths = [len(candidate.split()) for candidate in cands]
        scores = []
        for idx in range(num_candidates):
            pairwise = sum(
                self._pairwise_score_raw(cands[idx], cands[other_idx])
                for other_idx in range(num_candidates)
                if other_idx != idx
            ) / max(1, num_candidates - 1)
            length_bonus = self._length_bonus(lengths, idx)
            support_bonus = self._support_bonus(support, idx)
            total = (pairwise + self.w_length * length_bonus + self.w_support * support_bonus) / self._full_total
            scores.append(total)
        return cands[int(np.argmax(scores))]


class EnsembleDictEngine:
    def __init__(self, cfg: EnsembleDictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.postprocessor = VectorizedPostprocessor()
        self.mbr = MBRSelector(
            pool_cap=cfg.mbr_pool_cap,
            w_chrf=cfg.mbr_w_chrf,
            w_bleu=cfg.mbr_w_bleu,
            w_jaccard=cfg.mbr_w_jaccard,
            w_length=cfg.mbr_w_length,
            w_support=cfg.mbr_w_support,
        )

    def _adaptive_beams(self, attn: torch.Tensor) -> int:
        if not self.cfg.use_adaptive_beams:
            return self.cfg.num_beams
        attn_cpu = attn.detach().to("cpu")
        med = float(attn_cpu.sum(dim=1).float().median().item())
        short = max(self.cfg.num_beam_cands, self.cfg.num_beams // 2)
        return short if med < 100 else self.cfg.num_beams

    def _build_dataloader(self, dataset: DictionaryAkkadianDataset, wrapper: ModelWrapper):
        if self.cfg.use_bucket_batching:
            sampler = BucketBatchSampler(dataset, self.cfg.batch_size, self.cfg.num_buckets, self.logger)
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.cfg.num_workers,
                collate_fn=wrapper.collate,
                pin_memory=(wrapper.device.type == "cuda"),
            )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=wrapper.collate,
            pin_memory=(wrapper.device.type == "cuda"),
        )

    def _run_one_model(self, wrapper: ModelWrapper, dataset: DictionaryAkkadianDataset) -> dict[str, list[str]]:
        dataloader = self._build_dataloader(dataset, wrapper)
        pools_by_id: dict[str, list[str]] = {}
        with torch.inference_mode():
            for batch_ids, enc in tqdm(dataloader, desc=f"  [{wrapper.label}]"):
                input_ids = enc.input_ids.to(wrapper.device, non_blocking=True)
                attention_mask = enc.attention_mask.to(wrapper.device, non_blocking=True)
                beam_size = self._adaptive_beams(attention_mask)
                try:
                    batch_pools = wrapper.generate_candidates(input_ids, attention_mask, beam_size)
                    for sample_id, pool in zip(batch_ids, batch_pools, strict=True):
                        pools_by_id[str(sample_id)] = pool
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        self.logger.error(f"OOM in [{wrapper.label}] — skipping batch")
                        torch.cuda.empty_cache()
                        for sample_id in batch_ids:
                            pools_by_id.setdefault(str(sample_id), [])
                    else:
                        raise
                except Exception as error:
                    self.logger.error(f"[{wrapper.label}] batch error: {error}")
                    for sample_id in batch_ids:
                        pools_by_id.setdefault(str(sample_id), [])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return pools_by_id

    def _run_model_phase(
        self,
        model_path: str,
        label: str,
        device: torch.device,
        dataset: DictionaryAkkadianDataset,
    ) -> dict[str, list[str]]:
        wrapper = ModelWrapper(model_path, self.cfg, self.logger, label, device)
        try:
            return self._run_one_model(wrapper, dataset)
        finally:
            wrapper.unload()
            del wrapper

    def _run_loaded_model(self, wrapper: ModelWrapper, dataset: DictionaryAkkadianDataset) -> dict[str, list[str]]:
        return self._run_one_model(wrapper, dataset)

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        cands_per_sample = (
            cfg.num_beam_cands
            + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
            + cfg.num_sample_cands
        )
        self.logger.info("=" * 60)
        self.logger.info("Ensemble × Dictionary × MBR  |  Cross-model candidate pooling  v1")
        self.logger.info(f"  Model A           : {cfg.model_a_path}")
        self.logger.info(f"  Model B           : {cfg.model_b_path}")
        self.logger.info(
            f"  GPU mode          : {'dual' if cfg.dual_gpu else 'single'} "
            f"(count={cfg.cuda_device_count}, A={cfg.model_a_device}, B={cfg.model_b_device})"
        )
        self.logger.info(f"  Standard beam     : {cfg.num_beam_cands} (num_beams={cfg.num_beams})")
        self.logger.info(f"  no_repeat_ngram   : {cfg.no_repeat_ngram_size}")
        self.logger.info(
            f"  Diverse beam      : {'ON' if cfg.use_diverse_beam else 'OFF'} "
            f"cands={cfg.num_diverse_cands}, groups={cfg.num_beam_groups}, penalty={cfg.diversity_penalty}"
        )
        self.logger.info(
            f"  Multi-temp sample : {'ON' if cfg.use_sampling else 'OFF'} "
            f"temps={cfg.sample_temperatures}, {cfg.num_sample_per_temp}/temp → {cfg.num_sample_cands} total"
        )
        self.logger.info(f"  Cands/model/sample: {cands_per_sample} → pool ≈ {cands_per_sample * 2} (2 models, pre-dedup)")
        self.logger.info(f"  MBR pool cap      : {cfg.mbr_pool_cap}")
        self.logger.info(f"  BF16 AMP          : {cfg.use_bf16_amp}")
        self.logger.info(f"  batch_size        : {cfg.batch_size}")
        self.logger.info(f"  seed              : {cfg.seed}")
        self.logger.info(
            f"  Dictionary prompt : lexicon={cfg.lexicon_path}, gloss={cfg.gloss_path}, "
            f"onomasticon={cfg.onomasticon_path}, include_onomasticon={cfg.include_onomasticon}"
        )
        self.logger.info(
            f"  Hint config       : placement={cfg.hint_placement}, max_hints={cfg.max_dictionary_hints}, "
            f"max_entry_tokens={cfg.max_entry_token_length}, max_hint_words={cfg.max_hint_words}, "
            f"max_gloss_variants={cfg.max_gloss_variants}"
        )
        self.logger.info(
            f"  max_new_tokens    : {'dynamic' if cfg.use_dynamic_max_new_tokens else 'fixed'} "
            f"(base={cfg.max_target_length}, ratio={cfg.dynamic_max_new_tokens_ratio}, cap={cfg.dynamic_max_new_tokens_cap})"
        )
        self.logger.info("=" * 60)

        dataset = DictionaryAkkadianDataset(frame, self.logger)
        sample_ids = [str(sample_id) for sample_id in dataset.sample_ids]
        if cfg.dual_gpu:
            self.logger.info("Phase 1/2 — Dual-GPU inference (sequential load, parallel run)")
            wrapper_a = ModelWrapper(cfg.model_a_path, cfg, self.logger, "Model-A", cfg.model_a_device)
            wrapper_b = ModelWrapper(cfg.model_b_path, cfg, self.logger, "Model-B", cfg.model_b_device)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(self._run_loaded_model, wrapper_a, dataset)
                future_b = executor.submit(self._run_loaded_model, wrapper_b, dataset)
                pools_a = future_a.result()
                pools_b = future_b.result()
            wrapper_a.unload()
            wrapper_b.unload()
            del wrapper_a
            del wrapper_b
        else:
            self.logger.info("Phase 1/2 — Model A inference")
            pools_a = self._run_model_phase(cfg.model_a_path, "Model-A", cfg.model_a_device, dataset)
            self.logger.info("Phase 2/2 — Model B inference")
            pools_b = self._run_model_phase(cfg.model_b_path, "Model-B", cfg.model_b_device, dataset)

        self.logger.info("Phase 3/3 — Pool merge + MBR selection")
        results = []
        for sample_id in tqdm(sample_ids, desc="  MBR"):
            combined = pools_a.get(sample_id, []) + pools_b.get(sample_id, [])
            postprocessed = self.postprocessor.postprocess_batch(combined) if combined else []
            chosen = self.mbr.pick(postprocessed)
            if not chosen or not chosen.strip():
                chosen = "The tablet is too damaged to translate."
            results.append((sample_id, chosen))
            if len(results) % cfg.checkpoint_freq == 0:
                checkpoint_path = Path(cfg.output_dir) / f"checkpoint_{len(results)}.csv"
                pd.DataFrame(results, columns=["id", "translation"]).to_csv(checkpoint_path, index=False)
                self.logger.info(f"  Checkpoint: {len(results)} rows → {checkpoint_path}")
        result_df = pd.DataFrame(results, columns=["id", "translation"])
        self._validate(result_df)
        return result_df

    def _validate(self, df: pd.DataFrame) -> None:
        self.logger.info("=" * 60)
        empty = df["translation"].str.strip().eq("").sum()
        lengths = df["translation"].str.len()
        self.logger.info(f"Empty     : {empty} ({100 * empty / max(1, len(df)):.2f}%)")
        self.logger.info(
            f"Len mean  : {lengths.mean():.1f}  median: {lengths.median():.1f}  min: {lengths.min()}  max: {lengths.max()}"
        )
        for idx in [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]:
            row = df.iloc[idx]
            self.logger.info(f"  ID {row['id']}: {str(row['translation'])[:80]}")
        self.logger.info("=" * 60)


def print_env(cfg: EnsembleDictConfig) -> None:
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"GPU Mem  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Count: {cfg.cuda_device_count}")
        print(f"GPU Mode : {'dual' if cfg.dual_gpu else 'single'} (A={cfg.model_a_device}, B={cfg.model_b_device})")
        print(f"BF16     : {_cuda_bf16_supported()}")
    print(f"BF16 AMP : {cfg.use_bf16_amp}")
    print(f"Seed     : {cfg.seed}")
    print(f"Model A  : {cfg.model_a_path}")
    print(f"Model B  : {cfg.model_b_path}")
    print(f"Tokenizer: {cfg.tokenizer_path}")
    print(f"Lexicon  : {cfg.lexicon_path}")
    print(f"Gloss    : {cfg.gloss_path}")
    print(f"Onomast. : {cfg.onomasticon_path} (enabled={cfg.include_onomasticon})")
    print()
    cands_per_sample = (
        cfg.num_beam_cands
        + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
        + cfg.num_sample_cands
    )
    print(f"Candidates/sample/model : {cands_per_sample}")
    print(f"  ├─ standard beam      : {cfg.num_beam_cands} (num_beams={cfg.num_beams})")
    print(f"  ├─ no_repeat_ngram    : {cfg.no_repeat_ngram_size}")
    print(f"  ├─ diverse beam       : {cfg.num_diverse_cands if cfg.use_diverse_beam else 0} (groups={cfg.num_beam_groups}, penalty={cfg.diversity_penalty})")
    print(f"  └─ multi-temp sample  : {cfg.num_sample_cands} ({cfg.num_sample_per_temp}/temp × {cfg.sample_temperatures})")
    print(f"Total pool (2 models)   : ~{cands_per_sample * 2} (before dedup)")
    print()


def main() -> None:
    cfg = EnsembleDictConfig()
    _seed_everything(cfg.seed)
    logger = setup_logging(cfg.output_dir)
    print_env(cfg)

    logger.info(f"Loading test data: {cfg.test_data_path}")
    frame = load_input_frame(Path(cfg.test_data_path))
    logger.info(f"Test samples: {len(frame)}")

    logger.info("Preparing dictionary-augmented sources")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    dictionary_index = load_dictionary_entries(cfg)
    frame = prepare_dictionary_frame_for_inference(frame, dictionary_index, tokenizer, cfg)

    engine = EnsembleDictEngine(cfg, logger)
    results_df = engine.run(frame)

    out_path = Path(cfg.output_dir) / "submission.csv"
    write_submission(frame, results_df["translation"].tolist(), out_path)
    logger.info(f"Saved → {out_path}  ({len(results_df)} rows)")

    cfg_snap = {
        "test_data_path": cfg.test_data_path,
        "output_dir": cfg.output_dir,
        "model_a_path": cfg.model_a_path,
        "model_b_path": cfg.model_b_path,
        "tokenizer_path": cfg.tokenizer_path,
        "lexicon_path": str(cfg.lexicon_path),
        "gloss_path": str(cfg.gloss_path),
        "include_onomasticon": cfg.include_onomasticon,
        "onomasticon_path": str(cfg.onomasticon_path),
        "source_prefix": cfg.source_prefix,
        "hint_placement": cfg.hint_placement,
        "max_dictionary_hints": cfg.max_dictionary_hints,
        "max_entry_token_length": cfg.max_entry_token_length,
        "max_hint_words": cfg.max_hint_words,
        "max_gloss_variants": cfg.max_gloss_variants,
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
        "use_dynamic_max_new_tokens": cfg.use_dynamic_max_new_tokens,
        "dynamic_max_new_tokens_ratio": cfg.dynamic_max_new_tokens_ratio,
        "dynamic_max_new_tokens_cap": cfg.dynamic_max_new_tokens_cap,
        "num_beam_cands": cfg.num_beam_cands,
        "num_beams": cfg.num_beams,
        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        "length_penalty": cfg.length_penalty,
        "repetition_penalty": cfg.repetition_penalty,
        "use_diverse_beam": cfg.use_diverse_beam,
        "num_diverse_cands": cfg.num_diverse_cands,
        "num_diverse_beams": cfg.num_diverse_beams,
        "num_beam_groups": cfg.num_beam_groups,
        "diversity_penalty": cfg.diversity_penalty,
        "use_sampling": cfg.use_sampling,
        "sample_temperatures": cfg.sample_temperatures,
        "num_sample_per_temp": cfg.num_sample_per_temp,
        "mbr_top_p": cfg.mbr_top_p,
        "mbr_pool_cap": cfg.mbr_pool_cap,
        "mbr_w_chrf": cfg.mbr_w_chrf,
        "mbr_w_bleu": cfg.mbr_w_bleu,
        "mbr_w_jaccard": cfg.mbr_w_jaccard,
        "mbr_w_length": cfg.mbr_w_length,
        "mbr_w_support": cfg.mbr_w_support,
        "cuda_device_count": cfg.cuda_device_count,
        "dual_gpu": cfg.dual_gpu,
        "gpu_a": cfg.gpu_a,
        "gpu_b": cfg.gpu_b,
        "model_a_device": str(cfg.model_a_device),
        "model_b_device": str(cfg.model_b_device),
        "seed": cfg.seed,
    }
    with open(Path(cfg.output_dir) / "ensemble_dict_config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg_snap, handle, ensure_ascii=False, indent=2)

    print("Submission file:", out_path)
    print("Config saved to:", Path(cfg.output_dir) / "ensemble_dict_config.json")


if __name__ == "__main__":
    main()
