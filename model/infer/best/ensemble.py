#!/usr/bin/env python3
"""
Deep Past Challenge — Akkadian-to-English Translation
Ensemble MBR inference with corrected postprocessing (FIXED version)

Corrections applied based on host v3 update + community discussion:
  1. 5/12 shekel → ⅓ shekel 15 grains (was ⅔)
  2. Commodity regex respects word boundaries (no import-tax corruption)
  3. Parentheses preserved in translations (appear in test set)
  4. Curly quotes → straight quotes (not deleted)
  5. ḫ/Ḫ → h/H in postprocessing (not in test set)
  6. Safer slash-alternative regex (protects fractions)
  7. Stray mark removal (.., xx, <<>>, <>)
"""


# ======================================================================
# Better Candidate Diversity and Stronger Consensus Selection
# ======================================================================

# ======================================================================
# Notebook structure
# ======================================================================

# ======================================================================
# Corrections applied (based on host recommendations + community discussion)
# ======================================================================
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# 1. Mixed precision helpers
#
# I use BF16 autocast whenever available.
# For ByT5 inference this is generally safer than FP16 while still reducing memory pressure.

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

# 2. Configuration
#
# This configuration keeps the public notebook transparent:
#
# - two different ByT5 checkpoints,
# - standard beam search,
# - optional diverse beam search,
# - multi-temperature sampling,
# - weighted MBR reranking.

@dataclass
class EnsembleMBRConfig:
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    output_dir:     str = "/kaggle/working/"
    model_a_path:   str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    model_b_path:   str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1"
    lexicon_path:   str = "/kaggle/input/datasets/yokoinaba/word-dict-akkad/OA_Lexicon_eBL_refined_with_definition.csv"
    onomasticon_path: str = "/home/watas/kaggle/Deep_Past_Challenge/data/onomasticon.csv"
    gpu_a:          int = 0
    gpu_b:          int = 1
    seed:           int = 42
    use_proper_name_normalization: bool = True
    max_name_token_length: int = 4
    max_name_replacements: int = 8
    proper_name_pn_translit_aliases: List[str] = field(
        default_factory=lambda: [
            "en-um-a-šur",
            "ša-lim-a-šur",
            "e-lá-ma",
            "a-šur-ma-lik",
            "en-na-sú-in",
            "šu-IŠTAR",
            "a-šur-i-mì-tí",
            "lá-qé-ep",
        ]
    )
    proper_name_pn_translation_aliases: List[str] = field(
        default_factory=lambda: [
            "Ennam-Aššur",
            "Šalim-Aššur",
            "Elamma",
            "Aššur-mālik",
            "Ennam-Suen",
            "Šu-Ištar",
            "Aššur-imittī",
            "Lā-qēpu"
        ]
    )
    proper_name_gn_translit_aliases: List[str] = field(
        default_factory=lambda: [
            "kà-ni-iš",
            "wa-ah-šu-ša-na",
            "dur4-hu-mì-it",
            "pu-ru-uš-ha-dim",
            "ha-hi-im"
        ]
    )
    proper_name_gn_translation_aliases: List[str] = field(
        default_factory=lambda: [
            "Kaneš",
            "Wahšušana",
            "Durhumit",
            "Purušhattu",
            "Hahhum",
        ]
    )

    max_input_length: int = 512
    max_new_tokens:   int = 384
    use_dynamic_max_new_tokens: bool = True
    dynamic_max_new_tokens_ratio: float = 2.5
    dynamic_max_new_tokens_cap: int = 512
    batch_size:       int = 8
    num_workers:      int = 2
    num_buckets:      int = 6

    num_beam_cands:      int = 5
    num_beams:           int = 8
    length_penalty:      float = 1.3
    early_stopping:      bool = True
    repetition_penalty:  float = 1.2
    no_repeat_ngram_size: int = 0

    use_diverse_beam:    bool = True
    num_diverse_cands:   int = 5
    num_diverse_beams:   int = 8
    num_beam_groups:     int = 4
    diversity_penalty:   float = 0.9

    use_sampling:        bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.55, 0.75, 0.95])
    num_sample_per_temp: int = 2
    mbr_top_p:           float = 0.92

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    mbr_pool_cap: int = 32

    mbr_w_chrf:     float = 0.55
    mbr_w_bleu:     float = 0.25
    mbr_w_jaccard:  float = 0.20
    mbr_w_length:   float = 0.04
    mbr_w_support:  float = 0.00

    use_mixed_precision:     bool = True
    use_better_transformer:  bool = True
    use_bucket_batching:     bool = True
    use_adaptive_beams:      bool = True
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 200

    def __post_init__(self):
        self.cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.dual_gpu = self.cuda_device_count >= 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lexicon_path = Path(self.lexicon_path)
        self.onomasticon_path = Path(self.onomasticon_path)
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
            self.use_mixed_precision
            and self.device.type == "cuda"
            and _cuda_bf16_supported()
        )

        assert self.num_beams >= self.num_beam_cands, "num_beams must be >= num_beam_cands"

        if self.use_diverse_beam:
            assert self.num_diverse_beams % self.num_beam_groups == 0
            assert self.num_diverse_beams >= self.num_diverse_cands

# 3. Logging utilities

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "ensemble_mbr.log"),
        ],
    )
    return logging.getLogger("ensemble_mbr")

# 4. Preprocessing
#
# This block normalizes Akkadian transliteration before tokenization.
#
# Key steps:
#
# - convert ASCII-style diacritic notation into normalized Unicode,
# - normalize gap markers,
# - standardize determinatives,
# - normalize decimal/fraction expressions,
# - clean whitespace and symbol variants.
#
# This part is important because small notation mismatches can fragment the candidate space and hurt MBR consistency.

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")

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
    re.I
)

def _normalize_gaps_vec(ser: pd.Series) -> pd.Series:
    return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)

_CHAR_TRANS = str.maketrans({
    "ḫ":"h","Ḫ":"H","ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")

_PN_RE = re.compile(r"\bPN\b")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")

_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}
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

def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]


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

class OptimizedPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = _normalize_gaps_vec(ser)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(",", "", regex=False)
        ser = ser.map(_normalize_plus_usage)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


@dataclass(frozen=True)
class ProperNameEntry:
    source_tokens: tuple[str, ...]
    target_name: str
    entity_kind: str


class ProperNameNormalizer:
    def __init__(self, cfg: EnsembleMBRConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.index = self._load_entries()
        self.logger.info(
            "Proper-name normalization: "
            f"{len(self.index)} PN/GN key(s) loaded"
        )

    def _normalize_source_form(self, text: str) -> tuple[str, ...]:
        source_form = " ".join(str(text).split()).strip()
        if not source_form:
            return ()
        source_form = _ascii_to_diacritics(source_form)
        source_form = source_form.translate(_CHAR_TRANS).replace(_SUB_X, "")
        source_form = source_form.replace(",", "")
        source_form = _normalize_plus_usage(source_form)
        source_form = re.sub(_WS_RE, " ", source_form).strip()
        source_tokens = tuple(token for token in source_form.split() if token)
        if not source_tokens or len(source_tokens) > self.cfg.max_name_token_length:
            return ()
        return source_tokens

    def _load_entries(self) -> dict[tuple[str, ...], ProperNameEntry]:
        index: dict[tuple[str, ...], ProperNameEntry] = {}
        if not self.cfg.lexicon_path.exists():
            self.logger.warning(
                f"Proper-name normalization skipped: lexicon not found at {self.cfg.lexicon_path}"
            )
            return {}

        rows: list[tuple[tuple[str, ...], str, str]] = []
        type_sets_by_form: dict[tuple[str, ...], set[str]] = {}

        with self.cfg.lexicon_path.open("r", encoding="utf-8", newline="") as lexicon_file:
            reader = csv.DictReader(lexicon_file)
            for row in reader:
                entity_type = str(row.get("type", "")).strip()
                source_form = " ".join(str(row.get("form", "")).split()).strip()
                if not source_form:
                    continue

                source_tokens = self._normalize_source_form(source_form)
                if not source_tokens:
                    continue

                type_sets_by_form.setdefault(source_tokens, set()).add(entity_type)

                if entity_type not in {"PN", "GN"}:
                    continue

                target_name = " ".join(str(row.get("lexeme", "")).split()).strip()
                if not target_name:
                    continue

                rows.append((source_tokens, target_name, entity_type))

        if self.cfg.onomasticon_path.exists():
            with self.cfg.onomasticon_path.open("r", encoding="utf-8", newline="") as onomasticon_file:
                reader = csv.DictReader(onomasticon_file)
                for row in reader:
                    target_name = " ".join(str(row.get("Name", "")).split()).strip()
                    if not target_name:
                        continue

                    spellings = str(row.get("Spellings_semicolon_separated", "")).split(";")
                    for spelling in spellings:
                        spelling = spelling.strip()
                        if not spelling:
                            continue

                        source_tokens = self._normalize_source_form(spelling)
                        if not source_tokens:
                            continue

                        allowed_types = type_sets_by_form.get(source_tokens, set())
                        if allowed_types and allowed_types != {"PN"}:
                            continue

                        rows.append((source_tokens, target_name, "PN"))

        skipped_ambiguous = 0
        for source_tokens, target_name, entity_type in rows:
            allowed_types = type_sets_by_form.get(source_tokens, set())
            if entity_type == "PN" and allowed_types and allowed_types != {"PN"}:
                skipped_ambiguous += 1
                continue
            if entity_type == "GN" and allowed_types and allowed_types != {"GN"}:
                skipped_ambiguous += 1
                continue

            index.setdefault(
                source_tokens,
                ProperNameEntry(
                    source_tokens=source_tokens,
                    target_name=target_name,
                    entity_kind=entity_type,
                ),
            )

        if skipped_ambiguous:
            self.logger.info(
                "Proper-name normalization skipped ambiguous forms: "
                f"{skipped_ambiguous}"
            )

        return index

    def transform_text(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        tokens = [token for token in str(text).split() if token]
        if not tokens or not self.index:
            return text, []

        pn_translit_alias_pool = self.cfg.proper_name_pn_translit_aliases
        pn_translation_alias_pool = self.cfg.proper_name_pn_translation_aliases
        gn_translit_alias_pool = self.cfg.proper_name_gn_translit_aliases
        gn_translation_alias_pool = self.cfg.proper_name_gn_translation_aliases
        if (
            not pn_translit_alias_pool
            and not pn_translation_alias_pool
            and not gn_translit_alias_pool
            and not gn_translation_alias_pool
        ):
            return text, []

        rewritten: list[str] = []
        restore_pairs: list[tuple[str, str]] = []
        cursor = 0
        replacements = 0
        pn_replacements = 0
        gn_replacements = 0

        while cursor < len(tokens):
            matched_entry = None
            matched_length = 0
            max_length = min(self.cfg.max_name_token_length, len(tokens) - cursor)
            for length in range(max_length, 0, -1):
                candidate = tuple(tokens[cursor : cursor + length])
                entry = self.index.get(candidate)
                if entry is None:
                    continue
                matched_entry = entry
                matched_length = length
                break

            if matched_entry is None or replacements >= self.cfg.max_name_replacements:
                rewritten.append(tokens[cursor])
                cursor += 1
                continue

            original_name = matched_entry.target_name
            if matched_entry.entity_kind == "PN":
                translit_alias_pool = pn_translit_alias_pool
                translation_alias_pool = pn_translation_alias_pool
                alias_index = pn_replacements
            else:
                translit_alias_pool = gn_translit_alias_pool
                translation_alias_pool = gn_translation_alias_pool
                alias_index = gn_replacements

            if (
                alias_index >= len(translit_alias_pool)
                or alias_index >= len(translation_alias_pool)
            ):
                rewritten.extend(tokens[cursor : cursor + matched_length])
                cursor += matched_length
                continue

            translit_alias = translit_alias_pool[alias_index]
            translation_alias = translation_alias_pool[alias_index]
            if translit_alias and translation_alias:
                rewritten.extend(translit_alias.split())
                restore_pairs.append((translation_alias, original_name))
                replacements += 1
                if matched_entry.entity_kind == "PN":
                    pn_replacements += 1
                else:
                    gn_replacements += 1
            else:
                rewritten.extend(tokens[cursor : cursor + matched_length])

            cursor += matched_length

        return " ".join(rewritten), restore_pairs

    def build_batch(self, texts: List[str]) -> tuple[List[str], List[list[tuple[str, str]]]]:
        transformed_texts: list[str] = []
        restore_maps: list[list[tuple[str, str]]] = []
        replacement_counts: list[int] = []

        for text in texts:
            transformed_text, restore_pairs = self.transform_text(text)
            transformed_texts.append(transformed_text)
            restore_maps.append(restore_pairs)
            replacement_counts.append(len(restore_pairs))

        if replacement_counts:
            covered = sum(1 for count in replacement_counts if count > 0)
            self.logger.info(
                "Proper-name normalization coverage: "
                f"{covered}/{len(replacement_counts)} rows, "
                f"mean replacements={sum(replacement_counts)/len(replacement_counts):.2f}"
            )
        return transformed_texts, restore_maps

    def restore_translation(self, translation: str, restore_pairs: list[tuple[str, str]]) -> str:
        restored = str(translation)
        if not restore_pairs:
            return restored

        for canonical_name, original_name in sorted(
            restore_pairs,
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if not canonical_name or not original_name:
                continue

            regex = re.compile(rf"(?<![A-Za-z]){re.escape(canonical_name)}(?![A-Za-z])", re.IGNORECASE)
            restored = regex.sub(original_name, restored)

        return restored

# 5. Postprocessing
#
# This block is where a lot of the practical gain comes from.
#
# The idea is not to aggressively rewrite the model output, but to remove avoidable formatting noise so that MBR compares candidates on their *meaningful content* rather than superficial inconsistencies.
#
# I keep `/`, straight quotes, parentheses, and apostrophes when they may be meaningful.
#
# **Key corrections in this version:**
# - Parentheses `()` are **preserved** (they appear in the test set)
# - Curly quotes are **converted to straight quotes** (not deleted)
# - Commodity replacements only match space-prefixed hyphens
# - 5/12 shekel correctly maps to ⅓ shekel 15 grains
# - ḫ/Ḫ characters normalized to h/H

_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
# FIX #4: curly quotes → straight quotes (not deletion)
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")   # " " → "
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")   # ' ' → '
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])") 

# FIX #3: Removed '(' and ')' from forbidden chars — host confirmed
# parentheses appear in the test set, e.g. "(silver)", "(and)"
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]ʾ;')

# FIX #2: Commodity regex now requires a SPACE before the hyphen,
# so "import-tax" and "kutānu-textiles" are NOT corrupted.
# Only standalone " -gold", " -tax", " -textiles" are replaced.
_COMMODITY_RE = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}

def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]

# FIX #1: 5/12 shekel → ⅓ shekel 15 grains (was ⅔, host confirmed)
_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅓ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

# FIX #6: Safer slash-alternative regex — require word chars around the slash
# and explicitly exclude digit/digit patterns (fractions like 1/3)
_SLASH_ALT_RE = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')

_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')

# FIX #7: Additional stray marks the host recommends removing
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')

# FIX #5: ḫ/Ḫ normalization for translations (host confirmed not in test set)
_HACEK_TRANS = str.maketrans({"ḫ": "h", "Ḫ": "H"})

def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"

class VectorizedPostprocessor:
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations).fillna("").astype(str)
        s = _normalize_gaps_vec(s)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)

        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)

        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)
        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_EXTRA_STRAY_RE, "", regex=True)       # FIX #7
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)

        # FIX #4: curly → straight quotes (not deletion)
        s = s.str.replace(_CURLY_DQ_RE, '"', regex=True)
        s = s.str.replace(_CURLY_SQ_RE, "'", regex=True)

        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)

        # Protect <gap> markers before stripping forbidden chars
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)

        # FIX #5: ḫ/Ḫ → h/H in translations
        s = s.str.translate(_HACEK_TRANS)

        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)

        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()

# 6. Dataset and bucket batching
#
# Bucketed batching reduces padding waste and keeps throughput more stable for variable-length transliterations.

class AkkadianDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: OptimizedPreprocessor,
        logger: logging.Logger,
        proper_name_normalizer: ProperNameNormalizer | None = None,
    ):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.restore_maps: list[list[tuple[str, str]]] = [[] for _ in proc]
        if proper_name_normalizer is not None:
            proc, self.restore_maps = proper_name_normalizer.build_batch(proc)
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]
        logger.info(f"Dataset: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets, logger, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i*bsize : None if i == num_buckets-1 else (i+1)*bsize]
            for i in range(num_buckets)
        ]

        for i, b in enumerate(self.buckets):
            if b:
                bl = [lengths[x] for x in b]
                logger.info(f"  Bucket {i}: {len(b)} samples, len [{min(bl)}, {max(bl)}]")

    def __iter__(self):
        for bucket in self.buckets:
            b = list(bucket)
            if self.shuffle:
                random.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                yield b[i:i+self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)

# 7. Model wrapper and candidate generation
#
# This is the main inference change compared with the earlier public version.
#
# Candidate sources
#
# For each model and each sample, the pool can include:
#
# - **standard beam candidates**
# - **diverse beam candidates** (optional)
# - **multi-temperature sampled candidates**
#
# In my current public LB 34.9 configuration:
#
# - diverse beam is **prepared but disabled**
# - multi-temperature sampling is **enabled**
# - the effective per-model pool is:
#
# `4 beam + 6 sampled = 10 candidates`
#
# Across two models, that gives about **20 candidates before deduplication**.
#
# Why this helped
#
# The previous version already benefited from cross-model pooling.
# This version improved further because the candidate pool became more heterogeneous, which gives MBR a better chance to select a consensus output that is both faithful and fluent.

class ModelWrapper:
    def __init__(
        self,
        model_path: str,
        cfg: EnsembleMBRConfig,
        logger: logging.Logger,
        label: str,
        device: torch.device,
    ):
        self.cfg = cfg
        self.logger = logger
        self.label = label
        self.device = device
        self.tokenizer = None
        self.model = None

        logger.info(f"[{label}] Loading from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._load_model_on_device(model_path).eval()

        if self.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        n = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[{label}] {n:,} parameters")

        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated(self.device) / 1e9
            logger.info(f"[{label}] GPU mem used on {self.device}: {used:.2f} GB")

        if cfg.use_better_transformer and self.device.type == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info(f"[{label}] BetterTransformer applied")
            except Exception as e:
                logger.warning(f"[{label}] BetterTransformer skipped: {e}")

    def _load_model_on_device(self, model_path: str):
        if self.device.type == "cuda":
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    device_map={"": str(self.device)},
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                self.logger.warning(
                    f"[{self.label}] device_map load skipped ({e}), falling back to standard load"
                )

        try:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=False,
            ).to(self.device)
        except NotImplementedError as e:
            if "meta tensor" not in str(e):
                raise
            self.logger.warning(
                f"[{self.label}] standard load hit meta tensors ({e}), "
                "falling back to config + sharded checkpoint load"
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
        ids = [s[0] for s in batch_samples]
        texts = [s[1] for s in batch_samples]
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
                max_similarity = max(
                    self._token_jaccard(candidate, chosen) for chosen in selected
                )
                score = -max_similarity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def generate_candidates(self, input_ids, attention_mask, beam_size: int) -> List[List[str]]:
        cfg = self.cfg
        B = input_ids.shape[0]
        ctx = _bf16_ctx(self.device, cfg.use_bf16_amp)
        dynamic_max_new_tokens = cfg.max_new_tokens
        if cfg.use_dynamic_max_new_tokens:
            dynamic_max_new_tokens = min(
                cfg.dynamic_max_new_tokens_cap,
                max(1, int(input_ids.shape[1] * cfg.dynamic_max_new_tokens_ratio)),
            )

        Rb = cfg.num_beam_cands
        Rd = cfg.num_diverse_cands if cfg.use_diverse_beam else 0
        Rs = cfg.num_sample_per_temp

        with ctx:
            nb = max(beam_size, Rb)
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
                num_beams=nb,
                num_return_sequences=Rb,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                **common_generate_kwargs,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            diverse_texts = []
            actual_Rd = 0
            if cfg.use_diverse_beam:
                try:
                    diverse_beam_pool = max(cfg.num_diverse_beams, cfg.num_diverse_cands * 3)
                    div_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=diverse_beam_pool,
                        num_return_sequences=diverse_beam_pool,
                        length_penalty=cfg.length_penalty,
                        early_stopping=cfg.early_stopping,
                        **common_generate_kwargs,
                    )
                    diverse_pool_texts = self.tokenizer.batch_decode(div_out, skip_special_tokens=True)
                    diverse_texts = []
                    for batch_idx in range(B):
                        start = batch_idx * diverse_beam_pool
                        stop = start + diverse_beam_pool
                        subset = self._select_diverse_subset(
                            diverse_pool_texts[start:stop],
                            cfg.num_diverse_cands,
                        )
                        if len(subset) < cfg.num_diverse_cands:
                            subset.extend([""] * (cfg.num_diverse_cands - len(subset)))
                        diverse_texts.extend(subset)
                    actual_Rd = cfg.num_diverse_cands
                except Exception as e:
                    self.logger.warning(f"[{self.label}] Diverse beam failed ({e}), skipping")

            all_samp_texts = []
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
                            samp_out = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                do_sample=True,
                                num_beams=1,
                                top_p=cfg.mbr_top_p,
                                temperature=temp,
                                num_return_sequences=Rs,
                                **common_generate_kwargs,
                            )
                        temp_texts = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)
                        all_samp_texts.extend(temp_texts)
                    except Exception as e:
                        self.logger.warning(
                            f"[{self.label}] Sampling @ temp={temp:.2f} failed ({e}), padding with empty strings"
                        )
                        all_samp_texts.extend([""] * (B * Rs))

        pools = []
        for i in range(B):
            pool = []
            pool.extend(beam_texts[i * Rb : (i + 1) * Rb])

            if diverse_texts and actual_Rd > 0:
                pool.extend(diverse_texts[i * actual_Rd : (i + 1) * actual_Rd])

            if all_samp_texts and num_temps > 0:
                for t_idx in range(num_temps):
                    seg_start = t_idx * B * Rs + i * Rs
                    pool.extend(all_samp_texts[seg_start : seg_start + Rs])

            pools.append(pool)

        if pools:
            self.logger.info(
                f"[{self.label}] Pool per sample: "
                f"beam={Rb} + diverse={actual_Rd} + sample={num_temps}x{Rs}={num_temps*Rs} "
                f"= {len(pools[0])} total"
            )

        return pools

    def unload(self):
        label = self.label
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
            self.logger.info(f"[{label}] Unloaded from {self.device}. GPU free: {free:.2f} GB")

# 8. Weighted MBR selector
#
# This is the second major improvement.
#
# Earlier, I used **chrF++ only**.
# In this version, I use a weighted combination of:
#
# - **chrF++** for character-level robustness,
# - **BLEU** for word-level precision,
# - **Jaccard** for fast token overlap regularization,
# - **length bonus** to reduce pathological outputs.
#
# This is still a simple MBR implementation, but it is noticeably stronger in practice than relying on a single metric.

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
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)
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
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    def _dedup_with_support(self, xs: List[str]) -> tuple[List[str], List[int]]:
        seen = {}
        out = []
        support = []
        for x in xs:
            x = str(x).strip()
            if not x:
                continue
            key = self._normalize_key(x)
            if key in seen:
                support[seen[key]] += 1
                continue
            seen[key] = len(out)
            out.append(x)
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
            ranked = sorted(
                zip(cands, support),
                key=lambda item: (-item[1], item[0]),
            )
            cands = [item[0] for item in ranked]
            support = [item[1] for item in ranked]
        if self.pool_cap:
            cands = cands[:self.pool_cap]
            support = support[:self.pool_cap]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        lengths = [len(c.split()) for c in cands]
        scores = []

        for i in range(n):
            pw_raw = sum(
                self._pairwise_score_raw(cands[i], cands[j])
                for j in range(n) if j != i
            ) / max(1, n - 1)

            lb = self._length_bonus(lengths, i)
            sb = self._support_bonus(support, i)
            total = (pw_raw + self.w_length * lb + self.w_support * sb) / self._full_total
            scores.append(total)

        return cands[int(np.argmax(scores))]

# 9. End-to-end engine

class EnsembleMBREngine:
    def __init__(self, cfg: EnsembleMBRConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.preprocessor = OptimizedPreprocessor()
        self.proper_name_normalizer = None
        if cfg.use_proper_name_normalization:
            self.proper_name_normalizer = ProperNameNormalizer(cfg, logger)
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

    def _build_dataloader(self, dataset: AkkadianDataset, wrapper: ModelWrapper) -> DataLoader:
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

    def _run_one_model(self, wrapper: ModelWrapper, dataset: AkkadianDataset) -> dict:
        dl = self._build_dataloader(dataset, wrapper)
        pools_by_id = {}

        with torch.inference_mode():
            for batch_ids, enc in tqdm(dl, desc=f"  [{wrapper.label}]"):
                input_ids = enc.input_ids.to(wrapper.device, non_blocking=True)
                attn = enc.attention_mask.to(wrapper.device, non_blocking=True)
                beam_size = self._adaptive_beams(attn)

                try:
                    batch_pools = wrapper.generate_candidates(input_ids, attn, beam_size)
                    for sid, pool in zip(batch_ids, batch_pools):
                        pools_by_id[str(sid)] = pool
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"OOM in [{wrapper.label}] — skipping batch")
                        torch.cuda.empty_cache()
                        for sid in batch_ids:
                            pools_by_id.setdefault(str(sid), [])
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"[{wrapper.label}] batch error: {e}")
                    for sid in batch_ids:
                        pools_by_id.setdefault(str(sid), [])

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return pools_by_id

    def _run_model_phase(
        self,
        model_path: str,
        label: str,
        device: torch.device,
        dataset: AkkadianDataset,
    ) -> dict:
        wrapper = ModelWrapper(model_path, self.cfg, self.logger, label, device)
        try:
            return self._run_one_model(wrapper, dataset)
        finally:
            wrapper.unload()
            del wrapper

    def _run_loaded_model(self, wrapper: ModelWrapper, dataset: AkkadianDataset) -> dict:
        return self._run_one_model(wrapper, dataset)

    def run(self, test_df: pd.DataFrame) -> pd.DataFrame:
        cfg, logger = self.cfg, self.logger

        cands_per_sample = (
            cfg.num_beam_cands
            + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
            + cfg.num_sample_cands
        )

        logger.info("=" * 60)
        logger.info("Ensemble × MBR  |  Cross-model candidate pooling  v2")
        logger.info(f"  Model A           : {cfg.model_a_path}")
        logger.info(f"  Model B           : {cfg.model_b_path}")
        logger.info(
            f"  GPU mode          : {'dual' if cfg.dual_gpu else 'single'} "
            f"(count={cfg.cuda_device_count}, A={cfg.model_a_device}, B={cfg.model_b_device})"
        )
        logger.info(f"  Standard beam     : {cfg.num_beam_cands} (num_beams={cfg.num_beams})")
        logger.info(f"  no_repeat_ngram   : {cfg.no_repeat_ngram_size}")
        logger.info(
            f"  Diverse beam      : {'ON' if cfg.use_diverse_beam else 'OFF'} "
            f"cands={cfg.num_diverse_cands}, groups={cfg.num_beam_groups}, penalty={cfg.diversity_penalty}"
        )
        logger.info(
            f"  Multi-temp sample : {'ON' if cfg.use_sampling else 'OFF'} "
            f"temps={cfg.sample_temperatures}, {cfg.num_sample_per_temp}/temp "
            f"→ {cfg.num_sample_cands} total"
        )
        logger.info(
            f"  Cands/model/sample: {cands_per_sample} "
            f"→ pool ≈ {cands_per_sample * 2} (2 models, pre-dedup)"
        )
        logger.info(f"  MBR pool cap      : {cfg.mbr_pool_cap}")
        logger.info(f"  BF16 AMP          : {cfg.use_bf16_amp}")
        logger.info(f"  batch_size        : {cfg.batch_size}")
        logger.info(f"  seed              : {cfg.seed}")
        logger.info(f"  MBR support bonus : {cfg.mbr_w_support}")
        logger.info(
            f"  Proper-name norm  : {'ON' if cfg.use_proper_name_normalization else 'OFF'} "
            f"(lexicon={cfg.lexicon_path}, onomasticon={cfg.onomasticon_path}, "
            f"pn_t={len(cfg.proper_name_pn_translit_aliases)}, "
            f"pn_x={len(cfg.proper_name_pn_translation_aliases)}, "
            f"gn_t={len(cfg.proper_name_gn_translit_aliases)}, "
            f"gn_x={len(cfg.proper_name_gn_translation_aliases)})"
        )
        logger.info(
            f"  max_new_tokens    : "
            f"{'dynamic' if cfg.use_dynamic_max_new_tokens else 'fixed'} "
            f"(base={cfg.max_new_tokens}, ratio={cfg.dynamic_max_new_tokens_ratio}, "
            f"cap={cfg.dynamic_max_new_tokens_cap})"
        )
        logger.info("=" * 60)

        dataset = AkkadianDataset(
            test_df,
            self.preprocessor,
            logger,
            self.proper_name_normalizer,
        )
        sample_ids = [str(s) for s in dataset.sample_ids]
        restore_map_by_id = {
            str(sample_id): restore_pairs
            for sample_id, restore_pairs in zip(dataset.sample_ids, dataset.restore_maps)
        }

        if cfg.dual_gpu:
            logger.info("Phase 1/2 — Dual-GPU inference (sequential load, parallel run)")
            wrapper_a = ModelWrapper(cfg.model_a_path, cfg, logger, "Model-A", cfg.model_a_device)
            wrapper_b = ModelWrapper(cfg.model_b_path, cfg, logger, "Model-B", cfg.model_b_device)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(
                    self._run_loaded_model,
                    wrapper_a,
                    dataset,
                )
                future_b = executor.submit(
                    self._run_loaded_model,
                    wrapper_b,
                    dataset,
                )
                pools_a = future_a.result()
                pools_b = future_b.result()
            wrapper_a.unload()
            wrapper_b.unload()
            del wrapper_a
            del wrapper_b
        else:
            logger.info("Phase 1/2 — Model A inference")
            pools_a = self._run_model_phase(
                cfg.model_a_path,
                "Model-A",
                cfg.model_a_device,
                dataset,
            )

            logger.info("Phase 2/2 — Model B inference")
            pools_b = self._run_model_phase(
                cfg.model_b_path,
                "Model-B",
                cfg.model_b_device,
                dataset,
            )

        logger.info("Phase 3/3 — Pool merge + MBR selection")
        results = []

        for sid in tqdm(sample_ids, desc="  MBR"):
            combined = pools_a.get(sid, []) + pools_b.get(sid, [])
            pp = self.postprocessor.postprocess_batch(combined) if combined else []
            if self.proper_name_normalizer is not None:
                pp = [
                    self.proper_name_normalizer.restore_translation(
                        text,
                        restore_map_by_id.get(sid, []),
                    )
                    for text in pp
                ]
            chosen = self.mbr.pick(pp)

            if not chosen or not chosen.strip():
                chosen = "The tablet is too damaged to translate."

            results.append((sid, chosen))

            if len(results) % cfg.checkpoint_freq == 0:
                ckpt = Path(cfg.output_dir) / f"checkpoint_{len(results)}.csv"
                pd.DataFrame(results, columns=["id", "translation"]).to_csv(ckpt, index=False)
                logger.info(f"  Checkpoint: {len(results)} rows → {ckpt}")

        result_df = pd.DataFrame(results, columns=["id", "translation"])
        self._validate(result_df)
        return result_df

    def _validate(self, df: pd.DataFrame):
        logger = self.logger
        logger.info("=" * 60)
        empty = df["translation"].str.strip().eq("").sum()
        lens = df["translation"].str.len()
        logger.info(f"Empty     : {empty} ({100*empty/max(1,len(df)):.2f}%)")
        logger.info(f"Len mean  : {lens.mean():.1f}  median: {lens.median():.1f}  min: {lens.min()}  max: {lens.max()}")

        for idx in [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]:
            row = df.iloc[idx]
            logger.info(f"  ID {row['id']}: {str(row['translation'])[:80]}")

        logger.info("=" * 60)

# 10. Environment summary helper

def print_env(cfg: EnsembleMBRConfig):
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Mem  : {mem:.2f} GB")
        print(f"GPU Count: {cfg.cuda_device_count}")
        print(
            f"GPU Mode : {'dual' if cfg.dual_gpu else 'single'} "
            f"(A={cfg.model_a_device}, B={cfg.model_b_device})"
        )
        print(f"BF16     : {_cuda_bf16_supported()}")
    print(f"BF16 AMP : {cfg.use_bf16_amp}")
    print(f"Seed     : {cfg.seed}")
    print(
        f"PN/GN norm: {cfg.use_proper_name_normalization} "
        f"({cfg.lexicon_path}, {cfg.onomasticon_path}, pn_t={len(cfg.proper_name_pn_translit_aliases)}, "
        f"pn_x={len(cfg.proper_name_pn_translation_aliases)}, "
        f"gn_t={len(cfg.proper_name_gn_translit_aliases)}, "
        f"gn_x={len(cfg.proper_name_gn_translation_aliases)})"
    )
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

# 11. Run inference and export submission
#
# This is the exact end-to-end submission path.


# ======================================================================
# Final notes
# ======================================================================


# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":
    cfg = EnsembleMBRConfig()
    _seed_everything(cfg.seed)
    logger = setup_logging(cfg.output_dir)

    print_env(cfg)

    logger.info(f"Loading test data: {cfg.test_data_path}")
    test_df = pd.read_csv(cfg.test_data_path, encoding="utf-8")
    logger.info(f"Test samples: {len(test_df)}")

    engine = EnsembleMBREngine(cfg, logger)
    results_df = engine.run(test_df)

    out_path = Path(cfg.output_dir) / "submission.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  ({len(results_df)} rows)")
    results_df.head()
    cfg_snap = {
        "model_a": cfg.model_a_path,
        "model_b": cfg.model_b_path,
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
        "num_sample_cands": cfg.num_sample_cands,
        "mbr_top_p": cfg.mbr_top_p,
        "mbr_w_chrf": cfg.mbr_w_chrf,
        "mbr_w_bleu": cfg.mbr_w_bleu,
        "mbr_w_jaccard": cfg.mbr_w_jaccard,
        "mbr_w_length": cfg.mbr_w_length,
        "mbr_w_support": cfg.mbr_w_support,
        "mbr_pool_cap": cfg.mbr_pool_cap,
        "max_new_tokens": cfg.max_new_tokens,
        "use_dynamic_max_new_tokens": cfg.use_dynamic_max_new_tokens,
        "dynamic_max_new_tokens_ratio": cfg.dynamic_max_new_tokens_ratio,
        "dynamic_max_new_tokens_cap": cfg.dynamic_max_new_tokens_cap,
        "use_bf16_amp": cfg.use_bf16_amp,
        "batch_size": cfg.batch_size,
        "cuda_device_count": cfg.cuda_device_count,
        "dual_gpu": cfg.dual_gpu,
        "gpu_a": cfg.gpu_a,
        "gpu_b": cfg.gpu_b,
        "seed": cfg.seed,
        "use_proper_name_normalization": cfg.use_proper_name_normalization,
        "lexicon_path": str(cfg.lexicon_path),
        "onomasticon_path": str(cfg.onomasticon_path),
        "max_name_token_length": cfg.max_name_token_length,
        "max_name_replacements": cfg.max_name_replacements,
        "proper_name_pn_translit_aliases": cfg.proper_name_pn_translit_aliases,
        "proper_name_pn_translation_aliases": cfg.proper_name_pn_translation_aliases,
        "proper_name_gn_translit_aliases": cfg.proper_name_gn_translit_aliases,
        "proper_name_gn_translation_aliases": cfg.proper_name_gn_translation_aliases,
        "model_a_device": str(cfg.model_a_device),
        "model_b_device": str(cfg.model_b_device),
    }
    with open(Path(cfg.output_dir) / "ensemble_mbr_config.json", "w") as f:
        json.dump(cfg_snap, f, indent=2)

    print("Submission file:", out_path)
    print("Config saved to:", Path(cfg.output_dir) / "ensemble_mbr_config.json")
