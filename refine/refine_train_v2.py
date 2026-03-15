from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / 'data' / 'train_truncated.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'train_refined_v2.csv'
SHORT_TRANSLITERATION_WORD_LIMIT = 10
SHORT_TRANSLATION_WORD_LIMIT=10
# ---------------------------------------------------------------------------
# 4. Preprocessing  (unchanged from v3)
# ---------------------------------------------------------------------------


_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz","š").replace("SZ","Š")
    s = s.replace("s,","ṣ").replace("S,","Ṣ")
    s = s.replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_ALLOWED_FRACS = [
    (1/6,"0.16666"),(1/4,"0.25"),(1/3,"0.33333"),(1/2,"0.5"),
    (2/3,"0.66666"),(3/4,"0.75"),(5/6,"0.83333"),
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
_DET_UPPER_RE  = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE  = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")

_PN_RE         = re.compile(r"\bPN\b")
_KUBABBAR_RE   = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333":"⅚","0.6666":"⅔","0.3333":"⅓","0.1666":"⅙",
    "0.625":"⅝","0.75":"¾","0.25":"¼","0.5":"½",
}

def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]

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
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()

# ---------------------------------------------------------------------------
# 5. Postprocessing  (all v3 fixes preserved)
# ---------------------------------------------------------------------------

_SOFT_GRAM_RE  = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE  = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE  = re.compile(r"\(\?\)")
_CURLY_DQ_RE   = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE   = re.compile("[\u2018\u2019]")
_MONTH_RE      = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT     = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}
_REPEAT_WORD_RE  = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE  = re.compile(r"\s+([.,:])") 
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')
_COMMODITY_RE    = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL  = {"gold":"pašallum gold","tax":"šadduātum tax","textiles":"kutānum textiles"}

def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]

_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I),      '⅓ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I),      '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

_SLASH_ALT_RE   = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE   = re.compile(r'(?:<gap>\s*){2,}')
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')
_HACEK_TRANS    = str.maketrans({"ḫ":"h","Ḫ":"H"})

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
        s = s.str.replace(_EXTRA_STRAY_RE, "", regex=True)
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
        s = s.str.replace(_CURLY_DQ_RE, '"', regex=True)
        s = s.str.replace(_CURLY_SQ_RE, "'", regex=True)
        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        s = s.str.translate(_HACEK_TRANS)
        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)
        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()


preprocessor=OptimizedPreprocessor()
postprocessor=VectorizedPostprocessor()

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    transliteration = preprocessor.preprocess_batch(df["transliteration"].tolist())
    translation = postprocessor.postprocess_batch(df["translation"].tolist())
    df["transliteration"] = transliteration
    df["translation"] = translation
    df.to_csv(OUTPUT_PATH, index=False)