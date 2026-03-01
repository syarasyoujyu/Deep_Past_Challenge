from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_train_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"transliteration", "translation"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"train data is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["transliteration"] = frame["transliteration"].fillna("").astype(str).str.strip()
    frame["translation"] = frame["translation"].fillna("").astype(str).str.strip()
    frame = frame[(frame["transliteration"] != "") & (frame["translation"] != "")]
    frame = frame.reset_index(drop=True)
    return frame


def read_test_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"id", "transliteration"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"test data is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["transliteration"] = frame["transliteration"].fillna("").astype(str).str.strip()
    return frame


def build_generation_config(args) -> dict[str, int | float | bool]:
    return {
        "max_new_tokens": args.max_target_length,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
    }


def load_tokenizer(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
