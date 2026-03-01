from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM

from model.common import DATA_DIR, ROOT_DIR, build_generation_config, load_tokenizer, read_test_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run translation inference for the Deep Past challenge.")
    parser.add_argument("--test-path", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--submission-path", type=Path, default=ROOT_DIR / "submission.csv")
    parser.add_argument("--model-path", type=str, default=str(ROOT_DIR / "artifacts" / "byt5-small"))
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English: ")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batch_iterable(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    test_frame = read_test_frame(args.test_path)

    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    generation_config = build_generation_config(args)
    prefixed_inputs = [f"{args.source_prefix}{text}" for text in test_frame["transliteration"].tolist()]

    predictions: list[str] = []
    for batch in batch_iterable(prefixed_inputs, args.batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, **generation_config)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    submission = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "translation": [prediction.strip() for prediction in predictions],
        }
    )
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)


if __name__ == "__main__":
    main()
