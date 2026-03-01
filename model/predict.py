from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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


def main() -> None:
    args = parse_args()
    test_frame = read_test_frame(args.test_path)

    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    generation_config = build_generation_config(args)

    dataset = Dataset.from_dict(
        {
            "id": test_frame["id"].astype(str).tolist(),
            "input_text": [
                f"{args.source_prefix}{text}" for text in test_frame["transliteration"].tolist()
            ],
        }
    )

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            truncation=True,
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    predict_args = Seq2SeqTrainingArguments(
        output_dir=str(args.submission_path.parent / ".predict_tmp"),
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        report_to=[],
        fp16=False,
        bf16=False,
        use_cpu=args.device == "cpu" or not torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=predict_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    prediction_output = trainer.predict(tokenized, **generation_config)
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    submission = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "translation": [prediction.strip() for prediction in decoded_predictions],
        }
    )
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)


if __name__ == "__main__":
    main()
