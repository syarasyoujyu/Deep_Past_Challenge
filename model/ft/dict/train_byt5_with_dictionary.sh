#!/usr/bin/env bash

set -euo pipefail

TRAIN_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2_sentence_split_refined_refined.csv"
LEXICON_PATH="data/OA_Lexicon_eBL_refined_with_definition.csv"
OUTPUT_DIR="artifacts/byt5-small-dict"
MODEL_NAME="google/byt5-small"

uv run python -m model.ft.dict.train_byt5_with_dictionary \
  --train-path "$TRAIN_PATH" \
  --lexicon-path "$LEXICON_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --hint-placement prepend \
  --max-dictionary-hints 8 \
  --max-entry-token-length 4 \
  --learning-rate 5e-5 \
  --num-train-epochs 10 \
  --disable-wandb true \
  "$@"
