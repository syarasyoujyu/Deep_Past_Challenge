#!/usr/bin/env bash

set -euo pipefail

TRAIN_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2_sentence_split_refined_refined.csv"
MODEL_NAME_OR_PATH="artifacts/byt5-small"
OUTPUT_DIR="artifacts/byt5-small-continued"

if [[ ! -d "$MODEL_NAME_OR_PATH" ]]; then
  MODEL_NAME_OR_PATH="google/byt5-small"
fi

uv run python -m model.ft.train_byt5_small_continue \
  --train-path "$TRAIN_PATH" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --val-size 0.15 \
  --min-eval-examples 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --num-train-epochs 12 \
  --label-smoothing-factor 0.05 \
  --freeze-shared-embeddings true \
  --freeze-encoder true \
  --unfreeze-last-n-decoder-blocks 2 \
  --early-stopping-patience 3 \
  --disable-wandb true \
  "$@"
