#!/usr/bin/env bash

set -euo pipefail

TRAIN_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2_sentence_split_refined_refined.csv"
OUTPUT_DIR="artifacts/byt5-small-dict-rl"
LEXICON_PATH="data/OA_Lexicon_eBL_refined_with_definition.csv"
GLOSS_PATH="data/now/train_openai_gloss_compact.jsonl"
MODEL_NAME="google/byt5-small"

uv run python -m model.ft.dict.train_byt5_with_dictionary_rl \
  --train-path "$TRAIN_PATH" \
  --lexicon-path "$LEXICON_PATH" \
  --gloss-path "$GLOSS_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --learning-rate 2e-5 \
  --num-train-epochs 5 \
  --rl-loss-weight 0.2 \
  --ce-loss-weight 1.0 \
  --reward-bleu-weight 0.5 \
  --reward-chrfpp-weight 0.5 \
  --disable-wandb true \
  "$@"
