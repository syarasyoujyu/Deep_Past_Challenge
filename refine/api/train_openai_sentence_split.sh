#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"
INPUT_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2.csv"
OUTPUT_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2_sentence_split.csv"
CHECKPOINT_PATH="data/supplement/Michel_Old_Assyrian_Letters_Corpus/train_refined_v2_sentence_split_compact.jsonl"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

uv run --with tqdm python -m refine.api.train_openai_sentence_split \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --max-workers 30\
  --max-rows 1000\
  "$@"
