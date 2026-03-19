#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="data/supplement/Michel_Old_Assyrian_Letters_Corpus"
DICT_DIR="data"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  ./refine/refine.sh [input_csv] [truncated_csv] [refined_csv] [sentence_csv]

Defaults:
  input_csv      $DATA_DIR/train_refined_v2_sentence_split_refined.csv
  truncated_csv  $DATA_DIR/train_refined_v2_sentence_split_refined_truncated.csv
  refined_csv    $DATA_DIR/train_refined_v2_sentence_split_refined_refined.csv
  sentence_csv   $DICT_DIR/Sentences_Oare_FirstWord_LinNum.csv
EOF
  exit 0
fi

INPUT_CSV="${1:-$DATA_DIR/train_refined_v2_sentence_split_refined.csv}"
TRUNCATED_CSV="${2:-$DATA_DIR/train_refined_v2_sentence_split_refined_truncated.csv}"
REFINED_CSV="${3:-$DATA_DIR/train_refined_v2_sentence_split_refined_refined.csv}"
SENTENCE_CSV="${4:-$DICT_DIR/Sentences_Oare_FirstWord_LinNum.csv}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

echo "Input CSV: $INPUT_CSV"
echo "Truncated CSV: $TRUNCATED_CSV"
echo "Refined CSV: $REFINED_CSV"
echo "Sentence CSV: $SENTENCE_CSV"
echo "UV cache dir: $UV_CACHE_DIR"

uv run python -m refine.build_train_truncated \
  --train-path "$INPUT_CSV" \
  --sentence-path "$SENTENCE_CSV" \
  --output-path "$TRUNCATED_CSV"

uv run python -m refine.refine_train_v2 \
  --input-path "$TRUNCATED_CSV" \
  --output-path "$REFINED_CSV"
