#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

EDITION="${1:-a}"
if [[ $# -gt 0 ]]; then
  shift
fi

run_edition() {
  local edition="$1"
  shift

  case "$edition" in
    a|6a|AKT_6a|AKT6a)
      INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6a.pdf"
      OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a"
      START_PAGE=52
      END_PAGE=436
      OUTPUT_FILE="akt6a_parallel_openai.csv"
      CHECKPOINT_FILE="akt6a_parallel_openai_pages.jsonl"
      METRICS_FILE="akt6a_parallel_openai_metrics.csv"
      ;;
    b|6b|AKT_6b|AKT6b)
      INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6b.pdf"
      OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6b"
      START_PAGE=59
      END_PAGE=352
      OUTPUT_FILE="akt6b_parallel_openai.csv"
      CHECKPOINT_FILE="akt6b_parallel_openai_pages.jsonl"
      METRICS_FILE="akt6b_parallel_openai_metrics.csv"
      ;;
    c|6c|AKT_6c|AKT6c)
      INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6c.pdf"
      OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6c"
      START_PAGE=31
      END_PAGE=282
      OUTPUT_FILE="akt6c_parallel_openai.csv"
      CHECKPOINT_FILE="akt6c_parallel_openai_pages.jsonl"
      METRICS_FILE="akt6c_parallel_openai_metrics.csv"
      ;;
    d|6d|AKT_6d|AKT6d)
      INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6d.pdf"
      OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6d"
      START_PAGE=11
      END_PAGE=132
      OUTPUT_FILE="akt6d_parallel_openai.csv"
      CHECKPOINT_FILE="akt6d_parallel_openai_pages.jsonl"
      METRICS_FILE="akt6d_parallel_openai_metrics.csv"
      ;;
    e|6e|AKT_6e|AKT6e)
      INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6e.pdf"
      OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6e"
      START_PAGE=85
      END_PAGE=259
      OUTPUT_FILE="akt6e_parallel_openai.csv"
      CHECKPOINT_FILE="akt6e_parallel_openai_pages.jsonl"
      METRICS_FILE="akt6e_parallel_openai_metrics.csv"
      ;;
    *)
      echo "Unknown edition: $edition" >&2
      echo "Usage: bash refine/augment/pdf/extract_akt6a_parallel_table_openai.sh [a|b|c|d|e|all] [extra args...]" >&2
      exit 1
      ;;
  esac

  echo "Running AKT 6${edition#6}..."

  uv run --with pypdfium2 --with pillow --with tqdm python refine/augment/pdf/extract_akt6a_parallel_table_openai.py \
    --input-path "$INPUT_PDF" \
    --output-dir "$OUTPUT_DIR" \
    --output-file "$OUTPUT_FILE" \
    --checkpoint-file "$CHECKPOINT_FILE" \
    --metrics-file "$METRICS_FILE" \
    --start-page "$START_PAGE" \
    --end-page "$END_PAGE" \
    --render-scale 3.0 \
    --image-padding-px 96 \
    --overwrite \
    --max-workers 5 \
    "$@"
}

if [[ "$EDITION" == "all" ]]; then
  for edition in a b c d e; do
    run_edition "$edition" "$@"
  done
else
  run_edition "$EDITION" "$@"
fi
