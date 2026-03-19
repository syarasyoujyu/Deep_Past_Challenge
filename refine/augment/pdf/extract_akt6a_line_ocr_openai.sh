
#============a
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6a.pdf"
OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a"
LINE_OUTPUT_FILE="akt6b_parallel_lines.csv"

uv run --with pdfplumber --with pypdfium2 --with pillow python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 52 \
  --end-page 436 \
  --save-row-images \
  --input-path "$INPUT_PDF" \
  --output-dir "$OUTPUT_DIR" \
  --line-output-file "$LINE_OUTPUT_FILE"

uv run --with tqdm python refine/augment/pdf/extract_akt6a_line_ocr_openai.py \
  --overwrite \
  --prompt-tokens-per-minute 100000 \
  --max-workers 20 \
  --start-page 52 \
  --end-page 436 \
  --input-path "$OUTPUT_DIR/$LINE_OUTPUT_FILE" \
  --output-dir "$OUTPUT_DIR"


#============b
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6b.pdf"
OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6b"
LINE_OUTPUT_FILE="akt6b_parallel_lines.csv"

uv run --with pdfplumber --with pypdfium2 --with pillow python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 59 \
  --end-page 352 \
  --save-row-images \
  --input-path "$INPUT_PDF" \
  --output-dir "$OUTPUT_DIR" \
  --line-output-file "$LINE_OUTPUT_FILE"

uv run --with tqdm python refine/augment/pdf/extract_akt6a_line_ocr_openai.py \
  --overwrite \
  --prompt-tokens-per-minute 100000 \
  --max-workers 20 \
  --start-page 59 \
  --end-page 352 \
  --input-path "$OUTPUT_DIR/$LINE_OUTPUT_FILE" \
  --output-dir "$OUTPUT_DIR"

#============c
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6c.pdf"
OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6c"
LINE_OUTPUT_FILE="akt6b_parallel_lines.csv"

uv run --with pdfplumber --with pypdfium2 --with pillow python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 31 \
  --end-page 282 \
  --save-row-images \
  --input-path "$INPUT_PDF" \
  --output-dir "$OUTPUT_DIR" \
  --line-output-file "$LINE_OUTPUT_FILE"

uv run --with tqdm python refine/augment/pdf/extract_akt6a_line_ocr_openai.py \
  --overwrite \
  --prompt-tokens-per-minute 100000 \
  --max-workers 20 \
  --start-page 31 \
  --end-page 282 \
  --input-path "$OUTPUT_DIR/$LINE_OUTPUT_FILE" \
  --output-dir "$OUTPUT_DIR"


#============d
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6d.pdf"
OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6d"
LINE_OUTPUT_FILE="akt6b_parallel_lines.csv"

uv run --with pdfplumber --with pypdfium2 --with pillow python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 11 \
  --end-page 132 \
  --save-row-images \
  --input-path "$INPUT_PDF" \
  --output-dir "$OUTPUT_DIR" \
  --line-output-file "$LINE_OUTPUT_FILE"

uv run --with tqdm python refine/augment/pdf/extract_akt6a_line_ocr_openai.py \
  --overwrite \
  --prompt-tokens-per-minute 100000 \
  --max-workers 20 \
  --start-page 11 \
  --end-page 132 \
  --input-path "$OUTPUT_DIR/$LINE_OUTPUT_FILE" \
  --output-dir "$OUTPUT_DIR"


#============e
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

INPUT_PDF="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6e.pdf"
OUTPUT_DIR="data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6e"
LINE_OUTPUT_FILE="akt6b_parallel_lines.csv"

uv run --with pdfplumber --with pypdfium2 --with pillow python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 85 \
  --end-page 259 \
  --save-row-images \
  --input-path "$INPUT_PDF" \
  --output-dir "$OUTPUT_DIR" \
  --line-output-file "$LINE_OUTPUT_FILE"

uv run --with tqdm python refine/augment/pdf/extract_akt6a_line_ocr_openai.py \
  --overwrite \
  --prompt-tokens-per-minute 100000 \
  --max-workers 20 \
  --start-page 85 \
  --end-page 259 \
  --input-path "$OUTPUT_DIR/$LINE_OUTPUT_FILE" \
  --output-dir "$OUTPUT_DIR"
