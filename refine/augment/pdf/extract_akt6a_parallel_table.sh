#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/deep-past-uv-cache}"

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

cd "$ROOT_DIR"

uv run --with pdfplumber python refine/augment/pdf/extract_akt6a_parallel_table.py "$@"
