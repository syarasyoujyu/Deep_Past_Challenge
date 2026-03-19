from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional UX dependency
    tqdm = None


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
    / "akt6a_parallel_lines.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
)
DEFAULT_OUTPUT_FILE = "akt6a_parallel_lines_openai.csv"
DEFAULT_CHECKPOINT_FILE = "akt6a_parallel_lines_openai.jsonl"
DEFAULT_METRICS_FILE = "akt6a_parallel_lines_openai_metrics.csv"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "gpt-5"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_PROMPT_TOKENS_PER_MINUTE = 100_000
DEFAULT_ESTIMATED_INPUT_TOKENS_PER_ROW = 600
DEFAULT_ESTIMATED_OUTPUT_TOKENS_PER_ROW = 500
DEFAULT_ESTIMATED_CACHED_INPUT_TOKENS_PER_ROW = 0
DEFAULT_MAX_WORKERS = 8

MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float, float | None]] = {
    "gpt-5": (1.25, 10.00, 0.125),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    "gpt-5.4": (2.50, 15.00, 0.25),
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
    "gpt-5.4-nano": (0.20, 1.25, 0.02),
    "gpt-5.4-pro": (15.00, 90.00, None),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
}

SYSTEM_INSTRUCTIONS = """You perform OCR for one row from a bilingual Old Assyrian edition table.

The input is one full-row image containing the left transliteration, center line marker, and right translation.

Your job:
- Read the row exactly from the provided row image.
- Return the left transliteration in transliteration.
- Return the right English text in translation.
- Ignore center line markers such as 5, 10, 15, 20, or e.
- Preserve visible Assyriological diacritics and special letters exactly when visible.
- Important examples include corner brackets ˹ and ˺, accented vowels á à é è í ì ú ù, subscript forms a₂ a₃ e₂ e₃ i₂ i₃ u₂ u₃, subscript digits ₀-₉, subscript ₓ, š Š ṣ Ṣ ṭ Ṭ ḫ Ḫ, and glottal ʾ or ‘.
- Brackets, braces, apostrophes, and editorial marks are part of the text when shown and must not be dropped.
- Do not flatten special characters to ASCII when the image shows the marked character.
- If one side is blank, return an empty string for that side.
- Do not borrow text from neighboring rows.
- Do not normalize, reinterpret, or repair uncertain text unless the image clearly supports it.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "transliteration": {"type": "string"},
        "translation": {"type": "string"},
    },
    "required": ["transliteration", "translation"],
}


@dataclass(slots=True)
class LineRow:
    oare_id: str
    doc_label: str
    pdf_page: int
    y_top: str
    y_bottom: str
    x0: str
    x1: str
    marker: str
    transliteration: str
    translation: str
    row_image_path: str
    left_image_path: str
    right_image_path: str


@dataclass(slots=True)
class OcrResult:
    oare_id: str
    transliteration: str
    translation: str
    metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class CostEstimate:
    basis: str
    input_tokens_per_row: float
    cached_input_tokens_per_row: float
    output_tokens_per_row: float
    per_row_cost_usd: float | None
    completed_cost_usd: float
    pending_cost_usd: float | None
    projected_total_cost_usd: float | None


@dataclass(slots=True)
class CompletedRow:
    row: LineRow
    result: OcrResult


class TokenBucket:
    def __init__(self, tokens_per_minute: int, capacity: int | None = None) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        bucket_capacity = max(tokens_per_minute, capacity or tokens_per_minute)
        self.capacity = float(bucket_capacity)
        self.tokens = float(bucket_capacity)
        self.refill_rate = float(tokens_per_minute) / 60.0
        self.last_updated = time.monotonic()
        self.condition = threading.Condition()

    def acquire(self, amount: int) -> None:
        needed = float(max(amount, 1))
        with self.condition:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_updated
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self.last_updated = now
                if self.tokens >= needed:
                    self.tokens -= needed
                    return
                wait_seconds = (needed - self.tokens) / self.refill_rate
                self.condition.wait(timeout=max(wait_seconds, 0.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OCR saved AKT 6a row crop images one line at a time with OpenAI. "
            "Run extract_akt6a_parallel_table.py --save-row-images first."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--checkpoint-file", type=str, default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--metrics-file", type=str, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--start-page", type=int, default=None)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    parser.add_argument("--cached-input-price-per-1m", type=float, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--prompt-tokens-per-minute",
        type=int,
        default=DEFAULT_PROMPT_TOKENS_PER_MINUTE,
        help="Client-side prompt token budget used to throttle concurrent row OCR requests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent OpenAI row OCR requests.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    parser.add_argument(
        "--estimated-input-tokens-per-row",
        type=int,
        default=DEFAULT_ESTIMATED_INPUT_TOKENS_PER_ROW,
        help="Fallback input token estimate per row when no checkpoint metrics exist.",
    )
    parser.add_argument(
        "--estimated-output-tokens-per-row",
        type=int,
        default=DEFAULT_ESTIMATED_OUTPUT_TOKENS_PER_ROW,
        help="Fallback output token estimate per row when no checkpoint metrics exist.",
    )
    parser.add_argument(
        "--estimated-cached-input-tokens-per-row",
        type=int,
        default=DEFAULT_ESTIMATED_CACHED_INPUT_TOKENS_PER_ROW,
        help="Fallback cached input token estimate per row when no checkpoint metrics exist.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir
    return (
        output_dir / args.output_file,
        output_dir / args.checkpoint_file,
        output_dir / args.metrics_file,
    )


def strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> bool:
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = strip_wrapping_quotes(value.strip())
    return True


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_image_as_data_url(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def build_openai_request_payload(row: LineRow, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"OCR this single table row from AKT 6a. "
                f"oare_id={row.oare_id}. "
                "Use the full row image as the source of truth. "
                "Return the left transliteration and right translation exactly as visible in that row. "
                "Ignore the center line marker."
            ),
        }
    ]

    if row.row_image_path:
        content.append(
            {
                "type": "input_image",
                "image_url": read_image_as_data_url(row.row_image_path),
                "detail": "high",
            }
        )

    return {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "akt6a_line_ocr",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    }


def extract_response_text(response_payload: dict[str, Any]) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for output_item in response_payload.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                return text

    raise ValueError("OpenAI response did not contain output text")


def call_openai(
    request_payload: dict[str, Any],
    api_key: str,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        started_at = time.perf_counter()
        request = urllib.request.Request(
            OPENAI_API_URL,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                response_payload = json.load(response)
            response_text = extract_response_text(response_payload)
            return json.loads(response_text), response_payload, time.perf_counter() - started_at
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code == 429 or 500 <= exc.code < 600
            last_error = RuntimeError(f"OpenAI API request failed: {exc.code} {details}")
            if not should_retry or attempt == max_retries:
                raise last_error from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, json.JSONDecodeError) as exc:
            last_error = RuntimeError(f"OpenAI API request failed: {exc}")
            if attempt == max_retries:
                raise last_error from exc

        time.sleep(max(retry_backoff_seconds * attempt, 0.0))

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI API request failed without a captured exception")


def extract_usage_metrics(response_payload: dict[str, Any]) -> dict[str, Any]:
    usage = response_payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    input_details = usage.get("input_tokens_details", {})
    if not isinstance(input_details, dict):
        input_details = {}

    output_details = usage.get("output_tokens_details", {})
    if not isinstance(output_details, dict):
        output_details = {}

    cached_tokens = int(input_details.get("cached_tokens") or 0)
    text_input_tokens = input_details.get("text_tokens")
    image_input_tokens = input_details.get("image_tokens")
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "non_cached_input_tokens": max(input_tokens - cached_tokens, 0),
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "text_input_tokens": int(text_input_tokens) if text_input_tokens is not None else None,
        "image_input_tokens": int(image_input_tokens) if image_input_tokens is not None else None,
        "reasoning_output_tokens": reasoning_tokens,
    }


def resolve_model_pricing(
    model: str,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
) -> tuple[float | None, float | None, float | None]:
    default_prices = MODEL_PRICING_PER_1M_TOKENS.get(model.strip())
    resolved_input = (
        input_price_per_1m
        if input_price_per_1m is not None
        else (default_prices[0] if default_prices is not None else None)
    )
    resolved_output = (
        output_price_per_1m
        if output_price_per_1m is not None
        else (default_prices[1] if default_prices is not None else None)
    )
    resolved_cached = (
        cached_input_price_per_1m
        if cached_input_price_per_1m is not None
        else (default_prices[2] if default_prices is not None else None)
    )
    return resolved_input, resolved_output, resolved_cached


def compute_cost_metrics(
    usage_metrics: dict[str, Any],
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
) -> dict[str, Any]:
    if input_price_per_1m is None or output_price_per_1m is None:
        return {
            "input_cost_usd": None,
            "cached_input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
        }

    non_cached_input_cost = usage_metrics["non_cached_input_tokens"] / 1_000_000 * input_price_per_1m
    cached_input_cost = None
    if cached_input_price_per_1m is not None:
        cached_input_cost = (
            usage_metrics["cached_input_tokens"] / 1_000_000 * cached_input_price_per_1m
        )
    output_cost = usage_metrics["output_tokens"] / 1_000_000 * output_price_per_1m
    total_cost = non_cached_input_cost + output_cost + (cached_input_cost or 0.0)
    estimated_text_input_cost, estimated_image_input_cost = estimate_modality_input_costs(
        usage_metrics,
        input_price_per_1m=input_price_per_1m,
        cached_input_price_per_1m=cached_input_price_per_1m,
    )
    return {
        "input_cost_usd": non_cached_input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "estimated_text_input_cost_usd": estimated_text_input_cost,
        "estimated_image_input_cost_usd": estimated_image_input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def estimate_modality_input_costs(
    usage_metrics: dict[str, Any],
    *,
    input_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
) -> tuple[float | None, float | None]:
    if input_price_per_1m is None:
        return None, None

    text_input_tokens = usage_metrics.get("text_input_tokens")
    image_input_tokens = usage_metrics.get("image_input_tokens")
    if text_input_tokens is None and image_input_tokens is None:
        return None, None

    text_tokens = float(text_input_tokens or 0.0)
    image_tokens = float(image_input_tokens or 0.0)
    total_modal_tokens = text_tokens + image_tokens
    cached_tokens = float(usage_metrics.get("cached_input_tokens") or 0.0)
    cached_rate = cached_input_price_per_1m if cached_input_price_per_1m is not None else input_price_per_1m

    if total_modal_tokens <= 0:
        return 0.0, 0.0

    text_share = text_tokens / total_modal_tokens
    image_share = image_tokens / total_modal_tokens
    text_cached_tokens = cached_tokens * text_share
    image_cached_tokens = cached_tokens * image_share
    text_non_cached_tokens = max(text_tokens - text_cached_tokens, 0.0)
    image_non_cached_tokens = max(image_tokens - image_cached_tokens, 0.0)

    text_cost = (
        text_non_cached_tokens / 1_000_000 * input_price_per_1m
        + text_cached_tokens / 1_000_000 * cached_rate
    )
    image_cost = (
        image_non_cached_tokens / 1_000_000 * input_price_per_1m
        + image_cached_tokens / 1_000_000 * cached_rate
    )
    return text_cost, image_cost


def build_cost_estimate(
    *,
    checkpoint: dict[str, OcrResult],
    completed_rows: int,
    pending_rows: int,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
    estimated_input_tokens_per_row: int,
    estimated_output_tokens_per_row: int,
    estimated_cached_input_tokens_per_row: int,
) -> CostEstimate:
    metrics_list = [
        result.metrics
        for result in checkpoint.values()
        if result.metrics and result.metrics.get("total_cost_usd") is not None
    ]
    completed_cost_usd = sum(
        float(metrics.get("total_cost_usd") or 0.0)
        for metrics in metrics_list
    )

    if metrics_list:
        input_tokens_per_row = sum(
            float(metrics.get("input_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        cached_input_tokens_per_row = sum(
            float(metrics.get("cached_input_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        output_tokens_per_row = sum(
            float(metrics.get("output_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        per_row_cost_usd = completed_cost_usd / len(metrics_list)
        pending_cost_usd = per_row_cost_usd * pending_rows
        projected_total_cost_usd = completed_cost_usd + pending_cost_usd
        return CostEstimate(
            basis="checkpoint_average",
            input_tokens_per_row=input_tokens_per_row,
            cached_input_tokens_per_row=cached_input_tokens_per_row,
            output_tokens_per_row=output_tokens_per_row,
            per_row_cost_usd=per_row_cost_usd,
            completed_cost_usd=completed_cost_usd,
            pending_cost_usd=pending_cost_usd,
            projected_total_cost_usd=projected_total_cost_usd,
        )

    usage_metrics = {
        "input_tokens": max(estimated_input_tokens_per_row, 0),
        "cached_input_tokens": max(estimated_cached_input_tokens_per_row, 0),
        "non_cached_input_tokens": max(
            estimated_input_tokens_per_row - estimated_cached_input_tokens_per_row,
            0,
        ),
        "output_tokens": max(estimated_output_tokens_per_row, 0),
    }
    cost_metrics = compute_cost_metrics(
        usage_metrics,
        input_price_per_1m,
        output_price_per_1m,
        cached_input_price_per_1m,
    )
    per_row_cost_usd = cost_metrics["total_cost_usd"]
    pending_cost_usd = (
        per_row_cost_usd * pending_rows if per_row_cost_usd is not None else None
    )
    projected_total_cost_usd = (
        completed_cost_usd + pending_cost_usd
        if pending_cost_usd is not None
        else None
    )
    return CostEstimate(
        basis="default_estimate",
        input_tokens_per_row=float(estimated_input_tokens_per_row),
        cached_input_tokens_per_row=float(estimated_cached_input_tokens_per_row),
        output_tokens_per_row=float(estimated_output_tokens_per_row),
        per_row_cost_usd=per_row_cost_usd,
        completed_cost_usd=completed_cost_usd,
        pending_cost_usd=pending_cost_usd,
        projected_total_cost_usd=projected_total_cost_usd,
    )


def print_cost_estimate(
    *,
    total_rows: int,
    completed_rows: int,
    pending_rows: int,
    estimate: CostEstimate,
) -> None:
    print(
        "Run summary: "
        f"total_rows={total_rows} "
        f"completed_rows={completed_rows} "
        f"pending_rows={pending_rows} "
        f"estimate_basis={estimate.basis}"
    )
    print(
        "Token estimate per row: "
        f"input~{estimate.input_tokens_per_row:.1f} "
        f"cached_input~{estimate.cached_input_tokens_per_row:.1f} "
        f"output~{estimate.output_tokens_per_row:.1f}"
    )
    if estimate.per_row_cost_usd is not None:
        print(
            "Cost estimate: "
            f"per_row~${estimate.per_row_cost_usd:.6f} "
            f"completed=${estimate.completed_cost_usd:.6f} "
            f"pending~${(estimate.pending_cost_usd or 0.0):.6f} "
            f"projected_total~${(estimate.projected_total_cost_usd or 0.0):.6f}"
        )


def resolve_request_workers(max_workers: int, pending_rows: int) -> int:
    if pending_rows <= 0:
        return 1
    return max(1, min(max_workers, pending_rows))


def process_row(
    row: LineRow,
    *,
    model: str,
    api_key: str,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    token_bucket: TokenBucket,
    estimated_input_tokens_per_row: int,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
    sleep_seconds: float,
) -> CompletedRow:
    if not row.row_image_path:
        raise RuntimeError(
            "row_image_path is empty. Re-run extract_akt6a_parallel_table.py with --save-row-images first."
        )

    token_bucket.acquire(estimated_input_tokens_per_row)
    request_payload = build_openai_request_payload(row, model)
    response_result, response_payload, request_seconds = call_openai(
        request_payload=request_payload,
        api_key=api_key,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    usage_metrics = extract_usage_metrics(response_payload)
    cost_metrics = compute_cost_metrics(
        usage_metrics,
        input_price_per_1m,
        output_price_per_1m,
        cached_input_price_per_1m,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return CompletedRow(
        row=row,
        result=OcrResult(
            oare_id=row.oare_id,
            transliteration=normalize_text(str(response_result.get("transliteration", ""))),
            translation=normalize_text(str(response_result.get("translation", ""))),
            metrics={
                **usage_metrics,
                **cost_metrics,
                "request_seconds": request_seconds,
            },
        ),
    )


def load_line_rows(
    path: Path,
    limit: int | None,
    *,
    start_page: int | None,
    end_page: int | None,
) -> list[LineRow]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [
            LineRow(
                oare_id=str(payload["oare_id"]),
                doc_label=str(payload.get("doc_label", "")),
                pdf_page=int(payload.get("pdf_page") or 0),
                y_top=str(payload.get("y_top", "")),
                y_bottom=str(payload.get("y_bottom", "")),
                x0=str(payload.get("x0", "")),
                x1=str(payload.get("x1", "")),
                marker=str(payload.get("marker", "")),
                transliteration=str(payload.get("transliteration", "")),
                translation=str(payload.get("translation", "")),
                row_image_path=str(payload.get("row_image_path", "")),
                left_image_path=str(payload.get("left_image_path", "")),
                right_image_path=str(payload.get("right_image_path", "")),
            )
            for payload in reader
        ]
    if start_page is not None:
        rows = [row for row in rows if row.pdf_page >= start_page]
    if end_page is not None:
        rows = [row for row in rows if row.pdf_page <= end_page]
    if limit is not None:
        return rows[: max(limit, 0)]
    return rows


def load_checkpoint(path: Path) -> dict[str, OcrResult]:
    if not path.exists():
        return {}

    checkpoint: dict[str, OcrResult] = {}
    with path.open("r", encoding="utf-8") as checkpoint_file:
        for line in checkpoint_file:
            payload = json.loads(line)
            checkpoint[str(payload["oare_id"])] = OcrResult(
                oare_id=str(payload["oare_id"]),
                transliteration=normalize_text(str(payload.get("transliteration", ""))),
                translation=normalize_text(str(payload.get("translation", ""))),
                metrics=payload.get("metrics"),
            )
    return checkpoint


def append_checkpoint(path: Path, result: OcrResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as checkpoint_file:
        checkpoint_file.write(
            json.dumps(
                {
                    "oare_id": result.oare_id,
                    "transliteration": result.transliteration,
                    "translation": result.translation,
                    "metrics": result.metrics or {},
                },
                ensure_ascii=False,
            )
        )
        checkpoint_file.write("\n")


def write_output_csv(path: Path, rows: list[LineRow], results: dict[str, OcrResult]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "oare_id",
                "doc_label",
                "pdf_page",
                "y_top",
                "y_bottom",
                "marker",
                "transliteration",
                "translation",
                "detected_transliteration",
                "detected_translation",
                "row_image_path",
                "left_image_path",
                "right_image_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            result = results.get(row.oare_id)
            if result is None:
                continue
            writer.writerow(
                {
                    "oare_id": row.oare_id,
                    "doc_label": row.doc_label,
                    "pdf_page": row.pdf_page,
                    "y_top": row.y_top,
                    "y_bottom": row.y_bottom,
                    "marker": row.marker,
                    "transliteration": result.transliteration,
                    "translation": result.translation,
                    "detected_transliteration": row.transliteration,
                    "detected_translation": row.translation,
                    "row_image_path": row.row_image_path,
                    "left_image_path": row.left_image_path,
                    "right_image_path": row.right_image_path,
                }
            )
            row_count += 1
    return row_count


def format_float(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_metrics_csv(path: Path, rows: list[LineRow], results: dict[str, OcrResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "oare_id",
                "pdf_page",
                "request_seconds",
                "input_tokens",
                "cached_input_tokens",
                "non_cached_input_tokens",
                "text_input_tokens",
                "image_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
                "input_cost_usd",
                "cached_input_cost_usd",
                "estimated_text_input_cost_usd",
                "estimated_image_input_cost_usd",
                "output_cost_usd",
                "total_cost_usd",
            ],
        )
        writer.writeheader()
        for row in rows:
            result = results.get(row.oare_id)
            metrics = result.metrics if result is not None else None
            if not metrics:
                continue
            writer.writerow(
                {
                    "oare_id": row.oare_id,
                    "pdf_page": row.pdf_page,
                    "request_seconds": format_float(metrics.get("request_seconds"), digits=3),
                    "input_tokens": metrics.get("input_tokens"),
                    "cached_input_tokens": metrics.get("cached_input_tokens"),
                    "non_cached_input_tokens": metrics.get("non_cached_input_tokens"),
                    "text_input_tokens": metrics.get("text_input_tokens"),
                    "image_input_tokens": metrics.get("image_input_tokens"),
                    "output_tokens": metrics.get("output_tokens"),
                    "reasoning_output_tokens": metrics.get("reasoning_output_tokens"),
                    "total_tokens": metrics.get("total_tokens"),
                    "input_cost_usd": format_float(metrics.get("input_cost_usd")),
                    "cached_input_cost_usd": format_float(metrics.get("cached_input_cost_usd")),
                    "estimated_text_input_cost_usd": format_float(metrics.get("estimated_text_input_cost_usd")),
                    "estimated_image_input_cost_usd": format_float(metrics.get("estimated_image_input_cost_usd")),
                    "output_cost_usd": format_float(metrics.get("output_cost_usd")),
                    "total_cost_usd": format_float(metrics.get("total_cost_usd")),
                }
            )


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    output_path, checkpoint_path, metrics_path = resolve_output_paths(args)
    model = (args.model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    input_price_per_1m, output_price_per_1m, cached_input_price_per_1m = resolve_model_pricing(
        model,
        args.input_price_per_1m,
        args.output_price_per_1m,
        args.cached_input_price_per_1m,
    )

    rows = load_line_rows(
        args.input_path,
        args.limit,
        start_page=args.start_page,
        end_page=args.end_page,
    )
    checkpoint = {} if args.overwrite else load_checkpoint(checkpoint_path)
    completed_rows = sum(1 for row in rows if row.oare_id in checkpoint)
    pending_rows = len(rows) - completed_rows
    estimate = build_cost_estimate(
        checkpoint=checkpoint,
        completed_rows=completed_rows,
        pending_rows=pending_rows,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        cached_input_price_per_1m=cached_input_price_per_1m,
        estimated_input_tokens_per_row=args.estimated_input_tokens_per_row,
        estimated_output_tokens_per_row=args.estimated_output_tokens_per_row,
        estimated_cached_input_tokens_per_row=args.estimated_cached_input_tokens_per_row,
    )
    if args.dry_run:
        print(f"Would OCR {len(rows)} line images from {args.input_path} using model {model}")
        print(f"Output: {output_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Metrics: {metrics_path}")
        print_cost_estimate(
            total_rows=len(rows),
            completed_rows=completed_rows,
            pending_rows=pending_rows,
            estimate=estimate,
        )
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} or place it in {args.env_file} before calling OpenAI."
        )

    print_cost_estimate(
        total_rows=len(rows),
        completed_rows=completed_rows,
        pending_rows=pending_rows,
        estimate=estimate,
    )
    request_workers = resolve_request_workers(args.max_workers, pending_rows)
    print(
        "Parallel config: "
        f"request_workers={request_workers} "
        f"prompt_tokens_per_minute={args.prompt_tokens_per_minute} "
        f"estimated_input_tokens_per_row={args.estimated_input_tokens_per_row}"
    )

    results = dict(checkpoint)
    cumulative_cost = sum(
        float(result.metrics.get("total_cost_usd") or 0.0)
        for result in results.values()
        if result.metrics
    )
    pending_line_rows = [row for row in rows if row.oare_id not in results]
    token_bucket = TokenBucket(
        args.prompt_tokens_per_minute,
        capacity=max(
            args.prompt_tokens_per_minute,
            args.estimated_input_tokens_per_row * max(request_workers, 1),
        ),
    )
    progress_bar = None
    if tqdm is not None and not args.no_progress:
        progress_bar = tqdm(
            total=len(rows),
            initial=completed_rows,
            desc="AKT6a line OCR",
            unit="row",
        )

    try:
        if pending_line_rows:
            future_to_position: dict[Future[CompletedRow], int] = {}
            with ThreadPoolExecutor(max_workers=request_workers) as executor:
                for position, row in enumerate(pending_line_rows, start=completed_rows + 1):
                    future = executor.submit(
                        process_row,
                        row,
                        model=model,
                        api_key=api_key,
                        request_timeout_seconds=args.request_timeout_seconds,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        token_bucket=token_bucket,
                        estimated_input_tokens_per_row=args.estimated_input_tokens_per_row,
                        input_price_per_1m=input_price_per_1m,
                        output_price_per_1m=output_price_per_1m,
                        cached_input_price_per_1m=cached_input_price_per_1m,
                        sleep_seconds=args.sleep_seconds,
                    )
                    future_to_position[future] = position

                for future in as_completed(future_to_position):
                    completed_row = future.result()
                    row = completed_row.row
                    result = completed_row.result
                    results[row.oare_id] = result
                    append_checkpoint(checkpoint_path, result)
                    metrics = result.metrics or {}
                    row_cost = float(metrics.get("total_cost_usd") or 0.0)
                    cumulative_cost += row_cost
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(
                            f"page={row.pdf_page} cost=${row_cost:.6f} cumulative=${cumulative_cost:.6f}"
                        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    row_count = write_output_csv(output_path, rows, results)
    write_metrics_csv(metrics_path, rows, results)
    print(f"Wrote {row_count} OCR rows to {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
