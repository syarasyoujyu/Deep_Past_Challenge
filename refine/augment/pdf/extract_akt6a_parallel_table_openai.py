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
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import pypdfium2 as pdfium
    from PIL import ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local CLI use
    raise SystemExit(
        "pypdfium2 is required. Run this script with "
        "`uv run --with pypdfium2 pillow python "
        "refine/augment/pdf/extract_akt6a_parallel_table_openai.py`."
    ) from exc

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "AKT 6a.pdf"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
)
DEFAULT_OUTPUT_FILE = "akt6a_parallel_openai.csv"
DEFAULT_CHECKPOINT_FILE = "akt6a_parallel_openai_pages.jsonl"
DEFAULT_METRICS_FILE = "akt6a_parallel_openai_metrics.csv"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_START_PAGE = 52
DEFAULT_END_PAGE = None
DEFAULT_RENDER_SCALE = 2.4
DEFAULT_IMAGE_PADDING_PX = 64
DEFAULT_MODEL = "gpt-5.1"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_PROMPT_TOKENS_PER_MINUTE = 100_000
DEFAULT_ESTIMATED_INPUT_TOKENS_PER_PAGE = 12_000
DEFAULT_PREPROCESS_WORKERS = 4

MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float, float | None]] = {
    "gpt-5": (1.25, 10.00, 0.125),
    "gpt-5.1": (1.25, 10.00, 0.125),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    "gpt-5.4": (2.50, 15.00, 0.25),
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
    "gpt-5.4-nano": (0.20, 1.25, 0.02),
    "gpt-5.4-pro": (15.00, 90.00, None),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
}

MIDDLE_MARKER_ONLY_RE = re.compile(
    r"^(?:\d{1,3}|(?:\d{1,3}\s*)?[1Il]\.?\s*e\.?|(?:\d{1,3}\s*)?e\.?)$",
    re.IGNORECASE,
)

SYSTEM_INSTRUCTIONS = """You extract visual table rows from a scholarly Old Assyrian edition page.

The page may contain:
- a centered text label like "1. kt 94/k 1263"
- a left Akkadian transliteration column
- a narrow middle column with line numbers such as 5, 10, 15, 20, or e.
- a right English translation column
- Note: and Comment: sections that must be ignored
- more than one separate bilingual table block on the same page

Layout notes:
- The transliteration line may be visually indented and can extend toward the middle of the page.
- Some transliteration lines start near the center-left rather than the far left margin.
- Text that sits just left of the line-number column can still belong to the transliteration column.
- Do not drop or misclassify a transliteration line merely because it is short, indented, or horizontally centered within the left half.
- The translation line may also be indented or wrapped and can start near the center-right rather than the far right margin.
- Text that sits just right of the line-number column can still belong to the translation column.
- Wrapped continuation lines in either column may appear near the middle of that half-page rather than hugging the outer margin.
- Valid table rows can begin very near the top edge of the page or continue to the very bottom edge.
- Inspect the full image boundary, including the first visible row near the top margin and the last visible row near the bottom margin.

Your job:
- Read the page image carefully.
- Extract all visible bilingual table rows on the page in top-to-bottom order.
- Ignore page numbers, line numbers, running headers, footers, and commentary prose.
- Middle-column markers such as 5, 10, 15, 20, 40 e., 1.e., l.e., and e. are not transliteration or translation.
- Never return those middle-column markers as transliteration text or translation text.
- Ignore Note:/Comment: prose itself, but resume extraction if another bilingual table block appears later on the same page.
- Do not omit a row merely because it is close to the page edge.
- Do not use a strict left-margin or right-margin rule. Classify by the printed row and by language/script cues, not only by x-position.
- Preserve transliteration characters as accurately as possible from the image.
- Preserve English translation wording as accurately as possible from the image.
- Preserve Assyriological diacritics and special letters exactly when visible.
- Important examples include corner brackets ˹ and ˺, accented vowels á à é è í ì ú ù, subscript forms a₂ a₃ e₂ e₃ i₂ i₃ u₂ u₃, subscript digits ₀-₉, subscript ₓ, š Š ṣ Ṣ ṭ Ṭ ḫ Ḫ, and glottal ʾ or ‘.
- Brackets, braces, apostrophes, and editorial marks are part of the text when shown and must not be dropped.
- Editorial gap phrases are also visible text and must be preserved when printed, for example: break, break of two or three lines, ..., and similar wording.
- If such a phrase is printed on the left side, put it in transliteration; if it is printed on the right side, put it in translation.
- Do not flatten special characters to plain ASCII when the page shows the marked character.
- For example, if the page shows ša-al-ṭi, do not return ša-al-ti.
- When left-side text is indented or centered within the left half, still return it as transliteration if it is Akkadian/Assyriological text rather than English prose.
- When right-side text is indented or centered within the right half, still return it as translation if it is English text rather than Akkadian transliteration.
- Return one output row per visible table line in reading order from top to bottom.
- A row may have text on both sides, only on the transliteration side, or only on the translation side.
- If one side is visually blank for that line, return an empty string for that side.
- Do not borrow text from the line above or below to force alignment.
- Do not merge multiple physical lines into one row unless they are printed on the same visual line.
- Do not normalize, reinterpret, or repair uncertain text unless the page clearly supports it.
- If a page has no bilingual table rows, return has_parallel_table=false and rows=[].
- If a document label is visible on the page, return it in doc_label. Otherwise return an empty string.
"""

FALLBACK_SYSTEM_INSTRUCTIONS = """You extract visual table rows from a scholarly Old Assyrian edition page.

This is a fallback pass for a page that may have only a few visible table rows.

Your job:
- Assume the page may still contain valid bilingual table content even if it is sparse.
- Extract any visible table rows on the page, even if there are only one or two rows.
- Continuation pages are valid; a page does not need a full dense table to count.
- Ignore page numbers, line numbers, running headers, footers, and commentary prose.
- Middle-column markers such as 5, 10, 15, 20, 40 e., 1.e., l.e., and e. must be ignored, not returned as row text.
- Ignore Note:/Comment: prose itself, but if another table block appears later on the same page, keep extracting rows after it.
- Rows near the top edge or bottom edge of the page are valid and must still be extracted.
- The transliteration line may be indented and can extend close to the middle line-number column; it still belongs to the left column.
- The translation line may be indented or wrapped and can extend close to the middle line-number column; it still belongs to the right column.
- Preserve transliteration characters and English translation wording exactly as shown.
- Preserve Assyriological diacritics and special letters exactly when visible, including ˹ ˺, accented vowels, subscript forms, š Š ṣ Ṣ ṭ Ṭ ḫ Ḫ, and ʾ or ‘.
- Preserve editorial gap phrases exactly when they are printed, including break, break of two or three lines, and ....
- Put the phrase on the side where it is visually printed; do not drop it just because it is not a lexical word.
- Do not discard a left-side line merely because it begins near the page center or sits just to the left of the line numbers.
- Do not discard a right-side line merely because it begins near the page center or sits just to the right of the line numbers.
- Return one output row per visible table line in reading order from top to bottom.
- A row may have text on both sides, only on the transliteration side, or only on the translation side.
- If one side is visually blank for that line, return an empty string for that side.
- Do not borrow text from neighboring lines to force alignment.
- If truly no table rows are visible, return has_parallel_table=false and rows=[].
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_number": {"type": "integer"},
        "doc_label": {"type": "string"},
        "has_parallel_table": {"type": "boolean"},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "transliteration": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["index", "transliteration", "translation"],
            },
        },
    },
    "required": ["page_number", "doc_label", "has_parallel_table", "rows"],
}


@dataclass(slots=True)
class PageExtraction:
    page_number: int
    doc_label: str
    has_parallel_table: bool
    rows: list[dict[str, str]]
    metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class PreparedPage:
    page_number: int
    image_data_url: str
    preprocess_metrics: dict[str, Any]
    preprocess_started_at: float
    preprocess_finished_at: float


@dataclass(slots=True)
class CompletedPage:
    prepared_page: PreparedPage
    extraction: PageExtraction
    response_payload: dict[str, Any]
    usage_metrics: dict[str, Any]
    request_seconds: float
    request_started_at: float
    request_finished_at: float


@dataclass(slots=True)
class CostEstimate:
    basis: str
    input_tokens_per_page: float
    cached_input_tokens_per_page: float
    output_tokens_per_page: float
    per_page_cost_usd: float | None
    completed_cost_usd: float
    pending_cost_usd: float | None
    projected_total_cost_usd: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render AKT 6a PDF pages to images and use OpenAI Vision to extract high-quality "
            "transliteration/translation row pairs."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--checkpoint-file", type=str, default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--metrics-file", type=str, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE)
    parser.add_argument("--end-page", type=int, default=DEFAULT_END_PAGE)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument(
        "--prompt-tokens-per-minute",
        type=int,
        default=DEFAULT_PROMPT_TOKENS_PER_MINUTE,
        help="Client-side prompt token budget used to throttle concurrent page requests.",
    )
    parser.add_argument(
        "--estimated-input-tokens-per-page",
        type=int,
        default=None,
        help="Estimated input tokens per page used for token-bucket scheduling.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent OpenAI requests. If omitted, derive from token budget.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=DEFAULT_PREPROCESS_WORKERS,
        help="Maximum concurrent PDF page render workers.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Override standard input token price in USD per 1M tokens.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Override standard output token price in USD per 1M tokens.",
    )
    parser.add_argument(
        "--cached-input-price-per-1m",
        type=float,
        default=None,
        help="Override cached input token price in USD per 1M tokens.",
    )
    parser.add_argument("--render-scale", type=float, default=DEFAULT_RENDER_SCALE)
    parser.add_argument(
        "--image-padding-px",
        type=int,
        default=DEFAULT_IMAGE_PADDING_PX,
        help="White border added around the rendered page image to make edge rows easier to read.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Sleep between page requests to keep throughput conservative.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on number of pages processed after start-page.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore checkpoint and reprocess pages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render pages and print planned work without calling OpenAI.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
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


def is_middle_marker_only(text: str) -> bool:
    return bool(MIDDLE_MARKER_ONLY_RE.fullmatch(normalize_text(text)))


def build_oare_id(doc_label: str, page_number: int, row_index: int) -> str:
    seed = f"akt6a_openai|{normalize_text(doc_label)}|{page_number}|{row_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


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


def render_page_to_data_url(
    input_path: Path,
    page_number: int,
    scale: float,
    image_padding_px: int,
    ) -> PreparedPage:
    started_at = time.perf_counter()
    with pdfium.PdfDocument(input_path) as document:
        page = document[page_number - 1]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        if image_padding_px > 0:
            pil_image = ImageOps.expand(pil_image, border=image_padding_px, fill="white")
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    finished_at = time.perf_counter()
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return PreparedPage(
        page_number=page_number,
        image_data_url=f"data:image/png;base64,{encoded}",
        preprocess_metrics={
            "preprocess_seconds": finished_at - started_at,
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            "image_bytes": len(png_bytes),
        },
        preprocess_started_at=started_at,
        preprocess_finished_at=finished_at,
    )


def build_openai_request_payload(
    page_number: int,
    image_data_url: str,
    model: str,
    *,
    fallback: bool = False,
) -> dict[str, Any]:
    page_prompt = (
        f"Extract bilingual table rows from PDF page {page_number}. "
        "Return one row per visible table line. "
        "If a line has transliteration but no matching translation on that same printed line, keep translation as an empty string. "
        "If a line has translation but no transliteration on that same printed line, keep transliteration as an empty string. "
        "Do not force alignment across neighboring lines. "
        "Do not include Note or Comment sections. "
        "Keep transliteration and translation exactly as read from the page image, including diacritics such as ṭ."
    )
    if fallback:
        page_prompt = (
            f"Fallback extraction for PDF page {page_number}. "
            "This page may contain only a few continuation rows. "
            "Extract any visible bilingual table rows before Note or Comment. "
            "Do not reject the page just because the table is sparse. "
            "Keep one row per visible printed line."
        )
    return {
        "model": model,
        "instructions": FALLBACK_SYSTEM_INSTRUCTIONS if fallback else SYSTEM_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": page_prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "akt6a_parallel_page",
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


def process_prepared_page(
    prepared_page: PreparedPage,
    *,
    model: str,
    api_key: str,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    token_bucket: TokenBucket,
    estimated_input_tokens: int,
) -> CompletedPage:
    token_bucket.acquire(estimated_input_tokens)
    request_started_at = time.perf_counter()
    request_payload = build_openai_request_payload(
        prepared_page.page_number,
        prepared_page.image_data_url,
        model,
    )
    result, response_payload, request_seconds = call_openai(
        request_payload=request_payload,
        api_key=api_key,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    request_finished_at = time.perf_counter()
    extraction = normalize_page_result(result, prepared_page.page_number)
    usage_metrics = extract_usage_metrics(response_payload)
    if not extraction.rows:
        fallback_request_payload = build_openai_request_payload(
            prepared_page.page_number,
            prepared_page.image_data_url,
            model,
            fallback=True,
        )
        fallback_result, fallback_response_payload, fallback_request_seconds = call_openai(
            request_payload=fallback_request_payload,
            api_key=api_key,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        fallback_extraction = normalize_page_result(fallback_result, prepared_page.page_number)
        usage_metrics = merge_usage_metrics(
            usage_metrics,
            extract_usage_metrics(fallback_response_payload),
        )
        request_seconds += fallback_request_seconds
        request_finished_at = time.perf_counter()
        if fallback_extraction.rows:
            extraction = fallback_extraction
            response_payload = fallback_response_payload
    return CompletedPage(
        prepared_page=prepared_page,
        extraction=extraction,
        response_payload=response_payload,
        usage_metrics=usage_metrics,
        request_seconds=request_seconds,
        request_started_at=request_started_at,
        request_finished_at=request_finished_at,
    )


def load_checkpoint(path: Path) -> dict[int, PageExtraction]:
    if not path.exists():
        return {}

    checkpoint: dict[int, PageExtraction] = {}
    with path.open("r", encoding="utf-8") as checkpoint_file:
        for line in checkpoint_file:
            payload = json.loads(line)
            extraction = PageExtraction(
                page_number=int(payload["page_number"]),
                doc_label=str(payload.get("doc_label", "")),
                has_parallel_table=bool(payload.get("has_parallel_table", False)),
                rows=[
                    {
                        "transliteration": normalize_text(str(row.get("transliteration", ""))),
                        "translation": normalize_text(str(row.get("translation", ""))),
                    }
                    for row in payload.get("rows", [])
                ],
                metrics=payload.get("metrics"),
            )
            checkpoint[extraction.page_number] = extraction
    return checkpoint


def append_checkpoint(path: Path, extraction: PageExtraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as checkpoint_file:
        checkpoint_file.write(
            json.dumps(
                {
                    "page_number": extraction.page_number,
                    "doc_label": extraction.doc_label,
                    "has_parallel_table": extraction.has_parallel_table,
                    "rows": extraction.rows,
                    "metrics": extraction.metrics or {},
                },
                ensure_ascii=False,
            )
        )
        checkpoint_file.write("\n")


def normalize_page_result(result: dict[str, Any], page_number: int) -> PageExtraction:
    rows_payload = result.get("rows", [])
    rows: list[dict[str, str]] = []
    for row in rows_payload:
        transliteration = normalize_text(str(row.get("transliteration", "")))
        translation = normalize_text(str(row.get("translation", "")))
        if is_middle_marker_only(transliteration):
            transliteration = ""
        if is_middle_marker_only(translation):
            translation = ""
        if not transliteration and not translation:
            continue
        rows.append(
            {
                "transliteration": transliteration,
                "translation": translation,
            }
        )

    return PageExtraction(
        page_number=page_number,
        doc_label=normalize_text(str(result.get("doc_label", ""))),
        has_parallel_table=bool(result.get("has_parallel_table", False)) and bool(rows),
        rows=rows,
    )


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


def merge_usage_metrics(*usage_items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "non_cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "text_input_tokens": 0,
        "image_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    has_text_input_tokens = False
    has_image_input_tokens = False
    for usage in usage_items:
        merged["input_tokens"] += int(usage.get("input_tokens") or 0)
        merged["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
        merged["non_cached_input_tokens"] += int(usage.get("non_cached_input_tokens") or 0)
        merged["output_tokens"] += int(usage.get("output_tokens") or 0)
        merged["total_tokens"] += int(usage.get("total_tokens") or 0)
        merged["reasoning_output_tokens"] += int(usage.get("reasoning_output_tokens") or 0)
        if usage.get("text_input_tokens") is not None:
            merged["text_input_tokens"] += int(usage.get("text_input_tokens") or 0)
            has_text_input_tokens = True
        if usage.get("image_input_tokens") is not None:
            merged["image_input_tokens"] += int(usage.get("image_input_tokens") or 0)
            has_image_input_tokens = True
    if not has_text_input_tokens:
        merged["text_input_tokens"] = None
    if not has_image_input_tokens:
        merged["image_input_tokens"] = None
    return merged


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
    checkpoint: dict[int, PageExtraction],
    completed_pages: int,
    pending_pages: int,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
    estimated_input_tokens_per_page: int,
) -> CostEstimate:
    metrics_list = [
        page.metrics
        for page in checkpoint.values()
        if page.metrics and page.metrics.get("total_cost_usd") is not None
    ]
    completed_cost_usd = sum(
        float(metrics.get("total_cost_usd") or 0.0)
        for metrics in metrics_list
    )

    if metrics_list:
        input_tokens_per_page = sum(
            float(metrics.get("input_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        cached_input_tokens_per_page = sum(
            float(metrics.get("cached_input_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        output_tokens_per_page = sum(
            float(metrics.get("output_tokens") or 0.0) for metrics in metrics_list
        ) / len(metrics_list)
        per_page_cost_usd = completed_cost_usd / len(metrics_list)
        pending_cost_usd = per_page_cost_usd * pending_pages
        projected_total_cost_usd = completed_cost_usd + pending_cost_usd
        return CostEstimate(
            basis="checkpoint_average",
            input_tokens_per_page=input_tokens_per_page,
            cached_input_tokens_per_page=cached_input_tokens_per_page,
            output_tokens_per_page=output_tokens_per_page,
            per_page_cost_usd=per_page_cost_usd,
            completed_cost_usd=completed_cost_usd,
            pending_cost_usd=pending_cost_usd,
            projected_total_cost_usd=projected_total_cost_usd,
        )

    usage_metrics = {
        "input_tokens": max(estimated_input_tokens_per_page, 0),
        "cached_input_tokens": 0,
        "non_cached_input_tokens": max(estimated_input_tokens_per_page, 0),
        "output_tokens": 800,
    }
    cost_metrics = compute_cost_metrics(
        usage_metrics,
        input_price_per_1m,
        output_price_per_1m,
        cached_input_price_per_1m,
    )
    per_page_cost_usd = cost_metrics["total_cost_usd"]
    pending_cost_usd = (
        per_page_cost_usd * pending_pages if per_page_cost_usd is not None else None
    )
    projected_total_cost_usd = (
        completed_cost_usd + pending_cost_usd
        if pending_cost_usd is not None
        else None
    )
    return CostEstimate(
        basis="default_estimate",
        input_tokens_per_page=float(estimated_input_tokens_per_page),
        cached_input_tokens_per_page=0.0,
        output_tokens_per_page=800.0,
        per_page_cost_usd=per_page_cost_usd,
        completed_cost_usd=completed_cost_usd,
        pending_cost_usd=pending_cost_usd,
        projected_total_cost_usd=projected_total_cost_usd,
    )


def print_cost_estimate(
    *,
    total_pages: int,
    completed_pages: int,
    pending_pages: int,
    estimate: CostEstimate,
) -> None:
    print(
        "Run summary: "
        f"total_pages={total_pages} "
        f"completed_pages={completed_pages} "
        f"pending_pages={pending_pages} "
        f"estimate_basis={estimate.basis}"
    )
    print(
        "Token estimate per page: "
        f"input~{estimate.input_tokens_per_page:.1f} "
        f"cached_input~{estimate.cached_input_tokens_per_page:.1f} "
        f"output~{estimate.output_tokens_per_page:.1f}"
    )
    if estimate.per_page_cost_usd is not None:
        print(
            "Cost estimate: "
            f"per_page~${estimate.per_page_cost_usd:.6f} "
            f"completed=${estimate.completed_cost_usd:.6f} "
            f"pending~${(estimate.pending_cost_usd or 0.0):.6f} "
            f"projected_total~${(estimate.projected_total_cost_usd or 0.0):.6f}"
        )


def resolve_estimated_input_tokens_per_page(
    args: argparse.Namespace,
    checkpoint: dict[int, PageExtraction],
) -> int:
    if args.estimated_input_tokens_per_page is not None:
        return max(args.estimated_input_tokens_per_page, 1)

    observed_values = [
        int(page.metrics.get("input_tokens") or 0)
        for page in checkpoint.values()
        if page.metrics and page.metrics.get("input_tokens")
    ]
    if observed_values:
        average = sum(observed_values) / len(observed_values)
        return max(int(round(average * 1.1)), 1)

    return DEFAULT_ESTIMATED_INPUT_TOKENS_PER_PAGE


def resolve_request_workers(
    args: argparse.Namespace,
    pending_page_count: int,
    estimated_input_tokens_per_page: int,
) -> int:
    if pending_page_count <= 0:
        return 1
    if args.max_workers is not None:
        return max(1, min(args.max_workers, pending_page_count))
    derived_workers = max(1, args.prompt_tokens_per_minute // max(estimated_input_tokens_per_page, 1))
    return max(1, min(derived_workers, pending_page_count))


def resolve_page_numbers(args: argparse.Namespace, page_count: int) -> list[int]:
    start_page = max(args.start_page, 1)
    end_page = args.end_page or page_count
    end_page = min(end_page, page_count)
    page_numbers = list(range(start_page, end_page + 1))
    if args.max_pages is not None:
        page_numbers = page_numbers[: max(args.max_pages, 0)]
    return page_numbers


def write_output_csv(path: Path, pages: list[PageExtraction]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    current_doc_label = ""

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["oare_id", "doc_label", "pdf_page", "transliteration", "translation"],
        )
        writer.writeheader()
        for extraction in sorted(pages, key=lambda item: item.page_number):
            if extraction.doc_label:
                current_doc_label = extraction.doc_label
            doc_label = extraction.doc_label or current_doc_label
            for index, row in enumerate(extraction.rows, start=1):
                writer.writerow(
                    {
                        "oare_id": build_oare_id(doc_label, extraction.page_number, index),
                        "doc_label": doc_label,
                        "pdf_page": extraction.page_number,
                        "transliteration": row["transliteration"],
                        "translation": row["translation"],
                    }
                )
                row_count += 1

    return row_count


def write_metrics_csv(path: Path, pages: list[PageExtraction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "page_number",
                "doc_label",
                "has_parallel_table",
                "row_count",
                "image_width",
                "image_height",
                "image_bytes",
                "preprocess_seconds",
                "queue_wait_seconds",
                "request_seconds",
                "total_seconds",
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
        for extraction in sorted(pages, key=lambda item: item.page_number):
            metrics = extraction.metrics or {}
            writer.writerow(
                {
                    "page_number": extraction.page_number,
                    "doc_label": extraction.doc_label,
                    "has_parallel_table": extraction.has_parallel_table,
                    "row_count": len(extraction.rows),
                    "image_width": metrics.get("image_width"),
                    "image_height": metrics.get("image_height"),
                    "image_bytes": metrics.get("image_bytes"),
                    "preprocess_seconds": format_float(metrics.get("preprocess_seconds")),
                    "queue_wait_seconds": format_float(metrics.get("queue_wait_seconds")),
                    "request_seconds": format_float(metrics.get("request_seconds")),
                    "total_seconds": format_float(metrics.get("total_seconds")),
                    "input_tokens": metrics.get("input_tokens"),
                    "cached_input_tokens": metrics.get("cached_input_tokens"),
                    "non_cached_input_tokens": metrics.get("non_cached_input_tokens"),
                    "text_input_tokens": metrics.get("text_input_tokens"),
                    "image_input_tokens": metrics.get("image_input_tokens"),
                    "output_tokens": metrics.get("output_tokens"),
                    "reasoning_output_tokens": metrics.get("reasoning_output_tokens"),
                    "total_tokens": metrics.get("total_tokens"),
                    "input_cost_usd": format_float(metrics.get("input_cost_usd"), digits=6),
                    "cached_input_cost_usd": format_float(metrics.get("cached_input_cost_usd"), digits=6),
                    "estimated_text_input_cost_usd": format_float(metrics.get("estimated_text_input_cost_usd"), digits=6),
                    "estimated_image_input_cost_usd": format_float(metrics.get("estimated_image_input_cost_usd"), digits=6),
                    "output_cost_usd": format_float(metrics.get("output_cost_usd"), digits=6),
                    "total_cost_usd": format_float(metrics.get("total_cost_usd"), digits=6),
                }
            )


def format_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def print_metrics_summary(pages: list[PageExtraction]) -> None:
    metrics_list = [page.metrics for page in pages if page.metrics]
    if not metrics_list:
        return

    total_preprocess = sum(float(metrics.get("preprocess_seconds") or 0.0) for metrics in metrics_list)
    total_queue_wait = sum(float(metrics.get("queue_wait_seconds") or 0.0) for metrics in metrics_list)
    total_request = sum(float(metrics.get("request_seconds") or 0.0) for metrics in metrics_list)
    total_elapsed = sum(float(metrics.get("total_seconds") or 0.0) for metrics in metrics_list)
    total_cost = sum(float(metrics.get("total_cost_usd") or 0.0) for metrics in metrics_list)
    total_input_cost = sum(float(metrics.get("input_cost_usd") or 0.0) for metrics in metrics_list)
    total_cached_input_cost = sum(
        float(metrics.get("cached_input_cost_usd") or 0.0) for metrics in metrics_list
    )
    total_output_cost = sum(float(metrics.get("output_cost_usd") or 0.0) for metrics in metrics_list)
    total_input_tokens = sum(int(metrics.get("input_tokens") or 0) for metrics in metrics_list)
    total_output_tokens = sum(int(metrics.get("output_tokens") or 0) for metrics in metrics_list)

    print(
        "Timing summary: "
        f"preprocess={total_preprocess:.2f}s "
        f"queue_wait={total_queue_wait:.2f}s "
        f"request={total_request:.2f}s "
        f"total={total_elapsed:.2f}s"
    )
    print(
        "Token summary: "
        f"input={total_input_tokens} "
        f"output={total_output_tokens}"
    )
    if total_cost > 0:
        print(
            "Cost summary: "
            f"input=${total_input_cost:.6f} "
            f"cached_input=${total_cached_input_cost:.6f} "
            f"output=${total_output_cost:.6f} "
            f"total=${total_cost:.6f}"
        )


def has_metrics(extraction: PageExtraction) -> bool:
    metrics = extraction.metrics
    return bool(metrics) and metrics.get("total_seconds") is not None


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

    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = {} if args.overwrite else load_checkpoint(checkpoint_path)

    with pdfium.PdfDocument(args.input_path) as document:
        page_count = len(document)
    page_numbers = resolve_page_numbers(args, page_count)
    completed_pages = sum(1 for page_number in page_numbers if page_number in checkpoint)
    pending_page_count = len(page_numbers) - completed_pages
    estimated_input_tokens_per_page = resolve_estimated_input_tokens_per_page(args, checkpoint)
    pending_pages = [page_number for page_number in page_numbers if page_number not in checkpoint]
    estimate = build_cost_estimate(
        checkpoint=checkpoint,
        completed_pages=completed_pages,
        pending_pages=pending_page_count,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        cached_input_price_per_1m=cached_input_price_per_1m,
        estimated_input_tokens_per_page=estimated_input_tokens_per_page,
    )
    request_workers = resolve_request_workers(
        args,
        len(pending_pages),
        estimated_input_tokens_per_page,
    )
    preprocess_workers = max(1, min(args.preprocess_workers, len(pending_pages) or 1))
    if args.dry_run:
        print(
            f"Would process {len(page_numbers)} pages from {args.input_path} "
            f"using model {model} and render scale {args.render_scale}"
        )
        print(f"Output: {output_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Metrics: {metrics_path}")
        print_cost_estimate(
            total_pages=len(page_numbers),
            completed_pages=completed_pages,
            pending_pages=pending_page_count,
            estimate=estimate,
        )
        print(
            "Parallel config: "
            f"preprocess_workers={preprocess_workers} "
            f"request_workers={request_workers} "
            f"prompt_tokens_per_minute={args.prompt_tokens_per_minute} "
            f"estimated_input_tokens_per_page={estimated_input_tokens_per_page}"
        )
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} or place it in {args.env_file} before calling OpenAI."
        )

    print_cost_estimate(
        total_pages=len(page_numbers),
        completed_pages=completed_pages,
        pending_pages=pending_page_count,
        estimate=estimate,
    )
    print(
        "Parallel config: "
        f"preprocess_workers={preprocess_workers} "
        f"request_workers={request_workers} "
        f"prompt_tokens_per_minute={args.prompt_tokens_per_minute} "
        f"estimated_input_tokens_per_page={estimated_input_tokens_per_page}"
    )

    page_results = dict(checkpoint)
    cumulative_cost = sum(
        float(page.metrics.get("total_cost_usd") or 0.0)
        for page in page_results.values()
        if page.metrics
    )
    for page_number in page_numbers:
        if page_number in page_results:
            extraction = page_results[page_number]
            if has_metrics(extraction):
                page_cost = float(extraction.metrics.get("total_cost_usd") or 0.0)
                print(
                    f"Page {page_number}: using checkpoint "
                    f"rows={len(extraction.rows)} "
                    f"total_cost=${page_cost:.6f}"
                )
            else:
                print(
                    f"Page {page_number}: using checkpoint but metrics are unavailable. "
                    "Re-run with --overwrite to collect preprocess/request time and actual cost."
                )

    progress_bar = None
    if tqdm is not None and not args.no_progress:
        progress_bar = tqdm(
            total=len(page_numbers),
            initial=completed_pages,
            desc="AKT page OCR",
            unit="page",
        )

    if pending_pages:
        token_bucket = TokenBucket(
            args.prompt_tokens_per_minute,
            capacity=max(
                args.prompt_tokens_per_minute,
                estimated_input_tokens_per_page * max(request_workers, 1),
            ),
        )
        preprocess_futures: dict[Future[PreparedPage], int] = {}
        request_futures: dict[Future[CompletedPage], int] = {}

        with (
            ProcessPoolExecutor(max_workers=preprocess_workers) as preprocess_executor,
            ThreadPoolExecutor(max_workers=request_workers) as request_executor,
        ):
            for page_number in pending_pages:
                preprocess_futures[
                    preprocess_executor.submit(
                        render_page_to_data_url,
                        args.input_path,
                        page_number,
                        args.render_scale,
                        args.image_padding_px,
                    )
                ] = page_number

            for preprocess_future in as_completed(preprocess_futures):
                prepared_page = preprocess_future.result()
                request_futures[
                    request_executor.submit(
                        process_prepared_page,
                        prepared_page,
                        model=model,
                        api_key=api_key,
                        request_timeout_seconds=args.request_timeout_seconds,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        token_bucket=token_bucket,
                        estimated_input_tokens=estimated_input_tokens_per_page,
                    )
                ] = prepared_page.page_number

            for request_future in as_completed(request_futures):
                completed_page = request_future.result()
                extraction = completed_page.extraction
                usage_metrics = completed_page.usage_metrics
                cost_metrics = compute_cost_metrics(
                    usage_metrics,
                    input_price_per_1m,
                    output_price_per_1m,
                    cached_input_price_per_1m,
                )
                page_number = extraction.page_number
                prepared_page = completed_page.prepared_page
                preprocess_started_at = prepared_page.preprocess_started_at
                preprocess_finished_at = prepared_page.preprocess_finished_at
                preprocess_seconds = prepared_page.preprocess_metrics["preprocess_seconds"]
                image_width = prepared_page.preprocess_metrics["image_width"]
                image_height = prepared_page.preprocess_metrics["image_height"]
                image_bytes = prepared_page.preprocess_metrics["image_bytes"]
                queue_wait_seconds = max(
                    completed_page.request_started_at - preprocess_finished_at,
                    0.0,
                )
                total_seconds = max(
                    completed_page.request_finished_at - preprocess_started_at,
                    completed_page.request_seconds,
                )
                extraction.metrics = {
                    "preprocess_started_at": preprocess_started_at,
                    "preprocess_finished_at": preprocess_finished_at,
                    "preprocess_seconds": preprocess_seconds,
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_bytes": image_bytes,
                    "queue_wait_seconds": queue_wait_seconds,
                    **usage_metrics,
                    **cost_metrics,
                    "request_seconds": completed_page.request_seconds,
                    "total_seconds": total_seconds,
                }
                page_results[page_number] = extraction
                append_checkpoint(checkpoint_path, extraction)
                page_cost = float(extraction.metrics.get("total_cost_usd") or 0.0)
                cumulative_cost += page_cost
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"page={page_number} rows={len(extraction.rows)} "
                        f"cumulative~${cumulative_cost:.6f} "
                        f"projected~${(estimate.projected_total_cost_usd or cumulative_cost):.6f}"
                    )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
    if progress_bar is not None:
        progress_bar.close()

    selected_pages = [page_results[page_number] for page_number in sorted(page_results) if page_number in set(page_numbers)]
    row_count = write_output_csv(output_path, selected_pages)
    write_metrics_csv(metrics_path, selected_pages)
    print(f"Wrote {row_count} rows to {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    missing_metrics_pages = [page.page_number for page in selected_pages if not has_metrics(page)]
    if missing_metrics_pages:
        print(
            "Metrics unavailable for checkpoint-only pages: "
            f"{missing_metrics_pages}. "
            "Run with --overwrite to recompute timing and actual cost."
        )
    print_metrics_summary(selected_pages)


if __name__ == "__main__":
    main()
