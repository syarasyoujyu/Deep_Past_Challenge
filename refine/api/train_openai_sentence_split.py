from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from refine.api.clean_sentence_split_output import clean_segment_text

def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "train_refined_v2.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "train_refined_v2_sentence_split.csv"
DEFAULT_REFINED_OUTPUT_PATH = PROJECT_ROOT / "data" / "train_refined_v2_sentence_split_refined.csv"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "now" / "train_refined_v2_sentence_split_compact.jsonl"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
FALLBACK_MODEL = "gpt-5-mini"
DEFAULT_MAX_ROWS = 10
DEFAULT_PROMPT_TOKENS_PER_MINUTE = 100_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_ESTIMATED_OUTPUT_TOKEN_MULTIPLIER = 1.3
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-chat-latest": (1.25, 10.00),
    "gpt-5.1-chat-latest": (1.25, 10.00),
    "gpt-5.2-chat-latest": (1.75, 14.00),
}

SYSTEM_INSTRUCTIONS = """You split each training example into aligned sentence-level or clause-level pairs.

You receive one Akkadian transliteration line and its full English translation.

Return 1..N aligned segments.

Rules:
- Preserve the original order.
- Copy substrings from the source text; do not paraphrase or normalize.
- Each segment must be a contiguous span of the original transliteration.
- Each segment must be a contiguous span of the original translation.
- Split only at natural sentence boundaries or strong clause boundaries clearly supported by the English translation.
- Prefer fewer splits when alignment is uncertain.
- Cover all content exactly once with no overlap and no omission.
- If the pair should stay as one unit, return exactly one segment.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "oare_id": {"type": "string"},
        "segments": {
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
    "required": ["oare_id", "segments"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use OpenAI to split train_refined_v2.csv rows into aligned "
            "sentence/clause-level transliteration-translation pairs."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--refined-output-path",
        type=Path,
        default=None,
        help=(
            "Optional filtered CSV written after the main output. "
            "Defaults to <output-path stem>_refined.csv."
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--oare-id", type=str, default=None)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument(
        "--limit",
        type=int,
        dest="max_rows",
        help="Deprecated alias for --max-rows.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt-tokens-per-minute",
        type=int,
        default=DEFAULT_PROMPT_TOKENS_PER_MINUTE,
        help="Client-side prompt token budget used for throttling and parallelism.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent OpenAI requests. If omitted, it is derived from token budget.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for retryable OpenAI request failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Base backoff in seconds between retry attempts.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Override OpenAI input pricing in USD per 1M tokens.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Override OpenAI output pricing in USD per 1M tokens.",
    )
    parser.add_argument(
        "--estimated-output-token-multiplier",
        type=float,
        default=DEFAULT_ESTIMATED_OUTPUT_TOKEN_MULTIPLIER,
        help="Multiplier applied to minimal response token estimates.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=None,
        help="Abort before API calls if estimated total cost exceeds this amount.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print token and cost estimates, then exit without calling the API.",
    )
    return parser.parse_args()


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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def estimate_text_tokens(text: str, model: str) -> tuple[int, str]:
    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)), "tiktoken"
    except Exception:
        return max(1, math.ceil(len(text.encode("utf-8")) / 3.0)), "heuristic"


def ensure_train_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("train CSV has no rows")
    required_columns = {"oare_id", "transliteration", "translation"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"train CSV is missing columns: {sorted(missing_columns)}")


def select_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if args.oare_id:
        matched_rows = [row for row in rows if row["oare_id"] == args.oare_id]
        if not matched_rows:
            raise ValueError(f"No row found for oare_id={args.oare_id}")
        return matched_rows

    if args.all_rows:
        return rows

    return rows[: max(args.max_rows, 1)]


def build_prompt_payload(row: dict[str, str]) -> dict[str, str]:
    return {
        "oare_id": row["oare_id"].strip(),
        "transliteration": row["transliteration"].strip(),
        "translation": row["translation"].strip(),
    }


def build_openai_request_payload(prompt_payload: dict[str, str], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "akkadian_sentence_split",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    }


def estimate_request_prompt_tokens(
    request_payload: dict[str, Any],
    model: str,
) -> tuple[int, str]:
    schema_text = json.dumps(
        request_payload["text"]["format"]["schema"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    input_text = request_payload["input"][0]["content"][0]["text"]
    instruction_tokens, method = estimate_text_tokens(request_payload["instructions"], model)
    input_tokens, _ = estimate_text_tokens(input_text, model)
    schema_tokens, _ = estimate_text_tokens(schema_text, model)
    return instruction_tokens + input_tokens + schema_tokens + 64, method


def print_prompt_token_summary(prompt_token_values: list[int], method: str) -> None:
    if not prompt_token_values:
        return
    average_prompt_tokens = sum(prompt_token_values) / len(prompt_token_values)
    print(f"Prompt token estimation method: {method}")
    print(
        f"Estimated prompt tokens: avg={average_prompt_tokens:.1f}, "
        f"min={min(prompt_token_values)}, max={max(prompt_token_values)}, total={sum(prompt_token_values)}"
    )


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


def call_openai_splitter(
    request_payload: dict[str, Any],
    api_key: str,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
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
            return json.loads(response_text)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code == 429 or 500 <= exc.code < 600
            last_error = RuntimeError(f"OpenAI API request failed: {exc.code} {details}")
            if not should_retry or attempt == max_retries:
                raise last_error from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            last_error = RuntimeError(f"OpenAI API request failed: {exc}")
            if attempt == max_retries:
                raise last_error from exc

        time.sleep(max(retry_backoff_seconds * attempt, 0.0))

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI API request failed without a captured exception")


@dataclass
class ScheduledRequest:
    index: int
    source_oare_id: str
    request_payload: dict[str, Any]
    prompt_tokens: int
    estimated_output_tokens: int


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
        needed = float(amount)
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

    def try_acquire(self, amount: int) -> bool:
        needed = float(amount)
        with self.condition:
            now = time.monotonic()
            elapsed = now - self.last_updated
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_updated = now
            if self.tokens >= needed:
                self.tokens -= needed
                return True
            return False


def resolve_model_pricing(
    model: str,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
) -> tuple[float | None, float | None]:
    normalized_model = model.strip()
    default_prices = MODEL_PRICING_PER_1M_TOKENS.get(normalized_model)
    resolved_input_price = (
        input_price_per_1m
        if input_price_per_1m is not None
        else (default_prices[0] if default_prices is not None else None)
    )
    resolved_output_price = (
        output_price_per_1m
        if output_price_per_1m is not None
        else (default_prices[1] if default_prices is not None else None)
    )
    return resolved_input_price, resolved_output_price


def estimate_response_tokens(
    prompt_payload: dict[str, str],
    model: str,
    multiplier: float,
) -> tuple[int, str]:
    minimal_response_payload = {
        "oare_id": prompt_payload["oare_id"],
        "segments": [
            {
                "index": 1,
                "transliteration": prompt_payload["transliteration"],
                "translation": prompt_payload["translation"],
            }
        ],
    }
    output_text = json.dumps(
        minimal_response_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output_tokens, method = estimate_text_tokens(output_text, model)
    return max(1, math.ceil(output_tokens * multiplier)), method


def print_cost_estimate_summary(
    scheduled_requests: list[ScheduledRequest],
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    model: str,
) -> float | None:
    if not scheduled_requests:
        return 0.0

    prompt_token_values = [request.prompt_tokens for request in scheduled_requests]
    output_token_values = [request.estimated_output_tokens for request in scheduled_requests]
    total_prompt_tokens = sum(prompt_token_values)
    total_output_tokens = sum(output_token_values)
    avg_prompt_tokens = total_prompt_tokens / len(prompt_token_values)
    avg_output_tokens = total_output_tokens / len(output_token_values)

    print(
        f"Estimated response tokens: avg={avg_output_tokens:.1f}, "
        f"min={min(output_token_values)}, max={max(output_token_values)}, total={total_output_tokens}"
    )

    if input_price_per_1m is None or output_price_per_1m is None:
        print(
            f"Pricing for model {model!r} is not configured. "
            "Set --input-price-per-1m and --output-price-per-1m to estimate cost."
        )
        return None

    estimated_input_cost = total_prompt_tokens / 1_000_000 * input_price_per_1m
    estimated_output_cost = total_output_tokens / 1_000_000 * output_price_per_1m
    estimated_total_cost = estimated_input_cost + estimated_output_cost

    print(
        f"Estimated cost for model {model}: "
        f"input=${estimated_input_cost:.4f} + output=${estimated_output_cost:.4f} "
        f"= total=${estimated_total_cost:.4f}"
    )
    print(
        f"Per-row estimate: prompt={avg_prompt_tokens:.1f} tokens, "
        f"response={avg_output_tokens:.1f} tokens, "
        f"cost=${estimated_total_cost / len(scheduled_requests):.4f}"
    )
    return estimated_total_cost


def normalize_segments(result: dict[str, Any], source_oare_id: str) -> dict[str, Any]:
    segments = result.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"OpenAI returned no segments for oare_id={source_oare_id}")

    normalized_segments: list[dict[str, Any]] = []
    for segment in segments:
        transliteration = clean_segment_text(str(segment.get("transliteration", "")))
        translation = clean_segment_text(str(segment.get("translation", "")))
        if not transliteration or not translation:
            continue
        index = len(normalized_segments) + 1
        normalized_segments.append(
            {
                "index": index,
                "split_oare_id": f"{source_oare_id}--{index}",
                "transliteration": transliteration,
                "translation": translation,
            }
        )

    if not normalized_segments:
        raise ValueError(f"OpenAI returned only empty segments for oare_id={source_oare_id}")

    return {
        "oare_id": source_oare_id,
        "segments": normalized_segments,
    }


def append_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output_file.write("\n")
        output_file.flush()


def write_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")


def write_split_csv(path: Path, split_results: list[dict[str, Any]]) -> int:
    fieldnames = ["oare_id", "source_oare_id", "segment_index", "transliteration", "translation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in split_results:
            source_oare_id = result["oare_id"]
            for segment in result["segments"]:
                writer.writerow(
                    {
                        "oare_id": segment["split_oare_id"],
                        "source_oare_id": source_oare_id,
                        "segment_index": segment["index"],
                        "transliteration": segment["transliteration"],
                        "translation": segment["translation"],
                    }
                )
                written_rows += 1

    return written_rows


def resolve_refined_output_path(args: argparse.Namespace) -> Path:
    if args.refined_output_path is not None:
        return args.refined_output_path
    return args.output_path.with_name(f"{args.output_path.stem}_refined{args.output_path.suffix}")


def filter_split_results_with_terminal_period(
    split_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered_results: list[dict[str, Any]] = []
    for result in split_results:
        filtered_segments = [
            segment
            for segment in result["segments"]
            if str(segment.get("translation", "")).rstrip().endswith(".")
        ]
        if not filtered_segments:
            continue
        filtered_results.append(
            {
                "oare_id": result["oare_id"],
                "segments": filtered_segments,
            }
        )
    return filtered_results


def run_openai_requests(
    scheduled_requests: list[ScheduledRequest],
    api_key: str,
    prompt_tokens_per_minute: int,
    max_workers: int | None,
    checkpoint_path: Path,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[dict[str, Any]]:
    total_rows = len(scheduled_requests)
    if total_rows == 0:
        return []

    prompt_token_values = [request.prompt_tokens for request in scheduled_requests]
    average_prompt_tokens = sum(prompt_token_values) / total_rows
    derived_max_workers = max(1, math.floor(prompt_tokens_per_minute / max(average_prompt_tokens, 1.0)))
    worker_count = max(1, max_workers or derived_max_workers)
    worker_count = min(worker_count, total_rows)

    print(
        f"Throttle: {prompt_tokens_per_minute} prompt tokens/min, "
        f"workers={worker_count}, theoretical max rows/min={prompt_tokens_per_minute / max(average_prompt_tokens, 1.0):.2f}"
    )

    results: list[dict[str, Any] | None] = [None] * total_rows
    elapsed_times: list[float] = []
    segments_completed = 0
    token_bucket = TokenBucket(
        prompt_tokens_per_minute,
        capacity=max(prompt_tokens_per_minute, max(prompt_token_values)),
    )
    progress = tqdm(total=total_rows, desc="OpenAI split", unit="row", dynamic_ncols=True)
    outstanding: dict[Future[tuple[int, dict[str, Any], float]], ScheduledRequest] = {}
    next_request_index = 0
    submitted_count = 0
    completed_count = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("", encoding="utf-8")

    def update_progress_postfix(last_elapsed: float | None = None, prompt_tokens: int | None = None) -> None:
        average_seconds = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
        remaining_rows = total_rows - completed_count
        eta_seconds = remaining_rows * average_seconds if average_seconds else 0.0
        postfix = {
            "submitted": str(submitted_count),
            "done": str(completed_count),
            "in_flight": str(len(outstanding)),
            "segments": str(segments_completed),
        }
        if last_elapsed is not None:
            postfix["last"] = f"{last_elapsed:.1f}s"
        if average_seconds:
            postfix["avg"] = f"{average_seconds:.1f}s"
            postfix["eta"] = f"{eta_seconds:.1f}s"
        if prompt_tokens is not None:
            postfix["ptoks"] = str(prompt_tokens)
        progress.set_postfix(postfix)

    def submit_request(
        executor: ThreadPoolExecutor,
        scheduled_request: ScheduledRequest,
        *,
        tokens_reserved: bool = False,
    ) -> Future[tuple[int, dict[str, Any], float]] | None:
        if not tokens_reserved and not token_bucket.try_acquire(scheduled_request.prompt_tokens):
            return None

        def task() -> tuple[int, dict[str, Any], float]:
            row_start = time.monotonic()
            response = call_openai_splitter(
                request_payload=scheduled_request.request_payload,
                api_key=api_key,
                request_timeout_seconds=request_timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            normalized = normalize_segments(response, source_oare_id=scheduled_request.source_oare_id)
            row_elapsed = time.monotonic() - row_start
            return scheduled_request.index, normalized, row_elapsed

        future = executor.submit(task)
        outstanding[future] = scheduled_request
        return future

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while next_request_index < total_rows or outstanding:
                while next_request_index < total_rows and len(outstanding) < worker_count:
                    future = submit_request(executor, scheduled_requests[next_request_index])
                    if future is None:
                        break
                    submitted_count += 1
                    next_request_index += 1
                    update_progress_postfix()

                if outstanding:
                    completed, _ = wait(
                        outstanding.keys(),
                        timeout=0.2,
                        return_when=FIRST_COMPLETED,
                    )
                    if completed:
                        for future in completed:
                            scheduled_request = outstanding.pop(future)
                            request_index, normalized, row_elapsed = future.result()
                            results[request_index] = normalized
                            elapsed_times.append(row_elapsed)
                            completed_count += 1
                            segments_completed += len(normalized["segments"])
                            append_checkpoint_record(checkpoint_path, normalized)
                            progress.update(1)
                            update_progress_postfix(
                                last_elapsed=row_elapsed,
                                prompt_tokens=scheduled_request.prompt_tokens,
                            )
                        continue

                if next_request_index < total_rows and len(outstanding) < worker_count:
                    scheduled_request = scheduled_requests[next_request_index]
                    token_bucket.acquire(scheduled_request.prompt_tokens)
                    future = submit_request(executor, scheduled_request, tokens_reserved=True)
                    if future is None:
                        raise RuntimeError("Failed to submit request after reserving tokens")
                    submitted_count += 1
                    next_request_index += 1
                    update_progress_postfix()
    except Exception:
        print(f"Aborted early. Partial results were saved to {checkpoint_path}")
        raise
    finally:
        progress.close()

    return [result for result in results if result is not None]


def main() -> None:
    args = parse_args()
    if load_env_file(args.env_file):
        print(f"Loaded environment variables from {args.env_file}")

    model = args.model or os.environ.get("OPENAI_MODEL", FALLBACK_MODEL)
    train_rows = load_csv_rows(args.input_path)
    ensure_train_columns(train_rows)
    selected_rows = select_rows(args, train_rows)
    print(f"Selected {len(selected_rows)} row(s) for processing")

    prompt_payloads = [build_prompt_payload(row) for row in selected_rows]
    if args.dry_run:
        for payload in prompt_payloads:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}")

    scheduled_requests: list[ScheduledRequest] = []
    token_estimation_method = "heuristic"
    output_token_estimation_method = "heuristic"
    for index, payload in enumerate(prompt_payloads):
        request_payload = build_openai_request_payload(payload, model=model)
        prompt_tokens, token_estimation_method = estimate_request_prompt_tokens(
            request_payload=request_payload,
            model=model,
        )
        estimated_output_tokens, output_token_estimation_method = estimate_response_tokens(
            prompt_payload=payload,
            model=model,
            multiplier=args.estimated_output_token_multiplier,
        )
        scheduled_requests.append(
            ScheduledRequest(
                index=index,
                source_oare_id=payload["oare_id"],
                request_payload=request_payload,
                prompt_tokens=prompt_tokens,
                estimated_output_tokens=estimated_output_tokens,
            )
        )

    prompt_token_values = [request.prompt_tokens for request in scheduled_requests]
    print_prompt_token_summary(prompt_token_values, token_estimation_method)
    print(f"Response token estimation method: {output_token_estimation_method}")
    input_price_per_1m, output_price_per_1m = resolve_model_pricing(
        model=model,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )
    estimated_total_cost = print_cost_estimate_summary(
        scheduled_requests=scheduled_requests,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        model=model,
    )

    if (
        estimated_total_cost is not None
        and args.max_estimated_cost_usd is not None
        and estimated_total_cost > args.max_estimated_cost_usd
    ):
        raise SystemExit(
            f"Estimated total cost ${estimated_total_cost:.4f} exceeds "
            f"--max-estimated-cost-usd={args.max_estimated_cost_usd:.4f}. Aborting."
        )

    if args.estimate_only:
        return

    split_results = run_openai_requests(
        scheduled_requests=scheduled_requests,
        api_key=api_key,
        prompt_tokens_per_minute=args.prompt_tokens_per_minute,
        max_workers=args.max_workers,
        checkpoint_path=args.checkpoint_path,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )

    write_checkpoint(args.checkpoint_path, split_results)
    written_rows = write_split_csv(args.output_path, split_results)
    refined_output_path = resolve_refined_output_path(args)
    refined_split_results = filter_split_results_with_terminal_period(split_results)
    refined_written_rows = write_split_csv(refined_output_path, refined_split_results)
    print(f"Wrote {written_rows} split row(s) to {args.output_path}")
    print(
        f"Wrote {refined_written_rows} split row(s) with translation ending in '.' "
        f"to {refined_output_path}"
    )
    print(f"Wrote checkpoint JSONL to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
