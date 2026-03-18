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
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm

from refine.refine_train_v2 import preprocessor

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "train_refined_v2.csv"
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "glosses" / "train_openai_gloss.json"
FALLBACK_MODEL = "gpt-5-mini"#gpt-5-miniで十分行ける(4.1 nanoだと単語全部はやってくれない)
DEFAULT_MAX_ROWS = 10
DEFAULT_PROMPT_TOKENS_PER_MINUTE = 100_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
OPENAI_API_URL = "https://api.openai.com/v1/responses"
MAX_PHRASE_LENGTH = 4
MAX_DEFINITION_CHARS = 320
DEFAULT_ESTIMATED_OUTPUT_TOKEN_MULTIPLIER = 1.3
MATCH_FIELD_PRIORITY = {
    "form": 0,
    "form_original": 1,
    "norm": 2,
    "lexeme": 3,
    "word": 4,
}
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":(4.0,16.0),
    "gpt-4.1-nano":(0.2,0.8),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-chat-latest": (1.25, 10.00),
    "gpt-5.1-chat-latest": (1.25, 10.00),
    "gpt-5.2-chat-latest": (1.75, 14.00),
}
SYSTEM_INSTRUCTIONS = """You are an Akkadian glossing assistant.

You receive:
- an Akkadian transliteration
- the full English translation of the line
- lexicon candidates for each token
- lexicon candidates for multi-token phrases

Return a token-by-token English gloss.

Rules:
- Stay close to the supplied English translation.
- Prefer the supplied lexicon candidates when they fit.
- Keep each gloss short, ideally 1 to 4 English words.
- Do not invent unsupported meanings.
- If a token is a numeral, measure marker, broken sign, or cannot be glossed confidently, return an empty gloss or a very cautious gloss.
- Preserve token order exactly.
"""
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "oare_id": {"type": "string"},
        "transliteration": {"type": "string"},
        "translation": {"type": "string"},
        "token_glosses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "source_token": {"type": "string"},
                    "normalized_token": {"type": "string"},
                    "gloss": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "supporting_dictionary_forms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "index",
                    "source_token",
                    "normalized_token",
                    "gloss",
                    "confidence",
                    "supporting_dictionary_forms",
                ],
            },
        },
    },
    "required": ["oare_id", "transliteration", "translation", "token_glosses"],
}


@dataclass(frozen=True)
class LexiconCandidate:
    form: str
    norm: str
    lexeme: str
    word: str
    definition: str
    derived_from: str
    match_field: str

    def rank_key(self) -> tuple[int, int, str, str]:
        return (
            MATCH_FIELD_PRIORITY.get(self.match_field, 99),
            len(self.definition),
            self.form,
            self.lexeme,
        )

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "form": self.form,
            "norm": self.norm,
            "lexeme": self.lexeme,
            "word": self.word,
            "definition": shorten_text(self.definition, MAX_DEFINITION_CHARS),
            "derived_from": self.derived_from,
            "match_field": self.match_field,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use OA_Lexicon_eBL_refined_with_definition.csv and OpenAI to assign "
            "token-level English glosses to transliterations."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to a .env file to load before reading API credentials.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--oare-id", type=str, default=None)
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Process all rows from the input CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Number of rows to process when --oare-id is not specified.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        dest="max_rows",
        help="Deprecated alias for --max-rows.",
    )
    parser.add_argument("--transliteration", type=str, default=None)
    parser.add_argument("--translation", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model name. If omitted, use OPENAI_MODEL from the environment or .env.",
    )
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max-token-candidates", type=int, default=3)
    parser.add_argument("--max-phrase-candidates", type=int, default=12)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return preprocessor.preprocess_batch([text])[0].strip()


def split_tokens(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]


def shorten_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


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


def print_prompt_token_summary(prompt_token_values: list[int], method: str) -> None:
    if not prompt_token_values:
        return
    average_prompt_tokens = sum(prompt_token_values) / len(prompt_token_values)
    print(f"Prompt token estimation method: {method}")
    print(
        f"Estimated prompt tokens: avg={average_prompt_tokens:.1f}, "
        f"min={min(prompt_token_values)}, max={max(prompt_token_values)}, total={sum(prompt_token_values)}"
    )


def ensure_train_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("train CSV has no rows")
    required_columns = {"oare_id", "transliteration", "translation"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"train CSV is missing columns: {sorted(missing_columns)}")


def build_normalization_cache(lexicon_rows: list[dict[str, str]]) -> dict[str, str]:
    raw_values: list[str] = []
    seen: set[str] = set()

    for row in lexicon_rows:
        for field in ("form", "form_original", "norm", "lexeme", "word"):
            raw_value = row.get(field, "").strip()
            if not raw_value or raw_value in seen:
                continue
            seen.add(raw_value)
            raw_values.append(raw_value)

    normalized_values = preprocessor.preprocess_batch(raw_values)
    return dict(zip(raw_values, normalized_values, strict=True))


def iter_lookup_keys(
    row: dict[str, str],
    normalization_cache: dict[str, str],
) -> list[tuple[str, tuple[str, ...]]]:
    keys: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for field in ("form", "form_original", "norm", "lexeme", "word"):
        raw_value = row.get(field, "").strip()
        if not raw_value:
            continue
        normalized = normalization_cache.get(raw_value, "")
        tokens = tuple(split_tokens(normalized))
        if not tokens:
            continue
        key = (field, tokens)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    return keys


def build_lexicon_indexes(
    lexicon_rows: list[dict[str, str]],
) -> tuple[dict[str, list[LexiconCandidate]], dict[tuple[str, ...], list[LexiconCandidate]]]:
    token_index: dict[str, list[LexiconCandidate]] = defaultdict(list)
    phrase_index: dict[tuple[str, ...], list[LexiconCandidate]] = defaultdict(list)
    normalization_cache = build_normalization_cache(lexicon_rows)

    for row in lexicon_rows:
        definition = row.get("definition", "").strip()
        if not definition:
            continue

        for match_field, tokens in iter_lookup_keys(row, normalization_cache):
            candidate = LexiconCandidate(
                form=row.get("form", "").strip(),
                norm=row.get("norm", "").strip(),
                lexeme=row.get("lexeme", "").strip(),
                word=row.get("word", "").strip(),
                definition=definition,
                derived_from=row.get("derived_from", "").strip(),
                match_field=match_field,
            )
            if len(tokens) == 1:
                token_index[tokens[0]].append(candidate)
            elif len(tokens) <= MAX_PHRASE_LENGTH:
                phrase_index[tokens].append(candidate)

    return token_index, phrase_index


def dedupe_candidates(
    candidates: list[LexiconCandidate],
    limit: int,
) -> list[LexiconCandidate]:
    deduped: list[LexiconCandidate] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()

    for candidate in sorted(candidates, key=lambda item: item.rank_key()):
        key = (
            candidate.form,
            candidate.norm,
            candidate.lexeme,
            candidate.word,
            candidate.definition,
            candidate.match_field,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break

    return deduped


def find_phrase_candidates(
    normalized_tokens: list[str],
    phrase_index: dict[tuple[str, ...], list[LexiconCandidate]],
    max_phrase_candidates: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, str]] = set()

    for length in range(min(MAX_PHRASE_LENGTH, len(normalized_tokens)), 1, -1):
        for start in range(len(normalized_tokens) - length + 1):
            span = tuple(normalized_tokens[start : start + length])
            if span not in phrase_index:
                continue
            for candidate in dedupe_candidates(phrase_index[span], limit=2):
                dedupe_key = (start, length, candidate.form, candidate.definition)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                matches.append(
                    {
                        "start": start,
                        "length": length,
                        "source_phrase": " ".join(span),
                        "candidate": candidate.to_prompt_dict(),
                    }
                )
                if len(matches) >= max_phrase_candidates:
                    return matches

    return matches


def build_prompt_payload(
    row: dict[str, str],
    token_index: dict[str, list[LexiconCandidate]],
    phrase_index: dict[tuple[str, ...], list[LexiconCandidate]],
    max_token_candidates: int,
    max_phrase_candidates: int,
) -> dict[str, Any]:
    transliteration = row["transliteration"].strip()
    normalized_transliteration = normalize_text(transliteration)
    source_tokens = split_tokens(transliteration)
    normalized_tokens = split_tokens(normalized_transliteration)

    token_payloads: list[dict[str, Any]] = []
    for index, source_token in enumerate(source_tokens):
        normalized_token = normalized_tokens[index] if index < len(normalized_tokens) else ""
        candidates = dedupe_candidates(token_index.get(normalized_token, []), max_token_candidates)
        token_payloads.append(
            {
                "index": index,
                "source_token": source_token,
                "normalized_token": normalized_token,
                "dictionary_candidates": [candidate.to_prompt_dict() for candidate in candidates],
            }
        )

    return {
        "oare_id": row["oare_id"],
        "transliteration": transliteration,
        "normalized_transliteration": normalized_transliteration,
        "translation": row["translation"].strip(),
        "tokens": token_payloads,
        "phrase_candidates": find_phrase_candidates(
            normalized_tokens,
            phrase_index,
            max_phrase_candidates=max_phrase_candidates,
        ),
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


def build_openai_request_payload(
    prompt_payload: dict[str, Any],
    model: str,
    schema: dict[str, Any] = RESPONSE_SCHEMA,
) -> dict[str, Any]:
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
                "name": "akkadian_token_glosses",
                "schema": schema,
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
    # Add a small fixed margin for response-format scaffolding and message wrappers.
    return instruction_tokens + input_tokens + schema_tokens + 64, method


def call_openai_glosser(
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

        sleep_seconds = retry_backoff_seconds * attempt
        time.sleep(max(sleep_seconds, 0.0))

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI API request failed without a captured exception")


@dataclass
class ScheduledRequest:
    index: int
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


def select_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if args.transliteration is not None or args.translation is not None:
        if not args.transliteration or args.translation is None:
            raise ValueError("Provide both --transliteration and --translation together")
        return [
            {
                "oare_id": args.oare_id or "manual_input",
                "transliteration": args.transliteration,
                "translation": args.translation,
            }
        ]

    if args.oare_id:
        matched_rows = [row for row in rows if row["oare_id"] == args.oare_id]
        if not matched_rows:
            raise ValueError(f"No row found for oare_id={args.oare_id}")
        return matched_rows

    if args.all_rows:
        return rows

    max_rows = max(args.max_rows, 1)
    return rows[:max_rows]


def write_results(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        if path.suffix.lower() == ".jsonl":
            for index, record in enumerate(records):
                if index > 0:
                    output_file.write("\n")
                output_file.write(json.dumps(record, ensure_ascii=False, indent=4))
                output_file.write("\n")
            return

        if len(records) == 1:
            json.dump(records[0], output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            return

        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def build_compact_jsonl_path(path: Path) -> Path:
    if path.name.endswith("_compact.jsonl"):
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}_compact.jsonl")
    return path.with_name(f"{path.name}_compact.jsonl")


def write_compact_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")


def append_compact_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output_file.write("\n")
        output_file.flush()


def run_openai_requests(
    scheduled_requests: list[ScheduledRequest],
    api_key: str,
    prompt_tokens_per_minute: int,
    max_workers: int | None,
    partial_output_path: Path,
    request_timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[dict[str, Any]]:
    total_rows = len(scheduled_requests)
    if total_rows == 0:
        return []

    prompt_token_values = [request.prompt_tokens for request in scheduled_requests]
    total_prompt_tokens = sum(prompt_token_values)
    average_prompt_tokens = total_prompt_tokens / total_rows
    derived_max_workers = max(1, math.floor(prompt_tokens_per_minute / max(average_prompt_tokens, 1.0)))
    worker_count = max(1, max_workers or derived_max_workers)
    worker_count = min(worker_count, total_rows)

    print(
        f"Throttle: {prompt_tokens_per_minute} prompt tokens/min, "
        f"workers={worker_count}, theoretical max rows/min={prompt_tokens_per_minute / max(average_prompt_tokens, 1.0):.2f}"
    )

    results: list[dict[str, Any] | None] = [None] * total_rows
    elapsed_times: list[float] = []
    total_start = time.monotonic()
    token_bucket = TokenBucket(
        prompt_tokens_per_minute,
        capacity=max(prompt_tokens_per_minute, max(prompt_token_values)),
    )
    progress = tqdm(
        total=total_rows,
        desc="OpenAI API",
        unit="row",
        dynamic_ncols=True,
    )
    outstanding: dict[Future[tuple[int, dict[str, Any], float]], ScheduledRequest] = {}
    next_request_index = 0
    submitted_count = 0
    completed_count = 0

    partial_output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output_path.write_text("", encoding="utf-8")

    def update_progress_postfix(last_elapsed: float | None = None, prompt_tokens: int | None = None) -> None:
        average_seconds = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
        remaining_rows = total_rows - completed_count
        eta_seconds = remaining_rows * average_seconds if average_seconds else 0.0
        postfix = {
            "submitted": str(submitted_count),
            "done": str(completed_count),
            "in_flight": str(len(outstanding)),
        }
        if last_elapsed is not None:
            postfix["last"] = f"{last_elapsed:.1f}s"
        if average_seconds:
            postfix["avg"] = f"{average_seconds:.1f}s"
            postfix["eta"] = f"{eta_seconds:.1f}s"
        if prompt_tokens is not None:
            postfix["ptoks"] = f"{prompt_tokens}"
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
            result = call_openai_glosser(
                request_payload=scheduled_request.request_payload,
                api_key=api_key,
                request_timeout_seconds=request_timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            row_elapsed = time.monotonic() - row_start
            return scheduled_request.index, result, row_elapsed

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
                            request_index, result, row_elapsed = future.result()
                            results[request_index] = result
                            elapsed_times.append(row_elapsed)
                            completed_count += 1
                            append_compact_jsonl_record(partial_output_path, result)
                            progress.update(1)
                            update_progress_postfix(
                                last_elapsed=row_elapsed,
                                prompt_tokens=scheduled_request.prompt_tokens,
                            )
                        continue

                if next_request_index < total_rows and len(outstanding) < worker_count:
                    scheduled_request = scheduled_requests[next_request_index]
                    token_bucket.acquire(scheduled_request.prompt_tokens)
                    future = submit_request(
                        executor,
                        scheduled_request,
                        tokens_reserved=True,
                    )
                    if future is None:
                        raise RuntimeError("Failed to submit request after reserving tokens")
                    submitted_count += 1
                    next_request_index += 1
                    update_progress_postfix()
    except Exception:
        print(
            f"Aborted after {completed_count} completed row(s). "
            f"Partial results were saved to {partial_output_path}"
        )
        raise
    finally:
        progress.close()

    total_elapsed = time.monotonic() - total_start
    average_seconds = total_elapsed / total_rows
    print(
        f"API calls finished in {total_elapsed:.1f}s total "
        f"({average_seconds:.1f}s per row on average)"
    )
    return [result for result in results if result is not None]


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
    prompt_payload: dict[str, Any],
    model: str,
    multiplier: float,
) -> tuple[int, str]:
    minimal_response_payload = {
        "oare_id": prompt_payload["oare_id"],
        "transliteration": prompt_payload["transliteration"],
        "translation": prompt_payload["translation"],
        "token_glosses": [
            {
                "index": token["index"],
                "source_token": token["source_token"],
                "normalized_token": token["normalized_token"],
                "gloss": "",
                "confidence": "medium",
                "supporting_dictionary_forms": [],
            }
            for token in prompt_payload["tokens"]
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


def main() -> None:
    args = parse_args()
    env_loaded = load_env_file(args.env_file)
    if env_loaded:
        print(f"Loaded environment variables from {args.env_file}")
    if not args.model:
        args.model = os.environ.get("OPENAI_MODEL", FALLBACK_MODEL)
    train_rows = load_csv_rows(args.input_path)
    ensure_train_columns(train_rows)
    selected_rows = select_rows(args, train_rows)
    print(f"Selected {len(selected_rows)} row(s) for processing")
    lexicon_rows = load_csv_rows(args.lexicon_path)
    token_index, phrase_index = build_lexicon_indexes(lexicon_rows)

    prompt_payloads = [
        build_prompt_payload(
            row=row,
            token_index=token_index,
            phrase_index=phrase_index,
            max_token_candidates=args.max_token_candidates,
            max_phrase_candidates=args.max_phrase_candidates,
        )
        for row in selected_rows
    ]
    scheduled_requests: list[ScheduledRequest] = []
    token_estimation_method = "heuristic"
    output_token_estimation_method = "heuristic"
    for index, prompt_payload in enumerate(prompt_payloads):
        request_payload = build_openai_request_payload(
            prompt_payload=prompt_payload,
            model=args.model,
        )
        prompt_tokens, token_estimation_method = estimate_request_prompt_tokens(
            request_payload=request_payload,
            model=args.model,
        )
        estimated_output_tokens, output_token_estimation_method = estimate_response_tokens(
            prompt_payload=prompt_payload,
            model=args.model,
            multiplier=args.estimated_output_token_multiplier,
        )
        scheduled_requests.append(
            ScheduledRequest(
                index=index,
                request_payload=request_payload,
                prompt_tokens=prompt_tokens,
                estimated_output_tokens=estimated_output_tokens,
            )
        )
    prompt_token_values = [request.prompt_tokens for request in scheduled_requests]
    print_prompt_token_summary(prompt_token_values, token_estimation_method)
    print(f"Response token estimation method: {output_token_estimation_method}")
    input_price_per_1m, output_price_per_1m = resolve_model_pricing(
        model=args.model,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )
    estimated_total_cost = print_cost_estimate_summary(
        scheduled_requests=scheduled_requests,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        model=args.model,
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

    if args.dry_run:
        for payload in prompt_payloads:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}")

    compact_output_path = build_compact_jsonl_path(args.output_path)
    print(f"Checkpoint JSONL path: {compact_output_path}")
    results = run_openai_requests(
        scheduled_requests=scheduled_requests,
        api_key=api_key,
        prompt_tokens_per_minute=args.prompt_tokens_per_minute,
        max_workers=args.max_workers,
        partial_output_path=compact_output_path,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )

    if args.output_path:
        write_results(args.output_path, results)
        write_compact_jsonl(compact_output_path, results)
        print(f"Wrote {len(results)} glossed record(s) to {args.output_path}")
        print(f"Wrote compact JSONL to {compact_output_path}")
        return

    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
