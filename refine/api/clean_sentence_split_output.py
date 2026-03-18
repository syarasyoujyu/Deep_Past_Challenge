from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clean_segment_text(text: str) -> str:
    cleaned = text.replace("\\", "")
    cleaned = cleaned.replace('"', "")
    cleaned = re.sub(r"[!?]+(?=\s*$)", ".", cleaned)
    cleaned = re.sub(r"[!?]+", "", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def clean_record(record: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    segments = record.get("segments", [])
    if not isinstance(segments, list):
        return record

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for field in fields:
            value = segment.get(field)
            if isinstance(value, str):
                segment[field] = clean_segment_text(value)
    return record


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, list):
            return payload, "json"
        if isinstance(payload, dict):
            return [payload], "json"
        raise ValueError(f"Unsupported JSON payload type in {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records, "jsonl"


def write_records(path: Path, records: list[dict[str, Any]], file_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        if file_format == "json":
            json.dump(records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            return

        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean stray backslashes and double quotes from sentence-split JSON/JSONL output."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="If omitted, overwrite the input file.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["transliteration", "translation"],
        help="Segment fields to clean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.input_path
    records, file_format = load_records(args.input_path)
    cleaned_records = [clean_record(record, args.fields) for record in records]
    write_records(output_path, cleaned_records, file_format)
    print(f"Cleaned {len(cleaned_records)} record(s) to {output_path}")


if __name__ == "__main__":
    main()
