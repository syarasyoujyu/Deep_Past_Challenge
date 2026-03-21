from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
    / "akt6a_parallel_openai.csv"
)
PUZUR_RE = re.compile(r"\bPuzur-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove apostrophes from the transliteration column of an AKT CSV."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path or input_path

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        if "transliteration" not in fieldnames:
            raise ValueError(f"{input_path} does not contain a transliteration column")

        rows = list(reader)

    changed_count = 0
    for row in rows:
        original = row.get("transliteration", "")
        updated = original.replace("'", "")
        updated = PUZUR_RE.sub("Puzur4-", updated)
        if updated != original:
            changed_count += 1
            row["transliteration"] = updated

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Rows changed: {changed_count}")


if __name__ == "__main__":
    main()
