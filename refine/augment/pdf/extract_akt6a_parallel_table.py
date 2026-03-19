from __future__ import annotations

import argparse
import csv
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local CLI use
    raise SystemExit(
        "pdfplumber is required. Run this script with "
        "`uv run --with pdfplumber python refine/augment/pdf/extract_akt6a_parallel_table.py`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
DEFAULT_OUTPUT_FILE = "akt6a_parallel_sentences.csv"
DEFAULT_LINE_OUTPUT_FILE = "akt6a_parallel_lines.csv"
DEFAULT_IMAGE_OUTPUT_SUBDIR = "image"
DEFAULT_START_PAGE = 52
DEFAULT_LEFT_MAX_X = 230.0
DEFAULT_CENTER_MAX_X = 290.0
DEFAULT_TOP_MIN = 40.0
DEFAULT_BOTTOM_MAX = 645.0
DEFAULT_Y_TOLERANCE = 3.0
DEFAULT_NON_TABLE_STREAK_LIMIT = 2
DEFAULT_MIN_PAGE_PARALLEL_ROWS = 8
DEFAULT_LINE_GAP_SPLIT_THRESHOLD = 24.0
DEFAULT_IMAGE_RENDER_SCALE = 3.0
DEFAULT_ROW_PADDING_X = 12.0
DEFAULT_ROW_PADDING_Y = 6.0

TITLE_SOURCE_RE = re.compile(r"\b(?:kt|kts|tc|tcl|cct|ick|poat|ktm)\b", re.I)
TITLE_NUMBER_RE = re.compile(r"^\d+[a-z]?\.$", re.I)
NOTE_COMMENT_RE = re.compile(r"^(?:note|notes|comment)\s*:", re.I)
MARKER_RE = re.compile(r"^(?:\d+[a-z]?|e)\.?$", re.I)
TERMINAL_PUNCTUATION_RE = re.compile(r"(?:\.\.\.|[.!?;:])[\]\"')]*$")
TRANSLITERATION_SIGN_RE = re.compile(
    r"\b(?:DUMU|IGI|KI|URUDU|GIN|KU|KÙ|SES|ITI|LUGAL|TUG|TA)\b"
)
TRANSLITERATION_SYLLABLE_RE = re.compile(r"\b[\w!\[\]{}'/]+-[\w!\[\]{}'/]+\b")
TRANSLITERATION_FUNCTION_RE = re.compile(r"\b(?:a-na|i-na|ma-na|sa|ina|ana)\b", re.I)
ENGLISH_STOPWORDS = {
    "a",
    "again",
    "and",
    "at",
    "be",
    "by",
    "comment",
    "context",
    "date",
    "fact",
    "for",
    "from",
    "in",
    "is",
    "line",
    "meaning",
    "note",
    "of",
    "or",
    "reason",
    "reading",
    "son",
    "silver",
    "text",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "word",
    "owed",
    "witnessed",
}
TRANSLITERATION_LITERAL_REPLACEMENTS = {
    "{": "i",
    "!Star": "IStar",
}
TRANSLITERATION_REGEX_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9])1(?=-)"), "I"),
)


@dataclass(slots=True)
class PdfRow:
    page_number: int
    top: float
    bottom: float
    x0: float
    x1: float
    left_text: str
    marker_text: str
    right_text: str
    doc_label: str = ""
    left_x0: float | None = None
    left_x1: float | None = None
    right_x0: float | None = None
    right_x1: float | None = None
    marker_x0: float | None = None
    marker_x1: float | None = None
    row_image_path: str = ""
    left_image_path: str = ""
    right_image_path: str = ""

    @property
    def combined_text(self) -> str:
        return " ".join(
            fragment for fragment in (self.left_text, self.marker_text, self.right_text) if fragment
        )


@dataclass(slots=True)
class Segment:
    oare_id: str
    transliteration: str
    translation: str
    pdf_page_start: int
    pdf_page_end: int
    doc_label: str
    y_top: float
    y_bottom: float
    source_rows: list[PdfRow] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract candidate transliteration/translation pairs from the bilingual tables in "
            "AKT 6a. Outputs both line-level rows and heuristic sentence/block-level rows."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--line-output-file", type=str, default=DEFAULT_LINE_OUTPUT_FILE)
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--left-max-x", type=float, default=DEFAULT_LEFT_MAX_X)
    parser.add_argument("--center-max-x", type=float, default=DEFAULT_CENTER_MAX_X)
    parser.add_argument("--top-min", type=float, default=DEFAULT_TOP_MIN)
    parser.add_argument("--bottom-max", type=float, default=DEFAULT_BOTTOM_MAX)
    parser.add_argument("--y-tolerance", type=float, default=DEFAULT_Y_TOLERANCE)
    parser.add_argument(
        "--non-table-streak-limit",
        type=int,
        default=DEFAULT_NON_TABLE_STREAK_LIMIT,
        help="How many consecutive commentary-like rows cause the parser to leave table mode.",
    )
    parser.add_argument(
        "--min-page-parallel-rows",
        type=int,
        default=DEFAULT_MIN_PAGE_PARALLEL_ROWS,
        help="If a continuation page has fewer parallel-like rows than this, require a new title to re-enter table mode.",
    )
    parser.add_argument(
        "--line-gap-split-threshold",
        type=float,
        default=DEFAULT_LINE_GAP_SPLIT_THRESHOLD,
        help="If the y-gap between rows exceeds this, start a new output segment.",
    )
    parser.add_argument(
        "--save-row-images",
        action="store_true",
        help="Render and save one image per detected table row, plus left/right crops.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=None,
        help="Directory for optional row crop images. Defaults to <output-dir>/image.",
    )
    parser.add_argument(
        "--image-render-scale",
        type=float,
        default=DEFAULT_IMAGE_RENDER_SCALE,
        help="PDF render scale used when saving row crop images.",
    )
    parser.add_argument(
        "--row-padding-x",
        type=float,
        default=DEFAULT_ROW_PADDING_X,
        help="Horizontal padding in PDF points added to saved row crop images.",
    )
    parser.add_argument(
        "--row-padding-y",
        type=float,
        default=DEFAULT_ROW_PADDING_Y,
        help="Vertical padding in PDF points added to saved row crop images.",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir
    output_path = output_dir / args.output_file
    line_output_path = output_dir / args.line_output_file
    image_output_dir = args.image_output_dir or (output_dir / DEFAULT_IMAGE_OUTPUT_SUBDIR)
    return output_path, line_output_path, image_output_dir


def normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_transliteration_text(text: str) -> str:
    normalized = normalize_spacing(text)
    for source, target in TRANSLITERATION_LITERAL_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    for pattern, replacement in TRANSLITERATION_REGEX_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def append_fragment(current: str, fragment: str) -> str:
    fragment = normalize_spacing(fragment)
    if not fragment:
        return current
    if not current:
        return fragment
    if current.endswith("-") and fragment and fragment[0].islower():
        return f"{current[:-1]}{fragment}"
    return f"{current} {fragment}"


def english_stopword_hits(text: str) -> int:
    return sum(1 for token in re.findall(r"[A-Za-z]+", text.lower()) if token in ENGLISH_STOPWORDS)


def is_transliteration_like(text: str) -> bool:
    text = normalize_spacing(text)
    if not text:
        return False
    score = 0
    if TRANSLITERATION_SIGN_RE.search(text):
        score += 2
    if TRANSLITERATION_SYLLABLE_RE.search(text):
        score += 2
    if TRANSLITERATION_FUNCTION_RE.search(text):
        score += 1
    if any(char in text for char in "[]{}!/"):
        score += 1
    score += min(text.count("-"), 2)
    score -= english_stopword_hits(text)
    return score >= 2


def is_translation_like(text: str) -> bool:
    text = normalize_spacing(text)
    if not text:
        return False
    if TITLE_SOURCE_RE.search(text):
        return False
    stopword_hits = english_stopword_hits(text)
    alpha_words = re.findall(r"[A-Za-z]{2,}", text)
    return stopword_hits >= 1 or len(alpha_words) >= 3


def extract_page_title(rows: list[PdfRow]) -> str:
    title_rows = [row for row in rows if row.top <= 90 and row.combined_text]
    title_parts = [row.combined_text for row in title_rows]
    title = normalize_spacing(" ".join(title_parts))
    if TITLE_SOURCE_RE.search(title):
        return title
    return ""


def cluster_page_rows(
    page: pdfplumber.page.Page,
    *,
    left_max_x: float,
    center_max_x: float,
    top_min: float,
    bottom_max: float,
    y_tolerance: float,
) -> list[PdfRow]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    filtered_words = [
        word
        for word in words
        if top_min <= float(word["top"]) <= bottom_max
    ]
    filtered_words.sort(key=lambda word: (float(word["top"]), float(word["x0"])))

    buckets: list[tuple[float, list[dict[str, object]]]] = []
    for word in filtered_words:
        top = float(word["top"])
        if not buckets or abs(top - buckets[-1][0]) > y_tolerance:
            buckets.append((top, [word]))
        else:
            buckets[-1][1].append(word)

    rows: list[PdfRow] = []
    for top, row_words in buckets:
        row_words = sorted(row_words, key=lambda word: float(word["x0"]))
        left_words = [word for word in row_words if float(word["x0"]) < left_max_x]
        marker_words = [
            word
            for word in row_words
            if left_max_x <= float(word["x0"]) < center_max_x
        ]
        right_words = [word for word in row_words if float(word["x0"]) >= center_max_x]
        left = " ".join(word["text"] for word in left_words)
        marker = " ".join(word["text"] for word in marker_words)
        right = " ".join(word["text"] for word in right_words)
        row_x0 = min(float(word["x0"]) for word in row_words)
        row_x1 = max(float(word["x1"]) for word in row_words)
        row_bottom = max(float(word["bottom"]) for word in row_words)
        rows.append(
            PdfRow(
                page_number=page.page_number,
                top=top,
                bottom=row_bottom,
                x0=row_x0,
                x1=row_x1,
                left_text=normalize_transliteration_text(left),
                marker_text=normalize_spacing(marker),
                right_text=normalize_spacing(right),
                left_x0=min((float(word["x0"]) for word in left_words), default=None),
                left_x1=max((float(word["x1"]) for word in left_words), default=None),
                right_x0=min((float(word["x0"]) for word in right_words), default=None),
                right_x1=max((float(word["x1"]) for word in right_words), default=None),
                marker_x0=min((float(word["x0"]) for word in marker_words), default=None),
                marker_x1=max((float(word["x1"]) for word in marker_words), default=None),
            )
        )
    return rows


def row_starts_note_or_comment(row: PdfRow) -> bool:
    return any(
        NOTE_COMMENT_RE.match(fragment)
        for fragment in (row.left_text, row.marker_text, row.right_text)
        if fragment
    )


def row_is_title(row: PdfRow) -> bool:
    text = row.combined_text
    return bool(TITLE_SOURCE_RE.search(text) and (TITLE_NUMBER_RE.search(text) or "kt " in text.lower()))


def row_is_parallel_candidate(row: PdfRow) -> bool:
    left_like = is_transliteration_like(row.left_text)
    right_like = is_translation_like(row.right_text)
    if left_like and (right_like or not row.right_text):
        return True
    if right_like and not row.left_text:
        return True
    if left_like and row.marker_text and MARKER_RE.match(row.marker_text):
        return True
    return False


def filter_parallel_rows(
    rows: list[PdfRow],
    *,
    non_table_streak_limit: int,
    min_page_parallel_rows: int,
) -> list[PdfRow]:
    filtered_rows: list[PdfRow] = []
    current_doc_label = ""
    in_table = False
    require_title_to_reenter = False

    page_numbers = sorted({row.page_number for row in rows})
    rows_by_page = {page_number: [row for row in rows if row.page_number == page_number] for page_number in page_numbers}

    for page_number in page_numbers:
        page_rows = rows_by_page[page_number]
        page_has_title = any(row_is_title(row) for row in page_rows)
        page_parallel_candidates = sum(1 for row in page_rows if row_is_parallel_candidate(row))
        non_table_streak = 0

        if in_table and not page_has_title and page_parallel_candidates < min_page_parallel_rows:
            in_table = False
            require_title_to_reenter = True

        for row in page_rows:
            if row_is_title(row):
                current_doc_label = row.combined_text
                require_title_to_reenter = False
                non_table_streak = 0
                continue

            if row_starts_note_or_comment(row):
                in_table = False
                require_title_to_reenter = True
                non_table_streak = 0
                continue

            row.doc_label = current_doc_label
            candidate = row_is_parallel_candidate(row)
            if require_title_to_reenter:
                continue

            if candidate:
                in_table = True
                non_table_streak = 0
                if row.left_text or row.right_text:
                    filtered_rows.append(row)
                continue

            if not in_table:
                continue

            if row.left_text and not is_transliteration_like(row.left_text):
                non_table_streak += 1
            elif row.right_text and not is_translation_like(row.right_text):
                non_table_streak += 1
            elif not row.left_text and not row.right_text:
                non_table_streak += 1
            else:
                non_table_streak = 0

            if non_table_streak > non_table_streak_limit:
                in_table = False
                require_title_to_reenter = True
                non_table_streak = 0
                continue

            if row.left_text or row.right_text:
                filtered_rows.append(row)

    return filtered_rows


def row_should_start_new_segment(previous: Segment | None, row: PdfRow, gap_threshold: float) -> bool:
    if previous is None:
        return True
    if row.page_number != previous.pdf_page_end:
        return True
    if row.top - previous.y_bottom > gap_threshold:
        return True
    if row.left_text and row.right_text and previous.transliteration and previous.translation:
        return True
    if previous.translation and TERMINAL_PUNCTUATION_RE.search(previous.translation) and row.right_text:
        return True
    return False


def build_segments(rows: list[PdfRow], *, gap_threshold: float) -> list[Segment]:
    segments: list[Segment] = []
    current: Segment | None = None

    for row in rows:
        if row_should_start_new_segment(current, row, gap_threshold):
            if current and current.transliteration and current.translation:
                segments.append(current)
            current = Segment(
                oare_id="",
                transliteration="",
                translation="",
                pdf_page_start=row.page_number,
                pdf_page_end=row.page_number,
                doc_label=row.doc_label,
                y_top=row.top,
                y_bottom=row.top,
                source_rows=[],
            )

        assert current is not None
        current.transliteration = append_fragment(current.transliteration, row.left_text)
        current.translation = append_fragment(current.translation, row.right_text)
        current.pdf_page_end = row.page_number
        current.y_bottom = row.top
        current.doc_label = current.doc_label or row.doc_label
        current.source_rows.append(row)

    if current and current.transliteration and current.translation:
        segments.append(current)

    for index, segment in enumerate(segments, start=1):
        segment.oare_id = build_oare_id(index, segment)

    return segments


def build_oare_id(index: int, segment: Segment) -> str:
    seed = (
        "akt6a_segment|"
        f"{normalize_spacing(segment.doc_label)}|"
        f"{segment.pdf_page_start}|{segment.pdf_page_end}|"
        f"{index}|{segment.transliteration}|{segment.translation}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def build_line_oare_id(index: int, row: PdfRow) -> str:
    seed = (
        "akt6a_line|"
        f"{normalize_spacing(row.doc_label)}|"
        f"{row.page_number}|{row.top:.1f}|{index}|"
        f"{row.left_text}|{row.right_text}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def save_row_images(
    input_path: Path,
    rows: list[PdfRow],
    *,
    image_output_dir: Path,
    render_scale: float,
    padding_x: float,
    padding_y: float,
) -> int:
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency for image export
        raise SystemExit(
            "pypdfium2 is required when using --save-row-images. Run this script with "
            "`uv run --with pdfplumber --with pypdfium2 --with pillow python "
            "refine/augment/pdf/extract_akt6a_parallel_table.py --save-row-images ...`."
        ) from exc

    image_output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_page: dict[int, list[tuple[int, PdfRow]]] = {}
    for index, row in enumerate(rows, start=1):
        rows_by_page.setdefault(row.page_number, []).append((index, row))

    saved_count = 0
    with pdfium.PdfDocument(input_path) as document:
        for page_number, page_rows in rows_by_page.items():
            page = document[page_number - 1]
            bitmap = page.render(scale=render_scale)
            page_image = bitmap.to_pil()
            image_width, image_height = page_image.size

            for index, row in page_rows:
                line_id = build_line_oare_id(index, row)
                page_dir = image_output_dir / f"page_{page_number:03d}"
                page_dir.mkdir(parents=True, exist_ok=True)

                row_image = crop_pdf_region(
                    page_image,
                    x0=row.x0 - padding_x,
                    top=row.top - padding_y,
                    x1=row.x1 + padding_x,
                    bottom=row.bottom + padding_y,
                    render_scale=render_scale,
                    image_width=image_width,
                    image_height=image_height,
                )
                row_path = page_dir / f"{line_id}_row.png"
                row_image.save(row_path)
                row.row_image_path = str(row_path)

                if row.left_x0 is not None and row.left_x1 is not None:
                    left_image = crop_pdf_region(
                        page_image,
                        x0=row.left_x0 - padding_x,
                        top=row.top - padding_y,
                        x1=min(row.left_x1 + padding_x, row.marker_x0 - 2.0 if row.marker_x0 is not None else row.left_x1 + padding_x),
                        bottom=row.bottom + padding_y,
                        render_scale=render_scale,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    left_path = page_dir / f"{line_id}_left.png"
                    left_image.save(left_path)
                    row.left_image_path = str(left_path)

                if row.right_x0 is not None and row.right_x1 is not None:
                    right_image = crop_pdf_region(
                        page_image,
                        x0=max(row.right_x0 - padding_x, row.marker_x1 + 2.0 if row.marker_x1 is not None else row.right_x0 - padding_x),
                        top=row.top - padding_y,
                        x1=row.right_x1 + padding_x,
                        bottom=row.bottom + padding_y,
                        render_scale=render_scale,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    right_path = page_dir / f"{line_id}_right.png"
                    right_image.save(right_path)
                    row.right_image_path = str(right_path)

                saved_count += 1

    return saved_count


def crop_pdf_region(
    page_image: Any,
    *,
    x0: float,
    top: float,
    x1: float,
    bottom: float,
    render_scale: float,
    image_width: int,
    image_height: int,
) -> Any:
    left_px = max(int(round(x0 * render_scale)), 0)
    top_px = max(int(round(top * render_scale)), 0)
    right_px = min(int(round(x1 * render_scale)), image_width)
    bottom_px = min(int(round(bottom * render_scale)), image_height)
    if right_px <= left_px:
        right_px = min(left_px + 1, image_width)
    if bottom_px <= top_px:
        bottom_px = min(top_px + 1, image_height)
    return page_image.crop((left_px, top_px, right_px, bottom_px))


def write_line_csv(path: Path, rows: list[PdfRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "oare_id",
                "doc_label",
                "pdf_page",
                "y_top",
                "y_bottom",
                "x0",
                "x1",
                "marker",
                "transliteration",
                "translation",
                "row_image_path",
                "left_image_path",
                "right_image_path",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "oare_id": build_line_oare_id(index, row),
                    "doc_label": row.doc_label,
                    "pdf_page": row.page_number,
                    "y_top": f"{row.top:.1f}",
                    "y_bottom": f"{row.bottom:.1f}",
                    "x0": f"{row.x0:.1f}",
                    "x1": f"{row.x1:.1f}",
                    "marker": row.marker_text,
                    "transliteration": row.left_text,
                    "translation": row.right_text,
                    "row_image_path": row.row_image_path,
                    "left_image_path": row.left_image_path,
                    "right_image_path": row.right_image_path,
                }
            )


def write_segment_csv(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "oare_id",
                "doc_label",
                "pdf_page_start",
                "pdf_page_end",
                "y_top",
                "y_bottom",
                "transliteration",
                "translation",
            ],
        )
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "oare_id": segment.oare_id,
                    "doc_label": segment.doc_label,
                    "pdf_page_start": segment.pdf_page_start,
                    "pdf_page_end": segment.pdf_page_end,
                    "y_top": f"{segment.y_top:.1f}",
                    "y_bottom": f"{segment.y_bottom:.1f}",
                    "transliteration": segment.transliteration,
                    "translation": segment.translation,
                }
            )


def main() -> None:
    args = parse_args()
    output_path, line_output_path, image_output_dir = resolve_output_paths(args)
    all_rows: list[PdfRow] = []

    with pdfplumber.open(args.input_path) as pdf:
        end_page = args.end_page or len(pdf.pages)
        for page_index in range(args.start_page - 1, min(end_page, len(pdf.pages))):
            page_rows = cluster_page_rows(
                pdf.pages[page_index],
                left_max_x=args.left_max_x,
                center_max_x=args.center_max_x,
                top_min=args.top_min,
                bottom_max=args.bottom_max,
                y_tolerance=args.y_tolerance,
            )
            title = extract_page_title(page_rows)
            if title:
                for row in page_rows:
                    row.doc_label = title
            all_rows.extend(page_rows)

    parallel_rows = filter_parallel_rows(
        all_rows,
        non_table_streak_limit=args.non_table_streak_limit,
        min_page_parallel_rows=args.min_page_parallel_rows,
    )
    segments = build_segments(
        parallel_rows,
        gap_threshold=args.line_gap_split_threshold,
    )

    saved_image_count = 0
    if args.save_row_images:
        saved_image_count = save_row_images(
            args.input_path,
            parallel_rows,
            image_output_dir=image_output_dir,
            render_scale=args.image_render_scale,
            padding_x=args.row_padding_x,
            padding_y=args.row_padding_y,
        )

    write_line_csv(line_output_path, parallel_rows)
    write_segment_csv(output_path, segments)

    print(f"Wrote {len(parallel_rows)} line rows to {line_output_path}")
    print(f"Wrote {len(segments)} segment rows to {output_path}")
    if args.save_row_images:
        print(f"Saved {saved_image_count} row images to {image_output_dir}")


if __name__ == "__main__":
    main()
