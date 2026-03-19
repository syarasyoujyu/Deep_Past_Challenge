from __future__ import annotations

import argparse
import csv
import html
import os
import tempfile
import uuid
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

try:
    import pypdfium2 as pdfium
    from PIL import ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "pypdfium2 and pillow are required. Run this app with "
        "`uv run --with pypdfium2 --with pillow python "
        "refine/augment/pdf/parallel_table_editor.py`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "data"
    / "AKT_6a"
    / "akt6a_parallel_openai.csv"
)
DEFAULT_PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "supplement"
    / "Old_Assyrian_Kültepe_Tablets_in_PDF"
    / "AKT 6a.pdf"
)
DEFAULT_HOST = os.environ.get("AKT_EDITOR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("AKT_EDITOR_PORT", "8010"))
DEFAULT_RENDER_SCALE = 2.2
DEFAULT_IMAGE_PADDING_PX = 64
REQUIRED_COLUMNS = ["oare_id", "doc_label", "pdf_page", "transliteration", "translation"]
TRANSLITERATION_CHARACTERS = (
    "!+-.0123456789:<>ABDEGHIKLMNPQRSTUWZ_abdeghiklmnpqrstuwz{}¼½ÀÁÈÉÌÍÙÚ"
    "àáèéìíùúİışŠšṢṣṬṭ…⅓⅔⅙⅚"
)
TRANSLATION_CHARACTERS = (
    "!\"'()+,-.0123456789:;<>?ABCDEFGHIJKLMNOPQRSTUWYZ[]_abcdefghijklmnopqrstuvwxyz"
    "¼½àâāēğīışŠšūṢṣṬṭ–—‘’“”⅓⅔⅙⅚"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit page-level AKT parallel-table CSV while previewing the source PDF page."
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--pdf-path", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--render-scale", type=float, default=DEFAULT_RENDER_SCALE)
    parser.add_argument("--image-padding-px", type=int, default=DEFAULT_IMAGE_PADDING_PX)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)


def load_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for column in REQUIRED_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    return fieldnames, rows


def write_csv_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=csv_path.parent,
        delete=False,
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = temp_file.name
    os.replace(temp_name, csv_path)


def parse_request_lists(environ: dict[str, object]) -> dict[str, list[str]]:
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    if method == "GET":
        raw_query = str(environ.get("QUERY_STRING", ""))
    else:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw_query = environ["wsgi.input"].read(length).decode("utf-8")
    return parse_qs(raw_query, keep_blank_values=True)


def get_first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def get_pdf_page_count(pdf_path: Path) -> int:
    with pdfium.PdfDocument(pdf_path) as document:
        return len(document)


def filter_rows_for_page(rows: list[dict[str, str]], page_number: int) -> list[dict[str, str]]:
    target = str(page_number)
    return [dict(row) for row in rows if row.get("pdf_page") == target]


def render_pdf_page_png(
    pdf_path: Path,
    page_number: int,
    render_scale: float,
    image_padding_px: int,
) -> bytes:
    return _render_pdf_page_png_cached(
        str(pdf_path.resolve()),
        page_number,
        render_scale,
        image_padding_px,
    )


@lru_cache(maxsize=32)
def _render_pdf_page_png_cached(
    pdf_path: str,
    page_number: int,
    render_scale: float,
    image_padding_px: int,
) -> bytes:
    with pdfium.PdfDocument(pdf_path) as document:
        page = document[page_number - 1]
        bitmap = page.render(scale=render_scale)
        image = bitmap.to_pil()
        if image_padding_px > 0:
            image = ImageOps.expand(image, border=image_padding_px, fill="white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_submitted_page_rows(
    query: dict[str, list[str]],
    *,
    fieldnames: list[str],
    page_number: int,
    default_doc_label: str,
) -> list[dict[str, str]]:
    row_count = max(
        len(query.get("row_oare_id", [])),
        len(query.get("row_transliteration", [])),
        len(query.get("row_translation", [])),
    )
    submitted_rows: list[dict[str, str]] = []

    for index in range(row_count):
        deleted = get_list_value(query, "row_deleted", index)
        if deleted == "1":
            continue

        oare_id = normalize_text(get_list_value(query, "row_oare_id", index))
        transliteration = normalize_text(get_list_value(query, "row_transliteration", index))
        translation = normalize_text(get_list_value(query, "row_translation", index))

        if not oare_id:
            oare_id = str(uuid.uuid4())

        if not default_doc_label and not transliteration and not translation:
            continue

        row = {fieldname: "" for fieldname in fieldnames}
        row["oare_id"] = oare_id
        row["doc_label"] = default_doc_label
        row["pdf_page"] = str(page_number)
        row["transliteration"] = transliteration
        row["translation"] = translation
        submitted_rows.append(row)

    return submitted_rows


def get_list_value(query: dict[str, list[str]], key: str, index: int) -> str:
    values = query.get(key, [])
    if index >= len(values):
        return ""
    return values[index]


def replace_page_rows(
    existing_rows: list[dict[str, str]],
    page_number: int,
    new_page_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    page_key = str(page_number)
    indices = [index for index, row in enumerate(existing_rows) if row.get("pdf_page") == page_key]
    if indices:
        start = indices[0]
        end = indices[-1] + 1
        return existing_rows[:start] + new_page_rows + existing_rows[end:]

    insert_at = len(existing_rows)
    for index, row in enumerate(existing_rows):
        raw_page = str(row.get("pdf_page", "")).strip()
        if raw_page.isdigit() and int(raw_page) > page_number:
            insert_at = index
            break
    return existing_rows[:insert_at] + new_page_rows + existing_rows[insert_at:]


def build_page_options(available_pages: list[int], selected_page: int) -> str:
    return "\n".join(
        f'<option value="{page}"{" selected" if page == selected_page else ""}>{page}</option>'
        for page in available_pages
    )


def build_row_cards(rows: list[dict[str, str]], page_number: int, default_doc_label: str) -> str:
    if not rows:
        rows = [
            {
                "oare_id": "",
                "doc_label": default_doc_label,
                "pdf_page": str(page_number),
                "transliteration": "",
                "translation": "",
            }
        ]

    cards: list[str] = []
    for index, row in enumerate(rows):
        cards.append(render_row_card(index, row))
    return "\n".join(cards)


def render_row_card(index: int, row: dict[str, str]) -> str:
    oare_id = row.get("oare_id", "")
    transliteration = row.get("transliteration", "")
    translation = row.get("translation", "")
    card_title = oare_id or f"new-row-{index + 1}"
    return f"""
      <article class="row-card" data-row-card>
        <div class="row-card-head">
          <strong data-row-label>Row {index + 1}</strong>
          <span class="row-badge">{html_escape(card_title)}</span>
          <button type="button" class="ghost" onclick="addRowAfter(this)">Add below</button>
          <button type="button" class="ghost danger" onclick="removeRow(this)">Delete row</button>
        </div>
        <input type="hidden" name="row_deleted" value="0">
        <label>oare_id</label>
        <input type="text" name="row_oare_id" value="{html_escape(oare_id)}" placeholder="auto-generated if blank">
        <label>transliteration</label>
        <textarea name="row_transliteration">{html.escape(transliteration)}</textarea>
        <label>translation</label>
        <textarea name="row_translation">{html.escape(translation)}</textarea>
      </article>
    """


def render_template_row(default_doc_label: str) -> str:
    return f"""
      <template id="row-template">
        <article class="row-card" data-row-card>
          <div class="row-card-head">
            <strong data-row-label>New row</strong>
            <span class="row-badge">new</span>
            <button type="button" class="ghost" onclick="addRowAfter(this)">Add below</button>
            <button type="button" class="ghost danger" onclick="removeRow(this)">Delete row</button>
          </div>
          <input type="hidden" name="row_deleted" value="0">
          <label>oare_id</label>
          <input type="text" name="row_oare_id" value="" placeholder="auto-generated if blank">
          <label>transliteration</label>
          <textarea name="row_transliteration"></textarea>
          <label>translation</label>
          <textarea name="row_translation"></textarea>
        </article>
      </template>
    """


def render_editor_page(
    *,
    csv_path: Path,
    pdf_path: Path,
    available_pages: list[int],
    page_number: int,
    page_rows: list[dict[str, str]],
    message: str = "",
    message_type: str = "info",
) -> bytes:
    selected_index = available_pages.index(page_number) if page_number in available_pages else 0
    previous_page = available_pages[selected_index - 1] if selected_index > 0 else None
    next_page = available_pages[selected_index + 1] if selected_index + 1 < len(available_pages) else None
    default_doc_label = normalize_text(page_rows[0]["doc_label"]) if page_rows else ""
    rows_html = build_row_cards(page_rows, page_number, default_doc_label)
    options_html = build_page_options(available_pages, page_number)
    template_html = render_template_row(default_doc_label)
    message_html = ""
    if message:
        message_html = f'<div class="message {html_escape(message_type)}">{html.escape(message)}</div>'

    previous_link = (
        f'<a class="nav-pill" href="/?page={previous_page}">Prev page</a>'
        if previous_page is not None
        else '<span class="nav-pill disabled">Prev page</span>'
    )
    next_link = (
        f'<a class="nav-pill" href="/?page={next_page}">Next page</a>'
        if next_page is not None
        else '<span class="nav-pill disabled">Next page</span>'
    )

    page = f"""<!DOCTYPE html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AKT Parallel Table Editor</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3ede4;
        --panel: #fffaf2;
        --ink: #201812;
        --muted: #6a5c50;
        --line: #d7c6b2;
        --accent: #8e3b1f;
        --accent-strong: #6f2811;
        --ok-bg: #e8f3e7;
        --ok-text: #214122;
        --warn-bg: #fde9e2;
        --warn-text: #6a2b1a;
      }}

      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(142, 59, 31, 0.12), transparent 28%),
          linear-gradient(180deg, #f9f5ee 0%, var(--bg) 100%);
      }}
      main {{
        width: min(1520px, calc(100% - 24px));
        margin: 18px auto 28px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: clamp(2rem, 3vw, 3rem);
        letter-spacing: 0.02em;
      }}
      .lead {{
        margin: 0;
        color: var(--muted);
      }}
      .topbar {{
        display: flex;
        gap: 12px;
        align-items: end;
        flex-wrap: wrap;
        margin: 20px 0 18px;
      }}
      .topbar form {{
        display: flex;
        gap: 12px;
        align-items: end;
        flex-wrap: wrap;
      }}
      .split {{
        display: grid;
        grid-template-columns: minmax(420px, 1.05fr) minmax(420px, 1fr);
        gap: 18px;
        align-items: start;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 18px 40px rgba(61, 39, 22, 0.08);
        overflow: hidden;
      }}
      .panel-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 18px 22px;
        border-bottom: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.45);
      }}
      .panel-head h2 {{
        margin: 0;
        font-size: 1.15rem;
      }}
      .panel-head p {{
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .pdf-pane {{
        padding: 16px;
        max-height: calc(100vh - 180px);
        overflow: auto;
        background: #efe6d9;
      }}
      .pdf-pane img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 12px;
        background: white;
      }}
      .editor-pane {{
        padding: 18px;
        max-height: calc(100vh - 180px);
        overflow: auto;
      }}
      .message {{
        margin: 0 0 16px;
        padding: 12px 14px;
        border-radius: 12px;
      }}
      .message.info {{
        background: #f1eadf;
        color: var(--ink);
      }}
      .message.success {{
        background: var(--ok-bg);
        color: var(--ok-text);
      }}
      .message.error {{
        background: var(--warn-bg);
        color: var(--warn-text);
      }}
      label {{
        display: block;
        margin: 12px 0 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      select,
      input,
      textarea,
      button {{
        font: inherit;
      }}
      select,
      input,
      textarea {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        background: #fffdf9;
        color: var(--ink);
      }}
      textarea {{
        min-height: 110px;
        resize: vertical;
      }}
      button {{
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        background: var(--accent);
        color: white;
        cursor: pointer;
      }}
      button:hover {{
        background: var(--accent-strong);
      }}
      .ghost {{
        background: transparent;
        color: var(--ink);
        border: 1px solid var(--line);
      }}
      .ghost:hover {{
        background: #f4eadb;
      }}
      .ghost.danger {{
        color: var(--warn-text);
      }}
      .nav-group {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .nav-pill {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 98px;
        padding: 10px 14px;
        border: 1px solid var(--line);
        border-radius: 999px;
        text-decoration: none;
        color: var(--ink);
        background: #fffdf9;
      }}
      .nav-pill.disabled {{
        color: var(--muted);
        background: #efe4d4;
      }}
      .editor-actions {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
      }}
      .row-list {{
        display: grid;
        gap: 14px;
      }}
      .row-card {{
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
        background: #fffdf9;
      }}
      .row-card-head {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }}
      .row-badge {{
        display: inline-flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        background: #f1e5d8;
        color: var(--muted);
        font-size: 0.88rem;
      }}
      .footer-note {{
        margin-top: 14px;
        color: var(--muted);
        font-size: 0.9rem;
      }}
      .char-panel {{
        margin-top: 18px;
        padding: 22px;
      }}
      .char-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }}
      .char-box {{
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px;
        background: #fffdf9;
      }}
      .char-box h3 {{
        margin: 0 0 10px;
        font-size: 1rem;
      }}
      .char-box pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-all;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.95rem;
        line-height: 1.5;
      }}
      @media (max-width: 1100px) {{
        .split {{
          grid-template-columns: 1fr;
        }}
        .pdf-pane {{
          max-height: none;
        }}
        .editor-pane {{
          max-height: none;
        }}
        .char-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>AKT Parallel Table Editor</h1>
      <p class="lead">Left: PDF page preview. Right: editable CSV rows for the selected page.</p>

      <div class="topbar">
        <form method="get">
          <div>
            <label for="page">PDF page</label>
            <select id="page" name="page">{options_html}</select>
          </div>
          <button type="submit">Go</button>
        </form>
        <div class="nav-group">
          {previous_link}
          {next_link}
        </div>
      </div>

      <div class="split">
        <section class="panel">
          <div class="panel-head">
            <div>
              <h2>PDF</h2>
              <p>{html_escape(str(pdf_path))} | page {page_number}</p>
            </div>
          </div>
          <div class="pdf-pane">
            <img src="/page-image?page={page_number}" alt="PDF page {page_number}">
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <h2>CSV Rows</h2>
              <p>{html_escape(str(csv_path))} | page {page_number} | {len(page_rows)} row(s)</p>
            </div>
          </div>
          <div class="editor-pane">
            {message_html}
            <form method="post">
              <input type="hidden" name="page" value="{page_number}">
              <div class="editor-actions">
                <button type="button" class="ghost" onclick="addRowToEnd()">Add row to end</button>
                <button type="submit">Save</button>
              </div>
              <div id="row-list" class="row-list">
                {rows_html}
              </div>
            </form>
            <p class="footer-note">Blank `oare_id` is auto-filled on save. Use `Add below` on each row to insert a new row immediately after it.</p>
          </div>
        </section>
      </div>
      <section class="panel char-panel">
        <div class="panel-head">
          <div>
            <h2>Character Candidates</h2>
            <p>Reference characters for manual correction.</p>
          </div>
        </div>
        <div class="editor-pane">
          <div class="char-grid">
            <div class="char-box">
              <h3>Transliteration</h3>
              <pre>{html.escape(TRANSLITERATION_CHARACTERS)}</pre>
            </div>
            <div class="char-box">
              <h3>Translation</h3>
              <pre>{html.escape(TRANSLATION_CHARACTERS)}</pre>
            </div>
          </div>
        </div>
      </section>
      {template_html}
    </main>

    <script>
      function createRowFragment() {{
        const template = document.getElementById("row-template");
        return template.content.cloneNode(true);
      }}

      function renumberRows() {{
        const cards = document.querySelectorAll("[data-row-card]");
        cards.forEach((card, index) => {{
          const label = card.querySelector("[data-row-label]");
          if (label) {{
            label.textContent = `Row ${{index + 1}}`;
          }}
        }});
      }}

      function addRowAfter(button) {{
        const card = button.closest("[data-row-card]");
        if (!card) {{
          return;
        }}
        const fragment = createRowFragment();
        card.after(fragment);
        renumberRows();
      }}

      function addRowToEnd() {{
        const fragment = createRowFragment();
        document.getElementById("row-list").appendChild(fragment);
        renumberRows();
      }}

      function removeRow(button) {{
        const card = button.closest("[data-row-card]");
        if (card) {{
          card.remove();
          renumberRows();
        }}
      }}

      renumberRows();
    </script>
  </body>
</html>
"""
    return page.encode("utf-8")


def build_app(
    *,
    csv_path: Path,
    pdf_path: Path,
    page_count: int,
    render_scale: float,
    image_padding_px: int,
):
    def app(environ: dict[str, object], start_response):
        all_pages = list(range(1, page_count + 1))
        path = str(environ.get("PATH_INFO", "/"))
        if path == "/page-image":
            query = parse_request_lists(environ)
            requested_page = get_first(query, "page", "1")
            page_number = int(requested_page) if requested_page.isdigit() else 1
            page_number = min(max(page_number, 1), page_count)
            png_bytes = render_pdf_page_png(
                pdf_path,
                page_number,
                render_scale,
                image_padding_px,
            )
            start_response(
                "200 OK",
                [
                    ("Content-Type", "image/png"),
                    ("Content-Length", str(len(png_bytes))),
                    ("Cache-Control", "no-store"),
                ],
            )
            return [png_bytes]

        query = parse_request_lists(environ)
        fieldnames, rows = load_csv_rows(csv_path)
        requested_page = get_first(query, "page", "1")
        page_number = int(requested_page) if requested_page.isdigit() else 1
        page_number = min(max(page_number, 1), page_count)

        message = ""
        message_type = "info"
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        if method == "POST":
            existing_page_rows = filter_rows_for_page(rows, page_number)
            default_doc_label = normalize_text(existing_page_rows[0]["doc_label"]) if existing_page_rows else ""
            submitted_rows = build_submitted_page_rows(
                query,
                fieldnames=fieldnames,
                page_number=page_number,
                default_doc_label=default_doc_label,
            )
            new_rows = replace_page_rows(rows, page_number, submitted_rows)
            write_csv_rows(csv_path, fieldnames, new_rows)
            rows = new_rows
            message = f"Saved {len(submitted_rows)} row(s) for page {page_number}."
            message_type = "success"

        page_rows = filter_rows_for_page(rows, page_number)
        body = render_editor_page(
            csv_path=csv_path,
            pdf_path=pdf_path,
            available_pages=all_pages,
            page_number=page_number,
            page_rows=page_rows,
            message=message,
            message_type=message_type,
        )
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ],
        )
        return [body]

    return app


def main() -> None:
    args = parse_args()
    page_count = get_pdf_page_count(args.pdf_path)
    app = build_app(
        csv_path=args.csv_path,
        pdf_path=args.pdf_path,
        page_count=page_count,
        render_scale=args.render_scale,
        image_padding_px=args.image_padding_px,
    )
    print(f"CSV: {args.csv_path}")
    print(f"PDF: {args.pdf_path}")
    print(f"Listening on http://{args.host}:{args.port}")
    with make_server(args.host, args.port, app) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
