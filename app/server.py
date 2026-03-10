from __future__ import annotations

import csv
import html
import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

try:
    from .lexicon_lookup import extract_lexicon_matches
    from .sentence_lookup import count_texts_with_same_no, extract_sentence_matches
except ImportError:
    from lexicon_lookup import extract_lexicon_matches
    from sentence_lookup import count_texts_with_same_no, extract_sentence_matches


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_truncated.csv"
HOST = os.environ.get("APP_HOST", "127.0.0.1")
PORT = int(os.environ.get("APP_PORT", "8000"))


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def clip_text(text: str, limit: int = 56) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def build_oare_options(rows: list[dict[str, str]], selected_oare_id: str) -> str:
    options = ['<option value="">-- oare_id を選択してください --</option>']
    for row in rows:
        oare_id = row["oare_id"]
        selected = ' selected' if oare_id == selected_oare_id else ""
        label = f"{oare_id} | {clip_text(row['transliteration'])}"
        options.append(
            f'<option value="{html.escape(oare_id)}"{selected}>{html.escape(label)}</option>'
        )
    return "\n".join(options)


def get_neighbor_oare_ids(
    rows: list[dict[str, str]], selected_oare_id: str
) -> tuple[str | None, str | None]:
    if not selected_oare_id:
        return None, None

    for index, row in enumerate(rows):
        if row["oare_id"] != selected_oare_id:
            continue

        previous_oare_id = rows[index - 1]["oare_id"] if index > 0 else None
        next_oare_id = rows[index + 1]["oare_id"] if index + 1 < len(rows) else None
        return previous_oare_id, next_oare_id

    return None, None


def update_translation(oare_id: str, translation: str) -> bool:
    rows = load_rows()
    updated = False

    for row in rows:
        if row["oare_id"] == oare_id:
            row["translation"] = translation
            updated = True
            break

    if not updated:
        return False

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=DATA_PATH.parent,
        delete=False,
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=["oare_id", "transliteration", "translation"])
        writer.writeheader()
        writer.writerows(rows)
        temp_name = temp_file.name

    os.replace(temp_name, DATA_PATH)
    return True


def get_request_data(environ: dict[str, object]) -> dict[str, str]:
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()

    if method == "GET":
        parsed = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=True)
    else:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(length).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)

    return {key: values[0] if values else "" for key, values in parsed.items()}


def render_page(
    oare_options_html: str,
    same_no_text_count: int,
    oare_id: str = "",
    transliteration: str = "",
    translation: str = "",
    lexicon_matches: list[dict[str, str]] | None = None,
    sentence_matches: list[dict[str, str]] | None = None,
    current_same_no_count: int = 0,
    message: str = "",
    message_type: str = "info",
    not_found: bool = False,
    previous_oare_id: str | None = None,
    next_oare_id: str | None = None,
) -> bytes:
    message_html = ""
    if message:
        message_html = (
            f'<div class="message {html.escape(message_type)}">{html.escape(message)}</div>'
        )

    summary_html = f"""
      <section class="card summary-card">
        <p><strong>Same=No を含む文章数:</strong> {same_no_text_count}</p>
      </section>
    """
    if oare_id and not not_found:
        summary_html = f"""
          <section class="card summary-card">
            <p><strong>Same=No を含む文章数:</strong> {same_no_text_count}</p>
            <p><strong>選択中レコードの Same=No 件数:</strong> {current_same_no_count}</p>
          </section>
        """

    details_html = ""
    if oare_id and not not_found:
        lexicon_rows = ""
        for entry in lexicon_matches or []:
            definitions_html = "<br>".join(
                html.escape(part) for part in entry["definitions"].split("\n")
            ) or '<span class="muted">definition なし</span>'
            female_value = html.escape(entry["female"]) if entry["female"] else ""
            lexicon_rows += f"""
            <tr>
              <td>{html.escape(entry["form"])}</td>
              <td>{html.escape(entry["lexeme"])}</td>
              <td>{html.escape(entry["type"])}</td>
              <td>{female_value}</td>
              <td>{definitions_html}</td>
            </tr>
            """

        lexicon_html = """
          <div class="dictionary-block">
            <h3>OA Lexicon Matches</h3>
            <p class="empty">対応する OA Lexicon 項目は見つかりませんでした。</p>
          </div>
        """
        if lexicon_rows:
            lexicon_html = f"""
            <div class="dictionary-block">
              <h3>OA Lexicon Matches</h3>
              <table class="dictionary-table">
                <thead>
                  <tr>
                    <th>Form</th>
                    <th>Lexeme</th>
                    <th>Type</th>
                    <th>Female(f)</th>
                    <th>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {lexicon_rows}
                </tbody>
              </table>
            </div>
            """

        sentence_rows = ""
        for entry in sentence_matches or []:
            same_class = "same-no" if entry["same"] == "No" else "same-yes"
            sentence_rows += f"""
            <tr>
              <td>{html.escape(entry["sentence_obj_in_text"])}</td>
              <td>{html.escape(entry["first_word_number"])}</td>
              <td>{html.escape(entry["first_word_spelling"])}</td>
              <td>{html.escape(entry["train_word"])}</td>
              <td class="{same_class}">{html.escape(entry["same"])}</td>
              <td>{html.escape(entry["translation"])}</td>
            </tr>
            """

        sentence_html = """
          <div class="dictionary-block">
            <h3>Sentence Matches</h3>
            <p class="empty">対応する sentence 情報は見つかりませんでした。</p>
          </div>
        """
        if sentence_rows:
            sentence_html = f"""
            <div class="dictionary-block">
              <h3>Sentence Matches</h3>
              <table class="dictionary-table">
                <thead>
                  <tr>
                    <th>Sentence Obj</th>
                    <th>First Word #</th>
                    <th>First Word Spelling</th>
                    <th>Train Word</th>
                    <th>Same</th>
                    <th>Translation</th>
                  </tr>
                </thead>
                <tbody>
                  {sentence_rows}
                </tbody>
              </table>
            </div>
            """

        navigation_html = ""
        if previous_oare_id or next_oare_id:
            previous_link = (
                f'<a class="nav-link" href="/?oare_id={html.escape(previous_oare_id)}">Prev</a>'
                if previous_oare_id
                else '<span class="nav-link disabled">Prev</span>'
            )
            next_link = (
                f'<a class="nav-link" href="/?oare_id={html.escape(next_oare_id)}">Next</a>'
                if next_oare_id
                else '<span class="nav-link disabled">Next</span>'
            )
            navigation_html = f'<div class="record-nav">{previous_link}{next_link}</div>'

        details_html = f"""
        <section class="card">
          <h2>Record</h2>
          {navigation_html}
          <dl class="meta">
            <div>
              <dt>oare_id</dt>
              <dd>{html.escape(oare_id)}</dd>
            </div>
          </dl>
          <label for="transliteration">Transliteration</label>
          <textarea id="transliteration" readonly>{html.escape(transliteration)}</textarea>
          <form method="post" class="editor">
            <input type="hidden" name="oare_id" value="{html.escape(oare_id)}">
            <label for="translation">Translation</label>
            <textarea id="translation" name="translation">{html.escape(translation)}</textarea>
            <button type="submit">Save</button>
          </form>
          {lexicon_html}
          {sentence_html}
        </section>
        """
    elif not_found:
        details_html = """
        <section class="card">
          <h2>Record</h2>
          <p class="empty">指定した oare_id は見つかりませんでした。</p>
        </section>
        """

    page = f"""<!DOCTYPE html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Translation Editor</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5efe4;
        --panel: #fffaf2;
        --ink: #1f1a16;
        --muted: #6a5d52;
        --line: #d5c5b0;
        --accent: #9f3a24;
        --accent-strong: #7c2412;
        --ok-bg: #e8f3e7;
        --ok-text: #234525;
        --warn-bg: #fbe9e2;
        --warn-text: #6b2c1e;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(159, 58, 36, 0.12), transparent 30%),
          linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      }}

      main {{
        width: min(1100px, calc(100% - 32px));
        margin: 48px auto;
      }}

      h1 {{
        margin: 0 0 8px;
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: 0.02em;
      }}

      .lead {{
        margin: 0 0 24px;
        color: var(--muted);
      }}

      .summary-card {{
        margin-bottom: 20px;
      }}

      .summary-card p {{
        margin: 0;
      }}

      .summary-card p + p {{
        margin-top: 8px;
      }}

      .card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 18px 40px rgba(61, 39, 22, 0.08);
      }}

      .search {{
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        margin-bottom: 20px;
      }}

      label {{
        display: block;
        margin: 14px 0 8px;
        font-size: 0.95rem;
        color: var(--muted);
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
        padding: 14px 16px;
        background: #fffdf9;
        color: var(--ink);
      }}

      select:focus,
      input:focus,
      textarea:focus {{
        outline: 2px solid rgba(159, 58, 36, 0.2);
        border-color: var(--accent);
      }}

      textarea {{
        min-height: 132px;
        resize: vertical;
      }}

      textarea[readonly] {{
        background: #f8f1e7;
      }}

      button {{
        border: 0;
        border-radius: 999px;
        padding: 14px 22px;
        background: var(--accent);
        color: white;
        cursor: pointer;
        transition: background 0.2s ease, transform 0.2s ease;
      }}

      button:hover {{
        background: var(--accent-strong);
        transform: translateY(-1px);
      }}

      .editor button {{
        margin-top: 16px;
      }}

      .record-nav {{
        display: flex;
        gap: 10px;
        margin: 0 0 18px;
      }}

      .nav-link {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 88px;
        padding: 10px 14px;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--ink);
        background: #fffdf9;
        text-decoration: none;
      }}

      .nav-link.disabled {{
        color: var(--muted);
        background: #f1e7d9;
      }}

      .message {{
        margin-bottom: 20px;
        padding: 14px 16px;
        border-radius: 12px;
      }}

      .dictionary-block {{
        margin-top: 24px;
        border-top: 1px solid var(--line);
        padding-top: 20px;
      }}

      .dictionary-block h3 {{
        margin: 0 0 12px;
      }}

      .dictionary-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
      }}

      .dictionary-table th,
      .dictionary-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}

      .dictionary-table th {{
        color: var(--muted);
        font-weight: 600;
      }}

      .muted {{
        color: var(--muted);
      }}

      .same-yes {{
        color: var(--ok-text);
        font-weight: 600;
      }}

      .same-no {{
        color: var(--warn-text);
        font-weight: 600;
      }}

      .message.success {{
        background: var(--ok-bg);
        color: var(--ok-text);
      }}

      .message.error {{
        background: var(--warn-bg);
        color: var(--warn-text);
      }}

      .meta dt {{
        font-size: 0.85rem;
        color: var(--muted);
      }}

      .meta dd {{
        margin: 4px 0 0;
        font-family: "SFMono-Regular", Consolas, monospace;
      }}

      .empty {{
        margin-bottom: 0;
      }}

      @media (max-width: 640px) {{
        main {{
          width: min(100% - 20px, 1100px);
          margin: 20px auto;
        }}

        .card {{
          padding: 18px;
        }}

        .search {{
          grid-template-columns: 1fr;
        }}

        .dictionary-table,
        .dictionary-table thead,
        .dictionary-table tbody,
        .dictionary-table tr,
        .dictionary-table th,
        .dictionary-table td {{
          display: block;
        }}

        .dictionary-table thead {{
          display: none;
        }}

        .dictionary-table td {{
          padding: 8px 0;
        }}

        button {{
          width: 100%;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Translation Editor</h1>
      <p class="lead">`data/train_refined.csv` の `translation` を `oare_id` ごとに編集して保存します。</p>
      {message_html}
      {summary_html}
      <section class="card">
        <form method="get" class="search">
          <div>
            <label for="oare_id">oare_id</label>
            <select id="oare_id" name="oare_id" onchange="if (this.value) this.form.submit();">
              {oare_options_html}
            </select>
          </div>
          <div>
            <label>&nbsp;</label>
            <button type="submit">Load</button>
          </div>
        </form>
      </section>
      {details_html}
    </main>
  </body>
</html>
"""
    return page.encode("utf-8")


def app(environ: dict[str, object], start_response):
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    data = get_request_data(environ)
    rows = load_rows()
    oare_id = data.get("oare_id", "").strip()
    message = ""
    message_type = "info"
    not_found = False
    transliteration = ""
    translation = ""
    lexicon_matches: list[dict[str, str]] = []
    sentence_matches: list[dict[str, str]] = []
    current_same_no_count = 0
    previous_oare_id = None
    next_oare_id = None

    if method == "POST":
        if not oare_id:
            message = "oare_id を選択してください。"
            message_type = "error"
        else:
            saved = update_translation(oare_id, data.get("translation", ""))
            if saved:
                rows = load_rows()
                message = "translation を保存しました。"
                message_type = "success"
            else:
                message = "指定した oare_id は見つかりませんでした。"
                message_type = "error"
                not_found = True

    if oare_id:
        row = next((candidate for candidate in rows if candidate["oare_id"] == oare_id), None)
        if row:
            transliteration = row["transliteration"]
            translation = row["translation"]
            lexicon_matches = extract_lexicon_matches(transliteration)
            sentence_matches = extract_sentence_matches(oare_id)
            current_same_no_count = sum(1 for match in sentence_matches if match["same"] == "No")
            not_found = False
            previous_oare_id, next_oare_id = get_neighbor_oare_ids(rows, oare_id)
        elif method == "GET":
            message = "指定した oare_id は見つかりませんでした。"
            message_type = "error"
            not_found = True

    body = render_page(
        oare_options_html=build_oare_options(rows, oare_id),
        same_no_text_count=count_texts_with_same_no(),
        oare_id=oare_id,
        transliteration=transliteration,
        translation=translation,
        lexicon_matches=lexicon_matches,
        sentence_matches=sentence_matches,
        current_same_no_count=current_same_no_count,
        message=message,
        message_type=message_type,
        not_found=not_found,
        previous_oare_id=previous_oare_id,
        next_oare_id=next_oare_id,
    )

    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]


if __name__ == "__main__":
    print(f"Serving Translation Editor at http://{HOST}:{PORT}")
    with make_server(HOST, PORT, app) as httpd:
        httpd.serve_forever()
