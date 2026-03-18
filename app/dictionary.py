from __future__ import annotations

import csv
import html
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEXICON_PATH = PROJECT_ROOT / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
HOST = os.environ.get("APP_HOST", "127.0.0.1")
PORT = int(os.environ.get("APP_PORT", "8001"))
MAX_RESULTS = 200


@lru_cache(maxsize=1)
def load_lexicon_rows() -> list[dict[str, str]]:
    with LEXICON_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def search_rows(query: str) -> list[dict[str, str]]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    exact_matches: list[dict[str, str]] = []
    partial_matches: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in load_lexicon_rows():
        form = row.get("form", "").strip()
        definition = row.get("definition", "").strip()
        if not form:
            continue

        normalized_form = normalize_query(form)
        if normalized_query not in normalized_form:
            continue

        key = (row.get("type", ""), form, definition)
        if key in seen:
            continue
        seen.add(key)

        if normalized_form == normalized_query:
            exact_matches.append(row)
        else:
            partial_matches.append(row)

    exact_matches.sort(key=lambda row: (row.get("type", ""), row.get("form", "")))
    partial_matches.sort(key=lambda row: (row.get("type", ""), row.get("form", "")))
    return (exact_matches + partial_matches)[:MAX_RESULTS]


def get_request_data(environ: dict[str, object]) -> dict[str, str]:
    parsed = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def render_page(query: str, rows: list[dict[str, str]]) -> bytes:
    query_html = html.escape(query)
    result_count_html = ""
    rows_html = ""

    if query:
        result_count_html = f"<p><strong>{len(rows)}</strong> result(s)</p>"
        if not rows:
            rows_html = '<p class="empty">No matches found.</p>'
        else:
            for row in rows:
                definition = row.get("definition", "").strip() or "definition なし"
                rows_html += f"""
                <tr>
                  <td>{html.escape(row.get("type", ""))}</td>
                  <td>{html.escape(row.get("form", ""))}</td>
                  <td>{html.escape(row.get("norm", ""))}</td>
                  <td>{html.escape(row.get("lexeme", ""))}</td>
                  <td>{html.escape(definition)}</td>
                </tr>
                """
            rows_html = f"""
            <table class="results-table">
              <thead>
                <tr>
                  <th>type</th>
                  <th>form</th>
                  <th>norm</th>
                  <th>lexeme</th>
                  <th>definition</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
            """

    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OA Lexicon Dictionary</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f6f1e8;
        --card: #fffdf9;
        --border: #d6c7ae;
        --ink: #2d2417;
        --accent: #7d4e1d;
        --muted: #6f6555;
      }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        background: linear-gradient(180deg, #f1e6d6 0%, var(--bg) 100%);
        color: var(--ink);
      }}
      .page {{
        max-width: 1040px;
        margin: 0 auto;
        padding: 40px 20px 56px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(80, 53, 18, 0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      p {{
        color: var(--muted);
        line-height: 1.5;
      }}
      form {{
        display: flex;
        gap: 12px;
        margin: 24px 0 12px;
        flex-wrap: wrap;
      }}
      input[type="text"] {{
        flex: 1 1 360px;
        min-width: 0;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--border);
        font: inherit;
        background: #fff;
      }}
      button {{
        border: 0;
        border-radius: 12px;
        padding: 14px 20px;
        background: var(--accent);
        color: #fff;
        font: inherit;
        cursor: pointer;
      }}
      .results-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
      }}
      .results-table th,
      .results-table td {{
        text-align: left;
        padding: 12px 10px;
        border-top: 1px solid var(--border);
        vertical-align: top;
      }}
      .results-table th {{
        color: var(--accent);
      }}
      .empty {{
        margin-top: 18px;
      }}
      @media (max-width: 720px) {{
        .page {{
          padding: 20px 14px 40px;
        }}
        .card {{
          padding: 18px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="card">
        <h1>OA Lexicon Dictionary</h1>
        <p>Search by <code>form</code> and inspect <code>type</code>, <code>form</code>, <code>norm</code>, <code>lexeme</code>, and <code>definition</code>.</p>
        <form method="get">
          <input type="text" name="q" value="{query_html}" placeholder="e.g. GÍN, ma-na, a-na" />
          <button type="submit">Search</button>
        </form>
        {result_count_html}
        {rows_html}
      </section>
    </main>
  </body>
</html>
"""
    return document.encode("utf-8")


def application(environ: dict[str, object], start_response: object) -> list[bytes]:
    data = get_request_data(environ)
    query = data.get("q", "")
    rows = search_rows(query)

    body = render_page(query, rows)
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]


def main() -> None:
    with make_server(HOST, PORT, application) as server:
        print(f"Serving dictionary app on http://{HOST}:{PORT}")
        server.serve_forever()


if __name__ == "__main__":
    main()
