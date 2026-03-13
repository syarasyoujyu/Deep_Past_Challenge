from __future__ import annotations

import csv
import html
import os
import statistics
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
DATA_PATH = PROJECT_ROOT / "data" / "train_refined.csv"
HOST = os.environ.get("APP_HOST", "127.0.0.1")
PORT = int(os.environ.get("APP_PORT", "8000"))


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def count_words(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def compute_length_ratio_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    ratios: list[float] = []

    for row in rows:
        transliteration_word_count = count_words(row["transliteration"])
        if transliteration_word_count == 0:
            continue

        translation_word_count = count_words(row["translation"])
        ratios.append(translation_word_count / transliteration_word_count)

    if not ratios:
        return {
            "mean": 0.0,
            "median": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "max": 0.0,
            "lower_whisker": 0.0,
            "upper_whisker": 0.0,
            "left_outliers": 0,
            "right_outliers": 0,
        }

    sorted_ratios = sorted(ratios)
    median = statistics.median(sorted_ratios)
    if len(sorted_ratios) == 1:
        q1 = median
        q3 = median
    else:
        quartiles = statistics.quantiles(sorted_ratios, n=4, method="inclusive")
        q1 = quartiles[0]
        q3 = quartiles[2]

    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)

    non_outliers = [ratio for ratio in sorted_ratios if lower_fence <= ratio <= upper_fence]
    lower_whisker = non_outliers[0] if non_outliers else sorted_ratios[0]
    upper_whisker = non_outliers[-1] if non_outliers else sorted_ratios[-1]
    left_outliers = sum(1 for ratio in sorted_ratios if ratio < lower_fence)
    right_outliers = sum(1 for ratio in sorted_ratios if ratio > upper_fence)

    return {
        "mean": statistics.mean(sorted_ratios),
        "median": median,
        "variance": statistics.pvariance(sorted_ratios),
        "min": sorted_ratios[0],
        "q1": q1,
        "q3": q3,
        "max": sorted_ratios[-1],
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "left_outliers": left_outliers,
        "right_outliers": right_outliers,
    }


def render_ratio_boxplot(ratio_stats: dict[str, float]) -> str:
    width = 560
    height = 120
    plot_left = 52
    plot_right = width - 52
    line_y = 46
    box_top = 28
    box_height = 32
    minimum = ratio_stats["lower_whisker"]
    maximum = ratio_stats["upper_whisker"]

    def scale(value: float) -> float:
        if maximum <= minimum:
            return width / 2
        usable_width = plot_right - plot_left
        return plot_left + ((value - minimum) / (maximum - minimum)) * usable_width

    whisker_min_x = scale(ratio_stats["lower_whisker"])
    q1_x = scale(ratio_stats["q1"])
    median_x = scale(ratio_stats["median"])
    q3_x = scale(ratio_stats["q3"])
    whisker_max_x = scale(ratio_stats["upper_whisker"])
    left_outlier_x = 22
    right_outlier_x = width - 22
    left_outlier_html = ""
    right_outlier_html = ""
    if ratio_stats["left_outliers"]:
        left_outlier_html = (
            f'<circle cx="{left_outlier_x}" cy="{line_y}" r="5" class="boxplot-outlier" />'
            f'<text x="{left_outlier_x}" y="18" class="boxplot-label" text-anchor="middle">'
            f'{ratio_stats["left_outliers"]} low outlier(s)</text>'
        )
    if ratio_stats["right_outliers"]:
        right_outlier_html = (
            f'<circle cx="{right_outlier_x}" cy="{line_y}" r="5" class="boxplot-outlier" />'
            f'<text x="{right_outlier_x}" y="18" class="boxplot-label" text-anchor="middle">'
            f'{ratio_stats["right_outliers"]} high outlier(s)</text>'
        )

    return f"""
      <div class="boxplot-block">
        <h3>Ratio Box Plot(translation/transliteration)</h3>
        <svg viewBox="0 0 {width} {height}" class="boxplot" role="img" aria-label="translation/transliteration 単語数比の箱ひげ図">
          <line x1="{whisker_min_x:.2f}" y1="{line_y}" x2="{whisker_max_x:.2f}" y2="{line_y}" class="boxplot-line" />
          <line x1="{whisker_min_x:.2f}" y1="30" x2="{whisker_min_x:.2f}" y2="62" class="boxplot-line" />
          <line x1="{whisker_max_x:.2f}" y1="30" x2="{whisker_max_x:.2f}" y2="62" class="boxplot-line" />
          <rect x="{q1_x:.2f}" y="{box_top}" width="{max(q3_x - q1_x, 1):.2f}" height="{box_height}" class="boxplot-box" />
          <line x1="{median_x:.2f}" y1="{box_top}" x2="{median_x:.2f}" y2="{box_top + box_height}" class="boxplot-median" />
          {left_outlier_html}
          {right_outlier_html}
          <text x="{whisker_min_x:.2f}" y="82" class="boxplot-label" text-anchor="middle">whisker {ratio_stats["lower_whisker"]:.3f}</text>
          <text x="{q1_x:.2f}" y="100" class="boxplot-label" text-anchor="middle">Q1 {ratio_stats["q1"]:.3f}</text>
          <text x="{median_x:.2f}" y="82" class="boxplot-label" text-anchor="middle">med {ratio_stats["median"]:.3f}</text>
          <text x="{q3_x:.2f}" y="100" class="boxplot-label" text-anchor="middle">Q3 {ratio_stats["q3"]:.3f}</text>
          <text x="{whisker_max_x:.2f}" y="82" class="boxplot-label" text-anchor="middle">whisker {ratio_stats["upper_whisker"]:.3f}</text>
        </svg>
        <p class="boxplot-note">
          whisker は外れ値を除いた範囲です。全体の最小値は {ratio_stats["min"]:.3f}、最大値は {ratio_stats["max"]:.3f}。
        </p>
      </div>
    """


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
    ratio_stats: dict[str, float],
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

    boxplot_html = render_ratio_boxplot(ratio_stats)
    summary_html = f"""
      <section class="card summary-card">
        <p><strong>Same=No を含む文章数:</strong> {same_no_text_count}</p>
        {boxplot_html}
      </section>
    """
    if oare_id and not not_found:
        summary_html = f"""
          <section class="card summary-card">
            <p><strong>Same=No を含む文章数:</strong> {same_no_text_count}</p>
            <p><strong>選択中レコードの Same=No 件数:</strong> {current_same_no_count}</p>
            {boxplot_html}
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
              <td>{html.escape(entry["norm"])}</td>
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
                    <th>Norm</th>
                    <th>Lexeme (Dictionary Word)</th>
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

      .boxplot-block {{
        margin-top: 18px;
        padding-top: 16px;
        border-top: 1px solid var(--line);
      }}

      .boxplot-block h3 {{
        margin: 0 0 10px;
        font-size: 1rem;
      }}

      .boxplot {{
        width: 100%;
        height: auto;
        overflow: visible;
      }}

      .boxplot-line {{
        stroke: #7b6758;
        stroke-width: 2;
      }}

      .boxplot-box {{
        fill: rgba(159, 58, 36, 0.16);
        stroke: var(--accent);
        stroke-width: 2;
      }}

      .boxplot-median {{
        stroke: var(--accent-strong);
        stroke-width: 3;
      }}

      .boxplot-label {{
        fill: var(--muted);
        font-size: 12px;
        font-family: "SFMono-Regular", Consolas, monospace;
      }}

      .boxplot-outlier {{
        fill: var(--accent-strong);
        stroke: white;
        stroke-width: 1.5;
      }}

      .boxplot-note {{
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 0.92rem;
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
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
        gap: 12px;
        align-items: end;
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
            <label for="oare_id_input">oare_id を入力</label>
            <input
              id="oare_id_input"
              name="oare_id"
              type="text"
              value="{html.escape(oare_id)}"
              placeholder="例: 821b6253-72c8-43a5-93e1-0b40b5f9a7fb"
            >
          </div>
          <div>
            <label for="oare_id">oare_id</label>
            <select
              id="oare_id"
              onchange="if (this.value) {{ document.getElementById('oare_id_input').value = this.value; this.form.submit(); }}"
            >
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
    ratio_stats = compute_length_ratio_stats(rows)
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
        ratio_stats=ratio_stats,
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
