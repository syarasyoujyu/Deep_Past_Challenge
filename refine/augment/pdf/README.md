# AKT 6a PDF Extractor

`AKT 6a.pdf` の P52 以降にある二段組の対訳表から、`oare_id`, `transliteration`, `translation` を抜くための抽出スクリプトです。

対象 PDF:

- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6a.pdf`

実装:

- [extract_akt6a_parallel_table.py](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table.py)
- [extract_akt6a_parallel_table.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table.sh)
- [extract_akt6a_parallel_table_openai.py](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table_openai.py)
- [extract_akt6a_parallel_table_openai.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table_openai.sh)
- [extract_akt6a_line_ocr_openai.py](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_line_ocr_openai.py)
- [extract_akt6a_line_ocr_openai.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_line_ocr_openai.sh)

## どちらを使うか

- PDF の文字層をそのまま使う軽量版:
  [extract_akt6a_parallel_table.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table.sh)
- PDF のページ画像を OpenAI Vision に読ませる高品質版:
  [extract_akt6a_parallel_table_openai.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_parallel_table_openai.sh)
- 既存の行位置推定で row crop を切り出し、その 1 行画像を OpenAI に読ませる高品質版:
  [extract_akt6a_line_ocr_openai.sh](/home/watas/kaggle/Deep_Past_Challenge/refine/augment/pdf/extract_akt6a_line_ocr_openai.sh)

文字化けや特殊記号の崩れが強い場合は、高品質版を使う前提です。
特に「ページ全体では alignment がずれるが、行位置推定はかなり合っている」ケースでは line OCR 版を優先します。

## 実行

```bash
./refine/augment/pdf/extract_akt6a_parallel_table.sh
```

主な出力:

- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines.csv`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_sentences.csv`

任意のページ範囲だけ試す場合:

```bash
./refine/augment/pdf/extract_akt6a_parallel_table.sh \
  --start-page 52 \
  --end-page 60 \
  --output-dir /tmp/akt6a \
  --output-file akt6a_segments.csv \
  --line-output-file akt6a_lines.csv
```

row crop 画像も保存したい場合:

```bash
uv run --with pdfplumber --with pypdfium2 --with pillow \
  python refine/augment/pdf/extract_akt6a_parallel_table.py \
  --start-page 52 \
  --end-page 55 \
  --save-row-images
```

この場合、line CSV に `row_image_path`, `left_image_path`, `right_image_path` も出ます。
画像は既定で `<output-dir>/image` に保存します。

軽量版の出力先も `--output-dir` 基準で、ファイル名は
`--output-file` と `--line-output-file` で指定します。

## 高品質版の実行

```bash
./refine/augment/pdf/extract_akt6a_parallel_table_openai.sh
```

この `.sh` は既定で `6a` を対象にします。
`a|b|c|d|e` を先頭引数に渡すと `AKT 6a` から `AKT 6e` を切り替えられます。
加えて、特殊文字やページ端の行の視認性を少し上げるため `--render-scale 3.0` と `--image-padding-px 96` を既定で付けています。
別の範囲や設定を試したい場合は edition の後ろに追加引数で上書きしてください。
checkpoint を無視して前処理時間・request 時間・実費を取り直したい場合は `--overwrite` を付けてください。

例:

```bash
./refine/augment/pdf/extract_akt6a_parallel_table_openai.sh b
./refine/augment/pdf/extract_akt6a_parallel_table_openai.sh e --max-workers 10
./refine/augment/pdf/extract_akt6a_parallel_table_openai.sh all
```

補足:

- `--max-workers` は OpenAI request の同時実行数の上限です。
- 進捗は `tqdm` で「何ページ中の何ページまで終わったか」を表示します。
- checkpoint 済みのページがある場合は、その分を初期進捗に含めます。
- 実行前に概算コストを先に表示し、その後で `tqdm` に概算 cumulative / projected cost を載せて進捗可視化します。

高品質版の主な出力:

- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_openai.csv`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_openai_pages.jsonl`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_openai_metrics.csv`

高品質版の出力先も `--output-dir` 基準で、ファイル名は
`--output-file`, `--checkpoint-file`, `--metrics-file` で指定します。

`.env` または環境変数に `OPENAI_API_KEY` が必要です。

## 高品質版の読み取りフロー

1. `pypdfium2` で PDF ページを高解像度 PNG にレンダリングします。
   PDF の文字層ではなく、紙面画像そのものを Vision 入力にします。

2. 各ページ画像を OpenAI Responses API に送ります。
   request は並列に投げられますが、`--prompt-tokens-per-minute` と推定 input token に基づく token bucket で抑制します。

3. モデルには JSON schema つきで、以下を返すよう指示します。

- `doc_label`
- `has_parallel_table`
- `rows[{index, transliteration, translation}]`

4. プロンプトでは次を明示しています。

- 左列は transliteration
- 右列は translation
- 中央の `5`, `10`, `15`, `e.` などは無視
- `40 e.`, `1.e.`, `l.e.` など中央列 marker が OCR されても row text には入れない
- 左右どちらの列も、折り返しや短い行は各 half-page の中央寄りに出ても本文として拾う
- `Note:` / `Comment:` 本文は無視する
- ただし同一ページ内でその後に翻訳表が再開するなら、その後半ブロックも拾う
- 対訳表の物理行だけを上から順に返す
- 同じ印刷行に片側しか無い場合は、もう片側を空文字のまま残す
- 前後の行から text を借りて無理に alignment しない
- `˹`, `˺`, `á`, `à`, `é`, `è`, `í`, `ì`, `ú`, `ù`, `a₂`, `a₃`, `e₂`, `e₃`, `i₂`, `i₃`, `u₂`, `u₃`, `₀-₉`, `ₓ`, `š`, `Š`, `ṣ`, `Ṣ`, `ṭ`, `Ṭ`, `ḫ`, `Ḫ`, `ʾ`, `‘` などの特殊文字を見えている限り保持する

5. ページごとの JSON を checkpoint に保存します。
   `akt6a_parallel_openai_pages.jsonl` が再開用です。

6. 最後に全ページ分をまとめて CSV にします。
   `doc_label` がページに無い continuation page では、直前の `doc_label` を引き継いで `oare_id` を作ります。
   `oare_id` は `9f868ebe-6392-cf33-c6c6-5d9bc2297825` のような UUID 形式で、再実行しても同じ入力行からは同じ ID が出るようにしています。

7. ページごとの処理メトリクスを保存します。
   `akt6a_parallel_openai_metrics.csv` には以下を出します。

- 前処理時間: PDF ページのレンダリングと data URL 化にかかった秒数
- request 時間: OpenAI Responses API の往復時間
- total 時間: そのページ全体の処理時間
- input / output token 数
- cached input token 数
- ページごとの実費換算: `input_cost_usd`, `cached_input_cost_usd`, `output_cost_usd`, `total_cost_usd`
- `text_input_tokens` / `image_input_tokens` も保存し、概算の `estimated_text_input_cost_usd` / `estimated_image_input_cost_usd` も出します

実行中の標準出力では、開始前の cost estimate を出したあと、`tqdm` で進捗と概算 cumulative / projected cost を表示します。

並列度の主な調整引数:

- `--preprocess-workers`: PDF レンダリング並列数
- `--max-workers`: OpenAI request 並列数
- `--prompt-tokens-per-minute`: request スケジューリングで使う入力 token/分の上限
- `--estimated-input-tokens-per-page`: 1 ページあたりの推定 input token 数

並列化の注意:

- OpenAI request 側はスレッド並列で問題ありませんが、PDF 前処理側は `pypdfium2` のページ読み込みがスレッド並列だと不安定です。
- 実際に `Failed to load page.` が出ることがあるため、前処理は `ThreadPoolExecutor` ではなく `ProcessPoolExecutor` で並列化しています。
- そのため `--preprocess-workers` は「プロセス数」、`--max-workers` は「OpenAI request の並列数」として別物です。

## Line OCR 版の実行

```bash
./refine/augment/pdf/extract_akt6a_line_ocr_openai.sh
```

この `.sh` は既定で次を順に実行します。

1. `extract_akt6a_parallel_table.py` で P52-P55 の行位置を検出
2. `--save-row-images` で各行の full row / left / right crop を保存
3. `extract_akt6a_line_ocr_openai.py` で、前段が出した `akt6a_parallel_lines.csv` を読みつつ各行画像を 1 行ずつ OCR

既定の line OCR 実行設定:

- OpenAI request は並列実行します
- 既定の `--prompt-tokens-per-minute` は `100000`
- 既定の `--max-workers` は `8`
- 進捗は `tqdm` で表示します

主な出力:

- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines.csv`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/image/...`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines_openai.csv`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines_openai.jsonl`
- `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines_openai_metrics.csv`

line OCR 版の狙い:

- ページ全体 OCR では前後の行を混ぜてしまう問題を避ける
- 既にうまく出ている「行位置推定」をそのまま使う
- OCR 本体は full row 画像を source of truth にして、左右の対応関係を崩しにくくする

補足:

- `left_image_path` と `right_image_path` は保存されますが、現在の line OCR request では使っていません。
- これは右列だけの crop を単独で見せると alignment が不安定になりやすいためです。
- 現在は row 全体を見て 1 行として読む方式を既定にしています。

`akt6a_parallel_lines_openai.csv` には OpenAI OCR の結果に加えて、
元のヒューリスティック抽出値を `detected_transliteration`, `detected_translation` として残します。

ページ範囲を絞りたい場合は、`extract_akt6a_line_ocr_openai.py` に
`--start-page` / `--end-page` を付けると、input CSV の `pdf_page` で対象行を絞れます。

出力先は `--output-dir` を基準にし、ファイル名は
`--output-file`, `--checkpoint-file`, `--metrics-file` で指定します。

line OCR 実行時の cost 表示:

- 実行開始前に `total_rows`, `completed_rows`, `pending_rows` を表示します。
- あわせて、checkpoint の平均実費があればそれを使い、なければ既定 token 見積もりを使って `projected_total` を先に出します。
- その後に各行の実費を出すので、走らせる前にだいたいの総額を把握できます。

line OCR 実行時の並列化:

- request は `ThreadPoolExecutor` で並列に投げます。
- client 側では `TokenBucket` で prompt input token を制御し、既定で `100000 token/min` を超えないようにしています。
- `tqdm` には checkpoint 済み行も含めた全体進捗を出します。
- 行ごとの詳細ログは出さず、進捗は `tqdm` に集約しています。

## 読み取りフロー

1. `pdfplumber` で PDF の文字と座標を取得します。
   各単語について `x0`, `top` などの位置情報を取り、OCR ではなく PDF の文字層を直接使います。

2. ページ内の単語を y 座標でまとめて「行」にします。
   `--y-tolerance` の範囲内にある単語を同じ行として束ねます。

3. 行を 3 つの帯に分けます。
   左帯を transliteration、中央帯を行番号や補助マーカー、右帯を translation 候補として扱います。

4. 左帯の transliteration にだけ軽い字形補正をかけます。
   PDF の italic フォントで `{` が `i`、語頭の `1-` が `I-`、`!Star` が `IStar` のように崩れるケースがあるため、明らかに一貫しているものだけ補正します。

5. ページ上部のタイトルを拾います。
   例: `1. kt 94/k 1263`
   これを `doc_label` として保持し、後で `oare_id` の一部にも使います。

6. 各行が「対訳表の行っぽいか」を判定します。
   判定には以下を使っています。

- 左側に `DUMU`, `IGI`, `KI`, `URUDU`, `GIN` など transliteration で出やすい記号があるか
- 左側に `a-na`, `i-na`, `ma-na` やハイフン連結語があるか
- 右側に英語 stopword があるか
- 中央帯が `5`, `10`, `15`, `e.` などの行マーカーか

7. `Note:` / `Comment:` 本文は除外します。
   ただし同じページ内で後半に翻訳表が再開する場合は、そのブロックも再び拾います。
   つまり「注記本文を飛ばす」が正しく、「以降すべてを捨てる」ではありません。

8. 行レベル CSV を出します。
   `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_lines.csv` は、PDF 上の 1 行に近い粒度です。
   片側が薄い行や、中段の行番号を含むケースも残るので、まず確認用・後段整形用の出力として使います。

## CSV/PDF Editor

`parallel_table_editor.py` は、左に PDF ページ、右にそのページの CSV 行を並べて確認・編集するローカルアプリです。

- ページ指定で `pdf_page` が一致する行だけを右側に表示
- `transliteration` / `translation` / `oare_id` を編集可能
- 各 row の `Add below` で任意位置の直後へ新規行追加
- `Delete row` で行削除
- `Save` でそのページの行だけを CSV に反映

既定対象:

- CSV: `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_openai.csv`
- PDF: `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6a.pdf`

起動:

```bash
bash refine/augment/pdf/parallel_table_editor.sh
```

ブラウザで `http://127.0.0.1:8010` を開いて使います。

必要なら別ファイルにも向けられます。

```bash
bash refine/augment/pdf/parallel_table_editor.sh \
  --csv-path data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6b/akt6b_parallel_openai.csv \
  --pdf-path "data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/AKT 6b.pdf" \
  --port 8011
```

## Sentence Split

`build_parallel_openai_sentence_split.py` は `akt6a_parallel_openai.csv` の行を文単位へまとめます。
`data/train_refined_v2_sentence_split.csv` に近い形で、追加で `page_start` / `page_end` を持たせます。

```bash
bash refine/augment/pdf/build_parallel_openai_sentence_split.sh
```

9. 行を軽く連結して segment レベル CSV を出します。
   `data/supplement/Old_Assyrian_Kültepe_Tablets_in_PDF/data/AKT_6a/akt6a_parallel_sentences.csv` は、両側が揃っている行を基本単位にしつつ、片側だけの続き行を直前へ吸わせた近似出力です。
   厳密な sentence split ではなく、学習用に扱いやすい bilingual segment を作るためのヒューリスティックです。

## なぜ行番号やページ番号をそのまま使わないか

PDF の中央帯には以下が混ざります。

- 行番号: `5`, `10`, `15`, `20`, `e.`
- ページやテキスト番号
- レイアウト由来の断片

これをそのまま `translation` や `transliteration` に混ぜると学習に不向きなので、中央帯は基本的に本文から外しています。

## 制約

- 完全な sentence alignment を保証するものではありません。
- PDF 文字層の乱れで、一部の文字や語が崩れることがあります。
- 対訳表の continuation page と Note/Comment の境界はヒューリスティックです。
- OpenAI Vision 版は API コストがかかります。
- OpenAI Vision 版でも、判読困難な箇所は誤読や欠落が残る可能性があります。

そのため、まずは `akt6a_parallel_lines.csv` を見て品質確認し、その後必要なら追加の正規化や手直しを入れる前提です。
