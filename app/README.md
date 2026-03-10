# Translation Editor

`data/train_refined.csv` の `translation` を `oare_id` 単位で確認・編集するローカルアプリです。

## 起動方法

```bash
./.venv/bin/python app/server.py
```

ブラウザで `http://127.0.0.1:8000` を開いて使います。

ポートを変えたい場合は `APP_PORT=8080 ./.venv/bin/python app/server.py` のように起動できます。

## できること

- `oare_id` をプルダウンから選んで `transliteration` と `translation` を表示
- `transliteration` に含まれる `OA_Lexicon_eBL.csv` の `form` を検索
- 一致した `form` / `lexeme` / `type` / `Female(f)` を表示
- `lexeme` と `eBL_Dictionary.csv` の `word` から語義番号を除いた値を照合し、複数 `definition` を 1 カラムに `---------` 区切りで表示
- `Sentences_Oare_FirstWord_LinNum.csv` と `train.csv` を `text_uuid=oare_id` で照合し、`first_word_number` の単語一致と sentence translation を表示
- `Same=No` を含む文章数をアプリ上部に表示し、選択中レコードの `Same=No` 件数も表示
- `Prev` / `Next` で前後のレコードへ移動
- `translation` を編集して保存
- 保存内容を `data/train_refined.csv` に反映
