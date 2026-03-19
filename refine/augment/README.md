# Augment

`refine/augment` には、単文化した学習データをさらに増やすための段階的な augmentation スクリプトを置いています。

## Step 1

`build_train_refined_splitted.py`

- 入力: `data/now/train_refined_splitted.csv`
- 出力:
  - `data/now/train_refined_splitted_augmented_v1.csv`
  - `data/now/train_refined_splitted_augmented_v2.csv`
- 役割:
  - `v1`: `translation` が `.` で終わるところまでを 1 文として再結合する
  - `v2`: 文境界を無視して、連続する segment の window を列挙する
- `v2` は連結 segment 数と word 数に上限を掛ける

## Step 2

`build_pn_gn_swap_augment.py`

- 入力: `data/now/train_refined_splitted_augmented_v1.csv`
- 辞書: `data/OA_Lexicon_eBL_refined_with_definition.csv`
- 出力: `data/now/train_refined_splitted_augmented_v1_pn_gn_swap.csv`
- 役割:
  - transliteration 内の `PN` / `GN` を lexicon で検出する
  - 同じ type の別候補に置換する
  - translation 側に対応する英訳名が見つかるときだけ、translation も同時に置換する
  - 1 行あたり最大 3 件まで augmented row を作る
- translation 側に対応名が見つからない候補は捨てる

## Step 3

`build_item_swap_augment.py`

- 入力: `data/now/train_refined_splitted_augmented_v1.csv`
- 辞書: `data/OA_Lexicon_eBL_refined_with_definition.csv`
- 出力: `data/now/train_refined_splitted_augmented_v1_item_swap.csv`
- 役割:
  - 固有名詞ではなく、品目・素材名を置換する
  - 現在は保守的に `metal` と `textile` の一部だけを対象にする
  - 例:
    - `silver / gold / tin / copper`
    - `textile / garment / robe / shawl`
  - transliteration の token と translation の英語語彙を同時に差し替える
  - translation 側に対応語が無ければ置換しない
  - 1 行あたり最大 3 件まで augmented row を作る

## 方針

- まず `Step 1` で単文化・短い連結文を作る
- 次に `Step 2` で固有名詞のゆらぎを増やす
- さらに `Step 3` で品目・素材名のゆらぎを増やす

この順に増やすと、translation template を大きく壊さずにデータ件数を増やしやすいです。
