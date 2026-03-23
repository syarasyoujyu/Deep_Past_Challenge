# ByT5 Expansion + Distillation

`train.py` は、既存の `ByT5` checkpoint を teacher として固定し、そのモデルをベースに層を増やした student を追加学習するためのスクリプトです。

## 何をしているか

- `teacher`
  - 元の ByT5 checkpoint
  - 推論専用で固定
- `student`
  - teacher と同じ重みを引き継ぐ
  - encoder / decoder に追加 block を足す
  - 追加後に `cross-entropy + distillation` で学習する

狙いは、

- 元モデルの翻訳挙動をなるべく壊さない
- 追加した層に task-specific な表現力を持たせる
- 少量データでも full retrain より安定して伸ばす

ことです。

## 学習フロー

1. train CSV を読み込む
2. `model/byt5.py` と同じ source / target 正規化をかける
3. teacher 用の ByT5 をロードする
4. teacher config を複製し、`num_layers` と `num_decoder_layers` を増やして student config を作る
5. student model を新しく初期化する
6. teacher から student へ次をコピーする
   - shared embeddings
   - encoder の既存 block
   - decoder の既存 block
   - `lm_head`
   - final layer norm
7. 追加層は teacher の末尾 block を deepcopy して埋める
8. teacher を freeze し、student だけ更新する
9. `cross-entropy` と `KL distillation` を混ぜて最適化する

## Loss

概念的な loss は次です。

```text
total_loss = (1 - alpha) * CE(student, gold) + alpha * KL(student || teacher)
```

- `CE`
  - 正解 translation に対する通常の teacher-forced seq2seq loss
- `KL`
  - teacher logits と student logits の距離
- `alpha`
  - distillation の強さ
- `temperature`
  - logits をどれだけなだらかにして比較するか

## 層の増やし方

student は teacher をそのまま deep copy するのではなく、`teacher config` を拡張して新しく model を作り直します。

- encoder は `--extra-encoder-layers` 分だけ block を追加
- decoder は `--extra-decoder-layers` 分だけ block を追加
- 追加 block は teacher の末尾層からコピーして初期化

どの層をコピー元にするかは次で制御できます。

- `--copy-from-last-n-encoder-layers`
- `--copy-from-last-n-decoder-layers`

例えば `--copy-from-last-n-encoder-layers 2` なら、teacher の最後の 2 層を循環的に使って追加 encoder 層を埋めます。

## 更新するパラメータ

既定では student 全体を更新できますが、必要なら一部を固定できます。

- `--freeze-shared-embeddings`
- `--freeze-bottom-encoder-layers`
- `--freeze-bottom-decoder-layers`
- `--train-lm-head`
- `--train-final-layer-norms`

少量データでは、下位層をある程度固定した方が安定する場合があります。

## 主要引数

- `--model-name`
  - base model 名
- `--load-trained-model true --trained-model-path ...`
  - 既存 checkpoint を teacher の起点にしたいときに使う
- `--extra-encoder-layers`
  - encoder に追加する block 数
- `--extra-decoder-layers`
  - decoder に追加する block 数
- `--distill-alpha`
  - `CE` と `KL` の混合比
- `--distill-temperature`
  - distillation 温度
- `--teacher-device`
  - teacher を置く device

## 実行例

```bash
uv run python -m model.ft.distill.train \
  --train-path data/train.csv \
  --model-name google/byt5-small \
  --output-dir artifacts/byt5-small-expanded-distill \
  --extra-encoder-layers 2 \
  --extra-decoder-layers 2 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0
```

既学習 checkpoint を teacher にしたい場合:

```bash
uv run python -m model.ft.distill.train \
  --train-path data/train.csv \
  --load-trained-model true \
  --trained-model-path artifacts/byt5-small \
  --output-dir artifacts/byt5-small-expanded-distill \
  --extra-encoder-layers 1 \
  --extra-decoder-layers 1 \
  --distill-alpha 0.4 \
  --distill-temperature 2.0
```

## 保守的な開始設定

少量データなら、まずは次くらいが無難です。

- `--extra-encoder-layers 1`
- `--extra-decoder-layers 1`
- `--distill-alpha 0.3` から `0.5`
- `--distill-temperature 2.0`
- `--learning-rate 5e-5` 以下

追加層を増やしすぎると teacher から離れやすく、少量データでは崩れやすくなります。まずは 1-2 層ずつ増やして比較する方が安全です。
