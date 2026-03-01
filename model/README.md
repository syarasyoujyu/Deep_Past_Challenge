# Model Pipeline

## Train

```bash
uv run python -m model.train \
  --model-name google/byt5-small \
  --output-dir artifacts/byt5-small \
  --fp16 true \
  --disable-wandb false \
  --wandb-project deep-past-challenge \
  --wandb-run-name byt5-small-baseline
```

Python は 3.12 以上を前提にしています。`transformers` の TensorFlow backend は無効化しているため、PyTorch + `Seq2SeqTrainer` ベースで学習します。

`WANDB_API_KEY` を環境変数で渡すと `wandb` に自動で記録します。

Hugging Face Hub に push したい場合は `HF_TOKEN` を設定したうえで `--push-to-hub true --hub-model-id <user>/<repo>` を付けてください。

## Train MarianMT Arabic-English

`Helsinki-NLP/opus-mt-ar-en` をアッカド語転写文から英語への翻訳タスクに fine-tune する場合は、専用スクリプトを使います。

```bash
uv run python -m model.train_marian_ar_en \
  --train-path data/train.csv \
  --model-name Helsinki-NLP/opus-mt-ar-en \
  --output-dir artifacts/opus-mt-ar-en \
  --learning-rate 2e-5 \
  --num-train-epochs 15 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --fp16 true \
  --disable-wandb false \
  --wandb-project deep-past-challenge \
  --wandb-run-name opus-mt-ar-en
```

このスクリプトは `MarianTokenizer` と `MarianMTModel` を使い、`attn_implementation=sdpa` を既定値にしています。必要なら `--torch-dtype float16` や `--attn-implementation eager` を上書きできます。

## Predict

```bash
uv run python -m model.predict \
  --model-path artifacts/byt5-small \
  --batch-size 16 \
  --submission-path submission.csv
```

推論も `Seq2SeqTrainer.predict()` を使います。`--model-path` にはローカル保存済みモデルだけでなく、Hugging Face Hub 上のモデルIDも指定できます。
