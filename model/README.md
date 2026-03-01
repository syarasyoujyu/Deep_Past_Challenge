# Model Pipeline

## Train

```bash
uv run python -m model.train \
  --model-name google/byt5-small \
  --output-dir artifacts/byt5-small \
  --wandb-project deep-past-challenge \
  --wandb-run-name byt5-small-baseline
```

`WANDB_API_KEY` を環境変数で渡すと `wandb` に自動で記録します。

Hugging Face Hub に push したい場合は `HF_TOKEN` を設定したうえで `--push-to-hub --hub-model-id <user>/<repo>` を付けてください。

## Predict

```bash
uv run python -m model.predict \
  --model-path artifacts/byt5-small \
  --submission-path submission.csv
```

`--model-path` にはローカル保存済みモデルだけでなく、Hugging Face Hub 上のモデルIDも指定できます。

