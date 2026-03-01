# Model Pipeline

## Train

```bash
uv run python -m model.train \
  --model-name google/gemma-2-27b-it \
  --output-dir artifacts/gemma-2-27b-it \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --num-train-epochs 3 \
  --bf16 false \
  --fp16 true \
  --use-lora true \
  --use-qlora true \
  --disable-wandb false \
  --wandb-project deep-past-challenge \
  --wandb-run-name gemma-2-27b-it-qlora
```

Python は 3.12 以上を前提にしています。`transformers` の TensorFlow backend は無効化しており、Gemma instruction-tuned causal LM の supervised fine-tuning を PyTorch + `Trainer` ベースで行います。

`train.py` は `google/gemma-2-27b-it` のような instruction-tuned Gemma を、モデルカードどおり `AutoTokenizer` と `AutoModelForCausalLM` で読み込み、チャットテンプレート経由で

- system: 翻訳器としての役割
- user: アッカド語転写文
- assistant: 英訳

の形式で supervised fine-tuning します。既定では LoRA + QLoRA を有効にしています。

LoRA / QLoRA の主要引数:

- `--use-lora true|false`
- `--use-qlora true|false`
- `--lora-r 64`
- `--lora-alpha 128`
- `--lora-dropout 0.05`
- `--lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `--bnb-4bit-quant-type nf4`
- `--bnb-4bit-use-double-quant true`
- `--bnb-4bit-compute-dtype bfloat16`

Gemma 2 / 3 系を扱うため `transformers>=4.50.0` を前提にしています。QLoRA を使う場合は `bitsandbytes` も別途必要です。

T4 15GB では `gemma-2-27b-it` はかなり厳しいので、まずは `--per-device-train-batch-size 1` を維持したまま `google/gemma-2-9b-it` や `google/gemma-2-2b-it` で動作確認するのが安全です。

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
