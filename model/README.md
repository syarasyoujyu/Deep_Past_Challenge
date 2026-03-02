# Model Pipeline

## Train

```bash
uv run python -m model.train \
  --model-name unsloth/gemma-2-9b-bnb-4bit \
  --output-dir artifacts/gemma-2-9b-unsloth \
  --preprocess-num-proc auto \
  --dataloader-num-workers 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --num-train-epochs 3 \
  --bf16 false \
  --fp16 true \
  --use-lora true \
  --use-qlora true \
  --max-seq-length 512 \
  --disable-wandb false \
  --wandb-project deep-past-challenge \
  --wandb-run-name gemma-2-9b-unsloth
```

Python は 3.12 以上を前提にしています。`transformers` の TensorFlow backend は無効化しており、Unsloth の `FastLanguageModel` と `trl.SFTTrainer` ベースで supervised fine-tuning を行います。

`train.py` は `unsloth/gemma-2-9b-bnb-4bit` のような Unsloth の事前量子化モデルを読み込み、prompt-completion 形式で

- prompt: 翻訳器としての役割 + アッカド語転写文
- completion: 英訳

の形式で学習します。既定では LoRA + QLoRA を有効にしており、`train_on_inputs=false` のときは completion 部分だけで loss を計算します。

学習後の validation 生成では、次の指標を保存します。

- `BLEU`
- `ROUGE-2 Precision`
- `ROUGE-2 Recall`
- `ROUGE-2 F1`
- `BERTScore`
- `chrF++`
- `Geometric Mean of BLEU and chrF++`

Unsloth / LoRA の主要引数:

- `--use-lora true|false`
- `--use-qlora true|false`
- `--lora-r 16`
- `--lora-alpha 16`
- `--lora-dropout 0.0`
- `--lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `--packing true|false`
- `--train-on-inputs true|false`
- `--eval-max-new-tokens 256`
- `--bertscore-model-type <optional>`

Notebook 相当の環境として `unsloth` と `trl>=0.22.2` を前提にしています。Colab では Unsloth 側の推奨 install 手順に合わせて依存を入れる方が安定します。

T4 15GB では `unsloth/gemma-2-9b-bnb-4bit` でも `max_seq_length` を大きくすると厳しくなります。まずは `--per-device-train-batch-size 1` と `--max-seq-length 512` から始めるのが安全です。

前処理を速くしたい場合は `--preprocess-num-proc`、学習時の DataLoader 並列度は `--dataloader-num-workers` で調整できます。`auto` を渡すと `os.cpu_count() - 1` を使います。

`WANDB_API_KEY` を環境変数で渡すと `wandb` に自動で記録します。

Hugging Face Hub に push したい場合は `HF_TOKEN` を設定したうえで `--push-to-hub true --hub-model-id <user>/<repo>` を付けてください。保存されるのは Unsloth/PEFT の学習済みモデルです。

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
