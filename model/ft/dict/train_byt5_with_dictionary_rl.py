from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import sacrebleu
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup

from model.byt5 import parse_bool, parse_interval_strategy, parse_optional_torch_dtype
from model.common import ROOT_DIR, build_generation_config, load_tokenizer, seed_everything
from model.ft.dict.train_byt5_with_dictionary import (
    load_dictionary_entries,
    load_model,
    prepare_frame,
    preview_augmented_examples,
    resolve_model_source,
)


DEFAULT_MODEL_NAME = "google/byt5-small"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "byt5-small-dict-rl"
DEFAULT_TRAIN_PATH = (
    ROOT_DIR
    / "data"
    / "supplement"
    / "Michel_Old_Assyrian_Letters_Corpus"
    / "train_refined_v2_sentence_split_refined_refined.csv"
)
DEFAULT_LEXICON_PATH = ROOT_DIR / "data" / "OA_Lexicon_eBL_refined_with_definition.csv"
DEFAULT_GLOSS_PATH = ROOT_DIR / "data" / "now" / "train_openai_gloss_compact.jsonl"
DEFAULT_ONOMASTICON_PATH = ROOT_DIR / "data" / "onomasticon.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental RL fine-tuning for ByT5 with dictionary-augmented inputs. "
            "Uses a mixed loss: cross-entropy + self-critical sequence training reward."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--load-trained-model", type=parse_bool, default=False)
    parser.add_argument("--trained-model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--gloss-path", type=Path, default=DEFAULT_GLOSS_PATH)
    parser.add_argument("--include-onomasticon", type=parse_bool, default=False)
    parser.add_argument("--onomasticon-path", type=Path, default=DEFAULT_ONOMASTICON_PATH)
    parser.add_argument("--source-prefix", type=str, default="translate Akkadian to English with dictionary: ")
    parser.add_argument("--hint-placement", choices=["prepend", "append"], default="prepend")
    parser.add_argument("--max-dictionary-hints", type=int, default=8)
    parser.add_argument("--max-entry-token-length", type=int, default=4)
    parser.add_argument("--max-hint-words", type=int, default=6)
    parser.add_argument("--max-gloss-variants", type=int, default=4)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--dtype", dest="dtype", type=parse_optional_torch_dtype, default=None)
    parser.add_argument("--torch-dtype", dest="dtype", type=parse_optional_torch_dtype)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--normalize-source", type=parse_bool, default=True)
    parser.add_argument("--normalize-target", type=parse_bool, default=True)
    parser.add_argument("--normalize-h", type=parse_bool, default=True)
    parser.add_argument("--normalize-subscripts", type=parse_bool, default=True)
    parser.add_argument("--normalize-breaks", type=parse_bool, default=True)
    parser.add_argument("--remove-editorial-marks", type=parse_bool, default=True)
    parser.add_argument("--strip-word-dividers", type=parse_bool, default=False)
    parser.add_argument("--disable-wandb", type=parse_bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="deep-past-challenge")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--reward-bleu-weight", type=float, default=0.5)
    parser.add_argument("--reward-chrfpp-weight", type=float, default=0.5)
    parser.add_argument("--rl-loss-weight", type=float, default=0.2)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--baseline-num-beams", type=int, default=1)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--save-strategy", type=parse_interval_strategy, default="epoch")
    parser.add_argument("--eval-strategy", type=parse_interval_strategy, default="epoch")
    return parser.parse_args()


def setup_reporting(args: argparse.Namespace) -> list[str]:
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    return [value.strip() for value in args.report_to.split(",") if value.strip()]


def maybe_init_wandb(args: argparse.Namespace):
    if args.disable_wandb:
        return None

    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "--disable-wandb false was specified, but wandb is not installed."
        ) from error

    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
    )
    return run


def build_records(frame) -> list[dict[str, str]]:
    return frame.loc[:, ["augmented_source", "translation"]].to_dict("records")


def collate_records(records: list[dict[str, str]], tokenizer, args, device: torch.device) -> dict[str, torch.Tensor]:
    sources = [record["augmented_source"] for record in records]
    targets = [record["translation"] for record in records]

    model_inputs = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_source_length,
    )
    label_batch = tokenizer(
        text_target=targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_target_length,
    )
    labels = label_batch["input_ids"]
    if tokenizer.pad_token_id is not None:
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    model_inputs["labels"] = labels.to(device)
    model_inputs["references"] = targets
    return model_inputs


def sentence_reward(reference: str, hypothesis: str, args: argparse.Namespace) -> float:
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score / 100.0
    chrfpp = sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score / 100.0
    return (args.reward_bleu_weight * bleu) + (args.reward_chrfpp_weight * chrfpp)


def compute_rewards(references: list[str], hypotheses: list[str], args: argparse.Namespace) -> torch.Tensor:
    rewards = [sentence_reward(reference, hypothesis, args) for reference, hypothesis in zip(references, hypotheses, strict=True)]
    return torch.tensor(rewards, dtype=torch.float32)


def compute_sample_log_probs(generation_output, tokenizer, device: torch.device) -> torch.Tensor:
    sequences = generation_output.sequences
    scores = generation_output.scores
    batch_size = sequences.size(0)
    if not scores:
        return torch.zeros(batch_size, device=device)

    token_ids = sequences[:, 1 : 1 + len(scores)]
    log_probs = torch.zeros(batch_size, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_token_id = tokenizer.eos_token_id

    for step_index, step_scores in enumerate(scores):
        step_log_probs = torch.log_softmax(step_scores, dim=-1)
        step_token_ids = token_ids[:, step_index]
        step_token_log_probs = step_log_probs.gather(1, step_token_ids.unsqueeze(1)).squeeze(1)
        active_mask = ~finished
        log_probs = log_probs + (step_token_log_probs * active_mask)
        if eos_token_id is not None:
            finished = finished | (step_token_ids == eos_token_id)

    return log_probs


def save_model(output_dir: Path, model, tokenizer, metrics: dict[str, float] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if metrics is not None:
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def evaluate_model(model, tokenizer, records: list[dict[str, str]], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if not records:
        return {}

    model.eval()
    references = [record["translation"] for record in records]
    predictions: list[str] = []
    generation_config = build_generation_config(args)

    for start in range(0, len(records), args.per_device_eval_batch_size):
        batch_records = records[start : start + args.per_device_eval_batch_size]
        sources = [record["augmented_source"] for record in batch_records]
        encoded = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_config)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    chrfpp_result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    bleu = float(bleu_result.score)
    chrfpp = float(chrfpp_result.score)
    reward_mean = float(
        compute_rewards(references, predictions, args).mean().item()
    )
    geometric_mean = math.sqrt(bleu * chrfpp)
    return {
        "_bleu": round(bleu, 4),
        "chrf++": round(chrfpp, 4),
        "reward_mean": round(reward_mean, 6),
        "_bleu_chrfpp_geometric_mean": round(geometric_mean, 4),
    }


def build_optimizer(model, args: argparse.Namespace):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optim.lower() == "adafactor":
        return Adafactor(
            trainable_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )
    return AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_reporting(args)
    wandb_run = maybe_init_wandb(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_source = resolve_model_source(args)
    print(f"Loading ByT5 from: {model_source}")
    tokenizer = load_tokenizer(model_source)

    dictionary_index = load_dictionary_entries(args)
    frame = prepare_frame(args, dictionary_index, tokenizer)
    preview_augmented_examples(frame, args.preview_count)
    print(
        "Dictionary coverage: "
        f"{int((frame['dictionary_hint_count'] > 0).sum())}/{len(frame)} rows "
        f"have at least one hint."
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "dataset/row_count": len(frame),
                "dataset/dictionary_coverage_count": int((frame["dictionary_hint_count"] > 0).sum()),
                "dataset/dictionary_coverage_ratio": float((frame["dictionary_hint_count"] > 0).mean()),
                "dataset/mean_dictionary_hint_count": float(frame["dictionary_hint_count"].mean()),
            },
            step=0,
        )

    train_frame, val_frame = train_test_split(
        frame,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    ) if args.val_size > 0 else (frame, None)

    train_records = build_records(train_frame.reset_index(drop=True))
    val_records = [] if val_frame is None else build_records(val_frame.reset_index(drop=True))

    model = load_model(args)

    use_cpu = not torch.cuda.is_available()
    device = torch.device("cpu" if use_cpu else "cuda")
    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_loader = DataLoader(
        train_records,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_records(batch, tokenizer, args, device),
    )
    optimizer = build_optimizer(model, args)
    total_updates = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)) * int(args.num_train_epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )

    best_metric = float("-inf")
    global_step = 0
    running_loss = 0.0
    running_ce_loss = 0.0
    running_rl_loss = 0.0

    for epoch_index in range(int(math.ceil(args.num_train_epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step_index, batch in enumerate(train_loader, start=1):
            references = batch.pop("references")

            outputs = model(**batch)
            ce_loss = outputs.loss

            with torch.inference_mode():
                baseline_sequences = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=args.max_target_length,
                    num_beams=args.baseline_num_beams,
                    do_sample=False,
                )
            baseline_predictions = [
                text.strip()
                for text in tokenizer.batch_decode(baseline_sequences, skip_special_tokens=True)
            ]

            sampled_output = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.max_target_length,
                do_sample=True,
                top_p=args.sample_top_p,
                temperature=args.sample_temperature,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sampled_predictions = [
                text.strip()
                for text in tokenizer.batch_decode(sampled_output.sequences, skip_special_tokens=True)
            ]

            sample_rewards = compute_rewards(references, sampled_predictions, args).to(device)
            baseline_rewards = compute_rewards(references, baseline_predictions, args).to(device)
            advantages = sample_rewards - baseline_rewards
            sequence_log_probs = compute_sample_log_probs(sampled_output, tokenizer, device)
            rl_loss = -(advantages.detach() * sequence_log_probs).mean()

            loss = (args.ce_loss_weight * ce_loss) + (args.rl_loss_weight * rl_loss)
            loss = loss / max(args.gradient_accumulation_steps, 1)
            loss.backward()

            running_loss += float(loss.item())
            running_ce_loss += float(ce_loss.item())
            running_rl_loss += float(rl_loss.item())

            if step_index % args.gradient_accumulation_steps == 0 or step_index == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    log_payload = {
                        "train/loss": running_loss / args.logging_steps,
                        "train/ce_loss": running_ce_loss / args.logging_steps,
                        "train/rl_loss": running_rl_loss / args.logging_steps,
                        "train/epoch": epoch_index + 1,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    print(
                        f"[train] epoch={epoch_index + 1} step={global_step} "
                        f"loss={log_payload['train/loss']:.6f} "
                        f"ce={log_payload['train/ce_loss']:.6f} "
                        f"rl={log_payload['train/rl_loss']:.6f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=global_step)
                    running_loss = 0.0
                    running_ce_loss = 0.0
                    running_rl_loss = 0.0

        if val_records:
            train_eval_metrics = evaluate_model(model, tokenizer, train_records, args, device)
            print(f"[train_eval] epoch={epoch_index + 1} metrics={train_eval_metrics}")
            eval_metrics = evaluate_model(model, tokenizer, val_records, args, device)
            print(f"[eval] epoch={epoch_index + 1} metrics={eval_metrics}")
            if wandb_run is not None:
                wandb_run.log(
                    {f"train_eval/{key}": value for key, value in train_eval_metrics.items()} | {"train_eval/epoch": epoch_index + 1},
                    step=global_step,
                )
                wandb_run.log(
                    {f"eval/{key}": value for key, value in eval_metrics.items()} | {"eval/epoch": epoch_index + 1},
                    step=global_step,
                )
            current_metric = eval_metrics.get("_bleu_chrfpp_geometric_mean", float("-inf"))
            if current_metric > best_metric:
                best_metric = current_metric
                save_model(args.output_dir / "best_model", model, tokenizer, eval_metrics)
        else:
            train_eval_metrics = evaluate_model(model, tokenizer, train_records, args, device)
            print(f"[train_eval] epoch={epoch_index + 1} metrics={train_eval_metrics}")
            if wandb_run is not None:
                wandb_run.log(
                    {f"train_eval/{key}": value for key, value in train_eval_metrics.items()} | {"train_eval/epoch": epoch_index + 1},
                    step=global_step,
                )

        checkpoint_dir = args.output_dir / f"checkpoint-epoch-{epoch_index + 1}"
        save_model(checkpoint_dir, model, tokenizer)

    save_model(args.output_dir, model, tokenizer)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
