"""
Main QLoRA fine-tuning script for LLaMA 3 using HuggingFace TRL's SFTTrainer.

Reads defaults from training/config.yaml; any CLI flag overrides the file.

Usage:
    python training/train.py \
        --model meta-llama/Meta-Llama-3-8B \
        --data data/processed/train.jsonl \
        --output ./checkpoints \
        --epochs 3 --batch_size 4 --lora_r 16
"""

import argparse
import os

import yaml

from lora_config import build_bnb_config, build_lora_config


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model:
        cfg["model"]["name"] = args.model
    if args.data:
        cfg["data"]["train_file"] = args.data
    if args.eval_data:
        cfg["data"]["eval_file"] = args.eval_data
    if args.output:
        cfg["training"]["output_dir"] = args.output
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--eval_data")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    model_cfg, lora_cfg, data_cfg, train_cfg, track_cfg = (
        cfg["model"], cfg["lora"], cfg["data"], cfg["training"], cfg["tracking"]
    )

    # Imports are local so --help works without a GPU / heavy deps installed.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    print(f"Loading base model: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=model_cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = build_bnb_config(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if train_cfg.get("bf16") else torch.float16,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
        )

    peft_config = build_lora_config(lora_cfg)

    data_files = {"train": data_cfg["train_file"]}
    if data_cfg.get("eval_file") and os.path.exists(data_cfg["eval_file"]):
        data_files["validation"] = data_cfg["eval_file"]
    dataset = load_dataset("json", data_files=data_files)

    report_to = track_cfg.get("report_to", "none")
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", track_cfg.get("wandb_project", "llm-finetune"))

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_steps=train_cfg.get("eval_steps", 200),
        eval_strategy="steps" if "validation" in dataset else "no",
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dataset_text_field=data_cfg.get("text_field", "text"),
        seed=train_cfg.get("seed", 42),
        report_to=report_to if report_to != "none" else [],
        run_name=track_cfg.get("run_name", "llama3-qlora"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = os.path.join(train_cfg["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
