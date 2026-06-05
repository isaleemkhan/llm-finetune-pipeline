"""
Merge LoRA adapter weights back into the base model to produce a standalone,
full-precision model that can be served directly (e.g. by vLLM) without PEFT.

Usage:
    python serving/merge_weights.py \
        --checkpoint ./checkpoints/final \
        --base meta-llama/Meta-Llama-3-8B \
        --output ./merged
"""

import argparse
import json
import os


def detect_base(checkpoint: str) -> str:
    cfg_path = os.path.join(checkpoint, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--checkpoint", required=True, help="LoRA adapter directory.")
    parser.add_argument("--base", help="Base model id (auto-detected from adapter if omitted).")
    parser.add_argument("--output", required=True, help="Output directory for merged model.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = args.base or detect_base(args.checkpoint)
    if not base:
        raise ValueError("Could not determine base model; pass --base explicitly.")

    print(f"Loading base model: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"Applying LoRA adapter from: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    tok_src = args.checkpoint if os.path.exists(
        os.path.join(args.checkpoint, "tokenizer_config.json")
    ) else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.output)

    print(f"Merged model saved to: {args.output}")


if __name__ == "__main__":
    main()
