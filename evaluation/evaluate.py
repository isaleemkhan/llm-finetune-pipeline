"""
Evaluation suite: generates predictions from a fine-tuned model (base + LoRA
adapter or a merged model) over a benchmark file and reports ROUGE, BLEU, and
perplexity.

Benchmark format (JSONL), one example per line:
    {"instruction": "...", "input": "...", "output": "<reference answer>"}

Usage:
    python evaluation/evaluate.py --checkpoint ./checkpoints/final
    python evaluation/evaluate.py --checkpoint ./checkpoints/final \
        --base meta-llama/Meta-Llama-3-8B \
        --benchmark evaluation/benchmarks/sample_benchmark.jsonl
"""

import argparse
import json
import math
import os
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(ex: Dict) -> str:
    inp = ex.get("input", "")
    input_block = f"### Input:\n{inp}\n\n" if inp else ""
    return f"### Instruction:\n{ex['instruction']}\n\n{input_block}### Response:\n"


def load_model(checkpoint: str, base: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_adapter = os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    tokenizer_src = checkpoint if os.path.exists(
        os.path.join(checkpoint, "tokenizer_config.json")
    ) else (base or checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        if not base:
            raise ValueError("Checkpoint is a LoRA adapter; pass --base <base model id>.")
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def compute_perplexity(model, tokenizer, texts: List[str]) -> float:
    import torch

    losses = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        if not math.isnan(loss.item()):
            losses.append(loss.item())
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument("--checkpoint", required=True, help="Adapter dir or merged model dir.")
    parser.add_argument("--base", help="Base model id (required if checkpoint is a LoRA adapter).")
    parser.add_argument(
        "--benchmark",
        default=os.path.join(os.path.dirname(__file__), "benchmarks", "sample_benchmark.jsonl"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    import evaluate as hf_evaluate

    examples = read_jsonl(args.benchmark)
    print(f"Loaded {len(examples)} benchmark examples.")

    model, tokenizer = load_model(args.checkpoint, args.base)

    predictions, references = [], []
    for i, ex in enumerate(examples, 1):
        pred = generate(model, tokenizer, build_prompt(ex), args.max_new_tokens)
        predictions.append(pred)
        references.append(ex["output"])
        print(f"[{i}/{len(examples)}] generated {len(pred)} chars")

    rouge = hf_evaluate.load("rouge")
    bleu = hf_evaluate.load("sacrebleu")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])

    ppl = compute_perplexity(
        model, tokenizer, [build_prompt(ex) + ex["output"] for ex in examples]
    )

    results = {
        "num_examples": len(examples),
        "rouge1": round(rouge_scores["rouge1"], 4),
        "rouge2": round(rouge_scores["rouge2"], 4),
        "rougeL": round(rouge_scores["rougeL"], 4),
        "bleu": round(bleu_scores["score"], 4),
        "perplexity": round(ppl, 4),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k:>12}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
