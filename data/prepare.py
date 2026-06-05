"""
Dataset preprocessing for the LLM fine-tuning pipeline.

Loads raw data (JSONL, CSV, or a HuggingFace dataset), normalizes it into a
consistent {instruction, input, output} schema, renders it into a single
prompt/completion text field using a chosen template, performs a train/val
split, and writes the result back out as JSONL ready for training.

Usage:
    python data/prepare.py --input data/raw/dataset.jsonl --output data/processed/
    python data/prepare.py --input data/raw/data.csv --output data/processed/ --format completion
    python data/prepare.py --hf-dataset tatsu-lab/alpaca --output data/processed/
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

# Prompt templates. {instruction}/{input}/{output} are filled per-example.
TEMPLATES = {
    "instruction": (
        "### Instruction:\n{instruction}\n\n"
        "{input_block}"
        "### Response:\n{output}"
    ),
    "chat": (
        "<|start_header_id|>user<|end_header_id|>\n{instruction}{input_inline}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{output}<|eot_id|>"
    ),
    "completion": "{instruction}{input_inline}{output}",
}


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: str) -> List[Dict]:
    import pandas as pd

    df = pd.read_csv(path)
    return df.fillna("").to_dict(orient="records")


def _read_hf(name: str, split: str) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    return [dict(r) for r in ds]


def load_raw(args: argparse.Namespace) -> List[Dict]:
    if args.hf_dataset:
        return _read_hf(args.hf_dataset, args.hf_split)
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".jsonl":
        return _read_jsonl(args.input)
    if ext == ".csv":
        return _read_csv(args.input)
    if ext == ".json":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    raise ValueError(f"Unsupported input extension: {ext}")


def normalize(row: Dict, mapping: Dict[str, str]) -> Optional[Dict]:
    """Map arbitrary source columns to the canonical schema."""
    instruction = str(row.get(mapping["instruction"], "")).strip()
    inp = str(row.get(mapping["input"], "")).strip() if mapping["input"] else ""
    output = str(row.get(mapping["output"], "")).strip()
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": inp, "output": output}


def render(example: Dict, template_name: str) -> Dict:
    tpl = TEMPLATES[template_name]
    inp = example["input"]
    input_block = f"### Input:\n{inp}\n\n" if inp else ""
    input_inline = f"\n{inp}" if inp else ""
    text = tpl.format(
        instruction=example["instruction"],
        output=example["output"],
        input_block=input_block,
        input_inline=input_inline,
    )
    return {"text": text, **example}


def write_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning.")
    parser.add_argument("--input", help="Path to raw JSONL/CSV/JSON file.")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset id (alternative to --input).")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--format",
        choices=list(TEMPLATES.keys()),
        default="instruction",
        help="Prompt template to render.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction-col", default="instruction")
    parser.add_argument("--input-col", default="input")
    parser.add_argument("--output-col", default="output")
    args = parser.parse_args()

    if not args.input and not args.hf_dataset:
        parser.error("Provide either --input or --hf-dataset.")

    mapping = {
        "instruction": args.instruction_col,
        "input": args.input_col,
        "output": args.output_col,
    }

    raw = load_raw(args)
    print(f"Loaded {len(raw)} raw rows.")

    cleaned: List[Dict] = []
    skipped = 0
    for row in raw:
        norm = normalize(row, mapping)
        if norm is None:
            skipped += 1
            continue
        cleaned.append(render(norm, args.format))

    print(f"Kept {len(cleaned)} examples (skipped {skipped} incomplete rows).")

    random.seed(args.seed)
    random.shuffle(cleaned)
    n_val = max(1, int(len(cleaned) * args.val_ratio)) if cleaned else 0
    val, train = cleaned[:n_val], cleaned[n_val:]

    train_path = os.path.join(args.output, "train.jsonl")
    val_path = os.path.join(args.output, "val.jsonl")
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print(f"Wrote {len(train)} train -> {train_path}")
    print(f"Wrote {len(val)} val   -> {val_path}")


if __name__ == "__main__":
    main()
