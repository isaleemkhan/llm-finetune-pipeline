# llm-finetune-pipeline

**Train a custom LLM on your data. Evaluate it. Ship it.**

An end-to-end fine-tuning pipeline for LLaMA 3 using LoRA/QLoRA — from raw dataset to a served model endpoint. Designed for production, not just notebooks.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## What It Does

Most fine-tuning guides end at training. This pipeline goes all the way to a deployed endpoint:

1. **Prepare** - Load and format your dataset (JSONL, CSV, HuggingFace datasets)
2. **Train** - Fine-tune LLaMA 3 with QLoRA (4-bit quantization + LoRA adapters)
3. **Evaluate** - Run ROUGE, BLEU, perplexity, and custom benchmarks
4. **Merge** - Merge LoRA weights back into the base model
5. **Serve** - Deploy as a streaming FastAPI endpoint with vLLM backend

Designed for: domain-specific chatbots, instruction-following models, classification, summarization, and code generation.

---

## Architecture

```
Dataset (JSONL/CSV/HuggingFace)
      |
      v
Data Preprocessing
  |-- Formatting (instruction, chat, completion)
  +-- Tokenization + padding
      |
      v
QLoRA Fine-tuning (4-bit + LoRA adapters)
  |-- Base: LLaMA 3 8B or 70B
  |-- Trainer: HuggingFace TRL SFTTrainer
  +-- Logging: W&B / TensorBoard
      |
      v
Evaluation Suite
  |-- ROUGE / BLEU scores
  +-- Custom eval prompts
      |
      v
LoRA Weight Merge + GGUF Export
      |
      v
FastAPI Inference Server (vLLM backend)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Base Model | LLaMA 3 8B / 70B (Meta) |
| Fine-tuning | QLoRA (4-bit) + LoRA adapters |
| Training Framework | HuggingFace Transformers + TRL |
| Quantization | BitsAndBytes (4-bit NF4) |
| Inference | vLLM (PagedAttention) |
| API Server | FastAPI + Uvicorn |
| Experiment Tracking | Weights and Biases |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
llm-finetune-pipeline/
├── data/
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Formatted training data
│   └── prepare.py           # Dataset preprocessing script
├── training/
│   ├── train.py             # Main training script (QLoRA)
│   ├── config.yaml          # Training hyperparameters
│   └── lora_config.py       # LoRA adapter config
├── evaluation/
│   ├── evaluate.py          # ROUGE, BLEU, perplexity
│   └── benchmarks/          # Custom evaluation sets
├── serving/
│   ├── merge_weights.py     # Merge LoRA into base model
│   ├── server.py            # FastAPI inference server
│   └── client.py            # Example client
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── quickstart.ipynb
├── requirements.txt
└── README.md
```

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (24GB+ VRAM for 8B, 80GB for 70B with QLoRA)
- HuggingFace account with LLaMA 3 access

### Install

```bash
git clone https://github.com/isaleemkhan/llm-finetune-pipeline.git
cd llm-finetune-pipeline
pip install -r requirements.txt
```

### Prepare your dataset

Format your data as JSONL with instruction/output pairs:

```json
{"instruction": "Summarize this text", "input": "...", "output": "..."}
```

```bash
python data/prepare.py --input data/raw/dataset.jsonl --output data/processed/
```

### Train

```bash
python training/train.py   --model meta-llama/Meta-Llama-3-8B   --data data/processed/train.jsonl   --output ./checkpoints   --epochs 3   --batch_size 4   --lora_r 16
```

### Evaluate

```bash
python evaluation/evaluate.py --checkpoint ./checkpoints/final
```

### Serve

```bash
python serving/merge_weights.py --checkpoint ./checkpoints/final --output ./merged
uvicorn serving.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up --build
```

---

## Training Configuration

Key parameters in `training/config.yaml`:

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  lora_dropout: 0.05

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | /generate | Generate text from prompt |
| POST | /chat | Multi-turn chat completion |
| GET | /model/info | Model metadata and config |
| GET | /health | Health check |

---

## License

MIT - built by [Saleem Khan](https://github.com/isaleemkhan)
