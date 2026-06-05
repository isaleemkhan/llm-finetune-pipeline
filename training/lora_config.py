"""
Helpers that build BitsAndBytes quantization and LoRA/PEFT configs from a
plain config dict (typically loaded from training/config.yaml).
"""

from typing import Dict

import torch


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


def build_bnb_config(model_cfg: Dict):
    """Return a BitsAndBytesConfig for 4-bit QLoRA, or None if disabled."""
    if not model_cfg.get("load_in_4bit", False):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=_resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(lora_cfg: Dict):
    """Return a PEFT LoraConfig from the lora section of the config."""
    from peft import LoraConfig

    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
