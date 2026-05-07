"""Load a causal LM + tokenizer, optionally with LoRA adapters and 4-bit quant.

Keeping this in one place means every stage of the pipeline (SFT, GRPO,
evaluation) starts from identical model / tokenizer state, which is critical
for reproducible comparisons across the four system variants.

Supports two loading modes:
  1. Fresh base model + optional new LoRA adapter (used by SFT).
  2. Saved PEFT adapter checkpoint -> reload base model + existing adapter
     (used by GRPO when starting from an SFT LoRA checkpoint).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["LoaderConfig", "load_model_and_tokenizer"]


@dataclass
class LoaderConfig:
    name_or_path: str
    dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_impl: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    lora: Optional[dict[str, Any]] = None
    tokenizer_name_or_path: Optional[str] = None
    pad_token: str = "<|endoftext|>"
    chat_template_override: Optional[str] = None
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    if dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


def _maybe_bnb_config(cfg: LoaderConfig):
    if not (cfg.load_in_4bit or cfg.load_in_8bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("bitsandbytes / transformers quantization not available") from exc
    if cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _is_peft_adapter(path: str) -> bool:
    """Check if the given path is a saved PEFT adapter directory."""
    p = Path(path)
    return p.is_dir() and (p / "adapter_config.json").exists()


def _get_base_model_from_adapter(adapter_path: str) -> str:
    """Read the base model name from a saved adapter_config.json."""
    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    return adapter_cfg["base_model_name_or_path"]


def _apply_lora(model, lora_cfg: dict[str, Any]):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    target_modules = lora_cfg.get("target_modules")
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_cfg)
    return model


def _load_peft_adapter(adapter_path: str, model_kwargs: dict[str, Any], cfg: LoaderConfig, tokenizer):
    """Load a base model and attach a saved PEFT adapter for continued training."""
    from peft import PeftModel, prepare_model_for_kbit_training

    base_model_name = _get_base_model_from_adapter(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    # Resize embeddings to match the tokenizer saved with the adapter.
    # This is needed because SFT may have resized embeddings before saving.
    embed_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embed_size:
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model


def load_model_and_tokenizer(cfg: LoaderConfig):
    tok_src = cfg.tokenizer_name_or_path or cfg.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": cfg.pad_token})
    if cfg.chat_template_override:
        tokenizer.chat_template = cfg.chat_template_override

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": _resolve_dtype(cfg.dtype),
    }
    if cfg.device_map:
        model_kwargs["device_map"] = cfg.device_map
    if cfg.attn_impl:
        model_kwargs["attn_implementation"] = cfg.attn_impl
    bnb = _maybe_bnb_config(cfg)
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    model_kwargs.update(cfg.extra_model_kwargs or {})

    if _is_peft_adapter(cfg.name_or_path):
        model = _load_peft_adapter(cfg.name_or_path, model_kwargs, cfg, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **model_kwargs)

        # Only resize UP (when tokens were added). Never shrink — some models
        # (e.g. Qwen) intentionally have more embedding slots than vocab tokens.
        embed_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embed_size:
            model.resize_token_embeddings(len(tokenizer))

        if cfg.lora:
            model = _apply_lora(model, cfg.lora)

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
