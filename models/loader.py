"""Load a causal LM + tokenizer, optionally with LoRA adapters and 4-bit quant.

Keeping this in one place means every stage of the pipeline (SFT, GRPO,
evaluation) starts from identical model / tokenizer state, which is critical
for reproducible comparisons across the four system variants.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LOG = logging.getLogger(__name__)

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
    # Optional pin for the Hub revision when `name_or_path` is a PEFT dir (or base is Hub).
    base_model_revision: Optional[str] = None
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


def _is_peft_adapter_dir(name_or_path: str) -> bool:
    p = Path(name_or_path)
    if not p.is_dir():
        return False
    return (p / "adapter_config.json").is_file()


def _load_causal_backbone(cfg: LoaderConfig) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": _resolve_dtype(cfg.dtype),
    }
    if cfg.base_model_revision:
        model_kwargs["revision"] = cfg.base_model_revision
    if cfg.device_map:
        model_kwargs["device_map"] = cfg.device_map
    if cfg.attn_impl:
        model_kwargs["attn_implementation"] = cfg.attn_impl
    bnb = _maybe_bnb_config(cfg)
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    model_kwargs.update(cfg.extra_model_kwargs or {})
    return AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **model_kwargs)


def _load_tokenizer(cfg: LoaderConfig, tok_src: str):
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
    return tokenizer


def _peft_from_pretrained_matched(
    model: torch.nn.Module, peft_path: str, is_trainable: bool = False
) -> torch.nn.Module:
    from peft import PeftModel

    kwargs: dict[str, Any] = {"is_trainable": is_trainable}
    if "ignore_mismatched_sizes" in inspect.signature(PeftModel.from_pretrained).parameters:
        kwargs["ignore_mismatched_sizes"] = True
    return PeftModel.from_pretrained(model, peft_path, **kwargs)


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


def load_model_and_tokenizer(cfg: LoaderConfig):
    tok_src = cfg.tokenizer_name_or_path or cfg.name_or_path
    tokenizer = _load_tokenizer(cfg, tok_src)

    if _is_peft_adapter_dir(cfg.name_or_path):
        peft_path = str(Path(cfg.name_or_path).resolve())
        with open(Path(peft_path) / "adapter_config.json", encoding="utf-8") as f:
            adapter_info = json.load(f)
        base = adapter_info.get("base_model_name_or_path")
        if not base:
            raise ValueError("adapter_config.json is missing base_model_name_or_path")
        if cfg.lora is not None:
            _LOG.warning(
                "model.lora in config is ignored when name_or_path is a PEFT "
                "checkpoint (adapter is loaded from %s).",
                peft_path,
            )
        # Align base snapshot with the saved adapter when the Hub default moves (vocab/weights).
        rev = cfg.base_model_revision or adapter_info.get("revision")
        base_cfg = dataclasses.replace(
            cfg,
            name_or_path=base,
            lora=None,
            base_model_revision=rev,
        )
        model = _load_causal_backbone(base_cfg)
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        model = _peft_from_pretrained_matched(model, peft_path, is_trainable=False)
    else:
        if cfg.lora is not None:
            base_cfg = dataclasses.replace(cfg, lora=None)
        else:
            base_cfg = cfg
        model = _load_causal_backbone(base_cfg)
        if cfg.lora is not None:
            model = _apply_lora(model, cfg.lora)
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
