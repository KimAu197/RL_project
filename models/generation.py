"""Lightweight batched generation helpers used by evaluation.

Training (SFT / GRPO) uses the HuggingFace trainer loops, so this module
only targets the inference side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = ["GenerationConfigLite", "generate_batched"]


@dataclass
class GenerationConfigLite:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0


def _pad_left(tokenizer, input_ids: list[list[int]]) -> dict:
    """Left-pad a batch of token id sequences for causal LM generation."""
    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_token_id
    padded = []
    attn = []
    for ids in input_ids:
        pad = max_len - len(ids)
        padded.append([pad_id] * pad + ids)
        attn.append([0] * pad + [1] * len(ids))
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


@torch.no_grad()
def generate_batched(
    model,
    tokenizer,
    prompts: list[str],
    cfg: GenerationConfigLite,
    batch_size: int = 8,
) -> list[list[str]]:
    """Generate completions for each prompt.

    Returns a list (len == len(prompts)) of lists
    (len == cfg.num_return_sequences) containing only the newly generated
    text (the prompt is stripped off via slicing on the token ids).
    """
    model.eval()
    device = next(model.parameters()).device
    out_all: list[list[str]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in batch]
        padded = _pad_left(tokenizer, encoded)
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)
        prompt_len = input_ids.size(1)

        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature if cfg.do_sample else 1.0,
            "top_p": cfg.top_p,
            "num_return_sequences": cfg.num_return_sequences,
            "repetition_penalty": cfg.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
        }

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        # out shape: (batch * num_return, total_len). Strip prompt tokens.
        new_tokens = out[:, prompt_len:]
        # reshape (never view) to survive non-contiguous layouts.
        new_tokens = new_tokens.reshape(
            input_ids.size(0), cfg.num_return_sequences, -1
        )
        for i in range(input_ids.size(0)):
            group = [
                tokenizer.decode(new_tokens[i, k], skip_special_tokens=True)
                for k in range(cfg.num_return_sequences)
            ]
            out_all.append(group)

    return out_all
