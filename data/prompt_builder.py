"""Build chat-style prompts for the four system variants.

The exact wording is kept minimal and deterministic so that it can be used
both for SFT (target = assistant response) and GRPO (target is sampled).

Prompt variants:

* ``direct``: the assistant response is just the SQL in a fenced block.
* ``sketch``: the assistant response is a ``<sketch>...</sketch>`` block
  followed by ``<sql>...</sql>``.

Delimiters are intentionally explicit so reward computation and evaluation
can robustly locate the SQL no matter how much reasoning the model emits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .sketch_extractor import Sketch, extract_sketch, sketch_to_text

PromptMode = Literal["direct", "sketch"]

SYSTEM_DIRECT = (
    "You are an expert SQL programmer. Given a database schema and a question, "
    "return only the SQL query inside a fenced <sql>...</sql> block. "
    "Do not include any other text."
)

SYSTEM_SKETCH = (
    "You are an expert SQL programmer. Given a database schema and a question, "
    "first produce a structured sketch of the query inside <sketch>...</sketch>, "
    "then produce the final SQL inside <sql>...</sql>. The sketch must list the "
    "tables, joins, select expressions, aggregations, filters, grouping, ordering, "
    "limit, subqueries, and set operations needed to answer the question. Use "
    "SET_OP and SET_RHS for UNION, INTERSECT, or EXCEPT. Use '?' for concrete "
    "literal values."
)

USER_TEMPLATE = (
    "Schema:\n{schema}\n\n"
    "Question: {question}"
)


@dataclass
class PromptSpec:
    mode: PromptMode
    system: str
    user: str


def build_prompt(schema_text: str, question: str, mode: PromptMode) -> PromptSpec:
    if mode not in ("direct", "sketch"):
        raise ValueError(f"unknown prompt mode: {mode}")
    system = SYSTEM_DIRECT if mode == "direct" else SYSTEM_SKETCH
    user = USER_TEMPLATE.format(schema=schema_text.strip(), question=question.strip())
    return PromptSpec(mode=mode, system=system, user=user)


def build_target(sql: str, mode: PromptMode, sketch: Sketch | None = None, dialect: str = "sqlite") -> str:
    """Build the gold assistant response for SFT."""
    sql_clean = sql.strip().rstrip(";")
    if mode == "direct":
        return f"<sql>\n{sql_clean}\n</sql>"
    if mode == "sketch":
        if sketch is None:
            sketch = extract_sketch(sql_clean, dialect=dialect)
        return (
            "<sketch>\n"
            f"{sketch_to_text(sketch)}\n"
            "</sketch>\n"
            "<sql>\n"
            f"{sql_clean}\n"
            "</sql>"
        )
    raise ValueError(f"unknown prompt mode: {mode}")


def apply_chat_template(tokenizer, prompt: PromptSpec, target: str | None = None, add_generation_prompt: bool = False) -> str:
    """Render a prompt (and optional target) using the tokenizer's chat template.

    If the tokenizer has no chat template (rare for instruct models), we fall
    back to a plain concatenation that the SFT trainer can still consume.
    """
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and target is None,
        )

    rendered = [f"<|system|>\n{prompt.system}", f"<|user|>\n{prompt.user}"]
    if target is not None:
        rendered.append(f"<|assistant|>\n{target}")
    elif add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)
