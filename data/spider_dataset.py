"""Load Spider splits and convert them into prompt / target records.

This is deliberately framework-light: it produces a list of ``SpiderRecord``
dicts that downstream trainers (SFT, GRPO) can wrap into a HuggingFace
Dataset however they want.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Optional

from .prompt_builder import PromptMode, build_prompt, build_target
from .schema_linker import format_schema, load_tables_json
from .sketch_extractor import SketchExtractionError, extract_sketch

SpiderSplit = Literal["train", "dev"]


@dataclass
class SpiderRecord:
    db_id: str
    question: str
    sql: str
    schema_text: str
    prompt_system: str
    prompt_user: str
    target: str
    mode: PromptMode
    sketch_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _iter_split_files(spider_root: Path, split: SpiderSplit) -> Iterable[Path]:
    if split == "train":
        candidates = [spider_root / "train_spider.json", spider_root / "train_others.json"]
    elif split == "dev":
        candidates = [spider_root / "dev.json"]
    else:
        raise ValueError(f"unknown split {split!r}")
    for p in candidates:
        if p.exists():
            yield p


def load_spider_splits(
    spider_root: str | Path,
    split: SpiderSplit,
    mode: PromptMode,
    *,
    max_schema_chars: int | None = 4000,
    limit: Optional[int] = None,
    dialect: str = "sqlite",
    skip_unparseable: bool = True,
) -> list[SpiderRecord]:
    """Load one Spider split and materialize prompt / target pairs."""
    spider_root = Path(spider_root)
    tables = load_tables_json(spider_root / "tables.json")

    records: list[SpiderRecord] = []
    seen = 0
    for path in _iter_split_files(spider_root, split):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            if limit is not None and len(records) >= limit:
                break
            seen += 1
            db_id = ex["db_id"]
            if db_id not in tables:
                continue
            schema_text = format_schema(tables[db_id], max_chars=max_schema_chars)
            question = ex["question"]
            sql = ex["query"]

            sketch_ok = True
            if mode == "sketch":
                try:
                    sketch = extract_sketch(sql, dialect=dialect)
                except SketchExtractionError:
                    if skip_unparseable:
                        continue
                    sketch = None
                    sketch_ok = False
                target = build_target(sql, mode=mode, sketch=sketch, dialect=dialect)
            else:
                target = build_target(sql, mode=mode)

            prompt = build_prompt(schema_text, question, mode=mode)
            records.append(
                SpiderRecord(
                    db_id=db_id,
                    question=question,
                    sql=sql,
                    schema_text=schema_text,
                    prompt_system=prompt.system,
                    prompt_user=prompt.user,
                    target=target,
                    mode=mode,
                    sketch_ok=sketch_ok,
                )
            )
    return records


def write_jsonl(records: list[SpiderRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
