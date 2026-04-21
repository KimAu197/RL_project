"""Shared utilities: YAML config, logging, and output directory naming.

These helpers are imported by both training scripts and the evaluation
pipeline so all stages produce consistent run artifacts.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "load_config",
    "snapshot_config",
    "configure_logging",
    "resolve_output_dir",
    "deep_merge",
]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with ``!include base.yaml``-style inheritance.

    If the config has a top-level ``inherits`` key, we first load that file
    (relative to the config) and merge the current file into it.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("inherits", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base_cfg = load_config(parent_path)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (non-mutating)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def snapshot_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def configure_logging(log_file: str | Path, level: int = logging.INFO) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def resolve_output_dir(
    base: str | Path,
    round_name: str,
    model_name: str,
    dataset: str,
    count: int,
) -> Path:
    """Build ``<base>/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]``.

    Appends ``_HHMM`` only if a collision is detected, matching the user's
    workspace convention.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    mmdd = now.strftime("%m%d")
    model_tag = _sanitize(model_name)
    dataset_tag = _sanitize(dataset)
    round_tag = _sanitize(round_name)

    stem = f"{round_tag}_{model_tag}_{dataset_tag}_{count}_{mmdd}"
    candidate = base / stem
    if candidate.exists():
        hhmm = now.strftime("%H%M")
        candidate = base / f"{stem}_{hhmm}"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "log").mkdir(parents=True, exist_ok=True)
    return candidate


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", ".") else "_" for c in s)
