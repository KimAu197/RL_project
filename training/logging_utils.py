"""Training logging helpers shared by SFT and GRPO entry points."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_wandb_reporting(
    trainer_args: dict[str, Any],
    wandb_cfg: dict[str, Any] | None,
    output_dir: str | Path,
    run_name: str,
) -> dict[str, Any]:
    """Return Trainer args with W&B reporting configured.

    W&B is opt-in so local/offline training keeps working without a wandb login.
    """
    updated = dict(trainer_args)
    wandb_cfg = wandb_cfg or {}

    if not bool(wandb_cfg.get("enabled", False)):
        updated["report_to"] = ["none"]
        os.environ["WANDB_DISABLED"] = "true"
        for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_NAME", "WANDB_DIR", "WANDB_TAGS"):
            os.environ.pop(key, None)
        return updated

    output_dir = Path(output_dir).resolve()
    project = wandb_cfg.get("project") or "sketch-to-sql-rl"
    entity = wandb_cfg.get("entity")
    tags = wandb_cfg.get("tags") or []

    updated["report_to"] = ["wandb"]
    updated.setdefault("run_name", run_name)

    os.environ["WANDB_DISABLED"] = "false"
    os.environ["WANDB_PROJECT"] = str(project)
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_DIR"] = str(output_dir)

    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)
    else:
        os.environ.pop("WANDB_ENTITY", None)

    if tags:
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)
    else:
        os.environ.pop("WANDB_TAGS", None)

    return updated
