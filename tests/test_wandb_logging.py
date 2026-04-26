import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.training.logging_utils import configure_wandb_reporting


def test_disabled_wandb_uses_no_reporter(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)

    args_cfg = configure_wandb_reporting(
        trainer_args={"logging_steps": 10},
        wandb_cfg={"enabled": False, "project": "demo"},
        output_dir=Path("runs/test_run"),
        run_name="test_run",
    )

    assert args_cfg["report_to"] == ["none"]
    assert os.environ["WANDB_DISABLED"] == "true"
    assert "WANDB_PROJECT" not in os.environ
    assert "WANDB_NAME" not in os.environ
    assert "WANDB_DIR" not in os.environ


def test_enabled_wandb_sets_trainer_reporter_and_env(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    args_cfg = configure_wandb_reporting(
        trainer_args={"logging_steps": 10, "report_to": ["none"]},
        wandb_cfg={
            "enabled": True,
            "project": "sketch-to-sql-rl",
            "entity": "course-team",
            "tags": ["sft", "direct"],
        },
        output_dir=Path("runs/sft_direct_Qwen_spider_0_0425"),
        run_name="sft_direct_Qwen_spider_0_0425",
    )

    assert args_cfg["report_to"] == ["wandb"]
    assert os.environ["WANDB_PROJECT"] == "sketch-to-sql-rl"
    assert os.environ["WANDB_ENTITY"] == "course-team"
    assert os.environ["WANDB_NAME"] == "sft_direct_Qwen_spider_0_0425"
    assert os.environ["WANDB_DIR"] == str(Path("runs/sft_direct_Qwen_spider_0_0425").resolve())
    assert os.environ["WANDB_TAGS"] == "sft,direct"
    assert os.environ["WANDB_DISABLED"] == "false"
