"""Execution-based reward functions for GRPO.

We keep the default reward intentionally simple, matching the proposal:

    reward = 1.0 if pred_result_set == gold_result_set else 0.0

Optional shaping terms (validity bonus, sketch-present bonus) are wired in
behind flags so the reward-hacking ablation from the proposal is trivial to
run. The shaping terms are bounded so they can't dominate the exec-match
signal.

The callable returned by ``build_reward_fn`` matches TRL's GRPOTrainer
``reward_funcs`` contract: it accepts ``prompts``, ``completions``, plus
arbitrary keyword-args containing per-example metadata (forwarded by the
Dataset), and returns a list of floats.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..evaluation.sketch_metrics import compute_sketch_metrics
from ..sql.executor import execute_sql, spider_db_path
from ..sql.metrics import results_equal
from ..sql.validator import extract_sql, extract_sketch_text, is_valid_sql

__all__ = ["ExecutionReward", "RewardConfig", "build_reward_fn"]

LOGGER = logging.getLogger("reward")


@dataclass
class RewardConfig:
    spider_root: str
    mode: str = "direct"
    timeout_s: float = 5.0
    match_reward: float = 1.0
    validity_bonus: float = 0.0
    sketch_bonus: float = 0.0
    sketch_format_bonus: float = 0.0
    sketch_table_bonus: float = 0.0
    sketch_agg_bonus: float = 0.0
    no_sql_penalty: float = 0.0
    dialect: str = "sqlite"


class ExecutionReward:
    """Stateful reward: caches gold execution results per (db_id, gold_sql)."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._gold_cache: dict[tuple[str, str], Any] = {}

    def _gold_result(self, db_path: Path, gold_sql: str):
        key = (str(db_path), gold_sql)
        cached = self._gold_cache.get(key)
        if cached is not None:
            return cached
        res = execute_sql(gold_sql, db_path, timeout_s=self.cfg.timeout_s)
        self._gold_cache[key] = res
        return res

    def score_one(self, completion_text: str, db_id: str, gold_sql: str) -> float:
        cfg = self.cfg
        pred_sql = extract_sql(completion_text)
        if pred_sql is None:
            return cfg.no_sql_penalty

        reward = 0.0
        if is_valid_sql(pred_sql, dialect=cfg.dialect):
            reward += cfg.validity_bonus

        if cfg.mode == "sketch" and extract_sketch_text(completion_text):
            reward += self.sketch_reward(completion_text, gold_sql)

        db_path = spider_db_path(cfg.spider_root, db_id)
        gold_res = self._gold_result(db_path, gold_sql)
        pred_res = execute_sql(pred_sql, db_path, timeout_s=cfg.timeout_s)

        if results_equal(pred_res, gold_res, gold_sql):
            reward += cfg.match_reward

        return float(reward)

    def sketch_reward(self, completion_text: str, gold_sql: str) -> float:
        """Return bounded auxiliary reward for sketch presence and correctness."""
        cfg = self.cfg
        sketch_text = extract_sketch_text(completion_text)
        if cfg.mode != "sketch" or not sketch_text:
            return 0.0

        metrics = compute_sketch_metrics(sketch_text, gold_sql, dialect=cfg.dialect)
        reward = cfg.sketch_bonus
        if metrics.format_ok:
            reward += cfg.sketch_format_bonus
        reward += cfg.sketch_table_bonus * metrics.table_recall
        if metrics.agg_match:
            reward += cfg.sketch_agg_bonus
        return float(reward)


def _completion_text(completion: Any) -> str:
    """Normalize a TRL completion (string or chat-message list) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def build_reward_fn(cfg: RewardConfig) -> Callable[..., list[float]]:
    """Return a callable compatible with TRL's ``reward_funcs``.

    TRL passes ``prompts``, ``completions``, plus any extra columns from the
    Dataset as kwargs. We read ``db_id`` and ``gold_sql`` from those columns.
    """
    scorer = ExecutionReward(cfg)

    def reward_fn(
        prompts: Optional[Sequence] = None,
        completions: Optional[Sequence] = None,
        **kwargs,
    ) -> list[float]:
        db_ids = kwargs.get("db_id") or []
        gold_sqls = kwargs.get("gold_sql") or []
        out: list[float] = []
        completions = completions or []
        for i, comp in enumerate(completions):
            text = _completion_text(comp)
            db_id = db_ids[i] if i < len(db_ids) else ""
            gold = gold_sqls[i] if i < len(gold_sqls) else ""
            try:
                out.append(scorer.score_one(text, db_id, gold))
            except Exception as exc:
                # Never let a single reward error kill the batch, but log it
                # loud enough that it doesn't vanish during debugging.
                LOGGER.warning("reward error on db_id=%s: %s", db_id, exc)
                out.append(float(cfg.no_sql_penalty))
        return out

    reward_fn.__name__ = "execution_reward"
    return reward_fn
