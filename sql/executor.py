"""Safe SQLite execution for Spider databases.

The executor is designed to be fast (reused connections where possible) and
safe (read-only connections, per-query timeout, result truncation) so it can
be called thousands of times during a GRPO rollout.
"""
from __future__ import annotations

import signal
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = ["execute_sql", "ExecutionResult", "ExecutorError", "spider_db_path"]


class ExecutorError(Exception):
    """Raised for unrecoverable executor failures (e.g., missing DB)."""


@dataclass
class ExecutionResult:
    ok: bool
    rows: Optional[list[tuple]] = None
    error: Optional[str] = None
    timed_out: bool = False

    def as_set(self) -> frozenset:
        if self.rows is None:
            return frozenset()
        return frozenset(tuple(r) for r in self.rows)


def spider_db_path(spider_root: str | Path, db_id: str) -> Path:
    """Return the SQLite file path for a Spider db_id."""
    return Path(spider_root) / "database" / db_id / f"{db_id}.sqlite"


class _TimeoutError(Exception):
    pass


@contextmanager
def _time_limit(seconds: float):
    """Best-effort timeout. Uses SIGALRM in the main thread; falls back to a
    no-op elsewhere (worker threads / non-POSIX). We still rely on sqlite's
    own connection timeout as a second line of defense.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise _TimeoutError("sql execution timeout")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Not in main thread (e.g., some signal configurations); skip timeout.
        yield
        return

    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def execute_sql(
    sql: str,
    db_path: str | Path,
    *,
    timeout_s: float = 5.0,
    max_rows: int = 10000,
    read_only: bool = True,
) -> ExecutionResult:
    """Execute a SQL query and return the (truncated) rows or a structured error."""
    db_path = Path(db_path)
    if not db_path.exists():
        return ExecutionResult(ok=False, error=f"missing db: {db_path}")

    try:
        if read_only:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
        else:
            conn = sqlite3.connect(str(db_path), timeout=timeout_s)
    except sqlite3.Error as exc:
        return ExecutionResult(ok=False, error=f"connect error: {exc}")

    try:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        cur = conn.cursor()
        try:
            with _time_limit(timeout_s):
                cur.execute(sql)
                rows = cur.fetchmany(max_rows)
        except _TimeoutError:
            return ExecutionResult(ok=False, error="timeout", timed_out=True)
        except sqlite3.Error as exc:
            return ExecutionResult(ok=False, error=f"sqlite error: {exc}")
        return ExecutionResult(ok=True, rows=list(rows))
    finally:
        try:
            conn.close()
        except Exception:
            pass
