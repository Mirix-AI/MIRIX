"""
Global token-usage tracker for instrumenting MIRIX's LLM calls.

Designed so call-sites can blindly call ``record(...)`` and external code (evals,
benchmarks) decides what counts as "build" vs "query" via context-managed phases.

Why a tracker module instead of LangFuse:
  - LangFuse is heavy (network round-trips, project setup, env vars).
  - For evals we just want a per-run integer total. A process-global dict
    that records by ``(phase, user_id)`` is enough.

Usage in eval:

    from mirix.database.token_tracker import set_phase, snapshot, reset

    reset()  # at process start, optional
    with set_phase("build"):
        await client.add(...)   # all server LLM calls recorded under "build"
    with set_phase("query"):
        await task_agent.answer(...)
    stats = snapshot()  # {(phase, user_id): {prompt, completion, total, calls}}

Usage in call-sites (one-liner):

    from mirix.database.token_tracker import record
    record(prompt_tokens=..., completion_tokens=...)

Thread-safe via a single lock; concurrency-safe via contextvars for ``_phase_var``.
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import Optional

# Process-wide enable flag. Default OFF so the tracker is a true no-op for
# anyone not running an eval. Flip via enable()/disable() — typically called
# from an eval harness (see evals/main_eval.py) or from the
# /debug/token_stats/* REST endpoints.
_enabled: bool = False

# Current logical phase, propagated through asyncio tasks via contextvar.
# When tracker is enabled and no explicit phase is set, falls back to "server".
# Evals call set_phase("build") or set_phase("query") to bucket more finely.
_phase_var: ContextVar[Optional[str]] = ContextVar("mirix_token_phase", default=None)

# Stable buckets keyed by (phase, user_id). user_id is optional — calls from
# server endpoints that don't know the user just bucket as user_id="*".
_lock = threading.Lock()
_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
    lambda: {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
)


def enable() -> None:
    """Turn the tracker on. ``record()`` becomes a real write after this."""
    global _enabled
    _enabled = True


def disable() -> None:
    """Turn the tracker off. ``record()`` becomes a no-op."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def record(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: Optional[int] = None,
    user_id: str = "*",
) -> None:
    """Add one OpenAI/Anthropic ``usage`` payload to the active phase bucket.

    No-op unless ``enable()`` has been called. Phase defaults to "server"
    when enabled but no ``set_phase`` context is active.
    Robust to ``None`` / negative inputs.
    """
    if not _enabled:
        return
    phase = _phase_var.get() or "server"
    p = max(int(prompt_tokens or 0), 0)
    c = max(int(completion_tokens or 0), 0)
    t = int(total_tokens) if total_tokens is not None else p + c
    with _lock:
        bucket = _stats[(phase, user_id)]
        bucket["prompt"] += p
        bucket["completion"] += c
        bucket["total"] += t
        bucket["calls"] += 1


@contextlib.contextmanager
def set_phase(phase: str):
    """Context manager that sets ``_phase_var`` for the duration of the block.

    Nested calls are supported — inner phase wins, restored on exit. Cross-task
    propagation works because ``_phase_var`` is a contextvar (each asyncio Task
    inherits the calling task's context).
    """
    token = _phase_var.set(phase)
    try:
        yield
    finally:
        _phase_var.reset(token)


def snapshot() -> dict[str, dict[str, int]]:
    """Return a copy of current stats keyed by ``"phase|user_id"`` strings."""
    with _lock:
        return {f"{phase}|{uid}": dict(v) for (phase, uid), v in _stats.items()}


def reset() -> None:
    """Wipe all buckets. Use at the start of a fresh eval run."""
    with _lock:
        _stats.clear()
