"""Unit tests for the per-save write-count accumulator.

``dispatch_save`` publishes a fresh dict per save; ``_write_citation`` (the
single funnel every memory write passes through — LLM pipeline and direct
writes alike) bumps it; the worker's Meta Agent root span closes with
``writes_by_memory_type`` from it. Pure ContextVar mechanics — no I/O, no
locks (single event loop).

The concurrency-sharing test pins the pattern the design relies on: tasks
spawned under ``asyncio.gather`` COPY the context (so their token resets are
isolated) but share the parent's MUTABLE dict object — the same pattern as
the fault-injection active-source scope.
"""

import asyncio

import pytest

from mirix.observability.trace_attrs import (
    bump_write_count,
    get_write_counts,
    reset_save_write_counts,
    set_save_write_counts,
)


@pytest.fixture(autouse=True)
def _clean_counts():
    yield
    # Belt-and-braces: leave the ContextVar unset for the next test.
    reset_save_write_counts(None)


def test_set_gives_fresh_empty_dict():
    token = set_save_write_counts()
    try:
        assert get_write_counts() == {}
    finally:
        reset_save_write_counts(token)


def test_bump_is_silent_no_op_when_unset():
    # No save active: bump must neither raise nor create state.
    bump_write_count("episodic")
    assert get_write_counts() == {}


def test_counts_aggregate_per_memory_type():
    token = set_save_write_counts()
    try:
        bump_write_count("episodic")
        bump_write_count("episodic")
        bump_write_count("core")
        assert get_write_counts() == {"episodic": 2, "core": 1}
    finally:
        reset_save_write_counts(token)


def test_get_write_counts_returns_snapshot_not_live_dict():
    token = set_save_write_counts()
    try:
        bump_write_count("semantic")
        snap = get_write_counts()
        snap["semantic"] = 99
        assert get_write_counts() == {"semantic": 1}
    finally:
        reset_save_write_counts(token)


def test_reset_restores_prior_state():
    outer = set_save_write_counts()
    bump_write_count("core")
    inner = set_save_write_counts()  # nested save (defensive)
    bump_write_count("episodic")
    assert get_write_counts() == {"episodic": 1}
    reset_save_write_counts(inner)
    assert get_write_counts() == {"core": 1}
    reset_save_write_counts(outer)
    assert get_write_counts() == {}


@pytest.mark.asyncio
async def test_gathered_tasks_share_the_parent_dict():
    """Sub-agents run as tasks under asyncio.gather; each task's copied context
    points at the SAME dict object, so all their bumps land in one place."""
    token = set_save_write_counts()
    try:

        async def _sub_agent(memory_type: str, n: int) -> None:
            for _ in range(n):
                bump_write_count(memory_type)
                await asyncio.sleep(0)

        await asyncio.gather(
            _sub_agent("episodic", 3),
            _sub_agent("semantic", 2),
            _sub_agent("core", 1),
        )
        assert get_write_counts() == {"episodic": 3, "semantic": 2, "core": 1}
    finally:
        reset_save_write_counts(token)


@pytest.mark.asyncio
async def test_sequential_saves_do_not_leak_counts():
    """Back-to-back saves in the same task (the batch-worker shape) each see a
    fresh dict; the boundary is set/reset, mirroring dispatch_save."""

    async def _one_save(memory_type: str) -> dict:
        token = set_save_write_counts()
        try:
            bump_write_count(memory_type)
            return get_write_counts()
        finally:
            reset_save_write_counts(token)

    first = await _one_save("episodic")
    second = await _one_save("core")
    assert first == {"episodic": 1}
    assert second == {"core": 1}
    assert get_write_counts() == {}
