"""
Unit tests for the shared batch concurrency core
(``mirix/queue/batch.py::process_batch``).

Exercises the core's invariants directly with simple stand-in items:
group-by-user, gather across groups under a semaphore, serial-within-a-user,
and per-group first-error capture. Every batch consumer relies on these, so
verifying them here covers all of them at once.

By design, ``process_batch`` does NOT raise on a group error — it returns a
``BatchResult`` and lets the caller decide (re-raise ``errors[0]`` to trigger
broker redelivery, or re-enqueue the failed items).
"""

import asyncio
from types import SimpleNamespace

import pytest

from mirix.queue.batch import BatchResult, process_batch

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _item(item_id, user):
    """A minimal opaque batch item: an id + a grouping user attribute."""
    return SimpleNamespace(id=item_id, user=user)


def _user_key(item):
    return item.user


# ============================================================================
# Grouping + serial-within-a-user ordering
# ============================================================================


async def test_groups_by_user_and_runs_each_user_serially_in_arrival_order():
    """Items for the same user run SERIALLY in arrival order."""
    items = [
        _item("a1", "alice"),
        _item("b1", "bob"),
        _item("a2", "alice"),
        _item("a3", "alice"),
        _item("b2", "bob"),
    ]

    # Record the order each item *starts* processing.
    started_order = []

    async def process(item):
        started_order.append(item.id)
        # Yield so the scheduler could interleave if serialization were broken.
        await asyncio.sleep(0)

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    # alice's items must appear in arrival order relative to each other.
    alice_seen = [i for i in started_order if i.startswith("a")]
    assert alice_seen == ["a1", "a2", "a3"]
    # bob's items must appear in arrival order relative to each other.
    bob_seen = [i for i in started_order if i.startswith("b")]
    assert bob_seen == ["b1", "b2"]

    assert result.ok is True
    assert len(result.succeeded_items) == 5


# ============================================================================
# Cross-user concurrency
# ============================================================================


async def test_different_users_run_concurrently():
    """Two different users must be in flight at the same time (not serialized
    across groups)."""
    both_started = asyncio.Event()
    started = set()

    async def process(item):
        started.add(item.user)
        if len(started) >= 2:
            both_started.set()
        # Block until the sibling group has also started — proves concurrency.
        await asyncio.wait_for(both_started.wait(), timeout=2.0)

    items = [_item("a1", "alice"), _item("b1", "bob")]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    assert both_started.is_set()
    assert result.ok is True
    assert len(result.succeeded_items) == 2


# ============================================================================
# Semaphore cap
# ============================================================================


async def test_semaphore_caps_concurrent_user_groups():
    """With cap=2 and 5 distinct users, never more than 2 groups in flight."""
    cap = 2
    concurrent = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def process(item):
        nonlocal concurrent, max_observed
        async with lock:
            concurrent += 1
            max_observed = max(max_observed, concurrent)
        # Hold the slot long enough for others to pile up if the cap failed.
        await asyncio.sleep(0.05)
        async with lock:
            concurrent -= 1

    items = [_item(f"u{i}-1", f"user{i}") for i in range(5)]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=cap,
    )

    assert max_observed > 1, "expected real concurrency"
    assert max_observed <= cap, f"semaphore breached: {max_observed} > {cap}"
    assert result.ok is True
    assert len(result.succeeded_items) == 5


# ============================================================================
# First-error-in-a-group stops that group; siblings still complete
# ============================================================================


async def test_first_error_in_group_skips_rest_of_that_user_but_others_complete():
    """On the first exception in a user's stream, that user's later items are
    NOT processed; other users still complete; the error is captured and
    ``ok`` is False."""
    boom = RuntimeError("alice item 2 failed")
    processed = []

    async def process(item):
        if item.id == "a2":
            raise boom
        processed.append(item.id)
        await asyncio.sleep(0)

    items = [
        _item("a1", "alice"),
        _item("a2", "alice"),  # raises
        _item("a3", "alice"),  # must be skipped
        _item("b1", "bob"),
        _item("b2", "bob"),
    ]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    # a1 succeeded, a2 raised, a3 skipped.
    assert "a1" in processed
    assert "a3" not in processed
    # bob's stream is unaffected.
    assert "b1" in processed and "b2" in processed

    assert result.ok is False
    assert boom in result.errors
    # a1, b1, b2 succeeded; a2/a3 did not.
    succeeded_ids = {i.id for i in result.succeeded_items}
    assert succeeded_ids == {"a1", "b1", "b2"}


# ============================================================================
# on_item_success callback
# ============================================================================


async def test_on_item_success_called_once_per_succeeded_item_only():
    """``on_item_success`` fires exactly once per successful item and never for
    skipped/failed ones."""
    callbacks = []

    async def process(item):
        if item.id == "a2":
            raise RuntimeError("fail")
        await asyncio.sleep(0)

    items = [
        _item("a1", "alice"),
        _item("a2", "alice"),  # raises
        _item("a3", "alice"),  # skipped
        _item("b1", "bob"),
    ]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
        on_item_success=lambda item: callbacks.append(item.id),
    )

    assert sorted(callbacks) == ["a1", "b1"]
    assert result.ok is False
    assert sorted(i.id for i in result.succeeded_items) == ["a1", "b1"]


# ============================================================================
# Empty batch
# ============================================================================


async def test_empty_batch_returns_empty_ok_result():
    calls = []

    async def process(item):  # pragma: no cover - must never run
        calls.append(item)

    result = await process_batch(
        [],
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    assert isinstance(result, BatchResult)
    assert result.ok is True
    assert result.succeeded_items == []
    assert result.errors == []
    assert calls == []


# ============================================================================
# None key grouping
# ============================================================================


async def test_none_key_items_grouped_together_and_serial():
    """Items whose ``user_key`` is None share a single 'unknown' group and run
    serially among themselves (preserving arrival order)."""
    started_order = []

    async def process(item):
        started_order.append(item.id)
        await asyncio.sleep(0)

    items = [
        _item("n1", None),
        _item("n2", None),
        _item("n3", None),
    ]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    # All None-keyed items run serially in arrival order (one group).
    assert started_order == ["n1", "n2", "n3"]
    assert result.ok is True
    assert len(result.succeeded_items) == 3


async def test_none_key_group_failure_isolated_from_keyed_groups():
    """A failure in the None ('unknown') group must not stop keyed groups."""

    async def process(item):
        if item.id == "n1":
            raise RuntimeError("unknown-group failure")
        await asyncio.sleep(0)

    items = [
        _item("n1", None),  # raises
        _item("n2", None),  # skipped
        _item("k1", "keyed"),
    ]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    assert result.ok is False
    assert {i.id for i in result.succeeded_items} == {"k1"}


# ============================================================================
# All-clean
# ============================================================================


async def test_all_clean_records_every_item_and_no_errors():
    async def process(item):
        await asyncio.sleep(0)

    items = [
        _item("a1", "alice"),
        _item("b1", "bob"),
        _item("c1", "carol"),
    ]

    result = await process_batch(
        items,
        user_key=_user_key,
        process=process,
        max_in_flight_users=10,
    )

    assert result.ok is True
    assert result.errors == []
    assert {i.id for i in result.succeeded_items} == {"a1", "b1", "c1"}
