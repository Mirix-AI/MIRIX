"""Tests for the shared per-key lock registry.

The registry replaces three hand-rolled `dict[key, asyncio.Lock]` registries
(raw memory updates, per-agent skill evolution, per-user procedural dreams)
whose entries were never evicted — a slow leak on long-lived servers. These
tests pin the two properties the call sites rely on: per-key mutual exclusion,
and eviction back to empty once nobody holds or waits.
"""

import asyncio

import pytest

from mirix.helpers.keyed_locks import KeyedLocks


@pytest.mark.asyncio
async def test_mutual_exclusion_per_key():
    locks = KeyedLocks()
    order = []

    async def worker(tag: str):
        async with locks.acquire("k"):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)  # yield while holding — the other must wait
            order.append(f"{tag}-out")

    await asyncio.gather(worker("a"), worker("b"))

    # Critical sections must not interleave: each -in is followed by its -out.
    assert order in (["a-in", "a-out", "b-in", "b-out"],
                     ["b-in", "b-out", "a-in", "a-out"])


@pytest.mark.asyncio
async def test_different_keys_do_not_block_each_other():
    locks = KeyedLocks()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with locks.acquire("k1"):
            entered.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await entered.wait()
    # k2 is acquirable immediately even while k1 is held.
    async with locks.acquire("k2"):
        pass
    release.set()
    await task


@pytest.mark.asyncio
async def test_entries_are_evicted_when_idle():
    """The leak-fix property: the registry returns to empty after use."""
    locks = KeyedLocks()

    for i in range(100):
        async with locks.acquire(f"key-{i}"):
            pass

    assert len(locks) == 0


@pytest.mark.asyncio
async def test_entry_survives_while_a_waiter_is_queued():
    """Eviction must not drop a lock another coroutine is still waiting on —
    that would hand out a NEW lock for the same key and break exclusion."""
    locks = KeyedLocks()
    holding = asyncio.Event()
    proceed = asyncio.Event()
    order = []

    async def first():
        async with locks.acquire("k"):
            holding.set()
            await proceed.wait()
            order.append("first-out")

    async def second():
        await holding.wait()
        async with locks.acquire("k"):
            order.append("second-in")

    t1 = asyncio.create_task(first())
    t2 = asyncio.create_task(second())
    await holding.wait()
    await asyncio.sleep(0)  # let `second` queue up on the same entry
    assert len(locks) == 1  # one live entry, refcounted by holder + waiter
    proceed.set()
    await asyncio.gather(t1, t2)

    assert order == ["first-out", "second-in"]
    assert len(locks) == 0


@pytest.mark.asyncio
async def test_exception_inside_critical_section_still_evicts():
    locks = KeyedLocks()

    with pytest.raises(RuntimeError):
        async with locks.acquire("k"):
            raise RuntimeError("boom")

    assert len(locks) == 0
    # And the key is usable again afterwards.
    async with locks.acquire("k"):
        pass
