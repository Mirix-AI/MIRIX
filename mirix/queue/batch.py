"""Shared batch concurrency core for queue consumers.

A single implementation of the batch-processing pattern used by every consumer
that pulls messages in batches: group a batch by user, run different users
concurrently (bounded), and keep each user's messages strictly in order. Both
the in-memory queue consumer and any external batch consumer (e.g. a stream
processor delivering message batches) call this helper, so they exercise the
exact same group-by-user / gather / semaphore / serial-within-a-user path
rather than maintaining parallel copies that can drift.

It operates on opaque ``items`` via injected callables (``user_key`` to group,
``process`` to handle one item), so it imports nothing from any specific
consumer or transport. Callers pass whatever message object they hold.

Semantics:

  * Items are grouped by ``user_key(item)``, preserving arrival order within
    each group (Kafka's partition key is user_id, so same-user items arrive in
    order and must be processed in order).
  * A ``None`` key forms its own "unknown" group, processed serially among its
    members (safest fallback for items whose key could not be extracted).
  * Groups are gathered concurrently; an ``asyncio.Semaphore`` bounds how many
    groups run at once.
  * Within a group, items run SERIALLY. On the FIRST exception in a group, the
    remaining items in that group are skipped (they are NOT recorded as
    succeeded), the exception is captured, and the group coroutine returns.
  * Each group coroutine catches its own exceptions internally and stashes
    them; nothing escapes to ``asyncio.gather``. This is load-bearing — a
    leaked exception under ``gather(...)`` without ``return_exceptions`` would
    cancel sibling groups mid-write and break the all-or-nothing outcome.
  * ``on_item_success(item)`` is invoked after each successful ``process(item)``
    (a caller that must acknowledge each item individually uses this hook).

By design, ``process_batch`` does NOT raise on a group error. It returns a
:class:`BatchResult` and leaves the reaction to the caller, keeping the core
free of any redelivery/ack policy: a caller backed by a redelivering broker can
raise ``errors[0]`` to force the whole batch to be redelivered, while a caller
backed by a re-enqueueable queue can re-enqueue the failed items.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

__all__ = ["BatchResult", "process_batch"]


@dataclass
class BatchResult:
    """Outcome of a :func:`process_batch` run.

    Attributes:
        succeeded_items: Items whose ``process`` completed without raising, in
            no particular order (groups run concurrently). Each item is the
            opaque object the caller passed in.
        errors: One entry per group that failed — the FIRST exception raised in
            that group. The caller decides how to react (e.g. re-raise
            ``errors[0]`` to trigger broker redelivery, or re-enqueue).
        ok: ``True`` iff no group raised (``errors`` is empty).
    """

    succeeded_items: List[Any] = field(default_factory=list)
    errors: List[BaseException] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


async def process_batch(
    items: List[Any],
    *,
    user_key: Callable[[Any], Optional[str]],
    process: Callable[[Any], Awaitable[None]],
    max_in_flight_users: int,
    on_item_success: Optional[Callable[[Any], None]] = None,
) -> BatchResult:
    """Process a batch of opaque items, grouped by user, with bounded
    cross-group concurrency and serial-within-a-group ordering.

    See the module docstring for the full semantics. This helper never raises
    on a per-item/per-group failure — it returns a :class:`BatchResult` and
    lets the caller decide what to do with ``result.errors``.

    Args:
        items: Opaque batch items in arrival order.
        user_key: Extracts the grouping key for an item. ``None`` groups items
            into a single "unknown" group processed serially among themselves.
        process: Async per-item processor. Its return value is ignored; raising
            stops the rest of that item's group.
        max_in_flight_users: Upper bound on concurrently-processed groups
            (``asyncio.Semaphore``). Caps memory/connection pressure.
        on_item_success: Optional callback invoked (synchronously) after each
            successful ``process(item)`` with that item. Never called for
            skipped/failed items.

    Returns:
        A :class:`BatchResult` carrying the succeeded items and per-group first
        errors.
    """
    result = BatchResult()

    if not items:
        return result

    # Group by key, preserving arrival order within each group. OrderedDict so
    # per-user lists stay in submit order.
    groups: "OrderedDict[Optional[str], List[Any]]" = OrderedDict()
    for item in items:
        key = user_key(item)
        groups.setdefault(key, []).append(item)

    semaphore = asyncio.Semaphore(max_in_flight_users)

    async def _process_group(group_items: List[Any]) -> None:
        """Process one group's items serially under the semaphore.

        Stops at the first exception — remaining items for this group are NOT
        recorded as succeeded. The exception is stashed in ``result.errors``;
        nothing escapes to ``asyncio.gather``.
        """
        async with semaphore:
            for item in group_items:
                try:
                    await process(item)
                except Exception as exc:
                    result.errors.append(exc)
                    return
                result.succeeded_items.append(item)
                if on_item_success is not None:
                    on_item_success(item)

    tasks = [_process_group(group_items) for group_items in groups.values()]
    await asyncio.gather(*tasks)

    return result
