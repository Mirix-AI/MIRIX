"""Per-key asyncio.Lock registry with automatic eviction.

The naive pattern — a module-level ``dict[key, asyncio.Lock]`` filled lazily —
leaks: entries are never removed because a bare Lock cannot know when its last
user is done, so a long-lived server accumulates one Lock per key it has ever
seen. ``KeyedLocks`` reference-counts holders and waiters and drops an entry
the moment nobody holds or awaits it.

Used to serialize per-entity work across coroutines in ONE process (per-raw-
memory updates, per-agent skill evolution, per-user procedural dreams). It is
NOT a cross-process lock — pair it with a Postgres advisory lock when multiple
server processes contend (see auto_dream_manager._procedural_dream_guard).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Tuple


class KeyedLocks:
    """Lazily-created, self-evicting registry of per-key ``asyncio.Lock``s.

    All bookkeeping runs synchronously between awaits on the single event
    loop, so the get-or-create and the refcount updates are atomic without any
    extra locking of their own.
    """

    def __init__(self) -> None:
        # key -> (lock, number of coroutines holding or waiting on it)
        self._entries: Dict[str, Tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        """``async with locks.acquire(key):`` — mutual exclusion per key."""
        lock, refs = self._entries.get(key) or (asyncio.Lock(), 0)
        self._entries[key] = (lock, refs + 1)
        try:
            async with lock:
                yield
        finally:
            lock, refs = self._entries[key]
            if refs <= 1:
                del self._entries[key]
            else:
                self._entries[key] = (lock, refs - 1)

    def __len__(self) -> int:
        """Number of live entries — 0 when nothing is held or awaited."""
        return len(self._entries)
