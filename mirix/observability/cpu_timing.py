"""Synchronous CPU-segment timing probe (perf diagnostics).

The event-loop lag monitor proves the loop gets blocked under concurrency; this
names the culprit. A coroutine that runs a chunk of *synchronous* CPU work
between awaits (tokenization, big string assembly, embedding-array math, JSON
of large payloads) blocks the single event loop for that whole chunk -- every
other ready coroutine waits. ``time_cpu`` brackets such a block and logs how
long it held the loop, so the high-lag windows can be attributed to a specific
section.

Unlike ``timed_span`` (which creates a Langfuse child span and is for I/O
attribution), this is a plain synchronous context manager: cheap, no tracing
dependency, and it measures wall time of code that does NOT await. Use it ONLY
around synchronous stretches -- if the block awaits, its measurement is
meaningless (it would include the await).

Emits ``[CPU SEGMENT] name=<n> ms=<t> tid=<tid>`` at INFO when the block exceeds
``warn_ms`` (default 50ms), so a quick block stays silent. No PII (name + timing
only). Correlate by TID and timestamp with [EVENT LOOP LAG] windows.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Only log a segment that held the loop at least this long. 50ms is well below
# the event-loop monitor's 250ms warn threshold, so contributing segments
# surface before they alone would trip a lag warning.
_DEFAULT_WARN_MS = 50.0


@contextmanager
def time_cpu(name: str, warn_ms: float = _DEFAULT_WARN_MS) -> Iterator[None]:
    """Time a synchronous CPU block; log if it held the event loop > warn_ms.

    Args:
        name: short identifier for the block (e.g. "count_tokens",
            "prompt_assembly", "embedding_pad").
        warn_ms: only emit when the block ran at least this long.

    MUST wrap synchronous code only -- no ``await`` inside, or the number
    includes I/O wait and is meaningless.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms >= warn_ms:
            tid = _safe_tid()
            logger.info("[CPU SEGMENT] name=%s ms=%.1f tid=%s", name, elapsed_ms, tid)


def _safe_tid() -> Optional[str]:
    try:
        from mirix.observability.context import get_tid

        return get_tid()
    except Exception:  # noqa: BLE001 - timing must never fail the wrapped work
        return None
