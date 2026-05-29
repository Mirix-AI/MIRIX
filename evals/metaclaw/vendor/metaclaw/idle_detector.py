"""
Idle detection for MetaClaw scheduler.

Detects how long the user (or the proxy) has been inactive, to decide
when it is safe to run expensive slow RL updates without disrupting the
user's workflow.

Platform support:
  macOS  — parses IOHIDSystem HIDIdleTime via `ioreg` (stdlib only)
  Linux  — tries `xprintidle` (milliseconds), falls back to proxy activity
  other  — falls back to proxy activity tracker
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


class LastRequestTracker:
    """Thread-safe tracker of the last HTTP request received by the proxy.

    Used as an idle-detection fallback when platform-specific tools are
    unavailable (e.g. headless Linux, Windows, Docker containers).
    """

    def __init__(self) -> None:
        self._last: float = time.time()
        self._lock = threading.Lock()

    def touch(self) -> None:
        """Record that a request was received right now."""
        with self._lock:
            self._last = time.time()

    def seconds_since_last(self) -> int:
        """Return integer seconds since the last recorded request."""
        with self._lock:
            return int(time.time() - self._last)


class IdleDetector:
    """Detect system-level user idle time.

    Tries platform-specific methods first; silently falls back to
    ``last_request_tracker`` if they are unavailable or fail.

    Parameters
    ----------
    fallback_tracker:
        A :class:`LastRequestTracker` instance shared with the proxy server.
        Used when native idle detection is unavailable.
    """

    def __init__(self, fallback_tracker: LastRequestTracker | None = None) -> None:
        self._fallback = fallback_tracker

    def idle_seconds(self) -> int:
        """Return integer seconds the system has been idle.

        Falls back gracefully through the platform detection chain.
        Always returns a non-negative integer.
        """
        import sys
        try:
            if sys.platform == "darwin":
                return self._macos_idle()
            if sys.platform.startswith("linux"):
                return self._linux_idle()
        except Exception as exc:
            logger.debug("[IdleDetector] platform idle detection failed: %s", exc)
        return self._fallback_idle()

    # ------------------------------------------------------------------ #
    # Platform implementations                                            #
    # ------------------------------------------------------------------ #

    def _macos_idle(self) -> int:
        """Read HIDIdleTime from IOHIDSystem via `ioreg`.

        Returns idle time in seconds (nanoseconds ÷ 1e9).
        Raises RuntimeError if the value cannot be parsed.
        """
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "HIDIdleTime" in line:
                # Line format:   "HIDIdleTime" = 12345678900
                parts = line.split("=")
                nanoseconds = int(parts[-1].strip())
                return nanoseconds // 1_000_000_000
        raise RuntimeError("HIDIdleTime not found in ioreg output")

    def _linux_idle(self) -> int:
        """Use `xprintidle` (milliseconds → seconds).

        Requires an X11 session. Raises RuntimeError if unavailable.
        """
        result = subprocess.run(
            ["xprintidle"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(f"xprintidle failed: {result.stderr.strip()}")
        return int(result.stdout.strip()) // 1000

    def _fallback_idle(self) -> int:
        """Fall back to time since last proxy HTTP request."""
        if self._fallback is not None:
            return self._fallback.seconds_since_last()
        # No tracker available — conservatively report 0 (never idle)
        return 0
