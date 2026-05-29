"""
SlowUpdateScheduler for MetaClaw v0.3.0.

Gates expensive RL weight updates (slow outer loop) so they only run
when the user is unavailable, preserving a smooth interactive experience.

Three idle-window triggers (any one is sufficient):
  1. Sleep hours      — configurable start/end time (e.g. 23:00–07:00)
  2. System idle      — user has been away from keyboard for N minutes
  3. Calendar busy    — Google Calendar shows an active event (user in meeting)

State machine
─────────────
  IDLE_WAIT  ──(window opens)──►  WINDOW_OPEN
  WINDOW_OPEN ─(trainer acks)──►  UPDATING
  UPDATING   ──(user active) ──►  PAUSING
  PAUSING    ──(trainer done) ──►  IDLE_WAIT

  WINDOW_OPEN ─(window closes)──► IDLE_WAIT   (trainer not yet started)

Meta-learning interpretation
─────────────────────────────
  Fast inner loop  = skill evolution after each session (always on)
  Slow outer loop  = RL gradient update (only during idle windows)

The scheduler is the mechanism that identifies task gaps — periods between
active usage sessions — and uses them exclusively for outer-loop updates.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import MetaClawConfig
    from .idle_detector import IdleDetector
    from .calendar_client import GoogleCalendarClient

logger = logging.getLogger(__name__)

_STATE_FILE = Path.home() / ".metaclaw" / "scheduler_state.json"
_CHECK_INTERVAL_SECONDS = 60


class SchedulerState(enum.Enum):
    IDLE_WAIT   = "idle_wait"    # waiting for a qualifying window
    WINDOW_OPEN = "window_open"  # window detected, trainer signalled
    UPDATING    = "updating"     # trainer is actively running
    PAUSING     = "pausing"      # user became active, waiting for trainer to stop


class SlowUpdateScheduler:
    """Background scheduler that gates slow RL updates to idle time windows.

    Parameters
    ----------
    config:
        MetaClawConfig.  Reads all ``scheduler_*`` fields.
    trigger_event:
        asyncio.Event shared with MetaClawTrainer.  Set when a window opens.
    pause_event:
        asyncio.Event shared with MetaClawTrainer.  Set when the user returns
        and the trainer should abort the current collection round.
    idle_detector:
        IdleDetector instance for system idle time queries.
    calendar_client:
        Optional GoogleCalendarClient.  When None, calendar-based detection
        is skipped silently.
    """

    def __init__(
        self,
        config: "MetaClawConfig",
        trigger_event: asyncio.Event,
        pause_event: asyncio.Event,
        idle_detector: "IdleDetector",
        calendar_client: Optional["GoogleCalendarClient"] = None,
    ) -> None:
        self.config = config
        self._trigger_event = trigger_event
        self._pause_event = pause_event
        self._idle_detector = idle_detector
        self._calendar_client = calendar_client

        self.state = SchedulerState.IDLE_WAIT
        self._stop_requested = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Run the scheduler loop. Intended as an asyncio task alongside trainer.run()."""
        logger.info(
            "[Scheduler] started — sleep=%s–%s idle=%dmin calendar=%s",
            self.config.scheduler_sleep_start,
            self.config.scheduler_sleep_end,
            self.config.scheduler_idle_threshold_minutes,
            "on" if self._calendar_client else "off",
        )
        while not self._stop_requested.is_set():
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("[Scheduler] tick error (ignored): %s", exc)
            await asyncio.sleep(_CHECK_INTERVAL_SECONDS)

    async def _tick(self) -> None:
        """One scheduler check cycle."""
        window_open = self._is_window_open()

        if self.state == SchedulerState.IDLE_WAIT:
            if window_open:
                self._transition(SchedulerState.WINDOW_OPEN)
                self._trigger_event.set()

        elif self.state == SchedulerState.WINDOW_OPEN:
            if not window_open:
                # Window closed before trainer could start — rescind the trigger.
                self._trigger_event.clear()
                self._transition(SchedulerState.IDLE_WAIT)

        elif self.state == SchedulerState.UPDATING:
            if not window_open:
                # User became active — ask trainer to stop.
                self._transition(SchedulerState.PAUSING)
                self._pause_event.set()

        elif self.state == SchedulerState.PAUSING:
            # Wait until trainer acknowledges pause (clears both events).
            if not self._trigger_event.is_set() and not self._pause_event.is_set():
                self._transition(SchedulerState.IDLE_WAIT)

    # ------------------------------------------------------------------ #
    # Trainer callbacks                                                   #
    # ------------------------------------------------------------------ #

    def notify_trainer_started(self) -> None:
        """Called by MetaClawTrainer when it acknowledges the trigger and begins work."""
        if self.state == SchedulerState.WINDOW_OPEN:
            self._transition(SchedulerState.UPDATING)

    def notify_trainer_finished(self) -> None:
        """Called by MetaClawTrainer when it completes or aborts one update cycle."""
        self._trigger_event.clear()
        self._pause_event.clear()
        self._transition(SchedulerState.IDLE_WAIT)

    # ------------------------------------------------------------------ #
    # Window detection                                                    #
    # ------------------------------------------------------------------ #

    def _is_window_open(self) -> bool:
        """Return True if any idle condition is met right now."""
        if self._sleep_hours_active():
            logger.debug("[Scheduler] window: sleep hours")
            return True

        idle_secs = self._safe_idle_seconds()
        threshold_secs = self.config.scheduler_idle_threshold_minutes * 60
        if idle_secs >= threshold_secs:
            logger.debug("[Scheduler] window: system idle %ds >= %ds", idle_secs, threshold_secs)
            return True

        if self._calendar_busy():
            logger.debug("[Scheduler] window: calendar event active")
            return True

        return False

    def _sleep_hours_active(self) -> bool:
        """Return True if the current local time falls within the configured sleep window."""
        try:
            now = datetime.now().time()
            start = datetime.strptime(self.config.scheduler_sleep_start, "%H:%M").time()
            end   = datetime.strptime(self.config.scheduler_sleep_end,   "%H:%M").time()
            if start <= end:
                # Same-day window, e.g. 01:00–06:00
                return start <= now <= end
            else:
                # Wraps midnight, e.g. 23:00–07:00
                return now >= start or now <= end
        except Exception as exc:
            logger.warning("[Scheduler] could not parse sleep hours: %s", exc)
            return False

    def _safe_idle_seconds(self) -> int:
        try:
            return self._idle_detector.idle_seconds()
        except Exception as exc:
            logger.debug("[Scheduler] idle_detector error: %s", exc)
            return 0

    def _calendar_busy(self) -> bool:
        if self._calendar_client is None:
            return False
        try:
            return self._calendar_client.is_busy_now()
        except Exception as exc:
            logger.debug("[Scheduler] calendar check error: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # State management                                                    #
    # ------------------------------------------------------------------ #

    def _transition(self, new_state: SchedulerState) -> None:
        if new_state != self.state:
            logger.info(
                "[Scheduler] %s → %s", self.state.value, new_state.value
            )
            self.state = new_state
            self._write_state_file()

    def _write_state_file(self) -> None:
        """Write scheduler state to a JSON file for `metaclaw status` to read."""
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "state": self.state.value,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "sleep_window": (
                    f"{self.config.scheduler_sleep_start}–{self.config.scheduler_sleep_end}"
                ),
                "idle_threshold_minutes": self.config.scheduler_idle_threshold_minutes,
            }
            _STATE_FILE.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.debug("[Scheduler] could not write state file: %s", exc)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Return a dict describing current scheduler state (for CLI / monitoring)."""
        return {
            "state": self.state.value,
            "sleep_window": (
                f"{self.config.scheduler_sleep_start}–{self.config.scheduler_sleep_end}"
            ),
            "idle_threshold_minutes": self.config.scheduler_idle_threshold_minutes,
            "calendar_enabled": self._calendar_client is not None,
        }

    def stop(self) -> None:
        """Signal the scheduler loop to exit on its next iteration."""
        self._stop_requested.set()
