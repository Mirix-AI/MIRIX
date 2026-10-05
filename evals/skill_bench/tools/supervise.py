"""Keeps the matrix scheduler running, with retries and resumes, so that no single
interruption forces the experiment to start over.

::

    python -m evals.skill_bench.tools.supervise <every schedule.py option> \\
        [--notify-cmd CMD] [--no-caffeinate] [--no-drain-on-stop] [--drain-timeout 180] \\
        [--max-crash-restarts 20] [--restart-backoff-cap 300] [--heartbeat 30] \\
        [--budget-unknown-grace 1800] [--one-off-again ID]
    python -m evals.skill_bench.tools.supervise --status --runs-dir RUNS

It runs ``schedule.py`` with the given options plus ``--auto-resume`` (the scheduler
re-runs interrupted cells by itself, with backoff and caps) and:

1. takes ``runs/_supervise/supervisor.lock`` (another supervisor: exit 13);
2. on macOS keeps the machine awake with ``caffeinate -dimsu -w <own pid>`` (respawned if
   it dies; ``-s`` holds only on AC power, and a closed lid still sleeps the laptop);
3. starts the scheduler in its own session (``_supervise/scheduler.log``), or adopts a
   live one (its lock is held and ``state.json`` names a running process: a supervisor
   that was restarted), and while it runs writes ``_supervise/heartbeat.json`` every
   ``--heartbeat`` seconds (the scheduler's counts, pause and budget, the host's
   resources, the open alerts) and notes a host that slept (the wall clock ran ahead of
   the monotonic one). While the scheduler runs it alerts at once when a cell comes to
   need a person (``cell_needs_attention``: a HARNESS stop, a resume cap, a refused or
   violation cell, a MIRIX gateway it cannot supply; the others go on), when it adopted
   a gateway a live worker uses that is not healthy or runs with another budget
   (``gateway_needs_attention``: ``state.json`` ``gateways``' ``problem``), and when a
   running cell records a problem it went on past (``recorded_problem``:
   ``cell_records.recorded_problems`` of its highest attempt; a growing count again at
   most once per ``--stall-alert``). A gateway budget or balance that stays unknown is
   alerted every ``--budget-unknown-grace`` while the scheduler waits (nothing is sent
   meanwhile; it never stops the run). It raises an alert, and goes on waiting, when
   the run stalls for ``--stall-alert``: a pause the scheduler waits out (a
   low balance, a key limit that does not reset, an upstream 402), a cell the resource
   gate holds while nothing runs, or the scheduler restarted again and again on a pause
   (an unreachable gateway); and when the disk under the runs directory has less than
   the scheduler's ``--min-free-disk-gb`` free while workers run; and when the
   scheduler's ``state.json`` has not changed for ``--hung-alert`` (``max(10 * --poll,
   600)`` seconds) while its process lives: it may hang (nothing is killed);
4. when the scheduler exits: 0, every cell complete (exit 0); 2, some cells need a person
   (alert, exit 2); 3 on a spent budget (alert: a person raises the budget, and every
   host is relaunched with the new ``--budget``; exit 10) or an invalid one (the key's usage
   below the budget start: another key, the cap cannot be computed; exit 11); 3 on any
   other pause (an unreachable gateway), restart after ``--pause-restart-wait``; 4 (the
   scheduler was refused, e.g. another scheduler holds the runs directory), retry every
   ``--refused-retry``, and once the first refusal is ``--refused-timeout`` old one
   ``refused`` alert and retries at most every ``--restart-backoff-cap``; anything else
   (a crash, a signal), restart after ``min(cap, 10 * 2**(k-1))`` seconds for the k-th
   crash: more than ``--max-crash-restarts`` crashes is a crash loop, one ``crash_loop``
   alert, and the restarts go on (refusals and crash loops never end the supervisor).
   Every exit that is not the end belongs to one streak of runs that did nothing
   useful, whatever their codes: only a run of ``--crash-window`` (one hour) or a cell
   completed since the last exit ends it (and re-arms those alerts), so a scheduler
   alternating crashes, pauses and refusals still gets its alert, and a streak as long
   as ``--stall-alert`` raises one too. Before exit 10 or 11, a scheduler that crashed
   or was stopped left its workers and gateway behind: they are drained as in 5
   (``--no-drain-on-stop`` applies to a signal only);
5. on ``SIGTERM``/``SIGINT``: stops the scheduler (it launches nothing more and leaves its
   workers), then, with the default ``--drain-on-stop``, sends ``SIGTERM`` to every live
   ``run.py`` of the runs directory and waits up to ``--drain-timeout``. ``run.py`` exits
   ``interrupted``: a learning attempt becomes ``learning_interrupted`` with its
   checkpoints intact, a test phase leaves its items pending. Once the last worker is
   gone, the gateway a stopped scheduler launched is stopped too. A worker that started
   after the first scan (a scheduler launching as it was stopped) is signalled as soon
   as it shows; the drain ends only once no worker has shown for ``--drain-settle``
   seconds and no worker the stopped scheduler launched (its last ``state.json``
   ``active``) is still starting, so a worker that has not taken its cell lock yet does
   not lose its gateway. When the drain times out, a detached reaper
   (``--reap-gateway``) waits for the last worker and then stops that gateway, so it
   does not stay up with the key loaded. Exit 130. A second signal kills the scheduler
   and the workers (``SIGKILL``); everything stays resumable at the cost of at most one
   in-flight item per cell.

A spent budget and a low balance are handled differently: ``budget_spent`` is the
whole budget used up, so nothing relaunches (exit 10); a low balance, the key's daily
limit (``balance_below_floor``) and an upstream 402 are waited out by the scheduler
itself, which launches again when the gateway's pause lifts.

Events go to ``_supervise/events.jsonl``, the current alert text to ``_supervise/ALERT``
(cleared at start; the newest alert that needs a person on the first line, then the
newest ``recorded_problem`` alerts, which never take that line), and ``--notify-cmd``
runs on every alert (split with ``shlex``, never through a shell) with the text in
``SKILL_BENCH_ALERT``, first (a full disk must not keep the operator from hearing of
it) and in the workers' environment, with no provider key or other credential.
Writing a record never takes the supervisor down: a full disk only loses the record.
A notice (``--notify-cmd`` and an ``events.jsonl`` ``notice``, never ``ALERT``) tells
of what needs no action: the new attempts ``--new-attempt-cells`` starts, with their
estimate (what the attempts they replace cost; within the budget no separate
approval). The one-off scheduler options (``--reset-resume``,
``--rerun-harness-failures``, ``--new-attempt-cells``, whose ``--new-attempt-note`` is
part of its identity, and ``--widen-selection``) go to the schedulers this supervisor
starts until one has applied them (its ``state.json`` says ``one_off_applied``), and
never again from this runs directory, whichever supervisor starts it with the same
command line: ``_supervise/one_off.json`` records each one (option, cells and
``--one-off-again`` id) durably as ``pending`` before the scheduler that gets it
starts, then ``applied``. A supervisor that finds one recorded, ``pending`` included
(it died before it learned the outcome), leaves it out and raises an alert;
``--one-off-again ID`` (a new id each time) applies it once more on purpose. ``--status``
prints the heartbeat summary (exit 1 when it is missing or stale, or the scheduler's
``state.json`` is older than ``--hung-alert``). An unexpected error in the supervisor
itself raises an alert and exits :data:`SUPERVISE_INTERNAL_ERROR` 70, which the systemd
unit restarts (130 when it failed while carrying out a stop). A command line that
names a method the study removed (``study.REMOVED_METHODS``), or one of its cells, is
refused at start (exit 2, which the unit does not restart) after the same alert
(:func:`refused_start_alert`): it is most likely a unit's or a launch's command line
written before the removal. So is one that selects cells no earlier launch of the
runs directory selected, without ``--widen-selection`` (``schedule.selection_problem``:
e.g. a relaunch without the method list an earlier launch had). Exit codes:
:data:`SUPERVISE_COMPLETE` 0, :data:`SUPERVISE_ATTENTION` 2,
:data:`SUPERVISE_BUDGET_SPENT` 10, :data:`SUPERVISE_BUDGET_INVALID` 11,
:data:`SUPERVISE_LOCKED` 13 (another supervisor holds the runs directory, or it is a
merged view), :data:`SUPERVISE_INTERNAL_ERROR` 70, :data:`SUPERVISE_STOPPED` 130.
:data:`SUPERVISE_CRASH_LOOP` 12 is never returned.

On an x86_64 Linux host with Docker, a systemd user unit (``skill-bench@.service``,
rendered from ``infra/skill-bench@.service.in`` by ``tools/phase16.sh install-unit``:
``Restart=on-failure``, ``RestartPreventExitStatus=0 2 10 11 12 13 130``,
``KillSignal=SIGTERM``, ``KillMode=process``, ``TimeoutStopSec=240``) restarts the
supervisor after a reboot and a crash (exit 70 included), never after a stop it was
asked for (exit 130).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.skill_bench import cell_records, protocol, schedule, study
from evals.skill_bench.cell_records import (
    MERGED_MARKER,
    NON_CELL_DIRS,
    STATUS_RUNNING,
    CellPaths,
    host_info,
    now,
    read_json,
    write_json_atomic,
)
from evals.skill_bench.infra import resources
from evals.skill_bench.infra.gateway import (
    BUDGET_INVALID,
    BUDGET_SPENT,
    BUDGET_UNKNOWN,
)
from evals.skill_bench.infra.processes import (
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    live_cell_workers,
    lock_is_held,
    process_start_time,
    signal_group,
    stop_orphaned,
    sweep_owned,
    tail,
)

log = logging.getLogger("evals.skill_bench.supervise")

SCHEDULER_MODULE = "evals.skill_bench.schedule"
SUPERVISE_DIR = "_supervise"
ALERT_ENV = "SKILL_BENCH_ALERT"

SUPERVISE_COMPLETE = 0
SUPERVISE_ATTENTION = 2
SUPERVISE_BUDGET_SPENT = 10
SUPERVISE_BUDGET_INVALID = 11
#: Never returned (a crash loop only alerts and the supervisor keeps restarting); kept
#: so a systemd unit's ``RestartPreventExitStatus`` list that names it stays valid.
SUPERVISE_CRASH_LOOP = 12
SUPERVISE_LOCKED = 13
#: An unexpected error in the supervisor itself (EX_SOFTWARE). Not in the
#: systemd unit's ``RestartPreventExitStatus``: the unit restarts it, and the new
#: supervisor adopts the scheduler that went on running.
SUPERVISE_INTERNAL_ERROR = 70
SUPERVISE_STOPPED = 130
#: ``--status`` when there is no fresh heartbeat (or the scheduler seems hung).
STATUS_STALE = 1

#: Pauses that mean the budget or balance cannot be read at all: waited out by the
#: scheduler (nothing is sent meanwhile), with an alert every ``--budget-unknown-grace``
#: (never a stop).
UNKNOWN_PAUSES = frozenset({BUDGET_UNKNOWN, schedule.PAUSE_BALANCE_UNKNOWN})
#: Scheduler options that act once (at the first start that applies them), not at every
#: restart: a restart must neither clear the counters again nor release the HARNESS
#: stops that happened since the person looked, nor start the new attempts a person
#: asked for a second time.
ONE_OFF_SCHEDULER_OPTIONS: tuple[str, ...] = (
    "--reset-resume",
    "--rerun-harness-failures",
    "--new-attempt-cells",
    # A selection wider than every earlier launch's, once, on purpose.
    "--widen-selection",
)
#: The option whose value is part of a ``--new-attempt-cells`` request's identity: a
#: new note is a new request (the scheduler reruns a cell once per note).
NEW_ATTEMPT_NOTE_OPTION = "--new-attempt-note"
#: ``_supervise/one_off.json``: the one-off options handed to a scheduler of this runs
#: directory, by request.
ONE_OFF_FILE = "one_off.json"
#: Scheduler exits at an idle point, on its own: anything else (a crash, a signal) may
#: have left workers behind.
_ENDED_ON_ITS_OWN = frozenset(
    {
        schedule.EXIT_ALL_COMPLETE,
        schedule.EXIT_ATTENTION,
        schedule.EXIT_PAUSED,
        schedule.EXIT_REFUSED,
    }
)
#: How long a detached gateway reaper waits for the last worker after a drain timeout.
REAP_TIMEOUT_S = 3600.0
#: A drain ends once no worker has shown for this long.
DRAIN_SETTLE_S = 5.0
#: The scheduler rewrites ``state.json`` every ``--poll``: unchanged for this many polls
#: (and at least :data:`HUNG_MIN_S`) while it lives, it may hang.
HUNG_POLLS = 10
HUNG_MIN_S = 600.0
#: A heartbeat older than this many intervals is stale (``--status``).
STALE_HEARTBEATS = 3
NOTIFY_TIMEOUT_S = 30.0
#: Alerts of what a running cell recorded and went on past: raised and notified like
#: any other, but never ``ALERT``'s first line.
RECORDED_ALERTS = frozenset({"recorded_problem"})
#: ``ALERT`` lists at most this many recorded problems, the newest.
ALERT_FILE_RECORDED_MAX = 5
#: ``heartbeat.json`` keeps at most this many alerts, the newest; ``events.jsonl`` keeps
#: every one.
HEARTBEAT_ALERTS_MAX = 50


@dataclass(frozen=True)
class SuperviseOptions:
    runs_dir: Path
    #: The scheduler's own command-line options (``--auto-resume`` included).
    scheduler_args: tuple[str, ...] = ()
    python: str = sys.executable
    scheduler_module: str = SCHEDULER_MODULE
    notify_cmd: str | None = None
    caffeinate: bool = True
    drain_on_stop: bool = True
    drain_timeout_s: float = 180.0
    max_crash_restarts: int = 20
    restart_backoff_base_s: float = 10.0
    restart_backoff_cap_s: float = 300.0
    #: A scheduler that ran this long before crashing ends a crash loop's count.
    crash_window_s: float = 3600.0
    heartbeat_s: float = 30.0
    budget_unknown_grace_s: float = 1800.0
    #: Wait before restarting a scheduler that exited on a pause it does not wait out.
    pause_restart_wait_s: float = 300.0
    refused_retry_s: float = 60.0
    refused_timeout_s: float = 1800.0
    #: How long a stopping scheduler may take before it is killed (the drain waits).
    scheduler_stop_timeout_s: float = 30.0
    #: A wall-clock lead over the monotonic clock beyond this is a host sleep.
    sleep_drift_s: float = 60.0
    #: A stall this long raises an alert (the supervisor goes on waiting).
    stall_alert_s: float = 1800.0
    #: The scheduler's ``--min-free-disk-gb``: below it while workers run, an alert.
    min_free_disk_gb: float | None = None
    #: A drain ends once no worker has shown for this long.
    drain_settle_s: float = DRAIN_SETTLE_S
    #: The scheduler's ``--poll``: it rewrites ``state.json`` that often.
    scheduler_poll_s: float = schedule.DEFAULT_POLL_S
    #: ``state.json`` unchanged this long while the scheduler lives raises an alert
    #: (None: :attr:`hung_after_s`'s default).
    hung_alert_s: float | None = None
    #: ``--one-off-again``: the one-off options go once more, as a new request under
    #: this id, although the same ones were applied before.
    one_off_again: str | None = None
    #: The ``--new-attempt-cells`` that never ran under ``runs_dir`` while another named
    #: one did: they start as first attempts, recorded at start as an
    #: ``events.jsonl`` notice.
    first_attempt_cells: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_dir", Path(self.runs_dir).absolute())
        object.__setattr__(self, "scheduler_args", tuple(self.scheduler_args))
        object.__setattr__(self, "first_attempt_cells", tuple(self.first_attempt_cells))
        for name in (
            "drain_timeout_s",
            "restart_backoff_base_s",
            "restart_backoff_cap_s",
            "crash_window_s",
            "heartbeat_s",
            "budget_unknown_grace_s",
            "pause_restart_wait_s",
            "refused_retry_s",
            "refused_timeout_s",
            "scheduler_stop_timeout_s",
            "sleep_drift_s",
            "stall_alert_s",
            "drain_settle_s",
            "scheduler_poll_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not self.heartbeat_s > 0:
            raise ValueError("the heartbeat interval must be positive")
        if self.max_crash_restarts < 0:
            raise ValueError("--max-crash-restarts cannot be negative")
        if self.hung_alert_s is not None and not self.hung_alert_s > 0:
            raise ValueError("--hung-alert must be positive")
        if self.one_off_again is not None and not self.one_off_again.strip():
            raise ValueError("--one-off-again needs a non-empty id")

    @property
    def control(self) -> Path:
        return self.runs_dir / SUPERVISE_DIR

    @property
    def hung_after_s(self) -> float:
        """How long ``state.json`` may stay unchanged while the scheduler lives:
        ``--hung-alert``, else ten of its polls and at least ten minutes."""
        if self.hung_alert_s is not None:
            return self.hung_alert_s
        return max(HUNG_POLLS * self.scheduler_poll_s, HUNG_MIN_S)

    @property
    def schedule_dir(self) -> Path:
        return self.runs_dir / "_schedule"


# ---------------------------------------------------------------------------
# The scheduler process: spawned here, or adopted
# ---------------------------------------------------------------------------


@dataclass
class SchedulerHandle:
    """The scheduler the supervisor watches."""

    pid: int
    #: ``ps -o lstart`` of ``pid``: signals go only to that process (never a reused pid).
    started: str | None
    process: asyncio.subprocess.Process | None = None
    started_at: str = field(default_factory=now)
    #: The supervisor's monotonic clock at the start (how long it ran before a crash).
    started_mono: float = 0.0

    @property
    def adopted(self) -> bool:
        return self.process is None

    def alive(self) -> bool:
        """Blocking for an adopted scheduler (one ``ps``)."""
        if self.process is not None:
            return self.process.returncode is None
        current = process_start_time(self.pid)
        return current is not None and current == self.started

    def signal(self, sig: int, *, group: bool = False) -> bool:
        """Send ``sig`` to the scheduler (``group``: its whole session, which it leads).
        Blocking for an adopted scheduler."""
        if not self.alive():
            return False
        if group and self._leads_group():
            return signal_group(self.pid, sig)
        try:
            os.kill(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _leads_group(self) -> bool:
        try:
            return os.getpgid(self.pid) == self.pid
        except OSError:
            return False

    async def wait(self, state_path: Path, poll_s: float = 1.0) -> int:
        """Its exit code. An adopted scheduler is not our child: its final ``state.json``
        write carries the code; without one it died (a crash, 1)."""
        if self.process is not None:
            return await self.process.wait()
        while await asyncio.to_thread(self.alive):
            await asyncio.sleep(poll_s)
        try:
            state = await asyncio.to_thread(read_json, state_path)
        except (OSError, ValueError):
            state = None
        if (
            isinstance(state, dict)
            and state_is_of(state, self.pid, self.started)
            and isinstance(state.get("exit_code"), int)
        ):
            return int(state["exit_code"])
        return 1


def state_is_of(state: Mapping[str, Any], pid: Any, started: str | None) -> bool:
    """Whether ``state.json`` was written by the scheduler ``pid`` started at ``started``
    (``ps -o lstart``). A scheduler that reused a dead one's pid (likely after a reboot)
    never inherits its state: its exit code, its ``one_off_applied``. A state or a
    handle without a start time is compared by pid alone."""
    if not isinstance(pid, int) or isinstance(pid, bool) or state.get("pid") != pid:
        return False
    recorded = state.get("started")
    if not started or not recorded:
        return True
    return " ".join(str(recorded).split()) == " ".join(str(started).split())


def _kill_worker(pid: int, sig: int) -> bool:
    """Signal a live ``run.py`` and what runs with it: its whole group when it leads one (a
    scheduler-launched worker runs in its own session, with its ALFWorld games), else the
    process alone."""
    try:
        if os.getpgid(pid) == pid:
            return signal_group(pid, sig)
    except OSError:
        return False
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _starting_worker(pid: int, cell: str) -> bool:
    """``pid`` lives and its command line runs ``cell`` (a ``--out`` path ending in the
    cell id; never a stranger that reused the pid): a worker the stopped scheduler
    launched, which may not hold its cell lock yet. Blocking (one ``ps``)."""
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(Path(token).name == cell for token in result.stdout.split())


# ---------------------------------------------------------------------------
# One-off scheduler options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OneOffRequest:
    """A one-off scheduler option as a command line gives it: the option, its cells, and
    the ``--one-off-again`` id that came with it."""

    option: str
    values: tuple[str, ...] = ()
    again: str | None = None
    #: ``--new-attempt-cells`` only: its ``--new-attempt-note`` (a new note is a new
    #: request, as the scheduler reruns a cell once per note).
    note: str | None = None

    @property
    def key(self) -> str:
        """Its identity in ``one_off.json``: the same option, cells (in any order), id
        (and note) are the same request, however often the command line runs again."""
        parts: list[Any] = [self.option, sorted(set(self.values)), self.again]
        if self.note is not None:
            parts.append(self.note)  # other options keep the keys recorded before
        blob = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def describe(self) -> str:
        text = " ".join([self.option, *self.values])
        return text if self.note is None else f"{text} (note: {self.note})"


def one_off_requests(
    argv: Sequence[str], again: str | None = None
) -> list[OneOffRequest]:
    """The one-off options of a scheduler command line, the last occurrence of each (as
    argparse reads it), with the ``--one-off-again`` id ``again``."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    note: str | None = None
    taking_note = False
    for token in argv:
        if taking_note:
            note, taking_note = token, False
            continue
        if token.startswith("-"):
            name, eq, value = token.partition("=")
            current = None
            if name == NEW_ATTEMPT_NOTE_OPTION:
                if eq:
                    note = value
                else:
                    taking_note = True
            elif name in ONE_OFF_SCHEDULER_OPTIONS:
                current = [value] if eq else []
                found[name] = current
        elif current is not None:
            current.append(token)
    return [
        OneOffRequest(
            name,
            tuple(found[name]),
            again,
            note if name == "--new-attempt-cells" else None,
        )
        for name in ONE_OFF_SCHEDULER_OPTIONS
        if name in found
    ]


def read_one_off_record(path: Path) -> dict[str, dict[str, Any]]:
    """``one_off.json``'s entries by request key (empty when there is no file yet).
    Raises ``ValueError`` (or ``OSError``) when it cannot be read as one: nobody can then
    tell what was applied, and the caller withholds every one-off option. Blocking."""
    value = read_json(path)
    if value is None:
        return {}
    requests = value.get("requests") if isinstance(value, dict) else None
    if not isinstance(requests, dict) or not all(
        isinstance(entry, dict) for entry in requests.values()
    ):
        raise ValueError(f"{path} is not a one-off record")
    return {str(key): dict(entry) for key, entry in requests.items()}


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

#: ``(argv) -> process`` for caffeinate (tests inject one).
Spawner = Callable[[Sequence[str]], Awaitable[Any]]
#: ``(seconds) -> None``: the supervisor's waits between restarts (tests record them).
PauseHook = Callable[[float], Awaitable[None]]


async def _spawn_quiet(argv: Sequence[str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


class Supervisor:
    def __init__(
        self,
        options: SuperviseOptions,
        *,
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        pause: PauseHook | None = None,
        which: Callable[[str], str | None] = shutil.which,
        system: str = sys.platform,
        spawn_caffeinate: Spawner = _spawn_quiet,
        sampler: Callable[..., resources.HostResources] = resources.sample,
    ) -> None:
        self.options = options
        self.wall = wall
        self.mono = mono
        self._pause_hook = pause
        self.which = which
        self.system = system
        self.spawn_caffeinate = spawn_caffeinate
        self.sampler = sampler
        self.handle: SchedulerHandle | None = None
        self.restarts = 0
        self.crash_restarts = 0
        self.last_exit: dict[str, Any] | None = None
        self.alerts: list[dict[str, Any]] = []
        #: The newest alert that needs a person: ``ALERT``'s first line.
        self._current_alert: dict[str, Any] | None = None
        self.exit_code: int | None = None
        self.started_at = now()
        self.caffeinate: Any = None
        #: One streak of exits that were no useful run, whatever their codes: since
        #: when (monotonic), their codes, whether its alert went out; the
        #: crashes in it and its first refusal. Only a run of --crash-window or a cell
        #: completed since the last exit ends it.
        self._streak_since: float | None = None
        self._streak_codes: list[int] = []
        self._streak_longest_s = 0.0
        self._streak_alerted = False
        self._consecutive_crashes = 0
        self._refused_since: float | None = None
        #: The streak's crash-loop and refusal alerts went out (once per streak; the
        #: supervisor goes on restarting).
        self._crash_alerted = False
        self._refused_alerted = False
        #: The most complete cells a scheduler's state.json has shown (progress).
        self._complete_seen = 0
        self._unknown_since: float | None = None
        #: When (monotonic) the last unknown-budget alert went out: one every
        #: --budget-unknown-grace while it lasts (never a stop).
        self._unknown_alerted_at: float | None = None
        #: The cells that need a person as last seen in the watched scheduler's
        #: state.json: {cell: (status, reason)}.
        self._attention_seen: dict[str, tuple[Any, Any]] = {}
        #: The extra gateways the watched scheduler adopted with a problem, as last seen
        #: in its state.json: {port: problem} (schedule.adopt_live_gateways).
        self._gateway_problems_seen: dict[str, str] = {}
        #: Recorded problems: (cell, code) -> (count alerted, when).
        self._problems_alerted: dict[tuple[str, str], tuple[int, float]] = {}
        #: Stall watches: what is stalled, since when (monotonic), and whether its alert
        #: went out.
        self._stalls: dict[str, tuple[Any, float, bool]] = {}
        self._last_tick: tuple[float, float] | None = None
        self._stop: asyncio.Event | None = None
        self._escalate: asyncio.Event | None = None
        self.signals = 0
        self.stopping = False
        self.host: dict[str, Any] = {}
        self.own_started: str | None = None
        #: ``one_off.json`` as last read or written, and why it cannot be
        #: read (every one-off option is then withheld).
        self._one_off: dict[str, dict[str, Any]] = {}
        self._one_off_error: str | None = None
        self._one_off_lock = asyncio.Lock()
        #: Serializes :meth:`_note_one_off_applied` (the heartbeat and an exit).
        self._applied_lock = asyncio.Lock()
        #: Held for each heartbeat tick: the end waits for a tick that is alerting (its
        #: notify command) instead of cancelling it half-way (the alert, or a notice,
        #: would be lost with the state it already recorded as sent).
        self._tick_lock = asyncio.Lock()
        #: The one-off requests handed to the scheduler being watched (settled at its exit).
        self._one_off_passed: list[str] = []
        #: The last scheduler may have left workers behind: a terminal exit drains
        #: them.
        self._left_workers = False
        #: An unexpected error of the supervisor itself.
        self.error: str | None = None
        self._last_argv: list[str] = []

    # -- records -------------------------------------------------------------------------

    async def event(self, name: str, **fields: Any) -> None:
        """Append one record to ``events.jsonl``; a write that fails (a full disk) is
        logged and dropped, never raised."""
        record = {"at": now(), "event": name, "pid": os.getpid(), **fields}
        log.info("%s %s", name, json.dumps(fields, default=str)[:500])
        try:
            await asyncio.to_thread(
                append_jsonl, self.options.control / "events.jsonl", record
            )
        except OSError as exc:
            log.error("cannot record event %s: %s", name, exc)

    async def alert(self, kind: str, text: str, **fields: Any) -> None:
        """Raise an alert: ``--notify-cmd`` with the text in ``SKILL_BENCH_ALERT`` first
        (a full disk must not keep the operator from hearing of it), then
        ``events.jsonl`` and ``ALERT``, each best-effort.

        ``ALERT`` (what ``--status`` prints) holds the newest alert that
        needs a person on its first line, then the newest few recorded problems
        (:data:`RECORDED_ALERTS`, at most :data:`ALERT_FILE_RECORDED_MAX`): one never
        takes the first line from the other. The heartbeat's ``alerts`` keep one entry
        per recorded (cell, code), its latest count, and at most
        :data:`HEARTBEAT_ALERTS_MAX` in all (the oldest recorded problem goes first)."""
        entry: dict[str, Any] = {"at": now(), "kind": kind, "text": text}
        if kind in RECORDED_ALERTS:
            entry.update(
                {k: fields[k] for k in ("cell", "code", "count") if k in fields}
            )
            same = (entry.get("cell"), entry.get("code"))
            self.alerts = [
                a
                for a in self.alerts
                if a["kind"] != kind or (a.get("cell"), a.get("code")) != same
            ]
        else:
            self._current_alert = entry
        self.alerts.append(entry)
        while len(self.alerts) > HEARTBEAT_ALERTS_MAX:
            # The oldest recorded problem goes first: an alert that needs a person
            # stays while there is any.
            self.alerts.pop(
                next(
                    (
                        i
                        for i, a in enumerate(self.alerts)
                        if a["kind"] in RECORDED_ALERTS
                    ),
                    0,
                )
            )
        await self._notify(text)
        await self.event("alert", kind=kind, text=text, **fields)
        try:
            await asyncio.to_thread(
                _write_text_atomic, self.options.control / "ALERT", self._alert_text()
            )
        except OSError as exc:
            log.error("cannot write the ALERT file: %s", exc)

    def _alert_text(self) -> str:
        """The ``ALERT`` file: the newest alert that needs a person, then
        the newest recorded problems, one ``<at> <text>`` line each."""
        recorded = [a for a in reversed(self.alerts) if a["kind"] in RECORDED_ALERTS]
        shown = [self._current_alert] if self._current_alert is not None else []
        shown += recorded[:ALERT_FILE_RECORDED_MAX]
        return "".join(f"{a['at']} {a['text']}\n" for a in shown)

    async def notice(self, kind: str, text: str, **fields: Any) -> None:
        """A notification that needs no action: ``--notify-cmd`` with the text
        in ``SKILL_BENCH_ALERT`` and an ``events.jsonl`` ``notice`` record; never the
        ``ALERT`` file nor the heartbeat's open alerts."""
        await self._notify(text)
        await self.event("notice", kind=kind, text=text, **fields)

    async def _notify(self, text: str) -> None:
        command = self.options.notify_cmd
        if not command:
            return
        try:
            argv = shlex.split(command)
            process = await asyncio.create_subprocess_exec(
                *argv,
                # The workers' environment: no provider key or other credential goes to
                # a notifier, which may be a third party's program.
                env={**schedule.worker_env(), ALERT_ENV: text},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            await self.event(
                "notify_failed", error=f"{type(exc).__name__}: {exc}"[:300]
            )
            return
        try:
            code = await asyncio.wait_for(process.wait(), NOTIFY_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            await self.event("notify_failed", error="timeout")
            return
        if code != 0:
            await self.event("notify_failed", returncode=code)

    async def read_state(self) -> dict[str, Any]:
        """The scheduler's ``state.json`` (empty when missing or unreadable)."""
        try:
            value = await asyncio.to_thread(
                read_json, self.options.schedule_dir / "state.json"
            )
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    async def write_heartbeat(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        final: bool = False,
        reading: resources.HostResources | None = None,
    ) -> None:
        """``heartbeat.json`` (best-effort: a full disk only loses the beat)."""
        state = await self.read_state() if state is None else state
        try:
            if reading is None:
                reading = await asyncio.to_thread(self.sampler, self.options.runs_dir)
            host_resources: dict[str, Any] | None = reading.to_dict()
        except Exception as exc:  # a heartbeat never fails on a reading
            host_resources = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        handle = self.handle
        beat = {
            "at": now(),
            "pid": os.getpid(),
            "started": self.own_started,
            "started_at": self.started_at,
            "host": self.host,
            "heartbeat_s": self.options.heartbeat_s,
            "caffeinate_pid": getattr(self.caffeinate, "pid", None),
            "scheduler": {
                "pid": handle.pid if handle else None,
                # Its ps start time: --status tells it from a process that reused the pid.
                "started": handle.started if handle else None,
                "started_at": handle.started_at if handle else None,
                "adopted": handle.adopted if handle else None,
                "restarts": self.restarts,
                "crash_restarts": self.crash_restarts,
                "last_exit": self.last_exit,
                "hung_after_s": self.options.hung_after_s,
            },
            "state_at": state.get("at"),
            "counts": _counts(state),
            "waiting": state.get("waiting") or {},
            "pause_reason": state.get("pause_reason"),
            "budget": state.get("budget"),
            "resources": host_resources,
            "alerts": self.alerts,
            "stopping": self.stopping,
            "final": final,
            "exit_code": self.exit_code if final else None,
            "error": self.error,
        }
        try:
            await asyncio.to_thread(
                write_json_atomic, self.options.control / "heartbeat.json", beat
            )
        except OSError as exc:
            log.error("cannot write the heartbeat: %s", exc)

    # -- caffeinate ----------------------------------------------------------------------

    def _caffeinate_argv(self) -> list[str] | None:
        if not self.options.caffeinate or self.system != "darwin":
            return None
        path = self.which("caffeinate")
        return [path, "-dimsu", "-w", str(os.getpid())] if path else None

    async def start_caffeinate(self) -> None:
        argv = self._caffeinate_argv()
        if argv is None:
            return
        try:
            self.caffeinate = await self.spawn_caffeinate(argv)
        except OSError as exc:
            await self.event("caffeinate_failed", error=f"{type(exc).__name__}: {exc}")
            return
        await self.event("caffeinate", caffeinate_pid=self.caffeinate.pid, argv=argv)

    async def _check_caffeinate(self) -> None:
        if (
            self.stopping
            or self.caffeinate is None
            or self.caffeinate.returncode is None
        ):
            return
        old = self.caffeinate.returncode
        self.caffeinate = None
        await self.start_caffeinate()
        await self.event("caffeinate_respawned", returncode=old)

    async def stop_caffeinate(self) -> None:
        process, self.caffeinate = self.caffeinate, None
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), 5)

    # -- the heartbeat ---------------------------------------------------------------------

    async def tick(self) -> None:
        """One heartbeat: note a host sleep, keep caffeinate alive, watch an unknown
        budget or balance, the cells that need a person and the problems running cells
        recorded and went on past, write ``heartbeat.json``."""
        wall, mono = self.wall(), self.mono()
        if self._last_tick is not None:
            drift = (wall - self._last_tick[0]) - (mono - self._last_tick[1])
            if drift > self.options.sleep_drift_s:
                await self.event("host_slept", seconds=round(drift, 1))
        self._last_tick = (wall, mono)
        await self._check_caffeinate()
        state = await self.read_state()
        await self._watch_unknown_pause(state, mono)
        await self._note_one_off_applied(state)
        await self._watch_attention(state)
        await self._watch_gateways(state)
        await self._watch_recorded_problems(state, mono)
        try:
            reading: resources.HostResources | None = await asyncio.to_thread(
                self.sampler, self.options.runs_dir
            )
        except Exception:  # the heartbeat records the failure itself
            reading = None
        await self._watch_stalls(state, mono)
        await self._watch_disk(state, reading, mono)
        await self._watch_scheduler(state, mono)
        await self.write_heartbeat(state, reading=reading)

    async def _stall(
        self,
        key: str,
        what: Any,
        mono: float,
        text: str,
        *,
        after_s: float | None = None,
        **fields: Any,
    ) -> None:
        """One stall watch: ``what`` (None: not stalled) held since the first tick that
        saw it; after ``after_s`` (default ``--stall-alert``) it raises one alert, and
        waiting goes on."""
        if what is None:
            self._stalls.pop(key, None)
            return
        seen = self._stalls.get(key)
        if seen is None or seen[0] != what:
            self._stalls[key] = (what, mono, False)
            return
        _what, since, alerted = seen
        waited = mono - since
        limit = self.options.stall_alert_s if after_s is None else after_s
        if alerted or waited < limit:
            return
        self._stalls[key] = (what, since, True)
        await self.alert(
            key,
            text.format(minutes=waited / 60) + "; still waiting",
            waited_s=round(waited, 1),
            **fields,
        )

    async def _watch_stalls(self, state: Mapping[str, Any], mono: float) -> None:
        """A run that waits without end raises an alert (a run that needs no restart
        must say when it is stuck)."""
        handle = self.handle
        current = handle is not None and state_is_of(state, handle.pid, handle.started)
        reason = state.get("pause_reason") if current else None
        await self._stall(
            "stalled_pause",
            reason if reason not in UNKNOWN_PAUSES else None,
            mono,
            f"the scheduler has been paused on {reason} for {{minutes:.0f}} min (a "
            "top-up, a raised key limit or a daily reset lifts it)",
            budget=state.get("budget"),
        )
        held = state.get("resource_wait") if current else None
        idle = not (state.get("running") or [])
        what = (
            (held.get("cell"), held.get("gate"))
            if isinstance(held, Mapping) and idle
            else None
        )
        await self._stall(
            "stalled_resources",
            what,
            mono,
            f"nothing runs and the resource gate has held {what and what[0]} for "
            f"{{minutes:.0f}} min ({held.get('reason') if isinstance(held, Mapping) else ''}): "
            "free memory or disk on this host, or lower the gates",
        )

    async def _watch_disk(
        self,
        state: Mapping[str, Any],
        reading: resources.HostResources | None,
        mono: float,
    ) -> None:
        """The disk under the runs directory below the scheduler's
        ``--min-free-disk-gb`` while workers run (they write checkpoints and discarded
        HOME copies): an alert at once, once per episode."""
        floor = self.options.min_free_disk_gb
        free = reading.disk_free_gb if reading is not None else None
        low = (
            floor is not None
            and free is not None
            and free < floor
            and bool(state.get("running"))
        )
        seen = self._stalls.get("disk_low")
        if not low:
            self._stalls.pop("disk_low", None)
            return
        if seen is not None:
            return
        self._stalls["disk_low"] = (True, mono, True)
        await self.alert(
            "disk_low",
            f"only {free:.1f} GB free under {self.options.runs_dir} (below "
            f"--min-free-disk-gb {floor:g}) while {len(state.get('running') or [])} "
            "cell(s) run: free disk space now, or workers fail with disk_full",
            disk_free_gb=free,
        )

    async def _watch_scheduler(self, state: Mapping[str, Any], mono: float) -> None:
        """The scheduler rewrites ``state.json`` every ``--poll``; one that
        has not for ``--hung-alert`` while its process lives may hang (a deadlock, a call
        without a timeout). An alert, once per stretch; it is not killed. The measure is
        the supervisor's monotonic clock since it last saw ``state.json`` change, so a
        host sleep (which stops that clock and the scheduler alike) is no hang."""
        handle = self.handle
        what = None
        if (
            handle is not None
            and not self.stopping
            and await asyncio.to_thread(handle.alive)
        ):
            own = state_is_of(state, handle.pid, handle.started)
            what = (handle.pid, state.get("at") if own else None)
        pid = handle.pid if handle is not None else None
        await self._stall(
            "scheduler_hung",
            what,
            mono,
            f"the scheduler (pid {pid}) has not updated _schedule/state.json for "
            "{minutes:.0f} min although its process lives (it writes it every --poll): "
            f"it may hang. See {self.options.control / 'scheduler.log'}; if it stays so, "
            f"kill -KILL {pid} (the supervisor restarts it, the workers go on)",
            after_s=self.options.hung_after_s,
            scheduler=pid,
            state_at=state.get("at"),
        )

    async def _watch_unknown_pause(self, state: Mapping[str, Any], mono: float) -> None:
        """The gateway's budget or balance cannot be read (``UNKNOWN_PAUSES``): the
        scheduler sends nothing while it lasts, and resumes by itself once it is read.
        An alert at the first ``--budget-unknown-grace`` and then every grace; the
        scheduler is never signalled."""
        handle = self.handle
        reason = state.get("pause_reason")
        current = handle is not None and state_is_of(state, handle.pid, handle.started)
        if not current or reason not in UNKNOWN_PAUSES:
            self._unknown_since = None
            self._unknown_alerted_at = None
            return
        if self._unknown_since is None:
            self._unknown_since = mono
            return
        grace = self.options.budget_unknown_grace_s
        waited = mono - self._unknown_since
        if waited < grace:
            return
        if (
            self._unknown_alerted_at is not None
            and mono - self._unknown_alerted_at < max(grace, self.options.heartbeat_s)
        ):
            return  # one alert per grace (per heartbeat at the least)
        self._unknown_alerted_at = mono
        budget = state.get("budget") if isinstance(state.get("budget"), dict) else {}
        await self.alert(
            "budget_unknown",
            f"the gateway's budget or balance has been unknown ({reason}) for "
            f"{waited / 60:.0f} min: nothing is sent while it lasts (no spend); still "
            "waiting, it resumes by itself once read; check the key and "
            "_gateway/budget.json",
            reason=reason,
            seconds=round(waited, 1),
            budget=budget.get("spend") if isinstance(budget, dict) else None,
        )

    async def _watch_attention(self, state: Mapping[str, Any]) -> None:
        """A cell that needs a person (a HARNESS stop, a resume cap, a refused
        or violation cell, a gateway it cannot have ...) is alerted at once, while the
        scheduler goes on with the others, not only when it exits. Only the watched
        scheduler's own ``state.json`` counts. A cell that left the set and comes back
        is alerted again; a restarted supervisor reminds of the current set once."""
        handle = self.handle
        if handle is None or not state_is_of(state, handle.pid, handle.started):
            return
        reasons = state.get("attention_reasons")
        reasons = reasons if isinstance(reasons, Mapping) else {}
        attention = state.get("needs_attention")
        blocked = state.get("blocked")
        current: dict[str, tuple[Any, Any]] = {
            str(cell): (status, reasons.get(cell))
            for cell, status in (
                attention.items() if isinstance(attention, Mapping) else ()
            )
        }
        current.update(
            (str(cell), (status, "blocked"))
            for cell, status in (
                blocked.items() if isinstance(blocked, Mapping) else ()
            )
        )
        new = {
            cell: value
            for cell, value in sorted(current.items())
            if self._attention_seen.get(cell) != value
        }
        self._attention_seen = current
        if not new:
            return
        listed = ", ".join(
            f"{cell} ({status}: {reason})" for cell, (status, reason) in new.items()
        )
        await self.alert(
            "cell_needs_attention",
            f"cells need a person now: {listed}; the others go on",
            cells={cell: [status, reason] for cell, (status, reason) in new.items()},
        )

    async def _watch_gateways(self, state: Mapping[str, Any]) -> None:
        """An extra gateway the watched scheduler adopted at its start although it does
        not answer ``/health`` or runs with another budget than ``--budget``
        (``state.json`` ``gateways``' ``problem``, ``schedule.adopt_live_gateways``):
        ``--budget`` does not apply to it and the MIRIX worker that uses it may stall.
        Alerted at once (``gateway_needs_attention``), once per gateway and problem; a
        problem that clears and comes back is alerted again."""
        handle = self.handle
        if handle is None or not state_is_of(state, handle.pid, handle.started):
            return
        gateways = state.get("gateways")
        current = {
            str(port): str(entry["problem"])
            for port, entry in (
                gateways.items() if isinstance(gateways, Mapping) else ()
            )
            if isinstance(entry, Mapping) and entry.get("problem")
        }
        new = {
            port: problem
            for port, problem in sorted(current.items())
            if self._gateway_problems_seen.get(port) != problem
        }
        self._gateway_problems_seen = current
        if not new:
            return
        listed = "; ".join(f"port {port}: {problem}" for port, problem in new.items())
        await self.alert(
            "gateway_needs_attention",
            "the scheduler adopted a gateway a live worker still uses that does not "
            f"serve it as asked ({listed}); its cells may stall: stop that gateway once "
            "its cells are done, then restart the scheduler",
            gateways=new,
        )

    async def _watch_recorded_problems(
        self, state: Mapping[str, Any], mono: float
    ) -> None:
        """What a running cell recorded and went on past (a provider
        that could not serve an item, a refusal streak, an accepted MIRIX batch, a
        degraded retrieval, a rejected induction ...), read with
        ``cell_records.recorded_problems`` from each running cell's highest attempt.
        A count seen for the first time is alerted at once; a count that grows again
        at most once per ``--stall-alert``. Never raises: a watch that fails only
        loses its alerts for this beat."""
        reader = getattr(cell_records, "recorded_problems", None)
        if reader is None:  # before cell_records provides it: nothing to watch
            return
        try:
            found = await asyncio.to_thread(
                _recorded_problems, self.options.runs_dir, reader
            )
        except Exception as exc:
            log.warning("cannot read the recorded problems: %s", exc)
            return
        throttle = self.options.stall_alert_s
        for (cell, code), count in sorted(found.items()):
            alerted = self._problems_alerted.get((cell, code))
            if alerted is not None:
                if count <= alerted[0] or mono - alerted[1] < throttle:
                    continue
            self._problems_alerted[(cell, code)] = (count, mono)
            await self.alert(
                "recorded_problem",
                f"{cell}: {code} x{count} (recorded and continued; see its "
                "attempt.json)",
                cell=cell,
                code=code,
                count=count,
            )

    async def _heartbeat_loop(self) -> None:
        while True:
            async with self._tick_lock:
                try:
                    await self.tick()
                except Exception:  # the heartbeat never takes the supervisor down
                    log.exception("heartbeat failed")
            await asyncio.sleep(self.options.heartbeat_s)

    async def _stop_heartbeat(self, heartbeat: asyncio.Future[Any]) -> None:
        """Cancel the heartbeat between two ticks: a tick in progress (sending an alert
        or a notice) finishes first, for at most one notify timeout and a margin."""
        acquired = False
        try:
            await asyncio.wait_for(self._tick_lock.acquire(), NOTIFY_TIMEOUT_S + 5.0)
            acquired = True
        except asyncio.TimeoutError:
            log.warning("a heartbeat tick is still running: cancelled")
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            if acquired:
                self._tick_lock.release()

    # -- waiting -------------------------------------------------------------------------

    async def pause(self, seconds: float) -> bool:
        """Wait ``seconds`` unless a stop is asked for; True when it was."""
        assert self._stop is not None
        if self._pause_hook is not None:
            await self._pause_hook(seconds)
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), max(0.0, seconds))
        except asyncio.TimeoutError:
            return False
        return True

    def request_stop(self) -> None:
        """A signal: the first stops cleanly, the second escalates to SIGKILL."""
        assert self._stop is not None and self._escalate is not None
        self.signals += 1
        self._stop.set()
        if self.signals >= 2:
            self._escalate.set()

    # -- one-off options ---------------------------------------------------------------------

    @property
    def one_off_path(self) -> Path:
        return self.options.control / ONE_OFF_FILE

    def one_off_requests(self) -> list[OneOffRequest]:
        """The one-off options of this supervisor's scheduler command line."""
        return one_off_requests(self.options.scheduler_args, self.options.one_off_again)

    def one_off_withheld(self) -> list[OneOffRequest]:
        """The command line's one-off options that go to no scheduler: recorded as handed
        to one already (``pending`` or ``applied``), or all of them when the record
        cannot be read (nobody can tell what was applied)."""
        requests = self.one_off_requests()
        if self._one_off_error is not None:
            return requests
        return [r for r in requests if r.key in self._one_off]

    async def _load_one_off(self) -> None:
        try:
            self._one_off = await asyncio.to_thread(
                read_one_off_record, self.one_off_path
            )
            self._one_off_error = None
        except (OSError, ValueError) as exc:
            self._one_off = {}
            self._one_off_error = f"{type(exc).__name__}: {exc}"[:300]

    async def _write_one_off(
        self, changes: Mapping[str, dict[str, Any] | None]
    ) -> bool:
        """Apply ``changes`` (key -> entry; None removes it) to ``one_off.json`` durably
        (temporary file, fsync, rename, fsync of the directory). The in-memory record
        follows the file: a write that fails changes neither (False, and an event)."""
        async with self._one_off_lock:  # the heartbeat and an exit may both write
            record = dict(self._one_off)
            for key, entry in changes.items():
                if entry is None:
                    record.pop(key, None)
                else:
                    record[key] = entry
            try:
                await asyncio.to_thread(
                    write_json_atomic, self.one_off_path, {"requests": record}
                )
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
            else:
                self._one_off = record
                return True
        await self.event("one_off_record_failed", error=error, keys=sorted(changes))
        return False

    async def _warn_one_off_withheld(self) -> None:
        """At start: the command line carries one-off options that a scheduler of this
        runs directory got already (a supervisor restarted by systemd with the same
        ExecStart, or relaunched with the same command). They go to no scheduler, and a
        person hears of it: a restart must never release HARNESS stops nobody looked at,
        nor clear resume counters again."""
        requests = self.one_off_requests()
        if not requests:
            return
        if self._one_off_error is not None:
            await self.alert(
                "one_off_record_unreadable",
                f"cannot read {self.one_off_path} ({self._one_off_error}): not passing "
                f"{', '.join(r.describe() for r in requests)} to the scheduler (nobody can "
                "tell whether a scheduler applied them already). Check the file; delete "
                "it only if none of its options must stay applied, then relaunch",
                options=[r.describe() for r in requests],
            )
            return
        withheld = self.one_off_withheld()
        if not withheld:
            return
        parts = []
        for request in withheld:
            entry = self._one_off.get(request.key) or {}
            await self.event(
                "one_off_already_applied",
                option=request.option,
                values=list(request.values),
                again=request.again,
                key=request.key,
                state=entry.get("state"),
                recorded_at=entry.get("recorded_at"),
                applied_at=entry.get("applied_at"),
                scheduler=entry.get("scheduler"),
            )
            when = entry.get("applied_at") or entry.get("recorded_at")
            note = (
                ""
                if entry.get("state") == "applied"
                else ": its supervisor stopped before the scheduler said it applied "
                "it, which counts as applied"
            )
            parts.append(
                f"{request.describe()} ({entry.get('state')} {when}, scheduler pid "
                f"{entry.get('scheduler')}{note})"
            )
        await self.alert(
            "one_off_already_applied",
            f"not passing {'; '.join(parts)} to the scheduler: {self.one_off_path} "
            "records it as handed to a scheduler already, and a restart must not apply "
            "it twice. Remove it from the command line (a systemd unit's ExecStart "
            "too); to apply it once more on purpose, relaunch with --one-off-again "
            "<new id>",
            options=[r.describe() for r in withheld],
        )

    async def _one_off_for_spawn(self) -> tuple[list[str], list[OneOffRequest]]:
        """The next scheduler's command line and the one-off requests it carries. They
        are recorded ``pending`` BEFORE it starts: a supervisor that dies right after
        the start must not hand them out again (a restart counts ``pending`` as
        applied). A record that cannot be written withholds them (and alerts)."""
        argv = self.scheduler_argv()
        withheld = {r.key for r in self.one_off_withheld()}
        passing = [r for r in self.one_off_requests() if r.key not in withheld]
        if not passing:
            return argv, []
        at = now()
        entries: dict[str, dict[str, Any] | None] = {
            r.key: {
                "option": r.option,
                "values": list(r.values),
                "again": r.again,
                **({"note": r.note} if r.note is not None else {}),
                "state": "pending",
                "recorded_at": at,
                "applied_at": None,
                "scheduler": None,
                "supervisor": os.getpid(),
            }
            for r in passing
        }
        if await self._write_one_off(entries):
            return argv, passing
        await self.alert(
            "one_off_unrecorded",
            f"cannot record {', '.join(r.describe() for r in passing)} in "
            f"{self.one_off_path}: not passing it to the scheduler (a restart could apply "
            "it twice). Free disk space; the next scheduler start tries again",
            options=[r.describe() for r in passing],
        )
        return drop_option(argv, [r.option for r in passing]), []

    async def _note_one_off_applied(self, state: Mapping[str, Any]) -> None:
        """The scheduler ``state.json`` names says it applied its one-off options: the
        requests pending on it are ``applied``. Only the state of the scheduler watched
        now: a dead scheduler's state.json naming a pid a new one reused never marks the
        new one's requests applied."""
        pid = state.get("pid")
        handle = self.handle
        if not isinstance(pid, int) or not state.get("one_off_applied"):
            return
        if handle is None or not state_is_of(state, handle.pid, handle.started):
            return
        # The heartbeat and a scheduler's exit may both get here at once: one of them
        # marks the requests applied (and sends the notice), the other finds none.
        async with self._applied_lock:
            changes: dict[str, dict[str, Any] | None] = {
                key: {**entry, "state": "applied", "applied_at": now()}
                for key, entry in self._one_off.items()
                if entry.get("state") == "pending" and entry.get("scheduler") == pid
            }
            if not changes or not await self._write_one_off(changes):
                return
            await self.event("one_off_applied", scheduler=pid, keys=sorted(changes))
            for entry in changes.values():
                if entry is not None and entry.get("option") == "--new-attempt-cells":
                    await self._notice_new_attempts(entry, state)

    async def _notice_new_attempts(
        self, entry: Mapping[str, Any], state: Mapping[str, Any]
    ) -> None:
        """Reruns within the budget need no separate approval; the estimate is sent (a
        notice, not an alert) and they run."""
        requested = state.get("new_attempts")
        requested = requested if isinstance(requested, Mapping) else {}
        cells = [str(c) for c in entry.get("values") or ()]
        # Only this note's requests (a cell that never ran needs none; an earlier
        # note's request is not this one's estimate).
        found = {
            cell: dict(requested[cell])
            for cell in cells
            if isinstance(requested.get(cell), Mapping)
            and (
                entry.get("note") is None
                or requested[cell].get("note") == entry["note"]
            )
        }
        nothing = [c for c in cells if c not in found]
        estimates = [
            float(r["estimate_usd"])
            for r in found.values()
            if isinstance(r.get("estimate_usd"), (int, float))
            and not isinstance(r.get("estimate_usd"), bool)
        ]
        unknown = [c for c, r in found.items() if r.get("estimate_usd") is None]
        cost = f"estimated ${sum(estimates):.2f}" + (
            f" (unknown for {', '.join(unknown)})" if unknown else ""
        )
        note = entry.get("note") or next(
            (r.get("note") for r in found.values() if r.get("note")), None
        )
        text = (
            f"{', '.join(found)} start new attempts (note: {note}), {cost} from the "
            "attempts they replace; within the budget"
            if found
            else f"no new attempt to start (note: {note})"
        )
        if nothing:
            text += f"; nothing to rerun for {', '.join(nothing)} (never ran here)"
        await self.notice(
            "new_attempts_requested",
            text,
            cells=found,
            estimate_usd=round(sum(estimates), 6) if estimates else None,
            not_requested=nothing,
        )

    async def _settle_one_off(
        self, handle: SchedulerHandle | None, state: Mapping[str, Any], mine: bool
    ) -> None:
        """At a scheduler's exit: what it did with the one-off requests it got. Applied
        (its ``state.json`` says so): recorded. A scheduler this supervisor started that
        exited without applying them (refused, or a crash before): taken off the record,
        so the next start gets them again."""
        passed, self._one_off_passed = self._one_off_passed, []
        if mine and state.get("one_off_applied"):
            await self._note_one_off_applied(state)
            return
        if not passed or handle is None or handle.adopted:
            return
        unapplied = [
            key
            for key in passed
            if (self._one_off.get(key) or {}).get("state") == "pending"
        ]
        if unapplied and await self._write_one_off(dict.fromkeys(unapplied)):
            await self.event(
                "one_off_not_applied", scheduler=handle.pid, keys=sorted(unapplied)
            )

    # -- the scheduler ---------------------------------------------------------------------

    def scheduler_argv(self) -> list[str]:
        """The scheduler's command line: without the one-off options a scheduler of this
        runs directory got already (``one_off.json``)."""
        args = list(self.options.scheduler_args)
        withheld = self.one_off_withheld()
        if withheld:
            args = drop_option(args, [r.option for r in withheld])
        return [self.options.python, "-m", self.options.scheduler_module, *args]

    async def _adoptable(self) -> SchedulerHandle | None:
        """A live scheduler of this runs directory that is not ours (a restarted
        supervisor): its lock is held and ``state.json`` names a running process."""
        lock = self.options.schedule_dir / "scheduler.lock"
        if not await asyncio.to_thread(lock_is_held, lock):
            return None
        state = await self.read_state()
        pid, started = state.get("pid"), state.get("started")
        if (
            not isinstance(pid, int)
            or not started
            or state.get("exit_code") is not None
        ):
            return None
        current = await asyncio.to_thread(process_start_time, pid)
        if current is None or current != " ".join(str(started).split()):
            return None
        return SchedulerHandle(pid, current, started_mono=self.mono())

    async def _start_process(
        self, argv: Sequence[str], log_fh: Any, env: Mapping[str, str]
    ) -> asyncio.subprocess.Process:
        """Start the scheduler process (a seam for tests)."""
        return await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            env=dict(env),
            start_new_session=True,  # a supervisor restart never stops it
        )

    async def _spawn(self) -> SchedulerHandle:
        argv, passing = await self._one_off_for_spawn()
        self._last_argv = argv
        path = self.options.control / "scheduler.log"

        def open_log() -> Any:
            return path.open("ab")

        env = dict(os.environ)
        # -m evals.skill_bench... from any working directory (paths stay the caller's).
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(protocol.REPO_ROOT), env.get("PYTHONPATH")])
        )
        try:
            log_fh = await asyncio.to_thread(open_log)
            try:
                process = await self._start_process(argv, log_fh, env)
            finally:
                log_fh.close()
        except Exception:
            if passing:  # it never ran: the one-off requests were handed to nobody
                await self._write_one_off({r.key: None for r in passing})
            raise
        self._one_off_passed = [r.key for r in passing]
        if passing:
            # Which scheduler got them (a restarted supervisor that adopts it, or sees
            # its state.json say one_off_applied, records them applied).
            await self._write_one_off(
                {
                    r.key: {**self._one_off[r.key], "scheduler": process.pid}
                    for r in passing
                    if r.key in self._one_off
                }
            )
        started = await asyncio.to_thread(process_start_time, process.pid)
        return SchedulerHandle(process.pid, started, process, started_mono=self.mono())

    async def _run_scheduler(self) -> int | None:
        """Start or adopt the scheduler and wait for it; None when a stop was asked for
        first (the scheduler is then still running: :meth:`shutdown` handles it)."""
        assert self._stop is not None
        # A new scheduler: the grace of an unknown budget starts over.
        self._unknown_since = None
        self._unknown_alerted_at = None
        adopted = await self._adoptable()
        if adopted is not None:
            self.handle = adopted
            self._one_off_passed = []
            await self.event("scheduler_adopted", scheduler=adopted.pid)
            # An adoptable scheduler has applied its one-off options (its
            # state.json names it only from that write on): the requests a supervisor
            # that died handed to it are applied, and never go to a later start.
            await self._note_one_off_applied(await self.read_state())
        else:
            self.handle = await self._spawn()
            await self.event(
                "scheduler_started",
                scheduler=self.handle.pid,
                argv=self._last_argv,
                restarts=self.restarts,
            )
        await self.write_heartbeat()
        waiter = asyncio.ensure_future(
            self.handle.wait(self.options.schedule_dir / "state.json")
        )
        stopper = asyncio.ensure_future(self._stop.wait())
        done, _ = await asyncio.wait(
            {waiter, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        if waiter in done:
            stopper.cancel()
            return waiter.result()
        waiter.cancel()
        return None

    def _refusal(self) -> str | None:
        """The scheduler's last refusal (its ``{"error": ...}`` line in scheduler.log)."""
        for line in reversed(tail(self.options.control / "scheduler.log").splitlines()):
            with contextlib.suppress(ValueError):
                value = json.loads(line)
                if isinstance(value, dict) and value.get("error"):
                    return str(value["error"])
        return None

    async def after_exit(self, code: int) -> int | None:
        """What a scheduler exit means: the supervisor's exit code, or None to restart."""
        handle = self.handle
        state = await self.read_state()
        mine = handle is not None and state_is_of(state, handle.pid, handle.started)
        reason = state.get("pause_reason") if mine else None
        ran_s = self.mono() - handle.started_mono if handle is not None else 0.0
        self.last_exit = {"code": code, "at": now(), "pause_reason": reason}
        await self.event(
            "scheduler_exited",
            scheduler=handle.pid if handle else None,
            returncode=code,
            pause_reason=reason,
            ran_s=round(ran_s, 1),
        )
        # It applied its one-off options (its state.json says so): later starts go
        # without them. One that exited before applying them gets them again;
        # one_off.json keeps this across supervisors.
        await self._settle_one_off(handle, state, mine)
        # A scheduler that crashed, or was stopped while workers ran (its
        # final state.json lists them), left them and their gateway behind; a terminal
        # exit drains them.
        self._left_workers = code not in _ENDED_ON_ITS_OWN or (
            mine
            and bool(
                state.get("stopped") or state.get("running") or state.get("active")
            )
        )
        if code == schedule.EXIT_ALL_COMPLETE:
            await self.event("complete", complete=len(state.get("complete") or []))
            return SUPERVISE_COMPLETE
        if code == schedule.EXIT_ATTENTION:
            await self.alert("attention", _attention_text(state), **_attention(state))
            return SUPERVISE_ATTENTION
        self._note_streak(code, ran_s, state if mine else {})
        if code == schedule.EXIT_PAUSED:
            return await self._after_pause(state, reason)
        if code == schedule.EXIT_REFUSED:
            return await self._after_refusal()
        return await self._after_crash(code)

    def _note_streak(self, code: int, ran_s: float, state: Mapping[str, Any]) -> None:
        """Every exit that is not the end starts or extends one streak of
        runs that did nothing useful, whatever their codes. A crash that resets the pause
        and refusal timers, and a pause or refusal that resets the crash count, would let
        a scheduler alternating them restart forever without an alert. Only a useful run
        ends the streak: one of ``--crash-window``, or a cell completed since the last
        exit (``state``: this scheduler's own ``state.json``, else empty)."""
        complete = len(state.get("complete") or [])
        progressed = complete > self._complete_seen
        self._complete_seen = max(self._complete_seen, complete)
        if ran_s >= self.options.crash_window_s or progressed:
            self._streak_since = None
            self._streak_codes = []
            self._streak_longest_s = 0.0
            self._streak_alerted = False
            self._consecutive_crashes = 0
            self._refused_since = None
            self._crash_alerted = False
            self._refused_alerted = False
        elif (
            ran_s > 0
            and code != schedule.EXIT_REFUSED
            and self._refused_since is not None
        ):
            # The time a scheduler ran is no refusal. The refusal timeout counts the
            # refused starts and the waits, not the runs between them, so two transient
            # refusals around a healthy 45-minute run do not add up to a refusal alert.
            # A refused start's own time still counts (with --refused-retry 0 the
            # timeout would never come). The streak's alert
            # clock stays on wall time: a scheduler crashing every 50 minutes is still
            # worth an alert, which names its longest run.
            self._refused_since = min(self.mono(), self._refused_since + ran_s)
        if self._streak_since is None:
            self._streak_since = self.mono()
        self._streak_codes.append(code)
        self._streak_longest_s = max(self._streak_longest_s, ran_s)

    async def _alert_streak(self, last: str) -> None:
        """A streak of restarts as long as ``--stall-alert`` (a
        gateway's /balance failing for good; a scheduler alternating crashes and pauses)
        raises one alert; restarts go on."""
        if self._streak_since is None or self._streak_alerted:
            return
        waited = self.mono() - self._streak_since
        if waited < self.options.stall_alert_s:
            return
        self._streak_alerted = True
        codes = self._streak_codes
        summary = ", ".join(f"exit {c} x{n}" for c, n in sorted(Counter(codes).items()))
        await self.alert(
            "stalled_restarts",
            f"the scheduler keeps exiting ({last}) and being restarted: {len(codes)} "
            f"exits ({summary}) in {waited / 60:.0f} min now, the longest run "
            f"{self._streak_longest_s / 60:.0f} min, none a useful run (one of "
            f"{self.options.crash_window_s:g} s, or a cell completed); still restarting",
            exits=len(codes),
            codes=codes[-50:],
            longest_run_s=round(self._streak_longest_s, 1),
        )

    async def _after_pause(self, state: Mapping[str, Any], reason: Any) -> int | None:
        budget = state.get("budget") if isinstance(state.get("budget"), dict) else {}
        spend = budget.get("spend") if isinstance(budget, dict) else None
        if reason == BUDGET_SPENT:
            await self.alert(
                "budget_spent",
                "budget spent: a person must raise it; relaunch every host with "
                "--budget <new total>",
                budget=spend,
            )
            return SUPERVISE_BUDGET_SPENT
        if reason == BUDGET_INVALID:
            error = spend.get("error") if isinstance(spend, dict) else None
            await self.alert(
                "budget_invalid",
                f"budget invalid: {error or 'the key usage reads below the budget start'}",
                budget=spend,
            )
            return SUPERVISE_BUDGET_INVALID
        self.restarts += 1
        await self.event(
            "restart_after_pause",
            reason=reason,
            wait_s=self.options.pause_restart_wait_s,
        )
        await self._alert_streak(f"last on {reason}")
        await self.pause(self.options.pause_restart_wait_s)
        return None

    async def _after_refusal(self) -> int | None:
        """A refusal: retried every ``--refused-retry``. Once the streak's first refusal
        is ``--refused-timeout`` old (pause exits in between do not start it over; the
        time schedulers ran in between does not count), one alert per streak, and the
        retries go on at most every ``max(--refused-retry, --restart-backoff-cap)``.
        Another supervisor's lock and a merged view exit 13 at start."""
        mono = self.mono()
        if self._refused_since is None:
            self._refused_since = mono
        error = await asyncio.to_thread(self._refusal)
        waited = mono - self._refused_since
        retry = self.options.refused_retry_s
        if waited >= self.options.refused_timeout_s:
            retry = max(retry, self.options.restart_backoff_cap_s)
            if not self._refused_alerted:
                self._refused_alerted = True
                await self.alert(
                    "refused",
                    f"the scheduler has been refused for {waited / 60:.0f} min (with no "
                    f"useful run in between): {error}; still retrying every {retry:g} s",
                    error=error,
                    retry_s=retry,
                )
        await self.event("scheduler_refused", error=error, retry_s=retry)
        await self._alert_streak(f"last refused: {error}")
        await self.pause(retry)
        return None

    async def _after_crash(self, code: int) -> int | None:
        """A crash: restarted after ``min(cap, base * 2**(k-1))`` for the k-th crash of
        the streak (pause and refusal exits in between do not end it).
        More than ``--max-crash-restarts`` of them is a crash loop: one alert per
        streak, and the restarts go on at most every ``--restart-backoff-cap`` (the
        workers go on in their own sessions)."""
        self._consecutive_crashes += 1
        delay = min(
            self.options.restart_backoff_cap_s,
            self.options.restart_backoff_base_s * 2 ** (self._consecutive_crashes - 1),
        )
        if (
            self._consecutive_crashes > self.options.max_crash_restarts
            and not self._crash_alerted
        ):
            self._crash_alerted = True
            await self.alert(
                "crash_loop",
                f"the scheduler crashed {self._consecutive_crashes} times in a row with no "
                f"useful run (none ran {self.options.crash_window_s:g} s or completed a "
                "cell; pause or refused exits in between do not end the count; last exit "
                f"{code}): see {self.options.control / 'scheduler.log'}; still "
                f"restarting every <= {self.options.restart_backoff_cap_s:g} s, the "
                "workers go on in their own sessions",
                returncode=code,
            )
        self.restarts += 1
        self.crash_restarts += 1
        await self.event(
            "scheduler_crashed",
            returncode=code,
            consecutive=self._consecutive_crashes,
            restart_in_s=delay,
        )
        await self._alert_streak(f"last crashed with exit {code}")
        await self.pause(delay)
        return None

    # -- stopping --------------------------------------------------------------------------

    async def _wait_for(
        self, done: Callable[[], bool], timeout_s: float, poll_s: float = 0.2
    ) -> bool:
        """Poll ``done`` (blocking; run in a thread) until true, the timeout, or an
        escalation; True when it became true."""
        assert self._escalate is not None
        deadline = time.monotonic() + timeout_s
        while True:
            if await asyncio.to_thread(done):
                return True
            if self._escalate.is_set() or time.monotonic() >= deadline:
                return False
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._escalate.wait(), poll_s)

    async def shutdown(self) -> int:
        """The first signal: stop the scheduler, drain the workers (SIGTERM), stop the
        gateway nobody uses any more. A second signal: SIGKILL them all."""
        assert self._escalate is not None
        self.stopping = True
        handle = self.handle
        await self.event(
            "stopping", signals=self.signals, drain=self.options.drain_on_stop
        )
        if handle is not None and await asyncio.to_thread(handle.alive):
            await asyncio.to_thread(handle.signal, signal.SIGTERM)
            stopped = await self._wait_for(
                lambda: not handle.alive(), self.options.scheduler_stop_timeout_s
            )
            if not stopped and await asyncio.to_thread(handle.alive):
                await asyncio.to_thread(handle.signal, signal.SIGKILL, group=True)
                await self.event("scheduler_killed", scheduler=handle.pid)
        if handle is not None and handle.process is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(handle.process.wait(), 10)
        if self._escalate.is_set():
            await self._kill_everything()
            return SUPERVISE_STOPPED
        if not self.options.drain_on_stop:
            await self.event("stopped", drained=False)
            return SUPERVISE_STOPPED
        await self.drain("stop")
        return SUPERVISE_STOPPED

    async def drain_before_exit(self, code: int) -> int:
        """Before a terminal exit, the workers the last scheduler left
        behind are drained and their gateway stopped, as on a stop: otherwise they go on
        spending with no scheduler or supervisor watching them (only the gateway's own
        budget stops them), and the gateway stays up with the key loaded.
        A crash loop and an unknown budget never end the supervisor, so only a
        scheduler that exited on a spent or invalid budget (10, 11) with workers in its
        state reaches it.
        ``--no-drain-on-stop`` applies to a signal only. Never for exit 13: the workers
        there belong to the scheduler that holds the lock."""
        if code == SUPERVISE_LOCKED or not self._left_workers:
            return code
        self.stopping = True
        await self.event("stopping", exit_code=code, drain=True)
        await self.drain(f"exit {code}")
        return code

    async def drain(self, why: str) -> None:
        """SIGTERM every live ``run.py`` of the runs directory, wait for them (up to
        ``--drain-timeout``), then stop the gateway nobody uses any more; a drain that
        times out leaves a reaper for it. A second signal meanwhile kills
        everything instead. The scheduler is gone by now (stopped, or it exited)."""
        assert self._escalate is not None
        runs_dir = self.options.runs_dir
        settle_s = self.options.drain_settle_s
        signalled: dict[str, int] = {}
        late: dict[str, int] = {}
        # The workers the stopped scheduler launched (its last state.json's
        # "active"), one of them maybe just started and not holding its cell lock yet.
        state = await self.read_state()
        handle = self.handle
        launched: dict[str, int] = {}
        active = state.get("active")
        if (
            handle is not None
            and state_is_of(state, handle.pid, handle.started)
            and isinstance(active, Mapping)
        ):
            launched = {
                str(cell): pid
                for cell, pid in active.items()
                if isinstance(pid, int) and not isinstance(pid, bool)
            }
        quiet_since: list[float] = []

        def drain_new() -> bool:
            """SIGTERM every live worker not signalled yet; True once none has shown for
            ``--drain-settle`` and no worker the scheduler launched is still starting. A
            worker that took its cell after the first scan (a scheduler launching as it
            was stopped) is signalled as soon as it shows; one that has not taken it
            yet keeps the drain, and so the gateway, going."""
            found = live_cell_workers(runs_dir)
            for cell, pid in found:
                if signalled.get(cell) == pid:
                    continue
                # run.py alone: it stops its own children (games, MIRIX server) in order.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGTERM)
                signalled[cell] = pid
                late[cell] = pid
            starting = False
            for cell, pid in list(launched.items()):
                if signalled.get(cell) == pid:
                    continue
                if _starting_worker(pid, cell):
                    starting = True
                else:
                    del launched[cell]  # gone (or a stranger's pid now): never again
            if found or starting:
                quiet_since.clear()
                return False
            if not quiet_since:
                quiet_since.append(time.monotonic())
            return time.monotonic() - quiet_since[0] >= settle_s

        first = await asyncio.to_thread(live_cell_workers, runs_dir)
        for cell, pid in first:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
            signalled[cell] = pid
        await self.event(
            "drain",
            why=why,
            workers={cell: pid for cell, pid in first},
            launched=dict(launched),
        )
        drained = await self._wait_for(drain_new, self.options.drain_timeout_s)
        if late:
            await self.event("drain_late_workers", workers=dict(late))
        if self._escalate.is_set():
            await self._kill_everything()
            return
        if not drained:
            left = await asyncio.to_thread(live_cell_workers, runs_dir)
            reaper = await self._spawn_reaper()
            await self.event(
                "drain_timeout",
                why=why,
                workers={c: p for c, p in left},
                gateway_reaper=reaper,
            )
            return
        stopped = await asyncio.to_thread(
            stop_orphaned, runs_dir / "_gateway" / "owned_processes.jsonl"
        )
        await self.event("stopped", why=why, drained=True, gateway_stopped=stopped)

    async def _spawn_reaper(self) -> int | None:
        """After a drain timeout: a detached ``--reap-gateway`` process waits for the
        last worker and then stops the gateway nobody uses any more, so it
        does not stay up with the key loaded. Its pid, or None when it cannot start."""
        argv = [
            self.options.python,
            "-m",
            "evals.skill_bench.tools.supervise",
            "--reap-gateway",
            "--reap-timeout",
            f"{REAP_TIMEOUT_S:g}",
            "--runs-dir",
            str(self.options.runs_dir),
        ]
        env = dict(schedule.worker_env())
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(protocol.REPO_ROOT), env.get("PYTHONPATH")])
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
                start_new_session=True,  # it outlives this supervisor
            )
        except OSError as exc:
            await self.event(
                "reaper_failed", error=f"{type(exc).__name__}: {exc}"[:300]
            )
            return None
        return process.pid

    async def _kill_everything(self) -> None:
        handle = self.handle
        if handle is not None:
            await asyncio.to_thread(handle.signal, signal.SIGKILL, group=True)
        workers = await asyncio.to_thread(live_cell_workers, self.options.runs_dir)
        for _cell, pid in workers:
            await asyncio.to_thread(_kill_worker, pid, signal.SIGKILL)
        # The kernel frees a dead worker's locks at once; then the orphans (MIRIX servers,
        # an unused gateway) go the way a scheduler's start sweep would take them.
        await self._wait_for_quiet(5.0)
        swept = await asyncio.to_thread(sweep_owned, self.options.runs_dir)
        await self.event(
            "killed",
            scheduler=handle.pid if handle else None,
            workers={c: p for c, p in workers},
            swept=swept,
        )

    async def _wait_for_quiet(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not await asyncio.to_thread(live_cell_workers, self.options.runs_dir):
                return
            await asyncio.sleep(0.1)

    # -- main ------------------------------------------------------------------------------

    async def run(self, *, install_signals: bool = True) -> int:
        control = self.options.control
        await asyncio.to_thread(control.mkdir, parents=True, exist_ok=True)
        try:
            lock = await asyncio.to_thread(
                OwnerLock.acquire, control / "supervisor.lock", wait_s=0.0
            )
        except OwnedProcessError:
            print(
                json.dumps(
                    {
                        "error": f"another supervisor is running on {self.options.runs_dir}"
                    }
                ),
                file=sys.stderr,
            )
            return SUPERVISE_LOCKED
        self._stop = asyncio.Event()
        self._escalate = asyncio.Event()
        loop = asyncio.get_running_loop()
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.request_stop)
        try:
            self.exit_code = await self._run_locked()
            return self.exit_code
        except Exception as exc:  # never go down without a word
            return await self.report_failure(exc)
        finally:
            if install_signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
            lock.release()

    async def report_failure(self, exc: Exception) -> int:
        """An unexpected error of the supervisor itself. A person hears of
        it (``--notify-cmd``, ``ALERT``, ``events.jsonl``, a final heartbeat naming the
        error), and it exits :data:`SUPERVISE_INTERNAL_ERROR`, which the systemd unit
        restarts. The scheduler and its workers go on: a new supervisor adopts them. An
        error while it carries out a stop someone asked for (a signal) exits
        :data:`SUPERVISE_STOPPED` instead, so the unit does not restart what was to stop."""
        log.error("the supervisor failed", exc_info=exc)
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        self.error = detail[:500]
        code = SUPERVISE_STOPPED if self.signals else SUPERVISE_INTERNAL_ERROR
        self.exit_code = code
        handle = self.handle
        scheduler = handle.pid if handle is not None else None
        if code == SUPERVISE_STOPPED:
            text = (
                f"the supervisor failed while stopping ({self.error}): the scheduler (pid "
                f"{scheduler}), workers or the gateway may still run; see "
                f"{self.options.control / 'supervise.log'} and stop what is left by hand"
            )
        else:
            text = (
                f"the supervisor failed ({self.error}); the scheduler (pid {scheduler}) "
                "and its workers go on unsupervised. See "
                f"{self.options.control / 'supervise.log'}; relaunch the supervisor (the "
                "systemd unit does by itself): it adopts the scheduler"
            )
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for step in (
            lambda: self.alert("supervisor_failed", text, scheduler=scheduler),
            lambda: self.write_heartbeat(final=True),
            lambda: self.event(
                "exit", code=code, error=self.error, traceback=trace[-4000:]
            ),
        ):
            try:
                await step()
            except Exception:  # each record is best-effort; the exit code is not
                log.exception("cannot record the supervisor's failure")
        return code

    async def _run_locked(self) -> int:
        self.host = await asyncio.to_thread(host_info)
        self.own_started = await asyncio.to_thread(process_start_time, os.getpid())
        alert_path = self.options.control / "ALERT"
        if await asyncio.to_thread(alert_path.exists):
            old = await asyncio.to_thread(alert_path.read_text, encoding="utf-8")
            await asyncio.to_thread(alert_path.unlink)
            await self.event("alert_cleared", text=old.strip())
        await self._load_one_off()
        # Progress counts from the cells complete when it starts.
        self._complete_seen = len((await self.read_state()).get("complete") or [])
        await self.event(
            "start",
            host=self.host,
            scheduler_argv=self.scheduler_argv(),
            options={
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.options).items()
                if k != "notify_cmd"
            },
        )
        if await asyncio.to_thread((self.options.runs_dir / MERGED_MARKER).exists):
            await self.alert(
                "merged",
                f"{self.options.runs_dir} is a merged read-only view ({MERGED_MARKER}): "
                "supervise a host's own runs directory",
            )
            code = SUPERVISE_LOCKED
            await self.event("exit", code=code)
            return code
        if self.options.first_attempt_cells:
            # Recorded, not raised (no ALERT, no notification: the request's
            # own notice names them again once the scheduler applies it).
            cells = list(self.options.first_attempt_cells)
            await self.event(
                "notice",
                kind="new_attempt_cells_never_ran",
                text=f"--new-attempt-cells: {', '.join(cells)} never ran under "
                f"{self.options.runs_dir}: they start as first attempts",
                cells=cells,
            )
        await self._warn_one_off_withheld()
        await self.start_caffeinate()
        heartbeat = asyncio.ensure_future(self._heartbeat_loop())
        try:
            code = await self._supervise()
        finally:
            await self._stop_heartbeat(heartbeat)
            await self.stop_caffeinate()
        self.exit_code = code
        await self.write_heartbeat(final=True)
        await self.event("exit", code=code)
        return code

    async def _supervise(self) -> int:
        assert self._stop is not None
        while True:
            if self._stop.is_set():
                return await self.shutdown()
            code = await self._run_scheduler()
            if code is None or self._stop.is_set():
                return await self.shutdown()
            outcome = await self.after_exit(code)
            if outcome is not None:
                return await self.drain_before_exit(outcome)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _recorded_problems(
    runs_dir: Path, reader: Callable[[Any], Mapping[str, Any]]
) -> dict[tuple[str, str], int]:
    """``{(cell, code): count}`` of what each running cell recorded and went on past
    (``reader``: ``cell_records.recorded_problems``) in its highest attempt: the cells
    of ``runs_dir`` (``cell.json``, ``status.json`` status ``running``). Blocking."""
    found: dict[tuple[str, str], int] = {}
    if not runs_dir.is_dir():
        return found
    for child in sorted(runs_dir.iterdir()):
        if child.name in NON_CELL_DIRS or not (child / "cell.json").is_file():
            continue
        paths = CellPaths(child)
        try:
            status = read_json(paths.status)
            numbers = paths.attempt_numbers()
        except (OSError, ValueError):
            continue
        if not isinstance(status, dict) or status.get("status") != STATUS_RUNNING:
            continue
        if not numbers:
            continue
        try:
            counts = reader(paths.attempt(numbers[-1]))
        except Exception as exc:  # one unreadable cell never hides the others
            log.warning("cannot read the recorded problems of %s: %s", child.name, exc)
            continue
        for code, count in (counts if isinstance(counts, Mapping) else {}).items():
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                found[(child.name, str(code))] = count
    return found


def _counts(state: Mapping[str, Any]) -> dict[str, int]:
    return {
        "complete": len(state.get("complete") or []),
        "running": len(state.get("running") or []),
        "queued": len(state.get("queued") or []),
        "waiting": len(state.get("waiting") or {}),
        "needs_attention": len(state.get("needs_attention") or {}),
        "blocked": len(state.get("blocked") or {}),
    }


def _attention(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "needs_attention": state.get("needs_attention") or {},
        "attention_reasons": state.get("attention_reasons") or {},
        "blocked": state.get("blocked") or {},
    }


def _attention_text(state: Mapping[str, Any]) -> str:
    reasons = state.get("attention_reasons") or {}
    parts = [
        f"{cell} ({status}: {reasons.get(cell, '?')})"
        for cell, status in sorted((state.get("needs_attention") or {}).items())
    ]
    parts += [
        f"{cell} ({status})"
        for cell, status in sorted((state.get("blocked") or {}).items())
    ]
    return "cells need a person: " + (", ".join(parts) or "see _schedule/state.json")


def drop_option(argv: Sequence[str], names: Sequence[str]) -> list[str]:
    """``argv`` without the options ``names`` and their values (the tokens up to the next
    option)."""
    out: list[str] = []
    skipping = False
    for token in argv:
        if token.startswith("-"):
            skipping = token.split("=", 1)[0] in names
            if skipping:
                continue
        elif skipping:
            continue
        out.append(token)
    return out


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _age(at: Any, wall: float) -> float | None:
    value = schedule.epoch(at)
    return None if value is None else round(wall - value, 1)


def status_report(runs_dir: Path, *, wall: float | None = None) -> tuple[str, int]:
    """The ``--status`` text and exit code (0 with a fresh heartbeat of a live supervisor,
    1 otherwise). Blocking."""
    runs_dir = Path(runs_dir).absolute()
    wall = time.time() if wall is None else wall
    control = runs_dir / SUPERVISE_DIR
    try:
        beat = read_json(control / "heartbeat.json")
    except (OSError, ValueError):
        beat = None
    try:
        state = read_json(runs_dir / "_schedule" / "state.json") or {}
    except (OSError, ValueError):
        state = {}
    lines: list[str] = []
    code = STATUS_STALE
    if not isinstance(beat, dict):
        lines.append(f"supervisor: no heartbeat in {control}")
    else:
        age = _age(beat.get("at"), wall)
        pid = beat.get("pid")
        alive = (
            isinstance(pid, int)
            and bool(beat.get("started"))
            and process_start_time(pid) == beat.get("started")
        )
        interval = float(beat.get("heartbeat_s") or 30)
        fresh = age is not None and age <= STALE_HEARTBEATS * interval and alive
        code = 0 if fresh and not beat.get("final") else STATUS_STALE
        verdict = "fresh" if fresh else "STALE"
        if beat.get("final"):
            verdict = f"exited {beat.get('exit_code')}"
        host = beat.get("host") or {}
        lines.append(
            f"supervisor: pid {pid} ({'alive' if alive else 'gone'}), heartbeat "
            f"{age} s ago ({verdict}), host {host.get('hostname')} "
            f"({host.get('system')} {host.get('machine')}), caffeinate "
            f"{beat.get('caffeinate_pid')}"
        )
        sched = beat.get("scheduler") or {}
        spid = sched.get("pid")
        current = process_start_time(spid) if isinstance(spid, int) else None
        # The recorded start time too, never a process that reused the pid.
        recorded = " ".join(str(sched.get("started") or "").split())
        salive = current is not None and (not recorded or current == recorded)
        last = sched.get("last_exit") or {}
        # A live scheduler whose state.json (rewritten every --poll) is
        # older than the supervisor's --hung-alert may hang: exit 1.
        hung_after = float(sched.get("hung_after_s") or HUNG_MIN_S)
        own = state_is_of(state, spid, recorded or None)
        state_age = (
            _age(state.get("at"), wall) if own else _age(sched.get("started_at"), wall)
        )
        hung = (
            salive
            and not beat.get("final")
            and state_age is not None
            and state_age > hung_after
        )
        if hung:
            code = STATUS_STALE
        if own:
            written = f"state.json {state_age} s old"
        elif spid is None:
            written = "none started yet"
        else:
            written = f"no state.json of its own yet ({state_age} s since it started)"
        if hung:
            written += f" (HUNG? over {hung_after:g} s)"
        lines.append(
            f"scheduler: pid {spid} ({'alive' if salive else 'gone'}), {written}, "
            f"restarts {sched.get('restarts')} (crashes {sched.get('crash_restarts')}), "
            f"last exit {last.get('code')} {last.get('pause_reason') or ''}".rstrip()
        )
        if beat.get("error"):
            lines.append(f"supervisor error: {beat.get('error')}")
    counts = _counts(state)
    lines.append("cells: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    waiting = state.get("waiting") or {}
    if waiting:
        soon = sorted(waiting.items(), key=lambda item: str(item[1]))[:10]
        lines.append(
            "next wake-ups: "
            + ", ".join(
                f"{cell} in {max(0, -(_age(at, wall) or 0)):.0f} s" for cell, at in soon
            )
        )
    if state.get("needs_attention"):
        lines.append("attention: " + _attention_text(state))
    budget = state.get("budget") or {}
    spend = budget.get("spend") if isinstance(budget, dict) else None
    spent = (
        f", spent ${spend.get('spent_usd')} of ${spend.get('budget_usd')} "
        f"(left ${spend.get('left_usd')})"
        if isinstance(spend, dict) and spend.get("budget_usd") is not None
        else ""
    )
    remaining = budget.get("remaining_usd") if isinstance(budget, dict) else None
    lines.append(
        f"pause: {state.get('pause_reason') or 'none'}; balance ${remaining}{spent}"
    )
    reading = (beat or {}).get("resources") if isinstance(beat, dict) else None
    if isinstance(reading, dict) and "mem_available_gb" in reading:
        lines.append(
            f"resources: {reading.get('mem_available_gb')} GB available of "
            f"{reading.get('mem_total_gb')} GB, swap {reading.get('swap_used_gb')}/"
            f"{reading.get('swap_total_gb')} GB, disk {reading.get('disk_free_gb')} GB "
            f"free, load {(reading.get('load') or [None])[0]}"
        )
    alert = control / "ALERT"
    if alert.is_file():
        lines.append("ALERT: " + alert.read_text(encoding="utf-8").strip())
    return "\n".join(lines), code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _HelpAction(argparse.Action):
    """``--help``: the supervisor's own options, then every schedule.py option."""

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS)

    def __call__(
        self, parser: argparse.ArgumentParser, *_args: Any, **_kw: Any
    ) -> None:
        parser.print_help()
        print(
            "\nschedule.py options (passed through; --auto-resume is always added):\n"
        )
        schedule.build_parser().print_help()
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.tools.supervise",
        description="Keep the skill_bench scheduler running with --auto-resume.",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("-h", "--help", action=_HelpAction)
    parser.add_argument(
        "--status", action="store_true", help="print the heartbeat summary and exit"
    )
    parser.add_argument(
        "--notify-cmd",
        help="run on every alert (shlex-split, no shell), the text in $SKILL_BENCH_ALERT",
    )
    parser.add_argument(
        "--no-caffeinate", action="store_true", help="do not keep macOS awake"
    )
    parser.add_argument(
        "--no-drain-on-stop",
        action="store_true",
        help="on SIGTERM stop only the scheduler; its workers go on",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=180.0,
        help="seconds the workers get to stop on SIGTERM (default %(default)g)",
    )
    parser.add_argument(
        "--max-crash-restarts",
        type=int,
        default=20,
        help="crashes in a row, each within --crash-window of its start, before one "
        "crash_loop alert; the supervisor keeps restarting, at most every "
        "--restart-backoff-cap (default %(default)s)",
    )
    parser.add_argument(
        "--restart-backoff-cap",
        type=float,
        default=300.0,
        help="the longest wait before restarting a crashed scheduler (default %(default)g s)",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="seconds between heartbeat.json writes (default %(default)g)",
    )
    parser.add_argument(
        "--budget-unknown-grace",
        type=float,
        default=1800.0,
        help="seconds between alerts while the budget or balance stays unknown "
        "(default %(default)g)",
    )
    parser.add_argument(
        "--stall-alert",
        type=float,
        default=1800.0,
        help="alert (and go on waiting) when the run has stalled this many seconds: a "
        "pause the scheduler waits out, a cell the resource gate holds with nothing "
        "running, restarts on a pause (default %(default)g)",
    )
    parser.add_argument(
        "--one-off-again",
        metavar="ID",
        help="apply --reset-resume / --rerun-harness-failures / --new-attempt-cells once "
        "more although the same ones were applied in this runs directory before "
        "(_supervise/one_off.json; a new --new-attempt-note is a new request anyway); ID, "
        "any word not used before (e.g. the date), makes this a new request, so a "
        "restart with the same command line does not apply them yet again",
    )
    parser.add_argument(
        "--hung-alert",
        type=float,
        default=None,
        help="alert when the scheduler's state.json has not changed for this many "
        "seconds while it lives (default: 10 times its --poll, at least 600)",
    )
    parser.add_argument("--reap-gateway", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--reap-timeout", type=float, default=REAP_TIMEOUT_S, help=argparse.SUPPRESS
    )
    timing = parser.add_argument_group("timing (tuning and tests)")
    timing.add_argument(
        "--drain-settle",
        type=float,
        default=DRAIN_SETTLE_S,
        help="a drain ends once no worker has shown for this long, so one that has not "
        "taken its cell lock yet is signalled too (default %(default)g s)",
    )
    timing.add_argument(
        "--restart-backoff-base",
        type=float,
        default=10.0,
        help="the first wait before restarting a crashed scheduler; doubled per crash "
        "in a row (default %(default)g s)",
    )
    timing.add_argument(
        "--crash-window",
        type=float,
        default=3600.0,
        help="a scheduler that ran this long before crashing starts a new crash count "
        "(default %(default)g s)",
    )
    timing.add_argument(
        "--pause-restart-wait",
        type=float,
        default=300.0,
        help="wait before restarting a scheduler that exited paused on something it "
        "does not wait out, e.g. an unreachable gateway (default %(default)g s)",
    )
    timing.add_argument(
        "--refused-retry",
        type=float,
        default=60.0,
        help="wait between starts of a refused scheduler (default %(default)g s)",
    )
    timing.add_argument(
        "--refused-timeout",
        type=float,
        default=1800.0,
        help="alert once a scheduler has been refused this long; the retries go on, at "
        "most every --restart-backoff-cap (default %(default)g s)",
    )
    timing.add_argument(
        "--scheduler-stop-timeout",
        type=float,
        default=30.0,
        help="on SIGTERM, how long the scheduler may take to stop before it is killed "
        "(default %(default)g s)",
    )
    timing.add_argument(
        "--sleep-drift",
        type=float,
        default=60.0,
        help="a wall-clock lead over the monotonic clock beyond this is logged as "
        "host_slept (default %(default)g s)",
    )
    timing.add_argument(
        "--scheduler-module", default=SCHEDULER_MODULE, help=argparse.SUPPRESS
    )
    return parser


def _removed_start_probe(scheduler_args: Sequence[str]) -> tuple[list[str], Path]:
    """The removed methods (``study.REMOVED_METHODS``) a scheduler command line names with
    ``--method`` (every occurrence: the scheduler's parser checks each), and its
    ``--runs-dir``, read before the scheduler's parser, which would stop at the first."""
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--method", nargs="*", action="extend", default=[])
    probe.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    known, _ = probe.parse_known_args(list(scheduler_args))
    return [m for m in known.method if study.is_removed(m)], known.runs_dir


def refused_start_alert(runs_dir: Path, notify_cmd: str | None, text: str) -> bool:
    """Raise the alert of a supervisor refused at start because its command line names a
    method the study removed, or one of its cells; returns whether ``runs_dir`` got it.

    Such a command line is typically one written before the removal: a systemd
    unit's ``ExecStart`` or a ``nohup`` launch whose output goes nowhere. The refusal
    exits 2, which the unit does not restart (``RestartPreventExitStatus``), so the
    scheduler it would have adopted and its workers would go on unsupervised with nobody
    told. It is raised the way :meth:`Supervisor.alert` raises one: ``--notify-cmd``
    first (the text in ``SKILL_BENCH_ALERT``, the workers' environment), then
    ``events.jsonl`` and ``ALERT``, which ``--status`` prints and the next supervisor's
    start clears. Only in an existing runs directory that is not a merged view (a refused
    command line creates nothing); each step best-effort. Blocking: it runs before any
    event loop.

    While a live supervisor holds the runs directory's lock, nothing runs
    unsupervised and its ``ALERT`` is the one ``--status`` must keep showing: the refusal
    is only an ``events.jsonl`` row (no notification, ``ALERT`` untouched)."""
    runs_dir = Path(runs_dir).absolute()
    if not runs_dir.is_dir() or (runs_dir / MERGED_MARKER).exists():
        return False
    control = runs_dir / SUPERVISE_DIR
    supervised = lock_is_held(control / "supervisor.lock")
    text = (
        f"the supervisor of {runs_dir} refused to start: {text} Fix its command line "
        "(a systemd unit's ExecStart too) and start it again"
        + (
            " (a live supervisor runs this directory meanwhile)."
            if supervised
            else ": meanwhile a scheduler it would adopt runs unsupervised."
        )
    )
    at = now()
    if notify_cmd and not supervised:
        with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
            subprocess.run(
                shlex.split(notify_cmd),
                env={**schedule.worker_env(), ALERT_ENV: text},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=NOTIFY_TIMEOUT_S,
                check=False,
            )
    record = {
        "at": at,
        "event": "alert",
        "pid": os.getpid(),
        "kind": "refused_start",
        "text": text,
        "live_supervisor": supervised,
    }
    with contextlib.suppress(OSError):
        append_jsonl(control / "events.jsonl", record)
    if not supervised:
        with contextlib.suppress(OSError):
            _write_text_atomic(control / "ALERT", f"{at} {text}\n")
    return True


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[SuperviseOptions, argparse.Namespace]:
    """The supervisor's options and the scheduler's (validated by schedule.py's parser,
    whose errors name this command). A command line that names a removed method, or a
    cell of one, is refused (exit 2) after :func:`refused_start_alert`."""
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if "--dry-run" in rest:
        parser.error("--dry-run is schedule.py's; run it directly")
    scheduler_args = list(rest)
    if "--auto-resume" not in scheduler_args:
        scheduler_args.append("--auto-resume")
    # --status and --reap-gateway only read or tidy up; a stale command line
    # they were appended to is refused without an alert.
    starting = not (args.status or args.reap_gateway)
    removed, probed_runs_dir = _removed_start_probe(scheduler_args)
    if removed:
        text = f"argument --method: {study.removed_message(removed[0])}"
        if starting:
            refused_start_alert(probed_runs_dir, args.notify_cmd, text)
        parser.error(text)
    scheduler_parser = schedule.build_parser()
    # The scheduler's options are given to this command, so its refusals name it (not
    # `python -m evals.skill_bench.schedule`).
    scheduler_parser.prog = parser.prog
    sched = scheduler_parser.parse_args(scheduler_args)
    try:
        # The scheduler's own checks, before anything starts: a scheduler that refused
        # its arguments would exit 2, which reads as "cells need a person".
        cells = schedule.cells_from_args(sched)
        sched_options = schedule.options_from_args(sched)
        for flag, named in (
            ("--reset-resume", sched.reset_resume),
            ("--rerun-harness-failures", sched.rerun_harness_failures),
            ("--new-attempt-cells", sched.new_attempt_cells),
        ):
            problem = schedule.named_cells_problem(flag, named, cells)
            if problem:
                if starting and any(study.removed_cell_id(c) for c in named or ()):
                    refused_start_alert(sched.runs_dir, args.notify_cmd, problem)
                raise ValueError(problem)
        # A cell named for a rerun must have run under --runs-dir: the scheduler skips a
        # named cell without a HARNESS stop silently, so a wrong (or missing) --runs-dir
        # would start the named cells as first attempts in another directory, with none
        # of their options.
        never_run = sorted(
            c.cell_id
            for c in cells
            if c.cell_id in set(sched.rerun_harness_failures or ())
            and not CellPaths(sched_options.cell_dir(c)).status.is_file()
        )
        if never_run:
            raise ValueError(
                f"--rerun-harness-failures names cells that never ran under "
                f"{sched.runs_dir} (no status.json; the wrong --runs-dir?): {never_run}"
            )
        # The same for a rerun a person asks for: a cell that never ran here
        # starts as a first attempt anyway. Naming only such cells is most likely a
        # wrong --runs-dir (the rerun would then happen nowhere): refused. With a named
        # cell that ran, the others are new cells of this directory (e.g. cells this
        # host never ran): a warning and a start notice, and
        # the request goes on unchanged, so its identity (and its one-off record) is
        # the same at every restart, also once those cells have run a first attempt.
        named_new = [
            c for c in cells if c.cell_id in set(sched.new_attempt_cells or ())
        ]
        never_run = sorted(
            c.cell_id
            for c in named_new
            if not CellPaths(sched_options.cell_dir(c)).status.is_file()
        )
        if never_run and len(never_run) == len(named_new):
            raise ValueError(
                f"--new-attempt-cells names only cells that never ran under "
                f"{sched.runs_dir} (no status.json; the wrong --runs-dir?): "
                f"{never_run}; they would start as first attempts"
            )
        if never_run and starting:
            print(
                f"warning: --new-attempt-cells: {', '.join(never_run)} never ran under "
                f"{sched.runs_dir}: they start as first attempts",
                file=sys.stderr,
            )
        if starting:
            # A relaunch (tools/phase16.sh up or recover, the systemd
            # unit) must not widen the runs directory's selection by accident: the
            # scheduler would refuse it at every restart (exit 4); refused here at once
            # (exit 2, which the unit does not restart), with the alert.
            problem = schedule.selection_problem(
                sched_options.control, cells, sched.widen_selection
            )
            if problem:
                refused_start_alert(sched.runs_dir, args.notify_cmd, problem)
                raise ValueError(problem)
        options = SuperviseOptions(
            runs_dir=sched.runs_dir,
            scheduler_args=tuple(scheduler_args),
            scheduler_module=args.scheduler_module,
            notify_cmd=args.notify_cmd,
            caffeinate=not args.no_caffeinate,
            drain_on_stop=not args.no_drain_on_stop,
            drain_timeout_s=args.drain_timeout,
            max_crash_restarts=args.max_crash_restarts,
            restart_backoff_base_s=args.restart_backoff_base,
            restart_backoff_cap_s=args.restart_backoff_cap,
            crash_window_s=args.crash_window,
            heartbeat_s=args.heartbeat,
            budget_unknown_grace_s=args.budget_unknown_grace,
            pause_restart_wait_s=args.pause_restart_wait,
            refused_retry_s=args.refused_retry,
            refused_timeout_s=args.refused_timeout,
            scheduler_stop_timeout_s=args.scheduler_stop_timeout,
            sleep_drift_s=args.sleep_drift,
            stall_alert_s=args.stall_alert,
            min_free_disk_gb=sched.min_free_disk_gb,
            drain_settle_s=args.drain_settle,
            scheduler_poll_s=sched.poll_s,
            hung_alert_s=args.hung_alert,
            one_off_again=args.one_off_again,
            first_attempt_cells=tuple(never_run),
        )
        if args.notify_cmd:
            shlex.split(args.notify_cmd)
    except ValueError as exc:
        parser.error(str(exc))
    return options, args


async def reap_gateway(
    runs_dir: Path, *, timeout_s: float = REAP_TIMEOUT_S, poll_s: float = 2.0
) -> list[int]:
    """``--reap-gateway`` (the supervisor starts it detached after a drain timeout):
    wait until no worker of ``runs_dir`` is alive (at most ``timeout_s``),
    then stop the gateway nobody owns or uses any more (``processes.stop_orphaned``:
    never one a live scheduler owns or a worker uses). Returns the pids it stopped."""
    runs_dir = Path(runs_dir).absolute()
    deadline = time.monotonic() + timeout_s
    while await asyncio.to_thread(live_cell_workers, runs_dir):
        if time.monotonic() >= deadline:
            return []
        await asyncio.sleep(poll_s)
    stopped = await asyncio.to_thread(
        stop_orphaned, runs_dir / "_gateway" / "owned_processes.jsonl"
    )
    record = {
        "at": now(),
        "event": "gateway_reaped",
        "pid": os.getpid(),
        "stopped": stopped,
    }
    with contextlib.suppress(OSError):
        await asyncio.to_thread(
            append_jsonl, runs_dir / SUPERVISE_DIR / "events.jsonl", record
        )
    return stopped


def main(argv: Sequence[str] | None = None) -> int:
    options, args = parse_args(argv)
    if args.status:
        text, code = status_report(options.runs_dir)
        print(text)
        return code
    if args.reap_gateway:
        asyncio.run(reap_gateway(options.runs_dir, timeout_s=args.reap_timeout))
        return 0
    options.control.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(options.control / "supervise.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    supervisor = Supervisor(options)
    try:
        return asyncio.run(supervisor.run())
    except Exception as exc:  # before its lock, say: alert all the same
        return asyncio.run(supervisor.report_failure(exc))


if __name__ == "__main__":
    raise SystemExit(main())
