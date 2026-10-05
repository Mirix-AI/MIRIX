"""Host-wide coordination of the schedulers that share one host (for example several
replicate runs directories, each under its own scheduler and supervisor, run at once on
one machine).

A scheduler is alone in its runs directory (``_schedule/scheduler.lock``), and everything
a cell owns is unique to its cell directory (its database name and gateway run ids carry
the directory's random instance id). Three things belong to the host, not to a runs
directory:

* **Memory.** ``infra/resources.admit`` subtracts the headroom the running workers have not
  taken yet (a worker launched a moment ago does not show in the available memory). A
  scheduler knows only its own workers: three schedulers that each admit a cell on the
  same reading overcommit by two cells' peaks, and none reserves anything for the others'
  fresh workers.
* **The memory emergency.** Each scheduler stops its own newest worker after three low
  readings: on one low host, three workers are stopped where one was meant (and the next
  stop waits only for the scheduler's own victim).
* **The WebShop server.** One container with one global lock serves every WebShop episode
  of the host, so a per-scheduler cap (``--benchmark-concurrency``) multiplies
  by the number of schedulers.

So every scheduler keeps a small record in one host directory (:func:`default_host_dir`:
``$SKILL_BENCH_HOST_DIR``, else ``~/.cache/skill_bench/host``): its runs directory, its
pid and start time, its workers (each one's benchmark, method, admission estimate and the
resident memory last read) with the cells it has admitted but not launched yet, its
``--test-concurrency``, and its last memory emergency stop. Under the host lock
(``host.lock``, held for the time it takes to read a few small files), a scheduler reads
the other live schedulers' records (:meth:`HostRegistry.peers`) before it admits a cell
or stops a worker, and writes its own record with the admitted cell before it lets the
lock go:

* the memory gate reserves the headroom of every live scheduler's workers
  (:func:`peers_reserved_gb`), not only its own;
* ``--host-benchmark-concurrency BENCHMARK=N`` caps one benchmark's cells over every
  scheduler of the host (:func:`peers_cells`); by default WebShop's
  (``schedule.DEFAULT_HOST_BENCHMARK_CAPS``). Each scheduler publishes its caps in its
  record, and the lowest cap any live scheduler of the host publishes for a benchmark
  is the one every scheduler enforces (:func:`host_cap`: were each to enforce its own,
  with caps of 24 and 12 on one host the scheduler capped at 12 would be starved while
  the others went past its limit); a scheduler that sees a peer's cap differ from its
  own records
  ``host_cap_mismatch``. A scheduler the host cap holds back while it runs no cell of
  the benchmark says since when in its record (``waiting``), and every other scheduler
  keeps a slot free for it, first come first served (:func:`starved_peers`): a
  finishing cell's slot goes to the scheduler with none, not back to the one that had
  it. A record written before caps were published (an older scheduler still running)
  has none and counts only by its cells;
* a memory emergency stops one worker host-wide: a scheduler whose peer stopped one less
  than the settle period ago, or whose peer's victim still runs, waits. An orphaned
  record marks its victim (``emergency_victim``): the peers wait while it runs and for
  the settle period after they first see it gone.

A record whose process is gone (another pid, or another start time: a reused pid) is
ignored, and removed by the next scheduler that reads it under the lock (after reading it
again: a restarted scheduler of the same runs directory may have published its fresh
record at that path meanwhile). A scheduler that stops with no worker left
removes its own record; one that stops while its workers go on (a stop or a crash: the
supervisor drains them, or the next scheduler adopts them) leaves an **orphaned** record
of those workers, each with its pid and start time, so the peers keep counting them by
the workers' own liveness until they exit or a new scheduler of the runs directory
replaces the record. Nothing here ever blocks a launch by itself: a host
directory that cannot be used leaves each scheduler with its own view, and the scheduler
records that (``host_registry_unavailable``). Coordination covers the schedulers of one
user (the default directory is in the home directory); every scheduler of the host must
see the same directory.

This module is outside both fingerprints (the protocol's and the harness code digest,
``protocol.CODE_FINGERPRINT_DIRS``): it decides when cells start, never what a model sees.
Everything here is blocking (file reads and ``ps``); async callers use
``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.skill_bench.infra.processes import OwnerLock, process_start_time

#: Names the host directory every scheduler of the host shares (tests point it at a
#: temporary directory).
HOST_DIR_ENV = "SKILL_BENCH_HOST_DIR"
LOCK_FILE = "host.lock"
RECORDS_DIR = "schedulers"
#: The record format (a reader skips a record of another version). The fields added
#: since (``host_benchmark_caps``, ``waiting``, a worker's ``emergency_victim``) are
#: optional, so this stays 1: the schedulers of an older checkout running on the same
#: host still read these records, and these theirs.
RECORD_VERSION = 1
#: How long a scheduler waits for the host lock before it goes on with its own view (the
#: lock is held for a few file reads; a long wait means something is wrong with it).
LOCK_WAIT_S = 10.0
#: How long a peer's liveness (its pid running with the recorded start time) is trusted
#: before ``ps`` is asked again. A dead peer's reservation counts at most this long more.
LIVENESS_TTL_S = 10.0


def default_host_dir() -> Path:
    """The host directory: ``$SKILL_BENCH_HOST_DIR``, else ``~/.cache/skill_bench/host``."""
    value = os.environ.get(HOST_DIR_ENV)
    if value:
        return Path(value).absolute()
    return Path.home() / ".cache" / "skill_bench" / "host"


def record_key(runs_dir: Path) -> str:
    """One record per runs directory (a restarted scheduler replaces its predecessor's):
    the sha256 of its absolute, normalised spelling (not resolved: a symlinked runs
    directory keeps the operator's spelling everywhere else too), first 16 hex digits."""
    spelling = os.path.normpath(Path(runs_dir).absolute())
    return hashlib.sha256(spelling.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class Peer:
    """Another live scheduler of the host, as its record says."""

    runs_dir: str
    pid: int
    started: str
    at: str | None
    #: Its workers and its admitted, not yet launched cells: ``cell``, ``benchmark``,
    #: ``method``, ``estimate_gb``, ``rss_gb`` (None: not read yet), ``pending``.
    workers: tuple[Mapping[str, Any], ...] = ()
    test_concurrency: int | None = None
    #: When (epoch seconds) it last stopped a worker in a memory emergency, and whether
    #: that victim is still stopping.
    emergency_at: float | None = None
    emergency_pending: bool = False
    #: Its scheduler stopped and these are the workers it left running (each judged by
    #: its own pid and start time).
    orphaned: bool = False
    #: Its host-wide caps (``--host-benchmark-concurrency`` with the defaults), None for
    #: a record that publishes none (an older scheduler, or an orphaned record).
    host_benchmark_caps: Mapping[str, int] | None = None
    #: The benchmarks the host cap holds its cells back from while it runs none of them,
    #: with since when (epoch seconds).
    waiting: Mapping[str, float] = field(default_factory=dict)

    def reserved_gb(self) -> float:
        """The headroom its workers have not taken yet: each one's estimate minus the
        resident memory last read (a worker not read yet reserves its whole estimate)."""
        total = 0.0
        for worker in self.workers:
            estimate = _number(worker.get("estimate_gb")) or 0.0
            rss = _number(worker.get("rss_gb")) or 0.0
            total += max(0.0, estimate - rss)
        return round(total, 3)

    def cells(self, benchmark: str) -> int:
        """How many of its cells (running or admitted) are of ``benchmark``."""
        return sum(1 for w in self.workers if w.get("benchmark") == benchmark)

    def summary(self) -> dict[str, Any]:
        """For ``state.json``: who it is and what it holds."""
        by_benchmark: dict[str, int] = {}
        for worker in self.workers:
            name = str(worker.get("benchmark"))
            by_benchmark[name] = by_benchmark.get(name, 0) + 1
        return {
            "runs_dir": self.runs_dir,
            "pid": self.pid,
            "at": self.at,
            "cells": dict(sorted(by_benchmark.items())),
            "reserved_gb": self.reserved_gb(),
            "test_concurrency": self.test_concurrency,
            "emergency_at": self.emergency_at,
            "emergency_pending": self.emergency_pending,
            "orphaned": self.orphaned,
            "host_benchmark_caps": dict(self.host_benchmark_caps)
            if self.host_benchmark_caps is not None
            else None,
            "waiting": dict(self.waiting),
        }


def host_cap(
    own: int | None, peers: Iterable[Peer], benchmark: str
) -> tuple[int | None, list[Peer]]:
    """The host-wide cap of ``benchmark`` in force: the lowest of this
    scheduler's own (``own``, None: none) and every live peer's published one, and the
    peers that publish another cap than ``own`` (None when nobody caps it)."""
    caps = [own] if own is not None else []
    other: list[Peer] = []
    for peer in peers:
        theirs = (peer.host_benchmark_caps or {}).get(benchmark)
        if theirs is None:
            continue
        caps.append(theirs)
        if theirs != own:
            other.append(peer)
    return (min(caps) if caps else None), other


def starved_peers(
    peers: Iterable[Peer],
    benchmark: str,
    *,
    before: tuple[float, str] | None = None,
) -> list[Peer]:
    """The peers the host cap holds back from ``benchmark`` while they run none of its
    cells (their record's ``waiting``), for each of which a scheduler keeps a slot
    free. ``before``: this scheduler's own ``(since, runs_dir)`` when it waits too: only
    the peers waiting since earlier (then by runs directory) go first, so two waiting
    schedulers never hold back for each other."""
    out = []
    for peer in peers:
        since = peer.waiting.get(benchmark)
        if since is None or peer.cells(benchmark) != 0:
            continue
        if before is not None and (since, peer.runs_dir) >= before:
            continue
        out.append(peer)
    return out


def _caps(value: Any) -> dict[str, int] | None:
    """A record's ``host_benchmark_caps`` when it is one (benchmark -> positive int)."""
    if not isinstance(value, Mapping):
        return None
    caps = {
        str(k): v
        for k, v in value.items()
        if isinstance(v, int) and not isinstance(v, bool) and v > 0
    }
    return caps if len(caps) == len(value) else None


def peers_reserved_gb(peers: Iterable[Peer]) -> float:
    """The headroom the other schedulers' workers have not taken yet, in all."""
    return round(sum(p.reserved_gb() for p in peers), 3)


def peers_cells(peers: Iterable[Peer], benchmark: str) -> int:
    """How many cells of ``benchmark`` the other schedulers run or have admitted."""
    return sum(p.cells(benchmark) for p in peers)


def recent_emergency(
    peers: Iterable[Peer], *, now: float, settle_s: float
) -> list[Peer]:
    """The peers whose memory emergency victim is still stopping, or who stopped one
    less than ``settle_s`` ago: no other worker of the host is stopped meanwhile."""
    return [
        p
        for p in peers
        if p.emergency_pending
        or (p.emergency_at is not None and now - p.emergency_at < settle_s)
    ]


@dataclass
class HostRegistry:
    """One scheduler's view of the host directory (blocking methods)."""

    directory: Path
    runs_dir: Path
    #: ``pid -> started`` reader (``ps -o lstart``); injectable for tests.
    start_time: Callable[[int], str | None] = process_start_time
    clock: Callable[[], float] = time.monotonic
    #: Wall clock (epoch seconds, as ``emergency_at``); injectable for tests.
    wall: Callable[[], float] = time.time
    _liveness: dict[tuple[int, str], tuple[float, bool]] = field(
        default_factory=dict, repr=False
    )
    #: When (wall clock) this reader first saw an orphaned record's emergency victims
    #: gone, by record path and victims (the settle period runs from then, not from the
    #: stop that orphaned scheduler sent).
    _victims_gone: dict[tuple[str, tuple[Any, ...]], float] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).absolute()
        self.runs_dir = Path(self.runs_dir).absolute()

    @property
    def key(self) -> str:
        return record_key(self.runs_dir)

    @property
    def records(self) -> Path:
        return self.directory / RECORDS_DIR

    @property
    def path(self) -> Path:
        """This scheduler's own record."""
        return self.records / f"{self.key}.json"

    def lock(self, wait_s: float = LOCK_WAIT_S) -> OwnerLock:
        """The host lock (exclusive; raises ``OwnedProcessError`` after ``wait_s``, and
        ``OSError`` when the directory cannot be used). The caller releases it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return OwnerLock.acquire(self.directory / LOCK_FILE, wait_s=wait_s)

    def publish(
        self,
        *,
        pid: int,
        started: str,
        workers: Sequence[Mapping[str, Any]],
        test_concurrency: int | None = None,
        emergency_at: float | None = None,
        emergency_pending: bool = False,
        orphaned: bool = False,
        host_benchmark_caps: Mapping[str, int] | None = None,
        waiting: Mapping[str, float] | None = None,
    ) -> None:
        """Replace this scheduler's record (a temporary file renamed over it: a reader
        sees the old record or the new one, never half of one). Not fsync'd: the record
        is live state, worthless after a reboot. ``orphaned``: the scheduler stops and
        ``workers`` (each with its ``pid`` and ``started``; an emergency victim with
        ``emergency_victim``) go on without it. ``host_benchmark_caps``: its host-wide
        caps; ``waiting``: the benchmarks the host cap holds it back from while it runs
        none of their cells, with since when."""
        record = {
            "version": RECORD_VERSION,
            "runs_dir": str(self.runs_dir),
            "pid": int(pid),
            "started": " ".join(str(started).split()),
            "at": _now(),
            "workers": [dict(w) for w in workers],
            "test_concurrency": test_concurrency,
            "emergency_at": emergency_at,
            "emergency_pending": bool(emergency_pending),
            "orphaned": bool(orphaned),
            "host_benchmark_caps": dict(host_benchmark_caps)
            if host_benchmark_caps is not None
            else None,
            "waiting": dict(sorted((waiting or {}).items())),
        }
        self.records.mkdir(parents=True, exist_ok=True)
        tmp = self.records / f".{self.key}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def withdraw(self) -> None:
        """Remove this scheduler's record (it stops)."""
        self.path.unlink(missing_ok=True)

    def _alive(self, pid: int, started: str) -> bool:
        key = (pid, started)
        cached = self._liveness.get(key)
        clock = self.clock()
        if cached is not None and clock - cached[0] < LIVENESS_TTL_S:
            return cached[1]
        alive = bool(started) and self.start_time(pid) == started
        self._liveness[key] = (clock, alive)
        return alive

    def peers(self, *, prune: bool = False) -> list[Peer]:
        """The other live schedulers' records, by runs directory. A record of a process
        that is gone (or of another record version, or unreadable) is skipped; with
        ``prune`` (the caller holds the host lock) a dead one is removed."""
        found: list[Peer] = []
        if not self.records.is_dir():
            return found
        for path in sorted(self.records.glob("*.json")):
            if path.name == self.path.name:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                record = json.loads(text)
            except (OSError, ValueError):
                continue  # torn by a crash mid-rename is impossible; unreadable: skip
            if not isinstance(record, dict) or record.get("version") != RECORD_VERSION:
                continue
            pid = record.get("pid")
            started = " ".join(str(record.get("started") or "").split())
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                continue
            workers = tuple(
                w for w in record.get("workers") or () if isinstance(w, Mapping)
            )
            orphaned = record.get("orphaned") is True
            emergency_at = _number(record.get("emergency_at"))
            emergency_pending = record.get("emergency_pending") is True
            if orphaned:
                # Its scheduler is gone; the workers it left count while they live.
                victims = tuple(
                    (str(w.get("pid")), str(w.get("started")))
                    for w in workers
                    if w.get("emergency_victim") is True
                )
                workers = tuple(w for w in workers if self._worker_alive(w))
                alive = bool(workers)
                if victims:
                    # Its emergency stop holds while the victim runs, and for the
                    # settle period after it is first seen gone.
                    emergency_pending = any(
                        w.get("emergency_victim") is True for w in workers
                    )
                    key = (path.name, victims)
                    if emergency_pending:
                        self._victims_gone.pop(key, None)
                    else:
                        gone = self._victims_gone.setdefault(key, self.wall())
                        emergency_at = max(emergency_at or gone, gone)
            else:
                alive = self._alive(pid, started)
            if not alive:
                if prune:
                    self._prune(path, text)
                continue
            tc = record.get("test_concurrency")
            waiting = record.get("waiting")
            found.append(
                Peer(
                    runs_dir=str(record.get("runs_dir")),
                    pid=pid,
                    started=started,
                    at=record.get("at") if isinstance(record.get("at"), str) else None,
                    workers=workers,
                    test_concurrency=tc
                    if isinstance(tc, int) and not isinstance(tc, bool)
                    else None,
                    emergency_at=emergency_at,
                    emergency_pending=emergency_pending,
                    orphaned=orphaned,
                    # A scheduler that is gone caps nothing and waits for nothing.
                    host_benchmark_caps=None
                    if orphaned
                    else _caps(record.get("host_benchmark_caps")),
                    waiting={}
                    if orphaned or not isinstance(waiting, Mapping)
                    else {
                        str(b): float(t)
                        for b, t in waiting.items()
                        if _number(t) is not None
                    },
                )
            )
        return found

    def _worker_alive(self, worker: Mapping[str, Any]) -> bool:
        """An orphaned record's worker lives: its pid runs with its recorded start time."""
        pid = worker.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        return self._alive(pid, " ".join(str(worker.get("started") or "").split()))

    @staticmethod
    def _prune(path: Path, judged: str) -> None:
        """Remove a dead record, only if the file still holds the very record judged
        dead: a restarted scheduler of that runs directory publishes its fresh record at
        the same path, outside the lock, and must not lose it. The window
        left between this read and the unlink is a few microseconds, and a record lost
        in it comes back at its scheduler's next poll."""
        try:
            if path.read_text(encoding="utf-8") != judged:
                return
        except OSError:
            return
        path.unlink(missing_ok=True)
