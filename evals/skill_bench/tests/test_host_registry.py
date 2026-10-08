"""The host registry (``host_registry.py``): what the schedulers of one host tell each
other (for example three replicate schedulers on one host). The scheduler's use of it is
tested in ``test_schedule.py`` (two and three schedulers at once)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evals.skill_bench import host_registry
from evals.skill_bench.host_registry import (
    HostRegistry,
    Peer,
    peers_cells,
    peers_reserved_gb,
    recent_emergency,
    record_key,
)
from evals.skill_bench.infra.processes import OwnedProcessError, process_start_time

SELF_STARTED = process_start_time(os.getpid()) or ""


def registry(host: Path, runs: Path, **kw) -> HostRegistry:
    return HostRegistry(host, runs, **kw)


def worker(benchmark: str, estimate: float, rss: float | None, **kw) -> dict:
    return {
        "cell": f"{benchmark}-x",
        "benchmark": benchmark,
        "method": "mirix",
        "estimate_gb": estimate,
        "rss_gb": rss,
        "pending": False,
        **kw,
    }


def test_the_default_directory_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(host_registry.HOST_DIR_ENV, str(tmp_path / "h"))
    assert host_registry.default_host_dir() == tmp_path / "h"
    monkeypatch.delenv(host_registry.HOST_DIR_ENV)
    assert host_registry.default_host_dir() == (
        Path.home() / ".cache" / "skill_bench" / "host"
    )


def test_one_record_per_runs_directory_whatever_its_spelling(tmp_path):
    runs = tmp_path / "runs" / "phase16-r1"
    assert record_key(runs) == record_key(tmp_path / "runs" / "x" / ".." / "phase16-r1")
    assert record_key(runs) != record_key(tmp_path / "runs" / "phase16-r2")
    assert len(record_key(runs)) == 16


def test_a_peer_sees_the_record_and_never_its_own(tmp_path):
    host = tmp_path / "host"
    one = registry(host, tmp_path / "r1")
    two = registry(host, tmp_path / "r2")
    one.publish(
        pid=os.getpid(),
        started=SELF_STARTED,
        workers=[worker("webshop", 1.4, 0.4), worker("alfworld", 0.5, None)],
        test_concurrency=4,
        emergency_at=123.0,
        emergency_pending=True,
    )
    assert one.peers() == []
    (peer,) = two.peers()
    assert peer.runs_dir == str((tmp_path / "r1").absolute())
    assert peer.pid == os.getpid() and peer.test_concurrency == 4
    assert peer.cells("webshop") == 1 and peer.cells("alfworld") == 1
    assert peer.reserved_gb() == pytest.approx(1.0 + 0.5)
    assert peer.emergency_at == 123.0 and peer.emergency_pending is True
    summary = peer.summary()
    assert summary["cells"] == {"alfworld": 1, "webshop": 1}
    assert summary["reserved_gb"] == pytest.approx(1.5)
    one.withdraw()
    assert two.peers() == []
    one.withdraw()  # twice is harmless


def test_a_dead_or_reused_pid_and_a_foreign_record_are_skipped(tmp_path):
    host = tmp_path / "host"
    alive = registry(host, tmp_path / "r1")
    alive.publish(pid=os.getpid(), started=SELF_STARTED, workers=[])
    dead = registry(host, tmp_path / "r2")
    dead.publish(pid=os.getpid(), started="Mon Jan  1 00:00:00 2001", workers=[])
    records = host / host_registry.RECORDS_DIR
    (records / "other-version.json").write_text(
        json.dumps({"version": 99, "pid": os.getpid(), "started": SELF_STARTED})
    )
    (records / "torn.json").write_text("{")
    (records / "no-pid.json").write_text(
        json.dumps({"version": host_registry.RECORD_VERSION, "pid": True})
    )
    reader = registry(host, tmp_path / "r3")
    assert [p.runs_dir for p in reader.peers()] == [str((tmp_path / "r1").absolute())]
    assert dead.path.is_file()  # read without the lock: nothing removed
    reader.peers(prune=True)
    assert not dead.path.is_file() and alive.path.is_file()
    assert (
        records / "other-version.json"
    ).is_file()  # never removes what it cannot read


def test_liveness_is_asked_again_only_after_its_ttl(tmp_path):
    host = tmp_path / "host"
    registry(host, tmp_path / "r1").publish(pid=4242, started="then", workers=[])
    asked: list[int] = []
    clock = [0.0]

    def start_time(pid: int) -> str | None:
        asked.append(pid)
        return "then" if len(asked) == 1 else None  # it dies after the first look

    reader = registry(
        host, tmp_path / "r2", start_time=start_time, clock=lambda: clock[0]
    )
    assert len(reader.peers()) == 1
    clock[0] = host_registry.LIVENESS_TTL_S / 2
    assert len(reader.peers()) == 1 and asked == [4242]
    clock[0] = host_registry.LIVENESS_TTL_S + 1
    assert reader.peers() == [] and asked == [4242, 4242]


def test_a_dead_record_is_pruned_only_if_it_is_still_the_one_judged(tmp_path):
    """A peer reads the dead predecessor's record of runs directory
    X and asks ``ps``; meanwhile X's restarted scheduler publishes its fresh record at
    the same path, outside the lock. The prune must not delete the fresh record."""
    host = tmp_path / "host"
    restarted = registry(host, tmp_path / "phase16-r2")
    restarted.publish(
        pid=999_999,
        started="Mon Jan  1 00:00:00 2024",
        workers=[worker("webshop", 1.4, None)],
    )

    def start_time(pid: int) -> str | None:
        if pid == 999_999:
            # While the peer asks about the dead pid, the restarted scheduler publishes.
            restarted.publish(
                pid=os.getpid(),
                started=SELF_STARTED,
                workers=[worker("webshop", 1.4, None)],
            )
            return None
        return process_start_time(pid)

    reader = registry(host, tmp_path / "phase16-r1", start_time=start_time)
    assert reader.peers(prune=True) == []  # it judged the record it read: dead
    assert restarted.path.is_file()  # ... and kept the fresh one
    (peer,) = registry(host, tmp_path / "phase16-r3").peers(prune=True)
    assert peer.pid == os.getpid() and peer.cells("webshop") == 1


def test_an_orphaned_record_counts_the_workers_while_they_live(tmp_path):
    """A scheduler that stopped (or crashed) while its workers go on
    leaves an orphaned record of them; the peers judge it by the workers' own pid and
    start time, not the scheduler's, and drop each worker once it has exited."""
    host = tmp_path / "host"
    stopped = registry(host, tmp_path / "r1")
    stopped.publish(
        pid=999_998,  # the scheduler that stopped
        started="Mon Jan  1 00:00:00 2024",
        workers=[
            worker("webshop", 1.4, 0.4, pid=os.getpid(), started=SELF_STARTED),
            worker("alfworld", 0.5, None, pid=999_997, started="gone"),
        ],
        orphaned=True,
    )
    (peer,) = registry(host, tmp_path / "r2").peers(prune=True)
    assert peer.orphaned is True and peer.summary()["orphaned"] is True
    assert peer.cells("webshop") == 1 and peer.cells("alfworld") == 0
    assert peer.reserved_gb() == pytest.approx(1.0)
    # Every worker gone: the record is dead, and pruned.
    stopped.publish(
        pid=999_998,
        started="Mon Jan  1 00:00:00 2024",
        workers=[worker("webshop", 1.4, 0.4, pid=999_997, started="gone")],
        orphaned=True,
    )
    assert registry(host, tmp_path / "r2").peers(prune=True) == []
    assert not stopped.path.is_file()


def test_reservations_caps_and_emergencies_over_the_peers():
    a = Peer(
        "r1",
        1,
        "s",
        None,
        workers=(worker("webshop", 1.4, 2.0), worker("webshop", 0.5, None)),
        emergency_at=100.0,
    )
    b = Peer(
        "r2",
        2,
        "s",
        None,
        workers=(worker("alfworld", 1.4, 0.4), worker("webshop", 0.35, 0.1)),
        emergency_pending=True,
    )
    c = Peer("r3", 3, "s", None)
    # A worker above its estimate reserves nothing (never negative); one not read yet
    # reserves its whole estimate.
    assert a.reserved_gb() == pytest.approx(0.5)
    assert peers_reserved_gb([a, b, c]) == pytest.approx(0.5 + 1.0 + 0.25)
    assert peers_cells([a, b, c], "webshop") == 3
    assert peers_cells([a, b, c], "other_benchmark") == 0
    assert recent_emergency([a, b, c], now=130.0, settle_s=60.0) == [a, b]
    assert recent_emergency([a, b, c], now=161.0, settle_s=60.0) == [b]


def test_the_host_lock_is_exclusive(tmp_path):
    one = registry(tmp_path / "host", tmp_path / "r1")
    two = registry(tmp_path / "host", tmp_path / "r2")
    held = one.lock()
    try:
        with pytest.raises(OwnedProcessError):
            two.lock(wait_s=0.0)
    finally:
        held.release()
    two.lock(wait_s=0.0).release()


def test_the_shared_postgres_has_room_for_every_replicates_mirix_servers():
    """The schedulers of one host share one Postgres. Each MIRIX server
    pools up to MIRIX's ``pg_pool_size + pg_max_overflow`` connections (one worker). Only
    a MIRIX cell runs a MIRIX server, so a scheduler runs at most as many as its study has
    MIRIX cells (or its concurrency, if lower; ``protocol.SCHEDULE_CONCURRENCY_RANGE``),
    and three replicate schedulers may share the host with a development scheduler: the
    compose Postgres must allow them all (read from the sources, never by importing
    MIRIX's settings, which read the environment)."""
    import re

    from evals.skill_bench import protocol, study

    compose = (protocol.SKILL_BENCH_DIR / "infra" / "compose.yaml").read_text()
    allowed = int(re.search(r"max_connections=(\d+)", compose).group(1))
    settings = (protocol.REPO_ROOT / "mirix" / "settings.py").read_text()
    pool = int(re.search(r"pg_pool_size: int = (\d+)", settings).group(1))
    overflow = int(re.search(r"pg_max_overflow: int = (\d+)", settings).group(1))
    schedulers = 3 + 1  # the three replicates and a development pass
    mirix_cells = max(
        len(entry.benchmarks) * len(entry.models)
        for entry in study.STUDIES.values()
        if "mirix" in entry.methods
    )
    servers = min(mirix_cells, protocol.SCHEDULE_CONCURRENCY_RANGE[1])
    assert allowed >= schedulers * servers * (pool + overflow)


def test_each_record_publishes_its_caps_and_what_the_cap_holds_back(tmp_path):
    """A scheduler's record carries its host-wide caps and the
    benchmarks the cap holds it back from while it runs none of their cells; the lowest
    published cap is the one in force, and a peer that runs none and waits is starved.
    A record without them (an older scheduler) caps nothing; an orphaned one neither
    caps nor waits."""
    host = tmp_path / "host"
    live = dict(pid=os.getpid(), started=SELF_STARTED)
    registry(host, tmp_path / "r1").publish(
        **live,
        workers=[worker("webshop", 1.4, 0.4)],
        host_benchmark_caps={"webshop": 24},
    )
    registry(host, tmp_path / "r2").publish(
        **live,
        workers=[],
        host_benchmark_caps={"webshop": 12},
        waiting={"webshop": 100.0},
    )
    old = registry(host, tmp_path / "r3")
    old.publish(**live, workers=[worker("webshop", 1.4, 0.4)])
    text = json.loads(old.path.read_text())
    del text["host_benchmark_caps"], text["waiting"]  # as an older checkout writes it
    old.path.write_text(json.dumps(text))
    peers = {p.runs_dir: p for p in registry(host, tmp_path / "me").peers()}
    r1, r2, r3 = (peers[str(tmp_path / n)] for n in ("r1", "r2", "r3"))
    assert r1.host_benchmark_caps == {"webshop": 24} and r1.waiting == {}
    assert r2.waiting == {"webshop": 100.0}
    assert r2.summary()["waiting"] == {"webshop": 100.0}
    assert (
        r3.host_benchmark_caps is None and r3.summary()["host_benchmark_caps"] is None
    )
    cap, other = host_registry.host_cap(6, peers.values(), "webshop")
    assert cap == 6 and sorted(p.runs_dir for p in other) == [r1.runs_dir, r2.runs_dir]
    assert host_registry.host_cap(24, [r1, r3], "webshop") == (24, [])
    assert host_registry.host_cap(None, [r3], "webshop") == (None, [])
    assert host_registry.host_cap(None, [r1, r2], "alfworld") == (None, [])
    assert host_registry.starved_peers(peers.values(), "webshop") == [r2]
    assert host_registry.starved_peers([r1, r3], "webshop") == []
    # A scheduler waiting itself yields only to the ones waiting since earlier.
    assert host_registry.starved_peers([r2], "webshop", before=(99.0, "z")) == []
    assert host_registry.starved_peers([r2], "webshop", before=(101.0, "a")) == [r2]
    assert host_registry.starved_peers([r2], "webshop", before=(100.0, "~")) == [r2]
    # A malformed cap map is no cap.
    assert host_registry._caps({"webshop": 0}) is None
    assert host_registry._caps({"webshop": True}) is None
    assert host_registry._caps(["webshop"]) is None
    # The record version stays 1: an older checkout's schedulers keep reading them.
    assert host_registry.RECORD_VERSION == 1
    registry(host, tmp_path / "r4").publish(
        pid=999_998,
        started="gone",
        workers=[worker("webshop", 1.4, 0.4, **live)],
        orphaned=True,
        host_benchmark_caps={"webshop": 2},
        waiting={"webshop": 1.0},
    )
    orphan = next(
        p for p in registry(host, tmp_path / "me").peers() if p.runs_dir.endswith("r4")
    )
    assert orphan.host_benchmark_caps is None and orphan.waiting == {}


def test_an_orphaned_emergency_holds_while_its_victim_runs_and_settles_after(
    tmp_path,
):
    """A scheduler that stops while its emergency victim still
    captures its checkpoint leaves the victim marked in its orphaned record: the peers
    stop no other worker while it runs, and for the settle period after they first see
    it gone (not from the stop the scheduler sent)."""
    host = tmp_path / "host"
    clock = [1000.0]
    stopped = registry(host, tmp_path / "r1")
    victim = worker(
        "alfworld",
        1.4,
        0.4,
        pid=os.getpid(),
        started=SELF_STARTED,
        emergency_victim=True,
    )
    other = worker("webshop", 1.4, 0.4, pid=os.getpid(), started=SELF_STARTED)
    stopped.publish(
        pid=999_998,
        started="gone",
        workers=[victim, other],
        emergency_at=10.0,
        emergency_pending=True,
        orphaned=True,
    )
    reader = registry(host, tmp_path / "r2", wall=lambda: clock[0])
    (peer,) = reader.peers()
    assert peer.emergency_pending is True
    assert recent_emergency([peer], now=clock[0], settle_s=60.0) == [peer]
    # The victim exits (its pid gone); the other worker goes on.
    stopped.publish(
        pid=999_998,
        started="gone",
        workers=[{**victim, "pid": 999_997, "started": "gone"}, other],
        emergency_at=10.0,
        emergency_pending=True,
        orphaned=True,
    )
    reader = registry(host, tmp_path / "r2", wall=lambda: clock[0])
    (peer,) = reader.peers()
    assert peer.emergency_pending is False and peer.emergency_at == 1000.0
    clock[0] = 1030.0
    (peer,) = reader.peers()
    assert peer.emergency_at == 1000.0  # first seen gone, not re-stamped
    assert recent_emergency([peer], now=1030.0, settle_s=60.0) == [peer]
    assert recent_emergency([peer], now=1061.0, settle_s=60.0) == []
