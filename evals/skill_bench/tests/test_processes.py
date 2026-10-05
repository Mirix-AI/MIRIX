"""Owned-process cleanup never touches a live harness's children or unproven processes (H8; N1, N2)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.cell_records import read_jsonl, read_jsonl_counted
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    cleanup_owned,
    ends_torn,
    live_cell_workers,
    lock_is_held,
    parse_ps_table,
    process_group_rss,
    process_start_time,
    proven_group_members,
    registered_process_groups,
    row_in_use,
    signal_group,
    stop_orphaned,
    sweep_owned,
)

SLEEPER = f"{sys.executable} -c 'import time; time.sleep(60)'"


def rows(registry: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in registry.read_text().splitlines()]


def cleaned(registry: Path) -> dict[int, dict[str, Any]]:
    return {r["pid"]: r for r in rows(registry) if r["event"] == "cleaned"}


def wait_gone(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process_start_time(pid) is None:
            return True
        time.sleep(0.05)
    return False


def wait_for(predicate: Any, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.05)


def kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def orphaning_leader(tmp_path: Path) -> tuple[list[str], Path]:
    """A leader that starts one background member in its group, reports its PID and exits."""
    pid_file = tmp_path / "member.pid"
    script = f"{SLEEPER} & echo $! > {pid_file}; exit 0"
    return ["sh", "-c", script], pid_file


def read_pid(path: Path) -> int:
    wait_for(lambda: path.is_file() and path.read_text().strip().isdigit())
    return int(path.read_text().strip())


# --- owner locks ---------------------------------------------------------------------------


def test_owner_lock_is_exclusive_until_released(tmp_path):
    path = tmp_path / "locks" / "gateway.lock"
    assert not lock_is_held(path)  # a missing file is free
    owner = OwnerLock.acquire(path)
    try:
        assert owner.held and lock_is_held(path)
        with pytest.raises(OwnedProcessError, match="another live harness"):
            OwnerLock.acquire(path, wait_s=0.2)
    finally:
        owner.release()
    assert not owner.held and not lock_is_held(path)
    OwnerLock.acquire(path, wait_s=0.0).release()


def test_a_shared_users_lock_admits_many_users_and_blocks_an_exclusive_one(tmp_path):
    """X12: every live run.py holds its gateway's users lock shared at once; while any
    holds it the lock reads as held, and no exclusive owner can take it."""
    path = tmp_path / "_gateway" / "gateway-127.0.0.1-1.users.lock"
    first = OwnerLock.acquire(path, shared=True)
    second = OwnerLock.acquire(path, shared=True, wait_s=0.0)
    try:
        assert lock_is_held(path)
        with pytest.raises(OwnedProcessError):
            OwnerLock.acquire(path, wait_s=0.2)
        first.release()
        assert lock_is_held(path)  # the second user still holds it
    finally:
        first.release()
        second.release()
    assert not lock_is_held(path)


USER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from evals.skill_bench.infra.processes import OwnerLock
    users = OwnerLock.acquire(Path(sys.argv[1]), shared=True)
    print("holding", flush=True)
    time.sleep(60)
    """
)


def test_a_gateway_its_workers_still_use_is_never_cleaned(tmp_path):
    """X12: the gateway's owner died (a scheduler that stopped) while a run.py worker still
    holds the users lock: neither a sweep nor a cleanup kills it. Once the last user is
    gone, the orphan is cleaned as before."""
    runs = tmp_path / "runs"
    registry = runs / "_gateway" / "owned_processes.jsonl"
    owner_path = runs / "_gateway" / "gateway-127.0.0.1-1.lock"
    users_path = runs / "_gateway" / "gateway-127.0.0.1-1.users.lock"
    owner = OwnerLock.acquire(owner_path)
    child = OwnedProcess.spawn(
        "gateway",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin"},
        cwd=Path.cwd(),
        log_path=tmp_path / "g.log",
        registry=registry,
        owner_lock=owner_path,
        users_lock=users_path,
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", USER, str(users_path)],
        cwd=protocol.REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert (
            worker.stdout is not None and worker.stdout.readline().strip() == "holding"
        )
        assert rows(registry)[0]["users_lock"] == users_path.name
        owner.release()  # the owner is gone; its worker is not
        assert row_in_use(registry, rows(registry)[0])
        assert cleanup_owned(registry) == []
        assert sweep_owned(runs) == {}
        assert process_start_time(child.pid) is not None
        assert [r["event"] for r in rows(registry)] == ["started"]
        worker.kill()
        worker.wait(timeout=10)
        wait_for(lambda: not lock_is_held(users_path), 10)
        assert cleanup_owned(registry) == [child.pid]
        assert child.popen.wait(timeout=10) < 0  # our own child: reaped here
    finally:
        owner.release()
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
        kill_quietly(child.pid)


def test_an_attaching_sweep_leaves_the_gateway_registry_alone(tmp_path):
    """X12: a scheduler that only attaches to a gateway sweeps the cells, never
    ``_gateway`` (``skip_dirs``), whatever the state of the gateway's locks."""
    runs = tmp_path / "runs"
    registry = runs / "_gateway" / "owned_processes.jsonl"
    child = OwnedProcess.spawn(
        "gateway",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin"},
        cwd=Path.cwd(),
        log_path=tmp_path / "g.log",
        registry=registry,
    )
    try:
        assert sweep_owned(runs, skip_dirs=("_gateway",)) == {}
        assert process_start_time(child.pid) is not None
        assert sweep_owned(runs) == {"_gateway/owned_processes.jsonl": [child.pid]}
        assert child.popen.wait(timeout=10) < 0  # our own child: reaped here
    finally:
        kill_quietly(child.pid)


# --- JSON lines ------------------------------------------------------------------------------


def test_an_append_after_a_torn_line_keeps_every_later_record(tmp_path):
    """X15: a line torn by a crash or a full disk is closed before the next append, so
    the record after it never merges with it; readers skip the fragment and count it."""
    path = tmp_path / "gateway.jsonl"
    assert not ends_torn(path)  # missing
    append_jsonl(path, {"event": "call", "n": 1})
    assert not ends_torn(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "call", "n": 2, "cost_to')  # the crash
    assert ends_torn(path)
    append_jsonl(path, {"event": "call", "n": 3})
    append_jsonl(path, {"event": "call", "n": 4})
    assert not ends_torn(path)
    rows_read, torn = read_jsonl_counted(path)
    assert [r["n"] for r in rows_read] == [1, 3, 4] and torn == 1
    assert [r["n"] for r in read_jsonl(path)] == [1, 3, 4]
    # A fragment in the middle of a file (an older harness appended onto it) is skipped
    # too, not a reason to lose the records after it.
    path.write_text('{"n": 1}\n{"n": 2{"n": 3}\n{"n": 4}\n\xff\n', encoding="latin-1")
    rows_read, torn = read_jsonl_counted(path)
    assert [r["n"] for r in rows_read] == [1, 4] and torn == 2


HARNESS = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from evals.skill_bench.infra.processes import OwnedProcess, OwnerLock
    lock, registry, log = map(Path, sys.argv[1:4])
    owner = OwnerLock.acquire(lock)
    child = OwnedProcess.spawn(
        "gateway",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin"},
        cwd=Path.cwd(),
        log_path=log,
        registry=registry,
        owner_lock=lock,
    )
    print(child.pid, flush=True)
    time.sleep(60)
    """
)


def test_cleanup_spares_a_live_harness_and_cleans_up_after_it_dies(tmp_path):
    """N1: a registry shared by several harnesses; only a dead owner's child is killed."""
    runs = tmp_path / "runs"
    registry = runs / "_gateway" / "owned_processes.jsonl"
    lock = runs / "_gateway" / "gateway-127.0.0.1-1.lock"
    harness = subprocess.Popen(
        [
            sys.executable,
            "-c",
            HARNESS,
            str(lock),
            str(registry),
            str(tmp_path / "g.log"),
        ],
        cwd=protocol.REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = 0
    try:
        assert harness.stdout is not None
        child = int(harness.stdout.readline())
        started = rows(registry)[0]
        assert started["owner_lock"] == lock.name and started["token"]
        assert lock_is_held(lock)
        # Another harness starting a gateway (or the scheduler's sweep) must not touch it.
        assert cleanup_owned(registry) == []
        assert sweep_owned(runs) == {}
        assert process_start_time(child) is not None
        assert [r["event"] for r in rows(registry)] == ["started"]
        # The owner dies without stopping its child: the lock goes with it.
        harness.kill()
        harness.wait(timeout=10)
        wait_for(lambda: not lock_is_held(lock), 10)
        assert (
            process_start_time(child) is not None
        )  # the orphan lives on (own session)
        assert cleanup_owned(registry) == [child]
        assert wait_gone(child)
        assert cleaned(registry)[child]["reason"] == "killed"
        assert cleanup_owned(registry) == []  # settled
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait(timeout=10)
        if child:
            kill_quietly(child)


def test_a_gone_leaders_unrelated_group_is_never_signalled(tmp_path):
    """N2: a stale row whose PGID now names a stranger's orphaned group (reboot, wrap-around)."""
    argv, pid_file = orphaning_leader(tmp_path)
    leader = subprocess.Popen(argv, start_new_session=True)  # not ours: no token
    leader.wait(timeout=10)
    member = read_pid(pid_file)
    registry = tmp_path / "owned_processes.jsonl"
    try:
        assert process_start_time(leader.pid) is None
        assert process_start_time(member) is not None  # the group lives on
        stale = [
            {
                "event": "started",
                "name": "mirix",
                "pid": leader.pid,
                "pgid": leader.pid,
                "started": "Mon Jan  1 00:00:00 2024",
                "token": "0" * 32,  # a token the stranger does not carry
            },
        ]
        registry.write_text("".join(json.dumps(r) + "\n" for r in stale))
        assert proven_group_members(leader.pid, "0" * 32) == []
        assert cleanup_owned(registry) == []
        time.sleep(0.2)
        assert process_start_time(member) is not None  # never signalled
        assert cleaned(registry)[leader.pid]["reason"] == "gone"
        # A legacy row without any token cannot prove anything either.
        registry.write_text(json.dumps({**stale[0], "token": None}) + "\n")
        assert cleanup_owned(registry) == []
        assert process_start_time(member) is not None
    finally:
        kill_quietly(member)


def test_our_orphans_are_killed_when_their_leader_is_gone(tmp_path):
    """The members that carry the row's token are ours: killed one by one, nothing else."""
    argv, pid_file = orphaning_leader(tmp_path)
    registry = tmp_path / "owned_processes.jsonl"
    owned = OwnedProcess.spawn(
        "mirix",
        argv,
        env=dict(os.environ),
        cwd=tmp_path,
        log_path=tmp_path / "m.log",
        registry=registry,
    )
    owned.popen.wait(timeout=10)  # the leader exits and is reaped: its PID is free
    member = read_pid(pid_file)
    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        token = rows(registry)[0]["token"]
        assert owned.token == token
        wait_for(lambda: proven_group_members(owned.pid, token) == [member], 10)
        assert cleanup_owned(registry) == []  # no leader was running
        assert wait_gone(member)
        row = cleaned(registry)[owned.pid]
        assert (row["reason"], row["orphans"]) == ("orphans_killed", [member])
        assert stranger.poll() is None
    finally:
        kill_quietly(member)
        stranger.kill()
        stranger.wait(timeout=10)


@pytest.mark.asyncio
async def test_stop_kills_only_the_proven_orphans_of_an_exited_leader(tmp_path):
    argv, pid_file = orphaning_leader(tmp_path)
    owned = OwnedProcess.spawn(
        "server",
        argv,
        env=dict(os.environ),
        cwd=tmp_path,
        log_path=tmp_path / "s.log",
        registry=tmp_path / "r.jsonl",
    )
    owned.popen.wait(timeout=10)
    member = read_pid(pid_file)
    try:
        wait_for(lambda: proven_group_members(owned.pid, owned.token) == [member], 10)
        assert await owned.stop() == 0
        assert wait_gone(member)
        assert [r["event"] for r in rows(tmp_path / "r.jsonl")] == [
            "started",
            "stopped",
        ]
    finally:
        kill_quietly(member)


# --- resource guard and supervisor helpers --------------------------------------------------

HOLDER = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path
    from evals.skill_bench.infra.processes import OwnerLock, process_start_time
    cell = Path(sys.argv[1])
    cell.mkdir(parents=True, exist_ok=True)
    lock = OwnerLock.acquire(cell / "cell.lock")
    pid = os.getpid()
    (cell / "process.json").write_text(json.dumps({"pid": pid, "started": process_start_time(pid)}))
    blob = bytearray(40 * 1024 * 1024)  # resident: every page written
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    print("ready", flush=True)
    time.sleep(60)
    """
)


def start_holder(cell: Path) -> subprocess.Popen[str]:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(cell)],
        cwd=protocol.REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
    return holder


def test_ps_rows_are_summed_per_process_group():
    text = "    1     1  100\n  10    10  2048\n  11    10  1024\n  12    12  64\nnoise\n 13 13\n"
    assert parse_ps_table(text) == [
        (1, 1, 100),
        (10, 10, 2048),
        (11, 10, 1024),
        (12, 12, 64),
    ]
    assert process_group_rss([10, 12, 99], ps_output=text) == {
        10: 3072 * 1024,
        12: 64 * 1024,
    }
    assert process_group_rss([], ps_output=text) == {}


def test_live_cell_workers_names_only_proven_live_run_py(tmp_path):
    """The drain signals exactly the live run.py processes: cell.lock held and
    process.json naming the same process (start time), never a reused pid."""
    runs = tmp_path / "runs"
    live = start_holder(runs / "webshop-luna-mirix")
    try:
        # A cell whose process.json names a process that is not the lock holder's.
        stale = runs / "webshop-luna56-mirix"
        other = start_holder(stale)
        (stale / "process.json").write_text(
            json.dumps({"pid": other.pid, "started": "Mon Jan  1 00:00:00 2024"})
        )
        # A finished cell: process.json left behind, lock free.
        done = runs / "webshop-luna-none"
        done.mkdir()
        (done / "cell.lock").touch()
        (done / "process.json").write_text(
            json.dumps({"pid": os.getpid(), "started": "x"})
        )
        (runs / "_schedule").mkdir()
        assert live_cell_workers(runs) == [("webshop-luna-mirix", live.pid)]
        # The live worker's memory: its group's RSS from one ps call.
        rss = process_group_rss([live.pid])
        assert rss[live.pid] > 30 * 1024 * 1024
        other.kill()
        other.wait(timeout=10)
    finally:
        live.kill()
        live.wait(timeout=10)
    assert live_cell_workers(runs) == []
    assert live_cell_workers(tmp_path / "missing") == []


def test_registered_process_groups_lists_the_children_still_running(tmp_path):
    registry = tmp_path / "owned_processes.jsonl"
    assert registered_process_groups(registry) == []
    lines = [
        {"event": "started", "name": "mirix", "pid": 10, "pgid": 10},
        {"event": "started", "name": "mirix", "pid": 20, "pgid": 20},
        {"event": "stopped", "name": "mirix", "pid": 10},
        {"event": "started", "name": "mirix", "pid": 30},  # an older row: pgid = pid
    ]
    registry.write_text(
        "".join(json.dumps(r) + "\n" for r in lines) + '{"event": "clea'  # torn tail
    )
    assert registered_process_groups(registry) == [20, 30]


def test_signal_group_reports_a_group_that_is_gone():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    assert signal_group(child.pid, signal.SIGTERM)
    assert child.wait(timeout=10) == -signal.SIGTERM
    assert not signal_group(child.pid, signal.SIGTERM)


STUBBORN = f"{sys.executable} -c 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); time.sleep(60)'"


def test_stop_orphaned_terms_first_and_leaves_owned_or_used_children(tmp_path):
    """After a drain: a gateway nobody owns or uses gets SIGTERM (and time to exit), one
    that ignores it is killed by the cleanup; a child whose owner or users are alive is
    never touched."""
    runs = tmp_path / "runs"
    registry = runs / "_gateway" / "owned_processes.jsonl"
    owner_path = runs / "_gateway" / "gateway-127.0.0.1-1.lock"
    users_path = runs / "_gateway" / "gateway-127.0.0.1-1.users.lock"

    def spawn(argv: list[str], **locks: Path) -> OwnedProcess:
        return OwnedProcess.spawn(
            "gateway",
            argv,
            env={"PATH": "/usr/bin:/bin"},
            cwd=Path.cwd(),
            log_path=tmp_path / "g.log",
            registry=registry,
            **locks,
        )

    polite = spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    stubborn = spawn(["sh", "-c", STUBBORN])
    owner = OwnerLock.acquire(owner_path)
    owned = spawn(
        [sys.executable, "-c", "import time; time.sleep(60)"], owner_lock=owner_path
    )
    users = OwnerLock.acquire(users_path, shared=True)
    used = spawn(
        [sys.executable, "-c", "import time; time.sleep(60)"], users_lock=users_path
    )
    try:
        time.sleep(0.3)  # the stubborn child has installed its SIG_IGN
        termed = stop_orphaned(registry, grace_s=1.0)
        assert sorted(termed) == sorted([polite.pid, stubborn.pid])
        assert polite.popen.wait(timeout=10) == -signal.SIGTERM
        assert stubborn.popen.wait(timeout=10) == -signal.SIGKILL
        assert owned.popen.poll() is None and used.popen.poll() is None
        reasons = {pid: row["reason"] for pid, row in cleaned(registry).items()}
        assert set(reasons) == {polite.pid, stubborn.pid}
    finally:
        owner.release()
        users.release()
        for child in (polite, stubborn, owned, used):
            kill_quietly(child.pid)
            child.popen.wait(timeout=10)
