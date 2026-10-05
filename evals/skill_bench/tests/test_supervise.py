"""The supervisor (``tools/supervise.py``).

The scheduler it runs is ``tests/fake_scheduler.py`` (schedule.py's command line, a
scripted exit), passed through the ``--scheduler-module`` seam; the workers it drains are
``tests/fake_worker.py``. Its waits between restarts go through an injected hook that
records them and advances a fake monotonic clock, so a 30-minute timeout takes no time.
Signals are sent to a real supervisor process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.cell_records import (
    CellPaths,
    read_json,
    read_jsonl,
    write_json_atomic,
)
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    OwnerLock,
    lock_is_held,
    process_start_time,
)
from evals.skill_bench.infra.resources import HostResources
from evals.skill_bench.tests.run_support import (
    eventually,
    start_orphan_reaper,
    stop_test_processes,
)
from evals.skill_bench.tools import supervise
from evals.skill_bench.tools.supervise import (
    SUPERVISE_ATTENTION,
    SUPERVISE_BUDGET_INVALID,
    SUPERVISE_BUDGET_SPENT,
    SUPERVISE_COMPLETE,
    SUPERVISE_INTERNAL_ERROR,
    SUPERVISE_LOCKED,
    SUPERVISE_STOPPED,
    OneOffRequest,
    SchedulerHandle,
    SuperviseOptions,
    Supervisor,
    drop_option,
    one_off_requests,
    parse_args,
    status_report,
)

FAKE_SCHEDULER = "evals.skill_bench.tests.fake_scheduler"
FAKE_WORKER = "evals.skill_bench.tests.fake_worker"


@pytest.fixture(scope="session", autouse=True)
def reap_after_a_killed_session(tmp_path_factory):
    """However this pytest ends (a chunk timeout's SIGKILL too), what its tests left
    running under the session's basetemp is stopped."""
    start_orphan_reaper(tmp_path_factory.getbasetemp())


@pytest.fixture(autouse=True)
def no_leftover_processes(tmp_path):
    """After each test (passed, failed or timed out), the schedulers, gateways and
    workers it started and left running are stopped, and named in a warning."""
    yield
    left = stop_test_processes(tmp_path)
    if left:
        warnings.warn(f"the test left processes running, now stopped: {left}")


class Clock:
    """The supervisor's monotonic clock and wait hook: a wait returns at once, records
    its length and moves the clock by it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.waits: list[float] = []
        self.on_wait: Any = None

    def mono(self) -> float:
        return self.now

    async def pause(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds
        if self.on_wait is not None:
            self.on_wait(len(self.waits))


def reading(runs_dir: Path, workers: Any = None) -> HostResources:
    return HostResources(
        at="now", system="test", mem_available_gb=12.0, disk_free_gb=80.0
    )


def script(runs: Path, *steps: dict[str, Any]) -> None:
    write_json_atomic(runs / "_fake_scheduler.json", {"runs": list(steps)})


def scheduler_log(runs: Path, event: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(runs / "_fake_scheduler.jsonl")
    return [r for r in rows if event is None or r["event"] == event]


def events(runs: Path, name: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(runs / "_supervise" / "events.jsonl")
    return [r for r in rows if name is None or r["event"] == name]


def make(
    tmp_path: Path, *steps: dict[str, Any], args: tuple[str, ...] = (), **kw: Any
) -> tuple[Supervisor, Path, Clock]:
    runs = tmp_path / "runs"
    script(runs, *steps)
    clock = Clock()
    kw.setdefault("caffeinate", False)
    kw.setdefault("heartbeat_s", 0.05)
    kw.setdefault("drain_settle_s", 0.2)
    options = SuperviseOptions(
        runs_dir=runs,
        scheduler_args=("--runs-dir", str(runs), *args, "--auto-resume"),
        scheduler_module=FAKE_SCHEDULER,
        **kw,
    )
    sup = Supervisor(options, mono=clock.mono, pause=clock.pause, sampler=reading)
    return sup, runs, clock


async def run(sup: Supervisor) -> int:
    return await asyncio.wait_for(sup.run(install_signals=False), 120)


NOTIFY = textwrap.dedent(
    """
    import json, os, sys
    with open(sys.argv[1], "a") as fh:
        fh.write(json.dumps({"argv": sys.argv[2:], "alert": os.environ["SKILL_BENCH_ALERT"]}) + "\\n")
    """
)


# --- exits -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_and_attention_propagate(tmp_path):
    sup, runs, clock = make(tmp_path / "a", {"exit": 0})
    assert await run(sup) == SUPERVISE_COMPLETE
    assert [e["event"] for e in events(runs)][-2:] == ["complete", "exit"]
    assert not (runs / "_supervise" / "ALERT").exists() and clock.waits == []
    (start,) = scheduler_log(runs, "start")
    assert start["argv"].count("--auto-resume") == 1
    sup, runs, _ = make(
        tmp_path / "b",
        {
            "exit": 2,
            "attention": {"alfworld-luna-mirix": "learning_interrupted"},
            "attention_reasons": {"alfworld-luna-mirix": "resume_cap"},
        },
    )
    assert await run(sup) == SUPERVISE_ATTENTION
    alert = (runs / "_supervise" / "ALERT").read_text()
    assert "alfworld-luna-mirix (learning_interrupted: resume_cap)" in alert
    # The exit-time alert stays (a heartbeat that saw the cell first alerted it too,
    # cell_needs_attention).
    (row,) = alerts(runs, "attention")
    assert row["attention_reasons"] == {"alfworld-luna-mirix": "resume_cap"}
    assert {a["kind"] for a in alerts(runs)} <= {"attention", "cell_needs_attention"}


@pytest.mark.asyncio
async def test_a_spent_budget_stops_with_an_alert_and_the_notify_command(tmp_path):
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    spend = {"budget_usd": 100.0, "spent_usd": 100.4, "left_usd": -0.4}
    sup, runs, clock = make(
        tmp_path,
        {"exit": 3, "pause_reason": "budget_spent", "budget": {"spend": spend}},
        notify_cmd=f"{sys.executable} {notifier} {out} 'two words'",
    )
    assert await run(sup) == SUPERVISE_BUDGET_SPENT
    assert clock.waits == [] and len(scheduler_log(runs, "start")) == 1  # no relaunch
    (note,) = read_jsonl(out)
    assert note["argv"] == ["two words"]  # shlex, not a shell
    assert note["alert"].startswith("budget spent: a person must raise it")
    assert "--budget <new total>" in (runs / "_supervise" / "ALERT").read_text()
    (row,) = events(runs, "alert")
    assert row["kind"] == "budget_spent" and row["budget"] == spend
    beat = read_json(runs / "_supervise" / "heartbeat.json")
    assert beat["final"] and beat["exit_code"] == SUPERVISE_BUDGET_SPENT
    assert beat["alerts"][0]["kind"] == "budget_spent"


@pytest.mark.asyncio
async def test_an_invalid_budget_stops_and_a_failing_notify_is_recorded(tmp_path):
    sup, runs, _ = make(
        tmp_path,
        {"exit": 3, "pause_reason": "budget_invalid"},
        notify_cmd="/nonexistent/notifier",
    )
    assert await run(sup) == SUPERVISE_BUDGET_INVALID
    assert events(runs, "alert")[0]["kind"] == "budget_invalid"
    assert events(runs, "notify_failed")[0]["error"].startswith("FileNotFoundError")


@pytest.mark.asyncio
async def test_a_pause_the_scheduler_exited_on_is_restarted_after_a_wait(tmp_path):
    sup, runs, clock = make(
        tmp_path, {"exit": 3, "pause_reason": "gateway_unreachable"}, {"exit": 0}
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert clock.waits == [300.0]
    assert len(scheduler_log(runs, "start")) == 2
    (row,) = events(runs, "restart_after_pause")
    assert row["reason"] == "gateway_unreachable"
    assert (
        read_json(runs / "_supervise" / "heartbeat.json")["scheduler"]["restarts"] == 1
    )


@pytest.mark.asyncio
async def test_a_refused_scheduler_is_retried_and_alerted_once(tmp_path):
    """A scheduler refused for --refused-timeout gets one alert, and the retries go on
    (spaced by --restart-backoff-cap once alerted); the supervisor ends with the run
    that finally gets through."""
    sup, runs, clock = make(
        tmp_path,
        {"exit": 0},
        refused_retry_s=600.0,
        refused_timeout_s=1800.0,
        restart_backoff_cap_s=900.0,
        stall_alert_s=1e9,
    )
    lock = OwnerLock.acquire(runs / "_schedule" / "scheduler.lock")
    # Refused for three times the timeout, then the other scheduler is gone.
    clock.on_wait = lambda n: lock.release() if clock.now - 1000.0 >= 5400.0 else None
    try:
        assert await run(sup) == SUPERVISE_COMPLETE  # no exit at 1800 s
    finally:
        lock.release()
    assert clock.waits == [600.0, 600.0, 600.0, 900.0, 900.0, 900.0, 900.0]
    assert len(scheduler_log(runs, "refused")) == 7
    (row,) = events(runs, "alert")
    assert row["kind"] == "refused" and row["retry_s"] == 900.0
    assert "another scheduler is running" in row["text"]
    assert "still retrying every 900 s" in row["text"]
    assert events(runs, "scheduler_refused")[0]["error"].startswith("another scheduler")
    assert [e["retry_s"] for e in events(runs, "scheduler_refused")] == clock.waits


@pytest.mark.asyncio
async def test_a_refusal_that_clears_lets_the_scheduler_run(tmp_path):
    sup, runs, clock = make(tmp_path, {"exit": 0})
    lock = OwnerLock.acquire(runs / "_schedule" / "scheduler.lock")
    clock.on_wait = lambda n: lock.release() if n == 2 else None
    try:
        assert await run(sup) == SUPERVISE_COMPLETE
    finally:
        lock.release()
    assert clock.waits == [60.0, 60.0]
    assert [r["event"] for r in scheduler_log(runs)][-3:] == ["refused", "start", "end"]


@pytest.mark.asyncio
async def test_crashes_restart_with_backoff_and_a_crash_loop_only_alerts(tmp_path):
    """More than --max-crash-restarts crashes in a streak is one crash_loop alert; the
    supervisor goes on restarting (the backoff capped), and ends with the run that
    finally succeeds."""
    steps: list[dict[str, Any]] = [{"exit": 1}, {"kill_self": True}]
    steps += [{"exit": 1}] * 23 + [{"exit": 0}]
    sup, runs, clock = make(
        tmp_path, *steps, restart_backoff_cap_s=50.0, stall_alert_s=1e9
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert clock.waits == [10.0, 20.0, 40.0] + [50.0] * 22
    returncodes = [e["returncode"] for e in events(runs, "scheduler_crashed")]
    assert returncodes[:3] == [1, -9, 1] and len(returncodes) == 25
    (row,) = events(runs, "alert")
    assert row["kind"] == "crash_loop" and "21 times in a row" in row["text"]
    assert "still restarting every <= 50 s" in row["text"]
    assert len(scheduler_log(runs, "start")) == 26
    assert events(runs, "drain") == []


@pytest.mark.asyncio
async def test_crashes_far_apart_are_not_a_loop(tmp_path):
    # Every run lasted at least the crash window (0 here): each crash starts a new count.
    sup, runs, clock = make(
        tmp_path,
        {"exit": 1},
        {"exit": 1},
        {"exit": 1},
        {"exit": 1},
        {"exit": 0},
        max_crash_restarts=1,
        crash_window_s=0.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert clock.waits == [10.0] * 4
    assert (
        read_json(runs / "_supervise" / "heartbeat.json")["scheduler"]["crash_restarts"]
        == 4
    )


@pytest.mark.asyncio
async def test_a_pause_exit_in_between_does_not_end_a_crash_count(tmp_path):
    """A pause exit that was no useful run (short, no cell completed) does not start
    the crash count over: the third crash is the loop (one alert; the restarts go
    on)."""
    sup, runs, clock = make(
        tmp_path,
        {"exit": 1},
        {"exit": 1},
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 1},
        {"exit": 0},
        max_crash_restarts=2,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert clock.waits == [10.0, 20.0, 300.0, 40.0]
    (row,) = events(runs, "alert")
    assert row["kind"] == "crash_loop" and "3 times in a row" in row["text"]


@pytest.mark.asyncio
async def test_an_unknown_budget_is_alerted_periodically_and_never_stops(tmp_path):
    """A budget or balance that stays unknown is alerted at its first grace and then
    every grace; the scheduler is never signalled (nothing is sent while it lasts, and
    it resumes by itself once the budget is read)."""
    sup, runs, _ = make(tmp_path, budget_unknown_grace_s=100.0)
    sup.handle = SchedulerHandle(4242, None)
    spend = {"budget_usd": 100.0, "error": "usage unreadable"}
    unknown = {
        "pid": 4242,
        "pause_reason": "budget_unknown",
        "budget": {"spend": spend},
    }
    for mono in (0.0, 50.0, 100.0, 150.0, 199.0, 200.0, 250.0, 300.0, 350.0):
        await sup._watch_unknown_pause(unknown, mono)
    got = alerts(runs, "budget_unknown")
    assert [a["seconds"] for a in got] == [100.0, 200.0, 300.0]  # 3.5 graces: 3 alerts
    assert got[0]["reason"] == "budget_unknown" and got[0]["budget"] == spend
    assert "nothing is sent while it lasts" in got[0]["text"]
    assert "resumes by itself" in got[0]["text"]
    # The pause lifts: the timer starts over with the next unknown pause.
    await sup._watch_unknown_pause({"pid": 4242, "pause_reason": None}, 400.0)
    other = {"pid": 4242, "pause_reason": "balance_unknown"}
    for mono in (410.0, 500.0, 510.0):
        await sup._watch_unknown_pause(other, mono)
    assert [a["seconds"] for a in alerts(runs, "budget_unknown")][3:] == [100.0]
    # Another scheduler's state never counts.
    await sup._watch_unknown_pause({**other, "pid": 1}, 5000.0)
    assert len(alerts(runs, "budget_unknown")) == 4
    # End to end: the scheduler is never signalled and ends on its own.
    sup, runs, _ = make(
        tmp_path / "e2e",
        {"pause_reason": "budget_unknown", "sleep_s": 1.0, "exit": 0},
        budget_unknown_grace_s=0.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE  # never SIGTERM, never exit 11
    assert scheduler_log(runs, "signal") == []
    assert alerts(runs, "budget_unknown")
    assert events(runs, "unknown_pause_grace_exceeded") == []


@pytest.mark.asyncio
async def test_a_restarted_supervisor_adopts_the_live_scheduler(tmp_path):
    runs = tmp_path / "runs"
    release = tmp_path / "release"
    script(runs, {"until": str(release), "exit": 0})
    live = subprocess.Popen(
        [sys.executable, "-m", FAKE_SCHEDULER, "--runs-dir", str(runs)],
        cwd=protocol.REPO_ROOT,
        start_new_session=True,
    )
    try:
        state = runs / "_schedule" / "state.json"
        await eventually(
            lambda: state.is_file() and read_json(state).get("pid") == live.pid, 30
        )
        sup, _, _ = make(tmp_path)
        script(runs, {"exit": 1})  # a spawned one would crash: it must not be spawned
        task = asyncio.ensure_future(run(sup))
        await eventually(lambda: bool(events(runs, "scheduler_adopted")), 30)
        assert events(runs, "scheduler_adopted")[0]["scheduler"] == live.pid
        # This test is its parent: reap it (a dead supervisor's init does), or its zombie
        # would keep its pid and start time.
        reaper = asyncio.ensure_future(asyncio.to_thread(live.wait, 60))
        release.touch()
        assert await task == SUPERVISE_COMPLETE
        assert await reaper == 0
    finally:
        release.touch()
        live.wait(timeout=30)
    assert len(scheduler_log(runs, "start")) == 1


# --- heartbeat, status, sleep, caffeinate, locks -------------------------------------------------


@pytest.mark.asyncio
async def test_the_heartbeat_and_status(tmp_path):
    release = tmp_path / "release"
    sup, runs, _ = make(tmp_path, {"until": str(release), "exit": 0})
    task = asyncio.ensure_future(run(sup))
    beat_path = runs / "_supervise" / "heartbeat.json"
    try:
        await eventually(
            lambda: beat_path.is_file()
            and read_json(beat_path)["scheduler"]["pid"] is not None
            and read_json(beat_path)["state_at"] is not None,
            30,
        )
        beat = read_json(beat_path)
        assert beat["pid"] == os.getpid() and beat["host"]["hostname"]
        assert beat["resources"]["mem_available_gb"] == 12.0
        assert beat["counts"] == {
            "complete": 0,
            "running": 0,
            "queued": 0,
            "waiting": 0,
            "needs_attention": 0,
            "blocked": 0,
        }
        text, code = status_report(runs)
        assert code == 0, text
        assert "(fresh)" in text and "scheduler: pid" in text and "(alive)" in text
        assert "resources: 12.0 GB available" in text
    finally:
        release.touch()
    assert await task == SUPERVISE_COMPLETE
    text, code = status_report(runs)
    assert code == supervise.STATUS_STALE and "exited 0" in text
    missing, code = status_report(tmp_path / "nowhere")
    assert code == supervise.STATUS_STALE and "no heartbeat" in missing


def test_status_shows_waiting_cells_budget_and_the_alert(tmp_path, capsys):
    runs = tmp_path / "runs"
    at = time.time()
    write_json_atomic(
        runs / "_schedule" / "state.json",
        {
            "complete": ["a"],
            "running": ["b"],
            "queued": [],
            "waiting": {"c": supervise.schedule.iso(at + 120)},
            "needs_attention": {"d": "learning_failed"},
            "attention_reasons": {"d": "harness_failure"},
            "blocked": {},
            "pause_reason": "balance_below_floor",
            "budget": {
                "remaining_usd": 4.5,
                "spend": {"budget_usd": 100.0, "spent_usd": 10.0, "left_usd": 90.0},
            },
        },
    )
    (runs / "_supervise").mkdir(parents=True)
    (runs / "_supervise" / "ALERT").write_text("t0 cells need a person\n")
    text, code = status_report(runs, wall=at)
    assert code == supervise.STATUS_STALE  # no heartbeat
    assert "cells: complete 1, running 1, queued 0, waiting 1" in text
    assert "next wake-ups: c in 120 s" in text
    assert "d (learning_failed: harness_failure)" in text
    assert "pause: balance_below_floor; balance $4.5, spent $10.0 of $100.0" in text
    assert "ALERT: t0 cells need a person" in text
    assert (
        supervise.main(["--status", "--runs-dir", str(runs)]) == supervise.STATUS_STALE
    )
    assert "cells: complete 1" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_host_sleep_is_noted(tmp_path):
    walls = iter([1000.0, 1130.0, 1140.0])
    monos = iter([50.0, 60.0, 70.0])
    sup = Supervisor(
        SuperviseOptions(runs_dir=tmp_path / "runs", caffeinate=False),
        wall=lambda: next(walls),
        mono=lambda: next(monos),
        sampler=reading,
    )
    for _ in range(3):
        await sup.tick()
    (slept,) = events(tmp_path / "runs", "host_slept")
    assert slept["seconds"] == 120.0  # the wall clock ran 130 s, the monotonic one 10 s
    assert (tmp_path / "runs" / "_supervise" / "heartbeat.json").is_file()


class FakeCaffeinate:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    async def wait(self) -> int:
        return self.returncode or 0


@pytest.mark.asyncio
async def test_caffeinate_keeps_macos_awake_and_is_respawned(tmp_path):
    spawned: list[list[str]] = []

    async def spawn(argv: Any) -> FakeCaffeinate:
        spawned.append(list(argv))
        return FakeCaffeinate(4000 + len(spawned))

    def build(
        system: str, *, which: Any = lambda _n: "/usr/bin/caffeinate", **kw: Any
    ) -> Supervisor:
        return Supervisor(
            SuperviseOptions(runs_dir=tmp_path / system, **kw),
            system=system,
            which=which,
            spawn_caffeinate=spawn,
            sampler=reading,
        )

    mac = build("darwin")
    await mac.start_caffeinate()
    assert spawned == [["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())]]
    first = mac.caffeinate
    first.returncode = 0  # it died
    await mac.tick()
    assert len(spawned) == 2 and mac.caffeinate.pid == 4002
    assert events(tmp_path / "darwin", "caffeinate_respawned")[0]["returncode"] == 0
    beat = read_json(tmp_path / "darwin" / "_supervise" / "heartbeat.json")
    assert beat["caffeinate_pid"] == 4002
    current = mac.caffeinate
    await mac.stop_caffeinate()
    assert current.terminated and mac.caffeinate is None
    for other in (
        build("linux"),
        build("darwin", caffeinate=False),
        build("darwin", which=lambda _n: None),
    ):
        await other.start_caffeinate()
    assert len(spawned) == 2  # none of them spawned it


@pytest.mark.asyncio
async def test_one_supervisor_per_runs_directory_and_no_merged_view(tmp_path):
    sup, runs, _ = make(tmp_path, {"exit": 0})
    lock = OwnerLock.acquire(runs / "_supervise" / "supervisor.lock")
    try:
        assert await run(sup) == SUPERVISE_LOCKED
    finally:
        lock.release()
    assert scheduler_log(runs) == []
    (runs / "_merged.json").write_text("{}")
    sup, runs, _ = make(tmp_path, {"exit": 0})
    assert await run(sup) == SUPERVISE_LOCKED
    assert events(runs, "alert")[0]["kind"] == "merged" and scheduler_log(runs) == []


@pytest.mark.asyncio
async def test_a_stale_alert_is_cleared_at_start(tmp_path):
    sup, runs, _ = make(tmp_path, {"exit": 0})
    (runs / "_supervise").mkdir(parents=True)
    (runs / "_supervise" / "ALERT").write_text("old: budget spent\n")
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not (runs / "_supervise" / "ALERT").exists()
    assert events(runs, "alert_cleared")[0]["text"] == "old: budget spent"


# --- arguments ----------------------------------------------------------------------------------


def test_a_study_reaches_the_scheduler_and_is_checked_first(tmp_path):
    """``--study`` is the scheduler's; the supervisor checks it before it
    starts anything (a refused selection would exit 2, which reads as "cells need a
    person") and passes it on with the replicate's options."""
    runs = str(tmp_path / "phase16-r2")
    options, _ = parse_args(
        [
            "--runs-dir",
            runs,
            "--study",
            "phase16",
            "--gateway-port",
            "18542",
            "--budget",
            "100",
            "--budget-start-from",
            str(tmp_path / "phase16-r1"),
            "--host-benchmark-concurrency",
            "webshop=6",
        ]
    )
    assert options.runs_dir == Path(runs)
    assert options.scheduler_args[:2] == ("--runs-dir", runs)
    assert "--study" in options.scheduler_args and "phase16" in options.scheduler_args
    assert options.scheduler_args[-1] == "--auto-resume"
    with pytest.raises(SystemExit):
        parse_args(["--runs-dir", runs, "--study", "phase16", "--split-mode", "smoke"])


def test_the_scheduler_gets_every_other_option_and_auto_resume_once(tmp_path):
    runs = str(tmp_path / "runs")
    options, args = parse_args(
        [
            "--runs-dir",
            runs,
            "--benchmark",
            "alfworld",
            "--concurrency",
            "3",
            "--notify-cmd",
            "say 'x y'",
            "--heartbeat",
            "5",
            "--no-caffeinate",
            "--min-free-memory-gb",
            "4",
        ]
    )
    assert options.scheduler_args == (
        "--runs-dir",
        runs,
        "--benchmark",
        "alfworld",
        "--concurrency",
        "3",
        "--min-free-memory-gb",
        "4",
        "--auto-resume",
    )
    assert options.runs_dir == Path(runs) and options.heartbeat_s == 5.0
    assert not options.caffeinate and options.drain_on_stop
    assert options.notify_cmd == "say 'x y'" and not args.status
    again, _ = parse_args(["--runs-dir", runs, "--auto-resume"])
    assert again.scheduler_args.count("--auto-resume") == 1
    for bad in (
        ["--dry-run"],
        ["--benchmark", "webarena"],
        ["--concurrency", "25"],
        ["--benchmark", "alfworld", "--reset-resume", "webshop-luna-none"],
        ["--notify-cmd", "say 'unclosed"],
        # Abbreviated one-off options would escape one_off.json.
        ["--reset"],
        ["--rerun-harness", "alfworld-luna-none"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["--runs-dir", runs, *bad])


@pytest.mark.parametrize(
    "extra",
    [
        ["--method", "none", "{m}"],
        ["--rerun-harness-failures", "alfworld-luna56-{m}"],
        ["--reset-resume", "alfworld-luna-{m}"],
    ],
)
def test_the_supervisor_refuses_a_removed_method_before_it_starts(
    tmp_path, capsys, extra, removed_method
):
    """A command line naming a method the study removed (a method, or one of its cells
    left on disk) exits 2 naming the decision, under this command's name, before the
    supervisor's lock or any scheduler exists. Such a command line is most likely a
    systemd unit's written before the removal, which does not restart on 2: the
    refusal is an alert (``ALERT``, which ``--status`` prints, and ``events.jsonl``),
    not only a line on stderr."""
    extra = [a.format(m=removed_method) for a in extra]
    runs = tmp_path / "runs"
    (runs / f"alfworld-luna56-{removed_method}").mkdir(parents=True)
    (runs / f"alfworld-luna56-{removed_method}" / "status.json").write_text(
        '{"status": "interrupted"}', encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        parse_args(["--runs-dir", str(runs), *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert removed_method in err and "example removal" in err
    assert "invalid choice" not in err and "outside the selection" not in err
    assert "python -m evals.skill_bench.tools.supervise: error:" in err
    assert "evals.skill_bench.schedule" not in err
    alert = (runs / "_supervise" / "ALERT").read_text(encoding="utf-8")
    assert f"the supervisor of {runs} refused to start: " in alert
    assert "example removal" in alert and "runs unsupervised" in alert
    (row,) = events(runs, "alert")
    assert row["kind"] == "refused_start" and row["text"] in alert
    assert sorted(p.name for p in (runs / "_supervise").iterdir()) == [
        "ALERT",
        "events.jsonl",
    ]
    assert not (runs / "_schedule").exists()
    assert "ALERT: " in supervise.status_report(runs)[0]
    # Without a method the scheduler gets the study's methods (a bare --method too).
    for method in ([], ["--method"]):
        options, _ = parse_args(["--runs-dir", str(runs), *method])
        assert removed_method not in " ".join(options.scheduler_args)


@pytest.mark.parametrize("mode", ["--status", "--reap-gateway"])
def test_status_and_reap_on_a_stale_command_line_raise_no_alert(
    tmp_path, mode, removed_method
):
    """``--status`` and ``--reap-gateway`` appended to a command line naming the
    removed method (or one of its cells) are refused (exit 2) without an alert: nothing
    runs the notifier, and no ``_supervise`` directory appears."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    runs = tmp_path / "runs"
    runs.mkdir()
    notify = f"{sys.executable} {notifier} {out}"
    for stale in (
        ["--method", removed_method],
        ["--reset-resume", f"alfworld-luna56-{removed_method}"],
    ):
        with pytest.raises(SystemExit) as exc:
            parse_args([mode, "--runs-dir", str(runs), "--notify-cmd", notify, *stale])
        assert exc.value.code == 2, stale
    assert not out.exists()
    assert not (runs / "_supervise").exists()


def test_a_refused_start_beside_a_live_supervisor_keeps_its_alert(
    tmp_path, removed_method
):
    """While a live supervisor holds the runs directory, a refused stale command
    line is only an ``events.jsonl`` row: the notifier does not run and the live
    supervisor's ``ALERT`` stays the one ``--status`` prints."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    runs = tmp_path / "runs"
    control = runs / "_supervise"
    control.mkdir(parents=True)
    (control / "ALERT").write_text("t0 budget spent\n", encoding="utf-8")
    lock = OwnerLock.acquire(control / "supervisor.lock", wait_s=0.0)
    try:
        with pytest.raises(SystemExit) as exc:
            parse_args(
                [
                    *("--runs-dir", str(runs)),
                    *("--notify-cmd", f"{sys.executable} {notifier} {out}"),
                    *("--method", removed_method),
                ]
            )
    finally:
        lock.release()
    assert exc.value.code == 2
    assert not out.exists()
    assert (control / "ALERT").read_text(encoding="utf-8") == "t0 budget spent\n"
    (row,) = events(runs, "alert")
    assert row["kind"] == "refused_start" and row["live_supervisor"] is True
    assert "a live supervisor runs this directory" in row["text"]
    assert "runs unsupervised" not in row["text"]


def test_a_refused_start_runs_the_notify_command_and_creates_nothing_else(
    tmp_path, capsys, removed_method
):
    """The refused start's alert reaches ``--notify-cmd`` first (shlex-split, the text
    in SKILL_BENCH_ALERT), as every alert does; a later occurrence of ``--method`` is
    read too. A runs directory that does not exist, or a merged view, gets nothing (a
    refused command line creates no directory), and a refusal of anything else stays a
    parse error without an alert, under this command's name."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    runs = tmp_path / "runs"
    runs.mkdir()
    notify = f"{sys.executable} {notifier} {out} 'two words'"
    argv = ["--runs-dir", str(runs), "--notify-cmd", notify]
    with pytest.raises(SystemExit) as exc:
        parse_args([*argv, "--method", "none", f"--method={removed_method}"])
    assert exc.value.code == 2
    (note,) = read_jsonl(out)
    assert note["argv"] == ["two words"]
    assert note["alert"].startswith(f"the supervisor of {runs} refused to start: ")
    assert "example removal" in note["alert"]
    assert note["alert"] in (runs / "_supervise" / "ALERT").read_text(encoding="utf-8")
    # A notifier that cannot run leaves the refusal and its records as they are.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(SystemExit):
        parse_args(
            [
                *("--runs-dir", str(fresh), "--notify-cmd", "/nonexistent/notifier"),
                *("--method", removed_method),
            ]
        )
    assert (fresh / "_supervise" / "ALERT").is_file()
    missing = tmp_path / "missing"
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / supervise.MERGED_MARKER).write_text("{}", encoding="utf-8")
    for where in (missing, merged):
        with pytest.raises(SystemExit):
            parse_args(["--runs-dir", str(where), "--method", removed_method])
        assert not (where / "_supervise").exists(), where
    assert not missing.exists()
    capsys.readouterr()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit) as exc:
        parse_args(["--runs-dir", str(other), "--benchmark", "webarena"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    prog = "python -m evals.skill_bench.tools.supervise"
    assert f"{prog}: error: argument --benchmark" in err
    assert not (other / "_supervise").exists()


def test_drop_option_removes_an_option_and_its_values():
    argv = [
        "--runs-dir",
        "r",
        "--reset-resume",
        "a",
        "b",
        "--auto-resume",
        "--reset-resume",
    ]
    assert drop_option(argv, ("--reset-resume",)) == [
        "--runs-dir",
        "r",
        "--auto-resume",
    ]


@pytest.mark.asyncio
async def test_reset_resume_goes_to_the_first_scheduler_that_runs_only(tmp_path):
    sup, runs, clock = make(
        tmp_path,
        {},  # the refused start's (it ends before its step)
        {"exit": 1},
        {"exit": 0},
        args=("--reset-resume", "alfworld-luna-none"),
    )
    # The first start is refused (another scheduler holds the lock): it applied nothing.
    lock = OwnerLock.acquire(runs / "_schedule" / "scheduler.lock")
    clock.on_wait = lambda n: lock.release() if n == 1 else None
    try:
        assert await run(sup) == SUPERVISE_COMPLETE
    finally:
        lock.release()
    refused, first, second = scheduler_log(runs, "start")
    for start in (refused, first):
        assert (
            "--reset-resume" in start["argv"] and "alfworld-luna-none" in start["argv"]
        )
    assert "--reset-resume" not in second["argv"]
    assert "alfworld-luna-none" not in second["argv"]
    assert second["argv"].count("--auto-resume") == 1


# --- signals (a real supervisor process) ----------------------------------------------------------


def start_worker(runs: Path, cell: str, **behavior: Any) -> subprocess.Popen[bytes]:
    write_json_atomic(runs / cell / "behavior.json", behavior)
    worker = subprocess.Popen(
        [sys.executable, "-m", FAKE_WORKER, "--out", str(runs / cell)],
        cwd=protocol.REPO_ROOT,
        start_new_session=True,
    )
    return worker


def start_supervisor(runs: Path, *extra: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "evals.skill_bench.tools.supervise",
            "--runs-dir",
            str(runs),
            "--scheduler-module",
            FAKE_SCHEDULER,
            "--no-caffeinate",
            "--heartbeat",
            "0.1",
            "--drain-settle",
            "0.5",
            *extra,
        ],
        cwd=protocol.REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


@pytest.mark.asyncio
async def test_sigterm_stops_the_scheduler_and_drains_every_worker(tmp_path):
    runs = tmp_path / "runs"
    script(runs, {"sleep_s": 60})
    cells = ["alfworld-luna-mirix", "alfworld-luna56-awm"]
    workers = [
        start_worker(runs, c, until=str(tmp_path / "never"), until_timeout_s=60)
        for c in cells
    ]
    supervisor = None
    try:
        for cell in cells:
            await eventually(lambda c=cell: lock_is_held(runs / c / "cell.lock"), 30)
        supervisor = start_supervisor(runs, "--drain-timeout", "30")
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        supervisor.send_signal(signal.SIGTERM)
        assert await asyncio.to_thread(supervisor.wait, 60) == SUPERVISE_STOPPED
        assert [w.wait(timeout=30) for w in workers] == [130, 130]
    finally:
        for process in (*workers, supervisor):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert scheduler_log(runs, "signal") and scheduler_log(runs, "end")[0]["code"] == 3
    for cell in cells:
        assert read_json(CellPaths(runs / cell).status)["status"] == "interrupted"
    (drain,) = events(runs, "drain")
    assert sorted(drain["workers"]) == sorted(cells)
    assert events(runs, "stopped")[0]["drained"] is True
    assert events(runs, "exit")[-1]["code"] == SUPERVISE_STOPPED


@pytest.mark.asyncio
async def test_a_second_signal_kills_the_scheduler_and_the_workers(tmp_path):
    runs = tmp_path / "runs"
    script(runs, {"sleep_s": 60, "ignore_sigterm": True})
    worker = start_worker(
        runs, "alfworld-luna56-mirix", ignore_sigterm=True, sleep_s=60
    )
    supervisor = None
    try:
        await eventually(
            lambda: lock_is_held(runs / "alfworld-luna56-mirix" / "cell.lock"), 30
        )
        supervisor = start_supervisor(runs, "--scheduler-stop-timeout", "60")
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        supervisor.send_signal(signal.SIGTERM)
        await eventually(lambda: bool(scheduler_log(runs, "signal")), 30)
        supervisor.send_signal(signal.SIGTERM)
        assert await asyncio.to_thread(supervisor.wait, 60) == SUPERVISE_STOPPED
        assert worker.wait(timeout=30) == -signal.SIGKILL
    finally:
        for process in (worker, supervisor):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert scheduler_log(runs, "end") == []  # killed, never ended on its own
    (killed,) = events(runs, "killed")
    assert list(killed["workers"]) == ["alfworld-luna56-mirix"]
    assert not lock_is_held(runs / "alfworld-luna56-mirix" / "cell.lock")


@pytest.mark.asyncio
async def test_without_drain_the_workers_go_on(tmp_path):
    runs = tmp_path / "runs"
    release = tmp_path / "release"
    script(runs, {"sleep_s": 60})
    worker = start_worker(runs, "alfworld-luna-none", until=str(release))
    supervisor = None
    try:
        await eventually(
            lambda: lock_is_held(runs / "alfworld-luna-none" / "cell.lock"), 30
        )
        supervisor = start_supervisor(runs, "--no-drain-on-stop")
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        supervisor.send_signal(signal.SIGINT)
        assert await asyncio.to_thread(supervisor.wait, 60) == SUPERVISE_STOPPED
        assert worker.poll() is None  # still running
        release.touch()
        assert worker.wait(timeout=30) == 0
    finally:
        release.touch()
        for process in (worker, supervisor):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert events(runs, "stopped")[0]["drained"] is False


# --- one-off options, stalls, full disks, notifier environment, drains -------------------


@pytest.mark.asyncio
async def test_one_off_options_go_again_to_a_scheduler_that_never_applied_them(
    tmp_path,
):
    """--reset-resume and --rerun-harness-failures go to the
    schedulers this supervisor starts until one says (state.json one_off_applied) that it
    applied them; a scheduler that crashed before that gets them again."""
    sup, runs, _clock = make(
        tmp_path,
        {"crash_before_one_off": 1},  # e.g. ensure_gateway failed
        {"exit": 1},  # applied them, then crashed
        {"exit": 0},
        args=(
            "--reset-resume",
            "alfworld-luna-none",
            "--rerun-harness-failures",
            "alfworld-luna-mirix",
        ),
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    crashed, applied, later = scheduler_log(runs, "start")
    for start in (crashed, applied):
        assert "--reset-resume" in start["argv"]
        assert "--rerun-harness-failures" in start["argv"]
        assert "alfworld-luna-mirix" in start["argv"]
    assert "--reset-resume" not in later["argv"]
    assert "--rerun-harness-failures" not in later["argv"]
    assert "alfworld-luna-mirix" not in later["argv"]
    assert later["argv"].count("--auto-resume") == 1


def test_rerun_harness_failures_names_cells_of_the_selection(tmp_path, capsys):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--runs-dir",
                str(tmp_path),
                "--benchmark",
                "webshop",
                "--rerun-harness-failures",
                "alfworld-luna56-mirix",
            ]
        )
    # A named cell that never ran under --runs-dir (no status.json) is refused: the
    # scheduler would skip it silently and start it as a first attempt (e.g. a rerun
    # command without --runs-dir).
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--runs-dir",
                str(tmp_path),
                "--benchmark",
                "webshop",
                "--rerun-harness-failures",
                "webshop-luna56-mirix",
            ]
        )
    assert "never ran under" in capsys.readouterr().err
    (tmp_path / "webshop-luna56-mirix").mkdir()
    (tmp_path / "webshop-luna56-mirix" / "status.json").write_text(
        "{}", encoding="utf-8"
    )
    options, _ = parse_args(
        [
            "--runs-dir",
            str(tmp_path),
            "--benchmark",
            "webshop",
            "--rerun-harness-failures",
            "webshop-luna56-mirix",
            "--min-free-disk-gb",
            "10",
        ]
    )
    assert options.min_free_disk_gb == 10.0


@pytest.mark.asyncio
async def test_an_unknown_budget_pause_exit_is_restarted_like_any_pause(tmp_path):
    """A scheduler that exits paused on an unknown budget is restarted after the pause
    wait like any pause it does not wait out; an unknown budget never ends the
    supervisor."""
    sup, runs, clock = make(
        tmp_path,
        {"exit": 3, "pause_reason": "budget_unknown"},
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 0},
        budget_unknown_grace_s=0.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert len(scheduler_log(runs, "start")) == 3
    assert [e["reason"] for e in events(runs, "restart_after_pause")] == [
        "budget_unknown",
        "gateway_unreachable",
    ]
    assert clock.waits == [300.0, 300.0]
    assert scheduler_log(runs, "signal") == []


@pytest.mark.asyncio
async def test_a_waitable_pause_that_lasts_raises_an_alert_and_waiting_goes_on(
    tmp_path,
):
    """A pause the scheduler waits out (a key limit that does not reset)
    alerts after --stall-alert; the supervisor does not stop anything."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    sup, runs, _ = make(
        tmp_path,
        {"exit": 0, "pause_reason": "balance_below_floor", "sleep_s": 1.0},
        stall_alert_s=0.0,
        notify_cmd=f"{sys.executable} {notifier} {out}",
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    (row,) = events(runs, "alert")
    assert row["kind"] == "stalled_pause"
    assert "balance_below_floor" in row["text"] and "still waiting" in row["text"]
    assert [n["alert"] for n in read_jsonl(out)] == [row["text"]]
    assert not scheduler_log(runs, "signal")  # nothing was stopped


@pytest.mark.asyncio
async def test_a_resource_wait_with_nothing_running_raises_an_alert(tmp_path):
    held = {
        "cell": "alfworld-luna-mirix",
        "gate": "memory",
        "reason": "memory: 5.18 GB available - 1.4 GB ...",
        "since": "2026-09-29T00:00:00+00:00",
    }
    sup, runs, _ = make(
        tmp_path,
        {"exit": 0, "resource_wait": held, "sleep_s": 1.0},
        stall_alert_s=0.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    (row,) = events(runs, "alert")
    assert row["kind"] == "stalled_resources"
    assert "alfworld-luna-mirix" in row["text"] and "nothing runs" in row["text"]
    # With cells running, a held cell is no stall.
    sup, runs, _ = make(
        tmp_path / "busy",
        {"exit": 0, "resource_wait": held, "running": ["x"], "sleep_s": 1.0},
        stall_alert_s=0.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert events(runs, "alert") == []


@pytest.mark.asyncio
async def test_restarts_on_a_pause_that_go_on_raise_an_alert(tmp_path):
    sup, runs, clock = make(
        tmp_path,
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 0},
        stall_alert_s=600.0,
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert clock.waits == [300.0, 300.0, 300.0]
    (row,) = events(runs, "alert")
    assert row["kind"] == "stalled_restarts" and "still restarting" in row["text"]


@pytest.mark.asyncio
async def test_a_full_disk_never_takes_the_supervisor_down(tmp_path, monkeypatch):
    """With events.jsonl, ALERT and heartbeat.json unwritable (ENOSPC), the
    supervisor goes on, and the notify command still runs, first."""
    import errno

    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    sup, runs, _ = make(
        tmp_path,
        {"exit": 1},
        {"exit": 0},
        max_crash_restarts=0,
        notify_cmd=f"{sys.executable} {notifier} {out}",
    )

    def full(*_a: Any, **_kw: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(supervise, "append_jsonl", full)
    monkeypatch.setattr(supervise, "write_json_atomic", full)
    monkeypatch.setattr(supervise, "_write_text_atomic", full)
    assert await run(sup) == SUPERVISE_COMPLETE
    (note,) = read_jsonl(out)
    assert note["alert"].startswith("the scheduler crashed")


@pytest.mark.asyncio
async def test_a_disk_below_the_floor_while_cells_run_raises_an_alert(tmp_path):
    def low_disk(runs_dir: Path, workers: Any = None) -> HostResources:
        return HostResources(
            at="now", system="test", mem_available_gb=12.0, disk_free_gb=4.0
        )

    sup, runs, _ = make(
        tmp_path,
        {"exit": 0, "running": ["alfworld-luna-mirix"], "sleep_s": 1.0},
        min_free_disk_gb=10.0,
    )
    sup.sampler = low_disk
    assert await run(sup) == SUPERVISE_COMPLETE
    (row,) = events(runs, "alert")  # once per episode
    assert row["kind"] == "disk_low" and "4.0 GB free" in row["text"]


@pytest.mark.asyncio
async def test_the_notify_command_never_sees_a_provider_key(tmp_path, monkeypatch):
    """The notifier runs in the workers' environment."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-reach-the-notifier")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-the-notifier")
    out = tmp_path / "env.json"
    notifier = tmp_path / "notify.py"
    notifier.write_text(
        "import json, os, sys\njson.dump(sorted(os.environ), open(sys.argv[1], 'w'))\n"
    )
    sup, runs, _ = make(
        tmp_path,
        {"exit": 3, "pause_reason": "budget_invalid"},
        notify_cmd=f"{sys.executable} {notifier} {out}",
    )
    assert await run(sup) == SUPERVISE_BUDGET_INVALID
    seen = set(read_json(out))
    assert "SKILL_BENCH_ALERT" in seen and "PATH" in seen
    assert not {"OPENROUTER_API_KEY", "AWS_SECRET_ACCESS_KEY"} & seen


@pytest.mark.asyncio
async def test_a_worker_that_starts_during_the_drain_is_signalled_too(tmp_path):
    """The drain scans for workers until none is left, so one that took its cell after
    the first scan (a scheduler launching as it was stopped) gets SIGTERM too; a drain
    that times out leaves a reaper for the gateway."""
    runs = tmp_path / "runs"
    script(runs, {"sleep_s": 60})
    stubborn = start_worker(
        runs, "alfworld-luna-mirix", ignore_sigterm=True, sleep_s=60
    )
    late = None
    supervisor = None
    try:
        await eventually(
            lambda: lock_is_held(runs / "alfworld-luna-mirix" / "cell.lock"), 30
        )
        supervisor = start_supervisor(runs, "--drain-timeout", "4")
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        supervisor.send_signal(signal.SIGTERM)
        await eventually(lambda: bool(events(runs, "drain")), 30)
        late = start_worker(
            runs,
            "alfworld-luna56-awm",
            until=str(tmp_path / "never"),
            until_timeout_s=60,
        )
        assert late.wait(timeout=30) == 130  # signalled by the drain's rescan
        assert await asyncio.to_thread(supervisor.wait, 60) == SUPERVISE_STOPPED
        (timeout,) = events(runs, "drain_timeout")
        assert list(timeout["workers"]) == ["alfworld-luna-mirix"]
        assert isinstance(timeout["gateway_reaper"], int)
        stubborn.kill()
        stubborn.wait(timeout=10)
        await eventually(lambda: bool(events(runs, "gateway_reaped")), 30)
    finally:
        for process in (stubborn, late, supervisor):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    (drain,) = events(runs, "drain")
    assert list(drain["workers"]) == ["alfworld-luna-mirix"]
    assert list(events(runs, "drain_late_workers")[0]["workers"]) == [
        "alfworld-luna56-awm"
    ]
    assert (
        read_json(CellPaths(runs / "alfworld-luna56-awm").status)["signal"] == "SIGTERM"
    )


# --- one-off records, restart streaks, drains, internal errors, hung schedulers ------------

ONE_OFF_ARGS = (
    "--reset-resume",
    "alfworld-luna-none",
    "--rerun-harness-failures",
    "alfworld-luna-mirix",
)
ONE_OFF_TOKENS = ("--reset-resume", "--rerun-harness-failures", "alfworld-luna-none")


def one_off_record(runs: Path) -> dict[str, dict[str, Any]]:
    return read_json(runs / "_supervise" / "one_off.json")["requests"]


def alerts(runs: Path, kind: str | None = None) -> list[dict[str, Any]]:
    return [e for e in events(runs, "alert") if kind is None or e["kind"] == kind]


def orphan_gateway(runs: Path, tmp_path: Path) -> OwnedProcess:
    """A gateway a scheduler launched and left behind: registered in the runs directory,
    its owner lock free, no worker holding its users lock (stop_orphaned stops it)."""
    return OwnedProcess.spawn(
        "gateway",
        [sys.executable, "-c", "import time; time.sleep(120)"],
        env={"PATH": "/usr/bin:/bin"},
        cwd=Path.cwd(),
        log_path=tmp_path / "gateway.log",
        registry=runs / "_gateway" / "owned_processes.jsonl",
        owner_lock=runs / "_gateway" / "gateway-127.0.0.1-1.lock",
        users_lock=runs / "_gateway" / "gateway-127.0.0.1-1.users.lock",
    )


def test_a_state_is_its_schedulers_only_by_pid_and_start_time():
    """A scheduler that reused a dead one's pid (after a reboot)
    never inherits its state.json; without start times the pid decides."""
    state = {"pid": 4242, "started": "Tue Sep 29  09:50:22 2026"}
    assert supervise.state_is_of(state, 4242, "Tue Sep 29 09:50:22 2026")
    assert not supervise.state_is_of(state, 4242, "Wed Sep 30 01:02:03 2026")
    assert not supervise.state_is_of(state, 4243, "Tue Sep 29 09:50:22 2026")
    assert supervise.state_is_of({"pid": 4242}, 4242, "Wed Sep 30 01:02:03 2026")
    assert supervise.state_is_of(state, 4242, None)
    assert not supervise.state_is_of(state, None, None)


@pytest.mark.asyncio
async def test_a_dead_schedulers_state_never_marks_a_new_ones_one_offs_applied(
    tmp_path,
):
    """A new scheduler that reused the pid of a dead one whose last state.json said
    one_off_applied must not have its pending one-offs marked applied without applying
    them. Only the watched scheduler's own state counts."""
    sup, runs, _ = make(tmp_path)
    await sup._load_one_off()
    sup.handle = SchedulerHandle(4242, "Wed Sep 30 01:02:03 2026")
    assert await sup._write_one_off(
        {"k": {"state": "pending", "scheduler": 4242, "option": "--reset-resume"}}
    )
    dead = {"pid": 4242, "started": "Tue Sep 29 09:50:22 2026", "one_off_applied": True}
    await sup._note_one_off_applied(dead)
    assert one_off_record(runs)["k"]["state"] == "pending"  # not applied
    await sup._note_one_off_applied({**dead, "started": "Wed Sep 30 01:02:03 2026"})
    assert one_off_record(runs)["k"]["state"] == "applied"


def test_one_off_requests_are_read_as_argparse_reads_them():
    argv = [
        "--runs-dir",
        "r",
        "--reset-resume",
        "a",
        "--reset-resume=b",
        "--rerun-harness-failures",
        "--auto-resume",
    ]
    reset, rerun = one_off_requests(argv)
    assert reset == OneOffRequest("--reset-resume", ("b",))  # the last one wins
    assert rerun == OneOffRequest("--rerun-harness-failures", ())
    # The cells in any order are the same request; another --one-off-again id is not.
    same = OneOffRequest("--reset-resume", ("x", "y")).key
    assert OneOffRequest("--reset-resume", ("y", "x")).key == same
    assert OneOffRequest("--reset-resume", ("x", "y"), "2026-10-01").key != same
    assert one_off_requests(["--runs-dir", "r", "--auto-resume"]) == []
    # The keys recorded before --new-attempt-cells existed are unchanged.
    blob = json.dumps(["--reset-resume", ["x", "y"], None], ensure_ascii=False)
    assert same == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    # --new-attempt-cells: its note is part of the request, wherever it is.
    (rerun,) = one_off_requests(
        ["--new-attempt-note", "n1", "--new-attempt-cells", "a", "b", "--auto-resume"]
    )
    assert rerun == OneOffRequest("--new-attempt-cells", ("a", "b"), None, "n1")
    (later,) = one_off_requests(
        ["--new-attempt-cells", "b", "a", "--new-attempt-note=n1"]
    )
    assert later.key == rerun.key
    (other,) = one_off_requests(
        ["--new-attempt-cells", "a", "b", "--new-attempt-note", "n2"]
    )
    assert other.key != rerun.key
    assert rerun.describe() == "--new-attempt-cells a b (note: n1)"
    # A lone note is no request.
    assert one_off_requests(["--new-attempt-note", "n1"]) == []


@pytest.mark.asyncio
async def test_a_restarted_supervisor_never_applies_the_one_off_options_again(tmp_path):
    """A supervisor restarted with the same command line (systemd's
    ExecStart after a crash or a reboot) must not release the HARNESS stops nobody looked
    at since, nor clear the resume counters again: one_off.json says they were applied,
    so they go to no scheduler, and a person hears of it."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=ONE_OFF_ARGS)
    assert await run(sup) == SUPERVISE_COMPLETE
    (first,) = scheduler_log(runs, "start")
    assert all(token in first["argv"] for token in ONE_OFF_TOKENS)
    record = one_off_record(runs)
    assert sorted(e["option"] for e in record.values()) == [
        "--rerun-harness-failures",
        "--reset-resume",
    ]
    assert {e["state"] for e in record.values()} == {"applied"}
    assert {e["scheduler"] for e in record.values()} == {first["pid"]}
    assert alerts(runs) == []
    # The same command line again.
    sup, runs, _ = make(
        tmp_path,
        {"exit": 0},
        args=ONE_OFF_ARGS,
        notify_cmd=f"{sys.executable} {notifier} {out}",
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    _, again = scheduler_log(runs, "start")
    assert not [
        t for t in ONE_OFF_TOKENS + ("alfworld-luna-mirix",) if t in again["argv"]
    ]
    assert again["argv"].count("--auto-resume") == 1
    withheld = events(runs, "one_off_already_applied")
    assert sorted(e["option"] for e in withheld) == [
        "--rerun-harness-failures",
        "--reset-resume",
    ]
    assert {e["state"] for e in withheld} == {"applied"}
    (row,) = alerts(runs)
    assert row["kind"] == "one_off_already_applied"
    assert "--rerun-harness-failures alfworld-luna-mirix" in row["text"]
    assert "--one-off-again <new id>" in row["text"]
    assert "not passing --reset-resume" in (runs / "_supervise" / "ALERT").read_text()
    assert [n["alert"] for n in read_jsonl(out)] == [row["text"]]
    assert one_off_record(runs) == record  # nothing new was handed out


@pytest.mark.asyncio
async def test_one_off_again_applies_them_once_more_under_a_new_id(tmp_path):
    """--one-off-again ID is the deliberate re-application; the id makes it
    a new request, so a restart with that command line does not apply them a third time."""
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=ONE_OFF_ARGS)
    assert await run(sup) == SUPERVISE_COMPLETE
    for _ in range(2):
        sup, runs, _ = make(
            tmp_path, {"exit": 0}, args=ONE_OFF_ARGS, one_off_again="2026-10-01"
        )
        assert await run(sup) == SUPERVISE_COMPLETE
    first, again, restarted = scheduler_log(runs, "start")
    assert all(token in again["argv"] for token in ONE_OFF_TOKENS)
    assert not [t for t in ONE_OFF_TOKENS if t in restarted["argv"]]
    record = one_off_record(runs)
    assert len(record) == 4
    assert sorted(str(e["again"]) for e in record.values()) == [
        "2026-10-01",
        "2026-10-01",
        "None",
        "None",
    ]
    assert {e["scheduler"] for e in record.values() if e["again"]} == {again["pid"]}
    assert [e["again"] for e in events(runs, "one_off_already_applied")] == [
        "2026-10-01",
        "2026-10-01",
    ]


@pytest.mark.asyncio
async def test_one_off_options_are_recorded_pending_before_the_scheduler_starts(
    tmp_path,
):
    """The record is written before the scheduler that gets them starts
    (pending, then applied); a supervisor that died in between leaves a pending record,
    which its successor counts as applied (erring on the safe side)."""
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=ONE_OFF_ARGS)
    seen: list[Any] = []
    start_process = sup._start_process

    async def start(argv: Any, log_fh: Any, env: Any) -> Any:
        seen.append(read_json(sup.one_off_path))  # what a successor would find now
        return await start_process(argv, log_fh, env)

    sup._start_process = start  # type: ignore[method-assign]
    assert await run(sup) == SUPERVISE_COMPLETE
    (at_start,) = seen
    assert [e["state"] for e in at_start["requests"].values()] == ["pending"] * 2
    assert [e["scheduler"] for e in at_start["requests"].values()] == [None] * 2
    # The supervisor died right after the start: its successor finds that record.
    write_json_atomic(sup.one_off_path, at_start)
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=ONE_OFF_ARGS)
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not [
        t for t in ONE_OFF_TOKENS if t in scheduler_log(runs, "start")[-1]["argv"]
    ]
    (row,) = alerts(runs, "one_off_already_applied")
    assert "pending" in row["text"] and "counts as applied" in row["text"]


@pytest.mark.asyncio
async def test_a_one_off_record_that_cannot_be_written_or_read_withholds_them(
    tmp_path, monkeypatch
):
    """Options the record cannot hold go to no scheduler (a restart could
    apply them twice); an unreadable record withholds them all. Both alert."""
    real_write = supervise.write_json_atomic

    def no_record(path: Path, value: Any, **kw: Any) -> None:
        if path.name == supervise.ONE_OFF_FILE:
            raise OSError(28, "No space left on device")
        real_write(path, value, **kw)

    monkeypatch.setattr(supervise, "write_json_atomic", no_record)
    sup, runs, _ = make(tmp_path / "full", {"exit": 0}, args=ONE_OFF_ARGS)
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not [
        t for t in ONE_OFF_TOKENS if t in scheduler_log(runs, "start")[0]["argv"]
    ]
    (row,) = alerts(runs, "one_off_unrecorded")
    assert "not passing it" in row["text"]
    assert events(runs, "one_off_record_failed")
    monkeypatch.setattr(supervise, "write_json_atomic", real_write)
    sup, runs, _ = make(tmp_path / "torn", {"exit": 0}, args=ONE_OFF_ARGS)
    (runs / "_supervise").mkdir(parents=True)
    (runs / "_supervise" / "one_off.json").write_text('{"requests": ')
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not [
        t for t in ONE_OFF_TOKENS if t in scheduler_log(runs, "start")[0]["argv"]
    ]
    (row,) = alerts(runs, "one_off_record_unreadable")
    assert "cannot read" in row["text"]
    assert (runs / "_supervise" / "one_off.json").read_text() == '{"requests": '


@pytest.mark.asyncio
async def test_a_scheduler_that_never_started_takes_its_pending_record_back(tmp_path):
    """A start that failed handed the one-off options to nobody; the pending record
    goes, and the supervisor's own failure exits 70."""
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=ONE_OFF_ARGS)

    async def cannot_start(*_a: Any) -> Any:
        raise OSError(24, "Too many open files")

    sup._start_process = cannot_start  # type: ignore[method-assign]
    assert await run(sup) == SUPERVISE_INTERNAL_ERROR
    assert one_off_record(runs) == {}
    assert scheduler_log(runs) == []


@pytest.mark.asyncio
async def test_a_supervisor_that_died_hands_its_one_off_options_to_nobody_else(
    tmp_path,
):
    """A supervisor SIGKILLed while its scheduler runs; the next one (the
    same command line) adopts that scheduler, records the options applied by it, and
    starts the later scheduler without them."""
    runs = tmp_path / "runs"
    release = tmp_path / "release"
    script(
        runs,
        {"until": str(release), "exit": 3, "pause_reason": "gateway_unreachable"},
        {"exit": 0},
    )
    record_path = runs / "_supervise" / "one_off.json"
    first = start_supervisor(runs, "--reset-resume", "alfworld-luna-none")
    try:
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        (start,) = scheduler_log(runs, "start")

        def names_it() -> bool:
            with contextlib.suppress(OSError, ValueError, TypeError):
                entries = list(read_json(record_path)["requests"].values())
                return [e["scheduler"] for e in entries] == [start["pid"]]
            return False

        await eventually(names_it, 30)
        first.kill()  # a crash: nothing cleaned up, the scheduler goes on
        first.wait(timeout=10)
        assert process_start_time(start["pid"]) is not None
        clock = Clock()
        options = SuperviseOptions(
            runs_dir=runs,
            scheduler_args=(
                "--runs-dir",
                str(runs),
                "--reset-resume",
                "alfworld-luna-none",
                "--auto-resume",
            ),
            scheduler_module=FAKE_SCHEDULER,
            caffeinate=False,
            heartbeat_s=0.05,
            drain_settle_s=0.2,
        )
        sup = Supervisor(options, mono=clock.mono, pause=clock.pause, sampler=reading)
        task = asyncio.ensure_future(run(sup))
        await eventually(lambda: bool(events(runs, "scheduler_adopted")), 30)
        release.touch()
        assert await task == SUPERVISE_COMPLETE
    finally:
        release.touch()
        if first.poll() is None:
            first.kill()
            first.wait(timeout=10)
    adopted, later = scheduler_log(runs, "start")
    assert adopted["pid"] == start["pid"] and "--reset-resume" in adopted["argv"]
    assert "--reset-resume" not in later["argv"]
    assert "alfworld-luna-none" not in later["argv"]
    (entry,) = one_off_record(runs).values()
    assert entry["state"] == "applied" and entry["scheduler"] == adopted["pid"]
    assert alerts(runs, "one_off_already_applied")


@pytest.mark.asyncio
async def test_a_scheduler_alternating_crashes_and_pauses_is_a_crash_loop(tmp_path):
    """Crashes and pauses in turn are one streak, so neither resets the other's count:
    an alert once the streak is --stall-alert long, and one crash_loop alert at the
    fourth crash; the restarts go on (no exit 12) until a person stops it."""
    steps = []
    for _ in range(40):
        steps.append({"exit": 1, "no_final_state": True})
        steps.append({"exit": 3, "pause_reason": "gateway_unreachable"})
    sup, runs, clock = make(tmp_path, *steps, max_crash_restarts=3, stall_alert_s=600.0)
    clock.on_wait = lambda n: sup.request_stop() if n >= 12 else None
    assert await run(sup) == SUPERVISE_STOPPED
    assert clock.waits == [10.0, 300.0, 20.0, 300.0, 40.0, 300.0, 80.0, 300.0] + [
        160.0,
        300.0,
        300.0,
        300.0,
    ]
    stalled, loop = alerts(runs)
    assert (
        stalled["kind"] == "stalled_restarts" and "still restarting" in stalled["text"]
    )
    assert "exit 1 x3, exit 3 x2" in stalled["text"]
    assert loop["kind"] == "crash_loop" and "4 times in a row" in loop["text"]


@pytest.mark.asyncio
async def test_a_scheduler_alternating_pauses_and_refusals_is_alerted(tmp_path):
    """A pause exit does not restart the refusal timer: for a scheduler alternately
    paused and refused, the refusal timeout runs from the streak's first refusal: one
    refused alert (the retries go on, spaced by --restart-backoff-cap)."""
    sup, runs, clock = make(
        tmp_path,
        {"exit": 3, "pause_reason": "gateway_unreachable"},
        refused_timeout_s=1800.0,
    )
    lock_path = runs / "_schedule" / "scheduler.lock"
    held: list[OwnerLock] = []

    def on_wait(n: int) -> None:
        if n >= 16:
            sup.request_stop()
        elif clock.waits[-1] == 300.0 and not held:
            held.append(OwnerLock.acquire(lock_path))  # the next start is refused
        elif clock.waits[-1] == 60.0 and held:
            held.pop().release()

    clock.on_wait = on_wait
    try:
        assert await run(sup) == SUPERVISE_STOPPED  # never exit 13
    finally:
        for lock in held:
            lock.release()
    # Past the timeout the refused starts are retried every --restart-backoff-cap.
    assert clock.waits == [300.0, 60.0] * 5 + [300.0] * 6
    kinds = [a["kind"] for a in alerts(runs)]
    assert kinds == ["stalled_restarts", "refused"]
    assert "with no useful run in between" in alerts(runs, "refused")[0]["text"]


@pytest.mark.asyncio
async def test_only_a_useful_run_ends_the_streak(tmp_path):
    """A cell completed since the last exit, or a run of --crash-window,
    ends the streak (the crash count with it); a pause exit alone does not. The crash
    loop alert goes out once per streak; a new streak re-arms it."""
    sup, runs, clock = make(
        tmp_path, max_crash_restarts=2, crash_window_s=3600.0, stall_alert_s=1e9
    )
    sup._stop = asyncio.Event()
    sup._escalate = asyncio.Event()
    pids = iter(range(50000, 50100))

    async def exit_with(code: int, *, complete: int = 0, ran_s: float = 0.0) -> Any:
        pid = next(pids)
        write_json_atomic(
            runs / "_schedule" / "state.json",
            {
                "pid": pid,
                "complete": [f"cell-{i}" for i in range(complete)],
                "pause_reason": "gateway_unreachable" if code == 3 else None,
                "exit_code": code,
            },
        )
        sup.handle = SchedulerHandle(pid, None, started_mono=clock.now - ran_s)
        return await sup.after_exit(code)

    assert await exit_with(1) is None
    assert await exit_with(3) is None  # no progress: the count goes on
    assert await exit_with(1) is None
    assert await exit_with(3, complete=1) is None  # a cell completed: a new streak
    assert await exit_with(1, complete=1) is None
    assert await exit_with(1, complete=1) is None
    assert await exit_with(1, complete=1, ran_s=3600.0) is None  # a long run: new too
    assert await exit_with(1, complete=1) is None
    assert await exit_with(3, complete=1) is None
    assert alerts(runs) == []
    assert await exit_with(1, complete=1) is None  # the third crash: a crash loop
    assert [a["kind"] for a in alerts(runs)] == ["crash_loop"]
    assert await exit_with(1, complete=1) is None  # once per streak
    assert [a["kind"] for a in alerts(runs)] == ["crash_loop"]
    assert await exit_with(1, complete=2) is None  # progress: a new streak
    assert await exit_with(1, complete=2) is None
    assert await exit_with(1, complete=2) is None
    assert [a["kind"] for a in alerts(runs)] == ["crash_loop", "crash_loop"]


@pytest.mark.asyncio
async def test_a_healthy_run_between_two_refusals_is_not_refused_time(tmp_path):
    """One transient refusal, a healthy 45-minute run killed by the OOM killer, then one
    more transient refusal must not give the refusal verdict at once: the time a
    scheduler ran is not refused time (the stalled_restarts alert, on wall time, names
    the longest run)."""
    sup, runs, clock = make(
        tmp_path, refused_timeout_s=1800.0, crash_window_s=3600.0, stall_alert_s=1200.0
    )
    sup._stop = asyncio.Event()
    sup._escalate = asyncio.Event()
    pids = iter(range(50000, 50100))

    async def exit_with(code: int, *, ran_s: float = 0.0) -> Any:
        sup.handle = SchedulerHandle(next(pids), None, started_mono=clock.now - ran_s)
        return await sup.after_exit(code)

    assert await exit_with(4) is None  # a gateway /health slow after a crash
    clock.now += 45 * 60
    assert await exit_with(-9, ran_s=45 * 60) is None  # OOM-killed after 45 min
    assert await exit_with(4) is None  # no refused verdict here
    # The streak's alert stays on wall time, and says the scheduler did run.
    (stalled,) = alerts(runs)
    assert stalled["kind"] == "stalled_restarts"
    assert "the longest run 45 min" in stalled["text"]
    # Refusals in a row still reach the timeout: only the waits count. One alert,
    # the retries go on (no exit 13).
    refused_at = None
    for n in range(40):
        assert await exit_with(4) is None
        if refused_at is None and alerts(runs, "refused"):
            refused_at = n
    assert refused_at is not None and refused_at > 0
    assert [a["kind"] for a in alerts(runs)] == ["stalled_restarts", "refused"]


@pytest.mark.asyncio
async def test_refused_starts_without_a_wait_still_time_out(tmp_path):
    """With --refused-retry 0 a refused start's own time must not be subtracted as if
    it had been a run (the refusal timeout would never come). A refused start's time
    counts: the refused alert comes (once), the retries go on."""
    sup, runs, clock = make(tmp_path, refused_timeout_s=600.0, refused_retry_s=0.0)
    sup._stop = asyncio.Event()
    sup._escalate = asyncio.Event()
    pids = iter(range(50000, 51000))
    alerted_after = None
    for n in range(200):
        clock.now += 5.0  # each refused start takes 5 s
        sup.handle = SchedulerHandle(next(pids), None, started_mono=clock.now - 5.0)
        assert await sup.after_exit(4) is None
        if alerted_after is None and alerts(runs, "refused"):
            alerted_after = n
    assert alerted_after is not None
    assert len(alerts(runs, "refused")) == 1
    # Once alerted, the retries wait --restart-backoff-cap, not --refused-retry 0.
    assert clock.waits[: alerted_after + 1][-1] == 300.0
    assert set(clock.waits[:alerted_after]) == {0.0}


@pytest.mark.asyncio
async def test_a_crash_loop_never_drains_the_workers(tmp_path):
    """A crash loop only alerts. The workers the crashing scheduler left go on in their
    own sessions (no SIGTERM), their gateway stays up, and the next scheduler adopts
    them."""
    runs = tmp_path / "runs"
    cell = "alfworld-luna-mirix"
    release = tmp_path / "release"
    worker = start_worker(runs, cell, until=str(release), until_timeout_s=60)
    gateway = orphan_gateway(runs, tmp_path)
    try:
        await eventually(lambda: lock_is_held(runs / cell / "cell.lock"), 30)
        sup, runs, _ = make(tmp_path, {"exit": 1}, {"exit": 0}, max_crash_restarts=0)
        assert await run(sup) == SUPERVISE_COMPLETE  # no exit 12, nothing drained
        assert worker.poll() is None and gateway.popen.poll() is None
        assert [a["kind"] for a in alerts(runs)] == ["crash_loop"]
        assert events(runs, "drain") == []
        release.touch()
        assert await asyncio.to_thread(worker.wait, 30) == 0
    finally:
        release.touch()
        for process in (worker, gateway.popen):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)


@pytest.mark.asyncio
async def test_an_unknown_budget_never_drains(tmp_path):
    """An unknown budget is alerted, never a stop: the scheduler is not signalled and
    the workers are not drained."""
    runs = tmp_path / "runs"
    cell = "alfworld-luna-none"
    release = tmp_path / "release"
    worker = start_worker(runs, cell, until=str(release), until_timeout_s=60)
    try:
        await eventually(lambda: lock_is_held(runs / cell / "cell.lock"), 30)
        sup, runs, _ = make(
            tmp_path,
            {"pause_reason": "balance_unknown", "sleep_s": 1.0, "exit": 0},
            budget_unknown_grace_s=0.0,
        )
        assert await run(sup) == SUPERVISE_COMPLETE  # no exit 11, nothing drained
        assert worker.poll() is None
        assert alerts(runs, "budget_unknown")
        assert scheduler_log(runs, "signal") == [] and events(runs, "drain") == []
        release.touch()
        assert await asyncio.to_thread(worker.wait, 30) == 0
    finally:
        release.touch()
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=30)


@pytest.mark.asyncio
async def test_an_exit_at_an_idle_point_never_touches_other_workers(tmp_path):
    """A scheduler that ended on its own at an idle point left no worker;
    a run.py a person started by hand meanwhile is not drained (exit 11 on an invalid
    budget here; 0, 2 and 10 alike)."""
    runs = tmp_path / "runs"
    cell = "alfworld-luna56-none"
    release = tmp_path / "release"
    worker = start_worker(runs, cell, until=str(release))
    try:
        await eventually(lambda: lock_is_held(runs / cell / "cell.lock"), 30)
        sup, runs, _ = make(tmp_path, {"exit": 3, "pause_reason": "budget_invalid"})
        assert await run(sup) == SUPERVISE_BUDGET_INVALID
        assert worker.poll() is None
    finally:
        release.touch()
        assert worker.wait(timeout=30) == 0
    assert events(runs, "drain") == []


@pytest.mark.asyncio
async def test_the_drain_waits_for_a_worker_that_shows_after_an_empty_scan(tmp_path):
    """The drain must not end on its first empty scan, or a worker the scheduler had
    just launched (not holding its cell lock yet) loses its gateway: it ends once no
    worker has shown for --drain-settle."""
    runs = tmp_path / "runs"
    script(runs, {"sleep_s": 60})
    cell = "alfworld-luna56-awm"
    late = None
    supervisor = None
    try:
        supervisor = start_supervisor(runs, "--drain-settle", "15")
        await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
        supervisor.send_signal(signal.SIGTERM)
        await eventually(lambda: bool(events(runs, "drain")), 30)
        late = start_worker(
            runs, cell, until=str(tmp_path / "never"), until_timeout_s=60
        )
        assert await asyncio.to_thread(late.wait, 30) == 130
        assert await asyncio.to_thread(supervisor.wait, 60) == SUPERVISE_STOPPED
    finally:
        for process in (late, supervisor):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    (drain,) = events(runs, "drain")
    assert drain["workers"] == {} and drain["why"] == "stop"
    assert list(events(runs, "drain_late_workers")[0]["workers"]) == [cell]
    names = [e["event"] for e in events(runs)]
    assert names.index("drain_late_workers") < names.index("stopped")
    assert events(runs, "stopped")[0]["drained"] is True


#: run.py still importing: the pid the scheduler recorded, holding no cell lock for 2 s,
#: then the fake worker in the same process (os.execv keeps the pid).
DELAYED_WORKER = (
    "import os, sys, time; time.sleep(2); "
    f"os.execv(sys.executable, [sys.executable, '-m', '{FAKE_WORKER}', *sys.argv[1:]])"
)


@pytest.mark.asyncio
async def test_the_drain_waits_for_a_launched_worker_that_has_no_lock_yet(tmp_path):
    """A worker in the stopped scheduler's last state.json ("active")
    that is still starting keeps the drain (and so the gateway) going, whatever the
    settle time: it is signalled once it holds its lock."""
    sup, runs, _ = make(tmp_path, {"exit": 0}, drain_settle_s=0.0)
    cell = "alfworld-luna-mirix"
    write_json_atomic(
        runs / cell / "behavior.json",
        {"until": str(tmp_path / "never"), "until_timeout_s": 60},
    )
    slow = subprocess.Popen(
        [sys.executable, "-c", DELAYED_WORKER, "--out", str(runs / cell)],
        cwd=protocol.REPO_ROOT,
        start_new_session=True,
    )
    slow_done = asyncio.ensure_future(asyncio.to_thread(slow.wait, 90))
    gone = subprocess.Popen(["true"])
    gone.wait()
    write_json_atomic(
        runs / "_schedule" / "state.json",
        {"pid": gone.pid, "active": {cell: slow.pid}, "stopped": True, "exit_code": 3},
    )
    sup.handle = SchedulerHandle(gone.pid, "stopped")
    sup._stop = asyncio.Event()
    sup._escalate = asyncio.Event()
    sup._stop.set()
    try:
        assert await sup.shutdown() == SUPERVISE_STOPPED
        assert await asyncio.wait_for(slow_done, 30) == 130  # it was signalled
    finally:
        if slow.poll() is None:
            slow.kill()
        await asyncio.gather(slow_done, return_exceptions=True)
    (drain,) = events(runs, "drain")
    assert drain["workers"] == {} and drain["launched"] == {cell: slow.pid}
    assert events(runs, "drain_late_workers")[0]["workers"] == {cell: slow.pid}
    assert events(runs, "stopped")[0]["drained"] is True


@pytest.mark.asyncio
async def test_an_unexpected_error_alerts_and_exits_70(tmp_path):
    """An error of the supervisor itself never takes it down without a word: it alerts
    (notify, ALERT, events, a final heartbeat naming the error) and exits 70, which the
    systemd unit restarts (not in RestartPreventExitStatus)."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    sup, runs, _ = make(
        tmp_path, {"exit": 0}, notify_cmd=f"{sys.executable} {notifier} {out}"
    )

    async def boom(_code: int) -> Any:
        raise RuntimeError("boom in after_exit")

    sup.after_exit = boom  # type: ignore[method-assign]
    assert await run(sup) == SUPERVISE_INTERNAL_ERROR
    assert SUPERVISE_INTERNAL_ERROR not in {0, 2, 10, 11, 12, 13, 130}
    (note,) = read_jsonl(out)
    assert "the supervisor failed (RuntimeError: boom in after_exit)" in note["alert"]
    assert "boom in after_exit" in (runs / "_supervise" / "ALERT").read_text()
    (row,) = alerts(runs)
    assert row["kind"] == "supervisor_failed"
    exit_row = events(runs, "exit")[-1]
    assert exit_row["code"] == SUPERVISE_INTERNAL_ERROR
    assert "boom in after_exit" in exit_row["traceback"]
    beat = read_json(runs / "_supervise" / "heartbeat.json")
    assert beat["final"] and beat["exit_code"] == SUPERVISE_INTERNAL_ERROR
    assert beat["error"] == "RuntimeError: boom in after_exit"
    assert not lock_is_held(runs / "_supervise" / "supervisor.lock")
    text, code = status_report(runs)
    assert code == supervise.STATUS_STALE and "exited 70" in text
    assert "supervisor error: RuntimeError: boom in after_exit" in text


@pytest.mark.asyncio
async def test_an_error_while_stopping_exits_130_not_70(tmp_path):
    """An error while the supervisor carries out a stop someone asked for
    alerts too, but exits 130: a service manager must not restart what was to stop."""
    sup, runs, _ = make(tmp_path, {"sleep_s": 60})

    async def broken_drain(_why: str) -> None:
        raise PermissionError(13, "Permission denied", str(runs / "cell"))

    sup.drain = broken_drain  # type: ignore[method-assign]
    task = asyncio.ensure_future(run(sup))
    await eventually(lambda: bool(scheduler_log(runs, "start")), 30)
    sup.request_stop()
    assert await task == SUPERVISE_STOPPED
    (row,) = alerts(runs)
    assert row["kind"] == "supervisor_failed" and "failed while stopping" in row["text"]
    assert events(runs, "exit")[-1]["code"] == SUPERVISE_STOPPED
    assert scheduler_log(runs, "end")[0]["code"] == 3  # the scheduler was stopped first


def test_an_error_before_the_supervisor_runs_exits_70_too(tmp_path, monkeypatch):
    """main() reports an error that escapes Supervisor.run as well."""
    import logging

    async def broken(self: Supervisor, **_kw: Any) -> int:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Supervisor, "run", broken)
    monkeypatch.setattr(supervise.logging, "basicConfig", lambda **_kw: None)
    monkeypatch.setattr(
        supervise.logging, "FileHandler", lambda *_a, **_kw: logging.NullHandler()
    )
    runs = tmp_path / "runs"
    code = supervise.main(
        [
            "--runs-dir",
            str(runs),
            "--scheduler-module",
            FAKE_SCHEDULER,
            "--no-caffeinate",
        ]
    )
    assert code == SUPERVISE_INTERNAL_ERROR
    assert "Input/output error" in (runs / "_supervise" / "ALERT").read_text()


@pytest.mark.asyncio
async def test_a_scheduler_whose_state_stops_changing_raises_an_alert(tmp_path):
    """A scheduler that lives but has not rewritten state.json for
    --hung-alert (by the supervisor's monotonic clock) may hang: one alert per stretch,
    and nothing is killed."""
    runs = tmp_path / "runs"
    clock = Clock()
    sup = Supervisor(
        SuperviseOptions(runs_dir=runs, caffeinate=False, hung_alert_s=600.0),
        mono=clock.mono,
        sampler=reading,
    )
    me = os.getpid()
    sup.handle = SchedulerHandle(me, process_start_time(me))  # a live process
    state = runs / "_schedule" / "state.json"
    write_json_atomic(state, {"pid": me, "at": "2026-09-29T00:00:00+00:00"})
    await sup.tick()
    clock.now += 599.0
    await sup.tick()
    assert alerts(runs) == []
    clock.now += 2.0
    await sup.tick()
    (row,) = alerts(runs)
    assert row["kind"] == "scheduler_hung" and f"kill -KILL {me}" in row["text"]
    assert row["scheduler"] == me
    clock.now += 1000.0
    await sup.tick()
    assert len(alerts(runs)) == 1  # once per stretch
    write_json_atomic(state, {"pid": me, "at": "2026-09-29T00:20:00+00:00"})
    await sup.tick()  # it wrote again: the watch starts over
    clock.now += 599.0
    await sup.tick()
    assert len(alerts(runs)) == 1
    # A scheduler that is gone, or being stopped, is no hang.
    sup.handle = SchedulerHandle(me, "another process")
    clock.now += 5000.0
    await sup.tick()
    await sup.tick()
    assert len(alerts(runs)) == 1
    # The default: ten polls, and at least ten minutes.
    assert SuperviseOptions(runs_dir=runs).hung_after_s == 600.0
    assert SuperviseOptions(runs_dir=runs, scheduler_poll_s=90.0).hung_after_s == 900.0


def test_status_flags_a_hung_scheduler_and_a_reused_pid(tmp_path):
    """--status shows the age of the scheduler's state.json and exits 1
    when it is older than the supervisor's --hung-alert while the scheduler lives; a
    process that reused the scheduler's pid is not taken for it."""
    runs = tmp_path / "runs"
    at = time.time()
    me = os.getpid()
    started = process_start_time(me)
    iso = supervise.schedule.iso
    beat = {
        "at": iso(at),
        "pid": me,
        "started": started,
        "heartbeat_s": 30,
        "host": {},
        "final": False,
        "scheduler": {
            "pid": me,
            "started": started,
            "started_at": iso(at - 5000),
            "hung_after_s": 600.0,
            "restarts": 0,
            "crash_restarts": 0,
            "last_exit": None,
        },
    }
    write_json_atomic(runs / "_supervise" / "heartbeat.json", beat)
    state = runs / "_schedule" / "state.json"
    write_json_atomic(state, {"pid": me, "at": iso(at - 30)})
    text, code = status_report(runs, wall=at)
    assert code == 0, text
    assert "(alive), state.json 30.0 s old," in text
    write_json_atomic(state, {"pid": me, "at": iso(at - 1000)})
    text, code = status_report(runs, wall=at)
    assert code == supervise.STATUS_STALE  # although the heartbeat is fresh
    assert "(fresh)" in text and "state.json 1000.0 s old (HUNG? over 600 s)" in text
    # The state.json of an earlier scheduler: its age counts from this one's start.
    write_json_atomic(state, {"pid": 1, "at": iso(at - 10)})
    text, code = status_report(runs, wall=at)
    assert code == supervise.STATUS_STALE
    assert "no state.json of its own yet (5000.0 s since it started) (HUNG?" in text
    # Another process has the pid now: the scheduler is gone, and gone is no hang.
    beat["scheduler"]["started"] = "Thu Jan  1 00:00:00 1970"
    write_json_atomic(runs / "_supervise" / "heartbeat.json", beat)
    text, code = status_report(runs, wall=at)
    assert code == 0, text
    assert f"scheduler: pid {me} (gone)" in text and "HUNG" not in text
    # No scheduler started yet.
    beat["scheduler"] = {"pid": None, "started_at": None}
    write_json_atomic(runs / "_supervise" / "heartbeat.json", beat)
    text, code = status_report(runs, wall=at)
    assert code == 0 and "scheduler: pid None (gone), none started yet," in text


def test_the_new_options_reach_the_supervisor_not_the_scheduler(tmp_path):
    runs = str(tmp_path / "runs")
    options, _ = parse_args(
        [
            "--runs-dir",
            runs,
            "--rerun-harness-failures",
            "--one-off-again",
            "2026-10-01",
            "--hung-alert",
            "900",
            "--drain-settle",
            "2",
            "--poll",
            "5",
        ]
    )
    assert options.scheduler_args == (
        "--runs-dir",
        runs,
        "--rerun-harness-failures",
        "--poll",
        "5",
        "--auto-resume",
    )
    assert options.one_off_again == "2026-10-01" and options.hung_after_s == 900.0
    assert options.drain_settle_s == 2.0 and options.scheduler_poll_s == 5.0
    default, _ = parse_args(["--runs-dir", runs, "--poll", "90"])
    assert default.hung_after_s == 900.0 and default.one_off_again is None
    for bad in (["--one-off-again", " "], ["--hung-alert", "0"]):
        with pytest.raises(SystemExit):
            parse_args(["--runs-dir", runs, *bad])


# --- alerts while the scheduler runs ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cell_needing_a_person_is_alerted_at_once_while_the_scheduler_runs(
    tmp_path,
):
    """A cell that comes to need a person (a HARNESS stop) is alerted within one
    heartbeat while the scheduler goes on; the same set is not alerted again, a new cell
    is, and a cell that left and comes back is too."""
    release = tmp_path / "release"
    first = "alfworld-luna56-awm"
    sup, runs, _ = make(
        tmp_path,
        {
            "until": str(release),
            "exit": 0,
            "attention": {first: "learning_failed"},
            "attention_reasons": {first: "harness_failure"},
        },
    )
    task = asyncio.ensure_future(run(sup))
    try:
        await eventually(lambda: bool(alerts(runs, "cell_needs_attention")), 30)
        assert scheduler_log(runs, "end") == []  # before any exit
        (alert,) = alerts(runs, "cell_needs_attention")
        assert alert["cells"] == {first: ["learning_failed", "harness_failure"]}
        assert alert["text"] == (
            f"cells need a person now: {first} (learning_failed: harness_failure); "
            "the others go on"
        )
        assert "harness_failure" in (runs / "_supervise" / "ALERT").read_text()
        await asyncio.sleep(0.4)  # several heartbeats: the same set, no new alert
        assert len(alerts(runs, "cell_needs_attention")) == 1
        state_path = runs / "_schedule" / "state.json"
        state = read_json(state_path)
        write_json_atomic(
            state_path, {**state, "blocked": {"alfworld-luna-awm": "refused"}}
        )
        await eventually(lambda: len(alerts(runs, "cell_needs_attention")) == 2, 30)
        assert alerts(runs, "cell_needs_attention")[1]["cells"] == {
            "alfworld-luna-awm": ["refused", "blocked"]
        }
        # The first cell leaves (resumed) and comes back: alerted again.
        write_json_atomic(state_path, {**state, "needs_attention": {}})
        await asyncio.sleep(0.4)
        write_json_atomic(state_path, state)
        await eventually(lambda: len(alerts(runs, "cell_needs_attention")) == 3, 30)
        # Another process's state.json (a dead scheduler's) never counts.
        stranger = {**state, "pid": 1, "needs_attention": {"x": "crashed"}}
        write_json_atomic(state_path, stranger)
        await asyncio.sleep(0.4)
        assert len(alerts(runs, "cell_needs_attention")) == 3
        write_json_atomic(state_path, state)
    finally:
        release.touch()
    assert await task == SUPERVISE_COMPLETE


@pytest.mark.asyncio
async def test_an_adopted_gateway_with_a_problem_is_alerted_at_once(tmp_path):
    """An orphan gateway the scheduler adopted at its start although it runs with
    another budget than --budget (or is not healthy) is in state.json ``gateways`` with
    its ``problem``: alerted within one heartbeat, once per gateway and problem; one
    that clears and comes back is alerted again; a dead scheduler's state never
    counts."""
    release = tmp_path / "release"
    sup, runs, _ = make(tmp_path, {"until": str(release), "exit": 0})
    task = asyncio.ensure_future(run(sup))
    state_path = runs / "_schedule" / "state.json"
    problem = (
        "the gateway at http://127.0.0.1:9 runs with budget 100.0, not --budget 300"
    )
    try:
        await eventually(lambda: state_path.is_file(), 30)
        state = read_json(state_path)
        bad = {
            **state,
            "gateways": {
                "9": {
                    "url": "http://127.0.0.1:9",
                    "mode": "adopted",
                    "problem": problem,
                },
                "10": {
                    "url": "http://127.0.0.1:10",
                    "mode": "adopted",
                    "problem": None,
                },
            },
        }
        write_json_atomic(state_path, bad)
        await eventually(lambda: bool(alerts(runs, "gateway_needs_attention")), 30)
        (alert,) = alerts(runs, "gateway_needs_attention")
        assert alert["gateways"] == {"9": problem}
        assert "port 9: " + problem in alert["text"]
        assert "budget 100.0" in (runs / "_supervise" / "ALERT").read_text()
        await asyncio.sleep(0.4)  # several heartbeats: the same problem, no new alert
        assert len(alerts(runs, "gateway_needs_attention")) == 1
        write_json_atomic(state_path, state)  # cleared
        await asyncio.sleep(0.4)
        write_json_atomic(state_path, bad)  # back
        await eventually(lambda: len(alerts(runs, "gateway_needs_attention")) == 2, 30)
        write_json_atomic(state_path, {**bad, "pid": 1})  # not this scheduler's
        await asyncio.sleep(0.4)
        write_json_atomic(state_path, state)
        await asyncio.sleep(0.2)
        write_json_atomic(state_path, {**bad, "pid": 1})
        await asyncio.sleep(0.4)
        assert len(alerts(runs, "gateway_needs_attention")) == 2
    finally:
        release.touch()
    assert await task == SUPERVISE_COMPLETE


def running_cell(runs: Path, cell: str, *, attempts: int = 2, status: str = "running"):
    paths = CellPaths(runs / cell)
    write_json_atomic(paths.manifest, {"cell": {"cell_id": cell}})
    write_json_atomic(paths.status, {"status": status})
    for number in range(1, attempts + 1):
        write_json_atomic(paths.attempt(number).state, {"attempt": number})
    return paths


@pytest.mark.asyncio
async def test_a_growing_recorded_problem_is_alerted_then_throttled(
    tmp_path, monkeypatch
):
    """What a running cell recorded and went on past is alerted: a
    (cell, code) seen for the first time at once, a growing count at most once per
    --stall-alert. The counts come from cell_records.recorded_problems on the cell's
    highest attempt; a cell that does not run is not read."""
    from evals.skill_bench import cell_records

    sup, runs, _ = make(tmp_path, stall_alert_s=600.0)
    cell = "webshop-luna56-mirix"
    running_cell(runs, cell)
    running_cell(runs, "webshop-luna-mirix", status="complete")
    counts: dict[str, int] = {"provider_unserved": 1}
    read: list[str] = []

    def recorded_problems(paths: Any) -> dict[str, int]:
        read.append(f"{paths.root.parent.name}/{paths.root.name}")
        return dict(counts)

    monkeypatch.setattr(
        cell_records, "recorded_problems", recorded_problems, raising=False
    )
    await sup._watch_recorded_problems({}, 0.0)
    assert read == [f"{cell}/attempt-02"]
    (first,) = alerts(runs, "recorded_problem")
    assert (first["cell"], first["code"], first["count"]) == (
        cell,
        "provider_unserved",
        1,
    )
    assert first["text"] == (
        f"{cell}: provider_unserved x1 (recorded and continued; see its attempt.json)"
    )
    counts["provider_unserved"] = 2
    await sup._watch_recorded_problems({}, 100.0)  # grew, but within --stall-alert
    assert len(alerts(runs, "recorded_problem")) == 1
    counts["provider_unserved"] = 3
    await sup._watch_recorded_problems({}, 700.0)
    assert [a["count"] for a in alerts(runs, "recorded_problem")] == [1, 3]
    await sup._watch_recorded_problems({}, 1400.0)  # did not grow: nothing
    counts["retrieval_degraded"] = 1  # a new code: at once
    await sup._watch_recorded_problems({}, 1410.0)
    got = alerts(runs, "recorded_problem")
    assert [(a["code"], a["count"]) for a in got] == [
        ("provider_unserved", 1),
        ("provider_unserved", 3),
        ("retrieval_degraded", 1),
    ]
    # A reader that fails loses this beat's alerts, never the supervisor.

    def broken(paths: Any) -> dict[str, int]:
        raise RuntimeError("torn record")

    monkeypatch.setattr(cell_records, "recorded_problems", broken)
    await sup._watch_recorded_problems({}, 5000.0)
    # Without the reader (before cell_records provides it) nothing is watched.
    monkeypatch.delattr(cell_records, "recorded_problems")
    await sup._watch_recorded_problems({}, 9000.0)
    assert len(alerts(runs, "recorded_problem")) == 3


def alert_lines(runs: Path) -> list[str]:
    text = (runs / "_supervise" / "ALERT").read_text(encoding="utf-8")
    return [line.split(" ", 1)[1] for line in text.splitlines()]


@pytest.mark.asyncio
async def test_a_recorded_problem_never_takes_the_alert_file_from_a_cell_needing_a_person(
    tmp_path,
):
    """The ALERT file (what --status prints) keeps the alert that needs a
    person first; a recorded problem's alert (raised, notified, in events.jsonl) is
    listed after it, never in its place. The heartbeat's alerts keep one entry per
    (cell, code) of recorded problems and at most HEARTBEAT_ALERTS_MAX in all."""
    sup, runs, _ = make(tmp_path)
    (runs / "_supervise").mkdir(parents=True, exist_ok=True)
    need = "cells need a person now: alfworld-luna-mirix (harness_failure: x)"

    async def recorded(cell: str, code: str, count: int) -> None:
        text = f"{cell}: {code} x{count} (recorded and continued)"
        await sup.alert("recorded_problem", text, cell=cell, code=code, count=count)

    await recorded("alfworld-luna56-mirix", "provider_unserved", 1)
    assert alert_lines(runs) == [
        "alfworld-luna56-mirix: provider_unserved x1 (recorded and continued)"
    ]
    await sup.alert("cell_needs_attention", need)
    await recorded("webshop-luna56-mirix", "retrieval_degraded", 1)
    assert alert_lines(runs) == [
        need,
        "webshop-luna56-mirix: retrieval_degraded x1 (recorded and continued)",
        "alfworld-luna56-mirix: provider_unserved x1 (recorded and continued)",
    ]
    text, _code = status_report(runs)
    assert (
        f"ALERT: {(runs / '_supervise' / 'ALERT').read_text().splitlines()[0]}" in text
    )
    assert need in text.split("ALERT: ", 1)[1].splitlines()[0]
    # A growing count replaces its own entry, in the heartbeat and in the file.
    await recorded("webshop-luna56-mirix", "retrieval_degraded", 4)
    kinds = [(a["kind"], a.get("count")) for a in sup.alerts]
    assert kinds == [
        ("recorded_problem", 1),
        ("cell_needs_attention", None),
        ("recorded_problem", 4),
    ]
    assert alert_lines(runs)[:2] == [
        need,
        "webshop-luna56-mirix: retrieval_degraded x4 (recorded and continued)",
    ]
    # Many recorded problems: the file lists the newest few after the one that needs
    # a person; the heartbeat keeps a bounded list.
    for n in range(supervise.HEARTBEAT_ALERTS_MAX + 20):
        await recorded(f"cell-{n:03d}", "provider_unserved", 1)
    lines = alert_lines(runs)
    assert lines[0] == need
    assert len(lines) == 1 + supervise.ALERT_FILE_RECORDED_MAX
    assert lines[1].startswith(f"cell-{supervise.HEARTBEAT_ALERTS_MAX + 19:03d}:")
    assert len(sup.alerts) <= supervise.HEARTBEAT_ALERTS_MAX
    assert [a["text"] for a in sup.alerts if a["kind"] == "cell_needs_attention"] == [
        need
    ]
    # A newer alert that needs a person is the first line, as before.
    await sup.alert("budget_unknown", "the budget is unknown")
    assert alert_lines(runs)[0] == "the budget is unknown"


@pytest.mark.asyncio
async def test_one_unreadable_cell_never_hides_what_another_recorded(
    tmp_path, monkeypatch
):
    from evals.skill_bench import cell_records

    sup, runs, _ = make(tmp_path)
    running_cell(runs, "alfworld-luna56-mirix")  # sorted first: its reader raises
    running_cell(runs, "webshop-luna-mirix")

    def recorded_problems(paths: Any) -> dict[str, int]:
        if paths.root.parent.name == "alfworld-luna56-mirix":
            raise ValueError("torn attempt.json")
        return {"provider_unserved": 2}

    monkeypatch.setattr(
        cell_records, "recorded_problems", recorded_problems, raising=False
    )
    await sup._watch_recorded_problems({}, 0.0)
    got = alerts(runs, "recorded_problem")
    assert [(a["cell"], a["code"], a["count"]) for a in got] == [
        ("webshop-luna-mirix", "provider_unserved", 2)
    ]


@pytest.mark.asyncio
async def test_recorded_problems_are_watched_every_heartbeat(tmp_path, monkeypatch):
    from evals.skill_bench import cell_records

    release = tmp_path / "release"
    sup, runs, _ = make(tmp_path, {"until": str(release), "exit": 0})
    running_cell(runs, "alfworld-luna-mirix", attempts=1)
    monkeypatch.setattr(
        cell_records,
        "recorded_problems",
        lambda paths: {"mirix_batch_accepted": 1, "zero": 0},
        raising=False,
    )
    task = asyncio.ensure_future(run(sup))
    try:
        await eventually(lambda: bool(alerts(runs, "recorded_problem")), 30)
        await asyncio.sleep(0.3)
    finally:
        release.touch()
    assert await task == SUPERVISE_COMPLETE
    (alert,) = alerts(runs, "recorded_problem")
    assert (alert["cell"], alert["code"], alert["count"]) == (
        "alfworld-luna-mirix",
        "mirix_batch_accepted",
        1,
    )


def test_the_recorded_problems_reader_contract(tmp_path):
    """With cell_records' own reader: it never raises and a missing file counts 0,
    so a fresh attempt records nothing."""
    from evals.skill_bench import cell_records

    reader = getattr(cell_records, "recorded_problems", None)
    if reader is None:
        pytest.skip("cell_records.recorded_problems is not in this tree")
    runs = tmp_path / "runs"
    running_cell(runs, "alfworld-luna-mirix", attempts=1)
    assert supervise._recorded_problems(runs, reader) == {}


# --- --new-attempt-cells ------------------------------------------------------------------


NOTE = "rerun after a code fix"
RERUN = ("--benchmark", "alfworld", "--new-attempt-cells", "alfworld-luna-mirix")


@pytest.mark.asyncio
async def test_new_attempt_cells_go_to_one_scheduler_only(tmp_path):
    """--new-attempt-cells is a one-off. A supervisor restarted with the same
    command line (systemd, the same ExecStart) never hands it to a second scheduler
    (the scheduler would not rerun twice either); a new note is a new request."""
    args = (*RERUN, "--new-attempt-note", NOTE)
    sup, runs, _ = make(tmp_path, {"exit": 0}, args=args)
    assert await run(sup) == SUPERVISE_COMPLETE
    (first,) = scheduler_log(runs, "start")
    assert "--new-attempt-cells" in first["argv"] and NOTE in first["argv"]
    (entry,) = one_off_record(runs).values()
    assert entry["option"] == "--new-attempt-cells" and entry["state"] == "applied"
    assert entry["values"] == ["alfworld-luna-mirix"] and entry["note"] == NOTE
    again = Supervisor(sup.options, mono=Clock().mono, sampler=reading)
    assert await run(again) == SUPERVISE_COMPLETE
    second = scheduler_log(runs, "start")[-1]
    assert "--new-attempt-cells" not in second["argv"]
    assert "alfworld-luna-mirix" not in second["argv"]
    # The note alone goes on; the scheduler ignores it without the cells.
    assert second["argv"][second["argv"].index("--new-attempt-note") + 1] == NOTE
    (withheld,) = alerts(runs, "one_off_already_applied")
    assert f"(note: {NOTE})" in withheld["text"]
    # Another note: a new request, handed on once.
    other = SuperviseOptions(
        **{
            **{f: getattr(sup.options, f) for f in sup.options.__dataclass_fields__},
            "scheduler_args": tuple(
                "and once more" if a == NOTE else a for a in sup.options.scheduler_args
            ),
        }
    )
    assert await run(Supervisor(other, mono=Clock().mono, sampler=reading)) == 0
    assert "--new-attempt-cells" in scheduler_log(runs, "start")[-1]["argv"]
    assert len(one_off_record(runs)) == 2


@pytest.mark.asyncio
async def test_the_rerun_estimate_is_sent_as_a_notice_not_an_alert(tmp_path):
    """Reruns within the budget need no separate approval: the estimate is sent
    (--notify-cmd, an events.jsonl notice), never raised as an alert (no ALERT file),
    and the reruns run."""
    out = tmp_path / "notified.jsonl"
    notifier = tmp_path / "notify.py"
    notifier.write_text(NOTIFY)
    requests = {
        "alfworld-luna-mirix": {
            "note": NOTE,
            "from_attempt": 1,
            "estimate_usd": 1.25,
            "state": "pending",
            "new_attempt": None,
        },
        "alfworld-luna56-mirix": {
            "note": NOTE,
            "from_attempt": 2,
            "estimate_usd": 0.75,
            "state": "pending",
            "new_attempt": None,
        },
    }
    sup, runs, _ = make(
        tmp_path,
        {"exit": 0, "new_attempts": requests},
        args=(*RERUN, "alfworld-luna56-mirix", "--new-attempt-note", NOTE),
        notify_cmd=f"{sys.executable} {notifier} {out}",
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not (runs / "_supervise" / "ALERT").exists() and alerts(runs) == []
    (notice,) = events(runs, "notice")
    assert notice["kind"] == "new_attempts_requested"
    assert notice["estimate_usd"] == 2.0
    assert notice["text"] == (
        f"alfworld-luna-mirix, alfworld-luna56-mirix start new attempts (note: {NOTE}), "
        "estimated $2.00 from the attempts they replace; within the budget"
    )
    (note,) = read_jsonl(out)
    assert note["alert"] == notice["text"]
    beat = read_json(runs / "_supervise" / "heartbeat.json")
    assert beat["alerts"] == []


def test_new_attempt_cells_are_checked_before_the_supervisor_starts(tmp_path, capsys):
    base = ["--runs-dir", str(tmp_path), "--benchmark", "webshop"]
    with pytest.raises(SystemExit):  # outside the selection
        parse_args(
            [
                *base,
                "--new-attempt-cells",
                "alfworld-luna56-mirix",
                "--new-attempt-note",
                "x",
            ]
        )
    assert "outside the selection" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # no note
        parse_args([*base, "--new-attempt-cells", "webshop-luna56-mirix"])
    assert "--new-attempt-note" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # never ran under --runs-dir
        parse_args(
            [
                *base,
                "--new-attempt-cells",
                "webshop-luna56-mirix",
                "--new-attempt-note",
                "x",
            ]
        )
    assert "never ran under" in capsys.readouterr().err
    (tmp_path / "webshop-luna56-mirix").mkdir()
    (tmp_path / "webshop-luna56-mirix" / "status.json").write_text(
        "{}", encoding="utf-8"
    )
    options, _ = parse_args(
        [
            *base,
            "--new-attempt-cells",
            "webshop-luna56-mirix",
            "--new-attempt-note",
            "x",
        ]
    )
    (request,) = one_off_requests(options.scheduler_args)
    assert request == OneOffRequest(
        "--new-attempt-cells", ("webshop-luna56-mirix",), None, "x"
    )
    assert options.first_attempt_cells == ()
    capsys.readouterr()
    # With a cell that ran named too, one that never ran here is no wrong
    # --runs-dir; it starts as a first attempt (schedule.py records
    # new_attempt_not_needed): a warning, and the request goes on unchanged (the same
    # request identity at every restart, so a later restart never asks again for a
    # cell that has since run its first attempt).
    both = ["webshop-luna56-mirix", "webshop-luna-mirix"]
    options, _ = parse_args(
        [*base, "--new-attempt-cells", *both, "--new-attempt-note", "x"]
    )
    assert options.first_attempt_cells == ("webshop-luna-mirix",)
    (request,) = one_off_requests(options.scheduler_args)
    assert request == OneOffRequest("--new-attempt-cells", tuple(both), None, "x")
    err = capsys.readouterr().err
    assert "webshop-luna-mirix" in err and "first attempt" in err
    assert "webshop-luna56-mirix" not in err


@pytest.mark.asyncio
async def test_a_never_run_new_attempt_cell_is_a_start_notice_not_an_alert(tmp_path):
    # The supervisor records which named cells start as first attempts (an
    # events.jsonl notice), never an ALERT or a notification.
    sup, runs, _ = make(
        tmp_path, {"exit": 0}, args=RERUN, first_attempt_cells=("alfworld-luna-mirix",)
    )
    assert await run(sup) == SUPERVISE_COMPLETE
    assert not (runs / "_supervise" / "ALERT").exists() and alerts(runs) == []
    (notice,) = [
        e for e in events(runs, "notice") if e["kind"] == "new_attempt_cells_never_ran"
    ]
    assert notice["cells"] == ["alfworld-luna-mirix"]
    assert "start as first attempts" in notice["text"]


def test_a_relaunch_that_widens_the_selection_is_refused_at_start(tmp_path, capsys):
    """A runs directory launched with some methods and relaunched without its method
    list (tools/phase16.sh up or recover, the systemd unit) is refused at start (exit 2,
    the alert) instead of starting the other methods' cells; --widen-selection is a
    one-off option (recorded in one_off.json, never passed to a restart)."""
    from evals.skill_bench import schedule

    runs = tmp_path / "runs"
    control = runs / "_schedule"
    control.mkdir(parents=True)
    schedule.record_selection(
        control,
        schedule.study_matrix("phase16", methods=["none", "reasoningbank", "awm"]),
        None,
        pid=1,
    )
    with pytest.raises(SystemExit) as exc:
        parse_args(["--runs-dir", str(runs), "--study", "phase16"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "of methods it never selected: mirix" in err
    alert = (runs / "_supervise" / "ALERT").read_text(encoding="utf-8")
    assert "refused to start" in alert and "--widen-selection" in alert
    # Its own selection, or a narrower one, starts; so does MIRIX with a note.
    parse_args(
        ["--runs-dir", str(runs), "--study", "phase16", "--method", "none", "awm"]
    )
    options, _ = parse_args(
        [
            "--runs-dir",
            str(runs),
            "--study",
            "phase16",
            "--method",
            "mirix",
            "--widen-selection",
            "MIRIX final cells",
        ]
    )
    (request,) = supervise.one_off_requests(options.scheduler_args)
    assert request.option == "--widen-selection"
    assert request.values == ("MIRIX final cells",)
    assert "--widen-selection" in supervise.ONE_OFF_SCHEDULER_OPTIONS
    # --status only reads: never refused for the selection.
    _, args = parse_args(["--status", "--runs-dir", str(runs), "--study", "phase16"])
    assert args.status
