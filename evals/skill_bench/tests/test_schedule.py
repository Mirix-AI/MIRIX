"""The matrix scheduler (H8, H1.3).

Workers are real processes: ``tests/fake_worker.py`` (run.py's command line and cell
protocol, no model calls) for the scheduling rules. The gateway is a real one over the
fake upstream (``tests/fakes.py``); nothing reaches a provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.skill_bench import host_registry, protocol, schedule, study
from evals.skill_bench.cell_records import (
    CellPaths,
    read_json,
    read_jsonl,
    write_json_atomic,
)
from evals.skill_bench.infra.resources import HostResources
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    append_jsonl,
    OwnerLock,
    find_free_port,
    live_cell_workers,
    lock_is_held,
    process_start_time,
    sweep_owned,
)
from evals.skill_bench.run import CellSpec
from evals.skill_bench.schedule import (
    ATTENTION,
    BLOCKED,
    DONE,
    LIVE,
    QUEUED,
    ScheduleError,
    ScheduleOptions,
    Scheduler,
    classify_cell,
    ensure_gateway,
    live_cells,
    matrix,
    worker_argv,
    worker_env,
)
from evals.skill_bench.tests.fakes import FAKE_UPSTREAM_KEY, FakeUpstream
from evals.skill_bench.tests.run_support import (
    conformant_gateway,
    eventually,
    start_orphan_reaper,
    stop_test_processes,
)

FAKE_WORKER = "evals.skill_bench.tests.fake_worker"


def options(tmp_path: Path, url: str, **kw: Any) -> ScheduleOptions:
    kw.setdefault("poll_s", 0.05)
    kw.setdefault("balance_poll_s", 0.0)
    kw.setdefault("worker_module", FAKE_WORKER)
    return ScheduleOptions(runs_dir=tmp_path / "runs", gateway_url=url, **kw)


def behave(opts: ScheduleOptions, spec: CellSpec, **behavior: Any) -> None:
    write_json_atomic(opts.cell_dir(spec) / "behavior.json", behavior)


def set_status(opts: ScheduleOptions, spec: CellSpec, status: str) -> None:
    write_json_atomic(CellPaths(opts.cell_dir(spec)).status, {"status": status})


def worker_log(opts: ScheduleOptions) -> list[dict[str, Any]]:
    return read_jsonl(opts.runs_dir / "_fake_workers.jsonl")


def max_overlap(rows: list[dict[str, Any]]) -> int:
    live, peak = 0, 0
    for row in sorted(rows, key=lambda r: (r["t"], r["event"] == "start")):
        live += 1 if row["event"] == "start" else -1
        peak = max(peak, live)
    return peak


@pytest.fixture(scope="session", autouse=True)
def reap_after_a_killed_session(tmp_path_factory):
    """However this pytest ends (a chunk timeout's SIGKILL too), what its tests left
    running under the session's basetemp is stopped."""
    start_orphan_reaper(tmp_path_factory.getbasetemp())


@pytest.fixture(autouse=True)
def no_leftover_processes(tmp_path):
    """After each test (passed, failed or timed out), the gateways and fake workers it
    started and left running are stopped, and named in a warning."""
    yield
    left = stop_test_processes(tmp_path)
    if left:
        warnings.warn(f"the test left processes running, now stopped: {left}")


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    """The scheduler's docker stats never touch the host's Docker in these tests (a test
    injects its own where it checks them)."""
    monkeypatch.setattr(schedule.resources, "docker_memory", lambda: None)


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def gateway(tmp_path: Path, fake: FakeUpstream):
    with conformant_gateway(tmp_path, fake) as (gw, url):
        yield gw, url


# --- the matrix --------------------------------------------------------------------------------------


def test_the_matrix_is_h13_in_launch_order():
    cells = matrix()
    # 2 benchmarks x 2 models x 4 methods.
    assert len(cells) == 16 == len({c.cell_id for c in cells})
    assert {c.benchmark for c in cells} == set(protocol.BENCHMARKS)
    assert {c.method for c in cells} == set(protocol.METHODS)
    assert {c.model for c in cells} == set(protocol.CHAT_MODELS)
    # ALFWorld, then WebShop; learning cells before none.
    assert [c.benchmark for c in cells[::8]] == ["alfworld", "webshop"]
    assert [c.cell_id for c in cells[:4]] == [
        "alfworld-luna-mirix",
        "alfworld-luna56-mirix",
        "alfworld-luna-reasoningbank",
        "alfworld-luna56-reasoningbank",
    ]
    assert cells[5].method == "awm" and cells[6].method == cells[7].method == "none"
    batch = matrix(["alfworld"], ["openai/gpt-5.6-luna"], ["none", "awm"])
    assert [c.cell_id for c in batch] == ["alfworld-luna56-awm", "alfworld-luna56-none"]
    # A key selects its own cells only: luna never selects luna56, and back.
    assert [c.cell_id for c in matrix(["webshop"], ["luna"], ["none"])] == [
        "webshop-luna-none"
    ]
    picked = matrix(["webshop"], ["luna56", "openai/gpt-6-luna"], ["none"])
    assert [c.cell_id for c in picked] == ["webshop-luna-none", "webshop-luna56-none"]
    assert picked[1].model == "openai/gpt-5.6-luna"
    smoke = matrix(
        ["alfworld"], ["luna"], ["none"], split_mode="smoke", n_train=0, n_test=2
    )
    assert [c.cell_id for c in smoke] == ["alfworld-luna-none-smoke-l0-t2"]
    for bad in (dict(benchmarks=["webarena"]), dict(methods=["exprag"])):
        with pytest.raises(ValueError):
            matrix(**bad)  # type: ignore[arg-type]


def test_the_phase16_studies_select_their_cells():
    """The phase16 studies: ALFWorld and WebShop, luna and luna56, the four methods
    (``phase16``, the protocol cells, which replicate runs directories share); the
    WebShop pilot (``phase16-pilot``); development on the whole train and val splits
    (``phase16-dev``)."""
    final = schedule.study_matrix("phase16")
    assert len(final) == 16 == len({c.cell_id for c in final})
    assert [c.cell_id for c in final[:4]] == [
        "alfworld-luna-mirix",
        "alfworld-luna56-mirix",
        "alfworld-luna-reasoningbank",
        "alfworld-luna56-reasoningbank",
    ]
    assert {c.benchmark for c in final} == {"alfworld", "webshop"}
    assert {c.model_key for c in final} == {"luna", "luna56"}
    assert {c.split_mode for c in final} == {"protocol"}
    assert final == matrix(["alfworld", "webshop"], ["luna", "luna56"])
    pilot = schedule.study_matrix("phase16-pilot")
    assert [c.cell_id for c in pilot] == [
        "webshop-luna-none-pilot",
        "webshop-luna56-none-pilot",
    ]
    dev = schedule.study_matrix("phase16-dev")
    assert len(dev) == 16
    assert {c.cell_id for c in dev if c.method == "mirix"} == {
        "alfworld-luna-mirix-smoke-l39-t18",
        "alfworld-luna56-mirix-smoke-l39-t18",
        "webshop-luna-mirix-smoke-l40-t20",
        "webshop-luna56-mirix-smoke-l40-t20",
    }
    opts = ScheduleOptions(runs_dir=Path("/tmp/runs-dev"))
    argv = worker_argv(dev[-1], opts, "http://127.0.0.1:18541", Path("/tmp/l"))
    assert dev[-1].cell_id == "webshop-luna56-none-smoke-l40-t20"
    assert argv[argv.index("--n-train") + 1] == "40"
    assert argv[argv.index("--n-test") + 1] == "20"
    assert [
        c.cell_id
        for c in schedule.study_matrix("phase16-dev", ["alfworld"], ["luna"], ["mirix"])
    ] == ["alfworld-luna-mirix-smoke-l39-t18"]
    with pytest.raises(ValueError, match="outside --study phase16-pilot"):
        schedule.study_matrix("phase16-pilot", ["alfworld"])
    with pytest.raises(ValueError, match="unknown study"):
        schedule.study_matrix("no-such-study")


def test_the_cli_takes_a_study(tmp_path, capsys):
    parse = schedule.build_parser().parse_args
    cells = schedule.cells_from_args(
        parse(["--study", "phase16", "--benchmark", "webshop"])
    )
    assert [c.cell_id for c in cells][:2] == [
        "webshop-luna-mirix",
        "webshop-luna56-mirix",
    ]
    assert len(cells) == 8
    # Without --study the matrix and its default split mode are as before.
    plain = schedule.cells_from_args(
        parse(["--benchmark", "alfworld", "--model", "luna56"])
    )
    assert {c.split_mode for c in plain} == {"protocol"} and len(plain) == 4
    for bad in (
        ["--study", "phase16", "--split-mode", "smoke"],
        ["--study", "phase16-dev", "--n-train", "3"],
        ["--study", "phase16-pilot", "--benchmark", "alfworld"],
    ):
        with pytest.raises(ValueError):
            schedule.cells_from_args(parse(bad))
    with pytest.raises(SystemExit):
        parse(["--study", "no-such-study"])
    assert "unknown study 'no-such-study'" in capsys.readouterr().err
    assert (
        schedule.main(
            ["--study", "phase16-pilot", "--dry-run", "--runs-dir", str(tmp_path)]
        )
        == 0
    )
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["cell"] for r in rows] == [
        "webshop-luna-none-pilot",
        "webshop-luna56-none-pilot",
    ]
    with pytest.raises(SystemExit) as refused:
        schedule.main(["--study", "phase16", "--split-mode", "smoke", "--dry-run"])
    assert refused.value.code == 2
    assert "sets the split mode and sizes itself" in capsys.readouterr().err


def test_concurrency_is_one_to_twenty_four():
    # protocol.SCHEDULE_CONCURRENCY_RANGE: an operating limit, touching no cell.
    for bad in (0, 25):
        with pytest.raises(ValueError):
            ScheduleOptions(concurrency=bad)
    assert ScheduleOptions(concurrency=1).concurrency == 1
    assert ScheduleOptions(concurrency=7).concurrency == 7
    assert ScheduleOptions(concurrency=24).concurrency == 24


def test_workers_get_run_py_arguments_and_no_credentials(tmp_path):
    opts = ScheduleOptions(runs_dir=tmp_path, test_concurrency=2, idle_timeout_s=60)
    spec = CellSpec("alfworld", "luna56", "mirix", "smoke", 3, 2)
    argv = worker_argv(spec, opts, "http://127.0.0.1:1", tmp_path / "ledgers")
    assert argv[:3] == [sys.executable, "-m", "evals.skill_bench.run"]
    args = argv[3:]
    assert args[: args.index("--out")] == [
        "--benchmark",
        "alfworld",
        "--model",
        "luna56",
        "--method",
        "mirix",
        "--split-mode",
        "smoke",
        "--n-train",
        "3",
        "--n-test",
        "2",
    ]
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--out"] == str(tmp_path / spec.cell_id)
    assert pairs["--gateway-url"] == "http://127.0.0.1:1"
    assert pairs["--gateway-runs-dir"] == str(tmp_path / "ledgers")
    assert pairs["--test-concurrency"] == "2" and pairs["--idle-timeout"] == "60"
    assert pairs["--max-learning-attempts"] == "1"
    protocol_argv = worker_argv(
        CellSpec("alfworld", "luna", "none"), opts, "u", tmp_path
    )
    assert "--n-train" not in protocol_argv
    env = worker_env(
        {
            "PATH": "/bin",
            "ALFWORLD_DATA": "/data",
            "OPENROUTER_API_KEY": "sk-or-x",
            "OPENAI_API_KEY": "sk-x",
            "HF_TOKEN": "hf_x",
            "AWS_SECRET_ACCESS_KEY": "x",
            "NO_PROXY": "corp",
        }
    )
    assert env["PATH"] == "/bin" and env["ALFWORLD_DATA"] == "/data"
    assert not {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"} & set(env)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["NO_PROXY"] == "corp,127.0.0.1,localhost,::1"


def test_every_path_option_is_made_absolute_once(tmp_path, monkeypatch):
    """Workers and a launched gateway run in the repository root, and a
    worker's MIRIX server in its attempt directory; a relative path must reach none."""
    monkeypatch.chdir(tmp_path)
    opts = ScheduleOptions(
        runs_dir=Path("runs/smoke"),
        gateway_runs_dir=Path("ledgers"),
        env_file=Path("key.env"),
    )
    cwd = Path.cwd()
    assert opts.runs_dir == cwd / "runs/smoke" and opts.control.is_absolute()
    assert opts.ledger_dir == cwd / "ledgers" and opts.env_file == cwd / "key.env"
    assert ScheduleOptions(runs_dir=Path("r")).ledger_dir == cwd / "r"
    spec = CellSpec("alfworld", "luna", "mirix", "smoke", 5, 2)
    args = worker_argv(spec, opts, "http://127.0.0.1:1", opts.ledger_dir)[3:]
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--out"] == str(cwd / "runs/smoke" / spec.cell_id)
    assert pairs["--gateway-runs-dir"] == str(cwd / "ledgers")


def test_cells_are_classified_without_automatic_reruns(tmp_path):
    opts = ScheduleOptions(runs_dir=tmp_path)
    cases = {
        None: QUEUED,
        "budget_paused": QUEUED,
        "complete": DONE,
        "refused": BLOCKED,
        "violation": BLOCKED,
        "test_incomplete": ATTENTION,
        "learning_failed": ATTENTION,
        "interrupted": ATTENTION,
        "crashed": ATTENTION,
        "running": ATTENTION,  # its worker died without a word
    }
    for status, expected in cases.items():
        paths = CellPaths(tmp_path / f"cell-{status}")
        if status:
            write_json_atomic(paths.status, {"status": status})
        assert classify_cell(paths, rerun_interrupted=False)[0] == expected, status
        rerun = classify_cell(paths, rerun_interrupted=True)[0]
        assert rerun == (QUEUED if expected == ATTENTION else expected), status
    paths = CellPaths(tmp_path / "live")
    paths.root.mkdir()
    lock = OwnerLock.acquire(paths.lock)
    try:
        assert classify_cell(paths, rerun_interrupted=False) == (LIVE, "running")
    finally:
        lock.release()
    assert opts.cell_dir(CellSpec("alfworld", "luna", "none")) == (
        tmp_path / "alfworld-luna-none"
    )


def test_a_harness_learning_failure_is_never_rerun_in_bulk(tmp_path):
    """``--rerun-interrupted`` continues interrupted cells and restarts
    learning lost to infrastructure, but a learning phase that failed for a HARNESS
    reason (the harness's or MIRIX's own failure, which a new attempt repeats and pays
    for again) waits for a person: only ``rerun_harness_failures`` launches it."""

    def classify(paths: CellPaths, **kw: bool) -> str:
        return classify_cell(paths, **{"rerun_interrupted": True, **kw})[0]

    harness = CellPaths(tmp_path / "harness")
    write_json_atomic(
        harness.status, {"status": "learning_failed", "failure_kind": "harness"}
    )
    assert classify_cell(harness, rerun_interrupted=False)[0] == ATTENTION
    assert classify(harness) == ATTENTION
    assert classify(harness, rerun_harness_failures=True) == QUEUED
    infra = CellPaths(tmp_path / "infra")
    write_json_atomic(
        infra.status, {"status": "learning_failed", "failure_kind": "infra"}
    )
    assert classify(infra) == QUEUED
    # A status written before it carried the kind: the last attempt's failure decides.
    older = CellPaths(tmp_path / "older")
    write_json_atomic(older.status, {"status": "learning_failed"})
    write_json_atomic(
        older.attempt(1).state,
        {"state": "invalid", "failure": {"kind": "infra"}},
    )
    write_json_atomic(
        older.attempt(2).state,
        {"state": "invalid", "failure": {"kind": "harness", "code": "mirix_api"}},
    )
    assert schedule.learning_failure_kind(older, {}) == "harness"
    assert classify(older) == ATTENTION
    assert classify(older, rerun_harness_failures=True) == QUEUED
    # Not a learning failure: --rerun-interrupted continues it as before.
    tested = CellPaths(tmp_path / "tested")
    write_json_atomic(
        tested.status, {"status": "test_incomplete", "failure_kind": "harness"}
    )
    assert classify(tested) == QUEUED


@pytest.mark.asyncio
async def test_rerun_interrupted_launches_no_cell_whose_learning_harness_failed(
    tmp_path, gateway
):
    """The same in the scheduler's loop: the cell is reported, never launched."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["mirix"])
    opts = options(tmp_path, url, rerun_interrupted=True)
    write_json_atomic(
        CellPaths(opts.cell_dir(cell)).status,
        {"status": "learning_failed", "failure_kind": "harness"},
    )
    result = await Scheduler([cell], opts).run()
    assert result.attention == {cell.cell_id: "learning_failed"}
    assert result.launched == [] and result.exit_code == 2


# --- scheduling -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_scheduler_runs_every_cell_once_within_the_concurrency(
    tmp_path, gateway, monkeypatch
):
    _gw, url = gateway
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-reach-workers")
    monkeypatch.setenv("HF_TOKEN", "hf_must_not_reach_workers")
    cells = matrix(["alfworld"], None, ["mirix", "awm", "none"])
    opts = options(tmp_path, url, concurrency=2)
    for spec in cells:
        behave(opts, spec, sleep_s=0.4)
    result = await Scheduler(cells, opts).run()
    assert result.exit_code == 0
    assert sorted(result.complete) == sorted(c.cell_id for c in cells)
    assert sorted(result.launched) == sorted(c.cell_id for c in cells)
    rows = worker_log(opts)
    assert sorted(r["cell"] for r in rows if r["event"] == "start") == sorted(
        c.cell_id for c in cells
    )
    assert max_overlap(rows) == 2
    for spec in cells:
        seen = read_json(opts.cell_dir(spec) / "env.json")
        assert seen["keys"] == []  # the gateway alone reads the key
        assert seen["gateway_url"] == url
        assert seen["gateway_runs_dir"] == str(opts.runs_dir)
    state = read_json(opts.control / "state.json")
    assert state["active"] == {} and state["gateway"]["mode"] == "attach"
    events = [e["event"] for e in read_jsonl(opts.control / "events.jsonl")]
    assert events[0] == "start" and events[-1] == "stop"
    assert events.count("launched") == events.count("exited") == len(cells)


@pytest.mark.asyncio
async def test_a_relative_runs_dir_is_one_directory_for_scheduler_and_workers(
    tmp_path, gateway, monkeypatch
):
    """Workers run in the repository root, so a relative ``--runs-dir`` given from
    another working directory must still name one directory, not two."""
    _gw, url = gateway
    monkeypatch.chdir(tmp_path)
    relative = Path(f"relative-runs-{uuid.uuid4().hex[:8]}")
    stray = protocol.REPO_ROOT / relative
    cells = matrix(["alfworld"], ["luna"], ["none", "awm"])
    opts = ScheduleOptions(
        runs_dir=relative,
        gateway_url=url,
        poll_s=0.05,
        balance_poll_s=0.0,
        worker_module=FAKE_WORKER,
    )
    try:
        result = await Scheduler(cells, opts).run()
        assert not stray.exists()  # no worker wrote below the repository root
    finally:
        shutil.rmtree(stray, ignore_errors=True)
    assert result.exit_code == 0
    assert sorted(result.complete) == sorted(c.cell_id for c in cells)
    for spec in cells:
        cell = tmp_path / relative / spec.cell_id
        assert read_json(cell / "status.json")["status"] == "complete"
        seen = read_json(cell / "env.json")
        assert seen["gateway_runs_dir"] == str(tmp_path / relative)
        assert seen["argv"][seen["argv"].index("--out") + 1] == str(cell)


@pytest.mark.asyncio
async def test_state_json_shows_the_live_queue_and_no_exit_code_while_running(
    tmp_path, gateway
):
    """Mid-run, state.json lists the queued and running cells and no exit code (not 0,
    "all complete") while a worker is still testing."""
    _gw, url = gateway
    cells = matrix(["alfworld"], ["luna"], ["none", "awm"])
    opts = options(tmp_path, url, concurrency=1)
    release = tmp_path / "release"
    behave(opts, cells[0], until=str(release), sleep_s=0.0)
    behave(opts, cells[1], sleep_s=0.0)
    scheduler = Scheduler(cells, opts)
    task = asyncio.ensure_future(scheduler.run())
    path = opts.control / "state.json"

    def running_first() -> bool:
        state = read_json(path) if path.is_file() else None
        return bool(state) and state["running"] == [cells[0].cell_id]

    try:
        await eventually(running_first, 30)
        state = read_json(path)
        assert state["queued"] == [cells[1].cell_id]
        assert state["active"] == {
            cells[0].cell_id: scheduler.active[cells[0].cell_id].pid
        }
        assert state["exit_code"] is None and state["complete"] == []
        assert state["launched"] == [cells[0].cell_id]
    finally:
        release.touch()
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0
    final = read_json(path)
    assert final["exit_code"] == 0 and final["queued"] == final["running"] == []
    assert sorted(final["complete"]) == sorted(c.cell_id for c in cells)
    assert final["active"] == {}


@pytest.mark.asyncio
async def test_failed_cells_are_reported_not_rerun(tmp_path, gateway):
    _gw, url = gateway
    base = matrix(["alfworld"], ["luna"])
    by = {c.method: c for c in base}
    # A learning cell that ends learning_failed (another model's AWM cell).
    (failing,) = matrix(["alfworld"], ["luna56"], ["awm"])
    cells = [*base, failing]
    opts = options(tmp_path, url, concurrency=3)
    set_status(opts, by["mirix"], "test_incomplete")
    set_status(opts, by["reasoningbank"], "refused")
    set_status(opts, by["awm"], "complete")
    set_status(opts, by["none"], "budget_paused")  # stopped before it started: launched
    behave(opts, failing, status="learning_failed")
    result = await Scheduler(cells, opts).run()
    assert sorted(result.launched) == sorted([by["none"].cell_id, failing.cell_id])
    assert result.complete == [by["awm"].cell_id, by["none"].cell_id]
    assert result.attention == {
        by["mirix"].cell_id: "test_incomplete",
        failing.cell_id: "learning_failed",
    }
    assert result.blocked == {by["reasoningbank"].cell_id: "refused"}
    assert result.exit_code == schedule.EXIT_ATTENTION
    starts = [r["cell"] for r in worker_log(opts) if r["event"] == "start"]
    assert starts.count(failing.cell_id) == 1  # no automatic re-run

    # Only an explicit --rerun-interrupted continues them; refused cells stay blocked.
    behave(opts, by["mirix"], status="complete")
    behave(opts, failing, status="complete")
    again = await Scheduler(cells, options(tmp_path, url, rerun_interrupted=True)).run()
    assert sorted(again.launched) == sorted([by["mirix"].cell_id, failing.cell_id])
    assert again.blocked == {by["reasoningbank"].cell_id: "refused"}


@pytest.mark.asyncio
async def test_a_low_balance_launches_nothing_until_it_recovers(tmp_path):
    fake = FakeUpstream(limit_remaining=2.0)  # below the gateway's 10 USD floor
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        opts = options(tmp_path, url)
        result = await Scheduler(cells, opts).run()
        assert result.paused and result.exit_code == schedule.EXIT_PAUSED
        assert result.launched == [] and worker_log(opts) == []
        assert result.queued == [c.cell_id for c in cells]
        state = read_json(opts.control / "state.json")
        assert state["budget"]["paused"] is True
        assert state["budget"]["remaining_usd"] == 2.0
        assert "budget_paused" in [
            e["event"] for e in read_jsonl(opts.control / "events.jsonl")
        ]
        # An unknown balance counts as low too.
        fake.limit_remaining = None
        assert (await Scheduler(cells, opts).run()).paused
        # Topped up: the next scheduler launches.
        fake.limit_remaining = 50.0
        resumed = await Scheduler(cells, opts).run()
    assert sorted(resumed.complete) == sorted(c.cell_id for c in cells)


@pytest.mark.asyncio
async def test_a_restarted_scheduler_adopts_live_workers(tmp_path, gateway):
    """PID takeover: a cell whose worker still runs is adopted, not started twice, and it
    counts against the concurrency."""
    _gw, url = gateway
    first, second = matrix(["alfworld"], ["luna"], ["mirix", "awm"])
    opts = options(tmp_path, url, concurrency=1)
    release = tmp_path / "release"
    behave(opts, first, until=str(release), sleep_s=0.0)
    behave(opts, second, sleep_s=0.0)
    orphan = subprocess.Popen(
        [
            sys.executable,
            "-m",
            FAKE_WORKER,
            "--out",
            str(opts.cell_dir(first)),
            "--gateway-url",
            url,
        ],
        cwd=protocol.REPO_ROOT,
        start_new_session=True,
    )
    try:
        paths = CellPaths(opts.cell_dir(first))
        await eventually(lambda: lock_is_held(paths.lock), 30)
        scheduler = Scheduler([first, second], opts)
        task = asyncio.ensure_future(scheduler.run())
        await eventually(lambda: first.cell_id in scheduler.result.adopted, 30)
        await asyncio.sleep(0.5)
        assert (
            scheduler.result.launched == []
        )  # the slot is taken by the adopted worker
        assert scheduler.active[first.cell_id].pid == orphan.pid
        release.touch()
        result = await asyncio.wait_for(task, 60)
    finally:
        release.touch()
        orphan.wait(timeout=30)
    assert result.adopted == [first.cell_id] and result.launched == [second.cell_id]
    assert sorted(result.complete) == sorted([first.cell_id, second.cell_id])
    starts = [r["cell"] for r in worker_log(opts) if r["event"] == "start"]
    assert starts == [first.cell_id, second.cell_id]


@pytest.mark.asyncio
async def test_one_scheduler_per_runs_directory(tmp_path, gateway):
    _gw, url = gateway
    opts = options(tmp_path, url)
    opts.control.mkdir(parents=True)
    lock = OwnerLock.acquire(opts.control / "scheduler.lock")
    try:
        with pytest.raises(ScheduleError, match="another scheduler"):
            await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_a_stopped_scheduler_leaves_workers_running(tmp_path, gateway):
    _gw, url = gateway
    cells = matrix(["alfworld"], ["luna"], ["none", "awm"])
    opts = options(tmp_path, url, concurrency=1)
    release = tmp_path / "release"
    behave(opts, cells[0], until=str(release), sleep_s=0.0)
    scheduler = Scheduler(cells, opts)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: cells[0].cell_id in scheduler.active, 30)
    worker = scheduler.active[cells[0].cell_id]
    scheduler.stop.set()
    result = await asyncio.wait_for(task, 30)
    assert result.stopped and result.running == [cells[0].cell_id]
    assert result.queued == [cells[1].cell_id]
    assert worker.popen is not None and worker.popen.poll() is None  # still running
    release.touch()
    assert worker.popen.wait(timeout=30) == 0


# --- the gateway ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_launched_gateway_is_adopted_after_its_scheduler_died(tmp_path):
    """A scheduler that dies leaves its gateway to the workers; the next one adopts it (no
    second gateway, no killed calls) and stops it when nothing uses it any more."""
    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=tmp_path / "runs",
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
        )
        launched = await ensure_gateway(opts)
        assert launched.mode == "launched" and launched.pid is not None
        assert launched.process is not None
        child = launched.process.process.popen
        closed = False
        try:
            with pytest.raises(ScheduleError, match="another live harness"):
                await ensure_gateway(opts)
            launched.release()  # the scheduler "dies": lock gone, gateway alive
            adopted = await ensure_gateway(opts)
            assert adopted.mode == "adopted" and adopted.pid == launched.pid
            await adopted.close()
            closed = True
        finally:
            if not closed:
                await launched.process.stop()
        # This test process is the gateway's parent: reap it (a dead scheduler's init does).
        assert await asyncio.to_thread(child.wait, 30) == -9
        pid = launched.pid
        assert process_start_time(pid) is None
        # Something else on the port is never adopted.
        opts_busy = ScheduleOptions(
            runs_dir=tmp_path / "other",
            gateway_port=int(upstream.rsplit(":", 1)[1]),
            gateway_upstream=upstream,
            env_file=env_file,
        )
        with pytest.raises(ScheduleError, match="not this runs directory"):
            await ensure_gateway(opts_busy)
    rows = read_jsonl(tmp_path / "runs" / "_gateway" / "owned_processes.jsonl")
    assert [r["event"] for r in rows] == ["started", "cleaned"]


@pytest.mark.asyncio
async def test_an_orphaned_gateway_that_live_cells_may_use_is_never_killed(tmp_path):
    """A new gateway on another port would clean every orphaned gateway of the runs
    directory; while cells are live, the scheduler refuses instead."""
    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    with fake.serve() as upstream:

        def at(port: int) -> ScheduleOptions:
            return ScheduleOptions(
                runs_dir=runs,
                gateway_port=port,
                gateway_upstream=upstream,
                env_file=env_file,
            )

        first = await ensure_gateway(at(find_free_port()))
        assert first.process is not None
        first.release()  # its scheduler died; the gateway is an orphan now
        cell = CellPaths(runs / "alfworld-luna-mirix")
        cell.root.mkdir(parents=True)
        live = OwnerLock.acquire(cell.lock)  # a worker that may still use it
        try:
            with pytest.raises(ScheduleError, match="orphaned gateway"):
                await ensure_gateway(at(find_free_port()))
            assert first.process.process.alive()
        finally:
            live.release()
        second = await ensure_gateway(at(find_free_port()))  # no live cell: clean it
        try:
            assert await asyncio.to_thread(first.process.process.popen.wait, 30) == -9
        finally:
            await second.close()
    rows = read_jsonl(runs / "_gateway" / "owned_processes.jsonl")
    assert [r["event"] for r in rows] == ["started", "cleaned", "started", "stopped"]


@pytest.mark.asyncio
async def test_a_scheduler_launches_its_own_gateway_and_stops_it_at_the_end(tmp_path):
    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=tmp_path / "runs",
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
            poll_s=0.05,
            balance_poll_s=0.0,
            worker_module=FAKE_WORKER,
        )
        result = await Scheduler(cells, opts).run()
    assert sorted(result.complete) == sorted(c.cell_id for c in cells)
    url = f"http://127.0.0.1:{opts.gateway_port}"
    for spec in cells:
        assert read_json(opts.cell_dir(spec) / "env.json")["gateway_url"] == url
    rows = read_jsonl(opts.runs_dir / "_gateway" / "owned_processes.jsonl")
    assert [r["event"] for r in rows] == ["started", "stopped"]
    assert read_json(opts.control / "state.json")["gateway"]["mode"] == "launched"
    assert fake.key_calls >= 1  # the budget check asked the (fake) upstream


@pytest.mark.asyncio
async def test_an_unhealthy_gateway_url_is_refused(tmp_path):
    opts = options(tmp_path, f"http://127.0.0.1:{find_free_port()}")
    with pytest.raises(ScheduleError, match="not healthy"):
        await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()


def test_dry_run_shows_the_plan(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_json_atomic(
        CellPaths(runs / "alfworld-luna-none").status, {"status": "complete"}
    )
    code = schedule.main(
        [
            "--dry-run",
            "--runs-dir",
            str(runs),
            "--benchmark",
            "alfworld",
            "--method",
            "none",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        {"cell": "alfworld-luna-none", "plan": "done", "status": "complete"},
        {"cell": "alfworld-luna56-none", "plan": "queued", "status": None},
    ]
    assert not (runs / "_schedule").exists()


# --- a method removed from the study ----------------------------------------------------------
# ``study.REMOVED_METHODS`` names no method in this release; these tests install one of their
# own (conftest's ``removed_method`` fixture).


def test_a_removed_method_is_refused_by_name(removed_method):
    """A method that left the study: naming it is refused with the decision, alone or
    beside study methods (a selection is never silently narrowed); a typo is still a
    typo, and the default is every study method."""
    removal = study.REMOVED_METHODS[removed_method]
    assert set(schedule.METHOD_ORDER) == set(protocol.METHODS)
    assert removed_method not in schedule.METHOD_ORDER
    for methods in ([removed_method], ["none", removed_method]):
        with pytest.raises(ValueError, match=removal.decision) as exc:
            matrix(methods=methods)
        assert removed_method in str(exc.value) and removal.reason in str(exc.value)
    with pytest.raises(ValueError, match=removal.decision):
        matrix(
            ["alfworld"],
            ["luna"],
            [removed_method],
            split_mode="smoke",
            n_train=1,
            n_test=1,
        )
    with pytest.raises(ValueError, match="unknown methods"):
        matrix(methods=["exprag"])
    for methods in (None, []):
        assert {c.method for c in matrix(methods=methods)} == set(protocol.METHODS)


@pytest.mark.parametrize(
    "extra",
    [
        ["--method", "{m}"],
        ["--method", "none", "{m}"],
        ["--method", "{m}", "--dry-run"],
        ["--rerun-harness-failures", "alfworld-luna56-{m}", "--dry-run"],
        ["--reset-resume", "alfworld-luna-{m}"],
    ],
)
def test_the_cli_refuses_a_removed_method_with_the_decision(
    tmp_path, capsys, extra, removed_method
):
    """Exit 2 before anything starts or is written, and the message names the decision
    (never only argparse's "invalid choice" or "outside the selection")."""
    decision = study.REMOVED_METHODS[removed_method].decision
    extra = [arg.format(m=removed_method) for arg in extra]
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit) as exc:
        schedule.main(["--runs-dir", str(runs), "--benchmark", "alfworld", *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert removed_method in err and decision in err and "removed from the study" in err
    assert "invalid choice" not in err
    assert not runs.exists()


def _leftover_removed_cells(runs: Path, method: str) -> dict[Path, bytes]:
    """Cell directories of the removed ``method`` as an earlier study left them: one
    complete, one stopped mid-learning (``interrupted``) with a stale lock and process
    record, and its auto-resume entry; returns every file with its bytes."""
    for model, status in (("luna", "complete"), ("luna56", "interrupted")):
        cell = runs / f"alfworld-{model}-{method}"
        write_json_atomic(
            CellPaths(cell).manifest,
            {"cell": {"benchmark": "alfworld", "model": model, "method": method}},
        )
        write_json_atomic(CellPaths(cell).status, {"status": status})
        (cell / "cell.lock").write_text("")
        write_json_atomic(
            cell / "process.json", {"pid": 2**22 + 7, "started": "Mon Sep 28 01:02:03"}
        )
        (cell / "results.jsonl").write_text('{"raw_id": "q1"}\n')
    write_json_atomic(
        runs / "_schedule" / "resume_state.json",
        {f"alfworld-luna56-{method}": {"resumes": 1, "without_progress": 0}},
    )
    return {
        p: p.read_bytes()
        for p in sorted(runs.rglob("*"))
        if p.is_file() and method in p.as_posix()
    }


def test_leftover_cells_of_a_removed_method_are_never_planned(
    tmp_path, capsys, removed_method
):
    runs = tmp_path / "runs"
    before = _leftover_removed_cells(runs, removed_method)
    for method in ([], ["--method"]):
        code = schedule.main(
            [
                "--dry-run",
                "--auto-resume",
                "--runs-dir",
                str(runs),
                "--benchmark",
                "alfworld",
                *method,
            ]
        )
        assert code == 0
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [r["cell"] for r in rows] == [c.cell_id for c in matrix(["alfworld"])]
        assert len(rows) == 4 * len(protocol.MODELS)
        assert not [r for r in rows if removed_method in r["cell"]]
    # The runs-directory scans read them as any cell directory, without a crash: no
    # live worker (the lock is free), nothing to sweep, no gateway.
    assert live_cells(runs) == [] and live_cell_workers(runs) == []
    assert sweep_owned(runs) == {} and schedule.live_gateways(runs) == []
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.asyncio
async def test_a_scheduler_beside_a_removed_methods_cells_leaves_them_alone(
    tmp_path, gateway, removed_method
):
    """A runs directory holding the removed method's cells (one of them resumable):
    the auto-resuming scheduler runs the study cells to completion, never launches,
    resumes or counts the removed ones, and keeps their files and resume entry."""
    _gw, url = gateway
    opts = options(tmp_path, url, auto_resume=True, resume_backoff_base_s=0.02)
    before = _leftover_removed_cells(opts.runs_dir, removed_method)
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    result = await asyncio.wait_for(Scheduler(cells, opts).run(), 60)
    assert result.exit_code == 0
    assert sorted(result.complete) == sorted(c.cell_id for c in cells)
    assert not [c for c in result.launched if removed_method in c]
    assert not [r for r in worker_log(opts) if removed_method in r["cell"]]
    assert {p: p.read_bytes() for p in before if p.name != "resume_state.json"} == {
        p: b for p, b in before.items() if p.name != "resume_state.json"
    }
    kept = read_json(opts.control / "resume_state.json")
    assert kept[f"alfworld-luna56-{removed_method}"] == {
        "resumes": 1,
        "without_progress": 0,
    }
    state = json.dumps(read_json(opts.control / "state.json"))
    assert removed_method not in state


# --- budgets and gateways ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_budget_the_launched_gateway_has_spent_launches_nothing(tmp_path):
    """``--budget`` reaches the gateway the scheduler launches. Its runs directory has
    spent the budget (the key's usage grew by it since the budget was set), so the
    gateway pauses and the scheduler launches no cell and exits paused."""
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 20.0
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    write_json_atomic(
        runs / "_gateway" / "budget.json",
        {"start_usage_usd": 11.5, "started_at": "then", "budgets": []},
    )
    cells = matrix(["alfworld"], ["luna56"], ["none"])
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=runs,
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
            budget_usd=8.0,
            poll_s=0.05,
            balance_poll_s=0.0,
            worker_module=FAKE_WORKER,
        )
        result = await Scheduler(cells, opts).run()
    assert result.paused and result.exit_code == schedule.EXIT_PAUSED
    assert result.launched == [] and result.queued == [cells[0].cell_id]
    budget = read_json(opts.control / "state.json")["budget"]
    assert budget["pause_reason"] == "budget_spent"
    assert budget["spend"]["spent_usd"] == 8.5 and budget["spend"]["budget_usd"] == 8.0
    record = read_json(runs / "_gateway" / "budget.json")
    assert [b["budget_usd"] for b in record["budgets"]] == [8.0]
    # A later step whose command lost --budget is refused, not run without the cap.
    unbudgeted = ScheduleOptions(
        runs_dir=runs,
        gateway_port=find_free_port(),
        gateway_upstream="http://127.0.0.1:9",
        env_file=env_file,
        poll_s=0.05,
        worker_module=FAKE_WORKER,
    )
    with pytest.raises(ScheduleError, match="sets a spend budget"):
        await Scheduler(cells, unbudgeted).run()


@pytest.mark.asyncio
async def test_a_budget_the_attached_gateway_does_not_run_is_refused(tmp_path, gateway):
    """``--budget`` would silently not apply to a gateway the scheduler did not launch."""
    _gw, url = gateway
    opts = options(tmp_path, url, budget_usd=8.0)
    with pytest.raises(ScheduleError, match="runs with budget None, not --budget 8"):
        await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()
    fake = FakeUpstream()
    with conformant_gateway(tmp_path / "b", fake, budget_usd=8.0) as (_gw2, url2):
        result = await Scheduler(
            matrix(["alfworld"], ["luna"], ["none"]),
            options(tmp_path / "b", url2, budget_usd=8.0),
        ).run()
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_replicates_share_one_budget_start_and_its_total(tmp_path):
    """Replicate runs directories may use one key. A key-basis budget
    counts the key's whole usage since its start, so r2 started from r1's budget
    record (--budget-start-from) pauses when the key has spent --budget since r1's
    start, whoever spent it: --budget is the replicates' total."""
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 20.0
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    r1, r2 = tmp_path / "phase16-r1", tmp_path / "phase16-r2"
    write_json_atomic(
        r1 / "_gateway" / "budget.json",
        {"start_usage_usd": 11.5, "started_at": "then", "budgets": []},
    )
    cells = matrix(["alfworld"], ["luna"], ["none"])
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=r2,
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
            budget_usd=8.0,
            budget_start_from=r1,
            poll_s=0.05,
            balance_poll_s=0.0,
            worker_module=FAKE_WORKER,
        )
        result = await Scheduler(cells, opts).run()
    assert result.paused and result.exit_code == schedule.EXIT_PAUSED
    assert result.launched == []
    budget = read_json(opts.control / "state.json")["budget"]
    assert budget["pause_reason"] == "budget_spent"
    assert budget["spend"]["start_usage_usd"] == 11.5
    assert budget["spend"]["spent_usd"] == 8.5
    record = read_json(r2 / "_gateway" / "budget.json")
    assert record["start_usage_usd"] == 11.5 and record["started_at"] == "then"
    assert record[schedule.BUDGET_SHARED_FROM] == str(r1 / "_gateway" / "budget.json")
    (shared,) = events_of(opts, "budget_start_shared")
    assert shared["shared"] == "copied" and shared["start"] == 11.5


def test_sharing_a_budget_start_copies_once_and_records_a_difference(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    opts = ScheduleOptions(runs_dir=r2, budget_usd=100.0, budget_start_from=r1)
    assert schedule.share_budget_start(ScheduleOptions(runs_dir=r2)) is None
    # r1's gateway has not written its record yet: refused (the supervisor retries).
    with pytest.raises(ScheduleError, match="start the scheduler of"):
        schedule.share_budget_start(opts)
    assert not (r2 / "_gateway" / "budget.json").exists()
    write_json_atomic(
        r1 / "_gateway" / "budget.json",
        {"start_usage_usd": 3.25, "started_at": "t1", "budgets": [{"budget_usd": 100}]},
    )
    first = schedule.share_budget_start(opts)
    assert first is not None and first["shared"] == "copied"
    again = schedule.share_budget_start(opts)
    assert again is not None and again["shared"] == "same"
    # A record of its own is never overwritten: a different start is recorded.
    write_json_atomic(
        r2 / "_gateway" / "budget.json",
        {"start_usage_usd": 9.0, "started_at": "t2", "budgets": []},
    )
    differs = schedule.share_budget_start(opts)
    assert differs is not None and differs["shared"] == "differs"
    assert differs["start"] == 9.0 and differs["source_start"] == 3.25
    assert read_json(r2 / "_gateway" / "budget.json")["start_usage_usd"] == 9.0
    (r1 / "_gateway" / "budget.json").write_text("{")
    broken = schedule.share_budget_start(opts)
    assert broken is not None and broken["shared"] == "unreadable"
    # The option needs --budget and another directory.
    with pytest.raises(ValueError, match="needs --budget"):
        ScheduleOptions(runs_dir=r2, budget_start_from=r1)
    with pytest.raises(ValueError, match="names this runs directory"):
        ScheduleOptions(runs_dir=r2, budget_usd=1.0, budget_start_from=r2 / ".")
    args = schedule.build_parser().parse_args(
        ["--budget", "100", "--budget-start-from", str(r1)]
    )
    assert schedule.options_from_args(args).budget_start_from == r1.absolute()


def test_a_budget_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="budget must be positive"):
        ScheduleOptions(runs_dir=tmp_path, budget_usd=0.0)


@pytest.mark.asyncio
async def test_an_adopted_gateway_live_workers_use_is_never_stopped(tmp_path):
    """Another batch's run.py workers hold the gateway's users lock; the adopting
    scheduler's close() only lets go of it, and a gateway launch on another port leaves
    it alone. Once no worker uses it, the next close() stops it."""
    from evals.skill_bench.infra.gateway import gateway_users_lock, launch_gateway

    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=tmp_path / "runs",
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
        )
        launched = await ensure_gateway(opts)
        assert launched.process is not None
        child = launched.process.process.popen
        users_path = gateway_users_lock(opts.runs_dir, "127.0.0.1", opts.gateway_port)
        assert launched.users_lock == users_path
        worker = OwnerLock.acquire(users_path, shared=True)  # a live run.py
        other = None
        try:
            launched.release()  # its scheduler dies
            adopted = await ensure_gateway(opts)
            assert adopted.mode == "adopted" and adopted.in_use()
            await adopted.close()
            assert child.poll() is None  # still serving the worker
            # run.py in launch mode on another port cleans orphans: not this one.
            other = await launch_gateway(
                runs_dir=opts.runs_dir,
                port=find_free_port(),
                env_file=env_file,
                upstream=upstream,
            )
            assert child.poll() is None
            worker.release()
            again = await ensure_gateway(opts)
            assert not again.in_use()
            await again.close()
            assert await asyncio.to_thread(child.wait, 30) == -9
        finally:
            worker.release()
            if other is not None:
                await other.stop()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)


@pytest.mark.asyncio
async def test_an_attaching_scheduler_never_sweeps_the_gateway_registry(
    tmp_path, gateway
):
    """With --gateway-url the scheduler cleans crashed cells' children but never a
    gateway registered in runs/_gateway (not its gateway to stop)."""
    from evals.skill_bench.infra.processes import OwnedProcess

    _gw, url = gateway
    opts = options(tmp_path, url)
    orphan = OwnedProcess.spawn(
        "gateway",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin"},
        cwd=Path.cwd(),
        log_path=tmp_path / "g.log",
        registry=opts.runs_dir / "_gateway" / "owned_processes.jsonl",
    )
    try:
        result = await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()
        assert result.exit_code == 0
        assert orphan.popen.poll() is None
        events = [e["event"] for e in read_jsonl(opts.control / "events.jsonl")]
        assert "sweep" not in events
    finally:
        orphan.popen.kill()
        orphan.popen.wait(timeout=30)


@pytest.mark.asyncio
async def test_a_budget_paused_worker_is_launched_again_when_the_balance_allows(
    tmp_path, gateway
):
    """A worker that found the gateway paused ran nothing; the scheduler queues the
    cell again and launches it once the balance allows ("pause, then continue")."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(tmp_path, url)
    behave(opts, cell, statuses=["budget_paused", "complete"], sleep_s=0.0)
    result = await Scheduler([cell], opts).run()
    assert result.exit_code == 0
    assert result.complete == [cell.cell_id] and result.launched == [cell.cell_id]
    assert result.attention == {}
    starts = [r for r in worker_log(opts) if r["event"] == "start"]
    assert len(starts) == 2


@pytest.mark.asyncio
async def test_a_gateway_floor_of_zero_is_zero(tmp_path):
    """The gateway's balance floor 0 is not the 10 USD default."""
    fake = FakeUpstream(limit_remaining=2.0)
    with conformant_gateway(tmp_path, fake, balance_floor_usd=0.0) as (_gw, url):
        opts = options(tmp_path, url)
        result = await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()
    assert not result.paused and result.exit_code == 0
    state = read_json(opts.control / "state.json")
    assert state["budget"]["floor_usd"] == 0.0


# --- automatic resume ------------------------------------------------------------------------------


def progress(
    ordinal: int | None, *, attempt: int = 1, scored: int = 0, state: str = "learning"
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "state": state,
        "checkpoint_ordinal": ordinal,
        "learning_done": 0 if ordinal is None else ordinal // 2,
        "test_scored": scored,
    }


@pytest.fixture
def progress_file(monkeypatch):
    """cell_progress reads what the fake worker wrote to <cell>/progress.json."""

    def fake_progress(paths: CellPaths) -> Any:
        return read_json(paths.root / "progress.json")

    monkeypatch.setattr(schedule, "cell_progress", fake_progress)


def events_of(opts: ScheduleOptions, name: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(opts.control / "events.jsonl")
    return [r for r in rows if name is None or r["event"] == name]


def starts_of(opts: ScheduleOptions, spec: CellSpec) -> list[dict[str, Any]]:
    return [
        r
        for r in worker_log(opts)
        if r["event"] == "start" and r["cell"] == spec.cell_id
    ]


def tracker(tmp_path: Path, **kw: Any) -> schedule.ResumeTracker:
    kw.setdefault("base_s", 60.0)
    kw.setdefault("cap_s", 1800.0)
    return schedule.ResumeTracker(tmp_path / "resume_state.json", **kw)


def test_resume_backoff_doubles_to_its_cap_and_progress_resets_it(tmp_path):
    t = tracker(tmp_path, max_without_progress=10, max_resumes=100)
    same = progress(3)
    t.note_launch("c", same, at=0.0)
    delays = []
    for i in range(7):
        d = t.note_exit(
            "c",
            status="learning_interrupted",
            exit_code=8,
            progress=same,
            marker=f"m{i}",
            at=1000.0 * i,
        )
        assert d.new and d.attention is None and d.delay_s is not None
        assert d.next_at == pytest.approx(1000.0 * i + d.delay_s)
        delays.append(d.delay_s)
    assert delays == [60, 120, 240, 480, 960, 1800, 1800]
    entry = t.cells["c"]
    assert (entry["resumes"], entry["without_progress"], entry["backoff_index"]) == (
        7,
        7,
        6,
    )
    # Progress (a new checkpoint) resets the backoff; the history keeps the last 20.
    d = t.note_exit(
        "c",
        status="learning_interrupted",
        exit_code=8,
        progress=progress(4),
        marker="m7",
        at=9000.0,
    )
    assert d.delay_s == 60 and t.cells["c"]["without_progress"] == 0
    assert t.cells["c"]["history"][-1]["progressed"] is True
    # Persisted: a restarted scheduler reads the same counters.
    again = tracker(tmp_path)
    again.load()
    assert again.cells["c"]["resumes"] == 8 and again.cells["c"][
        "next_at"
    ] == schedule.iso(9060.0)


def test_resume_caps_send_a_cell_to_a_person(tmp_path):
    t = tracker(tmp_path)  # 5 without progress, 20 in all
    t.note_launch("c", progress(1), at=0)
    for i in range(5):
        d = t.note_exit(
            "c",
            status="test_incomplete",
            exit_code=2,
            progress=progress(1),
            marker=f"m{i}",
            at=i,
        )
        assert d.attention is None, i
    d = t.note_exit(
        "c",
        status="test_incomplete",
        exit_code=2,
        progress=progress(1),
        marker="m5",
        at=9,
    )
    assert d.attention == schedule.REASON_RESUME_CAP and d.next_at is None
    assert t.decision("c").attention == schedule.REASON_RESUME_CAP  # sticky until reset
    # crashed gets 2.
    t.note_launch("k", progress(1), at=0)
    outcomes = [
        t.note_exit(
            "k",
            status="crashed",
            exit_code=1,
            progress=progress(1),
            marker=f"k{i}",
            at=i,
        ).attention
        for i in range(3)
    ]
    assert outcomes == [None, None, schedule.REASON_RESUME_CAP]
    # The lifetime cap counts resumes that made progress too.
    t2 = tracker(tmp_path / "b", max_resumes=3)
    t2.note_launch("p", progress(0), at=0)
    caps = [
        t2.note_exit(
            "p",
            status="learning_interrupted",
            exit_code=8,
            progress=progress(i + 1),
            marker=f"p{i}",
            at=i,
        ).attention
        for i in range(4)
    ]
    assert caps == [None, None, None, schedule.REASON_RESUME_CAP_TOTAL]


def test_an_exit_is_counted_once_and_a_scheduler_stop_is_not_a_failure(tmp_path):
    t = tracker(tmp_path)
    t.note_launch("c", progress(1), at=0)
    first = t.note_exit(
        "c",
        status="interrupted",
        exit_code=130,
        progress=progress(1),
        marker="same",
        at=10,
    )
    again = t.note_exit(
        "c",
        status="interrupted",
        exit_code=130,
        progress=progress(1),
        marker="same",
        at=99,
    )
    assert first.new and not again.new
    assert again.next_at == first.next_at and t.cells["c"]["resumes"] == 1
    # The memory emergency stop: resumed after base, no count without progress.
    stopped = t.note_exit(
        "c",
        status="interrupted",
        exit_code=130,
        progress=progress(1),
        marker="other",
        at=20,
        counts=False,
        reason=schedule.STOPPED_MEMORY_EMERGENCY,
    )
    assert stopped.delay_s == 60 and t.cells["c"]["without_progress"] == 1
    assert t.cells["c"]["resumes"] == 2
    assert t.cells["c"]["history"][-1]["reason"] == schedule.STOPPED_MEMORY_EMERGENCY
    # A reset (a person looked) clears the counters and makes the cell due at once.
    for i in range(6):
        t.note_exit(
            "c",
            status="interrupted",
            exit_code=130,
            progress=progress(1),
            marker=f"x{i}",
            at=30 + i,
        )
    assert t.decision("c").attention == schedule.REASON_RESUME_CAP
    assert t.reset(["c", "unknown"], at=50) == ["c"]
    cleared = t.decision("c")
    assert cleared.attention is None and cleared.next_at is None
    assert (t.cells["c"]["resumes"], t.cells["c"]["without_progress"]) == (0, 0)


def test_progress_is_a_new_checkpoint_more_scored_items_or_a_later_state():
    grew = schedule.progress_grew
    assert grew(progress(3), progress(4))
    assert not grew(progress(4), progress(4))
    assert not grew(progress(5), progress(4))  # a rollback to an older checkpoint
    assert grew(
        progress(9, state="testing", scored=3), progress(9, state="testing", scored=4)
    )
    assert grew(progress(9), progress(9, state="frozen"))
    # A new attempt is progress once it got past its setup, not before (one
    # an infrastructure failure stops in its setup, again and again, reaches the cap).
    assert not grew(progress(9, attempt=1), progress(None, attempt=2))
    assert not grew(progress(9, attempt=1), progress(0, attempt=2))  # init only
    assert grew(progress(9, attempt=1), progress(1, attempt=2))  # item 0 played
    assert grew(progress(9, attempt=1), progress(None, attempt=2, state="testing"))
    assert grew(progress(9, attempt=1), progress(None, attempt=2, scored=1))
    assert not grew(progress(3, attempt=2), progress(9, attempt=1))  # not older ones
    assert grew({"attempt": None}, progress(1))  # a cell's first attempt
    assert not grew({"attempt": None}, progress(None))  # stopped in its setup
    assert grew(None, progress(0))  # no baseline: the benefit of the doubt
    assert not grew(progress(1), None)  # an unreadable cell made none


def test_auto_resume_classification(tmp_path):
    cases = {
        None: QUEUED,
        "budget_paused": QUEUED,
        "complete": DONE,
        "refused": BLOCKED,
        "violation": BLOCKED,
        "learning_interrupted": schedule.WAITING,
        "test_incomplete": schedule.WAITING,
        "infra_unavailable": schedule.WAITING,
        "interrupted": schedule.WAITING,
        "crashed": schedule.WAITING,
        "running": schedule.WAITING,  # its worker died
        "learning_failed": QUEUED,  # no checkpoints: run.py starts a new attempt
    }
    for status, expected in cases.items():
        paths = CellPaths(tmp_path / f"cell-{status}")
        if status:
            write_json_atomic(paths.status, {"status": status})
        got = classify_cell(paths, rerun_interrupted=False, auto_resume=True)[0]
        assert got == expected, status
        # Without auto-resume, learning_interrupted is reported like the others.
        if status == "learning_interrupted":
            assert classify_cell(paths, rerun_interrupted=False)[0] == ATTENTION
            assert classify_cell(paths, rerun_interrupted=True)[0] == QUEUED
    harness = CellPaths(tmp_path / "harness")
    write_json_atomic(
        harness.status, {"status": "learning_failed", "failure_kind": "harness"}
    )
    assert (
        classify_cell(harness, rerun_interrupted=False, auto_resume=True)[0]
        == ATTENTION
    )
    assert (
        classify_cell(
            harness,
            rerun_interrupted=False,
            auto_resume=True,
            rerun_harness_failures=True,
        )[0]
        == QUEUED
    )


@pytest.mark.asyncio
async def test_auto_resume_reruns_every_resumable_ending_until_complete(
    tmp_path, gateway, progress_file
):
    """Each auto-resumable status, a worker killed mid-run and one that died before its
    first status are re-queued by themselves (backoff base here: 0.05 s) and finish."""
    _gw, url = gateway
    endings = [
        "learning_interrupted",
        "test_incomplete",
        "infra_unavailable",
        "interrupted",
        "crashed",
    ]
    cells = matrix(["alfworld"], None, None)[:7]
    opts = options(
        tmp_path,
        url,
        concurrency=6,
        auto_resume=True,
        resume_backoff_base_s=0.05,
        resume_backoff_cap_s=0.2,
    )
    for spec, ending in zip(cells, endings):
        behave(
            opts,
            spec,
            statuses=[ending, "complete"],
            progress=[progress(3), progress(5)],
            sleep_s=0.0,
        )
    died, crashed_at_start = cells[5], cells[6]
    behave(opts, died, die=[1, None], progress=progress(2), sleep_s=0.0)
    behave(opts, crashed_at_start, crash_at_start=[1, None], sleep_s=0.0)
    result = await Scheduler(cells, opts).run()
    assert result.exit_code == 0, result.to_dict()
    assert sorted(result.complete) == sorted(c.cell_id for c in cells)
    for spec in cells:
        assert read_json(opts.cell_dir(spec) / "launches_seen.json")["n"] == 2, (
            spec.cell_id
        )
    state = read_json(opts.control / "resume_state.json")
    labels = {c: state[c]["history"][0]["status"] for c in state}
    assert [labels[c.cell_id] for c in cells[:5]] == endings
    assert (
        labels[died.cell_id]
        == labels[crashed_at_start.cell_id]
        == schedule.STATUS_WORKER_DIED
    )
    assert all(state[c.cell_id]["resumes"] == 1 for c in cells)
    scheduled = events_of(opts, "resume_scheduled")
    assert sorted(e["cell"] for e in scheduled) == sorted(c.cell_id for c in cells)
    assert {e["delay_s"] for e in scheduled} == {0.05}
    final = read_json(opts.control / "state.json")
    assert final["waiting"] == {} and final["auto_resume"] is True
    assert final["resume"][cells[0].cell_id]["resumes"] == 1


@pytest.mark.asyncio
async def test_a_cell_without_progress_goes_to_a_person_after_the_cap(
    tmp_path, gateway, progress_file
):
    _gw, url = gateway
    stuck, fine = matrix(["alfworld"], ["luna"], ["mirix", "none"])
    opts = options(
        tmp_path,
        url,
        concurrency=2,
        auto_resume=True,
        resume_backoff_base_s=0.02,
        resume_backoff_cap_s=0.05,
        max_resumes_without_progress=2,
    )
    behave(
        opts, stuck, status="learning_interrupted", progress=progress(3), sleep_s=0.0
    )
    behave(opts, fine, sleep_s=0.0)
    result = await Scheduler([stuck, fine], opts).run()
    assert result.exit_code == schedule.EXIT_ATTENTION
    assert result.complete == [fine.cell_id]
    assert result.attention == {stuck.cell_id: "learning_interrupted"}
    assert result.attention_reasons == {stuck.cell_id: schedule.REASON_RESUME_CAP}
    # The first launch set the baseline, the first exit made progress; then 2 more
    # resumes without progress, and the third exit without progress hit the cap.
    assert len(starts_of(opts, stuck)) == 4
    assert [e["reason"] for e in events_of(opts, "resume_cap")] == [
        schedule.REASON_RESUME_CAP
    ]
    # A person looks and resets it: the next scheduler resumes it at once.
    behave(opts, stuck, status="complete", sleep_s=0.0)
    again = await Scheduler(
        [stuck, fine], replace(opts, reset_resume=(stuck.cell_id,))
    ).run()
    assert again.exit_code == 0 and stuck.cell_id in again.complete
    assert events_of(opts, "resume_reset")[-1]["cells"] == [stuck.cell_id]


@pytest.mark.asyncio
async def test_a_signalled_worker_never_counts_towards_the_no_progress_cap(
    tmp_path, gateway, progress_file
):
    """``interrupted`` is a SIGTERM/SIGINT from outside (the supervisor's drain, a
    reboot, an operator): however often, it never sends the cell to a person."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["reasoningbank"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=0.02,
        max_resumes_without_progress=1,
    )
    behave(
        opts,
        cell,
        statuses=["interrupted"] * 4 + ["complete"],
        status_extra={"signal": "SIGTERM"},  # as run.py records it
        progress=progress(3),
        sleep_s=0.0,
    )
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert entry["resumes"] == 4 and entry["without_progress"] == 0
    assert {h.get("reason") for h in entry["history"]} == {"signal"}
    assert {e["delay_s"] for e in events_of(opts, "resume_scheduled")} == {0.02}


@pytest.mark.asyncio
async def test_an_interruption_nobody_signalled_counts_like_a_failure(
    tmp_path, gateway, progress_file
):
    """run.py writes ``interrupted`` for any cancellation of its main task; one
    without a recorded signal is a harness bug (a leaked cancellation), not someone
    else's stop: it counts towards the no-progress cap, with backoff."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["reasoningbank"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=0.02,
        max_resumes_without_progress=1,
    )
    behave(
        opts,
        cell,
        statuses=["interrupted"] * 4 + ["complete"],
        progress=progress(3),
        sleep_s=0.0,
    )
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
    entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert entry["without_progress"] == 2
    assert not {h.get("reason") for h in entry["history"]} - {None}


@pytest.mark.asyncio
async def test_resume_state_survives_a_scheduler_restart(
    tmp_path, gateway, progress_file
):
    """A cell waiting out a long backoff: the scheduler stops; the next one counts the
    same ending once, and resumes the cell when its wake-up has come."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["reasoningbank"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=3600.0,
        resume_backoff_cap_s=3600.0,
    )
    behave(
        opts,
        cell,
        statuses=["learning_interrupted", "complete"],
        progress=progress(4),
        sleep_s=0.0,
    )
    first = Scheduler([cell], opts)
    task = asyncio.ensure_future(first.run())
    path = opts.control / "state.json"
    await eventually(lambda: path.is_file() and bool(read_json(path)["waiting"]), 30)
    state = read_json(path)
    first.stop.set()
    result = await asyncio.wait_for(task, 30)
    assert result.stopped and result.exit_code == schedule.EXIT_PAUSED
    assert list(result.waiting) == [cell.cell_id]
    saved = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert saved["resumes"] == 1 and saved["last_status"] == "learning_interrupted"
    assert state["waiting"] == {cell.cell_id: saved["next_at"]}
    # Too early for a restarted scheduler too: it waits (and is stopped again).
    early = Scheduler([cell], opts)
    task = asyncio.ensure_future(early.run())
    await eventually(lambda: cell.cell_id in early.waiting, 30)
    await asyncio.sleep(0.3)
    assert len(starts_of(opts, cell)) == 1
    early.stop.set()
    await asyncio.wait_for(task, 30)
    # Two hours later (a fake clock): due, resumed, complete; counted once.
    later = Scheduler([cell], opts, clock=lambda: time.time() + 7200)
    result = await asyncio.wait_for(later.run(), 60)
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    assert read_json(opts.control / "resume_state.json")[cell.cell_id]["resumes"] == 1
    assert len(events_of(opts, "resume_scheduled")) == 1


@pytest.mark.asyncio
async def test_harness_failures_blocked_cells_and_new_attempts(
    tmp_path, gateway, progress_file
):
    """HARNESS learning failures, refused and violation cells are never re-queued and
    never hold the others back; --rerun-harness-failures relaunches a harness failure
    once, with --new-attempt, and a repeat waits for the person again."""
    _gw, url = gateway
    harness, refused, violation, fine = matrix(
        ["alfworld"], ["luna56"], ["mirix", "reasoningbank", "awm", "none"]
    )
    opts = options(
        tmp_path, url, concurrency=2, auto_resume=True, resume_backoff_base_s=0.02
    )
    write_json_atomic(
        CellPaths(opts.cell_dir(harness)).status,
        {"status": "learning_failed", "failure_kind": "harness", "updated_at": "t0"},
    )
    set_status(opts, refused, "refused")
    set_status(opts, violation, "violation")
    behave(opts, fine, sleep_s=0.0)
    result = await Scheduler([harness, refused, violation, fine], opts).run()
    assert result.exit_code == schedule.EXIT_ATTENTION
    assert result.launched == [fine.cell_id] and result.complete == [fine.cell_id]
    assert result.attention_reasons == {
        harness.cell_id: schedule.REASON_HARNESS_FAILURE
    }
    assert result.blocked == {
        refused.cell_id: "refused",
        violation.cell_id: "violation",
    }
    # A person looked: one new attempt, and the same failure again goes back to them.
    behave(
        opts,
        harness,
        status="learning_failed",
        status_extra={"failure_kind": "harness"},
        sleep_s=0.0,
    )
    again = await Scheduler([harness], replace(opts, rerun_harness_failures=True)).run()
    assert again.launched == [harness.cell_id]
    assert again.attention_reasons == {harness.cell_id: schedule.REASON_HARNESS_FAILURE}
    (start,) = starts_of(opts, harness)
    assert start["rest"][-1] == "--new-attempt"
    # A resume of that new attempt carries no --new-attempt (it would learn again).
    # (The fake worker counts launches across schedulers: its first was the one above.)
    behave(
        opts,
        harness,
        statuses=["learning_failed", "learning_interrupted", "complete"],
        progress=progress(2),
        sleep_s=0.0,
    )
    write_json_atomic(
        CellPaths(opts.cell_dir(harness)).status,
        {"status": "learning_failed", "failure_kind": "harness", "updated_at": "t1"},
    )
    third = await Scheduler([harness], replace(opts, rerun_harness_failures=True)).run()
    assert third.exit_code == 0
    later = starts_of(opts, harness)[1:]
    assert ["--new-attempt" in s["rest"] for s in later] == [True, False]


def test_an_adopted_worker_is_as_new_as_its_own_start(tmp_path):
    spec = CellSpec("alfworld", "luna", "mirix")
    paths = CellPaths(tmp_path / spec.cell_id)
    write_json_atomic(
        paths.process, {"pid": 4242, "at": schedule.iso(time.time() - 600)}
    )
    old = schedule.adopted_worker(spec, paths)
    assert old.pid == 4242
    assert time.monotonic() - old.started_at == pytest.approx(600, abs=5)
    fresh = CellPaths(tmp_path / "fresh")
    write_json_atomic(fresh.process, {"pid": 7})  # no launch time: adopted now
    assert time.monotonic() - schedule.adopted_worker(spec, fresh).started_at < 5
    assert schedule.adopted_worker(spec, CellPaths(tmp_path / "none")).pid == -1


def test_worker_argv_adds_new_attempt_only_when_asked(tmp_path):
    opts = ScheduleOptions(runs_dir=tmp_path)
    spec = CellSpec("alfworld", "luna", "mirix")
    assert "--new-attempt" not in worker_argv(spec, opts, "u", tmp_path)
    assert (
        worker_argv(spec, opts, "u", tmp_path, new_attempt=True)[-1] == "--new-attempt"
    )


@pytest.mark.asyncio
async def test_a_legacy_learning_failure_is_rerun_with_backoff_and_caps(
    tmp_path, gateway, progress_file
):
    """An INFRA learning_failed of an attempt without checkpoints (run.py
    starts a new attempt) waits out the backoff like any resumable ending; a new attempt
    that never gets past its setup is no progress, so a persistent failure reaches the
    cap, and a restarted scheduler keeps the cap instead of queueing the cell again."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["none"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=0.02,
        max_resumes_without_progress=2,
    )
    behave(
        opts,
        cell,
        status="learning_failed",
        progress=[progress(None, attempt=n) for n in range(1, 10)],
        sleep_s=0.0,
    )
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
    # The first run (no baseline: the benefit of the doubt), then two new attempts
    # that got nowhere, and the third over the cap goes to a person.
    assert len(starts_of(opts, cell)) == 4
    delays = [e["delay_s"] for e in events_of(opts, "resume_scheduled")]
    assert delays == [0.02, 0.02, 0.04]  # backed off, not at once
    again = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert again.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
    assert again.launched == [] and len(starts_of(opts, cell)) == 4


@pytest.mark.asyncio
async def test_a_busy_cell_is_adopted_not_launched_twice(tmp_path, gateway):
    """run.py finding another live harness on its cell exits ``busy`` without a status:
    the scheduler adopts that harness's run."""
    _gw, url = gateway
    first, second = matrix(["alfworld"], ["luna"], ["mirix", "awm"])
    opts = options(tmp_path, url, concurrency=1, auto_resume=True)
    release_first, release_second = tmp_path / "r1", tmp_path / "r2"
    behave(opts, first, until=str(release_first), sleep_s=0.0)
    behave(opts, second, until=str(release_second), sleep_s=0.0)
    scheduler = Scheduler([first, second], opts)
    task = asyncio.ensure_future(scheduler.run())
    orphan = None
    try:
        await eventually(lambda: first.cell_id in scheduler.active, 30)
        orphan = subprocess.Popen(
            [sys.executable, "-m", FAKE_WORKER, "--out", str(opts.cell_dir(second))],
            cwd=protocol.REPO_ROOT,
            start_new_session=True,
        )
        await eventually(
            lambda: lock_is_held(CellPaths(opts.cell_dir(second)).lock), 30
        )
        release_first.touch()
        await eventually(lambda: second.cell_id in scheduler.result.adopted, 30)
        assert scheduler.active[second.cell_id].pid == orphan.pid
        release_second.touch()
        result = await asyncio.wait_for(task, 60)
    finally:
        release_first.touch()
        release_second.touch()
        if orphan is not None:
            orphan.wait(timeout=30)
    assert result.exit_code == 0
    assert result.launched == [first.cell_id] and result.adopted == [second.cell_id]
    assert [e.get("reason") for e in events_of(opts, "adopted")] == ["live"]


@pytest.mark.asyncio
async def test_a_merged_view_is_never_scheduled(tmp_path, gateway):
    _gw, url = gateway
    opts = options(tmp_path, url)
    opts.runs_dir.mkdir(parents=True, exist_ok=True)
    (opts.runs_dir / "_merged.json").write_text("{}")
    with pytest.raises(ScheduleError, match="merged read-only view"):
        await Scheduler(matrix(["alfworld"], ["luna"], ["none"]), opts).run()
    assert not opts.control.exists()


# --- pauses under --auto-resume -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_spent_budget_stops_the_auto_resuming_scheduler(tmp_path):
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 20.0
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    write_json_atomic(
        runs / "_gateway" / "budget.json",
        {"start_usage_usd": 11.5, "started_at": "then", "budgets": []},
    )
    cells = matrix(["alfworld"], ["luna56"], ["none"])
    with fake.serve() as upstream:
        opts = ScheduleOptions(
            runs_dir=runs,
            gateway_port=find_free_port(),
            gateway_upstream=upstream,
            env_file=env_file,
            budget_usd=8.0,
            poll_s=0.05,
            balance_poll_s=0.0,
            worker_module=FAKE_WORKER,
            auto_resume=True,
        )
        result = await asyncio.wait_for(Scheduler(cells, opts).run(), 60)
    assert result.paused and result.pause_reason == "budget_spent"
    assert result.exit_code == schedule.EXIT_PAUSED and result.launched == []
    state = read_json(opts.control / "state.json")
    assert state["pause_reason"] == "budget_spent" and state["exit_code"] == 3
    (paused,) = events_of(opts, "budget_paused")
    assert paused["kind"] == "stop"


@pytest.mark.asyncio
async def test_a_low_balance_is_waited_out_and_launching_resumes(tmp_path):
    fake = FakeUpstream(limit_remaining=2.0)  # below the gateway's 10 USD floor
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        opts = options(tmp_path, url, auto_resume=True)
        scheduler = Scheduler(cells, opts)
        task = asyncio.ensure_future(scheduler.run())
        path = opts.control / "state.json"
        await eventually(
            lambda: path.is_file()
            and read_json(path)["pause_reason"] == "balance_below_floor",
            30,
        )
        await asyncio.sleep(0.5)
        assert not task.done() and worker_log(opts) == []  # polling, not exiting
        (paused,) = events_of(opts, "budget_paused")
        assert paused["kind"] == "wait"
        fake.limit_remaining = 50.0  # a top-up, or the daily limit reset
        result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0 and sorted(result.complete) == sorted(
        c.cell_id for c in cells
    )
    assert [
        e["event"] for e in events_of(opts) if e["event"].startswith("budget_")
    ] == [
        "budget_paused",
        "budget_resumed",
    ]


# --- the resource guard --------------------------------------------------------------------------


class FakeHost:
    """An injected host reading: ``mem`` (a number, or a callable) GB available."""

    def __init__(self, mem: float | Callable[[], float], disk: float = 100.0) -> None:
        self.mem = mem
        self.disk = disk
        self.calls = 0
        self.workers: list[dict[str, Any]] = []

    def __call__(self, runs_dir: Path, workers: Any = None) -> HostResources:
        self.calls += 1
        self.workers.append(dict(workers or {}))
        mem = self.mem() if callable(self.mem) else self.mem
        return HostResources(
            at="now",
            system="test",
            mem_available_gb=mem,
            disk_free_gb=self.disk,
            workers_rss_gb={c: 0.0 for c in (workers or {})},
        )


@pytest.mark.asyncio
async def test_the_memory_gate_holds_launches_until_memory_is_free(tmp_path, gateway):
    _gw, url = gateway
    heavy, light = matrix(["alfworld"], ["luna"], ["mirix", "none"])  # 1.4 GB, 0.5 GB
    opts = options(
        tmp_path, url, concurrency=2, min_free_memory_gb=4.0, min_free_disk_gb=10.0
    )
    behave(opts, heavy, sleep_s=0.0)
    behave(opts, light, sleep_s=0.0)
    host = FakeHost(4.2)  # the light cell would fit; the heavy one holds the line
    scheduler = Scheduler([heavy, light], opts, sampler=host)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: bool(events_of(opts, "resource_wait")), 30)
    await asyncio.sleep(0.3)
    assert worker_log(opts) == [] and not task.done()
    (wait,) = events_of(opts, "resource_wait")  # logged once, not every poll
    assert wait["cell"] == heavy.cell_id and wait["reason"].startswith("memory:")
    host.mem = 10.0
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0 and result.launched == [heavy.cell_id, light.cell_id]
    assert events_of(opts, "resource_wait_end")[0]["cell"] == heavy.cell_id
    # A full disk holds launches too.
    (other,) = matrix(["alfworld"], ["luna56"], ["none"])
    behave(opts, other, sleep_s=0.0)
    full = Scheduler([other], opts, sampler=FakeHost(50.0, disk=5.0))
    task = asyncio.ensure_future(full.run())
    await eventually(
        lambda: any(
            e["cell"] == other.cell_id for e in events_of(opts, "resource_wait")
        ),
        30,
    )
    full.stop.set()
    stopped = await asyncio.wait_for(task, 30)
    assert stopped.stopped and stopped.launched == []
    assert [e["reason"] for e in events_of(opts, "resource_wait")][
        -1
    ] == "disk: 5.0 GB free < 10 GB"


@pytest.mark.asyncio
async def test_a_memory_emergency_stops_the_newest_worker_which_resumes_later(
    tmp_path, gateway, progress_file
):
    _gw, url = gateway
    older, newer = matrix(["alfworld"], ["luna"], ["reasoningbank", "awm"])
    opts = options(
        tmp_path,
        url,
        concurrency=2,
        poll_s=0.1,
        auto_resume=True,
        resume_backoff_base_s=0.05,
        memory_emergency_gb=1.5,
    )
    release = tmp_path / "release"
    for spec in (older, newer):
        behave(opts, spec, until=str(release), sleep_s=0.0, progress=progress(1))

    def memory() -> float:
        stopped = any(e["event"] == "memory_emergency_stop" for e in events_of(opts))
        return 8.0 if stopped else 1.0

    scheduler = Scheduler([older, newer], opts, sampler=FakeHost(memory))
    task = asyncio.ensure_future(scheduler.run())
    try:
        await eventually(lambda: bool(events_of(opts, "memory_emergency_stop")), 30)
        (stop,) = events_of(opts, "memory_emergency_stop")
        assert stop["cell"] == newer.cell_id and stop["samples"] == 3
        assert stop["mem_available_gb"] == 1.0 and stop["signalled"] is True
        await eventually(lambda: len(starts_of(opts, newer)) == 2, 30)  # resumed
    finally:
        release.touch()
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0 and sorted(result.complete) == sorted(
        [older.cell_id, newer.cell_id]
    )
    assert len(starts_of(opts, older)) == 1
    entry = read_json(opts.control / "resume_state.json")[newer.cell_id]
    assert entry["without_progress"] == 0 and entry["resumes"] == 1
    assert entry["history"][0]["reason"] == schedule.STOPPED_MEMORY_EMERGENCY
    assert entry["history"][0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_benchmark_caps_and_the_resources_log(tmp_path, gateway):
    _gw, url = gateway
    capped = matrix(["alfworld"], None, ["awm"])
    (shop,) = matrix(["webshop"], ["luna"], ["none"])
    opts = options(
        tmp_path,
        url,
        concurrency=3,
        benchmark_concurrency=(("alfworld", 1),),
        resources_log_interval_s=3600.0,
    )
    for spec in capped:
        behave(opts, spec, sleep_s=0.5)
    behave(opts, shop, sleep_s=0.5)
    host = FakeHost(50.0)
    result = await Scheduler(
        [*capped, shop], opts, sampler=host, webshop_ready=_ready
    ).run()
    assert result.exit_code == 0
    rows = worker_log(opts)
    capped_ids = {c.cell_id for c in capped}
    assert max_overlap([r for r in rows if r["cell"] in capped_ids]) == 1
    assert max_overlap(rows) == 2  # WebShop ran beside the capped ALFWorld cells
    resources_rows = read_jsonl(opts.control / "resources.jsonl")
    assert len(resources_rows) == 1  # throttled: one per interval
    assert resources_rows[0]["mem_available_gb"] == 50.0
    assert (
        read_json(opts.control / "state.json")["resources"]["mem_available_gb"] == 50.0
    )
    # A frequent log writes a row per poll.
    (plain,) = matrix(["alfworld"], ["luna"], ["none"])
    behave(opts, plain, sleep_s=0.5)
    fast = replace(opts, resources_log_interval_s=0.01)
    await Scheduler([plain], fast, sampler=host).run()
    assert len(read_jsonl(opts.control / "resources.jsonl")) >= 3


@pytest.mark.asyncio
async def test_the_resource_reading_counts_each_cells_worker_and_mirix_server(
    tmp_path, gateway
):
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["mirix"])
    opts = options(tmp_path, url, resources_log_interval_s=0.05)
    release = tmp_path / "release"
    behave(opts, cell, until=str(release), sleep_s=0.0)
    host = FakeHost(50.0)
    scheduler = Scheduler([cell], opts, sampler=host)
    task = asyncio.ensure_future(scheduler.run())
    try:
        await eventually(lambda: cell.cell_id in scheduler.active, 30)
        worker = scheduler.active[cell.cell_id]
        # Its MIRIX server registers in the cell's registry once the worker runs.
        registry = opts.cell_dir(cell) / "owned_processes.jsonl"
        registry.write_text(
            json.dumps(
                {"event": "started", "name": "mirix", "pid": 424242, "pgid": 424242}
            )
            + "\n"
        )
        await eventually(
            lambda: any(424242 in w.get(cell.cell_id, []) for w in host.workers), 30
        )
    finally:
        release.touch()
    await asyncio.wait_for(task, 60)
    seen = next(w for w in host.workers if 424242 in w.get(cell.cell_id, []))
    assert seen == {cell.cell_id: sorted([worker.pid, 424242])}
    rows = read_jsonl(opts.control / "resources.jsonl")
    assert any(r["active"] == [cell.cell_id] for r in rows)


# --- options and CLI -------------------------------------------------------------------------


def test_resume_and_guard_options_are_checked(tmp_path):
    base: dict[str, Any] = dict(runs_dir=tmp_path)
    cases: list[tuple[dict[str, Any], str]] = [
        (dict(resume_backoff_base_s=10, resume_backoff_cap_s=5), "backoff"),
        (dict(resume_backoff_base_s=-1), "backoff"),
        (dict(max_resumes=-1), "cannot be negative"),
        (dict(min_free_memory_gb=-1.0), "min_free_memory_gb"),
        (dict(benchmark_concurrency=(("webarena", 1),)), "unknown benchmark"),
        (dict(benchmark_concurrency=(("alfworld", 0),)), "at least 1"),
        (dict(benchmark_concurrency=(("alfworld", 1), ("alfworld", 2))), "twice"),
        (dict(resources_log_interval_s=0), "must be positive"),
    ]
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            ScheduleOptions(**base, **bad)
    opts = ScheduleOptions(**base, benchmark_concurrency=[["alfworld", "3"]])  # type: ignore[list-item]
    assert opts.benchmark_caps == {"alfworld": 3, "webshop": 4}
    opts = ScheduleOptions(**base, benchmark_concurrency=(("webshop", 8),))
    assert opts.benchmark_caps == {"webshop": 8}  # the default yields to the option


def test_the_cli_carries_every_new_option(tmp_path):
    args = schedule.build_parser().parse_args(
        [
            "--runs-dir",
            str(tmp_path),
            "--auto-resume",
            "--resume-backoff-base",
            "30",
            "--resume-backoff-cap",
            "600",
            "--max-resumes-without-progress",
            "4",
            "--max-resumes",
            "9",
            "--reset-resume",
            "--min-free-memory-gb",
            "4",
            "--min-free-disk-gb",
            "10",
            "--memory-emergency-gb",
            "1.5",
            "--benchmark-concurrency",
            "webshop=3",
            "alfworld=2",
            "--resources-log-interval",
            "30",
        ]
    )
    opts = schedule.options_from_args(args)
    assert opts.auto_resume and opts.reset_resume == ()
    assert (opts.resume_backoff_base_s, opts.resume_backoff_cap_s) == (30.0, 600.0)
    assert (opts.max_resumes_without_progress, opts.max_resumes) == (4, 9)
    assert (
        opts.min_free_memory_gb,
        opts.min_free_disk_gb,
        opts.memory_emergency_gb,
    ) == (4.0, 10.0, 1.5)
    assert opts.benchmark_caps == {"webshop": 3, "alfworld": 2}
    assert opts.resources_log_interval_s == 30.0
    defaults = schedule.options_from_args(schedule.build_parser().parse_args([]))
    assert not defaults.auto_resume and defaults.reset_resume is None
    assert (defaults.resume_backoff_base_s, defaults.resume_backoff_cap_s) == (
        60.0,
        1800.0,
    )
    assert (defaults.max_resumes_without_progress, defaults.max_resumes) == (5, 20)
    assert defaults.min_free_memory_gb is None
    assert defaults.benchmark_caps == dict(schedule.DEFAULT_BENCHMARK_CAPS)
    with pytest.raises(SystemExit):
        schedule.build_parser().parse_args(["--benchmark-concurrency", "alfworld"])
    with pytest.raises(SystemExit):
        schedule.main(
            [
                "--dry-run",
                "--runs-dir",
                str(tmp_path),
                "--benchmark",
                "webshop",
                "--reset-resume",
                "alfworld-luna-none",
            ]
        )


def test_dry_run_with_auto_resume_shows_the_waiting_cells(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_json_atomic(
        CellPaths(runs / "alfworld-luna-none").status,
        {"status": "learning_interrupted"},
    )
    write_json_atomic(
        runs / "_schedule" / "resume_state.json",
        {
            "alfworld-luna-none": {
                "resumes": 2,
                "without_progress": 1,
                "next_at": "2026-09-29T10:00:00+00:00",
            }
        },
    )
    code = schedule.main(
        [
            "--dry-run",
            "--auto-resume",
            "--runs-dir",
            str(runs),
            "--benchmark",
            "alfworld",
            "--method",
            "none",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == {
        "cell": "alfworld-luna-none",
        "plan": "waiting",
        "status": "learning_interrupted",
        "resume": {
            "resumes": 2,
            "without_progress": 1,
            "next_at": "2026-09-29T10:00:00+00:00",
            "attention": None,
        },
    }
    assert rows[1]["plan"] == "queued"


# --- HARNESS releases, stop signals, budget pauses and the WebShop gate ------------------


@pytest.mark.asyncio
async def test_a_harness_stop_that_happens_during_the_run_waits_for_a_person(
    tmp_path, gateway
):
    """--rerun-harness-failures releases the HARNESS stops that exist when
    the scheduler starts. A cell nobody looked at, whose first launch ends in a HARNESS
    stop, goes to a person; it is not relaunched to learn again from scratch."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        rerun_harness_failures=True,
        resume_backoff_base_s=0.02,
    )
    behave(
        opts,
        cell,
        statuses=["learning_failed", "complete"],
        status_extra={"failure_kind": "harness"},
        sleep_s=0.0,
    )
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.attention_reasons == {cell.cell_id: schedule.REASON_HARNESS_FAILURE}
    (start,) = starts_of(opts, cell)
    assert "--new-attempt" not in start["rest"]
    assert events_of(opts, "harness_stops_released")[0]["cells"] == []


@pytest.mark.asyncio
async def test_a_released_harness_stop_gets_one_new_attempt_ever(
    tmp_path, gateway, progress_file
):
    """A released stop is kept by its status marker in resume_state.json. A later
    scheduler (a supervisor restart, an operator copying the command line) does not
    relaunch it again, and naming cells releases only those. The release is used up
    once its new attempt exists, not when its worker is spawned."""
    _gw, url = gateway
    first, second = matrix(["alfworld"], ["luna56"], ["mirix", "awm"])
    opts = options(tmp_path, url, auto_resume=True, resume_backoff_base_s=0.02)
    for n, spec in enumerate((first, second)):
        write_json_atomic(
            CellPaths(opts.cell_dir(spec)).status,
            {
                "status": "learning_failed",
                "failure_kind": "harness",
                "updated_at": f"t{n}",
            },
        )
        # The approved worker's new attempt (attempt 1; the cell had none) exists. It
        # lives 3 s, so the stop below comes before its end on a loaded box.
        behave(
            opts,
            spec,
            until=str(tmp_path / "never"),
            until_timeout_s=3.0,
            progress=progress(0, attempt=1),
        )
    named = replace(
        opts, rerun_harness_failures=True, rerun_harness_cells=(first.cell_id,)
    )
    scheduler = Scheduler([first, second], named)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: len(starts_of(opts, first)) == 1, 30)
    scheduler.stop.set()  # stopped before its worker ended: the next one adopts it
    stopped = await asyncio.wait_for(task, 30)
    assert stopped.launched == [first.cell_id]
    assert stopped.attention_reasons == {
        second.cell_id: schedule.REASON_HARNESS_FAILURE
    }
    assert starts_of(opts, first)[0]["rest"][-1] == "--new-attempt"
    entry = read_json(opts.control / "resume_state.json")[first.cell_id]
    assert entry["harness_attempt"] == 0 and "harness_relaunched" not in entry
    await eventually(lambda: not lock_is_held(CellPaths(opts.cell_dir(first)).lock))
    # The fake worker wrote a complete status; put the released stop back (a scheduler
    # that relaunched it, then died): its new attempt exists, so it is never relaunched
    # for that stop again, flag or not.
    write_json_atomic(
        CellPaths(opts.cell_dir(first)).status,
        {"status": "learning_failed", "failure_kind": "harness", "updated_at": "t0"},
    )
    again = await asyncio.wait_for(
        Scheduler([first], replace(opts, rerun_harness_failures=True)).run(), 60
    )
    assert again.launched == []
    assert again.attention_reasons == {first.cell_id: schedule.REASON_HARNESS_FAILURE}
    assert len(starts_of(opts, first)) == 1
    entry = read_json(opts.control / "resume_state.json")[first.cell_id]
    assert entry["harness_approved"] == entry["harness_relaunched"]
    assert entry["harness_new_attempt"] == 1
    (used,) = events_of(opts, "harness_release_used")
    assert used["reason"] == "new_attempt" and used["attempt"] == 1
    assert events_of(opts, "harness_stops_released")[-1]["cells"] == []


def test_the_rerun_harness_failures_flag_takes_cells(tmp_path, capsys):
    args = schedule.build_parser().parse_args(
        ["--rerun-harness-failures", "alfworld-luna56-mirix", "--auto-resume"]
    )
    opts = schedule.options_from_args(args)
    assert opts.rerun_harness_failures and opts.rerun_harness_cells == (
        "alfworld-luna56-mirix",
    )
    every = schedule.options_from_args(
        schedule.build_parser().parse_args(["--rerun-harness-failures"])
    )
    assert every.rerun_harness_failures and every.rerun_harness_cells == ()
    none = schedule.options_from_args(schedule.build_parser().parse_args([]))
    assert not none.rerun_harness_failures
    with pytest.raises(SystemExit):
        schedule.main(
            [
                "--runs-dir",
                str(tmp_path),
                "--benchmark",
                "webshop",
                "--rerun-harness-failures",
                "alfworld-luna56-mirix",
                "--dry-run",
            ]
        )
    assert "outside the selection" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_stop_during_the_gateway_poll_launches_nothing(
    tmp_path, gateway, monkeypatch
):
    """A SIGTERM that lands while the scheduler reads the gateway's balance
    must not start workers in the same iteration (nobody would drain them)."""
    _gw, url = gateway
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    opts = options(tmp_path, url, concurrency=2)
    for c in cells:
        behave(opts, c, sleep_s=3.0)
    real = Scheduler.gateway_pause

    async def pause_then_sigterm(self: Scheduler) -> str | None:
        reason = await real(self)
        self.stop.set()
        return reason

    monkeypatch.setattr(Scheduler, "gateway_pause", pause_then_sigterm)
    result = await asyncio.wait_for(Scheduler(cells, opts).run(), 60)
    assert result.stopped and result.launched == [] and worker_log(opts) == []


@pytest.mark.asyncio
async def test_a_stop_during_admission_launches_nothing(tmp_path, gateway):
    _gw, url = gateway
    cells = matrix(["alfworld"], ["luna56"], ["none", "awm"])
    opts = options(tmp_path, url, concurrency=2, min_free_memory_gb=1.0)
    scheduler: Scheduler | None = None

    def stopping_host(runs_dir: Path, workers: Any = None) -> HostResources:
        assert scheduler is not None
        scheduler.stop.set()  # SIGTERM while the host is read
        return FakeHost(50.0)(runs_dir, workers)

    scheduler = Scheduler(cells, opts, sampler=stopping_host)
    result = await asyncio.wait_for(scheduler.run(), 60)
    assert result.stopped and result.launched == []


@pytest.mark.asyncio
async def test_a_spent_budget_stops_the_scheduler_whatever_the_pause_reason(
    tmp_path, gateway
):
    """Belt and braces: a gateway whose pause reads balance_below_floor
    while its spend budget is used up (an older gateway checked the floor first) is a
    spent budget to the scheduler: a stop pause, which the supervisor reports (exit 10)."""
    gw, url = gateway

    async def no_refresh(*_a: Any, **_kw: Any) -> None:
        return None

    gw.refresh_balance = no_refresh
    gw.paused, gw.pause_reason = True, "balance_below_floor"
    gw.budget = {"budget_usd": 250.0, "spent_usd": 250.4, "left_usd": -0.4}
    (cell,) = matrix(["alfworld"], ["luna56"], ["none"])
    opts = options(tmp_path, url, auto_resume=True)
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.paused and result.pause_reason == "budget_spent"
    (paused,) = events_of(opts, "budget_paused")
    assert paused["reason"] == "budget_spent" and paused["kind"] == "stop"
    # With budget left, the floor's pause stays waitable.
    gw.budget = {"budget_usd": 250.0, "spent_usd": 10.0, "left_usd": 240.0}
    scheduler = Scheduler([cell], opts)
    scheduler.gateway = schedule.SchedulerGateway(url, "attach", opts.runs_dir)
    assert await scheduler.gateway_pause() == "balance_below_floor"


@pytest.mark.asyncio
async def test_webshop_cells_wait_for_the_server_and_hold_nothing_else(
    tmp_path, gateway, monkeypatch
):
    """No WebShop cell starts before its server answers ``ready`` (loading the
    product set takes minutes); the other benchmarks' cells start meanwhile, and the
    wait is recorded with the server's state as its reason."""
    _gw, url = gateway
    monkeypatch.setattr(schedule, "WEBSHOP_READY_POLL_S", 0.0)
    (shop,) = matrix(["webshop"], ["luna"], ["none"])
    (other,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(tmp_path, url, concurrency=2)
    behave(opts, shop, sleep_s=0.0)
    behave(opts, other, sleep_s=0.0)
    server = {"ready": False, "probes": 0}

    async def probe() -> tuple[bool, str]:
        server["probes"] += 1
        if server["ready"]:
            return True, "ready"
        return False, "webshop: server 'loading'"

    scheduler = Scheduler([shop, other], opts, webshop_ready=probe)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: bool(starts_of(opts, other)), 30)
    await eventually(
        lambda: any(
            e["cell"] == shop.cell_id for e in events_of(opts, "resource_wait")
        ),
        30,
    )
    await asyncio.sleep(0.2)
    assert starts_of(opts, shop) == [] and not task.done()
    (wait,) = [e for e in events_of(opts, "resource_wait") if e["cell"] == shop.cell_id]
    assert wait["reason"] == "webshop: server 'loading'"
    assert read_json(opts.control / "state.json")["resource_wait"]["gate"] == "webshop"
    server["ready"] = True
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0
    assert result.launched == [other.cell_id, shop.cell_id]
    assert server["probes"] >= 2


@pytest.mark.asyncio
async def test_webshop_cells_are_capped_and_their_server_memory_is_sampled(
    tmp_path, gateway
):
    """WebShop's cells share one server with one global lock: at most
    ``DEFAULT_BENCHMARK_CAPS["webshop"]`` of them run at once unless the operator caps
    them otherwise. A WebShop-only batch (a pilot) still samples the
    containers' memory, so ``resources.jsonl`` records the server's."""
    _gw, url = gateway
    cells = matrix(["webshop"], None, ["mirix", "reasoningbank", "awm", "none"])
    assert len(cells) == 4 * len(protocol.MODELS) == 8
    opts = options(tmp_path, url, concurrency=6, resources_log_interval_s=0.01)
    for spec in cells:
        behave(opts, spec, sleep_s=0.4)
    sampled: list[int] = []

    def docker() -> dict[str, float]:
        sampled.append(1)
        return {"skill-bench-webshop": 17.5}

    async def ready() -> tuple[bool, str]:
        return True, "ready"

    scheduler = Scheduler(
        cells,
        opts,
        sampler=FakeHost(50.0),
        docker_sampler=docker,
        webshop_ready=ready,
    )
    assert scheduler._has_containers
    result = await scheduler.run()
    assert result.exit_code == 0
    assert max_overlap(worker_log(opts)) == schedule.DEFAULT_BENCHMARK_CAPS["webshop"]
    assert sampled
    logged = [
        r for r in read_jsonl(opts.control / "resources.jsonl") if "docker_gb" in r
    ]
    assert logged and logged[-1]["docker_gb"]["containers"] == {
        "skill-bench-webshop": 17.5
    }
    assert read_json(opts.control / "state.json")["benchmark_concurrency"] == {
        "webshop": 4
    }


# --- schedulers that share one host (host_registry.py) -----------------------------------


def _replicate(tmp_path: Path, name: str, url: str, **kw: Any) -> ScheduleOptions:
    """One replicate's options: its own runs directory, the test's one host directory."""
    kw.setdefault("host_dir", tmp_path / "host")
    return options(tmp_path / name, url, **kw)


async def _ready() -> tuple[bool, str]:
    return True, "ready"


@pytest.mark.asyncio
async def test_two_schedulers_on_one_host_share_the_memory_gate(tmp_path, gateway):
    """Replicate schedulers may share one host. Each reads the same host
    memory; without the host registry each would admit a cell on the same reading and
    reserve nothing for the other's fresh worker. Here 6 GB stay available whatever runs
    (every worker reads 0 GB resident), so a 1.4 GB cell fits once host-wide
    (6.0 - 1.4 - 1.4 < 4.0): the two schedulers never run two at once."""
    _gw, url = gateway
    cells = [
        *matrix(["alfworld"], ["luna56"], ["mirix"]),
        *matrix(["alfworld"], ["luna"], ["mirix"]),
    ]
    a = _replicate(tmp_path, "r1", url, concurrency=2, min_free_memory_gb=4.0)
    b = _replicate(tmp_path, "r2", url, concurrency=2, min_free_memory_gb=4.0)
    for opts in (a, b):
        for spec in cells:
            behave(opts, spec, sleep_s=0.4)
    results = await asyncio.gather(
        Scheduler(cells, a, sampler=FakeHost(6.0)).run(),
        Scheduler(cells, b, sampler=FakeHost(6.0)).run(),
    )
    assert [r.exit_code for r in results] == [0, 0]
    assert max_overlap(worker_log(a) + worker_log(b)) == 1
    waits = events_of(a, "resource_wait") + events_of(b, "resource_wait")
    assert any("1 other scheduler(s) on this host" in w["reason"] for w in waits)
    # A stopped scheduler leaves no record behind.
    assert list((tmp_path / "host" / host_registry.RECORDS_DIR).glob("*.json")) == []


@pytest.mark.asyncio
async def test_a_stopped_scheduler_keeps_its_live_workers_on_the_host(
    tmp_path, gateway
):
    """``kill -TERM`` of one replicate's supervisor stops its scheduler at once while
    its workers drain. Were the scheduler to withdraw its host record, the peers would
    admit as if those cells were gone (past the WebShop host cap). It leaves an
    orphaned record of the workers, which the peers count until the workers exit."""
    _gw, url = gateway
    cells = matrix(["webshop"], ["luna"], ["none", "awm"])
    opts = _replicate(tmp_path, "r1", url, concurrency=2)
    release = tmp_path / "release"
    for spec in cells:
        behave(opts, spec, until=str(release), sleep_s=0.0)
    scheduler = Scheduler(cells, opts, sampler=FakeHost(50.0), webshop_ready=_ready)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: len(scheduler.active) == 2, 30)
    workers = dict(scheduler.active)
    scheduler.stop.set()
    result = await asyncio.wait_for(task, 30)
    assert result.stopped and sorted(result.running) == sorted(workers)
    (event,) = events_of(opts, "host_record_orphaned")
    assert sorted(event["workers"]) == sorted(workers)
    peer_view = host_registry.HostRegistry(tmp_path / "host", tmp_path / "r2" / "runs")
    (peer,) = peer_view.peers(prune=True)
    assert peer.orphaned and peer.cells("webshop") == 2
    assert {w["pid"] for w in peer.workers} == {w.pid for w in workers.values()}
    # The workers exit: the record no longer counts, and the next prune removes it.
    release.touch()
    for worker in workers.values():
        assert worker.popen is not None and worker.popen.wait(timeout=30) == 0
    peer_view = host_registry.HostRegistry(tmp_path / "host", tmp_path / "r3" / "runs")
    assert peer_view.peers(prune=True) == []
    assert list((tmp_path / "host" / host_registry.RECORDS_DIR).glob("*.json")) == []


@pytest.mark.asyncio
async def test_three_replicates_share_the_default_webshop_host_cap(tmp_path, gateway):
    """Three replicate schedulers with WebShop's per-scheduler cap of 4 would admit 12
    WebShop cells at once (48 test episodes at --test-concurrency 4) against one server
    with one global lock; the default host-wide cap keeps the host at 6."""
    _gw, url = gateway
    assert schedule.DEFAULT_HOST_BENCHMARK_CAPS == {"webshop": 6}
    cells = matrix(["webshop"], ["luna", "luna56"], ["reasoningbank", "awm"])
    assert len(cells) == 4
    replicates = [_replicate(tmp_path, f"r{n}", url, concurrency=4) for n in (1, 2, 3)]
    for opts in replicates:
        for spec in cells:
            behave(opts, spec, sleep_s=1.0)
    results = await asyncio.gather(
        *(
            Scheduler(cells, opts, sampler=FakeHost(50.0), webshop_ready=_ready).run()
            for opts in replicates
        )
    )
    assert [r.exit_code for r in results] == [0, 0, 0]
    assert max_overlap([r for o in replicates for r in worker_log(o)]) == 6
    # Each one alone stays within its own cap; the host cap held some back.
    assert all(max_overlap(worker_log(o)) <= 4 for o in replicates)
    held = [e for o in replicates for e in events_of(o, "host_cap_wait")]
    assert held and all(
        "--host-benchmark-concurrency webshop=6" in e["reason"] for e in held
    )
    state = read_json(replicates[0].control / "state.json")
    assert state["host"]["benchmark_concurrency"] == {"webshop": 6}
    assert state["host"]["directory"] == str(tmp_path / "host")


@pytest.mark.asyncio
async def test_the_host_cap_is_set_per_benchmark_and_counts_the_peers(
    tmp_path, gateway
):
    _gw, url = gateway
    cells = matrix(["webshop"], ["luna", "luna56"], ["reasoningbank", "awm"])
    caps = (("webshop", 2),)
    a = _replicate(tmp_path, "r1", url, concurrency=4, host_benchmark_concurrency=caps)
    b = _replicate(tmp_path, "r2", url, concurrency=4, host_benchmark_concurrency=caps)
    for opts in (a, b):
        for spec in cells:
            behave(opts, spec, sleep_s=0.4)
    sa = Scheduler(cells, a, sampler=FakeHost(50.0), webshop_ready=_ready)
    sb = Scheduler(cells, b, sampler=FakeHost(50.0), webshop_ready=_ready)
    peers_seen: list[int] = []

    async def watch() -> None:
        while True:
            state = read_json(a.control / "state.json") or {}
            peers_seen.append(len((state.get("host") or {}).get("peers") or ()))
            await asyncio.sleep(0.05)

    watcher = asyncio.ensure_future(watch())
    try:
        results = await asyncio.gather(sa.run(), sb.run())
    finally:
        watcher.cancel()
    assert [r.exit_code for r in results] == [0, 0]
    assert max_overlap(worker_log(a) + worker_log(b)) == 2
    assert max(peers_seen) == 1  # state.json names the other scheduler while it runs
    assert a.host_benchmark_caps == {"webshop": 2}


@pytest.mark.asyncio
async def test_one_memory_emergency_stops_one_worker_host_wide(
    tmp_path, gateway, progress_file
):
    """Both schedulers read the same low memory. The first to take the host lock stops
    its newest worker; the other leaves the stop to it for the settle period instead of
    stopping one of its own on the same reading."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["reasoningbank"])
    kw: dict[str, Any] = dict(
        concurrency=1,
        poll_s=0.05,
        auto_resume=True,
        resume_backoff_base_s=0.05,
        memory_emergency_gb=1.5,
    )
    a = _replicate(tmp_path, "r1", url, **kw)
    b = _replicate(tmp_path, "r2", url, **kw)
    release = tmp_path / "release"
    for opts in (a, b):
        behave(opts, cell, until=str(release), sleep_s=0.0, progress=progress(1))
    low_until: list[float] = []

    def memory() -> float:
        if not low_until and (
            events_of(a, "memory_emergency_stop")
            or events_of(b, "memory_emergency_stop")
        ):
            low_until.append(time.monotonic() + 2.0)  # low for 2 s after the stop
        return 1.0 if not low_until or time.monotonic() < low_until[0] else 8.0

    tasks = [
        asyncio.ensure_future(Scheduler([cell], o, sampler=FakeHost(memory)).run())
        for o in (a, b)
    ]
    try:
        await eventually(lambda: bool(low_until), 30)
        await eventually(lambda: time.monotonic() > low_until[0] + 0.5, 30)
        stops = events_of(a, "memory_emergency_stop") + events_of(
            b, "memory_emergency_stop"
        )
        assert len(stops) == 1 and stops[0]["signalled"] is True
        stopper, other = (a, b) if events_of(a, "memory_emergency_stop") else (b, a)
        (left,) = events_of(other, "memory_emergency_left_to_peer")
        assert left["peers"][0]["runs_dir"] == str(stopper.runs_dir)
        await eventually(lambda: len(starts_of(stopper, cell)) == 2, 30)  # resumed
    finally:
        release.touch()
    results = await asyncio.wait_for(asyncio.gather(*tasks), 60)
    assert [r.exit_code for r in results] == [0, 0]


@pytest.mark.asyncio
async def test_an_unusable_host_directory_is_recorded_and_blocks_nothing(
    tmp_path, gateway
):
    _gw, url = gateway
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(tmp_path, url, host_dir=blocker / "host", min_free_memory_gb=1.0)
    behave(opts, cell, sleep_s=0.1)
    result = await Scheduler([cell], opts, sampler=FakeHost(50.0)).run()
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    problems = events_of(opts, "host_registry_unavailable")
    # Each distinct problem once (the lock's and the record's), not once per poll.
    assert 1 <= len(problems) == len({p["error"] for p in problems}) <= 2
    assert {p["directory"] for p in problems} == {str(blocker / "host")}
    state = read_json(opts.control / "state.json")
    assert state["host"]["problem"] in {p["error"] for p in problems}


def test_the_cli_takes_host_wide_caps(tmp_path, capsys):
    args = schedule.build_parser().parse_args(
        ["--host-benchmark-concurrency", "webshop=3", "alfworld=5"]
    )
    opts = schedule.options_from_args(args)
    assert opts.host_benchmark_caps == {"webshop": 3, "alfworld": 5}
    default = schedule.options_from_args(schedule.build_parser().parse_args([]))
    assert default.host_benchmark_caps == {"webshop": 6}
    for bad in (["webshop=0"], ["webshop=2", "webshop=3"], ["webarena=1"]):
        with pytest.raises(ValueError):
            schedule.options_from_args(
                schedule.build_parser().parse_args(
                    ["--host-benchmark-concurrency", *bad]
                )
            )
    with pytest.raises(SystemExit):
        schedule.build_parser().parse_args(["--host-benchmark-concurrency", "webshop"])
    assert "host-benchmark-concurrency" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_failed_webshop_server_is_said_once_while_its_cells_wait(
    tmp_path, gateway, monkeypatch
):
    """A server that failed its own checks never becomes ready by itself;
    the gate holds its cells and records ``webshop_server_failed`` once per reason (and
    again when the reason changes), not on every reading."""
    _gw, url = gateway
    monkeypatch.setattr(schedule, "WEBSHOP_READY_POLL_S", 0.0)
    (shop,) = matrix(["webshop"], ["luna"], ["none"])
    opts = options(tmp_path, url)
    behave(opts, shop, sleep_s=0.0)
    readings = [
        "webshop: server 'loading'",
        "webshop: server 'failed' (price table differs)",
        "webshop: server 'failed' (price table differs)",
        "webshop: server 'failed' (seed 7)",
    ]
    state = {"at": 0}

    async def probe() -> tuple[bool, str]:
        at = state["at"]
        state["at"] += 1
        if at < len(readings):
            return False, readings[at]
        return True, "ready"

    scheduler = Scheduler([shop], opts, webshop_ready=probe)
    result = await asyncio.wait_for(scheduler.run(), 60)
    assert result.exit_code == 0 and result.launched == [shop.cell_id]
    assert [e["reason"] for e in events_of(opts, "webshop_server_failed")] == [
        "webshop: server 'failed' (price table differs)",
        "webshop: server 'failed' (seed 7)",
    ]


@pytest.mark.asyncio
async def test_the_webshop_gate_caches_its_reading_and_fails_closed(tmp_path, gateway):
    _gw, url = gateway
    calls: list[int] = []

    async def broken() -> tuple[bool, str]:
        calls.append(1)
        raise RuntimeError("no route to host")

    scheduler = Scheduler(
        matrix(["webshop"], ["luna"], ["none"]),
        options(tmp_path, url),
        webshop_ready=broken,
    )
    assert await scheduler._webshop_gate() == (False, "RuntimeError: no route to host")
    assert await scheduler._webshop_gate() == (False, "RuntimeError: no route to host")
    assert len(calls) == 1  # read again only after WEBSHOP_READY_POLL_S


@pytest.mark.asyncio
async def test_the_default_webshop_probe(tmp_path, monkeypatch):
    from evals.skill_bench.envs import webshop as ws
    from evals.skill_bench.tests.webshop_support import make_shop, server_module

    monkeypatch.setenv(ws.SERVER_URL_ENV, "http://127.0.0.1:9")  # nothing listens
    ready, why = await schedule.webshop_server_ready()
    assert not ready and why.startswith("webshop: server unreachable")
    import threading

    shop = make_shop(tmp_path)
    httpd = server_module().start_http(shop.service, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv(
            ws.SERVER_URL_ENV, f"http://127.0.0.1:{httpd.server_address[1]}"
        )
        assert await schedule.webshop_server_ready() == (True, "ready")
        shop.service.state, shop.service.problems = "failed", ["price table differs"]
        ready, why = await schedule.webshop_server_ready()
        assert not ready and why == "webshop: server 'failed' (price table differs)"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.asyncio
async def test_lighter_cells_always_start_past_a_held_heavy_one(tmp_path, gateway):
    """There is no skip limit: a heavy cell the memory gate holds never holds the
    queue: every lighter cell behind it starts while it waits
    (each launch past it is recorded, with the running count), and it starts the first
    time memory allows."""
    _gw, url = gateway
    heavy, *light = matrix(
        ["alfworld"], ["luna"], ["mirix", "reasoningbank", "awm", "none"]
    )
    opts = options(tmp_path, url, concurrency=4, min_free_memory_gb=1.0)
    for spec in (heavy, *light):
        behave(opts, spec, sleep_s=0.0)
    host = FakeHost(2.2)  # mirix (1.4) does not fit; each 0.5-0.6 GB cell does
    scheduler = Scheduler([heavy, *light], opts, sampler=host)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: len(events_of(opts, "resource_skip")) == len(light), 30)

    def started_cells() -> set[str]:
        return {r["cell"] for r in worker_log(opts) if r["event"] == "start"}

    # each launch is recorded before its worker writes its first row: wait for them
    await eventually(lambda: started_cells() >= {c.cell_id for c in light}, 30)
    await asyncio.sleep(0.3)  # the held heavy cell must still not start meanwhile
    started = started_cells()
    assert heavy.cell_id not in started
    assert started == {c.cell_id for c in light}
    skips = events_of(opts, "resource_skip")
    assert [e["launched"] for e in skips] == [c.cell_id for c in light]
    assert {e["held"] for e in skips} == {heavy.cell_id}
    assert [e["skips"] for e in skips] == [1, 2, 3]
    assert not hasattr(schedule, "RESOURCE_SKIP_LIMIT")
    state = read_json(opts.control / "state.json")
    assert state["resource_wait"]["cell"] == heavy.cell_id
    assert state["resource_wait"]["gate"] == "memory"
    assert not task.done()
    host.mem = 10.0
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0
    assert result.launched == [*(c.cell_id for c in light), heavy.cell_id]
    assert read_json(opts.control / "state.json")["resource_wait"] is None


@pytest.mark.asyncio
async def test_the_disk_gate_still_holds_every_cell(tmp_path, gateway):
    """Only the memory gate lets cells past: a full disk holds the whole queue."""
    _gw, url = gateway
    cells = matrix(["alfworld"], ["luna"], ["mirix", "none"])
    opts = options(tmp_path, url, concurrency=2, min_free_disk_gb=10.0)
    for spec in cells:
        behave(opts, spec, sleep_s=0.0)
    host = FakeHost(50.0, disk=5.0)
    scheduler = Scheduler(cells, opts, sampler=host)
    task = asyncio.ensure_future(scheduler.run())
    await eventually(lambda: bool(events_of(opts, "resource_wait")), 30)
    await asyncio.sleep(0.3)
    assert worker_log(opts) == [] and events_of(opts, "resource_skip") == []
    host.disk = 50.0
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0
    assert result.launched == [c.cell_id for c in cells]


class _StoppableWorker:
    def __init__(self, cell_id: str, started_at: float) -> None:
        self.spec = CellSpec("alfworld", "luna", cell_id)
        self.pid = abs(hash(cell_id)) % 10_000
        self.started_at = started_at
        self.terminated = 0

    def refresh(self) -> None:
        pass

    def terminate(self) -> bool:
        self.terminated += 1
        return True


@pytest.mark.asyncio
async def test_the_memory_emergency_waits_for_its_victim_and_rotates(tmp_path):
    """While a stopped worker is still capturing its checkpoint (it holds
    its memory) no second worker is stopped, nor right after it exited; and the victim
    is the newest of those stopped the fewest times."""
    opts = ScheduleOptions(runs_dir=tmp_path / "runs", memory_emergency_gb=1.5)
    (tmp_path / "runs" / "_schedule").mkdir(parents=True)
    scheduler = Scheduler([], opts)
    old = _StoppableWorker("reasoningbank", 1.0)
    new = _StoppableWorker("awm", 2.0)
    scheduler.active = {"old": old, "new": new}  # type: ignore[dict-item]
    low = HostResources(at="now", system="test", mem_available_gb=1.0)

    async def samples(n: int) -> None:
        for _ in range(n):
            await scheduler._emergency_check(low)

    await samples(schedule.EMERGENCY_SAMPLES)
    assert (old.terminated, new.terminated) == (0, 1)
    await samples(3 * schedule.EMERGENCY_SAMPLES)  # the victim is still stopping
    assert (old.terminated, new.terminated) == (0, 1)
    # It exited (reap), and relaunched as the newest worker: the settle period first.
    scheduler._stopping.clear()
    scheduler._emergency_settle_until = time.monotonic() + 60
    new.started_at = 3.0
    await samples(3 * schedule.EMERGENCY_SAMPLES)
    assert (old.terminated, new.terminated) == (0, 1)
    scheduler._emergency_settle_until = float("-inf")
    await samples(schedule.EMERGENCY_SAMPLES)
    # The newest was stopped once already: the other worker takes this stop.
    assert (old.terminated, new.terminated) == (1, 1)


# --- released HARNESS stops, worker deaths and emergency kills ----------------------------


def released_stop(opts: ScheduleOptions, spec: CellSpec, *, attempt: int = 1) -> None:
    """A HARNESS stop of ``attempt`` as run.py leaves it (its status names the attempt),
    with that attempt, stopped after its third checkpoint, the cell's latest."""
    write_json_atomic(
        CellPaths(opts.cell_dir(spec)).status,
        {
            "status": "learning_failed",
            "failure_kind": "harness",
            "attempt": attempt,
            "pid": 1,
            "updated_at": "t0",
        },
    )
    write_json_atomic(
        opts.cell_dir(spec) / "progress.json",
        progress(3, attempt=attempt, state="learning_interrupted"),
    )


def launched_with_new_attempt(opts: ScheduleOptions) -> list[bool]:
    return ["--new-attempt" in e["argv"] for e in events_of(opts, "launched")]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_end", ["infra_unavailable", "budget_paused", "died"])
async def test_an_approved_new_attempt_survives_a_launch_that_never_reached_it(
    tmp_path, gateway, progress_file, first_end
):
    """The approved worker can end before run.py's plan step reaches the
    stopped attempt (the gateway down at its start, a budget pause, a death before its
    first status). Its relaunch still carries --new-attempt (without it run.py refuses
    the needs_decision attempt and the cell ends blocked, beyond
    --rerun-harness-failures); once the new attempt exists, later launches resume it
    and never start a third."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        rerun_harness_failures=True,
        resume_backoff_base_s=0.02,
        resume_backoff_cap_s=0.05,
    )
    released_stop(opts, cell)
    stopped = progress(3, attempt=1, state="learning_interrupted")
    first = "budget_paused" if first_end == "budget_paused" else "infra_unavailable"
    behave(
        opts,
        cell,
        statuses=[first, "learning_interrupted", "complete"],
        crash_at_start=[1, None] if first_end == "died" else None,
        # The first launch leaves attempt 1 as it was; the second starts attempt 2.
        progress=[stopped, progress(0, attempt=2), progress(2, attempt=2)],
        sleep_s=0.0,
    )
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    assert launched_with_new_attempt(opts) == [True, True, False]
    entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert entry["harness_attempt"] == 1 and entry["harness_new_attempt"] == 2
    assert entry["harness_relaunched"] == entry["harness_approved"]
    (used,) = events_of(opts, "harness_release_used")
    assert used["reason"] == "new_attempt"
    assert (used["stopped_attempt"], used["attempt"]) == (1, 2)


@pytest.mark.asyncio
async def test_an_approved_new_attempt_survives_a_scheduler_restart(
    tmp_path, gateway, progress_file
):
    """The release stays in resume_state.json until its new attempt
    exists, so a restarted scheduler (the supervisor passes --rerun-harness-failures
    only until one scheduler has applied it) still launches that new attempt."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=30.0,
        resume_backoff_cap_s=30.0,
    )
    released_stop(opts, cell)
    behave(
        opts,
        cell,
        statuses=["infra_unavailable", "complete"],
        progress=[progress(3, attempt=1, state="learning_interrupted")]
        + [progress(0, attempt=2)],
        sleep_s=0.0,
    )
    first = Scheduler([cell], replace(opts, rerun_harness_failures=True))
    task = asyncio.ensure_future(first.run())
    await eventually(lambda: bool(events_of(opts, "resume_scheduled")), 30)
    first.stop.set()
    await asyncio.wait_for(task, 30)
    assert read_json(opts.control / "resume_state.json")[cell.cell_id][
        "harness_approved"
    ]
    # Restarted without the flag, once its backoff is over.
    later = Scheduler([cell], opts, clock=lambda: time.time() + 3600)
    result = await asyncio.wait_for(later.run(), 60)
    assert result.complete == [cell.cell_id]
    assert launched_with_new_attempt(opts) == [True, True]


def test_a_harness_release_is_used_up_only_by_its_new_attempt(tmp_path):
    """The release keeps --new-attempt coming until the cell's latest
    attempt is past the stopped one (never a second new attempt), or the stopped attempt
    went on without one (a person resumed it by hand)."""
    t = tracker(tmp_path)
    assert t.approve_harness("c", "m1", 0.0, attempt=3)
    # Launches that never reached attempt 3, or invalidated it before saving the next.
    for state in ("learning_interrupted", "invalid"):
        assert t.harness_release(
            "c", latest=3, state=state, marker="other", at=1.0
        ) == ("m1", None)
    # Released again by a restarted scheduler: the stopped attempt stays 3.
    assert t.approve_harness("c", "m1", 2.0, attempt=4)
    assert t.cells["c"]["harness_attempt"] == 3
    assert t.harness_release("c", latest=4, state="learning", marker="x", at=3.0) == (
        None,
        "new_attempt",
    )
    assert t.harness_pending("c") is None
    assert t.harness_release("c", latest=4, state="learning", marker="x", at=3.0) == (
        None,
        None,
    )
    assert not t.approve_harness("c", "m1", 4.0, attempt=3)  # never twice
    # A person resumed the stopped attempt by hand: --new-attempt would replace it.
    assert t.approve_harness("d", "m2", 0.0, attempt=1)
    assert t.harness_release("d", latest=1, state="frozen", marker="x", at=1.0) == (
        None,
        "resumed",
    )
    # Released by a scheduler that did not record the attempt: the latest one while
    # the status is still that stop, else used up (never a second new attempt).
    t.cells["e"] = {"harness_approved": "m3", "history": []}
    assert t.harness_release(
        "e", latest=2, state="learning_interrupted", marker="m3", at=0.0
    ) == ("m3", None)
    assert t.cells["e"]["harness_attempt"] == 2
    t.cells["f"] = {"harness_approved": "m4", "history": []}
    assert t.harness_release("f", latest=2, state="learning", marker="run", at=0.0) == (
        None,
        "unknown",
    )
    kept = tracker(tmp_path)
    kept.load()
    assert kept.cells["c"]["harness_new_attempt"] == 4
    assert kept.cells["c"]["history"][-1] == {
        "at": schedule.iso(3.0),
        "event": "harness_new_attempt",
        "attempt": 4,
        "reason": "new_attempt",
    }


async def run_bounded(
    scheduler: Scheduler, timeout_s: float
) -> schedule.ScheduleResult:
    """The scheduler's result, or a failure once it has run ``timeout_s`` (it is stopped
    first, so a relaunch loop never outlives the test)."""
    task = asyncio.ensure_future(scheduler.run())
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    if not done:
        scheduler.stop.set()
        await asyncio.wait_for(task, 30)
        pytest.fail(f"the scheduler still ran after {timeout_s} s")
    return task.result()


def test_a_hand_resume_uses_up_the_release_even_while_it_still_learns(tmp_path):
    """A released stop whose approved launch never reached it, then resumed by a person
    (--force-resume) and interrupted again while still learning, must not keep
    --new-attempt coming: run.py would throw the hand-resumed progress away. A hand
    resume since the release uses it up."""
    t = tracker(tmp_path)
    assert t.approve_harness("c", "m1", 0.0, attempt=1)
    assert t.harness_release(
        "c", latest=1, state="learning_interrupted", marker="m-new", at=1.0
    ) == ("m1", None)
    assert t.harness_release(
        "c",
        latest=1,
        state="learning_interrupted",
        marker="m-new",
        at=1.0,
        hand_resumed=True,
    ) == (None, "resumed")
    assert t.harness_pending("c") is None


def test_operator_resumed_since_reads_the_hand_resumes_after_the_release(tmp_path):
    """Only a --force-resume (operator_resume) at or after the release
    counts; an automatic resume, or a hand resume of an earlier stop, does not."""
    paths = CellPaths(tmp_path / "cell")
    since = "2026-09-29T12:00:00+00:00"
    assert not schedule.operator_resumed_since(paths, 1, since)  # no attempt yet

    def resumes(*entries: dict[str, Any]) -> None:
        write_json_atomic(
            paths.attempt(1).state, {"state": "learning", "resumes": list(entries)}
        )

    resumes({"at": "2026-09-29T12:30:00+00:00", "operator_resume": False})
    assert not schedule.operator_resumed_since(paths, 1, since)
    resumes({"at": "2026-09-29T11:00:00+00:00", "operator_resume": True})
    assert not schedule.operator_resumed_since(paths, 1, since)
    resumes(
        {"at": "2026-09-29T11:00:00+00:00", "operator_resume": True},
        {"at": "2026-09-29T12:30:00+00:00", "operator_resume": True},
    )
    assert schedule.operator_resumed_since(paths, 1, since)
    assert not schedule.operator_resumed_since(paths, 2, since)  # another attempt


@pytest.mark.asyncio
async def test_a_release_is_neither_used_nor_launched_on_an_unreadable_listing(
    tmp_path, monkeypatch
):
    """When the attempts cannot be listed (an EMFILE in the scheduler), the latest
    attempt must not read 0 and the next launch carry --new-attempt, which would
    invalidate a new attempt that exists. Such a launch goes without it, and the
    release waits for a reading. A cell directory moved out (below the
    stopped attempt) launches without it too and keeps the release; an attempt of the
    stopped number from another run (the cell started over) uses it up (reset)."""
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(tmp_path, "http://127.0.0.1:9", auto_resume=True)
    scheduler = Scheduler([cell], opts)
    tracker_ = scheduler.tracker
    assert tracker_.approve_harness(
        cell.cell_id, "m1", 0.0, attempt=2, run_id="run-old"
    )
    paths = CellPaths(opts.cell_dir(cell))
    old = {"state": "learning_interrupted", "run_id": "run-old"}
    write_json_atomic(paths.attempt(2).state, old)

    def unlistable(self: CellPaths) -> list[int]:
        raise OSError(24, "Too many open files")

    with monkeypatch.context() as m:
        m.setattr(CellPaths, "attempt_numbers", unlistable)
        assert await scheduler._harness_release(cell, paths) is None
    assert tracker_.harness_pending(cell.cell_id) == "m1"  # not used up
    assert cell.cell_id in scheduler._release_unreadable  # launch() defers it
    (event,) = events_of(opts, "harness_release_unreadable")
    assert event["stopped_attempt"] == 2
    # Readable again, and the new attempt is not there yet: --new-attempt once more.
    assert await scheduler._harness_release(cell, paths) == "m1"
    assert cell.cell_id not in scheduler._release_unreadable
    # Its run cannot be read: no --new-attempt, kept, the launch deferred.
    paths.attempt(2).state.write_text("{torn")
    assert await scheduler._harness_release(cell, paths) is None
    assert tracker_.harness_pending(cell.cell_id) == "m1"
    assert cell.cell_id in scheduler._release_unreadable
    write_json_atomic(paths.attempt(2).state, old)
    # Moved aside: no --new-attempt for a fresh cell, the release kept.
    aside = tmp_path / "aside"
    shutil.move(str(paths.root), aside)
    assert await scheduler._harness_release(cell, paths) is None
    assert tracker_.harness_pending(cell.cell_id) == "m1"
    shutil.move(str(aside), paths.root)  # back: the release is there for it
    assert await scheduler._harness_release(cell, paths) == "m1"
    # Started over and learned to attempt 2 again, in another run: used up.
    shutil.rmtree(paths.root)
    write_json_atomic(paths.attempt(2).state, {**old, "run_id": "run-new"})
    assert await scheduler._harness_release(cell, paths) is None
    assert tracker_.harness_pending(cell.cell_id) is None
    (used,) = events_of(opts, "harness_release_used")
    assert used["reason"] == "reset" and used["stopped_attempt"] == 2
    # A release used up without a new attempt may be given again.
    assert tracker_.approve_harness(cell.cell_id, "m1", 5.0, attempt=2, run_id="x")
    assert tracker_.harness_pending(cell.cell_id) == "m1"


@pytest.mark.asyncio
async def test_a_release_of_an_unreadable_attempt_is_kept_not_dropped(tmp_path):
    """A stop whose attempt could not be read is released without it (never
    dropped); the latest attempt is taken while the status is still that stop, and a
    directory moved aside meanwhile (no attempt, no status) never uses it up."""
    t = tracker(tmp_path)
    assert t.approve_harness("c", "m1", 0.0, attempt=None)
    assert "harness_attempt" not in t.cells["c"]
    assert t.harness_release(
        "c", latest=3, state="learning_interrupted", marker="m1", at=1.0
    ) == ("m1", None)
    assert t.cells["c"]["harness_attempt"] == 3
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(tmp_path, "http://127.0.0.1:9", auto_resume=True)
    scheduler = Scheduler([cell], opts)
    assert scheduler.tracker.approve_harness(cell.cell_id, "m1", 0.0, attempt=None)
    paths = CellPaths(opts.cell_dir(cell))  # moved aside: nothing there
    assert await scheduler._harness_release(cell, paths) is None
    assert scheduler.tracker.harness_pending(cell.cell_id) == "m1"


@pytest.mark.asyncio
async def test_a_launch_on_an_unreadable_release_is_deferred_then_handed_over(
    tmp_path, monkeypatch
):
    """A launch whose pending release cannot be read spawns nothing (without
    --new-attempt run.py would refuse and block the cell; with it, it could reroll a
    new attempt that exists): it waits and is tried again, and after
    RELEASE_UNREADABLE_MAX_DEFERRALS in a row goes to a person with the release kept."""
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(tmp_path, "http://127.0.0.1:9", auto_resume=True)
    scheduler = Scheduler([cell], opts)
    scheduler.gateway = SimpleNamespace(
        base_url="http://127.0.0.1:9", runs_dir=tmp_path
    )
    assert scheduler.tracker.approve_harness(cell.cell_id, "m1", 0.0, attempt=2)

    def unlistable(self: CellPaths) -> list[int]:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(CellPaths, "attempt_numbers", unlistable)
    for n in range(1, schedule.RELEASE_UNREADABLE_MAX_DEFERRALS):
        assert await scheduler.launch(cell) is False
        assert cell.cell_id in scheduler.waiting and not scheduler.active
        scheduler.waiting.pop(cell.cell_id)
        assert events_of(opts, "launch_deferred")[-1]["deferrals"] == n
    assert await scheduler.launch(cell) is False
    assert cell.cell_id not in scheduler.waiting
    assert scheduler.result.attention_reasons[cell.cell_id] == (
        schedule.REASON_RELEASE_UNREADABLE
    )
    assert scheduler.tracker.harness_pending(cell.cell_id) == "m1"  # kept
    assert worker_log(opts) == []  # nothing ever spawned


@pytest.mark.asyncio
async def test_whatever_a_release_cannot_read_defers_it(tmp_path, monkeypatch):
    """An OSError on any read the release decision needs (the status under
    EMFILE, EIO), a status that cannot be read while the stopped attempt is still to be
    recorded, or a stopped attempt whose state cannot be read, decides nothing: the
    cell is marked unreadable (its launch deferred) and the release kept."""
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(tmp_path, "http://127.0.0.1:9", auto_resume=True)
    scheduler = Scheduler([cell], opts)
    tr = scheduler.tracker
    paths = CellPaths(opts.cell_dir(cell))
    write_json_atomic(paths.attempt(1).state, {"state": "learning", "run_id": "r1"})

    async def deferred() -> bool:
        released = await scheduler._harness_release(cell, paths)
        return released is None and cell.cell_id in scheduler._release_unreadable

    # An OSError while reading the status.
    assert tr.approve_harness(cell.cell_id, "m1", 0.0, attempt=1, run_id="r1")

    def eio(paths: CellPaths) -> Any:
        raise OSError(5, "Input/output error")

    with monkeypatch.context() as m:
        m.setattr(schedule, "read_status", eio)
        assert await deferred()
    # The stopped attempt's state cannot be read.
    paths.attempt(1).state.write_text("{torn")
    assert await deferred()
    write_json_atomic(paths.attempt(1).state, {"state": "learning", "run_id": "r1"})
    assert await scheduler._harness_release(cell, paths) == "m1"
    # Released without its attempt, and the status cannot be read.
    tr.cells.pop(cell.cell_id)
    assert tr.approve_harness(cell.cell_id, "m2", 0.0, attempt=None)
    assert not paths.status.exists()
    assert await deferred()  # not used up, not launched without the flag
    assert tr.harness_pending(cell.cell_id) == "m2"


def test_a_release_recorded_late_keeps_its_run_id(tmp_path):
    """A release approved without its attempt records the attempt later; its
    run is recorded with it, so a started-over cell is still told apart."""
    t = tracker(tmp_path)
    assert t.approve_harness("c", "m1", 0.0, attempt=None)
    assert t.harness_release(
        "c",
        latest=1,
        state="learning_interrupted",
        marker="m1",
        at=1.0,
        latest_run_id="run-1",
    ) == ("m1", None)
    assert t.cells["c"]["harness_attempt"] == 1
    assert t.cells["c"]["harness_run_id"] == "run-1"


def test_a_kill_never_lands_on_a_run_that_took_the_cell_after_its_victim(tmp_path):
    """A hung victim's SIGKILL must not re-read process.json and switch to whoever holds
    the cell now (a run.py started by hand after the victim exited would be killed
    without a SIGTERM): kill() only ever kills the pid it has."""
    spec = CellSpec("alfworld", "luna", "awm")
    paths = CellPaths(tmp_path / spec.cell_id)
    paths.root.mkdir()
    newcomer = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        victim = schedule.Worker(spec, paths, 999_999)  # adopted, gone now
        started = process_start_time(newcomer.pid)
        assert started  # else the test would pass without testing anything
        write_json_atomic(paths.process, {"pid": newcomer.pid, "started": started})
        assert victim.kill() is False and victim.pid == 999_999
        assert newcomer.poll() is None
    finally:
        newcomer.kill()
        newcomer.wait()


def test_the_scheduler_takes_no_abbreviated_options():
    """argparse's abbreviations would let --reset and --rerun-harness stand for the
    one-off options, which the supervisor finds by their full names only: they would
    reach every restarted scheduler unrecorded. The scheduler refuses them."""
    parser = schedule.build_parser()
    for argv in (["--reset"], ["--rerun-harness", "x"], ["--auto"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    assert parser.parse_args(["--reset-resume"]).reset_resume == []


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_resume", [True, False])
async def test_a_worker_dead_before_its_first_status_is_not_its_stale_status(
    tmp_path, gateway, auto_resume
):
    """A worker that dies before its first status (an import error, a kill in run.py's
    _write_process) leaves an earlier run's status.json. Read as its own ending, a stale
    budget_paused would put the cell back at the front of the queue on every poll (no
    backoff, no cap, no alert). It is this worker's death:
    backoff and caps with --auto-resume, a person without."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path,
        url,
        auto_resume=auto_resume,
        max_resumes_without_progress=2,
        resume_backoff_base_s=0.02,
        resume_backoff_cap_s=0.05,
    )
    write_json_atomic(
        CellPaths(opts.cell_dir(cell)).status,
        {"status": "budget_paused", "pid": 1, "updated_at": "t0"},
    )
    behave(opts, cell, crash_at_start=1)
    result = await run_bounded(Scheduler([cell], opts), 30)
    launches = read_json(opts.cell_dir(cell) / "launches_seen.json")["n"]
    exits = events_of(opts, "exited")
    assert len(exits) == launches
    assert all(e["status"] is None for e in exits)
    assert all(e["stale_status"] == "budget_paused" for e in exits)
    if auto_resume:
        assert launches == 3  # two resumes without progress, then the cap
        assert result.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
        entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
        assert [h["status"] for h in entry["history"]] == ["worker_died"] * 3
        assert len(events_of(opts, "resume_scheduled")) == 2
    else:
        assert launches == 1
        assert result.attention == {cell.cell_id: "exit 1"}
        assert result.attention_reasons == {cell.cell_id: schedule.REASON_NOT_RERUN}


@pytest.mark.asyncio
async def test_a_death_on_a_stale_status_keeps_its_backoff_across_a_restart(
    tmp_path, gateway
):
    """A restarted scheduler that finds the stale budget_paused status a
    counted death left behind keeps that death's backoff and counters: it does not
    launch the cell at once, nor count the death again."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=30.0,
        resume_backoff_cap_s=30.0,
    )
    write_json_atomic(
        CellPaths(opts.cell_dir(cell)).status,
        {"status": "budget_paused", "pid": 1, "updated_at": "t0"},
    )
    behave(opts, cell, crash_at_start=1)
    first = Scheduler([cell], opts)
    task = asyncio.ensure_future(first.run())
    await eventually(lambda: bool(events_of(opts, "resume_scheduled")), 30)
    first.stop.set()
    await asyncio.wait_for(task, 30)
    before = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert before["without_progress"] == 1
    second = Scheduler([cell], opts)
    task = asyncio.ensure_future(second.run())
    try:
        await eventually(lambda: cell.cell_id in second.waiting, 10)
        await asyncio.sleep(0.3)  # a few polls: nothing is launched
    finally:
        second.stop.set()
        await asyncio.wait_for(task, 30)
    assert read_json(opts.cell_dir(cell) / "launches_seen.json")["n"] == 1
    assert second.result.launched == []
    assert read_json(opts.control / "resume_state.json")[cell.cell_id] == before


def test_a_death_marker_names_the_status_it_left_behind():
    """The same ending always gets the same marker, and it covers the
    stale status a restarted scheduler reads again."""
    stale = {"status": "infra_unavailable", "updated_at": "t0", "pid": 1}
    marker = schedule.died_marker(stale, 4242, 1)
    assert marker == schedule.died_marker(dict(stale), 4242, 1)
    assert marker != schedule.died_marker(stale, 4243, 1)
    assert schedule.marker_counted(marker, marker)
    assert schedule.marker_counted(marker, schedule.status_marker(stale))
    assert not schedule.marker_counted(schedule.status_marker(stale), marker)
    later = schedule.status_marker({**stale, "updated_at": "t1"})
    assert not schedule.marker_counted(marker, later)
    assert not schedule.marker_counted(None, marker)


@pytest.mark.asyncio
async def test_an_ending_is_counted_once_across_scheduler_restarts(tmp_path, gateway):
    """A worker that died before its first status is counted under a marker without
    the clock in it (with it, a restarted scheduler reading the status it left behind
    would count that earlier ending again: one resume without progress too many, a
    second resume_cap alert). The death's marker names that status."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        max_resumes_without_progress=1,
        resume_backoff_base_s=0.02,
        resume_backoff_cap_s=0.05,
    )
    stale = {"status": "infra_unavailable", "pid": 1, "updated_at": "t0"}
    write_json_atomic(CellPaths(opts.cell_dir(cell)).status, stale)
    behave(opts, cell, crash_at_start=1)
    first = await run_bounded(Scheduler([cell], opts), 30)
    assert first.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
    before = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert (before["without_progress"], before["resumes"]) == (2, 2)
    again = await run_bounded(Scheduler([cell], opts), 30)
    assert again.attention_reasons == {cell.cell_id: schedule.REASON_RESUME_CAP}
    assert again.launched == []
    assert read_json(opts.control / "resume_state.json")[cell.cell_id] == before
    assert len(events_of(opts, "resume_cap")) == 1
    last = events_of(opts, "exited")[-1]
    assert before["last_marker"] == schedule.died_marker(stale, last["worker"], 1)


class _EmergencyWorker:
    """A worker as the memory emergency sees it; ``terminate``/``kill`` deliver or not."""

    def __init__(
        self, method: str, started_at: float, *, term: bool = True, kill: bool = True
    ) -> None:
        self.spec = CellSpec("alfworld", "luna", method)
        self.pid = abs(hash(method)) % 10_000
        self.started_at = started_at
        self._term, self._kill = term, kill
        self.terminated = 0
        self.killed = 0

    def refresh(self) -> None:
        pass

    def terminate(self) -> bool:
        self.terminated += 1
        return self._term

    def kill(self) -> bool:
        self.killed += 1
        return self._kill


def emergency_scheduler(
    tmp_path: Path, *workers: _EmergencyWorker
) -> tuple[Scheduler, ScheduleOptions]:
    opts = ScheduleOptions(runs_dir=tmp_path / "runs", memory_emergency_gb=1.5)
    opts.control.mkdir(parents=True)
    scheduler = Scheduler([], opts)
    scheduler.active = {w.spec.cell_id: w for w in workers}  # type: ignore[misc]
    return scheduler, opts


LOW = HostResources(at="now", system="test", mem_available_gb=1.0)
FINE = HostResources(at="now", system="test", mem_available_gb=8.0)


async def emergency_samples(scheduler: Scheduler, n: int) -> None:
    for _ in range(n):
        await scheduler._emergency_check(LOW)


@pytest.mark.asyncio
async def test_a_victim_that_cannot_be_signalled_passes_the_stop_on(tmp_path):
    """An adopted worker whose pid is stale cannot be signalled; the
    emergency stop goes to the next worker instead of to that one forever."""
    stale = _EmergencyWorker("reasoningbank", 5.0, term=False)  # the newest
    other = _EmergencyWorker("awm", 1.0)
    scheduler, opts = emergency_scheduler(tmp_path, stale, other)
    await emergency_samples(scheduler, 10 * schedule.EMERGENCY_SAMPLES)
    assert (stale.terminated, other.terminated) == (1, 1)
    stops = events_of(opts, "memory_emergency_stop")
    assert [(e["cell"], e["signalled"]) for e in stops] == [
        (stale.spec.cell_id, False),
        (other.spec.cell_id, True),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("killable", [True, False])
async def test_a_hung_emergency_victim_is_killed_and_holds_nothing_back(
    tmp_path, killable
):
    """A victim still running EMERGENCY_KILL_AFTER_S after its SIGTERM
    gets SIGKILL, whatever the memory then, and no longer holds back the next stop (a
    delivered kill after the settle period: its memory is freed a moment later)."""
    hung = _EmergencyWorker("awm", 2.0, kill=killable)
    other = _EmergencyWorker("reasoningbank", 1.0)
    scheduler, opts = emergency_scheduler(tmp_path, hung, other)
    await emergency_samples(scheduler, schedule.EMERGENCY_SAMPLES)
    assert (hung.terminated, other.terminated) == (1, 0)
    await emergency_samples(scheduler, 3 * schedule.EMERGENCY_SAMPLES)
    assert (hung.killed, other.terminated) == (0, 0)  # still within its time
    scheduler._emergency_signalled[hung.spec.cell_id] -= schedule.EMERGENCY_KILL_AFTER_S
    await scheduler._emergency_check(FINE)
    assert hung.killed == 1
    (kill,) = events_of(opts, "memory_emergency_kill")
    assert kill["cell"] == hung.spec.cell_id and kill["killed"] is killable
    assert kill["waited_s"] >= schedule.EMERGENCY_KILL_AFTER_S
    await emergency_samples(scheduler, 3 * schedule.EMERGENCY_SAMPLES)
    if killable:
        assert other.terminated == 0  # the settle period first
        scheduler._emergency_settle_until = float("-inf")
        await emergency_samples(scheduler, schedule.EMERGENCY_SAMPLES)
    assert (hung.killed, other.terminated) == (1, 1)


def test_a_worker_is_killed_only_while_it_is_still_that_process(tmp_path):
    """SIGKILL goes to a launched worker's whole group (its own session:
    its ALFWorld games too), and to an adopted one only while its process.json start
    time still matches (never a stranger that reused the pid)."""
    spec = CellSpec("alfworld", "luna", "awm")
    paths = CellPaths(tmp_path / spec.cell_id)
    paths.root.mkdir()
    grandchild_file = tmp_path / "grandchild.pid"
    launched = subprocess.Popen(
        ["sh", "-c", f"sleep 60 & echo $! > {grandchild_file}; wait"],
        start_new_session=True,
    )
    adopted = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        worker = schedule.Worker(spec, paths, launched.pid, launched)
        deadline = time.monotonic() + 10
        while not grandchild_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        grandchild = int(grandchild_file.read_text().strip())
        assert worker.kill()
        assert launched.wait(10) == -signal.SIGKILL
        while process_start_time(grandchild) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process_start_time(grandchild) is None
        assert not worker.kill()  # gone
        stranger = schedule.Worker(spec, paths, adopted.pid)
        write_json_atomic(
            paths.process, {"pid": adopted.pid, "started": "Mon Jan  1 00:00:00 2001"}
        )
        assert not stranger.kill() and adopted.poll() is None
        write_json_atomic(
            paths.process,
            {"pid": adopted.pid, "started": process_start_time(adopted.pid)},
        )
        assert stranger.kill()
        assert adopted.wait(10) == -signal.SIGKILL
    finally:
        for child in (launched, adopted):
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.asyncio
async def test_a_hung_emergency_victim_is_killed_with_its_mirix_server(
    tmp_path, gateway, monkeypatch
):
    """End to end: a worker that ignores the emergency SIGTERM gets
    SIGKILL after EMERGENCY_KILL_AFTER_S; its MIRIX server (a group of its own, in the
    cell's registry) is cleaned up once it is gone, and the cell resumes."""
    monkeypatch.setattr(schedule, "EMERGENCY_KILL_AFTER_S", 0.5)
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(
        tmp_path,
        url,
        poll_s=0.1,
        auto_resume=True,
        resume_backoff_base_s=0.05,
        memory_emergency_gb=1.5,
    )
    behave(
        opts,
        cell,
        ignore_sigterm=True,
        until=str(tmp_path / "never"),
        until_timeout_s=20,
        sleep_s=0.0,
    )
    low = tmp_path / "low"

    def memory() -> float:
        return (
            1.0
            if low.exists() and not events_of(opts, "memory_emergency_kill")
            else 8.0
        )

    scheduler = Scheduler([cell], opts, sampler=FakeHost(memory))
    task = asyncio.ensure_future(scheduler.run())
    server = None
    try:
        await eventually(lambda: len(starts_of(opts, cell)) == 1, 30)
        behave(opts, cell, sleep_s=0.0)  # its relaunch completes
        server = await asyncio.to_thread(
            OwnedProcess.spawn,
            "mirix",
            ["sleep", "60"],
            env=worker_env(),
            cwd=tmp_path,
            log_path=tmp_path / "server.log",
            registry=opts.cell_dir(cell) / "owned_processes.jsonl",
        )  # as infra/mirix_server.py spawns it: no owner lock
        low.touch()
        await eventually(lambda: bool(events_of(opts, "memory_emergency_kill")), 20)
        result = await asyncio.wait_for(task, 60)
    finally:
        scheduler.stop.set()
        if not task.done():
            await asyncio.wait_for(task, 30)
        if server is not None and server.alive():
            server.popen.kill()
        if server is not None:
            await server.stop(grace_s=1.0)
    assert result.complete == [cell.cell_id]
    (stop,) = events_of(opts, "memory_emergency_stop")
    (kill,) = events_of(opts, "memory_emergency_kill")
    assert stop["signalled"] is True and kill["killed"] is True
    first = events_of(opts, "exited")[0]
    assert first["returncode"] == -signal.SIGKILL
    assert first["stopped_by"] == schedule.STOPPED_MEMORY_EMERGENCY
    assert first["swept"] == [server.pid]
    assert server.popen.returncode == -signal.SIGKILL
    entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
    assert entry["history"][0]["reason"] == schedule.STOPPED_MEMORY_EMERGENCY
    assert entry["without_progress"] == 0


@pytest.mark.asyncio
async def test_a_worker_killed_from_outside_has_its_mirix_server_swept(
    tmp_path, gateway
):
    """Not only a worker the scheduler itself SIGKILLed has its MIRIX server cleaned up
    at once: one the OOM killer took would leave it running, holding memory, until the
    cell's next run.py a resume backoff later. Any exit sweeps it (rows whose owner
    lock is held are never touched)."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["awm"])
    opts = options(
        tmp_path, url, poll_s=0.1, auto_resume=True, resume_backoff_base_s=0.05
    )
    behave(opts, cell, until=str(tmp_path / "never"), until_timeout_s=20, sleep_s=0.0)
    scheduler = Scheduler([cell], opts)
    task = asyncio.ensure_future(scheduler.run())
    server = None
    try:
        await eventually(lambda: len(starts_of(opts, cell)) == 1, 30)
        behave(opts, cell, sleep_s=0.0)  # its relaunch completes
        server = await asyncio.to_thread(
            OwnedProcess.spawn,
            "mirix",
            ["sleep", "60"],
            env=worker_env(),
            cwd=tmp_path,
            log_path=tmp_path / "server.log",
            registry=opts.cell_dir(cell) / "owned_processes.jsonl",
        )  # as infra/mirix_server.py spawns it: no owner lock
        os.kill(scheduler.active[cell.cell_id].pid, signal.SIGKILL)  # the OOM killer
        result = await asyncio.wait_for(task, 60)
    finally:
        scheduler.stop.set()
        if not task.done():
            await asyncio.wait_for(task, 30)
        if server is not None and server.alive():
            server.popen.kill()
        if server is not None:
            await server.stop(grace_s=1.0)
    assert result.complete == [cell.cell_id]
    first = events_of(opts, "exited")[0]
    assert first["returncode"] == -signal.SIGKILL
    assert first["swept"] == [server.pid]
    assert server.popen.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_a_busy_exit_never_sweeps_the_live_harness_mirix_server(tmp_path):
    """The sweep after an exit must not kill the MIRIX server of a live run.py (one a
    person started by hand, which made the scheduler's worker exit busy): a MIRIX
    server's registry row has no owner lock.
    The sweep runs only under the cell lock, which that run.py holds."""
    (cell,) = matrix(["alfworld"], ["luna"], ["mirix"])
    opts = options(tmp_path, "http://127.0.0.1:9")
    opts.control.mkdir(parents=True)
    paths = CellPaths(opts.cell_dir(cell))
    paths.root.mkdir(parents=True)
    cell_lock = OwnerLock.acquire(paths.lock)  # the live harness
    server = OwnedProcess.spawn(
        f"mirix-{cell.cell_id}-a1",
        ["sleep", "60"],
        env=worker_env(),
        cwd=tmp_path,
        log_path=tmp_path / "server.log",
        registry=paths.root / "owned_processes.jsonl",
    )  # as infra/mirix_server.py spawns it: no owner lock
    try:
        busy = subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({schedule.EXIT_CODES['busy']})"]
        )
        busy.wait()
        scheduler = Scheduler([cell], opts)
        scheduler.active[cell.cell_id] = schedule.Worker(cell, paths, busy.pid, busy)
        await scheduler.reap()
        await asyncio.sleep(0.3)
        assert server.popen.poll() is None
        (exited,) = events_of(opts, "exited")
        assert "swept" not in exited
        # Once that harness is gone (its lock free), its server is an orphan: swept.
        cell_lock.release()
        assert schedule.sweep_dead_cell(paths) == [server.pid]
        assert server.popen.wait(10) == -signal.SIGKILL
    finally:
        cell_lock.release()
        if server.popen.poll() is None:
            server.popen.kill()
        server.popen.wait()


@pytest.mark.asyncio
async def test_a_new_attempt_is_found_when_the_progress_is_unreadable(
    tmp_path, monkeypatch
):
    """A cell cell_progress cannot read (a torn results file) must not look
    like one whose new attempt is missing (that would start a second one): its attempt
    directories are read directly."""
    (cell,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    opts = ScheduleOptions(runs_dir=tmp_path / "runs")
    opts.control.mkdir(parents=True)
    paths = CellPaths(opts.cell_dir(cell))
    write_json_atomic(paths.attempt(1).state, {"state": "invalid"})
    scheduler = Scheduler([cell], opts)
    assert scheduler.tracker.approve_harness(cell.cell_id, "m", 0.0, attempt=1)

    def torn(_paths: CellPaths) -> Any:
        raise ValueError("torn results.jsonl")

    monkeypatch.setattr(schedule, "cell_progress", torn)
    # Attempt 1 invalidated, attempt 2 not saved yet: the launch still starts it.
    assert await scheduler._harness_release(cell, paths) == "m"
    write_json_atomic(paths.attempt(2).state, {"state": "learning"})
    assert await scheduler._harness_release(cell, paths) is None
    assert schedule.latest_attempt(paths) == (2, "learning")
    (used,) = events_of(opts, "harness_release_used")
    assert (used["attempt"], used["reason"]) == (2, "new_attempt")


def test_a_worker_adopted_before_its_process_record_can_still_be_stopped(tmp_path):
    """A worker adopted before its run.py wrote process.json (pid -1) takes the pid
    process.json names once it is there, under the start-time check. So does one
    adopted with the previous run's pid, dead now (the
    process.json it read was the last run's); a process.json naming a process that is
    not running (a reused pid, other start time) is never taken."""
    spec = CellSpec("alfworld", "luna", "awm")
    paths = CellPaths(tmp_path / spec.cell_id)
    paths.root.mkdir()
    children = [
        subprocess.Popen(["sleep", "60"], start_new_session=True) for _ in range(2)
    ]
    try:
        first, second = children
        adopted = schedule.adopted_worker(spec, paths)
        assert adopted.pid == -1 and not adopted.terminate()
        write_json_atomic(
            paths.process, {"pid": first.pid, "started": process_start_time(first.pid)}
        )
        assert adopted.terminate() and adopted.pid == first.pid
        assert first.wait(10) == -signal.SIGTERM
        # Adopted with the previous run's pid (dead).
        stale = schedule.adopted_worker(spec, paths)
        assert stale.pid == first.pid and not stale.terminate()
        write_json_atomic(paths.process, {"pid": second.pid, "started": "Jan 1 1999"})
        assert not stale.terminate() and stale.pid == first.pid  # not that process
        write_json_atomic(
            paths.process,
            {"pid": second.pid, "started": process_start_time(second.pid)},
        )
        assert stale.terminate() and stale.pid == second.pid
        assert second.wait(10) == -signal.SIGTERM
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.asyncio
async def test_a_clean_complete_ending_is_complete_whoever_wrote_it(tmp_path, gateway):
    """The ownership check never turns a clean complete ending into a
    death (a complete cell is never launched, so its status cannot be a stale one)."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(tmp_path, url, auto_resume=True)
    behave(opts, cell, sleep_s=0.0, status_extra={"pid": 1})
    result = await run_bounded(Scheduler([cell], opts), 30)
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    assert len(starts_of(opts, cell)) == 1
    assert not events_of(opts, "resume_scheduled")


# --- MIRIX gateways ----------------------------------------------------------------------


def mirix_attempt(
    opts: ScheduleOptions,
    spec: CellSpec,
    *,
    state: str,
    url: str | None,
    checkpoints: bool = False,
    number: int = 1,
) -> CellPaths:
    """Attempt ``number`` of ``spec`` as run.py leaves it, its agent set up with ``url``."""
    paths = CellPaths(opts.cell_dir(spec))
    record: dict[str, Any] = {
        "attempt": number,
        "run_id": f"{spec.cell_id}-a{number:02d}-0000",
        "state": state,
    }
    if url is not None:
        record["gateway_base_url"] = url
    write_json_atomic(paths.attempt(number).state, record)
    if checkpoints:
        paths.attempt(number).checkpoints.mkdir(parents=True, exist_ok=True)
    return paths


def test_mirix_gateway_url_agrees_with_run_resume_gateway_port(tmp_path):
    """The scheduler supplies the gateway run.py's resume needs: the same rule as
    run.CellRunner._resume_gateway_port, on every attempt state."""
    from evals.skill_bench import run

    (spec,) = matrix(["alfworld"], ["luna"], ["mirix"])
    opts = options(tmp_path, "http://127.0.0.1:1")
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    cases = [
        ("learning", True, port),
        ("learning", False, None),
        ("learning_interrupted", True, port),
        ("learning_interrupted", False, None),
        ("frozen", False, port),
        ("frozen", True, port),
        ("testing", False, port),
        ("complete", False, None),
        ("invalid", False, None),
        ("violation", False, None),
    ]
    for state, checkpoints, expected in cases:
        shutil.rmtree(opts.cell_dir(spec), ignore_errors=True)
        paths = mirix_attempt(opts, spec, state=state, url=url, checkpoints=checkpoints)
        runner = SimpleNamespace(
            options=SimpleNamespace(
                gateway_url=None, gateway_port=None, new_attempt=False
            ),
            spec=spec,
            paths=paths,
        )
        ours = schedule.mirix_gateway_url(paths)
        theirs = run.CellRunner._resume_gateway_port(runner)
        assert (int(ours.rsplit(":", 1)[1]) if ours else None) == theirs == expected, (
            state,
            checkpoints,
        )
    # No attempt, no URL recorded, an unreadable attempt.json: nothing to supply.
    shutil.rmtree(opts.cell_dir(spec))
    assert schedule.mirix_gateway_url(CellPaths(opts.cell_dir(spec))) is None
    paths = mirix_attempt(opts, spec, state="frozen", url=None)
    assert schedule.mirix_gateway_url(paths) is None
    paths.attempt(1).state.write_text("{")
    assert schedule.mirix_gateway_url(paths) is None
    # A later attempt decides (the latest one).
    mirix_attempt(opts, spec, state="complete", url=url, number=1)
    mirix_attempt(opts, spec, state="testing", url=url, number=2)
    assert schedule.mirix_gateway_url(paths) == url


@pytest.mark.asyncio
async def test_a_mirix_resume_gets_the_gateway_its_attempt_recorded(tmp_path):
    """A MIRIX cell whose attempt recorded another gateway (a standalone run.py,
    a scheduler on another port) resumes on that gateway, which this scheduler launches
    on the recorded port (and stops at its end); the other cells keep its own. A port
    taken by something else sends the cell to a person instead of a launch run.py
    would refuse."""
    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    mirix, plain = matrix(["alfworld"], ["luna"], ["mirix", "none"])
    (taken,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    with fake.serve() as upstream:
        main_port = find_free_port()
        other = find_free_port()
        while other == main_port:
            other = find_free_port()
        opts = ScheduleOptions(
            runs_dir=tmp_path / "runs",
            gateway_port=main_port,
            gateway_upstream=upstream,
            env_file=env_file,
            poll_s=0.05,
            balance_poll_s=0.0,
            worker_module=FAKE_WORKER,
            rerun_interrupted=True,
        )
        # "localhost" is the same loopback gateway as 127.0.0.1 (run.same_gateway).
        mirix_attempt(opts, mirix, state="testing", url=f"http://localhost:{other}")
        busy = int(upstream.rsplit(":", 1)[1])
        mirix_attempt(opts, taken, state="frozen", url=f"http://127.0.0.1:{busy}")
        for spec in (mirix, taken):
            set_status(opts, spec, "test_incomplete")
        for spec in (mirix, plain, taken):
            behave(opts, spec, sleep_s=0.0)
        scheduler = Scheduler([mirix, plain, taken], opts)
        try:
            result = await asyncio.wait_for(scheduler.run(), 120)
        finally:
            # A timeout leaves the gateways only released (a worker may still run):
            # stop them here, or they outlive the test in their own sessions.
            for gw in [scheduler.gateway, *scheduler.extra_gateways.values()]:
                if gw is not None and gw.process is not None:
                    if gw.process.process.alive():
                        await gw.process.stop()
    assert sorted(result.complete) == sorted([mirix.cell_id, plain.cell_id])
    env = read_json(opts.cell_dir(mirix) / "env.json")
    assert env["gateway_url"] == f"http://127.0.0.1:{other}"
    assert env["gateway_runs_dir"] == str(opts.runs_dir)
    assert read_json(opts.cell_dir(plain) / "env.json")["gateway_url"] == (
        f"http://127.0.0.1:{main_port}"
    )
    (extra,) = events_of(opts, "gateway_extra")
    assert (extra["port"], extra["mode"], extra["cell"]) == (
        other,
        "launched",
        mirix.cell_id,
    )
    assert extra["url"] == f"http://127.0.0.1:{other}"
    # Closed on the first poll after its only cell was done, not kept to the end;
    # state.json lists it no more.
    (closed,) = events_of(opts, "gateway_extra_closed")
    assert (closed["port"], closed["mode"]) == (other, "launched")
    assert read_json(opts.control / "state.json")["gateways"] == {}
    # The port the upstream holds: a person decides, nothing launched for it.
    assert result.attention == {taken.cell_id: "test_incomplete"}
    assert result.attention_reasons[taken.cell_id] == schedule.REASON_GATEWAY_PORT
    (refused,) = events_of(opts, "launch_refused_gateway")
    assert refused["cell"] == taken.cell_id and "in use" in refused["error"]
    assert not (opts.cell_dir(taken) / "env.json").exists()
    # Both gateways this scheduler launched are stopped at its end.
    rows = read_jsonl(opts.runs_dir / "_gateway" / "owned_processes.jsonl")
    assert sorted(r["event"] for r in rows) == [
        "started",
        "started",
        "stopped",
        "stopped",
    ]


@pytest.mark.asyncio
async def test_an_attaching_scheduler_sends_such_a_cell_to_a_person(tmp_path, gateway):
    """An attaching scheduler launches no gateway: a MIRIX cell whose attempt
    recorded another one needs a person at once (its own gateway's cells go on)."""
    _gw, url = gateway
    (mirix,) = matrix(["alfworld"], ["luna"], ["mirix"])
    (same,) = matrix(["alfworld"], ["luna56"], ["mirix"])
    opts = options(tmp_path, url, rerun_interrupted=True)
    other = f"http://127.0.0.1:{find_free_port()}"
    mirix_attempt(opts, mirix, state="frozen", url=other)
    mirix_attempt(opts, same, state="testing", url=url)
    for spec in (mirix, same):
        set_status(opts, spec, "test_incomplete")
        behave(opts, spec, sleep_s=0.0)
    result = await asyncio.wait_for(Scheduler([mirix, same], opts).run(), 60)
    assert result.complete == [same.cell_id]
    assert result.attention == {mirix.cell_id: "test_incomplete"}
    assert result.attention_reasons == {mirix.cell_id: "gateway_port_unavailable"}
    (refused,) = events_of(opts, "launch_refused_gateway")
    assert refused["recorded"] == other and "attaches" in refused["error"]
    assert result.exit_code == schedule.EXIT_ATTENTION
    assert read_json(opts.control / "state.json")["gateways"] == {}
    # A launch with --new-attempt sets up a new agent: any gateway will do.
    scheduler = Scheduler([mirix], opts)
    scheduler.gateway = schedule.SchedulerGateway(url, "attach", opts.runs_dir)
    paths = CellPaths(opts.cell_dir(mirix))
    got = await scheduler._worker_gateway(mirix, paths, new_attempt=True)
    assert got is scheduler.gateway


@pytest.mark.asyncio
async def test_a_live_orphan_gateway_is_adopted_not_refused(tmp_path):
    """A gateway on another port whose scheduler died while a worker still uses it (its
    users lock held) does not make the next scheduler refuse ('an orphaned gateway ...
    may serve the live cells'): it is adopted as an extra gateway: a MIRIX cell
    that recorded it resumes on it, and at the end it is released, not stopped, while
    its worker still uses it."""
    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    (mirix,) = matrix(["alfworld"], ["luna"], ["mirix"])
    with fake.serve() as upstream:

        def at(port: int) -> ScheduleOptions:
            return ScheduleOptions(
                runs_dir=runs,
                gateway_port=port,
                gateway_upstream=upstream,
                env_file=env_file,
                poll_s=0.05,
                balance_poll_s=0.0,
                worker_module=FAKE_WORKER,
                rerun_interrupted=True,
            )

        orphan_port = find_free_port()
        first = await ensure_gateway(at(orphan_port))
        held: list[OwnerLock] = []
        scheduler: Scheduler | None = None
        # Everything after the launch is inside the try: a failure in the set-up
        # stops ``first`` too.
        try:
            assert first.process is not None and first.users_lock is not None
            first.release()  # its scheduler died
            # A worker uses it.
            held.append(OwnerLock.acquire(first.users_lock, shared=True))
            outside = CellPaths(runs / "webshop-luna56-none")
            outside.root.mkdir(parents=True)
            # A live cell outside this matrix.
            held.append(OwnerLock.acquire(outside.lock))
            main_port = find_free_port()
            while main_port == orphan_port:
                main_port = find_free_port()
            opts = at(main_port)
            mirix_attempt(
                opts, mirix, state="testing", url=f"http://127.0.0.1:{orphan_port}"
            )
            set_status(opts, mirix, "test_incomplete")
            behave(opts, mirix, sleep_s=0.0)
            scheduler = Scheduler([mirix], opts)
            result = await asyncio.wait_for(scheduler.run(), 120)
            assert result.complete == [mirix.cell_id]
            (extra,) = events_of(opts, "gateway_extra")
            assert (extra["port"], extra["mode"], extra["cell"]) == (
                orphan_port,
                "adopted",
                None,
            )
            assert read_json(opts.cell_dir(mirix) / "env.json")["gateway_url"] == (
                f"http://127.0.0.1:{orphan_port}"
            )
            assert first.process.process.alive()  # released, not stopped
            owner = schedule.gateway_owner_lock(runs, "127.0.0.1", orphan_port)
            assert not lock_is_held(owner)  # the next scheduler adopts it again
        finally:
            for lock in held:
                lock.release()
            if first.process is not None:
                await first.process.stop()
            # The live cell outside the matrix keeps the scheduler's own gateway in use,
            # so the scheduler only released it: stop it here, or it outlives the
            # test in its own process group.
            main = None if scheduler is None else scheduler.gateway
            if main is not None and main.process is not None:
                await main.process.stop()


@pytest.mark.asyncio
async def test_an_adopted_orphan_gateway_is_checked_like_every_other_adoption(
    tmp_path, monkeypatch
):
    """Adopting an orphan extra gateway at the start runs the health and budget checks
    of ensure_gateway's and gateway_on_port's adoptions. One on another budget than
    --budget (the budget was raised) or not answering /health is still adopted (a
    live worker uses it: leaving it an orphan would refuse the whole matrix, stopping it
    would kill the worker), but carries its problem: the start event names it and the
    cells that recorded it, and state.json lists it, which the supervisor alerts."""
    runs = tmp_path / "runs"
    ports = []
    while len(ports) < 4:
        port = find_free_port()
        if port not in ports:
            ports.append(port)
    main, other_budget, unhealthy, fine = ports
    opts = ScheduleOptions(
        runs_dir=runs, gateway_port=main, budget_usd=300.0, worker_module=FAKE_WORKER
    )
    rows = []
    for port in (other_budget, unhealthy, fine):
        lock = schedule.gateway_owner_lock(runs, "127.0.0.1", port)
        lock.parent.mkdir(parents=True, exist_ok=True)
        rows.append(({"pid": 4242}, lock, "127.0.0.1", port))
    monkeypatch.setattr(schedule, "_orphans_in_use", lambda options: list(rows))
    health = {other_budget: {"budget_usd": 250.0}, unhealthy: None}
    health[fine] = {"budget_usd": 300.0}

    async def fake_healthy(url: str) -> dict[str, Any] | None:
        return health[int(url.rsplit(":", 1)[1])]

    monkeypatch.setattr(schedule, "_healthy", fake_healthy)
    adopted = await schedule.adopt_live_gateways(opts)
    try:
        assert sorted(adopted) == sorted([other_budget, unhealthy, fine])
        problem = adopted[other_budget].problem
        assert "budget 250.0" in problem and "--budget 300" in problem
        assert "not healthy" in adopted[unhealthy].problem
        assert adopted[fine].problem is None
        # Every one is adopted: its owner lock is ours, so no orphan refuses the matrix.
        for gw in adopted.values():
            assert gw.mode == "adopted" and gw.owner is not None
        (mirix, luna56) = matrix(["alfworld"], ["luna", "luna56"], ["mirix"])
        url = f"http://127.0.0.1:{other_budget}"
        mirix_attempt(opts, mirix, state="testing", url=url)
        scheduler = Scheduler([mirix, luna56], opts)
        scheduler.extra_gateways = adopted
        await scheduler._announce_extra_gateways()
        events = {e["port"]: e for e in events_of(opts, "gateway_extra")}
        assert events[other_budget]["problem"] == problem
        assert events[other_budget]["cells"] == [mirix.cell_id]
        assert "problem" not in events[fine] and events[fine]["cells"] == []
        await scheduler.write_state()
        state = read_json(opts.control / "state.json")
        assert state["gateways"][str(other_budget)]["problem"] == problem
        assert state["gateways"][str(fine)]["problem"] is None
    finally:
        for gw in adopted.values():
            gw.release()


class _StopRecorder:
    """A launched gateway's process for :class:`schedule.SchedulerGateway`: records
    :meth:`stop`."""

    def __init__(self) -> None:
        self.owner = None
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_an_extra_gateway_is_closed_once_no_cell_needs_it(tmp_path):
    """An extra gateway (a MIRIX cell's recorded port) does not live until the scheduler
    ends, nor is its close skipped whenever any unrelated cell of the runs directory is
    live. It is closed on the first poll
    after nothing needs it: no worker uses it (users lock free), no live cell records
    its port, and no queued, waiting or running MIRIX cell of this scheduler does. An
    unrelated live cell does not keep it."""
    runs = tmp_path / "runs"
    ports: list[int] = []
    while len(ports) < 6:
        port = find_free_port()
        if port not in ports:
            ports.append(port)
    main, used, queued_port, idle, live_port, waiting_port = ports
    opts = ScheduleOptions(runs_dir=runs, gateway_port=main, worker_module=FAKE_WORKER)
    luna, luna56 = matrix(["alfworld"], ["luna", "luna56"], ["mirix"])
    (idle_cell,) = matrix(["webshop"], ["luna"], ["mirix"])
    (outside,) = matrix(["webshop"], ["luna56"], ["none"])
    held = []
    try:
        # An unrelated live cell (a standalone run.py on another gateway).
        unrelated = CellPaths(opts.cell_dir(outside))
        unrelated.root.mkdir(parents=True)
        held.append(OwnerLock.acquire(unrelated.lock))
        # A live MIRIX cell outside the matrix whose attempt recorded live_port.
        (stranger,) = matrix(["webshop"], ["luna56"], ["mirix"])
        paths = mirix_attempt(
            opts, stranger, state="testing", url=f"http://127.0.0.1:{live_port}"
        )
        held.append(OwnerLock.acquire(paths.lock))
        mirix_attempt(opts, luna, state="frozen", url=f"http://localhost:{queued_port}")
        mirix_attempt(
            opts, luna56, state="testing", url=f"http://127.0.0.1:{waiting_port}"
        )
        gateways = {}
        for port in (used, queued_port, idle, live_port, waiting_port):
            gateways[port] = schedule.SchedulerGateway(
                f"http://127.0.0.1:{port}",
                "launched",
                runs,
                process=_StopRecorder(),
                users_lock=schedule.gateway_users_lock(runs, "127.0.0.1", port),
                extra_port=port,
            )
        gateways[used].users_lock.parent.mkdir(parents=True, exist_ok=True)
        held.append(OwnerLock.acquire(gateways[used].users_lock, shared=True))
        scheduler = Scheduler([luna, luna56, idle_cell], opts)
        scheduler.gateway = schedule.SchedulerGateway(
            f"http://127.0.0.1:{main}", "launched", runs, process=_StopRecorder()
        )
        scheduler.extra_gateways = dict(gateways)
        scheduler.queue = [luna]
        scheduler.waiting = {luna56.cell_id: time.time() + 60}
        await scheduler._close_idle_extra_gateways()
        assert sorted(scheduler.extra_gateways) == sorted(
            [used, queued_port, live_port, waiting_port]
        )
        assert gateways[idle].process.stopped
        assert not any(
            gateways[p].process.stopped for p in (used, queued_port, live_port)
        )
        (closed,) = events_of(opts, "gateway_extra_closed")
        assert (closed["port"], closed["mode"]) == (idle, "launched")
        # The queued and waiting cells leave, the worker lets go: closed too; the
        # port a live cell records stays.
        scheduler.queue, scheduler.waiting = [], {}
        held.pop().release()
        await scheduler._close_idle_extra_gateways()
        assert sorted(scheduler.extra_gateways) == [live_port]
        assert gateways[used].process.stopped and gateways[queued_port].process.stopped
        assert (
            not scheduler.gateway.process.stopped
        )  # the main one is never closed here
        # close() of an extra gateway decides on its own users: an unrelated live cell
        # does not keep it.
        lone = schedule.SchedulerGateway(
            f"http://127.0.0.1:{idle}",
            "launched",
            runs,
            process=_StopRecorder(),
            users_lock=schedule.gateway_users_lock(runs, "127.0.0.1", idle),
            extra_port=idle,
        )
        await lone.close()
        assert lone.process.stopped
        # The main gateway still counts every live cell of the runs directory.
        main_gw = schedule.SchedulerGateway(
            f"http://127.0.0.1:{main}", "launched", runs, process=_StopRecorder()
        )
        await main_gw.close()
        assert not main_gw.process.stopped
    finally:
        for lock in held:
            lock.release()


# --- new attempts a person asks for ------------------------------------------------------


NOTE = "rerun after a harness fix"


def complete_cell(
    opts: ScheduleOptions, spec: CellSpec, *, cost: float | None = 1.25
) -> CellPaths:
    """A complete, tested cell with its attempt 1 (and that attempt's costs.json)."""
    paths = CellPaths(opts.cell_dir(spec))
    write_json_atomic(
        paths.attempt(1).state,
        {"attempt": 1, "run_id": f"{spec.cell_id}-a01-fake", "state": "complete"},
    )
    if cost is not None:
        write_json_atomic(paths.attempt(1).costs, {"totals": {"cost_total": cost}})
    write_json_atomic(paths.status, {"status": "complete", "exit_code": 0})
    return paths


def launch_argvs(opts: ScheduleOptions) -> list[list[str]]:
    return [e["argv"] for e in events_of(opts, "launched")]


def rerun_flags(argv: list[str]) -> list[str]:
    """What of ``argv`` asks for a new attempt: ``--new-attempt`` and ``--note=TEXT``
    (read as ``--note``, ``TEXT``)."""
    out = []
    for i, arg in enumerate(argv):
        if arg == "--new-attempt":
            out.append(arg)
        elif arg == "--note":
            out += [arg, argv[i + 1]]
        elif arg.startswith("--note="):
            out += ["--note", arg.partition("=")[2]]
    return out


@pytest.mark.asyncio
async def test_new_attempt_cells_relaunch_a_complete_cell_once_with_its_note(
    tmp_path, gateway
):
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["mirix"])
    opts = options(
        tmp_path,
        url,
        auto_resume=True,
        resume_backoff_base_s=0.02,
        resume_backoff_cap_s=0.05,
        new_attempt_cells=(cell.cell_id,),
        new_attempt_note=NOTE,
    )
    paths = complete_cell(opts, cell)
    # The rerun's first launch is interrupted after its new attempt exists; the
    # resume that follows carries neither flag and completes it.
    behave(opts, cell, attempts=True, statuses=["interrupted", "complete"], sleep_s=0.0)
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.exit_code == 0 and result.complete == [cell.cell_id]
    assert [rerun_flags(a) for a in launch_argvs(opts)] == [
        ["--new-attempt", "--note", NOTE],
        [],
    ]
    assert paths.attempt_numbers() == [1, 2]
    assert read_json(paths.attempt(2).state)["note"] == {"text": NOTE}
    (requested,) = events_of(opts, "new_attempts_requested")
    assert requested["cells"] == [cell.cell_id] and requested["note"] == NOTE
    (started,) = events_of(opts, "new_attempt_started")
    assert (started["cell"], started["attempt"]) == (cell.cell_id, 2)
    entry = read_json(opts.control / "resume_state.json")[cell.cell_id]
    (request,) = entry["rerun_requests"].values()
    assert (request["state"], request["from_attempt"], request["new_attempt"]) == (
        "started",
        1,
        2,
    )
    assert read_json(opts.control / "state.json")["new_attempts"][cell.cell_id] == {
        "note": NOTE,
        "from_attempt": 1,
        "estimate_usd": 1.25,
        "state": "started",
        "new_attempt": 2,
    }
    # A scheduler restarted with the same command line never reruns it again.
    again = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert again.complete == [cell.cell_id] and len(launch_argvs(opts)) == 2
    last = events_of(opts, "new_attempts_requested")[-1]
    assert last["cells"] == [] and last["already_requested"] == [cell.cell_id]
    # A deliberate second rerun takes a new note.
    other = replace(opts, new_attempt_note="and once more")
    third = await asyncio.wait_for(Scheduler([cell], other).run(), 60)
    assert third.complete == [cell.cell_id]
    assert rerun_flags(launch_argvs(opts)[-1]) == [
        "--new-attempt",
        "--note",
        "and once more",
    ]
    assert paths.attempt_numbers() == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_rerun_estimate_is_the_replaced_attempts_cost(tmp_path, gateway):
    _gw, url = gateway
    first, second = matrix(["alfworld"], ["luna", "luna56"], ["mirix"])
    opts = options(
        tmp_path,
        url,
        new_attempt_cells=(first.cell_id, second.cell_id),
        new_attempt_note=NOTE,
    )
    complete_cell(opts, first, cost=2.5)
    complete_cell(opts, second, cost=None)  # no costs.json: estimate unknown
    for spec in (first, second):
        behave(opts, spec, attempts=True, sleep_s=0.0)
    result = await asyncio.wait_for(Scheduler([first, second], opts).run(), 60)
    assert sorted(result.complete) == sorted([first.cell_id, second.cell_id])
    (requested,) = events_of(opts, "new_attempts_requested")
    assert requested["estimates"] == {first.cell_id: 2.5, second.cell_id: None}
    assert requested["estimate_total_usd"] == 2.5
    new_attempts = read_json(opts.control / "state.json")["new_attempts"]
    assert new_attempts[first.cell_id]["estimate_usd"] == 2.5
    assert new_attempts[second.cell_id]["estimate_usd"] is None


@pytest.mark.asyncio
async def test_new_attempt_cells_of_a_never_run_cell_need_nothing(tmp_path, gateway):
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path, url, new_attempt_cells=(cell.cell_id,), new_attempt_note=NOTE
    )
    behave(opts, cell, attempts=True, sleep_s=0.0)
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.complete == [cell.cell_id]
    (not_needed,) = events_of(opts, "new_attempt_not_needed")
    assert not_needed["cell"] == cell.cell_id
    assert [rerun_flags(a) for a in launch_argvs(opts)] == [[]]
    assert read_json(opts.control / "state.json")["new_attempts"] == {}


@pytest.mark.asyncio
async def test_a_rerun_of_a_cell_that_was_started_over_is_used_up(tmp_path, gateway):
    """The request replaces the attempt it saw (its run id): the same number from
    another run (the cell started over meanwhile) is never replaced by it."""
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path, url, new_attempt_cells=(cell.cell_id,), new_attempt_note=NOTE
    )
    paths = complete_cell(opts, cell)
    scheduler = Scheduler([cell], opts)
    scheduler.gateway = schedule.SchedulerGateway(url, "attach", opts.runs_dir)
    await asyncio.to_thread(scheduler.tracker.load)
    await scheduler._request_new_attempts()
    assert await scheduler._rerun_release(cell, paths) == NOTE
    write_json_atomic(
        paths.attempt(1).state,
        {"attempt": 1, "run_id": "another-run", "state": "complete"},
    )
    assert await scheduler._rerun_release(cell, paths) is None
    (request,) = scheduler.tracker.cells[cell.cell_id]["rerun_requests"].values()
    assert request["state"] == "reset"
    assert events_of(opts, "new_attempt_reset")[0]["run_id"] == "another-run"


@pytest.mark.asyncio
async def test_a_rerun_whose_attempts_cannot_be_read_is_deferred_never_blind(
    tmp_path, gateway, monkeypatch
):
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(
        tmp_path, url, new_attempt_cells=(cell.cell_id,), new_attempt_note=NOTE
    )
    complete_cell(opts, cell)
    scheduler = Scheduler([cell], opts)
    scheduler.gateway = schedule.SchedulerGateway(url, "attach", opts.runs_dir)
    await asyncio.to_thread(scheduler.tracker.load)
    await scheduler._request_new_attempts()
    monkeypatch.setattr(schedule, "latest_attempt", lambda paths: (-1, None))
    assert await scheduler.launch(cell) is False
    (deferred,) = events_of(opts, "launch_deferred")
    assert deferred["reason"] == schedule.REASON_RERUN_UNREADABLE
    assert cell.cell_id in scheduler.waiting and scheduler.active == {}
    assert events_of(opts, "new_attempt_unreadable")
    # The request is kept for a launch that can read the attempts.
    assert scheduler.tracker.rerun_pending(cell.cell_id) is not None


def test_rerun_requests_are_one_per_note_and_one_pending_per_cell(tmp_path):
    tracker = schedule.ResumeTracker(tmp_path / "resume_state.json")
    kw: dict[str, Any] = dict(
        from_attempt=1, from_run_id="r1", estimate_usd=0.5, at=1.0
    )
    first = schedule.rerun_key("a")
    assert tracker.approve_rerun("c", key=first, note="a", **kw) is True
    assert tracker.approve_rerun("c", key=first, note="a", **kw) is False
    second = schedule.rerun_key("b")
    assert tracker.approve_rerun("c", key=second, note="b", **kw) is True
    requests = tracker.cells["c"]["rerun_requests"]
    assert requests[first]["state"] == "replaced"
    assert requests[first]["replaced_by"] == second
    assert tracker.rerun_pending("c") == (second, requests[second])
    tracker.note_rerun_started("c", second, 2, 2.0)
    assert tracker.rerun_pending("c") is None
    # Kept on disk: a restarted scheduler sees both notes as used.
    again = schedule.ResumeTracker(tmp_path / "resume_state.json")
    again.load()
    assert again.approve_rerun("c", key=second, note="b", **kw) is False
    assert again.reruns(["c", "d"]) == {
        "c": {
            "note": "b",
            "from_attempt": 1,
            "estimate_usd": 0.5,
            "state": "started",
            "new_attempt": 2,
        }
    }


def test_the_note_rules_and_the_worker_argv(tmp_path):
    (cell,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = options(tmp_path, "http://127.0.0.1:1")
    argv = worker_argv(cell, opts, "http://g", tmp_path, new_attempt=True, note=NOTE)
    assert argv[-2:] == ["--new-attempt", f"--note={NOTE}"]
    # run.py's own parser reads every note the scheduler accepts as it was given, one
    # that starts with '-' too.
    from evals.skill_bench import run

    for note in (NOTE, "-fixed", "--rerun", "- a dash and spaces", "x" * 1000):
        argv = worker_argv(
            cell, opts, "http://g", tmp_path, new_attempt=True, note=note
        )
        assert schedule.note_problem(note) is None
        _spec, parsed, _args = run.parse_args(argv[3:])
        assert (parsed.new_attempt, parsed.note) == (True, note)
    # A HARNESS release carries no note; a note never goes without --new-attempt.
    assert worker_argv(cell, opts, "http://g", tmp_path, new_attempt=True)[-1] == (
        "--new-attempt"
    )
    assert not [
        a
        for a in worker_argv(cell, opts, "http://g", tmp_path, note=NOTE)
        if a.startswith("--note")
    ]
    for bad in ("", "  ", "two\nlines", "x" * (schedule.NOTE_MAX_CHARS + 1), None):
        with pytest.raises(ValueError, match="--new-attempt-note"):
            ScheduleOptions(
                runs_dir=tmp_path, new_attempt_cells=("c",), new_attempt_note=bad
            )
    # A lone note is accepted (the supervisor may withhold the cells) and ignored.
    ScheduleOptions(runs_dir=tmp_path, new_attempt_note="x")


def test_the_cli_takes_new_attempt_cells_with_a_note(tmp_path, capsys, removed_method):
    runs = tmp_path / "runs"
    (cell,) = matrix(["webshop"], ["luna"], ["mirix"])
    base = ["--runs-dir", str(runs), "--benchmark", "webshop", "--dry-run"]
    with pytest.raises(SystemExit) as exc:
        schedule.main([*base, "--new-attempt-cells", cell.cell_id])
    assert exc.value.code == 2
    assert "--new-attempt-note" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        schedule.main(
            [
                *base,
                "--new-attempt-cells",
                "alfworld-luna-mirix",
                "--new-attempt-note",
                "x",
            ]
        )
    assert "outside the selection" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        schedule.main(
            [
                *base,
                "--new-attempt-cells",
                f"webshop-luna-{removed_method}",
                "--new-attempt-note",
                "x",
            ]
        )
    assert "removed method" in capsys.readouterr().err
    # A lone note is accepted and ignored.
    assert schedule.main([*base, "--new-attempt-note", "x"]) == 0
    capsys.readouterr()
    # The dry run shows the rerun a complete cell would get, with its estimate.
    opts = ScheduleOptions(runs_dir=runs)
    complete_cell(opts, cell, cost=3.0)
    assert (
        schedule.main(
            [*base, "--new-attempt-cells", cell.cell_id, "--new-attempt-note", NOTE]
        )
        == 0
    )
    rows = {r["cell"]: r for r in map(json.loads, capsys.readouterr().out.splitlines())}
    assert rows[cell.cell_id]["plan"] == schedule.PLAN_RERUN
    assert rows[cell.cell_id]["rerun"] == {
        "note": NOTE,
        "from_attempt": 1,
        "estimate_usd": 3.0,
        "state": "to_request",
    }
    assert rows["webshop-luna-none"]["plan"] == "queued"


@pytest.mark.asyncio
async def test_a_harness_release_launch_never_carries_a_note(
    tmp_path, gateway, progress_file
):
    _gw, url = gateway
    (cell,) = matrix(["alfworld"], ["luna56"], ["awm"])
    opts = options(tmp_path, url, rerun_interrupted=True, rerun_harness_failures=True)
    released_stop(opts, cell)
    behave(opts, cell, sleep_s=0.0, progress=[progress(0, attempt=2)])
    result = await asyncio.wait_for(Scheduler([cell], opts).run(), 60)
    assert result.complete == [cell.cell_id]
    assert [rerun_flags(a) for a in launch_argvs(opts)] == [["--new-attempt"]]


# --- what a test leaves running ----------------------------------------------------------


def _sleeper(*args: str) -> subprocess.Popen[bytes]:
    """A stand-in gateway or worker: in a session of its own, ``args`` on its command
    line."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", *args],
        start_new_session=True,
    )


def _gone(proc: subprocess.Popen[bytes], timeout_s: float = 15.0) -> bool:
    try:
        proc.wait(timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return True


def test_what_a_test_left_running_is_stopped_by_its_records(tmp_path):
    """A gateway the scheduler only released, the workers of a test that timed out:
    each test's teardown stops what its records under tmp_path prove it started (an
    unsettled owned_processes.jsonl row, a process.json), never a pid whose start time
    differs (reused) and never this process."""
    runs = tmp_path / "runs"
    gw, worker, stranger = _sleeper(), _sleeper(), _sleeper()
    try:
        registry = runs / "_gateway" / "owned_processes.jsonl"
        for proc, started in (
            (gw, process_start_time(gw.pid)),
            (stranger, "Thu Jan  1 00:00:00 1970"),  # the pid now names another
        ):
            append_jsonl(
                registry,
                {
                    "event": "started",
                    "pid": proc.pid,
                    "pgid": proc.pid,
                    "started": started,
                },
            )
        append_jsonl(
            registry,
            {
                "event": "started",
                "pid": os.getpid(),
                "pgid": 0,
                "started": process_start_time(os.getpid()),
            },
        )
        cell = runs / "alfworld-luna-mirix"
        write_json_atomic(
            cell / "process.json",
            {"pid": worker.pid, "started": process_start_time(worker.pid)},
        )
        stopped = stop_test_processes(tmp_path, grace_s=5.0)
        assert stopped == sorted([gw.pid, worker.pid])
        assert _gone(gw) and _gone(worker)
        assert stranger.poll() is None
    finally:
        for proc in (gw, worker, stranger):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_a_killed_pytest_leaves_no_gateway_behind(tmp_path):
    """A SIGKILLed pytest (a chunk timeout) runs no teardown, and a launched gateway in
    its own session would run on. The watchdog the
    session starts outlives it and stops every process whose command line names the
    session's basetemp."""
    basetemp = tmp_path / "bt"
    runs = basetemp / "test_x0" / "runs"
    runs.mkdir(parents=True)
    script = "\n".join(
        [
            "import subprocess, sys, time",
            "from pathlib import Path",
            "from evals.skill_bench.tests.run_support import start_orphan_reaper",
            f"start_orphan_reaper(Path({str(basetemp)!r}))",
            "child = subprocess.Popen([sys.executable, '-c', 'import time; "
            f"time.sleep(120)', '--runs-dir', {str(runs)!r}], start_new_session=True)",
            "print(child.pid, flush=True)",
            "time.sleep(120)",
        ]
    )
    session = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    gateway_pid = None
    try:
        assert session.stdout is not None
        gateway_pid = int(session.stdout.readline())
        assert process_start_time(gateway_pid) is not None
        session.kill()  # SIGKILL: no fixture teardown runs
        session.wait()
        deadline = time.monotonic() + 30
        while process_start_time(gateway_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert process_start_time(gateway_pid) is None
    finally:
        if session.poll() is None:
            session.kill()
            session.wait()
        if session.stdout is not None:
            session.stdout.close()
        if gateway_pid is not None and process_start_time(gateway_pid):
            os.killpg(gateway_pid, signal.SIGKILL)


# --- the selection a runs directory's launches made -------------------------------------


def baselines(name: str = "phase16") -> list[CellSpec]:
    return schedule.study_matrix(name, methods=["none", "reasoningbank", "awm"])


def test_a_wider_selection_than_every_earlier_launch_needs_a_note(tmp_path):
    """A deployment may run the baselines' replicates first and MIRIX's cells later in
    the same runs directories. The first launch's cells are recorded; a later launch
    selecting cells no launch selected (MIRIX's) is refused unless --widen-selection
    says why, which is recorded; a narrower one, or one within the recorded cells,
    starts."""
    control = tmp_path / "runs" / "_schedule"
    control.mkdir(parents=True)
    first = schedule.record_selection(control, baselines(), None, pid=7)
    assert first["kind"] == "first" and first["methods"] == [
        "awm",
        "none",
        "reasoningbank",
    ]
    record = read_json(control / schedule.SELECTION_FILE)
    assert len(record["cells"]) == 12 and record["widened"] == []
    # Again, or narrower: nothing to record.
    assert schedule.record_selection(control, baselines(), None, pid=8) is None
    narrow = schedule.study_matrix("phase16", methods=["none"])
    assert schedule.selection_problem(control, narrow, None) is None
    # Every method (a relaunch without P16_METHODS): refused, naming MIRIX.
    every = schedule.study_matrix("phase16")
    problem = schedule.selection_problem(control, every, None)
    assert problem is not None
    assert "selects 4 cell(s) no earlier launch" in problem
    assert "of methods it never selected: mirix" in problem
    assert "--widen-selection" in problem and "P16_METHODS" in problem
    with pytest.raises(ValueError, match="no earlier launch"):
        schedule.record_selection(control, every, None, pid=9)
    # On purpose: recorded once, then the wider selection is the record's.
    mirix = schedule.study_matrix("phase16", methods=["mirix"])
    widened = schedule.record_selection(
        control, mirix, "add MIRIX's cells after the baselines", pid=10
    )
    assert widened["kind"] == "widened" and widened["methods"] == ["mirix"]
    assert widened["note"] == "add MIRIX's cells after the baselines"
    record = read_json(control / schedule.SELECTION_FILE)
    assert len(record["cells"]) == 16 and len(record["widened"]) == 1
    assert record["methods"] == ["awm", "mirix", "none", "reasoningbank"]
    assert schedule.selection_problem(control, every, None) is None
    assert schedule.record_selection(control, every, None, pid=11) is None


def test_a_runs_directory_from_before_the_record_is_read_from_its_start_events(
    tmp_path,
):
    """Baselines started by a scheduler without the record (an older checkout): the
    cells of its ``start`` events are the earlier selection, written down once."""
    control = tmp_path / "runs" / "_schedule"
    control.mkdir(parents=True)
    cells = [c.cell_id for c in baselines()]
    append_jsonl(control / "events.jsonl", {"event": "start", "cells": cells[:6]})
    append_jsonl(control / "events.jsonl", {"event": "stop"})
    append_jsonl(control / "events.jsonl", {"event": "start", "cells": cells[6:]})
    record = schedule.recorded_selection(control)
    assert record["inferred_from"] == "events" and record["cells"] == sorted(cells)
    mirix = schedule.study_matrix("phase16", methods=["mirix"])
    assert "mirix" in schedule.selection_problem(control, mirix, None)
    written = schedule.record_selection(control, baselines(), None, pid=3)
    assert written["kind"] == "inferred"
    assert read_json(control / schedule.SELECTION_FILE)["inferred_from"] == "events"
    assert schedule.record_selection(control, baselines(), None, pid=4) is None
    # A record that cannot be read refuses (nobody can tell what was selected).
    (control / schedule.SELECTION_FILE).write_text("[]")
    assert "cannot be read" in schedule.selection_problem(control, baselines(), None)
    # No scheduler ever started there: the first launch.
    fresh = tmp_path / "fresh" / "_schedule"
    assert schedule.selection_problem(fresh, baselines(), None) is None


def test_the_scheduler_refuses_a_wider_selection_before_it_starts(tmp_path, capsys):
    """The command line records the selection and refuses a wider one (exit 4) before
    any gateway or worker; a note it refuses is a usage error."""
    runs = tmp_path / "runs"
    control = runs / "_schedule"
    control.mkdir(parents=True)
    schedule.record_selection(control, baselines(), None, pid=1)
    code = schedule.main(["--runs-dir", str(runs), "--study", "phase16"])
    assert code == schedule.EXIT_REFUSED
    error = json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]
    assert "of methods it never selected: mirix" in error
    assert not (runs / "_gateway").exists()
    with pytest.raises(SystemExit):
        schedule.main(
            ["--runs-dir", str(runs), "--study", "phase16", "--widen-selection", "a\nb"]
        )
    args = schedule.build_parser().parse_args(
        ["--runs-dir", str(runs), "--widen-selection", "why"]
    )
    options = schedule.options_from_args(args)
    assert options.guard_selection is True and options.widen_selection == "why"
    # A scheduler built by hand (the tests' own) keeps no record unless asked.
    assert schedule.ScheduleOptions(runs_dir=runs).guard_selection is False


# --- host caps every scheduler publishes; a restart keeps its predecessor's record ----


def _starts(opts: ScheduleOptions) -> list[float]:
    return sorted(r["t"] for r in worker_log(opts) if r["event"] == "start")


def _live_at_each_start(rows: list[dict[str, Any]], since: float) -> list[int]:
    """How many workers ran right after each start at or after ``since``."""
    ordered = sorted(rows, key=lambda r: (r["t"], r["event"] == "start"))
    live, out = 0, []
    for row in ordered:
        live += 1 if row["event"] == "start" else -1
        if row["event"] == "start" and row["t"] >= since:
            out.append(live)
    return out


@pytest.mark.asyncio
async def test_the_lowest_published_host_cap_is_enforced_and_recorded(
    tmp_path, gateway
):
    """With schedulers at webshop=4 and webshop=2 on one host, were each to enforce its
    own cap, the lower one would starve until the higher one's queue drained, while the
    higher one went past its load limit. Each publishes its cap; every one enforces the
    lowest (host_cap_mismatch recorded on both sides), and the one running
    none of the benchmark's cells gets the next free slot."""
    _gw, url = gateway
    a_cells = matrix(
        ["webshop"], ["luna", "luna56"], ["mirix", "reasoningbank", "awm", "none"]
    )
    b_cells = matrix(["webshop"], ["luna"], ["mirix", "none"])
    a = _replicate(
        tmp_path, "a", url, concurrency=4, host_benchmark_concurrency=(("webshop", 4),)
    )
    b = _replicate(
        tmp_path, "b", url, concurrency=4, host_benchmark_concurrency=(("webshop", 2),)
    )
    for spec in a_cells:
        behave(a, spec, sleep_s=1.5)
    for spec in b_cells:
        behave(b, spec, sleep_s=0.3)
    sa = Scheduler(a_cells, a, sampler=FakeHost(50.0), webshop_ready=_ready)
    sb = Scheduler(b_cells, b, sampler=FakeHost(50.0), webshop_ready=_ready)
    ta = asyncio.ensure_future(sa.run())
    await eventually(lambda: len(sa.active) == 4, 30)
    tb = asyncio.ensure_future(sb.run())
    results = await asyncio.gather(ta, tb)
    assert [r.exit_code for r in results] == [0, 0]
    a_starts, b_starts = _starts(a), _starts(b)
    # B is not starved until A's queue drains ...
    assert b_starts[0] < a_starts[-1]
    # ... and while B runs, the host never runs more than the lower cap (once B's
    # scheduler is gone, A's own cap is the lowest a live scheduler publishes).
    rows = worker_log(a) + worker_log(b)
    b_end = max(r["t"] for r in worker_log(b) if r["event"] != "start")
    during = [
        n
        for n, row in zip(
            _live_at_each_start(rows, b_starts[0]),
            sorted(
                r["t"] for r in rows if r["event"] == "start" and r["t"] >= b_starts[0]
            ),
        )
        if row <= b_end
    ]
    assert during and max(during) <= 2
    (theirs,) = events_of(a, "host_cap_mismatch")
    assert theirs["peer"] == str(b.runs_dir) and theirs["benchmark"] == "webshop"
    assert (theirs["theirs"], theirs["ours"], theirs["in_force"]) == (2, 4, 2)
    (ours,) = events_of(b, "host_cap_mismatch")
    assert (ours["theirs"], ours["ours"], ours["in_force"]) == (4, 2, 2)
    held = [e["reason"] for e in events_of(a, "host_cap_wait")]
    assert any(
        "webshop=2 (the lowest of this host's schedulers; this scheduler's own is 4)"
        in r
        for r in held
    )


@pytest.mark.asyncio
async def test_a_scheduler_running_none_of_a_benchmark_gets_the_next_slot(
    tmp_path, gateway
):
    """With equal caps, a scheduler whose cell finishes and is replaced within one loop
    iteration must not keep the slot while a peer running none of the benchmark's cells
    waits for its whole queue. The slot goes to the scheduler
    waiting with none, first come first served (two waiting ones never hold back for
    each other: with a cap of 1 the two alternate)."""
    _gw, url = gateway
    caps = (("webshop", 1),)
    a_cells = matrix(["webshop"], ["luna"], ["reasoningbank", "awm", "none"])
    (b_cell,) = matrix(["webshop"], ["luna"], ["mirix"])
    a = _replicate(tmp_path, "a", url, concurrency=4, host_benchmark_concurrency=caps)
    b = _replicate(tmp_path, "b", url, concurrency=4, host_benchmark_concurrency=caps)
    for spec in a_cells:
        behave(a, spec, sleep_s=0.6)
    behave(b, b_cell, sleep_s=0.2)
    sa = Scheduler(a_cells, a, sampler=FakeHost(50.0), webshop_ready=_ready)
    sb = Scheduler([b_cell], b, sampler=FakeHost(50.0), webshop_ready=_ready)
    ta = asyncio.ensure_future(sa.run())
    await eventually(lambda: len(sa.active) == 1, 30)
    tb = asyncio.ensure_future(sb.run())
    await eventually(lambda: bool(events_of(b, "host_cap_wait")), 30)
    results = await asyncio.gather(ta, tb)
    assert [r.exit_code for r in results] == [0, 0]
    a_starts, (b_start,) = _starts(a), _starts(b)
    assert len(a_starts) == 3
    assert b_start < a_starts[1]  # the slot A's first cell freed went to B
    assert max_overlap(worker_log(a) + worker_log(b)) == 1
    assert events_of(a, "host_cap_mismatch") == []
    state = read_json(a.control / "state.json")
    assert state["host"]["benchmark_concurrency_in_force"] == {"webshop": 1}
    # Why a cell is held while a peer waits with none (its state.json host.held).
    waiting = host_registry.Peer(str(b.runs_dir), 1, "s", None, waiting={"webshop": 1})
    reason = sa._host_cap_problem(a_cells[0], [waiting])
    assert reason is not None and reason.endswith(
        f"; 1 slot(s) kept for scheduler(s) waiting with none: {b.runs_dir}"
    )


@pytest.mark.asyncio
async def test_a_restarted_scheduler_keeps_its_predecessors_record_until_it_adopts(
    tmp_path, gateway
):
    """A restarted scheduler must not publish an empty record (with its write of
    state.json) before it adopted the workers its predecessor left: the peers would stop
    counting them for that window (the /balance read and housekeeping included). The
    orphaned record stays until the workers are adopted; a scheduler that stops
    before adopting them leaves it as it is."""
    _gw, url = gateway
    cells = matrix(["webshop"], ["luna"], ["none", "awm"])
    opts = _replicate(tmp_path, "r1", url, concurrency=2)
    release = tmp_path / "release"
    for spec in cells:
        behave(opts, spec, until=str(release), sleep_s=0.0)
    first = Scheduler(cells, opts, sampler=FakeHost(50.0), webshop_ready=_ready)
    task = asyncio.ensure_future(first.run())
    await eventually(lambda: len(first.active) == 2, 30)
    await eventually(lambda: len(_starts(opts)) == 2, 30)  # each holds its cell
    first.stop.set()
    await asyncio.wait_for(task, 30)

    def peer_count() -> int:
        view = host_registry.HostRegistry(tmp_path / "host", tmp_path / "rX" / "runs")
        return sum(p.cells("webshop") for p in view.peers())

    assert peer_count() == 2
    # One that stops before adopting: the record stays the predecessor's.
    broken = Scheduler(cells, opts, sampler=FakeHost(50.0), webshop_ready=_ready)

    async def fail() -> None:
        raise RuntimeError("classify failed")

    broken.classify = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="classify failed"):
        await broken.run()
    assert peer_count() == 2
    seen: dict[str, int] = {}
    second = Scheduler(cells, opts, sampler=FakeHost(50.0), webshop_ready=_ready)
    original = second.classify

    async def spy() -> None:
        seen["at_classify"] = peer_count()
        await original()
        seen["adopted"] = len(second.active)

    second.classify = spy  # type: ignore[method-assign]
    task = asyncio.ensure_future(second.run())
    await eventually(lambda: "adopted" in seen, 30)
    assert seen == {"at_classify": 2, "adopted": 2}
    await eventually(lambda: peer_count() == 2, 10)  # now its own, live record
    release.touch()
    result = await asyncio.wait_for(task, 60)
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_an_orphaned_record_marks_the_emergency_victim(tmp_path, monkeypatch):
    """A scheduler that stops while its emergency victim still runs
    marks it in the orphaned record, with emergency_pending, so the peers wait for it."""
    (spec,) = matrix(["alfworld"], ["luna"], ["none"])
    opts = ScheduleOptions(runs_dir=tmp_path / "runs", host_dir=tmp_path / "host")
    scheduler = Scheduler([spec], opts, sampler=FakeHost(50.0))
    scheduler.started = process_start_time(os.getpid())
    scheduler._host_ready = True
    scheduler.active = {
        spec.cell_id: SimpleNamespace(pid=os.getpid(), spec=spec)  # type: ignore[dict-item]
    }
    scheduler._stopping[spec.cell_id] = schedule.STOPPED_MEMORY_EMERGENCY
    scheduler._host_emergency_at = 5.0
    await scheduler._leave_host()
    record = json.loads(scheduler.host.path.read_text())
    assert record["orphaned"] is True and record["emergency_pending"] is True
    assert record["workers"][0]["emergency_victim"] is True
    view = host_registry.HostRegistry(tmp_path / "host", tmp_path / "other")
    (peer,) = view.peers()
    assert peer.emergency_pending is True
