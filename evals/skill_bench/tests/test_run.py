"""One cell end to end (H1, H8, H9 cost inputs, H10.4).

Every test runs the real ``run.py`` runner, rollout, actor and a real loopback gateway
over the fake upstream (``tests/fakes.py``); nothing reaches a provider. The benchmark is
:class:`~evals.skill_bench.tests.run_support.TinyEnv` (or ALFWorld with scripted games),
and the methods are ``none`` and
:class:`~evals.skill_bench.tests.run_support.StoringMethod`, a trivial method that
stores something.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol, run
from evals.skill_bench.cell_records import (
    ATTEMPT_COMPLETE,
    ATTEMPT_INVALID,
    ATTEMPT_LEARNING,
    ATTEMPT_VIOLATION,
    EXIT_CODES,
    CellPaths,
    read_json,
    read_jsonl,
    summarize_ledger,
    write_json_atomic,
)
from evals.skill_bench.envs import alfworld as alf
from evals.skill_bench.infra.mirix_server import CellServerSpec, MirixServer
from evals.skill_bench.infra.processes import OwnerLock, lock_is_held
from evals.skill_bench.methods.none import NoMemory
from evals.skill_bench.run import CellSpec, seeded_order, select_items
from evals.skill_bench.tests.fakes import (
    FakeUpstream,
    Reply,
    serve_gateway,
    write_stub_server,
)
from evals.skill_bench.tests.rollout_support import text_reply
from evals.skill_bench.tests.run_support import (
    FIXED_GIT,
    InfraFailure,
    TINY_SIZES,
    StoringMethod,
    TinyEnv,
    actor_questions,
    conformant_gateway,
    eventually,
    ledger,
    read_rows,
    runner,
    tiny_answer,
    tiny_responder,
)
from evals.skill_bench.tests.test_env_alfworld import Games, act

SPEC = CellSpec("alfworld", "luna", "reasoningbank")
#: The cell directory's instance id, fixed here so run ids are known in advance (run.py
#: draws a random one per cell directory; ``test_run_ids_carry_the_cell_instance``
#: runs with the real one).
INSTANCE = "0000feed"
RUN_A1 = f"{SPEC.cell_id}-a01-{INSTANCE}"
RUN_A2 = f"{SPEC.cell_id}-a02-{INSTANCE}"


@pytest.fixture(autouse=True)
def _fixed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "new_instance_id", lambda: INSTANCE)


@pytest.fixture(autouse=True)
def _frozen_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This test's own frozen record: a copy of the repository's, frozen to this tree (a
    no-op on a frozen tree). run.py freezes automatically, so a test that changes the
    protocol writes this copy, never the repository's record; CLI subprocesses inherit
    the variable."""
    from evals.skill_bench.tools import freeze_protocol

    copy = tmp_path / "_frozen_record" / "protocol_frozen.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(protocol.FROZEN_PROTOCOL_JSON, copy)
    freeze_protocol.freeze(
        reason="the tests baseline", path=copy, git=lambda: dict(FIXED_GIT)
    )
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, str(copy))
    return copy


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream(responder=tiny_responder)


@pytest.fixture
def gateway(tmp_path: Path, fake: FakeUpstream):
    with conformant_gateway(tmp_path, fake) as (gw, url):
        yield gw, url


def storing(events: list[Any], **kw: Any):
    return lambda: StoringMethod(events, **kw)


def cell_paths(tmp_path: Path, spec: CellSpec = SPEC) -> CellPaths:
    return CellPaths(tmp_path / "runs" / spec.cell_id)


def train_order() -> list[str]:
    return [t.raw_id for t in seeded_order(TinyEnv().load_tasks("train"))]


def expected_test_order() -> list[str]:
    """Protocol test items run in the seed-42 order of the test split, not in manifest
    order, which can group items by a scorer-side label (H5)."""
    return [t.raw_id for t in seeded_order(TinyEnv().load_tasks("test"))]


def question(raw_id: str) -> str:
    split, index = raw_id.split(":")
    return f"Q {split} {int(index)}?"


# --- cell definition ------------------------------------------------------------------------


def test_cell_ids_and_spec_validation():
    assert SPEC.cell_id == "alfworld-luna-reasoningbank" and SPEC.test_split == "test"
    smoke = CellSpec("alfworld", "openai/gpt-5.6-luna", "mirix", "smoke", 5, 3)
    assert smoke.cell_id == "alfworld-luna56-mirix-smoke-l5-t3"
    assert smoke.model == protocol.MODELS["luna56"] and smoke.test_split == "val"
    assert CellSpec("webshop", "luna56", "none").learns is False
    for bad in (
        dict(benchmark="webarena", model="luna", method="none"),
        dict(benchmark="webshop", model="gpt-4", method="none"),
        dict(benchmark="webshop", model="luna", method="exprag"),
        dict(benchmark="webshop", model="luna", method="none", n_test=3),
        dict(benchmark="webshop", model="luna", method="none", split_mode="smoke"),
        dict(
            benchmark="webshop",
            model="luna",
            method="none",
            split_mode="smoke",
            n_train=1,
            n_test=0,
        ),
        dict(
            benchmark="alfworld",
            model="luna",
            method="none",
            split_mode="smoke",
            n_train=1,
            n_test=19,  # ALFWorld val has 18 items
        ),
    ):
        with pytest.raises(ValueError):
            CellSpec(**bad)  # type: ignore[arg-type]


def test_a_removed_method_is_refused_with_the_decision(
    tmp_path, capsys, removed_method
):
    """A method the study removed: its cells stay on disk, but run.py starts none (the
    CLI and the Python API), and the refusal names the decision rather than an unknown
    method or argparse's "invalid choice"."""
    with pytest.raises(ValueError, match="removed from the study") as exc:
        CellSpec("alfworld", "luna56", removed_method)
    assert "example removal" in str(exc.value)
    out = tmp_path / f"alfworld-luna56-{removed_method}"
    with pytest.raises(SystemExit) as stop:
        run.parse_args(
            [
                *("--benchmark", "alfworld", "--model", "luna56"),
                *("--method", removed_method, "--out", str(out)),
                *("--gateway-url", "http://127.0.0.1:1"),
            ]
        )
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert removed_method in err and "example removal" in err
    assert "invalid choice" not in err and not out.exists()
    with pytest.raises(SystemExit):
        run.parse_args(
            [
                *("--benchmark", "alfworld", "--model", "luna56", "--method", "exprag"),
                *("--gateway-url", "http://127.0.0.1:1"),
            ]
        )
    assert "invalid choice: 'exprag'" in capsys.readouterr().err


def test_items_follow_the_protocol_order_and_smoke_never_loads_test():
    env = TinyEnv()
    learning, test = select_items(env, SPEC)
    manifest = [t.raw_id for t in env.load_tasks("train")]
    expected = list(manifest)
    random.Random(42).shuffle(expected)
    assert [t.raw_id for t in learning] == expected != manifest
    test_manifest = [t.raw_id for t in env.load_tasks("test")]
    assert [t.raw_id for t in test] == expected_test_order() != test_manifest
    assert sorted(expected_test_order()) == sorted(test_manifest)  # the whole split

    env = TinyEnv()
    learning, test = select_items(
        env, CellSpec("alfworld", "luna", "awm", "smoke", 2, 3)
    )
    assert [t.raw_id for t in learning] == expected[:2]
    val = [t.raw_id for t in TinyEnv().load_tasks("val")]
    random.Random(42).shuffle(val)
    assert [t.raw_id for t in test] == val[:3]
    assert (
        "test" not in env.loaded
    )  # the test split is not touched before the formal run

    env = TinyEnv()
    learning, _ = select_items(env, CellSpec("alfworld", "luna", "none"))
    assert learning == [] and "train" not in env.loaded  # H1.6


# --- a whole cell ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cell_learns_in_order_freezes_and_tests(tmp_path, fake, gateway):
    """H1.4-H1.6: hooks per learning item in the seed-42 order, freeze, test-only before_task."""
    _gw, url = gateway
    events: list[Any] = []
    made: list[StoringMethod] = []

    def factory() -> StoringMethod:
        made.append(StoringMethod(events))
        return made[-1]

    outcome = await runner(tmp_path, url, factory=factory).run()
    assert outcome.status == "complete" and outcome.exit_code == 0
    assert outcome.attempt == 1

    order = train_order()
    names = [e[0] for e in events]
    # Learning: the init checkpoint after setup; then per item before_task -> on_message
    # (question, answer) -> the played checkpoint -> after_task -> idle -> the done
    # checkpoint (each checkpoint records the method's state and snapshot).
    first_test = names.index("before_task", names.index("finalize"))
    assert events[1:3] == [("checkpoint",), ("snapshot",)]
    learning_part = events[3 : names.index("finalize")]
    per_item = [learning_part[i : i + 9] for i in range(0, len(learning_part), 9)]
    assert len(per_item) == len(order)
    for raw_id, hooks in zip(order, per_item):
        assert hooks == [
            ("before_task", "learning", question(raw_id)),
            ("on_message", "user"),
            ("on_message", "assistant"),
            ("checkpoint",),
            ("snapshot",),
            ("after_task", question(raw_id)),
            ("wait_until_idle",),
            ("checkpoint",),
            ("snapshot",),
        ]
    # Freeze (H1.5), then only before_task, then the H10.4 snapshot and close.
    assert names[names.index("finalize") : first_test] == [
        "finalize",
        "wait_until_idle",
        "snapshot",
    ]
    test_part = events[first_test:]
    assert sorted(e[2] for e in test_part if e[0] == "before_task") == sorted(
        question(r) for r in expected_test_order()
    )
    assert {e[0] for e in test_part} == {"before_task", "snapshot", "close"}

    paths = cell_paths(tmp_path)
    attempt = paths.attempt(1)
    # The method's only directory is its own: never the scorer's records (H10.1).
    cell = made[0].cell
    assert cell is not None and cell.work_dir == attempt.method_dir == cell.cell_dir
    assert cell.run_id == RUN_A1 and cell.options == {}
    # Learning curve: every item's score by stream position, scorer side only (H9).
    learning = read_rows(attempt.learning)
    assert [r["raw_id"] for r in learning] == order
    assert [r["position"] for r in learning] == list(range(len(order)))
    assert [r["primary"] for r in learning] == [
        1.0 if tiny_answer("train", int(r.split(":")[1])) else 0.0 for r in order
    ]
    # Learning items use the memory learned so far.
    injected = [r for r in read_rows(attempt.injections) if r["phase"] == "learning"]
    assert [r["text"] for r in injected] == [""] + [
        f"MEMORY holds {i} lessons" for i in range(1, len(order))
    ]
    assert injected[0]["tokens"] == 0 and all(r["tokens"] > 0 for r in injected[1:])
    # Test: the final results in test order, all scored, with the frozen memory.
    final = read_rows(paths.results)
    assert [r["raw_id"] for r in final] == expected_test_order()
    assert all(r["phase"] == "test" and r["score"] is not None for r in final)
    assert [r["primary"] for r in final] == [
        1.0 if tiny_answer("test", int(r.split(":")[1])) else 0.0
        for r in expected_test_order()
    ]
    tested = [r for r in read_rows(attempt.injections) if r["phase"] == "test"]
    assert {r["text"] for r in tested} == {f"MEMORY holds {len(order)} lessons"}
    # Freeze equality (H10.4) and the records.
    state = read_json(attempt.state)
    assert state["state"] == ATTEMPT_COMPLETE
    frozen = read_json(attempt.frozen_snapshot)
    assert frozen["snapshot"]["memory"] == [question(r) for r in order]
    assert state["frozen"]["sha256"] == frozen["sha256"]
    assert [c["kind"] for c in state["snapshot_checks"]] == ["after_test"]
    assert all(c["equal"] for c in state["snapshot_checks"])
    assert state["test_reruns"] == []
    for row in [*learning, *final]:
        assert (paths.root / row["trajectory"]).is_file()
        assert row["gateway_call_ids"] and row["run_id"] == RUN_A1
    summary = read_json(paths.summary)
    assert summary["test"]["items"] == TINY_SIZES["test"]
    assert summary["frozen_snapshot_sha256"] == frozen["sha256"]
    status = read_json(paths.status)
    assert status["status"] == "complete" and status["exit_code"] == 0
    # Learning checkpoints: the latest two are kept, LATEST names
    # the done checkpoint of the last item, and every one was marked in the ledger.
    last = len(order) - 1
    assert sorted(p.name for p in attempt.checkpoints.iterdir()) == [
        "LATEST",
        f"c{last:04d}-done",
        f"c{last:04d}-played",
    ]
    latest = read_json(attempt.latest_checkpoint)
    assert latest["id"] == f"c{last:04d}-done" and latest["ordinal"] == 2 * last + 2
    record = read_json(attempt.checkpoint(latest["id"]) / "checkpoint.json")
    assert record["method"]["state"] == {
        "memory": [question(r) for r in order],
        "learned": len(order),
    }
    assert record["method"]["snapshot_sha256"] == frozen["sha256"]
    assert record["attempt_files"]["learning.jsonl"] == attempt.learning.stat().st_size
    markers = [r for r in ledger(tmp_path, RUN_A1) if r.get("event") == "checkpoint"]
    assert [m["id"] for m in markers] == ["c0000-init"] + [
        f"c{p:04d}-{kind}" for p in range(len(order)) for kind in ("played", "done")
    ]
    assert state["host"] == run.host_info()
    assert state["ledger_rel"] == f"{RUN_A1}/gateway.jsonl"
    assert state["resumes"] == [] and state["checkpoint_format"] == 1
    assert state["learning"] == {
        "items": len(order),
        "done": len(order),
        "played": len(order),
        "checkpoint": latest["id"],
    }
    work = [r for r in read_rows(paths.launches) if r["event"] == "work"]
    assert work and all(r["host"] == run.host_info() for r in work)
    assert read_json(paths.manifest)["last_host"] == run.host_info()


@pytest.mark.asyncio
async def test_costs_providers_and_ledger_linkage_are_recorded(tmp_path, fake, gateway):
    """H9 cost inputs and the providers come from the attempt's gateway ledger."""
    _gw, url = gateway
    await runner(tmp_path, url, factory=storing([])).run()
    paths = cell_paths(tmp_path)
    costs = read_json(paths.attempt(1).costs)
    n_train, n_test = TINY_SIZES["train"], TINY_SIZES["test"]
    assert costs["run_id"] == RUN_A1
    assert costs["by_phase_role"]["learning/actor"]["calls"] == n_train
    assert costs["by_phase_role"]["test/actor"]["calls"] == n_test
    assert costs["by_phase_role"]["test/actor"]["cost_total"] == pytest.approx(
        0.000123 * n_test
    )
    assert costs["providers"] == {
        "learning/actor": {"FakeProvider": n_train},
        "test/actor": {"FakeProvider": n_test},
    }
    assert costs["phases"] == ["setup", "learning", "test"]
    assert costs["linkage"]["actor_calls_referenced"] == n_train + n_test
    assert costs["linkage"]["missing_from_ledger"] == []
    assert costs["protocol_conformant"] is True
    manifest = read_json(paths.manifest)
    assert manifest["providers"] == costs["providers"]
    assert manifest["cell"]["protocol_fingerprint"] == protocol.protocol_fingerprint()
    assert manifest["cell"]["learning_items"] == train_order()
    assert manifest["cell"]["test_items"] == expected_test_order()
    assert manifest["gateway"]["protocol_conformant"] is True
    assert manifest["git"] == FIXED_GIT
    assert [a["state"] for a in manifest["attempts"]] == ["complete"]
    # Every actor call of the cell is on the ledger with its opaque item and phase.
    calls = [r for r in ledger(tmp_path, RUN_A1) if r.get("event") == "call"]
    assert {(r["role"], r["phase"]) for r in calls} == {
        ("actor", "learning"),
        ("actor", "test"),
    }
    assert all(r["item"] and r["item"].startswith("t") for r in calls)


@pytest.mark.asyncio
async def test_the_no_memory_method_only_runs_the_test_phase(tmp_path, fake, gateway):
    _gw, url = gateway
    spec = CellSpec("alfworld", "luna56", "none")
    outcome = await runner(tmp_path, url, spec=spec, factory=NoMemory).run()
    assert outcome.status == "complete"
    paths = cell_paths(tmp_path, spec)
    assert read_rows(paths.attempt(1).learning) == []
    assert [r["raw_id"] for r in read_rows(paths.results)] == expected_test_order()
    assert set(actor_questions(fake)) == {question(r) for r in expected_test_order()}
    assert all(r["tokens"] == 0 for r in read_rows(paths.attempt(1).injections))
    assert read_json(paths.attempt(1).frozen_snapshot)["snapshot"] == {
        "method": "none",
        "memory": None,
    }


@pytest.mark.asyncio
async def test_a_complete_cell_is_not_run_again(tmp_path, fake, gateway):
    _gw, url = gateway
    events: list[Any] = []
    assert (await runner(tmp_path, url, factory=storing(events)).run()).exit_code == 0
    calls = len(fake.calls("chat/completions"))
    again = await runner(tmp_path, url, factory=storing(events)).run()
    assert again.status == "complete" and again.detail == "already complete"
    assert len(fake.calls("chat/completions")) == calls
    refused = await runner(
        tmp_path, url, factory=storing(events), new_attempt=True
    ).run()
    assert refused.status == "refused"
    # A refusal changes nothing: the finished cell's status stays complete.
    status = read_json(cell_paths(tmp_path).status)
    assert status["status"] == "complete" and status["exit_code"] == 0
    assert status["last_refusal"]["status"] == "refused"
    other = TinyEnv({"train": 6, "val": 4, "test": 6})  # e.g. a protocol change
    drift = await runner(tmp_path, url, env=other, factory=storing(events)).run()
    assert drift.status == "refused"
    assert read_json(cell_paths(tmp_path).status)["status"] == "complete"


@pytest.mark.asyncio
async def test_test_items_run_concurrently_within_the_bound(tmp_path, fake, gateway):
    _gw, url = gateway
    live = {"now": 0, "max": 0}

    async def slow(_task: Any) -> None:
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.05)
        live["now"] -= 1

    method = storing([], on_before_task=slow)
    env = TinyEnv({"train": 0, "val": 1, "test": 8})
    outcome = await runner(
        tmp_path, url, env=env, factory=method, test_concurrency=3
    ).run()
    assert outcome.status == "complete"
    assert live["max"] == 3


# --- H8: learning failures restart the cell ------------------------------------------------------


@pytest.mark.asyncio
async def test_an_infra_learning_failure_is_resumed_from_its_checkpoint(
    tmp_path, fake, gateway
):
    """An infrastructure failure in the learning phase does not restart the cell. The
    attempt stays (learning_interrupted, exit 8, its gateway run
    not abandoned) and the next invocation resumes it from the checkpoint: here the
    played checkpoint of the item whose after_task failed, so only that after_task runs
    again, on the view of the recorded rollout."""
    _gw, url = gateway
    events: list[Any] = []
    order = train_order()
    faults = {"after_task": {question(order[2])}}
    failed = await runner(
        tmp_path, url, factory=storing(events, fail_once=faults)
    ).run()
    assert failed.status == "learning_interrupted" and failed.exit_code == 8
    assert failed.failure_kind == "infra"
    paths = cell_paths(tmp_path)
    first = read_json(paths.attempt(1).state)
    assert isinstance(first, dict)
    assert first["state"] == "learning_interrupted" and "abandoned" not in first
    assert first["failure"]["kind"] == "infra"
    assert first["failure_stage"] == "after_task" and first["failure_position"] == 2
    assert first["needs_decision"] is None and first["resume_from"] is None
    assert first["learning"]["checkpoint"] == "c0002-played"
    # Item 2 was played and recorded before after_task: it is scored, once.
    rows = read_rows(paths.attempt(1).learning)
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert all(r["score"] is not None for r in rows)
    failures = read_rows(paths.attempt(1).failures)
    assert [f["recovery"] for f in failures] == ["resume_from_checkpoint"]
    assert failures[0]["item"] == rows[-1]["task_uid"]
    assert read_json(paths.status)["status"] == "learning_interrupted"
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.post(
            f"/runs/{RUN_A1}/actor/v1/chat/completions",
            json={"model": SPEC.model, "messages": [{"role": "user", "content": "x"}]},
            headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
        )
    assert response.status_code == 200  # the run goes on: nothing was abandoned

    # The next invocation resumes attempt 1 from c0002-played: item 2's after_task on
    # the recorded view, then items 3..5.
    events.clear()
    outcome = await runner(tmp_path, url, factory=storing(events)).run()
    assert outcome.status == "complete" and outcome.attempt == 1, outcome.detail
    assert events[0] == ("setup", "learning", 1, RUN_A1, "c0002-played")
    learned = [e[2] for e in events if e[0] == "before_task" and e[1] == "learning"]
    assert learned == [question(r) for r in order[3:]]
    assert [e[1] for e in events if e[0] == "after_task"] == [
        question(r) for r in order[2:]
    ]
    rows = read_rows(paths.attempt(1).learning)
    assert [r["position"] for r in rows] == list(range(len(order)))
    assert [r["raw_id"] for r in rows] == order
    state = read_json(paths.attempt(1).state)
    assert state["state"] == ATTEMPT_COMPLETE
    frozen = read_json(paths.attempt(1).frozen_snapshot)
    assert frozen["snapshot"]["memory"] == [question(r) for r in order]
    # One discarded segment: from c0002-played to the resume marker. This method's
    # after_task makes no call; the one call in it is the probe above, made under the
    # run while the attempt was interrupted.
    discarded = read_rows(paths.attempt(1).discarded_log)
    assert len(discarded) == 1
    segment = discarded[0]
    assert segment["segment"] == 1 and segment["checkpoint"]["id"] == "c0002-played"
    assert segment["from_state"] == "learning_interrupted"
    assert segment["reason"] == first["failure"]["code"] == "gateway_unreachable"
    assert segment["in_flight"] == {
        "position": 2,
        "task_uid": rows[2]["task_uid"],
        "stage": "after_task",
        "trajectory_recorded": True,
    }
    assert [(c["role"], c["status"]) for c in segment["calls"]] == [("actor", "ok")]
    assert segment["cost_total"] == pytest.approx(0.000123)
    assert segment["unknown_cost_calls"] == 0
    assert segment["restore"]["files_equal"] is True
    assert (
        segment["restore"]["snapshot_sha256_actual"]
        == segment["restore"]["snapshot_sha256_expected"]
    )
    marks = [r for r in ledger(tmp_path, RUN_A1) if r.get("event") == "resume"]
    assert [(m["segment"], m["checkpoint"]) for m in marks] == [(1, "c0002-played")]
    assert [r["segment"] for r in state["resumes"]] == [1]
    checks = [c for c in state["snapshot_checks"] if c["kind"] == "learning_resume"]
    assert len(checks) == 1 and checks[0]["equal"] is True
    assert checks[0]["checkpoint"] == "c0002-played"
    manifest = read_json(paths.manifest)
    assert [(a["attempt"], a["state"], a["resumes"]) for a in manifest["attempts"]] == [
        (1, "complete", 1)
    ]
    assert read_json(paths.summary)["attempt"] == 1


@pytest.mark.asyncio
async def test_a_hung_finalize_interrupts_the_learning_phase(
    tmp_path, fake, gateway, monkeypatch
):
    """``finalize`` and the idle waits are bounded: a hung method is INFRA, not a hang;
    the freeze resumes from the last item's done checkpoint."""
    _gw, url = gateway
    monkeypatch.setattr(run, "IDLE_WAIT_GRACE_S", 0.05)

    class Hangs(StoringMethod):
        async def finalize(self) -> None:
            await asyncio.sleep(3600)

    outcome = await runner(
        tmp_path, url, factory=lambda: Hangs([]), finalize_timeout_s=0.1
    ).run()
    assert outcome.status == "learning_interrupted"
    state = read_json(cell_paths(tmp_path).attempt(1).state)
    assert state["failure_stage"] == "finalize"
    assert state["failure"]["code"] == "idle_wait_timeout"
    assert state["failure"]["kind"] == "infra"
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete" and done.attempt == 1
    last = len(train_order()) - 1
    assert events[0] == ("setup", "learning", 1, RUN_A1, f"c{last:04d}-done")
    assert not [e for e in events if e[0] == "before_task" and e[1] == "learning"]


@pytest.mark.asyncio
async def test_max_learning_attempts_resumes_an_interrupted_learning_phase_at_once(
    tmp_path, fake, gateway
):
    """Opt-in (default 1): the same invocation resumes the learning phase from its
    checkpoint, in the same attempt."""
    _gw, url = gateway
    events: list[Any] = []
    made: list[StoringMethod] = []

    def factory() -> StoringMethod:
        method = StoringMethod(events, fail_after_task_at=1 if not made else None)
        made.append(method)
        return method

    outcome = await runner(
        tmp_path, url, factory=factory, max_learning_attempts=2
    ).run()
    assert outcome.status == "complete" and outcome.attempt == 1
    paths = cell_paths(tmp_path)
    assert paths.attempt_numbers() == [1]
    assert [e for e in events if e[0] == "setup"] == [
        ("setup", "learning", 1, RUN_A1),
        ("setup", "learning", 1, RUN_A1, "c0001-played"),
    ]
    assert len(read_rows(paths.attempt(1).discarded_log)) == 1
    with pytest.raises(ValueError):
        run.RunOptions(out_dir=tmp_path, max_learning_attempts=0)


@pytest.mark.asyncio
async def test_new_attempt_is_for_the_first_learning_step_only(tmp_path, fake, gateway):
    """With --max-learning-attempts above 1, --new-attempt applies to the first step
    only: an INFRA interruption of the new attempt in the same invocation is resumed by
    a later step, never invalidated for a third attempt."""
    _gw, url = gateway
    events: list[Any] = []
    first = await runner(
        tmp_path, url, factory=lambda: StoringMethod(events, fail_after_task_at=1)
    ).run()
    assert first.status == "learning_interrupted" and first.attempt == 1
    made: list[StoringMethod] = []

    def factory() -> StoringMethod:
        method = StoringMethod(events, fail_after_task_at=1 if not made else None)
        made.append(method)
        return method

    outcome = await runner(
        tmp_path, url, factory=factory, max_learning_attempts=2, new_attempt=True
    ).run()
    assert outcome.status == "complete" and outcome.attempt == 2
    paths = cell_paths(tmp_path)
    assert paths.attempt_numbers() == [1, 2]
    assert len(read_rows(paths.attempt(2).discarded_log)) == 1  # resumed once


@pytest.mark.asyncio
async def test_max_learning_attempts_never_resumes_a_harness_failure(
    tmp_path, fake, gateway
):
    """A HARNESS learning failure (the harness's or MIRIX's own, which a
    resume repeats) is not resumed even with steps left. The attempt waits for a
    person: a plain run is refused; --force-resume resumes it from its checkpoint."""
    _gw, url = gateway
    events: list[Any] = []
    made: list[StoringMethod] = []
    faults = {"after_task": {"*"}}

    def factory() -> StoringMethod:
        method = StoringMethod(events, harness_once=faults)
        made.append(method)
        return method

    outcome = await runner(
        tmp_path, url, factory=factory, max_learning_attempts=3
    ).run()
    assert outcome.status == "learning_failed" and outcome.exit_code == 3
    assert outcome.attempt == 1 and outcome.failure_kind == "harness"
    assert outcome.to_dict()["failure_kind"] == "harness"
    assert "nothing resumes it automatically" in outcome.detail
    assert len(made) == 1
    paths = cell_paths(tmp_path)
    assert paths.attempt_numbers() == [1]
    state = read_json(paths.attempt(1).state)
    assert isinstance(state, dict)
    assert state["state"] == "learning_interrupted"
    assert state["needs_decision"] == "harness" and "abandoned" not in state
    assert (state["failure"]["kind"], state["failure"]["code"]) == (
        "harness",
        "unexpected_exception",
    )
    assert read_rows(paths.attempt(1).failures)[0]["recovery"] == "restart_cell"
    # status.json carries the kind, so the scheduler leaves the cell for a person.
    status = read_json(paths.status)
    assert (status["status"], status["failure_kind"]) == ("learning_failed", "harness")
    # A plain run is refused: a person decides.
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused" and "--force-resume" in refused.detail
    assert read_json(paths.attempt(1).state)["needs_decision"] == "harness"
    # --force-resume resumes it from the played checkpoint of item 0.
    events.clear()
    again = await runner(
        tmp_path, url, factory=storing(events), force_resume=True
    ).run()
    assert again.status == "complete" and again.attempt == 1, again.detail
    assert events[0] == ("setup", "learning", 1, RUN_A1, "c0000-played")
    state = read_json(paths.attempt(1).state)
    assert isinstance(state, dict)
    assert "needs_decision" not in state
    assert state["resumes"][0]["operator_resume"] is True
    assert read_rows(paths.attempt(1).discarded_log)[0]["operator_resume"] is True
    assert "failure_kind" not in (read_json(paths.status) or {})


def test_the_scheduler_leaves_a_harness_learning_failure_for_a_person(tmp_path):
    """``status.json`` ``failure_kind: harness`` keeps the cell out of the automatic
    re-runs (the scheduler's classification; skipped while schedule.py cannot be
    imported)."""
    schedule = pytest.importorskip("evals.skill_bench.schedule", exc_type=ImportError)
    paths = cell_paths(tmp_path)
    paths.root.mkdir(parents=True)
    write_json_atomic(
        paths.status,
        {
            "cell_id": SPEC.cell_id,
            "status": "learning_failed",
            "failure_kind": "harness",
        },
    )
    assert (
        schedule.classify_cell(paths, rerun_interrupted=True)[0] == schedule.ATTENTION
    )
    assert (
        schedule.classify_cell(
            paths, rerun_interrupted=True, rerun_harness_failures=True
        )[0]
        == schedule.QUEUED
    )


@pytest.mark.asyncio
async def test_new_attempt_invalidates_an_interrupted_learning_phase(
    tmp_path, fake, gateway
):
    """--new-attempt on an interrupted (or harness-stopped) learning phase invalidates it
    (operator_reroll), abandons its gateway run and learns again from scratch."""
    _gw, url = gateway
    order = train_order()
    failed = await runner(
        tmp_path,
        url,
        factory=storing([], harness_once={"after_task": {question(order[1])}}),
    ).run()
    assert failed.status == "learning_failed"
    events: list[Any] = []
    outcome = await runner(
        tmp_path, url, factory=storing(events), new_attempt=True
    ).run()
    assert outcome.status == "complete" and outcome.attempt == 2
    paths = cell_paths(tmp_path)
    first = read_json(paths.attempt(1).state)
    assert first["state"] == ATTEMPT_INVALID and first["abandoned"] is True
    assert first["invalidation"] == run.INVALIDATION_OPERATOR
    assert first["previous_state"] == "learning_interrupted"
    assert events[0] == ("setup", "learning", 2, RUN_A2)
    assert any(r.get("event") == "abandoned" for r in ledger(tmp_path, RUN_A1))


@pytest.mark.asyncio
async def test_a_harness_interrupted_while_learning_resumes(tmp_path, fake, gateway):
    """SIGTERM (a cancelled task) in the learning phase leaves the attempt resumable
    (learning_interrupted, never abandoned); a SIGKILL leaves it in state learning, which
    resumes the same way. Each resume discards what ran after its checkpoint."""
    _gw, url = gateway
    events: list[Any] = []
    reached = asyncio.Event()
    order = train_order()

    async def block_second_item(task: Any) -> None:
        if sum(1 for e in events if e[0] == "before_task") == 2:
            reached.set()
            await asyncio.sleep(3600)

    cell = asyncio.ensure_future(
        runner(
            tmp_path, url, factory=storing(events, on_before_task=block_second_item)
        ).run()
    )
    await asyncio.wait_for(reached.wait(), 20)
    cell.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cell
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert isinstance(state, dict)
    assert state["state"] == "learning_interrupted" and "abandoned" not in state
    assert state["interruption"]["reason"] == "interrupted"
    assert state["failure"] is None
    assert read_json(paths.status)["status"] == "interrupted"
    assert read_json(paths.attempt(1).latest_checkpoint)["id"] == "c0000-done"

    # A SIGKILL leaves no chance to write that: the state stays "learning".
    state.update(state=ATTEMPT_LEARNING)
    state.pop("interruption")
    write_json_atomic(paths.attempt(1).state, state)
    events.clear()
    outcome = await runner(tmp_path, url, factory=storing(events)).run()
    assert outcome.status == "complete" and outcome.attempt == 1, outcome.detail
    assert events[0] == ("setup", "learning", 1, RUN_A1, "c0000-done")
    learned = [e[2] for e in events if e[0] == "before_task" and e[1] == "learning"]
    assert learned == [question(r) for r in order[1:]]  # item 1 again, from the start
    segment = read_rows(paths.attempt(1).discarded_log)[0]
    assert segment["reason"] == "process_killed" and segment["from_state"] == "learning"
    assert segment["in_flight"]["position"] == 1
    assert segment["in_flight"]["stage"] == "rollout"
    assert segment["in_flight"]["trajectory_recorded"] is False
    # It died in before_task, before its first actor call: nothing to discard.
    assert segment["calls"] == []
    assert not any(r.get("event") == "abandoned" for r in ledger(tmp_path, RUN_A1))
    rows = read_rows(paths.attempt(1).learning)
    assert [r["position"] for r in rows] == list(range(len(order)))


# --- H8: test-phase failures are re-run per item ------------------------------------------------------


@pytest.mark.asyncio
async def test_test_phase_infra_failures_resume_per_item(tmp_path, fake, gateway):
    _gw, url = gateway
    events: list[Any] = []
    broken = {question("test:01"), question("test:03")}
    first = await runner(
        tmp_path, url, factory=storing(events, fail_test_questions=set(broken))
    ).run()
    assert first.status == "test_incomplete" and first.exit_code == 2
    assert first.pending == 2
    paths = cell_paths(tmp_path)
    attempt = paths.attempt(1)
    rows = read_rows(attempt.results)
    failed = [r for r in rows if r["score"] is None]
    assert sorted(r["raw_id"] for r in failed) == ["test:01", "test:03"]
    assert {r["recovery"] for r in failed} == {"rerun_item"}
    assert {r["failure"]["kind"] for r in failed} == {"infra"}  # never a model failure
    assert (attempt.trajectories("test") / "failed").is_dir()
    assert not paths.results.exists()  # no final results before the cell is complete
    assert read_json(attempt.state)["state"] == "testing"
    asked = actor_questions(fake)

    # A restarted harness: a new method instance resumes the frozen memory, checks it, and
    # runs only the pending items.
    events.clear()
    second = await runner(tmp_path, url, factory=storing(events)).run()
    assert second.status == "complete" and second.attempt == 1
    assert events[0] == ("setup", "test", 1, RUN_A1)
    assert sorted(e[2] for e in events if e[0] == "before_task") == sorted(broken)
    assert sorted(actor_questions(fake)[len(asked) :]) == sorted(broken)
    state = read_json(attempt.state)
    assert [c["kind"] for c in state["snapshot_checks"]] == [
        "after_test",
        "resume",
        "after_test",
    ]
    assert all(c["equal"] for c in state["snapshot_checks"])
    reruns = state["test_reruns"]
    assert sorted(r["raw_id"] for r in reruns) == ["test:01", "test:03"]
    assert all(r["previous_failure"]["code"] == "gateway_unreachable" for r in reruns)
    final = read_rows(paths.results)
    assert [r["raw_id"] for r in final] == expected_test_order()
    assert all(r["score"] is not None for r in final)
    assert read_json(paths.summary)["test"]["reruns"] == 2


@pytest.mark.asyncio
async def test_an_item_cut_off_by_a_harness_crash_is_re_run_and_listed(tmp_path):
    """An item whose actor call reached the ledger but that has no result row (the harness
    died mid-call) is re-run and listed as a re-run (H10.3); an item that never reached the
    gateway is simply run."""
    held = {question("test:02")}

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        content = (body.get("messages") or [{}])[-1].get("content")
        reply = tiny_responder(endpoint, body)
        if content in held and reply is not None:
            held.discard(content)
            reply.delay_s = 3600.0  # the call is in flight when the harness dies
        return reply

    fake = FakeUpstream(responder=responder)
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        cell = asyncio.ensure_future(
            runner(tmp_path, url, factory=storing([]), test_concurrency=1).run()
        )
        paths = cell_paths(tmp_path)

        def in_flight() -> bool:
            return (
                any(
                    r.get("event") == "call_started"
                    and r.get("phase") == "test"
                    and r.get("item")
                    and not any(
                        x.get("task_uid") == r.get("item")
                        for x in read_rows(paths.attempt(1).results)
                    )
                    for r in ledger(tmp_path, RUN_A1)
                )
                and not held
            )

        await eventually(in_flight, 30)
        cell.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cell
        state = read_json(paths.attempt(1).state)
        assert state["state"] == "testing"  # resumable
        # The interrupted launch compared its in-memory state with the frozen one.
        cut = [c for c in state["snapshot_checks"] if c.get("interrupted")]
        assert [(c["kind"], c["equal"]) for c in cut] == [("after_test", True)]
        done = {r["raw_id"] for r in read_rows(paths.attempt(1).results)}
        assert "test:02" not in done and len(done) == 2
        outcome = await runner(tmp_path, url, factory=storing([])).run()
    assert outcome.status == "complete"
    reruns = read_json(paths.attempt(1).state)["test_reruns"]
    assert [(r["raw_id"], r["previous_failure"]) for r in reruns] == [("test:02", None)]
    assert [r["raw_id"] for r in read_rows(paths.results)] == expected_test_order()


@pytest.mark.asyncio
async def test_a_memory_written_during_the_test_phase_is_a_violation(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    outcome = await runner(tmp_path, url, factory=storing([], write_in_test=True)).run()
    assert outcome.status == "violation" and outcome.exit_code == 4
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert state["state"] == ATTEMPT_VIOLATION
    assert state["snapshot_checks"][-1]["equal"] is False
    assert "H10.4" in state["violation"]["detail"]
    assert not paths.results.exists() and not paths.summary.exists()
    # A violated cell is not resumed. Its test items were scored, so a new attempt needs
    # a note (H1.3); the violated attempt then stays,
    # marked superseded.
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert again.status == "refused" and "--new-attempt" in again.detail
    assert "--note" in again.detail and "superseded" in again.detail
    fresh = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert fresh.status == "refused" and "already run its test phase" in fresh.detail
    assert "--note" in fresh.detail
    assert paths.attempt_numbers() == [1]
    assert read_json(paths.attempt(1).state)["state"] == ATTEMPT_VIOLATION
    replaced = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note="rerun after the fix"
    ).run()
    assert (replaced.status, replaced.attempt) == ("complete", 2), replaced.detail
    old = read_json(paths.attempt(1).state)
    assert (old["state"], old["invalidation"]) == (ATTEMPT_INVALID, "superseded")
    assert old["previous_state"] == ATTEMPT_VIOLATION
    assert old["superseded"]["scored_items"] > 0
    assert old["superseded"]["cell_records_sha256"] is None  # it published none
    assert old["superseded"]["by_attempt"] == 2
    assert old["violation"]["detail"]  # its violation stays on record


@pytest.mark.asyncio
async def test_a_new_attempt_before_any_test_item_scored_is_a_recorded_reroll(
    tmp_path, fake, gateway
):
    """--new-attempt may replace a frozen attempt only while no test item has a
    score; the old attempt keeps its state and is marked an operator re-roll."""
    _gw, url = gateway
    every_test = {question(r) for r in expected_test_order()}
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions=set(every_test))
    ).run()
    assert first.status == "test_incomplete" and first.pending == len(every_test)
    paths = cell_paths(tmp_path)
    fresh = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert fresh.status == "complete" and fresh.attempt == 2
    old = read_json(paths.attempt(1).state)
    assert old["state"] == ATTEMPT_INVALID
    assert old["previous_state"] == "testing"
    assert old["invalidation"] == run.INVALIDATION_OPERATOR
    manifest = read_json(paths.manifest)
    assert manifest["attempts"][0]["invalidation"] == run.INVALIDATION_OPERATOR
    assert manifest["attempts"][0]["previous_state"] == "testing"


@pytest.mark.asyncio
async def test_a_new_attempt_after_test_phase_actor_calls_is_refused(tmp_path):
    """An attempt whose test phase already sent actor calls is tested even if
    none of its items got a score (every call failed): --new-attempt would re-roll a cell
    whose test items the model has seen, so it is refused, as the audit's
    ``tested_attempt_replaced`` would flag it."""
    tests = {question(r) for r in expected_test_order()}
    revoked = {"key": True}

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        content = (body.get("messages") or [{}])[-1].get("content")
        if endpoint == "chat/completions" and content in tests and revoked["key"]:
            # The upstream key refused: INFRA, not retried by the gateway.
            return Reply(status=401, json={"error": {"message": "bad key"}})
        return tiny_responder(endpoint, body)

    fake = FakeUpstream(responder=responder)
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        first = await runner(tmp_path, url, factory=storing([])).run()
        assert first.status == "test_incomplete" and first.pending == len(tests)
        paths = cell_paths(tmp_path)
        assert not [r for r in read_rows(paths.attempt(1).results) if r.get("score")]
        sent = [
            r
            for r in ledger(tmp_path, RUN_A1)
            if r.get("event") == "call"
            and r.get("role") == "actor"
            and r.get("run_phase") == "test"
        ]
        assert sent  # the model saw the test items
        fresh = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
        assert fresh.status == "refused", fresh.detail
        assert "0 scored test item(s)" in fresh.detail and "--note" in fresh.detail
        assert f"{len(sent)} test-phase actor call(s)" in fresh.detail
        assert paths.attempt_numbers() == [1]
        assert read_json(paths.attempt(1).state)["state"] == "testing"
        # With a note it is superseded: kept, with what its test phase did.
        revoked["key"] = False
        note = "the upstream key was revoked mid-test; rerun"
        replaced = await runner(
            tmp_path, url, factory=storing([]), new_attempt=True, note=note
        ).run()
    assert (replaced.status, replaced.attempt) == ("complete", 2), replaced.detail
    old = read_json(paths.attempt(1).state)
    assert old["invalidation"] == run.INVALIDATION_SUPERSEDED
    assert old["previous_state"] == "testing" and old["abandoned"] is True
    assert old["superseded"]["test_actor_calls"] == len(sent)
    assert old["superseded"]["scored_items"] == 0
    assert old["superseded"]["note"] == note


@pytest.mark.asyncio
async def test_a_resume_that_restores_other_memory_is_a_violation(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:00")})
    ).run()
    assert first.status == "test_incomplete"
    second = await runner(
        tmp_path, url, factory=storing([], tamper_on_resume=True)
    ).run()
    assert second.status == "violation"
    state = read_json(cell_paths(tmp_path).attempt(1).state)
    assert [c["kind"] for c in state["snapshot_checks"]][-1] == "resume"
    assert "resumed" in state["violation"]["detail"]


@pytest.mark.asyncio
async def test_a_resume_that_cannot_find_the_memory_stays_incomplete(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:04")})
    ).run()
    assert first.pending == 1
    paths = cell_paths(tmp_path)
    (paths.attempt(1).method_dir / "memory.json").unlink()  # the method raises in setup
    second = await runner(tmp_path, url, factory=storing([])).run()
    assert second.status == "test_incomplete" and second.pending == 1
    assert "could not be resumed" in second.detail
    failures = read_rows(paths.attempt(1).failures)
    assert failures[-1]["phase"] == "test" and failures[-1]["item"] is None


@pytest.mark.asyncio
async def test_a_test_resume_under_another_commit_is_recorded(tmp_path, fake, gateway):
    """A test resume at another commit runs (no flag needed) and
    opens a version segment saying from which test count it ran and what changed.
    ``--allow-code-change`` is still accepted (older command lines parse)."""
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:02")})
    ).run()
    assert first.status == "test_incomplete"
    moved = {**FIXED_GIT, "commit": "decafbad"}
    resumed = await runner(tmp_path, url, factory=storing([]), git=moved).run()
    assert (resumed.status, resumed.attempt) == ("complete", 1), resumed.detail
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    first_seg, second = state["version_segments"]
    assert first_seg["index"] == 0 and first_seg["changed"] is None
    assert first_seg["why"] == "first attempt" and first_seg["git"] == FIXED_GIT
    assert first_seg["learning_from"] == {
        "position": 0,
        "stage": "setup",
        "checkpoint": None,
    }
    assert second["index"] == 1 and second["phase"] == "test"
    assert second["why"] == "test resume of attempt 1" and second["git"] == moved
    assert second["test_from"] == {"scored": TINY_SIZES["test"] - 1, "pending": 1}
    assert second["learning_from"] is None
    assert second["changed"]["commit"] is True and second["changed"]["code"] == []
    assert second["changed"]["protocol"] is False
    assert second["changed"]["protocol_diff"] is None
    assert second["changed"]["tree"] is False
    assert second["changed"]["code_files"] == {
        "harness": {"changed": [], "added": [], "removed": []}
    }
    assert second["version_file"] == "attempt-01/versions/01/version.json"
    assert state["versions_mixed"] == {"protocol": False, "code": False, "commit": True}
    assert state["git"] == FIXED_GIT  # the attempt keeps the commit it started at
    works = [r for r in read_rows(paths.launches) if r["event"] == "work"]
    assert [(w["segment"], w["git"]["commit"]) for w in works] == [
        (0, "c0ffee0"),
        (1, "decafbad"),
    ]
    versions = read_json(paths.summary)["versions"]
    assert versions["segments"] == 2 and versions["mixed"]["commit"] is True
    assert versions["protocol_fingerprints"] == [_tree_protocol()["fingerprint"]]
    summary = read_json(paths.manifest)["attempts"][0]
    assert summary["version_segments"] == 2
    assert summary["versions_mixed"] == state["versions_mixed"]
    allowed = await runner(
        tmp_path, url, factory=storing([]), git=moved, allow_code_change=True
    ).run()
    assert (allowed.status, allowed.detail) == ("complete", "already complete")


@pytest.mark.asyncio
async def test_the_same_version_adds_its_launch_to_the_segment(tmp_path, fake, gateway):
    """A resume under the same protocol, code, commit and tree adds its launch to the
    last segment; it opens no new one."""
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:02")})
    ).run()
    assert first.status == "test_incomplete"
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete"
    paths = cell_paths(tmp_path)
    (segment,) = read_json(paths.attempt(1).state)["version_segments"]
    works = [r for r in read_rows(paths.launches) if r["event"] == "work"]
    assert segment["launches"] == [w["launch"] for w in works]
    assert [w["segment"] for w in works] == [0, 0]
    assert sorted(p.name for p in (paths.attempt(1).root / "versions").iterdir()) == [
        "00"
    ]


# --- protocol versions (recorded per attempt, frozen automatically) ---------------------------


def _tree_protocol() -> dict[str, Any]:
    manifest = protocol.fingerprint_manifest()
    return {k: manifest[k] for k in ("fingerprint", "constants_sha256")}


def _change_protocol(
    monkeypatch: pytest.MonkeyPatch, resamples: int = 10_001
) -> dict[str, Any]:
    """Change a real protocol constant only the report reads (the fingerprint and the
    constants hash change consistently, the gateway's settings do not); returns this
    tree's new protocol. run.py freezes it automatically into the test's record copy."""
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", resamples)
    return _tree_protocol()


def _strip_protocol(paths: CellPaths, number: int) -> None:
    """Make attempt ``number`` look as an older run.py wrote it: no ``protocol`` or
    version segments in its attempt.json, no ``protocol``, segment, note or automatic
    freeze in its work lines (and no protocol_history in cell.json)."""
    state = read_json(paths.attempt(number).state)
    for key in ("protocol", "version_segments", "versions_mixed", "note", "supersedes"):
        state.pop(key, None)
    write_json_atomic(paths.attempt(number).state, state)
    shutil.rmtree(paths.attempt(number).root / run.VERSIONS_DIR, ignore_errors=True)
    rows = read_rows(paths.launches)
    for row in rows:
        for key in ("protocol", "segment", "note", "auto_freeze"):
            row.pop(key, None)
    paths.launches.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _work_protocols(paths: CellPaths) -> list[tuple[int, str, str]]:
    return [
        (r["attempt"], r["phase"], r["protocol"]["fingerprint"])
        for r in read_rows(paths.launches)
        if r["event"] == "work"
    ]


@pytest.mark.asyncio
async def test_another_cell_definition_is_refused_and_a_test_resume_records_a_protocol_change(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    """Versions are recorded per attempt, not enforced. cell.json
    still refuses every other difference of the cell definition, whatever the protocol. A
    test phase resumed under another protocol runs (whatever the options), freezes the
    tree first and records a version segment naming the constant that changed; the
    attempt keeps the protocol it started under, the cell is labelled a mixed version, and
    a complete cell stays as it is when the tree moves on."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    awm = CellSpec("alfworld", "luna", "awm")
    luna56 = CellSpec("alfworld", "luna56", "reasoningbank")
    cells: list[tuple[CellSpec, dict[str, Any]]] = [
        (SPEC, {}),
        (awm, {"allow_code_change": True}),
        (luna56, {"force_resume": True}),
    ]
    for spec, _options in cells:
        first = await runner(
            tmp_path,
            url,
            spec=spec,
            factory=storing(
                [], name=spec.method, fail_test_questions={question("test:00")}
            ),
        ).run()
        assert first.status == "test_incomplete", (spec.cell_id, first.detail)
    ran = _tree_protocol()
    assert read_json(paths.attempt(1).state)["protocol"] == ran
    assert _work_protocols(paths) == [(1, "learning", ran["fingerprint"])]
    before = read_json(paths.manifest)
    assert before["cell"]["protocol_fingerprint"] == ran["fingerprint"]
    (summary,) = before["attempts"]
    assert summary["protocol"] == {**ran, "source": "attempt"}
    assert summary["code"] == read_json(paths.attempt(1).state)["code"]
    other = TinyEnv({"train": 6, "val": 4, "test": 6})  # the item list changed
    outcome = await runner(tmp_path, url, env=other, factory=storing([])).run()
    assert outcome.status == "refused" and "test_items" in outcome.detail
    assert "describes another cell" in outcome.detail

    changed = _change_protocol(monkeypatch)
    assert changed["fingerprint"] != ran["fingerprint"]
    # Another definition under another protocol: the definition is what refuses it.
    outcome = await runner(tmp_path, url, env=other, factory=storing([])).run()
    assert outcome.status == "refused" and "test_items" in outcome.detail
    assert "protocol_fingerprint" not in outcome.detail
    # The attempt has scored test items: a new attempt needs a note.
    tested = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert tested.status == "refused" and "--note" in tested.detail
    assert "already run its test phase" in tested.detail
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == ran["fingerprint"]  # nothing ran: nothing frozen

    for spec, options in cells:
        done = await runner(
            tmp_path, url, spec=spec, factory=storing([], name=spec.method), **options
        ).run()
        assert (done.status, done.attempt) == ("complete", 1), (spec, done.detail)
        cell = cell_paths(tmp_path, spec)
        state = read_json(cell.attempt(1).state)
        start, resumed = state["version_segments"]
        assert start["protocol"] == ran and resumed["protocol"] == changed
        assert resumed["why"] == "test resume of attempt 1"
        assert resumed["changed"]["protocol"] is True
        diff = resumed["changed"]["protocol_diff"]
        assert (
            diff["from"] == ran["fingerprint"] and diff["to"] == changed["fingerprint"]
        )
        assert diff["constants_changed_keys"] == ["bootstrap_resamples"]
        assert diff["changed"] == diff["added"] == diff["removed"] == []
        assert resumed["known_version"] is True
        assert state["versions_mixed"]["protocol"] is True
        # The attempt keeps the protocol it started under; nothing is adopted.
        assert state["protocol"] == ran
        manifest = read_json(cell.manifest)
        assert manifest["cell"]["protocol_fingerprint"] == ran["fingerprint"]
        assert "protocol_history" not in manifest
        versions = read_json(cell.summary)["versions"]
        assert versions["mixed"]["protocol"] is True
        assert versions["protocol_fingerprints"] == [
            ran["fingerprint"],
            changed["fingerprint"],
        ]
        assert read_json(cell.summary)["protocol_fingerprint"] == ran["fingerprint"]
    # The first launch under the changed tree froze it, the others found it frozen.
    statuses = [
        read_json(cell_paths(tmp_path, spec).attempt(1).state)["version_segments"][1][
            "auto_freeze"
        ]["status"]
        for spec, _options in cells
    ]
    assert statuses == ["frozen", "already_frozen", "already_frozen"]
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == changed["fingerprint"]
    assert record["auto_freeze"]["cell_id"] == SPEC.cell_id
    assert record["auto_freeze"]["why"] == "test resume of attempt 1"
    assert record["auto_freeze"]["constants_changed_keys"] == ["bootstrap_resamples"]
    assert record["history"][-1]["fingerprint"] == ran["fingerprint"]
    assert _work_protocols(paths) == [
        (1, "learning", ran["fingerprint"]),
        (1, "test", changed["fingerprint"]),
    ]
    # A complete cell stays under its version after the tree moved on.
    _change_protocol(monkeypatch, 10_002)
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert (again.status, again.detail) == ("complete", "already complete")
    kept = read_json(paths.manifest)
    assert kept["cell"]["protocol_fingerprint"] == ran["fingerprint"]
    assert "protocol_history" not in kept
    assert read_json(paths.summary)["protocol_fingerprint"] == ran["fingerprint"]
    # Nothing ran: nothing was frozen.
    assert (
        protocol.load_frozen_protocol(_frozen_record)["fingerprint"]
        == (changed["fingerprint"])
    )


@pytest.mark.asyncio
async def test_a_learning_resume_under_another_protocol_continues_and_a_new_attempt_adopts_it(
    tmp_path, fake, gateway, monkeypatch
):
    """A learning phase resumes under another protocol (from its
    checkpoint, not from item 0), recorded as a version segment with the position it
    resumed from; --new-attempt starts a new attempt that adopts it. The replaced values
    go to cell.json protocol_history with the attempts that ran under them, on disk
    before the attempt's number is reserved, so an invocation stopped right after the
    reservation leaves them recorded once."""
    _gw, url = gateway
    awm = CellSpec("alfworld", "luna", "awm")
    other = cell_paths(tmp_path, awm)
    stopped = await runner(
        tmp_path, url, spec=awm, factory=storing([], name="awm", fail_after_task_at=0)
    ).run()
    assert stopped.status == "learning_interrupted"
    paths = cell_paths(tmp_path)
    failed = await runner(
        tmp_path, url, factory=storing([], fail_after_task_at=0)
    ).run()
    assert failed.status == "learning_interrupted"
    ran = _tree_protocol()
    assert read_json(paths.attempt(1).state)["protocol"] == ran

    changed = _change_protocol(monkeypatch)
    resumed = await runner(
        tmp_path, url, spec=awm, factory=storing([], name="awm")
    ).run()
    assert (resumed.status, resumed.attempt) == ("complete", 1), resumed.detail
    state = read_json(other.attempt(1).state)
    assert len(state["resumes"]) == 1
    start, later = state["version_segments"]
    assert start["protocol"] == ran and later["protocol"] == changed
    assert (
        later["phase"] == "learning" and later["why"] == "learning resume of attempt 1"
    )
    assert later["learning_from"] == {
        "position": 0,
        "stage": "after_task",
        "checkpoint": "c0000-played",
    }
    assert later["changed"]["protocol"] is True
    assert later["changed"]["protocol_diff"]["constants_changed_keys"] == [
        "bootstrap_resamples"
    ]
    assert later["auto_freeze"]["status"] == "frozen"
    assert state["protocol"] == ran  # the protocol it started under

    # --new-attempt: attempt 1 is invalidated and the new attempt adopts the protocol.
    # The gateway refuses the new run: the adoption is on disk all the same.
    real_set_phase = run.GatewayLink.set_phase

    async def refused(self: run.GatewayLink, run_id: str, phase: str) -> None:
        if run_id == RUN_A2:
            raise httpx.ConnectError("the gateway is restarting")
        await real_set_phase(self, run_id, phase)

    monkeypatch.setattr(run.GatewayLink, "set_phase", refused)
    halted = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert halted.status == "infra_unavailable"
    assert read_json(paths.attempt(1).state)["state"] == ATTEMPT_INVALID
    manifest = read_json(paths.manifest)
    assert {k: manifest["cell"][k] for k in run.PROTOCOL_KEYS} == {
        "protocol_fingerprint": changed["fingerprint"],
        "protocol_constants_sha256": changed["constants_sha256"],
    }
    (entry,) = manifest["protocol_history"]
    assert {k: entry[k] for k in ("fingerprint", "constants_sha256", "attempts")} == {
        **ran,
        "attempts": [1],
    }
    assert entry["replaced_by"] == read_rows(paths.launches)[-1]["launch"]
    assert read_json(paths.attempt(2).reservation)["run_id"] == RUN_A2
    monkeypatch.setattr(run.GatewayLink, "set_phase", real_set_phase)

    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert (done.status, done.attempt) == ("complete", 3), done.detail
    manifest = read_json(paths.manifest)
    assert len(manifest["protocol_history"]) == 1  # adopted once
    third = read_json(paths.attempt(3).state)
    assert third["protocol"] == changed
    (segment,) = third["version_segments"]
    assert segment["why"] == "after invalid attempt 1"
    assert segment["auto_freeze"]["status"] == "already_frozen"
    assert third["supersedes"] is None and third["note"] is None
    assert _work_protocols(paths) == [
        (1, "learning", ran["fingerprint"]),
        (3, "learning", changed["fingerprint"]),
    ]
    summaries = {a["attempt"]: a for a in manifest["attempts"]}
    assert summaries[1]["protocol"] == {**ran, "source": "attempt"}
    assert summaries[3]["protocol"] == {**changed, "source": "attempt"}
    assert summaries[3]["code"] == third["code"]
    summary = read_json(paths.summary)
    assert summary["protocol_fingerprint"] == changed["fingerprint"]
    assert summary["code"] == third["code"]


@pytest.mark.asyncio
async def test_an_attempt_without_recorded_versions_resumes_with_its_legacy_segment(
    tmp_path, fake, gateway, monkeypatch
):
    """An attempt an older run.py wrote, before attempts recorded their protocol and
    versions: no ``protocol`` or ``version_segments`` in attempt.json, none in the work
    lines, no protocol_history. Such an attempt resumes under another protocol: its
    earlier launches become a legacy segment (the manifest's protocol), the resume a new
    one; a new attempt lists it in the history."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    ran = _tree_protocol()
    failed = await runner(
        tmp_path, url, factory=storing([], fail_after_task_at=0)
    ).run()
    assert failed.status == "learning_interrupted"
    _strip_protocol(paths, 1)
    first_launch = read_rows(paths.launches)[0]["launch"]

    changed = _change_protocol(monkeypatch)
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    state = read_json(paths.attempt(1).state)
    assert "protocol" not in state and len(state["resumes"]) == 1
    legacy, resumed = state["version_segments"]
    assert legacy["legacy"] is True and legacy["why"] == "legacy"
    assert legacy["protocol"]["fingerprint"] == ran["fingerprint"]
    assert legacy["launches"] == [first_launch] and legacy["version_file"] is None
    assert resumed["legacy"] is False and resumed["index"] == 1
    assert resumed["changed"]["protocol"] is True
    assert resumed["changed"]["protocol_diff"]["constants_changed_keys"] == [
        "bootstrap_resamples"
    ]
    assert resumed["changed"]["code_files"] is None  # no maps were recorded before
    assert state["versions_mixed"]["protocol"] is True
    assert [a["protocol"] for a in read_json(paths.manifest)["attempts"]] == [
        {**ran, "source": "cell"}
    ]
    assert read_json(paths.summary)["versions"]["protocol_fingerprints"] == [
        ran["fingerprint"],
        changed["fingerprint"],
    ]

    # A legacy learning attempt replaced by a new one under another protocol.
    spec = CellSpec("alfworld", "luna", "awm")
    other = cell_paths(tmp_path, spec)
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", 10_000)
    failed = await runner(
        tmp_path,
        url,
        spec=spec,
        factory=storing([], name="awm", fail_after_task_at=0),
    ).run()
    assert failed.status == "learning_interrupted"
    _strip_protocol(other, 1)
    _change_protocol(monkeypatch)
    done = await runner(
        tmp_path, url, spec=spec, factory=storing([], name="awm"), new_attempt=True
    ).run()
    assert (done.status, done.attempt) == ("complete", 2), done.detail
    manifest = read_json(other.manifest)
    (entry,) = manifest["protocol_history"]
    assert (entry["fingerprint"], entry["attempts"]) == (ran["fingerprint"], [1])
    assert [a["protocol"] for a in manifest["attempts"]] == [
        {**ran, "source": "history"},
        {**changed, "source": "attempt"},
    ]
    assert manifest["cell"]["protocol_fingerprint"] == changed["fingerprint"]


@pytest.mark.asyncio
async def test_a_new_attempt_from_an_unfrozen_tree_freezes_it_once(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    """A new attempt under a tree that is not the frozen value freezes it once, under
    the record's lock, with a reason naming the cell, the attempt and the commit, and
    runs; the next cell finds it frozen."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    failed = await runner(
        tmp_path, url, factory=storing([], fail_after_task_at=0)
    ).run()
    assert failed.status == "learning_interrupted"
    old = protocol.load_frozen_protocol(_frozen_record)
    changed = _change_protocol(monkeypatch)
    done = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert (done.status, done.attempt) == ("complete", 2), done.detail
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == changed["fingerprint"]
    assert "auto-freeze" in record["reason"]
    for needle in (
        SPEC.cell_id,
        "attempt 02",
        "c0ffee0",
        "constants [bootstrap_resamples]",
    ):
        assert needle in record["reason"], needle
    assert record["history"][-1]["fingerprint"] == old["fingerprint"]
    assert record["auto_freeze"]["previous_fingerprint"] == old["fingerprint"]
    assert record["auto_freeze"]["cell_dir"] == str(paths.root)
    (segment,) = read_json(paths.attempt(2).state)["version_segments"]
    assert segment["auto_freeze"]["status"] == "frozen"
    assert segment["why"] == "--new-attempt replacing interrupted attempt 1"
    assert segment["known_version"] is True
    assert read_json(paths.attempt(1).state)["invalidation"] == "operator_reroll"
    work = [r for r in read_rows(paths.launches) if r["event"] == "work"][-1]
    assert (work["attempt"], work["auto_freeze"], work["segment"]) == (2, "frozen", 0)
    version = read_json(paths.root / segment["version_file"])
    assert version["fingerprint"] == changed["fingerprint"]
    assert version["frozen"] == {
        "known": True,
        "current": True,
        "frozen_at": record["frozen_at"],
        "reason": record["reason"],
    }
    assert protocol.recorded_version(version)["fingerprint"] == changed["fingerprint"]
    # A second cell run afterwards finds it frozen.
    awm = CellSpec("alfworld", "luna", "awm")
    other = await runner(tmp_path, url, spec=awm, factory=storing([], name="awm")).run()
    assert other.status == "complete"
    (segment,) = read_json(cell_paths(tmp_path, awm).attempt(1).state)[
        "version_segments"
    ]
    assert segment["auto_freeze"]["status"] == "already_frozen"
    assert len(protocol.load_frozen_protocol(_frozen_record)["history"]) == len(
        record["history"]
    )


@pytest.mark.asyncio
async def test_an_auto_freeze_into_a_tracked_record_is_logged_and_recorded(
    tmp_path, fake, gateway, monkeypatch, _frozen_record, caplog
):
    """An automatic freeze into the git-tracked record leaves it changed on
    this host (a later ``git pull`` that brings another record stops there). run.py
    logs what to do and the version segment records it; the attempt runs on."""
    from evals.skill_bench.tools import freeze_protocol

    _gw, url = gateway
    seen: list[Path] = []

    def tracked(path: Path) -> bool:
        seen.append(path)
        return True

    monkeypatch.setattr(freeze_protocol, "record_tracked_by_git", tracked)
    _change_protocol(monkeypatch)
    with caplog.at_level("WARNING", logger=run.log.name):
        done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete", done.detail
    assert seen == [_frozen_record]
    (segment,) = read_json(cell_paths(tmp_path).attempt(1).state)["version_segments"]
    freeze = segment["auto_freeze"]
    assert (freeze["status"], freeze["record_tracked"]) == ("frozen", True)
    assert "git pull" in freeze["note"] and freeze["commit"] == "c0ffee0"
    assert freeze["note"] in caplog.text


@pytest.mark.asyncio
async def test_a_revert_freezes_back_to_the_earlier_version(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    _gw, url = gateway
    ran = _tree_protocol()
    changed = _change_protocol(monkeypatch)
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", 10_000)  # the revert
    awm = CellSpec("alfworld", "luna", "awm")
    done = await runner(tmp_path, url, spec=awm, factory=storing([], name="awm")).run()
    assert done.status == "complete"
    (segment,) = read_json(cell_paths(tmp_path, awm).attempt(1).state)[
        "version_segments"
    ]
    freeze = segment["auto_freeze"]
    assert freeze["status"] == "frozen" and freeze["revert_to_known"] is True
    assert "back to the earlier version frozen at" in freeze["reason"]
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == ran["fingerprint"]
    assert [e["fingerprint"] for e in record["history"]][-2:] == [
        ran["fingerprint"],
        changed["fingerprint"],
    ]


@pytest.mark.asyncio
async def test_two_cells_starting_at_once_freeze_once(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    _gw, url = gateway
    history = len(protocol.load_frozen_protocol(_frozen_record)["history"])
    changed = _change_protocol(monkeypatch)
    awm = CellSpec("alfworld", "luna", "awm")
    outcomes = await asyncio.gather(
        runner(tmp_path, url, factory=storing([])).run(),
        runner(tmp_path, url, spec=awm, factory=storing([], name="awm")).run(),
    )
    assert [o.status for o in outcomes] == ["complete", "complete"]
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == changed["fingerprint"]
    assert len(record["history"]) == history + 1
    statuses = sorted(
        read_json(cell_paths(tmp_path, spec).attempt(1).state)["version_segments"][0][
            "auto_freeze"
        ]["status"]
        for spec in (SPEC, awm)
    )
    assert statuses == ["already_frozen", "frozen"]


@pytest.mark.asyncio
async def test_a_held_record_lock_is_recorded_and_the_attempt_runs(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    """A freeze that cannot get the record's lock in time is recorded (the work line and
    the version segment say so) and the attempt runs on (recorded, not blocked)."""
    from evals.skill_bench.tools import freeze_protocol

    _gw, url = gateway
    monkeypatch.setattr(freeze_protocol, "FREEZE_LOCK_WAIT_S", 0.2)
    changed = _change_protocol(monkeypatch)
    before = _frozen_record.read_bytes()
    with freeze_protocol.record_lock(_frozen_record):
        done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete", done.detail
    paths = cell_paths(tmp_path)
    (segment,) = read_json(paths.attempt(1).state)["version_segments"]
    assert segment["auto_freeze"]["status"] == "error"
    assert "lock" in segment["auto_freeze"]["error"]
    assert segment["known_version"] is False  # the record does not know it (yet)
    assert segment["protocol"] == changed
    work = [r for r in read_rows(paths.launches) if r["event"] == "work"][-1]
    assert work["auto_freeze"] == "error" and "lock" in work["auto_freeze_error"]
    assert _frozen_record.read_bytes() == before
    # The version itself is recorded with the attempt: its version.json vouches for it.
    version = read_json(paths.root / segment["version_file"])
    assert version["frozen"]["known"] is False
    assert protocol.recorded_version(version)["fingerprint"] == changed["fingerprint"]


@pytest.mark.asyncio
async def test_a_missing_frozen_record_is_recorded_and_never_recreated(
    tmp_path, fake, gateway, monkeypatch
):
    _gw, url = gateway
    missing = tmp_path / "nowhere" / "protocol_frozen.json"
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, str(missing))
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete", done.detail
    (segment,) = read_json(cell_paths(tmp_path).attempt(1).state)["version_segments"]
    assert segment["auto_freeze"]["status"] == "error"
    assert "never recreated" in segment["auto_freeze"]["error"]
    assert segment["known_version"] is None  # the record could not be read
    assert not missing.exists()


@pytest.mark.asyncio
async def test_successive_refreezes_and_a_revert_keep_every_attempts_protocol(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    """Adoptions across versions A -> B -> C and a revert to A, each frozen automatically
    by the new attempt that first ran it. Each value is adopted by the next new attempt
    only; ``protocol_history`` lists each replaced value once, with the attempts that ran
    under them and no other, and every attempt (its attempt.json, its work lines, the
    manifest's summaries, and ``protocol.attempt_protocol`` from the history alone) keeps
    its own version."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    a = _tree_protocol()
    b = _change_protocol(monkeypatch, 10_001)
    c = _change_protocol(monkeypatch, 10_002)
    for number, resamples in enumerate((10_000, 10_001, 10_002), start=1):
        monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", resamples)
        stopped = await runner(
            tmp_path,
            url,
            factory=storing([], fail_after_task_at=0),
            new_attempt=number > 1,
        ).run()
        assert (stopped.status, stopped.attempt) == ("learning_interrupted", number), (
            stopped.detail
        )
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", 10_000)  # the revert to A
    done = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert (done.status, done.attempt) == ("complete", 4), done.detail

    ran = [a, b, c, a]
    manifest = read_json(paths.manifest)
    assert isinstance(manifest, dict)
    assert [
        (e["fingerprint"], e["constants_sha256"], e["attempts"])
        for e in manifest["protocol_history"]
    ] == [
        (a["fingerprint"], a["constants_sha256"], [1]),
        (b["fingerprint"], b["constants_sha256"], [2]),
        (c["fingerprint"], c["constants_sha256"], [3]),
    ]
    assert manifest["cell"]["protocol_fingerprint"] == a["fingerprint"]
    assert [s["protocol"] for s in manifest["attempts"]] == [
        {**value, "source": "attempt"} for value in ran
    ]
    for number, value in enumerate(ran, start=1):
        assert read_json(paths.attempt(number).state)["protocol"] == value
    assert _work_protocols(paths) == [
        (n, "learning", v["fingerprint"]) for n, v in enumerate(ran, start=1)
    ]
    statuses = [
        read_json(paths.attempt(n).state)["version_segments"][0]["auto_freeze"][
            "status"
        ]
        for n in (1, 2, 3, 4)
    ]
    assert statuses == ["already_frozen", "frozen", "frozen", "frozen"]
    record = protocol.load_frozen_protocol(_frozen_record)
    assert record["fingerprint"] == a["fingerprint"]
    assert record["auto_freeze"]["revert_to_known"] is True
    assert [e["fingerprint"] for e in record["history"]][-3:] == [
        a["fingerprint"],
        b["fingerprint"],
        c["fingerprint"],
    ]
    # An attempt that recorded no protocol (as an older run.py wrote them) is attributed
    # by the history, whichever re-freeze replaced its value.
    for number, value in enumerate(ran[:3], start=1):
        legacy = {**read_json(paths.attempt(number).state)}
        legacy.pop("protocol")
        found = protocol.attempt_protocol(manifest, legacy, number)
        assert (found["fingerprint"], found["source"]) == (
            value["fingerprint"],
            "history",
        )
    assert read_json(paths.summary)["protocol_fingerprint"] == a["fingerprint"]


# --- refusals before any attempt ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_nonconformant_gateway_is_refused(tmp_path, fake):
    with serve_gateway(tmp_path, fake) as (
        _gw,
        url,
    ):  # test back-off: not the protocol's
        outcome = await runner(tmp_path, url, factory=storing([])).run()
        assert outcome.status == "refused" and "protocol_conformant" in outcome.detail
        assert cell_paths(tmp_path).attempt_numbers() == []
        allowed = await runner(
            tmp_path, url, factory=storing([]), allow_nonconformant_gateway=True
        ).run()
        assert allowed.status == "complete"
        manifest = read_json(cell_paths(tmp_path).manifest)
        assert manifest["gateway"]["protocol_conformant"] is False


@pytest.mark.asyncio
async def test_a_live_gateway_of_other_model_settings_is_refused_before_any_call(
    tmp_path, fake, gateway, monkeypatch, _frozen_record
):
    """A live gateway started under other model settings than this tree's
    protocol gives (the scheduler keeps and adopts running gateways) conforms to its own
    import-time settings, so ``protocol_conformant`` alone passes it. The audit then fails
    every attempt it served (H10.6:gateway_start_settings, a validity check), so run.py
    refuses before the first call, freezes nothing and names the fix; a difference in a
    recorded-only setting (deadline, retries, back-off, margin) runs and is recorded."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    before = _frozen_record.read_bytes()
    served = protocol.REASONING_EFFORT
    monkeypatch.setattr(protocol, "REASONING_EFFORT", "high")
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused", refused.detail
    for needle in ("reasoning_effort", repr(served), "'high'", "restart", "H10.6"):
        assert needle in refused.detail, needle
    assert paths.attempt_numbers() == []
    assert fake.calls("chat/completions") == []
    assert _frozen_record.read_bytes() == before  # nothing auto-froze
    # --allow-nonconformant-gateway (tests and debugging only) runs it, recorded.
    allowed = await runner(
        tmp_path, url, factory=storing([]), allow_nonconformant_gateway=True
    ).run()
    assert allowed.status == "complete", allowed.detail
    gw = read_json(paths.manifest)["gateway"]
    assert gw["protocol_conformant"] is False
    assert gw["tree_settings_differ"] == ["reasoning_effort"]
    monkeypatch.setattr(protocol, "REASONING_EFFORT", served)
    # A recorded-only setting that differs runs, and is recorded as differing.
    monkeypatch.setattr(
        protocol, "GATEWAY_MAX_RETRIES", protocol.GATEWAY_MAX_RETRIES + 1
    )
    other = CellSpec("alfworld", "luna", "awm")
    done = await runner(
        tmp_path, url, spec=other, factory=storing([], name="awm")
    ).run()
    assert done.status == "complete", done.detail
    gw = read_json(cell_paths(tmp_path, other).manifest)["gateway"]
    assert (gw["protocol_conformant"], gw["tree_settings_differ"]) == (
        True,
        ["max_retries"],
    )


@pytest.mark.asyncio
async def test_a_paused_gateway_starts_nothing(tmp_path):
    fake = FakeUpstream(
        responder=tiny_responder, limit_remaining=1.0
    )  # below the floor
    with conformant_gateway(tmp_path, fake) as (gw, url):
        await asyncio.to_thread(
            lambda: httpx.get(url + "/balance", timeout=10, trust_env=False)
        )
        outcome = await runner(tmp_path, url, factory=storing([])).run()
    assert outcome.status == "budget_paused" and outcome.exit_code == 6
    assert cell_paths(tmp_path).attempt_numbers() == []
    assert fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_an_unlinked_ledger_is_refused(tmp_path, fake, gateway):
    _gw, url = gateway
    spec = SPEC
    cell = run.CellRunner(
        spec,
        run.RunOptions(
            out_dir=tmp_path / "runs" / spec.cell_id,
            gateway_url=url,
            gateway_runs_dir=tmp_path / "elsewhere",
            preflight=False,
        ),
        env_factory=lambda _b: TinyEnv(),
        method_factory=storing([]),
        git=lambda: dict(FIXED_GIT),
    )
    outcome = await cell.run()
    assert outcome.status == "refused" and "--gateway-runs-dir" in outcome.detail
    assert cell_paths(tmp_path).attempt_numbers() == []


@pytest.mark.asyncio
async def test_placeholder_and_misnamed_methods_are_refused(
    tmp_path, fake, gateway, monkeypatch
):
    _gw, url = gateway
    monkeypatch.setattr(run, "method_class", lambda name: None)
    outcome = await runner(tmp_path, url, factory=None).run()
    assert outcome.status == "refused" and "placeholder" in outcome.detail
    assert fake.calls("chat/completions") == []
    misnamed = await runner(
        tmp_path, url, factory=storing([], name="awm"), out_dir=tmp_path / "b"
    ).run()
    assert misnamed.status == "learning_failed"
    assert "named 'awm'" in misnamed.detail


@pytest.mark.asyncio
async def test_a_busy_cell_is_left_alone(tmp_path, fake, gateway):
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    paths.root.mkdir(parents=True)
    write_json_atomic(paths.status, {"status": "running", "pid": 1})
    lock = OwnerLock.acquire(paths.lock)
    try:
        outcome = await runner(tmp_path, url, factory=storing([])).run()
    finally:
        lock.release()
    assert outcome.status == "busy" and outcome.exit_code == 5
    assert read_json(paths.status) == {"status": "running", "pid": 1}


@pytest.mark.asyncio
async def test_preflight_problems_refuse_the_cell(tmp_path, fake, gateway):
    _gw, url = gateway

    class Checked(TinyEnv):
        def verify_gamefiles(self, split: str) -> list[str]:
            self.preflights.append(split)
            return [f"{split}/game.tw-pddl differs"] if split == "test" else []

    env = Checked()
    outcome = await runner(
        tmp_path, url, env=env, factory=storing([]), preflight=True
    ).run()
    assert outcome.status == "refused" and "game.tw-pddl differs" in outcome.detail
    assert env.preflights == ["train", "test"]
    assert fake.calls("chat/completions") == []

    class Broken(TinyEnv):
        async def preflight(self) -> dict[str, Any]:
            raise RuntimeError("sandbox image missing")

    outcome = await runner(
        tmp_path,
        url,
        env=Broken(),
        factory=storing([]),
        preflight=True,
        out_dir=tmp_path / "c",
    ).run()
    assert outcome.status == "refused" and "sandbox image missing" in outcome.detail

    class Fine(TinyEnv):
        async def preflight(self) -> dict[str, Any]:
            return {"image": "sha256:abc"}

    outcome = await runner(
        tmp_path,
        url,
        env=Fine(),
        factory=storing([]),
        preflight=True,
        out_dir=tmp_path / "d",
    ).run()
    assert outcome.status == "complete"
    manifest = read_json(CellPaths(tmp_path / "d").manifest)
    assert manifest["preflight"]["env"] == {"image": "sha256:abc"}


@pytest.mark.asyncio
async def test_a_webshop_server_down_at_cell_start_is_infra_unavailable(
    tmp_path, fake, gateway
):
    """The WebShop server unreachable or still loading when a cell starts
    (a restart between the scheduler's readiness reading and the cell) is transient:
    ``infra_unavailable``, which ``--auto-resume`` runs again, never ``refused``, which
    waits for a person. A server that failed its own checks still refuses the cell."""
    from evals.skill_bench.envs.webshop import WebShopEnvError, WebShopServerError

    _gw, url = gateway

    class Down(TinyEnv):
        async def preflight(self) -> dict[str, Any]:
            raise WebShopServerError("GET /ready: ConnectError: connection refused")

    outcome = await runner(
        tmp_path, url, env=Down(), factory=storing([]), preflight=True
    ).run()
    assert outcome.status == "infra_unavailable" and outcome.exit_code == 7
    assert "connection refused" in outcome.detail
    assert fake.calls("chat/completions") == []

    class Failed(TinyEnv):
        async def preflight(self) -> dict[str, Any]:
            raise WebShopEnvError("the WebShop server is 'failed': price table")

    outcome = await runner(
        tmp_path,
        url,
        env=Failed(),
        factory=storing([]),
        preflight=True,
        out_dir=tmp_path / "e",
    ).run()
    assert outcome.status == "refused" and "price table" in outcome.detail
    with pytest.raises(run._InfraUnavailable):
        await run.preflight_env(Down(), "webshop", ["test"])
    with pytest.raises(run._Refused):
        await run.preflight_env(Failed(), "webshop", ["test"])


@pytest.mark.asyncio
async def test_a_lost_salt_or_a_corrupt_frozen_record_refuses_the_resume(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:01")})
    ).run()
    assert first.status == "test_incomplete"
    paths = cell_paths(tmp_path)
    salt = paths.salt.read_bytes()
    paths.salt.unlink()
    lost = await runner(tmp_path, url, factory=storing([])).run()
    assert lost.status == "refused" and "uids" in lost.detail
    paths.salt.write_bytes(salt)
    frozen = read_json(paths.attempt(1).frozen_snapshot)
    frozen["snapshot"]["memory"].append("edited by hand")
    write_json_atomic(paths.attempt(1).frozen_snapshot, frozen)
    corrupt = await runner(tmp_path, url, factory=storing([])).run()
    assert corrupt.status == "refused" and "corrupt" in corrupt.detail
    assert read_json(paths.attempt(1).state)["state"] == "testing"


class FakeConnection:
    def __init__(self, template: bool = True) -> None:
        self.template = template
        self.closed = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        assert "pg_database" in query and args == ("skill_bench_template",)
        return 1 if self.template else None

    async def close(self) -> None:
        self.closed = True


def contract(values: dict[str, Any]):
    async def probe(env: Any) -> dict[str, Any]:
        assert env["MIRIX_LLM_MAX_RETRIES"] == "0" and "OPENROUTER_API_KEY" not in env
        return values

    return probe


HONOURED = {
    "llm_max_retries": 0,
    "llm_timeout_seconds": 630,
    "embedding_timeout_seconds": 630,
    "SESSION_IDLE_TIMEOUT_SECONDS": protocol.MIRIX_SESSION_IDLE_TIMEOUT_S,
}


@pytest.mark.asyncio
async def test_mirix_preflight_refuses_before_an_attempt_is_used(tmp_path, monkeypatch):
    async def connect_ok(**kwargs: Any) -> FakeConnection:
        assert kwargs["database"] == "postgres"
        return FakeConnection()

    async def unreachable(**kwargs: Any) -> Any:
        raise OSError("connection refused")

    async def no_template(**kwargs: Any) -> FakeConnection:
        return FakeConnection(template=False)

    info = await run.mirix_preflight(
        "c", tmp_path, connect=connect_ok, probe=contract(HONOURED)
    )
    assert (
        info["llm_contract"] == HONOURED and info["template"] == "skill_bench_template"
    )
    # An unreachable Postgres is infrastructure that is down: resumable, exit 7.
    with pytest.raises(run._InfraUnavailable, match="compose Postgres"):
        await run.mirix_preflight(
            "c", tmp_path, connect=unreachable, probe=contract(HONOURED)
        )
    for connect, probe, message in (
        (no_template, contract(HONOURED), "template database"),
        (
            connect_ok,
            contract({**HONOURED, "llm_max_retries": {"missing": True}}),
            "retry/timeout contract",
        ),
        (connect_ok, contract({**HONOURED, "llm_timeout_seconds": 60}), "630"),
        (
            connect_ok,
            contract({**HONOURED, "embedding_timeout_seconds": 60}),
            "MIRIX_EMBEDDING_TIMEOUT_SECONDS=630",
        ),
    ):
        with pytest.raises(run._Refused, match=message):
            await run.mirix_preflight("c", tmp_path, connect=connect, probe=probe)
    # MirixClient would authenticate as MIRIX_API_KEY's client, so every
    # MIRIX cell would fail at setup; the preflight says so first.
    monkeypatch.setenv("MIRIX_API_KEY", "a-key-from-other-mirix-work")
    with pytest.raises(run._Refused, match="MIRIX_API_KEY"):
        await run.mirix_preflight(
            "c", tmp_path, connect=connect_ok, probe=contract(HONOURED)
        )
    monkeypatch.delenv("MIRIX_API_KEY")
    monkeypatch.setattr(run, "dotenv_files_visible_to_mirix", lambda: [Path("/x/.env")])
    with pytest.raises(run._Refused, match="load_dotenv"):
        await run.mirix_preflight(
            "c", tmp_path, connect=connect_ok, probe=contract(HONOURED)
        )


@pytest.mark.asyncio
async def test_a_mirix_cell_that_fails_preflight_uses_no_attempt(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    spec = CellSpec("alfworld", "luna", "mirix")
    checks: list[str] = []

    async def failing(cell_id: str, cell_dir: Path) -> dict[str, Any]:
        checks.append(cell_id)
        raise run._Refused("this MIRIX does not honour the retry/timeout contract")

    cell = runner(
        tmp_path, url, spec=spec, factory=lambda: MirixProbe([]), preflight=True
    )
    cell.mirix_check = failing
    outcome = await cell.run()
    assert outcome.status == "refused" and "contract" in outcome.detail
    assert checks == [spec.cell_id]
    assert cell_paths(tmp_path, spec).attempt_numbers() == []
    assert fake.calls("chat/completions") == []


# --- MIRIX cells get a per-cell server (H7.2, H8) ----------------------------------------------------


class MirixProbe(StoringMethod):
    """A stand-in for the MIRIX method: talks to the per-cell server run.py started."""

    name = "mirix"

    async def setup(self, cell: Any) -> None:
        await super().setup(cell)
        base = cell.options["mirix_base_url"]
        assert cell.options["mirix_port"] == int(base.rsplit(":", 1)[1])
        assert (
            cell.options["mirix_finalize_timeout_s"] == run.DEFAULT_FINALIZE_TIMEOUT_S
        )
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            env = (await client.get(base + "/env")).json()
        self.events.append(("mirix_home", env["HOME"]))


@pytest.mark.asyncio
async def test_mirix_cells_get_a_fresh_server_per_attempt_and_resume_it(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    stub = write_stub_server(tmp_path)
    modes: list[tuple[int, str]] = []

    def stub_server(spec: CellServerSpec, mode: Any) -> MirixServer:
        modes.append((spec.attempt, mode))
        return MirixServer(
            spec,
            mode=mode,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            database=False,
        )

    spec = CellSpec("alfworld", "luna", "mirix")
    events: list[Any] = []

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: MirixProbe(events, **kw),
        )
        cell.mirix_server_factory = stub_server
        return cell

    failed = await make(fail_after_task_at=1).run()
    assert failed.status == "learning_interrupted"
    # The learning resume restarts the attempt's own server on its HOME (mode resume),
    # and so does the test-phase resume after it.
    incomplete = await make(fail_test_questions={question("test:00")}).run()
    assert incomplete.status == "test_incomplete" and incomplete.attempt == 1
    done = await make().run()
    assert done.status == "complete"
    assert modes == [(1, "fresh"), (1, "resume"), (1, "resume")]
    paths = cell_paths(tmp_path, spec)
    assert read_json(paths.attempt(1).state)["mirix"]["home"] == str(
        paths.attempt(1).root / "home"
    )
    homes = [e[1] for e in events if e[0] == "mirix_home"]
    assert homes == [str(paths.attempt(1).root / "home")] * 3
    # The checkpoints kept HOME's files (the stub server's marker among them).
    record = read_json(
        paths.attempt(1).checkpoint(read_json(paths.attempt(1).latest_checkpoint)["id"])
        / "checkpoint.json"
    )
    assert record["mirix"]["db_name"] is None  # a server without a database
    assert "home_files/.mirix/stub_marker" in record["files"]
    # Every server was stopped and its lock released when its invocation ended.
    rows = read_jsonl(paths.root / "owned_processes.jsonl")
    assert [r["event"] for r in rows].count("started") == 3
    assert [r["event"] for r in rows].count("stopped") == 3
    assert read_json(paths.attempt(1).state)["mirix"]["mode"] == "resume"


# --- the cell database's identity and owner ----------------------------------------------------


def stub_mirix_servers(
    tmp_path: Path, seen: list[tuple[CellServerSpec, str]], **server: Any
):
    """run.py's MIRIX server factory for a stub server (no database unless ``server``
    says so), recording every spec it is given."""
    stub = write_stub_server(tmp_path)

    def factory(spec: CellServerSpec, mode: Any) -> MirixServer:
        seen.append((spec, mode))
        server.setdefault("database", False)
        return MirixServer(
            spec,
            mode=mode,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            **server,
        )

    return factory


@pytest.mark.asyncio
async def test_two_runs_dirs_with_one_cell_id_get_distinct_databases(
    tmp_path, fake, gateway, monkeypatch
):
    """Two runs directories that run one cell id on one Postgres get distinct database
    names: the name carries the cell directory's instance id, like its gateway run ids,
    so one directory's resume never drops the other's database."""
    from evals.skill_bench.infra import mirix_server as ms

    _gw, url = gateway
    instances = iter(["aaaa0001", "bbbb0002"])
    monkeypatch.setattr(run, "new_instance_id", lambda: next(instances))
    seen: list[tuple[CellServerSpec, str]] = []
    factory = stub_mirix_servers(tmp_path, seen)
    spec = CellSpec("alfworld", "luna", "mirix", "smoke", 2, 1)
    recorded = []
    for runs, instance in (("runs-a", "aaaa0001"), ("runs-b", "bbbb0002")):
        paths = CellPaths(tmp_path / runs / spec.cell_id)
        cell = runner(
            tmp_path, url, spec=spec, factory=lambda: MirixProbe([]), out_dir=paths.root
        )
        cell.mirix_server_factory = factory
        outcome = await cell.run()
        assert outcome.status == "complete", outcome.detail
        state = read_json(paths.attempt(1).state)
        assert state["instance"] == instance
        assert state["mirix"]["db_name"] == ms.cell_db_name(spec.cell_id, 1, instance)
        recorded.append(state["mirix"]["db_name"])
    assert recorded[0] != recorded[1]
    # Every server of a cell directory ran on that directory's own database, marked
    # with its own attempt's run id.
    for s, _mode in seen:
        instance = "aaaa0001" if "/runs-a/" in str(s.cell_dir) + "/" else "bbbb0002"
        assert s.instance == instance and s.name_bound_to_instance
        # MIRIX names the session it serves in the gateway's item header.
        assert dict(s.extra_env) == {"MIRIX_LLM_SESSION_HEADER": "x-skill-bench-item"}
        assert s.owner == f"{spec.cell_id}-a01-{instance}"
        assert s.db_name == ms.cell_db_name(spec.cell_id, 1, instance)
    assert {s.db_name for s, _ in seen} == set(recorded)


@pytest.mark.asyncio
async def test_a_recorded_database_name_is_reused_on_resume(tmp_path, fake, gateway):
    """An attempt keeps the database name it recorded, whatever the naming rule of the
    harness that resumes it: an older harness's attempt recorded a name without an
    instance id, and its test phase must find that database."""
    from evals.skill_bench.infra import mirix_server as ms

    _gw, url = gateway
    seen: list[tuple[CellServerSpec, str]] = []
    spec = CellSpec("alfworld", "luna", "mirix")

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(tmp_path, url, spec=spec, factory=lambda: MirixProbe([], **kw))
        cell.mirix_server_factory = stub_mirix_servers(tmp_path, seen)
        return cell

    failed = await make(fail_after_task_at=1).run()
    assert failed.status == "learning_interrupted"
    paths = cell_paths(tmp_path, spec)
    state = read_json(paths.attempt(1).state)
    assert state["mirix"]["db_name"] == ms.cell_db_name(spec.cell_id, 1, INSTANCE)
    legacy = ms.cell_db_name(spec.cell_id, 1)  # e.g. sb_alfworld_luna_mirix_a1_938b2ade
    state["mirix"]["db_name"] = legacy
    write_json_atomic(paths.attempt(1).state, state)
    done = await make().run()
    assert done.status == "complete", done.detail
    fresh, *resumed = seen
    assert fresh[1] == "fresh" and fresh[0].database is None
    assert resumed and all(
        s.db_name == legacy == s.database
        and s.owner == f"{spec.cell_id}-a01-{INSTANCE}"
        for s, _mode in resumed
    )
    assert not any(s.name_bound_to_instance for s, _mode in resumed)
    assert read_json(paths.attempt(1).state)["mirix"]["db_name"] == legacy


@pytest.mark.asyncio
async def test_a_database_conflict_at_a_fresh_start_waits_for_a_person(
    tmp_path, fake, gateway, monkeypatch
):
    """A new attempt that finds its database already there stops for a person (HARNESS
    ``mirix_database_conflict``), never for an automatic resume that would roll back
    what it found (another runs directory's database, for example). Not even
    --force-resume moves another run's database."""
    from evals.skill_bench.cell_records import (
        LEDGER_EVENT_RESUME,
        NEEDS_DECISION_HARNESS,
        RECOVERY_RESUME_FROM_CHECKPOINT,
    )
    from evals.skill_bench.infra import mirix_server as ms
    from evals.skill_bench.tests.test_mirix_server import changes, fake_connect

    _gw, url = gateway
    monkeypatch.setenv(ms.PG_CONTAINER_ENV, "fake-postgres")  # never `docker ps`
    spec = CellSpec("alfworld", "luna", "mirix")
    name = ms.cell_db_name(spec.cell_id, 1, INSTANCE)
    other = ms.DatabaseOwner(f"{spec.cell_id}-a01-{INSTANCE}-copy", spec.cell_id, 1)
    log: list[tuple[str, str]] = []
    existing = {name}
    comments = {name: other.comment()}
    seen: list[tuple[CellServerSpec, str]] = []

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(tmp_path, url, spec=spec, factory=lambda: MirixProbe([]), **kw)
        cell.mirix_server_factory = stub_mirix_servers(
            tmp_path, seen, database=True, connect=fake_connect(log, existing, comments)
        )
        return cell

    stopped = await make().run()
    assert stopped.status == "learning_failed", stopped.detail
    assert stopped.failure_kind == "harness"
    paths = cell_paths(tmp_path, spec)
    (row,) = read_jsonl(paths.attempt(1).failures)
    assert (row["kind"], row["code"]) == ("harness", "mirix_database_conflict")
    assert row["recovery"] != RECOVERY_RESUME_FROM_CHECKPOINT
    state = read_json(paths.attempt(1).state)
    assert state["needs_decision"] == NEEDS_DECISION_HARNESS
    assert other.run_id in state["failure"]["detail"]
    # The next plain invocation waits for the person; nothing is rolled back.
    again = await make().run()
    assert again.status == "refused" and "--force-resume" in again.detail
    # A person's --force-resume claims only databases without a marker: another run's
    # stays, and the resume stops before its ledger marker.
    forced = await make(force_resume=True).run()
    assert forced.status == "learning_failed" and forced.failure_kind == "harness"
    rows = read_jsonl(paths.attempt(1).failures)
    assert [(r["code"], r["stage"]) for r in rows] == [
        ("mirix_database_conflict", "cell"),
        ("mirix_database_conflict", "resume"),
    ]
    run_id = f"{spec.cell_id}-a01-{INSTANCE}"
    assert not any(
        r.get("event") == LEDGER_EVENT_RESUME for r in ledger(tmp_path, run_id)
    )
    assert not paths.attempt(1).discarded_log.exists()
    assert changes(log) == [] and existing == {name}
    assert comments == {name: other.comment()}


@pytest.mark.asyncio
async def test_a_resume_never_rolls_back_a_foreign_database(
    tmp_path, fake, gateway, monkeypatch
):
    """An infrastructure stop resumes automatically, but only after the database under
    the attempt's name is proven the attempt's: another run's is a HARNESS stop before
    the resume's ledger marker, and no rollback touches it."""
    from evals.skill_bench.cell_records import LEDGER_EVENT_RESUME
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import StubMirix, StubMirixServer

    _gw, url = gateway
    stubs: dict[tuple[str, int], StubMirix] = {}

    def server(s: CellServerSpec, mode: Any) -> Any:
        key = (str(s.cell_dir), s.attempt)
        if mode == "fresh":
            stubs[key] = StubMirix(threshold=2, curator_errors=["infra"])
        return StubMirixServer(stubs[key], mode)

    spec = CellSpec("alfworld", "luna", "mirix")

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: MirixMethod(poll_s=0.01),
            **kw,
        )
        cell.mirix_server_factory = server
        return cell

    failed = await make().run()
    assert failed.status == "learning_interrupted", failed.detail
    (stub,) = stubs.values()
    store = stub._checkpoint_store  # type: ignore[attr-defined]
    store.owner_conflict = "sb_x: owned by run someone-else-a01-0a1b2c3d"
    paths = cell_paths(tmp_path, spec)
    run_id = f"{spec.cell_id}-a01-{INSTANCE}"
    before = read_json(paths.attempt(1).state)
    cause = before["failure"]
    assert cause["code"] != "mirix_database_conflict"
    # The dead invocation's leftovers are stopped even though the resume stops
    # before its rollback (an orphan would keep writing and spending).
    cleaned: list[Path] = []
    real_cleanup = run.cleanup_owned

    def recording_cleanup(path: Path) -> Any:
        cleaned.append(path)
        return real_cleanup(path)

    monkeypatch.setattr(run, "cleanup_owned", recording_cleanup)

    def resume_markers() -> list[dict[str, Any]]:
        return [
            r for r in ledger(tmp_path, run_id) if r.get("event") == LEDGER_EVENT_RESUME
        ]

    stopped = await make().run()
    assert stopped.status == "learning_failed", stopped.detail
    assert stopped.failure_kind == "harness"
    assert cleaned == [paths.root / "owned_processes.jsonl"]
    state = read_json(paths.attempt(1).state)
    # The attempt keeps what stopped it (the resume a person decides still needs
    # it); the conflict is recorded next to it.
    assert state["failure"] == cause
    assert state["failure_stage"] == before["failure_stage"]
    assert state.get("resume_from") == before.get("resume_from")
    assert state["state"] == before["state"]
    assert state["resume_conflict"]["failure"]["code"] == "mirix_database_conflict"
    assert state["resume_conflict"]["stage"] == "resume" and state["needs_decision"]
    assert store.owner_checks == [(False, 1)] and store.rollbacks == []
    assert resume_markers() == [] and not paths.attempt(1).discarded_log.exists()
    # The plain next invocation waits, naming the conflict.
    waiting = await make().run()
    assert waiting.status == "refused" and "mirix_database_conflict" in waiting.detail
    # --force-resume claims only unmarked databases, never another run's.
    forced = await make(force_resume=True).run()
    assert forced.status == "learning_failed" and store.owner_checks[-1] == (True, 1)
    assert store.rollbacks == [] and resume_markers() == []
    assert read_json(paths.attempt(1).state)["failure"] == cause
    # Once the name holds the attempt's own database again, the resume goes on.
    store.owner_conflict = None
    done = await make(force_resume=True).run()
    assert done.status == "complete", done.detail
    assert store.rollbacks == [1] and len(resume_markers()) == 1
    (row,) = read_rows(paths.attempt(1).discarded_log)
    assert row["mirix"]["owner"]["states"] == {"stub_mirix": "ours"}
    # The discarded segment names the real cause, with the conflict beside it.
    assert row["failure"] == cause and row["reason"] == cause["code"]
    assert row["resume_conflict"]["failure"]["code"] == "mirix_database_conflict"
    assert "resume_conflict" not in read_json(paths.attempt(1).state)


@pytest.mark.asyncio
async def test_a_later_stop_after_an_owner_conflict_names_its_own_failure(
    tmp_path, fake, gateway, monkeypatch
):
    """A resume that stopped on another run's database records
    ``resume_conflict``; when a person's --force-resume then stops on its own failure (a
    restore that does not match its checkpoint), the attempt's record and the next
    refusal name that failure and its advice ("fix the code ... or --new-attempt"), not
    the stale conflict's "pass --force-resume", which would repeat the rollback."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import StubMirix, StubMirixServer

    _gw, url = gateway
    stubs: dict[tuple[str, int], StubMirix] = {}

    def server(s: CellServerSpec, mode: Any) -> Any:
        key = (str(s.cell_dir), s.attempt)
        if mode == "fresh":
            stubs[key] = StubMirix(threshold=2, curator_errors=["infra"])
        return StubMirixServer(stubs[key], mode)

    spec = CellSpec("alfworld", "luna", "mirix")

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: MirixMethod(poll_s=0.01),
            **kw,
        )
        cell.mirix_server_factory = server
        return cell

    failed = await make().run()
    assert failed.status == "learning_interrupted", failed.detail
    (stub,) = stubs.values()
    store = stub._checkpoint_store  # type: ignore[attr-defined]
    store.owner_conflict = "sb_x: owned by run someone-else-a01-0a1b2c3d"
    paths = cell_paths(tmp_path, spec)
    stopped = await make().run()
    assert stopped.status == "learning_failed", stopped.detail
    assert read_json(paths.attempt(1).state)["resume_conflict"]
    # The name holds the attempt's own database again, but the restore does not match
    # its checkpoint.
    store.owner_conflict = None

    async def mismatched(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise run.CheckpointMismatch("the restored database does not match its digest")

    monkeypatch.setattr(run.CellRunner, "_restore_checkpoint", mismatched)
    forced = await make(force_resume=True).run()
    assert forced.status == "learning_failed", forced.detail
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == run.FAILURE_CHECKPOINT_MISMATCH
    assert "resume_conflict" not in state
    waiting = await make().run()
    assert waiting.status == "refused"
    assert run.FAILURE_CHECKPOINT_MISMATCH in waiting.detail
    assert "--force-resume would stop the same way" in waiting.detail
    assert "mirix_database_conflict" not in waiting.detail


def test_every_path_option_is_made_absolute_once(tmp_path, monkeypatch):
    """The MIRIX server runs in its attempt directory and a launched
    gateway in the repository root, so a relative path must not reach them."""
    monkeypatch.chdir(tmp_path)
    options = run.RunOptions(
        out_dir=Path("runs/smoke/cell"),
        gateway_runs_dir=Path("runs/smoke"),
        env_file=Path("key.env"),
        blob_dir=Path("blobs"),
        acknowledge_rerun=Path("note.txt"),
    )
    cwd = Path.cwd()
    assert options.out_dir == cwd / "runs/smoke/cell"
    assert options.gateway_runs_dir == cwd / "runs/smoke"
    assert options.env_file == cwd / "key.env"
    assert options.blobs == cwd / "blobs"
    assert options.acknowledge_rerun == cwd / "note.txt"
    assert run.RunOptions(out_dir=Path("c")).blobs == cwd / "_blobs"
    recorded = options.to_dict()
    assert all(
        Path(recorded[k]).is_absolute()
        for k in ("out_dir", "gateway_runs_dir", "blob_dir", "acknowledge_rerun")
    )
    _spec, parsed, _args = run.parse_args(
        [
            *("--benchmark", "alfworld", "--model", "luna", "--method", "mirix"),
            *("--out", "runs/smoke/x", "--gateway-runs-dir", "runs/smoke"),
            *("--gateway-url", "http://127.0.0.1:1"),
        ]
    )
    assert parsed.out_dir == cwd / "runs/smoke/x"
    assert parsed.gateway_runs_dir == cwd / "runs/smoke"
    # absolute(), not resolve(): a symlinked runs directory keeps its spelling.
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    assert run.RunOptions(out_dir=Path("linked/c")).out_dir == cwd / "linked/c"


@pytest.mark.asyncio
async def test_a_mirix_cell_under_a_relative_out_dir_resumes_its_test_phase(
    tmp_path, fake, gateway, monkeypatch
):
    """With a relative ``--out``, the server's HOME is absolute all the same: resolved
    against the server's working directory (its attempt directory), MIRIX's state would
    land in a nested directory and the test-phase resume would be refused ("missing or
    empty"). The state lands in the documented HOME and the resume goes through."""
    _gw, url = gateway
    stub = write_stub_server(tmp_path)
    modes: list[tuple[int, str]] = []

    def stub_server(spec: CellServerSpec, mode: Any) -> MirixServer:
        modes.append((spec.attempt, mode))
        return MirixServer(
            spec,
            mode=mode,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            database=False,
        )

    monkeypatch.chdir(tmp_path)
    spec = CellSpec("alfworld", "luna", "mirix")
    events: list[Any] = []

    def make(**kw: Any) -> run.CellRunner:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: MirixProbe(events, **kw),
            out_dir=Path("runs") / spec.cell_id,  # a relative --out
        )
        cell.mirix_server_factory = stub_server
        return cell

    first = await make(fail_test_questions={question("test:00")}).run()
    assert first.status == "test_incomplete"
    done = await make().run()
    assert done.status == "complete", done.detail
    assert modes == [(1, "fresh"), (1, "resume")]
    attempt = cell_paths(tmp_path, spec).attempt(1).root
    home = attempt / "home"
    assert [e[1] for e in events if e[0] == "mirix_home"] == [str(home)] * 2
    assert (home / ".mirix" / "stub_marker").is_file()
    assert not (attempt / "runs").exists()  # no nested HOME below the attempt
    mirix = read_json(cell_paths(tmp_path, spec).attempt(1).state)["mirix"]
    assert mirix["mode"] == "resume" and mirix["home"] == str(home)


@pytest.mark.asyncio
async def test_a_dead_mirix_server_stops_the_test_phase_early(tmp_path, fake, gateway):
    """After an infra failure the runner checks the cell's infrastructure and stops taking
    items while it is down; they stay pending for the next invocation."""
    _gw, url = gateway
    stub = write_stub_server(tmp_path)
    servers: list[MirixServer] = []

    def stub_server(spec: CellServerSpec, mode: Any) -> MirixServer:
        server = MirixServer(
            spec,
            mode=mode,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            database=False,
        )
        servers.append(server)
        return server

    env = TinyEnv({"train": 1, "val": 1, "test": 6})
    first = seeded_order(env.load_tasks("test"))[0].raw_id  # the first item run

    async def kill_server(task: Any) -> None:
        if task.instruction == question(first):
            process = servers[-1].process
            assert process is not None
            process.popen.kill()
            await asyncio.to_thread(process.popen.wait)
            raise run.MirixServerError("MIRIX went away")

    spec = CellSpec("alfworld", "luna", "mirix")
    cell = runner(
        tmp_path,
        url,
        spec=spec,
        env=env,
        factory=lambda: MirixProbe([], on_before_task=kill_server),
        test_concurrency=1,
    )
    cell.mirix_server_factory = stub_server
    outcome = await cell.run()
    assert outcome.status == "test_incomplete"
    assert (
        outcome.pending == 6
    )  # the first item failed, the other five were not started
    rows = read_rows(cell_paths(tmp_path, spec).attempt(1).results)
    assert [r["failure"]["code"] for r in rows] == ["mirix_server"]


@pytest.mark.asyncio
async def test_the_real_mirix_method_runs_a_cell_on_a_run_py_managed_server(
    tmp_path, fake, gateway
):
    """run.py + the real MIRIX method (native API only) + the stub MIRIX of the method's
    tests, on the server run.py starts per attempt: learning, flush, freeze, a test phase
    that crashes and resumes on the same "database", and the H10.4 check."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import (
        StubMirix,
        StubMirixServer,
    )

    _gw, url = gateway
    stubs: dict[tuple[str, int], StubMirix] = {}
    modes: list[str] = []

    def server(spec: CellServerSpec, mode: Any) -> Any:
        key = (spec.cell_id, spec.attempt)
        modes.append(mode)
        if mode == "fresh":
            assert key not in stubs
            stubs[key] = StubMirix(threshold=2)
        return StubMirixServer(stubs[key], mode)

    spec = CellSpec("alfworld", "luna", "mirix")
    broken = {question("test:03")}

    class FlakyMirix(MirixMethod):
        async def before_task(self, task: Any) -> Any:
            if task.instruction in broken:
                broken.discard(task.instruction)
                raise InfraFailure("MIRIX retrieval connection reset")
            return await super().before_task(task)

    def make() -> run.CellRunner:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: FlakyMirix(poll_s=0.01),
        )
        cell.mirix_server_factory = server
        return cell

    first = await make().run()
    assert first.status == "test_incomplete" and first.pending == 1
    second = await make().run()
    assert second.status == "complete", second.detail
    assert modes == ["fresh", "resume"]
    stub = stubs[(spec.cell_id, 1)]
    paths = cell_paths(tmp_path, spec)
    learning = read_rows(paths.attempt(1).learning)
    # Every learning session reached MIRIX's own session store, and MIRIX learned skills.
    added = {r.json["session_id"] for r in stub.requests if r.path == "/memory/add"}
    assert added == {r["task_uid"] for r in learning}
    assert stub.skills and all(
        run["status"] == "succeeded" for run in stub.runs.values()
    )
    frozen = read_json(paths.attempt(1).frozen_snapshot)["snapshot"]
    assert frozen["skills"]
    # The test phase injected MIRIX's rendering and changed nothing (H10.4).
    tested = [r for r in read_rows(paths.attempt(1).injections) if r["phase"] == "test"]
    assert tested and all("Relevant skills" in r["text"] for r in tested)
    state = read_json(paths.attempt(1).state)
    assert [c["equal"] for c in state["snapshot_checks"]] == [True, True, True]
    assert [r["raw_id"] for r in state["test_reruns"]] == ["test:03"]
    costs = read_json(paths.attempt(1).costs)
    assert costs["by_phase_role"]["learning/mirix"]["calls"] > 0  # MIRIX's distiller
    assert read_json(paths.attempt(1).state)["mirix"]["mode"] == "resume"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", [m for m in protocol.METHODS if m != "mirix"]
)  # MIRIX: the test above
async def test_every_implemented_method_runs_a_cell_and_resumes_its_test_phase(
    tmp_path, fake, gateway, method
):
    """The real method class (``method_class``) through run.py, the real gateway and the
    fake upstream: a whole cell, then a harness restart in the test phase, which the
    method must resume from its frozen memory with an unchanged snapshot (H6, H8, H10.4)."""
    cls = run.method_class(method)
    if cls is None:
        pytest.skip(f"methods/{method}.py is still a placeholder")
    _gw, url = gateway
    spec = CellSpec("alfworld", "luna56", method)
    env = TinyEnv({"train": 3, "val": 1, "test": 3}, fail_reset={question("test:01")})
    first = await runner(tmp_path, url, spec=spec, env=env, factory=None).run()
    assert first.status == "test_incomplete", first.detail
    assert first.pending == 1
    second = await runner(tmp_path, url, spec=spec, env=env, factory=None).run()
    assert second.status == "complete", second.detail
    paths = cell_paths(tmp_path, spec)
    state = read_json(paths.attempt(1).state)
    assert [c["kind"] for c in state["snapshot_checks"]] == [
        "after_test",
        "resume",
        "after_test",
    ]
    assert all(c["equal"] for c in state["snapshot_checks"])
    assert [r["raw_id"] for r in read_rows(paths.results)] == [
        t.raw_id for t in seeded_order(env.load_tasks("test"))
    ]
    learned = 3 if method != "none" else 0
    assert len(read_rows(paths.attempt(1).learning)) == learned
    costs = read_json(paths.attempt(1).costs)
    assert costs["linkage"]["missing_from_ledger"] == []
    roles = set(costs["by_phase_role"])
    # The methods' own calls went through the gateway with their roles (H6, H7.3-H7.8).
    expected = {
        "none": set(),
        # With the fake's "ok" replies nothing parses into memory items, so ReasoningBank
        # stores (and embeds) nothing; its judge and extraction calls still happen.
        "reasoningbank": {"learning/judge", "learning/learner"},
        "awm": {"learning/judge"},
    }[method]
    assert expected <= roles, roles
    assert not {r for r in roles if r.startswith("test/")} - {
        "test/actor",
        "test/embedding",
    }, roles  # the frozen test phase never learns
    assert {k.split("/")[0] for k in costs["by_phase_role"]} <= {
        "setup",
        "learning",
        "test",
    }


# --- real environments -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protocol_mode_on_alfworld_scripted_games(tmp_path):
    """Protocol mode over the real ALFWorld manifests: 39 learning items in the seed-42
    order, then the 134 test items (games scripted, one model call each)."""
    games = Games()
    env = alf.AlfWorldEnv(
        data_root=tmp_path / "alfworld", backend_factory=games.factory()
    )
    fake = FakeUpstream(
        responder=lambda endpoint, body: text_reply(act("win"))
        if endpoint == "chat/completions"
        else None
    )
    spec = CellSpec("alfworld", "luna", "reasoningbank")
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        outcome = await runner(
            tmp_path,
            url,
            spec=spec,
            env=env,
            factory=storing(events),
            test_concurrency=8,
        ).run()
    assert outcome.status == "complete"
    paths = cell_paths(tmp_path, spec)
    train = [t.raw_id for t in env.load_tasks("train")]
    expected = list(train)
    random.Random(protocol.SEED).shuffle(expected)
    assert [r["raw_id"] for r in read_rows(paths.attempt(1).learning)] == expected
    final = read_rows(paths.results)
    manifest_order = [t.raw_id for t in env.load_tasks("test")]
    # The seed-42 order, so test items do not arrive grouped by task type.
    assert [r["raw_id"] for r in final] == [
        t.raw_id for t in seeded_order(env.load_tasks("test"))
    ]
    assert sorted(r["raw_id"] for r in final) == sorted(manifest_order)
    first_types = [r["task_type"] for r in final[:20]]
    assert len(set(first_types)) > 1  # the manifest's first 20 are all one task type
    assert len(final) == protocol.SPLIT_SIZES["alfworld"]["test"]
    assert {r["primary"] for r in final} == {1.0}
    assert {r["task_type"] for r in final} >= {"pick_and_place_simple"}
    manifest = read_json(paths.manifest)
    assert manifest["cell"]["split_mode"] == "protocol"
    assert manifest["cell"]["learning_items"] == expected


def test_exit_codes_are_distinct_per_outcome_class():
    assert EXIT_CODES["complete"] == 0
    assert len({EXIT_CODES[s] for s in ("test_incomplete", "learning_failed")}) == 2
    assert EXIT_CODES["refused"] == EXIT_CODES["violation"] == 4


# --- records -------------------------------------------------------------------------------------------


def test_ledger_summary(tmp_path):
    path = tmp_path / "gateway.jsonl"
    rows = [
        {"event": "phase", "phase": "learning"},
        {"event": "call_started", "call_id": "a"},
        {
            "event": "call",
            "call_id": "a",
            "role": "actor",
            "phase": "learning",
            "item": "t1",
            "status": "ok",
            "cost_total": 0.5,
            "billed_attempts": 2,
            "retries": 1,
            "provider": "P1",
            "model": "m",
            "model_served": "m-2026",
            "input_tokens_actual": 10,
            "output_tokens": 3,
            "protocol_conformant": True,
        },
        {
            "event": "call",
            "call_id": "b",
            "role": "learner",
            "phase": "learning",
            "status": "infra_error",
            "cost_total": None,
            "repeat_of": "x",
            "protocol_conformant": True,
        },
        {"event": "phase", "phase": "test"},
        {
            "event": "call",
            "call_id": "c",
            "role": "actor",
            "phase": "test",
            "item": "t2",
            "status": "ok",
            "cost_total": 0.25,
            "provider": "P2",
            "protocol_conformant": False,
        },
        {"event": "abandoned"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"event": "ca')
    summary = summarize_ledger(path)
    assert summary["phases"] == ["learning", "test"] and summary["abandoned"] is True
    learning = summary["by_phase_role"]["learning/actor"]
    assert learning["cost_total"] == 0.5 and learning["billed_attempts"] == 2
    assert learning["retries"] == 1 and learning["input_tokens_actual"] == 10
    assert summary["by_phase_role"]["learning/learner"]["failed"] == 1
    assert summary["totals"]["cost_total"] == 0.75 and summary["totals"]["calls"] == 3
    assert summary["providers"] == {
        "learning/actor": {"P1": 1},
        "test/actor": {"P2": 1},
    }
    assert summary["model_served"] == {"m": {"m-2026": 1}}
    assert summary["items"] == {"t1": {"learning": 0.5}, "t2": {"test": 0.25}}
    assert summary["repeat_of"] == {"learner": 1}
    assert summary["protocol_conformant"] is False
    assert summary["call_ids"] == ["a", "b", "c"]


def test_salt_is_created_once_and_private(tmp_path):
    path = tmp_path / "cell" / "private" / "salt"
    salt = run.load_or_create_salt(path)
    assert run.load_or_create_salt(path) == salt and len(salt) == 32
    assert (path.stat().st_mode & 0o777) == 0o600
    assert (path.parent.stat().st_mode & 0o777) == 0o700


def test_snapshots_must_be_json_objects():
    record = run.snapshot_record({"b": (1, 2), "a": "x"})
    assert record["snapshot"] == {"a": "x", "b": [1, 2]}
    assert record["sha256"] == run.snapshot_record({"a": "x", "b": [1, 2]})["sha256"]
    with pytest.raises(TypeError):
        run.snapshot_record(["not", "an", "object"])
    with pytest.raises(TypeError):
        run.snapshot_record({"a": object()})


# --- gateway locks, run ids, re-runs and recovery ---------------------------------------------


@pytest.mark.asyncio
async def test_a_live_cell_holds_its_gateways_users_lock(tmp_path, fake, gateway):
    """For as long as run.py uses a gateway it holds the gateway's users lock
    (shared), so no scheduler sweep or gateway launch kills the gateway under it."""
    from evals.skill_bench.infra.gateway import gateway_users_lock

    _gw, url = gateway
    parsed = httpx.URL(url)
    lock = gateway_users_lock(tmp_path / "runs", parsed.host, parsed.port or 80)
    seen: list[bool] = []

    async def probe(task: Any) -> None:
        seen.append(await asyncio.to_thread(lock_is_held, lock))

    outcome = await runner(
        tmp_path, url, factory=storing([], on_before_task=probe)
    ).run()
    assert outcome.status == "complete"
    assert seen and all(seen)
    assert not lock_is_held(lock)  # released with the link


@pytest.mark.asyncio
async def test_run_ids_carry_the_cell_directorys_instance(
    tmp_path, fake, gateway, monkeypatch
):
    """A cell started over in a new directory on the same gateway gets new gateway
    runs, so its costs and audit never include the old cell's calls; a copied directory
    (same instance) is refused instead of writing into the other cell's ledger."""
    _gw, url = gateway
    ids = iter(["aaaa0001", "bbbb0002"])
    monkeypatch.setattr(run, "new_instance_id", lambda: next(ids))
    first = await runner(tmp_path, url, factory=storing([])).run()
    assert first.status == "complete"
    paths = cell_paths(tmp_path)
    one = read_json(paths.attempt(1).state)
    assert one["run_id"] == f"{SPEC.cell_id}-a01-aaaa0001"
    assert one["instance"] == read_json(paths.manifest)["instance"] == "aaaa0001"

    def calls() -> int:
        by = read_json(paths.attempt(1).costs)["by_phase_role"]
        return sum(v["calls"] for v in by.values())

    first_calls = calls()
    paths.root.rename(tmp_path / "moved-away")
    second = await runner(tmp_path, url, factory=storing([])).run()
    assert second.status == "complete"
    two = read_json(paths.attempt(1).state)
    assert two["run_id"] == f"{SPEC.cell_id}-a01-bbbb0002"
    assert calls() == first_calls  # its own calls only, not the moved cell's too
    # A copy of the moved cell (same instance id) cannot run a new attempt on it.
    shutil.rmtree(paths.root)
    shutil.copytree(tmp_path / "moved-away", paths.root)
    for number in paths.attempt_numbers():
        shutil.rmtree(paths.attempt(number).root)
    for name in ("status.json", "results.jsonl", "learning.jsonl", "summary.json"):
        (paths.root / name).unlink(missing_ok=True)
    manifest = read_json(paths.manifest)
    manifest["attempts"], manifest["current_attempt"] = [], None
    write_json_atomic(paths.manifest, manifest)
    copied = await runner(tmp_path, url, factory=storing([])).run()
    assert copied.status == "refused" and "shares its instance id" in copied.detail


@pytest.mark.asyncio
async def test_a_cell_made_before_instance_ids_resumes_its_recorded_run(
    tmp_path, fake, gateway
):
    """An attempt started as <cell>-aNN (before instance ids) keeps that run id when it
    resumes."""
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:01")})
    ).run()
    assert first.status == "test_incomplete"
    paths = cell_paths(tmp_path)
    old = f"{SPEC.cell_id}-a01"
    (tmp_path / "runs" / RUN_A1).rename(tmp_path / "runs" / old)
    state = read_json(paths.attempt(1).state)
    state.pop("instance")
    state.update(run_id=old, ledger=str(tmp_path / "runs" / old / "gateway.jsonl"))
    write_json_atomic(paths.attempt(1).state, state)
    manifest = read_json(paths.manifest)
    manifest.pop("instance")
    write_json_atomic(paths.manifest, manifest)
    outcome = await runner(tmp_path, url, factory=storing([])).run()
    assert outcome.status == "complete"
    rerun = next(r for r in read_rows(paths.results) if r["raw_id"] == "test:01")
    assert rerun["run_id"] == old
    assert read_json(paths.attempt(1).costs)["run_id"] == old
    assert read_json(paths.manifest)["instance"] == INSTANCE  # for its next attempts


@pytest.mark.asyncio
async def test_an_attempt_stopped_before_it_existed_does_not_block_the_cell(
    tmp_path, fake, gateway, monkeypatch
):
    """An invocation stopped (SIGTERM) after the gateway created the new attempt's run
    but before attempt.json existed leaves a gateway run the next invocation's run id
    would collide with. The number is reserved first: the next invocation skips it,
    abandons its run and goes on."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    failed = await runner(
        tmp_path, url, factory=storing([], fail_after_task_at=0)
    ).run()
    # cell.json now holds the instance id
    assert failed.status == "learning_interrupted"
    real = run.CellRunner._check_ledger_link

    async def stopped_after_set_phase(self: run.CellRunner, run_id: str) -> None:
        await real(self, run_id)
        if run_id == RUN_A2:
            raise asyncio.CancelledError()  # SIGTERM between set_phase and attempt.json

    monkeypatch.setattr(run.CellRunner, "_check_ledger_link", stopped_after_set_phase)
    with pytest.raises(asyncio.CancelledError):
        # The operator starts over instead of resuming: attempt 1 is invalidated and
        # attempt 2 reserved.
        await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    monkeypatch.setattr(run.CellRunner, "_check_ledger_link", real)
    assert (tmp_path / "runs" / RUN_A2 / "gateway_state.json").is_file()
    assert not paths.attempt(2).state.exists()
    reserved = read_json(paths.attempt(2).reservation)
    assert (reserved["run_id"], reserved["instance"]) == (RUN_A2, INSTANCE)
    assert read_json(paths.status)["status"] == "interrupted"

    done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 3)
    assert paths.attempt_numbers() == [1, 3]
    assert read_json(paths.attempt(3).state)["run_id"] == (
        f"{SPEC.cell_id}-a03-{INSTANCE}"
    )
    settled = read_json(paths.attempt(2).reservation)
    assert settled["abandoned"] is True and settled["settled_at"]
    assert settled["settled_by"] == read_json(paths.status)["launch"]
    assert [r["event"] for r in ledger(tmp_path, RUN_A2)] == ["phase", "abandoned"]
    assert [a["attempt"] for a in read_json(paths.manifest)["attempts"]] == [1, 3]
    # Settled once: the next invocation leaves it alone.
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert again.detail == "already complete"
    assert read_json(paths.attempt(2).reservation) == settled


@pytest.mark.asyncio
async def test_an_unstarted_attempts_run_is_abandoned_once_the_gateway_takes_it(
    tmp_path, fake, gateway, monkeypatch
):
    """When the gateway fails the abandon of an attempt that never started, the
    reservation keeps the error and stays unsettled; the next invocation tries again."""
    _gw, url = gateway
    paths = cell_paths(tmp_path)
    assert (
        await runner(tmp_path, url, factory=storing([], fail_after_task_at=0)).run()
    ).status == "learning_interrupted"
    real_link = run.CellRunner._check_ledger_link

    async def stopped_after_set_phase(self: run.CellRunner, run_id: str) -> None:
        await real_link(self, run_id)
        if run_id == RUN_A2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(run.CellRunner, "_check_ledger_link", stopped_after_set_phase)
    with pytest.raises(asyncio.CancelledError):
        await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    monkeypatch.setattr(run.CellRunner, "_check_ledger_link", real_link)
    real_abandon = run.GatewayLink.abandon

    async def hiccup(self: run.GatewayLink, run_id: str, reason: str = "") -> None:
        raise httpx.ReadTimeout("the gateway is busy")

    monkeypatch.setattr(run.GatewayLink, "abandon", hiccup)
    failed = await runner(
        tmp_path, url, factory=storing([], fail_after_task_at=0)
    ).run()
    assert (failed.status, failed.attempt) == ("learning_interrupted", 3)
    pending = read_json(paths.attempt(2).reservation)
    assert isinstance(pending, dict) and "settled_at" not in pending
    assert "ReadTimeout" in pending["abandon_error"]
    monkeypatch.setattr(run.GatewayLink, "abandon", real_abandon)
    # The learning resume settles the reservation too, then completes attempt 3.
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 3)
    settled = read_json(paths.attempt(2).reservation)
    assert isinstance(settled, dict) and "abandon_error" not in settled
    assert settled["abandoned"] is True and settled["settled_at"]
    assert ledger(tmp_path, RUN_A2)[-1]["event"] == "abandoned"


@pytest.mark.asyncio
async def test_a_first_attempt_the_gateway_refused_takes_the_next_number(
    tmp_path, fake, gateway, monkeypatch
):
    """The gateway refused the first attempt's run (nothing was created there): the
    reservation is settled as never created and the cell goes on with attempt 2, under
    the instance id cell.json recorded before the reservation."""
    _gw, url = gateway
    ids = iter(["aaaa0001", "bbbb0002"])
    monkeypatch.setattr(run, "new_instance_id", lambda: next(ids))
    real = run.GatewayLink.set_phase

    async def refused(self: run.GatewayLink, run_id: str, phase: str) -> None:
        raise httpx.ConnectError("the gateway is restarting")

    monkeypatch.setattr(run.GatewayLink, "set_phase", refused)
    first = await runner(tmp_path, url, factory=storing([])).run()
    assert first.status == "infra_unavailable"
    paths = cell_paths(tmp_path)
    assert read_json(paths.manifest)["instance"] == "aaaa0001"
    assert read_json(paths.attempt(1).reservation)["instance"] == "aaaa0001"
    monkeypatch.setattr(run.GatewayLink, "set_phase", real)
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 2)
    assert read_json(paths.attempt(2).state)["run_id"] == (
        f"{SPEC.cell_id}-a02-aaaa0001"
    )
    settled = read_json(paths.attempt(1).reservation)
    assert settled["gateway_run"] == "never created" and settled["abandoned"] is False
    assert not (tmp_path / "runs" / f"{SPEC.cell_id}-a01-aaaa0001").exists()


@pytest.mark.asyncio
async def test_a_rerun_note_records_the_earlier_runs_it_acknowledges(
    tmp_path, fake, gateway, monkeypatch
):
    """A cell started over in a new directory after its old directory ran test items
    is the cell run twice (H1.3). --acknowledge-rerun copies the operator's note into the
    new cell and records it in cell.json with the earlier runs it covers, for the audit."""
    _gw, url = gateway
    ids = iter(["aaaa0001", "bbbb0002"])
    monkeypatch.setattr(run, "new_instance_id", lambda: next(ids))
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    paths = cell_paths(tmp_path)
    old_run = read_json(paths.attempt(1).state)["run_id"]
    note = tmp_path / "rerun.md"
    note.write_text("")
    empty = await runner(tmp_path, url, acknowledge_rerun=note).run()
    assert empty.status == "refused" and "is empty" in empty.detail
    note.write_text(
        "2026-09-28, the operator: the harness mis-scored item 3; re-run.\n"
    )
    # The cell's own runs are no earlier run: recorded without any (an earlier run on
    # another host has none here either), and late (this directory already ran its test
    # items).
    nothing = await runner(tmp_path, url, acknowledge_rerun=note).run()
    assert (nothing.status, nothing.detail) == ("complete", "already complete")
    (recorded,) = read_json(paths.manifest)["operator_notes"]
    assert recorded["superseded"] == [] and recorded["after_test"] is True
    assert recorded["tested_runs"] == [f"gateway run {old_run}", "attempt 1"]

    paths.root.rename(tmp_path / "moved-away")
    done = await runner(
        tmp_path, url, factory=storing([]), acknowledge_rerun=note
    ).run()
    assert done.status == "complete"
    manifest = read_json(paths.manifest)
    assert manifest["instance"] == "bbbb0002"
    (entry,) = manifest["operator_notes"]
    assert "after_test" not in entry  # before this directory's test phase
    assert entry["kind"] == "rerun_acknowledged" and entry["source"] == str(note)
    copied = paths.root / entry["file"]
    assert copied.read_bytes() == note.read_bytes()
    assert entry["sha256"] == hashlib.sha256(note.read_bytes()).hexdigest()
    (earlier,) = entry["superseded"]
    assert (earlier["run_id"], earlier["instance"]) == (old_run, "aaaa0001")
    assert earlier["test_actor_calls"] == TINY_SIZES["test"]
    # The same note again over the same runs is recorded once.
    again = await runner(tmp_path, url, acknowledge_rerun=note).run()
    assert again.detail == "already complete"
    assert len(read_json(paths.manifest)["operator_notes"]) == 1
    # Another note now that this directory ran its own test items: both runs' scores
    # could be seen, so a note could pick the better run.
    late = tmp_path / "late.md"
    late.write_text("2026-09-29, the operator: keep the second run.\n")
    recorded_late = await runner(tmp_path, url, acknowledge_rerun=late).run()
    assert (recorded_late.status, recorded_late.detail) == (
        "complete",
        "already complete",
    )
    notes = read_json(paths.manifest)["operator_notes"]
    assert len(notes) == 2 and notes[1]["after_test"] is True
    new_run = read_json(paths.attempt(1).state)["run_id"]
    assert notes[1]["tested_runs"] == [f"gateway run {new_run}", "attempt 1"]
    assert [r["instance"] for r in notes[1]["superseded"]] == ["aaaa0001"]
    assert read_json(paths.status)["status"] == "complete"
    assert "--acknowledge-rerun" in " ".join(
        a.option_strings[0] for a in run.build_parser()._actions if a.option_strings
    )


@pytest.mark.asyncio
async def test_refusals_say_how_to_start_a_cell_over(
    tmp_path, fake, gateway, monkeypatch
):
    """A refusal tells the operator to move the old cell directory out of the runs
    directory (renamed in place, both copies stay cells there) and leave its gateway
    runs; a protocol cell needs an operator note before its test phase, while a smoke
    cell (val items) may simply be run again, and its note is recorded at any time."""
    _gw, url = gateway
    ids = iter(["aaaa0001", "bbbb0002", "cccc0003"])
    monkeypatch.setattr(run, "new_instance_id", lambda: next(ids))
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    protocol_cell = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True
    ).run()
    assert protocol_cell.status == "refused"
    assert "--note" in protocol_cell.detail
    assert "out of the runs directory" in protocol_cell.detail
    assert "runs-superseded/" in protocol_cell.detail
    assert "before its test phase" not in protocol_cell.detail
    assert "--acknowledge-rerun" in protocol_cell.detail
    assert "smoke" not in protocol_cell.detail

    smoke = CellSpec("alfworld", "luna", "reasoningbank", "smoke", 2, 2)
    first = await runner(tmp_path, url, spec=smoke, factory=storing([])).run()
    assert first.status == "complete"
    refused = await runner(
        tmp_path, url, spec=smoke, factory=storing([]), new_attempt=True
    ).run()
    assert refused.status == "refused"
    assert "A smoke cell (val items) may be run again" in refused.detail
    assert "out of the runs directory" in refused.detail
    assert "smoke_cell_rerun" in refused.detail and "--note" in refused.detail
    assert "--acknowledge-rerun" not in refused.detail
    # With a note a smoke cell's attempt is superseded too.
    noted_rerun = await runner(
        tmp_path, url, spec=smoke, factory=storing([]), new_attempt=True, note="again"
    ).run()
    assert (noted_rerun.status, noted_rerun.attempt) == ("complete", 2)
    smoke_paths = cell_paths(tmp_path, smoke)
    assert read_json(smoke_paths.attempt(1).state)["invalidation"] == "superseded"
    # The smoke cell run again from a new directory; a note after its test phase is
    # recorded (the audit only warns about a smoke re-run).
    root = cell_paths(tmp_path, smoke).root
    root.rename(tmp_path / "runs-superseded")
    again = await runner(tmp_path, url, spec=smoke, factory=storing([])).run()
    assert again.status == "complete"
    note = tmp_path / "smoke-rerun.md"
    note.write_text("2026-09-28, the operator: smoke re-run after a harness fix.\n")
    noted = await runner(tmp_path, url, spec=smoke, acknowledge_rerun=note).run()
    assert noted.detail == "already complete"
    (entry,) = read_json(cell_paths(tmp_path, smoke).manifest)["operator_notes"]
    # Both attempts of the earlier directory (the superseded one and its replacement).
    assert [r["instance"] for r in entry["superseded"]] == ["bbbb0002", "bbbb0002"]
    assert "after_test" not in entry  # a smoke cell's note is never late


@pytest.mark.asyncio
async def test_an_unreachable_gateway_is_infrastructure_not_a_refusal(tmp_path):
    """Nothing listens at the gateway URL: exit 7, resumable, no attempt used."""
    port = await asyncio.to_thread(run.find_free_port)
    outcome = await runner(
        tmp_path, f"http://127.0.0.1:{port}", factory=storing([])
    ).run()
    assert outcome.status == "infra_unavailable"
    assert outcome.exit_code == EXIT_CODES["infra_unavailable"] == 7
    paths = cell_paths(tmp_path)
    assert paths.attempt_numbers() == []
    assert read_json(paths.status)["status"] == "infra_unavailable"
    from evals.skill_bench.cell_records import RESUMABLE_STATUSES

    assert "infra_unavailable" in RESUMABLE_STATUSES


@pytest.mark.asyncio
async def test_a_complete_cell_without_its_published_records_rewrites_them(
    tmp_path, fake, gateway
):
    """A crash between marking the attempt complete and writing the cell's files (an
    older run.py wrote them after) must not lose the cell: the next run restores them."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    paths = cell_paths(tmp_path)
    before = {p: p.read_bytes() for p in (paths.results, paths.learning, paths.summary)}
    for path in before:
        path.unlink()
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert again.status == "complete" and again.detail == "already complete"
    for path, data in before.items():
        assert path.is_file(), path
    assert paths.results.read_bytes() == before[paths.results]
    assert paths.learning.read_bytes() == before[paths.learning]


@pytest.mark.asyncio
async def test_the_cell_records_are_written_before_the_attempt_is_complete(
    tmp_path, fake, gateway, monkeypatch
):
    """_complete writes results/learning/summary first, then flips the attempt."""
    _gw, url = gateway
    seen: list[bool] = []
    original = run.CellRunner._update_attempt

    async def spy(self: Any, number: int, **fields: Any) -> Any:
        if fields.get("state") == ATTEMPT_COMPLETE:
            seen.append(self.paths.results.is_file() and self.paths.summary.is_file())
        return await original(self, number, **fields)

    monkeypatch.setattr(run.CellRunner, "_update_attempt", spy)
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    assert seen == [True]


class _RefusingResume(StoringMethod):
    """A method whose test-phase setup finds its frozen memory changed."""

    async def setup(self, cell: Any) -> None:
        if cell.phase == "test":
            raise run.FrozenMemoryMismatch("the stored memory differs from frozen.json")
        await super().setup(cell)


@pytest.mark.asyncio
async def test_a_frozen_memory_the_method_cannot_restore_is_a_violation(
    tmp_path, fake, gateway
):
    """A method that finds its frozen memory missing or changed on resume raises
    FrozenMemoryMismatch: an H10.4 violation, never a pending item re-tried forever."""
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:02")})
    ).run()
    assert first.status == "test_incomplete"
    second = await runner(tmp_path, url, factory=lambda: _RefusingResume([])).run()
    assert second.status == "violation" and second.exit_code == 4
    state = read_json(cell_paths(tmp_path).attempt(1).state)
    assert state["state"] == ATTEMPT_VIOLATION
    assert "FrozenMemoryMismatch" in state["violation"]["detail"]


@pytest.mark.asyncio
async def test_a_resumed_test_phase_is_in_the_test_phase_from_its_first_call(
    tmp_path, fake, gateway
):
    """run.py puts the run in its test phase before the method's resume setup, so
    a call made while the method restarts is never tagged with a stale phase."""
    _gw, url = gateway
    first = await runner(
        tmp_path, url, factory=storing([], fail_test_questions={question("test:02")})
    ).run()
    assert first.status == "test_incomplete"
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        stale = await client.post(
            f"/runs/{RUN_A1}/phase",
            json={"phase": "learning"},
            headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
        )
    assert stale.status_code == 200
    phases: list[str] = []

    class Probe(StoringMethod):
        async def setup(self, cell: Any) -> None:
            state = json.loads(
                (tmp_path / "runs" / RUN_A1 / "gateway_state.json").read_text()
            )
            phases.append(state["phase"])
            await super().setup(cell)

    outcome = await runner(tmp_path, url, factory=lambda: Probe([])).run()
    assert outcome.status == "complete"
    assert phases == ["test"]


@pytest.mark.asyncio
async def test_a_memory_changed_before_an_interruption_is_a_violation(
    tmp_path, fake, gateway
):
    """The snapshot taken when the test phase is interrupted is compared with the
    frozen one; a memory the test phase wrote is caught there, not left to a warning."""
    _gw, url = gateway
    reached = asyncio.Event()
    count = 0

    async def block(task: Any) -> None:
        nonlocal count
        count += 1
        if count == TINY_SIZES["train"] + 2:  # the second test item
            reached.set()
            await asyncio.sleep(3600)

    cell = asyncio.ensure_future(
        runner(
            tmp_path,
            url,
            factory=storing([], write_in_test=True, on_before_task=block),
            test_concurrency=1,
        ).run()
    )
    await asyncio.wait_for(reached.wait(), 30)
    cell.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cell
    state = read_json(cell_paths(tmp_path).attempt(1).state)
    assert state["state"] == ATTEMPT_VIOLATION
    last = state["snapshot_checks"][-1]
    assert (last["kind"], last["equal"], last["interrupted"]) == (
        "after_test",
        False,
        True,
    )


@pytest.mark.asyncio
async def test_every_launch_that_runs_work_records_the_code_it_ran(
    tmp_path, fake, gateway
):
    """The model-visible harness code (and, for MIRIX, the MIRIX package) of every
    invocation that runs an attempt's work, for the audit's per-attempt and per-benchmark
    checks. The start line of an invocation carries no code."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    paths = cell_paths(tmp_path)
    expected = protocol.code_fingerprint()
    assert read_json(paths.attempt(1).state)["code"] == expected
    start, work = read_rows(paths.launches)
    assert start["event"] == "start" and "code" not in start
    assert start["git"] == FIXED_GIT and start["options"]["out_dir"]
    assert (work["event"], work["attempt"], work["phase"]) == ("work", 1, "learning")
    assert work["launch"] == start["launch"] and work["code"] == expected
    assert work["git"] == FIXED_GIT
    assert "mirix" not in expected
    # The version it ran under: segment 0, no note, the tree
    # already frozen; the per-key constants and per-file hashes in its version.json.
    assert (work["segment"], work["note"], work["auto_freeze"]) == (
        0,
        None,
        "already_frozen",
    )
    assert "auto_freeze_error" not in work
    state = read_json(paths.attempt(1).state)
    (segment,) = state["version_segments"]
    assert segment["code"] == expected and segment["launches"] == [work["launch"]]
    assert segment["tree"]["dirty"] is False and segment["tree"]["diff"] is None
    version = read_json(paths.root / segment["version_file"])
    manifest = protocol.fingerprint_manifest()
    assert version["constants"] == json.loads(json.dumps(manifest["constants"]))
    assert version["files"] == manifest["files"]
    assert version["code"]["harness"] == expected["harness"]
    assert (
        set(version["code"]["harness_sha256"]) and "mirix_sha256" not in version["code"]
    )
    assert version["tree"]["dirty"] is False
    assert not (paths.attempt(1).root / "versions" / "00" / "tree.diff").exists()
    assert state["versions_mixed"] == {
        "protocol": False,
        "code": False,
        "commit": False,
    }
    assert state["note"] is None and state["supersedes"] is None


@pytest.mark.asyncio
async def test_an_invocation_that_runs_nothing_records_no_code(
    tmp_path, fake, gateway, monkeypatch
):
    """A no-op invocation after a code change (the cell already complete, a refused
    --new-attempt) writes a start line without code and no work line, so the finished
    attempt keeps one code."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    paths = cell_paths(tmp_path)
    changed = {**protocol.code_fingerprint(), "harness": "f" * 64}
    monkeypatch.setattr(protocol, "code_fingerprint", lambda **_kw: dict(changed))
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert (again.status, again.detail) == ("complete", "already complete")
    refused = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert refused.status == "refused" and "--acknowledge-rerun" in refused.detail
    assert "--note" in refused.detail
    launches = read_rows(paths.launches)
    assert [r["event"] for r in launches] == ["start", "work", "start", "start"]
    assert not any(r.get("code") == changed for r in launches)
    assert all("code" not in r for r in launches if r["event"] == "start")
    assert read_json(paths.status)["status"] == "complete"


class _SwallowingMethod(StoringMethod):
    """A method whose own model call fails on infrastructure and that carries on, as MIRIX
    does when it defers a session: the ledger shows it."""

    async def after_task(self, trajectory: Any) -> None:
        await super().after_task(trajectory)
        if self.learned == 2:
            assert self.cell is not None
            async with httpx.AsyncClient(
                base_url=self.cell.gateway_base_url, trust_env=False, timeout=60
            ) as client:
                response = await client.post(
                    f"/runs/{self.cell.run_id}/embedding/v1/embeddings",
                    json={"model": protocol.EMBEDDING_MODEL, "input": "FAIL-ME"},
                    headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                )
            self.events.append(("embedding", response.status_code))  # and carries on


@pytest.mark.asyncio
async def test_a_learning_call_lost_to_an_outage_restarts_the_attempt(tmp_path):
    """An INFRA failure on the embedding (or mirix) role in the learning phase that
    no retry recovered restarts the attempt, even though the method carried on."""
    failing = {"on": True}

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        if (
            endpoint == "embeddings"
            and body.get("input") == "FAIL-ME"
            and failing["on"]
        ):
            # The gateway's upstream key refused: INFRA, and not retried by the gateway.
            return Reply(status=401, json={"error": {"message": "bad key"}})
        return tiny_responder(endpoint, body)

    fake = FakeUpstream(responder=responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        failed = await runner(
            tmp_path, url, factory=lambda: _SwallowingMethod(events)
        ).run()
        assert ("embedding", 401) in events
        assert failed.status == "learning_interrupted" and failed.exit_code == 8
        paths = cell_paths(tmp_path)
        state = read_json(paths.attempt(1).state)
        assert state["state"] == "learning_interrupted"
        assert state["failure"]["code"] == "mirix_learning", state["failure"]
        assert state["failure_stage"] == "learning_calls"
        # The lost call ran in item 1's after_task, after its played checkpoint: the
        # resume restores that checkpoint, so the memory never keeps its effect.
        assert state["resume_from"] == "c0001-played"
        failing["on"] = False
        events.clear()
        outcome = await runner(
            tmp_path, url, factory=lambda: _SwallowingMethod(events)
        ).run()
    assert ("embedding", 200) in events
    assert outcome.status == "complete" and outcome.attempt == 1, (
        outcome.detail,
        read_json(cell_paths(tmp_path).attempt(1).state).get("failure"),
    )
    assert events[0] == ("setup", "learning", 1, RUN_A1, "c0001-played")
    segment = read_rows(paths.attempt(1).discarded_log)[0]
    assert segment["checkpoint"]["id"] == "c0001-played"
    lost = [c for c in segment["calls"] if c["role"] == "embedding"]
    assert len(lost) == 1 and lost[0]["status"] != "ok"
    # The discarded call is not the live memory's: the ledger check passed after it.
    costs = read_json(paths.attempt(1).costs)
    assert costs["discarded"]["calls"] == len(segment["calls"])
    assert costs["discarded"]["by_role"] == {"embedding": 1}


@pytest.mark.asyncio
async def test_a_mirix_test_phase_resumes_only_on_the_gateway_its_agent_calls(
    tmp_path, fake
):
    """The frozen MIRIX agent keeps the gateway URL it was set up with; a resume
    through another gateway is refused with the fix (attach there, or launch on its
    port), and launch mode reuses the recorded port."""
    stub = write_stub_server(tmp_path)

    def stub_server(spec: CellServerSpec, mode: Any) -> MirixServer:
        return MirixServer(
            spec,
            mode=mode,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            database=False,
        )

    spec = CellSpec("alfworld", "luna", "mirix")
    env = TinyEnv({"train": 1, "val": 1, "test": 2})
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            env=env,
            factory=lambda: MirixProbe([], fail_test_questions={question("test:00")}),
        )
        cell.mirix_server_factory = stub_server
        first = await cell.run()
    assert first.status == "test_incomplete", first.detail
    paths = cell_paths(tmp_path, spec)
    recorded = read_json(paths.attempt(1).state)["gateway_base_url"]
    assert recorded == url
    with conformant_gateway(tmp_path, fake) as (_gw2, url2):  # same runs dir
        other = runner(
            tmp_path, url2, spec=spec, env=env, factory=lambda: MirixProbe([])
        )
        other.mirix_server_factory = stub_server
        refused = await other.run()
    assert refused.status == "refused"
    port = httpx.URL(url).port
    assert f"--gateway-port {port}" in refused.detail and url in refused.detail
    # Launch mode picks the recorded port for this attempt.
    launch = run.CellRunner(
        spec,
        run.RunOptions(out_dir=paths.root, gateway_runs_dir=tmp_path / "runs"),
        env_factory=lambda _b: env,
    )
    assert launch._resume_gateway_port() == port
    assert run.same_gateway(url, url.replace("127.0.0.1", "localhost"))
    assert not run.same_gateway(url, url2)


# --- superseding a tested attempt; version segments ----------------------------------------


@pytest.mark.asyncio
async def test_new_attempt_supersedes_a_tested_attempt_with_a_note(
    tmp_path, fake, gateway
):
    """H1.3: ``--new-attempt`` over a tested attempt needs ``--note``. The old
    attempt stays in the directory, marked superseded (its note, what it scored, the
    digest of the cell records it published, the attempt that replaced it), its gateway
    run is abandoned, and the new attempt names what it supersedes. Repeating the same
    request goes on with the attempt it started instead of replacing it again."""
    from evals.skill_bench.cell_records import records_digest

    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    paths = cell_paths(tmp_path)
    published = records_digest(paths)
    refused = await runner(tmp_path, url, factory=storing([]), new_attempt=True).run()
    assert refused.status == "refused"
    assert "--note" in refused.detail and "already run its test phase" in refused.detail
    assert read_json(paths.attempt(1).state)["state"] == ATTEMPT_COMPLETE
    status = read_json(paths.status)
    assert (
        status["status"] == "complete" and "--note" in status["last_refusal"]["detail"]
    )

    note = "rerun after the curator fix"
    events: list[Any] = []
    done = await runner(
        tmp_path, url, factory=storing(events), new_attempt=True, note=note
    ).run()
    assert (done.status, done.attempt) == ("complete", 2), done.detail
    old = read_json(paths.attempt(1).state)
    assert (old["state"], old["invalidation"]) == (ATTEMPT_INVALID, "superseded")
    assert old["previous_state"] == ATTEMPT_COMPLETE and old["abandoned"] is True
    assert note in old["invalid_reason"]
    superseded = old["superseded"]
    assert superseded["note"] == note and superseded["by_attempt"] == 2
    assert superseded["scored_items"] == TINY_SIZES["test"]
    assert superseded["test_actor_calls"] == TINY_SIZES["test"]
    assert superseded["cell_records_sha256"] == published
    new = read_json(paths.attempt(2).state)
    assert new["note"]["text"] == note and new["note"]["new_attempt"] is True
    assert new["supersedes"] == 1
    (segment,) = new["version_segments"]
    assert segment["why"].startswith("superseding tested attempt 1 (complete)")
    assert segment["note"] == note
    assert {r["run_id"] for r in read_rows(paths.results)} == {RUN_A2}
    assert any(r.get("event") == "abandoned" for r in ledger(tmp_path, RUN_A1))
    summaries = {a["attempt"]: a for a in read_json(paths.manifest)["attempts"]}
    assert summaries[1]["superseded"]["by_attempt"] == 2
    assert (summaries[2]["note"], summaries[2]["supersedes"]) == (note, 1)
    works = [r for r in read_rows(paths.launches) if r["event"] == "work"]
    assert [(w["attempt"], w["note"]) for w in works] == [(1, None), (2, note)]

    # The same request again (a supervisor restart of the same argv): not replaced.
    again = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note=note
    ).run()
    assert (again.status, again.detail) == ("complete", "already complete")
    assert paths.attempt_numbers() == [1, 2]
    # Another note replaces it again.
    third = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note="and once more"
    ).run()
    assert (third.status, third.attempt) == ("complete", 3), third.detail
    assert read_json(paths.attempt(2).state)["superseded"]["by_attempt"] == 3


@pytest.mark.asyncio
async def test_a_repeated_request_resumes_the_attempt_it_started(
    tmp_path, fake, gateway
):
    """The note guard also holds before the new attempt is complete: an interrupted
    learning phase of the attempt the request started is resumed, not replaced."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    note = "rerun after the curator fix"
    stopped = await runner(
        tmp_path,
        url,
        factory=storing([], fail_after_task_at=1),
        new_attempt=True,
        note=note,
    ).run()
    assert (stopped.status, stopped.attempt) == ("learning_interrupted", 2)
    resumed = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note=note
    ).run()
    assert (resumed.status, resumed.attempt) == ("complete", 2), resumed.detail
    paths = cell_paths(tmp_path)
    assert paths.attempt_numbers() == [1, 2]
    state = read_json(paths.attempt(2).state)
    assert len(state["resumes"]) == 1 and state["note"]["text"] == note


@pytest.mark.asyncio
async def test_a_supersede_interrupted_before_its_new_attempt_is_finished_later(
    tmp_path, fake, gateway, monkeypatch
):
    """A crash (here: the gateway refusing the new run) between superseding the tested
    attempt and writing its replacement's attempt.json: the next invocation's new attempt
    still names the one it supersedes."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    real_set_phase = run.GatewayLink.set_phase

    async def refused(self: run.GatewayLink, run_id: str, phase: str) -> None:
        if run_id == RUN_A2:
            raise httpx.ConnectError("the gateway is restarting")
        await real_set_phase(self, run_id, phase)

    monkeypatch.setattr(run.GatewayLink, "set_phase", refused)
    halted = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note="rerun"
    ).run()
    assert halted.status == "infra_unavailable"
    paths = cell_paths(tmp_path)
    old = read_json(paths.attempt(1).state)
    assert (
        old["invalidation"] == "superseded" and old["superseded"]["by_attempt"] is None
    )
    monkeypatch.setattr(run.GatewayLink, "set_phase", real_set_phase)
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 3), done.detail
    assert read_json(paths.attempt(3).state)["supersedes"] == 1
    assert read_json(paths.attempt(1).state)["superseded"]["by_attempt"] == 3
    (segment,) = read_json(paths.attempt(3).state)["version_segments"]
    assert segment["why"] == "after invalid attempt 1"


@pytest.mark.asyncio
async def test_a_replacement_written_before_its_link_is_linked_later(
    tmp_path, fake, gateway
):
    """A crash between the new attempt's attempt.json and the superseded attempt's
    ``by_attempt``: the next invocation names the replacement."""
    _gw, url = gateway
    assert (await runner(tmp_path, url, factory=storing([])).run()).status == "complete"
    done = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note="rerun"
    ).run()
    assert (done.status, done.attempt) == ("complete", 2)
    paths = cell_paths(tmp_path)
    old = read_json(paths.attempt(1).state)
    old["superseded"]["by_attempt"] = None
    write_json_atomic(paths.attempt(1).state, old)
    again = await runner(tmp_path, url, factory=storing([])).run()
    assert (again.status, again.detail) == ("complete", "already complete")
    assert read_json(paths.attempt(1).state)["superseded"]["by_attempt"] == 2
    # A stale pending diff of a crashed launch is dropped by the next one that records
    # a version.
    stale = paths.attempt(2).root / "versions" / ".pending-deadbeef.diff.tmp"
    stale.write_text("partial")
    note = "and again"
    third = await runner(
        tmp_path, url, factory=storing([]), new_attempt=True, note=note
    ).run()
    assert third.attempt == 3 and stale.exists()  # another attempt's directory
    runner_cell = run.CellRunner(SPEC, run.RunOptions(out_dir=paths.root))
    runner_cell._drop_pending_diffs(paths.attempt(2))
    assert not stale.exists()


def test_a_note_is_one_line():
    options = run.RunOptions(out_dir=Path("x"), note="  why this run  ")
    assert (
        options.note == "why this run" and options.to_dict()["note"] == "why this run"
    )
    assert run.RunOptions(out_dir=Path("x")).to_dict()["note"] is None
    for bad in ("", "   ", "two\nlines", "a\rb", "x" * (run.MAX_NOTE_CHARS + 1)):
        with pytest.raises(ValueError, match="--note is one line"):
            run.RunOptions(out_dir=Path("x"), note=bad)
    assert run.RunOptions(out_dir=Path("x"), note="x" * run.MAX_NOTE_CHARS).note


def _fake_capture(diff: bytes):
    """A ``run.capture_tree`` that writes ``diff`` as the tree diff."""

    def capture(dest: Path, **_kw: Any) -> dict[str, Any]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(diff)
        return {
            "dirty": True,
            "diff": str(dest),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "diff_bytes": len(diff),
            "untracked": 0,
            "untracked_skipped": [],
            "error": None,
        }

    return capture


@pytest.mark.asyncio
async def test_a_dirty_tree_is_saved_with_the_segment(tmp_path, fake, gateway):
    """A launch from a tree with uncommitted changes records them:
    the segment's tree.diff (git apply on the commit rebuilds the tree). The same diff
    again is the same version; another diff opens a segment that names only the tree."""
    _gw, url = gateway
    dirty = {**FIXED_GIT, "dirty": True}
    first_diff = b"diff --git a/x b/x\n+one\n"
    stopped = await runner(
        tmp_path,
        url,
        factory=storing(
            [], fail_test_questions={question("test:00"), question("test:01")}
        ),
        git=dirty,
        capture=_fake_capture(first_diff),
    ).run()
    assert stopped.status == "test_incomplete"
    paths = cell_paths(tmp_path)
    (segment,) = read_json(paths.attempt(1).state)["version_segments"]
    tree = segment["tree"]
    assert tree["diff"] == "attempt-01/versions/00/tree.diff"
    assert tree["diff_sha256"] == hashlib.sha256(first_diff).hexdigest()
    assert (paths.root / tree["diff"]).read_bytes() == first_diff
    assert read_json(paths.root / segment["version_file"])["tree"] == tree
    versions = paths.attempt(1).root / "versions"
    assert not [p for p in versions.iterdir() if p.name.startswith(".pending")]
    # The same diff again: the same version.
    fail_one = {question("test:01")}
    same = await runner(
        tmp_path,
        url,
        factory=storing([], fail_test_questions=fail_one),
        git=dirty,
        capture=_fake_capture(first_diff),
    ).run()
    assert same.status == "test_incomplete"
    (segment,) = read_json(paths.attempt(1).state)["version_segments"]
    assert len(segment["launches"]) == 2
    assert not [p for p in versions.iterdir() if p.name.startswith(".pending")]
    # Another diff: a new segment, only the tree changed.
    done = await runner(
        tmp_path,
        url,
        factory=storing([]),
        git=dirty,
        capture=_fake_capture(b"diff --git a/x b/x\n+two\n"),
    ).run()
    assert done.status == "complete"
    state = read_json(paths.attempt(1).state)
    _first, second = state["version_segments"]
    changed = second["changed"]
    assert changed["tree"] is True
    assert (changed["protocol"], changed["code"], changed["commit"]) == (
        False,
        [],
        False,
    )
    assert second["test_from"] == {"scored": TINY_SIZES["test"] - 1, "pending": 1}
    assert second["tree"]["diff"] == "attempt-01/versions/01/tree.diff"
    assert state["versions_mixed"] == {"protocol": False, "code": False, "commit": True}
    # The tests' runner never diffs the real checkout: a dirty tree without a capture
    # of its own is recorded as not captured.
    other = CellSpec("alfworld", "luna", "awm")
    plain = await runner(
        tmp_path, url, spec=other, factory=storing([], name="awm"), git=dirty
    ).run()
    assert plain.status == "complete"
    (segment,) = read_json(cell_paths(tmp_path, other).attempt(1).state)[
        "version_segments"
    ]
    assert (
        segment["tree"]["diff"] is None and "not captured" in segment["tree"]["error"]
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_capture_tree_reproduces_the_tree(tmp_path):
    """One binary patch (tracked changes and untracked files under the pathspecs)
    that git apply on the commit turns back into the tree; a clean tree writes nothing;
    what the pathspecs exclude is left out; a failing git is recorded, not raised."""
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "sub" / "a.txt").write_text("one\n")
    (repo / "sub" / "skip.txt").write_text("kept out\n")
    (repo / "sub" / "img.bin").write_bytes(bytes(range(256)))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    specs = ["sub", ":(exclude)sub/skip.txt"]
    clean = run.capture_tree(
        tmp_path / "out" / "clean.diff", pathspecs=specs, repo=repo
    )
    assert clean["dirty"] is False and clean["error"] is None
    assert not (tmp_path / "out" / "clean.diff").exists()

    (repo / "sub" / "a.txt").write_text("two\n")
    (repo / "sub" / "img.bin").write_bytes(bytes(reversed(range(256))))
    (repo / "sub" / "b.txt").write_text("new\n")
    (repo / "sub" / "skip.txt").write_text("changed, excluded\n")
    (repo / "outside.txt").write_text("not in the pathspecs\n")
    dest = tmp_path / "out" / "tree.diff"
    result = run.capture_tree(dest, pathspecs=specs, repo=repo)
    assert result["dirty"] is True and result["error"] is None, result
    assert result["untracked"] == 1 and result["untracked_skipped"] == []
    data = dest.read_bytes()
    assert result["diff_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["diff_bytes"] == len(data) and result["diff"] == str(dest)
    assert b"skip.txt" not in data and b"outside.txt" not in data

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    _git(clone, "apply", "--check", str(dest))
    _git(clone, "apply", str(dest))
    for rel in ("sub/a.txt", "sub/b.txt", "sub/img.bin"):
        assert (clone / rel).read_bytes() == (repo / rel).read_bytes(), rel
    assert (clone / "sub" / "skip.txt").read_text() == "kept out\n"

    broken = run.capture_tree(
        tmp_path / "out" / "none.diff", pathspecs=specs, repo=tmp_path / "not-a-repo"
    )
    assert broken["dirty"] is None and broken["error"]
    assert not (tmp_path / "out" / "none.diff").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_capture_tree_skips_untracked_files_it_cannot_carry(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    monkeypatch.setattr(run, "TREE_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(run, "TREE_MAX_UNTRACKED", 1)
    (repo / "big.txt").write_text("x" * 100)
    (repo / "b1.txt").write_text("1\n")
    (repo / "b2.txt").write_text("2\n")
    result = run.capture_tree(tmp_path / "t.diff", pathspecs=["."], repo=repo)
    assert result["dirty"] is True and result["untracked"] == 1
    assert sorted(result["untracked_skipped"]) == ["b2.txt", "big.txt"]


@pytest.mark.asyncio
async def test_the_tree_diff_covers_the_harness_and_mirix_only(tmp_path):
    spec = CellSpec("alfworld", "luna", "mirix")
    cell = run.CellRunner(spec, run.RunOptions(out_dir=tmp_path / "c"))
    specs = cell._tree_pathspecs()
    assert specs[0] == "evals/skill_bench" and "mirix" in specs
    for excluded in ("tests", "runs", "data", "config/protocol_frozen.json"):
        assert f":(exclude)evals/skill_bench/{excluded}" in specs
    plain = run.CellRunner(SPEC, run.RunOptions(out_dir=tmp_path / "d"))
    assert "mirix" not in plain._tree_pathspecs()
    # A MIRIX cell's tree diff carries the launcher of its server.
    assert "scripts/start_server.py" in specs
    assert "scripts/start_server.py" not in plain._tree_pathspecs()
    # The checkout check runs on the real checkout only: a runner given its
    # own git record has none to check.
    assert cell.tree_check is run.unreproducible_files
    seamed = run.CellRunner(SPEC, run.RunOptions(out_dir=tmp_path / "e"), git=dict)
    assert seamed.tree_check is None


def test_hashed_repo_paths_name_every_file_a_version_hashes():
    """The fingerprinted files and the harness code (relative to
    evals/skill_bench) and the MIRIX code with its launcher (relative to mirix/), as
    paths from the repository's top."""
    hashed = run.hashed_repo_paths(
        {"methods/prompts/awm/a.json": "1", "config/models.json": "2"},
        {
            "harness": "h",
            "harness_sha256": {"agent/x.py": "3"},
            "mirix_sha256": {"agent/y.py": "4", protocol.MIRIX_LAUNCHER_KEY: "5"},
        },
    )
    assert hashed == {
        "evals/skill_bench/methods/prompts/awm/a.json",
        "evals/skill_bench/config/models.json",
        "evals/skill_bench/agent/x.py",
        "mirix/agent/y.py",
        "scripts/start_server.py",
    }
    assert run.hashed_repo_paths(None, {}) == set()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_unreproducible_files_names_hashed_files_commit_and_diff_cannot_rebuild(
    tmp_path,
):
    """A hashed file that is gitignored (git status and the tree diff never
    see it), or untracked and not in the tree diff (skipped, or no diff written), is
    one the commit plus tree.diff cannot rebuild; tracked files and captured untracked
    files are not, nor are files no version hashes."""
    repo = tmp_path / "repo"
    (repo / "sub" / "debug").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("debug\n*.log\n")
    (repo / "sub" / "tracked.txt").write_text("one\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "sub" / "tracked.txt").write_text("two\n")  # tracked: the diff has it
    (repo / "sub" / "debug" / "prompt.txt").write_text("ignored\n")
    (repo / "sub" / "run.log").write_text("ignored, not hashed\n")
    (repo / "sub" / "new.txt").write_text("untracked, captured\n")
    (repo / "sub" / "big.txt").write_text("untracked, skipped\n")
    (repo / "sub" / "other.txt").write_text("untracked, not hashed\n")
    hashed = {
        "sub/tracked.txt",
        "sub/debug/prompt.txt",
        "sub/new.txt",
        "sub/big.txt",
        "sub/gone.txt",  # hashed, then deleted: nothing to rebuild
    }
    tree = {"dirty": True, "diff": "x/tree.diff", "untracked_skipped": ["sub/big.txt"]}
    result = run.unreproducible_files(hashed, tree, pathspecs=["sub"], repo=repo)
    assert result["error"] is None and result["count"] == 2
    assert result["files"] == [
        {"path": "sub/big.txt", "why": "not_captured"},
        {"path": "sub/debug/prompt.txt", "why": "ignored"},
    ]
    # No diff written (a clean status, or git failed): every untracked file is lost.
    lost = run.unreproducible_files(
        hashed, {"dirty": False, "diff": None}, pathspecs=["sub"], repo=repo
    )
    assert [f["path"] for f in lost["files"]] == [
        "sub/big.txt",
        "sub/debug/prompt.txt",
        "sub/new.txt",
    ]
    # Outside the pathspecs a version's diff covers: not listed by git, so not checked.
    narrow = run.unreproducible_files(hashed, tree, pathspecs=["nope"], repo=repo)
    assert narrow["count"] == 0
    # The list is bounded; the count is not.
    many = {f"sub/debug/p{i}.txt" for i in range(5)}
    for name in many:
        (repo / name).write_text("x\n")
    capped = run.unreproducible_files(
        many, tree, pathspecs=["sub"], repo=repo, max_files=3
    )
    assert capped["count"] == 5 and len(capped["files"]) == 3
    # A failing git is recorded, not raised.
    broken = run.unreproducible_files(
        hashed, tree, pathspecs=["sub"], repo=tmp_path / "not-a-repo"
    )
    assert broken["files"] is None and broken["count"] is None and broken["error"]


@pytest.mark.asyncio
async def test_a_version_records_hashed_files_its_commit_and_diff_cannot_rebuild(
    tmp_path, fake, gateway
):
    """A tree whose only change is a gitignored (or uncaptured untracked) file the
    version hashes would look clean with no diff, although the fingerprint or code
    digest includes it. The segment names such files in
    ``tree.unreproducible`` and counts the tree as dirty (the audit's dirty_tree
    warning); a runner with no checkout to check records None (not checked)."""
    _gw, url = gateway
    seen: list[tuple[set[str], dict[str, Any], list[str]]] = []

    def check(hashed: Any, tree: Any, *, pathspecs: Any, **_kw: Any) -> dict:
        seen.append((set(hashed), dict(tree), list(pathspecs)))
        lost = [
            {"path": "evals/skill_bench/methods/prompts/debug/p.txt", "why": "ignored"}
        ]
        return {"files": lost, "count": 1, "error": None}

    cell = runner(tmp_path, url, factory=storing([]))
    cell.tree_check = check
    done = await cell.run()
    assert done.status == "complete"
    paths = cell_paths(tmp_path)
    (segment,) = read_json(paths.attempt(1).state)["version_segments"]
    tree = segment["tree"]
    assert tree["dirty"] is True and tree["diff"] is None
    assert tree["unreproducible"] == {
        "files": [
            {"path": "evals/skill_bench/methods/prompts/debug/p.txt", "why": "ignored"}
        ],
        "count": 1,
        "error": None,
    }
    assert read_json(paths.root / segment["version_file"])["tree"] == tree
    ((hashed, before, specs),) = seen
    assert "evals/skill_bench/config/models.json" in hashed
    assert "evals/skill_bench/protocol.py" in hashed
    assert before["dirty"] is False and specs == cell._tree_pathspecs()
    # The tests' runners never check the real checkout: recorded as not checked.
    other = CellSpec("alfworld", "luna", "awm")
    plain = await runner(
        tmp_path, url, spec=other, factory=storing([], name="awm")
    ).run()
    assert plain.status == "complete"
    (segment,) = read_json(cell_paths(tmp_path, other).attempt(1).state)[
        "version_segments"
    ]
    assert (
        segment["tree"]["dirty"] is False and segment["tree"]["unreproducible"] is None
    )


# --- refusals and degraded retrievals: recorded, not stopped --------------------------------

PAYLOAD_413 = Reply(
    413,
    {"error": {"code": 413, "message": "Request payload too large"}},
)
NO_TOOL_ENDPOINT = Reply(
    404,
    {
        "error": {
            "code": 404,
            "message": "No endpoints found that support tool use. Try disabling tools.",
        }
    },
)


def refusing(questions: set[str], reply: Reply):
    """The tiny responder, refusing the actor's calls for ``questions`` with ``reply``."""

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        messages = body.get("messages") or []
        content = messages[-1].get("content") if messages else None
        if endpoint == "chat/completions" and content in questions:
            return reply
        return tiny_responder(endpoint, body)

    return responder


@pytest.mark.asyncio
async def test_a_refused_learning_item_is_scored_and_the_cell_goes_on(tmp_path):
    """The provider refuses
    an actor call in the learning phase (a 413 payload refusal): the item ends
    unanswered, is scored 0 and flagged, one failures.jsonl row says so, and the learning
    phase goes on."""
    refused = train_order()[1]
    fake = FakeUpstream(responder=refusing({question(refused)}, PAYLOAD_413))
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    paths = cell_paths(tmp_path)
    rows = read_rows(paths.attempt(1).learning)
    assert [r["position"] for r in rows] == list(range(TINY_SIZES["train"]))
    row = rows[1]
    assert row["raw_id"] == refused and row["provider_unserved"] is True
    assert row["end_reason"] == "provider_unserved" and row["primary"] == 0.0
    assert row["score"]["primary"] == 0.0  # (TinyEnv's scorer reads no answer as 0)
    assert (row["failure"]["code"], row["failure"]["status"]) == (
        "provider_unserved",
        413,
    )
    assert all("provider_unserved" not in r for i, r in enumerate(rows) if i != 1)
    assert all("retrieval_degraded" not in r for r in rows)
    (failure,) = read_rows(paths.attempt(1).failures)
    assert (failure["phase"], failure["stage"], failure["code"]) == (
        "learning",
        "actor",
        "provider_unserved",
    )
    assert failure["recovery"] == "score_and_continue" and failure["status"] == 413
    assert failure["item"] == row["task_uid"] and failure["role"] == "actor"
    # The cell's own copy keeps the flag, and costs.json counts the refusal.
    assert read_rows(paths.learning)[1]["provider_unserved"] is True
    costs = read_json(paths.attempt(1).costs)
    assert costs["refusals"] == {"learning/actor": {"payload/413": 1}}


@pytest.mark.asyncio
async def test_three_refused_learning_items_in_a_row_stop_for_a_person(tmp_path):
    """The same refusal on REFUSAL_STREAK_ITEMS consecutive items stops the learning phase
    (HARNESS upstream_refusal_streak, restart_cell) after the last one's done checkpoint;
    --force-resume goes on with the next item."""
    order_ = train_order()
    fake = FakeUpstream(
        responder=refusing({question(r) for r in order_[1:4]}, NO_TOOL_ENDPOINT)
    )
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        stopped = await runner(tmp_path, url, factory=storing([])).run()
        assert stopped.status == "learning_failed", stopped.detail
        assert stopped.failure_kind == "harness"
        paths = cell_paths(tmp_path)
        state = read_json(paths.attempt(1).state)
        assert state["failure"]["code"] == "upstream_refusal_streak"
        assert state["failure_stage"] == "refusal_streak"
        assert state["failure_position"] == 3 and state["needs_decision"] == "harness"
        assert state["learning"]["checkpoint"] == "c0003-done"
        failures = read_rows(paths.attempt(1).failures)
        assert [f["code"] for f in failures] == [
            "provider_unserved",
            "provider_unserved",
            "provider_unserved",
            "upstream_refusal_streak",
        ]
        assert failures[-1]["recovery"] == "restart_cell"
        assert failures[-1]["kind"] == "harness"
        waiting = await runner(tmp_path, url, factory=storing([])).run()
        assert waiting.status == "refused" and "--force-resume" in waiting.detail
        events: list[Any] = []
        done = await runner(
            tmp_path, url, factory=storing(events), force_resume=True
        ).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    assert events[0] == ("setup", "learning", 1, RUN_A1, "c0003-done")
    rows = read_rows(paths.attempt(1).learning)
    assert [r["position"] for r in rows] == list(range(TINY_SIZES["train"]))
    assert [bool(r.get("provider_unserved")) for r in rows] == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]


@pytest.mark.asyncio
async def test_a_test_item_refused_no_endpoint_twice_is_scored_on_its_second_try(
    tmp_path,
):
    """An item whose actor call the provider refuses
    with "no endpoints" is re-run like any failed item; its second
    consecutive such try is scored 0 and flagged provider_unserved."""
    item = "test:02"
    fake = FakeUpstream(responder=refusing({question(item)}, NO_TOOL_ENDPOINT))
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        first = await runner(tmp_path, url, factory=storing([])).run()
        assert (first.status, first.pending) == ("test_incomplete", 1), first.detail
        paths = cell_paths(tmp_path)
        (unscored,) = [
            r for r in read_rows(paths.attempt(1).results) if r["raw_id"] == item
        ]
        assert unscored["score"] is None and unscored["recovery"] == "rerun_item"
        assert (unscored["failure"]["code"], unscored["failure"]["status"]) == (
            "upstream_rejected",
            404,
        )
        assert "provider_unserved" not in unscored
        done = await runner(tmp_path, url, factory=storing([])).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    tries = [r for r in read_rows(paths.attempt(1).results) if r["raw_id"] == item]
    assert len(tries) == 2
    scored = tries[1]
    assert scored["provider_unserved"] is True and scored["primary"] == 0.0
    assert scored["end_reason"] == "provider_unserved"
    assert (scored["failure"]["code"], scored["failure"]["status"]) == (
        "provider_unserved",
        404,
    )
    failures = read_rows(paths.attempt(1).failures)
    assert [(f["code"], f["recovery"]) for f in failures] == [
        ("upstream_rejected", "rerun_item"),
        ("provider_unserved", "score_and_continue"),
    ]
    state = read_json(paths.attempt(1).state)
    assert [r["raw_id"] for r in state["test_reruns"]] == [item]
    (final,) = [r for r in read_rows(paths.results) if r["raw_id"] == item]
    assert final["provider_unserved"] is True


class _DegradedRetrieval(StoringMethod):
    """Injects a retrieval it knows lost part of its input (as MIRIX does for a degraded
    retrieval) for the questions in ``degraded``."""

    def __init__(self, events: list[Any], degraded: set[str]) -> None:
        super().__init__(events)
        self.degraded = degraded

    async def before_task(self, task: Any) -> Any:
        from dataclasses import replace as replaced

        injection = await super().before_task(task)
        if task.instruction in self.degraded:
            return replaced(injection, degraded=True)
        return injection


@pytest.mark.asyncio
async def test_a_degraded_retrieval_flags_its_rows(tmp_path, fake, gateway):
    _gw, url = gateway
    learning_item, test_item = train_order()[0], "test:01"
    degraded = {question(learning_item), question(test_item)}
    done = await runner(
        tmp_path, url, factory=lambda: _DegradedRetrieval([], degraded)
    ).run()
    assert done.status == "complete", done.detail
    paths = cell_paths(tmp_path)
    flagged = [
        r["raw_id"]
        for rows in (read_rows(paths.attempt(1).learning), read_rows(paths.results))
        for r in rows
        if r.get("retrieval_degraded")
    ]
    assert flagged == [learning_item, test_item]
    assert all(
        r.get("retrieval_degraded") in (None, True)
        for r in read_rows(paths.attempt(1).results)
    )
    injections = read_rows(paths.attempt(1).injections)
    assert [r["raw_id"] for r in injections if r.get("degraded")] == [
        learning_item,
        test_item,
    ]
    assert all("degraded" not in r for r in injections if r["raw_id"] not in flagged)


@pytest.mark.asyncio
async def test_a_mirix_batch_lost_again_after_three_rollbacks_is_accepted(tmp_path):
    """A checkpoint's batch of MIRIX (here: embedding) calls lost to
    infrastructure is rolled back and replayed; once the same checkpoint was restored
    MIRIX_BATCH_ROLLBACKS times in a row, the calls its replay loses again are accepted
    (attempt.json accepted_learning_calls, one failures.jsonl row) and the learning phase
    goes on."""
    from evals.skill_bench.infra.failures import MIRIX_BATCH_ROLLBACKS

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        if endpoint == "embeddings" and body.get("input") == "FAIL-ME":
            return Reply(status=401, json={"error": {"message": "bad key"}})
        return tiny_responder(endpoint, body)

    fake = FakeUpstream(responder=responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        done = await runner(
            tmp_path,
            url,
            factory=lambda: _SwallowingMethod(events),
            max_learning_attempts=MIRIX_BATCH_ROLLBACKS + 2,
        ).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    assert events.count(("embedding", 401)) == MIRIX_BATCH_ROLLBACKS + 1
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert [r["checkpoint"] for r in state["resumes"]] == ["c0001-played"] * 3
    (accepted,) = state["accepted_learning_calls"]
    assert accepted["role"] == "embedding" and accepted["http_status"] == 401
    assert (accepted["checkpoint"], accepted["rollbacks"], accepted["position"]) == (
        "c0001-played",
        MIRIX_BATCH_ROLLBACKS,
        1,
    )
    for key in ("call_id", "error_type", "launch", "at"):
        assert accepted[key], key
    failures = read_rows(paths.attempt(1).failures)
    accepted_rows = [f for f in failures if f["recovery"] == "accept_failed_batch"]
    (row,) = accepted_rows
    assert (row["code"], row["kind"], row["stage"], row["phase"]) == (
        "mirix_learning",
        "infra",
        "mirix_learning",
        "learning",
    )
    assert (row["role"], row["status"]) == ("embedding", 401)
    assert row["detail"].startswith(
        f"1 lost call(s) accepted after {MIRIX_BATCH_ROLLBACKS} rollbacks of checkpoint "
        "c0001-played: "
    )
    assert accepted["call_id"] in row["detail"]
    # The three rollbacks before were ordinary restarts of the batch.
    assert [f["recovery"] for f in failures].count("resume_from_checkpoint") == 3


class _DegradedLearningRetrieval(StoringMethod):
    """Loses the query vector of its second learning item's retrieval (an embedding call
    the upstream refuses, which no retry recovers) and records that retrieval's window in
    ``mirix_retrievals.jsonl`` as MIRIX's method does. ``flag``: it injects the retrieval
    flagged degraded; otherwise it records the
    window without the ``degraded`` evidence and injects a clean-looking retrieval."""

    def __init__(self, events: list[Any], *, flag: bool = True) -> None:
        super().__init__(events)
        self.flag = flag

    async def before_task(self, task: Any) -> Any:
        from dataclasses import replace as replaced

        from evals.skill_bench.cell_records import now
        from evals.skill_bench.infra.processes import append_jsonl

        injection = await super().before_task(task)
        if self.frozen or self.learned != 1:
            return injection
        assert self.cell is not None
        started = now()
        async with httpx.AsyncClient(
            base_url=self.cell.gateway_base_url, trust_env=False, timeout=60
        ) as client:
            response = await client.post(
                f"/runs/{self.cell.run_id}/embedding/v1/embeddings",
                json={"model": protocol.EMBEDDING_MODEL, "input": "FAIL-ME"},
                headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
            )
        self.events.append(("embedding", response.status_code))
        problem = {"kind": "infra", "code": "mirix_retrieval", "role": "embedding"}
        record = {
            "event": "retrieval",
            "started_at": started,
            "at": now(),
            "task_uid": task.task_uid,
            "frozen": False,
            "retrieval_id": "retrieval-1",
            "failure": None,
            **({"degraded": problem} if self.flag else {}),
        }
        await asyncio.to_thread(
            append_jsonl, self.cell.work_dir / "mirix_retrievals.jsonl", record
        )
        return replaced(injection, degraded=self.flag)


def _refusing_embeddings(endpoint: str, body: dict[str, Any]) -> Any:
    if endpoint == "embeddings" and body.get("input") == "FAIL-ME":
        # The gateway's upstream key refused: INFRA, and not retried by the gateway.
        return Reply(status=401, json={"error": {"message": "bad key"}})
    return tiny_responder(endpoint, body)


@pytest.mark.asyncio
async def test_a_degraded_learning_retrieval_is_learned_on_not_rolled_back(tmp_path):
    """End to end: the calls a learning retrieval lost, inside the window of a
    retrieval the method injected flagged degraded, are recorded (attempt.json
    degraded_retrieval_calls, one failures.jsonl row counted and continued) and the
    learning phase goes on: no rollback, no replay, the item's row keeps its
    retrieval_degraded flag."""
    fake = FakeUpstream(responder=_refusing_embeddings)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        done = await runner(
            tmp_path, url, factory=lambda: _DegradedLearningRetrieval(events)
        ).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    assert events.count(("embedding", 401)) == 1  # never replayed
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert not state.get("resumes")
    assert not state.get("accepted_learning_calls")
    (call,) = state["degraded_retrieval_calls"]
    assert (call["role"], call["http_status"], call["retrieval_id"]) == (
        "embedding",
        401,
        "retrieval-1",
    )
    learning = read_rows(paths.attempt(1).learning)
    flagged = [r for r in learning if r.get("retrieval_degraded")]
    assert [r["position"] for r in flagged] == [1] == [call["position"]]
    assert call["task_uid"] == flagged[0]["task_uid"]
    for key in ("call_id", "error_type", "started_at", "ended_at", "launch", "at"):
        assert call[key], key
    (row,) = read_rows(paths.attempt(1).failures)
    assert (row["code"], row["kind"], row["recovery"], row["stage"]) == (
        "mirix_retrieval",
        "infra",
        "count_and_continue",
        "mirix_retrieval",
    )
    assert (row["role"], row["status"], row["item"]) == (
        "embedding",
        401,
        call["task_uid"],
    )
    assert call["call_id"] in row["detail"] and "retrieval-1" in row["detail"]


@pytest.mark.asyncio
async def test_a_lost_call_in_an_unflagged_retrieval_still_rolls_back(tmp_path):
    """Only a retrieval the method injected flagged degraded excuses its lost calls: the
    same window recorded without the degraded evidence (or injected unflagged) is an
    lost call, and the attempt resumes from the checkpoint before it."""
    fake = FakeUpstream(responder=_refusing_embeddings)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        failed = await runner(
            tmp_path,
            url,
            factory=lambda: _DegradedLearningRetrieval(events, flag=False),
        ).run()
    assert failed.status == "learning_interrupted", failed.detail
    state = read_json(cell_paths(tmp_path).attempt(1).state)
    assert state["failure"]["code"] == "mirix_learning", state["failure"]
    assert state["failure_stage"] == "learning_calls"
    assert not state.get("degraded_retrieval_calls")


def test_only_a_flagged_degraded_retrieval_with_a_window_excuses_lost_calls():
    """Pure: a learning retrieval excuses lost calls only when its record
    carries the degraded evidence, no failure and a readable window, and the item's
    learning injection is flagged degraded; a test-phase injection, an unflagged one, a
    failed retrieval or a window ending before it starts excuses nothing."""
    from evals.skill_bench.run import degraded_retrievals

    base = {
        "event": "retrieval",
        "frozen": False,
        "degraded": {"embedding": "401"},
        "failure": None,
        "task_uid": "u1",
        "retrieval_id": "r1",
        "started_at": "2026-09-30T10:00:00+00:00",
        "at": "2026-09-30T10:00:05+00:00",
    }
    flagged = [{"phase": "learning", "degraded": True, "task_uid": "u1", "position": 3}]
    (found,) = degraded_retrievals([base], flagged)
    assert found.position == 3
    assert found.holds({"ts": "2026-09-30T10:00:02+00:00"})
    assert not found.holds({"ts": "2026-09-30T10:00:06+00:00"})
    assert not found.holds({"ts": None})
    for record, injections in (
        (base, [{**flagged[0], "phase": "test"}]),
        (base, [{**flagged[0], "degraded": False}]),
        (base, []),
        ({**base, "failure": {"code": "x"}}, flagged),
        ({**base, "degraded": None}, flagged),
        ({**base, "frozen": True}, flagged),
        ({**base, "at": "2026-09-30T09:59:59+00:00"}, flagged),
        ({**base, "started_at": "not a time"}, flagged),
    ):
        assert degraded_retrievals([record], injections) == [], (record, injections)
