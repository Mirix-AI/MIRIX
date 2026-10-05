"""Learning-phase checkpoints and resume.

Every test runs the real ``run.py`` runner, rollout, actor and a real loopback gateway
over the fake upstream (``tests/fakes.py``, whose ``install_mark_route`` gives the
gateway ``POST /runs/<run_id>/mark`` where its checkout lacks it); nothing reaches a
provider. The benchmark is :class:`~evals.skill_bench.tests.run_support.TinyEnv` and the
method :class:`~evals.skill_bench.tests.run_support.StoringMethod` (one-shot faults), so
the memory an uninterrupted run freezes is known: every learning instruction, in the
seed-42 order, once.
"""

from __future__ import annotations

import asyncio
import errno
import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol, run
from evals.skill_bench.cell_records import (
    ATTEMPT_INVALID,
    ATTEMPT_LEARNING,
    CellPaths,
    read_json,
    read_ledger_lines,
    segment_calls,
    write_json_atomic,
)
from evals.skill_bench.infra.mirix_server import CellServerSpec, MirixServer
from evals.skill_bench.run import CellSpec
from evals.skill_bench.tests.fakes import (
    FakeUpstream,
    Reply,
    write_stub_server,
)
from evals.skill_bench.tests.run_support import (
    FIXED_GIT,
    StoringMethod,
    conformant_gateway,
    ledger,
    order,
    question,
    read_rows,
    resumed_cleanly,
    runner,
    tiny_responder,
)

SPEC = CellSpec("alfworld", "luna", "reasoningbank")
INSTANCE = "0000feed"
RUN_A1 = f"{SPEC.cell_id}-a01-{INSTANCE}"
RUN_A2 = f"{SPEC.cell_id}-a02-{INSTANCE}"


@pytest.fixture(autouse=True)
def _fixed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "new_instance_id", lambda: INSTANCE)


@pytest.fixture(autouse=True)
def _frozen_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This test's own copy of the frozen record, frozen to this tree: run.py freezes
    automatically and must never write the repository's."""
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


def storing(events: list[Any], **kw: Any) -> Callable[[], StoringMethod]:
    return lambda: StoringMethod(events, **kw)


def cell_paths(tmp_path: Path, spec: CellSpec = SPEC) -> CellPaths:
    return CellPaths(tmp_path / "runs" / spec.cell_id)


def resumed_from(events: list[Any]) -> str | None:
    """The checkpoint the learning setup of an invocation restored (None: none)."""
    setups = [e for e in events if e[0] == "setup" and e[1] == "learning"]
    assert len(setups) == 1, setups
    return setups[0][4] if len(setups[0]) > 4 else None


def assert_resumed_cleanly(
    tmp_path: Path, paths: CellPaths, run_id: str = RUN_A1, *, storing: bool = True
) -> None:
    """:func:`~evals.skill_bench.tests.run_support.resumed_cleanly` for a run of this
    module's gateway (``tmp_path/runs``)."""
    resumed_cleanly(
        tmp_path / "runs" / run_id / "gateway.jsonl", paths, storing=storing
    )


# --- where an interruption lands --------------------------------------------------------


def _fail_checkpoint_once(
    monkeypatch: pytest.MonkeyPatch, kind: str, position: int, error: BaseException
) -> None:
    real = run.CellRunner._write_checkpoint
    fired = {"done": False}

    async def failing(self: Any, attempt: Any, k: str, p: int, **kw: Any) -> Any:
        if (k, p) == (kind, position) and not fired["done"]:
            fired["done"] = True
            raise error
        return await real(self, attempt, k, p, **kw)

    monkeypatch.setattr(run.CellRunner, "_write_checkpoint", failing)


def _fail_capture_once(
    monkeypatch: pytest.MonkeyPatch, kind: str, position: int, error: BaseException
) -> None:
    real = run.CellRunner._capture_checkpoint
    fired = {"done": False}

    async def failing(self: Any, attempt: Any, k: str, p: int, **kw: Any) -> Any:
        if (k, p) == (kind, position) and not fired["done"]:
            fired["done"] = True
            raise error
        return await real(self, attempt, k, p, **kw)

    monkeypatch.setattr(run.CellRunner, "_capture_checkpoint", failing)


CASES = {
    # stage: (resumed checkpoint, discarded learning rows, failure stage, code)
    "rollout": ("c0001-done", 1, "before_task", "gateway_unreachable"),
    "played": ("c0001-done", 1, "played", "gateway_unreachable"),
    "after_task": ("c0002-played", 0, "after_task", "gateway_unreachable"),
    "wait_until_idle": ("c0002-played", 0, "wait_until_idle", "gateway_unreachable"),
    "writing_done": ("c0002-played", 0, "checkpoint", "disk_full"),
    "freeze": ("c0005-done", 0, "finalize", "gateway_unreachable"),
    "setup": (None, 0, "cell", "gateway_unreachable"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", list(CASES))
async def test_an_infra_failure_resumes_from_the_right_checkpoint(
    tmp_path, fake, gateway, monkeypatch, stage
):
    """Design 1.6: per stage of item k=2 (or the freeze, or setup), the attempt ends
    learning_interrupted (exit 8), and the next invocation resumes from the checkpoint the
    table names, redoes only what came after it, and freezes the memory an uninterrupted
    run freezes."""
    _gw, url = gateway
    checkpoint, moved_rows, failure_stage, code = CASES[stage]
    k = question(order()[2])
    faults: dict[str, set[str]] = {}
    if stage == "rollout":
        faults = {"before_task": {k}}
    elif stage == "after_task":
        faults = {"after_task": {k}}
    elif stage == "wait_until_idle":
        faults = {"wait_until_idle": {k}}
    elif stage == "freeze":
        faults = {"finalize": {"*"}}
    elif stage == "played":
        _fail_checkpoint_once(monkeypatch, "played", 2, httpx.ConnectError("gone"))
    elif stage == "writing_done":
        _fail_capture_once(
            monkeypatch, "done", 2, OSError(errno.ENOSPC, "No space left on device")
        )
    elif stage == "setup":
        _fail_checkpoint_once(monkeypatch, "init", 0, httpx.ConnectError("gone"))
    events: list[Any] = []
    failed = await runner(
        tmp_path, url, factory=storing(events, fail_once=faults)
    ).run()
    assert (failed.status, failed.exit_code) == ("learning_interrupted", 8), (
        failed.detail
    )
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert isinstance(state, dict)
    assert state["state"] == "learning_interrupted" and "abandoned" not in state
    assert (state["failure_stage"], state["failure"]["code"]) == (failure_stage, code)
    assert state["failure"]["kind"] == "infra"

    events.clear()
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert (done.status, done.attempt) == ("complete", 1), done.detail
    assert resumed_from(events) == checkpoint
    segment = read_rows(paths.attempt(1).discarded_log)[0]
    assert segment["checkpoint"]["id"] == checkpoint
    assert segment["moved"]["learning_rows"] == moved_rows
    assert segment["failure_stage"] == failure_stage
    if stage == "setup":
        assert segment["in_flight"]["stage"] == "cell"
        assert segment["restore"]["snapshot_sha256_expected"] is None
    else:
        assert (
            segment["restore"]["snapshot_sha256_actual"]
            == (segment["restore"]["snapshot_sha256_expected"])
        )
    assert_resumed_cleanly(tmp_path, paths)
    if moved_rows:
        # The moved rows point at their moved trajectories.
        moved = paths.attempt(1).discarded_dir(1)
        rows = read_rows(moved / "learning_rows.jsonl")
        assert [r["position"] for r in rows] == [2]
        assert all((paths.root / r["trajectory_discarded"]).is_file() for r in rows)


@pytest.mark.asyncio
async def test_the_rows_are_recorded_before_after_task_and_a_later_failure_adds_none(
    tmp_path, fake, gateway, monkeypatch
):
    """An item is persisted as soon as it is scored (on_played),
    before the method learns from it; a failure after that writes no second row."""
    _gw, url = gateway
    events: list[Any] = []
    real = run.CellRunner._record

    async def record(
        self: Any, attempt: Any, phase: str, position: int, *a: Any, **kw: Any
    ) -> None:
        events.append(("record", phase, position))
        await real(self, attempt, phase, position, *a, **kw)

    monkeypatch.setattr(run.CellRunner, "_record", record)
    k = question(order()[1])
    failed = await runner(
        tmp_path, url, factory=storing(events, fail_once={"after_task": {k}})
    ).run()
    assert failed.status == "learning_interrupted"
    names = [(e[0], e[1] if len(e) > 1 else None) for e in events]
    for position, raw_id in enumerate(order()[:2]):
        assert names.index(("record", "learning")) < names.index(
            ("after_task", question(raw_id))
        )
        recorded = [
            i for i, e in enumerate(events) if e[:3] == ("record", "learning", position)
        ]
        after = [
            i
            for i, e in enumerate(events)
            if e[0] == "after_task" and e[1] == question(raw_id)
        ]
        assert len(recorded) == 1 and recorded[0] < after[0]
    rows = read_rows(cell_paths(tmp_path).attempt(1).learning)
    assert [(r["position"], r["score"] is not None) for r in rows] == [
        (0, True),
        (1, True),
    ]
    assert rows[1]["timings"]["played_s"] >= 0


@pytest.mark.asyncio
async def test_a_lost_call_before_the_played_checkpoint_resumes_from_the_item_before(
    tmp_path,
):
    """A lost call at item k before its played checkpoint (an embedding call in
    before_task): the checkpoint's own ledger check catches it, and ``resume_from`` is
    item k-1's done checkpoint, so item k is redone from its start."""
    failing = {"on": True}

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        if (
            endpoint == "embeddings"
            and body.get("input") == "FAIL-ME"
            and failing["on"]
        ):
            return Reply(status=401, json={"error": {"message": "bad key"}})
        return tiny_responder(endpoint, body)

    class LosesACall(StoringMethod):
        async def before_task(self, task: Any) -> Any:
            if not self.frozen and self.learned == 2:
                assert self.cell is not None
                async with httpx.AsyncClient(
                    base_url=self.cell.gateway_base_url, trust_env=False, timeout=60
                ) as client:
                    response = await client.post(
                        f"/runs/{self.cell.run_id}/embedding/v1/embeddings",
                        json={"model": protocol.EMBEDDING_MODEL, "input": "FAIL-ME"},
                        headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                    )
                self.events.append(("embedding", response.status_code))
            return await super().before_task(task)

    fake = FakeUpstream(responder=responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        failed = await runner(tmp_path, url, factory=lambda: LosesACall(events)).run()
        assert failed.status == "learning_interrupted"
        paths = cell_paths(tmp_path)
        state = read_json(paths.attempt(1).state)
        assert state["failure"]["code"] == "mirix_learning"
        assert state["failure_stage"] == "learning_calls"
        assert state["resume_from"] == "c0001-done"
        failing["on"] = False
        events.clear()
        done = await runner(tmp_path, url, factory=lambda: LosesACall(events)).run()
    assert done.status == "complete", done.detail
    assert resumed_from(events) == "c0001-done"
    assert ("embedding", 200) in events
    assert_resumed_cleanly(tmp_path, paths)


# --- decisions for a person --------------------------------------------------------------


async def _interrupted(tmp_path: Path, url: str, at: int = 2) -> CellPaths:
    k = question(order()[at])
    failed = await runner(
        tmp_path, url, factory=storing([], fail_once={"after_task": {k}})
    ).run()
    assert failed.status == "learning_interrupted"
    return cell_paths(tmp_path)


@pytest.mark.asyncio
async def test_a_resume_under_other_code_is_recorded_while_other_hosts_and_formats_are_refused(
    tmp_path, fake, gateway, monkeypatch
):
    """A learning resume under other code goes on and records a version segment.
    Another host (the checkpoints, database and HOME are local) and another checkpoint
    format stay refused."""
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    state = read_json(paths.attempt(1).state)
    # Checkpoints another run.py wrote in another layout.
    write_json_atomic(paths.attempt(1).state, {**state, "checkpoint_format": 0})
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused"
    assert "checkpoints are in format 0" in refused.detail
    assert f"check out commit {FIXED_GIT['commit']}" in refused.detail
    write_json_atomic(paths.attempt(1).state, state)

    here = run.host_info()
    # Another machine id and name: same_host compares the machine ids (with the system and
    # architecture) when both records have one, so another hostname alone is not enough.
    other = {**here, "hostname": "another-host", "machine_id": "d" * 32}
    monkeypatch.setattr(run, "host_info", lambda: other)
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused" and "another-host" in refused.detail
    other = {**here, "machine_id": "e" * 32}  # another machine under the same name
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused" and "id eeeeeeee" in refused.detail
    assert read_json(paths.attempt(1).state) == state  # refusals change nothing
    # Nothing was changed by the refusals: the resume goes through, also after this
    # machine's hostname changed (the machine id names the host), and under other
    # code, recorded as a new version segment.
    other = {**here, "hostname": "my-laptop.local"}
    # The attempt learned under other code (its start and its version segment say so).
    old_code = {**state["code"], "harness": "0" * 64}
    segments = [{**state["version_segments"][0], "code": old_code}]
    changed = {**state, "code": old_code, "version_segments": segments}
    write_json_atomic(paths.attempt(1).state, changed)
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete" and resumed_from(events) == "c0002-played"
    segments = read_json(paths.attempt(1).state)["version_segments"]
    assert len(segments) == 2 and segments[0]["code"] == old_code
    assert segments[-1]["changed"]["code"] == ["harness"]
    assert segments[-1]["changed"]["protocol"] is False
    assert segments[-1]["learning_from"] == {
        "position": 2,
        "stage": "after_task",
        "checkpoint": "c0002-played",
    }
    assert read_json(paths.attempt(1).state)["versions_mixed"]["code"] is True


@pytest.mark.asyncio
async def test_an_abandoned_run_is_not_resumed(tmp_path, fake, gateway):
    _gw, url = gateway
    await _interrupted(tmp_path, url)
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.post(
            f"/runs/{RUN_A1}/abandon",
            json={"reason": "operator"},
            headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
        )
        response.raise_for_status()
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused" and "abandoned" in refused.detail


@pytest.mark.asyncio
async def test_an_attempt_without_checkpoints_is_restarted(tmp_path, fake, gateway):
    """An attempt of an older harness (no ``checkpoints/``) keeps the H8 rule."""
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    shutil.rmtree(paths.attempt(1).checkpoints)
    state = read_json(paths.attempt(1).state)
    state["state"] = ATTEMPT_LEARNING
    write_json_atomic(paths.attempt(1).state, state)
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert (done.status, done.attempt) == ("complete", 2)
    first = read_json(paths.attempt(1).state)
    assert first["state"] == ATTEMPT_INVALID and first["abandoned"] is True
    assert "no learning checkpoints" in first["invalid_reason"]
    assert first["invalidation"] == "learning_failure"
    assert events[0] == ("setup", "learning", 2, RUN_A2)
    (segment,) = read_json(paths.attempt(2).state)["version_segments"]
    assert segment["why"] == "restart of attempt 1 (no learning checkpoints)"


@pytest.mark.asyncio
async def test_a_cell_without_learning_items_restarts_as_a_new_attempt(
    tmp_path, fake, gateway
):
    """The no-memory method (no learning items) writes no checkpoints: a failed
    freeze stops the attempt as any learning failure does (learning_interrupted for
    infrastructure, not invalidated on the spot), and the next invocation restarts it as
    a new attempt, since it has nothing to resume."""
    _gw, url = gateway
    spec = CellSpec("alfworld", "luna", "none")
    events: list[Any] = []
    failed = await runner(
        tmp_path,
        url,
        spec=spec,
        factory=storing(events, name="none", fail_once={"finalize": {"*"}}),
    ).run()
    assert failed.status == "learning_interrupted" and failed.failure_kind == "infra"
    assert "restarts it as a new attempt" in failed.detail
    paths = cell_paths(tmp_path, spec)
    assert not paths.attempt(1).checkpoints.exists()
    state = read_json(paths.attempt(1).state)
    assert state["state"] == "learning_interrupted" and not state.get("abandoned")
    assert state["needs_decision"] is None
    failures = read_rows(paths.attempt(1).failures)
    assert failures[-1]["recovery"] == "restart_cell"
    done = await runner(
        tmp_path, url, spec=spec, factory=storing([], name="none")
    ).run()
    assert (done.status, done.attempt) == ("complete", 2)
    first = read_json(paths.attempt(1).state)
    assert first["state"] == ATTEMPT_INVALID and first["abandoned"] is True
    assert first["invalidation"] == "learning_failure"
    assert "no learning checkpoints" in first["invalid_reason"]
    assert first["failure"]["kind"] == "infra"


@pytest.mark.asyncio
async def test_a_harness_failure_of_a_cell_without_learning_items_waits_for_a_person(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    spec = CellSpec("alfworld", "luna", "none")
    failed = await runner(
        tmp_path,
        url,
        spec=spec,
        factory=storing([], name="none", harness_once={"finalize": {"*"}}),
    ).run()
    assert failed.status == "learning_failed" and failed.failure_kind == "harness"
    assert "restarts as a new attempt" in failed.detail
    paths = cell_paths(tmp_path, spec)
    state = read_json(paths.attempt(1).state)
    assert state["needs_decision"] == "harness"
    refused = await runner(
        tmp_path, url, spec=spec, factory=storing([], name="none")
    ).run()
    assert refused.status == "refused" and "--force-resume" in refused.detail
    assert paths.attempt_numbers() == [1]
    done = await runner(
        tmp_path, url, spec=spec, factory=storing([], name="none"), force_resume=True
    ).run()
    assert (done.status, done.attempt) == ("complete", 2), done.detail
    assert read_json(paths.attempt(1).state)["invalidation"] == "learning_failure"


# --- corrupt checkpoints -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_corrupt_latest_falls_back_to_the_previous_checkpoint(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    attempt = paths.attempt(1)
    assert read_json(attempt.latest_checkpoint)["id"] == "c0002-played"
    (attempt.checkpoint("c0002-played") / "snapshot.json").write_text("{}")
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete", done.detail
    assert resumed_from(events) == "c0001-done"
    # Item 2 was redone from its start; its first try's row was moved aside.
    learned = [e[2] for e in events if e[0] == "before_task" and e[1] == "learning"]
    assert learned[0] == question(order()[2])
    segment = read_rows(attempt.discarded_log)[0]
    assert segment["checkpoint"]["id"] == "c0001-done"
    assert segment["moved"]["learning_rows"] == 1
    assert_resumed_cleanly(tmp_path, paths)


@pytest.mark.asyncio
async def test_no_verifiable_checkpoint_is_a_harness_stop(tmp_path, fake, gateway):
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    attempt = paths.attempt(1)
    for child in attempt.checkpoints.iterdir():
        if child.is_dir():
            (child / "snapshot.json").write_text("{}")
    failed = await runner(tmp_path, url, factory=storing([])).run()
    assert (failed.status, failed.failure_kind) == ("learning_failed", "harness")
    state = read_json(attempt.state)
    assert state["failure"]["code"] == "checkpoint_corrupt"
    assert state["needs_decision"] == "harness"
    # Nothing was marked or moved: the resume stopped before its ledger marker.
    assert not [r for r in ledger(tmp_path, RUN_A1) if r.get("event") == "resume"]
    assert not attempt.discarded_log.exists()


@pytest.mark.asyncio
async def test_a_restore_that_does_not_reproduce_the_checkpoint_is_a_harness_stop(
    tmp_path, fake, gateway
):
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    failed = await runner(
        tmp_path, url, factory=storing([], tamper_checkpoint=True)
    ).run()
    assert (failed.status, failed.failure_kind) == ("learning_failed", "harness")
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == "checkpoint_mismatch"
    assert state["needs_decision"] == "harness"
    # A person looks, then forces the resume with a method that restores correctly.
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events), force_resume=True).run()
    assert done.status == "complete", done.detail
    discarded = read_rows(paths.attempt(1).discarded_log)
    assert [r["segment"] for r in discarded] == [1, 2]
    assert discarded[1]["operator_resume"] is True
    assert_resumed_cleanly(tmp_path, paths)


# --- repeated and interrupted resumes -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_resume_killed_after_its_marker_is_finished_by_the_next(
    tmp_path, fake, gateway, monkeypatch
):
    """A resume that dies between its ledger marker and its discarded row leaves no row;
    the next resume takes the same segment number and records it once."""
    _gw, url = gateway
    paths = await _interrupted(tmp_path, url)
    real = run.CellRunner._restore_checkpoint

    async def killed(self: Any, *a: Any, **kw: Any) -> Any:
        raise asyncio.CancelledError()

    monkeypatch.setattr(run.CellRunner, "_restore_checkpoint", killed)
    with pytest.raises(asyncio.CancelledError):
        await runner(tmp_path, url, factory=storing([])).run()
    attempt = paths.attempt(1)
    assert not attempt.discarded_log.exists()
    assert attempt.discarded_dir(1).is_dir()  # the move aside was sealed
    assert read_json(attempt.state)["state"] == "learning_interrupted"
    monkeypatch.setattr(run.CellRunner, "_restore_checkpoint", real)
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete", done.detail
    marks = [r for r in ledger(tmp_path, RUN_A1) if r.get("event") == "resume"]
    assert [m["segment"] for m in marks] == [1, 1]
    assert [r["segment"] for r in read_rows(attempt.discarded_log)] == [1]
    assert_resumed_cleanly(tmp_path, paths)


@pytest.mark.asyncio
async def test_a_resume_killed_while_moving_aside_is_finished_by_the_next(
    tmp_path, fake, gateway, monkeypatch
):
    _gw, url = gateway
    k = question(order()[2])
    failed = await runner(
        tmp_path, url, factory=storing([], fail_once={"before_task": {k}})
    ).run()
    assert failed.status == "learning_interrupted"
    paths = cell_paths(tmp_path)
    attempt = paths.attempt(1)
    real = run._truncate_jsonl

    def killed(*a: Any, **kw: Any) -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(run, "_truncate_jsonl", killed)
    with pytest.raises(asyncio.CancelledError):
        await runner(tmp_path, url, factory=storing([])).run()
    partial = attempt.discarded_dir(1).with_name("seg-01.partial")
    assert partial.is_dir() and (partial / "method").is_dir()
    assert not attempt.method_dir.exists()
    monkeypatch.setattr(run, "_truncate_jsonl", real)
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete", done.detail
    assert not partial.exists() and (attempt.discarded_dir(1) / "method").is_dir()
    segment = read_rows(attempt.discarded_log)[0]
    assert segment["moved"]["learning_rows"] == 1  # the failed try's row
    assert_resumed_cleanly(tmp_path, paths)


@pytest.mark.asyncio
async def test_two_interruptions_make_two_segments(tmp_path, fake, gateway):
    _gw, url = gateway
    o = order()
    first = await runner(
        tmp_path, url, factory=storing([], fail_once={"after_task": {question(o[1])}})
    ).run()
    assert first.status == "learning_interrupted"
    second = await runner(
        tmp_path, url, factory=storing([], fail_once={"before_task": {question(o[4])}})
    ).run()
    assert second.status == "learning_interrupted"
    done = await runner(tmp_path, url, factory=storing([])).run()
    assert done.status == "complete", done.detail
    paths = cell_paths(tmp_path)
    discarded = read_rows(paths.attempt(1).discarded_log)
    assert [(r["segment"], r["checkpoint"]["id"]) for r in discarded] == [
        (1, "c0001-played"),
        (2, "c0003-done"),
    ]
    assert_resumed_cleanly(tmp_path, paths)
    costs = read_json(paths.attempt(1).costs)
    assert costs["discarded"]["calls"] == sum(len(r["calls"]) for r in discarded)


# --- quiescence -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_started_during_a_checkpoint_is_not_quiescent(
    tmp_path, fake, gateway, monkeypatch
):
    """A call that starts after the checkpoint's marker, while its capture reads the
    state, may leave part of its effect in the capture: the checkpoint is refused
    (``checkpoint_not_quiescent``, INFRA) and the resume goes back to the one before."""
    _gw, url = gateway
    real = run._copy_method_dir
    fired = {"done": False}

    def copy_and_call(src: Path, dest: Path, **kw: Any) -> Any:
        if dest.parent.name == "c0002-done.partial" and not fired["done"]:
            fired["done"] = True
            httpx.post(
                f"{url}/runs/{RUN_A1}/actor/v1/chat/completions",
                json={
                    "model": SPEC.model,
                    "messages": [{"role": "user", "content": "x"}],
                },
                headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                timeout=30,
                trust_env=False,
            )
        return real(src, dest, **kw)

    monkeypatch.setattr(run, "_copy_method_dir", copy_and_call)
    events: list[Any] = []
    failed = await runner(tmp_path, url, factory=storing(events)).run()
    assert failed.status == "learning_interrupted", failed.detail
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == "checkpoint_not_quiescent"
    assert state["failure_stage"] == "checkpoint"
    assert not paths.attempt(1).checkpoint("c0002-done.partial").exists()
    events.clear()
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete", done.detail
    assert resumed_from(events) == "c0002-played"
    assert_resumed_cleanly(tmp_path, paths)


@pytest.mark.asyncio
async def test_a_call_started_before_the_marker_while_the_state_is_read(
    tmp_path, fake, gateway
):
    """A call that starts after the run was found quiescent but before the
    checkpoint's marker (here while the method's state is read) may have changed the
    state the capture holds: the checkpoint is refused as well."""
    _gw, url = gateway
    fired = {"done": False}

    class CallsWhileCheckpointed(StoringMethod):
        async def checkpoint(self) -> dict[str, Any]:
            state = await super().checkpoint()
            if self.learned == 2 and not fired["done"]:
                fired["done"] = True
                assert self.cell is not None
                async with httpx.AsyncClient(
                    base_url=self.cell.gateway_base_url, trust_env=False, timeout=30
                ) as client:
                    await client.post(
                        f"/runs/{self.cell.run_id}/learner/v1/chat/completions",
                        json={
                            "model": SPEC.model,
                            "messages": [{"role": "user", "content": "background"}],
                        },
                        headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                    )
            return state

    events: list[Any] = []
    failed = await runner(
        tmp_path, url, factory=lambda: CallsWhileCheckpointed(events)
    ).run()
    assert failed.status == "learning_interrupted", failed.detail
    paths = cell_paths(tmp_path)
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == "checkpoint_not_quiescent"
    assert "started during the capture" in state["failure"]["detail"]
    # Item 1's done checkpoint (learned == 2) was refused; its played one is LATEST.
    assert state["learning"]["checkpoint"] == "c0001-played"
    events.clear()
    done = await runner(
        tmp_path, url, factory=lambda: CallsWhileCheckpointed(events)
    ).run()
    assert done.status == "complete", done.detail
    assert resumed_from(events) == "c0001-played"
    (segment,) = read_rows(paths.attempt(1).discarded_log)
    assert [c["role"] for c in segment["calls"]] == ["learner"]
    assert_resumed_cleanly(tmp_path, paths)


@pytest.mark.asyncio
async def test_a_mirix_write_between_the_snapshot_and_the_dump(
    tmp_path, fake, gateway, monkeypatch
):
    """MIRIX writes to its database without a gateway call (its sweeper)
    after the snapshot the marker records, before the dump: the memory no longer reads
    the same after the dump, so the checkpoint is refused (INFRA) instead of leaving a
    dump that a later resume would find does not rebuild its snapshot."""
    _gw, url = gateway
    stub = write_stub_server(tmp_path)
    spec = CellSpec("alfworld", "luna", "mirix")
    from evals.skill_bench.infra import mirix_server as ms

    def server(s: CellServerSpec, mode: Any) -> MirixServer:
        return MirixServer(
            s,
            mode=mode,
            command=[__import__("sys").executable, str(stub), "--port", str(s.port)],
            database=False,
        )

    methods: list[StoringMethod] = []

    def factory() -> StoringMethod:
        methods.append(StoringMethod([], name="mirix"))
        return methods[-1]

    real = ms.NoDatabase.checkpoint
    wrote = {"done": False}

    async def sweeper_writes(self: Any, out_dir: Path) -> Any:
        if out_dir.name == "c0001-done.partial" and not wrote["done"]:
            wrote["done"] = True
            methods[-1].memory.append("a sweeper's write")
        return await real(self, out_dir)

    monkeypatch.setattr(ms.NoDatabase, "checkpoint", sweeper_writes)

    def make() -> Any:
        cell = runner(tmp_path, url, spec=spec, factory=factory)
        cell.mirix_server_factory = server
        return cell

    failed = await make().run()
    assert failed.status == "learning_interrupted", failed.detail
    paths = cell_paths(tmp_path, spec)
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == "checkpoint_not_quiescent"
    assert "changed while it was captured" in state["failure"]["detail"]
    assert not paths.attempt(1).checkpoint("c0001-done.partial").exists()
    assert state["learning"]["checkpoint"] == "c0001-played"
    done = await make().run()
    assert done.status == "complete", done.detail
    frozen = read_json(paths.attempt(1).frozen_snapshot)["snapshot"]
    assert "a sweeper's write" not in frozen["memory"]


@pytest.mark.asyncio
async def test_a_call_still_in_flight_blocks_the_checkpoint(tmp_path, monkeypatch):
    """A method call still in flight when a checkpoint is due (here one after_task left
    running) makes the checkpoint wait, and after the bound fail
    ``checkpoint_not_quiescent``; the resume discards that call too."""
    monkeypatch.setattr(run, "CHECKPOINT_QUIESCE_S", 0.3)
    release = threading.Event()

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        messages = body.get("messages") or []
        if messages and messages[-1].get("content") == "LINGER":
            return Reply(delay_s=2.0)
        return tiny_responder(endpoint, body)

    tasks: list[asyncio.Task[Any]] = []

    class Lingers(StoringMethod):
        async def after_task(self, trajectory: Any) -> None:
            await super().after_task(trajectory)
            if self.learned == 2 and not release.is_set():
                release.set()
                assert self.cell is not None
                cell = self.cell

                async def linger() -> None:
                    async with httpx.AsyncClient(
                        base_url=cell.gateway_base_url, trust_env=False, timeout=60
                    ) as client:
                        await client.post(
                            f"/runs/{cell.run_id}/learner/v1/chat/completions",
                            json={
                                "model": SPEC.model,
                                "messages": [{"role": "user", "content": "LINGER"}],
                            },
                            headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                        )

                tasks.append(asyncio.ensure_future(linger()))
                # after_task returns once the call reached the upstream (so its
                # call_started line is in the ledger), leaving it in flight.
                while not any(
                    (r.json.get("messages") or [{}])[-1].get("content") == "LINGER"
                    for r in fake.calls("chat/completions")
                ):
                    await asyncio.sleep(0.01)

    fake = FakeUpstream(responder=responder)
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        failed = await runner(tmp_path, url, factory=lambda: Lingers([])).run()
        assert failed.status == "learning_interrupted", failed.detail
        paths = cell_paths(tmp_path)
        state = read_json(paths.attempt(1).state)
        assert state["failure"]["code"] == "checkpoint_not_quiescent"
        await asyncio.gather(*tasks, return_exceptions=True)
        done = await runner(tmp_path, url, factory=lambda: Lingers([])).run()
    assert done.status == "complete", done.detail
    segment = read_rows(paths.attempt(1).discarded_log)[0]
    assert segment["checkpoint"]["id"] == "c0001-played"
    assert [c["role"] for c in segment["calls"]] == ["learner"]


# --- signals ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sigterm_during_a_checkpoint_lets_it_finish(
    tmp_path, fake, gateway, monkeypatch
):
    """An interruption while a checkpoint is being written waits for it: LATEST names a
    complete checkpoint, never a half-written one."""
    _gw, url = gateway
    entered = threading.Event()
    real = run._copy_method_dir

    def slow_copy(src: Path, dest: Path, **kw: Any) -> Any:
        if dest.parent.name == "c0001-played.partial":
            entered.set()
            import time

            time.sleep(0.5)
        return real(src, dest, **kw)

    monkeypatch.setattr(run, "_copy_method_dir", slow_copy)
    cell = asyncio.ensure_future(runner(tmp_path, url, factory=storing([])).run())
    await asyncio.to_thread(entered.wait, 30)
    cell.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cell
    paths = cell_paths(tmp_path)
    attempt = paths.attempt(1)
    assert read_json(attempt.latest_checkpoint)["id"] == "c0001-played"
    assert (attempt.checkpoint("c0001-played") / "checkpoint.json").is_file()
    state = read_json(attempt.state)
    assert state["state"] == "learning_interrupted"
    assert read_json(paths.status)["status"] == "interrupted"
    events: list[Any] = []
    done = await runner(tmp_path, url, factory=storing(events)).run()
    assert done.status == "complete" and resumed_from(events) == "c0001-played"
    assert_resumed_cleanly(tmp_path, paths)


# --- other ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_merged_runs_directory_is_refused(tmp_path, fake, gateway):
    _gw, url = gateway
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / "_merged.json").write_text("{}")
    refused = await runner(tmp_path, url, factory=storing([])).run()
    assert refused.status == "refused" and "_merged.json" in refused.detail
    assert not cell_paths(tmp_path).root.exists()


def test_the_cli_takes_the_resume_flags(tmp_path):
    _spec, options, _args = run.parse_args(
        [
            *("--benchmark", "alfworld", "--model", "luna", "--method", "awm"),
            *("--out", str(tmp_path / "c"), "--gateway-url", "http://127.0.0.1:1"),
            "--force-resume",
            "--keep-discarded-db",
            "--allow-code-change",  # a no-op, still accepted
            *("--new-attempt", "--note", "rerun with the fixed MIRIX curator"),
        ]
    )
    assert options.force_resume and options.keep_discarded_db
    assert options.to_dict()["force_resume"] is True
    assert options.allow_code_change and options.new_attempt
    assert options.note == "rerun with the fixed MIRIX curator"
    assert options.to_dict()["note"] == options.note
    assert run.EXIT_CODES["learning_interrupted"] == 8
    base = [
        *("--benchmark", "alfworld", "--model", "luna", "--method", "awm"),
        *("--out", str(tmp_path / "c"), "--gateway-url", "http://127.0.0.1:1"),
    ]
    for bad in ("two\nlines", "", "x" * (run.MAX_NOTE_CHARS + 1)):
        with pytest.raises(SystemExit) as exc:
            run.parse_args([*base, "--note", bad])
        assert exc.value.code == 2
    _spec, plain, _args = run.parse_args(base)
    assert plain.note is None


def test_a_mirix_learning_resume_reuses_its_gateway_port(tmp_path):
    """Learning resumes: a launched gateway takes the port the MIRIX agent calls."""
    spec = CellSpec("alfworld", "luna", "mirix")
    options = run.RunOptions(out_dir=tmp_path / "c", gateway_runs_dir=tmp_path / "runs")
    cell = run.CellRunner(spec, options)
    attempt = cell.paths.attempt(1)
    attempt.checkpoints.mkdir(parents=True)
    write_json_atomic(
        attempt.state,
        {"state": "learning_interrupted", "gateway_base_url": "http://127.0.0.1:18777"},
    )
    assert cell._resume_gateway_port() == 18777
    reroll = run.CellRunner(
        spec,
        run.RunOptions(
            out_dir=tmp_path / "c", gateway_runs_dir=tmp_path / "runs", new_attempt=True
        ),
    )
    assert reroll._resume_gateway_port() is None


@pytest.mark.asyncio
async def test_a_mirix_cell_checkpoints_home_and_restores_it(tmp_path, fake, gateway):
    """A MIRIX cell (stub server without a database): HOME is part of every checkpoint,
    and a resume moves what the discarded segment added to HOME aside and restores the
    checkpoint's files before the server starts again."""
    _gw, url = gateway
    stub = write_stub_server(tmp_path)
    spec = CellSpec("alfworld", "luna", "mirix")

    def server(s: CellServerSpec, mode: Any) -> MirixServer:
        return MirixServer(
            s,
            mode=mode,
            command=[__import__("sys").executable, str(stub), "--port", str(s.port)],
            database=False,
        )

    k = question(order()[2])

    def make(**kw: Any) -> Any:
        cell = runner(tmp_path, url, spec=spec, factory=storing([], name="mirix", **kw))
        cell.mirix_server_factory = server
        return cell

    failed = await make(fail_once={"after_task": {k}}).run()
    assert failed.status == "learning_interrupted"
    paths = cell_paths(tmp_path, spec)
    home = paths.attempt(1).home
    (home / ".mirix" / "images").mkdir(parents=True, exist_ok=True)
    (home / ".mirix" / "images" / "late.png").write_bytes(
        b"written after the checkpoint"
    )
    (home / ".mirix" / "stub_marker").write_text("changed after the checkpoint")
    done = await make().run()
    assert done.status == "complete", done.detail
    seg = paths.attempt(1).discarded_dir(1)
    assert (seg / "home" / ".mirix" / "images" / "late.png").is_file()
    assert (seg / "home" / ".mirix" / "stub_marker").read_text() == (
        "changed after the checkpoint"
    )
    assert not (home / ".mirix" / "images" / "late.png").exists()
    row = read_rows(paths.attempt(1).discarded_log)[0]
    assert row["mirix"]["post_crash_db"] == "none"
    assert row["moved"]["home_files"] == 2
    assert row["restore"]["files_equal"] is True


def test_the_segment_rule_excludes_earlier_segments():
    """A resume that restores an older checkpoint covers ledger an earlier resume
    already discarded: those calls stay in the earlier segment only."""
    from evals.skill_bench.cell_records import LedgerLine

    lines = [
        LedgerLine(0, {"event": "checkpoint", "id": "c0001-done"}),
        LedgerLine(10, {"event": "call_started", "call_id": "a", "role": "actor"}),
        LedgerLine(
            20, {"event": "call", "call_id": "a", "status": "ok", "cost_total": 1}
        ),
        LedgerLine(30, {"event": "resume", "segment": 1}),
        LedgerLine(40, {"event": "call_started", "call_id": "b", "role": "actor"}),
        LedgerLine(50, {"event": "resume", "segment": 2}),
        LedgerLine(60, {"event": "call", "call_id": "b", "status": "ok"}),
    ]
    first = segment_calls(lines, checkpoint_offset=0, resume_offset=30)
    second = segment_calls(lines, checkpoint_offset=0, resume_offset=50, exclude={"a"})
    assert [c["call_id"] for c in first] == ["a"]
    assert [(c["call_id"], c["final_line"]) for c in second] == [("b", True)]


def test_json_dump_of_a_checkpoint_is_stable(tmp_path):
    """``checkpoint.json`` is verified by its own sha256 (LATEST) and every file's."""
    root = tmp_path / "c0000-init"
    root.mkdir()
    (root / "snapshot.json").write_text("{}")
    import hashlib

    record = {
        "id": "c0000-init",
        "files": {"snapshot.json": hashlib.sha256(b"{}").hexdigest()},
    }
    data = run._write_json_durable(root / "checkpoint.json", record)
    sha = hashlib.sha256(data).hexdigest()
    assert run._verify_checkpoint(root, sha)["id"] == "c0000-init"
    with pytest.raises(run._CheckpointCorrupt):
        run._verify_checkpoint(root, "0" * 64)
    (root / "snapshot.json").write_text("[]")
    with pytest.raises(run._CheckpointCorrupt):
        run._verify_checkpoint(root, sha)
    assert json.loads((root / "checkpoint.json").read_text())["id"] == "c0000-init"


# --- MIRIX (the stub MIRIX of the method's tests, its "database" checkpointed) -------------


def _without_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _without_times(v)
            for k, v in value.items()
            if not (k.endswith("_at") or k in ("at", "ts", "frozen_snapshot_sha256"))
        }
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


async def _mirix_cell(
    tmp_path: Path, url: str, root: Path, **stub_options: Any
) -> tuple[Any, dict[tuple[str, int], Any], list[str]]:
    """A MIRIX cell (the real method on the stub MIRIX) in ``root``, with a fixed salt so
    two cell directories see the same task uids."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import StubMirix, StubMirixServer

    spec = CellSpec("alfworld", "luna", "mirix")
    stubs: dict[tuple[str, int], Any] = {}
    modes: list[str] = []

    def server(s: CellServerSpec, mode: Any) -> Any:
        key = (str(s.cell_dir), s.attempt)
        modes.append(mode)
        if mode == "fresh":
            stubs[key] = StubMirix(threshold=2, **stub_options)
        return StubMirixServer(stubs[key], mode)

    paths = CellPaths(root / spec.cell_id)
    paths.salt.parent.mkdir(parents=True, exist_ok=True)
    if not paths.salt.exists():
        paths.salt.write_bytes(b"a fixed test salt, 32 bytes long!")

    def make() -> Any:
        cell = runner(
            tmp_path,
            url,
            spec=spec,
            factory=lambda: MirixMethod(poll_s=0.01),
            out_dir=paths.root,
        )
        cell.mirix_server_factory = server
        return cell

    return make, stubs, modes


@pytest.mark.asyncio
async def test_a_mirix_batch_that_failed_on_infrastructure_resumes_from_played(
    tmp_path, fake, gateway, monkeypatch
):
    """run.py with the real MIRIX method on the stub MIRIX: the curator batch the second
    item's close started fails on infrastructure; the resume restores that item's played
    checkpoint, database included, and the batch runs again. The frozen memory equals an
    uninterrupted cell's with the same replies."""
    _gw, url = gateway
    reference, _stubs, _modes = await _mirix_cell(tmp_path, url, tmp_path / "reference")
    done = await reference().run()
    assert done.status == "complete", done.detail
    spec = CellSpec("alfworld", "luna", "mirix")
    expected = read_json(
        CellPaths(tmp_path / "reference" / spec.cell_id).attempt(1).frozen_snapshot
    )["snapshot"]

    monkeypatch.setattr(run, "new_instance_id", lambda: "0000beef")  # another cell dir
    make, stubs, modes = await _mirix_cell(
        tmp_path, url, tmp_path / "cells", curator_errors=["infra"]
    )
    failed = await make().run()
    assert failed.status == "learning_interrupted", failed.detail
    paths = CellPaths(tmp_path / "cells" / spec.cell_id)
    state = read_json(paths.attempt(1).state)
    assert state["failure"]["code"] == "mirix_learning"
    assert state["failure_stage"] == "wait_until_idle"
    assert state["learning"]["checkpoint"] == "c0001-played"
    record = read_json(paths.attempt(1).checkpoint("c0001-played") / "checkpoint.json")
    assert (
        record["mirix"]["active_runs"] == 0
        and "stub_mirix_state.pickle" in record["files"]
    )
    (stub,) = stubs.values()
    # One run, whose curator batch MIRIX stopped on infrastructure.
    failed_runs = {k: v["status"] for k, v in stub.runs.items()}
    assert len(failed_runs) == 1

    resumed = await make().run()
    assert resumed.status == "complete", resumed.detail
    assert modes == ["fresh", "resume"]
    row = read_rows(paths.attempt(1).discarded_log)[0]
    assert row["checkpoint"]["id"] == "c0001-played"
    discarded_runs = [r["id"] for r in row["mirix"]["learning_runs"]]
    assert set(discarded_runs) == set(failed_runs)
    assert row["restore"]["db_digest_equal"] is True
    assert row["in_flight"]["stage"] == "wait_until_idle"
    assert [c["role"] for c in row["calls"]] and {c["role"] for c in row["calls"]} <= {
        "mirix",
        "embedding",
    }
    frozen = read_json(paths.attempt(1).frozen_snapshot)["snapshot"]
    assert _without_times(frozen) == _without_times(expected)
    assert all(run["status"] == "succeeded" for run in stub.runs.values())
    # The stub's id counter was restored with its state, so the redone batch's run reuses
    # the id; the discarded row records the stopped one.
    assert len(row["mirix"]["learning_runs"]) == 1
    assert_resumed_cleanly(
        tmp_path, paths, read_json(paths.attempt(1).state)["run_id"], storing=False
    )


def test_the_ledger_reader_parses_only_what_the_gateway_appended(tmp_path):
    """The checkpoint's quiescence and late-call checks read the ledger incrementally: a
    line being written is read once complete, a call is open until its final line, a
    call of another gateway start or of a discarded segment never counts as open, and a
    call started after a marker is found by the marker's offset."""
    path = tmp_path / "gateway.jsonl"
    reader = run._LedgerReader()

    def append(data: bytes) -> int:
        with path.open("ab") as handle:
            offset = handle.tell()
            handle.write(data)
        return offset

    def line(**row: Any) -> bytes:
        return json.dumps(row).encode() + b"\n"

    append(line(event="call_started", call_id="a", gateway_start="g1"))
    append(line(event="call_started", call_id="b", gateway_start="g0"))
    reader.read(path)
    assert reader.open_calls(exclude=frozenset(), gateway_start="g1") == ["a"]
    assert reader.open_calls(exclude=frozenset({"a"}), gateway_start="g1") == []
    assert reader.open_calls(exclude=frozenset(), gateway_start=None) == ["a", "b"]

    marker_at = append(line(event="checkpoint", id="c0000-init"))
    c_line = line(event="call_started", call_id="c", gateway_start="g1")
    c_at = append(c_line[:9])  # the gateway is writing this line
    reader.read(path)
    assert reader.offset == c_at
    marker = reader.marker("checkpoint", id="c0000-init")
    assert marker is not None and marker.offset == marker_at
    assert reader.started_after(marker_at) == []

    append(c_line[9:])
    append(line(event="call", call_id="a", status="ok", role="mirix"))
    append(b"{torn\n")  # a gateway killed mid-write: skipped
    reader.read(path)
    assert reader.offset == path.stat().st_size
    assert reader.started_after(marker_at) == ["c"]
    assert reader.started["c"].offset == c_at
    # From an offset the reader had reached, the lines written since.
    assert reader.started_since(c_at) == ["c"] and reader.started_after(c_at) == []
    assert reader.open_calls(exclude=frozenset(), gateway_start="g1") == ["c"]
    assert [r["call_id"] for r in reader.role_lines] == ["a"]
    reader.read(tmp_path / "another.jsonl")  # another run's ledger starts over
    assert reader.offset == 0 and not reader.started


# --- a gateway pause during learning ------------------------------------------------------


def _hold_pause(gw: Any, on: bool) -> None:
    """Pause or resume ``gw`` by hand: its balance refresh is disabled, so the test
    decides when the pause lifts (``pause_hold_s=0``: a held call is answered 402 at
    once, as one held past its deadline is)."""

    async def no_refresh(*_a: Any, **_kw: Any) -> None:
        return None

    gw.refresh_balance = no_refresh
    gw.paused = on
    gw.pause_reason = "balance_below_floor" if on else None


def _audit_violations(tmp_path: Path) -> list[str]:
    from evals.skill_bench import audit
    from evals.skill_bench.tests.run_support import TinyEnv

    env = TinyEnv()
    result = audit.audit_runs(
        tmp_path / "runs",
        env_factory=lambda _b: env,
        split_mode=None,
        expected_upstream="http://upstream",
    )
    return [
        f"{f.invariant}:{f.check}: {f.detail[:200]}"
        for c in result.cells
        for f in c.findings
        if f.severity == audit.VIOLATION and f.check != "memory_slot"
    ]


@pytest.mark.asyncio
async def test_a_swallowed_call_a_pause_held_resumes_from_the_item_before(tmp_path):
    """A learning call held by a gateway pause and answered 402 has its call_started line,
    so a method that swallows the failure (MIRIX: its calls carry ids, so no redo is
    byte-identical) resumes from the checkpoint before it, not from the start, and that
    resume discards it."""
    held: dict[str, Any] = {"done": False, "gw": None}

    class SwallowsAHeldEmbedding(StoringMethod):
        async def before_task(self, task: Any) -> Any:
            if not self.frozen and self.learned == 2 and not held["done"]:
                held["done"] = True
                _hold_pause(held["gw"], True)
                assert self.cell is not None
                async with httpx.AsyncClient(
                    base_url=self.cell.gateway_base_url, trust_env=False, timeout=60
                ) as client:
                    response = await client.post(
                        f"/runs/{self.cell.run_id}/embedding/v1/embeddings",
                        json={
                            "model": protocol.EMBEDDING_MODEL,
                            "input": "an id-bearing request",
                        },
                        headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                    )
                self.events.append(("embedding", response.status_code))
                _hold_pause(held["gw"], False)
            return await super().before_task(task)

    fake = FakeUpstream(responder=tiny_responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake, pause_hold_s=0.0) as (gw, url):
        held["gw"] = gw
        failed = await runner(
            tmp_path, url, factory=lambda: SwallowsAHeldEmbedding(events)
        ).run()
        assert failed.status == "learning_interrupted", failed.detail
        paths = cell_paths(tmp_path)
        state = read_json(paths.attempt(1).state)
        assert state["failure"]["code"] == "mirix_learning"
        assert state["resume_from"] == "c0001-done"
        assert ("embedding", 402) in events
        events.clear()
        done = await runner(
            tmp_path, url, factory=lambda: SwallowsAHeldEmbedding(events)
        ).run()
    assert done.status == "complete", done.detail
    assert resumed_from(events) == "c0001-done"
    (segment,) = read_rows(paths.attempt(1).discarded_log)
    paused = [
        r
        for r in ledger(tmp_path, RUN_A1)
        if r.get("event") == "call" and r.get("status") == "budget_paused"
    ]
    assert len(paused) == 1
    assert paused[0]["call_id"] in {c["call_id"] for c in segment["calls"]}
    assert_resumed_cleanly(tmp_path, paths)
    assert _audit_violations(tmp_path) == []


@pytest.mark.asyncio
async def test_a_paused_actor_call_is_discarded_with_its_try(tmp_path):
    """An actor call a pause answers 402 fails the learning item's rollout (INFRA); its
    failed try is moved aside with the call, and the redo of the same first request is
    the redo of a discarded call, not an actor retry (the audit passes)."""
    held: dict[str, Any] = {"done": False, "gw": None}

    class PausedBeforeItsActorCall(StoringMethod):
        async def before_task(self, task: Any) -> Any:
            if not self.frozen and self.learned == 2 and not held["done"]:
                held["done"] = True
                _hold_pause(held["gw"], True)
            return await super().before_task(task)

    fake = FakeUpstream(responder=tiny_responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake, pause_hold_s=0.0) as (gw, url):
        held["gw"] = gw
        failed = await runner(
            tmp_path, url, factory=lambda: PausedBeforeItsActorCall(events)
        ).run()
        assert failed.status == "learning_interrupted", failed.detail
        _hold_pause(gw, False)
        done = await runner(
            tmp_path, url, factory=lambda: PausedBeforeItsActorCall(events)
        ).run()
    assert done.status == "complete", done.detail
    paths = cell_paths(tmp_path)
    (segment,) = read_rows(paths.attempt(1).discarded_log)
    assert segment["checkpoint"]["id"] == "c0001-done"
    assert [(c["role"], c["status"]) for c in segment["calls"]] == [
        ("actor", "budget_paused")
    ]
    assert_resumed_cleanly(tmp_path, paths)
    assert _audit_violations(tmp_path) == []


@pytest.mark.asyncio
async def test_the_redo_of_a_judge_call_a_pause_failed_passes_the_audit(tmp_path):
    """A judge call in after_task that a pause answers 402 fails the item (INFRA); the
    resume from its played checkpoint sends the byte-identical request again, which the
    gateway flags ``repeat_of``: the redo of a discarded call, allowed."""
    held: dict[str, Any] = {"paused": False, "gw": None}

    class JudgesWhilePaused(StoringMethod):
        async def after_task(self, trajectory: Any) -> None:
            if not self.frozen and self.learned == 3:
                assert self.cell is not None
                if not held["paused"]:
                    held["paused"] = True
                    _hold_pause(held["gw"], True)
                async with httpx.AsyncClient(
                    base_url=self.cell.gateway_base_url, trust_env=False, timeout=60
                ) as client:
                    response = await client.post(
                        f"/runs/{self.cell.run_id}/judge/v1/chat/completions",
                        json={
                            "model": self.cell.model,
                            "messages": [{"role": "user", "content": "judge item 3"}],
                        },
                        headers={
                            "Authorization": "Bearer " + run.LOCAL_TOKEN,
                            "x-skill-bench-item": trajectory.task_uid,
                        },
                    )
                _hold_pause(held["gw"], False)
                self.events.append(("judge", response.status_code))
                if response.status_code != 200:
                    raise httpx.ConnectError("the judge call failed (402)")
            await super().after_task(trajectory)

    fake = FakeUpstream(responder=tiny_responder)
    events: list[Any] = []
    with conformant_gateway(tmp_path, fake, pause_hold_s=0.0) as (gw, url):
        held["gw"] = gw
        failed = await runner(
            tmp_path, url, factory=lambda: JudgesWhilePaused(events)
        ).run()
        assert failed.status == "learning_interrupted", failed.detail
        done = await runner(
            tmp_path, url, factory=lambda: JudgesWhilePaused(events)
        ).run()
    assert done.status == "complete", done.detail
    assert [e for e in events if e[0] == "judge"] == [("judge", 402), ("judge", 200)]
    redo = [
        r
        for r in ledger(tmp_path, RUN_A1)
        if r.get("event") == "call" and r.get("role") == "judge" and r.get("repeat_of")
    ]
    assert len(redo) == 1
    assert _audit_violations(tmp_path) == []


def test_a_final_line_without_call_started_is_placed_by_its_own_offset(tmp_path):
    """Defence in depth for ledgers an older gateway wrote (a call held by a pause was
    answered 402 without a call_started line): the segment rule, the ledger reader and
    the lost-call resume point place such a call by its final line."""
    from evals.skill_bench.cell_records import LedgerLine

    lines = [
        LedgerLine(0, {"event": "checkpoint", "id": "c0001-done"}),
        LedgerLine(
            10,
            {
                "event": "call",
                "call_id": "held",
                "role": "embedding",
                "status": "budget_paused",
            },
        ),
        LedgerLine(20, {"event": "resume", "segment": 1}),
        LedgerLine(30, {"event": "call", "call_id": "late", "status": "ok"}),
    ]
    (call,) = segment_calls(lines, checkpoint_offset=0, resume_offset=20)
    assert (call["call_id"], call["started_offset"], call["final_line"]) == (
        "held",
        10,
        True,
    )
    assert call["role"] == "embedding" and call["status"] == "budget_paused"

    path = tmp_path / "gateway.jsonl"
    rows = [
        {"event": "checkpoint", "id": "c0000-init"},
        {"event": "call_started", "call_id": "a", "gateway_start": "g"},
        {"event": "call", "call_id": "a", "role": "mirix", "status": "ok"},
        {"event": "call", "call_id": "b", "role": "embedding", "status": "x"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    reader = run._LedgerReader()
    reader.read(path)
    b_at = path.read_bytes().rindex(b'{"event": "call", "call_id": "b"')
    assert reader.started["b"].offset == b_at
    assert reader.started_after(0) == ["a", "b"]
    assert reader.open_calls(exclude=frozenset(), gateway_start="g") == []


# --- a call a resume discarded in flight --------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_finished_after_its_resume_is_priced_from_its_final_line(
    tmp_path, monkeypatch
):
    """A learner call still in flight at the resume (a pause, or a slow
    provider, holds it) is listed with ``final_line: false``, and its cost is not in the
    row's ``cost_total`` (what the ledger held at the resume). The row names it in
    ``pending_calls`` and stays as written; ``costs.json`` ``discarded_segments`` and the
    report (what summarize_learning prints) price it from the final line the gateway
    writes after the resume marker."""
    monkeypatch.setattr(run, "CHECKPOINT_QUIESCE_S", 0.3)
    hold = threading.Event()
    paths = cell_paths(tmp_path)

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        messages = body.get("messages") or []
        if messages and messages[-1].get("content") == "HELD":
            return Reply(hold=hold)
        return tiny_responder(endpoint, body)

    tasks: list[asyncio.Task[Any]] = []

    class HoldsACall(StoringMethod):
        async def after_task(self, trajectory: Any) -> None:
            await super().after_task(trajectory)
            if self.learned == 2 and not tasks:
                assert self.cell is not None
                cell = self.cell

                async def held() -> httpx.Response:
                    async with httpx.AsyncClient(
                        base_url=cell.gateway_base_url, trust_env=False, timeout=60
                    ) as client:
                        return await client.post(
                            f"/runs/{cell.run_id}/learner/v1/chat/completions",
                            json={
                                "model": SPEC.model,
                                "messages": [{"role": "user", "content": "HELD"}],
                            },
                            headers={"Authorization": "Bearer " + run.LOCAL_TOKEN},
                        )

                tasks.append(asyncio.ensure_future(held()))
                while not any(
                    (r.json.get("messages") or [{}])[-1].get("content") == "HELD"
                    for r in fake.calls("chat/completions")
                ):
                    await asyncio.sleep(0.01)

        async def before_task(self, task: Any) -> Any:
            # The resumed run, once its row is written: the held call finishes now.
            if (
                tasks
                and not hold.is_set()
                and read_rows(paths.attempt(1).discarded_log)
            ):
                hold.set()
                response = await asyncio.wait_for(asyncio.shield(tasks[0]), 30)
                assert response.status_code == 200
            return await super().before_task(task)

    fake = FakeUpstream(responder=responder)
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        failed = await runner(tmp_path, url, factory=lambda: HoldsACall([])).run()
        assert failed.status == "learning_interrupted", failed.detail
        done = await runner(tmp_path, url, factory=lambda: HoldsACall([])).run()
    assert done.status == "complete", done.detail
    (segment,) = read_rows(paths.attempt(1).discarded_log)
    (call,) = segment["calls"]
    assert (call["role"], call["final_line"], call["cost_total"]) == (
        "learner",
        False,
        0.0,
    )
    assert segment["pending_calls"] == [call["call_id"]]
    assert segment["cost_total"] == 0.0
    (final,) = [
        r
        for r in ledger(tmp_path, RUN_A1)
        if r.get("event") == "call" and r.get("call_id") == call["call_id"]
    ]
    assert final["status"] == "ok" and final["cost_total"] > 0
    lines = read_ledger_lines(tmp_path / "runs" / RUN_A1 / "gateway.jsonl")
    (at,) = [
        x.offset
        for x in lines
        if x.row.get("event") == "call" and x.row.get("call_id") == call["call_id"]
    ]
    assert at > segment["resume_marker_offset"]  # written after the resume
    costs = read_json(paths.attempt(1).costs)
    (priced,) = costs["discarded_segments"]
    assert priced["cost_total"] == pytest.approx(final["cost_total"])
    assert priced["cost_total"] == pytest.approx(costs["discarded"]["cost_total"])
    assert (priced["finished_after_resume"], priced["unfinished_calls"]) == (1, 0)
    assert priced["cost_total_at_resume"] == 0.0
    state = read_json(paths.attempt(1).state)
    assert state["resumes"][0]["pending_calls"] == 1
    # What report.py and summarize_learning show: the ledger's figure, not the row's.
    from evals.skill_bench import report

    (loaded,) = report.load_cells(tmp_path / "runs", split_mode=None)
    assert report.cell_costs(loaded)["learning_discarded"]["cost"] == pytest.approx(
        final["cost_total"]
    )
    (resume,) = report.cell_resumes(loaded)
    assert resume["cost"] == pytest.approx(final["cost_total"])
    assert_resumed_cleanly(tmp_path, paths)
