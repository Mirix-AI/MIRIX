"""The MIRIX method against a stub MIRIX server.

The method speaks to MIRIX through MIRIX's own ``MirixClient`` and renders with MIRIX's
own ``render_retrieved_memories`` (this checkout's ``mirix`` package). The stub
(``tests/mirix_stub.py``) implements MIRIX's endpoints with MIRIX's shapes and records
every request; its learning runs and retrievals call a real gateway over the fake
upstream, as MIRIX's distiller, topic extraction and query embedding would. The
integration tests at the end run the same cell against a real MIRIX server (the compose
Postgres, no model: the gateway runs over the fake upstream, scripted as MIRIX's model).
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.rollout import RolloutError, run_episode
from evals.skill_bench.infra.failures import (
    IDLE_WAIT_TIMEOUT,
    MIRIX_RETRIEVAL,
    MIRIX_SERVER,
    FailureKind,
    LearningAttemptInvalid,
    MirixRetrievalDegraded,
    classify_exception,
)
from evals.skill_bench.infra.mirix_server import MirixServer, MirixServerError
from evals.skill_bench.infra.processes import find_free_port
from evals.skill_bench.methods import mirix as M
from evals.skill_bench.methods.base import (
    FrozenMemoryMismatch,
    GATEWAY_LOCAL_TOKEN,
    CellContext,
    Injection,
    TaskView,
    method_class,
)
from evals.skill_bench.tests.fakes import FakeUpstream, serve_gateway
from evals.skill_bench.tests.mirix_db import (
    database_digest,
    retrieval_log_rows,
    session_agent_contexts,
    skill_feedback_rows,
)
from evals.skill_bench.tests.method_contract import (
    ContractItem,
    contract_cell,
    contract_items,
    exercise_method,
)
from evals.skill_bench.tests.mirix_stub import (
    STUB_DISTILLER_PROMPT,
    StubMirix,
    StubMirixServer,
    StubMirixServers,
    with_stub_mirix,
)
from evals.skill_bench.tests.mirix_model import SKILL, USAGE_SECTION_HEAD, MirixModel
from evals.skill_bench.tests.rollout_support import (
    SALT,
    Script,
    ledger_calls,
    text_reply,
)

LUNA = protocol.MODELS["luna"]
ADD, RETRIEVE, FLUSH = (
    "/memory/add",
    "/memory/retrieve/conversation",
    "/memory/skills/flush",
)
RUNS, COMPONENTS = "/memory/learning_runs", "/memory/components"
LOGS = "/memory/retrieval_logs"


def wire(value: Any) -> str:
    """The canonical wire form of a JSON value, keys in their original order."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[str]:
    with serve_gateway(tmp_path, FakeUpstream()) as (_gateway, base_url):
        yield base_url


def mirix_cell(
    tmp_path: Path, gateway_url: str, benchmark: str = "alfworld", **kw: Any
) -> CellContext:
    return contract_cell(
        tmp_path / "work",
        benchmark=benchmark,
        method="mirix",
        gateway_base_url=gateway_url,
        run_id="cell-test-a1",
        **kw,
    )


def method_for(servers: StubMirixServers, **kw: Any) -> M.MirixMethod:
    return M.MirixMethod(server_factory=servers, poll_s=0.01, **kw)


def learning_items(benchmark: str, n: int = 3) -> list[ContractItem]:
    """``n`` learning items with distinct uids (the contract items, repeated if needed)."""
    base = contract_items(benchmark)
    items = []
    for i in range(n):
        item = base[i % len(base)]
        uid = f"t{i:02d}" + item.view.task_uid[3:]
        items.append(
            ContractItem(replace(item.view, task_uid=uid), copy.deepcopy(item.messages))
        )
    assert len({i.view.task_uid for i in items}) == n
    return items


def task_views(benchmark: str) -> list[TaskView]:
    """Two test-phase views with uids no learning item has."""
    return [
        replace(item.view, task_uid=f"z{i:02d}" + item.view.task_uid[3:])
        for i, item in enumerate(contract_items(benchmark)[:2])
    ]


async def learn(
    method: M.MirixMethod, items: list[ContractItem], *, idle_s: float = 10.0
) -> list[Injection]:
    injections = []
    for item in items:
        injections.append(await method.before_task(copy.deepcopy(item.view)))
        for message in item.messages:
            await method.on_message(item.view.task_uid, copy.deepcopy(message))
        await method.after_task(item.trajectory())
        await method.wait_until_idle(idle_s)
    return injections


def calls(stub: StubMirix, start: int = 0) -> list[tuple[str, str]]:
    return [(r.method, r.path) for r in stub.requests[start:]]


# --- setup ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_creates_the_tenant_and_the_meta_agent_through_mirix(
    tmp_path, gateway
):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        assert calls(stub) == [
            ("POST", "/organizations/create_or_get"),
            ("POST", "/clients/create_or_get"),
            ("POST", "/agents/meta/initialize"),
            ("POST", "/users/create_or_get"),
            ("GET", COMPONENTS),
            ("GET", RUNS),
        ]
        assert all(r.client_id == "skill-bench-client" for r in stub.requests[2:])
        client = stub.requests[1].json
        assert client["write_scope"] == "skill_bench" and client["read_scopes"] == [
            "skill_bench"
        ]
        config = stub.requests[2].json["config"]
        limits = protocol.model_limits(LUNA)
        # H7.2: exactly these fields; everything else is MIRIX's default.
        assert config == {
            "llm_config": {
                "model_endpoint_type": "openai",
                "model": LUNA,
                "model_endpoint": f"{gateway}/runs/cell-test-a1/mirix/v1",
                "context_window": limits.context_length,
                "max_output_tokens": limits.max_completion_tokens,
                "api_key": GATEWAY_LOCAL_TOKEN,
            },
            "embedding_config": {
                "embedding_endpoint_type": "openrouter",
                "embedding_dim": 3072,
                "embedding_endpoint": f"{gateway}/runs/cell-test-a1/embedding/v1",
                "embedding_model": "google/gemini-embedding-001",
                "api_key": GATEWAY_LOCAL_TOKEN,
            },
        }
        assert stub.requests[2].json["update_agents"] is False
        state = json.loads((cell.work_dir / M.STATE_FILE).read_text())
        assert state["meta_agent_id"] == method.state.meta_agent_id
        assert "api_key" not in json.dumps(state)  # no credential in records
        assert not state["frozen"]
    finally:
        await method.close()
    assert servers.servers[0].stopped  # close() stops the server the method started


# --- the call sequence: 3 learning tasks, 2 test tasks -------------------------------------


@pytest.mark.asyncio
async def test_three_learning_and_two_test_tasks(tmp_path, gateway):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway)
    items = learning_items("alfworld")
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        start = len(stub.requests)
        learned = await learn(method, items)
        expected: list[tuple[str, str]] = []
        for item in items:
            expected += [("POST", RETRIEVE), ("GET", LOGS)]  # the retrieval and its row
            expected += [("POST", ADD)] * len(item.messages)
            expected += [
                ("POST", f"/memory/sessions/{item.view.task_uid}/close"),
                ("GET", RUNS),  # below the threshold: nothing to wait for
            ]
        assert calls(stub, start) == expected
        assert all(i == Injection() for i in learned)  # nothing learned yet

        start = len(stub.requests)
        await method.finalize()
        finalize_calls = calls(stub, start)
        (run,) = stub.runs.values()
        assert finalize_calls[0] == ("POST", FLUSH)
        second = finalize_calls.index(("POST", FLUSH), 1)
        assert set(finalize_calls[1 : second - 1]) == {("GET", RUNS)}  # the idle poll
        assert finalize_calls[second - 1] == ("GET", f"{RUNS}/{run['id']}")
        assert finalize_calls[second:] == [
            ("POST", FLUSH),  # nothing left to learn: MIRIX starts no run
            ("GET", RUNS),  # idle at once
            ("GET", COMPONENTS),
            ("GET", RUNS),
            ("GET", "/memory/skills/proc-0001/history"),
            ("GET", "/memory/skills/proc-0001/versions/1"),
        ]
        assert run["trigger"] == "flush" and run["status"] == "succeeded"
        assert run["session_ids"] == sorted(i.view.task_uid for i in items)
        await method.wait_until_idle(1.0)
        frozen = await method.snapshot()

        start = len(stub.requests)
        views = task_views("alfworld")
        injections = await asyncio.gather(*(method.before_task(v) for v in views))
        assert (
            sorted(calls(stub, start)) == [("GET", LOGS)] * 2 + [("POST", RETRIEVE)] * 2
        )
        assert stub.writes()[-1].path == FLUSH  # the test phase wrote nothing
        assert await method.snapshot() == frozen
    finally:
        await method.close()

    # before_task: MIRIX's retrieval with the raw first messages, rendered by MIRIX.
    retrievals = [r for r in stub.requests if r.path == RETRIEVE]
    assert len(retrievals) == 5
    for request, view in zip(retrievals, [i.view for i in items] + views, strict=True):
        assert request.json == {
            "user_id": "skill-bench-user",
            "messages": list(view.first_messages),
            "limit": 10,
            "local_model_for_retrieval": None,
            "memory_types": ["procedural"],
            "session_id": view.task_uid,
        }
        assert wire(request.json["messages"]) == wire(list(view.first_messages))
    for injection in injections:
        assert injection.text.startswith(
            "## Relevant skills from memory\n### stub skill 1"
        )
        assert injection.item_ids == ("proc-0001",)
    # The snapshot exports skills, their history, every version and the runs (M10).
    assert [s["id"] for s in frozen["skills"]] == ["proc-0001"]
    assert (
        frozen["skill_history"]["proc-0001"]["versions"][0]["change_type"] == "create"
    )
    assert frozen["skill_versions"] == {
        "proc-0001": {"1": stub.versions["proc-0001"][1]}
    }
    assert [r["status"] for r in frozen["learning_runs"]] == ["succeeded"]
    assert frozen["frozen"] and frozen["frozen_snapshot_sha256"]
    assert frozen["counters"] == {
        "messages_added": sum(len(i.messages) for i in items),
        "sessions_closed": 3,
        "runs_started": 1,
        "add_warnings": 0,
    }
    # MIRIX's own model traffic went through the gateway's mirix / embedding roles.
    roles = {(r["role"], r["model"]) for r in ledger_calls(tmp_path, "cell-test-a1")}
    assert ("mirix", LUNA) in roles
    assert {r for r, _ in roles} == {"mirix", "embedding"}
    assert STUB_DISTILLER_PROMPT in [c["text"] for c in stub.gateway_calls]
    records = [
        json.loads(line)
        for line in (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()
    ]
    assert len(records) == 5 and records[-1]["item_ids"] == ["proc-0001"]
    # Each record keeps the retrieval's log row evidence (M7.4) and its scores.
    assert {r["retrieval_id"] for r in records} == {
        row["id"] for row in stub.retrieval_logs
    }
    assert all(
        r["evidence"]["extraction_status"] == "ok"
        and r["evidence"]["embedding_status"] == "ok"
        and r["failure"] is None
        for r in records
    )
    assert records[-1]["retrieval_scores"] == {
        "proc-0001": {"bm25": 1.0, "cosine": 0.9}
    }
    # ... the topics and the task-level query that ranked the skills (B04) ...
    assert all(
        r["topics"] == "stub topics" and r["procedural_query"] == "stub task"
        for r in records
    )
    # ... and its ranking trace: the stub logs only returned_ids of PROCEDURAL_TRACE_KEYS.
    assert [r["procedural"] for r in records] == [
        {"returned_ids": r["item_ids"]} for r in records
    ]
    flush_run = [
        json.loads(line)
        for line in (cell.work_dir / M.LEARNING_RECORD_FILE).read_text().splitlines()
        if json.loads(line)["event"] == "flush_run"
    ]
    assert [(r["status"], r["sessions"], r["experiences"]) for r in flush_run] == [
        ("succeeded", 3, 3)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
async def test_messages_reach_mirix_byte_for_byte(tmp_path, gateway, benchmark):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway, benchmark)
    items = contract_items(benchmark)  # every Q20 shape, the empty assistant turn too
    method = method_for(servers)
    await method.setup(cell)
    try:
        await learn(method, items)
    finally:
        await method.close()
    stub = servers.stub_for(cell.work_dir)
    adds = [r for r in stub.requests if r.path == ADD]
    sent = [(i.view.task_uid, m) for i in items for m in i.messages]
    assert len(adds) == len(sent)
    for request, (uid, message) in zip(adds, sent, strict=True):
        body = request.json
        assert body == {
            "user_id": "skill-bench-user",
            "meta_agent_id": method.state.meta_agent_id,
            "messages": [message],
            "chaining": True,
            "verbose": False,
            "session_id": uid,
            "session_tag": "task",
        }
        assert wire(body["messages"][0]) == wire(message)
        assert wire(message).encode() in request.raw  # the exact bytes on the wire
    for item in items:
        assert stub.sessions[item.view.task_uid]["messages"] == list(item.messages)


def agent_context_of(benchmark: str) -> str:
    """The benchmark's agent context, as the rollout hands it to methods."""
    from evals.skill_bench.envs import alfworld as alf
    from evals.skill_bench.envs import webshop as ws

    if benchmark == "alfworld":
        return alf.agent_context(alf.AlfWorldTemplates.load())
    return ws.agent_context(ws.WebShopTemplates.load())


def with_agent_context(items: list[ContractItem], context: str) -> list[ContractItem]:
    return [replace(i, view=replace(i.view, agent_context=context)) for i in items]


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
async def test_each_learning_session_gets_the_agent_context_with_its_first_add(
    tmp_path, gateway, benchmark
):
    """The first add of each learning session carries the task view's
    agent context, which MIRIX stores on the session (the stub as MIRIX does: the latest
    non-blank one); the session's other adds send none, nor does a task whose view has
    none, nor the test phase (its sessions are never distilled)."""
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway, benchmark)
    context = agent_context_of(benchmark)
    items = learning_items(benchmark, 3)
    items = with_agent_context(items[:2], context) + items[2:]
    tests = [replace(v, agent_context=context) for v in task_views(benchmark)]
    method = method_for(servers)
    await method.setup(cell)
    meta_agent_id = method.state.meta_agent_id
    try:
        await learn(method, items)
        assert method._agent_contexts == {}  # nothing pending between items
        await method.finalize()
        await method.wait_until_idle(10.0)
        for view in tests:
            await method.before_task(view)
        assert method._agent_contexts == {}
    finally:
        await method.close()
    stub = servers.stub_for(cell.work_dir)
    adds = [r.json for r in stub.requests if r.path == ADD]
    for index, item in enumerate(items):
        uid = item.view.task_uid
        bodies = [b for b in adds if b["session_id"] == uid]
        assert len(bodies) == len(item.messages)
        later = [b for b in bodies[1:] if "agent_context" in b]
        assert later == []
        if index < 2:
            # The first add is the one it always was, plus the agent context.
            assert bodies[0] == {
                "user_id": "skill-bench-user",
                "meta_agent_id": meta_agent_id,
                "messages": [item.messages[0]],
                "chaining": True,
                "verbose": False,
                "session_id": uid,
                "session_tag": "task",
                "agent_context": context,
            }
            assert stub.sessions[uid]["agent_context"] == context
        else:
            assert "agent_context" not in bodies[0]
            assert "agent_context" not in stub.sessions[uid]
    assert not any(v.task_uid in stub.sessions for v in tests)
    assert len(adds) == sum(len(i.messages) for i in items)


@pytest.mark.asyncio
async def test_a_mirix_without_the_agent_context_api_is_refused(tmp_path, gateway):
    """An older MIRIX (``/memory/add`` without ``agent_context``) cannot take the agent
    context: setup refuses it before creating anything."""
    servers = StubMirixServers(no_agent_context=True)
    with pytest.raises(M.MirixApiError, match="'agent_context'"):
        await method_for(servers).setup(mirix_cell(tmp_path, gateway, "alfworld"))
    assert servers.servers[0].stopped
    assert servers.stubs[tmp_path / "work"].requests == []


@pytest.mark.asyncio
async def test_the_threshold_starts_learning_and_the_idle_wait_waits_for_it(
    tmp_path, gateway
):
    servers = StubMirixServers(threshold=2, run_delay_s=0.3)
    cell = mirix_cell(tmp_path, gateway)
    items = learning_items("alfworld")
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, items[:1])
        t0 = time.monotonic()
        await learn(method, items[1:2])  # the second close claims both sessions
        assert time.monotonic() - t0 >= 0.3
        (run,) = stub.runs.values()
        assert run["trigger"] == "close" and run["status"] == "succeeded"
        third = await method.before_task(items[2].view)
        assert "stub skill 1" in third.text  # learned before the next item (Q18)
    finally:
        await method.close()
    assert method.state.counters.runs_started == 1


# --- Q18: the idle wait is bounded ---------------------------------------------------------


@pytest.mark.asyncio
async def test_an_idle_timeout_raises_an_infra_failure(tmp_path, gateway):
    servers = StubMirixServers(threshold=1, hold_runs=True)
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    item = learning_items("alfworld", 1)[0]
    try:
        for message in item.messages:
            await method.on_message(item.view.task_uid, message)
        await method.after_task(item.trajectory())
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="still in flight") as info:
            await method.wait_until_idle(0.2)
        assert 0.2 <= time.monotonic() - t0 < 5
        failure = classify_exception(info.value)
        assert (failure.kind, failure.code) == (FailureKind.INFRA, IDLE_WAIT_TIMEOUT)
        stub.release()
        await method.wait_until_idle(5.0)
    finally:
        await method.close()


@pytest.mark.asyncio
async def test_an_idle_timeout_in_a_real_rollout_aborts_the_learning_item(tmp_path):
    import evals.skill_bench.envs.alfworld as alf
    from evals.skill_bench.tests.test_env_alfworld import Games, act

    env = alf.AlfWorldEnv(data_root=tmp_path, backend_factory=Games().factory())
    task = alf.AlfWorldEnv(backend_factory=Games().factory()).load_tasks("train")[0]
    actor_script = Script(alfworld=[text_reply(act("win"))])
    fake = FakeUpstream(responder=with_stub_mirix(actor_script))
    servers = StubMirixServers(threshold=1, hold_runs=True)
    from evals.skill_bench.agent.actor import Actor

    with serve_gateway(tmp_path, fake) as (_gateway, base_url):
        cell = mirix_cell(tmp_path, base_url, "alfworld")
        method = method_for(servers)
        await method.setup(cell)
        try:
            async with Actor(
                gateway_base_url=base_url, run_id=cell.run_id, model=cell.model
            ) as actor:
                with pytest.raises(RolloutError) as info:
                    await run_episode(
                        env,
                        task,
                        method=method,
                        actor=actor,
                        salt=SALT,
                        phase="learning",
                        idle_timeout_s=0.2,
                    )
        finally:
            servers.stub_for(cell.work_dir).release()
            await method.close()
    assert info.value.stage == "wait_until_idle"
    assert info.value.failure.kind is FailureKind.INFRA
    assert info.value.failure.code == IDLE_WAIT_TIMEOUT
    # MIRIX received the raw ALFWorld messages and nothing scored.
    stub = servers.stub_for(cell.work_dir)
    stored = stub.sessions[info.value.trajectory.task_uid]["messages"]
    assert stored == list(info.value.trajectory.messages)
    sent = json.dumps([r.json for r in stub.requests])
    assert all(label not in sent for label in env.label_strings(task))


# --- the frozen test phase and its resume ------------------------------------------------------


async def learned_cell(
    tmp_path: Path, gateway: str, servers: StubMirixServers
) -> tuple[CellContext, dict[str, Any]]:
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    try:
        await learn(method, learning_items("alfworld"))
        await method.finalize()
        return cell, await method.snapshot()
    finally:
        await method.close()


@pytest.mark.asyncio
async def test_the_test_phase_resumes_read_only(tmp_path, gateway):
    servers = StubMirixServers()
    cell, frozen = await learned_cell(tmp_path, gateway, servers)
    stub = servers.stub_for(cell.work_dir)
    writes_before = len(stub.writes())
    start = len(stub.requests)
    resumed = method_for(servers)
    await resumed.setup(replace(cell, phase="test"))
    try:
        assert servers.servers[-1].mode == "resume"
        assert {m for m, _ in calls(stub, start)} == {"GET"}  # resume reads only
        assert await resumed.snapshot() == frozen
        views = task_views("alfworld")
        injections = await asyncio.gather(*(resumed.before_task(v) for v in views))
        assert all("stub skill 1" in i.text for i in injections)
        item = learning_items("alfworld", 1)[0]
        with pytest.raises(RuntimeError, match="frozen"):
            await resumed.on_message(item.view.task_uid, item.messages[0])
        with pytest.raises(RuntimeError, match="frozen"):
            await resumed.after_task(item.trajectory())
        with pytest.raises(RuntimeError, match="frozen"):
            await resumed.finalize()
        assert await resumed.snapshot() == frozen
    finally:
        await resumed.close()
    assert len(stub.writes()) == writes_before  # the test phase wrote nothing


@pytest.mark.asyncio
async def test_resume_refuses_anything_but_the_frozen_memory(tmp_path, gateway):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway)
    with pytest.raises(MirixServerError, match="no MIRIX database"):
        await method_for(servers).setup(replace(cell, phase="test"))

    learning = method_for(servers)
    await learning.setup(cell)
    await learning.close()
    with pytest.raises(FrozenMemoryMismatch, match="never frozen"):
        await method_for(servers).setup(replace(cell, phase="test"))
    assert all(s.stopped for s in servers.servers)  # a failed setup stops its server

    tmp2 = tmp_path / "second"
    servers2 = StubMirixServers()
    cell2, _ = await learned_cell(tmp2, gateway, servers2)
    stub = servers2.stub_for(cell2.work_dir)
    stub.skills["proc-9999"] = {**stub.skills["proc-0001"], "id": "proc-9999"}
    stub.history["proc-9999"] = []
    with pytest.raises(FrozenMemoryMismatch, match="differs from the memory frozen"):
        await method_for(servers2).setup(replace(cell2, phase="test"))
    del stub.skills["proc-9999"]
    with pytest.raises(FrozenMemoryMismatch, match="not to this cell attempt"):
        await method_for(servers2).setup(replace(cell2, phase="test", attempt=2))
    state_file = cell2.work_dir / M.STATE_FILE
    state_file.unlink()
    with pytest.raises(FrozenMemoryMismatch, match="no frozen MIRIX state"):
        await method_for(servers2).setup(replace(cell2, phase="test"))


@pytest.mark.asyncio
async def test_a_resumed_agent_must_point_at_this_cells_gateway_run(tmp_path, gateway):
    servers = StubMirixServers()
    cell, _ = await learned_cell(tmp_path, gateway, servers)
    moved = replace(cell, phase="test", run_id="another-run-a1")
    with pytest.raises(M.MirixApiError, match="model_endpoint"):
        await method_for(servers).setup(moved)


# --- refusals and failures ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_mirix_without_the_needed_api_is_refused(tmp_path, gateway):
    servers = StubMirixServers(legacy=True)
    with pytest.raises(M.MirixApiError) as info:
        await method_for(servers).setup(mirix_cell(tmp_path, gateway))
    message = str(info.value)
    assert "GET /memory/skills/{skill_id}/history" in message
    assert "'memory_types'" in message and "'session_id'" in message
    assert servers.servers[0].stopped
    assert servers.stubs[tmp_path / "work"].requests == []  # nothing was created


def test_missing_features_reads_the_openapi_document():
    stub = StubMirix()
    from fastapi.testclient import TestClient

    document = TestClient(stub.app).get("/openapi.json").json()
    assert M.missing_features(document) == []
    assert M.missing_features(
        TestClient(StubMirix(legacy=True).app).get("/openapi.json").json()
    ) == [
        "GET /memory/retrieval_logs",
        "GET /memory/skills/{skill_id}/history",
        "GET /memory/skills/{skill_id}/versions/{version_no}",
        "/memory/retrieve/conversation request field 'memory_types' "
        "(retrieval by memory type and session)",
        "/memory/retrieve/conversation request field 'session_id' "
        "(retrieval by memory type and session)",
    ]
    assert M.missing_features(
        TestClient(StubMirix(no_agent_context=True).app).get("/openapi.json").json()
    ) == [
        "/memory/add request field 'agent_context' "
        "(task sessions with an agent context)"
    ]
    assert M.missing_features({}) == ["an OpenAPI document (/openapi.json)"]
    # Parameter names do not matter, $ref and allOf are followed.
    renamed = copy.deepcopy(document)
    renamed["paths"]["/memory/skills/{id}/history"] = renamed["paths"].pop(
        "/memory/skills/{skill_id}/history"
    )
    assert M.missing_features(renamed) == []


@pytest.mark.asyncio
async def test_an_old_mirix_that_drops_a_config_field_is_refused(tmp_path, gateway):
    # An older MIRIX has no max_output_tokens and would ignore the F9 budget.
    servers = StubMirixServers(drop_llm_fields=("max_output_tokens",))
    with pytest.raises(M.MirixApiError, match="max_output_tokens"):
        await method_for(servers).setup(mirix_cell(tmp_path, gateway))
    assert servers.servers[0].stopped


@pytest.mark.asyncio
async def test_a_learning_attempt_needs_an_empty_mirix(tmp_path, gateway):
    stub = StubMirix()
    stub.skills["proc-0001"] = {"id": "proc-0001", "name": "old"}
    stub.history["proc-0001"] = []
    server = StubMirixServer(stub)
    await server.start()
    try:
        cell = mirix_cell(tmp_path, gateway, mirix_base_url=server.base_url)
        method = M.MirixMethod(poll_s=0.01)
        with pytest.raises(RuntimeError, match="starts from empty memory"):
            await method.setup(cell)
        assert method.server is None  # an external server is never the method's to stop
        await server.ensure_alive()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_mirix_failures_are_infra_or_harness(tmp_path, gateway):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    item = learning_items("alfworld", 1)[0]
    uid = item.view.task_uid
    try:
        stub.fail_once[("POST", ADD)] = 503
        with pytest.raises(MirixServerError) as outage:
            await method.on_message(uid, item.messages[0])
        assert classify_exception(outage.value).code == MIRIX_SERVER
        assert classify_exception(outage.value).kind is FailureKind.INFRA

        stub.fail_once[("POST", f"/memory/sessions/{uid}/close")] = 404
        with pytest.raises(M.MirixApiError) as refused:
            await method.after_task(item.trajectory())
        assert refused.value.status == 404
        assert classify_exception(refused.value).kind is FailureKind.HARNESS

        stub.add_status = "queued"  # a MIRIX that sends task sessions to the meta agent
        with pytest.raises(M.MirixApiError, match="did not store"):
            await method.on_message(uid, item.messages[0])
        stub.add_status = "stored"

        await method.on_message(uid, item.messages[0])
        await method.after_task(item.trajectory())
        with pytest.raises(M.MirixApiError, match="already closed"):
            await method.after_task(item.trajectory())

        server = servers.servers[-1]
        await server.stop()
        with pytest.raises(MirixServerError, match="not running") as dead:
            await method.wait_until_idle(1.0)
        assert classify_exception(dead.value).kind is FailureKind.INFRA
    finally:
        await method.close()


def test_the_default_server_is_the_cells_own_mirix(tmp_path):
    cell = contract_cell(tmp_path / "work", benchmark="alfworld", method="mirix")
    with pytest.raises(ValueError, match="mirix_port"):
        M.default_server(cell, "fresh")
    cell = contract_cell(
        tmp_path / "work", benchmark="alfworld", method="mirix", mirix_port=19123
    )
    server = M.default_server(cell, "resume")
    assert isinstance(server, MirixServer)
    assert server.mode == "resume"
    assert (server.spec.cell_id, server.spec.attempt, server.spec.port) == (
        cell.cell_id,
        cell.attempt,
        19123,
    )
    assert server.spec.cell_dir == cell.cell_dir
    assert server.base_url == "http://127.0.0.1:19123"


def test_the_method_uses_this_checkouts_mirix_client_and_renderer():
    """MIRIX's own client and M7.5 renderer are imported, never re-implemented: the
    injection is exactly ``mirix.client.render.render_retrieved_memories``' text."""
    from mirix.client.remote_client import MirixClient
    from mirix.client.render import render_retrieved_memories

    library = M._load_library()
    assert issubclass(library.client_class, MirixClient)
    response = {
        "memories": {
            "procedural": {
                "items": [
                    {
                        "id": "proc-1",
                        "name": "check twice",
                        "description": "d",
                        "instructions": "1. look",
                        "triggers": ["t1", "t2"],
                        "examples": [{"a": 1}],
                    }
                ]
            }
        }
    }
    assert library.render(response) == render_retrieved_memories(response)
    assert library.render(response) == MirixClient.render_memories(response)
    assert library.render({"memories": {}}) == ""


def test_a_mirix_without_its_client_or_renderer_is_refused(monkeypatch):
    monkeypatch.setattr(M, "RENDER_MODULE", "mirix.client.no_such_renderer")
    with pytest.raises(M.MirixApiError, match="no_such_renderer"):
        M._load_library()
    monkeypatch.undo()
    monkeypatch.setattr(
        M, "REQUIRED_CLIENT_METHODS", (*M.REQUIRED_CLIENT_METHODS, "list_telepathy")
    )
    with pytest.raises(M.MirixApiError, match="lacks list_telepathy"):
        M._load_library()


def test_a_mirix_api_key_in_the_environment_is_refused(monkeypatch):
    """MirixClient falls back to MIRIX_API_KEY, which would authenticate as that key's
    client (another tenant) instead of the cell's client headers."""
    monkeypatch.setenv("MIRIX_API_KEY", "someone-elses-key")
    template = M.setup_template()
    with pytest.raises(M.MirixApiError, match="MIRIX_API_KEY"):
        M.MirixApi(
            M._load_library(),
            "http://127.0.0.1:9",
            client=template["client"],
            organization=template["organization"],
        )


def test_the_client_never_retries_and_never_creates_users_on_its_own():
    template = M.setup_template()
    api = M.MirixApi(
        M._load_library(),
        "http://127.0.0.1:9",
        client=template["client"],
        organization=template["organization"],
    )
    client = api.client
    assert client.max_retries == 0 and client._client._transport._max_retries == 0
    assert client.api_key is None
    assert client._client.headers["X-Client-ID"] == "skill-bench-client"
    assert client._client.headers["X-Org-ID"] == "skill-bench-org"
    assert asyncio.run(client._ensure_user_exists("anyone")) is None
    asyncio.run(api.aclose())


def test_mirix_is_the_registered_s0_implementation():
    assert method_class("mirix") is M.MirixMethod
    assert M.MirixMethod.signal_level == "S0"
    template = M.setup_template()
    assert (
        template["memory_types"] == ["procedural"] and template["session_tag"] == "task"
    )
    assert M.without_secrets({"a": {"api_key": "x", "b": 1}, "api_key": "y"}) == {
        "a": {"b": 1}
    }


# --- the whole contract through the rollout's own exercise --------------------------------------


@pytest.mark.asyncio
async def test_the_contract_exercise_with_learning(tmp_path, gateway):
    servers = StubMirixServers()
    cell = mirix_cell(tmp_path, gateway, "alfworld")
    report = await exercise_method(lambda: method_for(servers), cell)
    assert [s["id"] for s in report.snapshot["skills"]] == ["proc-0001"]
    assert report.resumed_snapshot == report.snapshot
    assert all("stub skill 1" in i.text for i in report.test_injections)


# --- the scripted MIRIX model's skill-feedback replies (tests/mirix_model.py) ----------------


def _distiller_body(transcript: str, *, served: int = 0) -> dict[str, Any]:
    """A distiller request as MIRIX composes a whole session (its own label and
    instructions): header, metadata, the served skills' block, the transcript."""
    from mirix.services.distill_transcript import (
        WHOLE_INSTRUCTION,
        WHOLE_INSTRUCTION_WITH_FEEDBACK,
        WHOLE_TRANSCRIPT_LABEL,
    )

    from evals.skill_bench.tests.mirix_model import DISTILLER_PROMPT_HEAD

    skills = "".join(
        f"[{n}] name: skill {n} (version 1; served 1 time)\nsteps\n\n"
        for n in range(1, served + 1)
    )
    block = f"<skills_served>\n{skills}</skills_served>\n" if served else ""
    instruction = WHOLE_INSTRUCTION_WITH_FEEDBACK if served else WHOLE_INSTRUCTION
    text = (
        "agent_id: agent-1\nsession_id: t05-item\nsession: 4 message(s)\n"
        f"{block}\n{WHOLE_TRANSCRIPT_LABEL}\n{transcript}\n\n{instruction}"
    )
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": DISTILLER_PROMPT_HEAD + "."},
            {"role": "user", "content": text},
        ],
    }


TRANSCRIPT = "\n".join(
    [
        "user: Your task is to: put a clean mug in the cabinet.",
        "Admissible actions: go to countertop 1, look around the room",
        "assistant: <think>t</think><action>take mug 1 from countertop 1</action>",
        "user: You pick up the mug 1 from the countertop 1.",
        "Admissible actions: go to countertop 1, look around the room",
        "assistant: I took the mug, the task is done now",
    ]
)


def test_the_scripted_distiller_quotes_the_transcript_it_was_shown():
    """With ``p16_replies`` the scripted distiller answers with the
    reply object, its quotes copied from the transcript MIRIX rendered: the agent's
    first line, and the environment's last line that neither the task statement nor
    an agent message holds. A served session gets one feedback entry per served skill
    (followed quoting the agent, effect the environment); any other session a failed
    outcome that only the agent's words show."""
    from evals.skill_bench.tests import mirix_model as MM

    assert MM.transcript_quotes(
        _distiller_body(TRANSCRIPT)["messages"][1]["content"]
    ) == (
        "<think>t</think><action>take mug 1 from countertop 1</action>",
        "You pick up the mug 1 from the countertop 1.",
    )
    model = MM.MirixModel(p16_replies=True)
    served = json.loads(
        model("chat/completions", _distiller_body(TRANSCRIPT, served=2)).json[
            "choices"
        ][0]["message"]["content"]
    )
    assert served["outcome"] == "succeeded"
    assert (
        served["outcome_evidence"] == '"You pick up the mug 1 from the countertop 1."'
    )
    assert [f["skill"] for f in served["skill_feedback"]] == ["1", "2"]
    assert served["skill_feedback"][0] == {
        "skill": "1",
        "followed": "yes",
        "followed_evidence": '"<think>t</think><action>take mug 1 from countertop 1'
        '</action>"',
        "effect": "helped",
        "effect_evidence": '"You pick up the mug 1 from the countertop 1."',
    }
    unserved = json.loads(
        model("chat/completions", _distiller_body(TRANSCRIPT)).json["choices"][0][
            "message"
        ]["content"]
    )
    assert unserved["outcome"] == "failed" and "skill_feedback" not in unserved
    assert (
        unserved["outcome_evidence"] == served["skill_feedback"][0]["followed_evidence"]
    )
    assert [g["served"] for g in model.p16_given[:2]] == [2, 0]
    assert {g["session_id"] for g in model.p16_given} == {"t05-item"}
    # Nothing to quote (an episode whose model calls got no reply): unclear values.
    silent = "user: Your task is to: put a clean mug in the cabinet.\nuser: again"
    assert MM.transcript_quotes(_distiller_body(silent)["messages"][1]["content"]) == (
        None,
        None,
    )
    nothing = json.loads(
        model("chat/completions", _distiller_body(silent, served=1)).json["choices"][0][
            "message"
        ]["content"]
    )
    assert (nothing["outcome"], nothing["outcome_evidence"]) == ("unclear", "")
    assert nothing["skill_feedback"] == [
        {
            "skill": "1",
            "followed": "unclear",
            "followed_evidence": "",
            "effect": "unclear",
            "effect_evidence": "",
        }
    ]
    # The default stays the legacy bare list (no outcome, no feedback).
    legacy = MM.MirixModel()("chat/completions", _distiller_body(TRANSCRIPT, served=1))
    assert isinstance(json.loads(legacy.json["choices"][0]["message"]["content"]), list)


def test_the_scripted_pool_curator_can_cite_without_a_verified_quote():
    """B12 for the harness: with ``unverified_citation`` the pool curator's first
    write of a batch without earlier lessons also cites its fourth experience with a
    quote no skill holds; every curator batch's payload is kept once. The usage
    section's heading the tests look for is MIRIX's."""
    from mirix.services.skill_experience_curator import SKILL_USAGE_HEADING

    from evals.skill_bench.tests import mirix_model as MM

    assert MM.USAGE_SECTION_HEAD == SKILL_USAGE_HEADING
    payload = (
        MM.CURATOR_BATCH_HEAD
        + "\n"
        + "".join(f'<experience id="e{n}">x</experience>\n' for n in range(5))
    )
    body = {
        "model": "m",
        "tools": [{"function": {"name": MM.CURATOR_FINISH_TOOL}}],
        "messages": [
            {"role": "system", "content": "curator"},
            {"role": "user", "content": payload},
        ],
    }
    for flag, expected in ((False, ["e0"]), (True, ["e0", "e3"])):
        model = MM.MirixModel(pool_curator=True, unverified_citation=flag)
        reply = model("chat/completions", copy.deepcopy(body))
        call = reply.json["choices"][0]["message"]["tool_calls"][0]["function"]
        cited = json.loads(call["arguments"])["experience_ids"]
        assert [c["id"] for c in cited] == expected
        assert cited[0]["quote"] in MM.SKILL["instructions"]
        if flag:
            assert cited[1]["quote"] == MM.UNVERIFIED_QUOTE
            assert MM.UNVERIFIED_QUOTE not in json.dumps(MM.SKILL)
        model("chat/completions", copy.deepcopy(body))
        assert model.curator_payloads == [payload]


# --- a real MIRIX: this checkout's server on the compose Postgres, no model -------------------


def _postgres_up() -> bool:
    from evals.skill_bench.infra.mirix_server import PostgresAdmin

    admin = PostgresAdmin()
    try:
        with socket.create_connection((admin.host, admin.port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def real_mirix(tmp_path: Path) -> Iterator[tuple[str, MirixModel]]:
    """A real gateway over the fake upstream, scripted as MIRIX's model (``mirix_model``).
    The cell databases the test created are dropped afterwards."""
    from evals.skill_bench.infra.mirix_server import (
        PostgresAdmin,
        cell_db_name,
        drop_cell_database,
    )

    if not _postgres_up():
        pytest.skip(
            "skill_bench Postgres not reachable "
            "(docker compose -f evals/skill_bench/infra/compose.yaml up -d)"
        )
    model = MirixModel()
    try:
        with serve_gateway(tmp_path, FakeUpstream(responder=model)) as (_gw, url):
            yield url, model
    finally:
        for cell_id in _REAL_CELLS.pop(str(tmp_path), []):
            asyncio.run(drop_cell_database(PostgresAdmin(), cell_db_name(cell_id, 1)))


#: tmp_path -> the cell ids a real-MIRIX test created (their databases are dropped).
_REAL_CELLS: dict[str, list[str]] = {}


async def learned_real_cell(
    tmp_path: Path, url: str
) -> tuple[CellContext, dict[str, Any], str]:
    """3 learning items through a real MIRIX (the method starts the cell's server, on a
    database of its own), the freeze, and the frozen snapshot; the server is stopped
    again. Returns the cell, the frozen snapshot and the cell database's name.

    The first item's first observation after a step carries a NUL character (G1: the
    harness forwards the environment's output byte for byte). Postgres cannot store it;
    MIRIX stores it as U+FFFD and says so in the add response's ``warnings`` instead of
    failing the write, which would restart the attempt as INFRA every time."""
    import secrets

    from evals.skill_bench.infra.mirix_server import cell_db_name

    cell = replace(
        mirix_cell(tmp_path, url, mirix_port=find_free_port()),
        cell_id=f"alfworld-realmirix-{secrets.token_hex(4)}",
    )
    _REAL_CELLS.setdefault(str(tmp_path), []).append(cell.cell_id)
    items = learning_items("alfworld")
    observation = items[0].messages[2]
    observation["content"] = f"{observation['content']}\x00\x00 (binary tail)"
    method = M.MirixMethod()
    await method.setup(cell)
    try:
        await learn(method, items)
        await method.finalize()
        await method.wait_until_idle(600.0)
        frozen = await method.snapshot()
    finally:
        await method.close()
    return cell, frozen, cell_db_name(cell.cell_id, cell.attempt)


def rendered(skill: Mapping[str, Any]) -> str:
    """MIRIX's own rendering of one retrieved skill (M7.5)."""
    from mirix.client.render import render_retrieved_memories

    return render_retrieved_memories({"memories": {"procedural": {"items": [skill]}}})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_against_a_real_mirix_server(tmp_path, real_mirix):
    """3 learning + 2 test items against this checkout's real MIRIX, which learns through
    its own distiller and curator (the fake upstream plays their model): the skill exists
    after the freeze with its history and stored version, the frozen memory resumes
    read-only (the test phase writes nothing but retrieval log rows, checked in the cell
    database itself), and retrieval injects MIRIX's own rendering of the skill."""
    url, model = real_mirix
    cell, frozen, db_name = await learned_real_cell(tmp_path, url)

    # The freeze: one skill, learned by one flush run from the three sessions (M1.7).
    (skill,) = frozen["skills"]
    assert skill["name"] == SKILL["name"]
    assert skill["instructions"] == SKILL["instructions"]
    history = frozen["skill_history"][skill["id"]]["versions"]
    assert [v["change_type"] for v in history] == ["create"]
    version = frozen["skill_versions"][skill["id"]]
    assert (
        set(version) == {"1"} and version["1"]["instructions"] == SKILL["instructions"]
    )
    (run,) = frozen["learning_runs"]
    assert (run["trigger"], run["status"]) == ("flush", "succeeded")
    assert sorted(run["session_ids"]) == sorted(
        i.view.task_uid for i in learning_items("alfworld")
    )
    assert sum(e["experiences"] for e in run["distill_metrics"]) == 3
    assert run["llm_usage"]["calls"] >= 4  # three distillations and the curator
    assert model.kinds["distiller"] == 3 and model.kinds["curator"] >= 2
    assert frozen["counters"]["sessions_closed"] == 3
    # The NUL in the first item's observation was stored as U+FFFD, with one warning.
    assert frozen["counters"]["add_warnings"] == 1

    # The test phase: a resumed MIRIX (a server start of its own), read only,
    # retrieving the skill.
    before = await database_digest(db_name)
    logs_before = await retrieval_log_rows(db_name)
    resumed = M.MirixMethod()
    await resumed.setup(replace(cell, phase="test"))
    try:
        assert await resumed.snapshot() == frozen
        views = task_views("alfworld")
        injections = await asyncio.gather(*(resumed.before_task(v) for v in views))
        assert await resumed.snapshot() == frozen
    finally:
        await resumed.close()
    assert await database_digest(db_name) == before  # H10.4 below the API
    assert await retrieval_log_rows(db_name) == logs_before + 2
    for injection in injections:
        assert injection.text == rendered(skill)
        assert injection.item_ids == (skill["id"],)
    records = [
        json.loads(line)
        for line in (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()
    ]
    tested = [r for r in records if r["frozen"]]
    assert len(tested) == 2 and all(r["failure"] is None for r in tested)
    assert all(
        r["evidence"]["extraction_status"] == "ok"
        and r["evidence"]["embedding_status"] == "ok"
        and r["retrieval_scores"][skill["id"]]["cosine"] == pytest.approx(1.0)
        for r in tested
    )
    # The real row's ranking trace (M7.4): every kept key, the returned skill in it.
    assert all(
        set(r["procedural"]) == set(M.PROCEDURAL_TRACE_KEYS)
        and r["procedural"]["returned_ids"] == [skill["id"]]
        for r in tested
    )
    # M3.2 / H7.2: MIRIX made each model call once (MIRIX_LLM_MAX_RETRIES=0): no
    # mirix-role call repeats an earlier one.
    calls = ledger_calls(tmp_path, cell.run_id)
    assert [r for r in calls if r["role"] == "mirix"]
    assert not [r for r in calls if r["role"] == "mirix" and r.get("repeat_of")]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
async def test_a_real_mirix_distiller_sees_the_agent_context_and_the_skill_served(
    tmp_path, real_mirix, benchmark
):
    """The agent context and MIRIX's skill feedback end to end, the same for ALFWorld
    and
    WebShop (WebShop's MIRIX module matches ALFWorld's). Six learning items through this
    checkout's real MIRIX, each task view carrying the benchmark's agent context: the
    first five close, MIRIX's threshold run learns the skill from them, the sixth is
    served that skill, and the freeze's flush distills it. MIRIX stored the agent context
    on every learning session; the distiller's transcript of each one shows it verbatim
    in ``<agent_context>``; the sixth's also shows the skill served, at the version
    served, in ``<skills_served>``, and MIRIX recorded its per-skill feedback row.

    The scripted distiller answers with the skill-feedback reply object, its quotes
    copied from each transcript: the sixth session's feedback entry (followed quoting
    the agent, effect quoting the environment) is stored with its values; the first five
    state a failure only the agent's words show, which MIRIX stores as unclear. The
    flush run's curator batch shows the served skill's usage record."""
    import secrets

    from evals.skill_bench.infra.mirix_server import cell_db_name

    url, model = real_mirix
    model.p16_replies = True
    context = agent_context_of(benchmark)
    cell = replace(
        mirix_cell(tmp_path, url, benchmark, mirix_port=find_free_port()),
        cell_id=f"{benchmark}-realmirix-a2-{secrets.token_hex(4)}",
    )
    _REAL_CELLS.setdefault(str(tmp_path), []).append(cell.cell_id)
    items = with_agent_context(learning_items(benchmark, 6), context)
    # The sixth (the one served the skill) is the contract item with the agent's
    # replies and the environment's answers to quote; the other shape (no reply at
    # all) is among the first five.
    items[4], items[5] = items[5], items[4]
    method = M.MirixMethod()
    await method.setup(cell)
    try:
        injections = await learn(method, items, idle_s=600.0)
        await method.finalize()
        await method.wait_until_idle(600.0)
        frozen = await method.snapshot()
    finally:
        await method.close()
    db_name = cell_db_name(cell.cell_id, cell.attempt)
    uids = [i.view.task_uid for i in items]

    # MIRIX learned the skill from the first five sessions (its threshold run) and
    # served it to the sixth.
    (skill,) = frozen["skills"]
    assert skill["name"] == SKILL["name"]
    assert [i.item_ids for i in injections[:5]] == [()] * 5
    assert injections[5].item_ids == (skill["id"],)
    assert injections[5].text == rendered(skill)
    # The fifth close reached MIRIX's session threshold (its run's trigger: "close").
    runs = {r["trigger"]: r for r in frozen["learning_runs"]}
    assert sorted(runs) == ["close", "flush"]
    assert {r["status"] for r in runs.values()} == {"succeeded"}
    assert sorted(runs["close"]["session_ids"]) == sorted(uids[:5])
    assert runs["flush"]["session_ids"] == [uids[5]]

    # MIRIX stored the agent context on every learning session.
    stored = await session_agent_contexts(db_name)
    assert {uid: stored.get(uid) for uid in uids} == {uid: context for uid in uids}

    # The distiller saw it verbatim, and the sixth session's served skill.
    block = f"<agent_context>\n{context}\n</agent_context>"
    header = f"[1] name: {SKILL['name']} (version 1; served 1 time)"
    for index, uid in enumerate(uids):
        (shown,) = model.distiller_input(uid)
        assert block in shown, uid
        if index < 5:
            assert "<skills_served>" not in shown
        else:
            served = shown.split("<skills_served>\n", 1)[1].split(
                "\n</skills_served>", 1
            )[0]
            assert served.splitlines()[0] == header
            assert SKILL["instructions"].splitlines()[1] in served
            # The served skill comes first, then the agent context, then the session.
            assert shown.index("<skills_served>") < shown.index(block)
            assert shown.index(block) < shown.index(
                str(items[5].messages[0]["content"])[:40]
            )
    # The per-skill feedback of the sixth session: one row for the skill served,
    # stored with the values the distiller gave, each verified by its quote (followed
    # by the agent's own words, effect by the environment's).
    given = {g["session_id"]: g for g in model.p16_given}
    quoting = [bool(given[uid]["agent_quote"]) for uid in uids]
    assert quoting == [True, False, True, False, False, True]
    assert given[uids[5]]["environment_quote"], model.distiller_input(uids[5])
    rows = await skill_feedback_rows(db_name)
    assert [
        (r["session_id"], r["skill_id"], r["skill_name"], r["skill_version_no"])
        for r in rows
    ] == [(uids[5], skill["id"], SKILL["name"], 1)]
    (row,) = rows
    assert (row["followed"], row["effect"], row["session_outcome"]) == (
        "yes",
        "helped",
        "succeeded",
    )
    assert (row["stated_followed"], row["stated_effect"]) == (None, None)
    assert (row["followed_source"], row["effect_source"]) == (
        "agent_only",
        "user_or_tool",
    )
    # S2: those of the first five with an agent reply stated a failure only the
    # agent's words showed (the others had nothing to quote: unclear); both stored
    # unclear. The sixth's success the environment showed: stored.
    distilled = {
        str(e["session_id"]): e
        for run in frozen["learning_runs"]
        for e in run["distill_metrics"]
        if e.get("status") == "ok"
    }
    assert [distilled[uid]["session_outcome"] for uid in uids] == ["unclear"] * 5 + [
        "succeeded"
    ]
    assert [distilled[uid]["outcomes_self_reported"] for uid in uids] == [
        int(q) for q in quoting[:5]
    ] + [0]
    assert [given[uid]["outcome"] for uid in uids] == [
        "failed" if q else "unclear" for q in quoting[:5]
    ] + ["succeeded"]
    assert distilled[uids[5]]["skill_feedback"]["skills"] == [
        {
            "skill_id": skill["id"],
            "name": SKILL["name"],
            "version_no": 1,
            "followed": "yes",
            "effect": "helped",
        }
    ]
    # S3: the flush run's batch shows the served skill's usage record; the threshold
    # run's batch (nothing served yet) shows none.
    first, flush = model.curator_payloads
    assert USAGE_SECTION_HEAD not in first
    usage = flush.split(USAGE_SECTION_HEAD, 1)[1]
    assert f"Skill {SKILL['name']} (ID: {skill['id']}), served in this batch's" in usage
    assert "- served in 1 distilled session(s); most recent: " in usage
    assert "followed yes 1/" in usage and "effect helped 1/" in usage
    summary = M.learning_context(frozen["learning_runs"])
    assert summary is not None
    assert (
        summary["with_agent_context"],
        summary["served_sessions"],
        summary["feedback"]["followed"],
        summary["feedback"]["effect"],
        summary["usage"]["shown"],
    ) == (6, 1, {"yes": 1}, {"helped": 1}, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_real_mirix_retrieval_that_lost_its_infrastructure_is_never_scored(
    tmp_path, real_mirix
):
    """On a real MIRIX: an upstream 5xx on the ``mirix`` role (topic
    extraction) or the ``embedding`` role (query vector) during the test phase makes
    MIRIX retrieve without it; the method reads the retrieval's log row and raises INFRA
    ``mirix_retrieval`` instead of injecting the degraded result. The item's other
    retrievals, and the same item once the upstream is back, inject the skill."""
    url, model = real_mirix
    cell, frozen, _db = await learned_real_cell(tmp_path, url)
    (skill,) = frozen["skills"]
    base = task_views("alfworld")
    view, other = (
        replace(
            v,
            first_messages=(
                {
                    "role": "user",
                    "content": f"Question: What were {what}, in millions of dollars?",
                },
            ),
        )
        for v, what in zip(base, ("total receipts in 1943", "customs in 1950"))
    )
    # A text of ``view``'s conversation (MIRIX's extraction input holds it verbatim).
    marker = "total receipts in 1943"
    resumed = M.MirixMethod()
    await resumed.setup(replace(cell, phase="test"))
    try:
        start = len(ledger_calls(tmp_path, cell.run_id))
        model.fail_extraction_for = {marker}
        with pytest.raises(MirixRetrievalDegraded) as lost:
            await resumed.before_task(view)
        failure = classify_exception(lost.value)
        assert (failure.kind, failure.code, failure.role, failure.status) == (
            FailureKind.INFRA,
            MIRIX_RETRIEVAL,
            "mirix",
            502,
        )
        new = ledger_calls(tmp_path, cell.run_id)[start:]
        extraction = [r for r in new if r["role"] == "mirix"]
        # One MIRIX call (no SDK retry), which the gateway retried upstream and lost.
        assert len(extraction) == 1 and extraction[0]["status"] != "ok"
        assert extraction[0]["retries"] == protocol.GATEWAY_MAX_RETRIES
        assert (await resumed.before_task(other)).text == rendered(skill)
        model.fail_extraction_for = set()
        assert (await resumed.before_task(view)).text == rendered(skill)

        model.fail_embeddings = True
        with pytest.raises(MirixRetrievalDegraded) as lost:
            await resumed.before_task(other)
        model.fail_embeddings = False
        failure = classify_exception(lost.value)
        assert (failure.kind, failure.code, failure.role) == (
            FailureKind.INFRA,
            MIRIX_RETRIEVAL,
            "embedding",
        )
        assert (await resumed.before_task(other)).text == rendered(skill)
        assert await resumed.snapshot() == frozen
    finally:
        await resumed.close()
    records = [
        json.loads(line)
        for line in (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()
        if json.loads(line)["frozen"]
    ]
    assert [bool(r["failure"]) for r in records] == [True, False, False, True, False]
    lost_extraction, lost_vector = records[0], records[3]
    assert lost_extraction["evidence"]["extraction_status"] == "fallback"
    assert "502" in lost_extraction["evidence"]["extraction_error"]
    assert lost_extraction["injection_sha256"] is None
    assert lost_vector["evidence"]["embedding_status"] == "failed"
    # MIRIX's embedding retries (M3.5) are in the row's retry events.
    assert {e["kind"] for e in lost_vector["evidence"]["retry_events"]} == {"embedding"}


# --- X3 / X13: what an outage cost MIRIX's learning restarts the attempt --------------------


def test_learning_problems_and_unsettled_sessions_read_mirix_run_records():
    runs = [
        {
            "id": "r1",
            "created_at": "2026-01-01T00:00:01",
            "status": "succeeded",
            "session_ids": ["a", "b", "c"],
            "distill_metrics": [
                {"session_id": "a", "status": "ok"},
                {"session_id": "b", "status": "failed"},  # a counted (model) failure
                {"session_id": "c", "status": "deferred", "failure_kind": "infra"},
            ],
            "curator_metrics": [
                {"batch_id": "r1:b0", "error_kind": "db", "error": "x"}
            ],
        },
        {
            "id": "r2",
            "created_at": "2026-01-01T00:00:02",
            "status": "failed",
            "error": "lease_expired",  # MIRIX's LEASE_EXPIRED_ERROR
            "session_ids": ["d"],
        },
        {
            "id": "r3",
            "created_at": "2026-01-01T00:00:03",
            "status": "succeeded",
            "session_ids": ["b"],
            "distill_metrics": [{"session_id": "b", "status": "failed_permanently"}],
            "curator_metrics": [{"batch_id": "r3:b0", "error_kind": "model"}],
        },
    ]
    problems = M.learning_problems(runs)
    assert len(problems) == 3
    assert "deferred session c" in problems[0] and "r1:b0" in problems[1]
    assert problems[2].startswith("[infra] learning run r2 failed (lease_expired")
    assert [k for k, _ in M.learning_problem_kinds(runs)] == [FailureKind.INFRA] * 3
    assert M.unsettled_sessions(runs, ["a", "b", "c", "d", "e"]) == {
        "c": "deferred",
        "d": "failed",
        "e": "unclaimed",
    }


#: A curator batch's tool failures as MIRIX records them (``mirix/agent/tool_failures.py``).
DB_TOOL_FAILURE = {
    "tool": "list_memory_within_timerange",
    "stage": "execution",
    "exception": "DBAPIError",
    "error": "DBAPIError: (sqlalchemy.dialects.postgresql.asyncpg.Error) invalid input "
    "for query argument $1: datetime.datetime(2025, 12, 31, 0, 0, tz... (can't "
    "subtract offset-naive and offset-aware datetimes)",
    "count": 11,
}
LOST_CONNECTION = {
    "tool": "skill_read",
    "stage": "execution",
    "exception": "OperationalError",
    "error": "OperationalError: (asyncpg.exceptions.ConnectionDoesNotExistError) "
    "connection was closed in the middle of operation",
    "count": 1,
}
ORDERING_GUARD = {
    "tool": "finish_memory_update",
    "stage": "execution",
    "exception": "AssertionError",
    "error": "AssertionError: finish_memory_update must be the last tool call",
    "count": 4,
}
MODEL_MISTAKES = [
    {
        "tool": "list_memory_within_timerange",
        "stage": "execution",
        "exception": "ValueError",
        "error": "ValueError: Invalid isoformat string: 'string'",
        "count": 6,
    },
    {
        "tool": "search_in_memory",
        "stage": "execution",
        "exception": "ValidationError",
        "error": "ValidationError: 1 validation error for SearchArgs",
        "count": 1,
    },
    {
        "tool": "skill_create",
        "stage": "arguments",
        "exception": "JSONDecodeError",
        "error": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        "count": 1,
    },
    {
        "tool": "skill_create",
        "stage": "validation",
        "exception": None,
        "error": "Validation error: 'name' cannot be empty.",
        "count": 5,
    },
    {
        "tool": "skill_nope",
        "stage": "unknown_tool",
        "exception": None,
        "error": "No function named skill_nope",
        "count": 1,
    },
]


def curated_runs(*failures: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One succeeded learning run whose curator batch settled after ``failures``."""
    return [
        {
            "id": "r1",
            "status": "succeeded",
            "session_ids": ["a"],
            "distill_metrics": [{"session_id": "a", "status": "ok"}],
            "curator_metrics": [
                {
                    "batch_id": "r1:b0",
                    "status": "settled",
                    "outcome": "consume",
                    "stop_reason": "finish",
                    "failed_tool_calls": sum(int(f["count"]) for f in failures),
                    "tool_failures": [dict(f) for f in failures],
                    "tool_failures_omitted": 0,
                }
            ],
        }
    ]


def test_tool_exceptions_inside_the_curator_are_warnings_never_invalid_attempts():
    """B31: MIRIX answers an exception inside a curator tool to the model as a failed
    tool result, so MIRIX's own database error (K2, a DBAPIError in
    list_memory_within_timerange) never failed a batch or a run and the attempt check
    never saw it. MIRIX now records why each tool call failed; the harness warns about
    MIRIX's own and infrastructure exceptions, not about the model's arguments, and
    the attempt stays valid."""
    runs = curated_runs(
        DB_TOOL_FAILURE, LOST_CONNECTION, ORDERING_GUARD, *MODEL_MISTAKES
    )
    assert M.learning_problem_kinds(runs) == []  # the attempt is accepted
    found = M.curator_tool_failures(runs)
    assert [label for label, _ in found] == ["mirix", "infra", "mirix"]
    warnings = M.learning_warnings(runs)
    assert warnings[0].startswith("[mirix] run r1 curator batch r1:b0: ")
    assert "11 call(s) of list_memory_within_timerange" in warnings[0]
    assert "DBAPIError" in warnings[0] and "offset-naive" in warnings[0]
    assert warnings[1].startswith("[infra] ") and "OperationalError" in warnings[1]
    # MIRIX's ordering guard fires after the tool ran: MIRIX's own (until B22).
    assert warnings[2].startswith("[mirix] ") and "AssertionError" in warnings[2]
    # The model's own mistakes are no warning, and neither is a record without tool
    # failures (a MIRIX before them, e.g. older snapshots).
    assert M.learning_warnings(curated_runs(*MODEL_MISTAKES)) == []
    older = [
        {
            "id": "r1",
            "status": "succeeded",
            "curator_metrics": [
                {
                    "batch_id": "r1:b0",
                    "failed_tool_calls": 4,
                    "rejected_tool_calls": 4,
                    "rejected_by_kind": {},
                }
            ],
        }
    ]
    assert M.learning_warnings(older) == [] and M.learning_problem_kinds(older) == []
    odd = [
        {
            "id": "r1",
            "curator_metrics": [
                {"tool_failures": [None, "x", {"stage": "execution"}, {"count": 1}]},
                {"tool_failures": "not a list"},
            ],
        }
    ]
    assert M.learning_warnings(odd) == []


def test_mirix_s_own_argument_verdict_decides_over_the_class_name():
    """B31: a TypeError for a missing argument, a ValueError subclass
    (dateutil's ParserError) or an unknown id (NoResultFound) read as MIRIX's own
    failure by name. MIRIX records ``argument_error`` on the exception object; the
    harness follows it, and matches names only for a record without it."""
    judged = [
        {
            "tool": "list_memory_within_timerange",
            "stage": "execution",
            "exception": "ParserError",
            "error": "ParserError: Unknown string format: string",
            "count": 2,
            "argument_error": True,
        },
        {
            "tool": "skill_read",
            "stage": "execution",
            "exception": "NoResultFound",
            "error": "NoResultFound: no skill proc-x",
            "count": 1,
            "argument_error": True,
        },
        {
            "tool": "skill_edit",
            "stage": "execution",
            "exception": "TypeError",
            "error": "TypeError: unsupported operand type(s)",
            "count": 1,
            "argument_error": False,
        },
        # MIRIX's verdict wins over the name: a ValueError it judged its own.
        {
            "tool": "skill_edit",
            "stage": "execution",
            "exception": "ValueError",
            "error": "ValueError: bad state",
            "count": 1,
            "argument_error": False,
        },
    ]
    found = M.curator_tool_failures(curated_runs(*judged))
    assert [
        (label, "TypeError" in text, "ValueError" in text) for label, text in found
    ] == [
        ("mirix", True, False),
        ("mirix", False, True),
    ]
    # Without the verdict (an older MIRIX) the names decide, as before.
    legacy = [{k: v for k, v in f.items() if k != "argument_error"} for f in judged]
    assert [
        text.split(" raised ")[1].split()[0]
        for _, text in M.curator_tool_failures(curated_runs(*legacy))
    ] == [
        "ParserError",
        "NoResultFound",
        "TypeError",
    ]


def method_with_state() -> M.MirixMethod:
    """A method whose learning checks run without a MIRIX server: ``_check_learning``
    keeps its refusal streak and accepted problems in the method's state (the rules
    triage), so the state must exist as setup() leaves it."""
    method = M.MirixMethod()
    method._state = M.MirixState(
        cell_id="alfworld-luna-mirix",
        attempt=1,
        benchmark="alfworld",
        model="luna",
        org_id="org",
        client_id="client",
        user_id="user",
        meta_agent_id="meta",
        config={},
    )
    return method


def test_the_method_logs_each_curator_tool_warning_once(caplog):
    import logging

    method = method_with_state()
    runs = curated_runs(DB_TOOL_FAILURE, *MODEL_MISTAKES)
    with caplog.at_level(logging.WARNING, logger=M.__name__):
        method._check_learning(runs)  # accepted: no raise
        method._check_learning(runs)  # the next item's idle check lists it again
    logged = [r.getMessage() for r in caplog.records if "DBAPIError" in r.getMessage()]
    assert len(logged) == 1 and logged[0].startswith(
        "MIRIX curator tool exception: [mirix]"
    )
    # A real learning problem still raises, after the warnings are logged.
    broken = curated_runs(LOST_CONNECTION)
    broken[0]["curator_metrics"][0].update(error_kind="db", error="OperationalError")
    with pytest.raises(LearningAttemptInvalid):
        method._check_learning(broken)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stub_options", "message"),
    [
        ({"session_outcomes": {"UID": ["deferred"]}}, "deferred session"),
        ({"curator_errors": ["infra"]}, "curator batch"),
    ],
)
async def test_an_infrastructure_loss_in_mirix_learning_invalidates_the_attempt(
    tmp_path, gateway, stub_options, message
):
    """X13: MIRIX deferred a session or stopped a curator batch on an infrastructure
    error: the memory is not what the attempt would have learned; H8 restarts it."""
    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    options = {
        k: ({uid: v["UID"]} if k == "session_outcomes" else v)
        for k, v in stub_options.items()
    }
    servers = StubMirixServers(threshold=1, **options)
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        with pytest.raises(M.LearningAttemptInvalid, match=message) as info:
            await learn(method, items)
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.INFRA, "mirix_learning")


@pytest.mark.asyncio
async def test_the_freeze_waits_until_every_learning_session_is_settled(
    tmp_path, gateway
):
    """X3: a session with a counted failure is taken up again by another flush before the
    memory is frozen; one MIRIX gave up on (failed_permanently) is settled."""
    items = learning_items("alfworld", 2)
    servers = StubMirixServers(
        threshold=99,
        session_outcomes={
            items[0].view.task_uid: ["failed", "ok"],
            items[1].view.task_uid: ["failed_permanently"],
        },
    )
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, items)
        await method.finalize()
        # Two runs, then a flush that finds nothing left to learn.
        assert method.state.frozen and method.state.flush_rounds == 3
        assert [r["trigger"] for r in stub.runs.values()] == ["flush", "flush"]
        assert method.state.flush is not None
        assert method.state.flush["learning_run_id"] is None
        assert method.state.flush["pending_experiences"] == 0
        assert method.state.closed_sessions == [i.view.task_uid for i in items]
    finally:
        await method.close()


@pytest.mark.asyncio
async def test_a_session_that_keeps_failing_stops_the_freeze(tmp_path, gateway):
    items = learning_items("alfworld", 1)
    servers = StubMirixServers(
        threshold=99,
        session_outcomes={
            items[0].view.task_uid: ["failed"] * (M.MAX_FLUSH_ROUNDS + 1)
        },
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        await learn(method, items)
        with pytest.raises(M.MirixApiError, match="counted failures"):
            await method.finalize()
        assert not method.state.frozen
        assert method.state.flush_rounds == M.MAX_FLUSH_ROUNDS
    finally:
        await method.close()


@pytest.mark.asyncio
async def test_a_flush_that_leaves_a_counted_failure_unclaimed_stops_the_freeze(
    tmp_path, gateway
):
    """After a counted failure the next flush must take the session up again
    (Q8: pending at once). A MIRIX whose flush starts no run then would have the memory
    frozen without that session's learning; the method raises instead (HARNESS)."""
    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    servers = StubMirixServers(
        threshold=99,
        session_outcomes={uid: ["failed", "ok"]},
        flush_claims=[True, False],
    )
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, items)
        with pytest.raises(M.MirixApiError, match="started no learning run") as info:
            await method.finalize()
        assert uid in str(info.value) and "counted failures" in str(info.value)
        assert not method.state.frozen and method.state.frozen_snapshot_sha256 is None
        assert method.state.flush_rounds == 2
        assert [r["status"] for r in stub.runs.values()] == ["succeeded"]
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert failure.kind is FailureKind.HARNESS


@pytest.mark.asyncio
async def test_a_flush_the_sweeper_preempted_is_repeated_not_a_harness_failure(
    tmp_path, gateway
):
    """MIRIX's sweeper claimed the session with a counted failure just before
    the flush, which then started no run, and its run failed the session again. MIRIX
    kept its claim rules (another run took the session up): the method flushes again
    instead of raising, and freezes once the session is learned."""
    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    servers = StubMirixServers(
        threshold=99,
        session_outcomes={uid: ["failed", "failed", "ok"]},
        sweep_before_flush=[False, True],
    )
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, items)
        await method.finalize()
        assert method.state.frozen and method.state.flush_rounds == 4
        runs = sorted(stub.runs.values(), key=lambda r: r["id"])
        assert [(r["trigger"], r["session_ids"]) for r in runs] == [
            ("flush", [uid]),
            ("sweep", [uid]),
            ("flush", [uid]),
        ]
        assert M.unsettled_sessions(list(stub.runs.values()), [uid]) == {}
    finally:
        await method.close()
    events = [
        json.loads(line)
        for line in (cell.work_dir / M.LEARNING_RECORD_FILE).read_text().splitlines()
    ]
    preempted = [e for e in events if e["event"] == "flush_preempted"]
    assert [(e["round"], e["runs"], e["sessions"]) for e in preempted] == [
        (2, [runs[1]["id"]], [uid])
    ]


@pytest.mark.asyncio
async def test_experiences_a_curator_batch_returned_are_curated_before_the_freeze(
    tmp_path, gateway
):
    """A curator batch that returned its experiences with a counted attempt
    leaves them waiting; the flush repeats (a curate-only run, no session) until MIRIX
    starts no run, so the frozen memory holds what MIRIX would have curated later."""
    items = learning_items("alfworld", 2)
    servers = StubMirixServers(threshold=99, curator_returns=[2, 1])
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, items)
        await method.finalize()
        runs = sorted(stub.runs.values(), key=lambda r: r["id"])
        assert [len(r["session_ids"]) for r in runs] == [2, 0, 0]  # 2 curate-only
        assert method.state.frozen and method.state.flush_rounds == 4
        assert method.state.flush is not None
        assert method.state.flush["learning_run_id"] is None
        assert method.state.flush["pending_experiences"] == 0
    finally:
        await method.close()
    flushes = [
        json.loads(line)
        for line in (cell.work_dir / M.LEARNING_RECORD_FILE).read_text().splitlines()
    ]
    waiting = [f["pending_experiences"] for f in flushes if f["event"] == "flush"]
    assert waiting == [0, 2, 1, 0]


@pytest.mark.asyncio
async def test_finalize_has_one_deadline_for_all_its_rounds(tmp_path, gateway):
    """Every round waits only for what is left of ``finalize_timeout_s``,
    and past it the method's own TimeoutError lists the runs still in flight."""
    items = learning_items("alfworld", 1)
    servers = StubMirixServers(
        threshold=99,
        run_delay_s=0.4,
        session_outcomes={items[0].view.task_uid: ["failed", "failed", "ok"]},
    )
    method = method_for(servers, finalize_timeout_s=1.0)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        await learn(method, items)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="still in flight") as info:
            await method.finalize()
        elapsed = time.monotonic() - started
    finally:
        await method.close()
    assert elapsed < 2.0  # not one full timeout per round
    assert "slr-" in str(info.value)
    assert classify_exception(info.value).code == IDLE_WAIT_TIMEOUT


@pytest.mark.asyncio
async def test_a_skill_the_curator_deleted_is_in_the_frozen_snapshot(tmp_path, gateway):
    """The listing has live skills only; the export adds each skill a run
    changed and MIRIX deleted, with its history and every version's content."""
    items = learning_items("alfworld", 2)
    servers = StubMirixServers(threshold=1, delete_skill_in_run={2: "proc-0001"})
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    try:
        await learn(method, items)
        await method.finalize()
        frozen = await method.snapshot()
    finally:
        await method.close()
    assert [s["id"] for s in frozen["skills"]] == ["proc-0002"]
    assert frozen["deleted_skills"] == ["proc-0001"]
    history = frozen["skill_history"]["proc-0001"]
    assert history["is_deleted"] is True
    assert [v["change_type"] for v in history["versions"]] == ["create", "delete"]
    assert set(frozen["skill_versions"]["proc-0001"]) == {"1", "2"}
    assert frozen["skills_without_vector"] == []


@pytest.mark.asyncio
async def test_a_skill_without_its_vector_is_never_frozen(tmp_path, gateway):
    """A live skill whose embedding failed (and its end-of-run re-embedding
    too) is invisible to gated retrieval, and the test-phase server's startup pass would
    write its vector into the frozen memory: the outage cost learning (INFRA)."""
    items = learning_items("alfworld", 1)
    servers = StubMirixServers(threshold=99, skills_without_vector={"proc-0001"})
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        await learn(method, items)
        with pytest.raises(M.LearningAttemptInvalid, match="proc-0001") as info:
            await method.finalize()
        assert not method.state.frozen
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.INFRA, "mirix_learning")


@pytest.mark.asyncio
async def test_a_vector_added_after_the_freeze_is_a_changed_memory(tmp_path, gateway):
    servers = StubMirixServers()
    cell, _frozen = await learned_cell(tmp_path, gateway, servers)
    stub = servers.stub_for(cell.work_dir)
    stub.skills_without_vector.add("proc-0001")  # what the frozen hash covers changed
    resumed = method_for(servers)
    with pytest.raises(FrozenMemoryMismatch):
        await resumed.setup(replace(cell, phase="test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 404])
async def test_a_retrieval_mirix_did_not_answer_is_a_refused_window(
    tmp_path, gateway, status
):
    """A retrieval MIRIX answered with a 5xx (INFRA) or a 4xx (HARNESS) is not
    scored, and its record gives the audit the window of the gateway calls it made."""
    servers = StubMirixServers()
    cell, _frozen = await learned_cell(tmp_path, gateway, servers)
    stub = servers.stub_for(cell.work_dir)
    resumed = method_for(servers)
    await resumed.setup(replace(cell, phase="test"))
    try:
        stub.fail_once[("POST", RETRIEVE)] = status
        with pytest.raises((MirixServerError, M.MirixApiError)):
            await resumed.before_task(task_views("alfworld")[0])
    finally:
        await resumed.close()
    records = [
        json.loads(line)
        for line in (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()
    ]
    last = records[-1]
    assert last["frozen"] is True and last["evidence"] is None
    assert last["procedural"] is None
    expected = "infra" if status >= 500 else "harness"
    assert last["failure"]["kind"] == expected and last["failure"]["role"] == "mirix"
    assert last["started_at"] <= last["at"]


# --- rule N1: a retrieval degraded by anything but the model is never scored --------------------

GATEWAY_502 = (
    "LLMServerError: INTERNAL_SERVER_ERROR: OpenAI API error: Error code: 502 - "
    "{'error': {'source': 'gateway', 'type': 'upstream_infra_failure', 'code': 502, "
    "'message': 'infrastructure failure persisted after 5 retries (last: http_502)', "
    "'retries': 5}}"
)


def _deadline_error(**error: Any) -> str:
    body = {
        "error": {
            "source": "gateway",
            "type": "deadline_exceeded",
            "code": 504,
            "deadline_s": 600.0,
            **error,
        }
    }
    return f"LLMServerError: INTERNAL_SERVER_ERROR: OpenAI API error: Error code: 504 - {body!r}"


@pytest.mark.parametrize(
    "evidence, expected",
    [
        ({"extraction_status": "ok", "embedding_status": "ok"}, None),
        ({"extraction_status": "skipped", "embedding_status": "skipped"}, None),
        (
            {"extraction_status": "fallback", "extraction_error": GATEWAY_502},
            ("infra", "mirix", 502),
        ),
        (  # MIRIX cut the text at 500 characters: the status alone decides
            {"extraction_status": "fallback", "extraction_error": GATEWAY_502[:120]},
            ("infra", "mirix", 502),
        ),
        (  # the model alone used the whole deadline: MIRIX's own fallback, scored
            {
                "extraction_status": "fallback",
                "extraction_error": _deadline_error(
                    reason="deadline_during_attempt",
                    retries=0,
                    upstream_attempts=1,
                    attempt_budget_s=599.5,
                ),
            },
            None,
        ),
        (
            {
                "extraction_status": "fallback",
                "extraction_error": _deadline_error(
                    reason="deadline_while_queued", retries=0, upstream_attempts=0
                ),
            },
            ("infra", "mirix", 504),
        ),
        (  # a request the gateway refused (409 phase_mismatch): the harness's
            {
                "extraction_status": "fallback",
                "extraction_error": "LLMBadRequestError: INVALID_ARGUMENT: OpenAI API "
                "error: Error code: 409 - {'error': {'source': 'gateway', 'type': "
                "'phase_mismatch', 'code': 409}}",
            },
            ("harness", "mirix", 409),
        ),
        (
            {
                "extraction_status": "fallback",
                "extraction_error": "LLMConnectionError: INTERNAL_SERVER_ERROR: Failed "
                "to connect to OpenAI: Connection error.",
            },
            ("infra", "mirix", None),
        ),
        (
            {"extraction_status": "fallback", "extraction_error": None},
            ("infra", "mirix", None),
        ),
        (
            {
                "extraction_status": "fallback",
                "extraction_error": "TopicExtractionError: the extraction answer has no "
                "usable topic: no tool call",
            },
            None,
        ),
        (
            {
                "extraction_status": "fallback",
                "extraction_error": "LLMOutputTruncatedError: the answer hit the cap",
            },
            None,
        ),
        (
            {
                "extraction_status": "ok",
                "embedding_status": "failed",
                "embedding_error": "HTTPStatusError: OpenRouter embeddings endpoint "
                "answered HTTP 502 for http://127.0.0.1:1/runs/r/embedding/v1/embeddings"
                ': {"error": {"source": "gateway", "type": "upstream_infra_failure"}}',
            },
            ("infra", "embedding", 502),
        ),
        (  # an embedding timeout is never the model's
            {
                "embedding_status": "failed",
                "embedding_error": _deadline_error(
                    reason="deadline_during_attempt",
                    retries=0,
                    upstream_attempts=1,
                    attempt_budget_s=599.5,
                ),
            },
            ("infra", "embedding", 504),
        ),
        (
            {"embedding_status": "failed", "embedding_error": "ReadTimeout: timed out"},
            ("infra", "embedding", None),
        ),
        (
            {"success": False, "errors": {"procedural": "OperationalError: db gone"}},
            ("infra", "mirix", None),
        ),
        ({"success": False, "errors": {}}, ("harness", "mirix", None)),
    ],
)
def test_what_a_retrieval_lost_and_to_whom(evidence, expected):
    problem = M.retrieval_problem(evidence)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None
        assert (problem.kind.value, problem.role, problem.status) == expected
        assert problem.code == MIRIX_RETRIEVAL


#: OpenRouter's moderation refusal (a 403): the gateway forwards its message twice (its own
#: ``message`` and ``upstream.message``), so MIRIX's 500-character cut of the error text
#: splits the body.
MODERATION_MESSAGE = (
    "openai/gpt-6-luna requires moderation on Azure. Your input was flagged for "
    '"violence", "self-harm" and "harassment" in the text of the conversation you '
    "sent; the request was not processed. No credits were charged."
)
OPENROUTER_MODERATION_403 = {
    "error": {
        "code": 403,
        "message": MODERATION_MESSAGE,
        "metadata": {
            "reasons": ["violence", "self-harm", "harassment"],
            "flagged_input": "a long excerpt of the flagged conversation ...",
            "provider_name": "Azure",
            "model_slug": "openai/gpt-6-luna",
        },
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize("cut_inside", ["the body", "the message"])
async def test_a_moderation_refusal_of_the_topic_extraction_is_the_models(
    tmp_path, cut_inside
):
    """End to end: MIRIX's own topic extraction (``extract_topics``, its
    OpenAI-compatible client and ``handle_llm_error``) calls the real gateway, whose
    upstream refuses with OpenRouter's moderation 403. MIRIX falls back and logs the
    error cut to 500 characters, too short for the gateway's body. The method still finds
    the model's content filter (H7.2: the retrieval is injected and scored), as the
    harness's classifier does on the full ledger line."""
    from evals.skill_bench.infra.failures import CONTENT_FILTER, ledger_line_failure
    from evals.skill_bench.tests.fakes import Reply
    from mirix.schemas.llm_config import LLMConfig
    from mirix.services.topic_extraction import (
        extract_topics,
        normalize_retrieval_messages,
    )

    body = copy.deepcopy(OPENROUTER_MODERATION_403)
    if cut_inside == "the message":  # the cut falls inside the gateway's own message
        body["error"]["message"] = MODERATION_MESSAGE + " Details: " + "x" * 450
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(403, body))
    with serve_gateway(tmp_path, fake) as (_gateway, url):
        cell = mirix_cell(tmp_path, url)
        config = M.meta_agent_config(cell)["llm_config"]
        extraction = await extract_topics(
            normalize_retrieval_messages(
                [{"role": "user", "content": "Where is the mug?"}]
            ),
            LLMConfig(**config),
        )
    assert extraction.status == "fallback"
    text = extraction.error
    assert text.startswith("LLMPermissionDeniedError") and len(text) == 500
    status, parsed = M.gateway_error_in(text)
    assert status == 403
    with pytest.raises((ValueError, SyntaxError)):
        __import__("ast").literal_eval(text[text.index("{") :])  # the cut body
    if cut_inside == "the body":  # the fields before the cut survive the repair
        assert parsed is not None and parsed["error"]["type"] == "upstream_error"
        assert "moderation" in parsed["error"]["message"]
    assert M.extraction_failure(text) is None  # the model's: scored
    assert (
        M.retrieval_problem({"extraction_status": "fallback", "extraction_error": text})
        is None
    )
    (line,) = [c for c in ledger_calls(tmp_path, cell.run_id) if c["role"] == "mirix"]
    judged = ledger_line_failure(line)
    assert judged is not None
    assert (judged.kind, judged.code) == (FailureKind.MODEL, CONTENT_FILTER)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (  # a complete Python-repr body
            "Error code: 502 - {'error': {'type': 'a', 'code': 502}}",
            {"error": {"type": "a", "code": 502}},
        ),
        (  # cut inside a nested string: the complete fields before it survive
            "Error code: 403 - {'error': {'type': 'upstream_error', 'code': 403, "
            "'message': 'm', 'upstream': {'code': 403, 'message': 'cut her",
            {
                "error": {
                    "type": "upstream_error",
                    "code": 403,
                    "message": "m",
                    "upstream": {"code": 403},
                }
            },
        ),
        (  # JSON, cut inside a list, quotes and escapes inside strings
            'answered HTTP 502 for u: {"error": {"message": "a \\"quoted\\", b", '
            '"list": [1, 2, 3',
            {"error": {"message": 'a "quoted", b', "list": [1, 2]}},
        ),
        ("Error code: 403 - {'error': {'mess", None),  # nothing complete survived
    ],
)
def test_a_body_cut_short_keeps_its_complete_fields(text, expected):
    assert M.gateway_error_in(text)[1] == expected


def test_a_gateway_refusal_that_mentions_moderation_is_not_the_models():
    text = (
        "LLMPermissionDeniedError: PERMISSION_DENIED: Permission denied by OpenAI: Error "
        "code: 403 - {'error': {'source': 'gateway', 'type': 'model_mismatch_for_run', "
        "'code': 403, 'message': 'run r is pinned to m (moderated)'"
    )
    problem = M.extraction_failure(text)
    assert problem is not None and problem.kind is FailureKind.HARNESS


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("TopicExtractionError: no tool call", None),
        (  # MIRIX's own bug while building the request: its behaviour, scored
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(KeyError: 'image')",
            None,
        ),
        (  # the network failed while MIRIX built it
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(ConnectError: [Errno 61] Connection refused)",
            "infra",
        ),
        (
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(OperationalError: (sqlalchemy.dialects.postgresql.asyncpg.Error) "
            "<class 'asyncpg.exceptions.ConnectionDoesNotExistError'>: connection was "
            "closed in the middle of operation\n[SQL: SELECT 1])",
            "infra",
        ),
        (  # the database error format MIRIX's retrieval reports
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(OperationalError (ConnectionDoesNotExistError): connection was closed)",
            "infra",
        ),
        (  # a bind parameter naming a network class is no network failure
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(IntegrityError: (sqlite3.IntegrityError) UNIQUE constraint failed\n"
            "[SQL: INSERT INTO t VALUES (?)]\n[parameters: ('httpx.ReadTimeout',)])",
            None,
        ),
        (  # nor is a dictionary key that is an exception's name
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "(KeyError: 'ConnectionError')",
            None,
        ),
        (  # MIRIX's own verdict wins over the one cause the text names
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "[kind=infra] (RuntimeError: fetching the image failed)",
            "infra",
        ),
        (
            "LLMRequestBuildError: building the OpenAIClient request failed "
            "[kind=db] (RuntimeError: reading the attachment failed)",
            "infra",
        ),
        (
            "LLMRequestBuildError: calling the OpenAIClient SDK failed "
            "[kind=internal] (TypeError: got an unexpected keyword argument 'x')",
            None,
        ),
    ],
)
def test_a_request_mirix_could_not_build_is_judged_by_what_it_wraps(error, kind):
    problem = M.extraction_failure(error)
    assert (problem.kind.value if problem else None) == kind


def _build_error_causes() -> list[tuple[str, BaseException]]:
    """What building a request can meet, each as MIRIX's code raises it."""
    import json as json_module
    import sqlite3

    import asyncpg.exceptions as pg_exc
    import httpx
    import sqlalchemy.exc as sa_exc

    request = httpx.Request("GET", "http://files.local/page.png")

    def status_error(code: int) -> httpx.HTTPStatusError:
        try:
            httpx.Response(code, request=request).raise_for_status()
        except httpx.HTTPStatusError as exc:
            return exc
        raise AssertionError(code)

    return [
        (
            "connect",
            httpx.ConnectError("[Errno 61] Connection refused", request=request),
        ),
        ("read_timeout", httpx.ReadTimeout("timed out", request=request)),
        ("pool", httpx.PoolTimeout("no connection", request=request)),
        ("protocol", httpx.RemoteProtocolError("peer closed", request=request)),
        ("http_503", status_error(503)),
        ("http_429", status_error(429)),
        ("http_408", status_error(408)),
        ("http_404", status_error(404)),
        ("http_401", status_error(401)),
        ("reset", ConnectionResetError(54, "Connection reset by peer")),
        ("refused", ConnectionRefusedError(61, "Connection refused")),
        ("broken_pipe", BrokenPipeError(32, "Broken pipe")),
        ("timeout", TimeoutError()),
        ("timeout_text", TimeoutError("reading the stored attachment timed out")),
        ("cancelled", asyncio.CancelledError()),
        ("key", KeyError("image")),
        ("file", FileNotFoundError(2, "No such file or directory", "/img/a.png")),
        ("json", json_module.JSONDecodeError("Expecting value", "", 0)),
        (
            "multiple_addresses",
            OSError(
                "Multiple exceptions: [Errno 61] Connect call failed ('::1', 5432), "
                "[Errno 61] Connect call failed ('127.0.0.1', 5432)"
            ),
        ),
        (
            "pool_timeout",
            sa_exc.TimeoutError(
                "QueuePool limit of size 5 overflow 10 reached, connection timed "
                "out, timeout 30.00"
            ),
        ),
        ("disconnected", sa_exc.DisconnectionError("connection invalidated")),
        (
            "pg_deadlock",
            sa_exc.OperationalError(
                "SELECT 1", {}, pg_exc.DeadlockDetectedError("deadlock detected")
            ),
        ),
        (
            "pg_gone",
            pg_exc.ConnectionDoesNotExistError(
                "connection was closed in the middle of operation"
            ),
        ),
        (
            "pg_unknown_transaction",
            pg_exc.TransactionResolutionUnknownError("lost it"),
        ),
        (
            "sqlite_locked",
            sa_exc.OperationalError(
                "SELECT 1", {}, sqlite3.OperationalError("database is locked")
            ),
        ),
        (
            "sqlite_no_table",
            sa_exc.OperationalError(
                "SELECT 1", {}, sqlite3.OperationalError("no such table: files")
            ),
        ),
        (  # the bind parameters say "locked", the driver does not (MIRIX: INTERNAL)
            "sqlite_bind_mentions_lock",
            sa_exc.OperationalError(
                "SELECT ? AS note",
                ("database is locked",),
                sqlite3.OperationalError("no such table: files"),
            ),
        ),
        (
            "integrity",
            sa_exc.IntegrityError("INSERT", {}, sqlite3.IntegrityError("UNIQUE")),
        ),
        (  # a bind parameter naming a network class (MIRIX: INTERNAL)
            "integrity_bind_names_httpx",
            sa_exc.IntegrityError(
                "INSERT INTO t VALUES (?)",
                ("httpx.ReadTimeout",),
                sqlite3.IntegrityError("UNIQUE constraint failed: t.id"),
            ),
        ),
        ("key_names_an_exception", KeyError("ConnectionError")),
    ]


def _chained(outer: BaseException, *inner: BaseException) -> BaseException:
    """``outer`` whose ``__cause__`` chain is ``inner``, in order."""
    current = outer
    for cause in inner:
        current.__cause__ = cause
        current = cause
    return outer


def _chained_build_error_causes() -> list[tuple[str, BaseException]]:
    """Causes whose transient failure sits deeper than the one cause MIRIX's text names:
    MIRIX's ``_classify_build_error`` follows the chain."""
    import asyncpg.exceptions as pg_exc
    import httpx

    request = httpx.Request("GET", "http://files.local/page.png")
    return [
        (
            "runtime_from_reset",
            _chained(
                RuntimeError("fetching the image failed"),
                ConnectionResetError(54, "Connection reset by peer"),
            ),
        ),
        (
            "value_from_runtime_from_connect",
            _chained(
                ValueError("no image"),
                RuntimeError("download failed"),
                httpx.ConnectError("refused", request=request),
            ),
        ),
        (
            "runtime_from_deadlock",
            _chained(
                RuntimeError("reading the attachment failed"),
                pg_exc.DeadlockDetectedError("deadlock detected"),
            ),
        ),
        (  # nothing transient anywhere in the chain
            "runtime_from_key",
            _chained(RuntimeError("bad state"), KeyError("image")),
        ),
    ]


@pytest.mark.parametrize(
    "cause",
    [c for _, c in _build_error_causes()],
    ids=[n for n, _ in _build_error_causes()],
)
def test_build_errors_are_judged_as_mirix_judges_them(cause):
    """For a request MIRIX could not build (``LLMRequestBuildError``) the
    harness says INFRA exactly when MIRIX's own ``_classify_build_error`` says INFRA or
    DB, and HARNESS for MIRIX's INTERNAL, a wrapped ``TimeoutError`` or
    ``CancelledError`` included (a re-run repeats it: no INFRA re-run is paid for it).
    The text is the one MIRIX records (``topic_extraction._error_text``) of the error as
    MIRIX raises it (``LLMRequestBuildError.wrapping``, with its verdict);
    without the verdict (an older MIRIX) the classes it names agree too."""
    from mirix.errors import LLMRequestBuildError
    from mirix.services.retry_policy import ErrorKind, classify_error
    from mirix.services.topic_extraction import _error_text

    error = LLMRequestBuildError.wrapping(
        "building the OpenAIClient request", cause, stage="build"
    )
    mirix_kind = classify_error(error)
    assert mirix_kind in (ErrorKind.INFRA, ErrorKind.DB, ErrorKind.INTERNAL)
    assert error.kind == mirix_kind.value
    expected = (
        FailureKind.INFRA
        if mirix_kind in (ErrorKind.INFRA, ErrorKind.DB)
        else FailureKind.HARNESS
    )
    text = _error_text(error)
    assert f"[kind={mirix_kind.value}]" in text
    assert M.build_error_kind(text) is expected, (mirix_kind, text)
    assert M.mirix_error_kind(text) is expected, (mirix_kind, text)
    # In a topic extraction MIRIX fell back from, its own failure is its behaviour.
    problem = M.extraction_failure(text)
    assert (problem.kind if problem else None) is (
        FailureKind.INFRA if expected is FailureKind.INFRA else None
    )
    unmarked = text.replace(f" [kind={mirix_kind.value}]", "")
    assert M.build_error_kind(unmarked) is expected, (mirix_kind, unmarked)


@pytest.mark.parametrize(
    "cause",
    [c for _, c in _chained_build_error_causes()],
    ids=[n for n, _ in _chained_build_error_causes()],
)
def test_a_transient_failure_deeper_in_the_chain_is_judged_as_mirix_does(cause):
    """MIRIX's ``_classify_build_error`` follows ``__cause__`` up to 5
    levels, the recorded text names one. The verdict MIRIX writes into the text makes the
    harness agree anyway; the classes the text names alone cannot (the known limit of the
    fallback for a MIRIX that writes no verdict)."""
    from mirix.errors import LLMRequestBuildError
    from mirix.services.retry_policy import ErrorKind, classify_error
    from mirix.services.topic_extraction import _error_text

    error = LLMRequestBuildError.wrapping(
        "building the OpenAIClient request", cause, stage="build"
    )
    mirix_kind = classify_error(error)
    expected = (
        FailureKind.INFRA
        if mirix_kind in (ErrorKind.INFRA, ErrorKind.DB)
        else FailureKind.HARNESS
    )
    text = _error_text(error)
    assert M.build_error_kind(text) is expected, (mirix_kind, text)
    problem = M.extraction_failure(text)
    assert (problem.kind if problem else None) is (
        FailureKind.INFRA if expected is FailureKind.INFRA else None
    )
    unmarked = text.replace(f" [kind={mirix_kind.value}]", "")
    assert M.build_error_kind(unmarked) is FailureKind.HARNESS  # the fallback's limit


def test_build_error_names_cover_what_mirix_calls_transient():
    """Every httpx transport error, built-in connection error and asyncpg
    class MIRIX's ``_classify_build_error`` treats as transient is in
    :data:`BUILD_ERROR_INFRA_NAMES`, and nothing else from those libraries is."""
    import inspect

    import asyncpg.exceptions as pg_exc
    import httpx
    import sqlalchemy.exc as sa_exc

    from mirix.services import retry_policy

    def family(root: type) -> set[type]:
        found = {root}
        for sub in root.__subclasses__():
            found |= family(sub)
        return found

    # httpx's and the standard library's own classes: other libraries' subclasses
    # (``http.client.RemoteDisconnected``, httpx-sse's ``SSEError``) depend on what the
    # process imported, and never occur while MIRIX builds a request.
    def own(classes: set[type], package: str) -> set[str]:
        return {c.__name__ for c in classes if c.__module__.split(".")[0] == package}

    network = own(family(httpx.TransportError), "httpx") | own(
        family(ConnectionError), "builtins"
    )

    def transient(cls: type) -> bool:
        """Raw, or as MIRIX meets it: wrapped by SQLAlchemy (judged by SQLSTATE)."""
        driver = cls("simulated")
        wrapped = sa_exc.OperationalError("SELECT 1", {}, driver)
        return retry_policy._is_transient_db_error(
            driver
        ) or retry_policy._is_transient_db_error(wrapped)

    database = {
        name
        for name, cls in inspect.getmembers(pg_exc, inspect.isclass)
        if issubclass(cls, Exception) and transient(cls)
    } | {"DisconnectionError"}
    assert network <= M.BUILD_ERROR_INFRA_NAMES
    assert database <= M.BUILD_ERROR_INFRA_NAMES
    assert M.BUILD_ERROR_INFRA_NAMES == network | database
    assert "TimeoutError" not in M.BUILD_ERROR_INFRA_NAMES
    assert "CancelledError" not in M.BUILD_ERROR_INFRA_NAMES
    assert M.BUILD_ERROR_INFRA_STATUSES == retry_policy._INFRA_STATUS
    assert M.TRANSIENT_DB_MESSAGES == retry_policy._TRANSIENT_DB_MESSAGES


@pytest.mark.parametrize(
    ("errors", "kind"),
    [
        (
            {"procedural": "OperationalError (ConnectionDoesNotExistError): gone"},
            "infra",
        ),
        ({"procedural": "ReadTimeout: timed out"}, "infra"),
        ({"procedural": "TypeError: 'NoneType' object is not iterable"}, "harness"),
        (
            {"procedural": "ProgrammingError (UndefinedColumnError): no column"},
            "harness",
        ),
        ({"procedural": "IntegrityError (UniqueViolationError): duplicate"}, "harness"),
    ],
)
def test_a_failed_search_is_infra_only_when_the_environment_failed(errors, kind):
    """A deterministic MIRIX failure in the skill search is HARNESS (a person
    looks), never an INFRA re-run that pays for an extraction and an embedding again."""
    problem = M.retrieval_problem({"success": False, "errors": errors})
    assert problem is not None and problem.kind.value == kind


def test_mirixs_own_failure_in_a_learning_run_is_harness():
    """A run MIRIX failed for its own failure (a session it could not build a
    request for, a curator batch it kept after a bug) is HARNESS: a restart repeats it."""
    internal_session = {
        "id": "r1",
        "status": "failed",
        "error": "LearningRunInternalError: 1 session(s) not distilled",
        "distill_metrics": [
            {"session_id": "a", "status": "internal_error", "error": "KeyError: x"}
        ],
    }
    internal_batch = {
        "id": "r2",
        "status": "failed",
        "error": "KeyError: 'skill'",
        "curator_metrics": [
            {"batch_id": "r2:b0", "error_kind": "other", "outcome": "keep"}
        ],
    }
    refused_batch = {  # a counted (returned) batch: the model's, not listed at all
        "id": "r3",
        "status": "succeeded",
        "curator_metrics": [
            {"batch_id": "r3:b0", "error_kind": "refused", "outcome": "return"},
            {"batch_id": "r3:b1", "error_kind": "other", "outcome": "return"},
        ],
    }
    database = {"id": "r4", "status": "failed", "error": "OperationalError: gone"}
    kinds = M.learning_problem_kinds(
        [internal_session, internal_batch, refused_batch, database]
    )
    assert [(k.value, t) for k, t in kinds] == [
        (
            "harness",
            "run r1 could not distill session a for MIRIX's own failure: KeyError: x",
        ),
        (
            "harness",
            "learning run r1 failed (LearningRunInternalError: 1 session(s) not "
            "distilled)",
        ),
        (
            "harness",
            "run r2 curator batch r2:b0 stopped on MIRIX's own failure (other): None",
        ),
        ("harness", "learning run r2 failed (KeyError: 'skill')"),
        ("infra", "learning run r4 failed (OperationalError: gone)"),
    ]


def test_a_curation_configuration_skip_is_harness_and_other_config_kinds_stay_infra():
    """MIRIX records a curator stage its configuration stopped (no write
    scope, no curator tool, a context window too small ...) with ``error_kind: config``.
    It is deterministic: an INFRA resume meets it again, run after run, so it is HARNESS
    (a person must look). The other producers of ``config`` — an LLM call MIRIX
    classified as a misconfigured deployment (401/403/404) that stopped a batch, a
    session deferred after one — stay INFRA: the same kind, another meaning."""
    from mirix.services.retry_policy import ErrorKind
    from mirix.services.skill_experience_curator import (
        CONFIGURATION_SKIPS,
        CuratorOutcome,
        curation_skip_entry,
    )

    assert CONFIGURATION_SKIPS
    for reason in sorted(CONFIGURATION_SKIPS):
        entry = curation_skip_entry(
            "r1",
            CuratorOutcome(skipped=reason, experiences=3),
            started_at="2026-01-01T00:00:00+00:00",
        )
        assert entry is not None and entry["error_kind"] == ErrorKind.CONFIG.value
        run = {"id": "r1", "status": "succeeded", "curator_metrics": [entry]}
        ((kind, text),) = M.learning_problem_kinds([run])
        assert kind is FailureKind.HARNESS, reason
        assert reason in text and "r1:curate" in text
        method = method_with_state()
        with pytest.raises(M.LearningAttemptInvalid) as info:
            method._check_learning([run])
        failure = classify_exception(info.value)
        assert (failure.kind, failure.code) == (FailureKind.HARNESS, "mirix_learning")
        assert "configuration" in failure.detail
    config = ErrorKind.CONFIG.value
    others = {
        "id": "r2",
        "status": "succeeded",
        "distill_metrics": [
            {"session_id": "s", "status": "deferred", "failure_kind": config},
            # A MIRIX-side name the harness does not know stays INFRA.
            {
                "session_id": "t",
                "status": "deferred",
                "failure_kind": config,
                "config_deferral": "some_future_reason",
            },
        ],
        "curator_metrics": [
            {
                "batch_id": "r2:b0",
                "error_kind": config,
                "error": "AuthenticationError: 401",
                "outcome": "keep",
            }
        ],
    }
    assert [k for k, _ in M.learning_problem_kinds([others])] == [
        FailureKind.INFRA,
        FailureKind.INFRA,
        FailureKind.INFRA,
    ]


def test_a_distill_configuration_deferral_is_harness():
    """The distiller defers a session for its configuration in two cases a
    resume never changes: an input budget below MIN_CHUNK_BUDGET_TOKENS, and no LLM
    client or config. It names them (``config_deferral``); every such deferral is
    HARNESS, so the attempt is not resumed into the same deferral (as for the curator's
    configuration skips)."""
    from mirix.services.session_experience_distiller import (
        DISTILL_CONFIGURATION_DEFERRALS,
    )

    assert DISTILL_CONFIGURATION_DEFERRALS == frozenset(
        {"input_budget_too_small", "no_llm_client"}
    )
    for reason in sorted(DISTILL_CONFIGURATION_DEFERRALS):
        entry = {
            "session_id": "s",
            "status": "deferred",
            "failure_kind": "config",
            "config_deferral": reason,
            "error": "the distiller's input budget is 50 tokens",
        }
        run = {"id": "r1", "status": "succeeded", "distill_metrics": [entry]}
        ((kind, text),) = M.learning_problem_kinds([run])
        assert kind is FailureKind.HARNESS, reason
        assert reason in text and "session s" in text
        with pytest.raises(M.LearningAttemptInvalid) as info:
            method_with_state()._check_learning([run])
        failure = classify_exception(info.value)
        assert (failure.kind, failure.code) == (FailureKind.HARNESS, "mirix_learning")


@pytest.mark.asyncio
async def test_mirixs_own_learning_failure_invalidates_the_attempt_as_harness(
    tmp_path, gateway
):
    items = learning_items("alfworld", 1)
    servers = StubMirixServers(threshold=1, curator_errors=["internal"])
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        with pytest.raises(M.LearningAttemptInvalid, match="own error") as info:
            await learn(method, items)
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.HARNESS, "mirix_learning")


def test_the_http_timeout_covers_mirixs_slowest_answer():
    """One topic extraction and the query embedding's whole M3.5 retry chain,
    every call bounded by MIRIX's own client timeout; the constants agree with MIRIX."""
    from mirix.services.retry_policy import RETRY_POLICIES, ErrorKind

    policy = RETRY_POLICIES[ErrorKind.EMBEDDING]
    assert M.MIRIX_EMBEDDING_ATTEMPTS == policy.in_run_retries + 1
    assert M.MIRIX_EMBEDDING_BACKOFF_S == sum(
        policy.delay(i) for i in range(policy.in_run_retries)
    )
    for variable in ("MIRIX_LLM_TIMEOUT_SECONDS", "MIRIX_EMBEDDING_TIMEOUT_SECONDS"):
        assert float(protocol.MIRIX_SERVER_ENV[variable]) == protocol.CLIENT_TIMEOUT_S
    slowest = (
        protocol.CLIENT_TIMEOUT_S
        + M.MIRIX_EMBEDDING_ATTEMPTS * protocol.CLIENT_TIMEOUT_S
        + M.MIRIX_EMBEDDING_BACKOFF_S
    )
    assert slowest < M.MIRIX_HTTP_TIMEOUT_S


def test_the_evidence_comes_from_the_retrieval_log_row():
    response = {"success": True, "extraction_status": "fallback", "errors": {}}
    row = {
        "id": "rlog-1",
        "extraction_status": "fallback",
        "embedding_status": "failed",
        "llm_usage": {"topic_extraction": {"status": "fallback", "error": GATEWAY_502}},
        "per_type": {"procedural": {"embedding_error": "ConnectError: refused"}},
        "retry_events": [
            {
                "kind": "embedding",
                "stage": "retrieval",
                "attempt": 1,
                "error": "x" * 900,
            }
        ],
    }
    evidence = M.retrieval_evidence(response, row)
    assert evidence["extraction_error"] == GATEWAY_502
    assert evidence["embedding_error"] == "ConnectError: refused"
    assert evidence["retry_events"][0]["kind"] == "embedding"
    assert len(evidence["retry_events"][0]["error"]) == 300
    assert M.gateway_error_in(GATEWAY_502)[0] == 502
    assert (
        M.gateway_error_in(GATEWAY_502)[1]["error"]["type"] == "upstream_infra_failure"
    )
    assert M.gateway_error_in("ConnectError: refused") == (None, None)


def mirix_ranking_trace() -> dict[str, Any]:
    """A procedural ``per_type`` entry exactly as MIRIX logs it: this checkout's
    ``SkillRetrievalDiagnostics.to_dict`` with the log's candidate bound (M7.4, R24).

    Four candidates: the gate (0.62, 0.90 x best) passes two, the limit returns both;
    one more is logged in fused order and the last one is counted as omitted."""
    from mirix.services.skill_retrieval import (
        BM25Trace,
        CandidateTrace,
        GateTrace,
        SkillRetrievalDiagnostics,
    )

    def candidate(n: int, cosine: float | None, passed: bool | None) -> CandidateTrace:
        return CandidateTrace(
            id=f"proc-{n:04d}",
            bm25=1.0 / n,
            bm25_rank=n,
            cosine=cosine,
            cosine_rank=n if cosine is not None else None,
            rrf=0.5 / n,
            rrf_rank=n,
            gate_passed=passed,
            returned=bool(passed),
        )

    candidates = [
        candidate(1, 0.80, True),
        candidate(2, 0.74, True),
        candidate(3, 0.70, False),  # above 0.62, below 0.90 x 0.80
        candidate(4, 0.40, False),
    ]
    diagnostics = SkillRetrievalDiagnostics(
        requested_method="hybrid",
        method="hybrid",
        lanes=["bm25", "cosine"],
        backend="postgresql",
        embedding_status="ok",
        embedding_error=None,
        candidate_count=5,
        missing_vector_count=1,
        bm25=BM25Trace(
            query_terms=["receipts"],
            n_docs=5,
            avgdl=40.0,
            doc_freq={"receipts": 3},
            k1=1.2,
            b=0.75,
            weights={"A": 3.0},
            prefiltered=False,
            max_docs=5000,
            scored_docs=5,
        ),
        rrf_k=60,
        gate=GateTrace(
            applied=True,
            min_similarity=0.62,
            relative_ratio=0.90,
            max_cosine=0.80,
            threshold=0.72,
            passed=2,
        ),
        limit=10,
        max_cosine_distance=None,
        candidates=candidates,
        returned_ids=["proc-0001", "proc-0002"],
    )
    return diagnostics.to_dict(max_candidates=1)


def test_the_procedural_trace_keeps_mirixs_gate_limit_and_candidates():
    """From the record alone, the R2.5 gate and the limit
    can be re-checked for every logged candidate, not only for the returned ones."""
    logged = mirix_ranking_trace()
    assert set(M.PROCEDURAL_TRACE_KEYS) <= set(logged)  # MIRIX logs every kept key
    trace = M.procedural_trace({"id": "rlog-1", "per_type": {"procedural": logged}})
    assert trace == {k: logged[k] for k in M.PROCEDURAL_TRACE_KEYS}
    assert set(logged) - set(trace) == {
        "lanes",
        "backend",
        "embedding_status",  # the evidence keeps the embedding status and error
        "embedding_error",
        "bm25",
        "rrf_k",
        "candidates_logged",
    }
    assert json.loads(json.dumps(trace)) == trace
    gate = trace["gate"]
    assert [c["id"] for c in trace["candidates"]] == [
        "proc-0001",
        "proc-0002",
        "proc-0003",
    ]
    assert trace["candidates_omitted"] == 1
    for c in trace["candidates"]:
        assert c["gate_passed"] is (c["cosine"] >= gate["threshold"])
    assert gate["threshold"] == pytest.approx(
        max(gate["min_similarity"], gate["max_cosine"] * gate["relative_ratio"])
    )
    returned = [c["id"] for c in trace["candidates"] if c["returned"]]
    assert returned == trace["returned_ids"] and len(returned) <= trace["limit"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, None),
        ({}, None),
        ({"per_type": None}, None),  # a listing without the trace (include_trace=false)
        ({"per_type": {}}, None),
        ({"per_type": {"procedural": None}}, None),
        ({"per_type": {"procedural": "lost"}}, None),
        ({"per_type": {"procedural": {"limit": 10, "lanes": ["bm25"]}}}, {"limit": 10}),
        (
            {"per_type": {"procedural": {"gate": None, "candidates": []}}},
            {"gate": None, "candidates": []},
        ),
    ],
)
def test_the_procedural_trace_tolerates_what_a_row_lacks(row, expected):
    assert M.procedural_trace(row) == expected


@pytest.mark.asyncio
async def test_a_retrieval_record_keeps_the_rows_ranking_trace(
    tmp_path, gateway, monkeypatch
):
    """The record written by before_task carries the row's trace (a MIRIX-shaped one
    put in the stub's rows) next to its evidence, for the frozen retrievals too."""
    servers = StubMirixServers()
    cell, _frozen = await learned_cell(tmp_path, gateway, servers)
    logged = mirix_ranking_trace()
    resumed = method_for(servers)
    await resumed.setup(replace(cell, phase="test"))
    try:
        api = resumed._api
        assert api is not None
        listed = api.list_retrieval_logs

        async def with_mirix_trace(**kw: Any) -> list[dict[str, Any]]:
            rows = await listed(**kw)
            for row in rows:
                row["per_type"]["procedural"].update(copy.deepcopy(logged))
            return rows

        monkeypatch.setattr(api, "list_retrieval_logs", with_mirix_trace)
        await resumed.before_task(task_views("alfworld")[0])
    finally:
        await resumed.close()
    record = json.loads(
        (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()[-1]
    )
    assert record["frozen"] is True and record["failure"] is None
    assert record["procedural"] == {k: logged[k] for k in M.PROCEDURAL_TRACE_KEYS}
    assert record["evidence"]["embedding_status"] == "ok"


@pytest.mark.asyncio
async def test_a_stub_retrieval_that_lost_a_call_raises_infra(tmp_path):
    """On the stub MIRIX, through the real gateway: a test-phase retrieval whose topic
    extraction (or query embedding) the upstream fails raises INFRA ``mirix_retrieval``;
    MIRIX's own fallback after the model's unusable answer is injected; a MIRIX whose
    retrieval log is off cannot be checked (HARNESS)."""
    fake = FakeUpstream(responder=with_stub_mirix(lambda endpoint, body: None))
    with serve_gateway(tmp_path, fake) as (_gateway, url):
        servers = StubMirixServers(degrade_test_retrievals={"mirix": 1, "embedding": 1})
        cell, _frozen = await learned_cell(tmp_path, url, servers)
        stub = servers.stub_for(cell.work_dir)
        resumed = method_for(servers)
        await resumed.setup(replace(cell, phase="test"))
        try:
            view = task_views("alfworld")[0]
            for role in ("mirix", "embedding"):
                with pytest.raises(MirixRetrievalDegraded) as lost:
                    await resumed.before_task(view)
                failure = classify_exception(lost.value)
                assert (failure.kind, failure.code, failure.role, failure.status) == (
                    FailureKind.INFRA,
                    MIRIX_RETRIEVAL,
                    role,
                    502,
                )
            stub.extraction_outcomes = ["model"]
            injection = await resumed.before_task(view)
            assert "stub skill 1" in injection.text  # scored: MIRIX's own fallback
            stub.retrieval_log_disabled = True
            with pytest.raises(M.MirixApiError, match="no retrieval_id") as unchecked:
                await resumed.before_task(view)
            assert classify_exception(unchecked.value).kind is FailureKind.HARNESS
        finally:
            await resumed.close()
    records = [
        json.loads(line)
        for line in (cell.work_dir / M.RETRIEVAL_RECORD_FILE).read_text().splitlines()
    ]
    tested = [r for r in records if r["frozen"]]
    assert [(r["failure"] or {}).get("role") for r in tested] == [
        "mirix",
        "embedding",
        None,
        "mirix",  # the retrieval whose log row is missing is refused too
    ]
    assert tested[2]["evidence"]["extraction_status"] == "fallback"
    assert tested[2]["injection_sha256"] is not None
    assert tested[3]["failure"]["kind"] == "harness" and tested[3]["evidence"] is None
    assert tested[3]["started_at"] <= tested[3]["at"]
    # A refused retrieval still records its row's ranking trace, for the audit.
    assert [r["procedural"] for r in tested] == [
        {"returned_ids": ["proc-0001"]}
    ] * 3 + [None]


# --- The experience pool -------------------------------------------------------------------


def pool_entry(**overrides: Any) -> dict[str, Any]:
    """A settled curator batch entry of a MIRIX with the experience pool (the pool keys
    of ``SkillExperienceManager.settle_curation_batch`` and the curator's history keys),
    everything zero unless overridden."""
    entry: dict[str, Any] = {
        "batch_id": "slrun-x:b0",
        "status": "settled",
        "outcome": "consume",
        "history_ids": [],
        "history_pushed": 0,
        "history_tokens": 0,
        "history_skipped": "empty_pool",
        "history": {
            "pool_active": 0,
            "pool_scan_truncated": False,
            "embedded": 0,
            "embedding_error_kind": None,
            "embedding_error": None,
            "embedding_failed": 0,
            "embedding_failed_ids": [],
            "candidates": 0,
            "unusable_vectors": 0,
            "above_min": 0,
            "recurrence_candidates": 0,
            "dropped_for_room": 0,
            "lessons_cut": [],
            "recurred_ids": [],
            "max_cosines": [],
            "max_cosine_quantiles": {},
            "thresholds": {
                "history_min_similarity": 0.6,
                "recurrence_similarity": 0.8,
                "top_k": 10,
                "decay": 0.5,
            },
        },
        "citations": {},
        "uncited_writes": 0,
        "uncited_deletes": 0,
        "triage_calls": 0,
        "triaged": {"covered": 0, "not_worth": 0, "deferred": 0},
        "dispositions": {
            "used": 0,
            "covered": 0,
            "not_worth": 0,
            "deferred": 0,
            "defaulted": 0,
            "unaddressed": 0,
        },
        "history_cited": 0,
        "history_covered": 0,
        "history_not_worth": 0,
        "history_ignored": 0,
        "recurrences": 0,
        "embedding_failures": 0,
        "retired": {
            "used": 0,
            "covered": 0,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        },
        "retired_ignored": 0,
        "pool_active_after": 0,
    }
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(entry.get(key), Mapping):
            entry[key] = {**entry[key], **value}
        else:
            entry[key] = value
    return entry


def pool_runs() -> list[dict[str, Any]]:
    """Four learning runs, listed out of order: r0 (a MIRIX before the pool), r1 (two
    batches: the first finds the pool empty, the second is returned and closes one
    experience out), r2 (shows three earlier lessons) and r3 (its only batch never
    settled)."""
    return [
        {
            "id": "r3",
            "created_at": "2026-01-01T00:00:03",
            "status": "failed",
            "curator_metrics": [{"batch_id": "r3:b0", "status": "running"}],
        },
        {
            "id": "r2",
            "created_at": "2026-01-01T00:00:02",
            "status": "succeeded",
            "curator_metrics": [
                pool_entry(
                    batch_id="r2:b0",
                    history_ids=["e2", "e3", "e4"],
                    history_pushed=3,
                    history_skipped=None,
                    history={
                        "pool_active": 5,
                        "embedded": 3,
                        "candidates": 5,
                        "above_min": 4,
                        "recurrence_candidates": 2,
                        "dropped_for_room": 1,
                        "max_cosines": [
                            {"id": "e2", "max_cosine": 0.91},
                            {"id": "e3", "max_cosine": 0.85},
                            {"id": "e4", "max_cosine": 0.62},
                            {"id": "e5", "max_cosine": float("nan")},
                        ],
                        "thresholds": {
                            "history_min_similarity": 0.55,
                            "recurrence_similarity": 0.8,
                            "top_k": 10,
                            "decay": 0.5,
                        },
                    },
                    triage_calls=1,
                    dispositions={"used": 1, "deferred": 2, "defaulted": 1},
                    history_cited=1,
                    history_covered=1,
                    history_ignored=1,
                    recurrences=2,
                    retired={"used": 1, "covered": 1},
                    pool_active_after=4,
                    # B12: two citations without a verified quote (r1's batches come
                    # from a MIRIX before B12: they report none, counted 0).
                    citations_unverified=2,
                )
            ],
        },
        {
            "id": "r0",
            "created_at": "2026-01-01T00:00:00",
            "status": "succeeded",
            "curator_metrics": [{"batch_id": "r0:b0", "outcome": "consume"}],
        },
        {
            "id": "r1",
            "created_at": "2026-01-01T00:00:01",
            "status": "succeeded",
            "curator_metrics": [
                pool_entry(
                    batch_id="r1:b0",
                    uncited_writes=1,
                    triage_calls=1,
                    dispositions={
                        "used": 1,
                        "not_worth": 1,
                        "deferred": 3,
                        "defaulted": 2,
                    },
                    pool_active_after=4,
                ),
                pool_entry(
                    batch_id="r1:b1",
                    outcome="return",
                    history_skipped="no_room",
                    history={"pool_active": 4, "pool_scan_truncated": True},
                    dispositions={"unaddressed": 1},
                    pool_active_after=4,
                ),
            ],
        },
    ]


def test_the_experience_pool_is_read_off_the_curator_batches():
    """MIRIX has no experience endpoint; the pool's state at the freeze is the
    sum of what its settled curator batches report, the size after the last one."""
    pool = M.experience_pool(pool_runs())
    assert pool is not None
    per_run = pool.pop("per_run")
    history = pool.pop("history")
    assert pool == {
        "batches": 3,
        "unreported_batches": 2,  # r0's (before the pool) and r3's (never settled)
        "dispositions": {
            "used": 2,
            "covered": 0,
            "not_worth": 1,
            "deferred": 5,
            "defaulted": 3,
            "unaddressed": 1,
        },
        "entered": 6,
        "retired": {
            "used": 1,
            "covered": 1,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        },
        "active_at_freeze": 4,  # r2 is the last run with a settled batch
        "unaccounted": 0,
        "history_pushed": 3,
        "history_cited": 1,
        "history_covered": 1,
        "history_not_worth": 0,
        "history_ignored": 1,
        "recurrences": 2,
        "embedding_failures": 0,
        "uncited_writes": 1,
        "uncited_deletes": 0,
        "triage_calls": 2,
        "citations_unverified": 2,
        "history_skipped": {"empty_pool": 1, "no_room": 1},
    }
    assert history == {
        "candidates": 5,
        "above_min": 4,
        "recurrence_candidates": 2,
        "dropped_for_room": 1,
        "embedded": 3,
        "embedding_failed": 0,
        "unusable_vectors": 0,
        "lessons_cut": 0,
        "scan_truncated_batches": 1,
        # A non-finite cosine is left out; nearest-rank quantiles of what is left.
        "max_cosines": {"logged": 3, "p50": 0.85, "p90": 0.91, "max": 0.91},
        "thresholds": {
            "history_min_similarity": 0.55,
            "recurrence_similarity": 0.8,
            "top_k": 10,
            "decay": 0.5,
        },
    }
    assert [r["run_id"] for r in per_run] == ["r1", "r2"]
    assert per_run[0] == {
        "run_id": "r1",
        "batches": 2,
        "unreported_batches": 0,
        "dispositions": {
            "used": 1,
            "covered": 0,
            "not_worth": 1,
            "deferred": 3,
            "defaulted": 2,
            "unaddressed": 1,
        },
        "history_pushed": 0,
        "history_cited": 0,
        "history_covered": 0,
        "history_not_worth": 0,
        "history_ignored": 0,
        "recurrences": 0,
        "embedding_failures": 0,
        "uncited_writes": 1,
        "uncited_deletes": 0,
        "triage_calls": 1,
        "citations_unverified": 0,
        "retired": {
            "used": 0,
            "covered": 0,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        },
        "history_skipped": {"empty_pool": 1, "no_room": 1},
        "pool_active_after": 4,
    }
    assert per_run[1]["history_pushed"] == 3 and per_run[1]["recurrences"] == 2
    assert per_run[1]["citations_unverified"] == 2
    # The input order does not matter; the runs' own summaries are in run_outcome.
    assert M.experience_pool(list(reversed(pool_runs()))) == M.experience_pool(
        pool_runs()
    )
    runs = {r["id"]: r for r in pool_runs()}
    assert M.run_outcome(runs["r2"])["experience_pool"] == per_run[1]
    assert M.run_outcome(runs["r0"])["experience_pool"] is None


def test_a_mirix_without_the_pool_reports_none():
    runs = [r for r in pool_runs() if r["id"] in ("r0", "r3")]
    assert M.experience_pool(runs) is None
    assert M.experience_pool([]) is None
    assert M.experience_pool([{"id": "r", "curator_metrics": "garbage"}]) is None


def test_the_learning_context_is_read_off_the_distilled_sessions():
    """What the distiller was shown with each
    learned session (its last ``ok`` entry; the failed try before it is not counted),
    the served skills' stored feedback (per skill too), the usage records the curator
    batches showed and the curator's citations without a verified quote. The input
    order does not matter; each run's own is in ``run_outcome``."""
    from evals.skill_bench.tests.test_report import CONTEXT_SUMMARY, context_runs

    assert M.learning_context(context_runs()) == CONTEXT_SUMMARY
    assert M.learning_context(list(reversed(context_runs()))) == CONTEXT_SUMMARY
    r2, r1 = context_runs()
    assert M.run_outcome(r2)["learning_context"]["sessions"] == 2
    assert M.run_outcome(r1)["learning_context"]["usage"]["batches"] == 0


def test_the_learning_context_names_sessions_without_the_agent_context():
    """A learned session MIRIX distilled without an agent context is named (one
    without messages had nothing to distil and is not); the lengths MIRIX stored are
    listed once each; a MIRIX that recorded no per-skill values says so, and a MIRIX
    without agent contexts reports no learning context at all."""
    from evals.skill_bench.tests.test_report import _distilled

    runs = [
        {
            "id": "r",
            "distill_metrics": [
                _distilled("lost", agent_context_chars=0),
                _distilled("empty", agent_context_chars=0, messages=0),
                _distilled("other", agent_context_chars=7),
                _distilled(
                    "old",
                    served_skills=1,
                    skill_feedback={"served": 1, "not_shown": 0, "given": 1},
                ),
            ],
        }
    ]
    context = M.learning_context(runs)
    assert context is not None
    assert context["without_agent_context"] == ["lost"]
    assert context["with_agent_context"] == 2
    assert context["agent_context_chars"] == [7, 5210]
    assert context["skills_recorded"] is False and context["skills"] == []
    assert context["feedback"]["given"] == 1
    before_s1 = [{"id": "r", "distill_metrics": [{"session_id": "s", "status": "ok"}]}]
    assert M.learning_context(before_s1) is None
    assert M.learning_context([]) is None
    assert M.learning_context([{"id": "r", "distill_metrics": "garbage"}]) is None
    assert M.run_outcome(before_s1[0])["learning_context"] is None


def test_pool_metrics_that_do_not_add_up_are_unaccounted():
    runs = pool_runs()
    runs[1]["curator_metrics"][0]["pool_active_after"] = 3  # r2: one row went missing
    pool = M.experience_pool(runs)
    assert pool is not None
    assert (pool["entered"], pool["active_at_freeze"], pool["unaccounted"]) == (6, 3, 1)
    runs[1]["curator_metrics"][0]["pool_active_after"] = None
    assert M.experience_pool(runs)["unaccounted"] is None


POOL_502 = (
    "APIStatusError: Error code: 502 - {'error': {'message': 'fake upstream: bad "
    "gateway', 'type': 'upstream_infra_failure', 'retries': 2}}"
)
POOL_400_GATEWAY = (
    "BadRequestError: Error code: 400 - {'error': {'message': 'role embedding may "
    "not use this model', 'type': 'protocol_violation', 'source': 'gateway'}}"
)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (POOL_502, FailureKind.INFRA),
        ("ConnectError: [Errno 61] Connection refused", FailureKind.INFRA),
        (None, FailureKind.INFRA),  # nothing recorded: never blamed on the model
        (POOL_400_GATEWAY, FailureKind.HARNESS),  # a request the gateway refused
    ],
)
def test_a_batch_that_lost_its_earlier_lessons_to_the_embedding_is_a_problem(
    error, kind
):
    """Rule N1: the pool's embedding failing after MIRIX's retries left a
    batch without its earlier lessons. The embedding model is infrastructure, never the
    model under test, so it is judged by the gateway's answer MIRIX recorded."""
    runs = [
        {
            "id": "r1",
            "status": "succeeded",
            "curator_metrics": [
                pool_entry(
                    batch_id="r1:b0",
                    history_skipped="embedding_failed",
                    history={
                        "embedding_error_kind": "embedding",
                        "embedding_error": error,
                    },
                )
            ],
        }
    ]
    ((found, text),) = M.learning_problem_kinds(runs)
    assert found is kind
    assert "r1:b0 showed no earlier lessons" in text and "(embedding)" in text


@pytest.mark.parametrize(
    ("error", "error_kind", "kind"),
    [
        # No gateway answer, and MIRIX judged the error its own (a vector it
        # could not store, an empty answer): a restart meets it again.
        (
            "ValueError: the embedding model returned no vector",
            "other",
            FailureKind.HARNESS,
        ),
        ("KeyError: 'data'", "internal", FailureKind.HARNESS),
        # MIRIX's own kind, but the gateway answered: the gateway's answer decides.
        (POOL_502, "other", FailureKind.INFRA),
        # No gateway answer and not MIRIX's own: INFRA (never the model's).
        ("ReadTimeout: timed out", "embedding", FailureKind.INFRA),
    ],
)
def test_a_pool_embedding_failure_of_mirix_itself_needs_a_person(
    error, error_kind, kind
):
    runs = [
        {
            "id": "r1",
            "status": "succeeded",
            "curator_metrics": [
                pool_entry(
                    batch_id="r1:b0",
                    history_skipped="embedding_failed",
                    history={
                        "embedding_error_kind": error_kind,
                        "embedding_error": error,
                    },
                )
            ],
        }
    ]
    ((found, _text),) = M.learning_problem_kinds(runs)
    assert found is kind


def test_a_batch_that_left_a_pool_experience_out_is_a_problem_too():
    """MIRIX now shows the rest of the pool when only a pool experience's
    embedding fails; the batch still ran without it, so the failure is judged as a
    batch that showed nothing would be."""
    runs = [
        {
            "id": "r1",
            "status": "succeeded",
            "curator_metrics": [
                pool_entry(
                    batch_id="r1:b0",
                    history_skipped=None,
                    history_ids=["e1"],
                    history={
                        "embedding_error_kind": "embedding",
                        "embedding_error": POOL_502,
                        "embedding_failed": 1,
                        "embedding_failed_ids": ["e9"],
                    },
                )
            ],
        }
    ]
    ((found, text),) = M.learning_problem_kinds(runs)
    assert found is FailureKind.INFRA
    assert "r1:b0 left 1 pool experience(s) out of its earlier lessons" in text
    assert M.pool_embedding_failure(runs[0]["curator_metrics"][0]) is not None
    assert M.pool_embedding_failure(pool_entry(history_skipped=None)) is None


@pytest.mark.parametrize(
    "skipped",
    [None, "empty_pool", "no_candidates", "no_room", "disabled", "embeddings_disabled"],
)
def test_a_batch_that_skipped_its_earlier_lessons_otherwise_is_no_problem(skipped):
    runs = [
        {
            "id": "r1",
            "status": "succeeded",
            "curator_metrics": [pool_entry(history_skipped=skipped)],
        }
    ]
    assert M.learning_problem_kinds(runs) == []


@pytest.mark.asyncio
async def test_the_frozen_snapshot_carries_the_experience_pool(tmp_path, gateway):
    """The snapshot holds the pool's summary (MIRIX has no experience endpoint), the
    frozen hash covers it, and the resumed test phase finds the same memory."""
    servers = StubMirixServers(
        curator_pool=[
            pool_entry(
                triage_calls=1,
                dispositions={"used": 1, "not_worth": 1, "deferred": 1},
                pool_active_after=2,
            )
        ]
    )
    cell, frozen = await learned_cell(tmp_path, gateway, servers)
    pool = frozen["experience_pool"]
    assert pool == M.experience_pool(frozen["learning_runs"])
    assert (pool["entered"], pool["active_at_freeze"], pool["unaccounted"]) == (2, 2, 0)
    assert pool["history_skipped"] == {"empty_pool": 1}
    (run,) = frozen["learning_runs"]
    assert pool["per_run"] == [M.run_pool_summary(run)]
    flush_runs = [
        json.loads(line)
        for line in (cell.work_dir / M.LEARNING_RECORD_FILE).read_text().splitlines()
        if json.loads(line)["event"] == "flush_run"
    ]
    assert [r["experience_pool"] for r in flush_runs] == pool["per_run"]
    resumed = method_for(servers)
    await resumed.setup(replace(cell, phase="test"))  # the frozen hash still matches
    try:
        assert (await resumed.snapshot())["experience_pool"] == pool
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_the_frozen_hash_never_covers_the_harness_derived_pool(
    tmp_path, gateway, monkeypatch
):
    """``experience_pool`` is the harness's own function of the learning runs,
    which the frozen hash covers. It stays outside the hash, so a memory frozen by a
    harness whose summary differed (or had none: every attempt frozen before the pool)
    still resumes, instead of a valid attempt ending in an H10.4 mismatch."""
    servers = StubMirixServers(
        curator_pool=[pool_entry(dispositions={"deferred": 1}, pool_active_after=1)]
    )
    cell, frozen = await learned_cell(tmp_path, gateway, servers)
    state = M.MirixState.from_dict(
        json.loads((cell.work_dir / M.STATE_FILE).read_text())
    )
    exported = {
        k: v
        for k, v in frozen.items()
        if k
        not in (
            "method",
            "experience_pool",
            "counters",
            "frozen",
            "frozen_snapshot_sha256",
        )
    }
    assert M.canonical_sha256(exported) == state.frozen_snapshot_sha256
    assert frozen["experience_pool"] == M.experience_pool(frozen["learning_runs"])
    # A harness whose pool summary changed between the freeze and the resume.
    monkeypatch.setattr(M, "experience_pool", lambda runs: {"changed": True})
    resumed = method_for(servers)
    await resumed.setup(replace(cell, phase="test"))
    try:
        assert (await resumed.snapshot())["experience_pool"] == {"changed": True}
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_a_snapshot_of_a_mirix_without_the_pool_says_so(tmp_path, gateway):
    servers = StubMirixServers()  # a MIRIX before M5.7: no batch reports the pool
    _cell, frozen = await learned_cell(tmp_path, gateway, servers)
    assert frozen["experience_pool"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "kind"),
    [(POOL_502, FailureKind.INFRA), (POOL_400_GATEWAY, None)],
)
async def test_earlier_lessons_lost_to_the_embedding_invalidate_the_attempt(
    tmp_path, gateway, error, kind
):
    """The memory is not what the attempt learns without the failure: H8 restarts the
    attempt (INFRA). A request the gateway refused is counted and learning goes on
    (recorded, and three such idle checks in a row stop the attempt)."""
    items = learning_items("alfworld", 1)
    servers = StubMirixServers(
        threshold=1,
        curator_pool=[
            pool_entry(
                history_skipped="embedding_failed",
                history={"embedding_error_kind": "embedding", "embedding_error": error},
                dispositions={"deferred": 1, "defaulted": 1},
                pool_active_after=3,
            )
        ],
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        if kind is None:
            await learn(method, items)
            snapshot = await method.snapshot()
        else:
            with pytest.raises(
                M.LearningAttemptInvalid, match="earlier lessons"
            ) as info:
                await learn(method, items)
    finally:
        await method.close()
    if kind is None:
        [problem] = snapshot["tolerated"]["problems"]
        assert problem["reason"] == M.TOLERATED_REFUSED
        assert "earlier lessons" in problem["text"]
        return
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (kind, "mirix_learning")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_real_mirix_experience_pool_across_two_learning_runs(
    tmp_path, real_mirix
):
    """The experience pool on this checkout's real MIRIX (the fake upstream plays its
    model and gives every text the same vector, so every pool experience resembles every
    new one).

    Run 1 (MIRIX's threshold: the 5th closed session) curates e0..e4 with an empty pool:
    it creates a skill citing e0 and triages e1 not worth a skill and e2 deferred; e3
    and e4 are deferred by default. The pool then holds 4. Run 2 (the freeze's flush,
    3 sessions) shows those 4 as earlier lessons l0..l3, every one recurring; it creates
    a second skill citing l0 and its own e0, triages l1 covered, l2 not worth and its e1
    deferred. l0 retires used, l1 covered; l2 and l3 stay (ignored +2 and +1). Its e1
    and e2 enter the pool deferred, but each recurred a lesson still in it, so each is
    merged into that lesson as it enters: the pool ends with l2 and l3 only. The frozen
    snapshot reports exactly that, and the test phase changes nothing: not the
    snapshot, not the database."""
    import secrets

    from evals.skill_bench.infra.mirix_server import cell_db_name
    from evals.skill_bench.tests.mirix_model import POOL_SKILL

    url, model = real_mirix
    model.pool_curator = True
    cell = replace(
        mirix_cell(tmp_path, url, mirix_port=find_free_port()),
        cell_id=f"alfworld-realpool-{secrets.token_hex(4)}",
    )
    _REAL_CELLS.setdefault(str(tmp_path), []).append(cell.cell_id)
    items = learning_items("alfworld", 8)
    method = M.MirixMethod()
    await method.setup(cell)
    try:
        await learn(method, items, idle_s=120.0)
        await method.finalize()
        frozen = await method.snapshot()
    finally:
        await method.close()

    first, second = model.curations
    assert (len(first["experiences"]), first["lessons"]) == (5, [])
    assert len(second["experiences"]) == 3 and len(second["lessons"]) == 4
    assert set(second["lessons"]) == set(first["experiences"][1:])
    assert sorted(s["name"] for s in frozen["skills"]) == sorted(
        [SKILL["name"], POOL_SKILL["name"]]
    )
    runs = sorted(frozen["learning_runs"], key=lambda r: r["created_at"])
    assert [r["status"] for r in runs] == ["succeeded", "succeeded"]
    assert runs[1]["trigger"] == "flush"
    (batch1,) = runs[0]["curator_metrics"]
    (batch2,) = runs[1]["curator_metrics"]
    assert batch1["history_skipped"] == "empty_pool"
    assert batch2["history_skipped"] is None
    assert batch2["history_ids"] == second["lessons"]
    assert batch2["citations"] == {
        second["lessons"][0]: [batch2["changes"]["created"][0]],
        second["experiences"][0]: [batch2["changes"]["created"][0]],
    }

    assert frozen["experience_pool"] == M.experience_pool(frozen["learning_runs"])
    pool = copy.deepcopy(frozen["experience_pool"])
    per_run = pool.pop("per_run")
    history = pool.pop("history")
    assert pool == {
        "batches": 2,
        "unreported_batches": 0,
        "dispositions": {
            "used": 2,
            "covered": 0,
            "not_worth": 1,
            "deferred": 5,
            "defaulted": 3,
            "unaddressed": 0,
        },
        "entered": 6,
        "retired": {
            "used": 1,
            "covered": 1,
            "ignored": 0,
            "merged": 2,
            "unembeddable": 0,
        },
        "active_at_freeze": 2,
        "unaccounted": 0,
        "history_pushed": 4,
        "history_cited": 1,
        "history_covered": 1,
        "history_not_worth": 1,
        "history_ignored": 1,
        "recurrences": 4,
        "embedding_failures": 0,
        "uncited_writes": 0,
        "uncited_deletes": 0,
        "triage_calls": 2,
        "citations_unverified": 0,  # every citation quoted the text it applies (B12)
        "history_skipped": {"empty_pool": 1},
    }
    assert [r["pool_active_after"] for r in per_run] == [4, 2]
    assert [r["retired"]["merged"] for r in per_run] == [0, 2]
    assert history["candidates"] == 4 and history["recurrence_candidates"] == 4
    assert history["embedded"] == 7  # run 2's 3 new experiences and the 4 in the pool
    assert history["max_cosines"]["logged"] == 4
    assert history["max_cosines"]["max"] == pytest.approx(1.0, abs=1e-4)

    # The test phase: resumed read only; the pool (its rows and vectors) is untouched.
    db_name = cell_db_name(cell.cell_id, cell.attempt)
    before = await database_digest(db_name)
    resumed = M.MirixMethod()
    await resumed.setup(replace(cell, phase="test"))
    try:
        assert await resumed.snapshot() == frozen
        await asyncio.gather(*(resumed.before_task(v) for v in task_views("alfworld")))
        assert await resumed.snapshot() == frozen
    finally:
        await resumed.close()
    assert await database_digest(db_name) == before  # H10.4 below the API


# --- learning checkpoints (contract item 6) ------------------------------------------------------


async def _checkpoint_of(
    method: M.MirixMethod, stub: StubMirix, work_dir: Path, saved: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """What run.py records at a checkpoint: the method's state, its snapshot hash, the
    work_dir copy, and (here) the stub's "database" instead of a pg_dump."""
    from evals.skill_bench.methods.base import snapshot_sha256
    from evals.skill_bench.run import _copy_method_dir

    state = await method.checkpoint()
    sha = snapshot_sha256(await method.snapshot())
    _copy_method_dir(work_dir, saved)
    return state, sha, stub.dump_state()


def _restore(
    stub: StubMirix, work_dir: Path, saved: Path, dump: dict[str, Any]
) -> None:
    import shutil

    from evals.skill_bench.run import _copy_method_dir

    shutil.rmtree(work_dir)
    _copy_method_dir(saved, work_dir)
    stub.load_state(dump)


@pytest.mark.asyncio
async def test_a_learning_resume_binds_the_agent_mirix_has(tmp_path, gateway):
    """setup with ``cell.resume`` on the restored database binds the existing meta agent
    (never initializes one), refuses an agent of another gateway run, and refuses a
    memory whose snapshot is not the checkpoint's."""
    from evals.skill_bench.methods.base import CheckpointMismatch, LearningResume

    servers = StubMirixServers(threshold=5)
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    items = learning_items("alfworld", 3)
    try:
        await learn(method, items[:2])
        state, sha, dump = await _checkpoint_of(
            method, stub, cell.work_dir, tmp_path / "saved"
        )
        await learn(method, items[2:])  # discarded
    finally:
        await method.close()
    assert state["meta_agent_id"] == method.state.meta_agent_id
    assert not state["frozen"]
    resume = LearningResume("c0001-done", "done", 1, state, sha)
    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    start = len(stub.requests)
    resumed = method_for(servers)
    await resumed.setup(replace(cell, resume=resume))
    try:
        assert servers.servers[-1].mode == "resume"
        assert ("POST", "/agents/meta/initialize") not in calls(stub, start)
        assert resumed.state.meta_agent_id == state["meta_agent_id"]
        assert sorted(stub.sessions) == sorted(i.view.task_uid for i in items[:2])
        await learn(resumed, items[2:])
        await resumed.finalize()
        frozen = await resumed.snapshot()
    finally:
        await resumed.close()
    assert frozen["frozen"] is True
    assert sorted(stub.sessions) == sorted(i.view.task_uid for i in items)

    # Another gateway run, and a memory that is not the checkpoint's, are refused.
    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    with pytest.raises(M.MirixApiError, match="model_endpoint"):
        await method_for(servers).setup(
            replace(cell, run_id="another-run-a1", resume=resume)
        )
    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    with pytest.raises(CheckpointMismatch, match="hashes to"):
        await method_for(servers).setup(
            replace(cell, resume=replace(resume, snapshot_sha256="0" * 64))
        )
    frozen_state = LearningResume(
        "c0001-done", "done", 1, {**state, "frozen": True}, sha
    )
    with pytest.raises(CheckpointMismatch, match="frozen"):
        await method_for(servers).setup(replace(cell, resume=frozen_state))


@pytest.mark.asyncio
async def test_a_batch_that_failed_on_infrastructure_is_redone_after_the_rollback(
    tmp_path, gateway
):
    """D3: MIRIX deferred the session of the batch the item's close started (INFRA). The
    resume restores the item's played checkpoint (database included) and runs the close
    again: the whole batch runs again from the pre-batch state and settles clean; the
    live state keeps no trace of the failed run."""
    from evals.skill_bench.infra.failures import classify_exception as classify
    from evals.skill_bench.methods.base import LearningResume

    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    servers = StubMirixServers(threshold=1, session_outcomes={uid: ["deferred"]})
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    item = items[0]
    try:
        await method.before_task(copy.deepcopy(item.view))
        for message in item.messages:
            await method.on_message(uid, copy.deepcopy(message))
        state, sha, dump = await _checkpoint_of(
            method, stub, cell.work_dir, tmp_path / "saved"
        )
        await method.after_task(item.trajectory())
        with pytest.raises(M.LearningAttemptInvalid) as info:
            await method.wait_until_idle(10.0)
    finally:
        await method.close()
    failure = classify(info.value)
    assert (failure.kind, failure.code) == (FailureKind.INFRA, "mirix_learning")
    failed_runs = sorted(stub.runs)
    assert failed_runs

    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    assert stub.runs == {}  # the failed batch is not in the restored database
    resumed = method_for(servers)
    await resumed.setup(
        replace(cell, resume=LearningResume("c0000-played", "played", 0, state, sha))
    )
    try:
        await resumed.after_task(item.trajectory())
        await resumed.wait_until_idle(10.0)
        await resumed.finalize()
        frozen = await resumed.snapshot()
    finally:
        await resumed.close()
    assert [run["status"] for run in stub.runs.values()] == ["succeeded"]
    assert [run["session_ids"] for run in stub.runs.values()] == [[uid]]
    assert frozen["frozen"] is True and frozen["skills"]


# --- tolerated failures: record, do not stop ----------------------------------------------------

REFUSED_NO_ENDPOINT = (
    "BadRequestError: Error code: 404 - {'error': {'source': 'upstream', 'code': 404, "
    "'message': 'No endpoints found that support image input'}}"
)
REFUSED_PAYLOAD = (
    "APIStatusError: Error code: 413 - {'error': {'source': 'upstream', 'code': 413, "
    "'message': 'Request payload too large'}}"
)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (REFUSED_NO_ENDPOINT, "mirix/no_endpoint/404"),
        (REFUSED_PAYLOAD, "mirix/payload/413"),
        (
            "BadRequestError: Error code: 400 - {'error': {'message': 'Provider returned error'}}",
            "mirix/payload/400",
        ),
        (POOL_400_GATEWAY, "mirix/gateway/400"),
        (
            "error: Error code: 409 - {'error': {'source': 'gateway', 'type': "
            "'phase_mismatch', 'code': 409}}",
            "mirix/gateway/409",
        ),
        # A body cut short: a 413 is a payload refusal whatever it says; a 400 only
        # when the text that survived names its size or images, read from
        # the raw text when MIRIX's cut left no complete message field.
        ("APIStatusError: Error code: 413 - {'error': {'mess", "mirix/payload/413"),
        ("BadRequestError: Error code: 400 - {'error': {'mess", None),
        (
            "BadRequestError: Error code: 400 - {'error': {'code': 400, 'message': "
            "'Image exceeds the maximum allowed si",
            "mirix/payload/400",
        ),
        (
            "BadRequestError: Error code: 400 - {'error': {'code': 400, 'message': "
            "'max_tokens: must be less than or equal to 131072'}}",
            None,
        ),
        (
            "BadRequestError: Error code: 400 - {'error': {'code': 400, 'message': "
            "'Invalid messages[3]: tool_call_id not found in previous messa",
            None,
        ),
        (POOL_502, None),
        (
            REFUSED_NO_ENDPOINT.replace(
                "support image input", "matching your data policy"
            ),
            None,
        ),
        (
            "BadRequestError: Error code: 404 - {'error': {'message': 'Not Found'}}",
            None,
        ),
        (
            "AuthenticationError: Error code: 401 - {'error': {'message': 'no key'}}",
            None,
        ),
        ("RateLimitError: Error code: 429 - {'error': {'message': 'slow down'}}", None),
        (
            "PermissionDeniedError: Error code: 403 - {'error': {'message': 'Input flagged by moderation'}}",
            None,
        ),
        ("KeyError: x", None),
        (None, None),
    ],
)
def test_refused_call_in_reads_the_gateways_answer(text, expected):
    from evals.skill_bench.infra.failures import refusal_signature

    refused = M.refused_call_in(text)
    if expected is None:
        assert refused is None
    else:
        assert refused is not None and refusal_signature(refused) == expected


def test_degraded_streaks_count_trailing_degraded_rows_per_item():
    evidence = {"embedding_status": "failed"}
    problem = {"kind": "infra", "code": "mirix_retrieval"}

    def row(uid: str, *, frozen: bool = True, **kw: Any) -> dict[str, Any]:
        return {"event": "retrieval", "task_uid": uid, "frozen": frozen, **kw}

    records = [
        row("a", evidence=evidence, failure=problem),
        row("a", evidence=evidence, failure=problem),  # a: two refused
        row("b", evidence=evidence, failure=problem),
        row("b", evidence={"embedding_status": "ok"}, failure=None),  # b: clean ends it
        row("c", evidence=evidence, failure=problem),
        row("c", evidence=None, failure=problem),  # c: MIRIX did not answer: ends it
        row(
            "d", evidence=evidence, failure=None, degraded=problem
        ),  # d: an injected one
        row("e", frozen=False, evidence=evidence, failure=problem),  # learning: ignored
        {"event": "other", "task_uid": "a", "frozen": True},
        row("f", evidence=evidence, failure=problem),
    ]
    assert M.degraded_streaks(records) == {"a": 2, "d": 1, "f": 1}
    assert M.degraded_streaks([]) == {}


def _stub_gateway(tmp_path: Path) -> Any:
    return serve_gateway(
        tmp_path, FakeUpstream(responder=with_stub_mirix(lambda e, b: None))
    )


@pytest.mark.asyncio
async def test_a_test_item_degraded_three_times_is_injected_and_flagged(tmp_path):
    with _stub_gateway(tmp_path) as (_gateway, url):
        servers = StubMirixServers(degrade_test_retrievals={"embedding": 3})
        cell, frozen = await learned_cell(tmp_path, url, servers)
        resumed = method_for(servers)
        await resumed.setup(replace(cell, phase="test"))
        try:
            view = task_views("alfworld")[0]
            for _ in range(2):
                with pytest.raises(MirixRetrievalDegraded):
                    await resumed.before_task(view)
            injection = await resumed.before_task(view)
            clean = await resumed.before_task(view)
            assert await resumed.snapshot() == frozen  # H10.4: nothing changed
        finally:
            await resumed.close()
    assert injection.degraded is True and "stub skill 1" in injection.text
    assert clean.degraded is False and clean.text == injection.text
    assert "tolerated" not in frozen
    tested = [
        r for r in _records(cell.work_dir / M.RETRIEVAL_RECORD_FILE) if r["frozen"]
    ]
    assert [r.get("degraded_streak") for r in tested] == [1, 2, 3, None]
    assert [r["failure"] is not None for r in tested] == [True, True, False, False]
    assert "degraded" not in tested[0] and "degraded" not in tested[3]
    assert tested[2]["degraded"]["role"] == "embedding"
    assert tested[2]["injection_sha256"] is not None
    assert tested[2]["injection_chars"] == len(injection.text)
    assert [r["injection_sha256"] is None for r in tested] == [True, True, False, False]
    assert M.degraded_streaks(tested) == {}  # the clean fourth retrieval ended it


@pytest.mark.asyncio
async def test_the_test_streak_survives_a_new_invocation(tmp_path):
    """The re-runs of a test item happen in later invocations (the scheduler): the
    method reads each item's run of degraded retrievals back from its own record."""
    with _stub_gateway(tmp_path) as (_gateway, url):
        servers = StubMirixServers(degrade_test_retrievals={"embedding": 4})
        cell, frozen = await learned_cell(tmp_path, url, servers)
        view, other = task_views("alfworld")
        for _ in range(2):
            first = method_for(servers)
            await first.setup(replace(cell, phase="test"))
            try:
                with pytest.raises(MirixRetrievalDegraded):
                    await first.before_task(view)
            finally:
                await first.close()
        again = method_for(servers)
        await again.setup(replace(cell, phase="test"))
        try:
            injection = await again.before_task(view)
            # Another item starts its own run: refused.
            with pytest.raises(MirixRetrievalDegraded):
                await again.before_task(other)
            assert await again.snapshot() == frozen
        finally:
            await again.close()
    assert injection.degraded is True


@pytest.mark.asyncio
async def test_a_learning_retrieval_degraded_is_injected_and_counted(tmp_path):
    with _stub_gateway(tmp_path) as (_gateway, url):
        servers = StubMirixServers(degrade_learning_retrievals={"embedding": 1})
        cell = mirix_cell(tmp_path, url)
        method = method_for(servers)
        await method.setup(cell)
        try:
            injections = await learn(method, learning_items("alfworld", 2))
            snapshot = await method.snapshot()
            state = await method.checkpoint()
            await method.finalize()
            frozen = await method.snapshot()
        finally:
            await method.close()
    assert [i.degraded for i in injections] == [True, False]
    assert state["tolerated"] == {"retrievals_degraded": 1}
    assert state["degraded_streak"] == 0  # the clean second retrieval ended it
    assert snapshot["tolerated"] == {
        "counts": {"retrievals_degraded": 1},
        "problems": [],
    }
    assert frozen["tolerated"] == snapshot["tolerated"]
    rows = [r for r in _records(cell.work_dir / M.RETRIEVAL_RECORD_FILE)]
    assert rows[0]["degraded_streak"] == 1 and rows[0]["failure"] is None
    assert rows[0]["degraded"]["role"] == "embedding"
    assert "degraded" not in rows[1]


@pytest.mark.asyncio
async def test_three_degraded_learning_items_in_a_row_raise(tmp_path):
    """The lead kept this safety: the third consecutive learning item with a degraded
    retrieval raises (INFRA: the learning phase resumes from its checkpoint)."""
    with _stub_gateway(tmp_path) as (_gateway, url):
        servers = StubMirixServers(degrade_learning_retrievals={"embedding": 3})
        method = method_for(servers)
        await method.setup(mirix_cell(tmp_path, url))
        items = learning_items("alfworld", 3)
        try:
            injections = await learn(method, items[:2])
            with pytest.raises(MirixRetrievalDegraded) as info:
                await method.before_task(copy.deepcopy(items[2].view))
            state = await method.checkpoint()
        finally:
            await method.close()
    assert [i.degraded for i in injections] == [True, True]
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.INFRA, MIRIX_RETRIEVAL)
    assert state["tolerated"] == {"retrievals_degraded": 2}
    assert state["degraded_streak"] == 3


@pytest.mark.asyncio
async def test_a_gateway_refusal_in_mirix_learning_is_counted_and_learning_goes_on(
    tmp_path, gateway
):
    servers = StubMirixServers(
        threshold=1,
        curator_errors=["config"],
        curator_error_texts=[REFUSED_NO_ENDPOINT],
    )
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    items = learning_items("alfworld", 2)
    try:
        await learn(method, items[:1])
        state = await method.checkpoint()
        await method.wait_until_idle(10.0)  # the same problem is not accepted twice
        assert method.state.tolerated == state["tolerated"]
        await learn(method, items[1:])
        after = await method.checkpoint()
        snapshot = await method.snapshot()
    finally:
        await method.close()
    assert state["tolerated"] == {"gateway_refusal": 1, "mirix_learning_failed": 1}
    assert state["refusal_streaks"] == {
        "mirix": {"signature": "mirix/no_endpoint/404", "count": 1}
    }
    [problem] = state["accepted_problems"]
    assert problem["reason"] == "gateway_refusal" and problem["rollbacks"] == 0
    assert "No endpoints found" in problem["text"]
    assert after["refusal_streaks"] == {}  # a check without a new refusal ends it
    assert snapshot["tolerated"] == {
        "counts": {"gateway_refusal": 1, "mirix_learning_failed": 1},
        "problems": [problem],
    }
    records = [
        r
        for r in _records(cell.work_dir / M.LEARNING_RECORD_FILE)
        if r["event"] == "tolerated"
    ]
    assert len(records) == 1
    assert records[0]["reason"] == "gateway_refusal"
    assert records[0]["problem"] == problem["text"] and records[0]["rollbacks"] == 0
    # The state file carries it too (the frozen memory's snapshot reads it back).
    saved = json.loads((cell.work_dir / M.STATE_FILE).read_text())
    assert saved["accepted_problems"] == [problem]


@pytest.mark.asyncio
async def test_three_idle_checks_in_a_row_with_a_new_refusal_stop_the_attempt(
    tmp_path, gateway
):
    from evals.skill_bench.infra.failures import REFUSAL_STREAK

    servers = StubMirixServers(
        threshold=1,
        curator_errors=["config"] * 3,
        curator_error_texts=[REFUSED_PAYLOAD] * 3,
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    items = learning_items("alfworld", 3)
    try:
        await learn(method, items[:2])
        assert method.state.tolerated["gateway_refusal"] == 2
        with pytest.raises(M.LearningAttemptInvalid) as info:
            await learn(method, items[2:])
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code, failure.role, failure.status) == (
        FailureKind.HARNESS,
        REFUSAL_STREAK,
        "mirix",
        413,
    )
    assert "mirix/payload/413" in failure.detail
    assert method.state.tolerated["gateway_refusal"] == 2  # the stop accepted nothing


@pytest.mark.asyncio
async def test_mirixs_own_failure_beside_a_refusal_still_needs_a_person(
    tmp_path, gateway
):
    """A curator batch MIRIX kept after its own bug fails the run too: the refusal in
    the batch's error is accepted only when nothing else is raised."""
    servers = StubMirixServers(
        threshold=1, curator_errors=["other"], curator_error_texts=[POOL_400_GATEWAY]
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        with pytest.raises(M.LearningAttemptInvalid, match="own error") as info:
            await learn(method, learning_items("alfworld", 1))
    finally:
        await method.close()
    assert classify_exception(info.value).kind is FailureKind.HARNESS
    assert method.state.tolerated == {} and method.state.accepted_problems == []


@pytest.mark.asyncio
async def test_a_refusal_on_every_batch_stops_at_the_default_threshold(
    tmp_path, gateway
):
    """MIRIX learns once a batch of sessions is pending, so most idle checks
    bring no new run. Such a check is no served MIRIX call: it leaves the streak as it
    is, and the third refused run in a row stops the attempt (not never)."""
    from evals.skill_bench.infra.failures import REFUSAL_STREAK

    servers = StubMirixServers(
        threshold=5,
        curator_errors=["config"] * 3,
        curator_error_texts=[REFUSED_PAYLOAD] * 3,
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    items = learning_items("alfworld", 15)
    try:
        await learn(method, items[:7])  # one refused run (item 5), two idle checks
        streak = {"mirix": {"signature": "mirix/payload/413", "count": 1}}
        assert method.state.refusal_streaks == streak
        await learn(method, items[7:10])  # the second refused run
        assert method.state.refusal_streaks["mirix"]["count"] == 2
        with pytest.raises(M.LearningAttemptInvalid) as info:
            await learn(method, items[10:])
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code, failure.status) == (
        FailureKind.HARNESS,
        REFUSAL_STREAK,
        413,
    )
    assert method.state.tolerated["gateway_refusal"] == 2


@pytest.mark.asyncio
async def test_a_served_run_between_refused_runs_ends_the_streak(tmp_path, gateway):
    """A new run without a refusal is MIRIX served again: it ends the
    streak, so refusals that do not come in a row never stop the attempt."""
    servers = StubMirixServers(
        threshold=2,
        curator_errors=["config", "config", "", "config", "config"],
        curator_error_texts=[REFUSED_PAYLOAD] * 2 + [""] + [REFUSED_PAYLOAD] * 2,
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        await learn(method, learning_items("alfworld", 10))
        state = await method.checkpoint()
    finally:
        await method.close()
    assert state["tolerated"]["gateway_refusal"] == 4
    assert state["refusal_streaks"] == {
        "mirix": {"signature": "mirix/payload/413", "count": 2}
    }
    assert len(state["refusal_runs"]) == 5  # each run observed once


REFUSED_PHASE_MISMATCH = (
    "ConflictError: Error code: 409 - {'error': {'source': 'gateway', 'type': "
    "'phase_mismatch', 'code': 409, 'message': 'x-skill-bench-phase is test'}}"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "text", "why"),
    [
        ("config", REFUSED_NO_ENDPOINT, "deferred"),
        (
            "config",
            POOL_400_GATEWAY.replace("protocol_violation", "model_not_allowed"),
            "deferred",
        ),
        ("infra", REFUSED_PHASE_MISMATCH, "deferred"),
    ],
)
async def test_a_refused_session_mirix_defers_stops_at_once(
    tmp_path, gateway, kind, text, why
):
    """MIRIX defers a session refused 401/403/404 (or by the gateway's 409)
    across runs and refuses it again on every retry: it is never learned, and the freeze
    could not happen. Accepting it would spend the whole learning phase first; the first
    such refusal stops the attempt for a person (HARNESS: no automatic resume)."""
    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    servers = StubMirixServers(
        threshold=1,
        session_outcomes={uid: ["deferred"]},
        deferred_errors={uid: (kind, text)},
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        with pytest.raises(M.LearningAttemptInvalid, match=why) as info:
            await learn(method, items)
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.HARNESS, M.MIRIX_LEARNING)
    assert uid in failure.detail
    assert method.state.tolerated == {} and method.state.accepted_problems == []


@pytest.mark.asyncio
async def test_a_protocol_rejection_of_a_mirix_call_stops_at_once(tmp_path, gateway):
    """A gateway refusal of a protocol type (a phase, model or run MIRIX's
    call broke) is a configuration bug the audit fails the cell for (H10.3): the first
    one stops the attempt instead of being counted."""
    servers = StubMirixServers(
        threshold=1,
        curator_errors=["infra"],
        curator_error_texts=[REFUSED_PHASE_MISMATCH],
    )
    method = method_for(servers)
    await method.setup(mirix_cell(tmp_path, gateway))
    try:
        with pytest.raises(M.LearningAttemptInvalid, match="phase_mismatch") as info:
            await learn(method, learning_items("alfworld", 1))
    finally:
        await method.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.HARNESS, M.MIRIX_LEARNING)
    assert method.state.tolerated == {} and method.state.accepted_problems == []


async def _rolled_back(
    tmp_path: Path, gateway: str, rollbacks: int
) -> tuple[M.MirixMethod, CellContext, StubMirix, Any]:
    """A cell whose first item's batch lost a curator batch to infrastructure, resumed
    from the item's played checkpoint as the ``rollbacks``-th consecutive rollback."""
    from evals.skill_bench.methods.base import LearningResume

    items = learning_items("alfworld", 1)
    uid = items[0].view.task_uid
    servers = StubMirixServers(threshold=1, curator_errors=["infra", "infra", "infra"])
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    item = items[0]
    try:
        await method.before_task(copy.deepcopy(item.view))
        for message in item.messages:
            await method.on_message(uid, copy.deepcopy(message))
        state, sha, dump = await _checkpoint_of(
            method, stub, cell.work_dir, tmp_path / "saved"
        )
        await method.after_task(item.trajectory())
        with pytest.raises(M.LearningAttemptInvalid):
            await method.wait_until_idle(10.0)
    finally:
        await method.close()
    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    resumed = method_for(servers)
    resume = LearningResume("c0000-played", "played", 0, state, sha, rollbacks)
    await resumed.setup(replace(cell, resume=resume))
    await resumed.after_task(item.trajectory())
    return resumed, cell, stub, item


@pytest.mark.asyncio
async def test_an_infrastructure_loss_is_accepted_after_three_rollbacks(
    tmp_path, gateway
):
    resumed, cell, stub, _item = await _rolled_back(tmp_path, gateway, 3)
    try:
        await resumed.wait_until_idle(10.0)  # accepted at the first check
        snapshot = await resumed.snapshot()
        # Accepted only at the first check after the resume: a new loss raises.
        stub.curator_errors = ["infra"]
        with pytest.raises(M.LearningAttemptInvalid, match="infrastructure"):
            await learn(resumed, learning_items("alfworld", 2)[1:])
    finally:
        await resumed.close()
    tolerated = snapshot["tolerated"]
    assert tolerated["counts"] == {"mirix_learning_failed": 1, "rolled_back": 1}
    [problem] = tolerated["problems"]
    assert problem["reason"] == "rolled_back" and problem["rollbacks"] == 3
    assert "stopped on infra" in problem["text"]
    records = [
        r
        for r in _records(cell.work_dir / M.LEARNING_RECORD_FILE)
        if r["event"] == "tolerated"
    ]
    assert [(r["reason"], r["rollbacks"]) for r in records] == [("rolled_back", 3)]


@pytest.mark.asyncio
async def test_an_infrastructure_loss_after_two_rollbacks_still_rolls_back(
    tmp_path, gateway
):
    resumed, _cell, _stub, _item = await _rolled_back(tmp_path, gateway, 2)
    try:
        with pytest.raises(M.LearningAttemptInvalid) as info:
            await resumed.wait_until_idle(10.0)
    finally:
        await resumed.close()
    failure = classify_exception(info.value)
    assert (failure.kind, failure.code) == (FailureKind.INFRA, "mirix_learning")
    assert resumed.state.tolerated == {}


@pytest.mark.asyncio
async def test_a_mirix_state_of_the_code_before_tolerance_restores(tmp_path, gateway):
    """A checkpoint (or frozen state) written before tolerance has none of its keys: it
    reads back with them empty, and its rebuilt snapshot has no ``tolerated`` block, so
    it still hashes to what the checkpoint recorded."""
    from evals.skill_bench.methods.base import LearningResume

    servers = StubMirixServers(threshold=5)
    cell = mirix_cell(tmp_path, gateway)
    method = method_for(servers)
    await method.setup(cell)
    stub = servers.stub_for(cell.work_dir)
    try:
        await learn(method, learning_items("alfworld", 2))
        state, sha, dump = await _checkpoint_of(
            method, stub, cell.work_dir, tmp_path / "saved"
        )
        snapshot = await method.snapshot()
    finally:
        await method.close()
    assert "tolerated" not in snapshot
    old = {
        k: v
        for k, v in state.items()
        if k
        not in ("tolerated", "accepted_problems", "degraded_streak", "refusal_streaks")
    }
    restored = M.MirixState.from_dict(old)
    assert (restored.tolerated, restored.accepted_problems) == ({}, [])
    assert (restored.degraded_streak, restored.refusal_streaks) == (0, {})
    _restore(stub, cell.work_dir, tmp_path / "saved", dump)
    resumed = method_for(servers)
    await resumed.setup(
        replace(cell, resume=LearningResume("c0002-done", "done", 2, old, sha))
    )
    try:
        assert "tolerated" not in await resumed.snapshot()
    finally:
        await resumed.close()
