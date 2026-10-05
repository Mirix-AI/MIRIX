"""Every memory method keeps the method contract (methods/base.py).

Register each method implemented in ``methods/`` in :data:`CONTRACT_METHODS` (a
zero-argument factory). The registry test fails for an implemented method that is
missing, so no method escapes the checks below: the whole-cell exercise on
both benchmarks (with the empty assistant turn and every other Q20 message
shape) and the real ALFWorld rollout whose model call gets no reply. The method's
own model calls reach a real gateway over the fake upstream, which answers every
call with a generic reply: a method must survive output it cannot parse.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.trajectory import visible_message
from evals.skill_bench.infra import gateway as gw_mod
from evals.skill_bench.methods import base
from evals.skill_bench.methods.base import (
    CellContext,
    Injection,
    MemoryMethod,
    MemoryMethodBase,
    TaskView,
    TrajectoryView,
    method_class,
)
from evals.skill_bench.methods.awm import AgentWorkflowMemory
from evals.skill_bench.methods.mirix import MirixMethod
from evals.skill_bench.methods.none import NoMemory
from evals.skill_bench.methods.reasoningbank import ReasoningBank
from evals.skill_bench.tests.fakes import FakeUpstream, serve_gateway
from evals.skill_bench.tests.method_contract import (
    EMPTY_ASSISTANT,
    CheckpointHooks,
    ObjectWatch,
    contract_cell,
    contract_items,
    exercise_checkpoint_resume,
    exercise_method,
    play_alfworld_no_reply_step,
    unregistered_methods,
)
from evals.skill_bench.tests.mirix_stub import StubMirixServers

#: The MIRIX method runs against the stub MIRIX server of ``tests/mirix_stub.py`` (one
#: stub "database" per cell directory, kept across the test-phase restart);
#: ``tests/test_mirix_method.py`` also has the real-MIRIX variant (integration).
_MIRIX_STUBS = StubMirixServers()


def _mirix_on_stub() -> MirixMethod:
    return MirixMethod(server_factory=_MIRIX_STUBS, poll_s=0.01)


#: Every implemented method, by its protocol name. Add yours here (a zero-argument
#: factory); methods that need more (MIRIX's Postgres) add a skip mark to
#: :data:`CONTRACT_MARKS`.
CONTRACT_METHODS: dict[str, Callable[[], Any]] = {
    "none": NoMemory,
    "mirix": _mirix_on_stub,
    "reasoningbank": ReasoningBank,
    "awm": AgentWorkflowMemory,
}
CONTRACT_MARKS: dict[str, list[pytest.MarkDecorator]] = {}


def _params() -> list[Any]:
    return [
        pytest.param(name, factory, marks=CONTRACT_MARKS.get(name, []), id=name)
        for name, factory in CONTRACT_METHODS.items()
    ]


# --- the registry -------------------------------------------------------------------------


def test_every_implemented_method_is_under_the_contract():
    assert unregistered_methods(CONTRACT_METHODS) == []
    assert set(CONTRACT_METHODS) <= set(protocol.METHODS)
    for name in CONTRACT_METHODS:
        assert method_class(name) is not None, (
            f"{name} is registered but not implemented"
        )


def test_method_class_finds_the_implementation_or_none():
    assert method_class("none") is NoMemory
    placeholders = [n for n in protocol.METHODS if method_class(n) is None]
    assert "none" not in placeholders
    with pytest.raises(ValueError, match="unknown method"):
        method_class("webarena")


def test_a_misnamed_method_does_not_pass_for_a_placeholder(monkeypatch):
    import sys
    import types

    module = types.ModuleType("evals.skill_bench.methods.awm")

    class Misnamed(MemoryMethodBase):
        name = "agent-workflow-memory"

    Misnamed.__module__ = module.__name__
    module.Misnamed = Misnamed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="none is named 'awm'"):
        method_class("awm")
    Misnamed.name = "awm"
    assert method_class("awm") is Misnamed
    registry = {k: v for k, v in CONTRACT_METHODS.items() if k != "awm"}
    with pytest.raises(AssertionError):  # implemented but not registered
        assert unregistered_methods(registry) == []


# --- the contract items ----------------------------------------------------------------------


@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
def test_contract_items_carry_every_q20_shape(benchmark):
    items = contract_items(benchmark)
    messages = [m for item in items for m in item.messages]
    for message in messages:
        assert visible_message(message) == message  # exactly what the rollout emits
    assert EMPTY_ASSISTANT in messages
    for item in items:
        assert (
            item.messages[: len(item.view.first_messages)] == item.view.first_messages
        )
    if benchmark == "webshop":
        from evals.skill_bench.envs.webshop import WebShopEnv

        env = WebShopEnv()
        last = items[0].messages[-1]["content"]
        assert (
            last
            == "Thank you for shopping with us! Your score (min 0.0, max 1.0): 0.75"
        )
        for item in items:  # what a real episode sends passes the structural leak check
            assert env.leak_problems(item.messages) == []
            assert item.view.instruction in item.view.first_messages[0]["content"]
            assert all(
                item.view.instruction not in m["content"] for m in item.messages[1:]
            )


# --- CellContext --------------------------------------------------------------------------------


def test_cell_context_carries_the_models_json_limits_and_the_phase(tmp_path):
    cell = contract_cell(
        tmp_path, benchmark="webshop", method="none", model=protocol.MODELS["luna56"]
    )
    limits = protocol.model_limits(protocol.MODELS["luna56"])
    assert cell.model_limits == limits and cell.limits == limits
    assert (cell.context_length, cell.max_output_tokens) == (1_050_000, 128_000)
    assert cell.model_limits.tokenizer_margin == protocol.TOKENIZER_MARGIN
    assert cell.phase == "learning" and cell.learning and cell.cell_dir == cell.work_dir
    test = contract_cell(tmp_path, benchmark="webshop", method="none", phase="test")
    assert not test.learning
    with pytest.raises(ValueError, match="phase"):
        contract_cell(tmp_path, benchmark="webshop", method="none", phase="frozen")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disagrees with config/models.json"):
        CellContext(
            cell_id="c",
            attempt=1,
            benchmark="alfworld",
            method="none",
            model=protocol.MODELS["luna"],
            gateway_base_url="http://127.0.0.1:1",
            run_id="c-a1",
            gateway_token="t",
            work_dir=tmp_path,
            context_length=128_000,
        )
    with pytest.raises(ValueError, match="embedding"):
        CellContext(
            cell_id="c",
            attempt=1,
            benchmark="alfworld",
            method="none",
            model=protocol.MODELS["luna"],
            gateway_base_url="http://127.0.0.1:1",
            run_id="c-a1",
            gateway_token="t",
            work_dir=tmp_path,
            embedding_model="openai/text-embedding-3-small",
        )


def test_cell_context_gateway_urls_and_headers(tmp_path):
    cell = contract_cell(
        tmp_path / "work",
        benchmark="alfworld",
        method="none",
        gateway_base_url="http://127.0.0.1:18540/",
        run_id="alfworld-luna-none-a1",
        mirix_port=19000,
    )
    assert dict(cell.role_urls) == {
        role: f"http://127.0.0.1:18540/runs/alfworld-luna-none-a1/{role}/v1"
        for role in ("learner", "judge", "embedding", "mirix")
    }
    for role in cell.role_urls:  # the same scheme as the gateway's own helper
        assert cell.role_url(role) == gw_mod.gateway_url(
            cell.gateway_base_url, cell.run_id, role
        )
    with pytest.raises(ValueError):
        cell.role_url("actor")
    assert cell.headers() == {"Authorization": f"Bearer {gw_mod.LOCAL_TOKEN}"}
    assert cell.headers("t" + "1" * 20)["x-skill-bench-item"] == "t" + "1" * 20
    assert base.GATEWAY_LOCAL_TOKEN == gw_mod.LOCAL_TOKEN
    assert cell.options["mirix_port"] == 19000
    with pytest.raises(TypeError):
        cell.options["mirix_port"] = 1  # type: ignore[index]
    assert gw_mod.LOCAL_TOKEN not in repr(cell)


# --- the whole-cell exercise ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
async def test_base_and_no_memory_methods_keep_the_contract(tmp_path, benchmark):
    for name, factory in (("none", NoMemory), ("base", MemoryMethodBase)):
        cell = contract_cell(tmp_path / name, benchmark=benchmark, method=name)
        report = await exercise_method(factory, cell)
        assert all(
            i == Injection()
            for i in report.learning_injections + report.test_injections
        )
        assert report.resumed_snapshot == report.snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
@pytest.mark.parametrize(("name", "factory"), _params())
async def test_registered_methods_keep_the_contract(tmp_path, benchmark, name, factory):
    fake = FakeUpstream()
    with serve_gateway(tmp_path, fake) as (_gateway, base_url):
        cell = contract_cell(
            tmp_path / "work",
            benchmark=benchmark,
            method=name,
            gateway_base_url=base_url,
        )
        report = await exercise_method(
            factory, cell, forbidden_strings=["ZX-GOLD-ANSWER"]
        )
    assert json.loads(json.dumps(report.snapshot)) == report.snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _params())
async def test_registered_methods_accept_the_rollouts_empty_assistant_turn(
    tmp_path, name, factory
):
    run, _method = await play_alfworld_no_reply_step(
        tmp_path, factory, method_name=name
    )
    assert run.result.score is not None and run.result.score.primary == 1.0
    assert [f["code"] for f in run.trajectory.model_failures] == ["model_timeout"]


# --- the agent context ----------------------------------------------------------------------------


def _agent_context(benchmark: str) -> str:
    from evals.skill_bench.envs import alfworld as alf
    from evals.skill_bench.envs import webshop as ws

    if benchmark == "alfworld":
        return alf.agent_context(alf.AlfWorldTemplates.load())
    return ws.agent_context(ws.WebShopTemplates.load())


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
@pytest.mark.parametrize("name", ["reasoningbank", "awm"])
async def test_the_baselines_ignore_the_agent_context(tmp_path, benchmark, name):
    """The agent context reaches every method, and ReasoningBank and AWM keep their
    published pipelines. With the benchmark's agent context in every ``TaskView``, a
    whole cell (learning, freeze, test phase, test-phase restart) sends the same model
    requests, injects the same texts and freezes the same memory as without it. The
    fake upstream answers their learners and judge in their own formats
    (``e2e_support.E2EModel``), so they really learn something to compare."""
    import shutil

    from evals.skill_bench.tests.e2e_support import E2EModel

    context = _agent_context(benchmark)
    assert "{current_observation}" in context
    runs: list[dict[str, Any]] = []
    for agent_context in ("", context):
        root = tmp_path / "cell"  # the same paths for both runs
        shutil.rmtree(root, ignore_errors=True)
        items = [
            replace(item, view=replace(item.view, agent_context=agent_context))
            for item in contract_items(benchmark)
        ]
        fake = FakeUpstream(responder=E2EModel())
        with serve_gateway(root, fake) as (_gateway, base_url):
            cell = contract_cell(
                root / "work",
                benchmark=benchmark,
                method=name,
                gateway_base_url=base_url,
            )
            report = await exercise_method(CONTRACT_METHODS[name], cell, items=items)
        runs.append(
            {
                # The test phase's calls run concurrently: compared as a multiset.
                "requests": sorted(
                    json.dumps([r.path, r.json], sort_keys=True)
                    for r in fake.requests
                    if r.path in ("chat/completions", "embeddings")
                ),
                "learning": [i.to_dict() for i in report.learning_injections],
                "test": [i.to_dict() for i in report.test_injections],
                "snapshot": _without_times(report.snapshot),
            }
        )
    without, with_context = runs
    assert with_context == without
    learned = [i for i in without["test"] if i["text"].strip()]
    assert learned, "the baseline learned nothing: the comparison would be empty"
    assert any("chat/completions" in r for r in without["requests"])
    assert all(context not in r for r in with_context["requests"])


# --- contract item 6: learning checkpoints ----------------------------------------------------


def _without_times(value: Any) -> Any:
    """A snapshot without the wall-clock times a run records (two runs differ in them),
    and without MIRIX's hash of its export, which covers those times."""
    if isinstance(value, dict):
        return {
            k: _without_times(v)
            for k, v in value.items()
            if not (
                k.endswith("_at")
                or k in ("at", "ts", "timestamp", "frozen_snapshot_sha256")
            )
        }
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


def _mirix_hooks() -> CheckpointHooks:
    """The stub MIRIX's "database" is dumped and restored with the checkpoint, as run.py
    dumps and restores the cell database."""

    def stub(cell: CellContext) -> Any:
        return _MIRIX_STUBS.stub_for(Path(cell.cell_dir or cell.work_dir))

    return CheckpointHooks(
        capture=lambda cell: stub(cell).dump_state(),
        restore=lambda cell, state: stub(cell).load_state(state),
        normalize=_without_times,
    )


CHECKPOINT_HOOKS: dict[str, Callable[[], CheckpointHooks]] = {"mirix": _mirix_hooks}


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [("init", 0), ("done", 0), ("played", 1)], ids=str)
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
@pytest.mark.parametrize(("name", "factory"), _params())
async def test_registered_methods_resume_from_a_learning_checkpoint(
    tmp_path, benchmark, name, factory, cut
):
    """Contract item 6: checkpointed, killed after doing more, restored and continued, a
    method freezes the memory an uninterrupted run with the same replies freezes."""
    fake = FakeUpstream()
    hooks = CHECKPOINT_HOOKS.get(name, CheckpointHooks)()
    with serve_gateway(tmp_path, fake) as (_gateway, base_url):
        cells = [
            contract_cell(
                tmp_path / where,
                benchmark=benchmark,
                method=name,
                gateway_base_url=base_url,
            )
            for where in ("work", "reference")
        ]
        report = await exercise_checkpoint_resume(
            factory, cells[0], cells[1], cut=cut, hooks=hooks
        )
    assert json.loads(json.dumps(report["state"])) == report["state"]


class _KeepsMemoryInProcess(MemoryMethodBase):
    """Learns into process memory and neither checkpoints nor restores it."""

    name = "in-process"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    async def after_task(self, trajectory: TrajectoryView) -> None:
        self.seen.append(trajectory.task_uid)

    async def snapshot(self) -> dict[str, Any]:
        return {"seen": list(self.seen)}


class _RewritesObjects(MemoryMethodBase):
    """Writes a file under objects/ and changes it later (item 6 forbids that)."""

    name = "rewrites"

    async def after_task(self, trajectory: TrajectoryView) -> None:
        assert self.cell is not None
        path = self.cell.work_dir / "objects" / "pool.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trajectory.task_uid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (_KeepsMemoryInProcess, "rebuilt from the checkpoint differs"),
        (_RewritesObjects, "modified after it was written"),
    ],
)
async def test_the_checkpoint_contract_catches_broken_methods(
    tmp_path, factory, message
):
    cells = [
        contract_cell(tmp_path / where, benchmark="alfworld", method="none")
        for where in ("work", "reference")
    ]
    with pytest.raises(AssertionError, match=message):
        await exercise_checkpoint_resume(factory, cells[0], cells[1], cut=("done", 0))


def test_the_object_watch_sees_a_rewrite(tmp_path):
    watch = ObjectWatch(tmp_path)
    assert watch.check() == 0
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "a").write_text("1")
    assert watch.check() == 1
    (tmp_path / "objects" / "a").write_text("2")
    with pytest.raises(AssertionError):
        watch.check()


# --- the checks themselves catch broken methods ------------------------------------------------


class _RejectsEmpty(MemoryMethodBase):
    name = "rejects-empty"

    async def on_message(self, task_uid: str, message: dict[str, Any]) -> None:
        if message["role"] == "assistant" and not message.get("content"):
            raise ValueError("empty assistant message")


class _Forgets(MemoryMethodBase):
    """Keeps memory in process only: a test-phase restart comes back empty."""

    name = "forgets"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    async def after_task(self, trajectory: TrajectoryView) -> None:
        self.seen.append(trajectory.task_uid)

    async def snapshot(self) -> dict[str, Any]:
        return {"seen": list(self.seen)}


class _WritesWhenTested(MemoryMethodBase):
    name = "writes"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def before_task(self, task: TaskView) -> Injection:
        self.calls += 1
        return Injection(f"call {self.calls}")

    async def snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls}


class _Leaks(MemoryMethodBase):
    name = "leaks"

    async def before_task(self, task: TaskView) -> Injection:
        return Injection("the answer is ZX-GOLD-ANSWER")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (_RejectsEmpty, "empty assistant message"),
        (_Forgets, "did not resume the frozen memory"),
        (_WritesWhenTested, "changed the frozen memory"),
        (_Leaks, "zx-gold-answer"),
    ],
)
async def test_the_contract_catches_broken_methods(tmp_path, factory, message):
    cell = contract_cell(tmp_path, benchmark="alfworld", method="none")
    with pytest.raises((AssertionError, ValueError), match=message):
        await exercise_method(factory, cell, forbidden_strings=["ZX-GOLD-ANSWER"])


@pytest.mark.asyncio
async def test_the_rollout_check_catches_a_method_that_rejects_the_empty_turn(tmp_path):
    from evals.skill_bench.agent.rollout import RolloutError

    with pytest.raises(RolloutError) as info:
        await play_alfworld_no_reply_step(tmp_path, _RejectsEmpty, method_name="none")
    assert info.value.stage == "on_message"


def test_views_handed_to_methods_are_copies(tmp_path):
    item = contract_items("alfworld")[0]
    trajectory = item.trajectory()
    trajectory.messages[0]["content"] = "tampered"
    assert item.messages[0]["content"] != "tampered"
    assert isinstance(item.view, TaskView)
    assert copy.deepcopy(item.view) == item.view
    assert isinstance(NoMemory(), MemoryMethod)
    assert (
        Path(contract_cell(tmp_path, benchmark="webshop", method="none").work_dir)
        == tmp_path
    )
