"""Agent Workflow Memory, online: prompt faithfulness, induction only on a changed success
pool with overwrite, budget trimming (the F9 input budget and the request cap
``max_request_bytes``), the frozen test phase and label isolation, offline through the
real gateway."""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import ActorCallError
from evals.skill_bench.infra.failures import FailureKind
from evals.skill_bench.methods import awm as A
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods.base import (
    FrozenMemoryMismatch,
    GATEWAY_LOCAL_TOKEN,
    CellContext,
    Injection,
    TaskView,
)
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, running_gateway
from evals.skill_bench.tests.method_contract import (
    ContractItem,
    contract_cell,
    contract_items,
    exercise_method,
)
from evals.skill_bench.tests.rollout_support import text_reply
from evals.skill_bench.tests.s1_support import (
    Router,
    first_text,
    judged,
    ledger,
    module_ast,
    normal,
    originals_match,
    read_jsonl,
)

PROMPTS = protocol.SKILL_BENCH_DIR / A.PROMPT_DIR
ORIGINALS = PROMPTS / "originals"
REPO = ORIGINALS / "agent-workflow-memory-8c0ff8c"
NOTICE = PROMPTS.parent / "NOTICE"
LUNA = protocol.MODELS["luna"]
RUN = "awm-test-a1"


# --- the induction prompt: instruction.txt, the output format for the one-shot ---------------------


def test_instruction_is_instruction_txt_with_only_the_two_substitutions():
    original = (REPO / "instruction.txt").read_text(encoding="utf-8")
    for sub in A.awm_format()["substitutions"]:
        assert original.count(sub["original"]) == 1, sub["original"]
        original = original.replace(sub["original"], sub["replacement"])
    assert A.instruction_text() == original
    assert "web" not in A.instruction_text().lower()
    assert A.instruction_text().startswith("Given a list of tasks solved by an agent,")
    assert "(input text, file or object names)" in A.instruction_text()


def line_class(line: str) -> str:
    if re.fullmatch(r"Workflow \w+: .+", line):
        return "workflow"
    if line in ("<think>", "</think>", "<action>", "</action>"):
        return line
    return "text" if line.strip() else "empty"


def test_the_output_format_describes_the_one_shots_workflow_layout():
    one_shot = (REPO / "one_shot.txt").read_text(encoding="utf-8")
    summary = one_shot.split("## Summary Workflows\n\n", 1)[1]
    # The one-shot's workflows: a title line, then think/action blocks, empty lines between.
    shown = [line_class(line) for line in summary.split("\n")]
    ours = A.output_format_text()
    sample = ours.split(":\n\n", 1)[1]
    described = [line_class(line) for line in sample.split("\n")]
    step = ["<think>", "text", "</think>", "<action>", "text", "</action>"]
    assert described == ["workflow", *step, "empty", *step]
    assert shown[: len(described)] == described
    assert ours.startswith("## Output Format\n\n")
    for web in ("click", "hover", "order", "shop", "send_msg_to_user"):
        assert web not in ours


def awm_functions() -> dict[str, Any]:
    """AWM's own format_trajectory / format_examples, executed from the verbatim copy."""
    tree = module_ast(REPO / "induce_prompt.py.txt")
    source = (REPO / "induce_prompt.py.txt").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "format_trajectory",
            "format_examples",
        ):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            exec(segment, namespace)
    return namespace


def without_observations(text: str) -> str:
    text = re.sub(
        r"Actions:\n<observation>\n.*?\n</observation>\n\n",
        "Actions:\n",
        text,
        flags=re.S,
    )
    return re.sub(r"\n<observation>\n.*?\n</observation>", "", text, flags=re.S)


def test_the_induction_prompt_is_awms_with_the_output_format_for_the_one_shot():
    items = contract_items("alfworld")[:1] + contract_items("webshop")[:1]
    trajectories = [
        (i.view.instruction, A.trajectory_steps(i.trajectory())) for i in items
    ]
    [message], cut = A.induction_messages(trajectories)
    assert cut == 0 and message["role"] == "user"
    ours = message["content"]
    awm = awm_functions()
    examples = [
        {
            "query": query,
            "think_list": [s.think for s in steps.steps],
            "action_list": [list(s.actions) for s in steps.steps],
        }
        for query, steps in trajectories
    ]
    expected = "\n\n".join(
        [A.instruction_text(), A.output_format_text(), awm["format_examples"](examples)]
    )
    assert without_observations(ours) == expected
    # The observations are there, each as its own block after its step.
    assert "<observation>\nYou arrive at countertop 1." in ours
    assert (
        "</action>\n<observation>\n'Back to Search' [SEP] 'Page 1 (Total results: 50)'"
        in ours
    )
    assert ours.endswith(
        "<action>\nclick[buy now]\n</action>\n<observation>\nThank you for shopping "
        "with us! Your score (min 0.0, max 1.0): 0.75\n</observation>\n\n"
        "## Summary Workflows"
    )
    # The prompt is one user message, as induce_prompt.py:91-96 sends it.
    source = (REPO / "induce_prompt.py.txt").read_text()
    assert 'messages=[{"role": "user", "content": prompt}]' in source
    assert "prompt = '\\n\\n'.join([args.INSTRUCTION, args.ONE_SHOT, prompt])" in source


def test_the_web_helper_lines_are_not_part_of_the_workflow():
    source = (REPO / "induce_prompt.py.txt").read_text()
    helper = A.awm_format()["dropped_helper_lines"]
    assert f'workflows += "{helper.encode("unicode_escape").decode()}"' in source
    fmt_text = json.dumps(dict(A.awm_format()))
    assert fmt_text.count("click('id')") == 1  # only in the record of what was dropped


def test_the_papers_prompt_is_an_earlier_form_of_instruction_txt():
    paper = normal((ORIGINALS / "paper_2409.07429_appendix_a1.txt").read_text())
    repo = normal((REPO / "instruction.txt").read_text())
    assert (
        "Given a list of web navigation tasks, your task is to extract the common workflows."
        in paper
    )
    assert repo.startswith(
        "Given a list of web navigation tasks, your task is to extract the common workflows "
        "to solve these tasks."
    )
    assert (
        "(input text, button strings)" in paper
        and "(input text, button strings)" in repo
    )


def test_originals_hashes_blobs_licence_and_fingerprint():
    provenance = A.awm_format()["provenance"]
    checked = originals_match(PROMPTS, provenance["repo"]["files"])
    checked += originals_match(PROMPTS, provenance["paper"]["originals"])
    assert len(checked) == 7
    assert provenance["repo"]["commit"].startswith("8c0ff8c")
    assert "Apache License" in (REPO / "LICENSE").read_text()
    assert "awm/originals/agent-workflow-memory-8c0ff8c/" in NOTICE.read_text()
    for rel in ("instruction.md", "output_format.md", "format.json"):
        assert protocol.is_fingerprinted(Path(A.PROMPT_DIR) / rel)


# --- example selection within the budgets (pure) -------------------------------------------------------


def big_item(uid: str, instruction: str, words: int, seed: int = 0) -> ContractItem:
    first = {"role": "user", "content": f"Your task is to: {instruction}"}
    obs = " ".join(f"w{seed}x{i}" for i in range(words))
    messages = (
        first,
        {"role": "assistant", "content": "<think>go</think><action>look</action>"},
        {"role": "user", "content": obs},
    )
    return ContractItem(TaskView(uid, instruction, (first,)), messages)


def pool_of(items: Sequence[ContractItem]) -> list[A.PoolEntry]:
    return [
        A.PoolEntry(
            i.view.task_uid, i.view.instruction, A.trajectory_steps(i.trajectory())
        )
        for i in items
    ]


def test_selection_keeps_the_most_recent_successes_that_fit():
    items = [
        big_item(f"t{c * 20}", f"task {c}", 1500, seed=n) for n, c in enumerate("abcd")
    ]
    pool = pool_of(items)
    everything = A.select_examples(
        pool,
        model=LUNA,
        budget_tokens=10**7,
        budget_bytes=10**9,
    )
    assert everything.dropped == 0 and everything.binding is None
    assert everything.used == tuple(i.view.task_uid for i in items)
    two = A.induction_messages([(e.instruction, e.trajectory) for e in pool[2:]])[0]
    budget = A.estimate_messages(two)
    selected = A.select_examples(
        pool,
        model=LUNA,
        budget_tokens=budget,
        budget_bytes=10**9,
    )
    assert selected.used == (items[2].view.task_uid, items[3].view.task_uid)
    assert (
        selected.dropped == 2 and selected.binding == "tokens" and selected.cap is None
    )
    assert selected.input_tokens_estimated <= budget
    size = A._measure(two, LUNA)[1]
    by_bytes = A.select_examples(
        pool,
        model=LUNA,
        budget_tokens=10**7,
        budget_bytes=size,
    )
    assert by_bytes.used == selected.used and by_bytes.binding == "bytes"
    assert by_bytes.request_bytes <= size


def test_a_single_success_over_the_budget_is_cut_head_and_tail():
    pool = pool_of(
        [
            big_item("t" + "a" * 20, "task a", 1500),
            big_item("t" + "b" * 20, "task b", 30_000),
        ]
    )
    base = A.estimate_messages(A.induction_messages([])[0])
    selected = A.select_examples(
        pool,
        model=LUNA,
        budget_tokens=base + 3_000,
        budget_bytes=10**9,
    )
    assert selected.used == ("t" + "b" * 20,) and selected.dropped == 1
    assert selected.cap is not None and selected.chars_cut > 0
    assert selected.input_tokens_estimated <= base + 3_000
    assert selected.binding == "tokens"
    assert "characters omitted" in first_text({"messages": selected.messages})


# --- the request cap ----------------------------------------------------------------------------


def test_the_request_cap_is_in_format_json():
    """29 MB, below the payload limits of the providers behind OpenRouter; format.json
    (fingerprinted) records the rule with the value."""
    fmt = A.awm_format()
    assert fmt["max_request_bytes"] == 29_000_000
    assert "HTTP 413" in fmt["max_request_bytes_rule"]
    assert A.LIMITS == ("tokens", "bytes")


def test_the_binding_limit_is_the_first_exceeded_in_the_order_tokens_bytes():
    pool = pool_of(
        [big_item(f"t{c * 20}", f"task {c}", 100, seed=n) for n, c in enumerate("ab")]
    )
    newest = A.induction_messages([(pool[1].instruction, pool[1].trajectory)])[0]
    tokens, size = A._measure(newest, LUNA)
    # The older example exceeds both limits the newest alone fills exactly.
    for budgets, binding in (((tokens, size), "tokens"), ((10**7, size), "bytes")):
        selected = A.select_examples(
            pool, model=LUNA, budget_tokens=budgets[0], budget_bytes=budgets[1]
        )
        assert selected.used == (pool[1].task_uid,) and selected.binding == binding
        assert selected.input_tokens_estimated <= budgets[0]
        assert selected.request_bytes <= budgets[1]
        assert selected.cap is None
    assert A._exceeded((9, 9), (9, 9)) is None
    assert A._exceeded((10, 10), (9, 9)) == "tokens"
    assert A._exceeded((9, 10), (9, 9)) == "bytes"


# --- the method through the gateway ---------------------------------------------------------------------


def alf_item(uid: str, instruction: str, action: str = "look") -> ContractItem:
    first = {
        "role": "user",
        "content": f"Your task is to: {instruction}\n\nAdmissible actions:\n'look'",
    }
    messages = (
        first,
        {
            "role": "assistant",
            "content": f"<think>try {action}</think><action>{action}</action>",
        },
        {"role": "user", "content": f"You did {action}."},
    )
    return ContractItem(TaskView(uid, instruction, (first,)), messages)


def awm_cell(
    tmp_path: Path,
    benchmark: str = "alfworld",
    *,
    phase: str = "learning",
    limits: protocol.ModelLimits | None = None,
) -> CellContext:
    return CellContext(
        cell_id=f"{benchmark}-test-awm",
        attempt=1,
        benchmark=benchmark,
        method="awm",
        model=LUNA,
        gateway_base_url="http://gateway",
        run_id=RUN,
        gateway_token=GATEWAY_LOCAL_TOKEN,
        work_dir=tmp_path / "work",
        phase=phase,  # type: ignore[arg-type]
        limits=limits,
    )


async def learn(
    method: A.AgentWorkflowMemory, items: Sequence[ContractItem]
) -> list[Injection]:
    injections = []
    for item in items:
        injections.append(await method.before_task(copy.deepcopy(item.view)))
        for message in item.messages:
            await method.on_message(item.view.task_uid, copy.deepcopy(message))
        await method.after_task(item.trajectory())
        await method.wait_until_idle(5.0)
    return injections


WORKFLOW_A = (
    "Workflow 1: Look around\n<think>\nlook first\n</think>\n<action>\nlook\n</action>"
)
WORKFLOW_B = "Workflow 1: Open then take\n<think>\nopen {receptacle}\n</think>\n<action>\nopen {receptacle}\n</action>"


@pytest.mark.asyncio
async def test_successes_reinduce_from_the_whole_pool_and_overwrite(tmp_path):
    router = Router(
        judge=[judged("success"), judged("failure"), judged("success")],
        learner=[WORKFLOW_A, WORKFLOW_B],
    )
    fake = FakeUpstream(responder=router)
    items = [alf_item(f"t{c * 20}", f"task {c}.", f"act {c}") for c in "abc"]
    async with running_gateway(tmp_path, fake) as (gw, _client):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            injections = await learn(method, items)
            after = await method.before_task(TaskView("t" + "d" * 20, "task d.", ()))
            other = await method.before_task(
                TaskView("t" + "e" * 20, "another task.", ())
            )
            snapshot = await method.snapshot()
        finally:
            await method.close()
    # Before the first success there is no workflow; afterwards the latest one, verbatim.
    assert injections[0] == Injection()
    assert injections[1].text == injections[2].text == WORKFLOW_A
    assert after.text == other.text == WORKFLOW_B  # overwritten, one workflow per cell
    assert after.item_ids == other.item_ids and len(after.item_ids) == 1
    assert "click('id')" not in after.text
    # Two inductions: after each success, none after the failure (the pool did not change).
    first, second = router.calls["learner"]
    assert len(router.calls["judge"]) == 3
    assert "Query: task a." in first_text(first) and "Query: task c." not in first_text(
        first
    )
    assert "Query: task a." in first_text(second) and "Query: task c." in first_text(
        second
    )
    assert "Query: task b." not in first_text(second)  # failures are discarded
    assert first_text(second).index("Query: task a.") < first_text(second).index(
        "Query: task c."
    )
    for call in router.calls["learner"]:
        assert [m["role"] for m in call["messages"]] == ["user"]
        assert "temperature" not in call
    assert snapshot["workflow"] == WORKFLOW_B and [
        p["task_uid"] for p in snapshot["pool"]
    ] == [
        "t" + "a" * 20,
        "t" + "c" * 20,
    ]
    assert [(i["used"], i["dropped"], i["wrote"]) for i in snapshot["inductions"]] == [
        (["t" + "a" * 20], 0, True),
        (["t" + "a" * 20, "t" + "c" * 20], 0, True),
    ]
    counters = snapshot["counters"]
    assert (
        counters["judged_success"],
        counters["judged_failure"],
        counters["inductions"],
    ) == (2, 1, 2)
    assert (tmp_path / "work" / A.WORKFLOW_FILE).read_text() == WORKFLOW_B
    rows = ledger(tmp_path, RUN)
    assert sorted(r["role"] for r in rows) == ["judge"] * 3 + ["learner"] * 2
    assert {r["item"] for r in rows if r["role"] == "learner"} == {
        "t" + "a" * 20,
        "t" + "c" * 20,
    }


@pytest.mark.asyncio
async def test_an_induction_without_text_keeps_the_previous_workflow(tmp_path):
    router = Router(
        judge=[judged("success")] * 4,
        learner=[
            WORKFLOW_A,
            Reply(delay_s=3.0),  # no reply: MODEL timeout
            text_reply("", finish_reason="content_filter"),
            text_reply(WORKFLOW_B, finish_reason="length"),  # truncated: used, counted
        ],
    )
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake, deadline_s=1.0) as (gw, _client):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            injections = await learn(
                method, [alf_item(f"t{c * 20}", f"task {c}.") for c in "abcd"]
            )
            final = await method.before_task(TaskView("t" + "e" * 20, "task e.", ()))
            snapshot = await method.snapshot()
        finally:
            await method.close()
    assert [i.text for i in injections] == ["", WORKFLOW_A, WORKFLOW_A, WORKFLOW_A]
    assert final.text == WORKFLOW_B
    c = snapshot["counters"]
    assert (c["inductions"], c["workflows_written"], c["induction_no_text"]) == (
        4,
        2,
        2,
    )
    assert c["induction_model_failures"] == 3
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    codes = [None, "model_timeout", "content_filter", "truncated"]
    assert [r["failure_code"] for r in rows] == codes
    assert [r["failure"]["code"] if r["failure"] else None for r in rows] == codes
    assert [i["failure_code"] for i in snapshot["inductions"]] == codes
    assert [r["wrote"] for r in rows] == [True, False, False, True]


@pytest.mark.asyncio
async def test_the_budget_drops_the_oldest_successes_and_counts_them(tmp_path):
    limits = protocol.ModelLimits(LUNA, 20_000, 4_000)  # input budget 14,000 tokens
    budget = J.input_budget(limits)
    items = [
        big_item(f"t{c * 20}", f"task {c}.", 1500, seed=n) for n, c in enumerate("abcd")
    ]
    one = A.estimate_messages(
        A.induction_messages(
            [(items[0].view.instruction, A.trajectory_steps(items[0].trajectory()))]
        )[0]
    )
    base = A.estimate_messages(A.induction_messages([])[0])
    fits = (budget - base) // (one - base)
    assert 1 <= fits < 4, "the example size must make the budget bind"
    router = Router(judge=[judged("success")] * 4, learner=[f"W{n}" for n in range(4)])
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path, limits=limits))
        try:
            await learn(method, items)
            snapshot = await method.snapshot()
        finally:
            await method.close()
    last = snapshot["inductions"][-1]
    assert last["used"] == [i.view.task_uid for i in items[4 - fits :]]
    assert last["dropped"] == 4 - fits and last["binding"] == "tokens"
    assert last["input_tokens_estimated"] <= budget
    counters = snapshot["counters"]
    assert counters["examples_dropped"] == sum(
        i["dropped"] for i in snapshot["inductions"]
    )
    assert (
        counters["max_examples_dropped"] == 4 - fits
        and counters["inductions_trimmed"] >= 1
    )
    assert "Query: task a." not in first_text(router.calls["learner"][-1])
    assert "Query: task d." in first_text(router.calls["learner"][-1])


@pytest.mark.asyncio
async def test_the_test_phase_is_read_only_and_a_restart_resumes_the_workflow(tmp_path):
    router = Router(judge=[judged("success")], learner=[WORKFLOW_A])
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, [alf_item("t" + "a" * 20, "task a.")])
            await method.finalize()
            frozen = await method.snapshot()
            injection = await method.before_task(
                TaskView("t" + "b" * 20, "task b.", ())
            )
            with pytest.raises(RuntimeError, match="read-only"):
                await method.after_task(
                    alf_item("t" + "c" * 20, "task c.").trajectory()
                )
            assert await method.snapshot() == frozen and frozen["frozen"] is True
        finally:
            await method.close()
    calls = len(fake.requests)
    resumed = A.AgentWorkflowMemory()
    await resumed.setup(awm_cell(tmp_path, phase="test"))
    try:
        assert await resumed.snapshot() == frozen
        assert (
            await resumed.before_task(TaskView("t" + "b" * 20, "task b.", ()))
            == injection
        )
    finally:
        await resumed.close()
    assert (
        injection.text == WORKFLOW_A and len(fake.requests) == calls
    )  # no call to inject
    (tmp_path / "work" / A.WORKFLOW_FILE).write_text("tampered")
    with pytest.raises(FrozenMemoryMismatch, match="does not match"):
        await A.AgentWorkflowMemory().setup(awm_cell(tmp_path, phase="test"))
    with pytest.raises(FrozenMemoryMismatch, match="no frozen memory"):
        await A.AgentWorkflowMemory().setup(awm_cell(tmp_path / "other", phase="test"))
    with pytest.raises(RuntimeError, match="empty work_dir"):
        await A.AgentWorkflowMemory().setup(awm_cell(tmp_path))


@pytest.mark.asyncio
async def test_infrastructure_failures_raise(tmp_path):
    fake = FakeUpstream(
        responder=Router(judge=[judged("success")], learner=[Reply(503)])
    )
    async with running_gateway(tmp_path, fake, max_retries=0) as (gw, _client):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            with pytest.raises(ActorCallError) as info:
                await learn(method, [alf_item("t" + "a" * 20, "task a.")])
        finally:
            await method.close()
    assert info.value.failure.kind is FailureKind.INFRA
    assert info.value.failure.role == "learner"


# --- the H6 contract with a workflow that actually changes -----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", protocol.BENCHMARKS)
async def test_the_contract_holds_with_an_induced_workflow(tmp_path, benchmark):
    router = Router(judge=[judged("success"), judged("failure")], learner=[WORKFLOW_A])
    fake = FakeUpstream(responder=router)
    async with running_gateway(tmp_path, fake) as (gw, _client):
        transport = httpx.ASGITransport(app=gw.app)
        report = await exercise_method(
            lambda: A.AgentWorkflowMemory(transport=transport),
            contract_cell(
                tmp_path / "work",
                benchmark=benchmark,
                method="awm",
                gateway_base_url="http://gateway",
                run_id=RUN,
            ),
            forbidden_strings=["ZX-GOLD-ANSWER"],
        )
    assert report.snapshot["workflow"] == WORKFLOW_A
    assert report.resumed_snapshot == report.snapshot
    assert all(i.text == WORKFLOW_A for i in report.test_injections)
    assert report.learning_injections[0] == Injection()


# --- learning checkpoints (contract item 6) ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_learning_checkpoint_rebuilds_the_pool(tmp_path):
    """The pool's successes are written once under objects/ and hard-linked by a
    checkpoint; a learning setup with ``cell.resume`` rebuilds them, so the next
    induction sends the whole pool byte for byte as an uninterrupted run does."""
    import os
    import shutil

    from evals.skill_bench.methods.base import (
        CheckpointMismatch,
        LearningResume,
        snapshot_sha256,
    )
    from evals.skill_bench.run import _copy_method_dir

    router = Router(
        judge=[judged("success")] * 4,
        learner=[WORKFLOW_A, WORKFLOW_B, WORKFLOW_A, WORKFLOW_A],
    )
    fake = FakeUpstream(responder=router)
    items = [alf_item(f"t{c * 20}", f"task {c}.", f"act {c}") for c in "abc"]
    work = tmp_path / "work"
    saved = tmp_path / "checkpoint"
    async with running_gateway(tmp_path, fake) as (gw, _client):
        transport = httpx.ASGITransport(app=gw.app)
        method = A.AgentWorkflowMemory(transport=transport)
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:2])
            state = await method.checkpoint()
            recorded = await method.snapshot()
            _copy_method_dir(work, saved)
            await learn(method, items[2:])  # discarded: its induction is the reference
        finally:
            await method.close()
        uninterrupted = router.calls["learner"][-1]["messages"]
        objects = sorted(
            p.relative_to(saved).as_posix()
            for p in (saved / "objects").rglob("*")
            if p.is_file()
        )
        assert objects == [f"objects/pool/t{c * 20}.json" for c in "ab"]
        for rel in objects:  # hard-linked, not copied
            assert os.stat(saved / rel).st_ino == os.stat(work / rel).st_ino
        assert set(state) == {"counters", "judge", "inductions", "refusal_streaks"}
        assert state["refusal_streaks"] == {}
        resume = LearningResume(
            "c0001-done", "done", 1, state, snapshot_sha256(recorded)
        )

        def restore() -> None:
            shutil.rmtree(work)
            _copy_method_dir(saved, work)

        restore()
        resumed = A.AgentWorkflowMemory(transport=transport)
        await resumed.setup(dataclasses.replace(awm_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            await learn(resumed, items[2:])
        finally:
            await resumed.close()
        assert router.calls["learner"][-1]["messages"] == uninterrupted
        assert all(
            f"Query: task {c}." in first_text({"messages": uninterrupted})
            for c in "abc"
        )

        # A missing or corrupt object is refused.
        restore()
        (work / "objects" / "pool" / f"t{'b' * 20}.json").unlink()
        with pytest.raises(CheckpointMismatch, match="missing"):
            await A.AgentWorkflowMemory(transport=transport).setup(
                dataclasses.replace(awm_cell(tmp_path), resume=resume)
            )
        restore()
        pool = work / "objects" / "pool"
        other = pool / f"t{'b' * 20}.json"
        other.unlink()  # the checkpoint's link stays intact
        other.write_bytes((pool / f"t{'a' * 20}.json").read_bytes())
        with pytest.raises(CheckpointMismatch, match="another task"):
            await A.AgentWorkflowMemory(transport=transport).setup(
                dataclasses.replace(awm_cell(tmp_path), resume=resume)
            )


def test_objects_are_write_once(tmp_path):
    path = tmp_path / "objects" / "x.json"
    A.write_once(path, b"one")
    A.write_once(path, b"one")  # the same bytes: nothing to do
    with pytest.raises(RuntimeError, match="write-once"):
        A.write_once(path, b"two")
    assert path.read_bytes() == b"one"


# --- refusals: recorded, not stopped (rule harness-stop-learning) -------------------------------

PAYLOAD_413 = Reply(
    413,
    {"error": {"code": 413, "message": "Request payload too large"}},
)
PROVIDER_400 = Reply(
    400, {"error": {"code": 400, "message": "Provider returned error"}}
)
NO_IMAGE_ENDPOINT = Reply(
    404,
    {"error": {"code": 404, "message": "No endpoints found that support image input"}},
)
#: The counters every snapshot carried before refusals were counted.
PRE_REFUSAL_COUNTERS = {
    "tasks_learned",
    "judged_success",
    "judged_failure",
    "inductions",
    "workflows_written",
    "induction_model_failures",
    "induction_no_text",
    "examples_dropped",
    "max_examples_dropped",
    "inductions_trimmed",
    "inductions_text_cut",
    "steps_skipped",
}


def alf_items(n: int) -> list[ContractItem]:
    return [alf_item(f"t{chr(97 + i) * 20}", f"task {i}.") for i in range(n)]


def test_refusal_counters_stay_out_of_a_snapshot_without_refusals():
    counters = A.AwmCounters()
    assert set(counters.to_dict()) == PRE_REFUSAL_COUNTERS
    assert set(A.REFUSAL_COUNTERS) == {
        "induction_rejected",
        "induction_retries",
        "judge_rejected",
    }
    counters.judge_rejected = 1
    assert set(counters.to_dict()) == PRE_REFUSAL_COUNTERS | {"judge_rejected"}
    assert A.AwmCounters(**counters.to_dict()) == counters
    # A counters dict of the code before refusals reads back with the new ones at 0.
    old = {key: 1 for key in PRE_REFUSAL_COUNTERS}
    restored = A.AwmCounters(**old)
    assert restored.induction_rejected == restored.judge_rejected == 0
    assert restored.to_dict() == old


def test_the_refused_retry_divisor_is_in_format_json():
    fmt = A.awm_format()
    assert fmt["refused_retry_divisor"] == 2
    rule = fmt["refused_retry_rule"]
    assert "HTTP 413" in rule and "Provider returned error" in rule
    assert "induction_rejected" in rule and "3 consecutive" in rule
    assert "refused_retry_rule" in fmt["induction_rule"]
    assert protocol.is_fingerprinted(Path(A.FORMAT_FILE))


@pytest.mark.asyncio
async def test_call_learner_returns_refusals_and_raises_the_rest(tmp_path, monkeypatch):
    from evals.skill_bench.infra import failures as F
    from evals.skill_bench.infra import tokens

    messages = [{"role": "user", "content": "induce"}]
    router = Router(learner=[PAYLOAD_413, Reply(503)])
    async with running_gateway(
        tmp_path, FakeUpstream(responder=router), max_retries=0
    ) as (gw, _client):
        learner = A.learner_client(awm_cell(tmp_path), httpx.ASGITransport(app=gw.app))
        try:
            # AWM's induction prompt tells the Router this is a learner call.
            prompt = [{"role": "user", "content": A.instruction_text() + " x"}]
            refused = await A.call_learner(learner, prompt, index=0, item="t")
            assert refused.content is None and refused.failure is None
            assert refused.refused is not None
            assert (refused.refused.code, refused.refused.status) == (
                F.UPSTREAM_REJECTED,
                413,
            )
            assert refused.record()["refused"] == refused.refused.to_dict()
            assert refused.record()["http_status"] == 413
            with pytest.raises(ActorCallError) as info:
                await A.call_learner(learner, prompt, index=1, item="t")
            assert info.value.failure.kind is FailureKind.INFRA
            monkeypatch.setattr(tokens, "count_text", lambda text: 10**7)
            with pytest.raises(ActorCallError) as info:
                await A.call_learner(learner, messages, index=2, item="t")
            assert info.value.failure.code == F.GATEWAY_REJECTED
        finally:
            await learner.aclose()
    served = A.LearnerResult("x", None, None)
    assert "refused" not in served.record()


@pytest.mark.asyncio
async def test_a_refused_verdict_skips_the_item(tmp_path):
    router = Router(
        judge=[PROVIDER_400, judged("success")],
        learner=[WORKFLOW_A],
    )
    items = alf_items(2)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:1])
            assert method.workflow == "" and method._pool == []
            await learn(method, items[1:])
            snapshot = await method.snapshot()
            state = await method.checkpoint()
        finally:
            await method.close()
    assert snapshot["workflow"] == WORKFLOW_A
    assert [p["task_uid"] for p in snapshot["pool"]] == [items[1].view.task_uid]
    counters = snapshot["counters"]
    assert counters["judge_rejected"] == 1
    assert (counters["tasks_learned"], counters["judged_success"]) == (1, 1)
    assert "induction_rejected" not in counters and "induction_retries" not in counters
    assert state["refusal_streaks"] == {}  # the served verdict ended the streak
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    assert [r["event"] for r in rows] == ["judge_refused", "induction"]
    event = rows[0]
    assert event["task_uid"] == items[0].view.task_uid
    assert event["http_status"] == 400 and event["gateway_call_id"]
    assert event["failure"]["code"] == "upstream_rejected"
    assert event["failure"]["role"] == "judge" and "provider" in event
    assert "refused" not in rows[1] and "retry" not in rows[1]
    assert len(router.calls["learner"]) == 1


@pytest.mark.asyncio
async def test_three_refused_verdicts_in_a_row_stop_the_learning_phase(tmp_path):
    from evals.skill_bench.infra import failures as F

    router = Router(
        judge=[
            PROVIDER_400,
            PROVIDER_400,
            judged("failure"),  # a served verdict resets the streak
            PROVIDER_400,
            PROVIDER_400,
            PROVIDER_400,
        ]
    )
    items = alf_items(6)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            # The third refusal in a row: after_task returns, the next hook stops (F19).
            await learn(method, items)
            snapshot = await method.snapshot()
            with pytest.raises(F.GatewayCallError) as info:
                await method.finalize()
        finally:
            await method.close()
    stop = info.value.failure
    assert (stop.kind, stop.code, stop.role, stop.status) == (
        FailureKind.HARNESS,
        F.REFUSAL_STREAK,
        "judge",
        400,
    )
    assert F.classify_exception(info.value) == stop
    assert isinstance(info.value.__cause__, ActorCallError)
    assert snapshot["counters"]["judge_rejected"] == 5
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    assert [r["event"] for r in rows] == ["judge_refused"] * 5


@pytest.mark.asyncio
async def test_a_streak_stop_waits_for_the_items_checkpoint_so_a_person_can_go_on(
    tmp_path,
):
    """The third refused J call in a row does not raise inside after_task,
    before ``run.py``'s ``done`` checkpoint of the item: a --force-resume from ``played``
    would replay the same refused call and stop again, so a person could never go on.
    after_task returns (the done checkpoint holds the streak), and the method's next
    hook (the next item's before_task, or finalize after the last item) raises the stop.
    A learning phase resumed from that checkpoint goes on with the next item."""
    from evals.skill_bench.infra import failures as F
    from evals.skill_bench.methods.base import LearningResume, snapshot_sha256

    router = Router(judge=[PROVIDER_400] * 3 + [judged("failure")])
    items = alf_items(4)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        transport = httpx.ASGITransport(app=gw.app)
        method = A.AgentWorkflowMemory(transport=transport)
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:3])  # the third refusal: after_task returns
            state = await method.checkpoint()
            recorded = await method.snapshot()
            with pytest.raises(F.GatewayCallError) as info:
                await method.before_task(copy.deepcopy(items[3].view))
            with pytest.raises(F.GatewayCallError) as at_freeze:
                await method.finalize()
        finally:
            await method.close()
        assert state["refusal_streaks"] == {
            "judge": {"signature": "judge/payload/400", "count": 3}
        }
        resume = LearningResume(
            "c0002-done",
            "done",
            2,
            json.loads(json.dumps(state)),
            snapshot_sha256(recorded),
        )
        resumed = A.AgentWorkflowMemory(transport=transport)
        await resumed.setup(dataclasses.replace(awm_cell(tmp_path), resume=resume))
        try:
            await learn(resumed, items[3:])  # a person went on: the next item is served
            after = await resumed.checkpoint()
        finally:
            await resumed.close()
    for raised in (info.value, at_freeze.value):
        stop = raised.failure
        assert (stop.kind, stop.code, stop.role, stop.status) == (
            FailureKind.HARNESS,
            F.REFUSAL_STREAK,
            "judge",
            400,
        )
        assert isinstance(raised.__cause__, ActorCallError)
    assert recorded["counters"]["judge_rejected"] == 3
    assert after["refusal_streaks"] == {}  # the served verdict ended the streak


def big_items(n: int) -> list[ContractItem]:
    return [
        big_item(f"t{chr(97 + i) * 20}", f"task {i}.", 1500, seed=i) for i in range(n)
    ]


@pytest.mark.asyncio
async def test_a_payload_refusal_is_retried_with_halved_budgets(tmp_path):
    router = Router(
        judge=[judged("success")] * 4,
        learner=[WORKFLOW_A, WORKFLOW_A, WORKFLOW_A, PAYLOAD_413, WORKFLOW_B],
    )
    items = big_items(4)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items)
            snapshot = await method.snapshot()
            state = await method.checkpoint()
        finally:
            await method.close()
    refused_request, retried_request = router.calls["learner"][-2:]
    last = snapshot["inductions"][-1]
    kept = len(last["used"])
    assert 1 <= kept <= 4 // 2
    # The newest successes are kept (the oldest are dropped first).
    assert all(f"Query: task {n}." in first_text(refused_request) for n in range(4))
    assert [f"Query: task {n}." in first_text(retried_request) for n in range(4)] == [
        n >= 4 - kept for n in range(4)
    ]
    assert snapshot["workflow"] == WORKFLOW_B
    uids = [i.view.task_uid for i in items]
    assert last["used"] == uids[4 - kept :]
    assert last["wrote"] is True
    retry = last["retry"]
    assert retry["refusal"]["status"] == 413 and retry["refusal"]["role"] == "learner"
    assert (retry["used"], retry["dropped"]) == (uids, 0)
    assert retry["request_bytes"] > last["request_bytes"] * 2 - 1
    assert retry["gateway_call_id"] and "provider" in retry
    assert "rejected" not in last and "provider" not in last
    assert all("retry" not in i for i in snapshot["inductions"][:-1])
    counters = snapshot["counters"]
    assert counters["induction_retries"] == 1 and "induction_rejected" not in counters
    assert counters["inductions"] == 4 and counters["workflows_written"] == 4
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    assert rows[-1]["retry"] == retry and "refused" not in rows[-1]
    assert state["inductions"][-1] == last and state["refusal_streaks"] == {}


@pytest.mark.asyncio
async def test_a_refused_retry_keeps_the_previous_workflow(tmp_path):
    router = Router(
        judge=[judged("success")] * 2,
        learner=[WORKFLOW_A, PROVIDER_400, PAYLOAD_413],
    )
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, alf_items(2))
            snapshot = await method.snapshot()
            state = await method.checkpoint()
        finally:
            await method.close()
    assert snapshot["workflow"] == WORKFLOW_A
    assert (tmp_path / "work" / A.WORKFLOW_FILE).read_text() == WORKFLOW_A
    last = snapshot["inductions"][-1]
    assert last["wrote"] is False and last["failure_code"] is None
    assert last["rejected"]["status"] == 413 and "provider" in last
    assert last["retry"]["refusal"]["status"] == 400
    counters = snapshot["counters"]
    assert (counters["induction_rejected"], counters["induction_retries"]) == (1, 1)
    assert counters["induction_no_text"] == 0 and counters["workflows_written"] == 1
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    assert rows[-1]["refused"]["status"] == 413 and rows[-1]["http_status"] == 413
    assert rows[-1]["rejected"] == last["rejected"]
    assert state["refusal_streaks"] == {
        "learner": {"signature": "learner/payload/413", "count": 1}
    }
    assert len(router.calls["learner"]) == 3


@pytest.mark.asyncio
async def test_a_no_endpoint_refusal_is_not_retried(tmp_path):
    router = Router(
        judge=[judged("success")] * 2, learner=[NO_IMAGE_ENDPOINT, WORKFLOW_A]
    )
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, alf_items(1))
            assert len(router.calls["learner"]) == 1  # no retry
            assert method.workflow == ""
            await learn(method, alf_items(2)[1:])
            snapshot = await method.snapshot()
        finally:
            await method.close()
    first = snapshot["inductions"][0]
    assert first["rejected"]["status"] == 404 and "retry" not in first
    counters = snapshot["counters"]
    assert counters["induction_rejected"] == 1 and "induction_retries" not in counters
    assert snapshot["workflow"] == WORKFLOW_A


@pytest.mark.asyncio
async def test_three_refused_inductions_in_a_row_stop_the_learning_phase(tmp_path):
    from evals.skill_bench.infra import failures as F

    router = Router(
        judge=[judged("success")] * 6,
        learner=[
            NO_IMAGE_ENDPOINT,
            NO_IMAGE_ENDPOINT,
            WORKFLOW_A,  # a served induction resets the streak
            NO_IMAGE_ENDPOINT,
            NO_IMAGE_ENDPOINT,
            NO_IMAGE_ENDPOINT,
        ],
    )
    items = alf_items(6)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        method = A.AgentWorkflowMemory(transport=httpx.ASGITransport(app=gw.app))
        await method.setup(awm_cell(tmp_path))
        try:
            # The third refusal in a row: after_task returns, the next hook stops (F19).
            await learn(method, items)
            snapshot = await method.snapshot()
            with pytest.raises(F.GatewayCallError) as info:
                await method.finalize()
        finally:
            await method.close()
    stop = info.value.failure
    assert (stop.kind, stop.code, stop.role, stop.status) == (
        FailureKind.HARNESS,
        F.REFUSAL_STREAK,
        "learner",
        404,
    )
    assert "learner/no_endpoint/404" in stop.detail
    # The third refusal was recorded before the stop.
    assert len(snapshot["inductions"]) == 6
    assert snapshot["counters"]["induction_rejected"] == 5
    assert snapshot["workflow"] == WORKFLOW_A
    rows = read_jsonl(tmp_path / "work" / A.INDUCTIONS_FILE)
    assert len(rows) == 6 and rows[-1]["rejected"]["status"] == 404


@pytest.mark.asyncio
async def test_a_checkpoint_of_the_code_before_refusals_restores_as_recorded(tmp_path):
    """A checkpoint written before refusals were counted has no ``refusal_streaks`` and
    no refusal counters: it restores to its recorded snapshot hash (resume across
    versions), and learning goes on."""
    from evals.skill_bench.methods.base import LearningResume, snapshot_sha256

    router = Router(judge=[judged("success")] * 3, learner=[WORKFLOW_A] * 3)
    items = alf_items(3)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        transport = httpx.ASGITransport(app=gw.app)
        method = A.AgentWorkflowMemory(transport=transport)
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:2])
            state = await method.checkpoint()
            recorded = await method.snapshot()
        finally:
            await method.close()
        del state["refusal_streaks"]  # as the code before refusals wrote it
        assert set(recorded["counters"]) == PRE_REFUSAL_COUNTERS
        resume = LearningResume(
            "c0002-done", "done", 2, state, snapshot_sha256(recorded)
        )
        resumed = A.AgentWorkflowMemory(transport=transport)
        await resumed.setup(dataclasses.replace(awm_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            await learn(resumed, items[2:])
            after = await resumed.checkpoint()
        finally:
            await resumed.close()
    assert after["refusal_streaks"] == {} and after["counters"]["inductions"] == 3


@pytest.mark.parametrize("key_present", [False, True], ids=["absent", "present"])
@pytest.mark.asyncio
async def test_a_checkpoint_with_or_without_a_counter_restores_as_recorded(
    tmp_path, key_present
):
    """A learning checkpoint of code from before a counter existed has no key
    for it; one written later holds it, often as 0. Both rebuild to the snapshot they
    recorded, so the resume goes on under this code instead of stopping on
    checkpoint_mismatch. The counter comes back once it counts something
    (``inductions_text_cut`` stands for such a counter here)."""
    from evals.skill_bench.methods.base import LearningResume, snapshot_sha256

    router = Router(judge=[judged("success")] * 3, learner=[WORKFLOW_A] * 3)
    items = alf_items(3)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        transport = httpx.ASGITransport(app=gw.app)
        method = A.AgentWorkflowMemory(transport=transport)
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:2])
            state = json.loads(json.dumps(await method.checkpoint()))
            recorded = json.loads(json.dumps(await method.snapshot()))
        finally:
            await method.close()
        assert state["counters"]["inductions_text_cut"] == 0
        if not key_present:  # as code from before the counter wrote them
            del state["counters"]["inductions_text_cut"]
            del recorded["counters"]["inductions_text_cut"]
            del state["refusal_streaks"]
        resume = LearningResume(
            "c0002-done", "done", 2, state, snapshot_sha256(recorded)
        )
        resumed = A.AgentWorkflowMemory(transport=transport)
        await resumed.setup(dataclasses.replace(awm_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            await learn(resumed, items[2:])
            after = await resumed.checkpoint()
        finally:
            await resumed.close()
    assert after["counters"]["inductions"] == 3
    assert ("inductions_text_cut" in after["counters"]) is key_present
    fresh = A.AwmCounters()
    assert fresh.to_dict()["inductions_text_cut"] == 0  # fresh counters: as before
    old = A.AwmCounters.restore(
        {k: 0 for k in PRE_REFUSAL_COUNTERS - {"inductions_text_cut"}}
    )
    assert "inductions_text_cut" not in old.to_dict()
    old.inductions_text_cut = 1
    assert old.to_dict()["inductions_text_cut"] == 1
    with pytest.raises(TypeError):
        A.AwmCounters.restore({"no_such_counter": 1})


def test_a_checkpoint_mismatch_does_not_recommend_force_resume():
    """A --force-resume rebuilds the memory the same way, so the stop of a
    checkpoint_mismatch names a code fix or --new-attempt, never --force-resume."""
    from evals.skill_bench import run as R

    text = R._harness_stop_choices(R.FAILURE_CHECKPOINT_MISMATCH, checkpointed=True)
    assert "--force-resume would stop the same way" in text
    assert "--new-attempt" in text
    other = R._harness_stop_choices("other_code", checkpointed=True)
    assert other.startswith("A person picks --force-resume")
    refusal = R.CellRunner._needs_decision_refusal(
        2, {"failure": {"code": R.FAILURE_CHECKPOINT_MISMATCH, "detail": "x"}}
    )
    assert "pass --force-resume" not in refusal and "--new-attempt" in refusal
    refusal = R.CellRunner._needs_decision_refusal(
        2, {"failure": {"code": "exception", "detail": "x"}}
    )
    assert "pass --force-resume" in refusal


@pytest.mark.asyncio
async def test_a_checkpoint_restores_the_refusal_streaks(tmp_path):
    from evals.skill_bench.infra import failures as F
    from evals.skill_bench.methods.base import LearningResume, snapshot_sha256

    router = Router(judge=[judged("success")] * 3, learner=[PAYLOAD_413] * 5)
    items = alf_items(3)
    async with running_gateway(tmp_path, FakeUpstream(responder=router)) as (gw, _):
        transport = httpx.ASGITransport(app=gw.app)
        method = A.AgentWorkflowMemory(transport=transport)
        await method.setup(awm_cell(tmp_path))
        try:
            await learn(method, items[:2])  # two refused inductions (each retried)
            state = await method.checkpoint()
            recorded = await method.snapshot()
        finally:
            await method.close()
        assert state["refusal_streaks"] == {
            "learner": {"signature": "learner/payload/413", "count": 2}
        }
        assert recorded["counters"]["induction_rejected"] == 2
        assert recorded["counters"]["induction_retries"] == 2
        resume = LearningResume(
            "c0002-done",
            "done",
            2,
            json.loads(json.dumps(state)),
            snapshot_sha256(recorded),
        )
        resumed = A.AgentWorkflowMemory(transport=transport)
        await resumed.setup(dataclasses.replace(awm_cell(tmp_path), resume=resume))
        try:
            assert await resumed.snapshot() == recorded
            router.learner = [PAYLOAD_413, PAYLOAD_413]
            await learn(resumed, items[2:])
            with pytest.raises(F.GatewayCallError) as info:
                await resumed.before_task(copy.deepcopy(items[0].view))
        finally:
            await resumed.close()
    assert info.value.failure.code == F.REFUSAL_STREAK
