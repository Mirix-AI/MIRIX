"""audit.py (H10, H9): every kind of leak or breach is caught.

Each test copies cells of the offline matrix (``e2e_matrix``, ``tests/e2e_support.py``)
with their gateway ledgers, injects one leak or breach into the copy's records, and checks
that the audit reports exactly the violations that injection must cause. The untouched
copies pass, including the signal the protocol accepts: the ALFWorld termination signal
(an episode shorter than 50 steps was won, H1.7).
A MIRIX cell on the stub MIRIX shows the MIRIX rules: its application-level retries are
allowed, and its learning runs' token usage must not exceed what the gateway saw.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import shutil
import types
from datetime import datetime, timedelta
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import audit, protocol, study
from evals.skill_bench.cell_records import (
    NOTE_RERUN_ACKNOWLEDGED,
    CellPaths,
    canonical_sha256,
    cell_gateway_runs,
    read_json,
    write_json_atomic,
)
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.envs.base import opaque_task_uid
from evals.skill_bench.infra import gateway
from evals.skill_bench.tests.e2e_support import (
    TEST_UPSTREAM,
    E2EModel,
    Matrix,
    alfworld_env,
    cell_spec,
    copy_cells,
    jsonl,
    run_cells,
    write_jsonl,
)
from evals.skill_bench.tests.conftest import REMOVED_METHOD
from evals.skill_bench.tests.fakes import FakeUpstream, serve_gateway
from evals.skill_bench.run import CellOutcome, CellRunner, CellSpec, RunOptions

ALF_NONE = cell_spec("alfworld", "none").cell_id
ALF_RB = cell_spec("alfworld", "reasoningbank").cell_id
ALF_MIRIX = cell_spec("alfworld", "mirix").cell_id
WS_NONE = cell_spec("webshop", "none").cell_id
WS_RB = cell_spec("webshop", "reasoningbank").cell_id
WS_AWM = cell_spec("webshop", "awm").cell_id


# --- a copy of some cells to tamper with ----------------------------------------------------


class Copy:
    def __init__(self, matrix: Matrix, root: Path, *cells: str) -> None:
        self.matrix = matrix
        self.runs = copy_cells(matrix, root, cells)

    def cell(self, cell_id: str) -> CellPaths:
        return CellPaths(self.runs / cell_id)

    def run_id(self, cell_id: str, number: int = 1) -> str:
        """The attempt's run id (``<cell>-aNN-<instance>``) as run.py recorded it."""
        return str(read_json(self.cell(cell_id).attempt(number).state)["run_id"])

    def ledger(self, cell_id: str) -> Path:
        return self.runs / self.run_id(cell_id) / "gateway.jsonl"

    def rows(self, cell_id: str, phase: str) -> list[dict[str, Any]]:
        attempt = self.cell(cell_id).attempt(1)
        return jsonl(attempt.learning if phase == "learning" else attempt.results)

    def trajectory(self, cell_id: str, phase: str, index: int = 0) -> Path:
        return self.cell(cell_id).root / self.rows(cell_id, phase)[index]["trajectory"]

    def longest(self, cell_id: str, phase: str) -> Path:
        rows = self.rows(cell_id, phase)
        best = max(rows, key=lambda r: r["actor_calls"])
        return self.cell(cell_id).root / best["trajectory"]

    def audit(self, **kw: Any) -> audit.AuditReport:
        kw.setdefault("expected_upstream", TEST_UPSTREAM)
        return audit.audit_runs(
            self.runs, env_factory=self.matrix.env_factory, split_mode=None, **kw
        )

    def edit(self, path: Path, change: Callable[[Any], Any]) -> None:
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def edit_ledger(
        self, cell_id: str, change: Callable[[list[dict[str, Any]]], Any]
    ) -> None:
        rows = jsonl(self.ledger(cell_id))
        change(rows)
        write_jsonl(self.ledger(cell_id), rows)

    def calls(self, cell_id: str, **match: Any) -> list[dict[str, Any]]:
        return [
            r
            for r in jsonl(self.ledger(cell_id))
            if r.get("event") == "call" and all(r.get(k) == v for k, v in match.items())
        ]

    def recorded_request(self, cell_id: str, call_id: str) -> dict[str, Any]:
        """The actor request a trajectory recorded for gateway call ``call_id``."""
        for phase in ("learning", "test"):
            for row in self.rows(cell_id, phase):
                data = json.loads(
                    (self.cell(cell_id).root / row["trajectory"]).read_text()
                )
                for call in data.get("calls") or []:
                    if call.get("gateway_call_id") == call_id:
                        return call["request"]
        raise KeyError(call_id)

    def add_rerun(self, cell_id: str, task_uid: str) -> None:
        state_path = self.cell(cell_id).attempt(1).state
        state = read_json(state_path)
        state["test_reruns"] = [
            *state.get("test_reruns", []),
            {"raw_id": "?", "task_uid": task_uid},
        ]
        write_json_atomic(state_path, state)


@pytest.fixture
def copy(e2e_matrix: Matrix, tmp_path: Path) -> Callable[..., Copy]:
    return lambda *cells: Copy(e2e_matrix, tmp_path, *cells)


def violations(result: audit.AuditReport) -> set[str]:
    """``invariant:check`` of every violation, in the cells and across them."""
    found = {
        f"{f.invariant}:{f.check}"
        for cell in result.cells
        for f in cell.findings
        if f.severity == audit.VIOLATION
    }
    found |= {
        f"{f.invariant}:{f.check}"
        for f in result.findings
        if f.severity == audit.VIOLATION
    }
    return found


def warnings(result: audit.AuditReport) -> set[str]:
    return {
        f"{f.invariant}:{f.check}"
        for cell in result.cells
        for f in cell.findings
        if f.severity == audit.WARNING
    }


def across_warnings(result: audit.AuditReport) -> set[str]:
    """``invariant:check`` of every warning across cells (run level)."""
    return {
        f"{f.invariant}:{f.check}"
        for f in result.findings
        if f.severity == audit.WARNING
    }


def recorded(result: audit.AuditReport, key: str) -> list[audit.Finding]:
    """Every finding of ``key`` (``invariant:check``), in the cells and across them."""
    return [
        f
        for f in [*result.findings, *(f for x in result.cells for f in x.findings)]
        if f"{f.invariant}:{f.check}" == key
    ]


#: The warning of cells built from a dirty checkout (the fixture cells record the git
#: state of the tree the tests run on): subtract it where a test counts every warning.
DIRTY = {"H10.6:dirty_tree"}


# --- the untouched records --------------------------------------------------------------------


def test_untouched_copies_pass_with_the_accepted_signals(copy):
    c = copy(ALF_NONE, ALF_RB, WS_AWM)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert result.exit_code == 0
    # H1.7: the ALFWorld trajectories reveal won by their length (1 step won, 50 lost),
    # and the audit does not count that as a leak.
    steps = {r["steps"] for r in c.rows(ALF_RB, "learning") + c.rows(ALF_RB, "test")}
    assert 1 in steps and protocol.ALFWORLD_MAX_STEPS in steps


# --- H10.1: labels ----------------------------------------------------------------------------


def test_a_label_key_in_a_learning_message(copy):
    c = copy(ALF_RB)

    def leak(data: dict[str, Any]) -> None:
        data["messages"][2]["won"] = True  # an observation the method received

    c.edit(c.trajectory(ALF_RB, "learning"), leak)
    assert violations(c.audit()) == {"H10.1:label_key", "H10.1:message_shape"}


def test_a_provider_extra_in_an_assistant_message(copy):
    c = copy(ALF_RB)

    def leak(data: dict[str, Any]) -> None:
        data["messages"][1]["reasoning"] = "the provider's hidden reasoning"

    c.edit(c.trajectory(ALF_RB, "learning"), leak)
    assert violations(c.audit()) == {"H10.1:message_shape"}


def test_a_game_file_in_the_first_message(copy, e2e_matrix):
    c = copy(ALF_RB)
    path = c.trajectory(ALF_RB, "learning")
    raw_id = json.loads(path.read_text())["raw_id"]
    tasks = {t.raw_id: t for t in e2e_matrix.envs["alfworld"].load_tasks("train")}
    gamefile = str(tasks[raw_id].raw["gamefile"])

    def leak(data: dict[str, Any]) -> None:
        for message in (data["first_messages"][0], data["messages"][0]):
            message["content"] += f"\n\nThe game is {gamefile}."

    c.edit(path, leak)
    result = c.audit()
    assert violations(result) == {"H10.1:label_string"}
    details = [f.detail for cell in result.cells for f in cell.findings]
    assert any(gamefile.lower() in d for d in details)


def test_a_raw_id_in_an_observation_and_an_altered_instruction(copy):
    c = copy(ALF_RB)
    path = c.trajectory(ALF_RB, "learning")
    raw_id = json.loads(path.read_text())["raw_id"]

    def leak(data: dict[str, Any]) -> None:
        data["messages"][2]["content"] += f"\n(item {raw_id})"
        data["instruction"] += " quickly"

    c.edit(path, leak)
    result = c.audit()
    # The instruction check is recorded; the leak still fails.
    assert violations(result) == {"H10.1:label_string"}
    assert "H10.1:instruction" in warnings(result)
    (instruction,) = recorded(result, "H10.1:instruction")
    assert instruction.downgraded == audit.RECORD


# --- WebShop ---------------------------------------------------------------------------------


def _ws_learning(c: Any, index: int = 0) -> tuple[Path, dict[str, Any]]:
    path = c.trajectory(WS_RB, "learning", index)
    return path, json.loads(path.read_text())


def test_webshop_learning_messages_end_with_the_score_line_and_pass(copy):
    """H1.7: a learning episode that bought ends
    with the score line, which methods see; the audit passes it."""
    c = copy(WS_RB)
    _path, data = _ws_learning(c)
    first = data["first_messages"][0]["content"]
    assert data["instruction"] in first and first.startswith(
        "WebShop [SEP] Instruction:"
    )
    assert all(data["instruction"] not in m["content"] for m in data["messages"][1:])
    last = data["messages"][-1]["content"]
    assert ws.final_message_pattern(ws.WebShopTemplates.load()).fullmatch(last)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()


def test_webshop_done_page_text_is_a_structural_leak(copy):
    c = copy(WS_RB)
    path, _ = _ws_learning(c)
    c.edit(
        path,
        lambda d: d["messages"][2].__setitem__(
            "content", d["messages"][2]["content"] + " [SEP] instuction text [SEP] x"
        ),
    )
    result = c.audit()
    assert violations(result) == {"H10.1:structural_leak"}
    (found,) = [
        f for x in result.cells for f in x.findings if f.check == "structural_leak"
    ]
    assert "instuction text" in found.detail and found.cell_id == WS_RB


def test_webshop_score_line_before_the_end_is_a_structural_leak(copy):
    c = copy(WS_RB)
    path, _ = _ws_learning(c)

    def early(d: dict[str, Any]) -> None:
        d["messages"].insert(2, dict(d["messages"][-1]))

    c.edit(path, early)
    result = c.audit()
    assert "H10.1:structural_leak" in violations(result)
    details = [
        f.detail
        for x in result.cells
        for f in x.findings
        if f.check == "structural_leak"
    ]
    assert any("not the last message" in d for d in details)


# --- the agent context -----------------------------------------------------------------------


def _every_trajectory(c: Copy, cell_id: str) -> list[Path]:
    return [
        c.trajectory(cell_id, phase, index)
        for phase in ("learning", "test")
        for index in range(len(c.rows(cell_id, phase)))
    ]


@pytest.mark.parametrize("cell_id", [ALF_RB, WS_RB])
def test_the_agent_context_channel_is_audited_and_passes(copy, cell_id):
    """Every item of an ALFWorld or WebShop cell received its environment's static
    prompt template (one text, this tree's). The label searches cover it and find
    nothing."""
    c = copy(cell_id)
    paths = _every_trajectory(c, cell_id)
    texts = {json.loads(p.read_text())["agent_context"] for p in paths}
    env = c.matrix.env_factory(cell_id.split("-")[0])
    assert texts == {audit.env_agent_context(env)}
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert not warnings(result) & {
        "H10.1:agent_context_varies",
        "H10.1:agent_context_not_this_tree",
    }
    (cell,) = result.cells
    info = cell.info["agent_context"]
    assert (info["texts"], info["items"], info["this_tree"]) == (1, len(paths), True)
    assert "{current_observation}" in next(iter(texts))
    assert info["chars"] == [len(next(iter(texts)))]


def test_a_label_in_the_agent_context_is_a_label_string_and_a_varying_channel(copy):
    """The agent context is searched like the rest of what a method received: the item's raw
    id in its agent context is found, and a text only some items got varies."""
    c = copy(ALF_RB)
    path = c.trajectory(ALF_RB, "test", 0)
    raw_id = c.rows(ALF_RB, "test")[0]["raw_id"]
    c.edit(
        path,
        lambda d: d.__setitem__("agent_context", d["agent_context"] + f"\n{raw_id}"),
    )
    result = c.audit()
    assert violations(result) == {
        "H10.1:label_string",
        "H10.1:agent_context_varies",
    }
    (label,) = recorded(result, "H10.1:label_string")
    assert ".agent_context" in label.detail
    (varies,) = recorded(result, "H10.1:agent_context_varies")
    assert "2 different agent contexts" in varies.detail


def test_an_item_without_the_agent_context_varies_the_channel(copy):
    c = copy(WS_RB)
    c.edit(c.trajectory(WS_RB, "learning", 1), lambda d: d.pop("agent_context"))
    result = c.audit()
    assert violations(result) == {"H10.1:agent_context_varies"}
    (varies,) = recorded(result, "H10.1:agent_context_varies")
    assert "1 item(s) none" in varies.detail


def test_records_without_an_agent_context_are_not_checked(copy):
    """Older trajectories have no ``agent_context``: nothing is checked, the cell stays
    auditable as it ran."""
    c = copy(ALF_RB)
    for path in _every_trajectory(c, ALF_RB):
        c.edit(path, lambda d: d.pop("agent_context"))
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    (cell,) = result.cells
    assert cell.info["agent_context"] is None


def test_an_agent_context_this_tree_does_not_render_is_a_warning(copy):
    """One text for every item, but not this tree's (a template or code change since the
    cell ran): a warning naming it, never a violation."""
    c = copy(ALF_RB)
    for path in _every_trajectory(c, ALF_RB):
        c.edit(
            path,
            lambda d: d.__setitem__("agent_context", d["agent_context"] + " (v0)"),
        )
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    (stale,) = recorded(result, "H10.1:agent_context_not_this_tree")
    assert stale.severity == audit.WARNING
    (cell,) = result.cells
    assert cell.info["agent_context"]["this_tree"] is False


def test_a_varying_agent_context_across_harness_code_versions_is_a_warning(copy):
    """The harness code renders the text: an attempt that ran several harness
    code versions explains two texts, so it is a warning there."""
    c = copy(ALF_RB)
    c.edit(
        c.trajectory(ALF_RB, "test", 1),
        lambda d: d.__setitem__("agent_context", d["agent_context"] + " (v2)"),
    )
    original = audit.CellAuditor._check_code

    def two_versions(self: Any, code: Any) -> None:
        original(self, code)
        self.harness_code_versions = {"a" * 64, "b" * 64}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(audit.CellAuditor, "_check_code", two_versions)
        result = c.audit()
    assert "H10.1:agent_context_varies" not in violations(result)
    (varies,) = recorded(result, "H10.1:agent_context_varies")
    assert varies.severity == audit.WARNING
    assert "2 harness code versions" in varies.detail


class _FailingReset:
    """An episode whose reset fails on infrastructure (the backend unreachable)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def reset(self) -> Any:
        from evals.skill_bench.tests.run_support import InfraFailure

        raise InfraFailure("the environment's backend is unreachable")

    async def close(self) -> None:
        await self._inner.close()


def _flaky_alfworld(root: Path, split: str) -> Any:
    """The e2e ALFWorld whose first reset of an item of ``split`` fails once (INFRA)."""
    env = alfworld_env(root)
    inner = env.episode
    failed: list[str] = []

    def episode(task: Any) -> Any:
        made = inner(task)
        if not failed and task.split == split:
            failed.append(task.raw_id)
            return _FailingReset(made)
        return made

    env.episode = episode  # type: ignore[method-assign]
    return env


def _run_until_complete(root: Path, env: Any, method: str, tries: int) -> Path:
    """One ALFWorld cell through run.py, launched again until it completes (an INFRA
    failure stops a phase; the next launch re-runs or resumes it)."""
    spec = cell_spec("alfworld", method)
    runs = root / "runs"
    fake = FakeUpstream(responder=E2EModel())
    outcomes = []
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        for _ in range(tries):
            outcome = asyncio.run(run_cells(runs, url, {"alfworld": env}, [spec]))
            outcomes.append(outcome[spec.cell_id])
            if outcomes[-1].status == "complete":
                break
    assert outcomes[-1].status == "complete", [(o.status, o.detail) for o in outcomes]
    assert len(outcomes) >= 2  # the failure stopped the first launch
    return runs


def _trajectory_files(runs: Path) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted(runs.rglob("*.json")):
        if "trajectories" not in path.parts:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "agent_context" in data:
            found.append((path, data))
    return found


def _context_audit(runs: Path, env: Any) -> audit.AuditReport:
    return audit.audit_runs(
        runs,
        env_factory=lambda _b: env,
        split_mode=None,
        expected_upstream=TEST_UPSTREAM,
    )


def _context_findings(result: audit.AuditReport) -> list[audit.Finding]:
    return [
        f
        for cell in result.cells
        for f in cell.findings
        if f.check in ("agent_context_varies", "agent_context_not_this_tree")
    ]


@pytest.mark.parametrize(
    ("method", "split", "tries"),
    [("none", "val", 2), ("reasoningbank", "train", 3)],
    ids=["test-phase-rerun", "learning-discarded-try"],
)
def test_a_reset_failure_does_not_vary_the_agent_context(
    tmp_path, method, split, tries
):
    """An item whose reset failed on infrastructure (a test item re-run, or a learning
    try a resume discarded) was handed no task view. The rollout sets the static text
    before the episode starts, and the audit leaves out a trajectory whose reset failed,
    so an older record that carries an empty text for it passes too; any other item
    without the text still varies the channel."""
    env = _flaky_alfworld(tmp_path, split)
    runs = _run_until_complete(tmp_path, env, method, tries)
    text = audit.env_agent_context(env)
    records = _trajectory_files(runs)
    failed = [p for p, d in records if d.get("failure_stage") == audit.RESET_STAGE]
    assert len(failed) == 1
    assert ("discarded" in failed[0].parts) == (split == "train")
    assert {d["agent_context"] for _p, d in records} == {text}

    result = _context_audit(runs, env)
    assert violations(result) == set(), result.to_markdown()
    assert _context_findings(result) == []
    (cell,) = result.cells
    info = cell.info["agent_context"]
    assert info["texts"] == 1 and info["items"] == len(records) - 1

    # An older rollout's record: the reset failure carried "".
    data = json.loads(failed[0].read_text())
    data["agent_context"] = ""
    failed[0].write_text(json.dumps(data))
    result = _context_audit(runs, env)
    assert violations(result) == set(), result.to_markdown()
    assert _context_findings(result) == []
    (cell,) = result.cells
    assert cell.info["agent_context"] == info

    # Any other item without the text still varies the channel.
    other = next(p for p, d in records if d.get("failure_stage") is None)
    data = json.loads(other.read_text())
    data["agent_context"] = ""
    other.write_text(json.dumps(data))
    assert "H10.1:agent_context_varies" in violations(_context_audit(runs, env))


def test_the_audit_reset_stage_is_the_rollouts():
    from evals.skill_bench.agent import rollout

    assert audit.RESET_STAGE == rollout.STAGE_RESET


def test_webshop_altered_instruction_is_recorded(copy):
    c = copy(WS_RB)
    path, _ = _ws_learning(c)
    c.edit(path, lambda d: d.__setitem__("instruction", d["instruction"] + " quickly"))
    result = c.audit()
    assert violations(result) == set()
    (instruction,) = recorded(result, "H10.1:instruction")
    assert instruction.downgraded == audit.RECORD


def test_webshop_memory_is_stripped_from_the_start_of_the_user_prompt():
    """A WebShop request is [user, (assistant, user)*], the memory in the slot of its
    first message only."""
    t = ws.WebShopTemplates.load()
    task = "i need red shoes"
    first = ws.first_prompt(t, task, "'Search'", ("search[<your query>]",))
    turn = ws.turn_prompt(t, task, 2, "'Back to Search'", ("click[back to search]",))
    reply = "<think>a</think><action>search[red shoes]</action>"
    memory = "## learned\nsearch the product name first"
    for conversation in ((first,), (first, reply, turn)):
        fields = {
            "conversation": tuple(
                {"role": "assistant" if i % 2 else "user", "content": text}
                for i, text in enumerate(conversation)
            ),
            "step_count": len(conversation) // 2,
        }
        with_memory = ws.render_actor_messages(t, fields, memory)
        stripped, why = audit.strip_memory("webshop", with_memory, memory)
        assert why is None and stripped == ws.render_actor_messages(t, fields, "")
        bare = ws.render_actor_messages(t, fields, "")
        assert "not at the start of the user prompt" in str(
            audit.strip_memory("webshop", bare, memory)[1]
        )
        assert audit.strip_memory("webshop", bare, "")[0] == bare
        system_first = [{"role": "system", "content": "x"}, *with_memory]
        assert "[user, (assistant, user)*]" in str(
            audit.strip_memory("webshop", system_first, memory)[1]
        )
    # The slot heads the first message only; two users in a row are no conversation.
    again = [*with_memory[:2], {"role": "user", "content": with_memory[0]["content"]}]
    assert "later user message carries the memory slot" in str(
        audit.strip_memory("webshop", again, memory)[1]
    )
    users = [with_memory[0], {"role": "user", "content": turn}]
    assert "[user, (assistant, user)*]" in str(
        audit.strip_memory("webshop", users, memory)[1]
    )
    ending_with_a_reply = with_memory[:2]
    assert audit.strip_memory("webshop", ending_with_a_reply, memory)[0] is None


def _change_call_request(
    c: Copy, path: Path, index: int, change: Callable[[dict[str, Any]], None]
) -> None:
    """Change actor call ``index``'s recorded request and make the ledger's
    ``request_sha256`` agree, as if the actor had really sent it."""
    sent: dict[str, dict[str, Any]] = {}

    def edit(data: dict[str, Any]) -> None:
        call = data["calls"][index]
        change(call["request"])
        sent[str(call["gateway_call_id"])] = call["request"]

    c.edit(path, edit)

    def resend(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            request = sent.get(str(row.get("call_id")))
            if row.get("event") == "call" and request is not None:
                row["request_sha256"] = audit.sent_request_sha256(request, row["wire"])

    c.edit_ledger(path.relative_to(c.runs).parts[0], resend)


def test_webshop_calls_are_one_append_only_conversation(copy):
    """Each WebShop actor call is the previous one plus one reply
    and one user message, the memory in the first message only; the audit passes that,
    and fails an earlier message changed, even when the ledger agrees."""
    c = copy(WS_RB)
    path = c.longest(WS_RB, "test")
    data = json.loads(path.read_text())
    memory = data["injection"]["text"]
    requests = [call["request"]["messages"] for call in data["calls"]]
    assert memory and len(requests) > 3
    for index, messages in enumerate(requests):
        assert len(messages) == 2 * index + 1
        assert messages[0]["content"].startswith(memory)
        assert all(memory not in m["content"] for m in messages[1:])
        if index:
            assert messages[: len(requests[index - 1])] == requests[index - 1]
    assert violations(c.audit()) == set()

    def rerendered(request: dict[str, Any]) -> None:
        request["messages"][1]["content"] = "a reply the model never gave"

    _change_call_request(c, path, 2, rerendered)
    result = c.audit()
    assert violations(result) == {"H10.2:conversation_not_append_only"}
    found = [
        f.detail
        for x in result.cells
        for f in x.findings
        if f.check == "conversation_not_append_only"
    ]
    assert len(found) == 2 and "actor call 2's messages are not call 1's" in found[0]


def test_a_webshop_call_that_drops_an_exchange_is_found(copy):
    """A window (the oldest exchange left out) is no longer the conversation."""
    c = copy(WS_RB)
    _change_call_request(
        c,
        c.longest(WS_RB, "test"),
        3,
        lambda r: r.__setitem__("messages", [r["messages"][0], *r["messages"][3:]]),
    )
    assert violations(c.audit()) == {"H10.2:conversation_not_append_only"}


def test_a_webshop_first_call_with_history_is_found(copy):
    c = copy(WS_RB)
    path = c.longest(WS_RB, "test")

    def with_history(d: dict[str, Any]) -> None:
        messages = d["calls"][0]["request"]["messages"]
        messages += d["calls"][1]["request"]["messages"][1:]
        d["first_prompt_sha256"] = audit.request_fingerprint(d["calls"][0]["request"])

    c.edit(path, with_history)
    _change_call_request(c, path, 0, lambda r: None)  # the ledger agrees
    result = c.audit()
    assert "H10.2:conversation_not_append_only" in violations(result)
    assert any(
        "actor call 0 sent 3 messages" in f.detail
        for x in result.cells
        for f in x.findings
    )


def test_a_webshop_memory_slot_in_a_later_message_is_found(copy):
    c = copy(WS_RB)
    path = c.longest(WS_RB, "test")
    memory = json.loads(path.read_text())["injection"]["text"]
    slot = ws.WebShopTemplates.load().memory_slot.format(memory=memory) + "\n"

    def again(request: dict[str, Any]) -> None:
        last = request["messages"][-1]
        last["content"] = slot + last["content"]

    _change_call_request(c, path, -1, again)
    # The appended message is then no longer rollout_turn.md either.
    assert violations(c.audit()) == {
        "H10.2:memory_slot",
        "H10.2:conversation_not_append_only",
    }


def _append_only_details(result: audit.AuditReport) -> list[str]:
    return [
        f.detail
        for x in result.cells
        for f in x.findings
        if f.check == "conversation_not_append_only"
    ]


def test_a_webshop_reply_replayed_as_other_text_is_found(copy):
    """The replayed assistant message is the reply the previous call got, as its recorded
    response holds it: a conversation that kept another text (here, from call 2 on,
    consistently, the ledger agreeing) is not the episode."""
    c = copy(WS_RB)
    path = c.longest(WS_RB, "test")
    calls = len(json.loads(path.read_text())["calls"])
    for index in range(2, calls):
        _change_call_request(
            c,
            path,
            index,
            lambda r: r["messages"][3].__setitem__(
                "content", r["messages"][3]["content"] + " (edited)"
            ),
        )
    result = c.audit()
    assert violations(result) == {"H10.2:conversation_not_append_only"}
    (detail,) = _append_only_details(result)
    assert "actor call 2 appended an exchange" in detail
    assert "assistant message is not the reply call 1 got" in detail


@pytest.mark.parametrize("change", ["step", "task"])
def test_a_webshop_turn_of_another_step_or_task_is_found(copy, change):
    """The appended user message is rollout_turn.md for the next step, of the task the
    first message states (the last call changed, the ledger agreeing)."""
    c = copy(WS_RB)
    path = c.longest(WS_RB, "test")
    data = json.loads(path.read_text())
    step, task = len(data["calls"]), data["instruction"]
    first = data["calls"][0]["request"]["messages"][0]["content"]
    t = ws.WebShopTemplates.load()
    turn = ws.turn_prompt(t, task, step, "'Back to Search'", ("click[back to search]",))
    assert audit._webshop_turn_problem(turn, first, step) is None
    assert audit._webshop_turn_problem(turn, first, step - 1) is not None
    assert audit._webshop_turn_problem(turn + "x", first, step) is not None
    if change == "step":
        old, new, said = (
            f"You are now at step {step} ",
            f"You are now at step {step - 1} ",
            f"not rollout_turn.md for step {step}",
        )
    else:
        old, new, said = (
            task,
            task + " quickly",
            "states another task than the first message",
        )

    def edit(request: dict[str, Any]) -> None:
        last = request["messages"][-1]
        assert old in last["content"]
        last["content"] = last["content"].replace(old, new, 1)

    _change_call_request(c, path, -1, edit)
    result = c.audit()
    assert violations(result) == {"H10.2:conversation_not_append_only"}
    (detail,) = _append_only_details(result)
    assert said in detail


def test_the_audit_rebuilds_the_replayed_reply_as_the_env_keeps_it(tmp_path):
    """The audit's reading of a recorded response gives the assistant message the env
    appends for the same reply (``ws.reply_turn``): the reply's text (text parts joined),
    U+FFFD for a lone surrogate, and the placeholder for an empty reply or none."""
    t = ws.WebShopTemplates.load()

    def call(content: Any, status: int | None = 200) -> dict[str, Any]:
        message = {"role": "assistant", "content": content}
        return {"http_status": status, "response": {"choices": [{"message": message}]}}

    for content, action in (
        ("<think>a</think><action>click[x]</action>", None),
        (
            [
                {"type": "text", "text": "<action>"},
                {"type": "text", "text": "x</action>"},
            ],
            "<action>x</action>",
        ),
        ("bad \ud83d escape", None),
        ("", None),
        (" \n\t", None),
        (None, ""),
    ):
        expected = ws.reply_turn(t, content if action is None else action)
        assert audit._webshop_reply_turn(call(content), tmp_path) == expected
    assert (
        audit._webshop_reply_turn(call("bad \ud83d escape"), tmp_path) == "bad � escape"
    )
    no_reply = t.no_reply_turn
    for failed in (
        call("<action>click[x]</action>", 500),
        call("<action>click[x]</action>", None),
        {"http_status": 200, "response": {"choices": []}},
        {"http_status": None, "response": None},
    ):
        assert audit._webshop_reply_turn(failed, tmp_path) == no_reply


def test_uids_that_are_not_the_salted_hmac(copy):
    c = copy(ALF_RB)
    c.edit(
        c.trajectory(ALF_RB, "test", 0),
        lambda d: d.__setitem__("task_uid", "t" + "0" * 20),
    )
    c.edit(
        c.trajectory(ALF_RB, "test", 1),
        lambda d: d.__setitem__("task_uid", d["raw_id"]),
    )
    assert violations(c.audit()) == {
        "H10.1:uid_salt",
        "H10.1:uid_format",
        # The raw id the method received as its uid is the item's own label.
        "H10.1:label_string",
        # The ledger's actor lines still carry the real uids.
        "H10.3:actor_call_mismatch",
    }


def colliding_salt(raw_id: str, limit: int = 5_000_000) -> bytes:
    """A salt whose uid for ``raw_id`` holds ``raw_id`` as a substring: about
    one salt in 16**n / (21 - n) for an n-digit id, 230 for 3 digits, 65,536 for 5."""
    for i in range(limit):
        salt = hashlib.sha256(f"collide:{raw_id}:{i}".encode()).digest()
        if raw_id in opaque_task_uid(salt, raw_id):
            return salt
    raise AssertionError(f"no colliding salt for {raw_id!r} in {limit} tries")


def test_uid_opacity_is_the_hmac_not_a_substring_test():
    """A short numeric raw id inside its 20-hex uid (0.44% of the uids of a 3-digit id)
    is no leak; a uid that is not the salted HMAC is."""
    salt = colliding_salt("377")
    uid = opaque_task_uid(salt, "377")
    assert "377" in uid and audit.UID_RE.fullmatch(uid)
    assert audit.uid_problem(uid, "377", salt) is None
    assert audit.uid_problem(uid, "377", None) is None  # salt unreadable: shape only
    other = opaque_task_uid(b"another cell's salt", "377")
    assert audit.uid_problem(other, "377", salt) == (
        "uid_salt",
        f"task uid {other} is not HMAC(cell salt, raw id)",
    )
    for bad in (uid.upper(), uid[:-1], uid + "0", "u" + uid[1:], uid + "\n", ""):
        problem = audit.uid_problem(bad, "377", salt)
        assert problem is not None and problem[0] == "uid_format", bad
    # A raw id shaped like a uid, used as its own uid: never opaque, with or without salt.
    shaped = "t" + "0123456789abcdef0123"
    assert audit.uid_problem(shaped, shaped, None) == (
        "uid_format",
        f"task uid {shaped!r} is the raw id",
    )


def test_another_items_game_file_in_an_observation(copy, e2e_matrix):
    c = copy(ALF_RB)
    other = e2e_matrix.envs["alfworld"].load_tasks("val")[
        -1
    ]  # not one of the cell's items

    def leak(data: dict[str, Any]) -> None:
        data["messages"][2]["content"] += f"\n{other.raw['gamefile']}"

    c.edit(c.trajectory(ALF_RB, "learning"), leak)
    result = c.audit()
    assert violations(result) == {"H10.1:cross_item_label"}
    assert any(other.raw_id in f.detail for cell in result.cells for f in cell.findings)


def test_a_method_output_that_holds_a_scorer_identifier(copy, e2e_matrix):
    """A method can only hold a trial directory name if it received it."""
    c = copy(ALF_RB)
    trial = str(
        e2e_matrix.envs["alfworld"].load_tasks("test")[5].raw["gamefile"]
    ).split("/")[-2]
    injections = c.cell(ALF_RB).attempt(1).injections
    rows = jsonl(injections)
    rows[-1]["text"] += f"\nRemember {trial}."
    write_jsonl(injections, rows)
    assert violations(c.audit()) == {"H10.1:method_output_label"}


# --- H10.2: the actor template -------------------------------------------------------------------


def _change_first_request(
    c: Copy, path: Path, change: Callable[[dict[str, Any]], None], *, rehash: bool
) -> None:
    """Change the first actor request a trajectory recorded. ``rehash`` also makes the
    records consistent, as if the actor had really sent it: the trajectory's first-prompt
    hash and the ledger's ``request_sha256`` of that call."""
    sent: dict[str, dict[str, Any]] = {}

    def edit(data: dict[str, Any]) -> None:
        request = data["calls"][0]["request"]
        change(request)
        if rehash:
            data["first_prompt_sha256"] = audit.request_fingerprint(request)
            sent[str(data["calls"][0]["gateway_call_id"])] = request

    c.edit(path, edit)
    cell_id = path.relative_to(c.runs).parts[0]

    def resend(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            request = sent.get(str(row.get("call_id")))
            if row.get("event") == "call" and request is not None:
                row["request_sha256"] = audit.sent_request_sha256(request, row["wire"])

    if sent:
        c.edit_ledger(cell_id, resend)


def test_a_first_prompt_that_differs_outside_the_memory_slot(copy):
    c = copy(ALF_RB)
    _change_first_request(
        c,
        c.trajectory(ALF_RB, "test"),
        lambda r: r["messages"][1].__setitem__(
            "content", r["messages"][1]["content"] + "\nHint: the answer is on page 3."
        ),
        rehash=True,
    )
    assert violations(c.audit()) == {"H10.2:outside_memory_slot"}


def test_a_memory_outside_its_slot(copy):
    c = copy(ALF_RB)

    def move(request: dict[str, Any]) -> None:
        request["messages"][0]["content"] += "\n(memory moved here)"

    _change_first_request(c, c.trajectory(ALF_RB, "test"), move, rehash=True)
    assert violations(c.audit()) == {"H10.2:memory_slot"}


def test_a_recorded_first_prompt_hash_that_is_not_the_request(copy):
    c = copy(ALF_RB)
    c.edit(
        c.trajectory(ALF_RB, "test"),
        lambda d: d.__setitem__("first_prompt_sha256", "0" * 64),
    )
    assert violations(c.audit()) == {"H10.2:first_prompt_hash"}


def test_an_extra_field_and_another_model_in_the_actor_request(copy):
    c = copy(ALF_RB)
    _change_first_request(
        c,
        c.trajectory(ALF_RB, "test"),
        lambda r: r.update(temperature=0.0, model="openai/other-model"),
        rehash=False,
    )
    result = c.audit()
    # Recorded: the gateway's field whitelist and the model checks of H10.3 guard what
    # was sent.
    assert violations(result) == set()
    assert {"H10.2:request_fields", "H10.2:request_model"} <= warnings(result)


def test_a_memory_that_changes_within_an_episode(copy):
    c = copy(ALF_RB)
    c.edit(
        c.longest(ALF_RB, "test"),
        lambda d: d["calls"][7].__setitem__("memory_sha256", "0" * 64),
    )
    assert violations(c.audit()) == {"H10.2:memory_changed"}


def test_one_item_with_another_no_memory_prompt_across_cells(copy):
    c = copy(ALF_NONE, ALF_RB)
    c.edit(
        c.trajectory(ALF_RB, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    result = c.audit()
    assert violations(result) == {
        "H10.2:outside_memory_slot",  # this cell's own record
        "H10.2:first_prompt_differs_across_cells",  # and against the no-memory cell
    }
    (across,) = result.findings
    assert ALF_NONE in across.detail and ALF_RB in across.detail


# --- H10.3: the gateway ------------------------------------------------------------------------------


def test_an_actor_call_missing_from_the_ledger(copy):
    c = copy(ALF_RB)
    call_id = json.loads(c.trajectory(ALF_RB, "test").read_text())["calls"][0][
        "gateway_call_id"
    ]
    c.edit_ledger(
        ALF_RB,
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r.get("call_id") != call_id]
        ),
    )
    assert violations(c.audit()) == {"H10.3:actor_call_not_in_ledger"}


def test_a_role_the_method_does_not_use_and_a_learning_call_in_the_test_phase(copy):
    c = copy(ALF_RB)
    learner = [r["call_id"] for r in c.calls(ALF_RB, role="learner")]

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("call_id") == learner[0] and row["event"] == "call":
                row["role"] = "mirix"  # ReasoningBank never calls MIRIX (H7.5)
            if row.get("call_id") == learner[1] and row["event"] == "call":
                # The frozen test phase never learns (H1.6): a learner call the gateway
                # recorded while the run was in its test phase.
                row["phase"] = row["run_phase"] = row["phase_header"] = "test"

    c.edit_ledger(ALF_RB, change)
    assert violations(c.audit()) == {
        "H10.3:role_not_allowed",
        "H10.4:learning_call_in_test",
    }


def test_the_run_phase_decides_not_the_callers_phase_header(copy):
    """A learner that tags its call 'learning' while run.py has put the run in its
    test phase is judged by the run's phase; the two disagreeing is itself a violation."""
    c = copy(ALF_RB)
    learner = c.calls(ALF_RB, role="learner")[0]["call_id"]

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("call_id") == learner and row["event"] == "call":
                assert row["run_phase"] == row["phase_header"] == "learning"
                row["run_phase"] = "test"

    c.edit_ledger(ALF_RB, change)
    result = c.audit()
    assert violations(result) == {
        "H10.3:role_not_allowed",  # RB's learner has no test-phase role
        "H10.4:learning_call_in_test",
    }
    # The tag that disagrees is recorded (H10.3-phase); the run's phase decides.
    assert "H10.3:phase_mismatch" in warnings(result)


def test_a_ledger_without_run_phases_uses_the_phase_timeline(copy):
    """For lines without ``run_phase`` the phase run.py set at the call's time (the
    ledger's phase events) decides, not the caller's tag."""
    c = copy(ALF_RB)
    learner = c.calls(ALF_RB, role="learner")[0]["call_id"]
    test_actor = c.calls(ALF_RB, role="actor", phase="test")[0]

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row.pop("run_phase", None)
            row.pop("phase_header", None)
            if row.get("call_id") == learner and row["event"] == "call":
                row["ts"] = test_actor["ts"]  # a learner call in the test phase

    c.edit_ledger(ALF_RB, change)
    result = c.audit()
    found = violations(result)
    assert "H10.4:learning_call_in_test" in found
    assert "H10.3:phase_mismatch" in warnings(result)
    assert "H10.4:learning_call_after_freeze" in found


def test_a_learning_call_made_for_a_test_item(copy):
    """A learner call's item header names a test item."""
    c = copy(ALF_RB)
    learner = c.calls(ALF_RB, role="learner")[0]["call_id"]
    test_uid = c.rows(ALF_RB, "test")[0]["task_uid"]

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("call_id") == learner and row["event"] == "call":
                row["item"] = test_uid

    c.edit_ledger(ALF_RB, change)
    assert violations(c.audit()) == {"H10.4:learning_call_for_test_item"}


def test_repeat_of_is_judged_by_role(copy):
    c = copy(ALF_RB)
    learner = c.calls(ALF_RB, role="learner")
    actors = c.calls(ALF_RB, role="actor", phase="test")
    long_item = json.loads(c.longest(ALF_RB, "test").read_text())["task_uid"]
    same_item = [r for r in actors if r["item"] == long_item]
    other_item = next(r for r in actors if r["item"] != long_item)
    embeddings = c.calls(ALF_RB, role="embedding", phase="test")
    embedded = next(r for r in embeddings if r["item"] != long_item)

    def flag(marks: dict[str, str]) -> None:
        def change(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                if row.get("event") == "call" and row.get("call_id") in marks:
                    row["repeat_of"] = marks[row["call_id"]]
                    row["repeat_after_status"] = "upstream_error"

        c.edit_ledger(ALF_RB, change)

    # A learner never retries; an actor repeat on an item that was not re-run neither; an
    # RB embedding repeat in the test phase neither (unless H8 re-ran the item).
    flag({learner[1]["call_id"]: "gone-call"})
    flag({same_item[3]["call_id"]: same_item[2]["call_id"]})
    flag({embedded["call_id"]: "gone-call"})
    result = c.audit()
    problems = [
        f.detail
        for cell in result.cells
        for f in cell.findings
        if f.check == "repeat_of"
    ]
    # Recorded: each repeat is a warning, counted by role; nothing of it fails the
    # cell.
    assert len(problems) == 3 and violations(result) == set()
    assert "H10.3:repeat_of" in warnings(result)
    (cell,) = result.cells
    assert cell.info["repeat_of_flagged"] == {"learner": 1, "actor": 1, "embedding": 1}
    assert cell.info["recorded_counts"]["H10.3:repeat_of"] == 3
    # H8 re-ran those test items: the actor and the embedding repeats are allowed.
    c.add_rerun(ALF_RB, long_item)
    c.add_rerun(ALF_RB, embedded["item"])
    result = c.audit()
    assert [
        f.detail
        for cell in result.cells
        for f in cell.findings
        if f.check == "repeat_of"
    ] == [p for p in problems if "learner" in p]
    # The same request sent for another item is not a retry at all.
    flag({learner[1]["call_id"]: learner[0]["call_id"]})
    flag({other_item["call_id"]: same_item[0]["call_id"]})
    c.edit(
        c.cell(ALF_RB).attempt(1).state,
        lambda s: s.__setitem__("test_reruns", []),
    )
    assert learner[0]["item"] != learner[1]["item"]
    # Still same_item[3]'s and the embedding's.
    result = c.audit()
    assert violations(result) == set() and "H10.3:repeat_of" in warnings(result)
    assert result.cells[0].info["repeat_of_flagged"] == {"actor": 1, "embedding": 1}
    c.add_rerun(ALF_RB, long_item)
    c.add_rerun(ALF_RB, embedded["item"])
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:repeat_of" not in warnings(result)
    assert result.cells[0].info["repeat_of"] == {
        "learner": 1,
        "actor": 2,
        "embedding": 1,
    }
    assert result.cells[0].info["repeat_of_flagged"] == {}


def test_a_call_with_another_model_or_model_family(copy):
    c = copy(ALF_RB)
    actors = c.calls(ALF_RB, role="actor", phase="test")

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("call_id") == actors[0]["call_id"] and row["event"] == "call":
                row["model_served"] = "anthropic/claude-other-9"
            if row.get("call_id") == actors[1]["call_id"] and row["event"] == "call":
                row["model"] = protocol.MODELS["luna56"]

    c.edit_ledger(ALF_RB, change)
    assert violations(c.audit()) == {"H10.3:model", "H10.3:model_served"}


def test_an_actor_call_no_trajectory_accounts_for(copy):
    c = copy(ALF_RB)
    extra = dict(c.calls(ALF_RB, role="actor", phase="test")[0], call_id="orphan-1")
    c.edit_ledger(ALF_RB, lambda rows: rows.append(extra))
    assert violations(c.audit()) == {"H10.3:unaccounted_actor_call"}
    c.add_rerun(ALF_RB, extra["item"])  # a call of an item H8 re-ran in the test phase
    assert violations(c.audit()) == set()


def test_requests_the_gateway_refused_for_breaking_the_protocol(copy):
    c = copy(ALF_RB)
    base = c.calls(ALF_RB, role="actor", phase="test")[0]
    rejected = [
        {
            **{k: base[k] for k in ("run_id", "item", "gateway_settings_sha256")},
            "event": "call",
            "call_id": f"rejected-{kind}",
            "role": "actor",
            "phase": "test",
            "status": "rejected",
            "protocol_conformant": True,
            "error": {"type": kind},
        }
        for kind in ("unauthorized", "context_budget_exceeded")
    ]
    c.edit_ledger(ALF_RB, lambda rows: rows.extend(rejected))
    result = c.audit()
    # Recorded (H10.3-protocol-rejection): the gateway refused it, nothing was sent.
    assert violations(result) == set()
    assert "H10.3:protocol_rejection" in warnings(result)
    assert result.cells[0].info["rejected"] == {
        "unauthorized": 1,
        "context_budget_exceeded": 1,
    }


def test_a_gateway_that_forwarded_elsewhere(copy):
    c = copy(ALF_RB)
    assert violations(c.audit(expected_upstream=audit.DEFAULT_UPSTREAM)) == {
        "H10.3:upstream"
    }


# --- MIRIX on the stub MIRIX -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mirix_matrix(tmp_path_factory: pytest.TempPathFactory) -> Matrix:
    """One ALFWorld MIRIX cell through run.py, on the stub MIRIX of the method's tests
    (its learning runs call the real gateway like MIRIX's distiller)."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import (
        StubMirix,
        StubMirixServer,
    )

    root = tmp_path_factory.mktemp("mirix")
    env = alfworld_env(root)
    stubs: dict[tuple[str, int], StubMirix] = {}

    def server(spec: Any, mode: str) -> StubMirixServer:
        key = (spec.cell_id, spec.attempt)
        if mode == "fresh":
            stubs[key] = StubMirix(threshold=2)
        return StubMirixServer(stubs[key], mode)

    model = E2EModel()
    fake = FakeUpstream(responder=model)
    runs = root / "runs"
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        outcomes = asyncio.run(
            run_cells(
                runs,
                url,
                {"alfworld": env},
                [cell_spec("alfworld", "mirix")],
                method_factories={"mirix": lambda: MirixMethod(poll_s=0.01)},
                mirix_server_factory=server,
            )
        )
    assert {o.status for o in outcomes.values()} == {"complete"}, outcomes
    return Matrix(root, runs, {"alfworld": env}, outcomes, model, fake)


def test_a_mirix_cell_on_the_stub_reconciles_its_usage(mirix_matrix, tmp_path):
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    usage = result.cells[0].info["mirix_usage"]
    assert usage["runs"] == usage["runs_with_usage"] == 2  # threshold 2, then the flush
    assert usage["learning_runs"]["calls"] == 2
    # Each run's calls fall in its own window; MIRIX's retrieval calls (one topic
    # extraction per learning item) fall outside every run.
    assert usage["ledger_in_runs"] == usage["learning_runs"]
    assert usage["calls_outside_runs"] == usage["learning_retrievals"] == 3
    # MIRIX's application-level retries are allowed and counted.
    mirix_calls = c.calls(ALF_MIRIX, role="mirix")
    marks = {mirix_calls[1]["call_id"]: mirix_calls[0]["call_id"]}

    def flag(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") in marks:
                row["repeat_of"] = marks[row["call_id"]]
                row["item"] = mirix_calls[0]["item"]

    c.edit_ledger(ALF_MIRIX, flag)
    result = c.audit()
    assert violations(result) == set()
    assert result.cells[0].info["repeat_of"] == {"mirix": 1}


def test_mirix_usage_the_gateway_never_saw(mirix_matrix, tmp_path):
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    c.edit_ledger(
        ALF_MIRIX,
        lambda rows: rows.__setitem__(
            slice(None),
            [
                r
                for r in rows
                if not (r.get("role") == "mirix" and r.get("phase") == "learning")
            ],
        ),
    )
    assert violations(c.audit()) == {"H10.3:mirix_usage_exceeds_ledger"}


def _at(value: Any) -> datetime:
    stamp = audit._ts(value)
    assert stamp is not None, value
    return stamp


def test_mirix_calls_of_one_learning_run_dropped_from_the_ledger(
    mirix_matrix, tmp_path
):
    """One run's calls missing is caught even though the other lines remain (a
    whole-phase total would hide it behind the retrieval calls)."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    snapshot = read_json(c.cell(ALF_MIRIX).attempt(1).frozen_snapshot)["snapshot"]
    run = snapshot["learning_runs"][0]
    start, end = run["started_at"], run["finished_at"]

    def drop(rows: list[dict[str, Any]]) -> None:
        rows[:] = [
            r
            for r in rows
            if not (
                r.get("event") == "call"
                and r.get("role") == "mirix"
                and _at(start) <= _at(r["ts"]) <= _at(end)
            )
        ]

    c.edit_ledger(ALF_MIRIX, drop)
    assert violations(c.audit()) == {"H10.3:mirix_usage_exceeds_ledger"}


def test_mirix_calls_no_run_or_retrieval_explains(mirix_matrix, tmp_path):
    """A mirix-role call outside every learning run, beyond one per retrieval."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    snapshot = read_json(c.cell(ALF_MIRIX).attempt(1).frozen_snapshot)["snapshot"]
    last = max(_at(r["finished_at"]) for r in snapshot["learning_runs"])
    template = c.calls(ALF_MIRIX, role="mirix")[0]

    def extra(rows: list[dict[str, Any]]) -> None:
        stray = {
            **template,
            "call_id": "stray-call",
            "ts": (last + timedelta(milliseconds=1)).isoformat(),
        }
        stray.pop("repeat_of", None)
        rows.append(stray)

    c.edit_ledger(ALF_MIRIX, extra)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_unexplained_calls" in warnings(result)


def test_mirix_learning_that_failed_or_left_a_session_unsettled(mirix_matrix, tmp_path):
    """The frozen record shows a failed run and an unsettled learning session."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)

    def break_learning(snapshot: dict[str, Any]) -> None:
        run = snapshot["learning_runs"][-1]
        run["status"] = "failed"
        run["error"] = "gateway 504"
        entry = run["distill_metrics"][0]
        entry.update(status="deferred", failure_kind="infra", error="gateway 504")

    _rewrite_snapshot(c.cell(ALF_MIRIX), break_learning)
    result = c.audit()
    # H10.3-mirix-reconcile as recorded: counted warnings; the H10.4 snapshot checks
    # decide whether the test phase learned anything.
    assert violations(result) == set()
    assert {
        "H10.3:mirix_learning_infra",
        "H10.3:mirix_session_unsettled",
    } <= warnings(result)


def test_mirix_earlier_lessons_lost_to_the_embedding(mirix_matrix, tmp_path):
    """A curator batch that showed no earlier lessons because the experience
    pool's embedding failed is in the frozen record: the memory is not what the attempt
    learns without the failure (the method's own ``learning_problems``)."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)

    def lose_history(snapshot: dict[str, Any]) -> None:
        batch = snapshot["learning_runs"][-1]["curator_metrics"][0]
        batch.update(
            history_skipped="embedding_failed",
            history={
                "embedding_error_kind": "embedding",
                "embedding_error": "APIStatusError: Error code: 502 - {'error': "
                "{'message': 'bad gateway', 'type': 'upstream_infra_failure'}}",
            },
        )

    _rewrite_snapshot(c.cell(ALF_MIRIX), lose_history)
    result = c.audit()
    assert not violations(result) and "H10.3:mirix_learning_infra" in warnings(result)
    assert any(
        f.check == "mirix_learning_infra" and "showed no earlier lessons" in f.detail
        for cell in result.cells
        for f in cell.findings
    )


def test_mirix_calls_tagged_with_the_session_they_serve_pass_the_audit(
    mirix_matrix, tmp_path
):
    """A MIRIX run with MIRIX_LLM_SESSION_HEADER names the session each of its
    calls serves, so the ledger's mirix and embedding calls carry items (learning uids
    in the learning phase, test uids in the test phase). The audit reads them without
    a new finding: MIRIX's roles stay exempt from the repeat rule, and its windows,
    not the items, reconcile MIRIX with the ledger."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    baseline = c.audit()
    learning = c.rows(ALF_MIRIX, "learning")[0]["task_uid"]
    tested = c.rows(ALF_MIRIX, "test")[0]["task_uid"]
    assert learning and tested
    tagged = []

    def tag(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("role") in ("mirix", "embedding") and row.get("event") in (
                "call_started",
                "call",
            ):
                row["item"] = tested if row.get("phase") == "test" else learning
                tagged.append(row["item"])

    c.edit_ledger(ALF_MIRIX, tag)
    assert learning in tagged and tested in tagged
    result = c.audit()
    assert violations(result) == violations(baseline) == set()
    assert warnings(result) == warnings(baseline)


def test_mirix_curator_tool_exceptions_are_reported_as_warnings(mirix_matrix, tmp_path):
    """An exception inside a curator tool (MIRIX's own DBAPIError) that MIRIX
    answered to the model as a failed tool result is in the frozen record; the audit
    reports it per attempt as a warning, never a violation (the attempt stays valid),
    and says nothing about the model's own bad arguments."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)

    def fail_a_tool(snapshot: dict[str, Any]) -> None:
        batch = snapshot["learning_runs"][-1]["curator_metrics"][0]
        batch.update(
            failed_tool_calls=3,
            tool_failures=[
                {
                    "tool": "list_memory_within_timerange",
                    "stage": "execution",
                    "exception": "DBAPIError",
                    "error": "DBAPIError: (sqlalchemy.dialects.postgresql.asyncpg."
                    "Error) can't subtract offset-naive and offset-aware datetimes",
                    "count": 2,
                },
                {
                    "tool": "list_memory_within_timerange",
                    "stage": "execution",
                    "exception": "ValueError",
                    "error": "ValueError: Invalid isoformat string: 'string'",
                    "count": 1,
                },
            ],
        )

    _rewrite_snapshot(c.cell(ALF_MIRIX), fail_a_tool)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_curator_tool_exception" in warnings(result)
    (finding,) = [
        f
        for cell in result.cells
        for f in cell.findings
        if f.check == "mirix_curator_tool_exception"
    ]
    assert finding.cell_id == ALF_MIRIX and finding.severity == audit.WARNING
    assert "DBAPIError" in finding.detail and "ValueError" not in finding.detail


def _rewrite_snapshot(
    cell: CellPaths, change: Callable[[dict[str, Any]], None]
) -> None:
    """Change the frozen memory and every record of it consistently (H10.4 stays intact)."""
    attempt = cell.attempt(1)
    frozen = read_json(attempt.frozen_snapshot)
    change(frozen["snapshot"])
    digest = canonical_sha256(frozen["snapshot"])
    frozen["sha256"] = digest
    write_json_atomic(attempt.frozen_snapshot, frozen)
    state = read_json(attempt.state)
    state["frozen"]["sha256"] = digest
    for check in state["snapshot_checks"]:
        check["sha256"] = check["frozen_sha256"] = digest
        stored = read_json(cell.root / check["path"])
        stored["snapshot"] = frozen["snapshot"]
        stored["sha256"] = digest
        write_json_atomic(cell.root / check["path"], stored)
    write_json_atomic(attempt.state, state)


def test_mirix_learning_runs_without_usage(mirix_matrix, tmp_path):
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)

    def drop(snapshot: dict[str, Any]) -> None:
        for run in snapshot["learning_runs"]:
            run.pop("llm_usage", None)

    _rewrite_snapshot(c.cell(ALF_MIRIX), drop)
    assert violations(c.audit()) == {"H10.3:mirix_usage_missing"}


def _lost_call(template: dict[str, Any], call_id: str, **extra: Any) -> dict[str, Any]:
    """``template`` failed on infrastructure (a gateway 504), no retry of it."""
    lost = {
        **template,
        "call_id": call_id,
        "status": "error",
        "http_status": 504,
        "error": {"type": "upstream_timeout", "message": "timed out"},
        "cost": 0,
        **extra,
    }
    lost.pop("repeat_of", None)
    return lost


@pytest.mark.parametrize("role", ["mirix", "embedding"])
def test_a_mirix_retrieval_that_lost_its_infrastructure_in_the_test_phase(
    mirix_matrix, tmp_path, role
):
    """At the cell level: MIRIX's test-phase retrieval swallows an infrastructure
    failure of its topic extraction (``mirix``) or query embedding (``embedding``) and
    the item is scored on the degraded retrieval. The ledger shows the lost call, so the
    cell cannot pass; a retry that succeeded recovers it."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    template = c.calls(ALF_MIRIX, role=role)[0]
    test_actor = [
        r
        for r in c.calls(ALF_MIRIX, role="actor")
        if (r.get("run_phase") or r.get("phase")) == "test"
    ]
    at = min(_at(r["ts"]) for r in test_actor) + timedelta(milliseconds=5)
    phase = {"phase": "test", "run_phase": "test", "ts": at.isoformat()}
    c.edit_ledger(
        ALF_MIRIX, lambda rows: rows.append(_lost_call(template, "lost", **phase))
    )
    result = c.audit()
    # Recorded: a warning naming the lost call.
    assert violations(result) == set()
    assert "H10.3:mirix_test_infra_unrecovered" in warnings(result)
    (found,) = [
        f
        for cell in result.cells
        for f in cell.findings
        if f.check == "mirix_test_infra_unrecovered"
    ]
    assert "call lost" in found.detail and at.isoformat() in found.detail
    assert f"({role}, test," in found.detail
    # MIRIX's own retry of the same request, which succeeded, recovers it.
    retried = {**template, **phase, "call_id": "retried", "repeat_of": "lost"}
    c.edit_ledger(ALF_MIRIX, lambda rows: rows.append(retried))
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_test_infra_unrecovered" not in warnings(result)


@pytest.fixture(scope="module")
def mirix_degraded_matrix(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Matrix, list[dict[str, CellOutcome]]]:
    """Through run.py: one ALFWorld MIRIX cell on the stub MIRIX whose first two
    test-phase retrievals lose, one their topic extraction (an upstream 502 on the
    ``mirix`` role, which the gateway retries and gives up on), the other their query
    vector (``embedding``). The first invocation leaves both items pending; the second
    re-runs them (H8) and completes the cell."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import (
        StubMirix,
        StubMirixServer,
        with_stub_mirix,
    )

    root = tmp_path_factory.mktemp("mirix-degraded")
    env = alfworld_env(root)
    stubs: dict[tuple[str, int], StubMirix] = {}

    def server(spec: Any, mode: str) -> StubMirixServer:
        key = (spec.cell_id, spec.attempt)
        if mode == "fresh":
            stubs[key] = StubMirix(
                threshold=2, degrade_test_retrievals={"mirix": 1, "embedding": 1}
            )
        return StubMirixServer(stubs[key], mode)

    model = E2EModel()
    fake = FakeUpstream(responder=with_stub_mirix(model))
    runs = root / "runs"
    outcomes = []
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        for _ in range(2):
            outcomes.append(
                asyncio.run(
                    run_cells(
                        runs,
                        url,
                        {"alfworld": env},
                        [cell_spec("alfworld", "mirix")],
                        method_factories={"mirix": lambda: MirixMethod(poll_s=0.01)},
                        mirix_server_factory=server,
                    )
                )
            )
    return Matrix(root, runs, {"alfworld": env}, outcomes[-1], model, fake), outcomes


def _retrieval_records(c: Copy, cell_id: str) -> Path:
    from evals.skill_bench.methods.mirix import RETRIEVAL_RECORD_FILE

    return c.cell(cell_id).attempt(1).method_dir / RETRIEVAL_RECORD_FILE


def test_a_mirix_item_whose_retrieval_lost_its_infrastructure_is_re_run(
    mirix_degraded_matrix, tmp_path
):
    """End to end: the method refuses both degraded retrievals (INFRA
    ``mirix_retrieval``), run.py leaves the items pending instead of scoring them, the
    next invocation re-runs them, and the audit passes: every lost call falls inside a
    retrieval the method refused, and every scored item was scored on a clean one."""
    matrix, outcomes = mirix_degraded_matrix
    first, second = (o[ALF_MIRIX] for o in outcomes)
    assert (first.status, first.pending) == ("test_incomplete", 2), first.detail
    assert second.status == "complete", second.detail
    paths = CellPaths(matrix.runs / ALF_MIRIX).attempt(1)
    failures = [r for r in jsonl(paths.failures) if r["phase"] == "test"]
    assert sorted((r["kind"], r["code"], r["role"]) for r in failures) == [
        ("infra", "mirix_retrieval", "embedding"),
        ("infra", "mirix_retrieval", "mirix"),
    ]
    assert {r["recovery"] for r in failures} == {"rerun_item"}
    rows = jsonl(paths.results)
    lost = [r for r in rows if r["score"] is None]
    assert len(lost) == 2 and all(r["failure_stage"] == "before_task" for r in lost)
    assert all(r["actor_calls"] == 0 for r in lost)  # refused before the actor played
    reruns = read_json(paths.state)["test_reruns"]
    assert sorted(r["task_uid"] for r in reruns) == sorted(r["task_uid"] for r in lost)
    c = Copy(matrix, tmp_path, ALF_MIRIX)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    info = result.cells[0].info["mirix_test_retrievals"]
    assert info["refused"] == 2 and info["lost_calls"] == 2
    assert info["lost_calls_unexplained"] == 0 and info["scored_items_checked"] == 3
    lost_roles = sorted(
        r["role"]
        for r in c.calls(ALF_MIRIX)
        if r.get("run_phase") == "test" and r.get("status") != "ok"
    )
    assert lost_roles == ["embedding", "mirix"]


def test_the_audit_judges_each_mirix_test_retrieval(mirix_degraded_matrix, tmp_path):
    """The audit re-judges every record with the method's own rules."""
    matrix, _ = mirix_degraded_matrix
    lost_extraction = (
        "LLMServerError: INTERNAL_SERVER_ERROR: OpenAI API error: Error code: 502 - "
        "{'error': {'source': 'gateway', 'type': 'upstream_infra_failure'}}"
    )

    def edit_records(c: Copy, change: Callable[[list[dict[str, Any]]], Any]) -> None:
        path = _retrieval_records(c, ALF_MIRIX)
        rows = jsonl(path)
        change(rows)
        write_jsonl(path, rows)

    def scored(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return [r for r in rows if r["frozen"] and not r["failure"]][-1]

    # A scored item's retrieval lost its extraction to the infrastructure.
    c = Copy(matrix, tmp_path / "a", ALF_MIRIX)
    edit_records(
        c,
        lambda rows: scored(rows)["evidence"].update(
            extraction_status="fallback", extraction_error=lost_extraction
        ),
    )
    result = c.audit()
    # Recorded: every case below is a warning.
    assert violations(result) == set()
    assert "H10.3:mirix_degraded_retrieval_scored" in warnings(result)
    # ... or MIRIX's own fallback after the model's answer failed: that is scored.
    c = Copy(matrix, tmp_path / "b", ALF_MIRIX)
    edit_records(
        c,
        lambda rows: scored(rows)["evidence"].update(
            extraction_status="fallback",
            extraction_error="TopicExtractionError: no usable topic: no tool call",
        ),
    )
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_degraded_retrieval_scored" not in warnings(result)
    # A method that did not refuse: the lost calls are unexplained.
    c = Copy(matrix, tmp_path / "c", ALF_MIRIX)

    def forget(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["failure"]:
                row["failure"] = None
                row["evidence"].update(
                    extraction_status="ok",
                    extraction_error=None,
                    embedding_status="ok",
                    embedding_error=None,
                )

    edit_records(c, forget)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_test_infra_unrecovered" in warnings(result)
    assert result.cells[0].info["mirix_test_retrievals"]["lost_calls_unexplained"] == 2
    # No record, or no evidence, for a scored item.
    c = Copy(matrix, tmp_path / "d", ALF_MIRIX)

    def drop_one_item(rows: list[dict[str, Any]]) -> None:
        refused = {r["task_uid"] for r in rows if r["failure"]}
        uid = next(
            r["task_uid"] for r in rows if r["frozen"] and r["task_uid"] not in refused
        )
        rows[:] = [r for r in rows if not (r["frozen"] and r["task_uid"] == uid)]

    edit_records(c, drop_one_item)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_retrieval_unrecorded" in warnings(result)
    c = Copy(matrix, tmp_path / "e", ALF_MIRIX)
    edit_records(c, lambda rows: scored(rows).pop("evidence"))
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_retrieval_unchecked" in warnings(result)
    # A learning retrieval the method refused: the attempt should have restarted.
    c = Copy(matrix, tmp_path / "f", ALF_MIRIX)
    edit_records(
        c,
        lambda rows: next(r for r in rows if not r["frozen"])["evidence"].update(
            embedding_status="failed", embedding_error="ConnectError: refused"
        ),
    )
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:mirix_learning_retrieval_degraded" in warnings(result)
    assert result.cells[0].info["recorded_counts"] == {
        "H10.3:mirix_learning_retrieval_degraded": 1
    }


def test_only_mirix_cells_answer_for_a_lost_test_phase_embedding(copy):
    """This check is MIRIX's: ReasoningBank's query embedding raises and H8 re-runs the
    item, whose failed line may stay unrecovered (a re-run after the repeat window)."""
    c = copy(ALF_RB)
    template = c.calls(ALF_RB, role="embedding", phase="test")[0]
    c.edit_ledger(ALF_RB, lambda rows: rows.append(_lost_call(template, "lost")))
    assert "H10.3:mirix_test_infra_unrecovered" not in violations(c.audit())


# --- H10.4: the frozen memory ------------------------------------------------------------------------


def test_a_memory_that_changed_in_the_test_phase(copy):
    c = copy(ALF_RB)

    def change(state: dict[str, Any]) -> None:
        state["snapshot_checks"][-1].update(equal=False, sha256="1" * 64)

    c.edit(c.cell(ALF_RB).attempt(1).state, change)
    assert violations(c.audit()) == {"H10.4:snapshot_changed", "H10.4:snapshot_record"}


def test_a_frozen_record_that_does_not_match_its_hash(copy):
    c = copy(ALF_RB)
    c.edit(
        c.cell(ALF_RB).attempt(1).frozen_snapshot,
        lambda d: d["snapshot"].__setitem__("notes", "rewritten after the freeze"),
    )
    assert violations(c.audit()) == {"H10.4:frozen_record"}


def test_a_stored_snapshot_edited_after_the_check(copy):
    c = copy(ALF_RB)
    state = read_json(c.cell(ALF_RB).attempt(1).state)
    stored = c.cell(ALF_RB).root / state["snapshot_checks"][-1]["path"]
    c.edit(stored, lambda d: d["snapshot"].__setitem__("notes", "edited"))
    assert violations(c.audit()) == {"H10.4:snapshot_record"}


def test_a_test_phase_without_an_after_test_snapshot(copy):
    c = copy(ALF_RB)
    c.edit(
        c.cell(ALF_RB).attempt(1).state,
        lambda s: s.__setitem__(
            "snapshot_checks",
            [x for x in s["snapshot_checks"] if x["kind"] != "after_test"],
        ),
    )
    assert violations(c.audit()) == {
        "H10.4:after_test_missing",
        "H10.4:unverified_launch",
    }


def test_an_interrupted_launch_verified_by_the_next_resume_is_a_warning(copy):
    c = copy(ALF_RB)
    attempt = c.cell(ALF_RB).attempt(1)
    rows = jsonl(attempt.results)
    mark = {"launch": "interrupted", "at": "2026-01-01T00:00:00+00:00"}
    rows[0].update(mark)
    write_jsonl(attempt.results, rows)
    # The cell's results.jsonl copies the attempt's rows (H9), so it carries them too.
    published = jsonl(c.cell(ALF_RB).results)
    for row in published:
        if row["raw_id"] == rows[0]["raw_id"]:
            row.update(mark)
    write_jsonl(c.cell(ALF_RB).results, published)

    def resume(state: dict[str, Any]) -> None:
        after = state["snapshot_checks"][-1]
        state["snapshot_checks"].append(
            {
                **after,
                "kind": "resume",
                "launch": "next",
                "at": "2026-01-01T00:05:00+00:00",
            }
        )

    c.edit(attempt.state, resume)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.4:interrupted_launch" in warnings(result)


# --- H10.6: the protocol ---------------------------------------------------------------------------------


def _earlier_fixture() -> tuple[str, dict[str, str]]:
    """:data:`EARLIER` and its file map: the frozen constants, and the frozen file map
    with one method prompt changed, recorded as ``freeze_protocol`` records a version:
    with the fingerprint both give (a version whose record does not reproduce its
    fingerprint is never read, ``protocol.version_diff``).

    The session's frozen record is this tree (``conftest._frozen_record_copy``), and this
    runs at collection, before that copy exists: the tree's constants and file map are
    the frozen ones (the repository's record lags a tree not yet re-frozen, such as one
    that adds a benchmark)."""
    tree = protocol.fingerprint_manifest()
    files = dict(tree["files"])
    prompt = next(p for p in sorted(files) if p.startswith("methods/prompts/"))
    files[prompt] = "0" * 64
    return protocol.fingerprint_of(tree["constants"], files), files


#: A protocol fingerprint some earlier freeze produced, in the tests' known versions.
EARLIER, EARLIER_FILES = _earlier_fixture()
EARLIER_FROZEN_AT = "2026-09-29T15:43:01+00:00"


def _earlier_versions() -> dict[str, dict[str, Any]]:
    """The frozen record's versions plus :data:`EARLIER`, whose file map differs from the
    frozen one in one prompt (what a re-freeze keeps in history)."""
    known = protocol.known_versions()
    frozen = protocol.frozen_fingerprint()
    files = dict(EARLIER_FILES)
    known[EARLIER] = {
        "fingerprint": EARLIER,
        "frozen_at": EARLIER_FROZEN_AT,
        "reason": "an earlier freeze",
        "git": {"commit": "0a1b2c3"},
        "constants_sha256": known[frozen]["constants_sha256"],
        "files": files,
        "current": False,
    }
    return known


def _set_cell_protocol(c: Copy, cell_id: str, fingerprint: str | None) -> None:
    """Make ``cell_id`` a cell run under ``fingerprint``, consistently: cell.json,
    attempt.json (its version segments too, whose ``version.json`` then no longer
    describes them) and every work line; None drops the attempt's and the work lines'
    record, as an older run.py wrote them (no ``protocol`` at all)."""
    paths = c.cell(cell_id)
    if fingerprint is not None:
        c.edit(
            paths.manifest,
            lambda m: m["cell"].__setitem__("protocol_fingerprint", fingerprint),
        )

    def attempt(state: dict[str, Any]) -> None:
        if fingerprint is None:
            state.pop("protocol", None)
            state.pop("version_segments", None)
        else:
            state["protocol"]["fingerprint"] = fingerprint
            for segment in state.get("version_segments") or ():
                segment["protocol"]["fingerprint"] = fingerprint
                segment["version_file"] = None

    c.edit(paths.attempt(1).state, attempt)
    rows = jsonl(paths.launches)
    for row in rows:
        if fingerprint is None:
            row.pop("protocol", None)
        elif "protocol" in row:
            row["protocol"]["fingerprint"] = fingerprint
    write_jsonl(paths.launches, rows)


def _finding(result: audit.AuditReport, check: str) -> audit.Finding:
    (found,) = [f for x in result.cells for f in x.findings if f.check == check] + [
        f for f in result.findings if f.check == check
    ]
    return found


def test_a_cell_run_under_another_protocol(copy):
    """A fingerprint no freeze produced and no record of the cell describes is recorded
    (a warning, "version not registered"); every check that reads
    this tree is then listed as not re-verified. The gateway that served the cell ran the
    frozen protocol, another known version: a warning."""
    c = copy(ALF_RB)
    _set_cell_protocol(c, ALF_RB, "0" * 64)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert {
        "H10.6:fingerprint",
        "H10.6:gateway_protocol_version",
        "H10.6:protocol_version_unauditable",
    } <= warnings(result)
    assert "not registered" in _finding(result, "fingerprint").detail
    (cell,) = result.cells
    assert cell.info["protocol"] == {
        "fingerprint": "0" * 64,
        "current": False,
        "known": False,
        "source": "attempt",
    }
    assert f"`{'0' * 12}` (**unknown**)" in result.to_markdown()


def test_a_cell_under_an_earlier_protocol_version_is_a_warning(copy):
    """Versions are recorded and reported, not enforced across cells. A cell under a known earlier version passes with a warning naming the version,
    its freeze reason and the files that changed since."""
    c = copy(ALF_RB, ALF_NONE)
    frozen = protocol.frozen_fingerprint()
    known = _earlier_versions()
    _set_cell_protocol(c, ALF_RB, EARLIER)
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    assert {
        "H10.6:protocol_version",
        "H10.6:gateway_protocol_version",
    } <= warnings(result)
    version = _finding(result, "protocol_version")
    assert version.cell_id == ALF_RB and version.severity == audit.WARNING
    assert EARLIER[:12] in version.detail and frozen[:12] in version.detail
    assert "it was frozen 2026-09-29T15:43:01+00:00: an earlier freeze" in (
        version.detail
    )
    # Why it was replaced: the reason of the freeze that came next, besides the earlier
    # version's own reason and the files changed since.
    after = sorted(
        (v for v in known.values() if str(v.get("frozen_at")) > EARLIER_FROZEN_AT),
        key=lambda v: str(v["frozen_at"]),
    )
    assert (
        f"replaced by the freeze of {after[0]['frozen_at']}: {after[0]['reason']}"
        in version.detail
    )
    changed = protocol.version_diff(EARLIER, frozen, known)
    assert changed is not None and changed["changed"]
    assert f"changed: {changed['changed'][0]}" in version.detail
    # Only a method prompt differs: the audit from this tree sees it as it ran.
    assert "H10.6:protocol_version_unauditable" not in warnings(result)
    across = _finding(result, "protocol_differs_across_cells")
    assert across.severity == audit.WARNING
    assert f"{EARLIER[:12]}: {ALF_RB}" in across.detail
    assert f"{frozen[:12]}: {ALF_NONE}" in across.detail
    cells = {x.cell_id: x for x in result.cells}
    assert cells[ALF_RB].info["protocol"]["current"] is False
    assert cells[ALF_RB].info["protocol"]["known"] is True
    assert cells[ALF_NONE].info["protocol"]["current"] is True
    assert result.passed and result.exit_code == 0
    md = result.to_markdown()
    assert "## Versions" in md
    assert f"| `{ALF_RB}` | `{EARLIER[:12]}` (earlier) |" in md
    assert f"| `{ALF_NONE}` | `{frozen[:12]}` (current) |" in md
    # Without it among the known versions, the same cell's version is not registered:
    # recorded, and its tree-reading checks are listed as not re-verified.
    unknown = c.audit()
    assert violations(unknown) == set()
    assert "H10.6:fingerprint" in warnings(unknown)


def test_diff_versions_keeps_the_slugs_of_the_entry_the_tree_replaces():
    old_sha, new_sha = "a" * 64, "b" * 64
    recorded = {
        "v1": {"fingerprint": "v1", "model_slugs": {}},
        "v2": {
            "fingerprint": "v2",
            "reason": "a freeze",
            "model_slugs": {old_sha: {"m": "m-1"}, new_sha: {"m": "stale"}},
        },
    }
    tree = {"fingerprint": "v2", "reason": "the auditing tree"}
    tree["model_slugs"] = {new_sha: {"m": "m-2", "n": "n-1"}}
    merged = audit.diff_versions(recorded, {}, tree)
    assert merged["v1"] is recorded["v1"]
    assert merged["v2"]["reason"] == "the auditing tree"
    # The record's slugs of the earlier file stay; the tree's own file's are the tree's.
    assert merged["v2"]["model_slugs"] == {
        old_sha: {"m": "m-1"},
        new_sha: {"m": "m-2", "n": "n-1"},
    }
    assert recorded["v2"]["model_slugs"][new_sha] == {"m": "stale"}  # not mutated
    assert tree["model_slugs"] == {new_sha: {"m": "m-2", "n": "n-1"}}
    # A tree no recorded or carried entry describes stands as it is.
    other = {"fingerprint": "v3", "model_slugs": {new_sha: {"m": "m-2"}}}
    assert audit.diff_versions(recorded, {}, other)["v3"] == other
    # A carried entry of the tree's fingerprint (auto-frozen elsewhere) keeps its slugs.
    carried = {"v3": {"fingerprint": "v3", "model_slugs": {old_sha: {"m": "m-0"}}}}
    assert audit.diff_versions(recorded, carried, other)["v3"]["model_slugs"] == {
        old_sha: {"m": "m-0"},
        new_sha: {"m": "m-2"},
    }


def test_an_earlier_version_of_other_constants_or_templates_is_named_unauditable(copy):
    """The checks after the version compare every call with this
    tree's constants and rebuild the first prompts from this tree's templates, so a
    known earlier version whose constants or shared files differ cannot be audited from
    this tree. A distinct warning says so and names the commit to audit it from; the
    violations those checks report stay (fail-closed)."""
    c = copy(ALF_RB)
    frozen = protocol.frozen_fingerprint()
    known = _earlier_versions()
    _set_cell_protocol(c, ALF_RB, EARLIER)
    other = dict(known)
    template = next(p for p in sorted(known[frozen]["files"]) if "/templates/" in p)
    other[EARLIER] = {
        **known[EARLIER],
        "constants_sha256": "0" * 64,
        "files": {**known[EARLIER]["files"], template: "0" * 64},
    }
    result = c.audit(versions=other)
    found = _finding(result, "protocol_version_unauditable")
    assert found.severity == audit.WARNING and found.cell_id == ALF_RB
    assert "constants of protocol.py" in found.detail and template in found.detail
    # The recorded commit is the one it was frozen from, not one that holds it; the
    # warning names how to find the commit that recorded it.
    assert (
        "Audit it from "
        + protocol.version_checkout_hint(EARLIER, {"commit": "0a1b2c3"})
    ) in found.detail
    assert protocol.recorded_commit_command(EARLIER) in found.detail
    assert "frozen from commit 0a1b2c3" in (found.detail)
    assert "H10.6:protocol_version" in warnings(result)
    # A version frozen before the record kept file maps: what differs is not known.
    legacy = dict(known)
    legacy[EARLIER] = {**known[EARLIER], "files": None, "constants_sha256": None}
    found = _finding(c.audit(versions=legacy), "protocol_version_unauditable")
    assert "a file map that is not recorded" in found.detail


# --- per-key constants through the audit ---------------------------------------------


def _earlier_with_constants(
    known: dict[str, dict[str, Any]], *, per_key: bool = True, **edit: Any
) -> str:
    """Add to ``known`` an earlier version as ``freeze_protocol`` records one: the frozen
    constants with ``edit``, a file map that differs from the frozen one in a method
    prompt, and the fingerprint both give. ``per_key`` False: an entry frozen before the
    record kept its constants per key (only their hash). Returns its fingerprint."""
    frozen = known[protocol.frozen_fingerprint()]
    constants = {**frozen["constants"], **edit}
    files = dict(frozen["files"])
    prompt = next(p for p in sorted(files) if p.startswith("methods/prompts/"))
    files[prompt] = "0" * 64
    fingerprint = protocol.fingerprint_of(constants, files)
    known[fingerprint] = {
        "fingerprint": fingerprint,
        "frozen_at": EARLIER_FROZEN_AT,
        "reason": "an earlier freeze",
        "git": {"commit": "0a1b2c3", "dirty": True},
        "constants_sha256": protocol.constants_sha256(constants),
        "constants": constants if per_key else None,
        "inconsistent": False,
        "files": files,
        "current": False,
    }
    return fingerprint


def _with_the_removed_method() -> list[str]:
    """The method list of a version from before a method's removal (that one last)."""
    return [*protocol.METHODS, REMOVED_METHOD]


def test_a_version_that_differs_only_in_the_method_list_is_audited_as_it_ran(copy):
    """A cell of a version whose constants differ from the frozen
    ones only in ``methods`` (the study's composition) and whose files differ only in
    method prompts gets the ``protocol_version`` warning, naming the key, and no
    ``protocol_version_unauditable``: the audit sees past it."""
    c = copy(ALF_RB, ALF_NONE)
    known = protocol.known_versions()
    earlier = _earlier_with_constants(known, methods=_with_the_removed_method())
    _set_cell_protocol(c, ALF_RB, earlier)
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    found = _finding(result, "protocol_version")
    assert found.cell_id == ALF_RB
    assert "since then: constants of protocol.py changed: methods; changed:" in (
        found.detail
    )
    assert "H10.6:protocol_version_unauditable" not in warnings(result)


def test_a_version_that_differs_in_another_constant_names_it_unauditable(copy):
    """The method list and ``seed`` differ: the warning names ``seed`` (every cell ran
    under it), never the method list, and says where to audit the cell from."""
    c = copy(ALF_RB)
    known = protocol.known_versions()
    earlier = _earlier_with_constants(
        known, methods=_with_the_removed_method(), seed=protocol.SEED + 1
    )
    _set_cell_protocol(c, ALF_RB, earlier)
    result = c.audit(versions=known)
    assert "constants of protocol.py changed: methods, seed" in (
        _finding(result, "protocol_version").detail
    )
    found = _finding(result, "protocol_version_unauditable")
    assert found.severity == audit.WARNING and found.cell_id == ALF_RB
    assert "differs from the frozen one in protocol.py constant seed:" in found.detail
    assert "protocol.py constant methods" not in found.detail
    assert protocol.recorded_commit_command(earlier) in found.detail


def test_a_version_without_per_key_constants_is_audited_by_those_of_its_hash(copy):
    """An older version kept only its constants hash; a version with the same hash and
    per-key constants (the backfilled one) says which keys changed, when those constants
    reproduce the earlier version's own fingerprint. A record whose hash does not match
    its fingerprint (edited) is never read: unauditable."""
    c = copy(ALF_RB)
    known = protocol.known_versions()
    methods = _with_the_removed_method()
    backfilled = _earlier_with_constants(known, methods=methods)
    older = dict(known[backfilled])
    files = {**older["files"]}
    template = next(p for p in sorted(files) if p.startswith("methods/prompts/awm/"))
    files[template] = "1" * 64
    fingerprint = protocol.fingerprint_of(older["constants"], files)
    known[fingerprint] = {
        **older,
        "fingerprint": fingerprint,
        "constants": None,  # frozen before the record kept them per key
        "files": files,
    }
    _set_cell_protocol(c, ALF_RB, fingerprint)
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    assert "constants of protocol.py changed: methods;" in (
        _finding(result, "protocol_version").detail
    )
    assert "H10.6:protocol_version_unauditable" not in warnings(result)
    # Without a version that keeps those constants per key (the backfilled one), which
    # keys changed is not known: fail closed.
    alone = {
        k: v
        for k, v in known.items()
        if k == fingerprint or v["constants_sha256"] != older["constants_sha256"]
    }
    assert backfilled not in alone
    found = _finding(c.audit(versions=alone), "protocol_version_unauditable")
    assert "differs from the frozen one in constants of protocol.py:" in found.detail
    # The same hash on a record its constants do not reproduce: never read.
    edited = "1" * 64
    known[edited] = {**known[fingerprint], "fingerprint": edited}
    _set_cell_protocol(c, ALF_RB, edited)
    found = _finding(c.audit(versions=known), "protocol_version_unauditable")
    assert (
        f"differs from the frozen one in a record that does not reproduce its "
        f"fingerprint ({edited[:12]})"
    ) in found.detail


def test_cells_without_recorded_protocols_pass(copy):
    """Cells written before attempts and work lines recorded their protocol: the
    attempt ran under cell.json's value. Under the frozen value they pass as before;
    under a known earlier one they get warnings only: the within-attempt check has
    nothing to compare."""
    c = copy(ALF_RB, ALF_NONE)
    _set_cell_protocol(c, ALF_RB, None)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    cells = {x.cell_id: x for x in result.cells}
    assert cells[ALF_RB].info["protocol"]["source"] == "cell"
    assert cells[ALF_RB].info["protocol"]["current"] is True
    assert not {
        "H10.6:protocol_version",
        "H10.6:protocol_changed_within_attempt",
    } & warnings(result)

    c.edit(
        c.cell(ALF_RB).manifest,
        lambda m: m["cell"].__setitem__("protocol_fingerprint", EARLIER),
    )
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set(), result.to_markdown()
    assert {
        "H10.6:protocol_version",
        "H10.6:gateway_protocol_version",
    } <= warnings(result)
    assert "H10.6:protocol_differs_across_cells" in {
        f"{f.invariant}:{f.check}" for f in result.findings
    }


def test_a_protocol_change_within_an_attempt_is_recorded(copy):
    """An attempt records every version it ran under
    (``version_segments``). A work line under a protocol that neither attempt.json nor a
    version segment records is a recorded warning, whatever versions are known; a work
    line that records none is not checked."""
    c = copy(ALF_RB)
    launches = c.cell(ALF_RB).launches
    rows = jsonl(launches)
    (work,) = [r for r in rows if r.get("event") == "work"]
    later = {
        **work,
        "at": "2999-01-01T00:00:00+00:00",
        "launch": "0123456789ab",
        "phase": "test",
        "protocol": {**work["protocol"], "fingerprint": EARLIER},
    }
    unrecorded = {**work, "launch": "ba9876543210", "phase": "test"}
    unrecorded.pop("protocol")
    write_jsonl(launches, [*rows, later, unrecorded])
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set(), result.to_markdown()
    assert "H10.6:protocol_changed_within_attempt" in warnings(result)
    found = _finding(result, "protocol_changed_within_attempt")
    assert "0123456789ab" in found.detail and EARLIER[:12] in found.detail
    assert found.downgraded == audit.RECORD
    write_jsonl(launches, [*rows, unrecorded])
    result = c.audit()
    assert violations(result) == set()
    assert "H10.6:protocol_changed_within_attempt" not in warnings(result)


def test_calls_off_the_protocol_settings(copy):
    c = copy(ALF_RB)
    actors = c.calls(ALF_RB, role="actor", phase="test")

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("event") != "call":
                continue
            if row["call_id"] == actors[0]["call_id"]:
                row["protocol_conformant"] = False
            if row["call_id"] == actors[1]["call_id"]:
                row["gateway_settings_sha256"] = "0" * 64
            if row["call_id"] == actors[2]["call_id"]:
                row["wire"]["reasoning"] = {"effort": "high"}
                row["wire"]["seed"] = 7
                # As if really sent so: the forwarded body's hash follows the wire.
                request = c.recorded_request(ALF_RB, row["call_id"])
                row["request_sha256"] = audit.sent_request_sha256(request, row["wire"])

    c.edit_ledger(ALF_RB, change)
    result = c.audit()
    # The model parameters stay validity checks (H10.6-gateway-settings); a line's
    # conformance flag and settings hash are recorded.
    assert violations(result) == {"H10.6:reasoning_effort", "H10.6:seed"}
    assert {"H10.6:nonconformant_call", "H10.6:gateway_settings"} <= warnings(result)
    effort = _finding(result, "reasoning_effort")
    assert "its gateway start record says effort 'medium'" in effort.detail


def test_a_gateway_started_under_another_protocol(copy):
    """The gateway's fingerprint is compared with the attempt's versions: one no freeze
    produced is recorded, another known version a warning. Its
    model settings are held to the attempt's own version (a violation when they
    differ); every other setting that differs is one warning (H10.6-gateway-settings)."""
    c = copy(ALF_RB)
    starts = c.runs / "_gateway" / "starts.jsonl"
    original = starts.read_text()
    rows = jsonl(starts)
    for row in rows:
        row["protocol_fingerprint"] = "0" * 64
        row["settings"] = {**row["settings"], "max_retries": 9}
    write_jsonl(starts, rows)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert {
        "H10.6:gateway_fingerprint",
        "H10.6:gateway_settings_differ",
    } <= warnings(result)
    differ = _finding(result, "gateway_settings_differ")
    assert "max_retries: start 9 / attempt 5 / tree 5" in differ.detail
    # A model setting the attempt's version does not record: calls went out with other
    # model parameters (validity).
    rows = jsonl(starts)
    for row in rows:
        row["settings"] = {**row["settings"], "reasoning_effort": "high"}
    write_jsonl(starts, rows)
    result = c.audit()
    assert violations(result) >= {"H10.6:gateway_start_settings"}
    assert (
        "reasoning_effort='high'" in _finding(result, "gateway_start_settings").detail
    )
    starts.write_text(original)
    rows = jsonl(starts)
    for row in rows:
        row["protocol_fingerprint"] = EARLIER
    write_jsonl(starts, rows)
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set(), result.to_markdown()
    assert "H10.6:gateway_protocol_version" in warnings(result)
    assert "H10.6:gateway_fingerprint" in warnings(c.audit())


def test_an_audit_against_another_frozen_value(copy):
    """An explicit frozen value is the only known version unless ``versions`` says
    otherwise: the cell's (the tree's) is unknown. Its gateway ran the attempt's
    protocol, which is all the start records are compared with."""
    c = copy(ALF_RB)
    result = c.audit(frozen="0" * 64)
    # No run-level auditor_tree: the cell's version is recorded as one its own records
    # describe (run.py writes the start version's version.json), else as not
    # registered; the tree is its own
    # version, so nothing of what the checks read differs and every check keeps its
    # severity.
    assert violations(result) == set(), result.to_markdown()
    assert result.cells[0].info["protocol_record"] == "attempt record"
    carried = _finding(result, "protocol_version")
    assert "registered only in its own attempt record" in carried.detail
    assert "H10.6:fingerprint" not in warnings(result)
    # Without the cell's own version records the version is not registered.
    attempt = c.cell(ALF_RB).attempt(1)
    for path in (c.cell(ALF_RB).root / "attempt-01").rglob("version.json"):
        path.unlink()
    c.edit(
        attempt.state,
        lambda s: s.__setitem__(
            "protocol",
            {k: v for k, v in s["protocol"].items() if k not in ("constants", "files")},
        ),
    )
    result = c.audit(frozen="0" * 64)
    assert violations(result) == set(), result.to_markdown()
    assert result.cells[0].info["protocol_record"] is None
    assert "H10.6:fingerprint" in warnings(result)
    assert not [
        f
        for f in [*result.findings, *(f for x in result.cells for f in x.findings)]
        if f.check == "auditor_tree"
    ]
    assert result.cells[0].info["not_reverified"] == []
    assert result.exit_code == 0 and result.tree_current is False


def test_an_audit_under_another_protocol_never_replaces_a_passing_record(
    copy, tmp_path
):
    """An audit of a tree whose protocol has moved on since the cells ran must not
    overwrite their passing audit.json (report.py would then mark every cell failed). A
    failing audit made under another protocol state (frozen or auditor fingerprint) is
    written beside the passing record instead; one made under the same state, or with
    ``replace_passing``, replaces it."""
    c = copy(ALF_RB)
    out = tmp_path / "_audit"
    passing = c.audit()
    assert passing.passed
    json_path, _ = audit.write_audit(passing, out)
    assert json_path == out / audit.AUDIT_JSON
    record = json_path.read_text()

    leaky = Copy(c.matrix, tmp_path / "leaky", ALF_RB)
    leaky.edit(
        leaky.trajectory(ALF_RB, "learning"),
        lambda d: d["messages"][2].__setitem__("won", True),
    )
    drifted = leaky.audit(frozen="0" * 64)
    assert not drifted.passed and "H10.1:label_key" in violations(drifted)
    assert audit.kept_passing_record(drifted, out) == json_path
    beside, beside_md = audit.write_audit(drifted, out)
    sub = out / f"protocol-{drifted.auditor_fingerprint[:12]}"
    assert (beside, beside_md) == (sub / audit.AUDIT_JSON, sub / audit.AUDIT_MD)
    assert json.loads(beside.read_text())["passed"] is False
    assert json_path.read_text() == record  # the passing record is untouched

    # The same protocol state: a failure is news about the runs and replaces the record.
    c.edit_ledger(ALF_RB, lambda rows: rows.append({"event": "call", "role": ""}))
    broken = c.audit()
    assert not broken.passed
    assert audit.kept_passing_record(broken, out) is None
    assert audit.write_audit(broken, out)[0] == json_path
    assert json.loads(json_path.read_text())["passed"] is False

    # A passing report always replaces; ``replace_passing`` forces a failing one in.
    audit.write_audit(passing, out)
    assert audit.kept_passing_record(passing, out) is None
    assert audit.write_audit(drifted, out, replace_passing=True)[0] == json_path
    assert json.loads(json_path.read_text())["passed"] is False
    # No record, or a failing one: nothing to keep.
    assert audit.kept_passing_record(drifted, tmp_path / "empty") is None
    assert audit.kept_passing_record(drifted, out) is None


def test_incomplete_cells_are_listed_not_audited(copy):
    c = copy(ALF_RB)
    status = c.cell(ALF_RB).status
    write_json_atomic(status, {**read_json(status), "status": "test_incomplete"})
    result = c.audit()
    assert [cell.audited for cell in result.cells] == [False]
    assert result.incomplete == [ALF_RB] and result.exit_code == 2
    assert "Not audited" in result.to_markdown()
    included = c.audit(include_incomplete=True)
    assert included.cells[0].audited and violations(included) == set()


# --- helpers ------------------------------------------------------------------------------------------


def test_role_policy_covers_every_method():
    assert set(audit.ROLE_POLICY) == set(protocol.METHODS)
    for method, phases in audit.ROLE_POLICY.items():
        assert set(phases) == {"setup", "learning", "test"}, method
        assert all(roles <= set(protocol.ROLES) for roles in phases.values())
        assert not phases["test"] & audit.LEARNING_ROLES, method


def test_identifiers_and_word_boundaries():
    assert audit.identifier_like("train:0023")
    assert audit.identifier_like("pick_and_place_simple")
    assert audit.identifier_like("pgjw0227") and audit.identifier_like("600 mg")
    assert not audit.identifier_like("62409") and not audit.identifier_like("layout")
    corpus = audit.Corpus()
    corpus.add("what is the number of pages?", "a", "q")
    corpus.add("see page=180 and page=18.", "b", "obs")
    corpus.add("", "c", "empty")
    assert list(corpus.find("no")) == []
    assert list(corpus.find("page=18")) == [("b", "obs")]
    assert list(corpus.find("pages")) == [("a", "q")]
    assert list(corpus.find("?")) == [("a", "q")]


def test_model_families():
    assert audit.model_family("openai/gpt-6-luna") == "gpt-6-luna"
    assert audit.model_family("openai/gpt-6-luna-2026-09-22") == "gpt-6-luna"
    assert audit.model_family("openai/gpt-6-luna-20260922") == "gpt-6-luna"
    # An older model of the same line is another family: a cell served both is caught.
    assert audit.model_family("openai/gpt-5.6-luna-20260709") == "gpt-5.6-luna"
    assert audit.model_family(protocol.MODELS["luna"]) == "gpt-6-luna"
    assert audit.model_family(protocol.MODELS["luna56"]) == "gpt-5.6-luna"
    assert audit.model_family("anthropic/claude-other-9") == "claude-other-9"
    # What each model may be served as: its id's family and its canonical slug's.
    assert audit.requested_families(protocol.MODELS["luna"]) == {"gpt-6-luna"}
    assert audit.requested_families(protocol.MODELS["luna56"]) == {"gpt-5.6-luna"}
    assert audit.requested_families(protocol.EMBEDDING_MODEL) == {
        "gemini-embedding-001"
    }
    # The same weights in "pro" reasoning mode are another model.
    assert audit.model_family("openai/gpt-6-luna-pro-20260922") == "gpt-6-luna-pro"


@pytest.mark.parametrize(
    "served",
    [
        "openai/gpt-5.6-luna-20260709",  # an older model of the same line
        "openai/gpt-6-luna-pro-20260922",  # the same weights in pro reasoning mode
    ],
)
def test_a_cell_served_entirely_as_another_model_is_caught(copy, served):
    """Every ok call of the cell served as one other model is one family, so
    the mixed-families check alone passes it; the served family must be the requested
    model's own."""
    c = copy(ALF_RB)
    assert ALF_RB.startswith("alfworld-luna-")  # the cell under test runs luna

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["event"] == "call" and row.get("model") == protocol.MODELS["luna"]:
                row["model_served"] = served

    c.edit_ledger(ALF_RB, change)
    result = c.audit()
    assert violations(result) == {"H10.3:model_served"}
    details = [
        f.detail
        for cell in result.cells
        for f in cell.findings
        if f.check == "model_served"
    ]
    assert len(details) == 1 and audit.model_family(served) in details[0]
    # Served as the requested model's dated snapshot, the same cell passes.
    c.edit_ledger(
        ALF_RB,
        lambda rows: [
            row.__setitem__("model_served", "openai/gpt-6-luna-20260922")
            for row in rows
            if row.get("model_served") == served
        ],
    )
    assert violations(c.audit()) == set()


LUNA_SLUG = "openai/gpt-6-luna-20260922"
#: A canonical slug only the edited models.json gives luna (another model family).
TURBO_SLUG = "openai/gpt-6-luna-turbo-20261001"
EDITED_MODELS = "5" * 64


def _models_json_edited(monkeypatch: pytest.MonkeyPatch, *, luna: str) -> None:
    """The auditing tree after an edit of config/models.json alone: its hash is
    :data:`EDITED_MODELS`, and it gives luna the canonical slug ``luna``."""
    real = protocol.fingerprint_manifest()
    files = {**real["files"], protocol.MODELS_JSON_REL: EDITED_MODELS}
    drifted = {
        **real,
        "files": files,
        "fingerprint": protocol.fingerprint_of(real["constants"], files),
    }
    slugs = dict(protocol.canonical_slugs(protocol.load_models()))
    slugs[protocol.MODELS["luna"]] = luna
    monkeypatch.setattr(protocol, "fingerprint_manifest", lambda *a, **kw: drifted)
    monkeypatch.setattr(
        protocol, "model_slugs_record", lambda *a, **kw: {EDITED_MODELS: slugs}
    )


def _serve_luna_as(c: Copy, served: str) -> None:
    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["event"] == "call" and row.get("model") == protocol.MODELS["luna"]:
                row["model_served"] = served

    c.edit_ledger(ALF_RB, change)


def _drop_cell_model_slugs(c: Copy, cell_id: str) -> int:
    """Remove ``model_slugs`` from every ``version.json`` of the cell's attempt 1, as
    run.py wrote them before it recorded them; returns how many it held."""
    held = 0
    for path in sorted(
        (c.cell(cell_id).attempt(1).root / "versions").glob("*/version.json")
    ):
        document = json.loads(path.read_text())
        held += bool(document.pop("model_slugs", None))
        path.write_text(json.dumps(document))
    return held


def _frozen_with_slugs(slugs: Mapping[str, Mapping[str, str]] | None):
    """The frozen record's versions, the frozen one keeping ``slugs`` (None: none)."""
    known = protocol.known_versions()
    frozen = known[protocol.frozen_fingerprint()]
    frozen["model_slugs"] = {} if slugs is None else dict(slugs)
    return known


def test_an_edit_of_models_json_keeps_the_model_substitution_check_a_violation(
    copy, monkeypatch
):
    """model_served reads no hash of config/models.json. After an edit of that file, a
    luna cell of the earlier version that
    OpenRouter served as another model family is still a violation: its own version's
    slugs are recorded (the frozen record's ``model_slugs``, by the file's hash), and a
    family no recorded models.json gives luna is one whatever the version."""
    c = copy(ALF_RB)
    real_sha = protocol.fingerprint_manifest()["files"][protocol.MODELS_JSON_REL]
    real_slugs = protocol.canonical_slugs(protocol.load_models())
    assert real_slugs[protocol.MODELS["luna"]] == LUNA_SLUG
    # The cell's own version.json records what its models.json requested (run.py).
    assert _drop_cell_model_slugs(c, ALF_RB) >= 1
    _models_json_edited(monkeypatch, luna=LUNA_SLUG)
    _serve_luna_as(c, "openai/gpt-5.6-luna-20260709")
    for slugs in ({real_sha: real_slugs}, None):
        result = c.audit(versions=_frozen_with_slugs(slugs))
        assert violations(result) == {"H10.3:model_served"}, result.to_markdown()
        (found,) = recorded(result, "H10.3:model_served")
        assert found.downgraded is None and "gpt-5.6-luna" in found.detail
        assert "H10.3:model_served" not in _not_reverified(result.cells[0])
        assert result.exit_code == 1


def test_a_family_only_another_models_json_requests(copy, monkeypatch):
    """The edited models.json requests luna as another family (turbo). A cell
    of the earlier version served as turbo is a violation when its version's slugs are
    recorded (it requested gpt-6-luna), and a not-re-verified warning only when they are
    not (the earlier file may have requested turbo too)."""
    c = copy(ALF_RB)
    real_sha = protocol.fingerprint_manifest()["files"][protocol.MODELS_JSON_REL]
    real_slugs = protocol.canonical_slugs(protocol.load_models())
    _drop_cell_model_slugs(c, ALF_RB)
    _models_json_edited(monkeypatch, luna=TURBO_SLUG)
    _serve_luna_as(c, TURBO_SLUG)
    result = c.audit(versions=_frozen_with_slugs({real_sha: real_slugs}))
    assert violations(result) == {"H10.3:model_served"}, result.to_markdown()

    result = c.audit(versions=_frozen_with_slugs(None))
    assert violations(result) == set(), result.to_markdown()
    (found,) = recorded(result, "H10.3:model_served")
    assert found.severity == audit.WARNING
    assert found.downgraded == audit.NOT_REVERIFIED
    (listed,) = _not_reverified(result.cells[0])["H10.3:model_served"]
    assert "canonical_slug of openai/gpt-6-luna" in listed and "not recorded" in listed


def test_the_cells_own_version_json_resolves_its_slugs(copy, monkeypatch):
    """Without the frozen record's slugs, the ``model_slugs`` run.py wrote in
    the cell's ``version.json`` say what its models.json requested."""
    c = copy(ALF_RB)
    path = next((c.cell(ALF_RB).attempt(1).root / "versions").glob("*/version.json"))
    document = json.loads(path.read_text())
    real_sha = protocol.fingerprint_manifest()["files"][protocol.MODELS_JSON_REL]
    assert document["model_slugs"] == {
        real_sha: protocol.canonical_slugs(protocol.load_models())
    }
    _models_json_edited(monkeypatch, luna=TURBO_SLUG)
    _serve_luna_as(c, TURBO_SLUG)
    result = c.audit(versions=_frozen_with_slugs(None))
    assert violations(result) == {"H10.3:model_served"}, result.to_markdown()
    _serve_luna_as(c, LUNA_SLUG)
    result = c.audit(versions=_frozen_with_slugs(None))
    assert "H10.3:model_served" not in violations(result) | warnings(result)


def _constant_edited(monkeypatch: pytest.MonkeyPatch, **edit: Any) -> None:
    """The auditing tree after an edit of protocol constants alone (``edit``), both as
    the manifest hashes them and as the module reads them."""
    real = protocol.fingerprint_manifest()
    constants = {**real["constants"], **edit}
    drifted = {
        **real,
        "constants": constants,
        "constants_sha256": protocol.constants_sha256(constants),
        "fingerprint": protocol.fingerprint_of(constants, real["files"]),
    }
    monkeypatch.setattr(protocol, "fingerprint_manifest", lambda *a, **kw: drifted)
    if "embedding_model" in edit:
        monkeypatch.setattr(protocol, "EMBEDDING_MODEL", edit["embedding_model"])
    if "roles" in edit:
        monkeypatch.setattr(protocol, "ROLES", tuple(edit["roles"]))


def test_a_changed_embedding_model_is_judged_against_the_cells_own_version(
    copy, monkeypatch
):
    """The embedding model and the roles are protocol constants each version
    records per key, so after this tree changes them a cell of the earlier version is
    still judged against its own values: nothing is listed as not re-verified, its own
    embedding calls pass, and a call that used the tree's new model is a violation."""
    c = copy(ALF_RB)
    real = protocol.fingerprint_manifest()["constants"]
    _constant_edited(
        monkeypatch,
        embedding_model="openai/text-embedding-9",
        roles=[r for r in real["roles"] if r != "judge"],
    )
    result = c.audit()
    listed = _not_reverified(result.cells[0])
    assert "H10.3:embedding_model" not in listed and "H10.3:role" not in listed
    assert violations(result) == set(), result.to_markdown()
    assert not recorded(result, "H10.3:embedding_model")
    assert not recorded(result, "H10.3:role")

    first = _first_call(c, ALF_RB, role="embedding")
    c.edit_ledger(
        ALF_RB, _set_on_call(first["call_id"], model="openai/text-embedding-9")
    )
    result = c.audit()
    assert "H10.3:embedding_model" in violations(result), result.to_markdown()
    (found,) = recorded(result, "H10.3:embedding_model")
    assert found.downgraded is None
    assert found.detail.endswith("not google/gemini-embedding-001")


def test_a_version_without_per_key_constants_still_downgrades_the_embedding_check(
    copy, monkeypatch
):
    """Only a version whose constants are not recorded per key is judged
    against this tree's embedding model, and then its findings are not re-verified."""
    c = copy(ALF_RB)
    known = protocol.known_versions()
    frozen = protocol.frozen_fingerprint()
    legacy = {
        **known,
        EARLIER: {
            **known[frozen],
            "fingerprint": EARLIER,
            "constants": None,
            "constants_sha256": "4" * 64,
            "current": False,
        },
    }
    _set_cell_protocol(c, ALF_RB, EARLIER)
    _constant_edited(monkeypatch, embedding_model="openai/text-embedding-9")
    result = c.audit(versions=legacy)
    assert "H10.3:embedding_model" in _not_reverified(result.cells[0])
    assert violations(result) == set(), result.to_markdown()
    found = recorded(result, "H10.3:embedding_model")
    assert found and all(x.downgraded == audit.NOT_REVERIFIED for x in found)


def test_keys_and_strings_the_audit_searches():
    received = {
        "first_messages": [
            {"role": "user", "content": [{"type": "text", "text": "q"}]}
        ],
        "messages": [
            {"role": "assistant", "content": "model text"},
            {"role": "tool", "tool_call_id": "1", "content": "document excerpt"},
            {"role": "user", "content": "data:image/png;base64,AAAA", "won": True},
        ],
    }
    assert audit.forbidden_keys(received) == ["$.messages[2].won"]
    assert [t for _, t in audit.harness_strings(received)] == [
        "user",
        "text",
        "q",
        "user",
    ]
    placeholder = audit.blob_placeholders({"url": {"$blob": "ab"}})
    assert placeholder == {"url": "data:application/x-skill-bench-blob;sha256,ab"}


def test_the_memory_slot_of_each_benchmark(e2e_matrix):
    memory = "## Lessons\n- look first"
    # ALFWorld: the start of the user prompt.
    env = e2e_matrix.envs["alfworld"]
    fields = {
        "observation": "obs",
        "admissible_actions": ["look"],
        "history": [],
        "task_description": "t",
    }
    with_memory = env.actor_messages(fields, memory)
    stripped, why = audit.strip_memory("alfworld", with_memory, memory)
    assert why is None and stripped == env.actor_messages(fields, "")
    moved = [
        with_memory[0],
        {**with_memory[1], "content": "x" + with_memory[1]["content"]},
    ]
    assert audit.strip_memory("alfworld", moved, memory)[1] is not None
    assert audit.strip_memory("webarena", with_memory, memory)[1] is not None


# --- category labels and gold answers in method outputs --------------------------------------


def test_a_memory_may_use_the_words_of_alfworld_task_types(copy):
    """'pick_and_place_simple' is a category shared by many items, not another
    item's identifier; an identifier of another item in the memory is still a leak."""
    c = copy(ALF_RB)
    _rewrite_snapshot(
        c.cell(ALF_RB),
        lambda s: s.__setitem__(
            "notes", "- For pick_and_place_simple tasks, look first."
        ),
    )
    assert violations(c.audit()) == set()
    env = c.matrix.envs["alfworld"]
    other = env.load_tasks("train")[-1].raw["gamefile"]
    _rewrite_snapshot(
        c.cell(ALF_RB),
        lambda s: s.__setitem__("notes", s["notes"] + f"\n- see {other}"),
    )
    assert violations(c.audit()) == {"H10.1:method_output_label"}


def _test_task(c: Copy, cell_id: str, index: int = 0) -> Any:
    """The item of the cell's ``index``-th test row (a smoke cell tests val items)."""
    raw_id = c.rows(cell_id, "test")[index]["raw_id"]
    env = c.matrix.env_factory(cell_id.split("-")[0])
    return next(t for t in env.load_tasks("val") if t.raw_id == raw_id)


def test_a_method_that_read_a_test_items_gold_answer(copy):
    """The frozen memory holds a test item's gold answer that no learning input
    held: the method read the scorer's records."""
    c = copy(WS_AWM)
    task = _test_task(c, WS_AWM)
    answer = str(task.raw["asin"])
    _rewrite_snapshot(
        c.cell(WS_AWM), lambda s: s.__setitem__("notes", f"remember {answer}")
    )
    assert violations(c.audit()) == {"H10.1:answer_in_method_output"}


def test_a_gold_answer_injected_for_its_own_test_item(copy):
    """A method that adds an item's gold answer to that item's injection only (read
    from the scorer's records at before_task) is caught although its memory is clean."""
    c = copy(WS_AWM)
    task = _test_task(c, WS_AWM)
    attempt = c.cell(WS_AWM).attempt(1)
    rows = jsonl(attempt.injections)
    for row in rows:
        if row["phase"] == "test" and row["raw_id"] == task.raw_id:
            row["text"] += f"\nHint: the expected answer is {task.raw['asin']}."
    write_jsonl(attempt.injections, rows)
    assert violations(c.audit()) == {"H10.1:answer_in_method_output"}


def test_a_short_answer_in_the_shared_memory_is_only_a_warning(copy, monkeypatch):
    """One memory serves every test item, so a short number in it may equal some
    item's answer by coincidence: a warning, not a violation."""
    c = copy(WS_AWM)
    task = _test_task(c, WS_AWM)
    real = audit._answers
    monkeypatch.setattr(
        audit,
        "_answers",
        lambda t: {"7349"} if t.raw_id == task.raw_id else real(t),
    )
    assert audit.specific_answer("7349") and not audit.strong_answer("7349")
    assert audit.strong_answer("256,368") and audit.strong_answer("ab-1234")
    _rewrite_snapshot(
        c.cell(WS_AWM), lambda s: s.__setitem__("notes", "a table of 7349 rows")
    )
    result = c.audit()
    assert violations(result) == set()
    assert "H10.1:answer_in_method_output" in warnings(result)


# --- text gold answers ----------------------------------------------------------------------


def _inject_for_item(c: Copy, cell_id: str, raw_id: str, text: str) -> None:
    """Append ``text`` to the test-phase injection of ``raw_id`` only (a method that read
    the item's gold answer at ``before_task``)."""
    attempt = c.cell(cell_id).attempt(1)
    rows = jsonl(attempt.injections)
    hit = False
    for row in rows:
        if row["phase"] == "test" and row["raw_id"] == raw_id:
            row["text"] += text
            hit = True
    assert hit, f"{raw_id} has no test-phase injection"
    write_jsonl(attempt.injections, rows)


def test_text_answer_tiers():
    """Which text answers are searched for, and how telling each is."""
    tier = audit.text_answer_tier
    for strong in (
        "scissors menthol fresh",
        "perkin elmer corp",
        "principal investigator",
        "the coca-cola company",
        "bradford hill",
    ):
        assert tier(strong) == audit.TEXT_STRONG, strong
    for weak in (
        "nashville",
        "dematerialisation",
        "university of maine",
        "tobacco company",
        "new york",
    ):
        assert tier(weak) == audit.TEXT_WEAK, weak
    assert audit.word_text("Scissors-Menthol\n Fresh!") == "scissors menthol fresh"
    for skipped in (
        "yes",
        "no",
        "winston",  # one word under 8 characters
        "four weeks",  # common words only
        "final approval",
        "confidential",
        "the total amount",
        "n i l",
        "",
    ):
        assert tier(skipped) is None, skipped
    # Answers with a digit or an identifier mark are specific_answer's.
    assert tier("256,368") is None and tier("form 10-k") is None
    assert audit.specific_answer("form 10-k")


def test_a_one_word_text_answer_in_its_injection_is_a_warning(copy, monkeypatch):
    """A single content word may be template text: a warning; common words only are
    not searched at all."""
    c = copy(WS_RB)
    raw_id = c.rows(WS_RB, "test")[0]["raw_id"]
    real = audit._answers
    answer = {"text": "nashville"}
    monkeypatch.setattr(
        audit,
        "_answers",
        lambda t: {answer["text"]} if t.raw_id == raw_id else real(t),
    )
    _inject_for_item(c, WS_RB, raw_id, "\nSee the Nashville office. Four weeks.")
    result = c.audit()
    assert violations(result) == set()
    assert "H10.1:answer_in_method_output" in warnings(result)
    answer["text"] = "four weeks"
    result = c.audit()
    assert "H10.1:answer_in_method_output" not in warnings(result) | violations(result)


def test_an_answer_after_a_line_break_in_the_memory_is_found(copy):
    """The memory is searched string by string too. In its JSON a value that follows
    a line break reads ``\\n256,368``, which hides the number's word boundary."""
    c = copy(WS_AWM)
    answer = str(_test_task(c, WS_AWM).raw["asin"])
    _rewrite_snapshot(
        c.cell(WS_AWM), lambda s: s.__setitem__("notes", f"remember:\n{answer}")
    )
    assert violations(c.audit()) == {"H10.1:answer_in_method_output"}


def test_text_answers_are_searched_without_the_common_words_around_them():
    """An article, a title or a common noun around a text answer can
    neither hide it nor explain it away; the core must still pass the tier rule."""
    needle = audit.text_answer_needle
    assert needle("the coca-cola company") == ("coca cola", audit.TEXT_STRONG)
    assert needle("miss kendra ellis") == ("kendra ellis", audit.TEXT_STRONG)
    assert needle("perkin elmer corp") == ("perkin elmer", audit.TEXT_STRONG)
    assert needle("scissors menthol fresh") == (
        "scissors menthol fresh",
        audit.TEXT_STRONG,
    )
    assert needle("tea and other retail products") == (
        "tea and other retail",
        audit.TEXT_STRONG,
    )
    # A one-word core says less than the phrase, and a core that loses the tier is not
    # the answer's equal: the whole answer is searched.
    assert needle("the cocacola company") == ("the cocacola company", audit.TEXT_WEAK)
    assert needle("university of maine") == ("university of maine", audit.TEXT_WEAK)
    assert needle("washington university") == (
        "washington university",
        audit.TEXT_WEAK,
    )
    assert needle("new york") == ("new york", audit.TEXT_WEAK)
    assert needle("tobacco company") == ("tobacco company", audit.TEXT_WEAK)
    assert needle("the abc def") == ("the abc def", audit.TEXT_STRONG)
    for skipped in ("four weeks", "the total amount", "yes", "256,368", "form 10-k"):
        assert needle(skipped) is None, skipped


def test_numbers_are_also_compared_without_separators_and_currency_signs():
    """A numeric gold answer written without its thousands separator,
    with a space for it, or without its currency sign is still the answer."""

    def form(text: str) -> str:
        return audit.digit_form(audit.normalise(text))

    for written in ("256865", "256,865", "256 865", "$256,865", "USD 256,865"):
        assert "256865" in form(written).split(), written
    assert form("1,234,567") == "1234567" and form("1\u2019000") == "1000"
    assert form("\u20ac 500.00") == "500.00" and form("us$500.00") == "us 500.00"
    # Not a thousands group: kept apart.
    assert form("12,34") == "12,34" and form("in 2019 2020") == "in 2019 2020"
    texts = audit.Texts()
    texts.add("Answer: 256865. Paid US$500.00 in 1998.", "-", "-")
    for gold in ("256,865", "256 865", "$500.00", "500.00", "1998"):
        assert texts.holds(gold, textual=False), gold
    for other in ("56,865", "2,568", "19,98", "865"):
        assert not texts.holds(other, textual=False), other


def test_a_gold_answer_written_another_way_in_its_injection(
    e2e_matrix, tmp_path, monkeypatch
):
    """A text answer without its leading article or
    title, and a number without its separator or currency sign, injected for the item
    alone, are caught as the answer written as it is."""
    real = audit._answers
    raw_id = jsonl(e2e_matrix.cell(WS_RB).attempt(1).results)[0]["raw_id"]
    for case, (gold, written) in enumerate(
        (
            ("the coca-cola company", "\nAnswer: Coca-Cola Company"),
            ("miss kendra ellis", "\nAddressee: Kendra Ellis."),
            ("256,865", "\nAnswer: 256865"),
            ("256,865", "\nAnswer: 256 865"),
            ("$500.00", "\nAnswer: 500.00"),
            ("256865", "\nAnswer: 256,865"),
        )
    ):
        c = Copy(e2e_matrix, tmp_path / str(case), WS_RB)
        monkeypatch.setattr(
            audit,
            "_answers",
            lambda t, gold=gold: {gold} if t.raw_id == raw_id else real(t),
        )
        assert violations(c.audit()) == set(), gold
        _inject_for_item(c, WS_RB, raw_id, written)
        result = c.audit()
        assert violations(result) == {"H10.1:answer_in_method_output"}, gold
        assert all(
            f.item == raw_id
            for cell in result.cells
            for f in cell.findings
            if f.check == "answer_in_method_output"
        )


def test_how_many_text_answers_a_memory_may_share_by_coincidence():
    allowed = audit.memory_text_answers_allowed
    assert allowed(350) == 7 and allowed(172) == 3 and allowed(3) == 3
    assert allowed(1000) == 20


def test_many_text_gold_answers_in_the_shared_memory_are_a_violation(copy, monkeypatch):
    """A memory holding the telling text answers of
    a few test items draws warnings, as documents of one source share names; holding
    those of more test items than ``memory_text_answers_allowed`` is a violation, since
    no coincidence explains that many."""
    c = copy(WS_RB)
    test_items = sorted(read_json(c.cell(WS_RB).manifest)["cell"]["test_items"])
    golds = dict(
        zip(test_items, ("zanzibar quokka", "marmoset tangerine", "obsidian pelican"))
    )
    assert len(golds) == 3
    real = audit._answers
    monkeypatch.setattr(
        audit,
        "_answers",
        lambda t: {golds[t.raw_id]} if t.raw_id in golds else real(t),
    )
    _rewrite_snapshot(
        c.cell(WS_RB),
        lambda s: s.__setitem__(
            "notes", "- Names: " + "; ".join(v.title() for v in golds.values())
        ),
    )
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.1:answer_in_method_output" in warnings(result)
    info = result.cells[0].info
    assert info["gold_answers_in_memory"] == {
        "specific": 0,
        "strong_text": 3,
        "strong_text_allowed": 3,
    }
    monkeypatch.setattr(audit, "MEMORY_TEXT_ANSWERS_MIN", 2)
    result = c.audit()
    assert violations(result) == {"H10.1:answer_in_method_output"}
    found = [
        f
        for cell in result.cells
        for f in cell.findings
        if f.check == "answer_in_method_output" and f.severity == audit.VIOLATION
    ]
    assert {f.item for f in found} == set(golds)
    assert all("3 test items, more than the 2" in f.detail for f in found)


# --- H10.1: gold answers in the frozen memory ------------------------------------------------


def test_how_much_a_specific_answer_in_the_frozen_memory_weighs():
    """H10.1: in the frozen memory a specific answer is a violation only when no
    memory holds it by coincidence (5+ digits, or a path/key/address mark); a name
    specific only through its punctuation, and a short number in a telling phrase, go
    through the count rule of the text answers; other short ones are warnings."""
    strong, counted = audit.strong_in_memory, audit.counted_in_memory
    for needle in (
        "256,865",
        "(555) 253 3012",
        "march 27, 1979",
        "3/31/75",
        "$85,000.00",
        "robotics/analytical chemist",
        "jdoe@example.com",
        # A short number with a word no label, unit or common word explains is a code
        # or an address.
        "ma 1253 ag",
        "arl va 155",
        "5954 b",
        "1455 broad street",
    ):
        assert strong(needle) and not counted(needle), needle
    # Judged on the digits, as the memory is searched in digit_form too: '$1,000' is
    # strong_answer (6 characters), yet a bare '1000' in the memory holds it.
    for needle in (
        "$1,000",
        "$ 1795",
        "$500.00",
        "200.00",
        "7349",
        "step 2",
        "page 14",
        "figure 2",
        "schedule 11",
        "attachment 3",
        "december 7",
        "2 or more",
        "1/16 tsp.",
        "2:30 pm",
        "2:30 p.m.",
        "march 21st",
        "10 fl oz",
        # A count of servings or containers is what any memory may write.
        "5 to 6 glasses",
        "2 cans",
        # Common units and count nouns are amounts, not codes.
        "100 dollars",
        "25 people",
        "15 miles",
        "5 points",
        "75 degrees",
        "8 grams",
        "250 pounds",
        "400 meters",
        "4 persons",
        "50 acres",
        "20 employees",
        "1,250 tons",
        "3.5 million",
        "12 units",
    ):
        assert audit.specific_answer(needle), needle
        assert not strong(needle) and not counted(needle), needle
    for needle in (
        "r.j. reynolds",
        "pathmark stores inc.",
        "johnson & johnson and subsidiaries",
        "washington, d.c.",
        "american cheese production by states, 1952",
    ):
        assert not strong(needle) and counted(needle), needle
    assert audit.strong_answer("$1,000") and audit.strong_answer("r.j. reynolds")
    digits = audit.answer_digits
    assert digits("$1,000") == 4 and digits("$500.00") == 3 and digits("4.4200") == 5
    # Not specific: neither (text answers have tiers of their own).
    assert not counted("nashville") and not strong("nashville")


@pytest.mark.parametrize(
    ("gold", "memory", "severity"),
    [
        ("$1,000", "a fee of 1000 per page", audit.WARNING),
        ("$7,349", "a table of 7349 rows", audit.WARNING),
        ("$ 1795", "filed 1795 units", audit.WARNING),
        # Not step 2: the agent context names the user message of step 2.
        ("step 3", "Step 3: read the table", audit.WARNING),
        ("r.j. reynolds", "sender: R.J. Reynolds", audit.WARNING),
        ("256,865", "remember 256865", audit.VIOLATION),
        ("ma 1253 ag", "the form code is MA 1253 AG", audit.VIOLATION),
        (
            "robotics/analytical chemist",
            "a robotics/analytical chemist",
            audit.VIOLATION,
        ),
    ],
)
def test_a_specific_answer_in_the_frozen_memory(
    copy, monkeypatch, gold, memory, severity
):
    """H10.1: the memory tiers on a cell (WebShop): the finding says no learning
    input held the answer."""
    c = copy(WS_AWM)
    task = _test_task(c, WS_AWM)
    real = audit._answers
    monkeypatch.setattr(
        audit, "_answers", lambda t: {gold} if t.raw_id == task.raw_id else real(t)
    )
    _rewrite_snapshot(c.cell(WS_AWM), lambda s: s.__setitem__("notes", memory))
    result = c.audit()
    # Only this item's gold is patched: another item's own gold is reported for that item.
    found = [
        f
        for cell in result.cells
        for f in cell.findings
        if f.check == "answer_in_method_output" and f.item == task.raw_id
    ]
    assert [f.severity for f in found] == [severity], found
    assert found[0].detail.endswith(
        "which no learning input held"
        if severity == audit.VIOLATION
        else (
            "which no learning input held (a name the learning pages may have shown)"
            if gold == "r.j. reynolds"
            else "which no learning input held (short: possibly a coincidence)"
        )
    )
    assert violations(result) == (
        {"H10.1:answer_in_method_output"} if severity == audit.VIOLATION else set()
    )
    if gold == "r.j. reynolds":
        # The count rule of the text answers, not the specific answers' tier: past the
        # names a memory may share by coincidence it is a violation.
        info = result.cells[0].info["gold_answers_in_memory"]
        assert info["specific"] == 0 and info["strong_text"] == 1
        monkeypatch.setattr(audit, "MEMORY_TEXT_ANSWERS_MIN", 0)
        assert violations(c.audit()) == {"H10.1:answer_in_method_output"}


def test_later_actor_calls_are_checked_against_the_ledger(copy):
    """Calls after the first are checked too: their memory slot, and their recorded
    request against the hash of what the gateway forwarded."""
    c = copy(ALF_RB)
    path = c.longest(ALF_RB, "test")
    data = json.loads(path.read_text())
    memory = data["injection"]["text"]
    assert memory and len(data["calls"]) > 3

    def swap(request: dict[str, Any]) -> None:
        for message in request["messages"]:
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].replace(memory, "OTHER MEMORY")

    def edit(d: dict[str, Any]) -> None:
        swap(d["calls"][2]["request"])

    c.edit(path, edit)
    assert violations(c.audit()) == {
        "H10.2:memory_slot",
        "H10.2:request_not_as_sent",
    }
    # Records made consistent (as if the actor had really sent it): the slot is still
    # checked on every call.
    request = json.loads(path.read_text())["calls"][2]
    c.edit_ledger(
        ALF_RB,
        lambda rows: [
            row.__setitem__(
                "request_sha256",
                audit.sent_request_sha256(request["request"], row["wire"]),
            )
            for row in rows
            if row.get("event") == "call"
            and row.get("call_id") == request["gateway_call_id"]
        ],
    )
    assert violations(c.audit()) == {"H10.2:memory_slot"}


def _add_invalid_attempt(
    c: Copy, cell_id: str, *, scored: bool, invalidation: str, previous: str
) -> None:
    """An invalidated attempt next to the audited one (attempt-02 of the copy)."""
    paths = c.cell(cell_id)
    manifest = read_json(paths.manifest)
    instance = manifest["instance"]
    attempt = paths.attempt(2)
    attempt.root.mkdir(parents=True)
    run_id = f"{cell_id}-a02-{instance}"
    write_json_atomic(
        attempt.state,
        {
            "attempt": 2,
            "run_id": run_id,
            "instance": instance,
            "state": "invalid",
            "previous_state": previous,
            "invalidation": invalidation,
            "invalid_reason": "the operator asked for a new attempt",
            "ledger": str(c.runs / run_id / "gateway.jsonl"),
        },
    )
    rows = c.rows(cell_id, "test")[:1] if scored else []
    write_jsonl(attempt.results, rows)


def test_a_tested_attempt_replaced_by_a_new_one(copy):
    """An invalidated attempt that had scored test items means the cell ran twice
    (H1.3); a re-roll before any score is only a warning, as is a replaced violation."""
    c = copy(ALF_RB)
    _add_invalid_attempt(
        c, ALF_RB, scored=True, invalidation="operator_reroll", previous="testing"
    )
    result = c.audit()
    # H1.3 as recorded: listed, never mixed into the scores.
    assert violations(result) == set()
    assert "H10.3:tested_attempt_replaced" in warnings(result)
    (listed,) = result.cells[0].info["superseded_attempts"]
    assert (listed["attempt"], listed["superseded"], listed["scored"]) == (2, False, 1)

    c2 = Copy(c.matrix, c.runs.parent / "second", ALF_RB)
    _add_invalid_attempt(
        c2, ALF_RB, scored=False, invalidation="operator_reroll", previous="violation"
    )
    result = c2.audit()
    assert violations(result) == set()
    assert {"H10.3:operator_reroll", "H10.4:violation_history"} <= warnings(result)


def test_an_item_scored_twice_or_re_run_without_a_failure(copy):
    """One scored row per test item; a re-run replaces only a row that failed on
    infrastructure or the harness, and attempt.json lists it."""
    c = copy(ALF_RB)
    attempt = c.cell(ALF_RB).attempt(1)
    rows = jsonl(attempt.results)
    write_jsonl(attempt.results, [rows[0], *rows])
    assert violations(c.audit()) == {"H10.3:scored_twice"}

    c2 = Copy(c.matrix, c.runs.parent / "second", ALF_RB)
    attempt = c2.cell(ALF_RB).attempt(1)
    rows = jsonl(attempt.results)
    lost = {**rows[0], "score": None, "failure": {"kind": "model", "code": "x"}}
    write_jsonl(attempt.results, [lost, *rows])
    result = c2.audit()
    assert violations(result) == {"H10.3:rerun_without_failure"}
    assert "H10.3:rerun_unlisted" in warnings(result)
    # A row that failed on infrastructure, listed as a re-run: H8's legitimate re-run.
    lost["failure"] = {"kind": "infra", "code": "upstream_infra"}
    write_jsonl(attempt.results, [lost, *rows])
    c2.add_rerun(ALF_RB, rows[0]["task_uid"])
    state = read_json(attempt.state)
    state["test_reruns"][-1]["raw_id"] = rows[0]["raw_id"]
    write_json_atomic(attempt.state, state)
    assert violations(c2.audit()) == set()


def test_an_attempt_run_id_without_the_cell_instance(copy):
    """Attempt N runs as <cell>-aNN-<instance>; another run id may hold another
    cell's calls."""
    c = copy(ALF_RB)
    c.edit(
        c.cell(ALF_RB).attempt(1).state,
        lambda s: s.__setitem__("instance", "0badc0de"),
    )
    result = c.audit()
    assert "H10.3:run_id" in warnings(result) and violations(result) == set()


# --- the cell ran once, across cell directories, and no-op invocations -----------------------


def _run_again(c: Copy, cell_id: str, **options: Any) -> CellOutcome:
    """run.py's runner on a copied cell, attached to a gateway over the copy's runs
    directory (the fake model answers as it did when the matrix was built)."""
    spec = (
        PILOT
        if cell_id == WS_PILOT
        else next(
            cell_spec(b, m)
            for b in protocol.BENCHMARKS
            for m in ("none", "reasoningbank", "awm")
            if cell_spec(b, m).cell_id == cell_id
        )
    )
    fake = FakeUpstream(responder=c.matrix.model)
    with serve_gateway(
        c.runs.parent, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
    ) as (_gw, url):
        runner = CellRunner(
            spec,
            RunOptions(
                out_dir=c.runs / cell_id,
                gateway_url=url,
                gateway_runs_dir=c.runs,
                preflight=False,
                blob_dir=c.matrix.runs / "_blobs",
                **options,
            ),
            env_factory=lambda _benchmark: c.matrix.envs[spec.benchmark],
        )
        outcome = asyncio.run(runner.run())
    assert not fake.calls("chat/completions"), "a no-op invocation called the model"
    return outcome


def test_a_complete_cell_invoked_again_after_a_code_change_still_passes(
    copy, monkeypatch
):
    """run.py invoked again on a complete cell after the harness code changed (one way to
    restore a cell's files) ran nothing: the audit still passes. So does a
    refused --new-attempt."""
    c = copy(ALF_NONE)
    changed = {**protocol.code_fingerprint(), "harness": "f" * 64}
    monkeypatch.setattr(protocol, "code_fingerprint", lambda **_kw: dict(changed))
    again = _run_again(c, ALF_NONE)
    assert (again.status, again.detail) == ("complete", "already complete")
    refused = _run_again(c, ALF_NONE, new_attempt=True)
    assert refused.status == "refused"
    events = [r.get("event") for r in jsonl(c.cell(ALF_NONE).launches)]
    assert events == ["start", "work", "start", "start"]
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()


def _earlier_run(
    c: Copy, cell_id: str, instance: str, *, tested: bool, attempt: int = 1
) -> str:
    """A gateway run of ``cell_id`` from another cell directory (instance ``instance``):
    this cell's own ledger, with its test phase or cut before it."""
    rows = jsonl(c.ledger(cell_id))
    if not tested:
        rows = [r for r in rows if (r.get("run_phase") or r.get("phase")) != "test"]
    run_id = f"{cell_id}-a{attempt:02d}-{instance}"
    (c.runs / run_id).mkdir()
    write_jsonl(c.runs / run_id / "gateway.jsonl", rows)
    return run_id


#: WebShop's pilot (the no-memory method on the pilot split): a cell outside
#: smoke mode, which H1.3 runs once.
PILOT = CellSpec("webshop", "luna", "none", "pilot")
WS_PILOT = PILOT.cell_id


@pytest.fixture(scope="module")
def pilot_matrix(
    e2e_matrix: Matrix, tmp_path_factory: pytest.TempPathFactory
) -> Matrix:
    """WebShop's pilot cell through run.py and the protocol gateway, on the matrix's
    scripted shop."""
    root = tmp_path_factory.mktemp("pilot")
    env = e2e_matrix.envs["webshop"]
    model = E2EModel()
    fake = FakeUpstream(responder=model)
    runs = root / "runs"
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        outcomes = asyncio.run(run_cells(runs, url, {"webshop": env}, [PILOT]))
    assert {o.status for o in outcomes.values()} == {"complete"}, outcomes
    return Matrix(root, runs, {"webshop": env}, outcomes, model, fake)


@pytest.fixture
def pilot(pilot_matrix: Matrix, tmp_path: Path) -> Callable[..., Copy]:
    return _copies(pilot_matrix, tmp_path)


def test_a_cell_run_again_from_a_new_directory_after_its_test_phase(pilot):
    """The cell directory was moved away after its test phase and
    the cell run again from scratch; the new directory has a clean record, but the old
    instance's runs stay in the gateway runs directory. A pilot cell (H1.3)."""
    c = pilot(WS_PILOT)
    assert read_json(c.cell(WS_PILOT).manifest)["cell"]["split_mode"] == "pilot"
    _earlier_run(c, WS_PILOT, "0dd0dd01", tested=True)
    result = c.audit()
    # H1.3 as recorded: listed for the report, not failed.
    assert violations(result) == set()
    assert "H10.3:cell_rerun_after_test" in warnings(result)
    (other,) = result.cells[0].info["other_gateway_runs"]
    assert other["instance"] == "0dd0dd01" and other["test_actor_calls"] > 0
    assert audit._ts(other["first_test_actor_at"]) is not None


def test_a_smoke_cell_run_again_is_a_warning(copy):
    """H1.3 runs each protocol cell once. A smoke cell (it tests val items) is outside
    that matrix and may be run again after a fix: an earlier directory's tested run is
    listed as a warning, with no operator note needed."""
    c = copy(ALF_NONE)
    assert read_json(c.cell(ALF_NONE).manifest)["cell"]["split_mode"] == "smoke"
    _earlier_run(c, ALF_NONE, "0dd0dd01", tested=True)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:smoke_cell_rerun" in warnings(result)
    assert "H10.3:cell_rerun_after_test" not in warnings(result)


def test_a_cell_started_over_before_its_test_phase_is_a_warning(copy):
    """An earlier directory that never ran a test item: started over before any test
    score could be seen, like an operator re-roll."""
    c = copy(ALF_NONE)
    _earlier_run(c, ALF_NONE, "0dd0dd01", tested=False)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:cell_restarted" in warnings(result)


def test_a_tested_run_of_this_directory_no_attempt_records(copy):
    """A run of the cell directory's own instance that no attempt.json names and that ran
    test items: an attempt's records were removed. One that ran nothing is an attempt
    that never started, and is not reported."""
    c = copy(ALF_NONE)
    instance = read_json(c.cell(ALF_NONE).manifest)["instance"]
    never_started = f"{ALF_NONE}-a02-{instance}"
    (c.runs / never_started).mkdir()
    write_jsonl(
        c.runs / never_started / "gateway.jsonl",
        [{"event": "phase", "ts": "2026-09-28T00:00:00+00:00", "phase": "setup"}],
    )
    assert violations(c.audit()) == set()
    _earlier_run(c, ALF_NONE, instance, tested=True, attempt=3)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:unrecorded_attempt" in warnings(result)


def _own_first_test_call(c: Copy, cell_id: str) -> datetime:
    """When the cell directory's own run made its first test-phase actor call."""
    (own,) = [
        r for r in cell_gateway_runs(c.runs, cell_id) if r.run_id == c.run_id(cell_id)
    ]
    return _at(own.first_test_actor_at)


def _record_note(c: Copy, cell_id: str, note: Path, at: str) -> dict[str, Any]:
    """An operator note in cell.json as run.py --acknowledge-rerun records it (copied
    into operator_notes/, with its sha256 and the earlier runs as they are), at ``at``:
    run.py records one only before the directory's test phase, which a copied complete
    cell is past."""
    paths = c.cell(cell_id)
    data = note.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    dest = paths.operator_notes / f"{digest[:12]}-{note.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest = read_json(paths.manifest)
    earlier = [
        r.to_dict()
        for r in cell_gateway_runs(c.runs, cell_id)
        if r.instance != manifest["instance"]
    ]
    entry = {
        "kind": NOTE_RERUN_ACKNOWLEDGED,
        "file": dest.relative_to(paths.root).as_posix(),
        "sha256": digest,
        "source": str(note),
        "at": at,
        "launch": "test-launch",
        "superseded": earlier,
    }
    manifest["operator_notes"] = [*manifest.get("operator_notes", []), entry]
    write_json_atomic(paths.manifest, manifest)
    return entry


def _set_note_at(c: Copy, cell_id: str, at: str | None) -> None:
    manifest = read_json(c.cell(cell_id).manifest)
    manifest["operator_notes"][0]["at"] = at
    write_json_atomic(c.cell(cell_id).manifest, manifest)


def test_an_acknowledged_rerun_is_a_warning(pilot, tmp_path):
    """An operator note in cell.json acknowledges the earlier runs it lists: a
    warning. It acknowledges nothing when it was recorded once this directory's own test
    scores could be seen (a note could then keep the better run; run.py records one
    with after_test), when its copy was changed, or for an earlier run that ran more test
    items after it."""
    c = pilot(WS_PILOT)
    earlier = _earlier_run(c, WS_PILOT, "0dd0dd01", tested=True)
    note = tmp_path / "rerun-note.md"
    note.write_text("2026-09-28, the operator: the first run used a broken sandbox.\n")
    # run.py on the complete cell: its test phase is over, so the note is too late.
    # run.py records it with after_test (recorded, not refused): a warning in the
    # audit, never an acknowledgement.
    late = _run_again(c, WS_PILOT, acknowledge_rerun=note)
    assert late.status != "refused", late.detail
    (late_note,) = (read_json(c.cell(WS_PILOT).manifest) or {})["operator_notes"]
    assert late_note["after_test"] is True and late_note["tested_runs"]
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert {
        "H10.3:operator_note_after_test",
        "H10.3:rerun_acknowledged_after_test",
    } <= warnings(result)
    assert "H10.3:cell_rerun_acknowledged" not in warnings(result)
    c.edit(c.cell(WS_PILOT).manifest, lambda m: m.pop("operator_notes"))
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:cell_rerun_after_test" in warnings(result)

    first = _own_first_test_call(c, WS_PILOT)
    entry = _record_note(c, WS_PILOT, note, (first - timedelta(minutes=5)).isoformat())
    assert [r["run_id"] for r in entry["superseded"]] == [earlier]
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:cell_rerun_acknowledged" in warnings(result)

    # Recorded when, or after, this directory's first test-phase actor call: both runs'
    # scores could be seen. A note without a time is no better.
    for at in (first.isoformat(), (first + timedelta(hours=1)).isoformat(), None):
        _set_note_at(c, WS_PILOT, at)
        result = c.audit()
        assert violations(result) == set(), at
        assert "H10.3:rerun_acknowledged_after_test" in warnings(result), at
        assert "H10.3:cell_rerun_acknowledged" not in warnings(result)
    _set_note_at(c, WS_PILOT, entry["at"])
    assert violations(c.audit()) == set()

    copied = c.cell(WS_PILOT).root / entry["file"]
    copied.write_text(copied.read_text() + "edited later\n")
    result = c.audit()
    assert violations(result) == set()
    assert {
        "H10.3:cell_rerun_after_test",
        "H10.3:operator_note_invalid",
    } <= warnings(result)
    copied.write_text(note.read_text())
    assert violations(c.audit()) == set()

    rows = jsonl(c.runs / earlier / "gateway.jsonl")
    extra = next(
        r
        for r in rows
        if r.get("event") == "call"
        and r.get("role") == "actor"
        and r.get("run_phase") == "test"
    )
    write_jsonl(c.runs / earlier / "gateway.jsonl", [*rows, {**extra, "call_id": "x"}])
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:cell_rerun_after_test" in warnings(result)
    assert any(
        "ran more test items after the note" in f.detail
        for cell in result.cells
        for f in cell.findings
    )


def test_a_rerun_run_py_acknowledged_before_its_test_phase(pilot, tmp_path):
    """End to end: the old directory moved out of the runs directory, the cell run
    again by run.py with --acknowledge-rerun on the invocation that starts it. The note
    is recorded before the new directory's first test-phase actor call, so the audit
    lists the old run as a warning and the cell passes."""
    c = pilot(WS_PILOT)
    old_run = c.run_id(WS_PILOT)
    c.cell(WS_PILOT).root.rename(tmp_path / "runs-superseded")
    note = tmp_path / "rerun-note.md"
    note.write_text("2026-09-28, the operator: re-run after the sandbox fix.\n")
    fake = FakeUpstream(responder=c.matrix.model)
    with serve_gateway(
        c.runs.parent, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
    ) as (_gw, url):
        runner = CellRunner(
            PILOT,
            RunOptions(
                out_dir=c.runs / WS_PILOT,
                gateway_url=url,
                gateway_runs_dir=c.runs,
                preflight=False,
                idle_timeout_s=60.0,
                test_concurrency=2,
                acknowledge_rerun=note,
            ),
            env_factory=lambda _benchmark: c.matrix.envs["webshop"],
        )
        outcome = asyncio.run(runner.run())
    assert outcome.status == "complete", outcome
    (entry,) = read_json(c.cell(WS_PILOT).manifest)["operator_notes"]
    assert [r["run_id"] for r in entry["superseded"]] == [old_run]
    assert _at(entry["at"]) < _own_first_test_call(c, WS_PILOT)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:cell_rerun_acknowledged" in warnings(result)


def test_a_cell_directory_renamed_inside_the_runs_directory(copy, tmp_path):
    """The old directory renamed in place
    (runs/<cell>.old) is still a cell directory of the runs directory. The audit reads
    the latest-started complete one, lists the other under ``duplicates`` as superseded
    and says so across cells (a warning, naming the cell). Moved out of the runs
    directory, with its gateway runs left in place, only the re-run is reported."""
    c = copy(ALF_NONE)
    new = c.cell(ALF_NONE).root
    old = c.runs / f"{ALF_NONE}.old"
    shutil.copytree(new, old)
    run_old = _earlier_run(c, ALF_NONE, "0dd0dd01", tested=True)
    manifest = read_json(old / "cell.json")
    manifest["instance"] = "0dd0dd01"
    write_json_atomic(old / "cell.json", manifest)
    state = read_json(old / "attempt-01" / "attempt.json")
    state.update(
        instance="0dd0dd01",
        run_id=run_old,
        ledger=str(c.runs / run_old / "gateway.jsonl"),
        started_at="2026-01-01T00:00:00+00:00",
    )
    write_json_atomic(old / "attempt-01" / "attempt.json", state)
    result = c.audit()
    assert [x.root.rsplit("/", 1)[-1] for x in result.cells] == [new.name]
    (duplicate,) = [
        f for f in result.findings if f.check == "duplicate_cell_directories"
    ]
    assert duplicate.severity == audit.WARNING and duplicate.invariant == "H10.3"
    assert duplicate.downgraded == audit.RECORD and duplicate.cells == (ALF_NONE,)
    assert old.name in duplicate.detail and f"reads {new.name}" in duplicate.detail
    assert [
        (Path(d.root).name, d.role, Path(d.chosen_root).name, d.complete)
        for d in result.duplicates
    ] == [(old.name, "superseded", new.name, True)]
    assert violations(result) == set(), result.to_markdown()
    assert result.exit_code == 0
    data = result.to_dict()
    assert data["duplicates"][0]["started_at"] == "2026-01-01T00:00:00+00:00"
    assert "## Duplicate cell directories" in result.to_markdown()

    old.rename(tmp_path / "runs-superseded")
    result = c.audit()
    assert [x.root.rsplit("/", 1)[-1] for x in result.cells] == [new.name]
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:smoke_cell_rerun" in warnings(result)


def _work_line(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (work,) = [r for r in rows if r.get("event") == "work"]
    return work


def test_model_visible_code_recorded_per_launch_and_across_cells(copy):
    """One harness code per attempt and per benchmark; an attempt that recorded none
    cannot be compared."""
    c = copy(ALF_RB, ALF_NONE)
    launches = c.cell(ALF_RB).launches
    rows = jsonl(launches)
    work = _work_line(rows)
    # A later invocation resumed the test phase of attempt 1 on other code.
    later = {
        **work,
        "at": "2999-01-01T00:00:00+00:00",
        "launch": "0123456789ab",
        "phase": "test",
        "code": {**work["code"], "harness": "f" * 64},
    }
    write_jsonl(launches, [*rows, later])
    result = c.audit()
    # Recorded: no version segment records the other code.
    assert violations(result) == set()
    assert "H10.6:model_visible_code_changed" in warnings(result)
    assert "no version_segments entry records" in (
        _finding(result, "model_visible_code_changed").detail
    )
    write_jsonl(launches, rows)

    # Both records of ALF_NONE say another harness: consistent within the cell, not with
    # the other ALFWorld cell it is compared with.
    none = c.cell(ALF_NONE)
    c.edit(none.attempt(1).state, lambda s: s["code"].__setitem__("harness", "e" * 64))
    rows = jsonl(none.launches)
    _work_line(rows)["code"]["harness"] = "e" * 64
    write_jsonl(none.launches, rows)
    # A baseline fixed and re-run alone ran other code: reported (a warning listing
    # each version's cells), not enforced; the report flags it.
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    (across,) = [f for f in result.findings if f.check == "code_differs_across_cells"]
    assert across.severity == audit.WARNING
    assert f"{'e' * 12}: {ALF_NONE}" in across.detail and ALF_RB in across.detail
    assert f"| `{ALF_NONE}` | " in result.to_markdown()
    assert f"`{'e' * 12}`" in result.to_markdown()

    c.edit(none.attempt(1).state, lambda s: s.pop("code"))
    _work_line(rows).pop("code")
    write_jsonl(none.launches, rows)
    result = c.audit()
    assert violations(result) == set() and "H10.6:code_unrecorded" in warnings(result)


def test_only_launches_that_ran_the_attempt_count_for_its_code(copy):
    """An invocation that ran nothing (its start line only) or that worked on another
    attempt does not count as a code change of the audited attempt; a line written before
    start and work lines existed (no ``event``) still counts from the attempt's start."""
    c = copy(ALF_RB)
    launches = c.cell(ALF_RB).launches
    rows = jsonl(launches)
    start, work = rows
    assert start["event"] == "start" and "code" not in start
    other = {**work["code"], "harness": "f" * 64}
    late = "2999-01-01T00:00:00+00:00"
    no_op = {**start, "launch": "0123456789ab", "at": late, "code": other}
    elsewhere = {**work, "launch": "ba9876543210", "at": late, "attempt": 2}
    elsewhere["code"] = other
    write_jsonl(launches, [*rows, no_op, elsewhere])
    assert violations(c.audit()) == set()

    # The invocation that wrote the attempt's rows must have its work line.
    write_jsonl(launches, [start, no_op])
    result = c.audit()
    assert violations(result) == set() and "H10.6:code_unrecorded" in warnings(result)

    legacy = {k: v for k, v in no_op.items() if k != "event"}
    write_jsonl(launches, [*rows, legacy])
    result = c.audit()
    assert violations(result) == set()
    assert "H10.6:model_visible_code_changed" in warnings(result)
    # ... but not when it came before the attempt started.
    legacy["at"] = "2000-01-01T00:00:00+00:00"
    write_jsonl(launches, [legacy, *rows])
    assert violations(c.audit()) == set()


def test_a_ledger_is_matched_to_the_gateway_that_wrote_it(copy):
    """The start record is found by the id every ledger line carries, not by time: a
    later conformant gateway cannot vouch for calls another gateway served."""
    c = copy(ALF_RB)
    starts_path = c.runs / "_gateway" / "starts.jsonl"
    starts = jsonl(starts_path)
    served = {r.get("gateway_start") for r in jsonl(c.ledger(ALF_RB))}
    assert served and None not in served
    rogue = {
        **starts[0],
        "start_id": "0123456789abcdef",
        "upstream": "http://elsewhere",
        "ts": "2000-01-01T00:00:00+00:00",
    }
    write_jsonl(starts_path, [rogue, *starts])
    assert violations(c.audit()) == set()  # nothing of the attempt came from it
    c.edit_ledger(
        ALF_RB,
        lambda rows: [
            row.__setitem__("gateway_start", rogue["start_id"])
            for row in rows[: len(rows) // 2]
        ],
    )
    assert violations(c.audit()) == {"H10.3:upstream"}
    c.edit_ledger(
        ALF_RB,
        lambda rows: [
            row.__setitem__("gateway_start", "feedfacefeedface") for row in rows
        ],
    )
    assert violations(c.audit()) == {"H10.6:gateway_start_missing"}


def test_an_embedding_failure_no_retry_recovered_in_the_learning_phase(copy):
    """An INFRA failure on the embedding role that RB's caller swallowed."""
    c = copy(ALF_RB)
    embedding = c.calls(ALF_RB, role="embedding", phase="learning")[0]

    def fail(rows: list[dict[str, Any]]) -> None:
        stray = {
            **embedding,
            "call_id": "lost-embedding",
            "status": "error",
            "http_status": 504,
            "error": {"type": "upstream_timeout", "message": "timed out"},
            "cost": 0,
        }
        stray.pop("repeat_of", None)
        rows.append(stray)

    c.edit_ledger(ALF_RB, fail)
    result = c.audit()
    # Recorded, naming the call.
    assert violations(result) == set()
    assert "H10.3:learning_infra_unrecovered" in warnings(result)
    assert "lost-embedding" in _finding(result, "learning_infra_unrecovered").detail


def test_the_published_results_are_the_scorers_records(copy):
    """H9: report.py reads the cell's results.jsonl. It must be the attempt's last
    scored row per test item; each row's score its trajectory's result; and scoring the
    trajectory's final answer again must give that score."""
    c = copy(ALF_RB)
    cell = c.cell(ALF_RB)
    published = jsonl(cell.results)
    flipped = [
        {**r, "score": {**r["score"], "primary": 1.0 - r["score"]["primary"]}}
        for r in published
    ]
    write_jsonl(cell.results, flipped)
    assert violations(c.audit()) == {"H9:cell_results"}
    write_jsonl(cell.results, published[:-1])
    assert violations(c.audit()) == {"H9:cell_results"}
    write_jsonl(cell.results, published)
    assert violations(c.audit()) == set()

    # The attempt row, then the trajectory's recorded result, edited to match.
    attempt = cell.attempt(1)
    rows = jsonl(attempt.results)
    target = rows[0]
    new_score = {**target["score"], "primary": 1.0 - target["score"]["primary"]}
    write_jsonl(attempt.results, [{**target, "score": new_score}, *rows[1:]])
    write_jsonl(
        cell.results,
        [
            {**r, "score": new_score} if r["raw_id"] == target["raw_id"] else r
            for r in published
        ],
    )
    # The row is neither its trajectory's result nor what the environment scores.
    assert violations(c.audit()) == {"H9:score_record", "H9:rescore"}
    c.edit(
        cell.root / target["trajectory"],
        lambda d: d["result"].__setitem__("score", new_score),
    )
    assert violations(c.audit()) == {"H9:rescore"}


def test_a_torn_results_line_is_reported(copy):
    """A fragment left by a crash is skipped by every reader and reported."""
    c = copy(ALF_RB)
    with c.cell(ALF_RB).attempt(1).results.open("a") as fh:
        fh.write('{"raw_id": "val:9999", "score": {"prim')
    result = c.audit()
    assert violations(result) == set()
    assert "H9:torn_lines" in warnings(result)


# --- resumed learning --------------------------------------------------------------------


ALF_AWM = cell_spec("alfworld", "awm").cell_id


class _Faults:
    """One-shot infrastructure failures shared by every method instance of a fixture
    (a resumed invocation builds a new instance and must not fail again)."""

    def __init__(self) -> None:
        self.fired: set[str] = set()
        self.after_task = 0
        self.before_task = 0
        self.messages = 0

    def fire(self, key: str) -> None:
        from evals.skill_bench.tests.run_support import InfraFailure

        if key not in self.fired:
            self.fired.add(key)
            raise InfraFailure(f"injected infrastructure failure ({key})")


def _flaky_reasoningbank(faults: _Faults) -> Any:
    """ReasoningBank whose second ``after_task`` fails on infrastructure after it learned
    (J judged, the learner extracted, the items were embedded): the resume restores item
    1's played checkpoint and those calls are discarded."""
    from evals.skill_bench.methods.reasoningbank import ReasoningBank

    class Flaky(ReasoningBank):
        async def after_task(self, trajectory: Any) -> None:
            await super().after_task(trajectory)
            faults.after_task += 1
            if faults.after_task == 2:
                faults.fire("rb_after_task")

    return Flaky


def _flaky_awm(faults: _Faults) -> Any:
    """AWM whose third learning item fails on infrastructure in its rollout, after two
    actor turns (the method's ``on_message``): the resume restores item 1's done
    checkpoint, and the try's actor calls (and its failed trajectory, when one was
    written) are discarded."""
    from evals.skill_bench.methods.awm import AgentWorkflowMemory

    class Flaky(AgentWorkflowMemory):
        async def before_task(self, task: Any) -> Any:
            faults.before_task += 1
            faults.messages = 0
            return await super().before_task(task)

        async def on_message(self, task_uid: str, message: Any) -> None:
            await super().on_message(task_uid, message)
            faults.messages += 1
            if faults.before_task == 3 and faults.messages == 4:
                faults.fire("awm_rollout")

    return Flaky


@pytest.fixture(scope="module")
def resumed_matrix(
    e2e_matrix: Matrix, tmp_path_factory: pytest.TempPathFactory
) -> Matrix:
    """Three ALFWorld cells whose learning phase an infrastructure failure interrupted
    and the next invocation of run.py resumed from a checkpoint:
    ReasoningBank after item 1 learned (from its played checkpoint), AWM in the rollout
    of item 2 (from item 1's done checkpoint), and MIRIX on the stub MIRIX, whose first
    curator batch fails on infrastructure (from item 1's played checkpoint, the stub's
    database restored)."""
    from evals.skill_bench.methods.mirix import MirixMethod
    from evals.skill_bench.tests.mirix_stub import StubMirix, StubMirixServer

    root = tmp_path_factory.mktemp("resumed")
    env = e2e_matrix.envs["alfworld"]
    stubs: dict[tuple[str, int], StubMirix] = {}

    def server(spec: Any, mode: str) -> StubMirixServer:
        key = (spec.cell_id, spec.attempt)
        if mode == "fresh":
            stubs[key] = StubMirix(threshold=2, curator_errors=["infra"])
        return StubMirixServer(stubs[key], mode)

    faults = _Faults()
    factories = {
        "reasoningbank": _flaky_reasoningbank(faults),
        "awm": _flaky_awm(faults),
        "mirix": lambda: MirixMethod(poll_s=0.01),
    }
    specs = [
        cell_spec("alfworld", "reasoningbank"),
        cell_spec("alfworld", "awm"),
        cell_spec("alfworld", "mirix"),
    ]
    model = E2EModel()
    fake = FakeUpstream(responder=model)
    runs = root / "runs"
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        first = asyncio.run(
            run_cells(
                runs,
                url,
                {"alfworld": env},
                specs,
                method_factories=factories,
                mirix_server_factory=server,
            )
        )
        assert {o.status for o in first.values()} == {"learning_interrupted"}, first
        outcomes = asyncio.run(
            run_cells(
                runs,
                url,
                {"alfworld": env},
                specs,
                method_factories=factories,
                mirix_server_factory=server,
            )
        )
    assert {o.status for o in outcomes.values()} == {"complete"}, outcomes
    assert faults.fired == {"rb_after_task", "awm_rollout"}
    return Matrix(root, runs, {"alfworld": env}, outcomes, model, fake)


def _copies(matrix: Matrix, tmp_path: Path) -> Callable[..., Copy]:
    """Copies of ``matrix``'s cells, each in a directory of its own."""
    made = iter(range(1_000))
    return lambda *cells: Copy(matrix, tmp_path / f"copy-{next(made)}", *cells)


@pytest.fixture
def resumed(resumed_matrix: Matrix, tmp_path: Path) -> Callable[..., Copy]:
    return _copies(resumed_matrix, tmp_path)


@pytest.fixture(scope="module")
def rerun_matrix(
    e2e_matrix: Matrix, tmp_path_factory: pytest.TempPathFactory
) -> Matrix:
    """The re-run of a baseline fixed after its run, through run.py. ALFWorld's no-memory and AWM cells run under an
    earlier protocol (:data:`EARLIER`, frozen then) on a first gateway, whose start record
    carries it; AWM's learning stops on an infrastructure failure. After the re-freeze a
    fresh gateway serves the frozen protocol, and ``--new-attempt --note`` invalidates
    AWM's attempt 1 and runs attempt 2 to completion under the frozen value (a resume
    under the new protocol would continue the attempt as a new version segment: run.py's
    own tests cover that)."""
    root = tmp_path_factory.mktemp("rerun")
    env = e2e_matrix.envs["alfworld"]
    model = E2EModel()
    fake = FakeUpstream(responder=model)
    runs = root / "runs"
    none, awm = cell_spec("alfworld", "none"), cell_spec("alfworld", "awm")
    real_manifest = protocol.fingerprint_manifest
    record = protocol.load_frozen_protocol()

    def earlier_manifest(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, Any]:
        return {**real_manifest(root), "fingerprint": EARLIER}

    def earlier_record(path: Path | None = None) -> dict[str, Any]:
        return {**record, "fingerprint": EARLIER, "history": []}

    faults = _Faults()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(protocol, "fingerprint_manifest", earlier_manifest)
        patch.setattr(protocol, "load_frozen_protocol", earlier_record)
        with serve_gateway(
            root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
        ) as (_gw, url):
            first = asyncio.run(
                run_cells(
                    runs,
                    url,
                    {"alfworld": env},
                    [none, awm],
                    method_factories={"awm": _flaky_awm(faults)},
                )
            )
    assert first[none.cell_id].status == "complete", first
    assert first[awm.cell_id].status == "learning_interrupted", first

    def rerun(base_url: str, **options: Any) -> CellOutcome:
        runner = CellRunner(
            awm,
            RunOptions(
                out_dir=runs / awm.cell_id,
                gateway_url=base_url,
                gateway_runs_dir=runs,
                preflight=False,
                idle_timeout_s=60.0,
                test_concurrency=2,
                **options,
            ),
            env_factory=lambda _benchmark: env,
        )
        return asyncio.run(runner.run())

    # RunOptions.note (run.py --note) where the run.py under test has it.
    note = (
        {"note": "re-run under the frozen protocol after the fix"}
        if "note" in {f.name for f in dataclasses.fields(RunOptions)}
        else {}
    )
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        second = rerun(url, new_attempt=True, **note)
    assert (second.status, second.attempt) == ("complete", 2), second.detail
    return Matrix(
        root,
        runs,
        {"alfworld": env},
        {**first, awm.cell_id: second},
        model,
        fake,
    )


def test_a_baseline_rerun_under_the_frozen_protocol_passes_with_warnings(
    rerun_matrix: Matrix, tmp_path: Path
):
    """A cell whose invalid attempt 1 ran under an earlier version on
    its own gateway and whose attempt 2 ran under the frozen one on a fresh gateway,
    next to a cell complete under the earlier version, passes the audit with warnings
    only: the no-memory cell's version, the operator's re-roll, the protocols across
    cells. Nothing of attempt 1 or of the first gateway counts against attempt 2."""
    c = _copies(rerun_matrix, tmp_path)(ALF_NONE, ALF_AWM)
    frozen = protocol.frozen_fingerprint()
    paths = c.cell(ALF_AWM)
    manifest = read_json(paths.manifest)
    assert [
        (e["fingerprint"], e["attempts"]) for e in manifest["protocol_history"]
    ] == [(EARLIER, [1])]
    assert read_json(paths.attempt(1).state)["invalidation"] == "operator_reroll"
    assert read_json(paths.attempt(2).state)["protocol"]["fingerprint"] == frozen
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set(), result.to_markdown()
    cells = {x.cell_id: x for x in result.cells}
    assert cells[ALF_AWM].info["protocol"]["fingerprint"] == frozen
    assert cells[ALF_AWM].info["protocol"]["current"] is True
    assert cells[ALF_NONE].info["protocol"]["fingerprint"] == EARLIER
    by_cell = {
        x.cell_id: {
            f"{f.invariant}:{f.check}"
            for f in x.findings
            if f.severity == audit.WARNING
        }
        - DIRTY
        for x in result.cells
    }
    assert "H10.6:protocol_version" in by_cell[ALF_NONE]
    assert "H10.3:operator_reroll" in by_cell[ALF_AWM]
    assert (
        not {"H10.6:protocol_version", "H10.6:gateway_protocol_version"}
        & (by_cell[ALF_AWM])
    )
    assert "H10.6:protocol_version_unauditable" not in warnings(result)
    across = _finding(result, "protocol_differs_across_cells")
    assert across.severity == audit.WARNING
    assert result.passed and result.exit_code == 0
    # Without EARLIER among the known versions, the cell that ran under it has a version
    # no record registers: recorded, nothing fails.
    unknown = c.audit()
    failing = {
        x.cell_id
        for x in unknown.cells
        if any(f.severity == audit.VIOLATION for f in x.findings)
    }
    assert failing == set(), unknown.to_markdown()
    cells = {x.cell_id: x for x in unknown.cells}
    assert "H10.6:fingerprint" in {
        f"{f.invariant}:{f.check}"
        for f in cells[ALF_NONE].findings
        if f.severity == audit.WARNING
    }


def _edit_resumed_ledger(
    c: Copy, cell_id: str, change: Callable[[list[dict[str, Any]]], Any]
) -> None:
    """``Copy.edit_ledger`` for a resumed cell: rewriting the ledger moves its lines, so
    the discarded row and the resume marker are re-pointed at the markers' new offsets."""
    c.edit_ledger(cell_id, change)
    _fix_offsets(c, cell_id)


def _segment(c: Copy, cell_id: str, number: int = 1) -> dict[str, Any]:
    rows = jsonl(c.cell(cell_id).attempt(1).discarded_log)
    return rows[number - 1]


def _rewrite_segment(
    c: Copy, cell_id: str, change: Callable[[dict[str, Any]], Any]
) -> None:
    path = c.cell(cell_id).attempt(1).discarded_log
    rows = jsonl(path)
    change(rows[0])
    write_jsonl(path, rows)


def test_a_resumed_learning_phase_passes_the_audit(resumed):
    c = resumed(ALF_RB, ALF_AWM, ALF_MIRIX)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    by_cell = {cell.cell_id: cell for cell in result.cells}
    rb = by_cell[ALF_RB].info["learning_resumes"]
    assert [(s["segment"], s["checkpoint"]) for s in rb] == [(1, "c0001-played")]
    assert rb[0]["calls"] > 0 and "actor" not in rb[0]["calls_by_role"]
    assert {"judge", "learner"} <= set(rb[0]["calls_by_role"])
    awm = by_cell[ALF_AWM].info["learning_resumes"]
    assert [(s["segment"], s["checkpoint"]) for s in awm] == [(1, "c0001-done")]
    assert awm[0]["calls_by_role"].get("actor", 0) >= 2
    mirix = by_cell[ALF_MIRIX].info["learning_resumes"]
    assert [(s["segment"], s["checkpoint"]) for s in mirix] == [(1, "c0001-played")]
    assert mirix[0]["runs"] == 1 and set(mirix[0]["calls_by_role"]) <= {
        "mirix",
        "embedding",
    }
    # The segments are what discarded.jsonl records, recomputed from the markers.
    for cell_id in (ALF_RB, ALF_AWM, ALF_MIRIX):
        row = _segment(c, cell_id)
        info = by_cell[cell_id].info["learning_resumes"][0]
        assert info["calls"] == len(row["calls"])
    # The AWM try's trajectory was recorded before it failed, so nothing is unscanned.
    assert "H10.3:discarded_inputs_unrecorded" not in warnings(result)
    assert by_cell[ALF_AWM].info.get("discarded_trajectories", 0) >= 1


def test_a_discarded_segment_that_lists_other_calls_than_its_markers(resumed):
    c = resumed(ALF_RB)
    segment = _segment(c, ALF_RB)
    dropped = next(x for x in segment["calls"] if x["role"] == "judge")

    def drop(row: dict[str, Any]) -> None:
        row["calls"] = [x for x in row["calls"] if x["call_id"] != dropped["call_id"]]

    _rewrite_segment(c, ALF_RB, drop)
    result = c.audit()
    # audit-resume as recorded.
    assert violations(result) == set()
    assert "H10.3:discarded_segment_mismatch" in warnings(result)
    live = c.calls(ALF_RB, role="actor", phase="test")[0]

    def claim(row: dict[str, Any]) -> None:
        row["calls"] = [*row["calls"], dropped, {"call_id": live["call_id"]}]

    c2 = resumed(ALF_RB)
    _rewrite_segment(c2, ALF_RB, claim)
    found = c2.audit()
    assert "H10.3:discarded_segment_mismatch" in warnings(found)
    detail = " ".join(
        f.detail
        for f in found.cells[0].findings
        if f.check == "discarded_segment_mismatch"
    )
    assert "1 listed that are not between them" in detail


def test_a_resume_marker_with_calls_after_it_and_no_row(resumed):
    """A resume whose segment was never recorded, although learning went on: its
    discarded calls are not accounted for."""
    c = resumed(ALF_AWM)
    c.cell(ALF_AWM).attempt(1).discarded_log.write_text("")
    result = c.audit()
    found = violations(result) | warnings(result)
    assert "H10.3:discarded_segment_unrecorded" in warnings(result)
    # The try's actor calls are live again: no trajectory accounts for them.
    assert "H10.3:unaccounted_actor_call" in found or (
        "H10.3:actor_call_not_in_ledger" in found
        or "H10.3:discarded_segment_mismatch" in found
    )


def test_a_resume_marker_a_dead_resume_left_is_allowed_without_calls(resumed):
    """A resume that died between its marker and its row leaves a second marker of the
    same segment: allowed while no call started between the two."""
    c = resumed(ALF_RB)
    ledger = c.ledger(ALF_RB)
    rows = jsonl(ledger)
    index = next(i for i, r in enumerate(rows) if r.get("event") == "resume")
    rows.insert(index, dict(rows[index], launch="dead-launch"))
    write_jsonl(ledger, rows)
    # The row's marker offset moved by the inserted line: point it at the real one.
    lines = audit.read_ledger_lines(ledger)
    real = [line for line in lines if line.row.get("event") == "resume"][-1]
    _rewrite_segment(
        c,
        ALF_RB,
        lambda row: row.__setitem__("resume_marker_offset", real.offset),
    )
    _fix_offsets(c, ALF_RB)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    # The same dead marker with a call after it is not.
    rows = jsonl(ledger)
    index = next(i for i, r in enumerate(rows) if r.get("event") == "resume")
    template = next(r for r in rows if r.get("event") == "call_started")
    rows.insert(index + 1, dict(template, call_id="zombie-call"))
    write_jsonl(ledger, rows)
    _fix_offsets(c, ALF_RB)
    assert "H10.3:discarded_segment_unrecorded" in warnings(c.audit())


def test_a_discarded_call_an_older_gateway_logged_without_call_started(resumed):
    """Defence in depth: an older gateway answered a call it held during a
    budget pause 402 before writing its call_started line. Such a call is placed by its
    final line, so it is still in its discarded segment (not a live call, and its redo
    not a retry)."""
    c = resumed(ALF_RB)
    segment = _segment(c, ALF_RB)
    held = next(x for x in segment["calls"] if x["role"] == "judge")["call_id"]
    _edit_resumed_ledger(
        c,
        ALF_RB,
        lambda rows: rows.__setitem__(
            slice(None),
            [
                r
                for r in rows
                if not (r.get("event") == "call_started" and r.get("call_id") == held)
            ],
        ),
    )
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    info = result.cells[0].info["learning_resumes"][0]
    assert info["calls"] == len(segment["calls"])


def _fix_offsets(c: Copy, cell_id: str) -> None:
    """Re-point a tampered ledger's discarded row at its markers' new byte offsets."""
    lines = audit.read_ledger_lines(c.ledger(cell_id))
    resumes = [line for line in lines if line.row.get("event") == "resume"]
    marks = {
        line.row.get("id"): line.offset
        for line in lines
        if line.row.get("event") == "checkpoint"
    }
    path = c.cell(cell_id).attempt(1).discarded_log
    rows = jsonl(path)
    for row in rows:
        row["resume_marker_offset"] = resumes[-1].offset
        row["checkpoint"]["offset"] = marks.get(row["checkpoint"]["id"])
    write_jsonl(path, rows)
    ledger = c.ledger(cell_id)
    fixed = []
    for row in jsonl(ledger):
        if row.get("event") == "resume":
            row["checkpoint_offset"] = marks.get(row.get("checkpoint"))
        fixed.append(row)
    write_jsonl(ledger, fixed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snapshot_sha256_actual", "0" * 64),
        ("snapshot_sha256_actual", None),  # the last resume must have compared it
        ("files_equal", False),
    ],
)
def test_a_resume_that_did_not_restore_its_checkpoint(resumed, field, value):
    c = resumed(ALF_RB)
    _rewrite_segment(c, ALF_RB, lambda row: row["restore"].__setitem__(field, value))
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:learning_resume_mismatch" in warnings(result)


def test_a_mirix_resume_whose_database_digest_differed(resumed):
    c = resumed(ALF_MIRIX)
    _rewrite_segment(
        c, ALF_MIRIX, lambda row: row["restore"].__setitem__("db_digest_equal", None)
    )
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:learning_resume_mismatch" in warnings(result)


def test_a_resumed_snapshot_check_that_differs(resumed):
    c = resumed(ALF_RB)
    state_path = c.cell(ALF_RB).attempt(1).state
    state = read_json(state_path)
    for check in state["snapshot_checks"]:
        if check["kind"] == "learning_resume":
            check["equal"] = False
    write_json_atomic(state_path, state)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:learning_resume_mismatch" in warnings(result)


def test_a_learning_resume_check_is_not_compared_with_the_frozen_memory(resumed):
    """The rebuilt memory of a resume is the checkpoint's, not the frozen one: H10.4
    leaves it to the resume rules."""
    c = resumed(ALF_RB)
    state = read_json(c.cell(ALF_RB).attempt(1).state)
    kinds = [check["kind"] for check in state["snapshot_checks"]]
    assert "learning_resume" in kinds
    resume = next(x for x in state["snapshot_checks"] if x["kind"] == "learning_resume")
    assert resume["sha256"] != state["frozen"]["sha256"]
    assert "H10.4:snapshot_changed" not in violations(c.audit())


def test_a_live_learning_position_played_twice(resumed):
    c = resumed(ALF_RB)
    path = c.cell(ALF_RB).attempt(1).learning
    rows = jsonl(path)
    write_jsonl(path, [*rows[:2], rows[1], *rows[2:]])
    result = c.audit()
    # An item of the stream played twice is recorded (audit-resume); only an item
    # outside the stream is a violation.
    assert violations(result) == set()
    assert {
        "H9:learning_position_repeated",
        "H9:learning_rows_order",
    } <= warnings(result)


def test_a_repeat_of_a_discarded_call_is_the_redo(resumed):
    """A later call that repeats a discarded one is the redo of the rolled-back try, for
    every role (learner and judge included); a repeat inside one discarded try is not."""
    c = resumed(ALF_RB)
    segment = _segment(c, ALF_RB)
    judged = [x for x in segment["calls"] if x["role"] == "judge"]
    live_judge = [
        r
        for r in c.calls(ALF_RB, role="judge")
        if r["call_id"] not in {x["call_id"] for x in segment["calls"]}
    ]
    redo = next(r for r in live_judge if r["item"] == judged[0]["item"])

    def flag(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") == redo["call_id"]:
                row["repeat_of"] = judged[0]["call_id"]

    _edit_resumed_ledger(c, ALF_RB, flag)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert result.cells[0].info["repeat_of_discarded"] == {"judge": 1}
    learners = [x for x in segment["calls"] if x["role"] == "learner"]
    assert learners

    def within(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if (
                row.get("event") == "call"
                and row.get("call_id") == learners[-1]["call_id"]
            ):
                row["repeat_of"] = judged[0]["call_id"]
                row["item"] = judged[0]["item"]

    _edit_resumed_ledger(c, ALF_RB, within)
    result = c.audit()
    assert violations(result) == set() and "H10.3:repeat_of" in warnings(result)


def test_a_discarded_infra_failure_is_not_a_learning_infra_failure(resumed):
    """A discarded embedding call that failed on infrastructure was rolled back and
    redone: it is left out of the learning-infrastructure check; the same failure
    on a live call is not."""
    c = resumed(ALF_RB)
    segment = _segment(c, ALF_RB)
    discarded = {x["call_id"] for x in segment["calls"]}
    lost = next(x for x in segment["calls"] if x["role"] == "embedding")
    live = next(
        r for r in c.calls(ALF_RB, role="embedding") if r["call_id"] not in discarded
    )

    def fail(call_id: str) -> Callable[[list[dict[str, Any]]], None]:
        def change(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                if row.get("event") == "call" and row.get("call_id") == call_id:
                    row.update(
                        status="infra_error",
                        http_status=502,
                        error={"type": "upstream_infra_failure", "code": 502},
                    )

        return change

    _edit_resumed_ledger(c, ALF_RB, fail(lost["call_id"]))
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    _edit_resumed_ledger(c, ALF_RB, fail(live["call_id"]))
    result = c.audit()
    assert violations(result) == set()
    assert "H10.3:learning_infra_unrecovered" in warnings(result)


def test_a_discarded_trajectory_is_scanned_for_labels(resumed):
    """What the method received before the rollback is checked like a live trajectory
    (H10.1): the item's raw id in its first message is a leak."""
    c = resumed(ALF_AWM)
    moved = sorted(
        (c.cell(ALF_AWM).attempt(1).root / "discarded").rglob("trajectories/**/*.json")
    )
    assert moved
    data = json.loads(moved[0].read_text())

    def leak(value: dict[str, Any]) -> None:
        value["first_messages"][0]["content"] += f"\n(item {value['raw_id']})"

    c.edit(moved[0], leak)
    found = c.audit()
    assert "H10.1:label_string" in violations(found)
    where = next(f.detail for f in found.cells[0].findings if f.check == "label_string")
    assert "discarded segment 1" in where and data["raw_id"]


def test_a_try_killed_before_its_trajectory_was_written_is_a_warning(resumed):
    """A SIGKILL mid-rollout leaves the try's actor calls in the ledger and no trajectory:
    rolled back, they cannot be scanned (a warning, not a violation)."""
    c = resumed(ALF_AWM)
    for path in (c.cell(ALF_AWM).attempt(1).root / "discarded").rglob(
        "trajectories/**/*.json"
    ):
        path.unlink()
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:discarded_inputs_unrecorded" in warnings(result)


def test_a_discarded_trajectory_whose_calls_are_live(resumed):
    c = resumed(ALF_AWM)
    moved = sorted(
        (c.cell(ALF_AWM).attempt(1).root / "discarded").rglob("trajectories/**/*.json")
    )
    live_call = c.calls(ALF_AWM, role="actor", phase="test")[0]["call_id"]

    def claim(data: dict[str, Any]) -> None:
        data["calls"][0]["gateway_call_id"] = live_call

    c.edit(moved[0], claim)
    assert "H10.3:discarded_segment_mismatch" in warnings(c.audit())


def test_mirix_calls_a_discarded_segment_cannot_explain(resumed):
    """MIRIX's discarded calls are checked against the segment's own runs and
    retrievals: without its discarded run, its batch's calls are
    unexplained."""
    c = resumed(ALF_MIRIX)
    assert violations(c.audit()) == set()
    _rewrite_segment(
        c, ALF_MIRIX, lambda row: row["mirix"].__setitem__("learning_runs", [])
    )
    assert "H10.3:mirix_unexplained_calls" in warnings(c.audit())


# --- provider policy ---------------------------------------------------------------------


def test_a_luna_call_with_a_provider_routing(copy):
    c = copy(ALF_RB)
    target = c.calls(ALF_RB, role="learner")[0]["call_id"]

    def route(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") == target:
                row["wire"]["provider"] = {"ignore": ["azure"]}

    c.edit_ledger(ALF_RB, route)
    assert violations(c.audit()) == {"H10.6:provider_policy_unexpected"}


# --- hosts -------------------------------------------------------------------------------


def _set_host(c: Copy, cell_id: str, **fields: Any) -> None:
    """Make the cell's attempt (and every line of its invocations) name another host."""
    paths = c.cell(cell_id)
    state = read_json(paths.attempt(1).state)
    state["host"] = {**state["host"], **fields}
    write_json_atomic(paths.attempt(1).state, state)
    lines = jsonl(paths.launches)
    for line in lines:
        if isinstance(line.get("host"), dict):
            line["host"] = {**line["host"], **fields}
    write_jsonl(paths.launches, lines)


def test_every_attempt_records_its_host(copy):
    c = copy(ALF_NONE, ALF_RB)
    result = c.audit()
    assert violations(result) == set()
    for cell in result.cells:
        host = cell.info["host"]
        assert set(host) >= {"hostname", "system", "machine"} and host["system"]


def test_cells_of_one_benchmark_on_two_platforms(copy, e2e_matrix, tmp_path):
    c = copy(ALF_NONE, ALF_RB, WS_RB)
    _set_host(c, ALF_RB, machine="x86_64-other", hostname="laptop-2.local")
    result = c.audit()
    # Hosts as recorded: a warning naming its cells.
    assert violations(result) == set()
    across = {f.check: f for f in result.findings}
    platform = across["platform_differs_across_cells"]
    assert platform.severity == audit.WARNING and platform.downgraded == audit.RECORD
    assert "alfworld" in platform.detail and platform.cells == (ALF_NONE, ALF_RB)
    # Another benchmark on another platform is how the matrix is split: fine.
    c2 = Copy(e2e_matrix, tmp_path / "split", ALF_NONE, WS_RB)
    _set_host(c2, WS_RB, machine="x86_64-other", hostname="laptop-2.local")
    assert violations(c2.audit()) == set()


def test_cells_of_one_benchmark_on_two_machines_of_one_platform_is_a_warning(
    copy, e2e_matrix, tmp_path
):
    c = copy(ALF_NONE, ALF_RB)
    _set_host(c, ALF_RB, hostname="another-mac", machine_id="e" * 32)
    result = c.audit()
    assert violations(result) == set()
    (found,) = [f for f in result.findings if f.check == "host_differs_across_cells"]
    assert (
        found.severity == audit.WARNING and "another-mac (id eeeeeeee)" in found.detail
    )
    # One machine whose hostname changed (a Mac on another network) is one.
    c2 = Copy(e2e_matrix, tmp_path / "renamed", ALF_NONE, ALF_RB)
    _set_host(c2, ALF_RB, hostname="my-laptop.local")
    result = c2.audit()
    assert violations(result) == set()
    assert not [f for f in result.findings if f.check == "host_differs_across_cells"]


def test_an_attempt_that_changed_host(copy, e2e_matrix, tmp_path):
    c = copy(ALF_RB)
    paths = c.cell(ALF_RB)
    lines = jsonl(paths.launches)
    work = next(line for line in lines if line.get("event") == "work")
    work["host"] = {**work["host"], "machine_id": "e" * 32}
    write_jsonl(paths.launches, lines)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.6:host_changed_within_attempt" in warnings(result)
    # The same machine under another hostname (a Mac that joined another
    # network between learning and testing) is the same host.
    c2 = Copy(e2e_matrix, tmp_path / "renamed", ALF_RB)
    lines = jsonl(c2.cell(ALF_RB).launches)
    for line in lines:
        if line.get("event") == "work":
            line["host"] = {**line["host"], "hostname": "my-laptop.local"}
    write_jsonl(c2.cell(ALF_RB).launches, lines)
    assert violations(c2.audit()) == set()
    # A record without a machine id (written before them) still compares hostnames.
    c3 = Copy(e2e_matrix, tmp_path / "legacy", ALF_RB)
    lines = jsonl(c3.cell(ALF_RB).launches)
    work = next(line for line in lines if line.get("event") == "work")
    work["host"] = {
        **{k: v for k, v in work["host"].items() if k != "machine_id"},
        "hostname": "elsewhere",
    }
    write_jsonl(c3.cell(ALF_RB).launches, lines)
    result = c3.audit()
    assert violations(result) == set()
    assert "H10.6:host_changed_within_attempt" in warnings(result)


def test_an_attempt_without_a_host_is_a_warning(copy):
    c = copy(ALF_RB)
    state_path = c.cell(ALF_RB).attempt(1).state
    state = read_json(state_path)
    state.pop("host")
    write_json_atomic(state_path, state)
    result = c.audit()
    assert violations(result) == set()
    assert "H10.6:host_unrecorded" in warnings(result)


def test_webshop_cells_with_two_sandbox_images(copy):
    c = copy(WS_NONE, WS_RB)
    for cell_id, image in (
        (WS_NONE, "sha256:" + "a" * 64),
        (WS_RB, "sha256:" + "b" * 64),
    ):
        path = c.cell(cell_id).manifest
        manifest = read_json(path)
        manifest["preflight"] = {"env": {"sandbox_image_id": image}}
        write_json_atomic(path, manifest)
    result = c.audit()
    assert violations(result) == set()
    (image,) = [
        f for f in result.findings if f.check == "sandbox_image_differs_across_cells"
    ]
    assert image.severity == audit.WARNING and image.cells == (WS_NONE, WS_RB)


# --- a runs directory read on another host ------------------------------------------------


def test_a_copied_runs_directory_reads_its_own_ledgers_not_stale_recorded_ones(
    copy, tmp_path
):
    """The attempts record absolute ledger paths of the host that ran them. A copy reads
    the ledger relative to its own runs directory first, so a stale file at the recorded
    path (another tree that has the same run directory) is never read."""
    c = copy(ALF_RB, WS_RB)
    stale = tmp_path / "other-host" / "runs"
    for cell_id in (ALF_RB, WS_RB):
        paths = c.cell(cell_id)
        state = read_json(paths.attempt(1).state)
        run_id = state["run_id"]
        (stale / run_id).mkdir(parents=True)
        # A stale ledger: the run's first line only.
        first = (c.runs / run_id / "gateway.jsonl").read_text().splitlines()[0]
        (stale / run_id / "gateway.jsonl").write_text(first + "\n")
        state["ledger"] = str(stale / run_id / "gateway.jsonl")
        write_json_atomic(paths.attempt(1).state, state)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert all(str(stale) not in str(cell.info) for cell in result.cells)


def test_a_recorded_ledger_of_another_run_directory_is_never_read(copy, tmp_path):
    """Without a relative ledger, a recorded path counts only when its directory is the
    run's own: another run's ledger at a recorded path is reported missing."""
    c = copy(ALF_RB)
    paths = c.cell(ALF_RB)
    state = read_json(paths.attempt(1).state)
    run_id = state["run_id"]
    elsewhere = tmp_path / "elsewhere" / "not-the-run"
    elsewhere.mkdir(parents=True)
    shutil.move(str(c.runs / run_id / "gateway.jsonl"), elsewhere / "gateway.jsonl")
    state["ledger"] = str(elsewhere / "gateway.jsonl")
    write_json_atomic(paths.attempt(1).state, state)
    assert "H10.3:ledger_missing" in violations(c.audit())
    # The same ledger in a --gateway-runs-dir is found.
    extra = tmp_path / "gateway-runs"
    (extra / run_id).mkdir(parents=True)
    shutil.move(str(elsewhere / "gateway.jsonl"), extra / run_id / "gateway.jsonl")
    assert not (c.runs / run_id / "gateway.jsonl").exists()
    assert "H10.3:ledger_missing" not in violations(c.audit(gateway_runs_dirs=[extra]))


def test_gateway_start_records_are_read_from_every_host_mirror(copy):
    """A merged view of two hosts keeps each host's _gateway under _hosts/<name>/: the
    start records of both are read (their union by start_id)."""
    c = copy(ALF_RB)
    mirror = c.runs / "_hosts" / "remote"
    mirror.mkdir(parents=True)
    shutil.move(str(c.runs / "_gateway"), mirror / "_gateway")
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert result.cells[0].info["gateway_starts"] >= 1
    # Without any start record the calls are uncovered.
    shutil.rmtree(mirror / "_gateway")
    assert "H10.6:gateway_start_missing" in violations(c.audit())


def test_a_merged_view_that_is_not_final_is_shown_and_warned(copy):
    """Over a merged view the audit shows what its _merged.json says
    (when it was built, each host's last sync) and warns (H9) when the view is not final:
    a --mirror-only sync since it was built, a host whose last rsync did not exit 0, and
    every conflict it lists. None of it is a violation: the verdict is unchanged."""
    c = copy(ALF_RB)
    before = c.audit()
    assert before.passed and before.merged_view is None
    assert before.to_dict()["merged_view"] is None
    write_json_atomic(
        c.runs / "_merged.json",
        {
            "schema": 1,
            "sources": {
                "remote": {
                    "spec": "user@remote:runs/remote",
                    "synced_at": "2026-09-29T12:00:00+00:00",
                    "rsync_exit": 0,
                },
                "local": {
                    "spec": "runs/local",
                    "synced_at": "2026-09-29T10:00:00+00:00",
                    "rsync_exit": 24,
                },
            },
            "view_built_at": "2026-09-29T10:00:05+00:00",
            "last_sync": {"at": "2026-09-29T12:00:00+00:00", "mode": "mirror-only"},
            "entries": {ALF_RB: "local"},
            "conflicts": [
                {
                    "entry": WS_NONE,
                    "hosts": ["local", "remote"],
                    "reason": "different content",
                }
            ],
        },
    )
    result = c.audit()
    assert result.passed and violations(result) == set(), result.to_markdown()
    assert result.exit_code == before.exit_code
    across = [
        (f.invariant, f.check, f.severity, f.detail)
        for f in result.findings
        if f.check.startswith("merged_view")
    ]
    assert [a[:3] for a in across] == [
        ("H9", "merged_view_not_final", audit.WARNING),
        ("H9", "merged_view_not_final", audit.WARNING),
        ("H9", "merged_view_conflict", audit.WARNING),
    ]
    assert "remote synced after the view was built" in across[0][3]
    assert "--mirror-only sync" in across[0][3]
    assert "the last sync of local exited 24" in across[1][3]
    assert f"{WS_NONE} from local, remote (different content)" in across[2][3]
    assert result.merged_view is not None and result.merged_view["final"] is False
    assert result.to_dict()["merged_view"]["conflicts"][0]["entry"] == WS_NONE
    md = result.to_markdown()
    assert "- Merged view (`_merged.json`): 2 host(s)" in md
    assert "view built 2026-09-29T10:00:05+00:00" in md
    assert "warning H9 `merged_view_conflict`" in md
    # A final view: shown, no warning.
    write_json_atomic(
        c.runs / "_merged.json",
        {
            "schema": 1,
            "sources": {
                "remote": {"synced_at": "2026-09-29T10:00:00+00:00", "rsync_exit": 0}
            },
            "view_built_at": "2026-09-29T10:00:05+00:00",
            "conflicts": [],
        },
    )
    final = c.audit()
    assert final.merged_view["final"] is True
    assert not [f for f in final.findings if f.check.startswith("merged_view")]


# --- removed methods (study.REMOVED_METHODS) -------------------------------------------------


def _removed_copy(c: Copy, source: str, removed: str) -> str:
    """A leftover cell of the removed method ``removed`` (the ``removed_method``
    fixture's), made of ``source``'s records under that method's cell id (its own
    directory; ``cell.json`` names the removed method). Returns its cell id."""
    method = read_json(c.cell(source).manifest)["cell"]["method"]
    cell_id = source.replace(f"-{method}", f"-{removed}", 1)
    shutil.copytree(c.cell(source).root, c.runs / cell_id)

    def rename(manifest: dict[str, Any]) -> None:
        manifest["cell"].update(method=removed, cell_id=cell_id)

    c.edit(c.cell(cell_id).manifest, rename)
    assert study.cell_id_method(cell_id) == removed
    return cell_id


def _mentions(result: audit.AuditReport, cell_id: str) -> list[audit.Finding]:
    """Every finding, in a cell or across cells, that names ``cell_id``."""
    return [
        f
        for f in [*result.findings, *(f for x in result.cells for f in x.findings)]
        if cell_id in f.detail or f.cell_id == cell_id
    ]


def test_a_removed_methods_cell_is_listed_and_never_audited(copy, removed_method):
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)
    result = c.audit()
    assert [x.cell_id for x in result.cells] == [ALF_NONE, ALF_RB]
    assert violations(result) == set() and _mentions(result, dc) == []
    assert result.incomplete == [] and result.passed and result.exit_code == 0
    (removed,) = result.removed
    removal = study.REMOVED_METHODS[removed_method]
    assert (removed.cell_id, removed.method, removed.status) == (
        dc,
        removed_method,
        "complete",
    )
    assert (removed.decision, removed.reason) == (removal.decision, removal.reason)
    assert removed.root == str(c.runs / dc)
    data = result.to_dict()
    assert data["removed"]["methods"] == {
        removed_method: {
            "decision": removal.decision,
            "reason": removal.reason,
            "cells": [dc],
        }
    }
    assert data["removed"]["cells"][0]["cell_id"] == dc
    assert dc not in json.dumps([data["cells"], data["findings"], data["incomplete"]])
    md = result.to_markdown()
    assert (
        "- Result: **PASS** (0 violation(s); 0 cell(s) not audited); 1 cell(s) of "
        "removed methods set aside (see 'Removed methods')"
    ) in md
    section = md.split("## Removed methods")[1]
    assert f"| `{dc}` | `{dc}` | complete |" in section
    assert f"- `{removed_method}`, {removal.decision}: {removal.reason}" in section
    assert "Not audited" not in md


def test_a_removed_cell_raises_no_finding_however_it_differs(copy, removed_method):
    """Every check of a cell and across cells leaves a removed method's cell out: its
    label leak, first prompt, platform, protocol and a second directory of its id make
    no finding. Without the removal record its method is in no matrix (set aside with
    one warning); the same records audited as a study cell (their method a study
    method again) show that each change is real, and how each is judged now."""
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)

    def leak(data: dict[str, Any]) -> None:
        data["messages"][2]["won"] = True

    c.edit(c.trajectory(dc, "learning"), leak)
    c.edit(
        c.trajectory(dc, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    _set_host(c, dc, machine="x86_64-other", hostname="laptop-2.local")
    _set_cell_protocol(c, dc, "0" * 64)
    shutil.copytree(c.cell(dc).root, c.runs / f"{dc}.old")
    result = c.audit()
    assert violations(result) == set() and _mentions(result, dc) == []
    assert [Path(r.root).name for r in result.removed] == [dc, f"{dc}.old"]
    assert result.to_dict()["removed"]["methods"][removed_method]["cells"] == [dc]
    assert result.exit_code == 0
    outside = c.audit(removed_methods={})
    assert outside.removed == [] and violations(outside) == set()
    assert [Path(r.root).name for r in outside.not_in_matrix] == [dc, f"{dc}.old"]
    (not_in_matrix,) = [
        f for f in outside.findings if f.check == "method_not_in_matrix"
    ]
    assert not_in_matrix.severity == audit.WARNING and not_in_matrix.cells == (dc,)
    assert [x.cell_id for x in outside.cells] == [ALF_NONE, ALF_RB]
    assert outside.to_dict()["not_in_matrix"]["methods"] == {removed_method: [dc]}
    assert outside.exit_code == 0
    for root in (c.cell(dc).root, c.runs / f"{dc}.old"):
        c.edit(root / "cell.json", lambda m: m["cell"].update(method="reasoningbank"))
    audited = c.audit(removed_methods={})
    assert audited.removed == [] and audited.not_in_matrix == []
    assert violations(audited) == {
        "H10.1:label_key",
        "H10.1:message_shape",
    }, audited.to_markdown()
    (slot,) = recorded(audited, "H10.2:outside_memory_slot")
    # Its fingerprint ("0" * 64) is registered nowhere: the checks that read this
    # tree's templates are not re-verified under it.
    assert slot.severity == audit.WARNING and slot.downgraded == audit.NOT_REVERIFIED
    assert "H10.6:fingerprint" in warnings(audited)
    assert {
        "H10.3:duplicate_cell_directories",
        "H10.6:platform_differs_across_cells",
        "H10.2:first_prompt_differs_across_versions",
    } <= across_warnings(audited)
    assert not [f for f in audited.findings if f.severity == audit.VIOLATION]


def test_a_removed_cell_that_stopped_mid_write_never_breaks_the_audit(
    copy, removed_method
):
    """A leftover cell of a removed method whose records were cut off mid-write
    (status, attempt record, results line, a trajectory) is listed as it is, and the
    audit of the study goes on."""
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)
    paths = c.cell(dc)
    broken = c.trajectory(dc, "test", 0)
    paths.status.write_text('{"status": "runn')
    paths.attempt(1).state.write_text('{"attempt": 1, "run_')
    with paths.attempt(1).results.open("a") as f:
        f.write('{"raw_id": "o')
    broken.write_bytes(b"\x00\xff not a trajectory")
    result = c.audit()
    assert violations(result) == set() and _mentions(result, dc) == []
    assert result.exit_code == 0
    (removed,) = result.removed
    assert removed.status is None


def test_a_removed_cell_whose_cell_json_names_no_method_is_set_aside_by_its_id(
    copy, removed_method
):
    """A removed method's cell whose cell.json lost its method is never audited as a
    study cell: its id names the method (``study.cell_method``), so it is listed and
    never audited."""
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)
    c.edit(c.cell(dc).manifest, lambda m: m["cell"].pop("method"))
    result = c.audit()
    assert [x.cell_id for x in result.cells] == [ALF_NONE, ALF_RB]
    assert violations(result) == set() and _mentions(result, dc) == []
    assert result.passed and result.exit_code == 0
    (removed,) = result.removed
    assert (removed.cell_id, removed.method) == (dc, removed_method)
    assert result.to_dict()["removed"]["methods"][removed_method]["cells"] == [dc]
    # A definition that names a study method keeps it, whatever the id says.
    c.edit(c.cell(dc).manifest, lambda m: m["cell"].update(method="reasoningbank"))
    assert dc in [x.cell_id for x in c.audit().cells]


def test_removed_cells_follow_the_selection(copy, removed_method):
    """``--cell``, the split mode and ``--include-incomplete`` select removed cells as
    they select any: never audited, listed only when selected."""
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)
    assert c.audit(cells=[ALF_NONE]).removed == []
    only = c.audit(cells=[dc])
    assert [r.cell_id for r in only.removed] == [dc] and only.cells == []
    assert (only.passed, only.exit_code, only.incomplete) == (False, 2, [])
    protocol_only = audit.audit_runs(
        c.runs,
        env_factory=c.matrix.env_factory,
        split_mode="protocol",
        expected_upstream=TEST_UPSTREAM,
    )
    assert protocol_only.removed == [] and protocol_only.cells == []  # smoke cells
    write_json_atomic(c.cell(dc).status, {"status": "test_incomplete"})
    included = c.audit(include_incomplete=True)
    assert [r.cell_id for r in included.removed] == [dc]
    assert [x.cell_id for x in included.cells] == [ALF_NONE, ALF_RB]


def test_the_removal_record_decides_what_is_set_aside(copy):
    """``removed_methods`` (default ``study.REMOVED_METHODS``) is the only switch: a
    method it names is set aside with the decision it records."""
    c = copy(ALF_NONE, ALF_AWM)
    result = c.audit(removed_methods={"awm": study.Removal("a decision", "a reason")})
    assert [x.cell_id for x in result.cells] == [ALF_NONE]
    (removed,) = result.removed
    assert (removed.cell_id, removed.decision, removed.reason) == (
        ALF_AWM,
        "a decision",
        "a reason",
    )
    assert set(result.to_dict()["removed"]["methods"]) == {"awm"}


def test_the_cli_lists_removed_cells(
    copy, tmp_path, capsys, monkeypatch, removed_method
):
    c = copy(ALF_NONE, ALF_RB)
    dc = _removed_copy(c, ALF_RB, removed_method)
    base = [
        "--runs-dir",
        str(c.runs),
        "--split-mode",
        "all",
        "--expected-upstream",
        TEST_UPSTREAM,
    ]
    # Only the removed cell: nothing to audit (exit 2), no environment needed.
    assert audit.main([*base, "--cell", dc, "--out", str(tmp_path / "a")]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert (printed["removed"], printed["incomplete"], printed["passed"]) == (
        [dc],
        [],
        False,
    )
    assert "## Removed methods" in (tmp_path / "a" / "audit.md").read_text()
    saved = json.loads((tmp_path / "a" / "audit.json").read_text())
    assert saved["cells"] == []
    assert saved["removed"]["methods"][removed_method]["cells"] == [dc]
    # With the study's cells (the matrix's environments): they pass, exit 0.
    real = audit.audit_runs
    monkeypatch.setattr(
        audit,
        "audit_runs",
        lambda *a, **kw: real(*a, env_factory=c.matrix.env_factory, **kw),
    )
    assert audit.main([*base, "--out", str(tmp_path / "b")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["removed"] == [dc] and printed["passed"] is True


# --- what blocks: only validity checks fail a cell ------------------------------------------


def _split_key(key: str) -> tuple[str, str]:
    invariant, check = key.split(":", 1)
    return invariant, check


def test_findings_gate():
    """The one gate. A validity check stays a violation, a recorded check is a
    warning (``downgraded: record``), a validity check whose inputs of this tree changed
    is a warning (``not_reverified``), an unclassified check fails closed. Downgrades
    are counted before the per-check cap; an explicit warning passes as it is."""
    assert len(audit.VALIDITY_CHECKS) == 61 and len(audit.RECORDED_CHECKS) == 43
    assert not audit.VALIDITY_CHECKS & set(audit.RECORDED_CHECKS)
    assert not audit.NOT_CHECKED_CHECKS & (
        audit.VALIDITY_CHECKS | set(audit.RECORDED_CHECKS)
    )
    for key in sorted(audit.VALIDITY_CHECKS):
        f = audit.Findings(None)
        f.add(*_split_key(key), "x")
        (found,) = f.items
        assert (found.severity, found.downgraded) == (audit.VIOLATION, None), key
        assert f.count(audit.VIOLATION) == 1 and not f.downgraded
    for key, rule in sorted(audit.RECORDED_CHECKS.items()):
        assert rule, key
        f = audit.Findings(None)
        f.add(*_split_key(key), "x")
        (found,) = f.items
        assert (found.severity, found.downgraded) == (audit.WARNING, audit.RECORD), key
        assert f.downgraded[(key, audit.RECORD)] == 1 and not f.count(audit.VIOLATION)
    f = audit.Findings(None)
    f.not_reverified["H10.2:memory_slot"] = ["envs/templates/alfworld/x.txt"]
    f.add("H10.2", "memory_slot", "x")
    f.add("H10.2", "outside_memory_slot", "x")
    assert [(x.check, x.severity, x.downgraded) for x in f.items] == [
        ("memory_slot", audit.WARNING, audit.NOT_REVERIFIED),
        ("outside_memory_slot", audit.VIOLATION, None),
    ]
    assert f.downgraded == {("H10.2:memory_slot", audit.NOT_REVERIFIED): 1}
    # Fail closed: a check nobody classified.
    f = audit.Findings("c")
    f.add("H10.3", "zz", "x")
    assert f.items[0].severity == audit.VIOLATION and f.items[0].downgraded is None
    # Exact counts beyond the cap.
    f = audit.Findings("c")
    for _ in range(30):
        f.add("H10.3", "repeat_of", "x")
    assert f.downgraded[("H10.3:repeat_of", audit.RECORD)] == 30
    assert len(f.items) == audit.MAX_FINDINGS_PER_CHECK
    assert f.suppressed[("H10.3", "repeat_of")] == 5 and f.count(audit.WARNING) == 30
    # An explicit warning of a validity check is a warning (its call site decided).
    f = audit.Findings("c")
    f.warn("H10.1", "trajectory_missing", "x")
    assert (f.items[0].severity, f.items[0].downgraded) == (audit.WARNING, None)
    # Not checked is a severity of its own.
    f.not_checked("H10.1", "items_unavailable", "x")
    assert f.items[-1].severity == audit.NOT_CHECKED and f.count(audit.NOT_CHECKED) == 1
    # Run-level violations that name cells are kept for attribution, uncapped.
    f = audit.Findings(None)
    for i in range(30):
        f.add("H10.2", "first_prompt_differs_across_cells", f"x{i}", cells=("a", "b"))
    f.add("H10.3", "duplicate_cell_directories", "x", cells=("a",))
    assert len(f.across) == 30 and all(x.cells == ("a", "b") for x in f.across)
    assert f.items[0].to_dict()["cells"] == ["a", "b"]


def test_findings_across_cells_fail_only_the_cells_they_name():
    """A run-level violation fails exactly the audited cells it names, each holding
    a copy with its cell id; one that names no audited cell fails the audit as a whole
    (``unattributed_violations``)."""

    def cell(cell_id: str, audited: bool = True) -> audit.CellAudit:
        return audit.CellAudit(
            cell_id, cell_id, "alfworld", "m", "none", "smoke", "complete", 1, audited
        )

    a, b, off = cell("a"), cell("b"), cell("c", audited=False)
    findings = audit.Findings(None)
    findings.add("H10.2", "first_prompt_differs_across_cells", "odd", cells=("a",))
    findings.add(
        "H10.2", "first_prompt_differs_across_cells", "odd too", cells=("a", "b")
    )
    findings.add(
        "H10.2", "first_prompt_differs_across_cells", "a ghost", cells=("ghost", "c")
    )
    findings.add("H10.3", "zz", "names nobody")
    findings.warn("H10.6", "code_differs_across_cells", "a warning", cells=("b",))
    unattributed = audit._attribute_across_cells([a, b, off], findings)
    assert unattributed == 2
    assert (a.violations, b.violations, off.violations) == (2, 1, 0)
    assert a.violations_by_invariant == {"H10.2": 2} and not a.passed
    assert [(x.cell_id, x.detail, x.cells) for x in b.findings] == [
        ("b", "odd too", ("a", "b"))
    ]
    report = audit.AuditReport(
        "runs",
        "f",
        "f",
        "u",
        [a, b, off],
        list(findings.items),
        "now",
        unattributed_violations=unattributed,
    )
    assert report.violations == 5 and report.exit_code == 1 and not report.passed
    assert report.to_dict()["unattributed_violations"] == 2
    md = report.to_markdown()
    assert "(across cells: a, b)" in md and "(cells: ghost, c)" in md


def _emitted_checks() -> tuple[set[str], set[str], set[str]]:
    """``invariant:check`` of every ``add`` (violation) and ``not_checked`` call site of
    audit.py, read from its source, and the dynamic sites not resolved."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(audit))
    #: The sites whose check is computed, with every value it takes (violations only;
    #: ``host_differs_across_cells`` passes severity=WARNING).
    dynamic = {
        "uid_issue[0]": {"H10.1:uid_format", "H10.1:uid_salt"},
        "check": {"H10.6:platform_differs_across_cells"},
    }
    added: set[str] = set()
    unchecked: set[str] = set()
    unresolved: set[str] = set()

    def receiver(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in ("f", "findings")
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "f"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("add", "not_checked")
            and receiver(node.func.value)
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        severity = next((k.value for k in node.keywords if k.arg == "severity"), None)
        if isinstance(severity, ast.Attribute) and severity.attr != "VIOLATION":
            continue
        invariant, arg = node.args[0].value, node.args[1]
        if isinstance(arg, ast.Constant):
            keys = {f"{invariant}:{arg.value}"}
        elif (
            isinstance(arg, ast.IfExp)
            and isinstance(arg.body, ast.Constant)
            and isinstance(arg.orelse, ast.Constant)
        ):
            keys = {f"{invariant}:{arg.body.value}", f"{invariant}:{arg.orelse.value}"}
        elif ast.unparse(arg) in dynamic:
            keys = dynamic[ast.unparse(arg)]
        else:
            unresolved.add(ast.unparse(arg))
            continue
        (unchecked if node.func.attr == "not_checked" else added).update(keys)
    return added, unchecked, unresolved


def test_every_check_the_audit_reports_is_classified():
    """Every check audit.py can report as a violation is either a validity
    check or a recorded one (a new check must be classified, or it fails closed), every
    not-checked site is one of NOT_CHECKED_CHECKS, and no classification is stale."""
    added, unchecked, unresolved = _emitted_checks()
    assert unresolved == set(), unresolved
    classified = audit.VALIDITY_CHECKS | set(audit.RECORDED_CHECKS)
    assert added - classified == set()
    assert unchecked == set(audit.NOT_CHECKED_CHECKS)
    # Classified but never emitted: only auditor_tree, kept for audit.json of schema 1.
    assert classified - added == {"H10.6:auditor_tree"}


def test_tree_inputs_are_validity_checks():
    """What the not-re-verified path reads is real: its checks are validity
    checks, its constants are protocol constants, and the gateway settings map onto
    constants of the same name."""
    constants = protocol.protocol_constants()
    assert set(audit.TREE_INPUTS) | set(audit.WITHIN_ATTEMPT_INPUTS) <= (
        audit.VALIDITY_CHECKS
    )
    for keys, prefixes in [
        *audit.TREE_INPUTS.values(),
        *audit.WITHIN_ATTEMPT_INPUTS.values(),
        audit.PROMPT_INPUTS,
    ]:
        assert set(keys) <= set(constants), keys
        for prefix in prefixes:
            # A directory of fingerprinted files, or one fingerprinted file.
            path = prefix.format(benchmark="alfworld")
            assert protocol.is_fingerprinted(
                path + "x" if path.endswith("/") else path
            ), prefix
    assert set(audit.SETTINGS_CONSTANTS) == set(gateway.PROTOCOL_SETTINGS)
    for setting, key in audit.SETTINGS_CONSTANTS.items():
        assert audit._canon(constants[key]) == audit._canon(
            gateway.PROTOCOL_SETTINGS[setting]
        ), setting
    assert set(audit.MODEL_SETTINGS) < set(audit.SETTINGS_CONSTANTS)


def test_changed_inputs_name_what_a_check_reads():
    """What of a version diff a check reads, failing closed on what is not
    recorded."""
    changed = audit._changed_inputs
    diff = {
        "constants_changed": True,
        "constants_changed_keys": ["methods", "seed"],
        "inconsistent": [],
        "changed": ["envs/templates/alfworld/a.txt", "methods/prompts/x.txt"],
        "added": ["envs/splits/alfworld/b.json"],
        "removed": [],
    }
    seed, splits = frozenset({"seed"}), ("envs/splits/{benchmark}/",)
    assert changed(diff, "alfworld", seed, splits) == [
        "protocol.py constant seed",
        "envs/splits/alfworld/b.json",
    ]
    assert changed(diff, "webshop", seed, splits) == ["protocol.py constant seed"]
    assert changed(diff, "alfworld", frozenset({"roles"}), ()) == []
    assert changed(None, "alfworld", frozenset(), ()) == [
        "a file map that is not recorded"
    ]
    assert changed({**diff, "constants_changed_keys": None}, "alfworld", seed, ()) == [
        "constants of protocol.py (which keys is not recorded)"
    ]
    assert changed({**diff, "constants_changed": None}, "alfworld", seed, ()) == [
        "a constants hash that is not recorded"
    ]
    assert changed({**diff, "inconsistent": ["f" * 64]}, "alfworld", seed, ()) == [
        f"a record that does not reproduce its fingerprint ({'f' * 12})"
    ]


# --- every validity check still fails the audit ----------------------------------------------

#: A scenario applies one mutation and returns the audit and the cell it touched.
Scenario = Callable[[pytest.FixtureRequest], tuple[audit.AuditReport, str]]


def _c(request: pytest.FixtureRequest, *cells: str) -> Copy:
    return request.getfixturevalue("copy")(*cells)


def _mirix(request: pytest.FixtureRequest) -> Copy:
    matrix = request.getfixturevalue("mirix_matrix")
    return Copy(matrix, request.getfixturevalue("tmp_path") / "mirix", ALF_MIRIX)


def _edited(cell_id: str, *, phase: str = "test", index: int = 0, **kw: Any) -> Any:
    """A scenario that edits one trajectory of ``cell_id`` (``change(data)``)."""

    def scenario(change: Callable[[dict[str, Any]], Any]) -> Scenario:
        def run(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
            c = _c(request, cell_id)
            c.edit(c.trajectory(cell_id, phase, index), change)
            return c.audit(**kw), cell_id

        return run

    return scenario


def _ledger_edit(
    cell_id: str, change: Callable[[Copy, list[dict[str, Any]]], Any], **kw: Any
) -> Scenario:
    def run(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
        c = _c(request, cell_id)
        c.edit_ledger(cell_id, lambda rows: change(c, rows))
        return c.audit(**kw), cell_id

    return run


def _first_call(c: Copy, cell_id: str, **match: Any) -> dict[str, Any]:
    return c.calls(cell_id, **match)[0]


def _set_on_call(call_id: str, **fields: Any) -> Callable[[list[dict[str, Any]]], None]:
    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") == call_id:
                row.update(fields)

    return change


def _s_test_items(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    c.edit(c.cell(ALF_RB).manifest, lambda m: m["cell"]["test_items"].pop())
    return c.audit(), ALF_RB


def _s_learning_items(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    tested = c.rows(ALF_RB, "test")[0]["raw_id"]
    c.edit(
        c.cell(ALF_RB).manifest, lambda m: m["cell"]["learning_items"].append(tested)
    )
    return c.audit(), ALF_RB


def _s_learning_rows(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.cell(ALF_RB).attempt(1).learning
    rows = jsonl(path)
    rows[0]["raw_id"] = c.rows(ALF_RB, "test")[0]["raw_id"]
    write_jsonl(path, rows)
    return c.audit(), ALF_RB


def _flip_score(c: Copy, cell_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The first scored test row's score flipped in the attempt and the cell records
    (not in its trajectory): the row and its new score."""
    cell = c.cell(cell_id)
    attempt = cell.attempt(1)
    rows = jsonl(attempt.results)
    target = rows[0]
    score = {**target["score"], "primary": 1.0 - target["score"]["primary"]}
    write_jsonl(attempt.results, [{**target, "score": score}, *rows[1:]])
    write_jsonl(
        cell.results,
        [
            {**r, "score": score} if r["raw_id"] == target["raw_id"] else r
            for r in jsonl(cell.results)
        ],
    )
    return target, score


def _s_score_record(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    _flip_score(c, ALF_RB)
    return c.audit(), ALF_RB


def _s_rescore(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    target, score = _flip_score(c, ALF_RB)
    c.edit(
        c.cell(ALF_RB).root / target["trajectory"],
        lambda d: d["result"].__setitem__("score", score),
    )
    return c.audit(), ALF_RB


def _s_cell_results(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.cell(ALF_RB).results
    rows = jsonl(path)
    write_jsonl(path, rows[:-1])
    return c.audit(), ALF_RB


def _s_cross_item(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    other = request.getfixturevalue("e2e_matrix").envs["alfworld"].load_tasks("val")[-1]
    return _edited(ALF_RB, phase="learning")(
        lambda d: d["messages"][2].__setitem__(
            "content", d["messages"][2]["content"] + f"\n{other.raw['gamefile']}"
        )
    )(request)


def _s_method_output(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    envs = request.getfixturevalue("e2e_matrix").envs
    trial = str(envs["alfworld"].load_tasks("test")[5].raw["gamefile"]).split("/")[-2]
    path = c.cell(ALF_RB).attempt(1).injections
    rows = jsonl(path)
    rows[-1]["text"] += f"\nRemember {trial}."
    write_jsonl(path, rows)
    return c.audit(), ALF_RB


def _s_answer(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, WS_AWM)
    task = _test_task(c, WS_AWM)
    _inject_for_item(
        c, WS_AWM, task.raw_id, f"\nThe expected answer is {task.raw['asin']}."
    )
    return c.audit(), WS_AWM


def _s_unknown_item(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.cell(ALF_RB).attempt(1).learning
    rows = jsonl(path)
    rows[0]["raw_id"] = "train:unknown"
    write_jsonl(path, rows)
    return c.audit(), ALF_RB


def _s_salt(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    c.cell(ALF_RB).salt.unlink()
    return c.audit(), ALF_RB


def _discarded(c: Copy, cell_id: str) -> list[Path]:
    root = c.cell(cell_id).attempt(1).root / "discarded"
    return sorted(root.rglob("trajectories/**/*.json"))


def _s_discarded_unread(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    c = request.getfixturevalue("resumed")(ALF_AWM)
    _discarded(c, ALF_AWM)[0].write_text("{")
    _rewrite_segment(
        c, ALF_AWM, lambda row: row["restore"].__setitem__("files_equal", False)
    )
    return c.audit(), ALF_AWM


def _s_prompt_varies(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = request.getfixturevalue("resumed")(ALF_AWM)
    moved = next(
        p for p in _discarded(c, ALF_AWM) if json.loads(p.read_text()).get("calls")
    )
    c.edit(moved, lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64))
    return c.audit(), ALF_AWM


def _s_request_unreadable(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    return _edited(ALF_RB)(
        lambda d: d["calls"][0]["request"]["messages"][0].__setitem__(
            "content", {"$blob": "0" * 64}
        )
    )(request)


def _s_first_request(
    cell_id: str, change: Callable[[dict[str, Any]], None]
) -> Scenario:
    def run(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
        c = _c(request, cell_id)
        _change_first_request(c, c.trajectory(cell_id, "test"), change, rehash=True)
        return c.audit(), cell_id

    return run


def _append_to(index: int, text: str) -> Callable[[dict[str, Any]], None]:
    def change(request: dict[str, Any]) -> None:
        message = request["messages"][index]
        message["content"] = message["content"] + text

    return change


def _s_request_not_as_sent(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.longest(ALF_RB, "test")
    memory = json.loads(path.read_text())["injection"]["text"]

    def swap(d: dict[str, Any]) -> None:
        for message in d["calls"][2]["request"]["messages"]:
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].replace(memory, "OTHER")

    c.edit(path, swap)
    return c.audit(), ALF_RB


def _s_conversation_window(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    """A WebShop call that dropped its oldest exchange (a window), the ledger agreeing."""
    c = _c(request, WS_RB)
    _change_call_request(
        c,
        c.longest(WS_RB, "test"),
        3,
        lambda r: r.__setitem__("messages", [r["messages"][0], *r["messages"][3:]]),
    )
    return c.audit(), WS_RB


def _s_across_prompts(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_NONE, ALF_RB)
    c.edit(
        c.trajectory(ALF_RB, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    return c.audit(), ALF_RB


def _s_trajectory_gone(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    c.trajectory(ALF_RB, "test").unlink()
    return c.audit(), ALF_RB


def _s_scored_twice(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.cell(ALF_RB).attempt(1).results
    rows = jsonl(path)
    write_jsonl(path, [rows[0], *rows])
    return c.audit(), ALF_RB


def _s_rerun_no_failure(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    path = c.cell(ALF_RB).attempt(1).results
    rows = jsonl(path)
    lost = {**rows[0], "score": None, "failure": {"kind": "model", "code": "x"}}
    write_jsonl(path, [lost, *rows])
    return c.audit(), ALF_RB


def _s_upstream(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    return c.audit(expected_upstream=audit.DEFAULT_UPSTREAM), ALF_RB


def _s_ledger_missing(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    elsewhere = request.getfixturevalue("tmp_path") / "elsewhere" / "not-the-run"
    elsewhere.mkdir(parents=True)
    shutil.move(str(c.ledger(ALF_RB)), elsewhere / "gateway.jsonl")
    state = c.cell(ALF_RB).attempt(1).state
    c.edit(state, lambda s: s.__setitem__("ledger", str(elsewhere / "gateway.jsonl")))
    return c.audit(), ALF_RB


def _s_unlinked(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    def unlink(d: dict[str, Any]) -> None:
        call = d["calls"][0]
        call["http_status"] = 200
        call.pop("gateway_call_id")

    return _edited(ALF_RB)(unlink)(request)


def _s_not_in_ledger(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    call_id = json.loads(c.trajectory(ALF_RB, "test").read_text())["calls"][0][
        "gateway_call_id"
    ]
    c.edit_ledger(
        ALF_RB,
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r.get("call_id") != call_id]
        ),
    )
    return c.audit(), ALF_RB


def _s_call_mismatch(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        call_id = json.loads(c.trajectory(ALF_RB, "test").read_text())["calls"][0][
            "gateway_call_id"
        ]
        _set_on_call(call_id, item="t" + "1" * 20)(rows)

    return _ledger_edit(ALF_RB, change)(request)


def _on_first(of_role: str, of_phase: str | None = None, **fields: Any) -> Any:
    """Set ``fields`` on the first ledger call of ``of_role`` (in ``of_phase``)."""

    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        match = {"role": of_role, **({"phase": of_phase} if of_phase else {})}
        _set_on_call(_first_call(c, ALF_RB, **match)["call_id"], **fields)(rows)

    return change


def _s_mirix_unwindowed(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    c = _mirix(request)

    def unwindow(snapshot: dict[str, Any]) -> None:
        run = next(
            r
            for r in snapshot["learning_runs"]
            if r.get("status") == "succeeded" and r.get("llm_usage")
        )
        run.pop("started_at")

    _rewrite_snapshot(c.cell(ALF_MIRIX), unwindow)
    return c.audit(), ALF_MIRIX


def _s_mirix_exceeds(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _mirix(request)
    c.edit_ledger(
        ALF_MIRIX,
        lambda rows: rows.__setitem__(
            slice(None),
            [
                r
                for r in rows
                if not (r.get("role") == "mirix" and r.get("phase") == "learning")
            ],
        ),
    )
    return c.audit(), ALF_MIRIX


def _s_mirix_no_usage(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _mirix(request)

    def drop(snapshot: dict[str, Any]) -> None:
        for run in snapshot["learning_runs"]:
            run.pop("llm_usage", None)

    _rewrite_snapshot(c.cell(ALF_MIRIX), drop)
    return c.audit(), ALF_MIRIX


def _s_state(change: Callable[[dict[str, Any]], Any]) -> Scenario:
    def run(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
        c = _c(request, ALF_RB)
        c.edit(c.cell(ALF_RB).attempt(1).state, change)
        return c.audit(), ALF_RB

    return run


def _no_after_test(state: dict[str, Any]) -> None:
    state["snapshot_checks"] = [
        x for x in state["snapshot_checks"] if x["kind"] != "after_test"
    ]


def _s_frozen_missing(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    c.cell(ALF_RB).attempt(1).frozen_snapshot.unlink()
    return c.audit(), ALF_RB


def _s_frozen_record(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    c.edit(
        c.cell(ALF_RB).attempt(1).frozen_snapshot,
        lambda d: d["snapshot"].__setitem__("notes", "rewritten after the freeze"),
    )
    return c.audit(), ALF_RB


def _s_snapshot_record(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    state = read_json(c.cell(ALF_RB).attempt(1).state)
    stored = c.cell(ALF_RB).root / state["snapshot_checks"][-1]["path"]
    c.edit(stored, lambda d: d["snapshot"].__setitem__("notes", "edited"))
    return c.audit(), ALF_RB


def _s_after_freeze(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        learner = _first_call(c, ALF_RB, role="learner")["call_id"]
        test_actor = _first_call(c, ALF_RB, role="actor", phase="test")
        for row in rows:
            row.pop("run_phase", None)
            row.pop("phase_header", None)
            if row.get("call_id") == learner and row["event"] == "call":
                row["ts"] = test_actor["ts"]

    return _ledger_edit(ALF_RB, change)(request)


def _s_for_test_item(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        learner = _first_call(c, ALF_RB, role="learner")["call_id"]
        _set_on_call(learner, item=c.rows(ALF_RB, "test")[0]["task_uid"])(rows)

    return _ledger_edit(ALF_RB, change)(request)


def _s_wire(**wire: Any) -> Scenario:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        call_id = _first_call(c, ALF_RB, role="actor", phase="test")["call_id"]
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") == call_id:
                row["wire"].update(wire)
                request = c.recorded_request(ALF_RB, call_id)
                row["request_sha256"] = audit.sent_request_sha256(request, row["wire"])

    return _ledger_edit(ALF_RB, change)


def _s_policy_missing(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    starts = c.runs / "_gateway" / "starts.jsonl"
    rows = jsonl(starts)
    routing = {protocol.MODELS["luna"]: {"ignore": ["wafer"]}}
    for row in rows:
        row["settings"] = {**row["settings"], "provider_policy": routing}
    write_jsonl(starts, rows)
    return c.audit(), ALF_RB


def _s_policy_unexpected(
    request: pytest.FixtureRequest,
) -> tuple[audit.AuditReport, str]:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        target = _first_call(c, ALF_RB, role="learner")["call_id"]
        for row in rows:
            if row.get("event") == "call" and row.get("call_id") == target:
                row["wire"]["provider"] = {"ignore": ["azure"]}

    return _ledger_edit(ALF_RB, change)(request)


def _s_start_settings(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    c = _c(request, ALF_RB)
    starts = c.runs / "_gateway" / "starts.jsonl"
    rows = jsonl(starts)
    for row in rows:
        row["settings"] = {**row["settings"], "reasoning_effort": "high"}
    write_jsonl(starts, rows)
    return c.audit(), ALF_RB


def _s_start_missing(request: pytest.FixtureRequest) -> tuple[audit.AuditReport, str]:
    def change(c: Copy, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["gateway_start"] = "feedfacefeedface"

    return _ledger_edit(ALF_RB, change)(request)


VALIDITY_SCENARIOS: dict[str, Scenario] = {
    "H9:test_items": _s_test_items,
    "H9:learning_items": _s_learning_items,
    "H9:learning_rows": _s_learning_rows,
    "H9:score_record": _s_score_record,
    "H9:rescore": _s_rescore,
    "H9:cell_results": _s_cell_results,
    "H10.1:uid_format": _edited(ALF_RB, index=1)(
        lambda d: d.__setitem__("task_uid", d["raw_id"])
    ),
    "H10.1:uid_salt": _edited(ALF_RB)(
        lambda d: d.__setitem__("task_uid", "t" + "0" * 20)
    ),
    "H10.1:message_shape": _edited(ALF_RB, phase="learning")(
        lambda d: d["messages"][1].__setitem__("reasoning", "hidden reasoning")
    ),
    "H10.1:label_key": _edited(ALF_RB, phase="learning")(
        lambda d: d["messages"][2].__setitem__("won", True)
    ),
    "H10.1:label_string": _edited(ALF_RB, phase="learning")(
        lambda d: d["messages"][2].__setitem__(
            "content", d["messages"][2]["content"] + f"\n(item {d['raw_id']})"
        )
    ),
    "H10.1:cross_item_label": _s_cross_item,
    # Done-page text in an environment message.
    "H10.1:structural_leak": _edited(WS_RB, phase="learning")(
        lambda d: d["messages"][2].__setitem__(
            "content", d["messages"][2]["content"] + " [SEP] Reward Details"
        )
    ),
    "H10.1:agent_context_varies": _edited(ALF_RB)(
        lambda d: d.__setitem__("agent_context", d["agent_context"] + "\n(look left)")
    ),
    "H10.1:method_output_label": _s_method_output,
    "H10.1:answer_in_method_output": _s_answer,
    "H10.1:unknown_item": _s_unknown_item,
    "H10.1:salt_missing": _s_salt,
    "H10.1:trajectory_missing": _s_discarded_unread,
    "H10.2:memory_changed": _edited(ALF_RB, index=0)(
        lambda d: d["calls"][-1].__setitem__("memory_sha256", "0" * 64)
    ),
    "H10.2:request_unreadable": _s_request_unreadable,
    "H10.2:first_prompt_hash": _edited(ALF_RB)(
        lambda d: d.__setitem__("first_prompt_sha256", "0" * 64)
    ),
    "H10.2:memory_slot": _s_first_request(ALF_RB, _append_to(0, "\n(memory here)")),
    "H10.2:outside_memory_slot": _s_first_request(
        ALF_RB, _append_to(1, "\nHint: the answer is on page 3.")
    ),
    "H10.2:first_prompt_varies": _s_prompt_varies,
    "H10.2:request_not_as_sent": _s_request_not_as_sent,
    "H10.2:first_prompt_differs_across_cells": _s_across_prompts,
    "H10.2:conversation_not_append_only": _s_conversation_window,
    "H10.3:trajectory_missing": _s_trajectory_gone,
    "H10.3:scored_twice": _s_scored_twice,
    "H10.3:rerun_without_failure": _s_rerun_no_failure,
    "H10.3:upstream": _s_upstream,
    "H10.3:ledger_missing": _s_ledger_missing,
    "H10.3:actor_call_unlinked": _s_unlinked,
    "H10.3:actor_call_not_in_ledger": _s_not_in_ledger,
    "H10.3:actor_call_mismatch": _s_call_mismatch,
    "H10.3:role": _ledger_edit(ALF_RB, _on_first("learner", role="wizard")),
    "H10.3:phase": _ledger_edit(ALF_RB, _on_first("learner", run_phase="bogus")),
    "H10.3:role_not_allowed": _ledger_edit(ALF_RB, _on_first("learner", role="mirix")),
    "H10.3:model": _ledger_edit(
        ALF_RB, _on_first("actor", "test", model=protocol.MODELS["luna56"])
    ),
    "H10.3:embedding_model": _ledger_edit(
        ALF_RB, _on_first("embedding", model="other/embed")
    ),
    "H10.3:unaccounted_actor_call": _ledger_edit(
        ALF_RB,
        lambda c, rows: rows.append(
            dict(_first_call(c, ALF_RB, role="actor", phase="test"), call_id="orphan")
        ),
    ),
    "H10.3:model_served": _ledger_edit(
        ALF_RB, _on_first("actor", "test", model_served="anthropic/claude-other-9")
    ),
    "H10.3:mirix_run_unwindowed": _s_mirix_unwindowed,
    "H10.3:mirix_usage_exceeds_ledger": _s_mirix_exceeds,
    "H10.3:mirix_usage_missing": _s_mirix_no_usage,
    "H10.4:frozen_missing": _s_frozen_missing,
    "H10.4:frozen_record": _s_frozen_record,
    "H10.4:after_test_missing": _s_state(_no_after_test),
    "H10.4:snapshot_changed": _s_state(
        lambda s: s["snapshot_checks"][-1].update(equal=False, sha256="1" * 64)
    ),
    "H10.4:snapshot_record": _s_snapshot_record,
    "H10.4:unverified_launch": _s_state(_no_after_test),
    "H10.4:learning_call_in_test": _ledger_edit(
        ALF_RB,
        _on_first("learner", phase="learning", run_phase="test", phase_header="test"),
    ),
    "H10.4:learning_call_after_freeze": _s_after_freeze,
    "H10.4:learning_call_for_test_item": _s_for_test_item,
    "H10.6:reasoning_effort": _s_wire(reasoning={"effort": "high"}),
    "H10.6:seed": _s_wire(seed=7),
    "H10.6:provider_policy_missing": _s_policy_missing,
    "H10.6:provider_policy_unexpected": _s_policy_unexpected,
    "H10.6:gateway_start_settings": _s_start_settings,
    "H10.6:gateway_start_missing": _s_start_missing,
}


def test_the_validity_scenarios_cover_the_whitelist():
    assert set(VALIDITY_SCENARIOS) == audit.VALIDITY_CHECKS


@pytest.mark.parametrize("key", sorted(VALIDITY_SCENARIOS))
def test_every_validity_check_still_fails_the_audit(request, key):
    """Whatever is recorded instead of blocking, each
    validity check still fails the cell it concerns, as a violation nobody downgraded,
    and the audit exits 1."""
    result, cell_id = VALIDITY_SCENARIOS[key](request)
    assert key in violations(result), result.to_markdown()
    hits = [
        f
        for f in [*result.findings, *(f for x in result.cells for f in x.findings)]
        if f"{f.invariant}:{f.check}" == key and f.severity == audit.VIOLATION
    ]
    assert hits and all(f.downgraded is None for f in hits)
    # In a cell, or across cells naming the cells it fails.
    assert all(f.cell_id is not None or f.cells for f in hits)
    assert result.exit_code == 1 and not result.passed
    (cell,) = [x for x in result.cells if x.cell_id == cell_id]
    assert cell.audited and not cell.passed and cell.violations > 0
    assert cell.to_dict()["invariants"][key.split(":")[0]] == "FAIL"


# --- what the harness recorded instead of stopping -------------------------------------------


def _failure_row(phase: str, item: str, kind: str, code: str, recovery: str, **kw: Any):
    """One ``failures.jsonl`` row as ``infra/failures.FailureLog`` writes it."""
    return {
        "at": "2026-09-30T10:00:00+00:00",
        "phase": phase,
        "item": item,
        "attempt": 1,
        "recovery": recovery,
        "kind": kind,
        "code": code,
        "role": "actor",
        "status": None,
        "detail": "",
        "retries": 0,
        **kw,
    }


def test_failures_recorded_and_continued_from_are_listed(copy):
    """Every ``failures.jsonl`` pair outside the stops and re-runs
    (``audit.LEGACY_RECOVERIES``) is one warning per (phase, role, code, kind, recovery),
    listed in ``info.counted_failures``; those pairs and other attempts' rows are not."""
    c = copy(ALF_RB)
    attempt = c.cell(ALF_RB).attempt(1)
    tested = [r["raw_id"] for r in c.rows(ALF_RB, "test")[:2]]
    rows = [
        *(
            _failure_row(
                "test", raw, "harness", "provider_unserved", "score_and_continue"
            )
            for raw in tested
        ),
        _failure_row(
            "learning",
            "3",
            "infra",
            "mirix_batch_lost",
            "accept_failed_batch",
            role="mirix",
        ),
        # What the harness always wrote: a re-run and a model failure it scored.
        _failure_row("test", tested[0], "infra", "upstream_infra", "rerun_item"),
        _failure_row("test", tested[1], "model", "no_answer", "score_and_continue"),
        # Another attempt's row belongs to that attempt.
        {
            **_failure_row(
                "test", tested[0], "harness", "provider_unserved", "score_and_continue"
            ),
            "attempt": 2,
        },
    ]
    write_jsonl(attempt.failures, rows)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    counted = recorded(result, "H10.3:failure_counted_and_continued")
    assert len(counted) == 2 and all(f.severity == audit.WARNING for f in counted)
    assert any("2 test-phase actor failure(s)" in f.detail for f in counted)
    assert any("accept failed batch" in f.detail for f in counted)
    (cell,) = result.cells
    assert [
        (x["phase"], x["kind"], x["recovery"], x["count"])
        for x in cell.info["counted_failures"]
    ] == [
        ("learning", "infra", "accept_failed_batch", 1),
        ("test", "harness", "score_and_continue", 2),
    ]
    assert cell.info["counted_failures"][1]["items"] == sorted(tested)
    # What the method tolerated inside itself is read off its records.
    tolerated = cell.info["tolerated"]
    assert set(tolerated) >= {
        "learning",
        "test",
        "method",
        "refusals",
        "accepted_learning_calls",
    }
    assert json.loads(json.dumps(cell.to_dict()))["info"]["counted_failures"]


def _accept_batch(c: Copy, cell_id: str, call_ids: list[str], **record: Any) -> None:
    """``attempt.json`` ``accepted_learning_calls`` as run.py records a MIRIX batch
    accepted after ``failures.MIRIX_BATCH_ROLLBACKS`` rollbacks, and its
    ``failures.jsonl`` row."""
    attempt = c.cell(cell_id).attempt(1)
    entry = {"checkpoint": "ckpt-0003", "rollbacks": 3, "position": 3, **record}
    c.edit(
        attempt.state,
        lambda s: s.__setitem__(
            "accepted_learning_calls",
            [{"call_id": call_id, **entry} for call_id in call_ids],
        ),
    )
    write_jsonl(
        attempt.failures,
        [
            _failure_row(
                "learning",
                str(entry["position"]),
                "infra",
                "mirix_batch_lost",
                "accept_failed_batch",
                role="mirix",
            )
        ],
    )


def test_a_mirix_batch_accepted_after_three_rollbacks_passes(mirix_matrix, tmp_path):
    """The calls of a MIRIX learning batch lost on infrastructure after run.py rolled its checkpoint back three
    times are accepted. The audit names them "accepted after 3 rollbacks" and the cell
    passes; the same lost calls without the record are still recorded, not blocking."""
    c = Copy(mirix_matrix, tmp_path, ALF_MIRIX)
    template = next(
        r
        for r in c.calls(ALF_MIRIX, role="mirix")
        if (r.get("run_phase") or r.get("phase")) == "learning"
    )
    at = _at(template["ts"]) + timedelta(milliseconds=1)
    lost = [
        _lost_call(
            template, f"lost-{i}", ts=(at + timedelta(microseconds=i)).isoformat()
        )
        for i in (1, 2)
    ]
    c.edit_ledger(ALF_MIRIX, lambda rows: rows.extend(lost))
    unrecorded = c.audit()
    assert violations(unrecorded) == set(), unrecorded.to_markdown()
    assert all(
        "no accepted_learning_calls record" in f.detail
        for f in recorded(unrecorded, "H10.3:learning_infra_unrecovered")
    )

    _accept_batch(c, ALF_MIRIX, ["lost-1", "lost-2"])
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    unrecovered = recorded(result, "H10.3:learning_infra_unrecovered")
    assert len(unrecovered) == 2
    assert all(
        f.severity == audit.WARNING
        and f.downgraded == audit.RECORD
        and "accepted after 3 rollbacks" in f.detail
        for f in unrecovered
    )
    accepted = [
        f
        for f in recorded(result, "H10.3:mirix_learning_infra")
        if "accepted after 3 rollbacks" in f.detail
    ]
    assert len(accepted) == 1
    assert "lost-1" in accepted[0].detail and "lost-2" in accepted[0].detail
    assert "ckpt-0003" in accepted[0].detail
    (cell,) = result.cells
    assert [x["recovery"] for x in cell.info["counted_failures"]] == [
        "accept_failed_batch"
    ]
    assert [
        x["call_id"] for x in cell.info["tolerated"]["accepted_learning_calls"]
    ] == ["lost-1", "lost-2"]
    assert result.exit_code == 0


NO_ENDPOINT_DETAIL = "No endpoints found that support tool use"


def _refused_line(line: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Ledger line ``line`` as the gateway writes an actor call the provider refused
    no-endpoint (gw-404-no-endpoint), under call id ``call_id``."""
    return {
        **line,
        "call_id": call_id,
        "status": "error",
        "http_status": 404,
        "error": {
            "source": "upstream",
            "type": "upstream_rejected",
            "code": 404,
            "message": NO_ENDPOINT_DETAIL,
        },
    }


def _score_unserved(
    c: Copy,
    cell_id: str,
    *,
    primary: float = 0.0,
    ledger: bool = True,
    earlier_tries: int = 1,
) -> str:
    """The first test item as run.py records an item no provider could serve:
    scored ``primary`` as unanswered, flagged, in the attempt, the cell records and its
    trajectory. With ``ledger`` the gateway's ledger backs the flag as run.py leaves it:
    the trajectory's last actor call refused no-endpoint, after
    ``earlier_tries`` unscored tries refused the same way (their rows, trajectories and
    ledger lines). Returns its raw id."""
    cell = c.cell(cell_id)
    attempt = cell.attempt(1)
    rows = jsonl(attempt.results)
    target = rows[0]
    score = {**target["score"], "primary": primary, "answered": False}
    failure = {
        "kind": "harness",
        "code": "provider_unserved",
        "role": "actor",
        "status": 404,
        "detail": NO_ENDPOINT_DETAIL,
        "retries": 0,
    }
    flags = {
        "score": score,
        "failure": failure,
        "end_reason": "provider_unserved",
        "provider_unserved": True,
    }

    def flag(row: dict[str, Any]) -> dict[str, Any]:
        return {**row, **flags} if row["raw_id"] == target["raw_id"] else row

    path = cell.root / target["trajectory"]
    data = json.loads(path.read_text())
    earlier: list[dict[str, Any]] = []
    if ledger:
        lines = jsonl(c.ledger(cell_id))

        def ledger_line(call: Mapping[str, Any]) -> dict[str, Any]:
            (line,) = [
                r
                for r in lines
                if r.get("event") == "call"
                and r.get("call_id") == call["gateway_call_id"]
            ]
            return line

        last = data["calls"][-1]["gateway_call_id"]
        line = ledger_line(data["calls"][-1])
        first = data["calls"][0]
        added: list[dict[str, Any]] = []
        for n in range(earlier_tries):
            # A refused try ends at its first actor call.
            call_id = f"{first['gateway_call_id']}-try{n}"
            added.append(_refused_line(ledger_line(first), call_id))
            rel = f"{target['trajectory']}.try{n}.json"
            tried = {**data, "calls": [{**first, "gateway_call_id": call_id}]}
            (cell.root / rel).write_text(json.dumps(tried))
            earlier.append(
                {
                    **target,
                    "score": None,
                    "trajectory": rel,
                    "recovery": "rerun_item",
                    "failure": {**failure, "code": "upstream_rejected"},
                }
            )
        at = lines.index(line)
        lines[at] = _refused_line(line, last)
        write_jsonl(c.ledger(cell_id), [*lines[:at], *added, *lines[at:]])
        if earlier_tries:
            state = read_json(attempt.state)
            state["test_reruns"] = [
                *state.get("test_reruns", []),
                {"raw_id": target["raw_id"], "task_uid": data.get("task_uid")},
            ]
            write_json_atomic(attempt.state, state)
    write_jsonl(attempt.results, [*earlier, *(flag(r) for r in rows)])
    write_jsonl(cell.results, [flag(r) for r in jsonl(cell.results)])
    c.edit(path, lambda d: d["result"].__setitem__("score", score))
    return str(target["raw_id"])


@pytest.mark.parametrize("cell_id", [ALF_RB])
def test_an_item_no_provider_served_is_scored_zero_and_recorded(copy, cell_id):
    """H9:provider_unserved_scored (gw-404-no-endpoint): the item is scored 0 as
    unanswered, never by the environment: its row is not scored again, and the audit
    lists it. A flagged row with another score is still the scorer's record broken."""
    c = copy(cell_id)
    raw_id = _score_unserved(c, cell_id)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    (found,) = recorded(result, "H9:provider_unserved_scored")
    assert found.severity == audit.WARNING and raw_id in found.detail
    assert result.cells[0].info["provider_unserved"] == {"test": 1, "learning": 0}

    c2 = Copy(c.matrix, c.runs.parent / "scored", cell_id)
    _score_unserved(c2, cell_id, primary=1.0)
    assert violations(c2.audit()) == {"H9:score_record"}


@pytest.mark.parametrize(
    ("backing", "why"),
    [
        ({"ledger": False}, "as ok (HTTP 200), not an upstream refusal"),
        ({"earlier_tries": 0}, "holds 0 earlier try left unscored"),
    ],
)
def test_a_provider_unserved_flag_the_ledger_does_not_back_is_a_violation(
    copy, backing, why
):
    """The flag skips ALFWorld's environment rescore (a validity check) and
    leaves the item out of the report's accuracy without unserved items, so the audit
    holds it to the ledger: the item's last actor call refused upstream no-endpoint, after
    an unscored try refused the same way. A flag the ledger does not back (a served call,
    or a first refusal scored at once) is the scorer's record broken."""
    c = copy(ALF_RB)
    raw_id = _score_unserved(c, ALF_RB, **backing)
    result = c.audit()
    assert violations(result) == {"H9:score_record"}, result.to_markdown()
    (found,) = [
        f for f in recorded(result, "H9:score_record") if f.severity == audit.VIOLATION
    ]
    assert found.item == raw_id and why in found.detail, found.detail


def test_a_learning_stream_in_another_order_is_recorded(copy):
    """A learning stream of train items in another order than the seed-42 stream is
    recorded, as is the live learning.jsonl that played it; an item outside the train
    split or played outside the stream stays a validity check."""
    c = copy(ALF_RB)

    def swap(manifest: dict[str, Any]) -> None:
        items = manifest["cell"]["learning_items"]
        items[0], items[1] = items[1], items[0]

    c.edit(c.cell(ALF_RB).manifest, swap)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert {"H9:learning_items_order", "H9:learning_rows_order"} <= warnings(result)
    assert all(
        f.downgraded == audit.RECORD
        for key in ("H9:learning_items_order", "H9:learning_rows_order")
        for f in recorded(result, key)
    )
    assert result.cells[0].info["recorded_counts"]["H9:learning_items_order"] == 1


def test_operator_notes_run_py_recorded_late_or_without_runs(copy, tmp_path):
    """run.py records a late or unmatched note too (H1.3 recorded, not refused): a note
    recorded after the directory's test phase, and one that names no earlier run here,
    are warnings that say what the note cannot vouch for."""
    c = copy(ALF_NONE)
    note = tmp_path / "rerun-note.md"
    note.write_text("2026-09-30, the operator: re-run on the second host.\n")
    first = _own_first_test_call(c, ALF_NONE)
    entry = _record_note(c, ALF_NONE, note, (first - timedelta(minutes=5)).isoformat())
    assert entry["superseded"] == []
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.3:operator_note_without_runs" in warnings(result)
    assert "H10.3:operator_note_after_test" not in warnings(result)

    def late(manifest: dict[str, Any]) -> None:
        manifest["operator_notes"][0].update(after_test=True, tested_runs=["r-1"])

    c.edit(c.cell(ALF_NONE).manifest, late)
    result = c.audit()
    assert violations(result) == set()
    (found,) = recorded(result, "H10.3:operator_note_after_test")
    assert "(r-1)" in found.detail and found.severity == audit.WARNING


def test_a_tested_attempt_superseded_by_a_new_one_is_listed(copy):
    """``--new-attempt --note`` on a tested cell: the old attempt keeps its
    records, marked superseded; the audit lists it and warns, never mixing its scores
    into the audited attempt's."""
    for marker, by in (
        ({"note": "sandbox image fixed", "at": "2026-09-30T09:00:00+00:00"}, None),
        (
            {
                "note": "sandbox image fixed",
                "at": "2026-09-30T09:00:00+00:00",
                "by_attempt": 1,
            },
            1,
        ),
        (None, None),
    ):
        c = copy(ALF_RB)
        _add_invalid_attempt(
            c, ALF_RB, scored=True, invalidation="superseded", previous="complete"
        )
        state_path = c.cell(ALF_RB).attempt(2).state
        state = read_json(state_path)
        if marker is not None:
            state["superseded"] = marker
        else:
            state["invalid_reason"] = "sandbox image fixed"
        write_json_atomic(state_path, state)
        result = c.audit()
        assert violations(result) == set(), result.to_markdown()
        (found,) = recorded(result, "H10.3:attempt_superseded")
        assert found.severity == audit.WARNING and "sandbox image fixed" in found.detail
        assert ("(not yet recorded)" in found.detail) is (by is None)
        assert "H10.3:tested_attempt_replaced" not in warnings(result)
        (listed,) = result.cells[0].info["superseded_attempts"]
        assert (listed["attempt"], listed["superseded"], listed["scored"]) == (
            2,
            True,
            1,
        )
        assert listed["by_attempt"] == by
        assert listed["note"] == "sandbox image fixed"
        shutil.rmtree(c.runs)


# --- run-level findings name their cells; set-aside directories -----------------------------


def _second_directory(
    c: Copy, cell_id: str, name: str, *, started_at: str, status: str = "complete"
) -> Path:
    """Another directory of ``cell_id`` in the runs directory (a second host's copy of
    the cell, or a re-run in a new directory), its audited attempt started at
    ``started_at`` and its status ``status``."""
    root = c.runs / name
    shutil.copytree(c.cell(cell_id).root, root)
    state_path = root / "attempt-01" / "attempt.json"
    state = read_json(state_path)
    state["started_at"] = started_at
    write_json_atomic(state_path, state)
    write_json_atomic(
        root / "status.json", {**read_json(root / "status.json"), "status": status}
    )
    return root


def test_two_directories_of_one_cell_id_the_complete_latest_is_read(copy):
    """Among directories of one cell id the audit reads the max by (complete,
    started_at, created_at, name): a newer incomplete run is listed as
    ``newer_incomplete``, an older complete one as ``superseded``."""
    c = copy(ALF_NONE)
    own = c.cell(ALF_NONE).root
    started = read_json(c.cell(ALF_NONE).attempt(1).state)["started_at"]
    later = (_at(started) + timedelta(days=1)).isoformat()
    newer = _second_directory(
        c, ALF_NONE, f"{ALF_NONE}.host-b", started_at=later, status="test_incomplete"
    )
    result = c.audit()
    assert [Path(x.root).name for x in result.cells] == [own.name]
    assert [(Path(d.root).name, d.role, d.complete) for d in result.duplicates] == [
        (newer.name, "newer_incomplete", False)
    ]
    assert violations(result) == set(), result.to_markdown()
    (duplicate,) = recorded(result, "H10.3:duplicate_cell_directories")
    assert (
        "(newer_incomplete)" in duplicate.detail and "complete one" in duplicate.detail
    )

    # The newer run completes: it is read, the first directory is superseded.
    write_json_atomic(
        newer / "status.json",
        {**read_json(newer / "status.json"), "status": "complete"},
    )
    result = c.audit()
    assert [Path(x.root).name for x in result.cells] == [newer.name]
    assert [(Path(d.root).name, d.role) for d in result.duplicates] == [
        (own.name, "superseded")
    ]
    assert result.to_dict()["duplicates"][0]["role"] == "superseded"
    assert result.cells[0].cell_id == ALF_NONE


def test_an_incomplete_directory_is_dated_by_its_last_attempt(copy):
    """A directory that is not complete is dated by its last attempt
    (``report.directory_started_at``), even when an earlier attempt of it completed; a
    complete one by its reported attempt."""
    c = copy(ALF_NONE)
    own = c.cell(ALF_NONE).root
    started = _at(read_json(c.cell(ALF_NONE).attempt(1).state)["started_at"])
    other = _second_directory(
        c,
        ALF_NONE,
        f"{ALF_NONE}.host-b",
        started_at=(started - timedelta(days=1)).isoformat(),
        status="test_incomplete",
    )
    # Its attempt 1 completed a day before the audited run; attempt 2 is still running.
    second = other / "attempt-02"
    second.mkdir()
    state = read_json(other / "attempt-01" / "attempt.json")
    write_json_atomic(
        second / "attempt.json",
        {
            **state,
            "state": "test_incomplete",
            "started_at": (started + timedelta(days=1)).isoformat(),
        },
    )
    result = c.audit()
    assert [Path(x.root).name for x in result.cells] == [own.name]
    (duplicate,) = result.duplicates
    assert (duplicate.role, duplicate.complete) == ("newer_incomplete", False)
    assert duplicate.started_at == (started + timedelta(days=1)).isoformat()
    facts = audit.directory_facts(other)
    assert facts.started == started + timedelta(days=1) and not facts.complete

    # Complete (status complete): dated by its reported attempt, the highest complete.
    write_json_atomic(
        other / "status.json",
        {**read_json(other / "status.json"), "status": "complete"},
    )
    assert audit.directory_facts(other).started == started - timedelta(days=1)
    result = c.audit()
    assert [Path(x.root).name for x in result.cells] == [own.name]
    assert [(d.role, d.complete) for d in result.duplicates] == [("superseded", True)]


def _other_method_copy(c: Copy, source: str, method: str) -> str:
    """``source``'s records under ``method``'s cell id, ``cell.json`` naming it."""
    own = read_json(c.cell(source).manifest)["cell"]["method"]
    cell_id = source.replace(f"-{own}", f"-{method}", 1)
    shutil.copytree(c.cell(source).root, c.runs / cell_id)
    c.edit(
        c.cell(cell_id).manifest,
        lambda m: m["cell"].update(method=method, cell_id=cell_id),
    )
    return cell_id


def test_a_cell_of_a_method_not_in_the_matrix_is_set_aside(copy):
    """H10.3-role-policy: a cell of a method ``protocol.METHODS`` does not list and
    no removal record names is set aside with one warning per method, never audited."""
    c = copy(ALF_NONE, ALF_AWM)
    other = _other_method_copy(c, ALF_AWM, "voyager")
    assert "voyager" not in protocol.METHODS and "voyager" not in study.REMOVED_METHODS
    result = c.audit()
    assert sorted(x.cell_id for x in result.cells) == sorted([ALF_NONE, ALF_AWM])
    assert violations(result) == set() and result.exit_code == 0
    (warned,) = recorded(result, "H10.3:method_not_in_matrix")
    assert warned.severity == audit.WARNING and warned.cells == (other,)
    assert "'voyager'" in warned.detail
    assert [(r.cell_id, r.method) for r in result.not_in_matrix] == [(other, "voyager")]
    data = result.to_dict()
    assert data["not_in_matrix"]["methods"] == {"voyager": [other]}
    assert data["not_in_matrix"]["cells"][0]["cell_id"] == other
    assert "## Methods not in the matrix" in result.to_markdown()
    assert "cell(s) of methods not in the matrix set aside" in result.to_markdown()


def _removal_manifest(
    base: Path,
    removal_id: str,
    method: str,
    cells: list[dict[str, Any]],
    *,
    state: str = "complete",
    hostname: str = "host-a",
) -> Path:
    """A removal manifest of ``method`` as ``tools/remove_method_data.py`` writes one
    under ``base/_removed/<method>/<removal_id>/``."""
    root = base / audit.REMOVED_DATA_DIR / method / removal_id
    root.mkdir(parents=True)
    write_json_atomic(
        root / "manifest.json",
        {
            "schema": 1,
            "kind": "skill_bench.removed_method_data",
            "removal_id": removal_id,
            "method": method,
            "state": state,
            "host": {"hostname": hostname},
            "started_at": "2026-09-30T11:59:00+00:00",
            "completed_at": "2026-09-30T12:00:00+00:00"
            if state == "complete"
            else None,
            "cells": cells,
        },
    )
    (root / "calls.jsonl").write_text("")
    return root / "manifest.json"


def test_a_deleted_removed_cell_still_counts_by_its_manifest(copy, removed_method):
    """Once ``tools/remove_method_data.py`` deleted a removed method's cell
    directory, its removal manifest lists the cell: the cell is listed under
    ``removed.deleted``, a mirror of the same removal in a host's ``_hosts/`` directory
    is read once, a removal still ``deleting`` counts, and a manifest that cannot be
    read is a warning."""
    c = copy(ALF_NONE, ALF_AWM)
    dc = ALF_AWM.replace("-awm", f"-{removed_method}", 1)
    entry = {
        "cell_id": dc,
        "method": removed_method,
        "status": "complete",
        "split_mode": "smoke",
        "cell_dir": dc,
    }
    manifest = _removal_manifest(c.runs, "r-0001", removed_method, [entry])
    mirror = c.runs / audit.HOSTS_DIR / "host-a"
    _removal_manifest(mirror, "r-0001", removed_method, [entry])
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert result.exit_code == 0 and result.removed == []
    (deleted,) = result.deleted
    assert {k: deleted[k] for k in ("method", "cell_id", "status", "host")} == {
        "method": removed_method,
        "cell_id": dc,
        "status": "complete",
        "host": "host-a",
    }
    assert deleted["deleted_at"] == "2026-09-30T12:00:00+00:00"
    assert deleted["manifest"] == manifest.relative_to(c.runs).as_posix()
    data = result.to_dict()
    assert data["removed"]["deleted"][0]["cell_id"] == dc
    assert data["removed"]["methods"][removed_method]["cells"] == [dc]
    md = result.to_markdown()
    assert (
        "## Removed methods" in md and f"| `{dc}` | deleted 2026-09-30T12:00:00" in md
    )

    # Selected like any cell: another cell selection leaves it out.
    assert c.audit(cells=[ALF_NONE]).deleted == []

    # A removal still deleting counts; an unreadable manifest is a warning.
    _removal_manifest(
        c.runs,
        "r-0002",
        removed_method,
        [{**entry, "cell_id": dc + "-2"}],
        state="deleting",
    )
    for base in (c.runs, mirror):  # the host's mirror of it: one warning
        broken = base / audit.REMOVED_DATA_DIR / removed_method / "r-0003"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{")
    result = c.audit()
    assert sorted(d["cell_id"] for d in result.deleted) == [dc, dc + "-2"]
    (unreadable,) = [
        f for f in result.findings if f.check == "removed_manifest_unreadable"
    ]
    assert unreadable.severity == audit.WARNING and "r-0003" in unreadable.detail
    assert str(c.runs) not in unreadable.detail  # named relative to the runs dir


def test_a_cell_the_machine_cannot_check_is_not_passed(
    copy, tmp_path, capsys, monkeypatch
):
    """Fail closed: a cell whose items or environment this machine cannot
    load is neither passed nor failed: its checks are ``not checked``, the audit exits
    2 and says so, in the report and on the command line."""
    c = copy(ALF_RB, WS_RB)

    def missing(benchmark: str) -> Any:
        if benchmark == "alfworld":
            raise FileNotFoundError("the ALFWorld data is not on this machine")
        return c.matrix.env_factory(benchmark)

    result = audit.audit_runs(
        c.runs, env_factory=missing, split_mode=None, expected_upstream=TEST_UPSTREAM
    )
    cells = {x.cell_id: x for x in result.cells}
    cell = cells[ALF_RB]
    assert cell.audited and cell.not_checked > 0 and not cell.passed
    assert cells[WS_RB].passed and cells[WS_RB].not_checked == 0
    assert violations(result) == set(), result.to_markdown()
    # Without the items every trajectory would look unknown: not checked, not unknown.
    assert not recorded(result, "H10.1:unknown_item")
    assert {
        f"{f.invariant}:{f.check}"
        for f in cell.findings
        if f.severity == audit.NOT_CHECKED
    } <= set(audit.NOT_CHECKED_CHECKS)
    assert result.not_checked == [ALF_RB]
    assert (result.passed, result.exit_code) == (False, 2)
    assert "not checked" in cell.invariants().values()
    data = result.to_dict()
    assert data["not_checked"] == [ALF_RB] and data["passed"] is False
    md = result.to_markdown()
    assert "1 cell(s) not checked" in md and "- not checked H10.1" in md

    real = audit.audit_runs
    monkeypatch.setattr(
        audit, "audit_runs", lambda *a, **kw: real(*a, env_factory=missing, **kw)
    )
    out = tmp_path / "cli"
    code = audit.main(
        [
            "--runs-dir",
            str(c.runs),
            "--split-mode",
            "all",
            "--expected-upstream",
            TEST_UPSTREAM,
            "--out",
            str(out),
        ]
    )
    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["not_checked"] == [ALF_RB] and printed["passed"] is False


# --- versions: what a check reads, per cell ---------------------------------------------


def _earlier_version(
    known: dict[str, dict[str, Any]],
    *,
    files: Mapping[str, str] | None = None,
    **edit: Any,
) -> str:
    """Add to ``known`` an earlier version as ``freeze_protocol`` records one: the
    frozen constants with ``edit`` and the frozen file map with ``files``, under the
    fingerprint both give. Returns its fingerprint."""
    frozen = known[protocol.frozen_fingerprint()]
    constants = {**frozen["constants"], **edit}
    file_map = {**frozen["files"], **(files or {})}
    fingerprint = protocol.fingerprint_of(constants, file_map)
    known[fingerprint] = {
        "fingerprint": fingerprint,
        "frozen_at": EARLIER_FROZEN_AT,
        "reason": "an earlier freeze",
        "git": {"commit": "0a1b2c3", "dirty": False},
        "constants_sha256": protocol.constants_sha256(constants),
        "constants": constants,
        "inconsistent": False,
        "files": file_map,
        "current": False,
    }
    return fingerprint


def _not_reverified(cell: audit.CellAudit) -> dict[str, list[str]]:
    return {e["check"]: e["inputs"] for e in cell.info["not_reverified"]}


ALF_TEMPLATE = "envs/templates/alfworld/rollout_no_history.md"
WS_TEMPLATE = "envs/templates/webshop/rollout_no_history.md"


def test_no_auditor_tree_on_a_tree_that_is_not_frozen(copy, monkeypatch):
    """An auditing tree that is not the frozen value is no run-level violation. Each
    cell lists the checks that read what
    differs (``info.not_reverified``), and their findings are warnings."""
    c = copy(ALF_RB)
    real = protocol.fingerprint_manifest()
    assert (
        protocol.fingerprint_of(real["constants"], real["files"]) == real["fingerprint"]
    )
    files = {**real["files"], ALF_TEMPLATE: "1" * 64}
    drifted = {
        **real,
        "files": files,
        "fingerprint": protocol.fingerprint_of(real["constants"], files),
    }
    monkeypatch.setattr(protocol, "fingerprint_manifest", lambda *a, **kw: drifted)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert not [
        f
        for f in [*result.findings, *(f for x in result.cells for f in x.findings)]
        if f.check == "auditor_tree"
    ]
    assert result.tree_current is False
    assert result.auditor_fingerprint == drifted["fingerprint"]
    (cell,) = result.cells
    listed = _not_reverified(cell)
    for key in ("H10.2:memory_slot", "H10.2:outside_memory_slot"):
        assert any(ALF_TEMPLATE in i for i in listed[key]), listed
    summary = _finding(result, "protocol_version_unauditable")
    assert summary.severity == audit.WARNING and ALF_TEMPLATE in summary.detail
    assert "the auditing tree's protocol" in summary.detail
    assert "## Not re-verified under the cell's own version" in result.to_markdown()
    assert "Auditor tree = frozen value: no" in result.to_markdown()

    # A memory outside its slot is a finding of such a check: not re-verified.
    _change_first_request(
        c, c.trajectory(ALF_RB, "test"), _append_to(0, "\n(memory here)"), rehash=True
    )
    result = c.audit()
    assert violations(result) == set()
    (moved,) = recorded(result, "H10.2:memory_slot")
    assert moved.severity == audit.WARNING and moved.downgraded == audit.NOT_REVERIFIED
    assert "(not re-verified under its version)" in result.to_markdown()
    entry = next(
        e
        for e in result.cells[0].info["not_reverified"]
        if e["check"] == "H10.2:memory_slot"
    )
    assert entry["findings"] == 1
    assert result.exit_code == 0


def test_a_changed_webshop_template_is_not_re_verified_only_for_webshop_cells(copy):
    """What a check reads is its benchmark's: an earlier version whose WebShop
    prompt template differs leaves the WebShop cell's prompt checks not re-verified, and
    an ALFWorld cell of the same version keeps every check."""
    c = copy(ALF_RB, WS_RB)
    known = protocol.known_versions()
    earlier = _earlier_version(known, files={WS_TEMPLATE: "2" * 64})
    for cell_id in (ALF_RB, WS_RB):
        _set_cell_protocol(c, cell_id, earlier)
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    cells = {x.cell_id: x for x in result.cells}
    listed = _not_reverified(cells[WS_RB])
    for key in ("H10.2:memory_slot", "H10.2:outside_memory_slot"):
        assert any(WS_TEMPLATE in i for i in listed[key]), (key, listed)
    assert _not_reverified(cells[ALF_RB]) == {}
    unauditable = [f for f in recorded(result, "H10.6:protocol_version_unauditable")]
    assert [f.cell_id for f in unauditable] == [WS_RB]
    assert {f.cell_id for f in recorded(result, "H10.6:protocol_version")} == {
        ALF_RB,
        WS_RB,
    }


def test_a_changed_gateway_constant_needs_no_re_verification(copy):
    """A version that differs only in a gateway constant (the call deadline) leaves
    every check as it is: the calls are judged against their own gateway start records."""
    c = copy(ALF_RB)
    known = protocol.known_versions()
    deadline = known[protocol.frozen_fingerprint()]["constants"]["call_deadline_s"]
    earlier = _earlier_version(known, call_deadline_s=deadline + 60)
    _set_cell_protocol(c, ALF_RB, earlier)
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    assert result.cells[0].info["not_reverified"] == []
    assert "H10.6:protocol_version_unauditable" not in warnings(result)
    assert "call_deadline_s" in _finding(result, "protocol_version").detail
    # The gateway ran this tree's deadline, not the version's: recorded, one warning.
    assert "H10.6:gateway_settings_differ" in warnings(result)
    assert result.cells[0].info["provider_policy"]["source"] == "attempt version"


def test_a_version_without_a_file_map_downgrades_every_tree_check(copy):
    """Fail closed: a version frozen before the record kept file maps differs from
    this tree in everything a check reads: its findings are not re-verified warnings,
    and the summary says the file map is not recorded."""
    c = copy(ALF_RB)
    legacy = _earlier_versions()
    legacy[EARLIER] = {**legacy[EARLIER], "files": None, "constants_sha256": None}
    _set_cell_protocol(c, ALF_RB, EARLIER)
    _change_first_request(
        c, c.trajectory(ALF_RB, "test"), _append_to(0, "\n(memory here)"), rehash=True
    )
    result = c.audit(versions=legacy)
    assert violations(result) == set(), result.to_markdown()
    (moved,) = recorded(result, "H10.2:memory_slot")
    assert moved.downgraded == audit.NOT_REVERIFIED
    listed = _not_reverified(result.cells[0])
    assert set(audit.TREE_INPUTS) - {
        "H10.2:conversation_not_append_only",  # WebShop's
    } <= set(listed)
    assert "H10.2:conversation_not_append_only" not in listed
    assert (
        "a file map that is not recorded"
        in _finding(result, "protocol_version_unauditable").detail
    )


def _carry_version(c: Copy, cell_id: str, *, tamper: bool = False) -> str:
    """The cell ran under a version another host froze: ``attempt.json`` ``protocol``
    keeps its constants and file map (one method prompt changed), which reproduce its
    fingerprint unless ``tamper`` edits a file hash after the fact."""
    frozen = protocol.load_frozen_protocol()
    constants = dict(frozen["constants"])
    files = dict(frozen["files"])
    prompt = next(p for p in sorted(files) if p.startswith("methods/prompts/"))
    files[prompt] = "3" * 64
    fingerprint = protocol.fingerprint_of(constants, files)
    _set_cell_protocol(c, cell_id, fingerprint)
    if tamper:
        files[ALF_TEMPLATE] = "4" * 64

    def carry(state: dict[str, Any]) -> None:
        state["protocol"] = {
            "fingerprint": fingerprint,
            "constants_sha256": protocol.constants_sha256(constants),
            "constants": constants,
            "files": files,
        }

    c.edit(c.cell(cell_id).attempt(1).state, carry)
    return fingerprint


def test_an_auto_frozen_version_carried_by_the_attempt_record(copy):
    """A version ``config/protocol_frozen.json`` of this checkout does not register
    but the attempt's own record describes (its constants and file map reproduce its
    fingerprint) is audited as it ran: a ``protocol_version`` warning, no
    ``fingerprint`` finding. A record that does not reproduce its fingerprint is never
    read: the version is not registered."""
    c = copy(ALF_RB)
    fingerprint = _carry_version(c, ALF_RB)
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert "H10.6:fingerprint" not in warnings(result)
    found = _finding(result, "protocol_version")
    assert "registered only in its own attempt record" in found.detail
    assert fingerprint[:12] in found.detail
    (cell,) = result.cells
    assert cell.info["protocol_record"] == "attempt record"
    assert cell.info["protocol"]["known"] is False
    assert cell.info["not_reverified"] == []

    c2 = Copy(c.matrix, c.runs.parent / "tampered", ALF_RB)
    _carry_version(c2, ALF_RB, tamper=True)
    result = c2.audit()
    assert violations(result) == set()
    assert "H10.6:fingerprint" in warnings(result)
    assert result.cells[0].info["protocol_record"] is None
    assert result.cells[0].info["not_reverified"]


def test_a_resume_across_versions_is_recorded_not_refused(copy):
    """Protocol versions are recorded per attempt: run.py resumed the attempt's test
    phase under another protocol and harness and recorded a version segment for it. The
    work line under the new version is explained by the segment (no
    ``protocol_changed_within_attempt``, no ``model_visible_code_changed``); the attempt
    is one ``mixed_version_attempt`` warning and ``info.version_segments`` lists it."""
    c = copy(ALF_RB)
    paths = c.cell(ALF_RB)
    rows = jsonl(paths.launches)
    work = _work_line(rows)
    state = read_json(paths.attempt(1).state)
    prompt = next(p for p in sorted(EARLIER_FILES) if p.startswith("methods/prompts/"))
    later = {
        **work,
        "launch": "0123456789ab",
        "at": "2999-01-01T00:00:00+00:00",
        "phase": "test",
        "segment": 1,
        "protocol": {**work["protocol"], "fingerprint": EARLIER},
        "code": {**work["code"], "harness": "f" * 64},
    }
    write_jsonl(paths.launches, [*rows, later])
    first = {
        "index": 0,
        "legacy": False,
        "why": "first attempt",
        "phase": "learning",
        "first_launch": work["launch"],
        "started_at": state["started_at"],
        "launches": [work["launch"]],
        "learning_from": None,
        "test_from": None,
        "protocol": dict(state["protocol"]),
        "code": dict(state["code"]),
        "git": state.get("git"),
        "host": state.get("host"),
        "tree": None,
        "version_file": None,
        "auto_freeze": None,
        "known_version": True,
        "note": None,
        "changed": None,
    }
    second = {
        **first,
        "index": 1,
        "why": "test resume of attempt 1",
        "phase": "test",
        "first_launch": later["launch"],
        "started_at": later["at"],
        "launches": [later["launch"]],
        "test_from": {"scored": 2, "pending": 1},
        "protocol": {"fingerprint": EARLIER, "constants_sha256": None},
        "code": {**state["code"], "harness": "f" * 64},
        "changed": {
            "protocol": True,
            "protocol_diff": {
                "changed": [prompt],
                "added": [],
                "removed": [],
                "constants_changed": False,
                "constants_changed_keys": [],
            },
            "code": ["harness"],
            "code_files": None,
            "commit": False,
            "tree": False,
        },
    }
    c.edit(
        paths.attempt(1).state,
        lambda s: s.__setitem__("version_segments", [first, second]),
    )
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set(), result.to_markdown()
    found = warnings(result)
    assert "H10.6:mixed_version_attempt" in found
    assert (
        not {
            "H10.6:protocol_changed_within_attempt",
            "H10.6:model_visible_code_changed",
        }
        & found
    )
    mixed = _finding(result, "mixed_version_attempt")
    assert "the test phase (2 scored, 1 pending)" in mixed.detail
    assert f"protocol files {prompt}" in mixed.detail
    (segment,) = result.cells[0].info["version_segments"]
    assert {k: segment[k] for k in ("segment", "phase", "fingerprint", "known")} == {
        "segment": 1,
        "phase": "test",
        "fingerprint": EARLIER,
        "known": True,
    }
    assert segment["from"] == {"scored": 2, "pending": 1}
    md = result.to_markdown()
    assert "| cell | from segment | at | phase | protocol | harness |" in md
    assert "+1 later version(s)" in md

    # The same work line with no segment recording its version: recorded.
    c.edit(paths.attempt(1).state, lambda s: s.__setitem__("version_segments", [first]))
    result = c.audit(versions=_earlier_versions())
    assert violations(result) == set()
    assert {
        "H10.6:protocol_changed_within_attempt",
        "H10.6:model_visible_code_changed",
    } <= warnings(result)


def test_a_later_provider_policy_never_fails_old_calls(copy, monkeypatch):
    """H10.6-gateway-settings: every call is held to the settings its own gateway
    started with, the attempt's version: a provider policy this tree changed later
    fails none of them. The gateway's start is one warning that names the policy."""
    c = copy(ALF_RB)
    later = {protocol.MODELS["luna"]: {"order": ["openai"], "allow_fallbacks": False}}
    monkeypatch.setattr(
        audit,
        "PROTOCOL_SETTINGS",
        {**audit.PROTOCOL_SETTINGS, "provider_policy": later},
    )
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    assert (
        "provider_policy: start" in _finding(result, "gateway_settings_differ").detail
    )
    info = result.cells[0].info["provider_policy"]
    assert info["source"] == "attempt version"
    assert info["policy"] == {}


def _settings_unrecorded(c: Copy, cell_id: str) -> dict[str, dict[str, Any]]:
    """Make ``cell_id`` an attempt of which nothing records the protocol settings: an
    earlier version without per-key constants (no known version shares its constants
    hash), whose gateways started under another fingerprint. Returns the
    versions to audit with."""
    known = protocol.known_versions()
    frozen = protocol.frozen_fingerprint()
    legacy = {
        **known,
        EARLIER: {
            **known[frozen],
            "fingerprint": EARLIER,
            "constants": None,
            "constants_sha256": "4" * 64,
            "current": False,
        },
    }
    _set_cell_protocol(c, cell_id, EARLIER)
    return legacy


def _with_effort(c: Copy, call_id: str, effort: str, start_id: str | None = None):
    """Rewrite call ``call_id`` as sent with reasoning ``effort`` (its request hash kept
    consistent), served by gateway start ``start_id`` when given."""

    def change(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("call_id") != call_id:
                continue
            if start_id is not None:
                row["gateway_start"] = start_id
            if row.get("event") == "call":
                row["wire"]["reasoning"] = {"effort": effort}
                request = c.recorded_request(ALF_RB, call_id)
                row["request_sha256"] = audit.sent_request_sha256(request, row["wire"])

    c.edit_ledger(ALF_RB, change)


def test_unrecorded_settings_downgrade_only_the_calls_no_start_record_covers(copy):
    """When nothing of the attempt records its protocol settings, only the
    calls without a gateway start record are judged against this tree's settings and not
    re-verified. A call its start record covers is held to that record as usual: a
    reasoning effort that breaks it stays a violation."""
    c = copy(ALF_RB)
    versions = _settings_unrecorded(c, ALF_RB)
    actors = c.calls(ALF_RB, role="actor", phase="test")
    orphan, covered = actors[0], actors[1]
    # The orphan's gateway start is unknown; it went out with its protocol's effort.
    c.edit_ledger(ALF_RB, _set_on_call(orphan["call_id"], gateway_start="z" * 16))
    result = c.audit(versions=versions)
    cell = result.cells[0]
    assert cell.info["provider_policy"]["source"] == "unknown"
    assert "the attempt's protocol settings are not recorded" in (
        _not_reverified(cell).get("H10.6:reasoning_effort") or []
    )
    assert "H10.6:reasoning_effort" not in violations(result), result.to_markdown()
    # A covered call that breaks its own start record: a violation, not a warning.
    _with_effort(c, covered["call_id"], "high")
    result = c.audit(versions=versions)
    assert "H10.6:reasoning_effort" in violations(result), result.to_markdown()
    (found,) = recorded(result, "H10.6:reasoning_effort")
    assert found.downgraded is None and covered["call_id"] in found.detail
    # The orphan breaking this tree's effort: a not-re-verified warning, alone.
    _with_effort(c, covered["call_id"], protocol.REASONING_EFFORT)
    _with_effort(c, orphan["call_id"], "high")
    result = c.audit(versions=versions)
    assert "H10.6:reasoning_effort" not in violations(result), result.to_markdown()
    (found,) = recorded(result, "H10.6:reasoning_effort")
    assert (
        found.downgraded == audit.NOT_REVERIFIED and orphan["call_id"] in found.detail
    )


def test_a_start_is_held_to_this_trees_settings_when_the_attempt_records_none(copy):
    """With no settings recorded for the attempt, a gateway that started with another
    reasoning effort is not skipped: gateway_start_settings compares it
    with this tree's settings and lists it as not re-verified."""
    c = copy(ALF_RB)
    versions = _settings_unrecorded(c, ALF_RB)
    starts_path = c.runs / "_gateway" / "starts.jsonl"
    starts = jsonl(starts_path)
    target = c.calls(ALF_RB, role="actor", phase="test")[0]
    original = next(s for s in starts if s.get("start_id") == target["gateway_start"])
    second = {
        **original,
        "start_id": "b" * 16,
        "settings": {**original["settings"], "reasoning_effort": "high"},
    }
    write_jsonl(starts_path, [*starts, second])
    _with_effort(c, target["call_id"], "high", second["start_id"])
    result = c.audit(versions=versions)
    assert result.cells[0].info["provider_policy"]["source"] == "unknown"
    assert violations(result) == set(), result.to_markdown()
    (found,) = recorded(result, "H10.6:gateway_start_settings")
    assert found.downgraded == audit.NOT_REVERIFIED
    assert (
        "reasoning_effort='high'" in found.detail and "this tree says" in found.detail
    )
    listed = _not_reverified(result.cells[0])
    assert listed["H10.6:gateway_start_settings"] == [
        "the attempt's protocol settings are not recorded"
    ]


def test_a_call_is_held_to_its_own_gateway_start(copy):
    """A call is compared with the start record of the gateway that served it. A
    second gateway started with another reasoning effort served one call with that
    effort: the call matches its gateway, and the gateway's model settings, which the
    attempt's version does not record, are the violation. The same call under the
    first gateway breaks its own start record."""
    c = copy(ALF_RB)
    starts_path = c.runs / "_gateway" / "starts.jsonl"
    starts = jsonl(starts_path)
    target = c.calls(ALF_RB, role="actor", phase="test")[0]
    served = str(target["gateway_start"])
    original = next(s for s in starts if s.get("start_id") == served)
    second = {
        **original,
        "start_id": "b" * 16,
        "settings": {**original["settings"], "reasoning_effort": "high"},
    }
    write_jsonl(starts_path, [*starts, second])

    def move(start_id: str) -> Callable[[list[dict[str, Any]]], None]:
        def change(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                if row.get("call_id") != target["call_id"]:
                    continue
                row["gateway_start"] = start_id
                if row.get("event") == "call":
                    row["wire"]["reasoning"] = {"effort": "high"}
                    request = c.recorded_request(ALF_RB, target["call_id"])
                    row["request_sha256"] = audit.sent_request_sha256(
                        request, row["wire"]
                    )

        return change

    c.edit_ledger(ALF_RB, move(second["start_id"]))
    result = c.audit()
    assert violations(result) == {"H10.6:gateway_start_settings"}, result.to_markdown()
    assert (
        "reasoning_effort='high'" in _finding(result, "gateway_start_settings").detail
    )
    c.edit_ledger(ALF_RB, move(served))
    result = c.audit()
    assert violations(result) == {"H10.6:reasoning_effort"}
    assert (
        "its gateway start record says effort 'medium'"
        in _finding(result, "reasoning_effort").detail
    )


def test_a_first_prompt_across_template_versions_is_a_warning(copy):
    """H10.2-first-prompt-across-cells: cells whose versions differ in the
    benchmark's templates may differ in an item's no-memory first prompt: a warning the
    report flags, never a violation of either cell."""
    c = copy(ALF_NONE, ALF_RB)
    known = protocol.known_versions()
    earlier = _earlier_version(known, files={ALF_TEMPLATE: "5" * 64})
    _set_cell_protocol(c, ALF_RB, earlier)
    c.edit(
        c.trajectory(ALF_RB, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    result = c.audit(versions=known)
    assert violations(result) == set(), result.to_markdown()
    (across,) = [
        f for f in result.findings if f.check == "first_prompt_differs_across_versions"
    ]
    assert across.severity == audit.WARNING
    assert set(across.cells) == {ALF_NONE, ALF_RB}
    cells = {x.cell_id: x for x in result.cells}
    assert cells[ALF_NONE].info["prompt_key"] != cells[ALF_RB].info["prompt_key"]
    assert all(x.passed for x in result.cells)


def test_one_odd_cell_out_of_three_is_the_one_blamed(copy):
    """Among cells of one prompt key, the cell whose no-memory first prompt differs
    from the majority fails; the others pass."""
    c = copy(ALF_NONE, ALF_RB, ALF_AWM)
    c.edit(
        c.trajectory(ALF_AWM, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    result = c.audit()
    assert "H10.2:first_prompt_differs_across_cells" in violations(result)
    (across,) = [
        f for f in result.findings if f.check == "first_prompt_differs_across_cells"
    ]
    assert across.cells == (ALF_AWM,)
    cells = {x.cell_id: x for x in result.cells}
    assert not cells[ALF_AWM].passed
    assert cells[ALF_NONE].passed and cells[ALF_RB].passed


def test_cells_of_one_fingerprint_without_a_prompt_key_are_still_compared(copy):
    """A version without a recorded file map has no prompt key, but cells
    that ran that one fingerprint share its templates, splits and seed: the odd cell's
    no-memory first prompt is still a violation (first_prompt_differs_across_cells),
    not a warning across versions."""
    c = copy(ALF_NONE, ALF_RB, ALF_AWM)
    legacy = _earlier_versions()
    legacy[EARLIER] = {**legacy[EARLIER], "files": None, "constants_sha256": None}
    for cell_id in (ALF_NONE, ALF_RB, ALF_AWM):
        _set_cell_protocol(c, cell_id, EARLIER)
    c.edit(
        c.trajectory(ALF_AWM, "test"),
        lambda d: d.__setitem__("first_prompt_no_memory_sha256", "0" * 64),
    )
    result = c.audit(versions=legacy)
    cells = {x.cell_id: x for x in result.cells}
    assert all(x.info["prompt_key"] is None for x in cells.values())
    assert {x.info["prompt_fingerprint"] for x in cells.values()} == {EARLIER}
    (across,) = [
        f for f in result.findings if f.check == "first_prompt_differs_across_cells"
    ]
    assert across.severity == audit.VIOLATION and across.cells == (ALF_AWM,)
    assert not [
        f for f in result.findings if f.check == "first_prompt_differs_across_versions"
    ]
    assert not cells[ALF_AWM].passed
    assert cells[ALF_NONE].passed and cells[ALF_RB].passed


def test_a_cell_without_a_prompt_key_joins_its_fingerprints_key():
    """A cell of one fingerprint whose key is unknown shares the group of a
    cell of the same fingerprint whose key is known; cells of several versions and no
    key stay groups of their own."""

    def cell(cell_id: str, key: str | None, fingerprint: str | None) -> Any:
        return types.SimpleNamespace(
            cell_id=cell_id,
            info={"prompt_key": key, "prompt_fingerprint": fingerprint},
        )

    groups = audit._prompt_groups(
        [
            cell("a", "k1", "f1"),
            cell("b", None, "f1"),
            cell("c", None, "f2"),
            cell("d", None, None),
        ]
    )
    assert groups == {"a": "k1", "b": "k1", "c": "fp:f2", "d": None}


# --- recorded, not blocking: rolled-back tries, flagged retrievals -------------------------


def test_an_unreadable_discarded_try_after_an_exact_restore_is_a_warning(resumed):
    """H10.1-discarded-scan: a discarded trajectory that cannot be read was not
    scanned; when the resume that discarded it restored its checkpoint exactly, the
    rolled-back try left nothing in the memory: a warning. Without that proof it stays
    a violation."""
    c = resumed(ALF_AWM)
    path = _discarded(c, ALF_AWM)[0]
    path.write_text("{")
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    (found,) = recorded(result, "H10.1:trajectory_missing")
    assert found.severity == audit.WARNING and "restored its checkpoint exactly" in (
        found.detail
    )
    _rewrite_segment(
        c, ALF_AWM, lambda row: row["restore"].__setitem__("files_equal", False)
    )
    result = c.audit()
    assert "H10.1:trajectory_missing" in violations(result)


def test_a_flagged_degraded_mirix_retrieval_is_recorded(
    mirix_degraded_matrix, tmp_path
):
    """Recorded: MIRIX injected a degraded retrieval
    after its retry limit and run.py flagged the row ``retrieval_degraded``. The item is
    scored on it: a warning per item, the retrieval's lost calls are explained by it,
    and ``info.tolerated`` lists the items."""
    matrix, _ = mirix_degraded_matrix
    c = Copy(matrix, tmp_path, ALF_MIRIX)
    path = _retrieval_records(c, ALF_MIRIX)
    records = jsonl(path)
    degraded = {r["task_uid"] for r in records if r["frozen"] and r["failure"]}
    assert len(degraded) == 2
    kept = []
    for record in records:
        if record["frozen"] and record["task_uid"] in degraded:
            if not record["failure"]:
                continue  # the clean retry: the method injected the degraded one
            record["failure"] = None
        kept.append(record)
    write_jsonl(path, kept)
    cell = c.cell(ALF_MIRIX)

    def flag(row: dict[str, Any]) -> dict[str, Any]:
        if row["task_uid"] in degraded and row["score"] is not None:
            return {**row, "retrieval_degraded": True}
        return row

    write_jsonl(
        cell.attempt(1).results, [flag(r) for r in jsonl(cell.attempt(1).results)]
    )
    write_jsonl(cell.results, [flag(r) for r in jsonl(cell.results)])
    result = c.audit()
    assert violations(result) == set(), result.to_markdown()
    found = recorded(result, "H10.3:mirix_degraded_retrieval_scored")
    assert len(found) == 2 and all(
        f.severity == audit.WARNING and "flagged retrieval_degraded" in f.detail
        for f in found
    )
    assert "H10.3:mirix_test_infra_unrecovered" not in warnings(result)
    info = result.cells[0].info
    assert info["mirix_test_retrievals"]["degraded_scored"] == 2
    assert info["mirix_test_retrievals"]["lost_calls_unexplained"] == 0
    assert len(info["tolerated"]["test"]["retrieval_degraded"]) == 2
