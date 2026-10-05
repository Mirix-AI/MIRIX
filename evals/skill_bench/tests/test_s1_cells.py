"""ReasoningBank and AWM in a whole cell run by ``run.py``.

The real runner, rollout, actor and a conformant loopback gateway over the fake upstream;
the benchmark is ``run_support.TinyEnv``. The methods are built by
``method_class`` exactly as ``run.py`` builds them, with no test transport: every J,
learner and embedding call goes through the gateway by URL.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench.cell_records import ATTEMPT_COMPLETE, CellPaths, read_json
from evals.skill_bench.run import CellSpec
from evals.skill_bench.tests.fakes import FakeUpstream, Reply
from evals.skill_bench.tests.run_support import (
    TINY_SIZES,
    conformant_gateway,
    ledger,
    read_rows,
    runner,
    tiny_responder,
)
from evals.skill_bench.tests.s1_support import (
    call_kind,
    first_text,
    items_reply,
    judged,
)
from evals.skill_bench.tests.rollout_support import text_reply

WORKFLOW = "Workflow 1: Answer from the page\n<think>\nread it\n</think>\n<action>\n<answer>{value}</answer>\n</action>"


def responder(endpoint: str, body: dict[str, Any]) -> Reply | None:
    """TinyEnv's actor answers; J says success when the bot answered, the learner replies."""
    if endpoint != "chat/completions":
        return None
    kind = call_kind(body)
    if kind == "judge":
        answered = re.search(r"<answer>A-", first_text(body))
        return text_reply(judged("success" if answered else "failure"))
    if kind == "learner":
        if body["messages"][0]["role"] == "system":  # ReasoningBank's extraction
            return text_reply(items_reply("Copy the exact span", "Check the question"))
        return text_reply(WORKFLOW)
    return tiny_responder(endpoint, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["reasoningbank", "awm"])
async def test_a_whole_cell_learns_freezes_and_tests(tmp_path: Path, method: str):
    fake = FakeUpstream(responder=responder)
    spec = CellSpec("alfworld", "luna", method)
    with conformant_gateway(tmp_path, fake) as (_gw, url):
        outcome = await runner(tmp_path, url, spec=spec).run()
    assert outcome.status == "complete" and outcome.exit_code == 0, outcome
    paths = CellPaths(tmp_path / "runs" / spec.cell_id)
    attempt = paths.attempt(1)
    state = read_json(attempt.state)
    assert state["state"] == ATTEMPT_COMPLETE
    assert state["snapshot_checks"] and all(
        c["equal"] for c in state["snapshot_checks"]
    )
    frozen = read_json(attempt.frozen_snapshot)["snapshot"]
    assert frozen["method"] == method and frozen["frozen"] is True
    learned = frozen["counters"]["tasks_learned"]
    assert learned == frozen["judge"]["calls"] > 0
    injections = read_rows(attempt.injections)
    tested = [r for r in injections if r["phase"] == "test"]
    assert len(tested) == len(read_rows(paths.results)) > 0
    rows = ledger(tmp_path, state["run_id"])  # <cell>-a01-<instance>
    calls = [r for r in rows if r.get("event") == "call"]
    roles = {r["role"] for r in calls}
    if method == "reasoningbank":
        assert len(frozen["entries"]) == learned  # every task stored two items
        assert all(len(e["memory_items"]) == 2 for e in frozen["entries"])
        assert roles == {"actor", "judge", "learner", "embedding"}
        test_embeddings = [
            r for r in calls if r["role"] == "embedding" and r["phase"] == "test"
        ]
        assert len(test_embeddings) == len(
            tested
        )  # one retrieval per test item, nothing else
        assert all("Copy the exact span" in r["text"] for r in tested)
    else:
        assert frozen["workflow"] == WORKFLOW
        assert (
            frozen["counters"]["inductions"] == frozen["counters"]["judged_success"] > 0
        )
        assert roles == {"actor", "judge", "learner"}
        assert {r["text"] for r in tested} == {WORKFLOW}
    # Learning-side calls happen in the learning phase only; the test phase is read-only.
    assert {r["phase"] for r in calls if r["role"] in ("judge", "learner")} == {
        "learning"
    }
    assert TINY_SIZES["train"] >= learned
