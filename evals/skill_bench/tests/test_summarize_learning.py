"""tools/summarize_learning.py: what every method learned by the freeze, what it injected at
test time and what the cell cost, on the offline matrix and on constructed cells.

The constructed cells use ``test_report.make_cell`` (``run.py``'s layout with a gateway
ledger per attempt), so every number the summary prints is known in advance.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import report, study
from evals.skill_bench.tests.conftest import REMOVED_METHOD
from evals.skill_bench.tests.e2e_support import Matrix
from evals.skill_bench.tests.test_report import (
    CONTEXT_CHARS,
    CONTEXT_SUMMARY,
    POOL_RUNS,
    _call,
    _deadline_call,
    _distilled,
    binary,
    context_runs,
    make_cell,
    pool_batch,
)
from evals.skill_bench.tools import remove_method_data
from evals.skill_bench.tools import summarize_learning as S

TEST = binary("q", [1, 0])


def _injection(raw_id: str, tokens: int, phase: str = "test") -> dict[str, Any]:
    return {
        "phase": phase,
        "raw_id": raw_id,
        "tokens": tokens,
        "injected_tokens_total": tokens * 2,
        "text": "x" * tokens,
    }


def mirix_snapshot(*, skills: int = 1, runs: tuple[str, ...] = ("succeeded",)) -> dict:
    ids = [f"proc-{n:04d}" for n in range(1, skills + 1)]
    return {
        "method": "mirix",
        "skills": [
            {
                "id": i,
                "name": f"skill {i}",
                "version": "0.1.0",
                "entry_type": "workflow",
            }
            for i in ids
        ],
        "skill_history": {
            i: {
                "skill_id": i,
                "versions": [
                    {"version_no": 1, "change_type": "create"},
                    {"version_no": 2, "change_type": "update"},
                ],
            }
            for i in ids
        },
        "skill_versions": {i: {"1": {"name": i}, "2": {"name": i}} for i in ids},
        "learning_runs": [
            {
                "id": f"run-{n}",
                "status": status,
                "error": "curator: upstream 502" if status == "failed" else None,
                "distill_metrics": [
                    {"status": "ok", "experiences": 2},
                    {"status": "ok", "experiences": 1},
                ],
                "curator_metrics": [{"writes": 1}],
            }
            for n, status in enumerate(runs)
        ],
        "counters": {"messages_added": 6, "sessions_closed": 3, "runs_started": 1},
    }


def rb_snapshot(titles: tuple[str, ...] = ("Check the fridge first", "Verify units")):
    return {
        "method": "reasoningbank",
        "entries": [
            {
                "task_uid": "t1",
                "status": "success",
                "memory_items": [
                    {"id": "t1#1", "title": t, "description": "d", "content": "c"}
                    for t in titles
                ],
            },
            {"task_uid": "t2", "status": "failure", "memory_items": []},
        ],
        "counters": {"tasks_learned": 2},
    }


def awm_snapshot(workflow: str = "## Workflow\n1. find\n2. take") -> dict:
    return {
        "method": "awm",
        "workflow": workflow,
        "pool": [{"task_uid": "t1", "instruction": "i"}],
        "inductions": [{"at": "x"}, {"at": "y"}],
        "counters": {
            "inductions": 2,
            "workflows_written": 1 if workflow else 0,
            "induction_model_failures": 1,
        },
    }


SNAPSHOTS = {
    "mirix": mirix_snapshot,
    "reasoningbank": rb_snapshot,
    "awm": awm_snapshot,
}


def cell(runs: Path, method: str, snapshot: dict | None = None, **kw: Any) -> Path:
    kw.setdefault(
        "calls",
        [
            _call("learner", "learning", 0.01),
            _call("actor", "learning", 0.02, item="u1"),
            _call("actor", "test", 0.03, item="u9"),
        ],
    )
    kw.setdefault("injections", [_injection("q000", 100), _injection("q001", 300)])
    return make_cell(
        runs,
        "alfworld",
        method,
        test=TEST,
        snapshot=snapshot if snapshot is not None else SNAPSHOTS[method](),
        **kw,
    )


def summary_of(runs: Path) -> dict[str, dict[str, Any]]:
    return {c["method"]: c for c in S.summarize(runs)["cells"]}


# --- what each method learned -------------------------------------------------------------


def test_what_each_method_learned(tmp_path):
    runs = tmp_path / "runs"
    for method in SNAPSHOTS:
        cell(runs, method)
    make_cell(
        runs, "alfworld", "none", test=TEST, snapshot={"method": "none", "memory": None}
    )
    cells = summary_of(runs)

    mirix = cells["mirix"]["learned"]
    assert mirix["skills"] == 1
    assert mirix["skill_list"] == [
        {
            "id": "proc-0001",
            "name": "skill proc-0001",
            "version": "0.1.0",
            "entry_type": "workflow",
            "versions": 2,
            "changes": ["create", "update"],
            "versions_stored": 2,
        }
    ]
    assert (mirix["experiences"], mirix["curator_writes"]) == (3, 1)
    assert mirix["runs_by_status"] == {"succeeded": 1}
    assert mirix["counters"]["sessions_closed"] == 3

    rb = cells["reasoningbank"]["learned"]
    assert rb["memory_items"] == 2
    assert rb["titles"] == ["Check the fridge first", "Verify units"]
    assert rb["entries_by_status"] == {"success": 1, "failure": 1}

    awm = cells["awm"]["learned"]
    assert awm["workflow_chars"] == len("## Workflow\n1. find\n2. take")
    assert (
        awm["inductions"],
        awm["workflows_written"],
        awm["induction_model_failures"],
    ) == (
        2,
        1,
        1,
    )

    assert cells["none"]["learned"] == {}
    assert all(c["nothing_learned"] is None for c in cells.values())
    assert all(c["frozen"] for c in cells.values())


def test_injections_and_cost_come_from_the_records(tmp_path):
    runs = tmp_path / "runs"
    cell(
        runs,
        "awm",
        injections=[
            _injection("q000", 100),
            _injection("q001", 300),
            _injection("q001", 500),  # a re-run item: the last row wins
            _injection("L1", 7, phase="learning"),
        ],
        invalid_calls=[_call("learner", "learning", 0.5)],
    )
    (summary,) = S.summarize(runs)["cells"]
    tokens = summary["test_injections"]["tokens"]
    assert summary["test_injections"]["items"] == 2
    assert (tokens["mean"], tokens["max"], tokens["total"]) == (300.0, 500.0, 600.0)
    cost = summary["cost"]
    assert cost["learning"] == pytest.approx(0.03)
    assert cost["test"] == pytest.approx(0.03)
    assert cost["other_attempts"] == pytest.approx(0.5)
    assert cost["total"] == pytest.approx(0.56)
    assert (cost["learning_calls"], cost["test_calls"]) == (2, 1)
    assert cost["by_role"]["learning"] == {
        "learner": pytest.approx(0.01),
        "actor": pytest.approx(0.02),
    }
    assert summary["reported_attempt"] == 2 and summary["attempt_state"] == "complete"
    # The same numbers report.py prints for the cell.
    loaded = report.load_cells(runs, split_mode=None)[0]
    assert cost["total"] == report.cell_costs(loaded)["total_cost"]
    assert cost["unknown_cost_calls"] == 0 and cost["not_counted"] == "-"


def test_deadline_calls_are_not_counted_and_say_so(tmp_path, capsys):
    """A deadline call is billed at a cost the ledger never learns. The
    summary counts it apart, per side, and prints it as report.py's 'not counted'."""
    from evals.skill_bench.tests.test_report import _deadline_call

    runs = tmp_path / "runs"
    cell(
        runs,
        "reasoningbank",
        calls=[
            _call("actor", "learning", 0.02, item="u1"),
            _deadline_call("learner", "learning", legacy=True),
            _call("actor", "test", 0.03, item="u9"),
            _deadline_call("actor", "test", item="u9"),
        ],
    )
    summary = S.summarize(runs)
    (only,) = summary["cells"]
    cost = only["cost"]
    assert cost["total"] == pytest.approx(0.05)
    assert cost["unknown_cost_calls"] == 2
    assert cost["unknown_cost_calls_by_side"] == {"learning": 1, "test": 1}
    assert cost["not_counted"] == "2 cut-off call(s), cost unknown"
    assert cost["not_counted"] == report.uncounted_costs(
        report.cell_costs(report.load_cells(runs, split_mode=None)[0])
    )
    assert summary["unknown_cost_calls"] == 2
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert "total $0.0500; not counted: 2 cut-off call(s), cost unknown" in out
    assert "total cost $0.0500 (not counting 2 cut-off call(s) billed at an " in out


# --- produced nothing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "snapshot", "why"),
    [
        ("mirix", mirix_snapshot(skills=0, runs=("failed", "succeeded")), "0 skills"),
        ("reasoningbank", rb_snapshot(titles=()), "0 memory items"),
        ("awm", awm_snapshot(workflow="  \n"), "empty workflow"),
    ],
)
def test_a_learning_method_that_produced_nothing_fails_the_summary(
    tmp_path, capsys, method, snapshot, why
):
    runs = tmp_path / "runs"
    cell(runs, method, snapshot)
    make_cell(
        runs, "alfworld", "none", test=TEST, snapshot={"method": "none", "memory": None}
    )
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_NOTHING_LEARNED
    out = capsys.readouterr().out
    assert f"NOTHING LEARNED: {why}" in out
    assert "1 learned nothing: alfworld-luna-" + method in out
    summary = S.summarize(runs)
    assert summary["nothing_learned"] == [f"alfworld-luna-{method}"]


def test_mirix_that_learned_nothing_says_what_its_runs_did(tmp_path):
    runs = tmp_path / "runs"
    cell(runs, "mirix", mirix_snapshot(skills=0, runs=("failed", "succeeded")))
    (summary,) = S.summarize(runs)["cells"]
    assert summary["nothing_learned"] == (
        "0 skills after the freeze (6 experiences, 2 learning runs: failed 1, succeeded 1)"
    )
    assert summary["learned"]["run_errors"] == ["curator: upstream 502"]


def test_mirix_reports_the_session_outcomes_its_distiller_stored(tmp_path, capsys):
    """Every succeeded / failed outcome must be quoted verbatim from the
    transcript, so a misquote costs the session its outcome (stored ``unclear``). The
    summary counts the outcomes the learning runs stored and the chunks whose stated
    outcome the transcript did not show, or only the agent's own words did, so what the
    check costs can be measured run by run."""
    runs = tmp_path / "runs"
    snapshot = mirix_snapshot(runs=("succeeded", "succeeded"))
    first, second = snapshot["learning_runs"]
    first["distill_metrics"] = [
        {"status": "ok", "experiences": 2, "session_outcome": "succeeded"},
        {
            "status": "ok",
            "experiences": 1,
            "session_outcome": "unclear",
            "outcomes_unverified": 1,
        },
    ]
    second["distill_metrics"] = [
        {
            "status": "ok",
            "experiences": 1,
            "session_outcome": "succeeded",
            "outcomes_self_reported": 2,
        },
        {"status": "deferred", "session_outcome": None},
        {"status": "ok", "experiences": 1},  # a MIRIX before the outcome
    ]
    cell(runs, "mirix", snapshot)
    (summary,) = S.summarize(runs)["cells"]
    learned = summary["learned"]
    assert learned["session_outcomes"] == {"succeeded": 2, "unclear": 1}
    assert (learned["outcomes_unverified"], learned["outcomes_self_reported"]) == (1, 2)
    assert S.main(["--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert (
        "    session outcomes: succeeded 2, unclear 1 (1 stated outcome the transcript "
        "did not show, 2 only the agent's own words showed)"
    ) in out
    # A MIRIX that stored no outcome prints no such line.
    cell(tmp_path / "old", "mirix")
    (old,) = S.summarize(tmp_path / "old")["cells"]
    assert old["learned"]["session_outcomes"] == {}
    assert "session outcomes" not in S.render_text(S.summarize(tmp_path / "old"))


def test_mirix_curator_tool_exceptions_are_listed(tmp_path, capsys):
    """Exceptions curator tools met that MIRIX answered to the model as failed tool
    results (its own DBAPIError here) are listed with what the cell learned; the model's
    own bad arguments are not."""
    runs = tmp_path / "runs"
    snapshot = mirix_snapshot()
    snapshot["learning_runs"][0]["curator_metrics"][0].update(
        batch_id="run-0:b0",
        failed_tool_calls=3,
        tool_failures=[
            {
                "tool": "list_memory_within_timerange",
                "stage": "execution",
                "exception": "DBAPIError",
                "error": "DBAPIError: can't subtract offset-naive and offset-aware "
                "datetimes",
                "count": 2,
            },
            {
                "tool": "skill_create",
                "stage": "validation",
                "exception": None,
                "error": "Validation error: 'name' cannot be empty.",
                "count": 1,
            },
        ],
    )
    cell(runs, "mirix", snapshot)
    (summary,) = S.summarize(runs)["cells"]
    (listed,) = summary["learned"]["curator_tool_exceptions"]
    assert listed.startswith("[mirix] run run-0 curator batch run-0:b0: 2 call(s)")
    assert "DBAPIError" in listed
    assert summary["nothing_learned"] is None
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert "curator tool exception: [mirix] run run-0" in capsys.readouterr().out
    # A snapshot of a MIRIX before tool failures were recorded lists none.
    other = tmp_path / "older"
    cell(other, "mirix", mirix_snapshot())
    (older,) = S.summarize(other)["cells"]
    assert older["learned"]["curator_tool_exceptions"] == []


def test_a_cell_whose_memory_never_froze_learned_nothing(tmp_path):
    runs = tmp_path / "runs"
    root = cell(runs, "reasoningbank", status="testing")
    (root / "attempt-01" / "snapshots" / "frozen.json").unlink()
    (summary,) = S.summarize(runs)["cells"]
    assert summary["frozen"] is False and summary["learned"] is None
    assert summary["nothing_learned"] == "no frozen memory: learning did not finish"
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_NOTHING_LEARNED


def test_the_no_memory_baseline_is_never_flagged(tmp_path, capsys):
    runs = tmp_path / "runs"
    root = make_cell(
        runs, "alfworld", "none", test=TEST, injections=[_injection("q000", 0)]
    )
    (root / "attempt-01" / "snapshots" / "frozen.json").unlink()
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert (
        "nothing, by design" in out and "every learning method learned something" in out
    )


def test_learned_but_injected_nothing_is_a_warning(tmp_path):
    runs = tmp_path / "runs"
    cell(runs, "mirix", injections=[_injection("q000", 0), _injection("q001", 0)])
    (summary,) = S.summarize(runs)["cells"]
    assert summary["nothing_learned"] is None
    assert summary["warnings"] == [
        "learned something but injected nothing in all 2 test items"
    ]


def test_no_cells_and_json_output(tmp_path, capsys):
    assert S.main(["--runs-dir", str(tmp_path / "missing")]) == S.EXIT_NO_CELLS
    capsys.readouterr()
    runs = tmp_path / "runs"
    cell(runs, "reasoningbank", split_mode="smoke", n_train=5, n_test=2)
    cell(runs, "awm")
    assert S.main(["--runs-dir", str(runs), "--json", "--split-mode", "smoke"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [c["cell_id"] for c in printed["cells"]] == [
        "alfworld-luna-reasoningbank-smoke-l5-t2"
    ]
    assert printed["total_cost"] == pytest.approx(0.06)


def test_text_lists_mirix_skills_and_rb_titles(tmp_path, capsys):
    runs = tmp_path / "runs"
    cell(runs, "mirix", mirix_snapshot(skills=S.LIST_LIMIT + 2))
    cell(runs, "reasoningbank")
    assert S.main(["--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert (
        "22 skills, 3 experiences, 1 curator writes; 1 learning runs (succeeded 1)"
        in out
    )
    assert "- skill proc-0001 (v0.1.0, 2 versions: create, update)" in out
    assert "... and 2 more" in out
    assert "- Check the fridge first" in out
    assert "test injections: 2 items, 0 empty; tokens mean 200, median 200" in out
    assert "cost: learning $0.0300 (2 calls), test $0.0300 (1 calls)" in out


def test_text_lists_the_skills_mirix_deleted_and_those_without_a_vector(
    tmp_path, capsys
):
    """A skill a run deleted is in the frozen snapshot with its history."""
    snapshot = mirix_snapshot(skills=1)
    snapshot["deleted_skills"] = ["proc-9999"]
    snapshot["skill_history"]["proc-9999"] = {
        "skill_id": "proc-9999",
        "name": "an old skill",
        "is_deleted": True,
        "versions": [{"change_type": "create"}, {"change_type": "delete"}],
    }
    snapshot["skills_without_vector"] = ["proc-0001"]
    runs = tmp_path / "runs"
    cell(runs, "mirix", snapshot)
    assert S.main(["--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "- deleted: an old skill (2 versions: create, delete)" in out
    assert "without a retrieval vector: proc-0001" in out


# --- the MIRIX experience pool ------------------------------------------------------------------


def pool_snapshot(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """A MIRIX snapshot whose learning runs report the pool, with the method's summary."""
    from evals.skill_bench.methods.mirix import experience_pool

    snapshot = mirix_snapshot(skills=2)
    snapshot["learning_runs"] = runs
    snapshot["experience_pool"] = experience_pool(runs)
    return snapshot


def test_mirix_reports_its_experience_pool(tmp_path, capsys):
    runs = tmp_path / "runs"
    snapshot = pool_snapshot(POOL_RUNS)
    cell(runs, "mirix", snapshot)
    (summary,) = S.summarize(runs)["cells"]
    assert summary["learned"]["experience_pool"] == snapshot["experience_pool"]
    assert summary["warnings"] == []
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert (
        "    experience pool: 4 active at the freeze (6 entered; retired: used 1, "
        "covered 1, ignored 0, merged 0, unembeddable 0); new experiences: used 2, "
        "covered 0, not worth 1, "
        "deferred 5 (3 by default), unaddressed 0; earlier lessons shown 4 (cited 1, "
        "covered 1, not worth 1, ignored 1), recurrences 4; citations without a "
        "verified quote 0\n"
    ) in capsys.readouterr().out


def test_a_mirix_without_the_pool_says_so(tmp_path, capsys):
    runs = tmp_path / "runs"
    cell(runs, "mirix")  # its batches report no pool: a MIRIX without it
    (summary,) = S.summarize(runs)["cells"]
    assert summary["learned"]["experience_pool"] is None
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert (
        "experience pool: not reported (a MIRIX without the experience pool)"
        in capsys.readouterr().out
    )


def test_what_in_the_pool_deserves_a_look_is_warned_about(tmp_path, capsys):
    """A batch that lost its earlier lessons to the embedding (the attempt should have
    restarted), a pool switched off (not MIRIX's default) and metrics that do
    not add up are warnings; they do not make the cell one that learned nothing."""
    broken = [
        {
            "id": "r1",
            "created_at": "2026-01-01T00:00:01",
            "status": "succeeded",
            "curator_metrics": [
                pool_batch(
                    batch_id="r1:b0",
                    history_skipped="embedding_failed",
                    dispositions={"deferred": 2, "defaulted": 2},
                ),
                pool_batch(batch_id="r1:b1", history_skipped="embeddings_disabled"),
                pool_batch(batch_id="r1:b2", history_skipped="disabled"),
            ],
        }
    ]
    runs = tmp_path / "runs"
    cell(runs, "mirix", pool_snapshot(broken))
    (summary,) = S.summarize(runs)["cells"]
    assert summary["warnings"] == [
        "1 curator batch(es) showed no earlier lessons because the experience pool's "
        "embedding failed: the attempt should have restarted",
        "the experience pool was off in 1 batch(es) (disabled), 1 batch(es) "
        "(embeddings_disabled): not MIRIX's default",
        "the experience pool's metrics do not add up: 2 entered, 0 retired, 0 active "
        "at the freeze",
    ]
    assert summary["nothing_learned"] is None
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert "warning: 1 curator batch(es) showed no earlier lessons" in (
        capsys.readouterr().out
    )


# --- what MIRIX's distiller was shown and said --------------------------------------------------


def context_cell(
    runs: Path,
    learning_runs: list[dict[str, Any]],
    *,
    benchmark: str = "alfworld",
    sent: int | None = CONTEXT_CHARS,
    model_key: str = "luna",
) -> Path:
    """A MIRIX cell on ``benchmark`` whose attempt recorded an agent context of ``sent``
    chars (None: an attempt from before it recorded one)."""
    snapshot = mirix_snapshot()
    snapshot["learning_runs"] = learning_runs
    extra = (
        {"agent_context": {"sha256": "c" * 64, "chars": sent}}
        if sent is not None
        else {}
    )
    return make_cell(
        runs,
        benchmark,
        "mirix",
        test=TEST,
        snapshot=snapshot,
        model_key=model_key,
        attempts=[("complete", [_call("actor", "test", 0.01, item="u9")], extra)],
    )


def test_mirix_reports_what_its_distiller_was_shown_and_said(tmp_path, capsys):
    """The learned sessions distilled with the agent context, the skills served, each
    served skill's stored followed / effect values, the usage records the curator
    batches showed and its citations without a verified quote: a run confirms the
    agent context reached MIRIX's distiller without SQL against the cell's database."""
    runs = tmp_path / "runs"
    context_cell(runs, context_runs())
    (summary,) = S.summarize(runs)["cells"]
    assert summary["learned"]["learning_context"] == CONTEXT_SUMMARY
    assert summary["warnings"] == []
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert (
        f"    session context: 3 of 3 learned sessions with the agent context "
        f"({CONTEXT_CHARS} chars), 1 cut to fit (120 tokens); 3 skills served in 2 "
        "sessions (per session: 0 x1, 1 x1, 2 x1; 0 left out of the context)\n"
        "    skill feedback: entries for 3 of 3 served (0 not shown); followed no 1, "
        "unclear 1, yes 1; effect helped 1, unclear 2; stated without a verifying "
        "quote: followed 0, effect 1\n"
        "      - Find then take: served 2; followed no 1, yes 1; effect helped 1, "
        "unclear 1\n"
        "      - Check the price: served 1; followed unclear 1; effect unclear 1\n"
        "    curator usage records: shown in 1 of 2 batches (2 records, 1 of skills "
        "served there; 1 batch(es) with one line per skill, 0 skills left out for "
        "room, 310 "
        "tokens; none shown: none 1); citations without a verified quote 2\n"
    ) in out
    assert "citations without a verified quote 2\n" in out.split("experience pool:")[1]


@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
def test_a_learned_session_without_the_agent_context_is_warned_about(
    tmp_path, benchmark
):
    """An ALFWorld or WebShop MIRIX cell with a learned session MIRIX
    distilled without the agent context (a wrong tree, a lost attach) is warned about,
    and so is a stored length that is not the one the attempt sent. An attempt from
    before it recorded its context is judged by its benchmark."""
    runs = [
        {
            "id": "r",
            "distill_metrics": [
                _distilled(f"s{n}", agent_context_chars=0) for n in range(7)
            ]
            + [_distilled("ok"), _distilled("short", agent_context_chars=4000)],
        }
    ]
    context_cell(tmp_path / "a", runs, benchmark=benchmark)
    (summary,) = S.summarize(tmp_path / "a")["cells"]
    assert summary["warnings"] == [
        "7 of 9 learned session(s) were distilled without the agent context although "
        "the benchmark offers it (the agent context did not reach MIRIX's distiller): "
        "s0, s1, s2, s3, s4 and 2 more",
        f"MIRIX stored an agent context of 4000 chars on learned sessions; the attempt "
        f"sent {CONTEXT_CHARS}",
    ]
    assert summary["nothing_learned"] is None
    context_cell(tmp_path / "b", runs, benchmark=benchmark, sent=None)
    (before,) = S.summarize(tmp_path / "b")["cells"]
    assert before["warnings"] == summary["warnings"][:1]


def test_no_agent_context_warning_where_none_was_offered_or_reported(tmp_path, capsys):
    """A benchmark that offers no agent context (the attempt sent 0 chars) and a MIRIX
    that records no learning context are not warned about; the latter's summary says it
    reports no session context."""
    lost = [{"id": "r", "distill_metrics": [_distilled("s", agent_context_chars=0)]}]
    context_cell(tmp_path / "a", lost, benchmark="alfworld", sent=0)
    old = [{"id": "r", "distill_metrics": [{"session_id": "s", "status": "ok"}]}]
    context_cell(tmp_path / "a", old, benchmark="alfworld", model_key="luna56")
    summaries = S.summarize(tmp_path / "a")["cells"]
    assert [c["warnings"] for c in summaries] == [[], []]
    assert S.main(["--runs-dir", str(tmp_path / "a")]) == S.EXIT_OK
    assert (
        "alfworld-luna56-mirix  [complete; attempt 1 complete]\n"
        in capsys.readouterr().out
    )
    text = S.render_text(S.summarize(tmp_path / "a"))
    assert "    session context: not reported (a MIRIX that records none)\n" in text


def test_a_mirix_that_recorded_no_per_skill_values_says_so(tmp_path):
    """A MIRIX whose runs recorded the feedback counts but not each skill's values:
    the summary says where they are."""
    runs = [
        {
            "id": "r",
            "distill_metrics": [
                _distilled(
                    "s",
                    served_skills=1,
                    skill_feedback={"served": 1, "not_shown": 0, "given": 1},
                )
            ],
        }
    ]
    context_cell(tmp_path / "runs", runs)
    text = S.render_text(S.summarize(tmp_path / "runs"))
    assert (
        "      (each skill's values not recorded in the run: a MIRIX before it "
        "recorded them; skill_session_feedback holds them)\n"
    ) in text


def test_the_agent_context_benchmarks_are_the_environments_that_offer_one():
    """:data:`summarize_learning.AGENT_CONTEXT_BENCHMARKS` (the fallback for attempts
    that recorded no context) names exactly the environments with ``agent_context()``."""
    from evals.skill_bench import protocol
    from evals.skill_bench.envs.alfworld import AlfWorldEnv
    from evals.skill_bench.envs.webshop import WebShopEnv

    classes = {"alfworld": AlfWorldEnv, "webshop": WebShopEnv}
    assert set(classes) == set(protocol.BENCHMARKS)
    assert set(S.AGENT_CONTEXT_BENCHMARKS) == {
        b for b, cls in classes.items() if callable(getattr(cls, "agent_context", None))
    }


# --- the offline matrix -----------------------------------------------------------------------


def test_every_method_of_the_offline_matrix_learned_something(
    e2e_matrix: Matrix, capsys
):
    summary = S.summarize(e2e_matrix.runs)
    cells = {c["cell_id"]: c for c in summary["cells"]}
    assert set(cells) == set(e2e_matrix.outcomes)
    assert summary["nothing_learned"] == []
    for c in cells.values():
        assert c["complete"] and c["frozen"], c["cell_id"]
        assert c["cost"]["test"] > 0 and c["cost"]["total"] > 0, c["cell_id"]
        injections = c["test_injections"]
        if c["method"] == "none":
            assert c["learned"] == {}
            assert injections["empty"] == injections["items"]
            continue
        assert injections["items"] and injections["empty"] < injections["items"]
        learned = c["learned"]
        if c["method"] == "reasoningbank":
            assert learned["memory_items"] > 0 and learned["titles"]
        elif c["method"] == "awm":
            assert learned["workflow_chars"] > 0 and learned["inductions"] > 0
    assert S.main(["--runs-dir", str(e2e_matrix.runs)]) == S.EXIT_OK
    assert "every learning method learned something" in capsys.readouterr().out


# --- resumes, hosts and a copied runs directory -------------------------------------------------


def test_a_resumed_learning_phase_says_what_its_resumes_discarded(tmp_path, capsys):
    """The discarded calls are part of the learning side's cost (counted once) and shown
    apart, with how many resumes there were and from which checkpoints."""
    runs = tmp_path / "runs"
    discarded = _call("learner", "learning", 0.04)
    cut = _call("actor", "learning", 0.0, item="u1")
    root = cell(
        runs,
        "reasoningbank",
        calls=[
            _call("actor", "learning", 0.02, item="u1"),
            discarded,
            cut,
            _call("learner", "learning", 0.01),
            _call("actor", "test", 0.03, item="u9"),
        ],
    )
    rows = [
        {
            "segment": 1,
            "attempt": 1,
            "checkpoint": {"id": "c0000-played", "offset": 0},
            "calls": [{"call_id": discarded["call_id"], "role": "learner"}],
        },
        {
            "segment": 2,
            "attempt": 1,
            "checkpoint": {"id": "c0000-done", "offset": 0},
            "calls": [{"call_id": cut["call_id"], "role": "actor"}],
        },
    ]
    attempt = root / "attempt-01"
    (attempt / "discarded.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    state = json.loads((attempt / "attempt.json").read_text())
    state["host"] = {"hostname": "host-1", "system": "Linux", "machine": "x86_64"}
    (attempt / "attempt.json").write_text(json.dumps(state))
    summary = S.summarize(runs)
    (only,) = summary["cells"]
    assert only["resumes"] == 2
    assert only["resume_checkpoints"] == ["c0000-played", "c0000-done"]
    # A host recorded before its machine id was has none.
    assert only["host"] == {
        "hostname": "host-1",
        "machine_id": None,
        "system": "Linux",
        "machine": "x86_64",
    }
    cost = only["cost"]
    assert cost["learning"] == pytest.approx(0.07)
    assert cost["learning_discarded"] == pytest.approx(0.04)
    assert cost["learning_discarded_calls"] == 2
    assert cost["total"] == pytest.approx(0.10)
    assert summary["resumes"] == 2
    assert summary["learning_discarded_cost"] == pytest.approx(0.04)
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert "; host host-1 (Linux x86_64)]" in out
    state["host"]["machine_id"] = "89abcdef" * 4
    (attempt / "attempt.json").write_text(json.dumps(state))
    assert S.summarize(runs)["cells"][0]["host"]["machine_id"] == "89abcdef" * 4
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert "; host host-1 (Linux x86_64, id 89abcdef)]" in capsys.readouterr().out
    assert (
        "  learning resumes: 2 (from c0000-played, c0000-done); discarded $0.0400 "
        "(2 calls, part of learning)"
    ) in out
    assert "2 learning resume(s) discarded $0.0400" in out
    S.main(["--runs-dir", str(runs), "--json"])
    assert json.loads(capsys.readouterr().out)["cells"][0]["resumes"] == 2


def test_a_discarded_call_that_finished_after_its_resume_is_counted(tmp_path, capsys):
    """A call a pause held past its resume
    is in its discarded.jsonl row with no final line, ``cost_total`` 0 and in
    ``pending_calls``; its final line lands later. The discarded cost printed is the
    ledger's, never the row's figure at the resume."""
    runs = tmp_path / "runs"
    held = _call("embedding", "learning", 0.05)
    root = cell(
        runs,
        "reasoningbank",
        calls=[
            _call("actor", "learning", 0.02, item="u1"),
            held,
            _call("actor", "test", 0.03, item="u9"),
        ],
    )
    row = {
        "segment": 1,
        "attempt": 1,
        "checkpoint": {"id": "c0000-done", "offset": 0},
        "calls": [
            {
                "call_id": held["call_id"],
                "role": "embedding",
                "final_line": False,
                "cost_total": 0.0,
            }
        ],
        "cost_total": 0.0,
        "pending_calls": [held["call_id"]],
    }
    (root / "attempt-01" / "discarded.jsonl").write_text(json.dumps(row) + "\n")
    (only,) = S.summarize(runs)["cells"]
    assert only["cost"]["learning_discarded"] == pytest.approx(0.05)
    assert only["cost"]["learning_discarded_calls"] == 1
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert "discarded $0.0500 (1 calls, part of learning)" in capsys.readouterr().out


def test_a_cell_without_resumes_prints_no_resume_line(tmp_path, capsys):
    runs = tmp_path / "runs"
    cell(runs, "awm")
    (only,) = S.summarize(runs)["cells"]
    assert only["resumes"] == 0 and only["cost"]["learning_discarded"] == 0
    assert only["host"] is None
    S.main(["--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "learning resumes" not in out and "resume(s)" not in out
    assert "host " not in out


def test_the_ledgers_of_a_copied_runs_directory_are_found(tmp_path, capsys):
    """The runs directory was copied without its gateway run directories, which this
    host keeps elsewhere: ``--gateway-runs-dir`` names that place (the attempts'
    recorded ledger paths name the original host's tree)."""
    original = tmp_path / "original" / "runs"
    cell(runs=original, method="awm")
    copy = tmp_path / "copy" / "runs"
    elsewhere = tmp_path / "gateway-runs"
    elsewhere.mkdir()
    copy.mkdir(parents=True)
    for child in original.iterdir():
        target = copy if (child / "cell.json").is_file() else elsewhere
        (target / child.name).parent.mkdir(parents=True, exist_ok=True)
        child.rename(target / child.name)
    expected = 0.06
    (only,) = S.summarize(copy)["cells"]
    assert only["cost"]["total"] == 0  # no ledger found
    assert any("attempts without a ledger" in w for w in only["warnings"])
    (only,) = S.summarize(copy, gateway_runs_dirs=[elsewhere])["cells"]
    assert only["cost"]["total"] == pytest.approx(expected)
    assert only["warnings"] == []
    assert S.main(["--runs-dir", str(copy), "--gateway-runs-dir", str(elsewhere)]) == 0
    assert "total cost $0.0600" in capsys.readouterr().out


# --- removed methods (study.REMOVED_METHODS) ----------------------------------------------------

#: The method the tests register as removed (the ``removed_method`` fixture).
DC = REMOVED_METHOD


@pytest.fixture
def deletion_decided(removed_method: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """:data:`DC` removed from the study with its deletion decided
    (``remove_method_data.DELETION_DECISIONS``), for one test."""
    monkeypatch.setattr(
        remove_method_data,
        "DELETION_DECISIONS",
        {removed_method: {"decision": "example decision", "quote": "delete it"}},
    )
    return removed_method


def _leftovers(runs: Path) -> list[str]:
    """The removed method's cells a runs directory still holds: one complete whose
    memory froze empty, one stopped mid-learning. Their ledgers: $0.04 and $0.01."""
    make_cell(
        runs,
        "webshop",
        DC,
        test=TEST,
        snapshot={"method": DC, "notes": ""},
        calls=[_call("learner", "learning", 0.04)],
    )
    root = make_cell(
        runs,
        "alfworld",
        DC,
        test=[],
        status="interrupted",
        calls=[_call("actor", "learning", 0.01, item="u1")],
    )
    (root / "attempt-01" / "snapshots" / "frozen.json").unlink()
    return [f"alfworld-luna-{DC}", f"webshop-luna-{DC}"]


def test_a_removed_methods_cells_are_listed_not_summarized(
    tmp_path, capsys, removed_method
):
    """A removed method's cells are set aside (not what they
    learned, never 'learned nothing', never the exit status) and listed with their
    status and cost, which the totals keep apart."""
    runs = tmp_path / "runs"
    cell(runs, "awm")
    make_cell(
        runs, "alfworld", "none", test=TEST, snapshot={"method": "none", "memory": None}
    )
    study_only = S.summarize(runs)
    removed_ids = _leftovers(runs)
    summary = S.summarize(runs)
    assert summary["cells"] == study_only["cells"]
    assert summary["nothing_learned"] == [] and summary["total_cost"] == pytest.approx(
        study_only["total_cost"]
    )
    assert [c["cell_id"] for c in summary["removed"]] == removed_ids
    by_id = {c["cell_id"]: c for c in summary["removed"]}
    decision = study.REMOVED_METHODS[DC].decision
    assert by_id[f"webshop-luna-{DC}"] == {
        "cell_id": f"webshop-luna-{DC}",
        "cell_dir": f"webshop-luna-{DC}",
        "benchmark": "webshop",
        "model_key": "luna",
        "method": DC,
        "split_mode": "protocol",
        "status": "complete",
        "complete": True,
        "reported_attempt": 1,
        "attempt_state": "complete",
        "decision": decision,
        "cost": {
            "total": pytest.approx(0.04),
            "unknown_cost_calls": 0,
            "unfinished_calls": 0,
            "not_counted": "-",
        },
        "on_disk": True,
        "deleted": None,
    }
    assert by_id[f"alfworld-luna-{DC}"]["status"] == "interrupted"
    removed_uncounted = (
        summary["removed_unknown_cost_calls"],
        summary["removed_unfinished_calls"],
    )
    assert removed_uncounted == (0, 0)
    assert summary["removed_methods"] == {
        DC: {
            "decision": decision,
            "reason": study.REMOVED_METHODS[DC].reason,
            "cells": removed_ids,
        }
    }
    assert summary["removed_cost"] == pytest.approx(0.05)
    assert summary["total_cost_all_cells"] == pytest.approx(
        summary["total_cost"] + 0.05
    )
    assert str(runs) not in json.dumps([summary["removed"], summary["removed_methods"]])
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert (
        f"removed from the study, set aside (not summarized): {DC}, {decision}: "
    ) in out
    assert f"  webshop-luna-{DC}  [complete; attempt 1 complete] cost $0.0400" in out
    assert f"  alfworld-luna-{DC}  [interrupted; attempt 1 testing] cost $0.0100" in out
    assert (
        "2 cells, total cost $0.0600; 2 cell(s) of removed methods set aside, $0.0500 "
        "more ($0.1100 with them); every learning method learned something"
    ) in out
    assert "NOTHING LEARNED" not in out


def test_only_removed_cells_are_no_cells_of_the_study(tmp_path, capsys, removed_method):
    runs = tmp_path / "runs"
    removed_ids = _leftovers(runs)
    assert S.main(["--runs-dir", str(runs), "--json"]) == S.EXIT_NO_CELLS
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["cells"] == [] and [c["cell_id"] for c in printed["removed"]] == (
        removed_ids
    )
    assert (
        f"no cells in {runs} but 2 of methods removed from the study, set aside: "
        + ", ".join(removed_ids)
    ) in captured.err


def _without_disk(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ("on_disk", "deleted")}


def test_deleted_removed_cells_are_listed_from_their_manifest_at_the_same_cost(
    tmp_path, capsys, deletion_decided
):
    """Once tools/remove_method_data.py deleted the
    leftovers, they are listed from the removal manifest with the cost it recorded, so
    every total is what it was before the deletion; on a merged view too (the manifest
    under a host mirror), and once however many places hold it."""
    runs = tmp_path / "runs"
    cell(runs, "awm")
    removed_ids = _leftovers(runs)
    before = S.summarize(runs)
    code, out = remove_method_data.run(runs, DC, yes=True, argv=["test"])
    assert code == remove_method_data.EXIT_OK, out
    assert not any((runs / cell_id).exists() for cell_id in removed_ids)
    after = S.summarize(runs)
    for key in (
        "cells",
        "nothing_learned",
        "total_cost",
        "removed_methods",
        "removed_cost",
        "removed_unknown_cost_calls",
        "removed_unfinished_calls",
        "total_cost_all_cells",
    ):
        assert after[key] == before[key], key
    assert [c["cell_id"] for c in after["removed"]] == removed_ids
    assert [_without_disk(c) for c in after["removed"]] == [
        _without_disk(c) for c in before["removed"]
    ]
    assert [c["on_disk"] for c in before["removed"]] == [True, True]
    assert [c["on_disk"] for c in after["removed"]] == [False, False]
    (manifest,) = remove_method_data.load_manifests(runs)
    assert {c["deleted"]["removal_id"] for c in after["removed"]} == {
        manifest.removal_id
    }
    assert after["removed_deleted"] == 2 and after["removed_manifest_problems"] == []
    assert S.summarize(runs, split_mode="smoke")["removed"] == []

    merged = tmp_path / "merged"
    shutil.copytree(runs / "_removed", merged / "_hosts" / "h1" / "_removed")
    assert S.summarize(merged)["removed"] == after["removed"]
    both = S.summarize(runs, gateway_runs_dirs=[merged])
    assert both["removed"] == after["removed"]
    assert both["removed_cost"] == before["removed_cost"]

    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    text = capsys.readouterr().out
    assert (
        f"  webshop-luna-{DC}  [complete; attempt 1 complete] cost $0.0400; data "
        "deleted "
    ) in text
    assert f"(removal {manifest.removal_id}; cost from its manifest)" in text
    assert "2 cell(s) of removed methods set aside, $0.0500 more" in text


def test_a_cell_its_manifest_lists_but_still_on_disk_is_listed_once(
    tmp_path, deletion_decided
):
    """A deletion that stopped after its manifest (the tool finishes it on its next run):
    the cells still on disk are listed from the disk, once, and cost the same."""
    runs = tmp_path / "runs"
    cell(runs, "awm")
    _leftovers(runs)
    before = S.summarize(runs)
    copy = tmp_path / "copy"
    shutil.copytree(runs, copy)
    code, out = remove_method_data.run(copy, DC, yes=True, argv=["test"])
    assert code == remove_method_data.EXIT_OK, out
    shutil.copytree(copy / "_removed", runs / "_removed")
    during = S.summarize(runs)
    assert during["removed"] == before["removed"]
    assert during["removed_cost"] == before["removed_cost"]
    assert during["removed_deleted"] == 0


def test_an_unreadable_removal_manifest_is_named(tmp_path, capsys, deletion_decided):
    runs = tmp_path / "runs"
    cell(runs, "awm")
    _leftovers(runs)
    code, out = remove_method_data.run(runs, DC, yes=True, argv=["test"])
    assert code == remove_method_data.EXIT_OK, out
    (manifest,) = remove_method_data.load_manifests(runs)
    manifest.path.write_text("{torn", encoding="utf-8")
    summary = S.summarize(runs)
    assert summary["removed"] == []
    (problem,) = summary["removed_manifest_problems"]
    assert "unreadable" in problem
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    assert "warning: a removal manifest could not be read: " in capsys.readouterr().out


def test_what_a_method_accepted_and_went_on_from_is_listed_and_warned(tmp_path, capsys):
    """Recorded, not stopped: MIRIX's snapshot ``tolerated`` block and
    the refusal counters AWM and ReasoningBank keep are listed per cell and warned
    about; a method without them says nothing."""
    runs = tmp_path / "runs"
    mirix = mirix_snapshot()
    mirix["tolerated"] = {
        "counts": {"gateway_refusal": 1, "mirix_learning_failed": 1},
        "problems": [
            {
                "reason": "gateway_refusal",
                "text": "curator: No endpoints found for luna56 (404)",
                "rollbacks": 0,
            }
        ],
    }
    cell(runs, "mirix", mirix)
    awm = awm_snapshot()
    awm["counters"] = {**awm["counters"], "induction_rejected": 2, "judge_rejected": 1}
    cell(runs, "awm", awm)
    cell(runs, "reasoningbank")
    by_method = summary_of(runs)
    assert by_method["mirix"]["tolerated"] == mirix["tolerated"]
    assert by_method["mirix"]["refusals"] == {}
    assert (
        "MIRIX learning went on past gateway_refusal 1, mirix_learning_failed 1 "
        "(recorded, not stopped)"
    ) in by_method["mirix"]["warnings"]
    assert by_method["awm"]["tolerated"] is None
    assert by_method["awm"]["refusals"] == {
        "induction_rejected": 2,
        "induction_retries": 0,
        "judge_rejected": 1,
    }
    assert (
        "the provider refused learning calls, counted and gone on from: "
        "induction_rejected 2, judge_rejected 1"
    ) in by_method["awm"]["warnings"]
    assert by_method["reasoningbank"]["refusals"] == {
        "extraction_rejected": 0,
        "judge_rejected": 0,
    }
    assert by_method["reasoningbank"]["warnings"] == []
    assert S.main(["--runs-dir", str(runs)]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert (
        "  accepted (gateway_refusal, after 0 rollback(s)): curator: No endpoints "
        "found for luna56 (404)"
    ) in out


def test_the_all_cells_total_names_the_removed_cells_calls_of_unknown_cost(
    tmp_path, capsys, removed_method
):
    """A removed cell's per-cell line names its unfinished calls, and the closing
    line's all-cells total says it leaves them out (and the cut-off ones)."""
    runs = tmp_path / "runs"
    cell(runs, "awm")
    make_cell(
        runs,
        "webshop",
        DC,
        test=[],
        status="interrupted",
        calls=[
            _call("learner", "learning", 0.04),
            _deadline_call("learner", "learning"),
            {"event": "call_started", "call_id": "cut-mid-call", "role": "learner"},
            {"event": "call_started", "call_id": "cut-mid-call-2", "role": "learner"},
        ],
    )
    summary = S.summarize(runs)
    (listed,) = summary["removed"]
    assert listed["cost"]["unknown_cost_calls"] == 1
    assert listed["cost"]["unfinished_calls"] == 2
    removed_uncounted = (
        summary["removed_unknown_cost_calls"],
        summary["removed_unfinished_calls"],
    )
    assert removed_uncounted == (1, 2)
    assert summary["unknown_cost_calls"] == 0  # the study's own: none
    S.main(["--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert (
        "1 cell(s) of removed methods set aside, $0.0400 more (not counting their 1 "
        "cut-off and 2 unfinished call(s) of unknown cost) ($"
    ) in out


def test_a_removed_cell_whose_cell_json_names_no_method_is_listed_by_its_id(
    tmp_path, removed_method
):
    """A removed method's cell whose cell.json lost its method does not crash the
    summary (no KeyError 'method'): its id names the method (``study.method_of``)."""
    runs = tmp_path / "runs"
    cell(runs, "awm")
    removed_ids = _leftovers(runs)
    before = S.summarize(runs)
    manifest = runs / f"webshop-luna-{DC}" / "cell.json"
    record = json.loads(manifest.read_text())
    del record["cell"]["method"]
    manifest.write_text(json.dumps(record))
    summary = S.summarize(runs)
    assert [c["cell_id"] for c in summary["removed"]] == removed_ids
    for key in ("removed", "removed_methods", "cells", "total_cost_all_cells"):
        assert summary[key] == before[key], key
