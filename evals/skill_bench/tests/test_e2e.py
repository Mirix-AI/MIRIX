"""The offline end-to-end run: the matrix through run.py, then report and audit.

``e2e_matrix`` (``conftest.py``, built by ``tests/e2e_support.py``) runs every benchmark x
{none, ReasoningBank, AWM} on 3 learning and 3 test items through
``run.py``'s runner and one real gateway over the fake upstream. These tests check that
every cell completes, that ``report.py`` covers the whole matrix, that ``audit.py`` finds
no violation of any H10 invariant, and that nothing reached a real provider. MIRIX joins
the matrix on its real per-cell server in ``test_mirix_joins_the_matrix`` (integration: it needs the
compose Postgres), and learns from the skills it served, through run.py, in
``test_mirix_learns_from_the_skills_it_served_through_run_py``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import audit, protocol, report
from evals.skill_bench.cell_records import read_json, read_jsonl
from evals.skill_bench.envs.agent_context import env_agent_context
from evals.skill_bench.tests.e2e_support import (
    E2E_METHODS,
    TEST_UPSTREAM,
    Matrix,
    build_matrix,
    cell_spec,
)
from evals.skill_bench.tools import summarize_learning

LEARNING_METHODS = [m for m in E2E_METHODS if m != "none"]


def benchmarks() -> list[str]:
    return list(protocol.BENCHMARKS)


def expected_cells() -> set[str]:
    return {cell_spec(b, m).cell_id for b in benchmarks() for m in E2E_METHODS}


def run_audit(matrix: Matrix, **kw: Any) -> audit.AuditReport:
    return audit.audit_runs(
        matrix.runs,
        env_factory=matrix.env_factory,
        expected_upstream=TEST_UPSTREAM,
        split_mode=None,
        **kw,
    )


def test_every_cell_of_the_matrix_completes(e2e_matrix):
    outcomes = e2e_matrix.outcomes
    assert set(outcomes) == expected_cells()
    assert {c: o.status for c, o in outcomes.items()} == {
        c: "complete" for c in expected_cells()
    }, {c: o.detail for c, o in outcomes.items()}
    kinds = e2e_matrix.model.kinds
    # Each baseline's own model calls happened (J, RB extraction and embeddings, AWM
    # induction), and the actor played every benchmark.
    for kind in ("judge", "rb_learner", "embedding", "awm_learner"):
        assert kinds.get(kind, 0) > 0, kinds
    for benchmark in benchmarks():
        assert kinds.get(f"actor:{benchmark}", 0) > 0, kinds
    # ALFWorld plays games that cannot be won to the 50-step cap.
    alf = e2e_matrix.cell(cell_spec("alfworld", "none").cell_id)
    steps = [r["steps"] for r in read_jsonl(alf.results)]
    assert protocol.ALFWORLD_MAX_STEPS in steps and min(steps) == 1


def test_the_report_covers_the_matrix(e2e_matrix, tmp_path):
    data = report.build_report(e2e_matrix.runs, split_mode=None)
    cells = {c["cell_id"]: c for c in data["cells"]}
    assert set(cells) == expected_cells()
    assert all(c["complete"] for c in cells.values())
    frozen = protocol.frozen_fingerprint()
    assert data["protocol"]["frozen_fingerprint"] == frozen
    assert data["protocol"]["cells_not_frozen"] == []
    assert set(data["protocol"]["cells"].values()) == {frozen}
    assert data["errors"] == [] and data["duplicates"] == []
    # Schema 4 on a clean run: one directory per cell, one attempt each, reported, run
    # under one version, nothing scored under a degraded condition.
    assert data["schema"] == report.SCHEMA_VERSION == 4
    for cell in cells.values():
        (attempt,) = cell["attempts_detail"]
        assert attempt["reported"] and attempt["test_items_scored"] == 3
        assert attempt["metrics"] == cell["test"]["metrics"]
        assert cell["versions"]["mixed"] is False and cell["newer_attempt"] is None
        assert cell["degraded"] == {"test": {}, "learning": {}}
        assert cell["record_problems"] == []
    # No audit.json yet: every cell is marked, and still compared without --require-audit.
    assert {c["audit"]["status"] for c in cells.values()} == {report.AUDIT_MISSING}
    assert data["audit"]["found"] is False
    # Every learning method is compared with the no-memory method on every metric. Smoke
    # groups are keyed by the test items only: '<benchmark>-luna-smoke-t3'.
    compared = [c for c in data["comparisons"] if "skipped" not in c]
    assert not [c for c in data["comparisons"] if "skipped" in c]
    groups = {b: f"{b}-luna-smoke-t3" for b in benchmarks()}
    assert {
        cells[cell_spec(b, m).cell_id]["group"] for b in groups for m in E2E_METHODS
    } == set(groups.values())
    expected = {
        (groups[b], m, metric.name)
        for b in benchmarks()
        for m in LEARNING_METHODS
        for metric in report.METRICS[b]
    }
    assert {(c["group"], c["method"], c["metric"]) for c in compared} == expected
    for c in compared:
        assert c["items"] == 3
        if c["test"] == "mcnemar_exact":
            assert 0.0 <= c["p_value"] <= 1.0
        else:
            assert c["ci_low"] <= c["delta"] <= c["ci_high"]
            assert c["resamples"] == protocol.BOOTSTRAP_RESAMPLES
    # The report's numbers are the scorer's records, recomputed here by hand.
    for c in compared:
        benchmark = c["group"].split("-")[0]
        ours = recorded_scores(
            e2e_matrix, cell_spec(benchmark, c["method"]).cell_id, c["metric"]
        )
        none = recorded_scores(
            e2e_matrix, cell_spec(benchmark, "none").cell_id, c["metric"]
        )
        assert set(ours) == set(none)
        assert c["method_value"] == pytest.approx(sum(ours.values()) / len(ours))
        assert c["none_value"] == pytest.approx(sum(none.values()) / len(none))
        if c["test"] == "mcnemar_exact":
            assert c["b"] == sum(1 for i in ours if ours[i] == 1.0 and none[i] == 0.0)
            assert c["c"] == sum(1 for i in ours if ours[i] == 0.0 and none[i] == 1.0)
    for cell_id, cell in cells.items():
        benchmark = cell_id.split("-")[0]
        for metric in report.METRICS[benchmark]:
            scores = recorded_scores(e2e_matrix, cell_id, metric.name)
            assert cell["test"]["metrics"][metric.name] == pytest.approx(
                sum(scores.values()) / len(scores)
            )
    for cell_id, cell in cells.items():
        method = cell_id.split("-")[2]
        learning = cell["costs"]["learning"]
        assert cell["costs"]["test"]["by_role"].get("actor", {}).get("calls", 0) > 0
        if method == "none":
            assert learning["calls"] == 0 and cell["learning_curve"] == []
            assert cell["injections"]["test"]["empty"] == 3
            assert cell["judge_agreement"] is None
        else:
            assert [w["items"] for w in cell["learning_curve"]] == [3]
            roles = set(learning["by_role"])
            assert {
                "reasoningbank": {"actor", "judge", "learner", "embedding"},
                "awm": {"actor", "judge", "learner"},
            }[method] <= roles
            assert cell["injections"]["test"]["empty"] == 0  # memory was learned
        if method in ("reasoningbank", "awm"):
            assert cell["judge_agreement"]["joined"] == 3
        assert cell["providers"]["overall"]
        if cell_id.startswith("alfworld"):
            assert sum(v["items"] for v in cell["task_types"].values()) == 3
    assert data["costs"]["total"] > 0 and data["costs"]["invalidated_attempts"] == 0
    json_path, md_path = report.write_report(data, tmp_path / "report")
    md = md_path.read_text()
    assert all(f"`{c}`" in md for c in expected_cells())
    assert report.CAVEAT_SINGLE_RUN in md and frozen in md
    assert "**No audit**" in md
    assert json.loads(json_path.read_text())["cells"]
    # With a passing audit of these very records, every cell is marked passed.
    audit_json, _ = audit.write_audit(run_audit(e2e_matrix), tmp_path / "audit")
    audited = report.build_report(
        e2e_matrix.runs, split_mode=None, audit_path=audit_json, require_audit=True
    )
    assert {c["audit"]["status"] for c in audited["cells"]} == {report.AUDIT_PASSED}
    assert audited["comparisons"] == data["comparisons"]
    assert audited["audit"]["cells_not_passed"] == []


#: Where each report metric reads a published row (independent of report.METRICS): a
#: key of the scorer's record (None: ``score.primary``). WebShop's success is its
#: ``success`` metric (reward == 1), its mean score the primary.
METRIC_KEYS: dict[str, str | None] = {"success_rate": None}
WEBSHOP_METRIC_KEYS: dict[str, str | None] = {
    "success_rate": "success",
    "score": None,
    "success_within_15": "success_within_15",
}


def recorded_scores(matrix: Matrix, cell_id: str, metric: str) -> dict[str, float]:
    """raw_id -> the metric of the cell's published test row, straight from
    results.jsonl."""
    webshop = cell_id.startswith("webshop-")
    key = WEBSHOP_METRIC_KEYS[metric] if webshop else METRIC_KEYS[metric]
    out = {}
    for row in read_jsonl(matrix.cell(cell_id).results):
        score = row["score"]
        value = score["primary"] if key is None else score["metrics"][key]
        out[str(row["raw_id"])] = float(value)
    return out


def test_the_audit_passes_on_the_matrix(e2e_matrix, tmp_path):
    result = run_audit(e2e_matrix)
    problems = [
        f"{f.cell_id} {f.invariant} {f.check}: {f.detail}"
        for cell in result.cells
        for f in cell.findings
        if f.severity == audit.VIOLATION
    ] + [f.detail for f in result.findings]
    assert not problems, "\n".join(problems)
    assert result.passed and result.exit_code == 0
    assert {c.cell_id for c in result.cells} == expected_cells()
    for cell in result.cells:
        marks = cell.invariants()
        expected = {inv: "pass" for inv in audit.INVARIANTS}
        assert marks == expected, (cell.cell_id, marks)
        assert cell.checked["H10.1"] >= 3 and cell.checked["H10.3"] > 0
        assert cell.info["trajectories"] == (3 if cell.method == "none" else 6)
    assert result.auditor_fingerprint == result.frozen_fingerprint
    out = audit.write_audit(result, tmp_path / "audit")
    assert "Result: **PASS**" in out[1].read_text()


def test_nothing_left_the_machine(e2e_matrix):
    """Every gateway forwarded to the fake upstream, and every call it billed is one the fake
    answered: no model call reached a provider, nothing was paid."""
    starts = read_jsonl(e2e_matrix.runs / "_gateway" / "starts.jsonl")
    assert starts and {s["upstream"] for s in starts} == {TEST_UPSTREAM}
    billed = 0
    for cell_id in expected_cells():
        state = read_json(e2e_matrix.cell(cell_id).attempt(1).state)
        rows = read_jsonl(e2e_matrix.runs / state["run_id"] / "gateway.jsonl")
        calls = [r for r in rows if r.get("event") == "call"]
        assert {r["provider"] for r in calls} <= {"FakeProvider", "Google"}
        billed += sum(r["billed_attempts"] for r in calls)
    assert billed == len(e2e_matrix.upstream.requests)


# --- Records one part of the harness writes and another reads ----------------------------


SEAM_NOTE = "rerun with the fixed learner"


class _RefusingUpstream:
    """The matrix's fake model, except: J calls every trajectory a success (so every
    learning item induces a workflow), the upstream refuses the first actor call (413,
    the first learning item's first step: it ends unanswered, scored 0) and the first
    ``learner_refusals`` AWM learner calls (413: the first induction and its retry with
    halved budgets, so that induction is rejected and learning goes on)."""

    def __init__(self, model: Any, *, learner_refusals: int = 2) -> None:
        self.model = model
        self.learner_refusals = learner_refusals
        self.actor_refusals = 1
        self.refused: list[str] = []

    @staticmethod
    def _refusal() -> Any:
        from evals.skill_bench.tests.fakes import Reply

        return Reply(413, {"error": {"code": 413, "message": "Payload Too Large"}})

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Any:
        from evals.skill_bench.methods import awm as A
        from evals.skill_bench.methods import judge as J
        from evals.skill_bench.tests.rollout_support import system_text, text_reply
        from evals.skill_bench.tests.s1_support import first_text, judged

        if endpoint == "chat/completions":
            if system_text(body) == J.system_prompt():
                return text_reply(judged("success"), model=body["model"])
            learner = len(body["messages"]) == 1 and first_text(body).startswith(
                A.instruction_text()
            )
            if learner and self.learner_refusals > 0:
                self.learner_refusals -= 1
                self.refused.append("learner")
                return self._refusal()
            if not learner and self.actor_refusals > 0:
                self.actor_refusals -= 1
                self.refused.append("actor")
                return self._refusal()
        return self.model(endpoint, body)


def _seam_frozen_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This test's own frozen record, a copy frozen to this tree: run.py's automatic
    freeze writes it, never the session's copy the other e2e tests compare with."""
    import shutil

    from evals.skill_bench.tests.run_support import FIXED_GIT
    from evals.skill_bench.tools import freeze_protocol

    copy = tmp_path / "_frozen_record" / "protocol_frozen.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(protocol.frozen_record_path(), copy)
    freeze_protocol.freeze(
        reason="the seam test's baseline", path=copy, git=lambda: dict(FIXED_GIT)
    )
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, str(copy))
    return copy


def test_versions_supersedes_refusals_and_removed_data_flow_end_to_end(
    e2e_matrix, tmp_path, monkeypatch, removed_method
):
    """The records one part of the harness writes, read by the parts that report them,
    end to end on a copy of the matrix's ALFWorld AWM cell and the no-memory cell it is
    compared with:

    * ``run.py --new-attempt --note`` over the tested attempt 1: attempt 1 is kept,
      marked superseded, and attempt 2 names it;
    * attempt 2's first actor call and its first induction (and the induction's retry
      with halved budgets) are refused by the provider (413): the item is scored 0 and
      flagged ``provider_unserved``, the induction is rejected and counted;
    * infrastructure stops attempt 2's learning; it resumes under another protocol,
      which run.py freezes automatically and records as version segment 1;
    * a removed method's leftover cell (conftest's synthetic ``removed_method``, its
      deletion decided for this test) is deleted by ``tools/remove_method_data.py``,
      which writes its manifest first;

    and the audit, the report, ``tools/summarize_learning.py`` and the supervisor's
    reader (``cell_records.recorded_problems``) read each record as specified: warnings
    and listings, never a validity violation, and the same money before and after the
    deletion."""
    from evals.skill_bench import cell_records, run
    from evals.skill_bench.methods.awm import AgentWorkflowMemory
    from evals.skill_bench.tests.e2e_support import copy_cells
    from evals.skill_bench.tests.fakes import FakeUpstream, serve_gateway
    from evals.skill_bench.tests.run_support import InfraFailure, uncaptured_tree
    from evals.skill_bench.tests.test_report import _call, _row, make_cell
    from evals.skill_bench.tools import freeze_protocol
    from evals.skill_bench.tools import remove_method_data as removal

    record = _seam_frozen_record(tmp_path, monkeypatch)
    monkeypatch.setattr(
        removal,
        "DELETION_DECISIONS",
        {removed_method: {"decision": "example decision", "quote": "delete it"}},
    )
    awm = cell_spec("alfworld", "awm")
    none = cell_spec("alfworld", "none")
    runs = copy_cells(e2e_matrix, tmp_path, [awm.cell_id, none.cell_id])
    paths = cell_records.CellPaths(runs / awm.cell_id)
    upstream = _RefusingUpstream(e2e_matrix.model)
    fake = FakeUpstream(responder=upstream)
    started = {"after_task": 0}

    class StoppedOnce(AgentWorkflowMemory):
        """AWM whose third ``after_task`` of this test fails once (infrastructure)."""

        async def after_task(self, trajectory: Any) -> None:
            started["after_task"] += 1
            if started["after_task"] == 3:
                raise InfraFailure("the learner's store is down")
            await super().after_task(trajectory)

    def launch(url: str, **options: Any) -> Any:
        runner = run.CellRunner(
            awm,
            run.RunOptions(
                out_dir=paths.root,
                gateway_url=url,
                gateway_runs_dir=runs,
                preflight=False,
                idle_timeout_s=60.0,
                test_concurrency=2,
                **options,
            ),
            env_factory=e2e_matrix.env_factory,
            method_factory=StoppedOnce,
        )
        runner.capture_tree = uncaptured_tree
        return asyncio.run(runner.run())

    tree = protocol.fingerprint_manifest()
    with serve_gateway(
        tmp_path, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
    ) as (_gw, url):
        stopped = launch(url, new_attempt=True, note=SEAM_NOTE)
        assert (stopped.status, stopped.attempt) == ("learning_interrupted", 2), (
            stopped.detail
        )
        # Another protocol (a constant only the report reads) for the resume.
        monkeypatch.setattr(
            protocol, "BOOTSTRAP_RESAMPLES", protocol.BOOTSTRAP_RESAMPLES + 1
        )
        changed = protocol.fingerprint_manifest()
        assert changed["fingerprint"] != tree["fingerprint"]
        resumed = launch(url)
        assert (resumed.status, resumed.attempt) == ("complete", 2), resumed.detail
    monkeypatch.setattr(
        protocol, "BOOTSTRAP_RESAMPLES", protocol.BOOTSTRAP_RESAMPLES - 1
    )
    assert protocol.fingerprint_manifest()["fingerprint"] == tree["fingerprint"]
    assert upstream.refused == ["actor", "learner", "learner"]
    # The tree is back at the earlier protocol: freeze it back (as the next launch would).
    freeze_protocol.freeze(reason="back to the tree", path=record)

    # --- what run.py and the method wrote -------------------------------------------
    first = cell_records.read_json(paths.attempt(1).state)
    second = cell_records.read_json(paths.attempt(2).state)
    assert (first["state"], first["invalidation"], first["previous_state"]) == (
        "invalid",
        run.INVALIDATION_SUPERSEDED,
        "complete",
    )
    assert first["superseded"]["note"] == SEAM_NOTE
    assert first["superseded"]["by_attempt"] == 2
    assert first["superseded"]["scored_items"] == 3
    assert second["supersedes"] == 1 and second["note"]["text"] == SEAM_NOTE
    start, later = second["version_segments"]
    assert start["protocol"]["fingerprint"] == tree["fingerprint"]
    assert start["changed"] is None
    assert later["index"] == 1 and later["phase"] == "learning"
    assert later["protocol"]["fingerprint"] == changed["fingerprint"]
    assert later["changed"]["protocol"] is True
    assert later["changed"]["protocol_diff"]["constants_changed_keys"] == [
        "bootstrap_resamples"
    ]
    assert later["auto_freeze"]["status"] == "frozen"
    assert later["learning_from"]["position"] == 2
    assert (paths.root / later["version_file"]).is_file()
    assert second["versions_mixed"]["protocol"] is True
    learning = cell_records.read_jsonl(paths.attempt(2).learning)
    unserved = [r for r in learning if r.get("provider_unserved")]
    assert len(unserved) == 1 and unserved[0]["position"] == 0
    assert unserved[0]["end_reason"] == "provider_unserved"
    assert unserved[0]["score"]["primary"] == 0.0
    assert unserved[0]["score"]["answered"] is False  # unanswered, not a lost game
    assert unserved[0]["failure"]["code"] == "provider_unserved"
    assert unserved[0]["failure"]["status"] == 413
    failures = cell_records.read_jsonl(paths.attempt(2).failures)
    assert [
        (r["code"], r["recovery"], r.get("stage"))
        for r in failures
        if r["code"] == "provider_unserved"
    ] == [("provider_unserved", "score_and_continue", "actor")]
    frozen = cell_records.read_json(paths.attempt(2).frozen_snapshot)["snapshot"]
    assert frozen["counters"]["induction_rejected"] == 1
    assert frozen["counters"]["induction_retries"] == 1
    tolerated = cell_records.tolerated_failures(paths.attempt(2))
    assert [r["position"] for r in tolerated["learning"]["provider_unserved"]] == [0]
    assert tolerated["method"]["counters"]["induction_rejected"] == 1
    # The supervisor's source: what the attempt went on despite.
    problems = cell_records.recorded_problems(paths.attempt(2))
    assert problems["provider_unserved"] == 1
    assert problems["induction_rejected"] == 1 and problems["induction_retries"] == 1

    # --- a removed method's leftover cell, then its deletion -------------------------
    removed_cell = make_cell(
        runs,
        "alfworld",
        removed_method,
        test=[_row("a1", 1.0)],
        calls=[_call("actor", "test", 0.02, item="u1")],
    )
    before = report.build_report(runs, split_mode=None)
    code, removed = removal.run(
        runs, removed_method, yes=True, note="seam test", argv=[]
    )
    assert code == removal.EXIT_OK, removed
    assert not removed_cell.exists()
    (manifest,) = removal.load_manifests(runs)
    assert manifest.problems == [] and manifest.data["state"] == removal.STATE_COMPLETE
    assert [c["cell_id"] for c in manifest.data["cells"]] == [removed_cell.name]

    # --- the audit ------------------------------------------------------------------
    result = audit.audit_runs(
        runs,
        env_factory=e2e_matrix.env_factory,
        expected_upstream=TEST_UPSTREAM,
        split_mode=None,
    )
    violations = [
        f"{f.cell_id} {f.invariant}:{f.check}: {f.detail}"
        for cell in result.cells
        for f in cell.findings
        if f.severity == audit.VIOLATION
    ] + [f.detail for f in result.findings if f.severity == audit.VIOLATION]
    assert not violations, "\n".join(violations)
    assert result.passed and result.exit_code == 0
    cells = {c.cell_id: c for c in result.cells}
    assert set(cells) == {awm.cell_id, none.cell_id}
    audited = cells[awm.cell_id]
    assert audited.attempt == 2
    warned = {f"{f.invariant}:{f.check}" for f in audited.findings}
    assert {
        "H10.3:attempt_superseded",
        "H10.6:mixed_version_attempt",
        "H10.3:failure_counted_and_continued",
    } <= warned, warned
    (superseded,) = audited.info["superseded_attempts"]
    assert (superseded["attempt"], superseded["by_attempt"]) == (1, 2)
    assert superseded["note"] == SEAM_NOTE
    (segment,) = audited.info["version_segments"]
    assert segment["segment"] == 1 and segment["fingerprint"] == changed["fingerprint"]
    assert segment["known"] is True
    assert audited.info["tolerated"]["method"]["counters"]["induction_rejected"] == 1
    assert audited.info["provider_unserved"] == {"learning": 1, "test": 0}
    assert [d["cell_id"] for d in result.deleted] == [removed_cell.name]
    written, _ = audit.write_audit(result, tmp_path / "audit")

    # --- the report -----------------------------------------------------------------
    after = report.build_report(runs, split_mode=None, audit_path=written)
    assert after["costs"]["all_cells_total"] == pytest.approx(
        before["costs"]["all_cells_total"]
    )
    assert after["costs"]["total"] == pytest.approx(before["costs"]["total"])
    assert [c["cell_id"] for c in after["removed"]["cells"]] == [removed_cell.name]
    assert after["removed"]["deleted"]["cells"] == 1
    assert after["removed"]["cells"][0]["on_disk"] is False
    reported = {c["cell_id"]: c for c in after["cells"]}[awm.cell_id]
    assert reported["complete"] and reported["reported_attempt"] == 2
    one, two = reported["attempts_detail"]
    assert (one["attempt"], one["reported"], one["invalidation"]) == (
        1,
        False,
        run.INVALIDATION_SUPERSEDED,
    )
    assert one["superseded"]["by_attempt"] == 2
    assert (two["attempt"], two["reported"], two["note"], two["supersedes"]) == (
        2,
        True,
        SEAM_NOTE,
        1,
    )
    assert reported["versions"]["mixed"] is True
    assert reported["method_refusals"]["counters"]["induction_rejected"] == 1
    assert reported["degraded"]["learning"]
    assert reported["audit"]["status"] == report.AUDIT_PASSED
    checks = {w["check"] for w in after["warnings"]}
    assert {"mixed_version", "method_refusals", "degraded_items"} <= checks, checks
    # The money of the superseded attempt stays counted, apart from the reported one.
    other = reported["costs"]["other_attempts"]
    assert other["superseded"] == [1]
    assert other["superseded_cost"] == pytest.approx(one["cost"]) and one["cost"] > 0
    md = report.render_markdown(after)
    assert SEAM_NOTE in md and removed_cell.name in md

    # --- what the learning summary prints -------------------------------------------
    learned = summarize_learning.summarize(runs)
    assert learned["removed_deleted"] == 1
    (gone,) = learned["removed"]
    assert (gone["cell_id"], gone["on_disk"]) == (removed_cell.name, False)
    assert gone["deleted"]["removal_id"] == manifest.removal_id
    (cell,) = [c for c in learned["cells"] if c["cell_id"] == awm.cell_id]
    assert cell["refusals"]["induction_rejected"] == 1


# --- MIRIX on its real per-cell server (integration: the compose Postgres) ------------------------


def await_(awaitable: Any) -> Any:
    """Run one coroutine to its end from a synchronous test."""
    return asyncio.run(awaitable)


def _postgres_up() -> bool:
    from evals.skill_bench.infra.mirix_server import PostgresAdmin

    admin = PostgresAdmin()
    try:
        with socket.create_connection((admin.host, admin.port), timeout=1):
            return True
    except OSError:
        return False


class _MirixCells:
    """run.py's MIRIX server factory for the e2e matrix: the real ``MirixServer`` of each
    cell attempt, on a database named for this test run (so it never meets a real run's
    database of the same cell id, and is dropped afterwards), plus a method that digests
    its database when the memory freezes and when the attempt ends: the test phase may
    write nothing but retrieval log rows (H1.6, H10.4, below MIRIX's API)."""

    def __init__(self) -> None:
        self.tag = f"e2e{secrets.token_hex(3)}"
        self.databases: dict[str, str] = {}  # cell id -> its database
        self.digests: dict[str, list[dict[str, Any]]] = {}

    def server(self, spec: Any, mode: Any) -> Any:
        from evals.skill_bench.infra.mirix_server import MirixServer

        renamed = replace(spec, cell_id=f"{self.tag}-{spec.cell_id}")
        self.databases[spec.cell_id] = renamed.db_name
        return MirixServer(renamed, mode=mode)

    def method(self) -> Any:
        from evals.skill_bench.methods.mirix import MirixMethod
        from evals.skill_bench.tests.mirix_db import database_digest

        cells = self

        class DigestingMirix(MirixMethod):
            async def finalize(self) -> None:
                await super().finalize()
                assert self.cell is not None
                db = cells.databases[self.cell.cell_id]
                cells.digests.setdefault(self.cell.cell_id, []).append(
                    await database_digest(db)
                )

            async def close(self) -> None:
                frozen = self._state is not None and self._state.frozen
                await super().close()
                if frozen and self.cell is not None:
                    db = cells.databases[self.cell.cell_id]
                    cells.digests.setdefault(self.cell.cell_id, []).append(
                        await database_digest(db)
                    )

        return DigestingMirix()

    def drop(self) -> None:
        from evals.skill_bench.infra.mirix_server import (
            PostgresAdmin,
            drop_cell_database,
        )

        for db in self.databases.values():
            asyncio.run(drop_cell_database(PostgresAdmin(), db))


@pytest.mark.integration
def test_mirix_joins_the_matrix(tmp_path):
    """The matrix with MIRIX on its real per-cell server (run.py starts it; the fake
    upstream plays MIRIX's model through the real gateway): ALFWorld and WebShop (the
    scripted shop behind the real server.py) x {none, MIRIX}; 3 learning items each, 2
    test items. Every cell completes; the agent context reached MIRIX's
    distiller for every ALFWorld and WebShop learning session, by the same code path;
    MIRIX learned a skill before the freeze; its test phase wrote nothing but retrieval
    log rows; its experience pool holds the three experiences its curator
    cited none of; every test item's injection is MIRIX's own rendering of that skill; the
    report compares MIRIX with the no-memory method; the audit (the MIRIX usage
    reconciliation and the per-item check included) passes, as do both CLIs; and
    ``tools/summarize_learning.py`` reports the skill, its experiences and its run."""
    if not _postgres_up():
        pytest.skip(
            "skill_bench Postgres not reachable "
            "(docker compose -f evals/skill_bench/infra/compose.yaml up -d)"
        )
    from mirix.client.render import render_retrieved_memories

    from evals.skill_bench.tests.mirix_model import SKILL

    cells = _MirixCells()
    benchmarks = ("alfworld", "webshop")
    try:
        matrix = build_matrix(
            tmp_path,
            methods=("none", "mirix"),
            benchmarks=benchmarks,
            n_test=2,
            method_factories={"mirix": cells.method},
            mirix_server_factory=cells.server,
        )
        assert {c: o.status for c, o in matrix.outcomes.items()} == {
            c: "complete" for c in matrix.outcomes
        }, {c: o.detail for c, o in matrix.outcomes.items()}
        # <benchmark>-<model>-<method>[-smoke-...].
        mirix_cells = [c for c in matrix.outcomes if c.split("-")[2] == "mirix"]
        assert len(mirix_cells) == len(benchmarks)
        assert matrix.mirix is not None
        assert matrix.mirix.kinds["distiller"] == 3 * len(benchmarks)
        for cell_id in mirix_cells:
            paths = matrix.cell(cell_id).attempt(1)
            frozen = read_json(paths.frozen_snapshot)["snapshot"]
            # MIRIX learned, by itself, before the freeze.
            (skill,) = frozen["skills"]
            assert skill["name"] == SKILL["name"]
            assert set(frozen["skill_versions"][skill["id"]]) == {"1"}
            assert [r["status"] for r in frozen["learning_runs"]] == ["succeeded"]
            # Its experience pool: the fake curator cites none of the three
            # experiences and triages none, so all three wait in the pool (deferred).
            pool = frozen["experience_pool"]
            assert (pool["active_at_freeze"], pool["entered"], pool["unaccounted"]) == (
                3,
                3,
                0,
            )
            assert pool["dispositions"]["defaulted"] == 3
            assert pool["uncited_writes"] == 1
            assert pool["history_skipped"] == {"empty_pool": 1}
            # The test phase wrote nothing (the snapshot after it, and the database).
            state = read_json(paths.state)
            assert [c["equal"] for c in state["snapshot_checks"]] == [True]
            before, after = cells.digests[cell_id]
            assert before == after
            # Every test item got MIRIX's own rendering of the skill.
            expected = render_retrieved_memories(
                {"memories": {"procedural": {"items": [skill]}}}
            )
            injected = [r for r in read_jsonl(paths.injections) if r["phase"] == "test"]
            assert len(injected) == 2
            assert all(r["text"] == expected for r in injected)
            assert all(r["item_ids"] == [skill["id"]] for r in injected)
            # Every learning session's distiller input shows the benchmark's agent
            # context verbatim (ALFWorld and WebShop alike).
            env = matrix.envs[cell_id.split("-")[0]]
            context = env_agent_context(env)
            assert context
            learned_uids = [r["task_uid"] for r in read_jsonl(paths.learning)]
            assert len(learned_uids) == 3
            for uid in learned_uids:
                (shown,) = matrix.mirix.distiller_input(uid)
                assert f"<agent_context>\n{context}\n</agent_context>" in shown

        data = report.build_report(matrix.runs, split_mode=None)
        compared = [c for c in data["comparisons"] if "skipped" not in c]
        assert {c["method"] for c in compared} == {"mirix"}
        assert {c["group"].split("-")[0] for c in compared} == set(benchmarks)
        for cell in data["cells"]:
            if cell["cell_id"] in mirix_cells:
                assert cell["injections"]["test"]["empty"] == 0
                assert cell["mirix"]["runs_by_status"] == {"succeeded": 1}
                assert cell["mirix"]["skills"] == 1
                assert cell["mirix"]["retrievals"]["test"]["refused"] == 0
                assert cell["mirix"]["experience_pool"]["active_at_freeze"] == 3

        result = run_audit(matrix)
        assert result.passed, result.to_markdown()
        for cell in result.cells:
            if cell.method == "mirix":
                usage = cell.info["mirix_usage"]
                assert usage["runs_with_usage"] == usage["runs"] == 1
                retrievals = cell.info["mirix_test_retrievals"]
                assert retrievals["scored_items_checked"] == 2
                assert retrievals["refused"] == retrievals["lost_calls"] == 0
        argv = ["--runs-dir", str(matrix.runs), "--split-mode", "smoke"]
        assert (
            audit.main(
                [
                    *argv,
                    "--expected-upstream",
                    TEST_UPSTREAM,
                    "--out",
                    str(tmp_path / "audit"),
                ]
            )
            == 0
        )
        assert (
            report.main(
                [
                    *argv,
                    "--out",
                    str(tmp_path / "report"),
                    "--resamples",
                    "1000",
                ]
            )
            == 0
        )
        # What each method learned, as the learning summary prints it.
        learned = summarize_learning.summarize(matrix.runs)
        assert learned["nothing_learned"] == []
        for cell in learned["cells"]:
            if cell["method"] == "mirix":
                assert cell["learned"]["skills"] == 1
                assert cell["learned"]["skill_list"][0]["name"] == SKILL["name"]
                assert cell["learned"]["experiences"] == 3
                assert cell["learned"]["runs_by_status"] == {"succeeded": 1}
                assert cell["learned"]["experience_pool"]["active_at_freeze"] == 3
                assert cell["test_injections"]["empty"] == 0
        assert summarize_learning.main(argv) == summarize_learning.EXIT_OK
    finally:
        cells.drop()


@pytest.mark.integration
@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
def test_mirix_learns_from_the_skills_it_served_through_run_py(tmp_path, benchmark):
    """MIRIX's learning from the skills it served, through ``run.py``'s runner, the
    same for ALFWorld and WebShop. Seven learning items: MIRIX's threshold run learns
    a skill from the first five, the sixth and seventh are served it (``<skills_served>``
    in their distiller input, next to the agent context), and the freeze's flush distils
    them. The scripted model answers as real models do: MIRIX's reply object with quotes
    copied from each transcript, and a pool curator whose first write also cites an
    experience with a quote the write does not hold.

    MIRIX stores the served sessions' feedback with the values given (followed: yes,
    effect: helped, each verified by its quote), and the first five's failures, which
    only the agent's words show, as unclear; the flush run's curator batch shows
    the served skill's usage record, its provenance counting only the verified
    citation; the citation without a verified quote is counted and its
    experience stays in the pool. ``summarize_learning`` and the report show all of it,
    with no agent-context warning, and the audit passes."""
    if not _postgres_up():
        pytest.skip(
            "skill_bench Postgres not reachable "
            "(docker compose -f evals/skill_bench/infra/compose.yaml up -d)"
        )
    from evals.skill_bench.methods.mirix import learning_context
    from evals.skill_bench.tests.mirix_db import experience_rows, skill_feedback_rows
    from evals.skill_bench.tests.mirix_model import (
        SKILL,
        UNVERIFIED_QUOTE,
        USAGE_SECTION_HEAD,
        MirixModel,
    )

    cells = _MirixCells()
    model = MirixModel(p16_replies=True, pool_curator=True, unverified_citation=True)
    try:
        matrix = build_matrix(
            tmp_path,
            methods=("mirix",),
            benchmarks=(benchmark,),
            n_train=7,
            n_test=2,
            method_factories={"mirix": cells.method},
            mirix_server_factory=cells.server,
            mirix_model=model,
        )
        ((cell_id, outcome),) = matrix.outcomes.items()
        assert outcome.status == "complete", outcome.detail
        paths = matrix.cell(cell_id).attempt(1)
        frozen = read_json(paths.frozen_snapshot)["snapshot"]
        skill = next(s for s in frozen["skills"] if s["name"] == SKILL["name"])
        uids = [r["task_uid"] for r in read_jsonl(paths.learning)]
        assert len(uids) == 7
        # The threshold run learned the skill from the first five; the last two were
        # served it, and the flush distilled them.
        learning = [r for r in read_jsonl(paths.injections) if r["phase"] == "learning"]
        assert [r["item_ids"] for r in learning] == [[]] * 5 + [[skill["id"]]] * 2
        runs = sorted(frozen["learning_runs"], key=lambda r: r["created_at"])
        assert [(r["trigger"], r["status"]) for r in runs] == [
            ("close", "succeeded"),
            ("flush", "succeeded"),
        ]
        assert sorted(runs[1]["session_ids"]) == sorted(uids[5:])
        context = env_agent_context(matrix.envs[benchmark])
        assert matrix.mirix is model
        for index, uid in enumerate(uids):
            (shown,) = model.distiller_input(uid)
            assert f"<agent_context>\n{context}\n</agent_context>" in shown
            assert ("<skills_served>" in shown) == (index >= 5)

        # The served sessions' feedback, stored with the values given.
        given = {g["session_id"]: g for g in model.p16_given}
        assert all(given[uid]["agent_quote"] for uid in uids), given
        assert all(given[uid]["environment_quote"] for uid in uids[5:]), given
        rows = await_(skill_feedback_rows(cells.databases[cell_id]))
        assert sorted(
            (
                r["session_id"],
                r["skill_id"],
                r["followed"],
                r["effect"],
                r["session_outcome"],
                r["stated_followed"],
                r["stated_effect"],
            )
            for r in rows
        ) == sorted(
            (uid, skill["id"], "yes", "helped", "succeeded", None, None)
            for uid in uids[5:]
        )
        # The first five stated failures only the agent's words showed: unclear.
        distilled = {
            str(e["session_id"]): e
            for run in runs
            for e in run["distill_metrics"]
            if e.get("status") == "ok"
        }
        assert [distilled[u]["session_outcome"] for u in uids] == ["unclear"] * 5 + [
            "succeeded"
        ] * 2
        assert [distilled[u]["outcomes_self_reported"] for u in uids] == [1] * 5 + [
            0
        ] * 2

        # The threshold run's write cited e0 with a verified quote and e3 without;
        # only e0's citation counts: e0 is used, and e3, which no counted citation or
        # triage addressed, stays pending: the flush run's batch curates it again (and
        # cites it, with a verified quote, for its own skill). The skill whose write did
        # not quote it never records it as a source.
        first, second = model.curations
        e0, e3 = first["experiences"][0], first["experiences"][3]
        (batch1,) = runs[0]["curator_metrics"]
        (created,) = batch1["changes"]["created"]
        assert batch1["citations"] == {e0: [created]}
        assert batch1["citations_unverified"] == 1
        assert batch1["unverified_citations"] == {e3: [created]}
        assert batch1["dispositions"] == {
            "used": 1,
            "covered": 0,
            "not_worth": 1,
            "deferred": 2,
            "defaulted": 1,
            "unaddressed": 0,
        }
        assert e3 in second["experiences"] and e3 not in second["lessons"]
        experiences = await_(experience_rows(cells.databases[cell_id], [e0, e3]))
        assert experiences[e0]["cited_skill_ids"] == [created]
        assert created not in (experiences[e3]["cited_skill_ids"] or [])
        assert frozen["experience_pool"]["citations_unverified"] == 1

        # The flush run's batch shows the served skill's usage record; its
        # provenance counts only the verified citation (one session's experience).
        payload1, payload2 = model.curator_payloads
        assert USAGE_SECTION_HEAD not in payload1
        usage = payload2.split(USAGE_SECTION_HEAD, 1)[1]
        assert f"Skill {SKILL['name']} (ID: {skill['id']}), served in this batch's" in (
            usage
        )
        assert "- served in 2 distilled session(s); most recent: " in usage
        assert "followed yes 2/" in usage and "effect helped 2/" in usage
        assert "cited, with a verified quote, by experiences of 1 distinct" in usage
        assert UNVERIFIED_QUOTE not in json.dumps(frozen["skills"])
        (batch2,) = runs[1]["curator_metrics"]
        assert batch2["skill_usage"]["skill_ids"][0] == skill["id"]
        assert batch2["skill_usage"]["served_ids"] == [skill["id"]]

        # What the learning summary and the report print.
        summary = learning_context(frozen["learning_runs"])
        assert summary is not None
        assert (
            summary["sessions"],
            summary["with_agent_context"],
            summary["without_agent_context"],
            summary["agent_context_chars"],
            summary["served_sessions"],
            summary["feedback"]["followed"],
            summary["feedback"]["effect"],
            summary["usage"]["shown"],
            summary["usage"]["batches"],
            summary["citations_unverified"],
        ) == (7, 7, [], [len(context)], 2, {"yes": 2}, {"helped": 2}, 1, 2, 1)
        assert summary["skills"][0]["skill_id"] == skill["id"]
        learned = summarize_learning.summarize(matrix.runs)
        (cell,) = learned["cells"]
        assert cell["learned"]["learning_context"] == summary
        assert not [w for w in cell["warnings"] if "agent context" in w]
        text = summarize_learning.render_text(learned)
        assert "skill feedback: entries for 2 of 2 served (0 not shown); followed " in (
            text
        )
        assert (
            f"      - {SKILL['name']}: served 2; followed yes 2; effect helped 2"
            in (text)
        )
        data = report.build_report(matrix.runs, split_mode=None)
        (reported,) = data["cells"]
        assert reported["mirix"]["learning_context"] == summary
        result = run_audit(matrix)
        assert result.passed, result.to_markdown()
    finally:
        cells.drop()
