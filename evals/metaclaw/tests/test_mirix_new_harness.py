"""C5 — new MetaClaw validation harness (message-by-message records ingestion).

These tests lock the unit-testable core of the new harness WITHOUT a live LLM,
MIRIX server, or Postgres. Every §3 leakage guard gets at least one test that
asserts on the ACTUAL payload / behaviour the harness produces:

  G1  no oracle field reaches /v1/skills/distill-round — assert the built payload.
  G2  no MIRIX code path writes the bench workspace / agent-state dir.
  G3  one-round lag preserved — the harness never sends a round's own grade.
  G4  the runner reads bench report.json summary.accuracy; no local re-scoring.
  G5  evolve fires at {5,10,15,…} (not {6,11,16}); watermark = round_index;
      records with round_index < watermark only.

Plus: arm selection (no-skills / mirix-old / mirix-new / native), the
distiller _FAILURE_MARKERS ↔ real bench template match (point 6), and the
adapter's distill_round / evolve_from_records HTTP wire behaviour.

All HTTP is mocked with httpx.MockTransport — no network, no server.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from typing import List

import httpx
import pytest

from evals.metaclaw.mirix_adapters.evolver_adapter import (
    DEFAULT_EVOLVE_EVERY_N_ROUNDS,
    DISTILL_FORBIDDEN_KEYS,
    DISTILL_ROUND_ENDPOINT_PATH,
    EVOLVE_FROM_RECORDS_ENDPOINT_PATH,
    MirixEvolverAdapter,
    RoundEvolutionTracker,
    build_distill_round_payload,
    should_evolve_at,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Helpers — an evolver adapter wired to a MockTransport recorder               #
# --------------------------------------------------------------------------- #

_OPENAPI_OK = {
    "openapi": "3.1.0",
    "paths": {
        "/v1/skills/evolve": {"post": {}},
        "/v1/skills/distill-round": {"post": {}},
        "/v1/skills/evolve-from-records": {"post": {}},
    },
}


class _Recorder:
    """Captures every request the adapter makes; returns canned JSON bodies."""

    def __init__(self):
        self.requests: List[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if request.url.path == DISTILL_ROUND_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={"success": True, "record": None, "session_flushed": False},
            )
        if request.url.path == EVOLVE_FROM_RECORDS_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "run_id": "evrun-x",
                    "skipped": False,
                    "changes": {"created": ["s1"], "edited": [], "deleted": []},
                    "summary": {"consumed_count": 3},
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    def posts_to(self, path: str) -> List[httpx.Request]:
        return [r for r in self.requests if r.url.path == path and r.method == "POST"]

    def body_of(self, request: httpx.Request) -> dict:
        return json.loads(request.content.decode("utf-8"))


def _make_adapter(rec: _Recorder, **kw) -> MirixEvolverAdapter:
    transport = httpx.MockTransport(rec.handler)
    return MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-eval",
        transport=transport,
        **kw,
    )


# =========================================================================== #
# G1 — distill-round payload carries NO oracle field                          #
# =========================================================================== #


def test_g1_payload_has_only_question_and_answer():
    """The built body's `turn` has EXACTLY {prompt_text, response_text}."""
    body = build_distill_round_payload(
        day="day03",
        round_id="r4",
        round_index=4,
        query="[Previous Feedback] You missed option E\n\nWhat is X? Answer \\bbox{}",
        answer="I think \\bbox{A}",
        user_id="u-eval",
        session_id="day03_uuid",
        session_done=False,
    )
    assert set(body["turn"].keys()) == {"prompt_text", "response_text"}
    assert body["turn"]["prompt_text"].startswith("[Previous Feedback]")
    assert body["turn"]["response_text"] == "I think \\bbox{A}"


def test_g1_no_oracle_key_anywhere_in_payload():
    """No oracle-derived key appears at ANY depth of the built payload, even if
    an oracle-laden kwargs dict is (accidentally) in scope — the builder only
    reads its explicit args, so leakage is structurally impossible."""
    body = build_distill_round_payload(
        day="day01",
        round_id="r1",
        round_index=1,
        query="question text only",
        answer="answer text only",
        user_id="u-eval",
    )
    flat = json.dumps(body).lower()
    # The MC oracle answer set, the file_check command, the inline score, the
    # per-option feedback hints, reward — none may appear.
    for forbidden in DISTILL_FORBIDDEN_KEYS:
        assert forbidden not in body, f"forbidden top-level key {forbidden!r} leaked"
    # And specifically the oracle *values* a leak would carry:
    for needle in (
        "inline_score",
        "expect_exit",
        "feedback",
        '"reward"',
        "eval.command",
    ):
        assert needle not in flat, f"oracle token {needle!r} leaked into payload"
    assert body["user_id"] == "u-eval"


def test_g1_query_passed_verbatim_no_tail_truncation():
    """The leading [Previous Feedback] block survives — query is mapped to
    prompt_text verbatim (no PROMPT_TAIL_CHARS tail-truncation that drops it)."""
    feedback = "[Previous Feedback] " + ("Z" * 5000) + "\n\nthe actual question"
    body = build_distill_round_payload(
        day="d",
        round_id="r1",
        round_index=1,
        query=feedback,
        answer="a",
        user_id="u",
    )
    assert body["turn"]["prompt_text"] == feedback  # untouched, head intact


# =========================================================================== #
# G3 — one-round lag: the harness never sends a round's OWN grade              #
# =========================================================================== #


@pytest.mark.asyncio
async def test_g3_distill_round_sends_only_query_and_answer_not_own_grade():
    """`distill_round` POSTs {day, round_id, round_index, turn} — the round's own
    inline_score / passed / eval.* is NEVER in the request body (the distiller
    derives t's outcome only when t+1 arrives, server-side)."""
    rec = _Recorder()
    adapter = _make_adapter(rec)
    await adapter.distill_round(
        day="day02",
        round_id="r3",
        round_index=3,
        query="[Previous Feedback] correct!\n\nnext question",
        answer="my answer \\bbox{B}",
        session_id="day02_uuid",
    )
    posts = rec.posts_to(DISTILL_ROUND_ENDPOINT_PATH)
    assert len(posts) == 1
    body = rec.body_of(posts[0])
    # The body carries the round's question + answer + index, but no grade.
    assert body["round_index"] == 3
    assert "inline_score" not in body
    assert "passed" not in body
    assert "eval" not in body
    assert set(body["turn"].keys()) == {"prompt_text", "response_text"}


def test_g3_round_t_feedback_is_for_t_minus_1_only():
    """The feedback the harness forwards lives INSIDE the query (round t's
    prompt), and the bench built it from round t-1's score. The payload itself
    never carries a separate 'feedback for round t' field."""
    body = build_distill_round_payload(
        day="d",
        round_id="r5",
        round_index=5,
        # The [Previous Feedback] here is r4's outcome (one-round lag), embedded
        # by the bench's with_feedback() — it is r5's prompt, not r5's grade.
        query="[Previous Feedback] You incorrectly selected option C\n\nQ5 text",
        answer="r5 answer",
        user_id="u",
    )
    # The only feedback present is the embedded prior-round block inside the
    # prompt; there is no standalone outcome/grade key for round 5.
    assert "feedback" not in body
    assert "inline_score" not in body
    assert body["turn"]["prompt_text"].count("[Previous Feedback]") == 1


# =========================================================================== #
# G5 — evolve fires at {5,10,15,…}; watermark = round_index                    #
# =========================================================================== #


def test_g5_should_evolve_at_fires_on_multiples_only():
    fires = [n for n in range(1, 21) if should_evolve_at(n, 5)]
    assert fires == [5, 10, 15, 20]
    # Never at {6,11,16}.
    assert not should_evolve_at(6, 5)
    assert not should_evolve_at(11, 5)
    assert not should_evolve_at(16, 5)
    # Non-positive cadence disables periodic evolution.
    assert not should_evolve_at(5, 0)
    assert not should_evolve_at(0, 5)


def test_g5_tracker_fires_at_5_10_15_per_session():
    t = RoundEvolutionTracker(every_n=5)
    fired = [i for i in range(1, 17) if t.note_round("day01")]
    assert fired == [5, 10, 15]
    # A second session is counted independently.
    fired2 = [i for i in range(1, 7) if t.note_round("day02")]
    assert fired2 == [5]
    assert t.completed("day01") == 16
    assert t.completed("day02") == 6


@pytest.mark.asyncio
async def test_g5_evolve_fires_at_round_5_with_watermark_5():
    """Feeding 5 rounds: exactly ONE evolve-from-records POST, at round 5, with
    before_round_index == 5 (the watermark = the just-completed round index)."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    for idx in range(1, 6):
        await adapter.distill_round(
            day="day01",
            round_id=f"r{idx}",
            round_index=idx,
            query=f"[Previous Feedback] ok\n\nQ{idx}",
            answer=f"a{idx}",
            session_id="day01_uuid",
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    assert len(evolves) == 1
    body = rec.body_of(evolves[0])
    assert body["before_round_index"] == 5  # watermark = current round index
    assert body["use_autonomous_budget"] is False  # validity run: formula-only
    assert adapter.records_evolution_events == 1


@pytest.mark.asyncio
async def test_g5_no_evolve_at_round_6_only_at_5_and_10():
    """11 rounds → evolves at 5 and 10 (watermarks 5,10), NOT at 6 or 11."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    for idx in range(1, 12):
        await adapter.distill_round(
            day="day01",
            round_id=f"r{idx}",
            round_index=idx,
            query=f"Q{idx}",
            answer=f"a{idx}",
            session_id="day01_uuid",
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    watermarks = [rec.body_of(e)["before_round_index"] for e in evolves]
    assert watermarks == [5, 10]
    assert adapter.records_evolution_events == 2


@pytest.mark.asyncio
async def test_g5_session_done_flush_does_not_count_or_evolve():
    """The session_done flush is NOT a graded round (codex HIGH #2): it must
    forward session_done=True to clear the server buffer, but MUST NOT increment
    the completed-round counter or fire an evolution (which would shift the
    {5,10,15} cadence by one per session)."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    res = await adapter.distill_round(
        day="day01",
        round_id="",
        round_index=4,
        query="",
        answer="",
        session_id="day01_uuid",
        session_done=True,
    )
    assert res["ok"] is True
    assert res["evolved"] is False
    posts = rec.posts_to(DISTILL_ROUND_ENDPOINT_PATH)
    assert len(posts) == 1
    assert rec.body_of(posts[0])["session_done"] is True
    # Counter NOT advanced; no evolve fired.
    assert adapter._round_tracker.completed("day01_uuid") == 0
    assert rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH) == []
    assert adapter.records_evolution_events == 0


@pytest.mark.asyncio
async def test_g5_flush_after_4_rounds_does_not_shift_next_session_cadence():
    """A 4-round session + flush leaves the count at 4 (not 5), so a flush never
    fabricates an evolution and never desyncs the {5,10,15} schedule."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    for idx in range(1, 5):  # 4 graded rounds
        await adapter.distill_round(
            day="day01",
            round_id=f"r{idx}",
            round_index=idx,
            query=f"Q{idx}",
            answer=f"a{idx}",
            session_id="day01_uuid",
        )
    await adapter.distill_round(
        day="day01",
        round_id="",
        round_index=4,
        query="",
        answer="",
        session_id="day01_uuid",
        session_done=True,
    )
    # 4 graded rounds → no evolve; the flush did not push the count to 5.
    assert rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH) == []
    assert adapter._round_tracker.completed("day01_uuid") == 4


# =========================================================================== #
# P1-B — resume/replay: evolve cadence keys off the deterministic global       #
# round_index, NOT an in-process counter, so a resumed run is correct          #
# =========================================================================== #


@pytest.mark.asyncio
async def test_p1b_resume_evolves_at_global_round_not_local_count():
    """codex P1-B. A run is interrupted after round 7 of a 13-round session; on
    resume the proxy is a FRESH process (tracker count == 0) and rounds 1-7 were
    distilled in the ORIGINAL run so they are NOT re-POSTed. The resumed process
    therefore only ever sees rounds 8..13 (round_index 8..13).

    The evolve cadence MUST fire at the SAME global rounds {5,10} as a fresh run:
    round 5/7 were already evolved in the original run (never re-fire on resume);
    round 10 fires now (it falls inside the resumed window). An in-process counter
    would instead fire at local-5 == global-12 (WRONG). This test would FAIL under
    the old `note_round`-counter cadence and PASSES with the round_index cadence.
    """
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    # Only the resumed (newly executed) rounds reach this fresh process.
    for round_index in range(8, 14):  # global rounds 8..13
        await adapter.distill_round(
            day="day05",
            round_id=f"r{round_index}",
            round_index=round_index,
            query=f"Q{round_index}",
            answer=f"a{round_index}",
            session_id="day05_uuid",
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    watermarks = [rec.body_of(e)["before_round_index"] for e in evolves]
    # Fires ONLY at global round 10 (5 was consumed by the original run; the
    # fresh-process internal count of 6 must NOT trigger a spurious evolve).
    assert watermarks == [10]
    assert adapter.records_evolution_events == 1


@pytest.mark.asyncio
async def test_p1b_watermark_is_monotonic_and_global_on_resume():
    """codex P1-B. Across a resumed window the watermark passed to
    evolve-from-records must be the deterministic global round_index (monotonic),
    never a local re-based index that could go backwards / forwards inconsistently.
    """
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    # Resume window covering two evolve boundaries: global rounds 6..15.
    for round_index in range(6, 16):
        await adapter.distill_round(
            day="day10",
            round_id=f"r{round_index}",
            round_index=round_index,
            query=f"Q{round_index}",
            answer=f"a{round_index}",
            session_id="day10_uuid",
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    watermarks = [rec.body_of(e)["before_round_index"] for e in evolves]
    assert watermarks == [10, 15]  # global boundaries, strictly increasing
    assert watermarks == sorted(watermarks)
    assert adapter.records_evolution_events == 2


@pytest.mark.asyncio
async def test_p1b_resume_skipped_round_5_does_not_double_evolve():
    """codex P1-B. If round 5 itself is resume-skipped (already distilled +
    evolved in the original run) it is never re-POSTed, so the resumed process
    must NOT re-fire round 5's evolution. Feeding rounds {1,2,3,4, 6,7,8,9,10}
    (5 omitted) evolves ONLY at 10, never twice at 5."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    for round_index in [1, 2, 3, 4, 6, 7, 8, 9, 10]:  # 5 was skipped on resume
        await adapter.distill_round(
            day="day01",
            round_id=f"r{round_index}",
            round_index=round_index,
            query=f"Q{round_index}",
            answer=f"a{round_index}",
            session_id="day01_uuid",
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    watermarks = [rec.body_of(e)["before_round_index"] for e in evolves]
    assert watermarks == [10]  # NOT [5, 10] — round 5 never reached this process
    assert adapter.records_evolution_events == 1


# =========================================================================== #
# P2-B — round_index is SESSION-GLOBAL across groups, not group-local          #
# =========================================================================== #


@pytest.mark.asyncio
async def test_p2b_multi_group_session_evolves_at_session_global_rounds():
    """codex P2-B. A session spanning two groups (group A: rounds 1-3, group B:
    rounds 4-7) must use SESSION-GLOBAL round indices 1..7 keyed by the SAME
    session_id. The every-5 evolve then fires once, at session-global round 5
    (which lives in group B), with watermark 5.

    If round_index were group-local, group B would restart at round_index=1 and
    the session would NEVER reach 5 in a single group → no evolve (or a watermark
    that resets to 1). This test asserts the session-global behaviour the bench
    now produces by threading round_index_base across groups.
    """
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)
    sid = "day07_uuid"
    # group A: session-global rounds 1,2,3
    for round_index in (1, 2, 3):
        await adapter.distill_round(
            day="day07",
            round_id=f"A-r{round_index}",
            round_index=round_index,
            query=f"Q{round_index}",
            answer=f"a{round_index}",
            session_id=sid,
        )
    # group B (SAME session): session-global rounds 4,5,6,7 (NOT restarting at 1)
    for round_index in (4, 5, 6, 7):
        await adapter.distill_round(
            day="day07",
            round_id=f"B-r{round_index}",
            round_index=round_index,
            query=f"Q{round_index}",
            answer=f"a{round_index}",
            session_id=sid,
        )
    evolves = rec.posts_to(EVOLVE_FROM_RECORDS_ENDPOINT_PATH)
    watermarks = [rec.body_of(e)["before_round_index"] for e in evolves]
    assert watermarks == [5]  # the session-global boundary, in group B
    # And the per-session completed-round telemetry counts ALL 7 rounds.
    assert adapter._round_tracker.completed(sid) == 7


def test_p2b_bench_round_index_is_session_global_offset_by_base():
    """codex P2-B (static). The bench's _run_group must offset its group-local
    enumerate by ``round_index_base`` (rounds seen in prior groups of the same
    session) and return the new running total, and _run_one_test must thread that
    total across groups + send the session_done flush ONCE after the last group.

    A group-local ``enumerate(rounds, start=1)`` feeding round_index directly would
    reset the watermark per group; this asserts the session-global wiring exists.
    """
    bench_infer = (
        REPO_ROOT
        / "evals"
        / "metaclaw"
        / "vendor"
        / "benchmark"
        / "src"
        / "infer"
        / "infer_cmd.py"
    )
    tree = ast.parse(bench_infer.read_text())

    run_group = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_group"
    )
    # 1) accepts round_index_base.
    arg_names = {a.arg for a in run_group.args.args} | {
        a.arg for a in run_group.args.kwonlyargs
    }
    assert "round_index_base" in arg_names, (
        "_run_group must take a round_index_base for session-global indexing"
    )
    # 2) round_index is round_index_base + local_index (not bare enumerate index).
    src = ast.get_source_segment(bench_infer.read_text(), run_group) or ""
    assert "round_index_base + local_index" in src, (
        "_run_group must add round_index_base to the group-local index"
    )
    # 3) it returns the running total so the caller can thread it.
    assert any(isinstance(n, ast.Return) for n in ast.walk(run_group))

    run_one_test = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_one_test"
    )
    one_src = ast.get_source_segment(bench_infer.read_text(), run_one_test) or ""
    # threads the running total into the next group...
    assert "round_index_base=session_round_total" in one_src
    assert "session_round_total = await _run_group(" in one_src
    # ...and sends session_done flush exactly once, AFTER the group loop, keyed by
    # the session-global total (the in-group per-group flush was removed).
    assert "True,  # session_done" in one_src
    group_src = ast.get_source_segment(bench_infer.read_text(), run_group) or ""
    assert "session_done" not in group_src or "NOT sent here" in group_src, (
        "_run_group must NOT send the per-group session_done flush anymore"
    )


# =========================================================================== #
# P2-A — the tokenizer-PRESENT evolve branch is gated by skill_evolution_mode  #
# =========================================================================== #


def test_p2a_both_evolve_branches_gated_by_raw_transcript_mode():
    """codex P2-A. The legacy every-N-turns evolve must be gated by
    ``skill_evolution_mode == 'raw_transcript'`` in BOTH the tokenizer-None
    (skills_only) branch AND the tokenizer-present (RL/teacher) branch — otherwise
    a ``mirix_records`` run with a real tokenizer loaded would double-evolve
    (legacy batch evolve + per-round distill).

    Static AST guard: every ``_want_evolution = ...`` assignment in
    ``api_server.py``'s OpenClaw chat handler includes a comparison against the
    literal ``"raw_transcript"``.
    """
    api_server = (
        REPO_ROOT / "evals" / "metaclaw" / "vendor" / "metaclaw" / "api_server.py"
    )
    text = api_server.read_text()
    tree = ast.parse(text)

    want_evo_assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_want_evolution" for t in n.targets
        )
    ]
    # There are exactly two such sites (one per tokenizer branch).
    assert len(want_evo_assigns) == 2, (
        f"expected 2 _want_evolution sites, found {len(want_evo_assigns)}"
    )
    for assign in want_evo_assigns:
        seg = ast.get_source_segment(text, assign) or ""
        assert '"raw_transcript"' in seg or "'raw_transcript'" in seg, (
            "a _want_evolution branch is NOT gated by raw_transcript mode "
            f"(double-evolve risk in mirix_records): {seg!r}"
        )


# =========================================================================== #
# default cadence is 5                                                         #
# =========================================================================== #


def test_default_cadence_is_five_rounds():
    assert DEFAULT_EVOLVE_EVERY_N_ROUNDS == 5
    rec = _Recorder()
    adapter = _make_adapter(rec)  # no explicit every_n
    assert adapter.evolve_every_n_rounds == 5


# =========================================================================== #
# evolve-from-records error handling keeps the eval alive                      #
# =========================================================================== #


@pytest.mark.asyncio
async def test_evolve_from_records_swallows_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        return httpx.Response(500, json={"detail": "db blip"})

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u",
        transport=httpx.MockTransport(handler),
        max_http_retries=0,  # one attempt, fail fast in test
        retry_sleep_s=0.0,
    )
    ok = await adapter.evolve_from_records(before_round_index=5)
    assert ok is False  # swallowed; event not counted
    assert adapter.records_evolution_events == 0


# =========================================================================== #
# OLD raw-transcript evolve path is UNCHANGED                                  #
# =========================================================================== #


@pytest.mark.asyncio
async def test_old_evolve_path_still_posts_to_v1_skills_evolve():
    """The legacy `evolve(samples, skills)` still POSTs to /v1/skills/evolve with
    {messages, user_id} — the new methods did not break the regression baseline."""
    from types import SimpleNamespace

    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if request.url.path == "/v1/skills/evolve":
            return httpx.Response(
                200, json={"changes": {"created": [], "edited": [], "deleted": []}}
            )
        return httpx.Response(404)

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-old",
        transport=httpx.MockTransport(handler),
        retry_sleep_s=0.0,
    )
    out = await adapter.evolve(
        [SimpleNamespace(prompt_text="p", response_text="r", reward=0.0)], {}
    )
    assert out == []  # MIRIX is sole writer
    evolve_posts = [r for r in rec.requests if r.url.path == "/v1/skills/evolve"]
    assert len(evolve_posts) == 1
    body = json.loads(evolve_posts[0].content.decode())
    assert set(body.keys()) == {"messages", "user_id"}
    assert body["user_id"] == "u-old"


# =========================================================================== #
# Arm selection (DESIGN §C5 P1-10 control arms)                               #
# =========================================================================== #


def test_arm_selection_taxonomy():
    from evals.metaclaw.runner import _resolve_arm

    native = _resolve_arm("metaclaw")
    assert native.skills_provider == "metaclaw"
    assert native.needs_mirix is False
    assert native.evolution_mode == "raw_transcript"
    assert native.skill_records is False

    # native is an explicit alias of metaclaw.
    assert _resolve_arm("native") == native

    old = _resolve_arm("mirix")
    assert old.skills_provider == "mirix"
    assert old.needs_mirix is True
    assert old.evolution_mode == "raw_transcript"  # the regression baseline
    assert old.skill_records is False

    new = _resolve_arm("mirix-records")
    assert new.skills_provider == "mirix"
    assert new.needs_mirix is True
    assert new.evolution_mode == "mirix_records"  # the C5 path
    assert new.skill_records is True

    floor = _resolve_arm("no-skills")
    assert floor.skills_enabled is False
    assert floor.auto_evolve is False
    assert floor.needs_mirix is False


def test_unknown_arm_rejected():
    from evals.metaclaw.runner import _resolve_arm

    with pytest.raises(ValueError):
        _resolve_arm("bogus-arm")


def test_old_arms_default_path_unchanged():
    """The pre-C5 arms resolve to raw_transcript + no skill-records, so the proxy
    YAML + bench invocation reproduce the old behaviour exactly (no new mode)."""
    from evals.metaclaw.runner import _resolve_arm

    for arm in ("metaclaw", "mirix", "native"):
        spec = _resolve_arm(arm)
        assert spec.evolution_mode == "raw_transcript"
        assert spec.skill_records is False


def test_old_arm_proxy_yaml_byte_identical_to_pre_c5(tmp_path: Path):
    """codex MED #2: an old-arm proxy.yaml (defaults) carries NO C5 keys — it is
    byte-identical to the pre-C5 YAML; the new keys appear ONLY in the new mode."""
    from evals.metaclaw.runner import _write_proxy_yaml

    bench_env = {
        "BENCHMARK_BASE_URL": "https://openrouter.ai/api/v1",
        "BENCHMARK_API_KEY": "sk-x",
        "BENCHMARK_MODEL": "openai/gpt-5.2",
    }
    old = tmp_path / "old.yaml"
    _write_proxy_yaml(old, Path("/skills"), 30000, bench_env, skill_top_k=6)
    text = old.read_text()
    # Exactly the pre-C5 skill block (no C5 keys), byte-for-byte.
    assert "evolution_mode" not in text
    assert "evolution_every_n_rounds" not in text
    assert "  enabled: true\n" in text
    assert "  auto_evolve: true\n" in text

    # The new-harness YAML DOES add the two C5 lines.
    new = tmp_path / "new.yaml"
    _write_proxy_yaml(
        new,
        Path("/skills"),
        30000,
        bench_env,
        skill_top_k=10,
        evolution_mode="mirix_records",
        evolution_every_n_rounds=5,
    )
    ntext = new.read_text()
    assert "  evolution_mode: mirix_records\n" in ntext
    assert "  evolution_every_n_rounds: 5\n" in ntext

    # no-skills arm flips enabled/auto_evolve off, still no C5 keys.
    floor = tmp_path / "floor.yaml"
    _write_proxy_yaml(
        floor,
        Path("/skills"),
        30000,
        bench_env,
        skills_enabled=False,
        auto_evolve=False,
    )
    ftext = floor.read_text()
    assert "  enabled: false\n" in ftext
    assert "  auto_evolve: false\n" in ftext
    assert "evolution_mode" not in ftext


# =========================================================================== #
# G4 — runner reads bench report.json; no local re-scoring                     #
# =========================================================================== #


def test_g4_parse_report_reads_summary_accuracy_only(tmp_path: Path):
    """The runner's _parse_report reads summary.accuracy verbatim — it does not
    recompute a score from raw round results."""
    from evals.metaclaw.runner import _parse_report

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "accuracy": 0.5621,  # partial-credit /346, authoritative
                    "tokens": {"agent": {"total_input": 100, "output": 50}},
                }
            }
        ),
        encoding="utf-8",
    )
    acc, tokens, summary = _parse_report(report)
    assert acc == pytest.approx(0.5621)
    assert tokens == 150
    assert summary["accuracy"] == pytest.approx(0.5621)


def test_g4_runner_has_no_local_scoring_logic():
    """Static guard: runner.py contains no inline scoring (no bbox extraction, no
    eval.command execution) — it only parses the bench's report.json."""
    src = (REPO_ROOT / "evals" / "metaclaw" / "runner.py").read_text()
    for banned in ("\\bbox", "expect_exit", "_compute_inline_score", "re.search"):
        assert banned not in src, f"runner.py unexpectedly contains {banned!r}"


# =========================================================================== #
# G2 — no MIRIX harness code path writes the bench workspace / agent-state     #
# =========================================================================== #


def test_g2_distill_path_makes_no_filesystem_writes(tmp_path, monkeypatch):
    """Calling distill_round / evolve_from_records performs ZERO filesystem
    writes: assert open(..., 'w'|'a'|'x') and os.* mutators are never invoked
    during the new-harness ingestion path."""
    rec = _Recorder()
    adapter = _make_adapter(rec, evolve_every_n_rounds=5)

    write_calls: list = []
    real_open = open

    def _tracking_open(file, mode="r", *a, **kw):
        if any(c in mode for c in ("w", "a", "x", "+")):
            write_calls.append((str(file), mode))
        return real_open(file, mode, *a, **kw)

    import builtins

    monkeypatch.setattr(builtins, "open", _tracking_open)
    for fn in ("remove", "unlink", "rename", "replace", "mkdir", "makedirs"):
        if hasattr(__import__("os"), fn):
            monkeypatch.setattr(
                f"os.{fn}",
                lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError(f"os.{fn} called on distill path (G2 violation)")
                ),
            )

    async def _drive():
        for idx in range(1, 6):
            await adapter.distill_round(
                day="day01",
                round_id=f"r{idx}",
                round_index=idx,
                query=f"Q{idx}",
                answer=f"a{idx}",
                session_id="day01_uuid",
            )

    asyncio.run(_drive())
    assert write_calls == [], f"new-harness path wrote files: {write_calls}"


def test_g2_bench_distill_trigger_only_does_http():
    """Static guard on the bench's _trigger_distill_round: its body opens no file
    and only issues an HTTP POST (urllib.request). A stray file write there could
    flip a later file_check round."""
    bench_infer = (
        REPO_ROOT
        / "evals"
        / "metaclaw"
        / "vendor"
        / "benchmark"
        / "src"
        / "infer"
        / "infer_cmd.py"
    )
    tree = ast.parse(bench_infer.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_trigger_distill_round"
    )
    # No `open(` call in the function body, and no Path.write_text / .touch.
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = []
    for c in calls:
        if isinstance(c.func, ast.Name):
            names.append(c.func.id)
        elif isinstance(c.func, ast.Attribute):
            names.append(c.func.attr)
    assert "open" not in names
    assert "write_text" not in names
    assert "touch" not in names
    # It DOES call urlopen (the only side effect is the HTTP POST).
    assert "urlopen" in names


# =========================================================================== #
# Point 6 — distiller _FAILURE_MARKERS ↔ REAL bench feedback templates         #
# =========================================================================== #


def _bench_prompts():
    """Import the upstream bench prompt templates (the source of truth)."""
    import sys

    bench = str(REPO_ROOT / "third_party" / "MetaClaw" / "benchmark")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    from src.infer.prompts import (  # noqa: E402
        CONTINUE_REMINDER,
        FILE_CHECK_INCORRECT_SUFFIX,
        FORMAT_ERROR,
        missed_option,
        wrong_option,
    )

    return {
        "CONTINUE_REMINDER": CONTINUE_REMINDER,
        "FILE_CHECK_INCORRECT_SUFFIX": FILE_CHECK_INCORRECT_SUFFIX,
        "FORMAT_ERROR": FORMAT_ERROR,
        "missed_option": missed_option,
        "wrong_option": wrong_option,
    }


def test_point6_failure_markers_match_real_multi_choice_incorrect():
    """A real multi_choice per-option incorrect feedback string → 'failure'."""
    from mirix.services.skill_session_distiller import derive_record_type

    p = _bench_prompts()
    mc_incorrect = "\n".join(
        [
            p["missed_option"]("E", "uses an explicit timezone offset"),
            p["wrong_option"]("B", "ambiguous local time"),
            p["CONTINUE_REMINDER"],
        ]
    )
    assert derive_record_type(mc_incorrect) == "failure"
    # The missing-\bbox format error is also a failure.
    assert derive_record_type(p["FORMAT_ERROR"]) == "failure"


def test_point6_failure_markers_match_real_file_check_incorrect():
    """A real file_check incorrect feedback (dataset string + the continue-reminder
    suffix the harness appends) → 'failure'."""
    from mirix.services.skill_session_distiller import derive_record_type

    p = _bench_prompts()
    # Exactly how _build_feedback_text composes a failed file_check feedback.
    fc_incorrect = (
        "The created_at timestamps are not in the correct format. Each task's "
        "created_at should look like: 2026-03-16T09:00:00+08:00"
        + "\n"
        + p["FILE_CHECK_INCORRECT_SUFFIX"]
    )
    assert derive_record_type(fc_incorrect) == "failure"


def test_point6_correct_feedback_is_success():
    """Real congratulatory feedback (no correction marker) → 'success'; empty
    feedback → None (no graded record)."""
    from mirix.services.skill_session_distiller import derive_record_type

    assert (
        derive_record_type(
            "The standup.json file is properly structured with all required fields."
        )
        == "success"
    )
    assert (
        derive_record_type(
            "Correct! A and E are both valid — they use the full datetime format."
        )
        == "success"
    )
    assert derive_record_type("") is None
    assert derive_record_type(None) is None


def test_point6_marker_set_covers_every_failure_template():
    """Every harness-authored incorrect-feedback shape contains at least one of
    the distiller's _FAILURE_MARKERS (so none silently scores as 'success')."""
    from mirix.services.skill_session_distiller import _FAILURE_MARKERS

    p = _bench_prompts()
    failure_strings = [
        p["missed_option"]("A", "x"),
        p["wrong_option"]("A", "x"),
        p["FORMAT_ERROR"],
        p["CONTINUE_REMINDER"],  # appended to every file_check incorrect
    ]
    for s in failure_strings:
        low = s.lower()
        assert any(m in low for m in _FAILURE_MARKERS), f"no marker matched {s!r}"


# =========================================================================== #
# Sanity: the adapter exposes the new surface (signature contract)            #
# =========================================================================== #


def test_adapter_exposes_records_methods():
    assert inspect.iscoroutinefunction(MirixEvolverAdapter.distill_round)
    assert inspect.iscoroutinefunction(MirixEvolverAdapter.evolve_from_records)
    sig = inspect.signature(MirixEvolverAdapter.distill_round)
    assert {"day", "round_id", "round_index", "query", "answer"} <= set(sig.parameters)


# =========================================================================== #
# FIX7 — silent-swallow defense + failure surfacing                            #
# =========================================================================== #
#
# These lock the four eval-harness hardening behaviours so a future refactor
# can't silently reintroduce the swallow:
#   1. The bench's _trigger_distill_round enforces ``ok is True`` (HTTP 200 +
#      {"ok": false} / bad-JSON / non-2xx all count as FAILURE) and records it.
#   2. The MirixEvolverAdapter surfaces evolve/distill failures LOUDLY (ERROR
#      + EVOLVE_FAILURE marker) and increments evolve_failures.
#   3. run_run returns the distill-failure count and writes distill_health.json.
#   4. runner._run_bench refuses (rc=2) a records arm with no proxy_port.


def _import_vendored_src():
    """Import the VENDORED bench ``src`` package (the one the runner actually
    runs), NOT the third_party/MetaClaw copy.

    Both ``evals/metaclaw/vendor/benchmark`` and ``third_party/MetaClaw/benchmark``
    expose a top-level ``src`` package. Whichever is imported first wins
    ``sys.modules['src']`` for the whole session — so a test that runs after one
    importing the third_party copy (e.g. via _bench_prompts) would otherwise get
    the OLD infer_cmd lacking the FIX7 functions. We defensively purge any cached
    ``src``* modules and reinsert the vendored root at the FRONT of sys.path so
    the reimport resolves to our copy regardless of test ordering.
    """
    import importlib
    import sys

    bench = str(REPO_ROOT / "evals" / "metaclaw" / "vendor" / "benchmark")
    cached = importlib.import_module("src.infer.infer_cmd") if "src.infer.infer_cmd" in sys.modules else None
    if cached is not None and getattr(cached, "__file__", "").startswith(bench):
        return cached  # already the vendored copy
    # Purge stale src* modules (the third_party copy) so the reimport is clean.
    for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    sys.path.insert(0, bench)
    return importlib.import_module("src.infer.infer_cmd")


@pytest.fixture
def vendored_src_sandbox():
    """Snapshot/restore ``sys.modules`` and ``sys.path`` around a test that
    imports the vendored ``src`` package (codex P1).

    ``_import_vendored_src`` permanently purges cached ``src*`` modules and
    prepends the vendored root — without isolation that would corrupt a later
    test which expects the third_party ``src`` (e.g. via ``_bench_prompts``).
    This fixture records the pre-test ``src*`` module table and ``sys.path``,
    yields, then restores both so the purge is confined to the test that opted
    into it. Tests using the vendored bench MUST request this fixture.
    """
    import sys

    saved_modules = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    saved_path = list(sys.path)
    try:
        yield
    finally:
        # Drop any src* modules imported during the test, then restore the
        # snapshot so the next test re-resolves `src` from a clean slate.
        for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _bench_infer_cmd():
    return _import_vendored_src()


class _FakeHTTPResponse:
    """Minimal context-manager stand-in for urllib.request.urlopen()."""

    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


@pytest.mark.parametrize(
    "body,expected_ok",
    [
        ('{"ok": true, "evolved": false}', True),
        ('{"ok": true, "evolved": true, "watermark": 5}', True),
        ('{"ok": false, "error": "distiller boom"}', False),  # HTTP 200 + ok:false
        ('{"evolved": true}', False),  # ok missing entirely -> NOT success
        ('{"ok": "true"}', False),  # ok is a string, not bool True
        ("not json at all", False),  # bad JSON body
    ],
)
def test_fix7_distill_ok_contract(monkeypatch, vendored_src_sandbox, body, expected_ok):
    """_trigger_distill_round treats a round as successful ONLY when the body
    carries ``ok is True``. Every other shape (ok:false, ok missing, ok as a
    non-True value, bad JSON) is a FAILURE and bumps distill_failures."""
    import urllib.request as _ureq

    infer_cmd = _bench_infer_cmd()
    infer_cmd.reset_distill_health()

    monkeypatch.setattr(
        _ureq,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(body),
    )
    ok = infer_cmd._trigger_distill_round(
        session_id="sess", day="day01", round_id="r1", round_index=1,
        query="Q", answer="A", proxy_port=12345, final=False,
    )
    assert ok is expected_ok
    health = infer_cmd.get_distill_health()
    assert health["distill_attempts"] == 1
    assert health["distill_failures"] == (0 if expected_ok else 1)


def test_fix7_distill_connection_error_counts_as_failure(monkeypatch, vendored_src_sandbox):
    """A connection error (urlopen raises) is a FAILURE, never swallowed."""
    import urllib.request as _ureq

    infer_cmd = _bench_infer_cmd()
    infer_cmd.reset_distill_health()

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(_ureq, "urlopen", _boom)
    ok = infer_cmd._trigger_distill_round(
        session_id="sess", day="day01", round_id="r1", round_index=1,
        query="Q", answer="A", proxy_port=12345,
    )
    assert ok is False
    assert infer_cmd.get_distill_health() == {
        "distill_attempts": 1,
        "distill_failures": 1,
    }


def test_fix7_distill_http_error_counts_as_failure(monkeypatch, vendored_src_sandbox):
    """A non-2xx (HTTPError) is a FAILURE."""
    import urllib.error as _uerr
    import urllib.request as _ureq

    infer_cmd = _bench_infer_cmd()
    infer_cmd.reset_distill_health()

    def _boom(*a, **k):
        raise _uerr.HTTPError("http://x", 503, "unavailable", {}, None)

    monkeypatch.setattr(_ureq, "urlopen", _boom)
    ok = infer_cmd._trigger_distill_round(
        session_id="sess", day="day01", round_id="r1", round_index=1,
        query="Q", answer="A", proxy_port=12345,
    )
    assert ok is False
    assert infer_cmd.get_distill_health()["distill_failures"] == 1


@pytest.mark.asyncio
async def test_fix7_evolve_from_records_failure_is_loud_and_counted(caplog):
    """On an HTTP error, evolve_from_records returns False BUT logs at ERROR
    with the EVOLVE_FAILURE marker and bumps evolve_failures (no silent swallow)."""
    import logging as _logging

    from evals.metaclaw.mirix_adapters.evolver_adapter import EVOLVE_FAILURE_MARKER

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        # evolve-from-records always 500s -> exhausts retries -> HTTPError
        return httpx.Response(500, json={"detail": "pool dead"})

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-eval",
        transport=httpx.MockTransport(_handler),
        max_http_retries=0,  # one attempt, fail fast
        retry_sleep_s=0.0,
    )
    with caplog.at_level(_logging.ERROR):
        result = await adapter.evolve_from_records(before_round_index=5)

    assert result is False
    assert adapter.evolve_failures == 1
    assert any(
        EVOLVE_FAILURE_MARKER in rec.getMessage() and rec.levelno >= _logging.ERROR
        for rec in caplog.records
    ), "evolve failure must log at ERROR with the EVOLVE_FAILURE marker"


@pytest.mark.asyncio
async def test_fix7_evolve_raw_transcript_failure_is_loud_and_counted(caplog):
    """The OLD-baseline raw-transcript evolve() path is also LOUD on failure
    (a swallowed failure here would inflate the new-minus-old delta wrongly)."""
    import logging as _logging
    from types import SimpleNamespace

    from evals.metaclaw.mirix_adapters.evolver_adapter import EVOLVE_FAILURE_MARKER

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        return httpx.Response(500, json={"detail": "boom"})

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-eval",
        transport=httpx.MockTransport(_handler),
        max_http_retries=0,
        retry_sleep_s=0.0,
    )
    samples = [SimpleNamespace(prompt_text="p", response_text="r", reward=0.0)]
    with caplog.at_level(_logging.ERROR):
        out = await adapter.evolve(samples, current_skills={})

    assert out == []  # contract preserved: MIRIX is sole writer
    assert adapter.evolve_failures == 1
    assert any(EVOLVE_FAILURE_MARKER in rec.getMessage() for rec in caplog.records)


def test_fix7_default_timeout_raised_to_1800():
    """Defense-in-depth: the per-request timeout ceiling is a generous 1800s."""
    from evals.metaclaw.mirix_adapters.evolver_adapter import DEFAULT_TIMEOUT_S

    assert DEFAULT_TIMEOUT_S == 1800.0


def test_fix7_payload_reports_failure_only_on_explicit_false():
    """codex P1 helper: ONLY an explicit success:false / ok:false is a failure;
    a missing key (older shapes carrying only `changes`) stays success-by-default
    so a healthy run is never false-FAILed."""
    from evals.metaclaw.mirix_adapters.evolver_adapter import _payload_reports_failure

    assert _payload_reports_failure({"success": False}) is True
    assert _payload_reports_failure({"ok": False}) is True
    assert _payload_reports_failure({"success": True}) is False
    assert _payload_reports_failure({"ok": True, "evolved": True}) is False
    # Missing both keys -> NOT a failure (back-compat with changes-only shape).
    assert _payload_reports_failure({"changes": {"created": ["s1"]}}) is False
    assert _payload_reports_failure({}) is False
    assert _payload_reports_failure(None) is False
    assert _payload_reports_failure("nonsense") is False


@pytest.mark.asyncio
async def test_fix7_evolve_from_records_2xx_body_failure_is_caught(caplog):
    """codex P1: a 2xx evolve-from-records body that reports success:false must
    NOT count as a successful evolution event — it logs EVOLVE_FAILURE, bumps
    evolve_failures, returns False, and does NOT increment records_evolution_events."""
    import logging as _logging

    from evals.metaclaw.mirix_adapters.evolver_adapter import EVOLVE_FAILURE_MARKER

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        # 200 OK but the body says it failed downstream.
        return httpx.Response(200, json={"success": False, "error": "curator boom"})

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-eval",
        transport=httpx.MockTransport(_handler),
    )
    with caplog.at_level(_logging.ERROR):
        result = await adapter.evolve_from_records(before_round_index=5)

    assert result is False
    assert adapter.evolve_failures == 1
    assert adapter.records_evolution_events == 0  # must NOT count as a success
    assert any(EVOLVE_FAILURE_MARKER in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_fix7_distill_round_2xx_body_failure_is_caught(caplog):
    """codex P1: a 2xx distill-round body reporting ok:false means the record was
    NOT written — return {"ok": False} so the bench ok-contract counts it."""
    import logging as _logging

    from evals.metaclaw.mirix_adapters.evolver_adapter import (
        DISTILL_ROUND_ENDPOINT_PATH,
        EVOLVE_FAILURE_MARKER,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if request.url.path == DISTILL_ROUND_ENDPOINT_PATH:
            return httpx.Response(200, json={"ok": False, "error": "distiller boom"})
        return httpx.Response(404, json={"detail": "nope"})

    adapter = MirixEvolverAdapter(
        base_url="http://mirix.test",
        user_id="u-eval",
        transport=httpx.MockTransport(_handler),
    )
    with caplog.at_level(_logging.ERROR):
        out = await adapter.distill_round(
            day="day01", round_id="r1", round_index=1, query="Q", answer="A",
            session_id="s",
        )

    assert out == {"ok": False, "evolved": False}
    assert adapter.evolve_failures == 1
    assert any(EVOLVE_FAILURE_MARKER in rec.getMessage() for rec in caplog.records)


def test_fix7_run_bench_refuses_records_arm_without_proxy_port():
    """runner._run_bench returns rc=2 (does NOT fall back to dead :30000) when a
    records arm is launched with no proxy_port — a misconfig must fail LOUD."""
    from evals.metaclaw import runner

    called = {"subprocess": False}

    def _fake_call(*a, **k):  # pragma: no cover — must NOT be reached
        called["subprocess"] = True
        return 0

    import subprocess as _sub

    orig = _sub.call
    _sub.call = _fake_call
    try:
        rc = runner._run_bench(
            tests_used=Path("/tmp/x.json"),
            out_dir=Path("/tmp/out"),
            env={},
            skill_records=True,
            proxy_port=None,
        )
    finally:
        _sub.call = orig

    assert rc == 2
    assert called["subprocess"] is False, "must not launch the bench subprocess"


def test_fix7_run_run_writes_health_sidecar_and_returns_failures(
    tmp_path, monkeypatch, vendored_src_sandbox
):
    """run_run writes distill_health.json and returns the distill-failure count
    so the runner / sanity gate can detect a degenerate records pipeline."""
    import importlib

    # Ensure the VENDORED src is the cached one (purges any third_party copy),
    # then import run_cmd from that same package so its module-level
    # ``from src.infer.infer_cmd import get_distill_health`` binds to our copy.
    infer_cmd = _import_vendored_src()
    run_cmd = importlib.import_module("src.run.run_cmd")

    # Build a minimal all_tests.json the run pipeline can discover.
    ws = tmp_path / "ws"
    ws.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    evald = tmp_path / "evaldir"
    evald.mkdir()
    tests_file = tmp_path / "all_tests.json"
    tests_file.write_text(
        json.dumps(
            {
                "name": "fix7set",
                "workspace_src": str(ws),
                "openclaw_state_dir": str(state),
                "eval_dir": str(evald),
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    # Stub the heavy stages: infer becomes a no-op that just records 2 failures;
    # scoring/report are no-ops. We only exercise the sidecar + return contract.
    infer_cmd.reset_distill_health()

    async def _fake_infer(**kw):
        infer_cmd._record_distill_result(True)
        infer_cmd._record_distill_result(False)
        infer_cmd._record_distill_result(False)

    monkeypatch.setattr(run_cmd, "_run_one_all_tests", _fake_infer)
    monkeypatch.setattr(run_cmd, "run_scoring", lambda **kw: None)
    monkeypatch.setattr(run_cmd, "run_report", lambda **kw: None)

    failures = run_cmd.run_run(
        input_arg=str(tests_file),
        output_arg=str(out_dir),
        skill_records=True,
    )

    assert failures == 2
    health_files = list(out_dir.rglob("distill_health.json"))
    assert health_files, "distill_health.json sidecar must be written"
    data = json.loads(health_files[0].read_text())
    assert data["distill_failures"] == 2
    assert data["distill_attempts"] == 3
    assert data["skill_records"] is True
