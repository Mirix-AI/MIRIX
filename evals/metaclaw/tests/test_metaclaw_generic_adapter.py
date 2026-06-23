"""Unit tests for :mod:`evals.metaclaw.mirix_adapters.generic_adapter`.

Uses :class:`httpx.MockTransport` so every assertion is at the HTTP wire level —
no live MIRIX server, no API key, no patching of private methods.

Coverage (per the generic-arm spec):

  1.  Per-turn ``session_id`` tagging — ``distill_round(day, round_index)`` POSTs
      ``/memory/add_sync`` with ``session_id == f"{day}-r{round_index}"``.
  2.  Every-5 barrier cadence — 11 graded turns fire ``/memory/auto_dream`` at
      round_index 5 and 10 only.
  3.  Remainder flush — after a barrier at 5 then 2 more turns, ``session_done``
      fires ``auto_dream(last_n_sessions=2)``.
  4.  Leakage — no :data:`DISTILL_FORBIDDEN_KEYS` appears anywhere in any outgoing
      body; the add_sync messages are exactly the two role/content turns.
  5.  Retrieval delegation — the generic adapter performs NO retrieval HTTP.
  6.  Health gate — all-zero barriers ⇒ ``degenerate_run is True`` + an
      ``EVOLVE_FAILURE`` marker logged.
  7.  meta_agent_id resolution — ``GET /agents`` row used + cached; missing ⇒ loud
      failure (turn not ingested).
  8.  ``session_done`` does not shift cadence — a flush after 4 turns must not
      consume the next session's 5th-turn barrier.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List

import httpx
import pytest

from evals.metaclaw.mirix_adapters.evolver_adapter import (
    DISTILL_FORBIDDEN_KEYS,
    EVOLVE_FAILURE_MARKER,
)
from evals.metaclaw.mirix_adapters.generic_adapter import (
    ADD_SYNC_ENDPOINT_PATH,
    AGENTS_ENDPOINT_PATH,
    AUTO_DREAM_ENDPOINT_PATH,
    HEALTH_ENDPOINT_PATH,
    IN_BAND_TRIGGER_DISABLED_MIN,
    MirixGenericMemoryAdapter,
    build_add_sync_payload,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_META_AGENT_ID = "agent-meta-0001"

_OPENAPI_OK = {
    "openapi": "3.1.0",
    "paths": {
        ADD_SYNC_ENDPOINT_PATH: {"post": {}},
        AUTO_DREAM_ENDPOINT_PATH: {"post": {}},
    },
}

_OPENAPI_MISSING = {
    "openapi": "3.1.0",
    "paths": {ADD_SYNC_ENDPOINT_PATH: {"post": {}}},  # auto_dream absent
}

_AGENTS_ROWS = [
    {"id": "agent-other", "agent_type": "episodic_memory_agent"},
    {"id": _META_AGENT_ID, "agent_type": "meta_memory_agent"},
]


def _auto_dream_body(*, experiences: int, skills_changed: int) -> dict:
    return {
        "start_date": None,
        "end_date": None,
        "processed": {"procedural": {"total": experiences}},
        "last_dream_at": "2026-06-23T00:00:00",
        "dry_run": False,
        "skills_changed": skills_changed,
        "changes": {"created": [], "edited": [], "deleted": []},
        "message": "ok",
    }


def _make_handler(
    captured: List[httpx.Request],
    *,
    openapi=_OPENAPI_OK,
    agents=_AGENTS_ROWS,
    experiences: int = 2,
    skills_changed: int = 1,
    trigger_threshold: int = IN_BAND_TRIGGER_DISABLED_MIN,
    retain_n: int = 100,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that ack's preflight (openapi + health) +
    agents + the two production endpoints, recording every request.

    ``trigger_threshold`` is what ``/health`` reports for
    ``skill_trigger_session_threshold`` — defaults to the disabled sentinel so the
    in-band-trigger preflight passes; set it low (e.g. 5) to exercise the refusal.
    ``retain_n`` is the reported ``message_retain_last_n_sessions`` — defaults
    above the cadence so the retention-slack warning stays quiet; set it <= cadence
    to exercise the warning.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if path == "/openapi.json":
            return httpx.Response(200, json=openapi)
        if path == HEALTH_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "service": "mirix-api",
                    "skill_trigger_session_threshold": trigger_threshold,
                    "message_retain_last_n_sessions": retain_n,
                },
            )
        if path == AGENTS_ENDPOINT_PATH:
            return httpx.Response(200, json=agents)
        if path == ADD_SYNC_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={"success": True, "status": "processed", "message_count": 2},
            )
        if path == AUTO_DREAM_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json=_auto_dream_body(
                    experiences=experiences, skills_changed=skills_changed
                ),
            )
        return httpx.Response(404, text=f"unexpected path {path}")

    return handler


def _make_adapter(handler, **kw) -> MirixGenericMemoryAdapter:
    transport = httpx.MockTransport(handler)
    return MirixGenericMemoryAdapter(
        base_url=kw.pop("base_url", "http://mock.test"),
        user_id=kw.pop("user_id", "u-test"),
        transport=transport,
        retry_sleep_s=kw.pop("retry_sleep_s", 0.0),
        **kw,
    )


def _bodies_for(captured: List[httpx.Request], path: str) -> List[dict]:
    out = []
    for r in captured:
        if r.url.path == path and r.content:
            out.append(json.loads(r.content.decode()))
    return out


# --------------------------------------------------------------------------- #
# 1. Per-turn session_id tagging                                              #
# --------------------------------------------------------------------------- #


def test_per_turn_session_id_tagging():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        res = asyncio.run(
            adapter.distill_round(
                day="day01",
                round_id="r-abc",
                round_index=3,
                query="what is 2+2?",
                answer="4",
            )
        )
    finally:
        adapter.close()

    assert res["ok"] is True
    add_bodies = _bodies_for(captured, ADD_SYNC_ENDPOINT_PATH)
    assert len(add_bodies) == 1
    assert add_bodies[0]["session_id"] == "day01-r3"
    assert add_bodies[0]["meta_agent_id"] == _META_AGENT_ID
    assert add_bodies[0]["user_id"] == "u-test"


def test_incoming_session_id_is_ignored_for_add_sync():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        asyncio.run(
            adapter.distill_round(
                day="day02",
                round_id="r-x",
                round_index=7,
                query="q",
                answer="a",
                session_id="WHOLE-DAY-SESSION",  # must NOT be used in add_sync body
            )
        )
    finally:
        adapter.close()
    add_bodies = _bodies_for(captured, ADD_SYNC_ENDPOINT_PATH)
    assert add_bodies[0]["session_id"] == "day02-r7"


# --------------------------------------------------------------------------- #
# 2. Every-5 barrier cadence                                                  #
# --------------------------------------------------------------------------- #


def test_every_5_barrier_cadence():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 12):  # 11 graded turns, round_index 1..11
            asyncio.run(
                adapter.distill_round(
                    day="day01",
                    round_id=f"r{ri}",
                    round_index=ri,
                    query=f"q{ri}",
                    answer=f"a{ri}",
                )
            )
    finally:
        adapter.close()

    dream_bodies = _bodies_for(captured, AUTO_DREAM_ENDPOINT_PATH)
    # Fires at round_index 5 and 10 only.
    assert len(dream_bodies) == 2
    assert adapter.barriers_fired == 2
    for b in dream_bodies:
        assert b["mode"] == "procedural"
        assert b["last_n_sessions"] == 5


def test_auto_dream_user_id_is_query_param():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 6):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()
    dream_reqs = [r for r in captured if r.url.path == AUTO_DREAM_ENDPOINT_PATH]
    assert len(dream_reqs) == 1
    assert dream_reqs[0].url.params.get("user_id") == "u-test"
    # user_id must NOT be in the body.
    body = json.loads(dream_reqs[0].content.decode())
    assert "user_id" not in body


# --------------------------------------------------------------------------- #
# 3. Remainder flush on session_done                                          #
# --------------------------------------------------------------------------- #


def test_remainder_flush_after_barrier():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 6):  # 5 turns -> barrier at 5, remainder reset to 0
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        for ri in range(6, 8):  # 2 more turns, no barrier
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        res = asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=999, query="", answer="",
                session_done=True,
            )
        )
    finally:
        adapter.close()

    assert res["ok"] is True
    assert res["flush"] is True
    assert res["remainder"] == 2
    dream_bodies = _bodies_for(captured, AUTO_DREAM_ENDPOINT_PATH)
    # barrier at 5 (last_n=5) + flush (last_n=2).
    assert [b["last_n_sessions"] for b in dream_bodies] == [5, 2]
    # session_done sent NO add_sync (no graded query/answer forwarded).
    add_bodies = _bodies_for(captured, ADD_SYNC_ENDPOINT_PATH)
    assert len(add_bodies) == 7  # exactly the 7 graded turns


def test_flush_with_zero_remainder_is_noop():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 6):  # barrier at 5 -> remainder back to 0
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        res = asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=6, query="", answer="",
                session_done=True,
            )
        )
    finally:
        adapter.close()
    assert res["remainder"] == 0
    dream_bodies = _bodies_for(captured, AUTO_DREAM_ENDPOINT_PATH)
    assert [b["last_n_sessions"] for b in dream_bodies] == [5]  # only the barrier


# --------------------------------------------------------------------------- #
# 4. Leakage discipline                                                       #
# --------------------------------------------------------------------------- #


def test_no_oracle_keys_in_any_outgoing_body():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 6):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri,
                    query="[Previous Feedback] correct\nNow answer:", answer="42",
                )
            )
        asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=6, query="", answer="",
                session_done=True,
            )
        )
    finally:
        adapter.close()

    for r in captured:
        if not r.content:
            continue
        raw = r.content.decode().lower()
        for forbidden in DISTILL_FORBIDDEN_KEYS:
            # The forbidden tuple includes substrings like "answer"/"score" that
            # could appear inside agent-visible content; assert they are not
            # present as a JSON KEY. Parse + walk the structure for dict keys.
            body = json.loads(r.content.decode())
            assert not _has_key(body, forbidden), (
                f"forbidden key {forbidden!r} leaked into {r.url.path}: {body}"
            )


def test_add_sync_messages_are_exactly_two_role_turns():
    body = build_add_sync_payload(
        meta_agent_id="m", user_id="u", query="Q", answer="A", session_id="d-r1"
    )
    assert body["messages"] == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    # routing fields only — no oracle/grade fields, no filter_tags.
    assert set(body.keys()) == {
        "meta_agent_id", "user_id", "session_id", "chaining", "use_cache", "messages"
    }
    assert not _has_key(body, "filter_tags")
    for forbidden in DISTILL_FORBIDDEN_KEYS:
        assert not _has_key(body, forbidden)


def _has_key(obj, key) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(v, key) for v in obj)
    return False


# --------------------------------------------------------------------------- #
# 5. Retrieval delegation (the generic adapter does NO retrieval HTTP)        #
# --------------------------------------------------------------------------- #


def test_adapter_does_no_retrieval():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 6):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()
    # Only preflight + agents + add_sync + auto_dream paths — never a retrieval
    # endpoint like /v1/skills (that is MirixSkillsAdapter's job, wired by
    # METACLAW_SKILLS_PROVIDER=mirix, identical to the records arm).
    paths = {r.url.path for r in captured}
    assert paths <= {
        "/openapi.json", HEALTH_ENDPOINT_PATH, AGENTS_ENDPOINT_PATH,
        ADD_SYNC_ENDPOINT_PATH, AUTO_DREAM_ENDPOINT_PATH,
    }
    assert "/v1/skills" not in paths
    # The adapter has no retrieve() method — retrieval is not its surface.
    assert not hasattr(adapter, "retrieve")


# --------------------------------------------------------------------------- #
# 6. Health gate                                                              #
# --------------------------------------------------------------------------- #


def test_degenerate_run_marker_on_all_zero(caplog):
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, experiences=0, skills_changed=0)
    adapter = _make_adapter(handler)
    try:
        with caplog.at_level(logging.ERROR):
            for ri in range(1, 11):  # barriers at 5 and 10, both no-op
                asyncio.run(
                    adapter.distill_round(
                        day="d", round_id=f"r{ri}", round_index=ri,
                        query="q", answer="a",
                    )
                )
            assert adapter.barriers_fired == 2
            assert adapter.total_experiences == 0
            assert adapter.total_skills_changed == 0
            assert adapter.degenerate_run is True
    finally:
        adapter.close()
    assert any(EVOLVE_FAILURE_MARKER in rec.message for rec in caplog.records)


def test_healthy_run_is_not_degenerate():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured, experiences=3, skills_changed=2))
    try:
        for ri in range(1, 6):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        assert adapter.barriers_fired == 1
        assert adapter.total_experiences == 3
        assert adapter.total_skills_changed == 2
        assert adapter.degenerate_run is False
    finally:
        adapter.close()


def test_no_barriers_is_not_degenerate():
    # A run with <5 turns never fires a barrier; degenerate_run must be False
    # (no false-positive on a short run).
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 4):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        assert adapter.barriers_fired == 0
        assert adapter.degenerate_run is False
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# 7. meta_agent_id resolution                                                 #
# --------------------------------------------------------------------------- #


def test_meta_agent_id_resolved_and_cached():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 4):
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()
    assert adapter._meta_agent_id == _META_AGENT_ID
    # GET /agents happened exactly once (cached after first resolve).
    agents_reqs = [r for r in captured if r.url.path == AGENTS_ENDPOINT_PATH]
    assert len(agents_reqs) == 1


def test_missing_meta_agent_fails_loud(caplog):
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, agents=[{"id": "x", "agent_type": "episodic_memory_agent"}])
    adapter = _make_adapter(handler)
    try:
        with caplog.at_level(logging.ERROR):
            res = asyncio.run(
                adapter.distill_round(
                    day="d", round_id="r1", round_index=1, query="q", answer="a"
                )
            )
    finally:
        adapter.close()
    assert res["ok"] is False
    assert adapter.evolve_failures == 1
    assert any(EVOLVE_FAILURE_MARKER in rec.message for rec in caplog.records)
    # No add_sync was sent (turn not ingested).
    assert not _bodies_for(captured, ADD_SYNC_ENDPOINT_PATH)


# --------------------------------------------------------------------------- #
# 8. A flush resets the un-evolved window (no double-count after a flush)      #
# --------------------------------------------------------------------------- #


def test_flush_resets_window_no_double_count():
    # After a session_done flush evolves the remainder, the NEXT periodic barrier
    # must cover ONLY genuinely-new turns — it must NOT re-distill the already-
    # flushed sessions (codex P1/P2 2026-06-23). The cadence keys off the count of
    # un-evolved turns, so a flush at 4 turns resets the window, and the next
    # barrier fires on the 5th NEW turn (global round_index 9) with last_n=5 — it
    # does NOT fire at round_index 5 (only 1 new turn had accrued by then).
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        for ri in range(1, 5):  # round_index 1..4, no periodic barrier yet
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        flush = asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=4, query="", answer="",
                session_done=True,
            )
        )
        # The round-5 turn must NOT trigger a barrier (only 1 new turn since flush).
        res5 = asyncio.run(
            adapter.distill_round(
                day="d", round_id="r5", round_index=5, query="q", answer="a"
            )
        )
        # 4 more NEW turns (round_index 6..9) — the barrier fires on the 5th new
        # turn (round_index 9), covering exactly the 5 post-flush sessions.
        last = None
        for ri in range(6, 10):
            last = asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()

    assert flush["remainder"] == 4
    assert res5["evolved"] is False  # no early barrier on the 1st post-flush turn
    dream_bodies = _bodies_for(captured, AUTO_DREAM_ENDPOINT_PATH)
    # flush(last_n=4) then exactly one periodic barrier(last_n=5) on the 5th NEW
    # turn — NOT last_n>5 spanning the already-flushed sessions.
    assert [b["last_n_sessions"] for b in dream_bodies] == [4, 5]
    assert last["evolved"] is True
    assert last.get("watermark") == 9


# --------------------------------------------------------------------------- #
# 9. Barrier-failure handling (codex P1 — not swallowed as ok:True)           #
# --------------------------------------------------------------------------- #


def _failing_dream_handler(captured: List[httpx.Request]) -> Callable:
    """add_sync ok, but every auto_dream returns HTTP 500 (after retries)."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if path == HEALTH_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "skill_trigger_session_threshold": IN_BAND_TRIGGER_DISABLED_MIN,
                },
            )
        if path == AGENTS_ENDPOINT_PATH:
            return httpx.Response(200, json=_AGENTS_ROWS)
        if path == ADD_SYNC_ENDPOINT_PATH:
            return httpx.Response(200, json={"success": True, "status": "processed"})
        if path == AUTO_DREAM_ENDPOINT_PATH:
            return httpx.Response(500, text="boom")
        return httpx.Response(404, text=f"unexpected path {path}")

    return handler


def test_periodic_barrier_failure_reports_not_ok_and_retains_remainder():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(
        _failing_dream_handler(captured), max_http_retries=0
    )
    try:
        last = None
        for ri in range(1, 6):  # round 5 fires a barrier, which 500s
            last = asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        # close() also evaluates degenerate_run, but barriers_fired with a failed
        # 500 barrier: total_experiences==0 + skills_changed==0 AND barriers_fired>0
        # would trip degenerate — that is the intended health signal here.
        adapter.close()

    # The barrier failed → ok must be False (bench counts a failure).
    assert last["ok"] is False
    assert last["evolved"] is False
    assert adapter.evolve_failures >= 1
    # Remainder NOT reset → a later flush would retry the un-evolved window.
    assert adapter._turns_since_last_barrier == 5


def test_flush_barrier_failure_reports_not_ok_and_retains_remainder():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(
        _failing_dream_handler(captured), max_http_retries=0
    )
    try:
        for ri in range(1, 4):  # 3 turns, no periodic barrier
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        res = asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=3, query="", answer="",
                session_done=True,
            )
        )
    finally:
        adapter.close()
    assert res["ok"] is False
    assert res["flush"] is True
    assert res["remainder"] == 3
    assert adapter._turns_since_last_barrier == 3  # not reset on failure


def test_failed_barrier_retries_before_retention_prunes():
    # Under retention=5 a failed periodic barrier must RETRY on the next turn (the
    # counter is not reset on failure and the cadence keys off the un-evolved
    # count), widening its window so the un-evolved sessions are re-attempted
    # before their messages are pruned — never silently skipped (codex P1/P2
    # 2026-06-23).
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_failing_dream_handler(captured), max_http_retries=0)
    try:
        for ri in range(1, 7):  # round 5 fires a barrier (500); round 6 retries it
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()

    dream_reqs = [r for r in captured if r.url.path == AUTO_DREAM_ENDPOINT_PATH]
    # Barrier fired at round 5 (counter 5) and RETRIED at round 6 (counter 6 — the
    # window widened, the un-evolved turns were not skipped).
    assert len(dream_reqs) == 2
    assert [
        json.loads(r.content.decode())["last_n_sessions"] for r in dream_reqs
    ] == [5, 6]
    assert adapter._turns_since_last_barrier == 6  # never reset (all failed)
    assert adapter.evolve_failures >= 2


def test_degenerate_marker_emitted_on_session_done(caplog):
    # The vendored proxy NEVER calls close(); it DOES call distill_round with
    # session_done=True at each session boundary. The degenerate-run marker must
    # therefore be emitted on the session_done flush (the active production hook),
    # not only at teardown (codex P1 2026-06-23 — close()-only gate was dead code).
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured, experiences=0, skills_changed=0))
    with caplog.at_level(logging.ERROR):
        for ri in range(1, 6):  # one all-zero barrier at round 5
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
        # session_done flush — must emit the marker WITHOUT close() being called.
        asyncio.run(
            adapter.distill_round(
                day="d", round_id="done", round_index=6, query="", answer="",
                session_done=True,
            )
        )
    assert any(EVOLVE_FAILURE_MARKER in rec.message for rec in caplog.records)
    # The handler under test never had close() invoked, proving session_done is
    # the active path. Tidy up the socket now that the assertion holds.
    adapter.close()


def test_degenerate_marker_emitted_on_close(caplog):
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured, experiences=0, skills_changed=0))
    for ri in range(1, 6):  # one all-zero barrier at round 5
        asyncio.run(
            adapter.distill_round(
                day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
            )
        )
    # Do NOT read degenerate_run explicitly — close() must emit the marker so the
    # proxy.log grep gate trips without any reader (codex P1).
    with caplog.at_level(logging.ERROR):
        adapter.close()
    assert any(EVOLVE_FAILURE_MARKER in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Preflight                                                                   #
# --------------------------------------------------------------------------- #


def test_preflight_raises_when_endpoint_missing():
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, openapi=_OPENAPI_MISSING)
    with pytest.raises(RuntimeError, match="auto_dream"):
        MirixGenericMemoryAdapter(
            base_url="http://mock.test",
            user_id="u",
            transport=httpx.MockTransport(handler),
        )


def test_preflight_raises_when_in_band_trigger_enabled():
    # /health reports an ENABLED in-band trigger (threshold below the disabled
    # sentinel) -> the generic arm must REFUSE to construct, since it would
    # double-evolve alongside the explicit barrier (codex P1 2026-06-23).
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, trigger_threshold=5)
    with pytest.raises(RuntimeError, match="SKILL_TRIGGER_SESSION_THRESHOLD"):
        MirixGenericMemoryAdapter(
            base_url="http://mock.test",
            user_id="u",
            transport=httpx.MockTransport(handler),
        )


def test_preflight_warns_when_retention_not_above_cadence(caplog):
    # retention (5) <= cadence (5 default): healthy runs are lossless, but a failed
    # barrier can't retry losslessly — the preflight must WARN (not raise) so the
    # operator can raise MESSAGE_RETAIN_LAST_N_SESSIONS (codex P2 2026-06-23).
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, retain_n=5)
    with caplog.at_level(logging.WARNING):
        adapter = MirixGenericMemoryAdapter(
            base_url="http://mock.test",
            user_id="u",
            transport=httpx.MockTransport(handler),
        )
    adapter.close()
    assert any(
        "message_retain_last_n_sessions" in rec.message
        and "FAILED barrier" in rec.message
        for rec in caplog.records
    )


def test_preflight_no_retention_warning_when_slack_present(caplog):
    # retention (100) > cadence (5): no retention warning.
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, retain_n=100)
    with caplog.at_level(logging.WARNING):
        adapter = MirixGenericMemoryAdapter(
            base_url="http://mock.test",
            user_id="u",
            transport=httpx.MockTransport(handler),
        )
    adapter.close()
    assert not any(
        "message_retain_last_n_sessions" in rec.message for rec in caplog.records
    )


def test_preflight_accepts_disabled_trigger_at_sentinel():
    # Exactly at the disabled sentinel must be accepted (boundary).
    captured: List[httpx.Request] = []
    handler = _make_handler(captured, trigger_threshold=IN_BAND_TRIGGER_DISABLED_MIN)
    adapter = MirixGenericMemoryAdapter(
        base_url="http://mock.test",
        user_id="u",
        transport=httpx.MockTransport(handler),
    )
    adapter.close()


def test_evolve_every_n_rounds_alias_is_honored():
    # Drop-in compat: passing the records adapter's kwarg name must set the
    # cadence (not be swallowed into **_paper_kwargs) (codex P2 2026-06-23).
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured), evolve_every_n_rounds=3)
    try:
        assert adapter.evolve_every_n_turns == 3
        for ri in range(1, 4):  # 3 turns -> one barrier at the 3rd
            asyncio.run(
                adapter.distill_round(
                    day="d", round_id=f"r{ri}", round_index=ri, query="q", answer="a"
                )
            )
    finally:
        adapter.close()
    assert adapter.barriers_fired == 1
    dream_bodies = _bodies_for(captured, AUTO_DREAM_ENDPOINT_PATH)
    assert [b["last_n_sessions"] for b in dream_bodies] == [3]


# --------------------------------------------------------------------------- #
# Paper-compat surface                                                        #
# --------------------------------------------------------------------------- #


def test_paper_compat_surface():
    captured: List[httpx.Request] = []
    adapter = _make_adapter(_make_handler(captured))
    try:
        assert adapter.update_history == []
        assert adapter.history_path is None
        assert adapter.should_evolve([], threshold=0.0) is True
        assert asyncio.run(adapter.evolve([], {})) == []
        # evolve() is a pure no-op: no HTTP beyond preflight (openapi + health).
        assert {r.url.path for r in captured} <= {"/openapi.json", HEALTH_ENDPOINT_PATH}
    finally:
        adapter.close()
