"""Tests for the C1 per-round experience distiller.

Four layers, mirroring the C2 record-store suite:

1. Leakage filter (DB-free, LLM-free) — the §3 guard #1 hard assertion. A turn's
   `prompt_text` carries a `[Previous Feedback]` head + the round question + a
   long body; the sanitizer PRESERVES the feedback head (never tail-truncates it
   away) and surfaces ONLY {question, answer, prev_feedback}. Oracle-derived keys
   (`inline_score`, `reward`, `eval`, `feedback.options`) fed alongside MUST NOT
   appear anywhere in the sanitized output.
2. record_type derivation (DB-free) — correct feedback -> success, the various
   MetaClaw "incorrect" feedback shapes -> failure.
3. JSON parse robustness (DB-free) — fenced ```json blocks, surrounding prose,
   bare objects.
4. Distiller lag + persistence (DB-backed, LLM mocked) — feeding round t buffers
   it (no record yet); feeding round t+1 produces exactly one record for round t
   whose record_type comes from t+1's feedback prefix; flush_session on the last
   buffered round produces NO record. State is per-agent.

The LLM client is mocked everywhere (no network). DB-backed tests use the same
hermetic throwaway-SQLite sessionmaker pattern as test_skill_evolution_record.py.
"""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio

from mirix.services.skill_session_distiller import (
    SkillSessionDistiller,
    derive_record_type,
    parse_distiller_json,
    sanitize_turn,
)


# ============================ Leakage filter (§3 guard #1) ===================


# The exact MetaClaw injection format: "[Previous Feedback] <text>\n\n<question>"
# (benchmark/src/infer/prompts.py::with_feedback). The buffered prompt_text
# (api_server.py:1336) is the role-prefixed flattening of the message list, so the
# feedback + question live inside a "user: ..." line.
PREV_FEEDBACK = (
    "The time format in log_summary.json does not conform to the required "
    "standard. All time fields must include date, time, and timezone offset.\n"
    "Keep this in mind as you continue with the next task."
)
QUESTION = (
    "When processing server logs, various raw time formats need to be "
    "standardized. Which of the following conversions are correct?\n"
    "A. Apache format ...\nB. Unix timestamp ...\n"
    "Answer using \\bbox{X,Y,...} format."
)
# A long body simulating accumulated openclaw history AFTER the feedback head.
LONG_BODY = "\n".join(f"assistant: prior turn line {i} " + "x" * 60 for i in range(400))


def _prompt_text(prev_feedback: str = PREV_FEEDBACK, question: str = QUESTION) -> str:
    """Build a realistic flattened prompt_text whose head is the feedback+question."""
    head = f"user: [Previous Feedback] {prev_feedback}\n\n{question}"
    return f"{head}\n{LONG_BODY}"


class TestLeakageFilter:
    def test_preserves_feedback_head(self):
        # The legitimate pass/fail source is the [Previous Feedback] head; the
        # sanitizer must keep it even though the body is huge.
        turn = {"prompt_text": _prompt_text(), "response_text": "\\bbox{A,B}"}
        out = sanitize_turn(turn)
        assert "The time format in log_summary.json" in out["prev_feedback"]
        assert "Keep this in mind as you continue" in out["prev_feedback"]

    def test_extracts_question_and_answer(self):
        turn = {"prompt_text": _prompt_text(), "response_text": "\\bbox{A,B}"}
        out = sanitize_turn(turn)
        assert "Which of the following conversions are correct?" in out["question"]
        # The feedback marker must NOT bleed into the question field.
        assert "[Previous Feedback]" not in out["question"]
        assert out["answer"] == "\\bbox{A,B}"

    def test_truncation_bounds_the_middle_not_the_head(self):
        # A pathologically long prompt must still keep its feedback head. We force
        # truncation by exceeding the cap; the head (feedback) survives, the tail
        # may be dropped, and an elision marker is present.
        turn = {
            "prompt_text": _prompt_text() + "\n" + ("z" * 50000),
            "response_text": "ans",
        }
        out = sanitize_turn(turn, max_chars=2000)
        assert "The time format in log_summary.json" in out["prev_feedback"]
        # The question head is preserved too (it sits right after the feedback).
        assert "Which of the following conversions" in out["question"]

    def test_no_feedback_head_yields_empty_prev_feedback(self):
        # Round 1 has no predecessor -> no [Previous Feedback] prefix.
        turn = {"prompt_text": f"user: {QUESTION}\n{LONG_BODY}", "response_text": "a"}
        out = sanitize_turn(turn)
        assert out["prev_feedback"] == ""
        assert "Which of the following conversions" in out["question"]

    def test_forbids_oracle_fields_hard_assertion(self):
        # §3 guard #1: even if oracle-derived fields are present on the input dict,
        # the sanitizer must NEVER surface them. This is the load-bearing test.
        turn = {
            "prompt_text": _prompt_text(),
            "response_text": "\\bbox{A,B}",
            # All of these are FORBIDDEN and must be ignored/stripped:
            "inline_score": {"passed": False, "selected": ["A"], "score": 0.5},
            "reward": 1.0,
            "eval": {"command": "test -f out.json", "answer": ["A", "E"]},
            "feedback": {"options": {"A": "...", "E": "..."}, "correct": "Well done!"},
        }
        out = sanitize_turn(turn)

        # Output keys are strictly the allow-listed set.
        assert set(out.keys()) == {"question", "answer", "prev_feedback"}

        # No forbidden key name and no forbidden value appears anywhere.
        blob = repr(out)
        for forbidden in (
            "inline_score",
            "reward",
            "selected",
            "expect_exit",
            "test -f out.json",
            '"answer"',
            "options",
            "score",
        ):
            assert forbidden not in blob, f"oracle leak: {forbidden!r} in {blob!r}"

    def test_rejects_non_dict_or_missing_prompt(self):
        # Defensive: a malformed turn must not explode (returns empty fields).
        out = sanitize_turn({"response_text": "x"})
        assert out["question"] == ""
        assert out["prev_feedback"] == ""
        assert out["answer"] == "x"


# ============================ record_type derivation ========================


class TestDeriveRecordType:
    def test_correct_feedback_is_success(self):
        # feedback.correct text is congratulatory with no error markers.
        for fb in ("Well done!", "Perfect!", "Correct — nicely formatted."):
            assert derive_record_type(fb) == "success"

    def test_missed_option_is_failure(self):
        fb = "You missed option A: should have selected it.\nKeep this in mind as you continue with the next task."
        assert derive_record_type(fb) == "failure"

    def test_wrong_option_is_failure(self):
        fb = "You incorrectly selected option C: it is wrong.\nKeep this in mind as you continue with the next task."
        assert derive_record_type(fb) == "failure"

    def test_format_error_is_failure(self):
        fb = (
            "Note: your previous response did not include a \\bbox{X} answer "
            "(e.g. \\bbox{A}). Keep this in mind as you continue with the next task."
        )
        assert derive_record_type(fb) == "failure"

    def test_file_check_incorrect_suffix_is_failure(self):
        # file_check incorrect feedback always carries the continue-reminder suffix.
        fb = "The output file is missing field X.\nKeep this in mind as you continue with the next task."
        assert derive_record_type(fb) == "failure"

    def test_empty_feedback_is_none(self):
        # No feedback string -> outcome unknown -> no graded record.
        assert derive_record_type("") is None
        assert derive_record_type(None) is None


# ============================ JSON parse robustness =========================


class TestParseDistillerJson:
    def test_bare_object(self):
        out = parse_distiller_json('{"record_type": "success", "title": "t"}')
        assert out["record_type"] == "success"

    def test_fenced_json_block(self):
        raw = 'Here you go:\n```json\n{"record_type": "failure", "title": "x"}\n```\nDone.'
        out = parse_distiller_json(raw)
        assert out["record_type"] == "failure"
        assert out["title"] == "x"

    def test_fenced_plain_block(self):
        raw = '```\n{"record_type": "success", "quality_score": 0.8}\n```'
        out = parse_distiller_json(raw)
        assert out["quality_score"] == 0.8

    def test_object_embedded_in_prose(self):
        raw = (
            'The distilled record is {"record_type": "failure", "title": "y"} as shown.'
        )
        out = parse_distiller_json(raw)
        assert out["title"] == "y"

    def test_unparseable_returns_none(self):
        assert parse_distiller_json("no json here at all") is None
        assert parse_distiller_json("") is None

    def test_skips_earlier_non_json_brace_pair(self):
        # An earlier brace pair that is NOT valid JSON must not block a later real
        # object (codex LOW hardening of _balanced_objects).
        raw = 'Note {not json here} then {"record_type": "success", "title": "z"}.'
        out = parse_distiller_json(raw)
        assert out is not None
        assert out["title"] == "z"


# ===================== Distiller lag + persistence (DB) =====================


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_maker(tmp_path_factory):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import mirix.orm  # noqa: F401 -- ensure all ORM classes are registered
    from mirix.orm.base import Base

    db_path = tmp_path_factory.mktemp("distiller") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _ctx():
        async with local() as session:
            try:
                yield session
            finally:
                await session.close()

    yield _ctx
    await engine.dispose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def org(session_maker):
    from mirix.orm.organization import Organization as OrganizationORM

    org_id = f"distiller-org-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        await session.commit()
    return type("Org", (), {"id": org_id})()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def user_a(session_maker, org):
    from mirix.orm.user import User as UserORM

    uid = f"distiller-user-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(
            UserORM(
                id=uid,
                name=uid,
                organization_id=org.id,
                status="active",
                timezone="UTC",
            )
        )
        await session.commit()
    return type("User", (), {"id": uid, "organization_id": org.id})()


async def _insert_minimal_agent(session_maker, org_id: str) -> str:
    from mirix.orm.agent import Agent as AgentORM

    agent_id = f"agent-{uuid.uuid4()}"
    async with session_maker() as session:
        session.add(AgentORM(id=agent_id, organization_id=org_id))
        await session.commit()
    return agent_id


@pytest_asyncio.fixture(loop_scope="module")
async def agent_id(session_maker, org):
    return await _insert_minimal_agent(session_maker, org.id)


@pytest_asyncio.fixture(loop_scope="module")
async def other_agent_id(session_maker, org):
    return await _insert_minimal_agent(session_maker, org.id)


class _FakeLLMClient:
    """A stand-in for an LLMClientBase: records calls, returns a canned reply.

    The distiller only consumes `response.choices[0].message.content`, so we mimic
    just that shape via SimpleNamespace.
    """

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    async def send_llm_request(self, messages, **kwargs):
        self.calls.append(messages)
        msg = SimpleNamespace(content=self.reply)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


def _distiller(
    session_maker, reply: str
) -> tuple[SkillSessionDistiller, _FakeLLMClient]:
    """Build a distiller wired to the hermetic sessionmaker + a fake LLM client."""
    from mirix.services.skill_evolution_record_manager import (
        SkillEvolutionRecordManager,
    )

    mgr = SkillEvolutionRecordManager()
    mgr.session_maker = session_maker
    fake = _FakeLLMClient(reply)
    distiller = SkillSessionDistiller(record_manager=mgr, llm_client=fake)
    return distiller, fake


_SUCCESS_REPLY = (
    '```json\n{"record_type": "success", "title": "boxed answer accepted", '
    '"description": "answer matched the option set", '
    '"detail": "what_worked: emitted \\\\bbox{A,B} in the exact format", '
    '"evidence_round_ids": ["r1", "r2"], "quality_score": 0.7, "generality": 0.6}\n```'
)
_FAILURE_REPLY = (
    '{"record_type": "failure", "title": "missed an option", '
    '"description": "did not select all correct options", '
    '"detail": "root_cause: under-selected; what_to_avoid: re-check every option", '
    '"evidence_round_ids": ["r2", "r3"], "quality_score": 0.8, "generality": 0.5}'
)


def _turn(
    prev_feedback: str, question: str = QUESTION, answer: str = "\\bbox{A,B}"
) -> dict:
    return {
        "prompt_text": _prompt_text(prev_feedback=prev_feedback, question=question),
        "response_text": answer,
    }


def _llm_payload_text(fake: "_FakeLLMClient", call_index: int = 0) -> str:
    """Flatten the user-turn text the distiller sent the LLM on `call_index`."""
    messages = fake.calls[call_index]
    parts = []
    for m in messages:
        for c in m.content:
            parts.append(getattr(c, "text", ""))
    return "\n".join(parts)


@pytest.mark.asyncio(loop_scope="module")
class TestOneRoundLag:
    async def test_first_round_buffers_no_record(self, session_maker, agent_id, user_a):
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        # Round 1: no predecessor feedback -> just buffered, nothing distilled.
        rec = await distiller.distill_round(
            **owner, day="day01", round_id="r1", round_index=1, turn=_turn("")
        )
        assert rec is None
        assert fake.calls == []  # LLM not called yet
        # Nothing persisted.
        pending = await distiller.record_manager.list_pending(agent_id=agent_id)
        assert pending == []

    async def test_second_round_distills_first(self, session_maker, org, user_a):
        # Fresh agent to isolate buffer state.
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # Round 1 buffered.
        await distiller.distill_round(
            **owner, day="day01", round_id="r1", round_index=1, turn=_turn("")
        )
        # Round 2 arrives carrying r1's correctness feedback (incorrect -> failure).
        incorrect_fb = "You missed option B: required.\nKeep this in mind as you continue with the next task."
        rec = await distiller.distill_round(
            **owner,
            day="day01",
            round_id="r2",
            round_index=2,
            turn=_turn(incorrect_fb),
        )
        # Exactly one record, FOR ROUND 1, derived from r2's feedback prefix.
        assert rec is not None
        assert rec.round_id == "r1"
        assert rec.round_index == 1
        assert rec.record_type == "failure"
        assert len(fake.calls) == 1  # one LLM completion

        pending = await distiller.record_manager.list_pending(agent_id=agent)
        assert len(pending) == 1
        assert pending[0].round_id == "r1"

    async def test_record_type_follows_feedback_not_llm(
        self, session_maker, org, user_a
    ):
        # The LLM reply claims "failure", but r(t+1) feedback is the *correct*
        # text -> record_type MUST be success (derived from feedback, not LLM).
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, _ = _distiller(session_maker, _FAILURE_REPLY)  # LLM says failure
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await distiller.distill_round(
            **owner, day="day02", round_id="r1", round_index=1, turn=_turn("")
        )
        rec = await distiller.distill_round(
            **owner,
            day="day02",
            round_id="r2",
            round_index=2,
            turn=_turn("Well done! Correctly boxed."),
        )
        assert rec is not None
        assert rec.record_type == "success"  # from feedback, overriding LLM's claim

    async def test_flush_session_drops_final_round(self, session_maker, org, user_a):
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # Buffer a single (final) round.
        await distiller.distill_round(
            **owner, day="day03", round_id="r1", round_index=1, turn=_turn("")
        )
        # Flush: the last buffered round has no successor -> NO graded record.
        dropped = await distiller.flush_session(agent_id=agent)
        assert dropped is None
        assert fake.calls == []
        pending = await distiller.record_manager.list_pending(agent_id=agent)
        assert pending == []

    async def test_buffer_is_per_agent(self, session_maker, org, user_a):
        # Two agents interleave; each buffers its own previous round.
        a1 = await _insert_minimal_agent(session_maker, org.id)
        a2 = await _insert_minimal_agent(session_maker, org.id)
        distiller, _ = _distiller(session_maker, _FAILURE_REPLY)

        await distiller.distill_round(
            agent_id=a1,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            day="day04",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        await distiller.distill_round(
            agent_id=a2,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            day="day04",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        # Now round 2 for a1 -> distills a1's r1 ONLY.
        incorrect_fb = "You missed option A: required.\nKeep this in mind as you continue with the next task."
        rec = await distiller.distill_round(
            agent_id=a1,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            day="day04",
            round_id="r2",
            round_index=2,
            turn=_turn(incorrect_fb),
        )
        assert rec is not None and rec.agent_id == a1
        # a2 still has only its buffered (un-distilled) r1 -> nothing pending.
        a2_pending = await distiller.record_manager.list_pending(agent_id=a2)
        assert a2_pending == []
        a1_pending = await distiller.record_manager.list_pending(agent_id=a1)
        assert len(a1_pending) == 1

    async def test_unparseable_llm_reply_yields_no_record(
        self, session_maker, org, user_a
    ):
        # If the distiller LLM returns garbage, no record is persisted (the round
        # is consumed; we don't crash the per-round ingest path).
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, _ = _distiller(session_maker, "totally not json")
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await distiller.distill_round(
            **owner, day="day05", round_id="r1", round_index=1, turn=_turn("")
        )
        rec = await distiller.distill_round(
            **owner,
            day="day05",
            round_id="r2",
            round_index=2,
            turn=_turn(
                "You missed option A.\nKeep this in mind as you continue with the next task."
            ),
        )
        assert rec is None
        pending = await distiller.record_manager.list_pending(agent_id=agent)
        assert pending == []

    async def test_llm_sees_outcome_feedback_of_distilled_round(
        self, session_maker, org, user_a
    ):
        # The round being distilled is r1; its OUTCOME feedback is what arrives in
        # r2's prompt. The LLM payload must carry THAT feedback (r1's outcome),
        # NOT r1's own prev_feedback (which was r0's, i.e. empty here). This is the
        # codex HIGH "stale feedback to LLM" regression guard.
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # r1 carries a DISTINCTIVE (but stale, r0->r1) feedback string; if the
        # distiller leaked r1's own prev_feedback it would show up.
        stale = "STALE_R0_FEEDBACK should never reach the distiller LLM."
        await distiller.distill_round(
            **owner, day="day06", round_id="r1", round_index=1, turn=_turn(stale)
        )
        outcome_fb = "You missed option B: OUTCOME_OF_R1.\nKeep this in mind as you continue with the next task."
        await distiller.distill_round(
            **owner, day="day06", round_id="r2", round_index=2, turn=_turn(outcome_fb)
        )
        payload = _llm_payload_text(fake, 0)
        # The LLM must see r1's OUTCOME feedback (from r2's prompt)...
        assert "OUTCOME_OF_R1" in payload
        # ...and must NOT see r1's own stale predecessor feedback.
        assert "STALE_R0_FEEDBACK" not in payload

    async def test_llm_question_excludes_accumulated_history(
        self, session_maker, org, user_a
    ):
        # The buffered prompt_text contains the round's question PLUS a long tail
        # of accumulated role-flattened history (LONG_BODY). The distiller must
        # send only the single round's question, not the trailing history.
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await distiller.distill_round(
            **owner, day="day07", round_id="r1", round_index=1, turn=_turn("")
        )
        await distiller.distill_round(
            **owner,
            day="day07",
            round_id="r2",
            round_index=2,
            turn=_turn(
                "You missed option A.\nKeep this in mind as you continue with the next task."
            ),
        )
        payload = _llm_payload_text(fake, 0)
        assert "Which of the following conversions" in payload  # the question
        assert "prior turn line" not in payload  # the accumulated history tail

    async def test_new_session_does_not_distill_prior_session_final_round(
        self, session_maker, org, user_a
    ):
        # Codex MEDIUM: buffer keyed by (agent, session). A new session's round 1
        # (empty feedback) must NOT finalize the previous session's final buffered
        # round using the new session's (absent) feedback. The prior session's
        # final round is simply dropped.
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # Session day08: a single round, left un-flushed (its final round).
        await distiller.distill_round(
            **owner,
            session_id="day08",
            day="day08",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        # New session day09 begins with round 1 (no feedback).
        rec = await distiller.distill_round(
            **owner,
            session_id="day09",
            day="day09",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        # Nothing distilled across the session boundary; day08's r1 is dropped.
        assert rec is None
        assert fake.calls == []
        pending = await distiller.record_manager.list_pending(agent_id=agent)
        assert pending == []

    async def test_session_scoped_flush_clears_only_that_session(
        self, session_maker, org, user_a
    ):
        # flush_session(session_id=X) drops only session X's buffer; another
        # session's buffered round survives and still distills on its successor.
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, _ = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await distiller.distill_round(
            **owner,
            session_id="sA",
            day="dayA",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        await distiller.distill_round(
            **owner,
            session_id="sB",
            day="dayB",
            round_id="r1",
            round_index=1,
            turn=_turn(""),
        )
        # Flush only session sA.
        await distiller.flush_session(agent_id=agent, session_id="sA")
        # sB's round 1 still buffered -> its round 2 distills it.
        rec = await distiller.distill_round(
            **owner,
            session_id="sB",
            day="dayB",
            round_id="r2",
            round_index=2,
            turn=_turn(
                "You missed option A.\nKeep this in mind as you continue with the next task."
            ),
        )
        assert rec is not None
        assert rec.round_id == "r1"
        assert rec.day == "dayB"

    async def test_whole_agent_flush_clears_all_sessions(
        self, session_maker, org, user_a
    ):
        agent = await _insert_minimal_agent(session_maker, org.id)
        distiller, fake = _distiller(session_maker, _FAILURE_REPLY)
        owner = dict(
            agent_id=agent, user_id=user_a.id, organization_id=user_a.organization_id
        )
        for sid in ("sX", "sY"):
            await distiller.distill_round(
                **owner,
                session_id=sid,
                day=sid,
                round_id="r1",
                round_index=1,
                turn=_turn(""),
            )
        # Whole-agent flush (no session_id) drops every session's buffer.
        await distiller.flush_session(agent_id=agent)
        # A fresh round 1 in sX now has no predecessor -> nothing distilled.
        rec = await distiller.distill_round(
            **owner,
            session_id="sX",
            day="sX",
            round_id="r1",
            round_index=1,
            turn=_turn(
                "You missed option A.\nKeep this in mind as you continue with the next task."
            ),
        )
        assert rec is None
        assert fake.calls == []


# ============================== Distiller surface ===========================


class TestDistillerSurface:
    def test_distill_round_is_async(self):
        assert inspect.iscoroutinefunction(
            inspect.unwrap(SkillSessionDistiller.distill_round)
        )

    def test_flush_session_is_async(self):
        assert inspect.iscoroutinefunction(
            inspect.unwrap(SkillSessionDistiller.flush_session)
        )

    def test_no_asyncio_run(self):
        import mirix.services.skill_session_distiller as mod

        src = inspect.getsource(mod)
        assert "asyncio.run(" not in src, "distiller must never call asyncio.run()"

    def test_no_forbidden_oracle_imports(self):
        # The distiller must not import the leakage-bearing adapters.
        import mirix.services.skill_session_distiller as mod

        src = inspect.getsource(mod)
        assert "evolver_adapter" not in src
        assert "round_to_message" not in src
        assert "PROMPT_TAIL_CHARS" not in src
