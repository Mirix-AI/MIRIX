"""Tests for the general session-experience store (Goal 2 storage layer).

Three layers, mirroring test_skill_evolution_record.py:

1. Pydantic schema validation (DB-free) — rejects bad experience_type/status,
   enforces length caps, defaults status='pending', CLAMPS importance/credibility
   into [0,1] (garbage -> 0.0).
2. ORM shape (DB-free) — table name, mixin + explicit columns, sexp- id prefix,
   agent+status index, ORM package registration, migration SQL.
3. Manager behavior (DB-backed, integration) — create, list ordering by
   importance*credibility DESC, [0,1] enforcement at persistence, mark_consumed
   idempotency + lineage, mark_superseded, aggregate, per-agent isolation.

The DB-backed layer runs on a hermetic throwaway-SQLite sessionmaker (no
Postgres) and is marked `integration` per the TESTS-phase spec.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import inspect as sa_inspect

from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
from mirix.schemas.skill_experience import (
    SKILL_EXPERIENCE_MAX_CONTENT_LEN,
    SKILL_EXPERIENCE_MAX_EVIDENCE_LEN,
    SKILL_EXPERIENCE_MAX_TITLE_LEN,
    SkillExperience as PydanticSkillExperience,
    SkillExperienceBase,
    SkillExperienceCreate,
    SkillExperienceResponse,
    SkillExperienceUpdate,
    _clamp01,
)
from mirix.services.skill_experience_manager import SkillExperienceManager


# ============================ Schema validation ===========================


class TestSkillExperienceSchema:
    def _kwargs(self, **overrides) -> dict:
        base = dict(
            session_id="sess-1",
            experience_type="worth_learning",
            title="batch independent calls",
            content="when calls are independent, batch them",
            importance=0.8,
            credibility=0.9,
            evidence='{"quote":"great","signal_type":"user_confirmation"}',
        )
        base.update(overrides)
        return base

    def test_valid_base(self):
        rec = SkillExperienceBase(**self._kwargs())
        assert rec.experience_type == "worth_learning"
        assert rec.status == "pending"

    def test_experience_type_accepts_avoiding(self):
        rec = SkillExperienceBase(**self._kwargs(experience_type="worth_avoiding"))
        assert rec.experience_type == "worth_avoiding"

    def test_experience_type_rejects_unknown(self):
        with pytest.raises(ValidationError):
            SkillExperienceBase(**self._kwargs(experience_type="partial"))

    def test_status_rejects_unknown(self):
        with pytest.raises(ValidationError):
            SkillExperienceBase(**self._kwargs(status="archived"))

    def test_status_accepts_known(self):
        for s in ("pending", "consumed", "superseded"):
            assert SkillExperienceBase(**self._kwargs(status=s)).status == s

    def test_importance_clamped_high(self):
        rec = SkillExperienceBase(**self._kwargs(importance=5.0))
        assert rec.importance == 1.0

    def test_credibility_clamped_low(self):
        rec = SkillExperienceBase(**self._kwargs(credibility=-3.0))
        assert rec.credibility == 0.0

    def test_garbage_score_becomes_zero(self):
        rec = SkillExperienceBase(**self._kwargs(importance="garbage"))
        assert rec.importance == 0.0

    def test_clamp01_helper(self):
        assert _clamp01(5.0) == 1.0
        assert _clamp01(-3) == 0.0
        assert _clamp01("nope") == 0.0
        assert _clamp01(0.5) == 0.5
        assert _clamp01(float("inf")) == 1.0

    def test_clamp01_nan_must_become_zero(self):
        # Regression for the NaN gap (codex P1-1, fixed in _clamp01): NaN slips past
        # both `< 0` and `> 1` (all NaN comparisons are False) and would poison the
        # importance*credibility ordering. The [0,1] invariant requires NaN -> 0.0.
        assert _clamp01(float("nan")) == 0.0
        assert _clamp01("nan") == 0.0

    def test_title_length_cap(self):
        with pytest.raises(ValidationError):
            SkillExperienceBase(
                **self._kwargs(title="x" * (SKILL_EXPERIENCE_MAX_TITLE_LEN + 1))
            )

    def test_content_length_cap(self):
        with pytest.raises(ValidationError):
            SkillExperienceBase(
                **self._kwargs(content="x" * (SKILL_EXPERIENCE_MAX_CONTENT_LEN + 1))
            )

    def test_evidence_length_cap(self):
        with pytest.raises(ValidationError):
            SkillExperienceBase(
                **self._kwargs(evidence="x" * (SKILL_EXPERIENCE_MAX_EVIDENCE_LEN + 1))
            )

    def test_create_requires_owners(self):
        rec = SkillExperienceCreate(
            **self._kwargs(), agent_id="a", user_id="u", organization_id="o"
        )
        assert rec.agent_id == "a"

    def test_full_schema_carries_db_fields(self):
        rec = PydanticSkillExperience(
            **self._kwargs(), id="sexp-abc", agent_id="a",
            user_id="u", organization_id="o",
        )
        assert rec.id == "sexp-abc"
        assert rec.consumed_by is None
        assert rec.influenced_skill_ids is None

    def test_update_validates_status(self):
        with pytest.raises(ValidationError):
            SkillExperienceUpdate(id="sexp-1", status="bogus")

    def test_update_accepts_consumed(self):
        upd = SkillExperienceUpdate(id="sexp-1", status="consumed", consumed_by="run-1")
        assert upd.status == "consumed"

    def test_response_is_subclass(self):
        assert issubclass(SkillExperienceResponse, PydanticSkillExperience)

    def test_id_prefix(self):
        assert SkillExperienceBase.__id_prefix__ == "sexp"


# ================================ ORM shape ===============================


class TestSkillExperienceORM:
    def _cols(self):
        return {c.key for c in sa_inspect(SkillExperienceORM).columns}

    def test_table_name(self):
        assert SkillExperienceORM.__tablename__ == "skill_experience"

    def test_inherits_mixin_columns(self):
        cols = self._cols()
        for c in ["id", "organization_id", "user_id", "agent_id", "created_at"]:
            assert c in cols, f"missing inherited column: {c}"

    def test_has_explicit_columns(self):
        cols = self._cols()
        for c in [
            "session_id", "experience_type", "title", "content",
            "importance", "credibility", "evidence", "status",
            "consumed_by", "influenced_skill_ids",
        ]:
            assert c in cols, f"missing explicit column: {c}"

    def test_id_default_uses_sexp_prefix(self):
        default = SkillExperienceORM.__table__.c.id.default
        assert default is not None
        assert default.arg(None).startswith("sexp-")

    def test_status_defaults_to_pending(self):
        assert SkillExperienceORM.__table__.c.status.default.arg == "pending"

    def test_pydantic_model_link(self):
        assert SkillExperienceORM.__pydantic_model__ is PydanticSkillExperience

    def test_has_agent_status_index(self):
        idx = {i.name for i in SkillExperienceORM.__table__.indexes}
        assert "ix_skill_experience_agent_status" in idx

    def test_registered_in_orm_package(self):
        import mirix.orm as orm_pkg

        assert "SkillExperience" in orm_pkg.__all__
        assert orm_pkg.SkillExperience is SkillExperienceORM


# =============================== Migration SQL =============================


class TestMigrationSql:
    SQL_PATH = Path("scripts/migrate_add_skill_experience.sql")

    def _sql(self) -> str:
        return self.SQL_PATH.read_text()

    def test_creates_table(self):
        assert "CREATE TABLE IF NOT EXISTS skill_experience" in self._sql()

    def test_idempotent_block(self):
        sql = self._sql()
        assert "CREATE INDEX IF NOT EXISTS" in sql

    def test_creates_agent_status_index(self):
        assert "ix_skill_experience_agent_status" in self._sql()


# ============================== Manager surface ============================


class TestManagerSurface:
    def test_exposes_async_methods(self):
        for name in (
            "create_experience", "list_experiences",
            "mark_consumed", "mark_superseded", "aggregate",
        ):
            assert hasattr(SkillExperienceManager, name)
            unwrapped = inspect.unwrap(getattr(SkillExperienceManager, name))
            assert inspect.iscoroutinefunction(unwrapped), f"{name} must be async"

    def test_no_asyncio_run(self):
        src = inspect.getsource(SkillExperienceManager)
        assert "asyncio.run(" not in src


# ============================ Manager (DB-backed) ==========================
# Hermetic throwaway-SQLite (no Postgres). Marked integration per the spec.


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_maker(tmp_path_factory):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import mirix.orm  # noqa: F401 -- register all ORM classes
    from mirix.orm.base import Base

    db_path = tmp_path_factory.mktemp("sexp") / "test.db"
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

    org_id = f"sexp-org-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        await session.commit()
    return type("Org", (), {"id": org_id})()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def user_a(session_maker, org):
    from mirix.orm.user import User as UserORM

    uid = f"sexp-user-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(
            UserORM(id=uid, name=uid, organization_id=org.id,
                    status="active", timezone="UTC")
        )
        await session.commit()
    return type("User", (), {"id": uid, "organization_id": org.id})()


async def _insert_agent(session_maker, org_id: str) -> str:
    from mirix.orm.agent import Agent as AgentORM

    agent_id = f"agent-{uuid.uuid4()}"
    async with session_maker() as session:
        session.add(AgentORM(id=agent_id, organization_id=org_id))
        await session.commit()
    return agent_id


def _manager(session_maker):
    mgr = SkillExperienceManager()
    mgr.session_maker = session_maker
    return mgr


def _exp_kwargs(idx, etype="worth_learning", importance=0.6, credibility=0.6, **ov):
    base = dict(
        session_id=f"sess-{idx}",
        experience_type=etype,
        title=f"title-{idx}",
        content=f"content-{idx}",
        importance=importance,
        credibility=credibility,
        evidence='{"quote":"q","signal_type":"inferred"}',
    )
    base.update(ov)
    return base


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestCreateAndList:
    async def test_create_and_read_back(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        rec = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(1),
        )
        assert rec.id.startswith("sexp-")
        assert rec.status == "pending"
        assert rec.consumed_by is None
        assert rec.experience_type == "worth_learning"

    async def test_rejects_bad_experience_type(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        with pytest.raises(ValidationError):
            await mgr.create_experience(
                agent_id=agent_id, user_id=user_a.id,
                organization_id=user_a.organization_id,
                **_exp_kwargs(1, etype="partial"),
            )
        listed = await mgr.list_experiences(agent_id=agent_id)
        assert listed == []

    async def test_clamps_at_persistence(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        rec = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=user_a.organization_id,
            **_exp_kwargs(1, importance=9.0, credibility=-2.0),
        )
        assert rec.importance == 1.0
        assert rec.credibility == 0.0

    async def test_list_ordered_by_priority_desc(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        # priorities: low=0.09, high=0.72, mid=0.30
        await mgr.create_experience(**owner, **_exp_kwargs("low", importance=0.9, credibility=0.1))
        await mgr.create_experience(**owner, **_exp_kwargs("high", importance=0.8, credibility=0.9))
        await mgr.create_experience(**owner, **_exp_kwargs("mid", importance=0.5, credibility=0.6))
        listed = await mgr.list_experiences(agent_id=agent_id)
        sessions = [e.session_id for e in listed]
        assert sessions == ["sess-high", "sess-mid", "sess-low"]

    async def test_isolated_per_agent(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        a1 = await _insert_agent(session_maker, org.id)
        a2 = await _insert_agent(session_maker, org.id)
        await mgr.create_experience(
            agent_id=a1, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(1),
        )
        await mgr.create_experience(
            agent_id=a2, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(2),
        )
        listed = await mgr.list_experiences(agent_id=a2)
        assert len(listed) == 1
        assert all(e.agent_id == a2 for e in listed)

    async def test_respects_limit(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        for i in range(4):
            await mgr.create_experience(**owner, **_exp_kwargs(i))
        listed = await mgr.list_experiences(agent_id=agent_id, limit=2)
        assert len(listed) == 2


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestMarkTransitions:
    async def test_mark_consumed_with_lineage(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        r1 = await mgr.create_experience(**owner, **_exp_kwargs(1))
        r2 = await mgr.create_experience(**owner, **_exp_kwargs(2))
        n = await mgr.mark_consumed(
            ids=[r1.id, r2.id], run_id="xprun-1",
            influenced_skill_ids=["skill-a", "skill-b"],
        )
        assert n == 2
        # No longer pending.
        assert await mgr.list_experiences(agent_id=agent_id) == []
        # Lineage + consumed_by persisted.
        consumed = await mgr.list_experiences(agent_id=agent_id, status="consumed")
        assert len(consumed) == 2
        assert all(c.consumed_by == "xprun-1" for c in consumed)
        assert all(set(c.influenced_skill_ids) == {"skill-a", "skill-b"} for c in consumed)

    async def test_mark_consumed_idempotent(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        r1 = await mgr.create_experience(**owner, **_exp_kwargs(1))
        first = await mgr.mark_consumed(ids=[r1.id], run_id="run-A")
        assert first == 1
        second = await mgr.mark_consumed(ids=[r1.id], run_id="run-B")
        assert second == 0  # already consumed; consumed_by not clobbered
        consumed = await mgr.list_experiences(agent_id=agent_id, status="consumed")
        assert consumed[0].consumed_by == "run-A"

    async def test_mark_consumed_empty_noop(self, session_maker):
        mgr = _manager(session_maker)
        assert await mgr.mark_consumed(ids=[], run_id="run-0") == 0

    async def test_mark_superseded(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        r1 = await mgr.create_experience(**owner, **_exp_kwargs(1))
        n = await mgr.mark_superseded(ids=[r1.id])
        assert n == 1
        assert await mgr.list_experiences(agent_id=agent_id) == []
        sup = await mgr.list_experiences(agent_id=agent_id, status="superseded")
        assert len(sup) == 1


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestAggregate:
    async def test_counts_by_type_and_priority(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        a = await mgr.create_experience(
            **owner, **_exp_kwargs(1, etype="worth_avoiding", importance=1.0, credibility=1.0)
        )
        b = await mgr.create_experience(
            **owner, **_exp_kwargs(2, etype="worth_avoiding", importance=0.5, credibility=0.5)
        )
        c = await mgr.create_experience(
            **owner, **_exp_kwargs(3, etype="worth_learning", importance=0.4, credibility=0.5)
        )
        agg = await mgr.aggregate(ids=[a.id, b.id, c.id])
        assert agg["n"] == 3
        assert agg["n_worth_avoiding"] == 2
        assert agg["n_worth_learning"] == 1
        # 1.0 + 0.25 + 0.20 = 1.45
        assert agg["sum_priority"] == pytest.approx(1.45)

    async def test_empty_ids(self, session_maker):
        mgr = _manager(session_maker)
        agg = await mgr.aggregate(ids=[])
        assert agg == {"n": 0, "n_worth_learning": 0,
                       "n_worth_avoiding": 0, "sum_priority": 0.0}


# ====================== End-to-end-ish distill -> consume ==================


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestDistillThenConsume:
    async def test_distilled_experiences_are_consumable(self, session_maker, org, user_a):
        """Persist via the distiller's _persist_experiences (stubbed LLM upstream),
        then drive the Goal-3 core with mock snapshot/step to consume them."""
        import json as _json

        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )
        from mirix.services.skill_experience_curator import (
            _run_experience_evolution_core,
        )

        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        meta = type("M", (), {"id": agent_id})()
        user = type("U", (), {"id": user_a.id})()
        actor = type("A", (), {"organization_id": user_a.organization_id})()

        d = SessionExperienceDistiller(experience_manager=mgr)
        parsed = [
            {"experience_type": "worth_avoiding", "title": "no missing column",
             "importance": 0.9, "credibility": 0.9,
             "evidence": {"quote": "no such column", "signal_type": "tool_error"}},
            {"experience_type": "worth_learning", "title": "batch calls",
             "importance": 0.7, "credibility": 0.8,
             "evidence": {"quote": "great", "signal_type": "user_confirmation"}},
        ]
        created = await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="sess-e2e", parsed=parsed,
        )
        assert len(created) == 2

        # Now consume via the Goal-3 core (mock agent/step; real manager+DB).
        agent = type("Ag", (), {})()

        async def snapshot():
            return []

        stepped = {}

        async def run_step(a, payload, budget):
            stepped["payload"] = payload
            stepped["budget"] = budget

        result = await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent, agent_id="proc-e2e",
            meta_agent_id=agent_id, user_id=user_a.id,
            snapshot_skills=snapshot, run_step=run_step,
        )
        assert result["skipped"] is False
        assert result["consumed_count"] == 2
        # Payload carries both kinds, priority-ordered (avoid 0.81 before learn 0.56).
        assert stepped["payload"].index("[AVOID]") < stepped["payload"].index("[LEARN]")
        # All pending consumed.
        assert await mgr.list_experiences(agent_id=agent_id, status="pending") == []
        consumed = await mgr.list_experiences(agent_id=agent_id, status="consumed")
        assert {c.consumed_by for c in consumed} == {result["run_id"]}
        _ = _json  # silence unused if branch above changes
