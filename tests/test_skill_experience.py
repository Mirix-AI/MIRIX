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

import numpy as np
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
        # Regression for the NaN gap (fixed in _clamp01): NaN slips past
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

    def test_has_the_experience_pool_columns_and_index(self):
        cols = self._cols()
        for c in [
            "disposition", "cited_skill_ids", "pool_scope", "pool_embedding",
            "pool_embedding_model", "pool_embedding_failures", "recurrence_count",
            "ignored_pushes", "pushed_count", "last_pushed_at", "pool_retired_at",
            "pool_retire_reason",
        ]:
            assert c in cols, f"missing pool column: {c}"
        idx = {i.name: i for i in SkillExperienceORM.__table__.indexes}
        pool_index = idx["ix_skill_experience_agent_disposition"]
        assert [c.name for c in pool_index.columns] == ["agent_id", "disposition"]
        # The vector is deferred: listing and settling never load it.
        assert sa_inspect(SkillExperienceORM).attrs.pool_embedding.deferred

    def test_the_disposition_values_are_validated(self):
        base = dict(
            id="sexp-1", agent_id="a", user_id="u", organization_id="o",
            session_id="s", experience_type="worth_learning", title="t",
        )
        for value in ("used", "covered", "not_worth", "deferred", "unaddressed"):
            assert PydanticSkillExperience(**base, disposition=value).disposition == value
        with pytest.raises(ValidationError):
            PydanticSkillExperience(**base, disposition="superseded")
        with pytest.raises(ValidationError):
            PydanticSkillExperience(**base, pool_retire_reason="deferred")
        for reason in ("used", "covered", "ignored", "merged", "unembeddable"):
            assert (
                PydanticSkillExperience(**base, pool_retire_reason=reason).pool_retire_reason
                == reason
            )
        with pytest.raises(ValidationError):
            PydanticSkillExperience(**base, recurrence_count=0)
        with pytest.raises(ValidationError):
            PydanticSkillExperience(**base, pool_embedding_failures=-1)

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
            "create_experience", "list_experiences", "list_pending_for_curation",
            "settle_curation_batch", "mark_consumed", "mark_superseded", "aggregate",
            "delete_by_user_id", "delete_by_client_id",
            # The experience pool.
            "list_active_pool", "load_pool_vectors", "store_pool_vectors",
            "count_active_pool",
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
    from mirix.settings import settings

    if settings.mirix_pg_uri_no_default:
        # MIRIX_PG_URI (read at import) puts Postgres-only indexes into the ORM
        # metadata, which a SQLite file cannot create: this hermetic layer runs
        # in the integration selection without Postgres (isolated HOME).
        pytest.skip("SQLite-only store tests; run them without MIRIX_PG_URI")

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

    async def test_ids_scopes_to_given_set(self, session_maker, org, user_a):
        # The Goal-3 curator passes THIS round's freshly-distilled ids so an
        # evolve only sees its own batch, never the whole pending pool.
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        a = await mgr.create_experience(**owner, **_exp_kwargs("a", importance=0.9, credibility=0.9))
        await mgr.create_experience(**owner, **_exp_kwargs("b", importance=0.8, credibility=0.8))
        c = await mgr.create_experience(**owner, **_exp_kwargs("c", importance=0.7, credibility=0.7))
        # Scope to a + c only; b must stay invisible.
        scoped = await mgr.list_experiences(agent_id=agent_id, ids=[a.id, c.id])
        # Priority-ordered WITHIN the scope (a=0.81 before c=0.49).
        assert [e.id for e in scoped] == [a.id, c.id]

    async def test_empty_ids_returns_nothing(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(1),
        )
        # ids=[] is an explicit "scope to nothing" (distinct from ids=None).
        assert await mgr.list_experiences(agent_id=agent_id, ids=[]) == []
        # ids=None (default) still returns the row.
        assert len(await mgr.list_experiences(agent_id=agent_id)) == 1


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
class TestDistillThenCurate:
    """Distilled experiences reach the curator: every
    pending one, oldest first, with no per-run cap and no row limit; a batch
    settles in the caller's transaction."""

    async def test_distilled_experiences_are_listed_and_consumed_by_their_batch(
        self, session_maker, org, user_a
    ):
        import datetime as dt

        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )
        from mirix.services.skill_experience_manager import BATCH_CONSUME

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

        pending = await mgr.list_pending_for_curation(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        assert {e.id for e in pending} == {c.id for c in created}
        # Oldest first: (created_at, id), never the importance ranking.
        assert [e.id for e in pending] == [
            e.id for e in sorted(pending, key=lambda e: (e.created_at, e.id))
        ]

        now = dt.datetime.now(dt.timezone.utc)
        async with session_maker() as db:
            settled = await SkillExperienceManager.settle_curation_batch(
                db, ids=[e.id for e in pending], agent_id=agent_id,
                user_id=user_a.id, outcome=BATCH_CONSUME, batch_id="run-x:b0",
                max_attempts=3, now=now,
            )
            await db.commit()
        assert sorted(settled["consumed"]) == sorted(c.id for c in created)
        assert await mgr.list_pending_for_curation(
            agent_id=agent_id, user_id=user_a.id,
        ) == []
        consumed = await mgr.list_experiences(agent_id=agent_id, status="consumed")
        assert {c.consumed_by for c in consumed} == {"run-x:b0"}

    async def test_every_pending_experience_is_listed_no_cap_no_row_limit(
        self, session_maker, org, user_a
    ):
        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        owner = dict(agent_id=agent_id, user_id=user_a.id,
                     organization_id=user_a.organization_id)
        n = 130  # above the old per-run cap (25) and the old 100-row limit
        ids = [
            (await mgr.create_experience(**owner, **_exp_kwargs(i))).id
            for i in range(n)
        ]
        pending = await mgr.list_pending_for_curation(
            agent_id=agent_id, user_id=user_a.id,
        )
        assert sorted(e.id for e in pending) == sorted(ids)
        assert await mgr.list_experiences(
            agent_id=agent_id, status="superseded", limit=n
        ) == []

    async def test_a_returned_batch_counts_attempts_and_closes_out_at_the_limit(
        self, session_maker, org, user_a
    ):
        import datetime as dt

        from mirix.services.skill_experience_manager import (
            BATCH_KEEP,
            BATCH_RETURN,
            is_unaddressed,
        )

        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        rec = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(1),
        )

        async def settle(outcome, batch_id):
            async with session_maker() as db:
                result = await SkillExperienceManager.settle_curation_batch(
                    db, ids=[rec.id], agent_id=agent_id, user_id=user_a.id,
                    outcome=outcome, batch_id=batch_id, max_attempts=3,
                    now=dt.datetime.now(dt.timezone.utc),
                )
                await db.commit()
            [row] = await mgr.list_experiences(agent_id=agent_id, ids=[rec.id], status=None)
            return result, row

        result, row = await settle(BATCH_KEEP, "r0:b0")  # infrastructure: untouched
        assert result == {"consumed": [], "returned": [], "unaddressed": []}
        assert row.status == "pending" and row.curation_attempts == 0
        for attempt, batch_id in ((1, "r1:b0"), (2, "r2:b0")):
            result, row = await settle(BATCH_RETURN, batch_id)
            assert result["returned"] == [rec.id]
            assert row.status == "pending" and row.curation_attempts == attempt
        result, row = await settle(BATCH_RETURN, "r3:b0")
        assert result["unaddressed"] == [rec.id]
        assert row.status == "consumed" and row.curation_attempts == 3
        assert row.consumed_by == "r3:b0:unaddressed" and is_unaddressed(row)
        # Settled rows are no longer pending: a repeat touches nothing.
        result, row = await settle(BATCH_RETURN, "r4:b0")
        assert result == {"consumed": [], "returned": [], "unaddressed": []}
        assert row.curation_attempts == 3

    async def test_another_owner_s_rows_are_never_settled(self, session_maker, org, user_a):
        import datetime as dt

        from mirix.services.skill_experience_manager import BATCH_CONSUME

        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        other_agent = await _insert_agent(session_maker, org.id)
        rec = await mgr.create_experience(
            agent_id=other_agent, user_id=user_a.id,
            organization_id=user_a.organization_id, **_exp_kwargs(2),
        )
        async with session_maker() as db:
            result = await SkillExperienceManager.settle_curation_batch(
                db, ids=[rec.id], agent_id=agent_id, user_id=user_a.id,
                outcome=BATCH_CONSUME, batch_id="r:b0", max_attempts=3,
                now=dt.datetime.now(dt.timezone.utc),
            )
            await db.commit()
        assert result["consumed"] == []
        [row] = await mgr.list_experiences(agent_id=other_agent, ids=[rec.id], status=None)
        assert row.status == "pending"
        # The right agent and user, but another organization: untouched too.
        async with session_maker() as db:
            result = await SkillExperienceManager.settle_curation_batch(
                db, ids=[rec.id], agent_id=other_agent, user_id=user_a.id,
                outcome=BATCH_CONSUME, batch_id="r:b1", max_attempts=3,
                now=dt.datetime.now(dt.timezone.utc), organization_id="org-elsewhere",
            )
            await db.commit()
        assert result["consumed"] == []
        [row] = await mgr.list_experiences(agent_id=other_agent, ids=[rec.id], status=None)
        assert row.status == "pending"


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestErasure:
    """User/client erasure: experiences are distilled from verbatim user
    conversations, so the irreversible purge paths must cover them."""

    async def test_delete_by_user_id_removes_all_statuses_for_that_user_only(
        self, session_maker, org, user_a
    ):
        from mirix.orm.user import User as UserORM

        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        other_uid = f"sexp-user-{uuid.uuid4().hex[:8]}"
        async with session_maker() as session:
            session.add(
                UserORM(id=other_uid, name=other_uid, organization_id=org.id,
                        status="active", timezone="UTC")
            )
            await session.commit()

        kept = await mgr.create_experience(
            agent_id=agent_id, user_id=other_uid,
            organization_id=org.id, **_exp_kwargs("erase-keep"),
        )
        target = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=org.id, **_exp_kwargs("erase-pending"),
        )
        consumed = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id,
            organization_id=org.id, **_exp_kwargs("erase-consumed"),
        )
        await mgr.mark_consumed(ids=[consumed.id], run_id="run-x")

        deleted = await mgr.delete_by_user_id(user_id=user_a.id)

        assert deleted >= 2  # pending AND consumed rows both erased
        for status in ("pending", "consumed", "superseded"):
            assert await mgr.list_experiences(
                agent_id=agent_id, user_id=user_a.id, status=status, limit=50
            ) == []
        # The other user's experience is untouched.
        assert [e.id for e in await mgr.list_experiences(
            agent_id=agent_id, user_id=other_uid, status="pending", limit=50
        )] == [kept.id]

    async def test_delete_by_client_id_uses_creation_attribution(
        self, session_maker, org, user_a
    ):
        from mirix.schemas.client import Client as PydanticClient

        mgr = _manager(session_maker)
        agent_id = await _insert_agent(session_maker, org.id)
        client_a = PydanticClient(
            id=f"sexp-client-{uuid.uuid4().hex[:8]}", organization_id=org.id,
            name="A", write_scope="t", read_scopes=["t"],
        )
        client_b = PydanticClient(
            id=f"sexp-client-{uuid.uuid4().hex[:8]}", organization_id=org.id,
            name="B", write_scope="t", read_scopes=["t"],
        )
        mine = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id, organization_id=org.id,
            created_by_id=client_a.id, **_exp_kwargs("client-a"),
        )
        theirs = await mgr.create_experience(
            agent_id=agent_id, user_id=user_a.id, organization_id=org.id,
            created_by_id=client_b.id, **_exp_kwargs("client-b"),
        )

        deleted = await mgr.delete_by_client_id(actor=client_a)

        assert deleted == 1
        remaining = {e.id for e in await mgr.list_experiences(
            agent_id=agent_id, user_id=user_a.id, status="pending", limit=50
        )}
        assert mine.id not in remaining
        assert theirs.id in remaining


# ============================ Experience pool =======================
# The pool reads, writes and the batch settle on the same hermetic SQLite
# store. Not marked integration: nothing here needs a server or Postgres.


async def _pool_owner(session_maker, org_id=None):
    """A fresh (organization, user, meta agent) owner."""
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM

    org_id = org_id or f"pool-org-{uuid.uuid4().hex[:8]}"
    uid = f"pool-user-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        if await session.get(OrganizationORM, org_id) is None:
            session.add(OrganizationORM(id=org_id, name=org_id))
            await session.flush()
        session.add(
            UserORM(id=uid, name=uid, organization_id=org_id, status="active", timezone="UTC")
        )
        await session.commit()
    agent_id = await _insert_agent(session_maker, org_id)
    return type("Owner", (), {"org": org_id, "user": uid, "agent": agent_id})()


async def _pool_row(
    session_maker,
    owner,
    *,
    seconds,
    status="consumed",
    disposition="deferred",
    scope=...,
    cited=None,
    retired=False,
    deleted=False,
    vector=None,
    model=None,
    ignored=0,
    recurrence=1,
    pushed=0,
    failures=0,
    title=None,
    user=None,
    agent=None,
    org=None,
):
    import datetime as dt

    if scope is ...:  # a pending experience has not entered any pool yet
        scope = None if status == "pending" else "scope-a"
    row = SkillExperienceORM(
        agent_id=agent or owner.agent,
        user_id=user or owner.user,
        organization_id=org or owner.org,
        session_id=f"s-{seconds}",
        experience_type="worth_learning",
        title=title or f"lesson {seconds}",
        content=f"content {seconds}",
        importance=0.5,
        credibility=0.5,
        evidence="",
        status=status,
        disposition=disposition,
        pool_scope=scope,
        cited_skill_ids=cited,
        pool_embedding=vector,
        pool_embedding_model=model,
        ignored_pushes=ignored,
        recurrence_count=recurrence,
        pushed_count=pushed,
        pool_embedding_failures=failures,
        pool_retired_at=(
            dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc) if retired else None
        ),
        is_deleted=deleted,
    )
    row.created_at = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        seconds=seconds
    )
    async with session_maker() as session:
        session.add(row)
        await session.commit()
    return row.id


async def _pool_rows(session_maker, ids):
    from sqlalchemy import select
    from sqlalchemy.orm import undefer

    async with session_maker() as session:
        rows = (
            await session.execute(
                select(SkillExperienceORM)
                .options(undefer(SkillExperienceORM.pool_embedding))
                .where(SkillExperienceORM.id.in_(list(ids)))
            )
        ).scalars().all()
        return {row.id: row for row in rows}


async def _settle(session_maker, owner, ids, outcome, pool, *, batch_id="r:b0", now=None):
    import datetime as dt

    now = now or dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    async with session_maker() as db:
        result = await SkillExperienceManager.settle_curation_batch(
            db,
            ids=list(ids),
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            outcome=outcome,
            batch_id=batch_id,
            max_attempts=3,
            now=now,
            pool=pool,
        )
        await db.commit()
    return result


@pytest.mark.asyncio(loop_scope="module")
class TestExperiencePool:
    async def test_the_active_pool_filters_owner_scope_state_and_citations(
        self, session_maker
    ):
        owner = await _pool_owner(session_maker)
        other_user = await _pool_owner(session_maker, org_id=owner.org)
        other_org = await _pool_owner(session_maker)
        other_agent = await _insert_agent(session_maker, owner.org)
        mgr = _manager(session_maker)
        vec = [1.0, 0.0, 0.0]
        keep_deferred = await _pool_row(session_maker, owner, seconds=1, vector=vec, model="m")
        keep_not_worth = await _pool_row(
            session_maker, owner, seconds=2, disposition="not_worth", ignored=1
        )
        excluded = [
            await _pool_row(session_maker, owner, seconds=3, status="pending", disposition=None),
            await _pool_row(session_maker, owner, seconds=4, disposition="used"),
            await _pool_row(session_maker, owner, seconds=5, disposition="covered"),
            await _pool_row(session_maker, owner, seconds=6, disposition="unaddressed"),
            await _pool_row(session_maker, owner, seconds=7, disposition=None),  # legacy
            await _pool_row(session_maker, owner, seconds=8, retired=True),
            await _pool_row(session_maker, owner, seconds=9, deleted=True),
            await _pool_row(session_maker, owner, seconds=10, scope="scope-b"),
            await _pool_row(session_maker, owner, seconds=11, scope=None),
            await _pool_row(session_maker, owner, seconds=12, cited=["proc-x"]),
            await _pool_row(session_maker, owner, seconds=13, agent=other_agent),
            await _pool_row(
                session_maker, owner, seconds=14, user=other_user.user
            ),
            await _pool_row(
                session_maker, other_org, seconds=15,
            ),
        ]
        candidates, truncated = await mgr.list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=100,
        )
        assert [c.id for c in candidates] == [keep_deferred, keep_not_worth]
        assert not truncated
        assert not set(excluded) & {c.id for c in candidates}
        first = candidates[0]
        assert first.vector is not None and list(first.vector[:3]) == vec
        assert first.vector_model == "m"
        assert candidates[1].vector is None and candidates[1].ignored_pushes == 1
        # A scope-less owner never pools.
        assert await mgr.list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope=None,
            limit=100,
        ) == ([], False)
        # Another organization's owner id with this organization: nothing.
        assert (
            await mgr.list_active_pool(
                agent_id=other_org.agent,
                user_id=other_org.user,
                organization_id=owner.org,
                scope="scope-a",
                limit=100,
            )
        )[0] == []

    async def test_the_scan_is_newest_first_bounded_and_returned_oldest_first(
        self, session_maker
    ):
        owner = await _pool_owner(session_maker)
        mgr = _manager(session_maker)
        ids = [await _pool_row(session_maker, owner, seconds=i) for i in range(5)]
        candidates, truncated = await mgr.list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=3,
        )
        assert truncated
        assert [c.id for c in candidates] == ids[2:]  # the 3 newest, oldest first
        _, truncated = await mgr.list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=5,
        )
        assert not truncated

    async def test_vectors_are_stored_for_the_owner_only_under_the_guard(
        self, session_maker
    ):
        from mirix.constants import MAX_EMBEDDING_DIM

        owner = await _pool_owner(session_maker)
        stranger = await _pool_owner(session_maker, org_id=owner.org)
        mgr = _manager(session_maker)
        mine = await _pool_row(session_maker, owner, seconds=1, status="pending", disposition=None)
        theirs = await _pool_row(session_maker, stranger, seconds=2)
        calls = []

        async def guard(db):
            calls.append(db)

        stored = await mgr.store_pool_vectors(
            {mine: [0.5, 0.5], theirs: [1.0, 0.0]},
            model="m-1",
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            write_guard=guard,
        )
        assert stored == 1 and len(calls) == 1
        rows = await _pool_rows(session_maker, [mine, theirs])
        assert len(rows[mine].pool_embedding) == MAX_EMBEDDING_DIM
        assert list(rows[mine].pool_embedding[:2]) == [0.5, 0.5]
        assert rows[mine].pool_embedding_model == "m-1"
        assert rows[theirs].pool_embedding is None
        loaded = await mgr.load_pool_vectors(
            ids=[mine, theirs],
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
        )
        assert set(loaded) == {mine} and loaded[mine][1] == "m-1"

        class Lost(Exception):
            pass

        async def lost(db):
            raise Lost()

        with pytest.raises(Lost):
            await mgr.store_pool_vectors(
                {mine: [0.0, 1.0]},
                model="m-2",
                agent_id=owner.agent,
                user_id=owner.user,
                organization_id=owner.org,
                write_guard=lost,
            )
        rows = await _pool_rows(session_maker, [mine])
        assert rows[mine].pool_embedding_model == "m-1"  # nothing written

    async def test_defaults_and_to_pydantic_never_load_the_vector(self, session_maker):
        owner = await _pool_owner(session_maker)
        mgr = _manager(session_maker)
        created = await mgr.create_experience(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            **_exp_kwargs("defaults"),
        )
        assert created.recurrence_count == 1
        assert created.ignored_pushes == 0 and created.pushed_count == 0
        assert created.disposition is None and created.cited_skill_ids is None
        assert created.pool_scope is None and created.pool_retired_at is None
        assert "pool_embedding" not in type(created).model_fields
        # An ORM row loaded normally leaves the vector unloaded (deferred), and
        # to_pydantic works on it without a lazy load (async-safe).
        from sqlalchemy import inspect as sa_inspect, select

        async with session_maker() as session:
            row = (
                await session.execute(
                    select(SkillExperienceORM).where(SkillExperienceORM.id == created.id)
                )
            ).scalar_one()
            assert "pool_embedding" in sa_inspect(row).unloaded
            assert row.to_pydantic().id == created.id

    async def test_consume_records_every_disposition(self, session_maker):
        from mirix.services.skill_experience_manager import BATCH_CONSUME, PoolSettlement

        owner = await _pool_owner(session_maker)
        cited = await _pool_row(
            session_maker, owner, seconds=1, status="pending", disposition=None,
            cited=["proc-earlier"],  # cited by an earlier, returned attempt
        )
        covered = await _pool_row(session_maker, owner, seconds=2, status="pending", disposition=None)
        not_worth = await _pool_row(session_maker, owner, seconds=3, status="pending", disposition=None)
        deferred = await _pool_row(session_maker, owner, seconds=4, status="pending", disposition=None)
        untriaged = await _pool_row(session_maker, owner, seconds=5, status="pending", disposition=None)
        ids = [cited, covered, not_worth, deferred, untriaged]
        pool = PoolSettlement(
            scope="scope-a",
            triage={
                cited: "not_worth",  # a citation wins over the triage
                covered: "covered",
                not_worth: "not_worth",
                deferred: "deferred",
            },
        )
        result = await _settle(session_maker, owner, ids, BATCH_CONSUME, pool)
        assert result["consumed"] == ids
        rows = await _pool_rows(session_maker, ids)
        assert rows[cited].disposition == "used" and rows[cited].pool_scope is None
        assert rows[covered].disposition == "covered" and rows[covered].pool_scope is None
        assert rows[not_worth].disposition == "not_worth"
        assert rows[not_worth].ignored_pushes == 1
        assert rows[not_worth].pool_scope == "scope-a"
        for experience_id in (deferred, untriaged):
            assert rows[experience_id].disposition == "deferred"
            assert rows[experience_id].ignored_pushes == 0
            assert rows[experience_id].pool_scope == "scope-a"
        summary = result["pool"]
        assert summary["dispositions"] == {
            "used": 1,
            "covered": 1,
            "not_worth": 1,
            "deferred": 2,
            "defaulted": 1,
            "unaddressed": 0,
        }
        assert summary["pool_active_after"] == 3  # not_worth + 2 deferred
        # They are the pool now.
        candidates, _ = await _manager(session_maker).list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=10,
        )
        assert [c.id for c in candidates] == [not_worth, deferred, untriaged]

    async def test_consume_keeps_a_citation_without_a_quote_pending(
        self, session_maker
    ):
        """An experience a write cited only without a quote that shows
        where it applies it is not consumed by the batch: it stays pending
        (an attempt counted, closed out at max_attempts); an earlier lesson
        cited so is neither used nor ignored. A triage or a verified citation
        still decides."""
        from sqlalchemy import update

        from mirix.services.skill_experience_manager import BATCH_CONSUME, PoolSettlement

        owner = await _pool_owner(session_maker)
        pending = dict(status="pending", disposition=None)
        unquoted = await _pool_row(session_maker, owner, seconds=1, **pending)
        triaged = await _pool_row(session_maker, owner, seconds=2, **pending)
        also_cited = await _pool_row(
            session_maker, owner, seconds=3, cited=["proc-q"], **pending
        )
        last_try = await _pool_row(session_maker, owner, seconds=4, **pending)
        untouched = await _pool_row(session_maker, owner, seconds=5, **pending)
        lesson = await _pool_row(session_maker, owner, seconds=6, ignored=1)
        async with session_maker() as session:
            await session.execute(
                update(SkillExperienceORM)
                .where(SkillExperienceORM.id == last_try)
                .values(curation_attempts=2)
            )
            await session.commit()
        ids = [unquoted, triaged, also_cited, last_try, untouched]
        pool = PoolSettlement(
            scope="scope-a",
            pushed_ids=(lesson,),
            triage={triaged: "not_worth"},
            unquoted_ids=(unquoted, triaged, also_cited, last_try, lesson),
        )
        result = await _settle(session_maker, owner, ids, BATCH_CONSUME, pool)
        assert result["consumed"] == [triaged, also_cited, untouched]
        assert result["returned"] == [unquoted]
        assert result["unaddressed"] == [last_try]
        rows = await _pool_rows(session_maker, ids + [lesson])
        assert rows[unquoted].status == "pending"
        assert rows[unquoted].curation_attempts == 1
        assert rows[unquoted].disposition is None and rows[unquoted].pool_scope is None
        assert rows[triaged].disposition == "not_worth"  # the triage decides
        assert rows[also_cited].disposition == "used"  # a verified citation
        assert rows[last_try].status == "consumed"
        assert rows[last_try].disposition == "unaddressed"
        assert rows[untouched].disposition == "deferred"  # neither: as before
        # The lesson stays as it was: pushed, but not ignored.
        assert rows[lesson].ignored_pushes == 1 and rows[lesson].pool_retired_at is None
        assert rows[lesson].pushed_count == 1
        summary = result["pool"]
        assert summary["history_ignored"] == 0
        assert summary["dispositions"]["defaulted"] == 1
        assert summary["dispositions"]["unaddressed"] == 1

    async def test_consume_decays_retires_and_counts_the_shown_lessons(
        self, session_maker
    ):
        import datetime as dt

        from mirix.services.skill_experience_manager import BATCH_CONSUME, PoolSettlement

        owner = await _pool_owner(session_maker)
        ignored = await _pool_row(session_maker, owner, seconds=1)
        not_worth = await _pool_row(session_maker, owner, seconds=2)
        covered = await _pool_row(session_maker, owner, seconds=3)
        cited = await _pool_row(session_maker, owner, seconds=4, cited=["proc-now"])
        worn_out = await _pool_row(session_maker, owner, seconds=5, ignored=2, pushed=2)
        deferred = await _pool_row(session_maker, owner, seconds=6)
        recurred = await _pool_row(session_maker, owner, seconds=7, ignored=2, recurrence=1)
        recurred_unshown = await _pool_row(session_maker, owner, seconds=8, ignored=1, recurrence=3)
        untouched = await _pool_row(session_maker, owner, seconds=9, ignored=1)
        new = await _pool_row(session_maker, owner, seconds=10, status="pending", disposition=None)
        now = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
        pool = PoolSettlement(
            scope="scope-a",
            pushed_ids=(ignored, not_worth, covered, cited, worn_out, deferred, recurred),
            recurred_ids=(recurred, recurred_unshown),
            triage={not_worth: "not_worth", covered: "covered", deferred: "deferred"},
        )
        result = await _settle(session_maker, owner, [new], BATCH_CONSUME, pool, now=now)
        rows = await _pool_rows(
            session_maker,
            [ignored, not_worth, covered, cited, worn_out, deferred, recurred,
             recurred_unshown, untouched],
        )
        assert rows[ignored].ignored_pushes == 1 and rows[ignored].pool_retired_at is None
        assert rows[not_worth].ignored_pushes == 2 and rows[not_worth].pool_retired_at is None
        assert rows[covered].pool_retire_reason == "covered"
        assert rows[cited].pool_retire_reason == "used"
        assert rows[worn_out].ignored_pushes == 3
        assert rows[worn_out].pool_retire_reason == "ignored"
        assert rows[deferred].ignored_pushes == 1  # listed deferred: +1
        # Recurrence first (ignored 2 -> 0), then the push (+1).
        assert rows[recurred].recurrence_count == 2 and rows[recurred].ignored_pushes == 1
        assert rows[recurred_unshown].recurrence_count == 4
        assert rows[recurred_unshown].ignored_pushes == 0
        assert rows[recurred_unshown].pushed_count == 0
        for pushed in (ignored, not_worth, covered, cited, deferred, recurred):
            assert rows[pushed].pushed_count == 1
            assert rows[pushed].last_pushed_at is not None
        assert rows[worn_out].pushed_count == 3
        assert rows[untouched].ignored_pushes == 1 and rows[untouched].pushed_count == 0
        summary = result["pool"]
        assert summary["history_cited"] == 1 and summary["history_covered"] == 1
        assert summary["history_not_worth"] == 1
        assert summary["history_ignored"] == 4  # ignored, worn_out, deferred, recurred
        assert summary["recurrences"] == 2
        assert summary["retired"] == {
            "used": 1,
            "covered": 1,
            "ignored": 1,
            "merged": 0,
            "unembeddable": 0,
        }
        assert summary["retired_ignored"] == 1
        # ignored, not_worth, deferred, recurred, recurred_unshown, untouched,
        # and the new experience (deferred by default).
        assert summary["pool_active_after"] == 7

    async def test_a_lesson_that_keeps_coming_up_is_not_retired_for_being_ignored(
        self, session_maker
    ):
        """Recurrence resets the ignore count before the batch's pushes
        are counted, so a lesson that recurs in every batch and is
        ignored every time stays in the pool, while one that does not recur
        retires at its third ignore. The prompt and the batch header say so."""
        from mirix.services.skill_experience_manager import (
            BATCH_CONSUME,
            PoolSettlement,
        )

        owner = await _pool_owner(session_maker)
        recurring = await _pool_row(session_maker, owner, seconds=1)
        quiet = await _pool_row(session_maker, owner, seconds=2)
        for index in range(3):
            new = await _pool_row(
                session_maker,
                owner,
                seconds=10 + index,
                status="pending",
                disposition=None,
            )
            await _settle(
                session_maker,
                owner,
                [new],
                BATCH_CONSUME,
                PoolSettlement(
                    scope="scope-a",
                    pushed_ids=(recurring, quiet),
                    recurred_ids=(recurring,),
                ),
                batch_id=f"r:b{index}",
            )
            rows = await _pool_rows(session_maker, [recurring, quiet])
            assert rows[recurring].ignored_pushes == 1
            assert rows[quiet].ignored_pushes == index + 1
        rows = await _pool_rows(session_maker, [recurring, quiet])
        assert rows[recurring].pool_retired_at is None
        assert (
            rows[recurring].recurrence_count == 4 and rows[recurring].pushed_count == 3
        )
        assert rows[quiet].pool_retire_reason == "ignored"

    async def test_a_returned_batch_leaves_the_pool_but_retires_cited_lessons(
        self, session_maker
    ):
        from mirix.services.skill_experience_manager import (
            BATCH_KEEP,
            BATCH_RETURN,
            PoolSettlement,
        )

        owner = await _pool_owner(session_maker)
        shown = await _pool_row(session_maker, owner, seconds=1, ignored=1)
        cited = await _pool_row(session_maker, owner, seconds=2, cited=["proc-y"])
        recurred = await _pool_row(session_maker, owner, seconds=3, ignored=2)
        new = await _pool_row(session_maker, owner, seconds=4, status="pending", disposition=None)
        pool = PoolSettlement(
            scope="scope-a",
            pushed_ids=(shown, cited),
            recurred_ids=(recurred,),
            triage={shown: "covered"},
        )
        result = await _settle(session_maker, owner, [new], BATCH_RETURN, pool)
        rows = await _pool_rows(session_maker, [shown, cited, recurred, new])
        assert result["returned"] == [new]
        assert rows[new].status == "pending" and rows[new].disposition is None
        assert rows[shown].ignored_pushes == 1 and rows[shown].pushed_count == 0
        assert rows[shown].pool_retired_at is None  # triaged covered, but returned
        assert rows[recurred].recurrence_count == 1 and rows[recurred].ignored_pushes == 2
        assert rows[cited].pool_retire_reason == "used"  # a citation is a fact
        assert result["pool"]["retired"]["used"] == 1
        assert result["pool"]["recurrences"] == 0

        # keep: only the citation retirement (nothing left to retire here).
        again = await _pool_row(session_maker, owner, seconds=5, cited=["proc-z"])
        result = await _settle(
            session_maker,
            owner,
            [new],
            BATCH_KEEP,
            PoolSettlement(scope="scope-a", pushed_ids=(shown, again)),
            batch_id="r:b1",
        )
        rows = await _pool_rows(session_maker, [shown, again, new])
        assert rows[again].pool_retire_reason == "used"
        assert rows[shown].pool_retired_at is None and rows[shown].ignored_pushes == 1
        assert rows[new].curation_attempts == 1  # keep: unchanged
        assert result["pool"]["retired"] == {
            "used": 1,
            "covered": 0,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        }

    async def test_a_closed_out_experience_is_unaddressed_and_never_pools(
        self, session_maker
    ):
        from mirix.services.skill_experience_manager import BATCH_RETURN, PoolSettlement

        owner = await _pool_owner(session_maker)
        new = await _pool_row(session_maker, owner, seconds=1, status="pending", disposition=None)
        for i in range(3):
            result = await _settle(
                session_maker,
                owner,
                [new],
                BATCH_RETURN,
                PoolSettlement(scope="scope-a"),
                batch_id=f"r{i}:b0",
            )
        assert result["unaddressed"] == [new]
        assert result["pool"]["dispositions"]["unaddressed"] == 1
        rows = await _pool_rows(session_maker, [new])
        assert rows[new].disposition == "unaddressed" and rows[new].pool_scope is None
        candidates, _ = await _manager(session_maker).list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=10,
        )
        assert candidates == []

    async def test_a_closed_out_experience_a_write_cited_is_used(self, session_maker):
        """A skill write of an earlier (returned) attempt cited it and
        committed; closing it out at the third return does not undo that."""
        from mirix.services.skill_experience_manager import BATCH_RETURN, PoolSettlement

        owner = await _pool_owner(session_maker)
        cited = await _pool_row(
            session_maker, owner, seconds=1, status="pending", disposition=None,
            cited=["proc-attempt-1"],
        )
        plain = await _pool_row(session_maker, owner, seconds=2, status="pending", disposition=None)
        for i in range(3):
            result = await _settle(
                session_maker,
                owner,
                [cited, plain],
                BATCH_RETURN,
                PoolSettlement(scope="scope-a"),
                batch_id=f"f5-{i}:b0",
            )
        assert result["unaddressed"] == [cited, plain]  # both closed out
        assert result["pool"]["dispositions"]["used"] == 1
        assert result["pool"]["dispositions"]["unaddressed"] == 1
        rows = await _pool_rows(session_maker, [cited, plain])
        assert rows[cited].disposition == "used" and rows[cited].status == "consumed"
        assert rows[cited].consumed_by.endswith(":unaddressed")
        assert rows[plain].disposition == "unaddressed"
        assert rows[cited].pool_scope is None and rows[plain].pool_scope is None

    async def test_a_new_experience_that_recurred_a_lesson_in_the_pool_is_merged_into_it(
        self, session_maker
    ):
        """One lesson stays one pool row. A new experience that
        recurred a pool lesson still active after the settle is retired
        ``merged`` as it enters (the lesson counted it); one whose lesson retired
        in this settle, or that recurred nothing, stays."""
        from mirix.services.skill_experience_manager import BATCH_CONSUME, PoolSettlement

        owner = await _pool_owner(session_maker)
        lesson = await _pool_row(session_maker, owner, seconds=1, ignored=2)
        used_lesson = await _pool_row(session_maker, owner, seconds=2, cited=["proc-u"])
        foreign = await _pool_row(session_maker, owner, seconds=3, scope="scope-b")
        again = await _pool_row(session_maker, owner, seconds=4, status="pending", disposition=None)
        not_worth_again = await _pool_row(
            session_maker, owner, seconds=5, status="pending", disposition=None
        )
        after_used = await _pool_row(
            session_maker, owner, seconds=6, status="pending", disposition=None
        )
        after_foreign = await _pool_row(
            session_maker, owner, seconds=7, status="pending", disposition=None
        )
        covered_again = await _pool_row(
            session_maker, owner, seconds=8, status="pending", disposition=None
        )
        fresh = await _pool_row(session_maker, owner, seconds=9, status="pending", disposition=None)
        ids = [again, not_worth_again, after_used, after_foreign, covered_again, fresh]
        pool = PoolSettlement(
            scope="scope-a",
            pushed_ids=(lesson, used_lesson),
            recurred_ids=(lesson, used_lesson, foreign),
            recurred_by={
                again: (lesson,),
                not_worth_again: (lesson,),
                after_used: (used_lesson,),
                after_foreign: (foreign,),
                covered_again: (lesson,),
            },
            triage={not_worth_again: "not_worth", covered_again: "covered"},
        )
        result = await _settle(session_maker, owner, ids, BATCH_CONSUME, pool)
        rows = await _pool_rows(session_maker, ids + [lesson, used_lesson, foreign])
        # The lesson counted both recurrences of this batch once (a batch's
        # recurrence), was shown and ignored (+1 after the reset).
        assert rows[lesson].recurrence_count == 2 and rows[lesson].ignored_pushes == 1
        assert rows[lesson].pool_retired_at is None
        for merged in (again, not_worth_again):
            assert rows[merged].disposition in ("deferred", "not_worth")
            assert rows[merged].pool_retire_reason == "merged"
            assert rows[merged].pool_scope == "scope-a"
        assert rows[used_lesson].pool_retire_reason == "used"
        assert rows[after_used].pool_retired_at is None  # its lesson retired
        assert rows[after_foreign].pool_retired_at is None  # another scope's
        assert rows[covered_again].disposition == "covered"
        assert rows[covered_again].pool_retired_at is None
        assert rows[fresh].pool_retired_at is None and rows[fresh].disposition == "deferred"
        summary = result["pool"]
        assert summary["retired"]["merged"] == 2 and summary["retired"]["used"] == 1
        # lesson, after_used, after_foreign, fresh: entered 5 (4 deferred + 1 not
        # worth), 2 merged; with the lesson that stays.
        assert summary["dispositions"]["deferred"] == 4
        assert summary["dispositions"]["not_worth"] == 1
        assert summary["pool_active_after"] == 4
        candidates, _ = await _manager(session_maker).list_active_pool(
            agent_id=owner.agent,
            user_id=owner.user,
            organization_id=owner.org,
            scope="scope-a",
            limit=10,
        )
        assert [c.id for c in candidates] == [lesson, after_used, after_foreign, fresh]

    async def test_a_returned_batch_merges_nothing(self, session_maker):
        from mirix.services.skill_experience_manager import BATCH_RETURN, PoolSettlement

        owner = await _pool_owner(session_maker)
        lesson = await _pool_row(session_maker, owner, seconds=1)
        again = await _pool_row(session_maker, owner, seconds=2, status="pending", disposition=None)
        result = await _settle(
            session_maker,
            owner,
            [again],
            BATCH_RETURN,
            PoolSettlement(
                scope="scope-a", recurred_ids=(lesson,), recurred_by={again: (lesson,)}
            ),
        )
        rows = await _pool_rows(session_maker, [lesson, again])
        assert rows[again].status == "pending" and rows[again].pool_retired_at is None
        assert rows[lesson].recurrence_count == 1
        assert result["pool"]["retired"]["merged"] == 0

    async def test_a_lesson_whose_embedding_keeps_failing_retires_unembeddable(
        self, session_maker
    ):
        """A pool experience the embedding model refuses in three
        finished batches leaves the pool instead of being retried in every
        later batch. A returned or kept batch does not count."""
        from mirix.constants import SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES
        from mirix.services.skill_experience_manager import (
            BATCH_CONSUME,
            BATCH_KEEP,
            BATCH_RETURN,
            PoolSettlement,
        )

        assert SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES == 3
        owner = await _pool_owner(session_maker)
        poison = await _pool_row(session_maker, owner, seconds=1)
        foreign = await _pool_row(session_maker, owner, seconds=2, scope="scope-b")
        pool = PoolSettlement(scope="scope-a", embedding_failed_ids=(poison, foreign))
        for outcome in (BATCH_RETURN, BATCH_KEEP):
            new = await _pool_row(
                session_maker, owner, seconds=10, status="pending", disposition=None
            )
            result = await _settle(
                session_maker, owner, [new], outcome, pool, batch_id=f"f1-{outcome}:b0"
            )
            assert result["pool"]["embedding_failures"] == 0
        assert (await _pool_rows(session_maker, [poison]))[poison].pool_embedding_failures == 0
        for i in range(3):
            new = await _pool_row(
                session_maker, owner, seconds=20 + i, status="pending", disposition=None
            )
            result = await _settle(
                session_maker, owner, [new], BATCH_CONSUME, pool, batch_id=f"f1-{i}:b0"
            )
            assert result["pool"]["embedding_failures"] == 1
            row = (await _pool_rows(session_maker, [poison]))[poison]
            assert row.pool_embedding_failures == i + 1
            if i < 2:
                assert row.pool_retired_at is None
        assert row.pool_retire_reason == "unembeddable"
        assert result["pool"]["retired"]["unembeddable"] == 1
        other = (await _pool_rows(session_maker, [foreign]))[foreign]
        assert other.pool_embedding_failures == 0 and other.pool_retired_at is None

    async def test_a_large_scan_parses_its_vectors_off_the_event_loop(
        self, session_maker, monkeypatch
    ):
        """Up to MIRIX_SKILL_EXPERIENCE_POOL_SCAN_LIMIT vectors of
        4,096 dimensions are parsed by the column type's own processor in a
        worker thread, never on the event loop; a small scan stays inline."""
        import threading

        from mirix.services import skill_experience_manager as sem

        owner = await _pool_owner(session_maker)
        ids = [
            await _pool_row(session_maker, owner, seconds=i, vector=[float(i + 1), 1.0], model="m")
            for i in range(4)
        ]
        await _pool_row(session_maker, owner, seconds=9)  # no vector yet
        threads = []
        real = sem._parse_vectors

        def spy(raw_values, process):
            threads.append(threading.get_ident())
            return real(raw_values, process)

        monkeypatch.setattr(sem, "_parse_vectors", spy)
        loop_thread = threading.get_ident()
        mgr = _manager(session_maker)

        async def scan():
            return await mgr.list_active_pool(
                agent_id=owner.agent,
                user_id=owner.user,
                organization_id=owner.org,
                scope="scope-a",
                limit=10,
            )

        small, _ = await scan()
        monkeypatch.setattr(sem, "_VECTOR_PARSE_THREAD_THRESHOLD", 3)
        large, _ = await scan()
        assert threads[0] == loop_thread and threads[1] != loop_thread
        for candidates in (small, large):
            assert [c.id for c in candidates[:4]] == ids
            for i, candidate in enumerate(candidates[:4]):
                assert list(candidate.vector[:2]) == [float(i + 1), 1.0]
                assert candidate.vector.dtype == np.float32
            assert candidates[4].vector is None

    async def test_foreign_ids_are_never_touched(self, session_maker):
        from mirix.services.skill_experience_manager import BATCH_CONSUME, PoolSettlement

        owner = await _pool_owner(session_maker)
        stranger = await _pool_owner(session_maker, org_id=owner.org)
        foreign = await _pool_row(session_maker, stranger, seconds=1, cited=["proc-f"])
        other_scope = await _pool_row(session_maker, owner, seconds=2, scope="scope-b")
        new = await _pool_row(session_maker, owner, seconds=3, status="pending", disposition=None)
        foreign_pending = await _pool_row(
            session_maker, stranger, seconds=4, status="pending", disposition=None
        )
        result = await _settle(
            session_maker,
            owner,
            [new, foreign_pending],
            BATCH_CONSUME,
            PoolSettlement(
                scope="scope-a",
                pushed_ids=(foreign, other_scope),
                recurred_ids=(foreign, other_scope),
                triage={foreign: "covered", other_scope: "covered", foreign_pending: "covered"},
            ),
        )
        assert result["consumed"] == [new]
        rows = await _pool_rows(session_maker, [foreign, other_scope, foreign_pending])
        for row in rows.values():
            assert row.pool_retired_at is None and row.pushed_count == 0
            assert row.recurrence_count == 1
        assert rows[foreign_pending].status == "pending"
        assert result["pool"]["retired"] == {
            "used": 0,
            "covered": 0,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        }

    async def test_without_a_pool_the_settle_keeps_its_shape_and_records_dispositions(
        self, session_maker
    ):
        from mirix.services.skill_experience_manager import BATCH_CONSUME

        owner = await _pool_owner(session_maker)
        cited = await _pool_row(
            session_maker, owner, seconds=1, status="pending", disposition=None,
            cited=["proc-c"],
        )
        plain = await _pool_row(session_maker, owner, seconds=2, status="pending", disposition=None)
        result = await _settle(session_maker, owner, [cited, plain], BATCH_CONSUME, None)
        assert result == {"consumed": [cited, plain], "returned": [], "unaddressed": []}
        rows = await _pool_rows(session_maker, [cited, plain])
        assert rows[cited].disposition == "used"
        assert rows[plain].disposition == "deferred" and rows[plain].pool_scope is None

    async def test_an_unknown_outcome_is_refused(self, session_maker):
        owner = await _pool_owner(session_maker)
        with pytest.raises(ValueError):
            await _settle(session_maker, owner, [], "later", None)
