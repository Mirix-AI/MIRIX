"""Tests for the skill-evolution record store (C2).

Three layers, mirroring the existing skill test suites:

1. Pydantic schema validation (DB-free) — rejects bad record_type/status,
   enforces length caps, defaults status='pending'.
2. ORM shape (DB-free) — table name, inherited mixin columns present,
   explicit columns present, id prefix, indexes, ORM package registration.
3. Manager behavior (DB-backed) — runs against the default SQLite engine
   (no Postgres needed: the manager uses plain SELECT/UPDATE, no FOR UPDATE,
   so SQLite is faithful here). Covers record_round_result, list_pending
   watermark + failures-first ordering, mark_consumed / mark_superseded
   transitions, and the aggregate structural gate.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import inspect as sa_inspect

from mirix.orm.skill_evolution_record import (
    SkillEvolutionRecord as SkillEvolutionRecordORM,
)
from mirix.schemas.skill_evolution_record import (
    SKILL_RECORD_MAX_DESCRIPTION_LEN,
    SKILL_RECORD_MAX_DETAIL_LEN,
    SKILL_RECORD_MAX_TITLE_LEN,
    SkillEvolutionRecord as PydanticSkillEvolutionRecord,
    SkillEvolutionRecordBase,
    SkillEvolutionRecordCreate,
    SkillEvolutionRecordResponse,
    SkillEvolutionRecordUpdate,
)
from mirix.services.skill_evolution_record_manager import SkillEvolutionRecordManager


# ============================ Schema validation ===========================


class TestSkillEvolutionRecordSchema:
    """Pydantic validation: record_type/status enums, length caps, defaults."""

    def _valid_kwargs(self, **overrides) -> dict:
        base = dict(
            day="day03",
            round_id="r4",
            round_index=4,
            record_type="failure",
            title="missed the box format",
            description="answer was right but not boxed",
            detail="root_cause: forgot \\boxed{}; what_to_avoid: always box",
            evidence_round_ids=["r3", "r4"],
            quality_score=0.7,
            generality=0.5,
        )
        base.update(overrides)
        return base

    def test_valid_base_record(self):
        rec = SkillEvolutionRecordBase(**self._valid_kwargs())
        assert rec.record_type == "failure"
        assert rec.round_index == 4
        assert rec.evidence_round_ids == ["r3", "r4"]

    def test_record_type_accepts_success(self):
        rec = SkillEvolutionRecordBase(**self._valid_kwargs(record_type="success"))
        assert rec.record_type == "success"

    def test_record_type_rejects_unknown(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordBase(**self._valid_kwargs(record_type="partial"))

    def test_status_defaults_to_pending(self):
        rec = SkillEvolutionRecordBase(**self._valid_kwargs())
        assert rec.status == "pending"

    def test_status_accepts_known_values(self):
        for status in ("pending", "consumed", "superseded"):
            rec = SkillEvolutionRecordBase(**self._valid_kwargs(status=status))
            assert rec.status == status

    def test_status_rejects_unknown(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordBase(**self._valid_kwargs(status="archived"))

    def test_title_length_cap(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordBase(
                **self._valid_kwargs(title="x" * (SKILL_RECORD_MAX_TITLE_LEN + 1))
            )

    def test_description_length_cap(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordBase(
                **self._valid_kwargs(
                    description="x" * (SKILL_RECORD_MAX_DESCRIPTION_LEN + 1)
                )
            )

    def test_detail_length_cap(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordBase(
                **self._valid_kwargs(detail="x" * (SKILL_RECORD_MAX_DETAIL_LEN + 1))
            )

    def test_evidence_round_ids_defaults_to_empty(self):
        kwargs = self._valid_kwargs()
        kwargs.pop("evidence_round_ids")
        rec = SkillEvolutionRecordBase(**kwargs)
        assert rec.evidence_round_ids == []

    def test_full_schema_carries_db_fields(self):
        rec = PydanticSkillEvolutionRecord(
            **self._valid_kwargs(),
            id="sevr-abc12345",
            agent_id="agent-1",
            user_id="user-1",
            organization_id="org-1",
        )
        assert rec.id == "sevr-abc12345"
        assert rec.agent_id == "agent-1"
        assert rec.consumed_by is None
        assert rec.influenced_skill_ids is None

    def test_create_schema_requires_owners(self):
        # Create schema must carry the owner ids needed to persist the row.
        rec = SkillEvolutionRecordCreate(
            **self._valid_kwargs(),
            agent_id="agent-1",
            user_id="user-1",
            organization_id="org-1",
        )
        assert rec.agent_id == "agent-1"

    def test_update_schema_validates_status(self):
        with pytest.raises(ValidationError):
            SkillEvolutionRecordUpdate(id="sevr-abc12345", status="bogus")

    def test_update_schema_accepts_consumed(self):
        upd = SkillEvolutionRecordUpdate(
            id="sevr-abc12345", status="consumed", consumed_by="run-9"
        )
        assert upd.status == "consumed"
        assert upd.consumed_by == "run-9"

    def test_response_is_subclass(self):
        assert issubclass(SkillEvolutionRecordResponse, PydanticSkillEvolutionRecord)

    def test_id_prefix(self):
        assert SkillEvolutionRecordBase.__id_prefix__ == "sevr"


# ================================ ORM shape ===============================


class TestSkillEvolutionRecordORM:
    def _column_names(self):
        mapper = sa_inspect(SkillEvolutionRecordORM)
        return {col.key for col in mapper.columns}

    def test_table_name(self):
        assert SkillEvolutionRecordORM.__tablename__ == "skill_evolution_record"

    def test_inherits_mixin_columns(self):
        cols = self._column_names()
        for expected in ["id", "organization_id", "user_id", "agent_id", "created_at"]:
            assert expected in cols, f"missing inherited column: {expected}"

    def test_has_explicit_columns(self):
        cols = self._column_names()
        for expected in [
            "day",
            "round_id",
            "round_index",
            "record_type",
            "title",
            "description",
            "detail",
            "evidence_round_ids",
            "quality_score",
            "generality",
            "status",
            "consumed_by",
            "influenced_skill_ids",
        ]:
            assert expected in cols, f"missing explicit column: {expected}"

    def test_id_default_uses_sevr_prefix(self):
        # The PK default is a callable applied at flush; assert it mints a
        # sevr-prefixed id (mirrors `proc-` / `ats-` on sibling tables).
        default = SkillEvolutionRecordORM.__table__.c.id.default
        assert default is not None
        minted = default.arg(None)
        assert minted.startswith("sevr-")

    def test_status_defaults_to_pending(self):
        # status defaults to 'pending' at the column level.
        assert SkillEvolutionRecordORM.__table__.c.status.default.arg == "pending"

    def test_pydantic_model_link(self):
        assert (
            SkillEvolutionRecordORM.__pydantic_model__ is PydanticSkillEvolutionRecord
        )

    def test_has_status_index(self):
        idx_names = {i.name for i in SkillEvolutionRecordORM.__table__.indexes}
        assert "ix_skill_evolution_record_agent_status" in idx_names

    def test_is_registered_in_orm_package(self):
        import mirix.orm as orm_pkg

        assert "SkillEvolutionRecord" in orm_pkg.__all__
        assert orm_pkg.SkillEvolutionRecord is SkillEvolutionRecordORM


# ============================ Manager (DB-backed) ==========================


# The DB-backed manager tests run on a hermetic, throwaway SQLite file rather
# than the process-global engine. The global engine binds to ~/.mirix/sqlite.db
# at import time, which on a dev box is often a stale schema (e.g. missing
# columns added since it was created). A fresh engine + create_all gives a
# correct schema every run with zero Postgres dependency. The manager reads its
# sessions from `self.session_maker`, so we just point that at our test
# sessionmaker — no monkeypatching of module globals required.
#
# This is faithful to the manager's real behavior: it uses plain SELECT/UPDATE
# (no SELECT ... FOR UPDATE), so SQLite is a true substitute here.


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_maker(tmp_path_factory):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import mirix.orm  # noqa: F401 -- ensure all ORM classes are registered
    from mirix.orm.base import Base

    db_path = tmp_path_factory.mktemp("sevr") / "test.db"
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

    org_id = f"sevr-org-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        await session.commit()
    return type("Org", (), {"id": org_id})()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def user_a(session_maker, org):
    from mirix.orm.user import User as UserORM

    uid = f"sevr-user-{uuid.uuid4().hex[:8]}"
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


def _manager(session_maker):
    """Build a manager wired to the hermetic test sessionmaker."""
    mgr = SkillEvolutionRecordManager()
    mgr.session_maker = session_maker
    return mgr


@pytest_asyncio.fixture(loop_scope="module")
async def agent_id(session_maker, org):
    return await _insert_minimal_agent(session_maker, org.id)


@pytest_asyncio.fixture(loop_scope="module")
async def other_agent_id(session_maker, org):
    return await _insert_minimal_agent(session_maker, org.id)


def _record_kwargs(round_index, record_type="failure", **overrides):
    base = dict(
        day="day01",
        round_id=f"r{round_index}",
        round_index=round_index,
        record_type=record_type,
        title=f"title-{round_index}",
        description=f"desc-{round_index}",
        detail=f"detail-{round_index}",
        evidence_round_ids=[f"r{round_index - 1}", f"r{round_index}"],
        quality_score=0.6,
        generality=0.5,
    )
    base.update(overrides)
    return base


@pytest.mark.asyncio(loop_scope="module")
class TestRecordRoundResult:
    async def test_insert_and_read_back(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        rec = await mgr.record_round_result(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            **_record_kwargs(2, record_type="success"),
        )
        assert rec.id.startswith("sevr-")
        assert rec.record_type == "success"
        assert rec.status == "pending"
        assert rec.consumed_by is None
        assert rec.influenced_skill_ids is None
        assert rec.round_index == 2


@pytest.mark.asyncio(loop_scope="module")
class TestListPending:
    async def test_watermark_excludes_at_or_above(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        # rounds 5,6,7 — only those strictly below the watermark should return.
        for ri in (5, 6, 7):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))

        pending = await mgr.list_pending(agent_id=agent_id, before_round_index=7)
        indices = {r.round_index for r in pending}
        # round 7 (== watermark) and anything >= 7 excluded; <7 included.
        assert 7 not in indices
        assert {5, 6}.issubset(indices)

    async def test_no_watermark_returns_all_pending(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        for ri in (1, 2, 3):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))
        # With no watermark, every pending record for this agent is returned.
        pending = await mgr.list_pending(agent_id=agent_id)
        assert len(pending) == 3
        assert all(r.status == "pending" for r in pending)

    async def test_failures_first_then_round_index(
        self, session_maker, other_agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=other_agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        # Insert in mixed order: a success at round 2, failures at 1 and 3.
        await mgr.record_round_result(
            **owner, **_record_kwargs(2, record_type="success")
        )
        await mgr.record_round_result(
            **owner, **_record_kwargs(3, record_type="failure")
        )
        await mgr.record_round_result(
            **owner, **_record_kwargs(1, record_type="failure")
        )

        pending = await mgr.list_pending(agent_id=other_agent_id)
        types = [r.record_type for r in pending]
        # Failures must precede successes.
        assert types == sorted(types, key=lambda t: 0 if t == "failure" else 1)
        # Within failures, ascending round_index.
        failures = [r.round_index for r in pending if r.record_type == "failure"]
        assert failures == sorted(failures)

    async def test_respects_limit(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
        )
        for ri in (1, 2, 3, 4):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))
        pending = await mgr.list_pending(agent_id=agent_id, limit=2)
        assert len(pending) == 2

    async def test_isolated_per_agent(
        self, session_maker, agent_id, other_agent_id, user_a
    ):
        mgr = _manager(session_maker)
        # Records for `agent_id` must NOT surface under `other_agent_id`.
        await mgr.record_round_result(
            agent_id=agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            **_record_kwargs(1),
        )
        await mgr.record_round_result(
            agent_id=other_agent_id,
            user_id=user_a.id,
            organization_id=user_a.organization_id,
            **_record_kwargs(1),
        )
        pending = await mgr.list_pending(agent_id=other_agent_id)
        assert len(pending) == 1
        assert all(r.agent_id == other_agent_id for r in pending)


@pytest.mark.asyncio(loop_scope="module")
class TestMarkTransitions:
    async def test_mark_consumed(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_minimal_agent(session_maker, org.id)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        r1 = await mgr.record_round_result(**owner, **_record_kwargs(1))
        r2 = await mgr.record_round_result(**owner, **_record_kwargs(2))

        n = await mgr.mark_consumed(ids=[r1.id, r2.id], run_id="run-42")
        assert n == 2

        # Consumed records no longer appear in pending; consumed_by recorded.
        pending = await mgr.list_pending(agent_id=agent_id)
        assert pending == []
        agg = await mgr.aggregate(ids=[r1.id, r2.id])  # still queryable
        assert agg["n"] == 2

    async def test_mark_consumed_only_flips_pending(self, session_maker, org, user_a):
        # A second consume must NOT clobber an existing consumed_by: only
        # still-pending rows transition.
        mgr = _manager(session_maker)
        agent_id = await _insert_minimal_agent(session_maker, org.id)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        r1 = await mgr.record_round_result(**owner, **_record_kwargs(1))

        first = await mgr.mark_consumed(ids=[r1.id], run_id="run-A")
        assert first == 1
        # Re-consume the same (already-consumed) id: no-op, count 0.
        second = await mgr.mark_consumed(ids=[r1.id], run_id="run-B")
        assert second == 0

    async def test_mark_superseded(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_minimal_agent(session_maker, org.id)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        r1 = await mgr.record_round_result(**owner, **_record_kwargs(1))

        n = await mgr.mark_superseded(ids=[r1.id])
        assert n == 1

        # Excluded from pending...
        pending = await mgr.list_pending(agent_id=agent_id)
        assert pending == []
        # ...but still queryable with status flipped (NOT is_deleted): the C4
        # anti-thrash buffer reads superseded records' signatures later.
        async with session_maker() as session:
            from sqlalchemy import select as _select

            from mirix.orm.skill_evolution_record import SkillEvolutionRecord as _ORM

            row = (
                await session.execute(_select(_ORM).where(_ORM.id == r1.id))
            ).scalar_one()
            assert row.status == "superseded"
            assert row.is_deleted is False

    async def test_mark_consumed_empty_is_noop(self, session_maker):
        mgr = _manager(session_maker)
        n = await mgr.mark_consumed(ids=[], run_id="run-0")
        assert n == 0

    async def test_record_round_result_rejects_bad_record_type(
        self, session_maker, org, user_a
    ):
        # Invalid record_type must be rejected before it ever hits the plain
        # String column (the column itself would happily store garbage).
        mgr = _manager(session_maker)
        agent_id = await _insert_minimal_agent(session_maker, org.id)
        with pytest.raises(ValidationError):
            await mgr.record_round_result(
                agent_id=agent_id,
                user_id=user_a.id,
                organization_id=user_a.organization_id,
                **_record_kwargs(1, record_type="partial"),
            )
        # Nothing was persisted.
        pending = await mgr.list_pending(agent_id=agent_id)
        assert pending == []


@pytest.mark.asyncio(loop_scope="module")
class TestAggregate:
    async def test_structural_gate_counts_only_gated(self, session_maker, org, user_a):
        mgr = _manager(session_maker)
        agent_id = await _insert_minimal_agent(session_maker, org.id)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # Gated failure: has evidence + non-empty detail.
        gated_fail = await mgr.record_round_result(
            **owner,
            **_record_kwargs(1, record_type="failure"),
        )
        # Gated success.
        gated_succ = await mgr.record_round_result(
            **owner,
            **_record_kwargs(2, record_type="success"),
        )
        # Ungated: empty detail -> does NOT count as high.
        no_detail = await mgr.record_round_result(
            **owner,
            **_record_kwargs(3, record_type="failure", detail=""),
        )
        # Ungated: no evidence -> does NOT count as high.
        no_evidence = await mgr.record_round_result(
            **owner,
            **_record_kwargs(4, record_type="failure", evidence_round_ids=[]),
        )

        ids = [gated_fail.id, gated_succ.id, no_detail.id, no_evidence.id]
        agg = await mgr.aggregate(ids=ids)

        assert agg["n"] == 4
        assert agg["n_high_fail"] == 1  # only gated_fail
        assert agg["n_high_succ"] == 1  # only gated_succ
        assert agg["mean_q"] == pytest.approx(0.6)

    async def test_empty_ids(self, session_maker):
        mgr = _manager(session_maker)
        agg = await mgr.aggregate(ids=[])
        assert agg == {"n": 0, "n_high_fail": 0, "n_high_succ": 0, "mean_q": 0.0}


# ============================== Manager surface ============================


class TestManagerSurface:
    def test_exposes_async_methods(self):
        mgr = SkillEvolutionRecordManager
        for name in (
            "record_round_result",
            "list_pending",
            "mark_consumed",
            "mark_superseded",
            "aggregate",
        ):
            assert hasattr(mgr, name), f"manager missing {name}"
            method = getattr(mgr, name)
            unwrapped = inspect.unwrap(method)
            assert inspect.iscoroutinefunction(unwrapped), (
                f"{name} must be async under its decorators"
            )

    def test_no_asyncio_run(self):
        src = inspect.getsource(SkillEvolutionRecordManager)
        assert "asyncio.run(" not in src, "manager must never call asyncio.run()"


# =============================== Migration SQL =============================


class TestMigrationSql:
    SQL_PATH = Path("scripts/migrate_add_skill_evolution_record.sql")

    def _sql(self) -> str:
        return self.SQL_PATH.read_text()

    def test_creates_table(self):
        assert "CREATE TABLE IF NOT EXISTS skill_evolution_record" in self._sql()

    def test_is_idempotent_block(self):
        sql = self._sql()
        assert "BEGIN;" in sql
        assert "COMMIT;" in sql
        assert "CREATE INDEX IF NOT EXISTS" in sql

    def test_agent_fk_cascades(self):
        assert "REFERENCES agents(id) ON DELETE CASCADE" in self._sql()
