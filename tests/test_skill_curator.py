"""Tests for C3 — the records-based curator evolution path.

The curator orchestration core (`run_records_evolution`) is dependency-injected
so it can be exercised with NO server, NO network, NO real agent step. We inject:

* a record-manager-like object (the hermetic SQLite C2 manager, reused from
  test_skill_evolution_record's fixture pattern),
* a `run_step` async callable standing in for `ProceduralMemoryAgent.step` — it
  mutates a fake skill bank and returns,
* before/after skill snapshots so the diff (created/edited/deleted) is computed
  exactly as the production endpoint does.

Coverage:
* B_min=0 skip: an all-noise window (no structurally-gated records) skips the
  curator entirely (run_step is NEVER called) and returns an empty diff.
* Budget is set on the agent instance BEFORE step, from the count-driven formula.
* Bookkeeping is OUTSIDE the (mocked) step: AFTER the diff, records flip to
  consumed and `influenced_skill_ids` lineage is written — proving it is not a
  tool side-effect that a context reset would wipe.
* Payload is failure-first, record-shaped (title+detail+evidence), NOT raw
  transcripts; includes an anti-thrash "do NOT re-propose" block from superseded
  records.
* Per-agent lock serializes concurrent evolves on the same agent.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

from mirix.services.skill_curator import (
    build_curator_payload,
    run_records_evolution,
)


# ---------- hermetic SQLite C2 manager (same pattern as test_skill_evolution_record) ----------


@pytest_asyncio.fixture(loop_scope="function")
async def session_maker(tmp_path_factory):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import mirix.orm  # noqa: F401 -- register all ORM classes
    from mirix.orm.base import Base

    db_path = tmp_path_factory.mktemp("curator") / f"{uuid.uuid4().hex}.db"
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


@pytest_asyncio.fixture(loop_scope="function")
async def org(session_maker):
    from mirix.orm.organization import Organization as OrganizationORM

    org_id = f"cur-org-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        await session.commit()
    return type("Org", (), {"id": org_id})()


@pytest_asyncio.fixture(loop_scope="function")
async def user_a(session_maker, org):
    from mirix.orm.user import User as UserORM

    uid = f"cur-user-{uuid.uuid4().hex[:8]}"
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


async def _insert_agent(session_maker, org_id):
    from mirix.orm.agent import Agent as AgentORM

    agent_id = f"agent-{uuid.uuid4()}"
    async with session_maker() as session:
        session.add(AgentORM(id=agent_id, organization_id=org_id))
        await session.commit()
    return agent_id


@pytest_asyncio.fixture(loop_scope="function")
async def agent_id(session_maker, org):
    return await _insert_agent(session_maker, org.id)


def _manager(session_maker):
    from mirix.services.skill_evolution_record_manager import (
        SkillEvolutionRecordManager,
    )

    mgr = SkillEvolutionRecordManager()
    mgr.session_maker = session_maker
    return mgr


def _record_kwargs(round_index, record_type="failure", **overrides):
    base = dict(
        day="day01",
        round_id=f"r{round_index}",
        round_index=round_index,
        record_type=record_type,
        title=f"title-{round_index}",
        description=f"desc-{round_index}",
        detail=f"detail for round {round_index}: root_cause + what_to_avoid",
        evidence_round_ids=[f"r{round_index}"],
        quality_score=0.6,
        generality=0.5,
    )
    base.update(overrides)
    return base


# ---------------------- fake agent / step ----------------------


class _Skill:
    def __init__(
        self,
        sid,
        name,
        instructions,
        version="0.1.0",
        description="d",
        entry_type="guide",
    ):
        self.id = sid
        self.name = name
        self.instructions = instructions
        self.version = version
        self.description = description
        self.entry_type = entry_type


class _FakeAgent:
    """Carries the budget instance attrs the curator sets before step()."""

    def __init__(self):
        self.agent_state = type("S", (), {"id": "agent-1", "parent_id": None})()


def _diff_snapshots(before, after):
    """Mirror the production created/edited/deleted diff (rest_api evolve)."""
    before_ids = {s.id for s in before}
    after_ids = {s.id for s in after}
    before_map = {s.id: s for s in before}
    after_map = {s.id: s for s in after}
    created = after_ids - before_ids
    deleted = before_ids - after_ids
    edited = set()
    for sid in before_ids & after_ids:
        if (
            before_map[sid].version != after_map[sid].version
            or before_map[sid].instructions != after_map[sid].instructions
            or before_map[sid].description != after_map[sid].description
        ):
            edited.add(sid)
    return {"created": created, "edited": edited, "deleted": deleted}


# =============================== Tests =================================


@pytest.mark.asyncio(loop_scope="function")
class TestBminZeroSkip:
    async def test_all_noise_window_skips_curator(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # All records are ungated (empty detail OR no evidence) -> n_high_* == 0.
        for ri in (1, 2, 3):
            await mgr.record_round_result(
                **owner, **_record_kwargs(ri, detail="", evidence_round_ids=[])
            )

        step_called = {"n": 0}

        async def run_step(agent, payload, budget):
            step_called["n"] += 1

        result = await run_records_evolution(
            record_manager=mgr,
            agent=_FakeAgent(),
            agent_id=agent_id,
            run_id="run-skip",
            watermark=10,
            snapshot_skills=lambda: [],
            run_step=run_step,
            apply_lineage=_noop_lineage,
        )

        assert step_called["n"] == 0, "curator must be SKIPPED on a noise window"
        assert result["skipped"] is True
        assert result["budget"] == 0
        assert result["changes"]["created"] == []
        assert result["changes"]["edited"] == []
        assert result["changes"]["deleted"] == []
        # Nothing consumed when skipped.
        pending = await mgr.list_pending(agent_id=agent_id)
        assert len(pending) == 3


@pytest.mark.asyncio(loop_scope="function")
class TestBudgetSetBeforeStep:
    async def test_all_fail_budget_is_b_max(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        for ri in range(1, 6):  # 5 gated failures
            await mgr.record_round_result(**owner, **_record_kwargs(ri))

        seen = {}

        async def run_step(agent, payload, budget):
            # Budget must already be on the agent instance BEFORE step runs.
            seen["instance_budget"] = agent._edit_budget_remaining
            seen["arg_budget"] = budget

        agent = _FakeAgent()
        result = await run_records_evolution(
            record_manager=mgr,
            agent=agent,
            agent_id=agent_id,
            run_id="run-fail",
            watermark=10,
            snapshot_skills=lambda: [],
            run_step=run_step,
            apply_lineage=_noop_lineage,
        )
        assert result["budget"] == 6
        assert seen["instance_budget"] == 6
        assert seen["arg_budget"] == 6

    async def test_mixed_budget_is_mid(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        # 2 gated failures + 2 gated successes -> raw = 1 + 2 + 1 = 4.
        for ri in (1, 2):
            await mgr.record_round_result(
                **owner, **_record_kwargs(ri, record_type="failure")
            )
        for ri in (3, 4):
            await mgr.record_round_result(
                **owner, **_record_kwargs(ri, record_type="success")
            )

        async def run_step(agent, payload, budget):
            pass

        result = await run_records_evolution(
            record_manager=mgr,
            agent=_FakeAgent(),
            agent_id=agent_id,
            run_id="run-mixed",
            watermark=10,
            snapshot_skills=lambda: [],
            run_step=run_step,
            apply_lineage=_noop_lineage,
        )
        assert result["budget"] == 4

    async def test_hybrid_autonomous_only_reduces(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        for ri in range(1, 6):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))  # formula=6

        async def run_step(agent, payload, budget):
            pass

        # autonomous_budget=2 -> final = min(6, 2) = 2.
        async def autonomous(aggregate):
            return 2

        result = await run_records_evolution(
            record_manager=mgr,
            agent=_FakeAgent(),
            agent_id=agent_id,
            run_id="run-hybrid",
            watermark=10,
            snapshot_skills=lambda: [],
            run_step=run_step,
            apply_lineage=_noop_lineage,
            autonomous_budget_fn=autonomous,
            use_autonomous_budget=True,
        )
        assert result["budget"] == 2

    async def test_autonomous_off_by_default(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        for ri in range(1, 6):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))

        called = {"auto": False}

        async def autonomous(aggregate):
            called["auto"] = True
            return 1

        async def run_step(agent, payload, budget):
            pass

        # use_autonomous_budget defaults to False -> autonomous fn must NOT run.
        result = await run_records_evolution(
            record_manager=mgr,
            agent=_FakeAgent(),
            agent_id=agent_id,
            run_id="run-noauto",
            watermark=10,
            snapshot_skills=lambda: [],
            run_step=run_step,
            apply_lineage=_noop_lineage,
            autonomous_budget_fn=autonomous,
        )
        assert called["auto"] is False
        assert result["budget"] == 6  # formula-only


@pytest.mark.asyncio(loop_scope="function")
class TestBookkeepingOutsideStep:
    async def test_consume_and_lineage_after_diff(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        recs = []
        for ri in range(1, 4):
            recs.append(await mgr.record_round_result(**owner, **_record_kwargs(ri)))
        record_ids = [r.id for r in recs]

        # before/after snapshots: step "creates" proc-new + "edits" proc-old.
        before = [_Skill("proc-old", "old", "v1 instructions", version="0.1.0")]

        async def run_step(agent, payload, budget):
            # The step would mutate the bank; we simulate the result via snapshot.
            pass

        # after-snapshot reflects a create + an edit (version bump).
        after = [
            _Skill("proc-old", "old", "v2 instructions", version="0.1.1"),
            _Skill("proc-new", "new", "fresh", version="0.1.0"),
        ]
        snapshots = iter([before, after])

        lineage_calls = []

        async def apply_lineage(record_ids_arg, influenced_ids):
            lineage_calls.append((list(record_ids_arg), sorted(influenced_ids)))

        result = await run_records_evolution(
            record_manager=mgr,
            agent=_FakeAgent(),
            agent_id=agent_id,
            run_id="run-bk",
            watermark=10,
            snapshot_skills=lambda: next(snapshots),
            run_step=run_step,
            apply_lineage=apply_lineage,
        )

        # Diff computed AFTER step.
        assert set(result["changes"]["created"]) == {"proc-new"}
        assert set(result["changes"]["edited"]) == {"proc-old"}
        influenced = set(result["influenced_skill_ids"])
        assert influenced == {"proc-new", "proc-old"}

        # Records consumed AFTER the diff (bookkeeping outside the step / reset).
        pending = await mgr.list_pending(agent_id=agent_id)
        assert pending == []
        # consumed_by stamped with the run id.
        agg = await mgr.aggregate(ids=record_ids)
        assert agg["n"] == 3
        async with session_maker() as session:
            from sqlalchemy import select

            from mirix.orm.skill_evolution_record import SkillEvolutionRecord as ORM

            rows = (
                (await session.execute(select(ORM).where(ORM.id.in_(record_ids))))
                .scalars()
                .all()
            )
            assert all(r.status == "consumed" for r in rows)
            assert all(r.consumed_by == "run-bk" for r in rows)

        # Lineage write happened with the influenced skill ids.
        assert len(lineage_calls) == 1
        assert lineage_calls[0][1] == ["proc-new", "proc-old"]


@pytest.mark.asyncio(loop_scope="function")
class TestPayloadShape:
    async def test_failure_first_record_shaped_not_transcript(
        self, session_maker, agent_id, user_a
    ):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await mgr.record_round_result(
            **owner, **_record_kwargs(1, record_type="success", title="ok-thing")
        )
        await mgr.record_round_result(
            **owner, **_record_kwargs(2, record_type="failure", title="bad-thing")
        )
        pending = await mgr.list_pending(agent_id=agent_id, before_round_index=10)
        payload = build_curator_payload(pending, superseded_signatures=[])

        # Failure appears before success (list_pending already orders failures
        # first; the payload must preserve that).
        assert payload.index("bad-thing") < payload.index("ok-thing")
        # Record-shaped: carries title + detail + evidence, NOT raw prompt/response.
        assert "detail for round 2" in payload
        assert "r2" in payload  # evidence round id
        # No transcript markers leak in.
        assert "prompt_text" not in payload
        assert "response_text" not in payload

    async def test_anti_thrash_block_included(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        await mgr.record_round_result(**owner, **_record_kwargs(1))
        pending = await mgr.list_pending(agent_id=agent_id, before_round_index=10)
        sigs = ["round r9: do-not-redo-this-edit"]
        payload = build_curator_payload(pending, superseded_signatures=sigs)
        assert "do not re-propose" in payload.lower() or "do NOT re-propose" in payload
        assert "do-not-redo-this-edit" in payload


@pytest.mark.asyncio(loop_scope="function")
class TestPerAgentLock:
    async def test_concurrent_evolves_serialized(self, session_maker, agent_id, user_a):
        mgr = _manager(session_maker)
        owner = dict(
            agent_id=agent_id, user_id=user_a.id, organization_id=user_a.organization_id
        )
        for ri in range(1, 4):
            await mgr.record_round_result(**owner, **_record_kwargs(ri))

        # Detect overlap: if the lock works, the two run_steps never overlap.
        state = {"in_flight": 0, "max_overlap": 0}

        async def run_step(agent, payload, budget):
            state["in_flight"] += 1
            state["max_overlap"] = max(state["max_overlap"], state["in_flight"])
            await asyncio.sleep(0.02)
            state["in_flight"] -= 1

        snaps = iter([[], [], [], []])

        async def evolve(run_id):
            return await run_records_evolution(
                record_manager=mgr,
                agent=_FakeAgent(),
                agent_id=agent_id,
                run_id=run_id,
                watermark=10,
                snapshot_skills=lambda: next(snaps),
                run_step=run_step,
                apply_lineage=_noop_lineage,
            )

        await asyncio.gather(evolve("run-x"), evolve("run-y"))
        assert state["max_overlap"] == 1, "per-agent lock must serialize evolves"


# A no-op lineage writer for tests that don't assert on lineage.
async def _noop_lineage(record_ids, influenced_ids):
    return None


@pytest.mark.asyncio(loop_scope="function")
class TestSoftDeleteExcludedFromRetrieval:
    """Regression for the codex HIGH finding: a curator soft-delete
    (`is_deleted=True`) must be EXCLUDED from `list_procedures(use_cache=False)`,
    otherwise it would still appear in the evolve before/after snapshot and in
    retrieval. The raw fallback queries previously did not filter `~is_deleted`.
    """

    async def test_empty_query_excludes_soft_deleted(
        self, session_maker, agent_id, user_a
    ):
        from mirix.orm.procedural_memory import (
            ProceduralMemoryItem as ProcORM,
        )
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        # Insert one live + one soft-deleted skill row directly (bypassing
        # insert_procedure so we need no embedder).
        async with session_maker() as session:
            session.add(
                ProcORM(
                    id="proc-live",
                    agent_id=agent_id,
                    user_id=user_a.id,
                    organization_id=user_a.organization_id,
                    name="live-skill",
                    entry_type="guide",
                    description="a live skill",
                    instructions="do the thing",
                    is_deleted=False,
                )
            )
            session.add(
                ProcORM(
                    id="proc-gone",
                    agent_id=agent_id,
                    user_id=user_a.id,
                    organization_id=user_a.organization_id,
                    name="gone-skill",
                    entry_type="guide",
                    description="a soft-deleted skill",
                    instructions="do not surface me",
                    is_deleted=True,
                )
            )
            await session.commit()

        mgr = ProceduralMemoryManager()
        mgr.session_maker = session_maker

        from mirix.schemas.agent import AgentState, AgentType
        from mirix.schemas.embedding_config import EmbeddingConfig
        from mirix.schemas.llm_config import LLMConfig
        from mirix.schemas.user import User as PydanticUser

        agent_state = AgentState(
            id=agent_id,
            name="proc",
            system="s",
            agent_type=AgentType.procedural_memory_agent,
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
            tools=[],
        )
        user = PydanticUser(
            id=user_a.id,
            name="u",
            timezone="UTC",
            organization_id=user_a.organization_id,
        )

        # Empty-query path (the snapshot path) excludes the soft-deleted skill.
        results = await mgr.list_procedures(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="description",
            search_method="bm25",
            limit=1000,
            use_cache=False,
        )
        ids = {r.id for r in results}
        assert "proc-live" in ids
        assert "proc-gone" not in ids, "soft-deleted skill must be excluded"

        # The fuzzy_match fallback (SQLite, in-memory) must also exclude it.
        fuzzy = await mgr.list_procedures(
            agent_state=agent_state,
            user=user,
            query="skill",
            search_field="description",
            search_method="fuzzy_match",
            limit=1000,
            use_cache=False,
        )
        fuzzy_ids = {r.id for r in fuzzy}
        assert "proc-gone" not in fuzzy_ids, "fuzzy_match must exclude soft-deleted"

        # The string_match path (base_query) must exclude it too.
        sm = await mgr.list_procedures(
            agent_state=agent_state,
            user=user,
            query="skill",
            search_field="description",
            search_method="string_match",
            limit=1000,
            use_cache=False,
        )
        assert "proc-gone" not in {r.id for r in sm}, (
            "string_match must exclude soft-deleted"
        )

    async def test_list_procedures_by_org_excludes_soft_deleted(
        self, session_maker, agent_id, user_a
    ):
        """Regression for BUG A: the org-wide retrieval
        (`list_procedures_by_org`, served via `GET /memory/search_all_users`)
        SQLAlchemy fallback previously omitted the `~is_deleted` predicate, so a
        curator soft-delete (`is_deleted=True`) could leak across the whole org.
        It must match the per-user `list_procedures` behaviour and exclude
        soft-deleted skills.
        """
        from mirix.orm.procedural_memory import (
            ProceduralMemoryItem as ProcORM,
        )
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        # Insert one live + one soft-deleted skill row directly (bypassing
        # insert_procedure so we need no embedder).
        async with session_maker() as session:
            session.add(
                ProcORM(
                    id="org-proc-live",
                    agent_id=agent_id,
                    user_id=user_a.id,
                    organization_id=user_a.organization_id,
                    name="org-live-skill",
                    entry_type="guide",
                    description="a live skill",
                    instructions="do the thing",
                    is_deleted=False,
                )
            )
            session.add(
                ProcORM(
                    id="org-proc-gone",
                    agent_id=agent_id,
                    user_id=user_a.id,
                    organization_id=user_a.organization_id,
                    name="org-gone-skill",
                    entry_type="guide",
                    description="a soft-deleted skill",
                    instructions="do not surface me",
                    is_deleted=True,
                )
            )
            await session.commit()

        mgr = ProceduralMemoryManager()
        mgr.session_maker = session_maker

        from mirix.schemas.agent import AgentState, AgentType
        from mirix.schemas.embedding_config import EmbeddingConfig
        from mirix.schemas.llm_config import LLMConfig

        agent_state = AgentState(
            id=agent_id,
            name="proc",
            system="s",
            agent_type=AgentType.procedural_memory_agent,
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(provider="openai"),
            tools=[],
        )

        # Empty-query path (recent sort) exercises the shared base_query that the
        # fix patched and must exclude the soft-deleted skill org-wide.
        results = await mgr.list_procedures_by_org(
            agent_state=agent_state,
            organization_id=user_a.organization_id,
            query="",
            search_field="description",
            search_method="bm25",
            limit=1000,
            use_cache=False,
        )
        ids = {r.id for r in results}
        assert "org-proc-live" in ids
        assert "org-proc-gone" not in ids, (
            "soft-deleted skill must be excluded from list_procedures_by_org"
        )

        # A non-empty query that falls through to the patched base_query (no PG
        # full-text on SQLite, no embedding) must also exclude it.
        text_results = await mgr.list_procedures_by_org(
            agent_state=agent_state,
            organization_id=user_a.organization_id,
            query="skill",
            search_field="description",
            search_method="string_match",
            limit=1000,
            use_cache=False,
        )
        assert "org-proc-gone" not in {r.id for r in text_results}, (
            "soft-deleted skill must be excluded from org-wide text search"
        )
