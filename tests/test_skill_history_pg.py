"""Postgres integration tests for the skill write path and history.

Run through the fresh-database runner (MIRIX_PG_URI must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_skill_history_pg.py

What only Postgres can show (the SQLite behavior is in tests/test_skill_history.py):

* concurrent writers — ``asyncio.gather`` over separate pooled connections —
  never produce a duplicate ``version_no``, never two baselines, never two
  live skills with one name (advisory locks + ``SELECT ... FOR UPDATE`` +
  the partial unique index);
* a hard delete that collects objects waits for an in-flight writer that
  re-references one of its objects (exclusive vs shared org advisory lock), so
  no version ever points at a missing object — a deterministic interleaving;
* ``INSERT ... ON CONFLICT DO UPDATE`` stores identical content once and
  refreshes ``last_referenced_at``;
* a failure mid-transaction rolls everything back on Postgres too;
* the baseline backfill is re-runnable, also concurrently with writers;
* name reuse and soft-delete exclusion on the Postgres-only read paths
  (native full-text, org-wide keyword, pgvector) and user erasure without
  foreign-key violations.

Embeddings come from a deterministic in-process fake; nothing reaches a
model provider.
"""

import asyncio
import datetime as dt
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import text

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


async def _fetch(sql: str, **params):
    from mirix.server.server import engine

    async with engine.connect() as conn:
        return (await conn.execute(text(sql), params)).mappings().all()


async def _execute(sql: str, **params):
    from mirix.server.server import engine

    async with engine.begin() as conn:
        await conn.execute(text(sql), params)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def schema():
    from mirix.orm.base import Base
    from mirix.server.server import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


class _FakeEmbedModel:
    async def get_text_embedding(self, text_value):
        # Deterministic, non-zero, text-dependent.
        seed = sum(text_value.encode("utf-8")) % 97 + 1
        return [float(seed)] + [0.25] * 15


@pytest.fixture
def fake_embeddings(monkeypatch):
    import mirix.services.procedural_memory_manager as pmm

    import mirix.services.utils as service_utils

    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel()

    # The write path embeds in the manager; query vectors are built in
    # mirix.services.utils.build_query.
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(service_utils, "embedding_model", fake_embedding_model)


@pytest_asyncio.fixture(loop_scope="module")
async def t(schema, fake_embeddings):
    from mirix.schemas.agent import AgentState, AgentType
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.schemas.user import User as PydanticUser
    from mirix.services.procedural_memory_manager import ProceduralMemoryManager

    org = f"org-{uuid.uuid4().hex[:8]}"
    u1, u2 = (f"user-{uuid.uuid4().hex[:8]}" for _ in range(2))
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    await _execute("INSERT INTO organizations (id, name) VALUES (:o, :o)", o=org)
    for uid in (u1, u2):
        await _execute(
            "INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
            "VALUES (:u, :u, :o, 'active', 'UTC', FALSE)",
            u=uid,
            o=org,
        )
    await _execute(
        "INSERT INTO clients (id, organization_id, name, status, write_scope, read_scopes) "
        "VALUES (:c, :o, 'c', 'active', 'test', '[\"test\"]')",
        c=client_id,
        o=org,
    )
    await _execute(
        "INSERT INTO agents (id, organization_id, agent_type) VALUES (:a, :o, 'procedural_memory_agent')",
        a=agent_id,
        o=org,
    )

    async def no_sleep(_s):
        return None

    actor = PydanticClient(
        id=client_id,
        organization_id=org,
        name="c",
        write_scope="test",
        read_scopes=["test"],
    )
    agent_state = AgentState(
        id=agent_id,
        name="proc",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=16,
        ),
        tools=[],
    )
    return SimpleNamespace(
        org=org,
        u1=u1,
        u2=u2,
        agent_id=agent_id,
        actor=actor,
        agent_state=agent_state,
        mgr=ProceduralMemoryManager(sleep=no_sleep),
        user=lambda uid=u1: PydanticUser(
            id=uid, name=uid, organization_id=org, timezone="UTC"
        ),
    )


async def _create(t, name, *, user_id=None, **overrides):
    fields = dict(
        name=name,
        entry_type="workflow",
        description=f"How to {name}",
        instructions=f"1. {name}",
        triggers=[f"when {name}"],
        examples=[],
        filter_tags={"scope": "test"},
        agent_id=t.agent_id,
        organization_id=t.org,
    )
    fields.update(overrides)
    return await t.mgr.apply_skill_change(
        "create",
        actor=t.actor,
        user_id=user_id or t.u1,
        fields=fields,
        agent_state=t.agent_state,
    )


async def _edit(t, skill_id, **fields):
    return await t.mgr.apply_skill_change(
        "edit", actor=t.actor, user_id=t.u1, skill_id=skill_id, fields=fields
    )


async def _version_rows(skill_id):
    return await _fetch(
        "SELECT id, version_no, parent_version_id, change_type FROM skill_version "
        "WHERE skill_id = :s ORDER BY version_no",
        s=skill_id,
    )


async def _dangling_references(org):
    """Version hashes of the organization with no object behind them."""
    rows = await _fetch(
        """
        SELECT v.id, h.sha
        FROM skill_version v
        CROSS JOIN LATERAL (VALUES (v.name_sha256), (v.entry_type_sha256),
            (v.description_sha256), (v.instructions_sha256), (v.triggers_sha256),
            (v.examples_sha256), (v.filter_tags_sha256)) AS h(sha)
        WHERE v.organization_id = :o
          AND NOT EXISTS (
            SELECT 1 FROM skill_object o
            WHERE o.organization_id = v.organization_id AND o.sha256 = h.sha)
        """,
        o=org,
    )
    return [dict(r) for r in rows]


async def _legacy_skill(t, name):
    skill_id = f"proc-{uuid.uuid4().hex[:8]}"
    await _execute(
        "INSERT INTO procedural_memory (id, organization_id, user_id, agent_id, name, entry_type, "
        "description, instructions, triggers, examples, version, last_modify, is_deleted) VALUES "
        "(:id, :o, :u, :a, :n, 'guide', 'legacy d', 'legacy i', '[]', '[]', '0.1.0', '{}', FALSE)",
        id=skill_id,
        o=t.org,
        u=t.u1,
        a=t.agent_id,
        n=name,
    )
    return skill_id


# --------------------------------------------------------------------------- #
# Concurrent writers                                                           #
# --------------------------------------------------------------------------- #


async def test_concurrent_edits_never_duplicate_a_version_number(t):
    created = await _create(t, "hot-skill")
    writers = 16
    results = await asyncio.gather(
        *(
            _edit(t, created.skill.id, instructions=f"variant {i}")
            for i in range(writers)
        )
    )
    assert all(r.changed for r in results)
    rows = await _version_rows(created.skill.id)
    assert [r["version_no"] for r in rows] == list(range(1, writers + 2))
    # A single chain: every version's parent is the one before it.
    for previous, current in zip(rows, rows[1:]):
        assert current["parent_version_id"] == previous["id"]
    assert sorted(r.version.version_no for r in results) == list(range(2, writers + 2))


async def test_concurrent_read_modify_write_patches_lose_nothing(t):
    """Every writer reads the skill, patches what it read and writes with
    ``expected_fields`` — a writer whose read went stale is refused (never a
    silent overwrite), reads again and re-applies its patch. Every patch ends
    up in the content."""
    from mirix.services.procedural_memory_manager import SkillChangedMeanwhile

    created = await _create(t, "rmw-hot", instructions="base")
    user = t.user()

    async def append(i):
        refusals = 0
        while True:
            skill = await t.mgr.get_item_by_id(
                item_id=created.skill.id, user=user, timezone_str="UTC", actor=t.actor
            )
            try:
                await t.mgr.apply_skill_change(
                    "edit",
                    actor=t.actor,
                    user_id=t.u1,
                    skill_id=created.skill.id,
                    fields={"instructions": f"{skill.instructions} +{i}"},
                    expected_fields={"instructions": skill.instructions},
                    bump_version=True,
                )
                return refusals
            except SkillChangedMeanwhile:
                refusals += 1

    writers = 8
    refusals = await asyncio.gather(*(append(i) for i in range(writers)))
    final = await t.mgr.get_item_by_id(
        item_id=created.skill.id, user=user, timezone_str="UTC", actor=t.actor
    )
    assert sorted(final.instructions.split(" +")[1:], key=int) == [
        str(i) for i in range(writers)
    ]
    assert final.version == f"0.1.{writers}"
    assert len(await _version_rows(created.skill.id)) == writers + 1
    assert sum(refusals) > 0  # the race was real


async def test_an_ambiguous_commit_is_not_written_twice(t, monkeypatch):
    """A create / delete whose COMMIT went through but reported a
    transient error (a serialization failure here) is repeated and reports
    its own success — not a duplicate name, not "not found"."""
    from sqlalchemy.exc import OperationalError

    from mirix.services.procedural_memory_manager import ProceduralMemoryManager

    class _SerializationFailure(Exception):
        sqlstate = "40001"

    real = ProceduralMemoryManager._commit_change
    calls = {"n": 0}

    async def commit_then_fail(self, plan, **kwargs):
        calls["n"] += 1
        result = await real(self, plan, **kwargs)
        if calls["n"] % 2 == 1:
            raise OperationalError("COMMIT", {}, _SerializationFailure("x"))
        return result

    monkeypatch.setattr(ProceduralMemoryManager, "_commit_change", commit_then_fail)
    created = await _create(t, "ambiguous-pg")
    deleted = await t.mgr.soft_delete_procedure_by_id(
        created.skill.id, actor=t.actor, user=t.user(), reason="x"
    )
    assert calls["n"] == 4
    assert created.version.version_no == 1 and deleted.version.version_no == 2
    rows = await _version_rows(created.skill.id)
    assert [r["change_type"] for r in rows] == ["create", "delete"]


async def test_concurrent_first_changes_write_exactly_one_baseline(t):
    skill_id = await _legacy_skill(t, "legacy-hot")
    await asyncio.gather(*(_edit(t, skill_id, instructions=f"v{i}") for i in range(8)))
    rows = await _version_rows(skill_id)
    assert [r["change_type"] for r in rows] == ["baseline"] + ["edit"] * 8
    assert [r["version_no"] for r in rows] == list(range(1, 10))


async def test_concurrent_creates_of_one_name_let_exactly_one_win(t):
    from mirix.services.procedural_memory_manager import SkillNameConflictError

    outcomes = await asyncio.gather(
        *(_create(t, "only-one", description=f"attempt {i}") for i in range(8)),
        return_exceptions=True,
    )
    winners = [o for o in outcomes if not isinstance(o, BaseException)]
    losers = [o for o in outcomes if isinstance(o, BaseException)]
    assert len(winners) == 1
    assert all(isinstance(o, SkillNameConflictError) for o in losers), losers
    live = await _fetch(
        "SELECT count(*) AS n FROM procedural_memory WHERE organization_id = :o "
        "AND user_id = :u AND name = 'only-one' AND NOT is_deleted",
        o=t.org,
        u=t.u1,
    )
    assert live[0]["n"] == 1


async def test_restore_racing_a_create_of_the_same_name(t):
    from mirix.services.procedural_memory_manager import SkillNameConflictError

    for round_no in range(5):
        name = f"contested-{round_no}"
        old = await _create(t, name)
        await t.mgr.soft_delete_procedure_by_id(
            old.skill.id, actor=t.actor, user=t.user()
        )
        outcomes = await asyncio.gather(
            t.mgr.restore_skill(old.skill.id, 1, actor=t.actor, user_id=t.u1),
            _create(t, name, description="newcomer"),
            return_exceptions=True,
        )
        errors = [o for o in outcomes if isinstance(o, BaseException)]
        assert len(errors) == 1 and isinstance(errors[0], SkillNameConflictError), (
            outcomes
        )
        live = await _fetch(
            "SELECT count(*) AS n FROM procedural_memory WHERE organization_id = :o "
            "AND user_id = :u AND name = :n AND NOT is_deleted",
            o=t.org,
            u=t.u1,
            n=name,
        )
        assert live[0]["n"] == 1


async def test_writers_advisory_locks_are_taken(t, monkeypatch):
    """While a write is inside its transaction, the org lock is held SHARED
    and the claimed name's lock EXCLUSIVELY."""
    from mirix.services import skill_history as history

    entered, release = asyncio.Event(), asyncio.Event()
    real_insert = history.insert_version

    async def paused_insert(db, **kwargs):
        entered.set()
        await release.wait()
        return await real_insert(db, **kwargs)

    monkeypatch.setattr(history, "insert_version", paused_insert)
    task = asyncio.create_task(_create(t, "locked-create"))
    await asyncio.wait_for(entered.wait(), 10)
    try:
        locks = await _fetch(
            "SELECT mode, ((classid::bigint << 32) | objid::bigint) AS key "
            "FROM pg_locks WHERE locktype = 'advisory' AND granted"
        )
        modes = {}
        for row in locks:
            key = row["key"]
            signed = key - (1 << 64) if key >= (1 << 63) else key
            modes[signed] = row["mode"]
        assert modes.get(history.org_lock_key(t.org)) == "ShareLock"
        assert (
            modes.get(history.name_lock_key(t.org, t.u1, "locked-create"))
            == "ExclusiveLock"
        )
    finally:
        release.set()
        await task


# --------------------------------------------------------------------------- #
# Object collection vs. writers                                                #
# --------------------------------------------------------------------------- #


async def test_erasure_waits_for_a_writer_that_references_its_objects(t, monkeypatch):
    """Deterministic interleaving: a writer re-references an object only the
    doomed skill had and pauses before its version row; the hard delete must
    wait for it (exclusive org lock) and then keep that object."""
    from mirix.services import skill_history as history

    doomed = await _create(t, "doomed", instructions="only the doomed skill says this")
    keeper = await _create(t, "keeper")

    entered, release = asyncio.Event(), asyncio.Event()
    real_insert = history.insert_version

    async def paused_insert(db, **kwargs):
        if (
            kwargs.get("skill_id") == keeper.skill.id
            and kwargs["change_type"] == "edit"
        ):
            entered.set()
            await release.wait()
        return await real_insert(db, **kwargs)

    monkeypatch.setattr(history, "insert_version", paused_insert)
    writer = asyncio.create_task(
        _edit(t, keeper.skill.id, instructions="only the doomed skill says this")
    )
    await asyncio.wait_for(entered.wait(), 10)
    eraser = asyncio.create_task(
        t.mgr.delete_procedure_by_id(doomed.skill.id, actor=t.actor, user_id=t.u1)
    )
    await asyncio.sleep(0.5)
    assert not eraser.done(), "the collector must wait for the in-flight writer"
    release.set()
    await asyncio.gather(writer, eraser)

    assert await _dangling_references(t.org) == []
    rebuilt = await t.mgr.get_skill_version(
        keeper.skill.id, 2, organization_id=t.org, user_id=t.u1
    )
    assert rebuilt["content"]["instructions"] == "only the doomed skill says this"
    assert await _version_rows(doomed.skill.id) == []


async def test_erasures_racing_writers_never_leave_dangling_versions(t):
    shared = "shared across the org"
    skills = [
        (
            await _create(t, f"racer-{i}", instructions=shared, description=shared)
        ).skill.id
        for i in range(6)
    ]

    async def churn(skill_id, rounds):
        for r in range(rounds):
            await _edit(t, skill_id, instructions=f"{shared} {r % 2}")

    async def erase(skill_id):
        await t.mgr.delete_procedure_by_id(skill_id, actor=t.actor, user_id=t.u1)

    await asyncio.gather(
        churn(skills[0], 6),
        churn(skills[1], 6),
        erase(skills[2]),
        churn(skills[3], 6),
        erase(skills[4]),
        erase(skills[5]),
    )
    assert await _dangling_references(t.org) == []


async def test_a_purge_takes_the_lock_of_every_organization_it_collects(t, monkeypatch):
    """A skill of the user written in another organization after the
    purge's unlocked first read is only seen under the lock; the purge starts
    over with that organization's EXCLUSIVE lock too before it collects its
    objects."""
    from mirix.services import skill_history as history

    other_org = f"org-{uuid.uuid4().hex[:8]}"
    await _execute("INSERT INTO organizations (id, name) VALUES (:o, :o)", o=other_org)
    await _create(t, "purge-home", user_id=t.u2)
    real_lock = history.lock_for_purge
    rounds = []

    async def lock_after_a_concurrent_create(db, *, organization_ids):
        rounds.append(sorted(organization_ids))
        if len(rounds) == 1:
            await _create(
                t,
                "purge-elsewhere",
                user_id=t.u2,
                organization_id=other_org,
                instructions="only in the other organization",
            )
        await real_lock(db, organization_ids=organization_ids)
        held = await db.execute(
            text(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                "AND granted AND mode = 'ExclusiveLock' AND pid = pg_backend_pid()"
            )
        )
        rounds[-1] = (rounds[-1], held.scalar_one())

    monkeypatch.setattr(history, "lock_for_purge", lock_after_a_concurrent_create)
    assert await t.mgr.delete_by_user_id(t.u2) == 2
    assert rounds == [
        ([t.org], 1),
        (sorted([t.org, other_org]), 2),
    ]
    assert await _dangling_references(other_org) == []
    left = await _fetch(
        "SELECT count(*) AS n FROM skill_object WHERE organization_id = :o", o=other_org
    )
    assert left[0]["n"] == 0


# --------------------------------------------------------------------------- #
# Content store, atomicity, backfill                                           #
# --------------------------------------------------------------------------- #


async def test_identical_content_is_stored_once_and_touched(t):
    a = await _create(
        t, "twin-a", description="twin text", instructions="twin steps", triggers=[]
    )
    before = await _fetch(
        "SELECT sha256, last_referenced_at FROM skill_object WHERE organization_id = :o",
        o=t.org,
    )
    await asyncio.sleep(0.05)
    await _create(
        t, "twin-b", description="twin text", instructions="twin steps", triggers=[]
    )
    after = {
        r["sha256"]: r["last_referenced_at"]
        for r in await _fetch(
            "SELECT sha256, last_referenced_at FROM skill_object WHERE organization_id = :o",
            o=t.org,
        )
    }
    assert len(after) == len(before) + 1  # only the new name
    shared_sha = (await _version_rows(a.skill.id)) and (
        await _fetch(
            "SELECT description_sha256 AS s FROM skill_version WHERE skill_id = :s",
            s=a.skill.id,
        )
    )[0]["s"]
    first_seen = {r["sha256"]: r["last_referenced_at"] for r in before}[shared_sha]
    assert after[shared_sha] > first_seen


async def test_a_failure_mid_transaction_rolls_everything_back(t, monkeypatch):
    from mirix.services import skill_history as history

    skill_id = await _legacy_skill(t, "atomic-pg")
    objects_before = await _fetch(
        "SELECT count(*) AS n FROM skill_object WHERE organization_id = :o", o=t.org
    )
    real_insert = history.insert_version

    async def failing(db, **kwargs):
        if kwargs["change_type"] == "edit":
            raise RuntimeError("boom after the baseline and the objects")
        return await real_insert(db, **kwargs)

    monkeypatch.setattr(history, "insert_version", failing)
    with pytest.raises(RuntimeError):
        await _edit(t, skill_id, instructions="brand new text", description="brand new")
    assert await _version_rows(skill_id) == []
    row = await _fetch(
        "SELECT instructions FROM procedural_memory WHERE id = :s", s=skill_id
    )
    assert row[0]["instructions"] == "legacy i"
    objects_after = await _fetch(
        "SELECT count(*) AS n FROM skill_object WHERE organization_id = :o", o=t.org
    )
    assert objects_after[0]["n"] == objects_before[0]["n"]


async def test_backfill_is_rerunnable_and_safe_next_to_writers(t, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "backfill_skill_baseline_versions.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_script_pg", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    legacy = [await _legacy_skill(t, f"bf-{i}") for i in range(5)]
    report, _ = await asyncio.gather(
        module.backfill_baseline_versions(
            manager=t.mgr, organization_id=t.org, batch_size=2
        ),
        _edit(t, legacy[0], instructions="changed during the backfill"),
    )
    # Whoever came first wrote the one baseline of legacy[0] (the edit, or the
    # backfill — which then skips or never sees it); every skill has exactly
    # one baseline, at version 1, and an unbroken numbering.
    for skill_id in legacy:
        rows = await _version_rows(skill_id)
        assert [r["change_type"] for r in rows].count("baseline") == 1
        assert rows[0]["change_type"] == "baseline"
        assert [r["version_no"] for r in rows] == list(range(1, len(rows) + 1))
    assert 4 <= report.written + report.skipped <= 5
    again = await module.backfill_baseline_versions(
        manager=t.mgr, organization_id=t.org
    )
    assert (again.scanned, again.written) == (0, 0)


# --------------------------------------------------------------------------- #
# Name reuse, soft deletes and erasure on Postgres                            #
# --------------------------------------------------------------------------- #


async def test_deleted_skills_leave_every_postgres_read_path(t):
    keep = await _create(t, "pg-keep", instructions="rollout restart kubectl pods")
    gone = await _create(t, "pg-gone", instructions="rollout restart kubectl nodes")
    await t.mgr.soft_delete_procedure_by_id(gone.skill.id, actor=t.actor, user=t.user())
    user, agent_state = t.user(), t.agent_state
    for method, field in [
        ("bm25", "instructions"),  # native full-text (ts_rank_cd)
        ("bm25", ""),
        ("embedding", "description"),  # pgvector
        ("hybrid", "description"),
        ("string_match", "instructions"),
    ]:
        ids = {
            r.id
            for r in await t.mgr.list_procedures(
                agent_state=agent_state,
                user=user,
                query="rollout restart kubectl",
                search_field=field,
                search_method=method,
                timezone_str="UTC",
            )
        }
        assert gone.skill.id not in ids, method
        assert keep.skill.id in ids, method
    for method in ("bm25", "embedding"):
        ids = {
            r.id
            for r in await t.mgr.list_procedures_by_org(
                agent_state=agent_state,
                organization_id=t.org,
                query="rollout restart kubectl",
                search_field="instructions" if method == "bm25" else "description",
                search_method=method,
            )
        }
        assert gone.skill.id not in ids, method
    # The partial unique index frees the name.
    again = await _create(t, "pg-gone")
    assert again.skill.id != gone.skill.id


async def test_user_erasure_then_user_row_delete_has_no_fk_violation(t):
    await _create(t, "erase-1", user_id=t.u2)
    doomed = await _create(t, "erase-2", user_id=t.u2)
    await t.mgr.apply_skill_change(
        "delete", actor=t.actor, user_id=t.u2, skill_id=doomed.skill.id, reason="gone"
    )
    assert await t.mgr.delete_by_user_id(t.u2) == 2
    rows = await _fetch(
        "SELECT count(*) AS n FROM skill_version WHERE user_id = :u", u=t.u2
    )
    assert rows[0]["n"] == 0
    # skill_version.user_id references users: nothing may block the user row.
    await _execute("DELETE FROM users WHERE id = :u", u=t.u2)
    assert await _dangling_references(t.org) == []


async def test_versions_carry_database_timestamps_in_order(t):
    created = await _create(t, "timeline")
    await _edit(t, created.skill.id, instructions="later")
    rows = await _fetch(
        "SELECT created_at FROM skill_version WHERE skill_id = :s ORDER BY version_no",
        s=created.skill.id,
    )
    assert rows[0]["created_at"] <= rows[1]["created_at"]
    assert rows[1]["created_at"].tzinfo is not None
    assert abs(rows[1]["created_at"] - dt.datetime.now(dt.timezone.utc)) < dt.timedelta(
        minutes=5
    )


# --------------------------------------------------------------------------- #
# Missing retrieval vectors                                                    #
# --------------------------------------------------------------------------- #


async def _vectorless_skill(t, name, *, is_deleted=False):
    """A skill as scripts/migrate_skill_retrieval_fields.sql leaves it: its
    own embedding_config, no retrieval vector, no history."""
    import json

    skill_id = f"proc-{uuid.uuid4().hex[:8]}"
    await _execute(
        "INSERT INTO procedural_memory (id, organization_id, user_id, agent_id, name, "
        "entry_type, description, instructions, triggers, examples, version, "
        "last_modify, is_deleted, embedding_config) VALUES (:id, :o, :u, :a, :n, "
        "'guide', :d, 'legacy i', CAST(:tr AS JSON), '[]', '0.1.0', '{}', :del, "
        "CAST(:cfg AS JSON))",
        id=skill_id,
        o=t.org,
        u=t.u1,
        a=t.agent_id,
        n=name,
        d=f"legacy {name}",
        tr=json.dumps([f"when {name}"]),
        **{"del": is_deleted},
        cfg=json.dumps(t.agent_state.embedding_config.model_dump(mode="json")),
    )
    return skill_id


async def test_reembedding_fills_missing_vectors_next_to_a_writer(t):
    # The startup pass / scripts/reembed_skill_retrieval.py on Postgres: the
    # live NULL-vector rows are found (pgvector IS NULL), filled from their
    # retrieval text with their own config, concurrently (bounded), and a text
    # edit racing the pass is never overwritten by a stale vector.
    from mirix.services import skill_history as history
    from mirix.services.skill_retrieval_reembed import (
        reembed_missing_retrieval_embeddings,
    )

    live = [await _vectorless_skill(t, f"re-{i}") for i in range(6)]
    deleted = await _vectorless_skill(t, "re-deleted", is_deleted=True)
    report, _ = await asyncio.gather(
        reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org, batch_size=2, concurrency=3
        ),
        _edit(t, live[0], description="edited during the pass"),
    )
    # The edit may land before the pass lists the rows (it embeds the skill
    # itself), while the pass embeds it, or after it wrote its vector.
    assert report.scanned in (5, 6)
    assert set(report.outcomes) <= {"filled", "changed", "present"}
    assert sum(report.outcomes.values()) == report.scanned
    rows = {
        r["id"]: r
        for r in await _fetch(
            "SELECT id, name, description, triggers, is_deleted, "
            "retrieval_embedding::text AS vector FROM procedural_memory "
            "WHERE organization_id = :o",
            o=t.org,
        )
    }
    for skill_id in live:
        row = rows[skill_id]
        assert row["vector"] is not None
        expected = await _FakeEmbedModel().get_text_embedding(
            history.skill_retrieval_text(dict(row))
        )
        stored = [float(v) for v in row["vector"].strip("[]").split(",")]
        assert stored[: len(expected)] == pytest.approx(expected)
        assert not any(stored[len(expected) :])
    assert rows[live[0]]["description"] == "edited during the pass"
    assert rows[deleted]["vector"] is None  # nothing retrieves a deleted skill
    # No versions for the vectors: only the baseline of the one edited skill
    # and its edit.
    versions = await _fetch(
        "SELECT skill_id, change_type FROM skill_version WHERE skill_id = ANY(:ids) "
        "ORDER BY skill_id, version_no",
        ids=live,
    )
    edits = [v for v in versions if v["change_type"] == "edit"]
    assert [v["skill_id"] for v in edits] == [live[0]]
    again = await reembed_missing_retrieval_embeddings(
        manager=t.mgr, organization_id=t.org
    )
    assert again.scanned == 0
