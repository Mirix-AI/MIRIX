"""Skill version history and the single skill write path.

Version history (``apply_skill_change``), soft delete, no Redis, and the hard
delete that erases the history. Hermetic: a
throwaway SQLite database, a recording fake embedding model, no Redis, no
model provider. Postgres-only behavior (advisory locks, concurrent writers,
``ON CONFLICT`` on Postgres) is in tests/test_skill_history_pg.py.

Covered here:

* canonical encoding: text = UTF-8, JSON = sorted/compact/non-ASCII, the
  sha256 taken before zlib; the retrieval text;
* create / edit / delete / restore versions: numbering, parent chain,
  provenance, ``changed_fields``, the lazily written ``baseline``;
* identical content is stored once (per organization);
* a failure at any step rolls back EVERYTHING (row, baseline, objects,
  version, consumed experiences);
* the retrieval vector is computed outside every transaction, only when the
  retrieval text changes, and a failing embedding leaves NULL after the
  retries instead of failing the write;
* restore: byte-identical content, un-delete, 409 name conflict, unknown
  version, other owners;
* soft delete: every read / count / search excludes it, its name can be
  reused;
* hard delete / erasure: skill + all versions + unreferenced objects,
  objects still referenced by another skill survive; user and client erasure;
  organization deletion;
* bulk soft deletes write ``delete`` versions;
* experiences consumed in the write's transaction;
* the backfill script is idempotent.
"""

import sys
import uuid
import zlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.errors import NoResultFound, UniqueConstraintViolationError
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
from mirix.orm.skill_object import SkillObject as SkillObjectORM
from mirix.orm.skill_version import SkillVersion as SkillVersionORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate
from mirix.schemas.skill_version import SKILL_VERSIONED_FIELDS
from mirix.schemas.user import User as PydanticUser
from mirix.services import skill_history as history
from mirix.services.procedural_memory_manager import (
    ACTOR_AGENT,
    ACTOR_CURATOR,
    ProceduralMemoryManager,
    SkillChangedMeanwhile,
    SkillNameConflictError,
)
from mirix.services.skill_history import SkillVersionNotFound
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

# An unreachable local endpoint: any code path that escaped the monkeypatched
# embedding factory fails fast on localhost instead of reaching a provider.
_OFFLINE_EMBEDDING = {
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": "http://127.0.0.1:9/v1",
    "embedding_model": "x",
    "embedding_dim": 8,
}


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


class _CountingSessionMaker:
    """The test database's session factory, counting the sessions open now —
    the embedding fake asserts it runs while none is."""

    def __init__(self, local):
        self._local = local
        self.open = 0

    @asynccontextmanager
    async def __call__(self):
        async with self._local() as session:
            self.open += 1
            try:
                yield session
            finally:
                self.open -= 1
                await session.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_history") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = _CountingSessionMaker(async_sessionmaker(engine, expire_on_commit=False))
    yield maker
    await engine.dispose()


class _FakeEmbedModel:
    def __init__(self, state):
        self.state = state

    async def get_text_embedding(self, text):
        self.state.calls.append(text)
        # Never inside a transaction (or any open session).
        self.state.open_sessions_seen.append(self.state.sm.open)
        if self.state.fail:
            raise ConnectionError("embedding endpoint down")
        # Deterministic: first dims encode the text length.
        return [float(len(text) % 7 + 1)] + [0.5] * 7


@pytest.fixture
def embed(monkeypatch, sm):
    state = SimpleNamespace(calls=[], open_sessions_seen=[], fail=False, sm=sm)

    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel(state)

    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    return state


def _agent_state():
    return AgentState(
        id="agent-hist",
        name="hist",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(**_OFFLINE_EMBEDDING),
        tools=[],
    )


async def _no_sleep(_seconds):
    return None


@pytest_asyncio.fixture(loop_scope="module")
async def t(sm, embed):
    """A fresh tenant per test: org, two users, two clients, an agent."""
    org_id = f"org-{uuid.uuid4().hex[:8]}"
    other_org = f"org-{uuid.uuid4().hex[:8]}"
    ids = SimpleNamespace(
        org=org_id,
        other_org=other_org,
        u1=f"user-{uuid.uuid4().hex[:8]}",
        u2=f"user-{uuid.uuid4().hex[:8]}",
        c1=f"client-{uuid.uuid4().hex[:8]}",
        c2=f"client-{uuid.uuid4().hex[:8]}",
        agent=f"agent-{uuid.uuid4()}",
    )
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        db.add(OrganizationORM(id=other_org, name=other_org))
        await db.flush()
        for uid in (ids.u1, ids.u2):
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org_id,
                    status="active",
                    timezone="UTC",
                )
            )
        for cid in (ids.c1, ids.c2):
            db.add(
                ClientORM(
                    id=cid,
                    organization_id=org_id,
                    name=cid,
                    status="active",
                    write_scope="test",
                    read_scopes=["test"],
                )
            )
        db.add(AgentORM(id=ids.agent, organization_id=org_id))
        await db.commit()
    mgr = ProceduralMemoryManager(session_maker=sm, sleep=_no_sleep)

    def client(cid=ids.c1, org=org_id):
        return PydanticClient(
            id=cid,
            organization_id=org,
            name=cid,
            write_scope="test",
            read_scopes=["test"],
        )

    def user(uid=ids.u1):
        return PydanticUser(id=uid, name=uid, organization_id=org_id, timezone="UTC")

    return SimpleNamespace(
        ids=ids, mgr=mgr, client=client, user=user, agent_state=_agent_state(), sm=sm
    )


async def _create(
    t, name="deploy", *, user_id=None, actor=None, run_id=None, **overrides
):
    fields = dict(
        name=name,
        entry_type="workflow",
        description=f"How to {name}",
        instructions=f"Step 1: {name}.",
        triggers=[f"when {name}"],
        examples=[{"input": "q", "output": "a"}],
        filter_tags={"scope": "test"},
        agent_id=t.ids.agent,
        organization_id=t.ids.org,
    )
    fields.update(overrides)
    return await t.mgr.apply_skill_change(
        "create",
        actor=actor or t.client(),
        user_id=user_id or t.ids.u1,
        fields=fields,
        agent_state=t.agent_state,
        actor_kind=ACTOR_AGENT,
        run_id=run_id,
    )


async def _versions(t, skill_id):
    async with t.sm() as db:
        return await history.list_versions(db, skill_id)


async def _count(t, model, *where):
    async with t.sm() as db:
        return (
            await db.execute(select(func.count()).select_from(model).where(*where))
        ).scalar_one()


async def _objects(t, org=None):
    return await _count(
        t, SkillObjectORM, SkillObjectORM.organization_id == (org or t.ids.org)
    )


async def _legacy_skill(t, name="legacy", *, is_deleted=False, user_id=None):
    """A skill as it exists before this change: a row with no history."""
    skill_id = f"proc-{uuid.uuid4().hex[:8]}"
    async with t.sm() as db:
        db.add(
            ProceduralORM(
                id=skill_id,
                organization_id=t.ids.org,
                user_id=user_id or t.ids.u1,
                agent_id=t.ids.agent,
                client_id=t.ids.c1,
                name=name,
                entry_type="guide",
                description="legacy description",
                instructions="legacy instructions",
                triggers=["legacy trigger"],
                examples=[],
                filter_tags={"scope": "test"},
                version="0.3.0",
                last_modify={
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "operation": "created",
                },
                is_deleted=is_deleted,
            )
        )
        await db.commit()
    return skill_id


# --------------------------------------------------------------------------- #
# Canonical encoding and the retrieval text                                    #
# --------------------------------------------------------------------------- #


class TestCanonicalEncoding:
    def test_text_fields_are_their_utf8_bytes(self):
        assert history.canonical_bytes("description", "héllo ✓") == "héllo ✓".encode(
            "utf-8"
        )

    def test_json_fields_are_sorted_compact_and_not_ascii_escaped(self):
        raw = history.canonical_bytes("examples", [{"b": 1, "a": "é"}])
        assert raw == '[{"a":"é","b":1}]'.encode("utf-8")
        assert history.canonical_bytes("filter_tags", None) == b"null"

    def test_decode_is_the_inverse(self):
        for field, value in [
            ("name", "x-y"),
            ("triggers", ["a", "ü"]),
            ("examples", [{"input": "i", "output": {"k": [1, 2]}}]),
            ("filter_tags", {"scope": "s", "tags": ["a"]}),
        ]:
            assert (
                history.decode_field(field, history.canonical_bytes(field, value))
                == value
            )

    def test_the_hash_is_of_the_uncompressed_bytes(self):
        encoded = history.encode_content(
            {
                f: ("v" if f in history.TEXT_FIELDS else [])
                for f in SKILL_VERSIONED_FIELDS
            }
        )
        sha, raw = encoded["name"]
        import hashlib

        assert sha == hashlib.sha256(b"v").hexdigest() and raw == b"v"

    def test_retrieval_text_is_name_triggers_description(self):
        text = history.skill_retrieval_text(
            {
                "name": "n",
                "triggers": ["t1", "t2"],
                "description": "d",
                "instructions": "i",
            }
        )
        assert text == "n\nt1\nt2\nd"


# --------------------------------------------------------------------------- #
# create / edit                                                                #
# --------------------------------------------------------------------------- #


class TestCreateAndEdit:
    async def test_create_writes_version_1(self, t):
        result = await _create(t, "ship", run_id="run-1:b0")
        versions = await _versions(t, result.skill.id)
        assert [(v.version_no, v.change_type) for v in versions] == [(1, "create")]
        v1 = versions[0]
        assert v1.parent_version_id is None
        assert v1.run_id == "run-1:b0" and v1.actor_kind == ACTOR_AGENT
        assert v1.organization_id == t.ids.org and v1.user_id == t.ids.u1
        assert v1._created_by_id == t.ids.c1
        # Seven objects (all field values differ), zlib-compressed.
        assert await _objects(t) == 7
        async with t.sm() as db:
            obj = (
                await db.execute(
                    select(SkillObjectORM).where(
                        SkillObjectORM.sha256 == v1.instructions_sha256
                    )
                )
            ).scalar_one()
        assert zlib.decompress(obj.content_zlib) == b"Step 1: ship."
        assert obj.raw_length == len(b"Step 1: ship.")

    async def test_create_rejects_a_live_name_and_writes_nothing(self, t):
        await _create(t, "dup")
        objects, versions = await _objects(t), await _count(t, SkillVersionORM)
        with pytest.raises(SkillNameConflictError):
            await _create(t, "dup", description="other")
        assert isinstance(SkillNameConflictError("x"), UniqueConstraintViolationError)
        assert await _objects(t) == objects
        assert await _count(t, SkillVersionORM) == versions

    async def test_the_same_name_is_fine_for_another_user(self, t):
        await _create(t, "shared-name")
        other = await _create(t, "shared-name", user_id=t.ids.u2)
        assert other.skill.user_id == t.ids.u2

    async def test_edit_versions_and_changed_fields(self, t):
        created = await _create(t, "edit-me")
        edited = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "Step 1: better."},
            actor_kind=ACTOR_CURATOR,
            run_id="run-2:b1",
        )
        assert edited.version.version_no == 2
        assert edited.version.parent_version_id == created.version.id
        assert edited.skill.instructions == "Step 1: better."
        hist = await t.mgr.get_skill_history(
            created.skill.id, organization_id=t.ids.org, user_id=t.ids.u1
        )
        assert [v["changed_fields"] for v in hist["versions"]] == [
            list(SKILL_VERSIONED_FIELDS),
            ["instructions"],
        ]
        assert hist["versions"][1]["run_id"] == "run-2:b1"
        assert hist["versions"][1]["actor_kind"] == ACTOR_CURATOR

    async def test_first_change_of_a_legacy_skill_writes_its_baseline(self, t):
        skill_id = await _legacy_skill(t)
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=skill_id,
            fields={"description": "new description"},
            agent_state=t.agent_state,
        )
        versions = await _versions(t, skill_id)
        assert [(v.version_no, v.change_type) for v in versions] == [
            (1, "baseline"),
            (2, "edit"),
        ]
        assert result.baseline.id == versions[0].id
        assert versions[1].parent_version_id == versions[0].id
        # The baseline holds the content as it was.
        v1 = await t.mgr.get_skill_version(
            skill_id, 1, organization_id=t.ids.org, user_id=t.ids.u1
        )
        assert v1["content"]["description"] == "legacy description"

    async def test_identical_content_writes_no_version(self, t):
        created = await _create(t, "same")
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": created.skill.instructions},
        )
        assert result.version is None and not result.changed
        assert len(await _versions(t, created.skill.id)) == 1

    async def test_a_version_string_only_edit_updates_the_row_without_a_version(
        self, t
    ):
        created = await _create(t, "bump")
        result = await t.mgr.update_item(
            ProceduralMemoryItemUpdate(id=created.skill.id, version="0.1.9"),
            user=t.user(),
            actor=t.client(),
        )
        assert result.version == "0.1.9"
        assert len(await _versions(t, created.skill.id)) == 1

    async def test_an_explicit_none_is_not_given(self, t):
        created = await _create(t, "keep-fields")
        updated = await t.mgr.update_item(
            ProceduralMemoryItemUpdate(
                id=created.skill.id, description=None, triggers=None, instructions="x"
            ),
            user=t.user(),
            actor=t.client(),
        )
        assert updated.description == created.skill.description
        assert updated.triggers == created.skill.triggers
        assert updated.instructions == "x"

    async def test_create_many_items_versions_every_skill(self, t):
        from mirix.schemas.procedural_memory import ProceduralMemoryItem as Item

        items = [
            Item(
                name=f"many-{i}",
                entry_type="workflow",
                description="d",
                instructions="i",
                user_id=t.ids.u1,
                organization_id=t.ids.org,
            )
            for i in range(2)
        ]
        with_actor = await t.mgr.create_many_items(items, t.user(), actor=t.client())
        assert {s.client_id for s in with_actor} == {t.ids.c1}
        # The legacy call shape (no actor) works too (it no longer crashes).
        legacy = await t.mgr.create_many_items(
            [items[0].model_copy(update={"name": "many-legacy"})], t.user()
        )
        for skill in [*with_actor, *legacy]:
            assert [v.change_type for v in await _versions(t, skill.id)] == ["create"]

    async def test_an_edit_never_moves_a_skill_to_another_organization(self, t):
        created = await _create(t, "stay")
        updated = await t.mgr.update_item(
            ProceduralMemoryItemUpdate(
                id=created.skill.id, organization_id=t.ids.other_org, description="x"
            ),
            user=t.user(),
            actor=t.client(),
        )
        assert updated.organization_id == t.ids.org

    async def test_renaming_onto_a_live_name_conflicts(self, t):
        await _create(t, "taken")
        mine = await _create(t, "mine")
        with pytest.raises(SkillNameConflictError):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id=mine.skill.id,
                fields={"name": "taken"},
            )
        assert len(await _versions(t, mine.skill.id)) == 1

    async def test_other_owners_and_organizations_cannot_edit(self, t):
        created = await _create(t, "private")
        with pytest.raises(NoResultFound):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u2,
                skill_id=created.skill.id,
                fields={"description": "hijack"},
            )
        with pytest.raises(NoResultFound):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(org=t.ids.other_org),
                user_id=t.ids.u1,
                skill_id=created.skill.id,
                fields={"description": "hijack"},
            )


# --------------------------------------------------------------------------- #
# Content addressing                                                           #
# --------------------------------------------------------------------------- #


class TestContentAddressing:
    async def test_identical_content_is_stored_once(self, t):
        a = await _create(
            t, "alpha", description="same text", instructions="same steps", triggers=[]
        )
        objects_after_a = await _objects(t)
        # A second skill sharing six field values adds only its name.
        await _create(
            t, "beta", description="same text", instructions="same steps", triggers=[]
        )
        assert await _objects(t) == objects_after_a + 1
        # An instructions-only edit adds exactly one object; editing back adds none.
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=a.skill.id,
            fields={"instructions": "new steps"},
        )
        assert await _objects(t) == objects_after_a + 2
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=a.skill.id,
            fields={"instructions": "same steps"},
        )
        assert await _objects(t) == objects_after_a + 2
        async with t.sm() as db:
            shas = (
                (
                    await db.execute(
                        select(SkillObjectORM.sha256).where(
                            SkillObjectORM.organization_id == t.ids.org
                        )
                    )
                )
                .scalars()
                .all()
            )
        assert len(shas) == len(set(shas))

    async def test_objects_never_dedupe_across_organizations(self, t):
        await _create(t, "same-everywhere")
        other = await _create(
            t,
            "same-everywhere",
            actor=t.client(org=t.ids.other_org),
            organization_id=t.ids.other_org,
        )
        assert other.version.organization_id == t.ids.other_org
        assert await _objects(t, t.ids.other_org) == 7


# --------------------------------------------------------------------------- #
# Atomicity                                                                    #
# --------------------------------------------------------------------------- #


class TestRollback:
    async def _state(self, t, skill_id):
        async with t.sm() as db:
            row = (
                await db.execute(
                    select(ProceduralORM).where(ProceduralORM.id == skill_id)
                )
            ).scalar_one()
            return (
                row.description,
                row.is_deleted,
                await _count(t, SkillVersionORM, SkillVersionORM.skill_id == skill_id),
                await _objects(t),
            )

    @pytest.mark.parametrize("fail_at", ["insert_version", "consume"])
    async def test_a_failure_mid_way_rolls_everything_back(
        self, t, monkeypatch, fail_at
    ):
        skill_id = await _legacy_skill(t, f"atomic-{fail_at}")
        exp_id = f"sexp-{uuid.uuid4().hex[:8]}"
        async with t.sm() as db:
            db.add(
                SkillExperienceORM(
                    id=exp_id,
                    organization_id=t.ids.org,
                    user_id=t.ids.u1,
                    agent_id=t.ids.agent,
                    session_id="s",
                    experience_type="worth_learning",
                    title="t",
                    content="c",
                    importance=0.5,
                    credibility=0.5,
                    evidence="",
                    status="pending",
                )
            )
            await db.commit()
        before = await self._state(t, skill_id)
        assert before[2] == 0  # no history yet

        calls = {"n": 0}
        real_insert = history.insert_version

        async def failing_insert(db, **kwargs):
            calls["n"] += 1
            if kwargs["change_type"] == "edit":  # after the baseline + objects
                raise RuntimeError("boom in the middle")
            return await real_insert(db, **kwargs)

        async def failing_consume(self, db, **kwargs):
            raise RuntimeError("boom at the end")

        if fail_at == "insert_version":
            monkeypatch.setattr(history, "insert_version", failing_insert)
        else:
            monkeypatch.setattr(
                ProceduralMemoryManager, "_consume_experiences", failing_consume
            )

        with pytest.raises(RuntimeError, match="boom"):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id=skill_id,
                fields={"description": "changed", "instructions": "all new"},
                consume_experience_ids=[exp_id],
                run_id="run-x:b0",
            )
        # Row unchanged, no baseline, no version, no object — and the
        # experience still pending.
        assert await self._state(t, skill_id) == before
        async with t.sm() as db:
            status = (
                await db.execute(
                    select(SkillExperienceORM.status).where(
                        SkillExperienceORM.id == exp_id
                    )
                )
            ).scalar_one()
        assert status == "pending"

    async def test_a_failed_create_leaves_no_row(self, t, monkeypatch):
        async def failing_insert(db, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(history, "insert_version", failing_insert)
        objects = await _objects(t)
        with pytest.raises(RuntimeError):
            await _create(t, "never")
        assert await _count(t, ProceduralORM, ProceduralORM.name == "never") == 0
        assert await _objects(t) == objects


class TestConsumeExperiences:
    async def _experience(self, t, user_id):
        exp_id = f"sexp-{uuid.uuid4().hex[:8]}"
        async with t.sm() as db:
            db.add(
                SkillExperienceORM(
                    id=exp_id,
                    organization_id=t.ids.org,
                    user_id=user_id,
                    agent_id=t.ids.agent,
                    session_id="s",
                    experience_type="worth_learning",
                    title="t",
                    content="c",
                    importance=0.5,
                    credibility=0.5,
                    evidence="",
                    status="pending",
                )
            )
            await db.commit()
        return exp_id

    async def test_the_owners_pending_experiences_are_consumed_with_the_write(self, t):
        mine = await self._experience(t, t.ids.u1)
        foreign = await self._experience(t, t.ids.u2)
        created = await _create(t, "learned")
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "learned more"},
            run_id="run-9:b0",
            consume_experience_ids=[mine, foreign],
        )
        assert result.changed
        async with t.sm() as db:
            rows = {
                r.id: r
                for r in (
                    await db.execute(
                        select(SkillExperienceORM).where(
                            SkillExperienceORM.id.in_([mine, foreign])
                        )
                    )
                ).scalars()
            }
        assert rows[mine].status == "consumed" and rows[mine].consumed_by == "run-9:b0"
        assert rows[mine].influenced_skill_ids == [created.skill.id]
        assert rows[foreign].status == "pending"


# --------------------------------------------------------------------------- #
# The retrieval vector                                                         #
# --------------------------------------------------------------------------- #


class TestRetrievalVector:
    async def test_embedded_outside_any_session_and_only_when_the_text_changes(
        self, t, embed
    ):
        created = await _create(t, "vec", triggers=["t1"])
        assert embed.calls == ["vec\nt1\nHow to vec"]
        assert len(created.skill.retrieval_embedding) == MAX_EMBEDDING_DIM
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "no vector for this"},
            agent_state=t.agent_state,
        )
        assert len(embed.calls) == 1
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"triggers": ["t1", "t2"]},
        )  # no agent_state: the row's own embedding_config
        assert embed.calls[-1] == "vec\nt1\nt2\nHow to vec"
        assert set(embed.open_sessions_seen) == {0}

    async def test_a_failing_embedding_is_retried_then_stored_as_null(self, t, embed):
        embed.fail = True
        created = await _create(t, "no-vector")
        assert len(embed.calls) == 4  # 1 call + 3 retries
        assert created.skill.retrieval_embedding is None
        assert created.version.version_no == 1  # the write itself succeeded

    async def test_a_concurrent_change_of_the_text_re_prepares(
        self, t, embed, monkeypatch
    ):
        created = await _create(t, "racy")
        real_prepare = ProceduralMemoryManager._prepare_change
        state = {"n": 0}

        async def racing_prepare(self, *args, **kwargs):
            plan = await real_prepare(self, *args, **kwargs)
            state["n"] += 1
            if state["n"] == 1:
                # Another writer renames the skill after the vector was computed.
                async with t.sm() as db:
                    row = (
                        await db.execute(
                            select(ProceduralORM).where(
                                ProceduralORM.id == created.skill.id
                            )
                        )
                    ).scalar_one()
                    row.name = "racy-renamed"
                    await db.commit()
            return plan

        monkeypatch.setattr(ProceduralMemoryManager, "_prepare_change", racing_prepare)
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"description": "fresh"},
        )
        assert state["n"] == 2  # prepared again after the stale plan
        assert result.skill.name == "racy-renamed"
        assert embed.calls[-1] == "racy-renamed\nwhen racy\nfresh"


# --------------------------------------------------------------------------- #
# Soft delete                                                                  #
# --------------------------------------------------------------------------- #


class TestSoftDelete:
    async def test_delete_version_reason_and_row_state(self, t):
        created = await _create(t, "retire")
        result = await t.mgr.soft_delete_procedure_by_id(
            created.skill.id,
            actor=t.client(),
            user=t.user(),
            reason="superseded by deploy-v2",
            run_id="run-3:b0",
            actor_kind=ACTOR_CURATOR,
        )
        assert result.version.change_type == "delete"
        assert result.version.reason == "superseded by deploy-v2"
        async with t.sm() as db:
            row = (
                await db.execute(
                    select(ProceduralORM).where(ProceduralORM.id == created.skill.id)
                )
            ).scalar_one()
        assert row.is_deleted is True
        assert row.last_modify["operation"] == "deleted"
        assert row.last_modify["reason"] == "superseded by deploy-v2"
        hist = await t.mgr.get_skill_history(
            created.skill.id, organization_id=t.ids.org, user_id=t.ids.u1
        )
        assert hist["is_deleted"] is True
        assert hist["delete_reason"] == "superseded by deploy-v2"
        assert hist["deleted_at"] is not None

    async def test_every_read_path_excludes_a_deleted_skill(self, t):
        keep = await _create(t, "keep-me", instructions="kubectl keep")
        gone = await _create(t, "gone-soon", instructions="kubectl gone")
        await t.mgr.soft_delete_procedure_by_id(
            gone.skill.id, actor=t.client(), user=t.user()
        )
        user, actor, agent_state = t.user(), t.client(), t.agent_state

        with pytest.raises(NoResultFound):
            await t.mgr.get_item_by_id(
                item_id=gone.skill.id, user=user, timezone_str="UTC", actor=actor
            )
        for method, field in [
            ("bm25", "instructions"),
            ("string_match", "instructions"),
            ("fuzzy_match", "instructions"),
            ("bm25", ""),
        ]:
            results = await t.mgr.list_procedures(
                agent_state=agent_state,
                user=user,
                query="kubectl",
                search_field=field,
                search_method=method,
            )
            ids = {r.id for r in results}
            assert keep.skill.id in ids and gone.skill.id not in ids, method
        recent = await t.mgr.list_procedures(
            agent_state=agent_state, user=user, query=""
        )
        assert gone.skill.id not in {r.id for r in recent}
        by_org = await t.mgr.list_procedures_by_org(
            agent_state=agent_state, organization_id=t.ids.org, query=""
        )
        assert gone.skill.id not in {r.id for r in by_org}
        # The count and the "most recent" read.
        assert await t.mgr.get_total_number_of_items(user) == 1
        latest = await t.mgr.get_most_recently_updated_item(user)
        assert [r.id for r in latest] == [keep.skill.id]

    async def test_a_users_only_skill_deleted_leaves_nothing_to_read(self, t):
        created = await _create(t, "lonely", user_id=t.ids.u2)
        await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user(t.ids.u2)
        )
        # Neither the "most recent" read nor the count sees it.
        assert await t.mgr.get_most_recently_updated_item(t.user(t.ids.u2)) is None
        assert await t.mgr.get_total_number_of_items(t.user(t.ids.u2)) == 0

    async def test_a_deleted_name_can_be_reused(self, t):
        first = await _create(t, "phoenix")
        await t.mgr.soft_delete_procedure_by_id(
            first.skill.id, actor=t.client(), user=t.user()
        )
        again = await t.mgr.insert_procedure(
            agent_state=t.agent_state,
            agent_id=t.ids.agent,
            name="phoenix",
            description="reborn",
            instructions="i",
            entry_type="workflow",
            actor=t.client(),
            organization_id=t.ids.org,
            user_id=t.ids.u1,
        )
        assert again.id != first.skill.id and again.name == "phoenix"

    async def test_deleting_or_editing_a_deleted_skill_is_not_found(self, t):
        created = await _create(t, "twice")
        await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user()
        )
        with pytest.raises(NoResultFound):
            await t.mgr.soft_delete_procedure_by_id(
                created.skill.id, actor=t.client(), user=t.user()
            )
        with pytest.raises(NoResultFound):
            await t.mgr.update_item(
                ProceduralMemoryItemUpdate(id=created.skill.id, description="x"),
                user=t.user(),
                actor=t.client(),
            )

    async def test_bulk_soft_deletes_write_delete_versions(self, t):
        a = await _create(t, "bulk-a")
        b = await _create(t, "bulk-b", user_id=t.ids.u2)
        assert await t.mgr.soft_delete_by_user_id(t.ids.u1) == 1
        assert [v.change_type for v in await _versions(t, a.skill.id)] == [
            "create",
            "delete",
        ]
        assert await t.mgr.soft_delete_by_client_id(t.client()) == 1  # b (a is gone)
        vb = await _versions(t, b.skill.id)
        assert [v.change_type for v in vb] == ["create", "delete"]
        assert vb[-1].reason == "client deleted"


# --------------------------------------------------------------------------- #
# Restore                                                                      #
# --------------------------------------------------------------------------- #


class TestRestore:
    async def test_restored_content_is_byte_identical(self, t):
        original = dict(
            description="Déployer — the original ✓",
            instructions="1. build\n2. ship\n",
            triggers=["on release", "ünïcode"],
            examples=[{"output": {"z": 1, "a": [1, 2]}, "input": "go"}],
            filter_tags={"scope": "test", "tags": ["b", "a"]},
        )
        created = await _create(t, "restore-me", **original)
        original_content = history.content_of(created.skill)
        original_canonical = {
            f: history.canonical_bytes(f, original_content[f])
            for f in SKILL_VERSIONED_FIELDS
        }
        for step in range(2):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id=created.skill.id,
                fields={
                    "description": f"changed {step}",
                    "triggers": [f"t{step}"],
                    "examples": [],
                    "instructions": f"other {step}",
                },
            )
        stored = await t.mgr.get_skill_version(
            created.skill.id, 1, organization_id=t.ids.org, user_id=t.ids.u1
        )
        assert stored["canonical"] == original_canonical
        assert stored["content"] == original_content

        result = await t.mgr.restore_skill(
            created.skill.id, 1, actor=t.client(), user_id=t.ids.u1
        )
        assert result.version.change_type == "restore"
        assert result.version.version_no == 4
        assert result.version.reason == "restored version 1"
        restored = history.content_of(result.skill)
        assert {
            f: history.canonical_bytes(f, restored[f]) for f in SKILL_VERSIONED_FIELDS
        } == original_canonical
        assert result.skill.version == "0.1.1"  # a restore is a new change
        # The hashes of the restore equal version 1's.
        versions = await _versions(t, created.skill.id)
        assert history.version_hashes(versions[-1]) == history.version_hashes(
            versions[0]
        )

    async def test_restoring_a_deleted_skill_undeletes_it(self, t):
        created = await _create(t, "lazarus")
        await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user()
        )
        result = await t.mgr.restore_skill(
            created.skill.id, 1, actor=t.client(), user_id=t.ids.u1
        )
        assert result.changed
        found = await t.mgr.get_item_by_id(
            item_id=created.skill.id,
            user=t.user(),
            timezone_str="UTC",
            actor=t.client(),
        )
        assert found.name == "lazarus"
        assert [v.change_type for v in await _versions(t, created.skill.id)] == [
            "create",
            "delete",
            "restore",
        ]

    async def test_restoring_onto_a_live_name_conflicts_and_writes_nothing(self, t):
        created = await _create(t, "contested")
        await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user()
        )
        holder = await _create(t, "contested", description="the new holder")
        # The explicit check (not only the unique index) knows the holder, but
        # its message never names it: the holder may be of a scope the caller
        # cannot read.
        with pytest.raises(SkillNameConflictError) as conflict:
            await t.mgr.restore_skill(
                created.skill.id, 1, actor=t.client(), user_id=t.ids.u1
            )
        assert conflict.value.holder_id == holder.skill.id
        assert holder.skill.id not in str(conflict.value)
        assert len(await _versions(t, created.skill.id)) == 2

    async def test_a_scoped_restore_needs_the_skill_and_the_version_in_scope(self, t):
        # With scopes, the skill (on the locked row) and the restored
        # version must both be of one of them.
        moved = await _create(t, "moved", filter_tags={"scope": "other"})
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=moved.skill.id,
            fields={"filter_tags": {"scope": "test"}, "description": "now test"},
        )
        with pytest.raises(SkillVersionNotFound):
            await t.mgr.restore_skill(
                moved.skill.id, 1, actor=t.client(), user_id=t.ids.u1, scopes=["test"]
            )
        foreign = await _create(t, "foreign", filter_tags={"scope": "other"})
        with pytest.raises(NoResultFound):
            await t.mgr.restore_skill(
                foreign.skill.id, 1, actor=t.client(), user_id=t.ids.u1, scopes=["test"]
            )
        assert len(await _versions(t, moved.skill.id)) == 2
        assert len(await _versions(t, foreign.skill.id)) == 1
        restored = await t.mgr.restore_skill(
            moved.skill.id, 2, actor=t.client(), user_id=t.ids.u1, scopes=["test"]
        )
        assert not restored.changed  # version 2 is the current content

    async def test_restoring_an_old_name_taken_by_another_skill_conflicts(self, t):
        renamed = await _create(t, "old-name")
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=renamed.skill.id,
            fields={"name": "new-name"},
        )
        await _create(t, "old-name")
        with pytest.raises(SkillNameConflictError):
            await t.mgr.restore_skill(
                renamed.skill.id, 1, actor=t.client(), user_id=t.ids.u1
            )

    async def test_unknown_version_and_other_owner(self, t):
        created = await _create(t, "bounded")
        with pytest.raises(SkillVersionNotFound):
            await t.mgr.restore_skill(
                created.skill.id, 9, actor=t.client(), user_id=t.ids.u1
            )
        assert issubclass(SkillVersionNotFound, NoResultFound)
        with pytest.raises(NoResultFound):
            await t.mgr.restore_skill(
                created.skill.id, 1, actor=t.client(), user_id=t.ids.u2
            )

    async def test_restoring_the_current_content_of_a_live_skill_changes_nothing(
        self, t
    ):
        created = await _create(t, "steady")
        result = await t.mgr.restore_skill(
            created.skill.id, 1, actor=t.client(), user_id=t.ids.u1
        )
        assert not result.changed
        assert len(await _versions(t, created.skill.id)) == 1


# --------------------------------------------------------------------------- #
# Hard delete = erasure                                                        #
# --------------------------------------------------------------------------- #


class TestErasure:
    async def test_hard_delete_purges_history_but_keeps_shared_objects(self, t):
        shared = dict(
            description="shared d", instructions="shared i", triggers=[], examples=[]
        )
        doomed = await _create(t, "doomed", **shared)
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=doomed.skill.id,
            fields={"instructions": "only doomed has this"},
        )
        survivor = await _create(t, "survivor", **shared)
        await t.mgr.delete_procedure_by_id(
            doomed.skill.id, actor=t.client(), user=t.user()
        )

        assert await _count(t, ProceduralORM, ProceduralORM.id == doomed.skill.id) == 0
        assert (
            await _count(
                t, SkillVersionORM, SkillVersionORM.skill_id == doomed.skill.id
            )
            == 0
        )
        # Only the survivor's objects remain, and its history still rebuilds.
        survivor_version = (await _versions(t, survivor.skill.id))[0]
        assert await _objects(t) == len(
            set(history.version_hashes(survivor_version).values())
        )
        rebuilt = await t.mgr.get_skill_version(
            survivor.skill.id, 1, organization_id=t.ids.org, user_id=t.ids.u1
        )
        assert rebuilt["content"]["instructions"] == "shared i"

    async def test_hard_delete_checks_ownership(self, t):
        created = await _create(t, "owned")
        with pytest.raises(NoResultFound):
            await t.mgr.delete_procedure_by_id(
                created.skill.id, actor=t.client(), user_id=t.ids.u2
            )
        with pytest.raises(NoResultFound):
            await t.mgr.delete_procedure_by_id(
                created.skill.id, actor=t.client(org=t.ids.other_org), user_id=t.ids.u1
            )
        assert len(await _versions(t, created.skill.id)) == 1

    async def test_hard_delete_of_a_soft_deleted_skill(self, t):
        created = await _create(t, "soft-then-hard")
        await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user()
        )
        await t.mgr.delete_procedure_by_id(
            created.skill.id, actor=t.client(), user_id=t.ids.u1
        )
        assert (
            await _count(
                t, SkillVersionORM, SkillVersionORM.skill_id == created.skill.id
            )
            == 0
        )
        with pytest.raises(NoResultFound):
            await t.mgr.get_skill_history(
                created.skill.id, organization_id=t.ids.org, user_id=t.ids.u1
            )

    async def test_user_erasure_empties_the_users_history(self, t):
        a = await _create(t, "erase-a")
        await t.mgr.soft_delete_procedure_by_id(
            a.skill.id, actor=t.client(), user=t.user()
        )
        await _create(t, "erase-b")
        other = await _create(
            t, "erase-a", user_id=t.ids.u2
        )  # same content except owner
        assert await t.mgr.delete_by_user_id(t.ids.u1) == 2
        assert (
            await _count(t, SkillVersionORM, SkillVersionORM.user_id == t.ids.u1) == 0
        )
        assert await _count(t, ProceduralORM, ProceduralORM.user_id == t.ids.u1) == 0
        # The other user's skill, history and objects are intact.
        rebuilt = await t.mgr.get_skill_version(
            other.skill.id, 1, organization_id=t.ids.org, user_id=t.ids.u2
        )
        assert rebuilt["content"]["name"] == "erase-a"
        assert await _objects(t) == 7

    async def test_client_erasure(self, t):
        mine = await _create(t, "client-a")
        theirs = await _create(t, "client-b", actor=t.client(t.ids.c2))
        assert await t.mgr.delete_by_client_id(t.client()) == 1
        assert (
            await _count(t, SkillVersionORM, SkillVersionORM.skill_id == mine.skill.id)
            == 0
        )
        assert len(await _versions(t, theirs.skill.id)) == 1

    async def test_organization_deletion_purges_its_history(self, t, monkeypatch):
        from mirix.services.organization_manager import OrganizationManager

        await _create(t, "org-bound")
        manager = OrganizationManager()
        manager.session_maker = t.sm
        await manager.delete_organization_by_id(t.ids.org)
        assert (
            await _count(
                t, SkillVersionORM, SkillVersionORM.organization_id == t.ids.org
            )
            == 0
        )
        assert await _objects(t) == 0

    async def test_a_scoped_hard_delete_spares_other_scopes(self, t):
        # The scope is part of the purge's own (locked) selection.
        foreign = await _create(t, "foreign-erase", filter_tags={"scope": "other"})
        with pytest.raises(NoResultFound):
            await t.mgr.delete_procedure_by_id(
                foreign.skill.id, actor=t.client(), user_id=t.ids.u1, scopes=["test"]
            )
        assert len(await _versions(t, foreign.skill.id)) == 1
        mine = await _create(t, "mine-erase")
        await t.mgr.delete_procedure_by_id(
            mine.skill.id, actor=t.client(), user_id=t.ids.u1, scopes=["test"]
        )
        assert await _count(t, ProceduralORM, ProceduralORM.id == mine.skill.id) == 0

    async def test_a_purge_that_reaches_an_unlocked_organization_locks_it_first(
        self, t, monkeypatch
    ):
        # The organizations to lock come from an unlocked first read. A
        # skill written meanwhile in ANOTHER organization is then only seen by
        # the locked re-read — the purge must not collect that organization's
        # objects without its exclusive lock: it starts over with it locked.
        await _create(t, "home-org")
        real_lock = history.lock_for_purge
        locked = []
        state = {"injected": False}

        async def lock_after_a_concurrent_create(db, *, organization_ids):
            locked.append(sorted(organization_ids))
            if not state["injected"]:
                state["injected"] = True
                await _create(
                    t,
                    "elsewhere",
                    organization_id=t.ids.other_org,
                    instructions="only in the other organization",
                )
            await real_lock(db, organization_ids=organization_ids)

        monkeypatch.setattr(history, "lock_for_purge", lock_after_a_concurrent_create)
        assert await t.mgr.delete_by_user_id(t.ids.u1) == 2
        assert locked == [[t.ids.org], sorted([t.ids.org, t.ids.other_org])]
        assert (
            await _count(t, SkillVersionORM, SkillVersionORM.user_id == t.ids.u1) == 0
        )
        assert await _objects(t, t.ids.other_org) == 0


# --------------------------------------------------------------------------- #
# Read-modify-write edits and ambiguous commits                                #
# --------------------------------------------------------------------------- #


class TestOptimisticEdit:
    async def _instructions(self, t, skill_id):
        async with t.sm() as db:
            return (
                await db.execute(
                    select(ProceduralORM.instructions, ProceduralORM.version).where(
                        ProceduralORM.id == skill_id
                    )
                )
            ).one()

    async def test_an_edit_from_a_stale_read_is_refused_and_writes_nothing(self, t):
        created = await _create(t, "rmw", instructions="step A\nstep B\nstep C")
        stale = created.skill.instructions
        # A concurrent writer commits first.
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "step A\nstep B\nstep C\nstep D"},
        )
        with pytest.raises(SkillChangedMeanwhile) as changed:
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id=created.skill.id,
                fields={"instructions": stale.replace("step B", "step X")},
                expected_fields={"instructions": stale},
            )
        assert changed.value.fields == ["instructions"]
        instructions, _ = await self._instructions(t, created.skill.id)
        assert instructions.endswith("step D")  # the concurrent change survives
        assert len(await _versions(t, created.skill.id)) == 2

    async def test_the_locked_row_is_checked_not_only_the_first_read(
        self, t, monkeypatch
    ):
        created = await _create(t, "rmw-locked", instructions="one two three")
        real_commit = ProceduralMemoryManager._commit_change
        state = {"raced": False}

        async def race_then_commit(self, plan, **kwargs):
            if not state["raced"]:
                state["raced"] = True  # after the prepare step read the row
                await real_commit(
                    self,
                    await self._prepare_change(
                        "edit",
                        actor=t.client(),
                        user_id=t.ids.u1,
                        skill_id=plan.skill_id,
                        fields={"instructions": "one two three four"},
                        version_no=None,
                        agent_state=None,
                        organization_id=None,
                        allow_stale_embedding=True,
                    ),
                    actor=t.client(),
                    run_id=None,
                    actor_kind=None,
                    reason=None,
                    consume_experience_ids=None,
                )
            return await real_commit(self, plan, **kwargs)

        monkeypatch.setattr(ProceduralMemoryManager, "_commit_change", race_then_commit)
        with pytest.raises(SkillChangedMeanwhile):
            await t.mgr.apply_skill_change(
                "edit",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id=created.skill.id,
                fields={"instructions": "one 2 three"},
                expected_fields={"instructions": "one two three"},
            )
        instructions, _ = await self._instructions(t, created.skill.id)
        assert instructions == "one two three four"

    async def test_the_version_string_is_bumped_from_the_locked_row(self, t):
        created = await _create(t, "bumped", version="0.1.0")
        await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"description": "concurrent", "version": "0.1.1"},
        )
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "fresh"},
            expected_fields={"instructions": created.skill.instructions},
            bump_version=True,
        )
        assert result.skill.version == "0.1.2"
        # No version written -> no bump.
        same = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "fresh"},
            bump_version=True,
        )
        assert not same.changed and same.skill.version == "0.1.2"

    async def test_expected_fields_are_only_for_an_edit(self, t):
        with pytest.raises(ValueError):
            await t.mgr.apply_skill_change(
                "delete",
                actor=t.client(),
                user_id=t.ids.u1,
                skill_id="proc-x",
                expected_fields={"name": "x"},
            )


class _Busy(Exception):
    sqlite_errorcode = 5  # SQLITE_BUSY: transient


def _commit_then_fail(monkeypatch, *, times=1):
    """Every skill write's transaction commits, then (the first ``times``
    times) reports a transient failure — a COMMIT whose success the client
    never saw. The write is repeated and must report the success."""
    from sqlalchemy.exc import OperationalError

    real = ProceduralMemoryManager._commit_change
    state = {"left": times, "calls": 0}

    async def commit_then_fail(self, plan, **kwargs):
        state["calls"] += 1
        result = await real(self, plan, **kwargs)
        if state["left"] > 0:
            state["left"] -= 1
            raise OperationalError("COMMIT", {}, _Busy("database is locked"))
        return result

    monkeypatch.setattr(ProceduralMemoryManager, "_commit_change", commit_then_fail)
    return state


class TestAmbiguousCommit:
    async def test_a_repeated_create_returns_the_created_skill(self, t, monkeypatch):
        state = _commit_then_fail(monkeypatch)
        result = await _create(t, "created-once")
        assert state["calls"] == 2
        assert result.changed and result.version.version_no == 1
        assert result.skill.name == "created-once"
        assert await _count(t, ProceduralORM, ProceduralORM.name == "created-once") == 1
        assert len(await _versions(t, result.skill.id)) == 1

    async def test_a_repeated_delete_reports_the_delete(self, t, monkeypatch):
        created = await _create(t, "deleted-once")
        state = _commit_then_fail(monkeypatch)
        result = await t.mgr.soft_delete_procedure_by_id(
            created.skill.id, actor=t.client(), user=t.user(), reason="gone"
        )
        assert state["calls"] == 2
        assert result.changed and result.version.change_type == "delete"
        assert (
            await _count(
                t,
                ProceduralORM,
                ProceduralORM.id == created.skill.id,
                ProceduralORM.is_deleted,
            )
            == 1
        )
        assert [v.change_type for v in await _versions(t, created.skill.id)] == [
            "create",
            "delete",
        ]

    async def test_a_repeated_edit_writes_one_version(self, t, monkeypatch):
        created = await _create(t, "edited-once")
        _commit_then_fail(monkeypatch)
        result = await t.mgr.apply_skill_change(
            "edit",
            actor=t.client(),
            user_id=t.ids.u1,
            skill_id=created.skill.id,
            fields={"instructions": "edited"},
            expected_fields={"instructions": created.skill.instructions},
        )
        assert result.changed and result.skill.instructions == "edited"
        assert len(await _versions(t, created.skill.id)) == 2


# --------------------------------------------------------------------------- #
# Backfill script (idempotent)                                                 #
# --------------------------------------------------------------------------- #


class TestBackfillScript:
    async def test_backfill_is_idempotent(self, t, monkeypatch):
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "backfill_skill_baseline_versions.py"
        )
        spec = importlib.util.spec_from_file_location("backfill_script", path)
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolve annotations through sys.modules.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)

        live = await _legacy_skill(t, "bf-live")
        deleted = await _legacy_skill(t, "bf-deleted", is_deleted=True)
        versioned = await _create(t, "bf-versioned")

        dry = await module.backfill_baseline_versions(
            manager=t.mgr, organization_id=t.ids.org, dry_run=True
        )
        assert (dry.scanned, dry.written) == (2, 2)
        assert (
            await _count(
                t, SkillVersionORM, SkillVersionORM.skill_id.in_([live, deleted])
            )
            == 0
        )

        first = await module.backfill_baseline_versions(
            manager=t.mgr, organization_id=t.ids.org, batch_size=1
        )
        assert (first.scanned, first.written, first.skipped) == (2, 2, 0)
        for skill_id in (live, deleted):
            versions = await _versions(t, skill_id)
            assert [(v.version_no, v.change_type, v.actor_kind) for v in versions] == [
                (1, "baseline", "backfill")
            ]
            rebuilt = await t.mgr.get_skill_version(
                skill_id, 1, organization_id=t.ids.org, user_id=t.ids.u1
            )
            assert rebuilt["content"]["instructions"] == "legacy instructions"
        assert len(await _versions(t, versioned.skill.id)) == 1

        second = await module.backfill_baseline_versions(
            manager=t.mgr, organization_id=t.ids.org
        )
        assert (second.scanned, second.written) == (0, 0)
        # The deleted skill can now be restored.
        restored = await t.mgr.restore_skill(
            deleted, 1, actor=t.client(), user_id=t.ids.u1
        )
        assert restored.changed
        found = await t.mgr.get_item_by_id(
            item_id=deleted, user=t.user(), timezone_str="UTC", actor=t.client()
        )
        assert found.name == "bf-deleted"
