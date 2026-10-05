"""A skill's retrieval vector.

Hermetic: a throwaway SQLite database and a recording fake embedding model
whose vector depends on the text (and names the config it was built from); no
model provider, no Redis.

* the vector is the retrieval text's (name, triggers one per line,
  description) and is recomputed when any of the three changes — a triggers
  change moves the stored vector — never for an instructions-only edit; with
  no agent_state the row's own ``embedding_config`` is used;
* ``reembed_skill`` fills a live skill left without a vector, through the one
  write path, without a new version and without touching its modification
  stamp, and loses no concurrent change;
* a re-embedding pass (the startup task and ``scripts/reembed_skill_retrieval.py``)
  is idempotent, bounded in concurrency, always ends, and can be cancelled.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services import skill_history as history
from mirix.services import skill_retrieval_reembed as reembed
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

REPO = Path(__file__).resolve().parent.parent


def _config(model: str) -> EmbeddingConfig:
    # An unreachable local endpoint: a path escaping the fake fails fast.
    return EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="http://127.0.0.1:9/v1",
        embedding_model=model,
        embedding_dim=8,
    )


def _vector(text: str, model: str):
    digest = hashlib.sha256(f"{model}|{text}".encode()).digest()
    return [b / 255.0 + 0.01 for b in digest[:8]]


def _stored(text: str, model: str):
    return _vector(text, model) + [0.0] * (MAX_EMBEDDING_DIM - 8)


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("reembed") / "test.db"
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


class _Embed:
    """The fake provider: records (model, text); fails for texts in ``fail``;
    may wait on ``gate`` (to hold calls in flight)."""

    def __init__(self):
        self.calls = []
        self.fail = set()
        self.gate = None
        self.in_flight = 0
        self.max_in_flight = 0

    def factory(self):
        state = self

        class Model:
            def __init__(self, config):
                self.model = config.embedding_model

            async def get_text_embedding(self, text):
                state.calls.append((self.model, text))
                state.in_flight += 1
                state.max_in_flight = max(state.max_in_flight, state.in_flight)
                try:
                    if state.gate is not None:
                        await state.gate.wait()
                    else:
                        await asyncio.sleep(0)
                    if text in state.fail:
                        raise ConnectionError("embedding endpoint down")
                    return _vector(text, self.model)
                finally:
                    state.in_flight -= 1

        async def embedding_model(config, *args, **kwargs):
            return Model(config)

        return embedding_model


@pytest.fixture
def embed(monkeypatch):
    state = _Embed()
    monkeypatch.setattr(pmm, "embedding_model", state.factory())
    return state


async def _no_sleep(_seconds):
    return None


@pytest_asyncio.fixture(loop_scope="module")
async def t(sm, embed):
    org = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.flush()
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org,
                status="active",
                timezone="UTC",
            )
        )
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org,
                name=client_id,
                status="active",
                write_scope="test",
                read_scopes=["test"],
            )
        )
        db.add(AgentORM(id=agent_id, organization_id=org))
        await db.commit()

    def agent_state(model="agent-model"):
        return AgentState(
            id=agent_id,
            name="proc",
            system="s",
            agent_type=AgentType.procedural_memory_agent,
            llm_config=LLMConfig(
                model="x", model_endpoint_type="openai", context_window=8000
            ),
            embedding_config=_config(model),
            tools=[],
        )

    return SimpleNamespace(
        org=org,
        user_id=user_id,
        agent_id=agent_id,
        actor=PydanticClient(
            id=client_id,
            organization_id=org,
            name=client_id,
            write_scope="test",
            read_scopes=["test"],
        ),
        agent_state=agent_state,
        mgr=ProceduralMemoryManager(session_maker=sm, sleep=_no_sleep),
        sm=sm,
        embed=embed,
    )


async def _create(t, name, **overrides):
    fields = dict(
        name=name,
        entry_type="workflow",
        description=f"How to {name}",
        instructions=f"Step 1: {name}.",
        triggers=[f"when {name}"],
        examples=[],
        filter_tags={"scope": "test"},
        agent_id=t.agent_id,
        organization_id=t.org,
    )
    fields.update(overrides)
    result = await t.mgr.apply_skill_change(
        "create",
        actor=t.actor,
        user_id=t.user_id,
        fields=fields,
        agent_state=t.agent_state(),
    )
    return result.skill


async def _edit(t, skill_id, *, agent_state=None, **fields):
    result = await t.mgr.apply_skill_change(
        "edit",
        actor=t.actor,
        user_id=t.user_id,
        skill_id=skill_id,
        fields=fields,
        agent_state=agent_state,
    )
    return result.skill


async def _row(t, skill_id):
    async with t.sm() as db:
        return (
            await db.execute(select(ProceduralORM).where(ProceduralORM.id == skill_id))
        ).scalar_one()


async def _legacy(t, name, *, config="row-model", deleted=False, triggers=None):
    """A skill as the retrieval-field migration leaves it: no vector (and no
    history)."""
    skill_id = f"proc-{uuid.uuid4().hex[:8]}"
    async with t.sm() as db:
        db.add(
            ProceduralORM(
                id=skill_id,
                organization_id=t.org,
                user_id=t.user_id,
                agent_id=t.agent_id,
                name=name,
                entry_type="guide",
                description=f"legacy {name}",
                instructions="legacy instructions",
                triggers=triggers if triggers is not None else [f"when {name}"],
                examples=[],
                filter_tags={"scope": "test"},
                version="0.3.0",
                last_modify={
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "operation": "created",
                },
                embedding_config=_config(config) if config else None,
                is_deleted=deleted,
            )
        )
        await db.commit()
    return skill_id


def _text(name, triggers, description):
    return history.skill_retrieval_text(
        {"name": name, "triggers": triggers, "description": description}
    )


# --------------------------------------------------------------------------- #
# The vector follows the retrieval text                                        #
# --------------------------------------------------------------------------- #


class TestRetrievalTextVector:
    def test_the_retrieval_text(self):
        assert _text("deploy", ["on release", "when asked"], "Ship it.") == (
            "deploy\non release\nwhen asked\nShip it."
        )
        assert _text("n", [], "") == "n"

    async def test_a_triggers_change_moves_the_stored_vector(self, t):
        skill = await _create(t, "vec-a", triggers=["t1"])
        first = _text("vec-a", ["t1"], "How to vec-a")
        assert skill.retrieval_embedding == _stored(first, "agent-model")

        edited = await _edit(t, skill.id, triggers=["t1", "t2"])
        second = _text("vec-a", ["t1", "t2"], "How to vec-a")
        assert edited.retrieval_embedding == _stored(second, "agent-model")
        assert edited.retrieval_embedding != skill.retrieval_embedding
        row = await _row(t, skill.id)
        assert list(row.retrieval_embedding) == pytest.approx(
            _stored(second, "agent-model")
        )

    @pytest.mark.parametrize(
        "field, value",
        [("name", "renamed"), ("description", "New description")],
    )
    async def test_a_name_or_description_change_moves_it_too(self, t, field, value):
        skill = await _create(t, f"vec-{field}")
        edited = await _edit(t, skill.id, **{field: value})
        values = {
            "name": skill.name,
            "triggers": skill.triggers,
            "description": skill.description,
        }
        values[field] = value
        assert edited.retrieval_embedding == _stored(
            history.skill_retrieval_text(values), "agent-model"
        )

    async def test_an_instructions_edit_keeps_the_vector_and_calls_nothing(self, t):
        skill = await _create(t, "vec-i")
        t.embed.calls.clear()
        edited = await _edit(t, skill.id, instructions="Other steps.")
        assert t.embed.calls == []
        assert edited.retrieval_embedding == pytest.approx(skill.retrieval_embedding)

    async def test_without_an_agent_state_the_rows_own_config_is_used(self, t):
        skill = await _create(t, "vec-cfg")  # the row keeps agent-model's config
        t.embed.calls.clear()
        # With an agent_state its config is used (and becomes the row's) ...
        await _edit(
            t, skill.id, agent_state=t.agent_state("other-model"), triggers=["x"]
        )
        assert [m for m, _ in t.embed.calls] == ["other-model"]
        # ... without one, the row's own.
        edited = await _edit(t, skill.id, triggers=["y"])
        assert [m for m, _ in t.embed.calls] == ["other-model", "other-model"]
        assert edited.retrieval_embedding == _stored(
            _text("vec-cfg", ["y"], "How to vec-cfg"), "other-model"
        )


# --------------------------------------------------------------------------- #
# reembed_skill                                                               #
# --------------------------------------------------------------------------- #


class TestReembedSkill:
    async def test_a_skill_without_a_vector_is_filled_quietly(self, t):
        skill_id = await _legacy(t, "q-fill", triggers=["a", "b"])
        before = await _row(t, skill_id)
        assert await t.mgr.reembed_skill(skill_id) == pmm.REEMBED_FILLED
        after = await _row(t, skill_id)
        # The retrieval text's vector, with the row's OWN config.
        text = _text("q-fill", ["a", "b"], "legacy q-fill")
        assert t.embed.calls == [("row-model", text)]
        assert list(after.retrieval_embedding) == pytest.approx(
            _stored(text, "row-model")
        )
        # A vector is not a change of the skill: no version of its own (only
        # the baseline every first write records), same modification stamp.
        async with t.sm() as db:
            versions = await history.list_versions(db, skill_id)
        assert [v.change_type for v in versions] == ["baseline"]
        assert after.last_modify == before.last_modify
        assert after.updated_at == before.updated_at
        assert after.version == before.version
        # Idempotent.
        assert await t.mgr.reembed_skill(skill_id) == pmm.REEMBED_PRESENT
        assert len(t.embed.calls) == 1

    async def test_what_cannot_be_filled(self, t):
        no_config = await _legacy(t, "q-noconf", config=None)
        assert await t.mgr.reembed_skill(no_config) == pmm.REEMBED_NO_CONFIG
        deleted = await _legacy(t, "q-deleted", deleted=True)
        assert await t.mgr.reembed_skill(deleted) == pmm.REEMBED_GONE
        assert await t.mgr.reembed_skill("proc-missing") == pmm.REEMBED_GONE
        assert t.embed.calls == []
        failing = await _legacy(t, "q-fail")
        t.embed.fail.add(_text("q-fail", ["when q-fail"], "legacy q-fail"))
        assert await t.mgr.reembed_skill(failing) == pmm.REEMBED_FAILED
        assert (await _row(t, failing)).retrieval_embedding is None
        assert len(t.embed.calls) == 4  # the call and its 3 retries

    async def test_embeddings_off(self, t, monkeypatch):
        skill_id = await _legacy(t, "q-off")
        monkeypatch.setattr(pmm, "BUILD_EMBEDDINGS_FOR_MEMORY", False)
        assert await t.mgr.reembed_skill(skill_id) == pmm.REEMBED_DISABLED
        report = await reembed.reembed_missing_retrieval_embeddings(manager=t.mgr)
        assert report.scanned == 0 and t.embed.calls == []

    async def test_a_concurrent_change_of_the_text_wins(self, t, monkeypatch):
        skill_id = await _legacy(t, "q-race")
        real = ProceduralMemoryManager._embed

        raced = []

        async def embed_then_someone_edits(self, text_value, config):
            vector = await real(self, text_value, config)
            if not raced:
                raced.append(text_value)
                # While the re-embedding computed its vector, a writer changed
                # the description (and embedded the new text itself).
                await _edit(t, skill_id, description="changed meanwhile")
            return vector

        monkeypatch.setattr(ProceduralMemoryManager, "_embed", embed_then_someone_edits)
        assert await t.mgr.reembed_skill(skill_id) == pmm.REEMBED_CHANGED
        row = await _row(t, skill_id)
        assert row.description == "changed meanwhile"
        # The writer's vector (of the new text) stands, not the stale one.
        new_text = _text("q-race", ["when q-race"], "changed meanwhile")
        assert list(row.retrieval_embedding) == pytest.approx(
            _stored(new_text, "row-model")
        )

    async def test_listing_the_skills_without_a_vector(self, t):
        a = await _legacy(t, "l-a")
        b = await _legacy(t, "l-b")
        await _legacy(t, "l-deleted", deleted=True)
        await _create(t, "l-has-vector")
        mine = await t.mgr.list_skill_ids_missing_retrieval_embedding(
            organization_id=t.org
        )
        assert set(mine) >= {a, b}
        for skill_id in mine:
            row = await _row(t, skill_id)
            assert row.retrieval_embedding is None and not row.is_deleted
        assert mine == sorted(mine)
        first = await t.mgr.list_skill_ids_missing_retrieval_embedding(
            organization_id=t.org, limit=1
        )
        rest = await t.mgr.list_skill_ids_missing_retrieval_embedding(
            organization_id=t.org, after_id=first[0]
        )
        assert first + rest == mine
        assert (
            await t.mgr.list_skill_ids_missing_retrieval_embedding(
                organization_id="org-none"
            )
            == []
        )


# --------------------------------------------------------------------------- #
# The pass (startup task and script)                                           #
# --------------------------------------------------------------------------- #


class TestPass:
    async def test_a_pass_fills_what_it_can_ends_and_is_idempotent(self, t):
        good = [await _legacy(t, f"p-{i}") for i in range(5)]
        failing = await _legacy(t, "p-fail")
        no_config = await _legacy(t, "p-noconf", config=None)
        await _legacy(t, "p-deleted", deleted=True)
        t.embed.fail.add(_text("p-fail", ["when p-fail"], "legacy p-fail"))

        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org, batch_size=2, concurrency=3
        )
        assert report.scanned == 7
        assert report.outcomes == {"filled": 5, "failed": 1, "no_config": 1}
        assert report.failed == 1
        for skill_id in good:
            assert (await _row(t, skill_id)).retrieval_embedding is not None
        assert "filled 5" in report.summary()

        # Again: only what is still without a vector is looked at.
        t.embed.fail.clear()
        again = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org
        )
        assert again.outcomes == {"filled": 1, "no_config": 1}
        assert (await _row(t, failing)).retrieval_embedding is not None
        assert (await _row(t, no_config)).retrieval_embedding is None

    async def test_concurrency_is_bounded(self, t):
        for i in range(6):
            await _legacy(t, f"c-{i}")
        t.embed.gate = asyncio.Event()
        task = asyncio.ensure_future(
            reembed.reembed_missing_retrieval_embeddings(
                manager=t.mgr, organization_id=t.org, concurrency=2
            )
        )
        for _ in range(50):
            await asyncio.sleep(0.01)
            if t.embed.in_flight == 2:
                break
        assert t.embed.in_flight == 2
        t.embed.gate.set()
        report = await task
        assert report.outcomes == {"filled": 6}
        assert t.embed.max_in_flight == 2

    async def test_a_dry_run_only_counts(self, t):
        for i in range(3):
            await _legacy(t, f"d-{i}")
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org, dry_run=True
        )
        assert report.scanned == 3 and report.outcomes == {} and t.embed.calls == []
        assert "3 live skill(s)" in report.summary()

    async def test_bad_arguments(self, t):
        with pytest.raises(ValueError):
            await reembed.reembed_missing_retrieval_embeddings(
                manager=t.mgr, batch_size=0
            )
        with pytest.raises(ValueError):
            await reembed.reembed_missing_retrieval_embeddings(
                manager=t.mgr, concurrency=0
            )

    async def test_an_unexpected_error_of_one_skill_does_not_stop_the_pass(
        self, t, monkeypatch
    ):
        ids = [await _legacy(t, f"e-{i}") for i in range(3)]
        real = ProceduralMemoryManager.reembed_skill

        async def flaky(self, skill_id):
            if skill_id == ids[1]:
                raise RuntimeError("bug")
            return await real(self, skill_id)

        monkeypatch.setattr(ProceduralMemoryManager, "reembed_skill", flaky)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org
        )
        assert report.outcomes == {"filled": 2, "error": 1} and report.failed == 1


class TestOutageAndOverlap:
    """A pass during a sustained embedding outage stops
    after a few failures in a row instead of paying 4 calls (and ~7 s of
    backoff) for every skill; two passes that meet embed each skill once."""

    def _fail_all(self, t, names):
        for name in names:
            t.embed.fail.add(_text(name, [f"when {name}"], f"legacy {name}"))

    async def test_the_breaker_stops_a_pass_after_failures_in_a_row(self, t):
        names = [f"out-{i}" for i in range(10)]
        for name in names:
            await _legacy(t, name)
        self._fail_all(t, names)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr,
            organization_id=t.org,
            concurrency=1,
            max_consecutive_failures=3,
        )
        assert report.outcomes == {"failed": 3, "skipped": 7}
        assert report.stopped_early and report.skipped == 7
        assert report.still_missing == 10
        assert "stopped early" in report.summary()
        # 3 skills x (1 + 3 retries) provider calls, not 10 x 4.
        assert len(t.embed.calls) == 3 * 4

    async def test_it_stops_reading_pages_too(self, t):
        names = [f"page-{i}" for i in range(10)]
        for name in names:
            await _legacy(t, name)
        self._fail_all(t, names)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr,
            organization_id=t.org,
            batch_size=2,
            concurrency=1,
            max_consecutive_failures=3,
        )
        # Pages of 2: the third failure is on page 2, whose other skill is
        # skipped; pages 3-5 are never read.
        assert report.scanned == 4
        assert report.outcomes == {"failed": 3, "skipped": 1}
        assert report.stopped_early

    async def test_with_concurrency_at_most_the_started_skills_follow(self, t):
        names = [f"conc-{i}" for i in range(12)]
        for name in names:
            await _legacy(t, name)
        self._fail_all(t, names)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr,
            organization_id=t.org,
            concurrency=4,
            max_consecutive_failures=3,
        )
        tried = report.outcomes["failed"]
        assert 3 <= tried <= 3 + (4 - 1)
        assert report.outcomes["skipped"] == 12 - tried
        assert len(t.embed.calls) == tried * 4

    async def test_opening_on_the_last_skill_is_not_stopping_early(self, t):
        names = [f"last-{i}" for i in range(3)]
        for name in names:
            await _legacy(t, name)
        self._fail_all(t, names)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr,
            organization_id=t.org,
            concurrency=1,
            max_consecutive_failures=3,
        )
        # Every skill was tried; nothing was left for a later pass.
        assert report.outcomes == {"failed": 3}
        assert not report.stopped_early and report.skipped == 0
        assert "stopped early" not in report.summary()

    async def test_a_provider_that_answers_restarts_the_count(self, t):
        names = [f"alt-{i}" for i in range(8)]
        by_id = sorted([(await _legacy(t, name), name) for name in names])
        # In the pass's order (skill id): two failures, one success, two
        # failures, ... never three in a row.
        self._fail_all(t, [n for i, (_, n) in enumerate(by_id) if i % 3 != 2])
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr,
            organization_id=t.org,
            concurrency=1,
            max_consecutive_failures=3,
        )
        assert not report.stopped_early
        assert report.outcomes == {"failed": 6, "filled": 2}

    async def test_without_a_breaker_every_skill_is_tried(self, t):
        # The startup pass and the script: every skill (no max).
        names = [f"all-{i}" for i in range(5)]
        for name in names:
            await _legacy(t, name)
        self._fail_all(t, names)
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org, concurrency=1
        )
        assert report.outcomes == {"failed": 5} and not report.stopped_early
        with pytest.raises(ValueError):
            await reembed.reembed_missing_retrieval_embeddings(
                manager=t.mgr, max_consecutive_failures=0
            )

    async def test_two_calls_for_one_skill_make_one_provider_call(self, t):
        skill_id = await _legacy(t, "overlap-one")
        t.embed.gate = asyncio.Event()
        # Two managers, like the startup pass and a learning run's pass.
        other = ProceduralMemoryManager(session_maker=t.sm, sleep=_no_sleep)
        first = asyncio.ensure_future(t.mgr.reembed_skill(skill_id))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if t.embed.in_flight == 1:
                break
        second = asyncio.ensure_future(other.reembed_skill(skill_id))
        await asyncio.sleep(0.05)
        assert t.embed.in_flight == 1  # the second waits, it does not call
        t.embed.gate.set()
        assert await first == pmm.REEMBED_FILLED
        assert await second == pmm.REEMBED_FILLED
        assert len(t.embed.calls) == 1
        assert pmm._REEMBED_IN_FLIGHT == {}

    async def test_the_startup_pass_and_a_run_pass_embed_each_skill_once(self, t):
        ids = [await _legacy(t, f"meet-{i}") for i in range(6)]
        t.embed.gate = asyncio.Event()
        startup = asyncio.ensure_future(
            reembed.reembed_missing_retrieval_embeddings(
                manager=t.mgr, organization_id=t.org, concurrency=3
            )
        )
        run_pass = asyncio.ensure_future(
            reembed.reembed_missing_retrieval_embeddings(
                manager=ProceduralMemoryManager(session_maker=t.sm, sleep=_no_sleep),
                organization_id=t.org,
                user_id=t.user_id,
                concurrency=3,
                max_consecutive_failures=3,
            )
        )
        await asyncio.sleep(0.05)
        t.embed.gate.set()
        reports = await asyncio.gather(startup, run_pass)
        texts = [text for _, text in t.embed.calls]
        assert len(texts) == len(set(texts)) == 6  # one call per skill
        for skill_id in ids:
            assert (await _row(t, skill_id)).retrieval_embedding is not None
        assert sum(r.filled for r in reports) >= 6
        assert pmm._REEMBED_IN_FLIGHT == {}

    async def test_a_cancelled_first_call_leaves_the_skill_to_the_waiter(self, t):
        skill_id = await _legacy(t, "overlap-cancel")
        t.embed.gate = asyncio.Event()
        first = asyncio.ensure_future(t.mgr.reembed_skill(skill_id))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if t.embed.in_flight == 1:
                break
        second = asyncio.ensure_future(t.mgr.reembed_skill(skill_id))
        await asyncio.sleep(0.05)
        first.cancel()  # e.g. the startup pass stopped by a shutdown
        with pytest.raises(asyncio.CancelledError):
            await first
        t.embed.gate.set()
        assert await second == pmm.REEMBED_FILLED
        assert len(t.embed.calls) == 2  # the waiter made its own call
        assert (await _row(t, skill_id)).retrieval_embedding is not None
        assert pmm._REEMBED_IN_FLIGHT == {}

    async def test_cancelling_the_waiter_leaves_the_first_call_alone(self, t):
        skill_id = await _legacy(t, "overlap-waiter-cancel")
        t.embed.gate = asyncio.Event()
        first = asyncio.ensure_future(t.mgr.reembed_skill(skill_id))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if t.embed.in_flight == 1:
                break
        second = asyncio.ensure_future(t.mgr.reembed_skill(skill_id))
        await asyncio.sleep(0.05)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        t.embed.gate.set()
        assert await first == pmm.REEMBED_FILLED
        assert len(t.embed.calls) == 1
        assert pmm._REEMBED_IN_FLIGHT == {}


class TestStartupTask:
    async def test_it_runs_one_pass_in_the_background(self, t):
        for i in range(2):
            await _legacy(t, f"s-{i}")
        reembedder = reembed.SkillRetrievalReembedder(manager=t.mgr, concurrency=1)
        task = reembedder.start()
        assert reembedder.start() is task  # started once
        await task
        assert not reembedder.running
        assert reembedder.report.outcomes.get("filled", 0) >= 2
        await reembedder.stop()  # a no-op once it is done

    async def test_stop_cancels_a_pass_in_flight(self, t):
        await _legacy(t, "s-blocked")
        t.embed.gate = asyncio.Event()  # never set: the embedding hangs
        reembedder = reembed.SkillRetrievalReembedder(manager=t.mgr)
        reembedder.start()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if t.embed.in_flight:
                break
        assert reembedder.running and t.embed.in_flight == 1
        await reembedder.stop(timeout=5)
        assert not reembedder.running and reembedder.task.cancelled()
        assert t.embed.in_flight == 0

    async def test_a_failing_pass_is_logged_not_raised(self, t, monkeypatch):
        async def broken(**kwargs):
            raise RuntimeError("database gone")

        monkeypatch.setattr(reembed, "reembed_missing_retrieval_embeddings", broken)
        reembedder = reembed.SkillRetrievalReembedder(manager=t.mgr)
        assert await reembedder.start() is None
        assert reembedder.report is None


# --------------------------------------------------------------------------- #
# scripts/reembed_skill_retrieval.py                                           #
# --------------------------------------------------------------------------- #


def _load_script(monkeypatch):
    path = REPO / "scripts" / "reembed_skill_retrieval.py"
    spec = importlib.util.spec_from_file_location("reembed_script", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class TestScript:
    def test_arguments_and_exit_codes(self, monkeypatch, capsys):
        import mirix.server.server as server_module

        module = _load_script(monkeypatch)
        seen = []
        outcome = {}

        async def fake_pass(**kwargs):
            seen.append(kwargs)
            report = reembed.ReembedReport(dry_run=kwargs["dry_run"], scanned=2)
            report.outcomes = dict(outcome)
            return report

        class Engine:
            disposed = 0

            async def dispose(self):
                Engine.disposed += 1

        monkeypatch.setattr(reembed, "reembed_missing_retrieval_embeddings", fake_pass)
        monkeypatch.setattr(server_module, "engine", Engine())

        assert module.main(["--dry-run"]) == 0
        assert "2 live skill(s)" in capsys.readouterr().out
        outcome.update({"filled": 1, "failed": 1})
        assert (
            module.main(
                [
                    "--organization-id",
                    "org-x",
                    "--batch-size",
                    "5",
                    "--concurrency",
                    "2",
                ]
            )
            == 2
        )
        assert seen[-1] == {
            "organization_id": "org-x",
            "batch_size": 5,
            "concurrency": 2,
            "dry_run": False,
        }
        # A skill with no embedding config can never be embedded — that
        # is not "done" either.
        outcome.update({"no_config": 1})
        assert module.main([]) == 2  # a retryable failure takes precedence
        outcome.pop("failed")
        assert module.main([]) == 3
        assert "no embedding config" in capsys.readouterr().out
        outcome.pop("no_config")
        assert module.main([]) == 0
        assert Engine.disposed == 5

    async def test_the_real_pass_through_the_script_module(self, t, monkeypatch):
        module = _load_script(monkeypatch)
        skill_id = await _legacy(t, "script-real")
        report = await reembed.reembed_missing_retrieval_embeddings(
            manager=t.mgr, organization_id=t.org
        )
        assert report.outcomes == {"filled": 1}
        assert (await _row(t, skill_id)).retrieval_embedding is not None
        assert module.__doc__ and "idempotent" in module.__doc__


# --------------------------------------------------------------------------- #
# A vector left missing is filled without a restart                            #
# --------------------------------------------------------------------------- #


class TestFilledWithoutARestart:
    async def test_any_edit_fills_a_missing_vector(self, t):
        # An instructions-only edit does not change the retrieval text; it
        # still gives a skill left without a vector its vector.
        skill_id = await _legacy(t, "fill-on-edit", triggers=["x"])
        await _edit(t, skill_id, instructions="New steps.")
        row = await _row(t, skill_id)
        text = _text("fill-on-edit", ["x"], "legacy fill-on-edit")
        assert list(row.retrieval_embedding) == pytest.approx(
            _stored(text, "row-model")
        )
        # With a vector, an instructions edit calls nothing (the retrieval
        # text is unchanged).
        calls = len(t.embed.calls)
        await _edit(t, skill_id, instructions="Newer steps.")
        assert len(t.embed.calls) == calls

    async def test_an_edit_during_the_outage_leaves_it_missing(self, t):
        skill_id = await _legacy(t, "fill-still-down", triggers=["y"])
        t.embed.fail.add(_text("fill-still-down", ["y"], "legacy fill-still-down"))
        await _edit(t, skill_id, instructions="Other steps.")
        row = await _row(t, skill_id)
        assert row.retrieval_embedding is None
        assert row.instructions == "Other steps."  # the edit itself landed

    async def test_the_end_of_a_learning_run_fills_the_users_missing_vectors(
        self, t, monkeypatch
    ):
        """A skill the curator stored during an embedding outage is invisible
        to every gated retrieval; the next learning run of its user gives it
        its vector (and says so on the run), in the same process."""
        from mirix.schemas.user import User as PydanticUser
        from mirix.services.auto_dream_manager import AutoDreamManager

        original = reembed.reembed_missing_retrieval_embeddings

        async def on_this_database(**kwargs):
            return await original(manager=t.mgr, **kwargs)

        monkeypatch.setattr(
            reembed, "reembed_missing_retrieval_embeddings", on_this_database
        )

        class Runs:
            def __init__(self):
                self.entries, self.events = [], []

            async def append_curator_metrics(self, run_id, entries):
                self.entries += [(run_id, dict(e)) for e in entries]
                return True

            async def append_retry_events(self, run_id, events):
                self.events += list(events)
                return True

        user = PydanticUser(
            id=t.user_id, name=t.user_id, organization_id=t.org, timezone="UTC"
        )
        name = "outage-skill"
        text = _text(name, [f"when {name}"], f"How to {name}")
        t.embed.fail.add(text)
        skill = await _create(t, name)  # written while the provider is down
        assert (await _row(t, skill.id)).retrieval_embedding is None
        vector = _vector(text, "agent-model")

        async def gated():
            result = await t.mgr.retrieve_skills(
                name, user, ["test"], apply_gate=True, query_embedding=vector
            )
            return [r.skill.id for r in result.skills]

        assert await gated() == []  # no vector: the gate never passes it

        # The provider is still down at the end of the next run: recorded.
        runs = Runs()
        await AutoDreamManager()._reembed_missing_vectors(
            run_id="run-1", user=user, organization_id=t.org, run_mgr=runs
        )
        [(run_id, entry)] = runs.entries
        assert run_id == "run-1" and entry["stage"] == "reembed"
        assert entry["batch_id"] == "run-1:reembed"
        assert entry["error_kind"] == "embedding" and entry["filled"] == 0
        assert entry["outcomes"].get("failed", 0) >= 1
        assert [e["kind"] for e in runs.events][:3] == ["embedding"] * 3

        # It answers again: the next run fills it, and it is retrievable.
        t.embed.fail.clear()
        runs = Runs()
        await AutoDreamManager()._reembed_missing_vectors(
            run_id="run-2", user=user, organization_id=t.org, run_mgr=runs
        )
        [(_, entry)] = runs.entries
        assert "error_kind" not in entry and entry["filled"] >= 1
        assert (await _row(t, skill.id)).retrieval_embedding is not None
        assert await gated() == [skill.id]

        # Nothing left: the next run records nothing.
        runs = Runs()
        await AutoDreamManager()._reembed_missing_vectors(
            run_id="run-3", user=user, organization_id=t.org, run_mgr=runs
        )
        assert runs.entries == []

    async def test_only_the_runs_user_is_re_embedded(self, t, monkeypatch):
        from mirix.schemas.user import User as PydanticUser
        from mirix.services.auto_dream_manager import AutoDreamManager

        other_user = f"user-{uuid.uuid4().hex[:8]}"
        async with t.sm() as db:
            db.add(
                UserORM(
                    id=other_user,
                    name=other_user,
                    organization_id=t.org,
                    status="active",
                    timezone="UTC",
                )
            )
            await db.commit()
        mine = await _legacy(t, "user-filter-mine")
        theirs = await _legacy(t, "user-filter-theirs")
        async with t.sm() as db:
            row = (
                await db.execute(
                    select(ProceduralORM).where(ProceduralORM.id == theirs)
                )
            ).scalar_one()
            row.user_id = other_user
            await db.commit()
        original = reembed.reembed_missing_retrieval_embeddings

        async def on_this_database(**kwargs):
            return await original(manager=t.mgr, **kwargs)

        monkeypatch.setattr(
            reembed, "reembed_missing_retrieval_embeddings", on_this_database
        )

        class Runs:
            async def append_curator_metrics(self, run_id, entries):
                return True

            async def append_retry_events(self, run_id, events):
                return True

        await AutoDreamManager()._reembed_missing_vectors(
            run_id="run-u",
            user=PydanticUser(
                id=t.user_id, name=t.user_id, organization_id=t.org, timezone="UTC"
            ),
            organization_id=t.org,
            run_mgr=Runs(),
        )
        assert (await _row(t, mine)).retrieval_embedding is not None
        assert (await _row(t, theirs)).retrieval_embedding is None
