"""Unit tests for the backward-compatible AutoDreamResponse schema extension.

The generic-arm driver health-gates off structured evolution counts rather than
parsing the human-readable `message`, so AutoDreamResponse gained `skills_changed`
and `changes`. These tests pin the contract:

  * both new fields are OPTIONAL with sane defaults (0 / {}), so every existing
    caller (and non-procedural modes) keeps working unchanged;
  * when populated they round-trip through model_dump.

No live server / API key needed.
"""

import datetime as dt
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mirix.schemas.agent import AgentType
from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.schemas.auto_dream import AutoDreamResponse, MemoryTypeStats


def _now() -> dt.datetime:
    return dt.datetime(2026, 6, 23, 12, 0, 0)


def test_defaults_are_zero_and_empty():
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={},
        last_dream_at=_now(),
        dry_run=False,
    )
    assert resp.skills_changed == 0
    assert resp.changes == {}


def test_populated_fields_round_trip():
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={"procedural": MemoryTypeStats(total=4)},
        last_dream_at=_now(),
        dry_run=False,
        skills_changed=2,
        changes={"created": ["s1"], "edited": ["s2"], "deleted": []},
    )
    assert resp.skills_changed == 2
    assert resp.changes == {"created": ["s1"], "edited": ["s2"], "deleted": []}

    dumped = resp.model_dump()
    assert dumped["skills_changed"] == 2
    assert dumped["changes"]["created"] == ["s1"]
    assert dumped["processed"]["procedural"]["total"] == 4

    # Re-parse the dump → identical structured counts (wire-stable for the driver).
    reparsed = AutoDreamResponse.model_validate(dumped)
    assert reparsed.skills_changed == 2
    assert reparsed.changes == resp.changes


def test_existing_caller_without_new_fields_unaffected():
    # Mirrors the exact shape pre-existing callers build (no skills_changed/changes).
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={"procedural": MemoryTypeStats(total=0)},
        last_dream_at=_now(),
        dry_run=True,
        message="Dry run — distilled 0 experience(s).",
    )
    assert resp.message.startswith("Dry run")
    assert resp.skills_changed == 0
    assert resp.changes == {}


def test_run_ids_default_empty_and_round_trip():
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={},
        last_dream_at=_now(),
        dry_run=False,
    )
    assert resp.run_ids == []
    populated = AutoDreamResponse.model_validate(
        {**resp.model_dump(), "run_ids": ["slrun-1"]}
    )
    assert populated.model_dump()["run_ids"] == ["slrun-1"]


def test_auto_dream_request_wait_defaults_to_true():
    assert AutoDreamRequest(mode="procedural").wait is True
    assert AutoDreamRequest(mode="procedural", wait=False).wait is False


def test_auto_dream_request_accepts_meta_agent_id():
    req = AutoDreamRequest(mode="procedural", meta_agent_id="agent-meta-1")
    assert req.meta_agent_id == "agent-meta-1"


@pytest.mark.asyncio
async def test_auto_dream_handler_uses_explicit_meta_agent(monkeypatch):
    from mirix.server import rest_api

    client = SimpleNamespace(id="client-1")
    user = SimpleNamespace(id="user-1")
    explicit_meta = SimpleNamespace(
        id="agent-meta-2",
        agent_type=AgentType.meta_memory_agent,
    )
    captured = {}

    async def fake_auth(authorization=None, http_request=None):
        return client, "api_key"

    class UserManager:
        async def get_user_by_id(self, user_id=None):
            assert user_id == "user-1"
            return user

    class AgentManager:
        async def get_agent_by_id(self, agent_id, actor):
            assert agent_id == "agent-meta-2"
            assert actor is client
            return explicit_meta

        async def list_agents(self, actor):
            raise AssertionError("explicit meta_agent_id should skip list_agents")

    class FakeAutoDreamManager:
        async def run(self, *, request, user, actor, meta_agent_state):
            captured["request"] = request
            captured["user"] = user
            captured["actor"] = actor
            captured["meta_agent_state"] = meta_agent_state
            return {"ok": True}

    fake_server = SimpleNamespace(
        user_manager=UserManager(),
        agent_manager=AgentManager(),
    )

    monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", fake_auth)
    monkeypatch.setattr(rest_api, "get_server", lambda: fake_server)

    import mirix.services.auto_dream_manager as auto_dream_manager

    monkeypatch.setattr(auto_dream_manager, "AutoDreamManager", FakeAutoDreamManager)

    resp = await rest_api.auto_dream_handler(
        AutoDreamRequest(mode="procedural", meta_agent_id="agent-meta-2", wait=False),
        user_id="user-1",
        authorization="Bearer test",
        http_request=None,
    )

    assert resp == {"ok": True}
    assert captured["request"].wait is False
    assert captured["meta_agent_state"] is explicit_meta
    assert captured["user"] is user
    assert captured["actor"] is client
    assert captured["request"].meta_agent_id == "agent-meta-2"


@pytest.mark.asyncio
async def test_auto_dream_handler_rejects_non_meta_agent(monkeypatch):
    from mirix.server import rest_api

    client = SimpleNamespace(id="client-1")
    user = SimpleNamespace(id="user-1")
    chat_agent = SimpleNamespace(id="agent-chat-1", agent_type=AgentType.chat_agent)

    async def fake_auth(authorization=None, http_request=None):
        return client, "api_key"

    class UserManager:
        async def get_user_by_id(self, user_id=None):
            return user

    class AgentManager:
        async def get_agent_by_id(self, agent_id, actor):
            return chat_agent

    fake_server = SimpleNamespace(
        user_manager=UserManager(),
        agent_manager=AgentManager(),
    )

    monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", fake_auth)
    monkeypatch.setattr(rest_api, "get_server", lambda: fake_server)

    with pytest.raises(HTTPException) as exc_info:
        await rest_api.auto_dream_handler(
            AutoDreamRequest(mode="procedural", meta_agent_id="agent-chat-1"),
            user_id="user-1",
            authorization="Bearer test",
            http_request=None,
        )

    assert exc_info.value.status_code == 400
    assert "not a meta_memory_agent" in exc_info.value.detail


@pytest.mark.asyncio
async def test_the_dream_checkpoint_belongs_to_the_dreaming_user(monkeypatch):
    """``create_episodic_memory`` stamps its own ``user_id`` / ``client_id``
    arguments over the event's (falling back to the ADMIN user), so the
    checkpoint must pass them: otherwise it lands in the admin user's memory
    and ``get_last_dream_time(user)`` never finds it (found by the offline
    end-to-end test, tests/test_skill_learning_e2e_pg.py)."""
    import mirix.services.episodic_memory_manager as episodic_module
    from mirix.services.auto_dream_manager import AutoDreamManager

    created = []

    class RecordingEpisodicManager:
        async def create_episodic_memory(self, **kwargs):
            created.append(kwargs)
            return kwargs["episodic_memory"]

    monkeypatch.setattr(
        episodic_module, "EpisodicMemoryManager", RecordingEpisodicManager
    )
    user = SimpleNamespace(id="user-dreamer")
    actor = SimpleNamespace(id="client-7", organization_id="org-7")
    await AutoDreamManager().write_checkpoint(user, actor, None, _now())

    [call] = created
    assert call["user_id"] == "user-dreamer" and call["client_id"] == "client-7"
    event = call["episodic_memory"]
    assert event.user_id == "user-dreamer" and event.organization_id == "org-7"
    assert event.event_type == "auto_dream_checkpoint"


@pytest.mark.asyncio
async def test_checkpoints_never_crowd_real_events_out_of_the_dream_window(
    monkeypatch, tmp_path
):
    """One checkpoint per learning run lands in the user's episodic
    memory; the episodic dream's window of newest events leaves them out in
    the query, so hundreds of checkpoints never hide the real events."""
    import uuid
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import mirix.orm  # noqa: F401 -- register every ORM class
    import mirix.server.server as server_module
    import mirix.services.auto_dream_manager as adm
    from mirix.orm.base import Base
    from mirix.orm.episodic_memory import EpisodicEvent as EventORM
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM
    from mirix.schemas.user import User as PydanticUser

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'dream.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def sm():
        async with local() as session:
            yield session

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(adm, "_EPISODIC_DREAM_WINDOW", 5)
    org, uid = "org-dream", "user-dream"
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    try:
        async with sm() as db:
            db.add(OrganizationORM(id=org, name=org))
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org,
                    status="active",
                    timezone="UTC",
                )
            )
            await db.commit()

        def event(i, event_type):
            return EventORM(
                id=f"ep_{uuid.uuid4().hex[:12]}",
                organization_id=org,
                user_id=uid,
                occurred_at=base + dt.timedelta(minutes=i),
                actor="user",
                event_type=event_type,
                summary=f"{event_type} {i}",
                details="d",
                filter_tags={},
            )

        async with sm() as db:
            db.add_all([event(i, "activity") for i in range(3)])  # the oldest
            db.add(event(3, "conversation"))
            # Ten newer checkpoints: twice the (patched) window.
            db.add_all([event(10 + i, adm._CHECKPOINT_EVENT_TYPE) for i in range(10)])
            await db.commit()

        user = PydanticUser(id=uid, name=uid, organization_id=org, timezone="UTC")
        events = await adm.AutoDreamManager()._fetch_episodic(
            user, None, base, base + dt.timedelta(days=1)
        )
        assert [e.summary for e in events] == [
            "conversation 3",
            "activity 2",
            "activity 1",
            "activity 0",
        ]
    finally:
        await engine.dispose()
