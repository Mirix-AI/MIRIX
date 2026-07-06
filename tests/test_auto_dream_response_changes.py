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
        AutoDreamRequest(mode="procedural", meta_agent_id="agent-meta-2"),
        user_id="user-1",
        authorization="Bearer test",
        http_request=None,
    )

    assert resp == {"ok": True}
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
