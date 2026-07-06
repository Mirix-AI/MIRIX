from types import SimpleNamespace

import pytest

from mirix.client.remote_client import MirixClient


def _messages():
    return [
        {"role": "user", "content": [{"type": "text", "text": "remember this"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "noted"}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("async_add", "endpoint"),
    [(True, "/memory/add"), (False, "/memory/add_sync")],
)
async def test_add_sends_top_level_session_id(monkeypatch, async_add, endpoint):
    client = MirixClient(api_key="test-key", base_url="http://test")
    client._meta_agent = SimpleNamespace(id="agent-meta-1")
    calls = []

    async def fake_ensure_user_exists(user_id=None, headers=None):
        return None

    async def fake_request(method, endpoint, json=None, params=None, headers=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "json": json,
                "params": params,
            }
        )
        return {"success": True}

    monkeypatch.setattr(client, "_ensure_user_exists", fake_ensure_user_exists)
    monkeypatch.setattr(client, "_request", fake_request)

    try:
        await client.add(
            user_id="user-1",
            messages=_messages(),
            session_id="sess-1",
            async_add=async_add,
        )
    finally:
        await client.close()

    assert calls == [
        {
            "method": "POST",
            "endpoint": endpoint,
            "json": {
                "user_id": "user-1",
                "meta_agent_id": "agent-meta-1",
                "messages": _messages(),
                "chaining": True,
                "verbose": False,
                "session_id": "sess-1",
            },
            "params": None,
        }
    ]


@pytest.mark.asyncio
async def test_add_rejects_mismatched_session_id_filter_tag(monkeypatch):
    client = MirixClient(api_key="test-key", base_url="http://test")
    client._meta_agent = SimpleNamespace(id="agent-meta-1")

    async def fail_request(*args, **kwargs):
        raise AssertionError("_request should not be called")

    monkeypatch.setattr(client, "_request", fail_request)

    try:
        with pytest.raises(ValueError, match="must agree"):
            await client.add(
                user_id="user-1",
                messages=_messages(),
                session_id="sess-1",
                filter_tags={"session_id": "sess-2"},
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_dream_sends_meta_agent_and_last_n_sessions(monkeypatch):
    client = MirixClient(api_key="test-key", base_url="http://test")
    calls = []

    async def fake_ensure_user_exists(user_id=None, headers=None):
        return None

    async def fake_request(method, endpoint, json=None, params=None, headers=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "json": json,
                "params": params,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(client, "_ensure_user_exists", fake_ensure_user_exists)
    monkeypatch.setattr(client, "_request", fake_request)

    try:
        await client.auto_dream(
            user_id="user-1",
            mode="procedural",
            meta_agent_id="agent-meta-1",
            last_n_sessions=3,
        )
    finally:
        await client.close()

    assert calls == [
        {
            "method": "POST",
            "endpoint": "/memory/auto_dream",
            "json": {
                "mode": "procedural",
                "dry_run": False,
                "meta_agent_id": "agent-meta-1",
                "last_n_sessions": 3,
            },
            "params": {"user_id": "user-1"},
        }
    ]
