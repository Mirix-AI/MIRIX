"""Tests for evals.metaclaw.mirix_client."""
import json

import httpx
import pytest

from evals.metaclaw.mirix_client import MirixClient


@pytest.mark.asyncio
async def test_evolve_posts_messages_and_user_id():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content.decode())
        return httpx.Response(
            200,
            json={"success": True,
                  "changes": {
                      "created": [{"id": "proc-1", "name": "iso8601",
                                   "description": "d", "instructions": "i",
                                   "entry_type": "guide", "version": "0.1.0"}],
                      "edited": [], "deleted": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="user-1", _client=ac)
        result = await client.evolve(["msg1", "msg2"])

    assert seen["url"].endswith("/v1/skills/evolve")
    assert seen["body"] == {"messages": ["msg1", "msg2"], "user_id": "user-1"}
    assert result["created"][0]["name"] == "iso8601"
    assert result["edited"] == []
    assert result["deleted"] == []


@pytest.mark.asyncio
async def test_search_skills_uses_unified_search_and_returns_results():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        # Unified search envelope: procedural rows under "results" (a superset
        # of the old skill row), not "skills".
        return httpx.Response(
            200,
            json={"results": [{"memory_type": "procedural", "id": "proc-1",
                               "name": "iso8601", "description": "d",
                               "instructions": "i", "entry_type": "guide",
                               "version": "0.1.0"}],
                  "count": 1, "total_count": 1},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="user-1", _client=ac)
        skills = await client.search_skills("datetime format", limit=3)

    # Retrieval goes through the unified /memory/search endpoint (GET /v1/skills removed).
    assert "/memory/search" in seen["url"]
    assert "memory_type=procedural" in seen["url"]
    assert "query=datetime+format" in seen["url"] or "query=datetime%20format" in seen["url"]
    assert "limit=3" in seen["url"]
    assert "user_id=user-1" in seen["url"]
    # search_field/search_method ARE now sent (the server honors them);
    # search_method="" defers to the server's per-type default (hybrid).
    assert "search_field=description" in seen["url"]
    assert "search_method=" in seen["url"]
    assert skills[0]["name"] == "iso8601"


@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is True
    assert seen["url"].endswith("/health")


@pytest.mark.asyncio
async def test_health_returns_false_on_500():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is False
    assert seen["url"].endswith("/health")
