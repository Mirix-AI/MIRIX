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
async def test_search_skills_calls_bm25_and_returns_skills():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(
            200,
            json={"skills": [{"id": "proc-1", "name": "iso8601",
                              "description": "d", "instructions": "i",
                              "entry_type": "guide", "version": "0.1.0"}],
                  "total_count": 1},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="user-1", _client=ac)
        skills = await client.search_skills("datetime format", limit=3)

    assert "/v1/skills" in seen["url"]
    assert "search_method=bm25" in seen["url"]
    assert "query=datetime+format" in seen["url"] or "query=datetime%20format" in seen["url"]
    assert "limit=3" in seen["url"]
    assert skills[0]["name"] == "iso8601"


@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"}))
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_500():
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is False
