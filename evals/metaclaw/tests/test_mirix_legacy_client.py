"""Unit tests for LegacyMirixClient using httpx.MockTransport."""
import json

import httpx
import pytest

from evals.metaclaw.mirix_legacy_client import (
    DEFAULT_CLIENT_ID,
    DEFAULT_META_AGENT_NAME,
    LegacyMirixClient,
)


def _mock_transport(handler):
    return httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
        headers={"X-Client-Id": DEFAULT_CLIENT_ID},
    )


@pytest.mark.asyncio
async def test_resolve_meta_agent_id_picks_correct_agent():
    """GET /v1/agents must be filtered to the meta_memory_agent row."""
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["path"] = req.url.path
        return httpx.Response(
            200,
            json=[
                {"id": "agent-1", "name": "core_memory_agent"},
                {"id": "agent-2", "name": DEFAULT_META_AGENT_NAME},
                {"id": "agent-3", "name": "episodic_memory_agent"},
            ],
        )

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        agent_id = await cli._resolve_meta_agent_id()
        assert agent_id == "agent-2"
        assert captured == {"method": "GET", "path": "/v1/agents"}


@pytest.mark.asyncio
async def test_resolve_meta_agent_raises_when_missing():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "agent-1", "name": "core_memory_agent"}])

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        with pytest.raises(RuntimeError, match="meta_memory_agent"):
            await cli._resolve_meta_agent_id()


@pytest.mark.asyncio
async def test_add_memory_posts_to_add_sync_with_correct_body():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/agents":
            return httpx.Response(200, json=[{"id": "ma-1", "name": DEFAULT_META_AGENT_NAME}])
        captured["path"] = req.url.path
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"success": True, "status": "processed"})

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client, user_id="eval-legacy-3day")
        out = await cli.add_memory("round-1 transcript text")

    assert captured["path"] == "/v1/memory/add_sync"
    assert captured["body"] == {
        "meta_agent_id": "ma-1",
        "messages": [{"role": "user", "content": "round-1 transcript text"}],
        "user_id": "eval-legacy-3day",
        "chaining": True,
    }
    assert out["status"] == "processed"


@pytest.mark.asyncio
async def test_search_procedural_sends_correct_query_params():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "memory_type": "procedural",
                        "content": {
                            "id": "p-1",
                            "summary": "Format dates as ISO 8601",
                            "steps": "Use YYYY-MM-DD",
                            "entry_type": "guide",
                        },
                    }
                ]
            },
        )

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client, user_id="eval-legacy-3day")
        rows = await cli.search_procedural(query="what is today's date format", limit=6)

    assert captured["path"] == "/v1/memory/search"
    assert captured["params"]["memory_type"] == "procedural"
    assert captured["params"]["search_method"] == "bm25"
    assert captured["params"]["search_field"] == "summary"
    assert captured["params"]["query"] == "what is today's date format"
    assert captured["params"]["limit"] == "6"
    assert captured["params"]["user_id"] == "eval-legacy-3day"
    assert rows == [
        {
            "id": "p-1",
            "summary": "Format dates as ISO 8601",
            "steps": "Use YYYY-MM-DD",
            "entry_type": "guide",
        }
    ]


@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        assert await cli.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        assert await cli.health() is False
