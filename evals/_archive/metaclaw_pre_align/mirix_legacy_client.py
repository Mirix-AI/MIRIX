"""Async wrapper around the *original* MIRIX REST endpoints used by this
eval harness — the pre-skill-evolve world.

Endpoints used (main's rest_api.py mounts routes unprefixed):
    GET  /agents                        — find meta_memory_agent id
    POST /memory/add_sync               — synchronously feed a round to
                                            meta_agent (routes to
                                            procedural_memory_agent which
                                            writes summary+steps rows)
    GET  /memory/search?memory_type=procedural
                                        — BM25 retrieval over procedural
                                            memory by summary
    GET  /health                         — liveness probe

We deliberately use /memory/add_sync (not /memory/add) — the latter
queues via Kafka and would let writes lag a round behind retrieves,
silently degrading this arm to the no-skills baseline.
"""
from __future__ import annotations

from typing import Any

import httpx

DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_META_AGENT_NAME = "meta_memory_agent"


class LegacyMirixClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8532",
        user_id: str = "eval-legacy-3day",
        client_id: str = DEFAULT_CLIENT_ID,
        timeout: float = 600.0,
        _client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.client_id = client_id
        self._timeout = timeout
        self._client = _client
        self._owns_client = _client is None
        self._meta_agent_id: str | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers={"X-Client-Id": self.client_id},
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _resolve_meta_agent_id(self) -> str:
        """One-shot bootstrap: list agents, pick the one named meta_memory_agent.
        Cached on first success."""
        if self._meta_agent_id is not None:
            return self._meta_agent_id
        client = await self._get_client()
        resp = await client.get("/agents")
        resp.raise_for_status()
        agents = resp.json()
        for a in agents:
            if a.get("name") == DEFAULT_META_AGENT_NAME:
                self._meta_agent_id = a["id"]
                return self._meta_agent_id
        raise RuntimeError(
            f"No agent named {DEFAULT_META_AGENT_NAME!r} found on the server. "
            f"Call POST /agents/meta/initialize first."
        )

    async def add_memory(self, message_text: str) -> dict[str, Any]:
        """POST /memory/add_sync. Synchronous — by the time this returns,
        any procedural-memory writes from the meta_agent are durable."""
        agent_id = await self._resolve_meta_agent_id()
        client = await self._get_client()
        resp = await client.post(
            "/memory/add_sync",
            json={
                "meta_agent_id": agent_id,
                "messages": [{"role": "user", "content": message_text}],
                "user_id": self.user_id,
                "chaining": True,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def search_procedural(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """GET /memory/search?memory_type=procedural&search_method=bm25
        &search_field=summary&query=...&limit=...&user_id=..."""
        client = await self._get_client()
        resp = await client.get(
            "/memory/search",
            params={
                "memory_type": "procedural",
                "search_method": "bm25",
                "search_field": "summary",
                "query": query,
                "limit": limit,
                "user_id": self.user_id,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results", body.get("memories", []))
        rows: list[dict[str, Any]] = []
        for r in results:
            if r.get("memory_type") and r.get("memory_type") != "procedural":
                continue
            content = r.get("content", r)
            rows.append(content)
        return rows

    async def health(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
