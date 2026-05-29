"""Thin async wrapper around MIRIX REST endpoints used by this eval harness.

Endpoints used:
    POST /v1/skills/evolve   — trigger ProceduralMemoryAgent on a batch of
                               round messages, returns created/edited/deleted diff
    GET  /v1/skills?...      — search skills (BM25); used for retrieval
    GET  /health             — liveness probe

Auth: REST endpoints require a client identity. We send X-Client-Id on
every request, pointing at the default admin client that MIRIX creates
on first server boot. This is the dev-mode shortcut used by the auth
middleware (see rest_api.py: get_client_from_jwt_or_api_key — direct
X-Client-Id is honored without needing an API key for local dev).
"""
from __future__ import annotations

from typing import Any

import httpx

# MIRIX seeds this client row on first boot. See
# `mirix.services.client_manager.create_default_client` and the
# `default_client` row in the `clients` table.
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"


class MirixClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8531",
        user_id: str = "eval-metaclaw-3day",
        client_id: str = DEFAULT_CLIENT_ID,
        timeout: float = 600.0,
        _client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.client_id = client_id
        self._timeout = timeout
        self._client = _client            # injectable for tests
        self._owns_client = _client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers={"X-Client-Id": self.client_id},
            )
        return self._client

    async def aclose(self):
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evolve(self, messages: list[str]) -> dict[str, Any]:
        """POST /v1/skills/evolve. Returns {created: [...], edited: [...], deleted: [...]}."""
        client = await self._get_client()
        resp = await client.post(
            "/v1/skills/evolve",
            json={"messages": messages, "user_id": self.user_id},
        )
        resp.raise_for_status()
        body = resp.json()
        # rest_api.py returns {success, changes: {created, edited, deleted}}
        return body.get("changes", body)

    async def search_skills(
        self,
        query: str,
        limit: int = 6,
        search_method: str = "bm25",      # kept on signature; server hard-codes bm25
        search_field: str = "description",  # kept on signature; server hard-codes description
    ) -> list[dict[str, Any]]:
        """GET /v1/skills?query=...&limit=N&user_id=...

        The MIRIX endpoint currently hard-codes search_method=bm25 and
        search_field=description internally, so we don't send them on the
        wire. Method kwargs are kept for forward-compat if/when the route
        adds them as accepted query params.
        """
        client = await self._get_client()
        resp = await client.get(
            "/v1/skills",
            params={
                "query": query,
                "limit": limit,
                "user_id": self.user_id,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("skills", [])

    async def health(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
