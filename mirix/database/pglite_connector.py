"""
PGlite connector for Python backend.
Provides an async bridge between the Python backend and PGlite database via HTTP.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class PGliteConnector:
    """Async connector for PGlite database through HTTP bridge."""

    def __init__(self):
        self.bridge_url = os.environ.get("MIRIX_PGLITE_BRIDGE_URL", "http://localhost:8001")
        self.use_pglite = os.environ.get("MIRIX_USE_PGLITE", "false").lower() == "true"

    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make async HTTP request to PGlite bridge."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.bridge_url}{endpoint}", json=data, timeout=30.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error("PGlite bridge request failed: %s", e)
            raise

    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Execute a SQL query (async)."""
        if not self.use_pglite:
            raise ValueError("PGlite not enabled")
        data = {"query": query, "params": params or []}
        return await self._make_request("/query", data)

    async def execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL statements (async)."""
        if not self.use_pglite:
            raise ValueError("PGlite not enabled")
        data = {"sql": sql}
        return await self._make_request("/exec", data)

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections (API compatibility)."""
        try:
            yield self
        finally:
            pass

    async def health_check(self) -> bool:
        """Check if PGlite bridge is healthy (async)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.bridge_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False


# Global connector instance
pglite_connector = PGliteConnector()
