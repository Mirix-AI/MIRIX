"""
PGlite connector for Python backend
This module provides a bridge between the Python backend and PGlite database
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class PGliteConnector:
    """Connector for PGlite database through HTTP bridge"""

    def __init__(self):
        self.bridge_url = os.environ.get("MIRIX_PGLITE_BRIDGE_URL", "http://localhost:8001")
        self.use_pglite = os.environ.get("MIRIX_USE_PGLITE", "false").lower() == "true"

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to PGlite bridge"""
        try:
            response = requests.post(f"{self.bridge_url}{endpoint}", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("PGlite bridge request failed: %s", e)
            raise

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Execute a SQL query"""
        if not self.use_pglite:
            raise ValueError("PGlite not enabled")

        data = {"query": query, "params": params or []}

        return self._make_request("/query", data)

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL statements"""
        if not self.use_pglite:
            raise ValueError("PGlite not enabled")

        data = {"sql": sql}
        return self._make_request("/exec", data)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        try:
            # For PGlite, we don't need actual connections
            # Just yield self to maintain API compatibility
            yield self
        finally:
            pass

    def health_check(self) -> bool:
        """Check if PGlite bridge is healthy"""
        try:
            response = requests.get(f"{self.bridge_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# Global connector instance
pglite_connector = PGliteConnector()
