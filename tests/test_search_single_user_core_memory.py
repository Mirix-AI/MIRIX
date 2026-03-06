#!/usr/bin/env python3
"""
Test cases for single-user search (/memory/search) with include_core_memory.

This test suite verifies:
1. include_core_memory=False (default) returns no core blocks
2. include_core_memory=True returns core blocks scoped to the user and client
3. Core blocks have the expected shape (memory_type, id, user_id, label, value, scope)
4. Scope isolation: only blocks matching the client's read_scopes are returned

Prerequisites:
- Server must be running: python scripts/start_server.py
- Optional: Set MIRIX_API_URL in .env file (defaults to http://localhost:8000)
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import pytest
import pytest_asyncio

from mirix.client import MirixClient

pytestmark = [
    pytest.mark.integration,
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = Path(__file__).parent.parent / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


@pytest.mark.asyncio
class TestSearchSingleUserCoreMemory:
    """Test suite for single-user search with include_core_memory."""

    @pytest.fixture(scope="class")
    def org_id(self):
        return f"test-org-single-search-{int(time.time())}"

    @pytest_asyncio.fixture(scope="class")
    async def client1(self, org_id):
        client_id = f"test-client-single-1-{int(time.time())}"
        client = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Test Client Single 1",
            client_scope="scope_a",
            org_id=org_id,
            debug=True,
        )
        await client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)
        logger.info("Client 1 initialized: client_id=%s, org_id=%s, scope=scope_a", client_id, org_id)
        return client

    @pytest_asyncio.fixture(scope="class")
    async def client2(self, org_id):
        """Second client in the same org with a different scope."""
        client_id = f"test-client-single-2-{int(time.time())}"
        client = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Test Client Single 2",
            client_scope="scope_b",
            org_id=org_id,
            debug=True,
        )
        await client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)
        logger.info("Client 2 initialized: client_id=%s, org_id=%s, scope=scope_b", client_id, org_id)
        return client

    @pytest_asyncio.fixture(scope="class")
    async def user_id(self, client1, org_id):
        uid = f"test-user-single-{int(time.time())}"
        await client1.create_or_get_user(user_id=uid, user_name="Test User Single", org_id=org_id)
        logger.info("User created: %s", uid)
        return uid

    @pytest_asyncio.fixture(scope="class", autouse=True)
    async def setup_memories(self, client1, client2, user_id, org_id):
        """Add personal profile messages so the core_memory_agent creates blocks."""
        logger.info("Adding memories for user %s via client1 (scope_a)...", user_id)
        await client1.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "My name is Jordan and I'm a data scientist. "
                                "I prefer visual explanations and work from Austin, TX."
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Nice to meet you, Jordan! I've noted your preferences."}],
                },
            ],
            occurred_at="2025-11-20T14:00:00",
            chaining=True,
            filter_tags={"scope": "scope_a"},
        )
        logger.info("Waiting 50 seconds for async memory processing...")
        await asyncio.sleep(50)

        # Also create user for client2 so blocks exist under scope_b
        await client2.create_or_get_user(user_id=user_id, user_name="Test User Single", org_id=org_id)
        await client2.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I also go by JD and I love cycling on weekends. "
                                "My favorite programming language is Rust."
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Got it, JD! Noted your hobbies and language preference."}],
                },
            ],
            occurred_at="2025-11-20T15:00:00",
            chaining=True,
            filter_tags={"scope": "scope_b"},
        )
        logger.info("Waiting 50 seconds for async memory processing (client2)...")
        await asyncio.sleep(50)
        logger.info("Setup complete.")

    async def test_default_no_core_memory(self, client1, user_id):
        """By default (include_core_memory=False), no core blocks are returned."""
        results = await client1.search(
            user_id=user_id,
            query="",
            memory_type="all",
            limit=50,
        )

        assert results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]
        assert len(core_results) == 0, "Core memory should not be returned when include_core_memory is not set"
        logger.info("Test passed: No core blocks returned by default")

    async def test_include_core_memory_returns_blocks(self, client1, user_id):
        """include_core_memory=True returns core blocks for the user."""
        results = await client1.search(
            user_id=user_id,
            query="",
            memory_type="all",
            limit=50,
            include_core_memory=True,
        )

        assert results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]
        assert len(core_results) > 0, "Core blocks should be returned when include_core_memory=True"

        for item in core_results:
            assert "id" in item
            assert "label" in item
            assert "value" in item
            assert "user_id" in item
            assert "scope" in item
            assert item["memory_type"] == "core"

        logger.info("Test passed: %d core blocks returned with correct shape", len(core_results))

    async def test_core_memory_scope_isolation(self, client1, client2, user_id):
        """Core blocks returned are scoped to the calling client's read_scopes."""
        results_a = await client1.search(
            user_id=user_id,
            query="",
            memory_type="all",
            limit=50,
            include_core_memory=True,
        )
        results_b = await client2.search(
            user_id=user_id,
            query="",
            memory_type="all",
            limit=50,
            include_core_memory=True,
        )

        core_a = [r for r in results_a["results"] if r.get("memory_type") == "core"]
        core_b = [r for r in results_b["results"] if r.get("memory_type") == "core"]

        scopes_a = set(r["scope"] for r in core_a)
        scopes_b = set(r["scope"] for r in core_b)

        assert "scope_b" not in scopes_a, (
            f"Client1 (scope_a) should not see scope_b blocks. Scopes returned: {scopes_a}"
        )
        assert "scope_a" not in scopes_b, (
            f"Client2 (scope_b) should not see scope_a blocks. Scopes returned: {scopes_b}"
        )

        if core_a:
            assert "scope_a" in scopes_a, f"Client1 should see scope_a blocks. Scopes: {scopes_a}"
        if core_b:
            assert "scope_b" in scopes_b, f"Client2 should see scope_b blocks. Scopes: {scopes_b}"

        logger.info(
            "Test passed: Scope isolation verified. client1 scopes=%s, client2 scopes=%s",
            scopes_a,
            scopes_b,
        )

    async def test_include_core_memory_with_specific_memory_type(self, client1, user_id):
        """include_core_memory=True works even when searching a specific memory type (not 'all')."""
        results = await client1.search(
            user_id=user_id,
            query="",
            memory_type="episodic",
            limit=50,
            include_core_memory=True,
        )

        assert results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]
        non_core = [r for r in results["results"] if r.get("memory_type") != "core"]

        for r in non_core:
            assert r["memory_type"] == "episodic", "Non-core results should all be episodic"

        assert len(core_results) > 0, (
            "Core blocks should be returned even when memory_type='episodic' and include_core_memory=True"
        )
        logger.info(
            "Test passed: %d core blocks + %d episodic results returned",
            len(core_results),
            len(non_core),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
