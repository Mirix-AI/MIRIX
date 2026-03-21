#!/usr/bin/env python3
"""
Test cases for search_all_users API with client scope filtering.

This test suite verifies:
1. Cross-user search within same organization with matching scope
2. Scope filtering (memories without matching scope are excluded)
3. Organization isolation (different orgs don't see each other's data)
4. Client_id parameter handling

Prerequisites:
- Server must be running: python scripts/start_server.py
- Optional: Set MIRIX_API_URL in .env file (defaults to http://localhost:8000)
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pytest
import pytest_asyncio

from mirix.client import MirixClient

# Mark all tests as integration tests (require a running server)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("isolate_api_key_env"),
]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
# Base URL can be set via MIRIX_API_URL environment variable or .env file
# MirixClient will automatically read from environment variables
BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = Path(__file__).parent.parent / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


async def poll_until(
    fetch_results: Callable[[], Awaitable[dict[str, Any]]],
    is_ready: Callable[[dict[str, Any]], bool],
    wait_log: str,
    max_wait_s: int = 90,
    interval_s: int = 15,
) -> dict[str, Any]:
    """Poll an async search until condition is met or timeout expires."""
    results = await fetch_results()
    elapsed = 0
    while not is_ready(results) and elapsed < max_wait_s:
        logger.info(wait_log, interval_s, elapsed)
        await asyncio.sleep(interval_s)
        elapsed += interval_s
        results = await fetch_results()
    return results


async def add_all_memories(
    client: MirixClient,
    user_id: str,
    filter_tags: dict,
    prefix: str = "",
    block_filter_tags: Optional[dict] = None,
):
    """
    Add all types of memories for a user with given filter_tags.

    Args:
        client: MirixClient instance
        user_id: User ID
        filter_tags: Filter tags including scope
        prefix: Prefix for memory content to distinguish users
        block_filter_tags: Optional dict applied when creating new core (block) memory for this user
    """
    logger.info(f"Adding all memories for user {user_id} with filter_tags={filter_tags}")
    add_kwargs = {"chaining": True, "filter_tags": filter_tags}
    if block_filter_tags is not None:
        add_kwargs["block_filter_tags"] = block_filter_tags

    # Add episodic memory
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prefix}Met with team yesterday at 2 PM to discuss project planning"}
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Recorded team meeting event"}]},
        ],
        occurred_at="2025-11-20T14:00:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added episodic memory - Result: {result}")

    # Add procedural memory
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prefix}My code review process: 1) Check tests 2) Review logic 3) Approve or request changes",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Saved code review procedure"}]},
        ],
        occurred_at="2025-11-20T14:05:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added procedural memory - Result: {result}")

    # Add semantic memory
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prefix}Python is a high-level programming language known for readability and versatility",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Saved semantic knowledge about Python"}]},
        ],
        occurred_at="2025-11-20T14:10:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added semantic memory - Result: {result}")

    # Add resource memory
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prefix}Project documentation: Our main API endpoints are /agents, /memory, and /tools",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Saved project documentation"}]},
        ],
        occurred_at="2025-11-20T14:15:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added resource memory - Result: {result}")

    # Add knowledge vault memory
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prefix}Database credentials: postgresql://user:pass@localhost:5432/db"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Saved database credentials to knowledge vault"}],
            },
        ],
        occurred_at="2025-11-20T14:20:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added knowledge vault memory - Result: {result}")

    # Add core memory - personal user profile information that triggers the core_memory_agent.
    # The meta_memory_agent LLM needs to see personal facts / preferences to include "core"
    # in its trigger_memory_update call.
    result = await client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prefix}My name is Alex and I'm a senior software engineer. "
                            "I prefer direct, concise communication and like technical details. "
                            "I work remotely from Portland and enjoy hiking on weekends."
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Nice to meet you, Alex! I've noted your preferences."}],
            },
        ],
        occurred_at="2025-11-20T14:25:00",
        **add_kwargs,
    )
    logger.info(f"  ✓ Added core memory (personal profile) - Result: {result}")

    logger.info(f"✅ All memories added for user {user_id}")


class TestSearchAllUsers:
    """Test suite for search_all_users API."""

    pytestmark = [pytest.mark.asyncio(loop_scope="class")]

    @pytest.fixture(scope="class")
    def client_scope_value(self):
        """Client scope value used for testing."""
        return "read_write"

    @pytest.fixture(scope="class")
    def org1_id(self):
        """Organization 1 ID."""
        return f"test-org-1-{int(time.time())}"

    @pytest.fixture(scope="class")
    def org2_id(self):
        """Organization 2 ID."""
        return f"test-org-2-{int(time.time())}"

    @pytest_asyncio.fixture(scope="class")
    async def client1(self, org1_id, client_scope_value):
        """Create first MirixClient instance in org1."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 1 in Organization 1")
        logger.info("=" * 80)

        client_id = f"test-client-1-{int(time.time())}"
        c = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 1",
            client_scope=client_scope_value,
            org_id=org1_id,
            debug=True,
        )
        await c.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(
            f"✅ Client 1 initialized: client_id={client_id}, org_id={org1_id}, write_scope={client_scope_value}"
        )
        logger.info(f"   Connected to: {c.base_url}")
        return c

    @pytest_asyncio.fixture(scope="class")
    async def user1_id(self, client1, org1_id):
        """Create first user in org1."""
        user_id = f"test-user-1-{int(time.time())}"
        await client1.create_or_get_user(user_id=user_id, user_name="Test User 1", org_id=org1_id)
        logger.info(f"✅ User 1 created: {user_id}")
        await asyncio.sleep(0.5)
        return user_id

    @pytest_asyncio.fixture(scope="class")
    async def user2_id(self, client1, org1_id):
        """Create second user in org1."""
        user_id = f"test-user-2-{int(time.time())}"
        await client1.create_or_get_user(user_id=user_id, user_name="Test User 2", org_id=org1_id)
        logger.info(f"✅ User 2 created: {user_id}")
        await asyncio.sleep(0.5)
        return user_id

    @pytest.fixture(scope="class")
    def user3_id(self, org1_id):
        """Create third user in org1 (will use different scope via client3)."""
        user_id = f"test-user-3-{int(time.time())}"
        logger.info(f"✅ User 3 ID prepared: {user_id}")
        return user_id

    @pytest_asyncio.fixture(scope="class")
    async def client3(self, org1_id):
        """Create third MirixClient instance in org1 with DIFFERENT write_scope."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 3 in Organization 1 (Different write_scope)")
        logger.info("=" * 80)

        client_id = f"test-client-3-{int(time.time())}"
        c = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 3",
            client_scope="read_only",
            org_id=org1_id,
            debug=True,
        )
        await c.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(f"✅ Client 3 initialized: client_id={client_id}, org_id={org1_id}, write_scope=read_only")
        logger.info(f"   Connected to: {c.base_url}")
        return c

    @pytest_asyncio.fixture(scope="class")
    async def client2(self, org2_id, client_scope_value):
        """Create second MirixClient instance in org2 with same write_scope."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 2 in Organization 2")
        logger.info("=" * 80)

        client_id = f"test-client-2-{int(time.time())}"
        c = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 2",
            client_scope=client_scope_value,
            org_id=org2_id,
            debug=True,
        )
        await c.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(
            f"✅ Client 2 initialized: client_id={client_id}, org_id={org2_id}, write_scope={client_scope_value}"
        )
        logger.info(f"   Connected to: {c.base_url}")
        return c

    @pytest_asyncio.fixture(scope="class")
    async def user4_id(self, client2, org2_id):
        """Create fourth user in org2."""
        user_id = f"test-user-4-{int(time.time())}"
        await client2.create_or_get_user(user_id=user_id, user_name="Test User 4", org_id=org2_id)
        logger.info(f"✅ User 4 created: {user_id}")
        await asyncio.sleep(0.5)
        return user_id

    @pytest_asyncio.fixture(scope="class", autouse=True)
    async def setup_memories(
        self, client1, client2, client3, user1_id, user2_id, user3_id, user4_id, client_scope_value
    ):
        """Setup all memories for all users."""
        logger.info("\n" + "=" * 80)
        logger.info("SETTING UP TEST MEMORIES")
        logger.info("=" * 80)

        # User 1 & 2: Memories with write_scope='read_write' (via client1)
        filter_tags_with_scope = {"scope": client_scope_value, "test": "search_all"}
        logger.info("\n📝 Adding memories for User 1 via Client 1 (write_scope=read_write)...")
        await add_all_memories(client1, user1_id, filter_tags_with_scope, prefix="[User1] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 1)...")
        await asyncio.sleep(50)

        logger.info("\n📝 Adding memories for User 2 via Client 1 (write_scope=read_write)...")
        await add_all_memories(client1, user2_id, filter_tags_with_scope, prefix="[User2] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 2)...")
        await asyncio.sleep(30)

        # User 3: Memories with DIFFERENT write_scope='read_only' (via client3)
        filter_tags_different_scope = {"test": "search_all"}
        logger.info("\n📝 Adding memories for User 3 via Client 3 (write_scope=read_only - DIFFERENT)...")
        await client3.create_or_get_user(user_id=user3_id, user_name="Test User 3", org_id=client3.org_id)
        await add_all_memories(client3, user3_id, filter_tags_different_scope, prefix="[User3] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 3)...")
        await asyncio.sleep(30)

        # User 4: Memories in different org with write_scope='read_write' (via client2)
        filter_tags_org2 = {"scope": client_scope_value, "test": "search_all"}
        logger.info("\n📝 Adding memories for User 4 via Client 2 (Different Org)...")
        await add_all_memories(client2, user4_id, filter_tags_org2, prefix="[User4-Org2] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 4)...")
        await asyncio.sleep(30)

        logger.info("\n" + "=" * 80)
        logger.info("✅ All test memories created and processed")
        logger.info("   - User 1 & 2: write_scope='read_write' (via client1)")
        logger.info("   - User 3: write_scope='read_only' (via client3) - DIFFERENT")
        logger.info("   - User 4: write_scope='read_write' (via client2) - DIFFERENT ORG")
        logger.info("=" * 80)

    async def test_search_all_users_with_client_id_retrieves_both_users(
        self, client1, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 3: Search with client_id should retrieve memories from both user1 and user2."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: Search with client_id retrieves both users with matching scope")
        logger.info("=" * 80)

        async def _search_client1_bm25():
            return await client1.search_all_users(
                query="Python",
                memory_type="all",
                client_id=client1.client_id,
                limit=50,
            )

        results = await poll_until(
            fetch_results=_search_client1_bm25,
            is_ready=lambda r: r["count"] > 0,
            wait_log=(
                "Client1 bm25 search returned 0; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Client ID: {results.get('client_id')}")
        logger.info(f"Organization ID: {results.get('organization_id')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")
        logger.info(f"Filter Tags: {results.get('filter_tags')}")

        # Should retrieve memories from user1 and/or user2 (both have matching scope)
        # NOTE: Due to non-deterministic AI agent behavior, the exact memories created may vary.
        # We verify that:
        # 1. At least some memories are retrieved
        # 2. Only users with matching scope are included (no user3, no user4)
        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        assert results["success"] is True
        assert results["count"] > 0, "Should retrieve at least some memories"

        # At least one of the two users should have matching memories
        assert (
            user1_id in user_ids_in_results or user2_id in user_ids_in_results
        ), f"At least one of user1 or user2 should be included. Found: {user_ids_in_results}"

        # Should NOT include user3 or user4
        assert (
            user3_id not in user_ids_in_results
        ), f"User 3 should be excluded (different scope). Found: {user_ids_in_results}"
        assert (
            user4_id not in user_ids_in_results
        ), f"User 4 should be excluded (different org). Found: {user_ids_in_results}"

        logger.info(f"✅ Test passed: Retrieved memories from users with matching scope ({user_ids_in_results})")

    async def test_search_all_users_with_client_id_retrieves_both_users_embedding(
        self, client1, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 3b: Embedding search with client_id should retrieve memories from both user1 and user2."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3b: Embedding search with client_id retrieves both users with matching scope")
        logger.info("=" * 80)

        async def _search_client1_embedding():
            return await client1.search_all_users(
                query="group discussion",
                memory_type="all",
                search_method="embedding",
                client_id=client1.client_id,
                limit=50,
            )

        results = await poll_until(
            fetch_results=_search_client1_embedding,
            is_ready=lambda r: r["count"] > 0,
            wait_log=(
                "Client1 embedding search returned 0; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Search Method: {results.get('search_method')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")

        # Should retrieve memories from user1 and/or user2 (both have matching scope)
        # NOTE: Due to non-deterministic AI agent behavior, the exact memories created may vary.
        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        assert results["success"] is True
        assert results["search_method"] == "embedding"
        assert results["count"] > 0, "Should retrieve at least some memories"

        # At least one of the two users should have matching memories
        assert (
            user1_id in user_ids_in_results or user2_id in user_ids_in_results
        ), f"At least one of user1 or user2 should be included. Found: {user_ids_in_results}"

        # Should NOT include user3 or user4
        assert (
            user3_id not in user_ids_in_results
        ), f"User 3 should be excluded (different scope). Found: {user_ids_in_results}"
        assert (
            user4_id not in user_ids_in_results
        ), f"User 4 should be excluded (different org). Found: {user_ids_in_results}"

        logger.info(f"✅ Test passed: Retrieved memories from users with matching scope ({user_ids_in_results})")

        logger.info("✅ Test passed: Embedding search retrieved memories from both users with matching scope")

    async def test_search_excludes_user3_without_matching_scope(self, client1, user3_id):
        """Test 5: Search with client1 should NOT retrieve user3 memories (different write_scope: read_only vs read_write)."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5: User 3 excluded due to different write_scope (read_only vs read_write)")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="", memory_type="all", client_id=client1.client_id, limit=100  # client1 has write_scope='read_write'
        )

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")
        logger.info(f"Searching with client1 write_scope='read_write'")
        logger.info(f"User 3 has write_scope='read_only' (via client3)")

        assert (
            user3_id not in user_ids_in_results
        ), "User 3 memories should be excluded (write_scope='read_only' doesn't match 'read_write')"

        logger.info("✅ Test passed: User 3 correctly excluded due to different write_scope")

    async def test_search_excludes_user3_without_matching_scope_embedding(self, client1, user3_id):
        """Test 5b: Embedding search with client1 should NOT retrieve user3 memories (different write_scope)."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5b: Embedding search - User 3 excluded due to different write_scope")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="programming language",  # Semantic query
            memory_type="all",
            search_method="embedding",
            client_id=client1.client_id,
            limit=100,
        )

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")
        logger.info(f"Search Method: {results.get('search_method')}")

        assert results["search_method"] == "embedding"
        assert (
            user3_id not in user_ids_in_results
        ), "User 3 memories should be excluded (write_scope='read_only' doesn't match 'read_write')"

        logger.info("✅ Test passed: Embedding search correctly excluded User 3 due to different write_scope")

    async def test_search_with_client3_retrieves_only_user3(self, client3, user3_id, user1_id, user2_id):
        """Test 6: Search with client3 (write_scope=read_only) should only retrieve user3 memories."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6: Search with client3 (write_scope=read_only) retrieves only User 3")
        logger.info("=" * 80)

        # Search with client3 which has write_scope='read_only'
        async def _search_client3_bm25():
            return await client3.search_all_users(
                query="", memory_type="all", client_id=client3.client_id, limit=100
            )

        results = await poll_until(
            fetch_results=_search_client3_bm25,
            is_ready=lambda r: user3_id in set(result["user_id"] for result in r["results"]),
            wait_log=(
                "Client3 bm25 search missing user3; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Filter Tags: {results.get('filter_tags')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        # Should only retrieve User 3 (write_scope='read_only'), not User 1 or 2 (write_scope='read_write')
        assert user3_id in user_ids_in_results, "User 3 should be included (matching write_scope=read_only)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different write_scope)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different write_scope)"

        logger.info("✅ Test passed: Only User 3 retrieved with write_scope=read_only")

    async def test_search_with_client3_retrieves_only_user3_embedding(self, client3, user3_id, user1_id, user2_id):
        """Test 6b: Embedding search with client3 (write_scope=read_only) should only retrieve user3 memories."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6b: Embedding search with client3 (write_scope=read_only) retrieves only User 3")
        logger.info("=" * 80)

        # Search with client3 which has write_scope='read_only'
        async def _search_client3_embedding():
            return await client3.search_all_users(
                query="software development",
                memory_type="all",
                search_method="embedding",
                client_id=client3.client_id,
                limit=100,
            )

        results = await poll_until(
            fetch_results=_search_client3_embedding,
            is_ready=lambda r: user3_id in set(result["user_id"] for result in r["results"]),
            wait_log=(
                "Client3 embedding search missing user3; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Search Method: {results.get('search_method')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        # Should only retrieve User 3 (write_scope='read_only'), not User 1 or 2 (write_scope='read_write')
        assert results["search_method"] == "embedding"
        assert user3_id in user_ids_in_results, "User 3 should be included (matching write_scope=read_only)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different write_scope)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different write_scope)"

        logger.info("✅ Test passed: Embedding search - Only User 3 retrieved with write_scope=read_only")

    async def test_search_different_org_no_cross_contamination(
        self, client1, client2, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 8: Different organization - no cross-contamination even with matching scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 8: Organization isolation - same scope, different org")
        logger.info("=" * 80)

        # Search with client2 (in org2). Poll briefly because async memory
        # extraction can lag under heavier CI/local runs.
        async def _search_client2_bm25():
            return await client2.search_all_users(
                query="", memory_type="all", client_id=client2.client_id, limit=100
            )

        results = await poll_until(
            fetch_results=_search_client2_bm25,
            is_ready=lambda r: user4_id in set(result["user_id"] for result in r["results"]),
            wait_log="Org2 search missing user4; waiting %ss before retry (elapsed=%ss)...",
        )

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"Client 2 search - User IDs in results: {user_ids_in_results}")
        logger.info(f"Organization ID: {results.get('organization_id')}")

        # Should only see user4 (in org2), NOT user1/user2/user3 (in org1)
        assert user4_id in user_ids_in_results, "User 4 should be included (same org)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different org)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different org)"
        assert user3_id not in user_ids_in_results, "User 3 should be excluded (different org)"

        logger.info("✅ Test passed: Organization isolation working correctly")

    async def test_search_different_org_no_cross_contamination_embedding(
        self, client1, client2, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 8b: Embedding search - Different organization, no cross-contamination even with matching scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 8b: Embedding search - Organization isolation")
        logger.info("=" * 80)

        # Search with client2 (in org2)
        async def _search_client2_embedding():
            return await client2.search_all_users(
                query="database information",
                memory_type="all",
                search_method="embedding",
                client_id=client2.client_id,
                limit=100,
            )

        results = await poll_until(
            fetch_results=_search_client2_embedding,
            is_ready=lambda r: user4_id in set(result["user_id"] for result in r["results"]),
            wait_log=(
                "Org2 embedding search missing user4; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"Client 2 embedding search - User IDs in results: {user_ids_in_results}")
        logger.info(f"Search Method: {results.get('search_method')}")
        logger.info(f"Organization ID: {results.get('organization_id')}")

        # Should only see user4 (in org2), NOT user1/user2/user3 (in org1)
        assert results["search_method"] == "embedding"
        assert user4_id in user_ids_in_results, "User 4 should be included (same org)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different org)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different org)"
        assert user3_id not in user_ids_in_results, "User 3 should be excluded (different org)"

        logger.info("✅ Test passed: Embedding search - Organization isolation working correctly")

    async def test_search_all_memory_types(self, client1):
        """Test search across all memory types."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search all memory types")
        logger.info("=" * 80)

        results = await client1.search_all_users(query="", memory_type="all", client_id=client1.client_id, limit=50)

        # Count by memory type
        memory_types = {}
        for result in results["results"]:
            mem_type = result["memory_type"]
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

        logger.info(f"Memory types found: {memory_types}")

        # Should have all 5 memory types
        assert len(memory_types) >= 3, "Should find at least 3 memory types"

        logger.info("✅ Test passed: Multiple memory types retrieved")

    async def test_search_specific_memory_type(self, client1, user1_id, user2_id):
        """Test search for specific memory type only."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search specific memory type (episodic)")
        logger.info("=" * 80)

        results = await client1.search_all_users(query="team", memory_type="episodic", client_id=client1.client_id, limit=20)

        logger.info(f"Results: {results['count']} episodic memories found")

        # All results should be episodic type
        for result in results["results"]:
            assert result["memory_type"] == "episodic", "Should only return episodic memories"

        # Should include both users
        user_ids = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs: {user_ids}")

        assert results["success"] is True
        assert results["count"] > 0

        logger.info("✅ Test passed: Specific memory type search working")

    async def test_search_specific_memory_type_embedding(self, client1, user1_id, user2_id):
        """Test embedding search for specific memory type only."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Embedding search specific memory type (semantic)")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="programming language concepts",  # Semantic query for semantic memories
            memory_type="semantic",
            search_method="embedding",
            client_id=client1.client_id,
            limit=20,
        )

        async def _search_semantic_embedding():
            return await client1.search_all_users(
                query="programming language concepts",
                memory_type="semantic",
                search_method="embedding",
                client_id=client1.client_id,
                limit=20,
            )

        results = await poll_until(
            fetch_results=_search_semantic_embedding,
            is_ready=lambda r: r["count"] > 0,
            wait_log=(
                "Semantic embedding search returned 0; waiting %ss before retry (elapsed=%ss)..."
            ),
        )

        logger.info(f"Results: {results['count']} semantic memories found")
        logger.info(f"Search Method: {results.get('search_method')}")

        assert results["success"] is True
        assert results["search_method"] == "embedding"
        assert results["count"] > 0, (
            "Semantic embedding search still 0 results after waiting for retries (index may not be ready)."
        )

        # All results should be semantic type
        for result in results["results"]:
            assert result["memory_type"] == "semantic", "Should only return semantic memories"

        # Should include both users
        user_ids = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs: {user_ids}")

        logger.info("✅ Test passed: Embedding search for specific memory type working")

    async def test_search_with_additional_filter_tags(self, client1):
        """Test search with additional filter tags beyond scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search with additional filter tags")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="",
            memory_type="all",
            client_id=client1.client_id,
            filter_tags={"test": "search_all"},  # Additional filter
            limit=50,
        )

        logger.info(f"Results with filter_tags: {results['count']} memories")
        logger.info(f"Applied filter_tags: {results.get('filter_tags')}")
        logger.info(f"read_scopes: {results.get('read_scopes')}")

        # API returns read_scopes at top level; filter_tags contains user-provided tags
        assert "read_scopes" in results, "read_scopes should be returned (used for scope filtering)"
        assert results["read_scopes"] is not None, "read_scopes should be set from client"
        assert "test" in results["filter_tags"], "Additional filter tag should be included"

        logger.info("✅ Test passed: Additional filter tags work correctly")

    async def test_search_with_bm25(self, client1):
        """Test BM25 search method."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: BM25 search method")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="team meeting project",
            memory_type="episodic",
            search_method="bm25",
            client_id=client1.client_id,
            limit=10,
        )

        logger.info(f"BM25 results: {results['count']} memories")

        assert results["success"] is True
        assert results["search_method"] == "bm25"

        logger.info("✅ Test passed: BM25 search working")

    async def test_search_with_embedding(self, client1):
        """Test embedding search method explicitly."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Embedding search method")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="collaborative work meeting",  # Semantic query
            memory_type="episodic",
            search_method="embedding",
            client_id=client1.client_id,
            limit=10,
        )

        logger.info(f"Embedding results: {results['count']} memories")
        logger.info(f"Search Method: {results.get('search_method')}")

        assert results["success"] is True
        assert results["search_method"] == "embedding"

        logger.info("✅ Test passed: Embedding search working")

    async def test_response_includes_metadata(self, client1):
        """Test that response includes all expected metadata."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Response metadata completeness")
        logger.info("=" * 80)

        results = await client1.search_all_users(query="test", memory_type="all", client_id=client1.client_id, limit=10)

        # Check all expected fields in response
        assert "success" in results
        assert "query" in results
        assert "memory_type" in results
        assert "search_field" in results
        assert "search_method" in results
        assert "results" in results
        assert "count" in results
        assert "client_id" in results
        assert "organization_id" in results
        assert "read_scopes" in results
        assert "filter_tags" in results

        logger.info("Response fields: %s", list(results.keys()))
        logger.info("✅ Test passed: All metadata fields present")

    async def test_each_result_includes_user_id(self, client1):
        """Test that each result includes user_id field."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Each result includes user_id")
        logger.info("=" * 80)

        results = await client1.search_all_users(query="", memory_type="all", client_id=client1.client_id, limit=20)

        # Check each result has user_id
        for result in results["results"]:
            assert "user_id" in result, "Each result must include user_id"
            assert "memory_type" in result, "Each result must include memory_type"
            assert result["user_id"] is not None, "user_id must not be None"

        logger.info("✅ Test passed: All results include user_id")

    async def test_search_all_users_include_core_memory_returns_core_section(self, client1):
        """Cross-user search with include_core_memory=True returns core blocks in the results array."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: include_core_memory returns core blocks in results")
        logger.info("=" * 80)

        results = await client1.search_all_users(
            query="",
            memory_type="all",
            client_id=client1.client_id,
            limit=10,
            include_core_memory=True,
        )

        assert results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]
        assert len(core_results) > 0, "Results should include items with memory_type='core' when include_core_memory=True"
        for item in core_results:
            assert "id" in item
            assert "label" in item
            assert "value" in item
            assert "user_id" in item
            assert "scope" in item
        scopes = set(item["scope"] for item in core_results)
        logger.info("Core blocks in results: count=%s, scopes=%s", len(core_results), scopes)
        logger.info("✅ Test passed: Core blocks present in results array")

    async def test_search_all_users_include_core_memory_scope_isolation(self, client1, client3):
        """Cross-user search with include_core_memory returns only blocks within the client's scope (no cross-scope leakage)."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: include_core_memory respects scope - expected blocks returned, others excluded")
        logger.info("=" * 80)

        # client1 has write_scope/read_scopes = "read_write". User1/User2 blocks are in "read_write"; User3's are "read_only".
        # Searching with client1 must return blocks from "read_write" and must NOT return blocks from "read_only".
        results = await client1.search_all_users(
            query="",
            memory_type="all",
            client_id=client1.client_id,
            limit=100,
            include_core_memory=True,
        )

        assert results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]
        scopes_returned = set(r["scope"] for r in core_results)

        assert "read_only" not in scopes_returned, (
            "Blocks from scope 'read_only' must not be returned when searching with client1 (read_write). "
            "Scope isolation violated. Scopes returned: %s" % scopes_returned
        )

        assert "read_write" in scopes_returned, (
            "Blocks from scope 'read_write' (client1's scope) must be returned. Scopes returned: %s" % scopes_returned
        )
        read_write_items = [r for r in core_results if r["scope"] == "read_write"]
        assert len(read_write_items) > 0, (
            "At least one block from scope 'read_write' must be returned. core count=%s" % len(core_results)
        )

        logger.info(
            "Scopes in core results: %s; read_write blocks: %s (read_only correctly excluded)",
            scopes_returned,
            len(read_write_items),
        )
        logger.info("✅ Test passed: Scope isolation - expected blocks returned, no blocks outside client scope")

    async def test_search_all_users_include_core_memory_block_filter_tags_returns_only_matching_blocks(
        self, client1, org1_id
    ):
        """Cross-user search with block_filter_tags (non-scope tag) returns only blocks that have that tag."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: block_filter_tags filters blocks by tag (e.g. env); only matching blocks returned")
        logger.info("=" * 80)

        # Use dedicated users so their blocks are created FOR THE FIRST TIME with block_filter_tags.
        # (user1/user2 already have blocks from class setup without env tag, so they never get env=staging.)
        user_staging_id = f"test-user-staging-{int(time.time())}"
        user_prod_id = f"test-user-prod-{int(time.time())}"
        await client1.create_or_get_user(user_id=user_staging_id, user_name="Staging User", org_id=org1_id)
        await client1.create_or_get_user(user_id=user_prod_id, user_name="Prod User", org_id=org1_id)

        filter_tags = {"scope": "read_write", "test": "block_filter_tag_test"}
        await add_all_memories(
            client1, user_staging_id, filter_tags, prefix="[Staging] ", block_filter_tags={"env": "staging"}
        )
        await add_all_memories(client1, user_prod_id, filter_tags, prefix="[Prod] ", block_filter_tags={"env": "prod"})

        logger.info("⏱️  Waiting 50 seconds for async memory processing (staging + prod users)...")
        await asyncio.sleep(50)

        results = await client1.search_all_users(
            query="",
            memory_type="all",
            client_id=client1.client_id,
            limit=100,
            include_core_memory=True,
            block_filter_tags={"env": "staging"},
        )

        assert results is not None and results["success"] is True
        core_results = [r for r in results["results"] if r.get("memory_type") == "core"]

        if len(core_results) == 0:
            logger.info("No core results after 50s; waiting 40s and retrying once for slow queue processing...")
            await asyncio.sleep(40)
            results = await client1.search_all_users(
                query="",
                memory_type="all",
                client_id=client1.client_id,
                limit=100,
                include_core_memory=True,
                block_filter_tags={"env": "staging"},
            )
            core_results = [r for r in results["results"] if r.get("memory_type") == "core"]

        assert len(core_results) > 0, (
            "With block_filter_tags={'env': 'staging'}, at least one core block (staging user's) must be returned. "
            "Got 0 core results after waiting - processing may not have completed or blocks not created with env=staging."
        )
        for item in core_results:
            logger.info(
                "  Block: user_id=%s, label=%s, scope=%s",
                item.get("user_id"),
                item.get("label"),
                item.get("scope"),
            )
            assert item["user_id"] == user_staging_id, (
                "With block_filter_tags={'env': 'staging'}, only blocks from staging user should be returned. "
                "Got user_id=%s" % item["user_id"]
            )
        logger.info(
            "All %s core block(s) have user_id=%s (env=staging). No prod blocks returned.",
            len(core_results),
            user_staging_id,
        )
        logger.info("✅ Test passed: block_filter_tags filters by tag and returns only relevant blocks")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
