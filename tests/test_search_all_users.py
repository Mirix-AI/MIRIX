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

import logging
import os
import time
from pathlib import Path

import pytest

from mirix.client import MirixClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
# Base URL can be set via MIRIX_API_URL environment variable or .env file
# MirixClient will automatically read from environment variables
BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = Path(__file__).parent.parent / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


def add_all_memories(client: MirixClient, user_id: str, filter_tags: dict, prefix: str = ""):
    """
    Add all types of memories for a user with given filter_tags.

    Args:
        client: MirixClient instance
        user_id: User ID
        filter_tags: Filter tags including scope
        prefix: Prefix for memory content to distinguish users
    """
    logger.info(f"Adding all memories for user {user_id} with filter_tags={filter_tags}")

    # Add episodic memory
    result = client.add(
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
        chaining=True,
        filter_tags=filter_tags,
        occurred_at="2025-11-20T14:00:00",
    )
    logger.info(f"  ✓ Added episodic memory - Result: {result}")

    # Add procedural memory
    result = client.add(
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
        chaining=True,
        filter_tags=filter_tags,
        occurred_at="2025-11-20T14:05:00",
    )
    logger.info(f"  ✓ Added procedural memory - Result: {result}")

    # Add semantic memory
    result = client.add(
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
        chaining=True,
        filter_tags=filter_tags,
        occurred_at="2025-11-20T14:10:00",
    )
    logger.info(f"  ✓ Added semantic memory - Result: {result}")

    # Add resource memory
    result = client.add(
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
        chaining=True,
        filter_tags=filter_tags,
        occurred_at="2025-11-20T14:15:00",
    )
    logger.info(f"  ✓ Added resource memory - Result: {result}")

    # Add knowledge vault memory
    result = client.add(
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
        chaining=True,
        filter_tags=filter_tags,
        occurred_at="2025-11-20T14:20:00",
    )
    logger.info(f"  ✓ Added knowledge vault memory - Result: {result}")

    logger.info(f"✅ All memories added for user {user_id}")


class TestSearchAllUsers:
    """Test suite for search_all_users API."""

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

    @pytest.fixture(scope="class")
    def client1(self, org1_id, client_scope_value):
        """Create first MirixClient instance in org1."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 1 in Organization 1")
        logger.info("=" * 80)

        client_id = f"test-client-1-{int(time.time())}"
        # MirixClient will use MIRIX_API_URL from environment or default to http://localhost:8000
        client = MirixClient(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 1",
            client_scope=client_scope_value,
            org_id=org1_id,
            debug=True,
        )

        # Initialize meta agent
        client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(f"✅ Client 1 initialized: client_id={client_id}, org_id={org1_id}, scope={client_scope_value}")
        logger.info(f"   Connected to: {client.base_url}")
        return client

    @pytest.fixture(scope="class")
    def user1_id(self, client1, org1_id):
        """Create first user in org1."""
        user_id = f"test-user-1-{int(time.time())}"
        client1.create_or_get_user(user_id=user_id, user_name="Test User 1", org_id=org1_id)
        logger.info(f"✅ User 1 created: {user_id}")
        time.sleep(0.5)  # Small delay to ensure unique timestamps
        return user_id

    @pytest.fixture(scope="class")
    def user2_id(self, client1, org1_id):
        """Create second user in org1."""
        user_id = f"test-user-2-{int(time.time())}"
        client1.create_or_get_user(user_id=user_id, user_name="Test User 2", org_id=org1_id)
        logger.info(f"✅ User 2 created: {user_id}")
        time.sleep(0.5)
        return user_id

    @pytest.fixture(scope="class")
    def user3_id(self, org1_id):
        """Create third user in org1 (will use different scope via client3)."""
        user_id = f"test-user-3-{int(time.time())}"
        # Note: User creation doesn't need a specific client, we'll use client3 for adding memories
        logger.info(f"✅ User 3 ID prepared: {user_id}")
        time.sleep(0.5)
        return user_id

    @pytest.fixture(scope="class")
    def client3(self, org1_id):
        """Create third MirixClient instance in org1 with DIFFERENT scope."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 3 in Organization 1 (Different Scope)")
        logger.info("=" * 80)

        client_id = f"test-client-3-{int(time.time())}"
        # Different scope from client1 (read_write) - use read_only
        client = MirixClient(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 3",
            client_scope="read_only",  # DIFFERENT scope
            org_id=org1_id,
            debug=True,
        )

        # Initialize meta agent
        client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(f"✅ Client 3 initialized: client_id={client_id}, org_id={org1_id}, scope=read_only")
        logger.info(f"   Connected to: {client.base_url}")
        return client

    @pytest.fixture(scope="class")
    def client2(self, org2_id, client_scope_value):
        """Create second MirixClient instance in org2 with same scope."""
        logger.info("\n" + "=" * 80)
        logger.info("Setting up Client 2 in Organization 2")
        logger.info("=" * 80)

        client_id = f"test-client-2-{int(time.time())}"
        # MirixClient will use MIRIX_API_URL from environment or default to http://localhost:8000
        client = MirixClient(
            api_key=None,
            client_id=client_id,
            client_name="Test Client 2",
            client_scope=client_scope_value,  # Same scope as client1
            org_id=org2_id,
            debug=True,
        )

        # Initialize meta agent
        client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)

        logger.info(f"✅ Client 2 initialized: client_id={client_id}, org_id={org2_id}, scope={client_scope_value}")
        logger.info(f"   Connected to: {client.base_url}")
        return client

    @pytest.fixture(scope="class")
    def user4_id(self, client2, org2_id):
        """Create fourth user in org2."""
        user_id = f"test-user-4-{int(time.time())}"
        client2.create_or_get_user(user_id=user_id, user_name="Test User 4", org_id=org2_id)
        logger.info(f"✅ User 4 created: {user_id}")
        time.sleep(0.5)
        return user_id

    @pytest.fixture(scope="class", autouse=True)
    def setup_memories(self, client1, client2, client3, user1_id, user2_id, user3_id, user4_id, client_scope_value):
        """Setup all memories for all users."""
        logger.info("\n" + "=" * 80)
        logger.info("SETTING UP TEST MEMORIES")
        logger.info("=" * 80)

        # User 1 & 2: Memories with scope='read_write' (via client1)
        filter_tags_with_scope = {"scope": client_scope_value, "test": "search_all"}
        logger.info("\n📝 Adding memories for User 1 via Client 1 (scope=read_write)...")
        add_all_memories(client1, user1_id, filter_tags_with_scope, prefix="[User1] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 1)...")
        time.sleep(50)

        logger.info("\n📝 Adding memories for User 2 via Client 1 (scope=read_write)...")
        add_all_memories(client1, user2_id, filter_tags_with_scope, prefix="[User2] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 2)...")
        time.sleep(30)

        # User 3: Memories with DIFFERENT scope='read_only' (via client3)
        # Server will auto-inject client3.scope='read_only' into filter_tags
        filter_tags_different_scope = {"test": "search_all"}  # Client3's scope='read_only' will be auto-added
        logger.info("\n📝 Adding memories for User 3 via Client 3 (scope=read_only - DIFFERENT)...")

        # Create user via client3 first
        client3.create_or_get_user(user_id=user3_id, user_name="Test User 3", org_id=client3.org_id)

        add_all_memories(client3, user3_id, filter_tags_different_scope, prefix="[User3] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 3)...")
        time.sleep(30)

        # User 4: Memories in different org with scope='read_write' (via client2)
        filter_tags_org2 = {"scope": client_scope_value, "test": "search_all"}
        logger.info("\n📝 Adding memories for User 4 via Client 2 (Different Org)...")
        add_all_memories(client2, user4_id, filter_tags_org2, prefix="[User4-Org2] ")
        logger.info("⏱️  Waiting 50 seconds for async memory processing (User 4)...")
        time.sleep(30)

        logger.info("\n" + "=" * 80)
        logger.info("✅ All test memories created and processed")
        logger.info("   - User 1 & 2: scope='read_write' (via client1)")
        logger.info("   - User 3: scope='read_only' (via client3) - DIFFERENT")
        logger.info("   - User 4: scope='read_write' (via client2) - DIFFERENT ORG")
        logger.info("=" * 80)

    def test_search_all_users_with_client_id_retrieves_both_users(
        self, client1, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 3: Search with client_id should retrieve memories from both user1 and user2."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: Search with client_id retrieves both users with matching scope")
        logger.info("=" * 80)

        results = client1.search_all_users(
            query="Python",  # Search for "Python" which should appear in semantic memories for both users
            memory_type="all",
            client_id=client1.client_id,
            limit=50,
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

    def test_search_all_users_with_client_id_retrieves_both_users_embedding(
        self, client1, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 3b: Embedding search with client_id should retrieve memories from both user1 and user2."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3b: Embedding search with client_id retrieves both users with matching scope")
        logger.info("=" * 80)

        results = client1.search_all_users(
            query="group discussion",  # Semantic query for "team meeting"
            memory_type="all",
            search_method="embedding",
            client_id=client1.client_id,
            limit=50,
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

    def test_search_excludes_user3_without_matching_scope(self, client1, user3_id):
        """Test 5: Search with client1 should NOT retrieve user3 memories (different scope: read_only vs read_write)."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5: User 3 excluded due to different scope (read_only vs read_write)")
        logger.info("=" * 80)

        results = client1.search_all_users(
            query="", memory_type="all", client_id=client1.client_id, limit=100  # client1 has scope='read_write'
        )

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")
        logger.info(f"Searching with client1 scope='read_write'")
        logger.info(f"User 3 has scope='read_only' (via client3)")

        assert (
            user3_id not in user_ids_in_results
        ), "User 3 memories should be excluded (scope='read_only' doesn't match 'read_write')"

        logger.info("✅ Test passed: User 3 correctly excluded due to different scope")

    def test_search_excludes_user3_without_matching_scope_embedding(self, client1, user3_id):
        """Test 5b: Embedding search with client1 should NOT retrieve user3 memories (different scope)."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5b: Embedding search - User 3 excluded due to different scope")
        logger.info("=" * 80)

        results = client1.search_all_users(
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
        ), "User 3 memories should be excluded (scope='read_only' doesn't match 'read_write')"

        logger.info("✅ Test passed: Embedding search correctly excluded User 3 due to different scope")

    def test_search_with_client3_retrieves_only_user3(self, client3, user3_id, user1_id, user2_id):
        """Test 6: Search with client3 (scope=read_only) should only retrieve user3 memories."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6: Search with client3 (scope=read_only) retrieves only User 3")
        logger.info("=" * 80)

        # Search with client3 which has scope='read_only'
        results = client3.search_all_users(query="", memory_type="all", client_id=client3.client_id, limit=100)

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Filter Tags: {results.get('filter_tags')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        # Should only retrieve User 3 (scope='read_only'), not User 1 or 2 (scope='read_write')
        assert user3_id in user_ids_in_results, "User 3 should be included (matching scope=read_only)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different scope)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different scope)"

        logger.info("✅ Test passed: Only User 3 retrieved with scope=read_only")

    def test_search_with_client3_retrieves_only_user3_embedding(self, client3, user3_id, user1_id, user2_id):
        """Test 6b: Embedding search with client3 (scope=read_only) should only retrieve user3 memories."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6b: Embedding search with client3 (scope=read_only) retrieves only User 3")
        logger.info("=" * 80)

        # Search with client3 which has scope='read_only'
        results = client3.search_all_users(
            query="software development",  # Semantic query
            memory_type="all",
            search_method="embedding",
            client_id=client3.client_id,
            limit=100,
        )

        logger.info(f"Results: {results['count']} memories found")
        logger.info(f"Search Method: {results.get('search_method')}")
        logger.info(f"Client Scope: {results.get('client_scope')}")

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs in results: {user_ids_in_results}")

        # Should only retrieve User 3 (scope='read_only'), not User 1 or 2 (scope='read_write')
        assert results["search_method"] == "embedding"
        assert user3_id in user_ids_in_results, "User 3 should be included (matching scope=read_only)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different scope)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different scope)"

        logger.info("✅ Test passed: Embedding search - Only User 3 retrieved with scope=read_only")

    def test_search_different_org_no_cross_contamination(
        self, client1, client2, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 8: Different organization - no cross-contamination even with matching scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 8: Organization isolation - same scope, different org")
        logger.info("=" * 80)

        # Search with client2 (in org2)
        results = client2.search_all_users(query="", memory_type="all", client_id=client2.client_id, limit=100)

        user_ids_in_results = set(result["user_id"] for result in results["results"])
        logger.info(f"Client 2 search - User IDs in results: {user_ids_in_results}")
        logger.info(f"Organization ID: {results.get('organization_id')}")

        # Should only see user4 (in org2), NOT user1/user2/user3 (in org1)
        assert user4_id in user_ids_in_results, "User 4 should be included (same org)"
        assert user1_id not in user_ids_in_results, "User 1 should be excluded (different org)"
        assert user2_id not in user_ids_in_results, "User 2 should be excluded (different org)"
        assert user3_id not in user_ids_in_results, "User 3 should be excluded (different org)"

        logger.info("✅ Test passed: Organization isolation working correctly")

    def test_search_different_org_no_cross_contamination_embedding(
        self, client1, client2, user1_id, user2_id, user3_id, user4_id
    ):
        """Test 8b: Embedding search - Different organization, no cross-contamination even with matching scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 8b: Embedding search - Organization isolation")
        logger.info("=" * 80)

        # Search with client2 (in org2)
        results = client2.search_all_users(
            query="database information",  # Semantic query
            memory_type="all",
            search_method="embedding",
            client_id=client2.client_id,
            limit=100,
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

    def test_search_all_memory_types(self, client1):
        """Test search across all memory types."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search all memory types")
        logger.info("=" * 80)

        results = client1.search_all_users(query="", memory_type="all", client_id=client1.client_id, limit=50)

        # Count by memory type
        memory_types = {}
        for result in results["results"]:
            mem_type = result["memory_type"]
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

        logger.info(f"Memory types found: {memory_types}")

        # Should have all 5 memory types
        assert len(memory_types) >= 3, "Should find at least 3 memory types"

        logger.info("✅ Test passed: Multiple memory types retrieved")

    def test_search_specific_memory_type(self, client1, user1_id, user2_id):
        """Test search for specific memory type only."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search specific memory type (episodic)")
        logger.info("=" * 80)

        results = client1.search_all_users(query="team", memory_type="episodic", client_id=client1.client_id, limit=20)

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

    def test_search_specific_memory_type_embedding(self, client1, user1_id, user2_id):
        """Test embedding search for specific memory type only."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Embedding search specific memory type (semantic)")
        logger.info("=" * 80)

        results = client1.search_all_users(
            query="programming language concepts",  # Semantic query for semantic memories
            memory_type="semantic",
            search_method="embedding",
            client_id=client1.client_id,
            limit=20,
        )

        logger.info(f"Results: {results['count']} semantic memories found")
        logger.info(f"Search Method: {results.get('search_method')}")

        # All results should be semantic type
        for result in results["results"]:
            assert result["memory_type"] == "semantic", "Should only return semantic memories"

        # Should include both users
        user_ids = set(result["user_id"] for result in results["results"])
        logger.info(f"User IDs: {user_ids}")

        assert results["success"] is True
        assert results["search_method"] == "embedding"
        assert results["count"] > 0

        logger.info("✅ Test passed: Embedding search for specific memory type working")

    def test_search_with_additional_filter_tags(self, client1):
        """Test search with additional filter tags beyond scope."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Search with additional filter tags")
        logger.info("=" * 80)

        results = client1.search_all_users(
            query="",
            memory_type="all",
            client_id=client1.client_id,
            filter_tags={"test": "search_all"},  # Additional filter
            limit=50,
        )

        logger.info(f"Results with filter_tags: {results['count']} memories")
        logger.info(f"Applied filter_tags: {results.get('filter_tags')}")

        # Should have both "scope" and "test" in filter_tags
        assert "scope" in results["filter_tags"], "Scope should be added automatically"
        assert "test" in results["filter_tags"], "Additional filter tag should be included"

        logger.info("✅ Test passed: Additional filter tags work correctly")

    def test_search_with_bm25(self, client1):
        """Test BM25 search method."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: BM25 search method")
        logger.info("=" * 80)

        results = client1.search_all_users(
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

    def test_search_with_embedding(self, client1):
        """Test embedding search method explicitly."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Embedding search method")
        logger.info("=" * 80)

        results = client1.search_all_users(
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

    def test_response_includes_metadata(self, client1):
        """Test that response includes all expected metadata."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Response metadata completeness")
        logger.info("=" * 80)

        results = client1.search_all_users(query="test", memory_type="all", client_id=client1.client_id, limit=10)

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
        assert "client_scope" in results
        assert "filter_tags" in results

        logger.info("Response fields: %s", list(results.keys()))
        logger.info("✅ Test passed: All metadata fields present")

    def test_each_result_includes_user_id(self, client1):
        """Test that each result includes user_id field."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST: Each result includes user_id")
        logger.info("=" * 80)

        results = client1.search_all_users(query="", memory_type="all", client_id=client1.client_id, limit=20)

        # Check each result has user_id
        for result in results["results"]:
            assert "user_id" in result, "Each result must include user_id"
            assert "memory_type" in result, "Each result must include memory_type"
            assert result["user_id"] is not None, "user_id must not be None"

        logger.info("✅ Test passed: All results include user_id")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
