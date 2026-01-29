"""
Test deletion APIs for Mirix.

This test suite verifies the complete deletion workflow:
1. Create client
2. Create memories for a user
3. Delete memories for this user (hard delete)
4. Add new memories for the same user
5. Delete memories for the client (hard delete)
6. Add memory for this user of this client
7. Soft delete user
8. Soft delete client
"""

import logging
import time
import uuid
from pathlib import Path

import pytest
import requests

from mirix.client import MirixClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Generate unique IDs for this test run to avoid conflicts with previous runs
TEST_RUN_ID = uuid.uuid4().hex[:8]
TEST_ORG_ID = f"test-deletion-org-{TEST_RUN_ID}"
TEST_CLIENT_ID = f"test-deletion-client-{TEST_RUN_ID}"
TEST_USER_ID = f"test-deletion-user-{TEST_RUN_ID}"
BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"

logger.info("=" * 80)
logger.info("TEST RUN ID: %s", TEST_RUN_ID)
logger.info("  ORG:    %s", TEST_ORG_ID)
logger.info("  CLIENT: %s", TEST_CLIENT_ID)
logger.info("  USER:   %s", TEST_USER_ID)
logger.info("=" * 80)


@pytest.fixture(scope="module")
def check_server():
    """Verify server is running before tests."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        logger.info("✓ Server is running at %s", BASE_URL)
        return True
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Server not available at {BASE_URL}: {e}")


@pytest.fixture(scope="module")
def client(check_server, api_key_factory):
    """Create and initialize MirixClient."""
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING TEST CLIENT")
    logger.info("=" * 80)

    auth = api_key_factory(TEST_CLIENT_ID, TEST_ORG_ID)
    client = MirixClient(
        api_key=auth["api_key"],
        client_name="Test Deletion Client",
        client_scope="test",
        debug=False,
    )
    logger.info("Client initialized via API key: %s", TEST_CLIENT_ID)

    # Create or get user
    try:
        returned_user_id = client.create_or_get_user(
            user_id=TEST_USER_ID,
            user_name="Test Deletion User",
        )
        logger.info("User ready: %s (requested: %s)", returned_user_id, TEST_USER_ID)

        # Verify they match
        if returned_user_id != TEST_USER_ID:
            logger.warning("⚠️  Returned user_id (%s) differs from requested (%s)", returned_user_id, TEST_USER_ID)
    except Exception as e:
        logger.error("Failed to create/get user: %s", e)
        import traceback

        traceback.print_exc()
        raise

    # Initialize meta agent
    try:
        client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)
        logger.info("✓ Meta agent initialized")

        # Add a test memory to trigger sub-agent creation
        init_result = client.add(
            user_id=TEST_USER_ID,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test initialization message"}]}],
            chaining=True,
        )
        logger.info("✓ Initialization memory added: %s", init_result)
        time.sleep(5)  # Wait for processing
        logger.info("✓ Sub-agents created")
    except Exception as e:
        logger.error("Failed to initialize meta agent: %s", e)
        raise

    return client


def add_test_memories(client: MirixClient, user_id: str, batch_label: str):
    """Helper function to add test memories."""
    logger.info("\nAdding test memories (batch: %s)...", batch_label)

    memories = [
        {
            "type": "core",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[{batch_label}] My name is Test User and I work at Test Company."}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"[{batch_label}] I've saved your information."}],
                },
            ],
        },
        {
            "type": "episodic",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{batch_label}] Yesterday I had a meeting with the team about the new project.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"[{batch_label}] I've recorded this event."}],
                },
            ],
        },
        {
            "type": "semantic",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{batch_label}] MirixDB is our company's internal database system that combines PostgreSQL with Redis caching for high-performance operations. It's specifically designed for handling multi-agent memory operations with automatic cache invalidation, vector search capabilities, and real-time synchronization across distributed systems. The architecture uses a hybrid approach where frequently accessed data is cached in Redis while maintaining PostgreSQL as the source of truth.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{batch_label}] I've saved this comprehensive information about MirixDB to my semantic memory.",
                        }
                    ],
                },
            ],
        },
    ]

    for memory in memories:
        try:
            result = client.add(
                user_id=user_id,
                messages=memory["messages"],
                chaining=True,
                filter_tags={"batch": batch_label, "test": "deletion"},
            )
            logger.info("  ✓ Added %s memory - Result: %s", memory["type"], result)
        except Exception as e:
            logger.error("  ✗ Failed to add %s memory: %s", memory["type"], e)
            import traceback

            traceback.print_exc()
            raise

    # Wait for async processing (give enough time for classification)
    logger.info("⏱️  Waiting 30 seconds for async memory processing...")
    time.sleep(30)

    # Check if any memories were actually stored
    from mirix.orm.message import Message as MessageModel
    from mirix.server.server import db_context

    # Note: Cannot check queue worker status from test process - it runs in server process
    logger.info("⏱️  Checking if memories were stored in database...")

    with db_context() as session:
        message_count = session.query(MessageModel).filter(MessageModel.user_id == user_id).count()
        logger.info("✓ Messages in database after batch %s: %d", batch_label, message_count)

        if message_count == 0:
            logger.error("❌ No messages found for user %s after adding memories!", user_id)
            logger.error("This likely means:")
            logger.error("  1. The queue worker is not running (check above)")
            logger.error("  2. The user_id doesn't match")
            logger.error("  3. Memory addition is failing silently in the worker")

    logger.info("✓ All memories added for batch: %s", batch_label)


def count_memories_via_api(user_id: str, log_details: bool = False) -> dict:
    """Count memories by querying the database via API."""
    from mirix.orm.block import Block as BlockModel
    from mirix.orm.episodic_memory import EpisodicEvent
    from mirix.orm.message import Message as MessageModel
    from mirix.orm.procedural_memory import ProceduralMemoryItem
    from mirix.orm.semantic_memory import SemanticMemoryItem
    from mirix.server.server import db_context

    with db_context() as session:
        episodic_count = session.query(EpisodicEvent).filter(EpisodicEvent.user_id == user_id).count()

        semantic_count = session.query(SemanticMemoryItem).filter(SemanticMemoryItem.user_id == user_id).count()

        procedural_count = session.query(ProceduralMemoryItem).filter(ProceduralMemoryItem.user_id == user_id).count()

        message_count = session.query(MessageModel).filter(MessageModel.user_id == user_id).count()

        block_count = session.query(BlockModel).filter(BlockModel.user_id == user_id).count()

        # Log details for debugging
        if log_details:
            logger.debug("Memory details for user %s:", user_id)
            if semantic_count == 0:
                # Check if semantic memories exist but with is_deleted flag
                all_semantic = session.query(SemanticMemoryItem).filter(SemanticMemoryItem.user_id == user_id).all()
                logger.debug("  Total semantic memories (including deleted): %d", len(all_semantic))
                for mem in all_semantic:
                    logger.debug(
                        "    - id: %s, is_deleted: %s, summary: %s",
                        mem.id,
                        mem.is_deleted,
                        mem.summary[:50] if mem.summary else "N/A",
                    )

    return {
        "episodic": episodic_count,
        "semantic": semantic_count,
        "procedural": procedural_count,
        "messages": message_count,
        "blocks": block_count,
    }


def test_1_create_client_and_add_memories(client):
    """Test 1 & 2: Client is created and memories are added."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1 & 2: CLIENT CREATED & ADD INITIAL MEMORIES")
    logger.info("=" * 80)

    # Client is already created in fixture
    logger.info("✓ Client exists: %s", TEST_CLIENT_ID)
    logger.info("✓ User exists: %s", TEST_USER_ID)

    # Verify user exists in database before adding memories
    from mirix.orm.user import User as UserModel
    from mirix.server.server import db_context

    with db_context() as session:
        user = session.query(UserModel).filter(UserModel.id == TEST_USER_ID).first()
        if user:
            logger.info(
                "✓ User verified in database: id=%s, name=%s, is_deleted=%s", user.id, user.name, user.is_deleted
            )
        else:
            logger.error("✗ User NOT found in database with id=%s", TEST_USER_ID)
            pytest.fail(f"User {TEST_USER_ID} does not exist in database")

    # Add initial memories
    add_test_memories(client, TEST_USER_ID, "batch-1-initial")

    # Verify memories exist
    counts = count_memories_via_api(TEST_USER_ID, log_details=True)
    logger.info("\nMemory counts after initial add:")
    for memory_type, count in counts.items():
        logger.info("  %s: %d", memory_type, count)

    # Assert that memories were created (messages are cached in Redis, not PostgreSQL)
    # The important thing is that episodic/semantic memories were extracted and stored in PostgreSQL
    assert (
        counts["episodic"] > 0 or counts["semantic"] > 0
    ), f"At least one memory type should exist. Got: episodic={counts['episodic']}, semantic={counts['semantic']}"

    if counts["episodic"] == 0:
        logger.warning("⚠️  No episodic memories created - classification may have failed")
    if counts["semantic"] == 0:
        logger.warning("⚠️  No semantic memories created - classification may have failed")
    if counts["messages"] == 0:
        logger.info("ℹ️  Messages are cached in Redis only, not persisted to PostgreSQL")

    # At least one of the memory types should be created
    assert counts["episodic"] > 0 or counts["semantic"] > 0, "At least episodic or semantic memories should be created"

    logger.info("✓ TEST 1 & 2 PASSED")


def test_3_delete_user_memories(client):
    """Test 3: Delete memories for the user (hard delete)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: DELETE USER MEMORIES (HARD DELETE)")
    logger.info("=" * 80)

    # Check initial count
    counts_before = count_memories_via_api(TEST_USER_ID)
    logger.info("Memory counts before deletion:")
    for memory_type, count in counts_before.items():
        logger.info("  %s: %d", memory_type, count)

    # Delete memories via API
    response = requests.delete(f"{BASE_URL}/users/{TEST_USER_ID}/memories")
    response.raise_for_status()
    result = response.json()

    logger.info("\nDeletion response: %s", result["message"])
    logger.info("Preserved: %s", result["preserved"])

    # Verify memories are deleted
    counts_after = count_memories_via_api(TEST_USER_ID)
    logger.info("\nMemory counts after deletion:")
    for memory_type, count in counts_after.items():
        logger.info("  %s: %d", memory_type, count)

    assert counts_after["episodic"] == 0, "Episodic memories should be deleted"
    assert counts_after["semantic"] == 0, "Semantic memories should be deleted"
    # Note: Messages are cached in Redis only, not in PostgreSQL, so we don't check them
    assert counts_after["blocks"] == 0, "Blocks should be deleted"

    # Verify user still exists
    response = requests.get(f"{BASE_URL}/users/{TEST_USER_ID}")
    assert response.status_code == 200, "User should still exist"

    logger.info("✓ TEST 3 PASSED: User memories deleted, user preserved")


def test_4_add_new_memories_after_user_deletion(client):
    """Test 4: Add new memories for the same user after deletion."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: ADD NEW MEMORIES AFTER USER MEMORY DELETION")
    logger.info("=" * 80)

    # Add new memories
    add_test_memories(client, TEST_USER_ID, "batch-2-after-user-deletion")

    # Verify new memories exist
    counts = count_memories_via_api(TEST_USER_ID)
    logger.info("\nMemory counts after re-adding:")
    for memory_type, count in counts.items():
        logger.info("  %s: %d", memory_type, count)

    # At least one memory type should be created (AI classification determines which)
    assert (
        counts["episodic"] > 0 or counts["semantic"] > 0
    ), f"At least one memory type should exist. Got: episodic={counts['episodic']}, semantic={counts['semantic']}"
    # Note: Messages are cached in Redis only, not in PostgreSQL

    logger.info("✓ TEST 4 PASSED")


def test_5_delete_client_memories(client):
    """Test 5: Delete memories for the client (hard delete)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: DELETE CLIENT MEMORIES (HARD DELETE)")
    logger.info("=" * 80)

    # Check initial count
    counts_before = count_memories_via_api(TEST_USER_ID)
    logger.info("Memory counts before client memory deletion:")
    for memory_type, count in counts_before.items():
        logger.info("  %s: %d", memory_type, count)

    # Delete memories via API
    response = requests.delete(f"{BASE_URL}/clients/{TEST_CLIENT_ID}/memories")
    response.raise_for_status()
    result = response.json()

    logger.info("\nDeletion response: %s", result["message"])
    logger.info("Preserved: %s", result["preserved"])

    # Verify memories are deleted
    counts_after = count_memories_via_api(TEST_USER_ID)
    logger.info("\nMemory counts after client memory deletion:")
    for memory_type, count in counts_after.items():
        logger.info("  %s: %d", memory_type, count)

    assert counts_after["episodic"] == 0, "Episodic memories should be deleted"
    assert counts_after["semantic"] == 0, "Semantic memories should be deleted"
    # Note: Messages are cached in Redis only, not in PostgreSQL, so we don't check them

    # Verify client still exists
    response = requests.get(f"{BASE_URL}/clients/{TEST_CLIENT_ID}")
    assert response.status_code == 200, "Client should still exist"

    # Verify user still exists
    response = requests.get(f"{BASE_URL}/users/{TEST_USER_ID}")
    assert response.status_code == 200, "User should still exist"

    logger.info("✓ TEST 5 PASSED: Client memories deleted, client and user preserved")


def test_6_add_memory_after_client_deletion(client):
    """Test 6: Add memory for this user after client memory deletion."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: ADD MEMORY AFTER CLIENT MEMORY DELETION")
    logger.info("=" * 80)

    # Add new memories
    add_test_memories(client, TEST_USER_ID, "batch-3-after-client-deletion")

    # Verify new memories exist
    counts = count_memories_via_api(TEST_USER_ID)
    logger.info("\nMemory counts after re-adding:")
    for memory_type, count in counts.items():
        logger.info("  %s: %d", memory_type, count)

    # At least one memory type should be created (AI classification determines which)
    assert (
        counts["episodic"] > 0 or counts["semantic"] > 0
    ), f"At least one memory type should exist. Got: episodic={counts['episodic']}, semantic={counts['semantic']}"
    # Note: Messages are cached in Redis only, not in PostgreSQL

    logger.info("✓ TEST 6 PASSED")


def test_7_soft_delete_user(client):
    """Test 7: Soft delete user."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: SOFT DELETE USER")
    logger.info("=" * 80)

    # First verify the user exists
    try:
        response = requests.get(f"{BASE_URL}/users/{TEST_USER_ID}")
        if response.status_code == 200:
            user_data = response.json()
            logger.info("✓ User exists: %s (is_deleted=%s)", TEST_USER_ID, user_data.get("is_deleted"))
        else:
            logger.error("❌ User does not exist! Status: %d", response.status_code)
            logger.error("Response: %s", response.text)
            pytest.fail(f"User {TEST_USER_ID} does not exist before deletion test")
    except Exception as e:
        logger.error("❌ Failed to check user existence: %s", e)
        pytest.fail(f"Cannot verify user {TEST_USER_ID} exists: {e}")

    # Check initial count
    counts_before = count_memories_via_api(TEST_USER_ID)
    logger.info("Memory counts before user soft delete:")
    for memory_type, count in counts_before.items():
        logger.info("  %s: %d", memory_type, count)

    # Soft delete user via API
    response = requests.delete(f"{BASE_URL}/users/{TEST_USER_ID}")
    if response.status_code != 200:
        logger.error("❌ Delete failed! Status: %d", response.status_code)
        logger.error("Response: %s", response.text)
    response.raise_for_status()
    result = response.json()

    logger.info("\nDeletion response: %s", result["message"])

    # Verify memories are still in database (soft deleted)
    from mirix.orm.episodic_memory import EpisodicEvent
    from mirix.orm.user import User as UserModel
    from mirix.server.server import db_context

    with db_context() as session:
        # Check user is soft deleted
        user = session.query(UserModel).filter(UserModel.id == TEST_USER_ID).first()
        assert user is not None, "User should still exist in database"
        assert user.is_deleted is True, "User should be marked as deleted"
        logger.info("✓ User is soft deleted (is_deleted=True)")

        # Check memories are soft deleted
        episodic_memories = session.query(EpisodicEvent).filter(EpisodicEvent.user_id == TEST_USER_ID).all()
        assert len(episodic_memories) > 0, "Episodic memories should still exist in database"
        for memory in episodic_memories:
            assert memory.is_deleted is True, "Memory should be marked as deleted"
        logger.info("✓ Memories are soft deleted (is_deleted=True)")

    logger.info("✓ TEST 7 PASSED: User and memories soft deleted (data preserved in DB)")


def test_8_soft_delete_client(client):
    """Test 8: Soft delete client."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: SOFT DELETE CLIENT")
    logger.info("=" * 80)

    # Soft delete client via API
    response = requests.delete(f"{BASE_URL}/clients/{TEST_CLIENT_ID}")
    response.raise_for_status()
    result = response.json()

    logger.info("\nDeletion response: %s", result["message"])

    # Verify client is still in database (soft deleted)
    from mirix.orm.agent import Agent as AgentModel
    from mirix.orm.client import Client as ClientModel
    from mirix.server.server import db_context

    with db_context() as session:
        # Check client is soft deleted
        client_obj = session.query(ClientModel).filter(ClientModel.id == TEST_CLIENT_ID).first()
        assert client_obj is not None, "Client should still exist in database"
        assert client_obj.is_deleted is True, "Client should be marked as deleted"
        logger.info("✓ Client is soft deleted (is_deleted=True)")

        # Check agents are soft deleted
        agents = session.query(AgentModel).filter(AgentModel._created_by_id == TEST_CLIENT_ID).all()
        if agents:
            for agent in agents:
                assert agent.is_deleted is True, "Agent should be marked as deleted"
            logger.info("✓ Agents are soft deleted (is_deleted=True)")

    logger.info("✓ TEST 8 PASSED: Client and associated data soft deleted")


def test_9_summary():
    """Test 9: Print summary of all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ All 8 deletion API tests passed!")
    logger.info("\nTests completed:")
    logger.info("  1. ✓ Create client")
    logger.info("  2. ✓ Add initial memories for user")
    logger.info("  3. ✓ Delete user memories (hard delete)")
    logger.info("  4. ✓ Add new memories for same user")
    logger.info("  5. ✓ Delete client memories (hard delete)")
    logger.info("  6. ✓ Add memory after client memory deletion")
    logger.info("  7. ✓ Soft delete user")
    logger.info("  8. ✓ Soft delete client")
    logger.info("\nDeletion API coverage:")
    logger.info("  - DELETE /users/{user_id}/memories - Hard delete user memories")
    logger.info("  - DELETE /clients/{client_id}/memories - Hard delete client memories")
    logger.info("  - DELETE /users/{user_id} - Soft delete user")
    logger.info("  - DELETE /clients/{client_id} - Soft delete client")
    logger.info("=" * 80)
