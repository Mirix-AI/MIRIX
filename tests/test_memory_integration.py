"""
Memory System Integration Tests for Mirix

Integration tests for core memory operations via REST API.
Requires a manually started server. Automatically initializes users and agents on first run.

Prerequisites:
    export GEMINI_API_KEY=your_api_key_here

Run tests:
    Terminal 1: python scripts/start_server.py --port 8000
    Terminal 2: pytest tests/test_memory_integration.py -v -m integration

Test Coverage:
- client.add(): Add memories via conversation
- client.retrieve_with_conversation(): Retrieve memories with context
- client.retrieve_with_topic(): Retrieve memories by topic
- client.search(): Search across memory types
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import requests
from dotenv import load_dotenv

# Load .env file (optional - Mirix now loads .env automatically in mirix/settings.py)
# Kept here for backward compatibility
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix import EmbeddingConfig, LLMConfig
from mirix.client import MirixClient

TEST_USER_ID = "demo-user"
TEST_CLIENT_ID = "demo-client"
TEST_ORG_ID = "demo-org"

# Mark all tests as integration tests; one event loop per module so client and tests share it.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest.fixture(scope="module")
def server_process():
    """Check if server is running (requires manual server start)."""
    # Check if server is already running on port 8000
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("\n[OK] Server is running on port 8000")
            yield None  # No process to manage
            return
    except (requests.ConnectionError, requests.Timeout):
        pass

    # If not, fail with helpful message
    pytest.fail(
        "\n" + "=" * 70 + "\n"
        "Server is not running on port 8000!\n\n"
        "Integration tests require a manually started server:\n"
        "  Terminal 1: python scripts/start_server.py --port 8000\n"
        "  Terminal 2: pytest tests/test_memory_integration.py -v -m integration\n\n"
        "See tests/README.md for details.\n" + "=" * 70
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def api_auth(server_process):
    """Create org and client in DB once per module; yield auth for per-test client creation."""
    from conftest import _create_client_and_key

    auth = await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Demo Org")
    os.environ.setdefault("MIRIX_API_URL", "http://localhost:8000")
    os.environ["MIRIX_API_KEY"] = auth["api_key"]
    return auth


@pytest_asyncio.fixture
async def client(server_process, api_auth):
    """Create a new MirixClient per test in the current loop (avoids shared httpx + closed loop)."""
    c = await MirixClient.create(
        api_key=api_auth["api_key"],
        base_url="http://localhost:8000",
        debug=False,
    )
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"
    await c.initialize_meta_agent(config_path=str(config_path), update_agents=False)
    if c._meta_agent:
        print(f"[OK] Meta agent ready: {c._meta_agent.id}")
    return c


# =================================================================
# CORE INTEGRATION TESTS
# =================================================================


@pytest.mark.asyncio
async def test_add(client):
    """Test adding memories using client.add()."""
    print("\n[TEST] Adding memory via client.add()...")

    result = await client.add(
        user_id=TEST_USER_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I had a meeting with Sarah from design team at 2 PM. We discussed new UI mockups and selected the blue color scheme.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Got it! I've recorded your meeting with Sarah about the UI design and color selection.",
                    }
                ],
            },
        ],
    )

    assert result is not None
    assert result.get("success") is True
    print(f"[OK] Memory added successfully")


@pytest.mark.asyncio
async def test_retrieve_with_conversation(client):
    """Test retrieving memories with conversation context."""
    print("\n[TEST] Retrieving memories with conversation...")

    await client.add(
        user_id=TEST_USER_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I completed the database migration project yesterday. It took 3 hours and everything went smoothly.",
                    }
                ],
            }
        ],
    )

    await asyncio.sleep(2)  # Wait for processing

    result = await client.retrieve_with_conversation(
        user_id=TEST_USER_ID,
        messages=[{"role": "user", "content": [{"type": "text", "text": "What work did I complete recently?"}]}],
        limit=10,
    )

    assert result is not None
    assert result.get("success") is True
    assert "memories" in result
    print(f"[OK] Retrieved memories successfully")

    # Display results
    if result.get("memories"):
        for memory_type, items in result["memories"].items():
            if items and items.get("total_count", 0) > 0:
                print(f"  - {memory_type}: {items['total_count']} items")


@pytest.mark.asyncio
async def test_retrieve_with_topic(client):
    """Test retrieving memories by topic."""
    print("\n[TEST] Retrieving memories by topic...")

    await client.add(
        user_id=TEST_USER_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need to deploy the application to production. The deployment process includes running tests, building artifacts, and deploying to the server.",
                    }
                ],
            }
        ],
    )

    await asyncio.sleep(2)  # Wait for processing

    result = await client.retrieve_with_topic(user_id=TEST_USER_ID, topic="deployment", limit=5)

    assert result is not None
    assert result.get("success") is True
    assert "memories" in result
    print(f"[OK] Retrieved by topic: {result.get('topic')}")

    # Display results
    if result.get("memories"):
        for memory_type, items in result["memories"].items():
            if items and items.get("total_count", 0) > 0:
                print(f"  - {memory_type}: {items['total_count']} items")


@pytest.mark.asyncio
async def test_search(client):
    """Test searching memories."""
    print("\n[TEST] Searching memories...")

    await client.add(
        user_id=TEST_USER_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Team meeting scheduled for next Monday at 10 AM. We will discuss Q1 planning and budget allocation.",
                    }
                ],
            }
        ],
    )

    await asyncio.sleep(2)  # Wait for processing

    print("  [1] Searching across all memory types...")
    result_all = await client.search(user_id=TEST_USER_ID, query="meeting planning", memory_type="all", limit=10)

    assert result_all is not None
    assert result_all.get("success") is True
    print(f"  [OK] Found {result_all.get('count', 0)} results across all types")

    print("  [2] Searching episodic memory...")
    result_episodic = await client.search(
        user_id=TEST_USER_ID,
        query="meeting",
        memory_type="episodic",
        search_field="summary",
        search_method="bm25",
        limit=5,
    )

    assert result_episodic is not None
    assert result_episodic.get("success") is True
    print(f"  [OK] Found {result_episodic.get('count', 0)} episodic results")

    print("  [3] Searching with embedding method...")
    result_embedding = await client.search(
        user_id=TEST_USER_ID,
        query="team collaboration",
        memory_type="episodic",
        search_field="details",
        search_method="embedding",
        limit=5,
    )

    assert result_embedding is not None
    assert result_embedding.get("success") is True
    print(f"  [OK] Found {result_embedding.get('count', 0)} results with embedding search")

    print("[OK] All search tests completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
