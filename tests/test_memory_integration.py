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
    previous_api_key = os.environ.get("MIRIX_API_KEY")
    os.environ["MIRIX_API_KEY"] = auth["api_key"]
    try:
        yield auth
    finally:
        if previous_api_key is None:
            os.environ.pop("MIRIX_API_KEY", None)
        else:
            os.environ["MIRIX_API_KEY"] = previous_api_key


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
    try:
        yield c
    finally:
        await c.close()


# =================================================================
# CORE INTEGRATION TESTS
# =================================================================


@pytest.mark.asyncio(loop_scope="module")
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


@pytest.mark.asyncio(loop_scope="module")
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


@pytest.mark.asyncio(loop_scope="module")
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


@pytest.mark.asyncio(loop_scope="module")
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


# =================================================================
# MESSAGE LIFECYCLE INTEGRATION TESTS
#
# Verify the system's message persistence contracts:
# - System prompts live on the agent, not as message rows
# - Retention=0 clients leave no message rows after processing
# - Retention=N clients keep exactly N message-sets, pruning older ones
# - Failed processing (e.g. context overflow) leaves no partial state
# =================================================================

MSG_TEST_USER_ID = "msg-lifecycle-user"
MSG_TEST_CLIENT_ID = "msg-lifecycle-client"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def msg_api_auth(server_process):
    """Provision a dedicated client for message lifecycle tests."""
    from conftest import _create_client_and_key

    auth = await _create_client_and_key(
        MSG_TEST_CLIENT_ID, TEST_ORG_ID, org_name="Demo Org"
    )
    return auth


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def msg_client(server_process, msg_api_auth):
    """MirixClient for message lifecycle tests, initialized once per module."""
    c = await MirixClient.create(
        api_key=msg_api_auth["api_key"],
        base_url="http://localhost:8000",
        debug=False,
    )
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"
    await c.initialize_meta_agent(
        config_path=str(config_path), update_agents=False
    )
    await c.create_or_get_user(
        user_id=MSG_TEST_USER_ID, user_name="Message Lifecycle User"
    )
    try:
        yield c
    finally:
        await c.close()


def _get_server():
    """Import and return the singleton AsyncServer."""
    from mirix.server.rest_api import get_server

    return get_server()


async def _get_message_rows(agent_id: str, user_id: str, org_id: str):
    """Query the messages table for a given (agent, user) pair.

    Returns all non-deleted message rows in chronological order.
    """
    from mirix.services.message_manager import MessageManager
    from mirix.schemas.client import Client

    mm = MessageManager()
    actor = Client(
        id="query-actor",
        organization_id=org_id,
        name="query",
        status="active",
        write_scope="test",
        read_scopes=["test"],
    )
    return await mm.get_messages_for_agent_user(
        agent_id=agent_id,
        user_id=user_id,
        actor=actor,
        limit=10000,
    )


async def _get_sub_agent_ids(client: MirixClient):
    """Return a dict mapping short agent name -> agent_id."""
    top_level = await client.list_agents()
    meta = next((a for a in top_level if a.name == "meta_memory_agent"), None)
    if not meta:
        return {}

    from mirix.schemas.agent import AgentState

    resp = await client._request(
        "GET", f"/agents?parent_id={meta.id}&limit=1000"
    )
    sub_agents = resp if isinstance(resp, list) else resp.get("agents", [])
    result = {"meta_memory_agent": meta.id}
    for data in sub_agents:
        agent = AgentState(**data)
        short = agent.name
        if "meta_memory_agent_" in short:
            short = (
                short.replace("meta_memory_agent_", "")
                .replace("_memory_agent", "")
                .replace("_agent", "")
            )
        result[short] = agent.id
    return result


# -----------------------------------------------------------------
# System prompt is stored on the agent, not as a message row
# -----------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_system_prompt_stored_on_agent_not_as_message(msg_client):
    """The system prompt lives in agent_state.system. Updating it should
    never create a message row with role='system' in the messages table.
    """
    client = msg_client
    agent_map = await _get_sub_agent_ids(client)

    agent_name = "episodic"
    if agent_name not in agent_map:
        pytest.skip(f"Agent '{agent_name}' not found")

    agent_id = agent_map[agent_name]

    new_prompt = (
        "You are an episodic memory agent for integration testing. "
        "Extract episodic events from conversations."
    )
    updated = await client.update_system_prompt(
        agent_name=agent_name, system_prompt=new_prompt
    )

    assert updated.system == new_prompt

    await asyncio.sleep(1)

    messages = await _get_message_rows(
        agent_id=agent_id,
        user_id=MSG_TEST_USER_ID,
        org_id=TEST_ORG_ID,
    )
    system_msgs = [m for m in messages if m.role == "system"]
    assert len(system_msgs) == 0, (
        f"System prompt should not be stored as a message row; "
        f"found {len(system_msgs)} system message(s)"
    )


# -----------------------------------------------------------------
# Retention=0: no message rows persist after processing
# -----------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_no_messages_persisted_with_zero_retention(msg_client):
    """When a client has message_set_retention_count=0, processing a
    conversation should leave zero message rows in the DB for every
    agent in the pipeline.
    """
    client = msg_client
    agent_map = await _get_sub_agent_ids(client)

    server = _get_server()
    db_client = await server.client_manager.get_client_by_id(
        MSG_TEST_CLIENT_ID
    )
    assert (db_client.message_set_retention_count or 0) == 0, (
        "Test client should default to retention=0"
    )

    result = await client.add(
        user_id=MSG_TEST_USER_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I had lunch with Alex at the Italian place on 5th Ave.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Got it, I'll remember your lunch with Alex.",
                    }
                ],
            },
        ],
    )
    assert result.get("success") is True

    print("  Waiting for queue processing (15s)...")
    await asyncio.sleep(15)

    for name, aid in agent_map.items():
        messages = await _get_message_rows(
            agent_id=aid,
            user_id=MSG_TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        assert len(messages) == 0, (
            f"Agent '{name}' should have 0 message rows with "
            f"retention=0, found {len(messages)}"
        )


# -----------------------------------------------------------------
# Retention=N: keeps at most N message-sets, prunes older ones
# -----------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_message_retention_prunes_to_limit(msg_client):
    """With message_set_retention_count=2, sending 3 conversations should
    leave at most 2 retained message rows for the meta agent. The oldest
    message-set is pruned after each save.
    """
    client = msg_client
    agent_map = await _get_sub_agent_ids(client)

    if "meta_memory_agent" not in agent_map:
        pytest.skip("Meta agent not found")

    meta_agent_id = agent_map["meta_memory_agent"]

    server = _get_server()
    from mirix.schemas.client import ClientUpdate

    updated_client = await server.client_manager.update_client(
        ClientUpdate(id=MSG_TEST_CLIENT_ID, message_set_retention_count=2)
    )
    assert updated_client.message_set_retention_count == 2

    try:
        conversations = [
            "I went hiking at Mount Tamalpais this morning.",
            "I finished reading The Great Gatsby last night.",
            "I started learning to play guitar today.",
        ]

        for i, text in enumerate(conversations):
            result = await client.add(
                user_id=MSG_TEST_USER_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Noted. (conversation {i+1})",
                            }
                        ],
                    },
                ],
            )
            assert result.get("success") is True

            print(f"  Sent conversation {i+1}/3, waiting 15s...")
            await asyncio.sleep(15)

        messages = await _get_message_rows(
            agent_id=meta_agent_id,
            user_id=MSG_TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        assert len(messages) <= 2, (
            f"Expected at most 2 retained message rows with "
            f"retention=2, found {len(messages)}"
        )

    finally:
        reset_client = await server.client_manager.update_client(
            ClientUpdate(id=MSG_TEST_CLIENT_ID, message_set_retention_count=0)
        )
        assert (reset_client.message_set_retention_count or 0) == 0

        from mirix.services.message_manager import MessageManager

        mm = MessageManager()
        await mm.hard_delete_user_messages_for_agent(
            agent_id=meta_agent_id,
            user_id=MSG_TEST_USER_ID,
            actor=reset_client,
            keep_newest_n=0,
        )


# -----------------------------------------------------------------
# Failed processing leaves no partial message state
# -----------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_failed_processing_leaves_no_messages(msg_client):
    """When processing fails (e.g. input exceeds the context window),
    no partial message rows should remain in the DB.
    """
    client = msg_client
    agent_map = await _get_sub_agent_ids(client)

    if "meta_memory_agent" not in agent_map:
        pytest.skip("Meta agent not found")

    server = _get_server()
    db_client = await server.client_manager.get_client_by_id(
        MSG_TEST_CLIENT_ID
    )
    assert (db_client.message_set_retention_count or 0) == 0

    # ~2M chars / ~500k tokens — well beyond any model's context window
    huge_text = "overflow " * 200_000

    try:
        await client.add(
            user_id=MSG_TEST_USER_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": huge_text}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Acknowledged."}
                    ],
                },
            ],
        )
    except Exception:
        pass

    print("  Waiting for processing attempt (20s)...")
    await asyncio.sleep(20)

    for name, aid in agent_map.items():
        messages = await _get_message_rows(
            agent_id=aid,
            user_id=MSG_TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        assert len(messages) == 0, (
            f"Agent '{name}' should have 0 message rows after a "
            f"failed processing attempt, found {len(messages)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
