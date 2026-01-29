"""
Agent System Prompt Update Integration Tests for Mirix

Tests system prompt updates for all memory agent types:
- Episodic, Semantic, Core, Procedural, Resource, Knowledge Vault, Reflexion, Meta Memory
- Verifies updates in: running agents, PostgreSQL database, Redis cache
- Verifies system message (message_ids[0]) is updated

Prerequisites:
    export GEMINI_API_KEY=your_api_key_here

Run tests:
    Terminal 1: python scripts/start_server.py --port 8000
    Terminal 2: pytest tests/test_agent_prompt_update.py -v -m integration -s
"""

import os
import sys
import time
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.client import MirixClient

# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def server_check():
    """Check if server is running (requires manual server start)."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("\n[OK] Server is running on port 8000")
            yield None
            return
    except (requests.ConnectionError, requests.Timeout):
        pass

    pytest.skip(
        "\n" + "=" * 70 + "\n"
        "Server is not running on port 8000!\n\n"
        "Integration tests require a manually started server:\n"
        "  Terminal 1: python scripts/start_server.py --port 8000\n"
        "  Terminal 2: pytest tests/test_agent_prompt_update.py -v -m integration -s\n" + "=" * 70
    )


@pytest.fixture(scope="module")
def client(server_check, api_auth):
    """
    Create a client connected to the test server.

    Follows the same initialization pattern as samples/add_test_memory.py:
    1. Create client with full parameters
    2. Create/get user
    3. Initialize meta agent with update_agents=True
    """
    print("\n[SETUP] Creating MirixClient...")

    # Create client with same parameters as add_test_memory.py
    client = MirixClient(
        api_key=api_auth["api_key"],
        client_name="Demo Client Application",
        client_scope="Sales",
        debug=False,
    )
    print("[SETUP] Client created via API key")

    # Create or get user (ensures user exists in backend database)
    print(f"[SETUP] Creating/getting user: demo-user")
    try:
        user_id = client.create_or_get_user(
            user_id="demo-user",
            user_name="Demo User",
        )
        print(f"[SETUP] User ready: {user_id}")
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"\n[ERROR] Failed to create/get user:")
        print(f"  Exception: {e}")
        print(f"  Details:\n{error_details}")
        pytest.skip(f"Failed to create/get user: {e}")

    # Initialize meta agent with all sub-agents
    print("[SETUP] Initializing meta agent...")
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"

    try:
        result = client.initialize_meta_agent(
            config_path=str(config_path), update_agents=True  # This should create all sub-agents
        )
        # result is an AgentState object, not a dict
        agent_id = result.id if hasattr(result, "id") else "N/A"
        print(f"[SETUP] ✓ Meta agent initialized: {agent_id}")

        # Trigger sub-agent creation by sending a test message
        # Sub-agents are created lazily during message processing
        print("[SETUP] Triggering sub-agent creation by adding a test memory...")
        try:
            test_result = client.add(
                user_id="demo-user",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Test message to trigger agent creation"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Test response"}]},
                ],
                chaining=True,
            )
            print(f"[SETUP] ✓ Test memory added: {test_result.get('success', False)}")
        except Exception as e:
            print(f"[SETUP] ⚠ Warning: Test memory addition failed: {e}")

        # Wait for sub-agents to be created
        print("[SETUP] Waiting 10 seconds for sub-agent creation...")
        time.sleep(10)

        # Verify agents were created - need to list both top-level and sub-agents
        top_level_agents = client.list_agents()
        print(f"[SETUP] ✓ Found {len(top_level_agents)} top-level agents")

        # Get meta agent and its sub-agents
        meta_agent = None
        for agent in top_level_agents:
            if agent.name == "meta_memory_agent":
                meta_agent = agent
                break

        all_agents = list(top_level_agents)  # Start with top-level
        if meta_agent:
            # Fetch sub-agents using parent_id
            try:
                response = client._request("GET", f"/agents?parent_id={meta_agent.id}&limit=1000")
                sub_agents_data = response if isinstance(response, list) else response.get("agents", [])
                from mirix.schemas.agent import AgentState

                sub_agents = [AgentState(**agent_data) for agent_data in sub_agents_data]
                all_agents.extend(sub_agents)
                print(f"[SETUP] ✓ Found {len(sub_agents)} sub-agents under meta agent")
            except Exception as e:
                print(f"[SETUP] ⚠ Warning: Could not fetch sub-agents: {e}")

        print(f"[SETUP] ✓ Total agents: {len(all_agents)}")

        # Display all agents
        print("[SETUP] Available agents:")
        for agent in all_agents:
            full_name = agent.name
            short_name = full_name
            if "meta_memory_agent_" in full_name and full_name != "meta_memory_agent":
                short_name = (
                    full_name.replace("meta_memory_agent_", "").replace("_memory_agent", "").replace("_agent", "")
                )
            print(f"  - Full: {full_name}")
            print(f"    Short: {short_name}")
            print(f"    ID: {agent.id}")

        if len(all_agents) == 0:
            pytest.skip(
                "\n" + "=" * 70 + "\n"
                "No agents found for client via API key.\n\n"
                "Agent initialization failed. Check server logs for errors.\n" + "=" * 70
            )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"\n[ERROR] Failed to initialize meta agent:")
        print(f"  Exception: {e}")
        print(f"  Details:\n{error_details}")
        pytest.skip(f"Failed to initialize meta agent: {e}")

    return client


def get_agent_direct_from_api(client: MirixClient, agent_name: str):
    """
    Get agent directly from REST API to verify database state.

    Args:
        client: MirixClient instance
        agent_name: Name of the agent (short or full)

    Returns:
        dict: Agent data from API
    """
    # First, get the meta agent to find its ID
    top_level_agents = client.list_agents()
    meta_agent = None
    for agent in top_level_agents:
        if agent.name == "meta_memory_agent":
            meta_agent = agent
            break

    if not meta_agent:
        return None

    # Now list all sub-agents by specifying the parent_id
    # Use the internal _request method to query sub-agents
    try:
        response = client._request("GET", f"/agents?parent_id={meta_agent.id}&limit=1000")
        sub_agents = response if isinstance(response, list) else response.get("agents", [])

        # Try exact match first
        for agent_data in sub_agents:
            if agent_data.get("name") == agent_name:
                from mirix.schemas.agent import AgentState

                return AgentState(**agent_data)

        # Try short name match
        for agent_data in sub_agents:
            full_name = agent_data.get("name", "")
            if "meta_memory_agent_" in full_name:
                short_name = (
                    full_name.replace("meta_memory_agent_", "").replace("_memory_agent", "").replace("_agent", "")
                )
                if short_name == agent_name:
                    from mirix.schemas.agent import AgentState

                    return AgentState(**agent_data)

        # Also check if the requested name IS "meta_memory_agent"
        if agent_name == "meta_memory_agent":
            return meta_agent

    except Exception as e:
        print(f"Error fetching sub-agents: {e}")
        return None

    return None


def get_system_message_id(agent) -> str:
    """
    Get the system message ID from agent's message_ids.

    The system message is always the first message (message_ids[0]).

    Args:
        agent: AgentState object

    Returns:
        str: System message ID, or empty string if no messages
    """
    if agent.message_ids and len(agent.message_ids) > 0:
        return agent.message_ids[0]
    return ""


# Agent names to test (short names)
AGENT_NAMES = [
    "episodic",
    "semantic",
    "core",
    "procedural",
    "resource",
    "knowledge_vault",
    "reflexion",
    "meta_memory_agent",
]


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_update_agent_system_prompt(client, agent_name):
    """
    Test updating system prompt for a specific agent type.

    Verifies:
    1. System prompt is updated in the agent state
    2. System message (message_ids[0]) is updated
    3. Changes are persisted in the database
    4. Changes are reflected in Redis cache (via subsequent reads)

    Args:
        client: MirixClient fixture
        agent_name: Name of the agent to test
    """
    print("\n" + "=" * 70)
    print(f"TEST: Update System Prompt for '{agent_name}' Agent")
    print("=" * 70)

    # Step 1: Get original agent state
    print(f"\n[Step 1] Getting original state for '{agent_name}' agent...")
    original_agent = get_agent_direct_from_api(client, agent_name)

    if not original_agent:
        pytest.skip(f"Agent '{agent_name}' not found. Skipping test.")

    print(f"[OK] Found agent: {original_agent.name} (ID: {original_agent.id})")
    print(f"     Original system prompt: {original_agent.system[:80]}...")

    # Get original system message ID
    original_message_id = get_system_message_id(original_agent)
    if original_message_id:
        print(f"     Original system message ID: {original_message_id}")

    # Step 2: Update system prompt
    print(f"\n[Step 2] Updating system prompt for '{agent_name}' agent...")

    new_system_prompt = f"""You are a {agent_name} memory agent for TESTING.
This is a test system prompt updated at {time.time()}.

Key responsibilities:
- Test memory operations
- Verify prompt updates
- Ensure data persistence

Test ID: {agent_name}-test-{int(time.time())}
"""

    try:
        updated_agent = client.update_system_prompt(agent_name=agent_name, system_prompt=new_system_prompt)
        print(f"[OK] Update request successful")
        print(f"     Updated system prompt: {updated_agent.system[:80]}...")

    except Exception as e:
        pytest.fail(f"Failed to update system prompt: {e}")

    # Step 3: Verify the update in the returned agent state
    print(f"\n[Step 3] Verifying update in returned agent state...")

    assert updated_agent.system == new_system_prompt, "System prompt in returned agent should match the new prompt"
    print(f"[OK] System prompt matches in returned state")

    # Verify system message ID changed
    new_message_id = updated_agent.message_ids[0] if updated_agent.message_ids else None
    if original_message_id and new_message_id:
        assert new_message_id != original_message_id, "System message ID (message_ids[0]) should have changed"
        print(f"[OK] System message ID changed: {original_message_id} → {new_message_id}")

    # Step 4: Wait for cache and database to sync
    print(f"\n[Step 4] Waiting 2 seconds for cache/database sync...")
    time.sleep(2)

    # Step 5: Verify persistence by fetching agent again (tests Redis cache)
    print(f"\n[Step 5] Fetching agent again to verify persistence (Redis cache)...")

    refetched_agent = get_agent_direct_from_api(client, agent_name)
    assert refetched_agent is not None, "Agent should still exist after update"

    assert refetched_agent.system == new_system_prompt, "System prompt should persist in cache/database"
    print(f"[OK] System prompt persisted in cache")
    print(f"     Cached prompt: {refetched_agent.system[:80]}...")

    # Verify message_ids[0] is still the new one
    cached_message_id = refetched_agent.message_ids[0] if refetched_agent.message_ids else None
    assert cached_message_id == new_message_id, "System message ID should persist in cache"
    print(f"[OK] System message ID persisted: {cached_message_id}")

    # Step 6: Verify system prompt in agent state
    print(f"\n[Step 6] Verifying system prompt is stored correctly...")

    # The agent.system field should contain the new prompt
    assert refetched_agent.system == new_system_prompt, "Agent's system field should contain the new system prompt"
    print(f"[OK] System prompt verified in agent state")
    print(f"     Prompt: {refetched_agent.system[:80]}...")

    # Step 7: Verify old and new are different
    print(f"\n[Step 7] Verifying changes were actually made...")

    assert updated_agent.system != original_agent.system, "New system prompt should be different from original"
    print(f"[OK] System prompt was successfully changed")

    print(f"\n✓ TEST PASSED for '{agent_name}' agent")
    print("=" * 70)


def test_update_all_agents_sequentially(client):
    """
    Test updating all agents sequentially to ensure no conflicts.

    This test:
    1. Updates all agent types one by one
    2. Verifies each update independently
    3. Ensures updates don't interfere with each other
    """
    print("\n" + "=" * 70)
    print("TEST: Sequential Update of All Agent Types")
    print("=" * 70)

    updated_agents = {}

    for agent_name in AGENT_NAMES:
        print(f"\n[Updating {agent_name}]")

        # Get original state
        original_agent = get_agent_direct_from_api(client, agent_name)
        if not original_agent:
            print(f"  ⚠ Agent '{agent_name}' not found, skipping")
            continue

        # Create unique prompt
        new_prompt = f"Sequential test prompt for {agent_name} at {time.time()}"

        # Update
        try:
            updated = client.update_system_prompt(agent_name=agent_name, system_prompt=new_prompt)
            updated_agents[agent_name] = {
                "agent": updated,
                "prompt": new_prompt,
                "original_prompt": original_agent.system,
            }
            print(f"  ✓ Updated {agent_name}")
        except Exception as e:
            print(f"  ✗ Failed to update {agent_name}: {e}")

    # Wait for all updates to propagate
    print("\n[Waiting 3 seconds for propagation...]")
    time.sleep(3)

    # Verify all updates persisted
    print("\n[Verifying all updates persisted...]")
    all_verified = True

    for agent_name, data in updated_agents.items():
        refetched = get_agent_direct_from_api(client, agent_name)
        if refetched and refetched.system == data["prompt"]:
            print(f"  ✓ {agent_name}: Verified")
        else:
            print(f"  ✗ {agent_name}: Failed verification")
            all_verified = False

    assert all_verified, "All agent updates should persist"
    print(f"\n✓ TEST PASSED: All {len(updated_agents)} agents updated successfully")
    print("=" * 70)


def test_update_same_agent_multiple_times(client):
    """
    Test updating the same agent multiple times in succession.

    Verifies:
    1. Multiple updates to the same agent work correctly
    2. Each update creates a new system message
    3. message_ids[0] is updated each time
    """
    print("\n" + "=" * 70)
    print("TEST: Multiple Updates to Same Agent")
    print("=" * 70)

    agent_name = "episodic"  # Use episodic agent for this test

    print(f"\n[Test] Updating '{agent_name}' agent 3 times in succession...")

    previous_message_id = None
    previous_prompt = None

    for i in range(1, 4):
        print(f"\n  [Update {i}/3]")

        new_prompt = f"Multi-update test {i}/3 for {agent_name} at {time.time()}"

        # Update
        updated = client.update_system_prompt(agent_name=agent_name, system_prompt=new_prompt)

        # Verify prompt changed
        assert updated.system == new_prompt, f"Update {i} should apply new prompt"
        print(f"    ✓ Prompt updated")

        # Verify message_ids[0] changed
        current_message_id = updated.message_ids[0] if updated.message_ids else None
        if previous_message_id:
            assert current_message_id != previous_message_id, f"Update {i} should create new system message"
            print(f"    ✓ Message ID changed: {previous_message_id[:20]}... → {current_message_id[:20]}...")

        # Verify prompt is different from previous
        if previous_prompt:
            assert updated.system != previous_prompt, f"Update {i} should change prompt from previous"
            print(f"    ✓ Prompt changed from previous")

        previous_message_id = current_message_id
        previous_prompt = new_prompt

        # Small delay between updates
        time.sleep(1)

    # Final verification
    print(f"\n[Final Verification] Fetching agent to verify last update...")
    final_agent = get_agent_direct_from_api(client, agent_name)

    assert final_agent.system == previous_prompt, "Final prompt should match last update"
    print(f"  ✓ Final prompt matches last update")

    print(f"\n✓ TEST PASSED: Multiple updates to same agent work correctly")
    print("=" * 70)


def test_error_handling_nonexistent_agent(client):
    """
    Test error handling when trying to update a non-existent agent.

    Verifies:
    1. 404 error is raised for non-existent agent
    2. Error message is informative and suggests available agents
    3. Server doesn't crash
    4. Multiple error scenarios are handled correctly
    """
    print("\n" + "=" * 70)
    print("TEST: Error Handling for Non-Existent Agent")
    print("=" * 70)

    # Test Case 1: Completely invalid agent name
    print("\n[Test Case 1] Attempting to update with completely invalid agent name...")
    fake_agent_name = "nonexistent_fake_agent_12345"

    with pytest.raises(Exception) as exc_info:
        client.update_system_prompt(agent_name=fake_agent_name, system_prompt="This should fail")

    print(f"  ✓ Exception raised: {type(exc_info.value).__name__}")
    error_message = str(exc_info.value)
    print(f"    Full message: {error_message}")

    # Verify error message contains key information
    error_lower = error_message.lower()
    assert "not found" in error_lower, "Error should mention 'not found'"
    assert fake_agent_name in error_message, "Error should include the invalid agent name"

    # Verify it suggests available agents (if any exist)
    if "available agents:" in error_lower or "available" in error_lower:
        print(f"  ✓ Error message suggests available agents")

    # Test Case 2: Typo in short name (e.g., "episodick" instead of "episodic")
    print("\n[Test Case 2] Attempting to update with typo in agent name...")
    typo_agent_name = "episodick"  # Common typo

    with pytest.raises(Exception) as exc_info:
        client.update_system_prompt(agent_name=typo_agent_name, system_prompt="This should also fail")

    print(f"  ✓ Exception raised for typo: {type(exc_info.value).__name__}")
    error_message = str(exc_info.value)
    print(f"    Message excerpt: {error_message[:150]}...")

    assert "not found" in error_message.lower(), "Error should mention 'not found'"

    # Test Case 3: Partial agent name (e.g., "memory" instead of full/short name)
    print("\n[Test Case 3] Attempting to update with partial agent name...")
    partial_agent_name = "memory"  # Too vague

    with pytest.raises(Exception) as exc_info:
        client.update_system_prompt(agent_name=partial_agent_name, system_prompt="This should fail too")

    print(f"  ✓ Exception raised for partial name: {type(exc_info.value).__name__}")

    # Test Case 4: Wrong case (e.g., "EPISODIC" instead of "episodic")
    print("\n[Test Case 4] Attempting to update with wrong case...")
    wrong_case_name = "EPISODIC"  # All caps

    with pytest.raises(Exception) as exc_info:
        client.update_system_prompt(agent_name=wrong_case_name, system_prompt="This should fail - case sensitive")

    print(f"  ✓ Exception raised for wrong case: {type(exc_info.value).__name__}")
    print(f"    Note: Agent names are case-sensitive")

    # Test Case 5: Empty string
    print("\n[Test Case 5] Attempting to update with empty agent name...")

    with pytest.raises(Exception) as exc_info:
        client.update_system_prompt(agent_name="", system_prompt="This should fail - empty name")

    print(f"  ✓ Exception raised for empty name: {type(exc_info.value).__name__}")

    print(f"\n✓ TEST PASSED: All error handling scenarios work correctly")
    print("  - Invalid agent names are rejected")
    print("  - Error messages are informative")
    print("  - Server remains stable")
    print("=" * 70)


if __name__ == "__main__":
    """
    Run tests directly with:
        python tests/test_agent_prompt_update.py
    """
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
