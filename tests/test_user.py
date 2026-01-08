"""
User Creation and Auto-Creation Integration Tests for Mirix

Tests two scenarios:
1. Explicit user creation with create_or_get_user() before adding memory
2. Automatic user creation when add() is called with a non-existent user_id

Prerequisites:
    export GEMINI_API_KEY=your_api_key_here

Run tests:
    Terminal 1: python scripts/start_server.py --port 8000
    Terminal 2: pytest tests/test_user.py -v -m integration -s
"""

import os
import sys
import time
import uuid
import requests
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.client import MirixClient

TEST_ORG_ID = "test-user-org"
TEST_CLIENT_ID = "test-user-client"

# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
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
        "\n" + "="*70 + "\n"
        "Server is not running on port 8000!\n\n"
        "Integration tests require a manually started server:\n"
        "  Terminal 1: python scripts/start_server.py --port 8000\n"
        "  Terminal 2: pytest tests/test_user.py -v -m integration -s\n"
        + "="*70
    )


@pytest.fixture(scope="module")
def client(server_check, api_auth):
    """Create a client connected to the test server."""
    client = MirixClient(
        api_key=api_auth["api_key"],
        debug=False,
    )
    
    # Initialize meta agent
    print("\n[SETUP] Initializing meta agent...")
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"
    
    try:
        result = client.initialize_meta_agent(
            config_path=str(config_path),
            update_agents=False
        )
        print(f"[OK] Meta agent initialized: {result.get('agent_id', 'N/A')}")
    except Exception as e:
        print(f"[WARNING] Meta agent initialization: {e}")
    
    return client


def user_exists(client: MirixClient, user_id: str) -> bool:
    """
    Check if a user exists in the backend database.
    
    Args:
        client: MirixClient instance
        user_id: User ID to check
        
    Returns:
        bool: True if user exists, False otherwise
    """
    try:
        response = client._request("GET", f"/users/{user_id}")
        return response is not None and "id" in response
    except Exception:
        return False


def test_explicit_user_creation_then_add_memory(client):
    """
    Scenario 1: Create user explicitly with create_or_get_user(), then add memory.
    
    Steps:
        1. Generate a unique user_id
        2. Call create_or_get_user() to create the user
        3. Verify user exists in database
        4. Add memory for this user
        5. Wait for processing (10-15 seconds)
        6. Verify memory was added successfully
    """
    print("\n" + "="*70)
    print("TEST 1: Explicit User Creation with create_or_get_user()")
    print("="*70)
    
    # Step 1: Generate unique user ID
    user_id = f"test-explicit-user-{uuid.uuid4().hex[:8]}"
    print(f"\n[Step 1] Generated user_id: {user_id}")
    
    # Step 2: Create user explicitly
    print(f"[Step 2] Creating user with create_or_get_user()...")
    created_user_id = client.create_or_get_user(
        user_id=user_id,
        user_name=f"Test User {user_id}",
        org_id=TEST_ORG_ID
    )
    print(f"[OK] User created: {created_user_id}")
    assert created_user_id == user_id, "Returned user_id should match requested user_id"
    
    # Step 3: Verify user exists in database
    print(f"[Step 3] Verifying user exists in database...")
    time.sleep(1)  # Small delay to ensure database write is complete
    assert user_exists(client, user_id), f"User {user_id} should exist in database"
    print(f"[OK] User {user_id} verified in database")
    
    # Step 4: Add memory for this user
    print(f"[Step 4] Adding memory for user {user_id}...")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "My favorite color is blue and I work at Acme Corp."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Got it! I've noted that your favorite color is blue and you work at Acme Corp."}]
        }
    ]
    
    filter_tags = {
        "test_type": "explicit_creation",
        "account_id": "ACC-001"
    }
    
    response = client.add(
        user_id=user_id,
        messages=messages,
        filter_tags=filter_tags,
        chaining=False,
        verbose=False
    )
    
    print(f"[OK] Memory add request submitted")
    print(f"     Response: {response}")
    
    # Step 5: Wait for processing
    print(f"[Step 5] Waiting 15 seconds for memory processing...")
    time.sleep(15)
    print(f"[OK] Processing complete")
    
    # Step 6: Verify memory can be retrieved
    print(f"[Step 6] Retrieving memory to verify...")
    retrieve_response = client.retrieve_with_conversation(
        user_id=user_id,
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is my favorite color?"}]}],
        limit=5
    )
    
    print(f"[OK] Memory retrieval successful")
    print(f"     Retrieved {len(retrieve_response.get('memories', []))} memories")
    
    # Verify we got some memories back
    assert "memories" in retrieve_response, "Response should contain memories"
    print("\n✓ TEST 1 PASSED: User was created explicitly and memory was added successfully")


def test_auto_user_creation_on_add_memory(client):
    """
    Scenario 2: Add memory with non-existent user_id, verify auto-creation.
    
    Steps:
        1. Generate a unique user_id
        2. Verify user does NOT exist yet
        3. Call add() directly without create_or_get_user()
        4. Wait for processing (10-15 seconds)
        5. Verify user was auto-created in database
        6. Verify memory was added successfully
    """
    print("\n" + "="*70)
    print("TEST 2: Automatic User Creation on add()")
    print("="*70)
    
    # Step 1: Generate unique user ID
    user_id = f"test-auto-user-{uuid.uuid4().hex[:8]}"
    print(f"\n[Step 1] Generated user_id: {user_id}")
    
    # Step 2: Verify user does NOT exist yet
    print(f"[Step 2] Verifying user does NOT exist yet...")
    assert not user_exists(client, user_id), f"User {user_id} should NOT exist yet"
    print(f"[OK] Confirmed user {user_id} does not exist")
    
    # Step 3: Add memory WITHOUT creating user first
    print(f"[Step 3] Adding memory WITHOUT calling create_or_get_user()...")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "I just moved to San Francisco and I'm looking for an apartment."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Welcome to San Francisco! I'll help you find an apartment."}]
        }
    ]
    
    filter_tags = {
        "test_type": "auto_creation",
        "region": "West"
    }
    
    response = client.add(
        user_id=user_id,
        messages=messages,
        filter_tags=filter_tags,
        chaining=False,
        verbose=False
    )
    
    print(f"[OK] Memory add request submitted")
    print(f"     Response: {response}")
    
    # Step 4: Wait for processing
    print(f"[Step 4] Waiting 15 seconds for memory processing and user auto-creation...")
    time.sleep(15)
    print(f"[OK] Processing complete")
    
    # Step 5: Verify user was auto-created
    print(f"[Step 5] Verifying user was auto-created in database...")
    assert user_exists(client, user_id), f"User {user_id} should have been auto-created"
    print(f"[OK] User {user_id} was auto-created successfully")
    
    # Step 6: Verify memory can be retrieved
    print(f"[Step 6] Retrieving memory to verify...")
    retrieve_response = client.retrieve_with_conversation(
        user_id=user_id,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Where am I moving to?"}]}],
        limit=5
    )
    
    print(f"[OK] Memory retrieval successful")
    print(f"     Retrieved {len(retrieve_response.get('memories', []))} memories")
    
    # Verify we got some memories back
    assert "memories" in retrieve_response, "Response should contain memories"
    print("\n✓ TEST 2 PASSED: User was auto-created and memory was added successfully")


def test_idempotent_create_or_get_user(client):
    """
    Bonus test: Verify create_or_get_user() is idempotent.
    
    Steps:
        1. Create a user
        2. Call create_or_get_user() again with same user_id
        3. Verify it returns the existing user without error
    """
    print("\n" + "="*70)
    print("TEST 3: Idempotent create_or_get_user()")
    print("="*70)
    
    # Step 1: Create user
    user_id = f"test-idempotent-user-{uuid.uuid4().hex[:8]}"
    print(f"\n[Step 1] Creating user: {user_id}")
    
    created_user_id_1 = client.create_or_get_user(
        user_id=user_id,
        user_name="Idempotent Test User",
        org_id=TEST_ORG_ID
    )
    print(f"[OK] User created (1st call): {created_user_id_1}")
    
    # Step 2: Call again with same user_id
    print(f"[Step 2] Calling create_or_get_user() again with same user_id...")
    time.sleep(1)  # Small delay
    
    created_user_id_2 = client.create_or_get_user(
        user_id=user_id,
        user_name="Idempotent Test User",
        org_id=TEST_ORG_ID
    )
    print(f"[OK] User retrieved (2nd call): {created_user_id_2}")
    
    # Step 3: Verify same user_id returned
    assert created_user_id_1 == created_user_id_2, "Should return same user_id on repeated calls"
    assert user_exists(client, user_id), "User should still exist in database"
    
    print("\n✓ TEST 3 PASSED: create_or_get_user() is idempotent")


if __name__ == "__main__":
    """
    Run tests directly with:
        python tests/test_user.py
    """
    pytest.main([__file__, "-v", "-s", "-m", "integration"])


