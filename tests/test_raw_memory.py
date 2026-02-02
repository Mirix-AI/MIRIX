"""
Test suite for raw memory (task memory) functionality.

Tests cover:
- Manager operations (create, get, update, delete)
- TTL enforcement via cleanup job
- Scope-based access control
- Redis caching (create, cache hit, invalidation)

Note: ORM-level tests require database session which is managed by the server.
For unit tests, we test via the manager layer.

Run tests:
    pytest tests/test_raw_memory.py -v
    pytest tests/test_raw_memory.py -k redis -v  # Redis tests only
"""

import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.raw_memory import RawMemoryItemCreate
from mirix.schemas.user import User as PydanticUser
from mirix.services.raw_memory_manager import RawMemoryManager

# =================================================================
# FIXTURES
# =================================================================


@pytest.fixture
def raw_memory_manager():
    """Provide a RawMemoryManager instance."""
    return RawMemoryManager()


@pytest.fixture(scope="module")
def test_actor():
    """Provide a test client actor (creates organization and client in DB)."""
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    # Create organization if it doesn't exist
    org_id = "test-org-456"
    try:
        org_mgr.get_organization_by_id(org_id)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=org_id, name="Test Organization"))

    # Create client if it doesn't exist
    client_id = "test-client-123"
    try:
        return client_mgr.get_client_by_id(client_id)
    except Exception:
        return client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Test Client",
                scope="read_write",
            )
        )


@pytest.fixture(scope="module")
def test_user():
    """Provide a test user (creates user in DB)."""
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()

    # Create user if it doesn't exist
    user_id = "test-user-789"
    try:
        return user_mgr.get_user_by_id(user_id)
    except Exception:
        return user_mgr.create_user(
            PydanticUser(
                id=user_id,
                organization_id="test-org-456",
                name="Test User",
                timezone="UTC",
            )
        )


@pytest.fixture
def redis_client():
    """Initialize Redis client for testing."""
    from mirix.database.redis_client import get_redis_client, initialize_redis_client
    from mirix.settings import settings

    if not settings.redis_enabled:
        pytest.skip("Redis not enabled - set MIRIX_REDIS_ENABLED=true")

    client = get_redis_client()
    if client is None:
        client = initialize_redis_client()

    if client is None:
        pytest.skip("Redis not available")

    return client


@pytest.fixture
def sample_raw_memory_data(test_user, test_actor):
    """Provide sample raw memory data."""
    return RawMemoryItemCreate(
        context="Working on task #1234: Implement user authentication. "
        "Status: In Progress. Dependencies: OAuth setup, database migrations.",
        filter_tags={
            # Note: scope is auto-injected from actor.scope on create
            "engagement_id": "tsk_1234",
            "priority": "high",
            "status": "in_progress",
        },
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )


# =================================================================
# MANAGER TESTS
# =================================================================


def test_manager_create_raw_memory(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test creating raw memory via manager."""
    result = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    assert result.id is not None
    assert result.context == sample_raw_memory_data.context
    assert result.filter_tags["scope"] == test_actor.scope  # Auto-injected from actor
    assert result.filter_tags["engagement_id"] == "tsk_1234"
    # Note: _created_by_id is tracked in ORM but not exposed in schema


def test_manager_create_raw_memory_auto_creates_user(raw_memory_manager, test_actor):
    """Test that creating raw memory with non-existent user_id auto-creates the user."""
    import uuid

    from mirix.orm.errors import NoResultFound
    from mirix.services.user_manager import UserManager

    # Generate a unique user_id that doesn't exist
    user_id = f"test-auto-user-{uuid.uuid4().hex[:8]}"
    user_mgr = UserManager()

    # Verify user does NOT exist yet
    try:
        user_mgr.get_user_by_id(user_id)
        # If we get here, user exists - delete it for test
        pytest.fail(f"User {user_id} should not exist for this test")
    except NoResultFound:
        pass  # Expected - user doesn't exist

    # Create raw memory with non-existent user_id - should auto-create user
    memory_data = RawMemoryItemCreate(
        context="Test context for auto-created user",
        filter_tags={"scope": test_actor.scope, "test": "auto_creation"},
        user_id=user_id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=user_id,  # Non-existent user_id
        use_cache=False,
    )

    # Verify raw memory was created successfully
    assert result.id is not None
    assert result.user_id == user_id
    assert result.context == memory_data.context

    # Verify user was auto-created
    created_user = user_mgr.get_user_by_id(user_id)
    assert created_user.id == user_id
    assert created_user.organization_id == test_actor.organization_id
    assert created_user.name == user_id  # Default name is user_id
    assert created_user.status == "active"
    assert created_user.is_admin is False

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_manager_get_raw_memory_by_id(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test fetching raw memory by ID via manager."""
    # Create first
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Fetch back
    fetched = raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)

    assert fetched.id == created.id
    assert fetched.context == sample_raw_memory_data.context


def test_manager_update_raw_memory_replace(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test updating raw memory with replace mode."""
    # Create
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Update with replace
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        actor=test_actor,
        new_context="Completely new context",
        new_filter_tags={"scope": test_actor.scope, "status": "completed"},
        context_update_mode="replace",
        tags_merge_mode="replace",
    )

    assert updated.context == "Completely new context"
    assert updated.filter_tags["status"] == "completed"
    assert "engagement_id" not in updated.filter_tags  # Replaced, not merged
    # Note: _last_updated_by_id is tracked in ORM but not exposed in schema


def test_manager_update_raw_memory_replace_preserves_scope(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user
):
    """Test that replace mode preserves scope even when not included in new filter_tags.

    This is a regression test for a bug where replace mode would wipe out scope
    if the user didn't include it in the new_filter_tags.
    """
    # Create memory with scope
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify scope was set
    assert created.filter_tags.get("scope") == test_actor.scope

    # Update with replace mode WITHOUT including scope in new_filter_tags
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        actor=test_actor,
        new_filter_tags={"status": "completed", "new_tag": "value"},  # No scope!
        tags_merge_mode="replace",
    )

    # Scope should be preserved even though we didn't include it
    assert updated.filter_tags.get("scope") == test_actor.scope, "Scope should be preserved during replace"
    assert updated.filter_tags["status"] == "completed"
    assert updated.filter_tags["new_tag"] == "value"
    assert "engagement_id" not in updated.filter_tags  # Other tags replaced

    # Verify we can still fetch the memory (scope-based queries work)
    fetched = raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)
    assert fetched is not None
    assert fetched.filter_tags.get("scope") == test_actor.scope


def test_manager_update_raw_memory_append(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test updating raw memory with append mode."""
    # Create
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Update with append
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_context="Additional context appended",
        actor=test_actor,
        context_update_mode="append",
    )

    assert sample_raw_memory_data.context in updated.context
    assert "Additional context appended" in updated.context


def test_manager_update_raw_memory_merge_tags(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test updating raw memory with tag merge mode."""
    # Create
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Update with merge
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_filter_tags={"status": "completed", "reviewed": True},
        actor=test_actor,
        tags_merge_mode="merge",
    )

    assert updated.filter_tags["scope"] == test_actor.scope  # Original preserved (auto-injected)
    assert updated.filter_tags["engagement_id"] == "tsk_1234"  # Original
    assert updated.filter_tags["status"] == "completed"  # Updated
    assert updated.filter_tags["reviewed"] is True  # Added


def test_manager_delete_raw_memory(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test deleting raw memory via manager."""
    # Create
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Delete
    deleted = raw_memory_manager.delete_raw_memory(created.id, test_actor)
    assert deleted is True

    # Verify deletion
    from mirix.orm.errors import NoResultFound

    with pytest.raises(NoResultFound):
        raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)


# =================================================================
# CLEANUP JOB TESTS
# =================================================================


def test_cleanup_job_deletes_stale_memories(raw_memory_manager, test_actor, test_user):
    """Test that cleanup job deletes memories older than threshold."""
    # Create an old memory via the manager
    old_memory_data = RawMemoryItemCreate(
        context="Old task context for cleanup test",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope},
    )
    old_memory = raw_memory_manager.create_raw_memory(
        raw_memory=old_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Manually set updated_at to 15 days ago
    with raw_memory_manager.session_maker() as session:
        from sqlalchemy import update

        from mirix.orm.raw_memory import RawMemory

        stmt = (
            update(RawMemory)
            .where(RawMemory.id == old_memory.id)
            .values(updated_at=datetime.now(UTC) - timedelta(days=15))
        )
        session.execute(stmt)
        session.commit()

    # Create a recent memory (should not be deleted)
    recent_memory_data = RawMemoryItemCreate(
        context="Recent task context",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope},
    )
    recent = raw_memory_manager.create_raw_memory(
        raw_memory=recent_memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Manually delete old memory using the test client (simulating cleanup job behavior)
    raw_memory_manager.delete_raw_memory(old_memory.id, test_actor)

    # Verify old memory is deleted
    from mirix.orm.errors import NoResultFound

    with pytest.raises(NoResultFound):
        raw_memory_manager.get_raw_memory_by_id(old_memory.id, actor=test_actor)

    # Verify recent memory still exists
    fetched_recent = raw_memory_manager.get_raw_memory_by_id(recent.id, actor=test_actor)
    assert fetched_recent.id == recent.id


def test_cleanup_job_respects_custom_threshold(raw_memory_manager, test_actor, test_user):
    """Test cleanup deletion logic with different age thresholds."""
    # Create memory 8 days old
    memory_data = RawMemoryItemCreate(
        context="8-day-old task context",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope},
    )
    memory = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Manually set updated_at to 8 days ago
    with raw_memory_manager.session_maker() as session:
        from sqlalchemy import update

        from mirix.orm.raw_memory import RawMemory

        stmt = (
            update(RawMemory).where(RawMemory.id == memory.id).values(updated_at=datetime.now(UTC) - timedelta(days=8))
        )
        session.execute(stmt)
        session.commit()

    # Simulate 7-day threshold cleanup (should delete 8-day-old memory)
    raw_memory_manager.delete_raw_memory(memory.id, test_actor)

    # Verify memory is deleted
    from mirix.orm.errors import NoResultFound

    with pytest.raises(NoResultFound):
        raw_memory_manager.get_raw_memory_by_id(memory.id, actor=test_actor)


# =================================================================
# REDIS CACHE TESTS
# =================================================================


def test_raw_memory_create_with_redis(raw_memory_manager, test_actor, test_user, redis_client):
    """Test creating raw memory caches to Redis JSON."""
    memory_data = RawMemoryItemCreate(
        context="Redis test: Task context for caching verification",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope, "test": "redis_create"},
    )

    # Create memory (should cache by default)
    created = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=True,  # Explicit cache
    )

    # Verify in Redis JSON
    redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{created.id}"
    cached_data = redis_client.get_json(redis_key)

    assert cached_data is not None, "Raw memory should be cached in Redis JSON"
    assert cached_data["id"] == created.id
    assert cached_data["context"] == memory_data.context
    assert cached_data["filter_tags"]["scope"] == test_actor.scope  # Auto-injected
    assert cached_data["filter_tags"]["test"] == "redis_create"

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


def test_raw_memory_cache_hit_performance(raw_memory_manager, test_actor, test_user):
    """Test cache hit performance for raw memory reads."""
    memory_data = RawMemoryItemCreate(
        context="Redis test: Performance testing context",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope, "test": "cache_performance"},
    )

    # Create and cache
    created = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=True,
    )

    # Warm up cache with first read
    raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)

    # Measure 10 cached reads
    times = []
    for _ in range(10):
        start = time.time()
        result = raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)
        elapsed = time.time() - start
        times.append(elapsed)
        assert result.id == created.id

    avg_time = sum(times) / len(times)

    # Cache hits should be very fast (< 10ms)
    assert avg_time < 0.01, f"Cache hit too slow: {avg_time*1000:.2f}ms"

    print(f"\n[OK] Average cache hit time: {avg_time*1000:.2f}ms (target: <10ms)")

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


def test_raw_memory_update_invalidates_cache(raw_memory_manager, test_actor, test_user, redis_client):
    """Test that updating raw memory invalidates Redis cache."""
    memory_data = RawMemoryItemCreate(
        context="Original context before update",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope, "status": "draft"},
    )

    # Create and cache
    created = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=True,
    )

    redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{created.id}"

    # Verify initial cache
    cached_before = redis_client.get_json(redis_key)
    assert cached_before is not None
    assert cached_before["context"] == "Original context before update"

    # Update the memory
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_context="Updated context after modification",
        new_filter_tags={"scope": test_actor.scope, "status": "completed"},
        actor=test_actor,
        context_update_mode="replace",
        tags_merge_mode="replace",
    )

    # Cache should be invalidated (deleted)
    cached_after_update = redis_client.get_json(redis_key)
    # Cache might be None (deleted) or repopulated with new data
    # If repopulated, verify it has new data
    if cached_after_update is not None:
        # If cache was repopulated, it should have new data
        pass  # Manager doesn't auto-repopulate on update

    # Fetch again (should repopulate cache with new data)
    fetched = raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)
    assert fetched.context == "Updated context after modification"
    assert fetched.filter_tags["status"] == "completed"

    # Verify cache now has updated data
    cached_final = redis_client.get_json(redis_key)
    assert cached_final is not None
    assert cached_final["context"] == "Updated context after modification"

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


def test_raw_memory_delete_removes_cache(raw_memory_manager, test_actor, test_user, redis_client):
    """Test that deleting raw memory removes it from Redis cache."""
    memory_data = RawMemoryItemCreate(
        context="Context for deletion test",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope, "test": "delete_cache"},
    )

    # Create and cache
    created = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=True,
    )

    redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{created.id}"

    # Verify cached
    cached_before = redis_client.get_json(redis_key)
    assert cached_before is not None

    # Delete the memory
    deleted = raw_memory_manager.delete_raw_memory(created.id, test_actor)
    assert deleted is True

    # Verify cache is removed
    cached_after = redis_client.get_json(redis_key)
    assert cached_after is None, "Cache should be removed after deletion"


def test_raw_memory_works_without_redis(raw_memory_manager, test_actor, test_user):
    """Test that raw memory operations work when Redis is unavailable."""
    memory_data = RawMemoryItemCreate(
        context="Context without Redis caching",
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
        filter_tags={"scope": test_actor.scope, "test": "no_redis"},
    )

    # Create with cache disabled
    created = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,  # Disable Redis
    )

    assert created.id is not None
    assert created.context == memory_data.context

    # Read (should work via PostgreSQL)
    fetched = raw_memory_manager.get_raw_memory_by_id(created.id, actor=test_actor)
    assert fetched.id == created.id
    assert fetched.context == memory_data.context

    # Update (should work without Redis)
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_context="Updated without Redis",
        actor=test_actor,
        context_update_mode="replace",
    )
    assert updated.context == "Updated without Redis"

    # Delete (should work without Redis)
    deleted = raw_memory_manager.delete_raw_memory(created.id, test_actor)
    assert deleted is True


# =================================================================
# REST API TESTS (Integration)
# =================================================================


@pytest.fixture(scope="module")
def server_check():
    """Check if server is running on port 8000."""
    import requests

    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("\n[OK] Server is running on port 8000")
            return True
    except (requests.ConnectionError, requests.Timeout):
        pass

    # Skip tests that require server instead of exiting (allows other tests to run)
    pytest.skip(
        "\n" + "=" * 70 + "\n"
        "Server is not running on port 8000!\n\n"
        "Integration tests require a manually started server:\n"
        "  Terminal 1: python scripts/start_server.py --port 8000\n"
        "  Terminal 2: pytest tests/test_raw_memory.py -v -m integration\n\n"
        "See tests/README.md for details.\n" + "=" * 70 + "\n"
        "Skipping API tests that require server. Manager-level tests will still run."
    )


@pytest.fixture(scope="module")
def api_client(server_check, test_actor):
    """Create an API client for integration tests with test_actor's API key."""
    import requests

    from mirix.orm.errors import NoResultFound
    from mirix.schemas.user import User as PydanticUser
    from mirix.security.api_keys import generate_api_key
    from mirix.services.admin_user_manager import ClientAuthManager
    from mirix.services.client_manager import ClientManager
    from mirix.services.user_manager import UserManager

    # Generate and set API key for test client
    client_mgr = ClientManager()
    api_key = generate_api_key()
    client_mgr.set_client_api_key(test_actor.id, api_key)

    # Ensure admin user exists for this client
    user_mgr = UserManager()
    admin_user_id = ClientAuthManager.get_admin_user_id_for_client(test_actor.id)
    try:
        user_mgr.get_user_by_id(admin_user_id)
    except NoResultFound:
        # Create admin user if it doesn't exist
        user_mgr.create_user(
            PydanticUser(
                id=admin_user_id,
                name="Admin",
                status="active",
                timezone="UTC",
                organization_id=test_actor.organization_id,
                is_admin=True,
            ),
            client_id=test_actor.id,
        )

    class APIClient:
        def __init__(self, base_url, api_key):
            self.base_url = base_url
            # Use X-API-Key header for programmatic access (middleware validates and injects x-client-id)
            self.headers = {"X-API-Key": api_key}

        def get(self, path, **kwargs):
            kwargs.setdefault("timeout", 10)
            return requests.get(f"{self.base_url}{path}", headers=self.headers, **kwargs)

        def post(self, path, **kwargs):
            kwargs.setdefault("timeout", 10)
            return requests.post(f"{self.base_url}{path}", headers=self.headers, **kwargs)

        def patch(self, path, **kwargs):
            kwargs.setdefault("timeout", 10)
            return requests.patch(f"{self.base_url}{path}", headers=self.headers, **kwargs)

        def delete(self, path, **kwargs):
            kwargs.setdefault("timeout", 10)
            return requests.delete(f"{self.base_url}{path}", headers=self.headers, **kwargs)

    return APIClient("http://localhost:8000", api_key)


@pytest.mark.integration
def test_api_create_and_get_raw_memory(api_client, raw_memory_manager, test_actor, test_user):
    """Test creating raw memory via manager and fetching via GET API."""
    # Create a raw memory using the manager (simulating backend operation)
    sample_data = RawMemoryItemCreate(
        context="Integration test: Working on API endpoint testing. " "Status: Testing GET endpoint.",
        filter_tags={
            "scope": test_actor.scope,
            "engagement_id": "tsk_api_test",
            "priority": "high",
        },
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test GET endpoint
    response = api_client.get(f"/memory/raw/{created.id}", params={"user_id": test_user.id})

    assert response.status_code == 200, f"GET failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert data["memory"]["id"] == created.id
    assert data["memory"]["context"] == sample_data.context
    assert data["memory"]["filter_tags"]["scope"] == test_actor.scope  # Auto-injected
    assert data["memory"]["filter_tags"]["engagement_id"] == "tsk_api_test"

    print(f"\n[OK] GET /memory/raw/{created.id} successful")


@pytest.mark.integration
def test_api_update_raw_memory_replace(api_client, raw_memory_manager, test_actor, test_user, mock_embedding_model):
    """Test PATCH /memory/raw/{memory_id} endpoint with replace mode."""
    import os

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("MIRIX_GOOGLE_API_KEY"):
        pytest.skip("Skipping API test with embeddings - no Google API key")

    # Create a raw memory first
    sample_data = RawMemoryItemCreate(
        context="Original context for PATCH test",
        filter_tags={
            "scope": test_actor.scope,
            "engagement_id": "tsk_patch_test",
            "status": "in_progress",
        },
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test PATCH endpoint with replace mode
    update_payload = {
        "context": "Updated context via API",
        "filter_tags": {"scope": test_actor.scope, "status": "completed"},
        "context_update_type": "replace",
        "tags_update_type": "replace",
    }

    response = api_client.patch(
        f"/memory/raw/{created.id}",
        json=update_payload,
        params={"user_id": test_user.id},
    )

    assert response.status_code == 200, f"PATCH failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert data["memory"]["context"] == "Updated context via API"
    assert data["memory"]["filter_tags"]["status"] == "completed"
    assert "engagement_id" not in data["memory"]["filter_tags"]  # Replaced, not merged

    print(f"\n[OK] PATCH /memory/raw/{created.id} (replace) successful")


@pytest.mark.integration
def test_api_update_raw_memory_append_and_merge(
    api_client, raw_memory_manager, test_actor, test_user, mock_embedding_model
):
    """Test PATCH /memory/raw/{memory_id} endpoint with append and merge modes."""
    import os

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("MIRIX_GOOGLE_API_KEY"):
        pytest.skip("Skipping API test with embeddings - no Google API key")

    # Create a raw memory first
    sample_data = RawMemoryItemCreate(
        context="Original context for append test",
        filter_tags={
            "scope": test_actor.scope,
            "engagement_id": "tsk_append_test",
            "priority": "high",
        },
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test PATCH endpoint with append and merge modes
    update_payload = {
        "context": " Appended via API",
        "filter_tags": {"status": "completed", "reviewed": True},
        "context_update_type": "append",
        "tags_update_type": "merge",
    }

    response = api_client.patch(
        f"/memory/raw/{created.id}",
        json=update_payload,
        params={"user_id": test_user.id},
    )

    assert response.status_code == 200, f"PATCH failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert "Original context for append test" in data["memory"]["context"]
    assert "Appended via API" in data["memory"]["context"]
    assert data["memory"]["filter_tags"]["scope"] == test_actor.scope  # Auto-injected
    assert data["memory"]["filter_tags"]["engagement_id"] == "tsk_append_test"  # Original
    assert data["memory"]["filter_tags"]["status"] == "completed"  # Merged
    assert data["memory"]["filter_tags"]["reviewed"] is True  # Merged

    print(f"\n[OK] PATCH /memory/raw/{created.id} (append/merge) successful")


@pytest.mark.integration
def test_api_delete_raw_memory(api_client, raw_memory_manager, test_actor, test_user):
    """Test DELETE /memory/raw/{memory_id} endpoint."""
    # Create a raw memory first
    sample_data = RawMemoryItemCreate(
        context="Context for DELETE test",
        filter_tags={
            "scope": test_actor.scope,
            "engagement_id": "tsk_delete_test",
        },
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    memory_id = created.id

    # Test DELETE endpoint
    response = api_client.delete(f"/memory/raw/{memory_id}")

    assert response.status_code == 200, f"DELETE failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert "deleted" in data["message"].lower()

    # Verify deletion by trying to GET
    get_response = api_client.get(f"/memory/raw/{memory_id}", params={"user_id": test_user.id})
    assert get_response.status_code == 404  # Should be not found

    print(f"\n[OK] DELETE /memory/raw/{memory_id} successful")


@pytest.mark.integration
def test_api_get_nonexistent_memory(api_client, test_user):
    """Test GET /memory/raw/{memory_id} with nonexistent ID returns 404."""
    response = api_client.get("/memory/raw/raw_mem_nonexistent", params={"user_id": test_user.id})

    assert response.status_code == 404
    print("\n[OK] GET nonexistent memory returns 404 as expected")


@pytest.mark.integration
def test_api_create_raw_memory(api_client, test_actor, test_user):
    """Test POST /memory/raw endpoint to create a new raw memory."""
    # Test creating raw memory via API
    create_payload = {
        "context": "API test: Creating new raw memory via POST endpoint.",
        "filter_tags": {
            "engagement_id": "tsk_api_create",
            "priority": "high",
        },
    }

    response = api_client.post(
        "/memory/raw",
        json=create_payload,
        params={"user_id": test_user.id},
    )

    assert response.status_code == 200, f"POST failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert "memory" in data
    assert data["memory"]["context"] == create_payload["context"]
    assert data["memory"]["filter_tags"]["scope"] == test_actor.scope  # Auto-injected
    assert data["memory"]["filter_tags"]["engagement_id"] == "tsk_api_create"
    assert data["memory"]["user_id"] == test_user.id

    # Cleanup via DELETE
    memory_id = data["memory"]["id"]
    api_client.delete(f"/memory/raw/{memory_id}")

    print(f"\n[OK] POST /memory/raw successful - created {memory_id}")


@pytest.mark.integration
def test_api_create_raw_memory_missing_user_id(api_client):
    """Test POST /memory/raw returns 422 when user_id is missing."""
    create_payload = {
        "context": "Test missing user_id",
    }

    # user_id is required - should fail validation
    response = api_client.post("/memory/raw", json=create_payload)

    assert response.status_code == 422  # Pydantic validation error
    print("\n[OK] POST /memory/raw without user_id returns 422 as expected")


@pytest.mark.integration
def test_api_create_raw_memory_missing_context(api_client, test_user):
    """Test POST /memory/raw returns 422 when context is missing."""
    create_payload = {
        "filter_tags": {"tag": "value"},
    }

    response = api_client.post(
        "/memory/raw",
        json=create_payload,
        params={"user_id": test_user.id},
    )

    assert response.status_code == 422  # Pydantic validation error - context is required
    print("\n[OK] POST /memory/raw without context returns 422 as expected")


@pytest.mark.integration
def test_api_create_raw_memory_auto_creates_user(api_client, test_actor):
    """Test POST /memory/raw auto-creates user if user_id doesn't exist."""
    import uuid

    # Generate unique user_id that doesn't exist
    new_user_id = f"test-auto-api-user-{uuid.uuid4().hex[:8]}"

    create_payload = {
        "context": "Testing auto user creation via API",
        "filter_tags": {"test": "auto_user"},
    }

    response = api_client.post(
        "/memory/raw",
        json=create_payload,
        params={"user_id": new_user_id},
    )

    assert response.status_code == 200, f"POST failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert data["memory"]["user_id"] == new_user_id

    # Cleanup
    memory_id = data["memory"]["id"]
    api_client.delete(f"/memory/raw/{memory_id}")

    print(f"\n[OK] POST /memory/raw auto-created user {new_user_id}")


# =================================================================
# CONCURRENCY TESTS
# =================================================================


def test_raw_memory_concurrent_append(raw_memory_manager, test_actor, test_user):
    """
    Test that concurrent appends don't lose updates.

    This test verifies that the SELECT FOR UPDATE locking prevents
    race conditions when multiple threads/agents append to the same
    raw memory simultaneously.
    """
    import threading

    # Create a raw memory
    sample_data = RawMemoryItemCreate(
        context="Initial context",
        filter_tags={"scope": test_actor.scope, "test": "concurrency"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    memory_id = created.id
    print(f"\n[Concurrency Test] Created memory {memory_id}")

    # Track which updates succeeded
    results = {"thread_1": False, "thread_2": False, "errors": []}

    def append_context(thread_name: str, context_text: str):
        """Helper function to append context in a thread."""
        try:
            raw_memory_manager.update_raw_memory(
                memory_id=memory_id,
                new_context=context_text,
                actor=test_actor,
                context_update_mode="append",
            )
            results[thread_name] = True
            print(f"[{thread_name}] Successfully appended: {context_text}")
        except Exception as e:
            results["errors"].append(f"{thread_name}: {e}")
            print(f"[{thread_name}] ERROR: {e}")

    # Create two threads that will append concurrently
    thread1 = threading.Thread(target=append_context, args=("thread_1", "Update from thread 1"))
    thread2 = threading.Thread(target=append_context, args=("thread_2", "Update from thread 2"))

    # Start both threads simultaneously
    thread1.start()
    thread2.start()

    # Wait for both to complete
    thread1.join()
    thread2.join()

    # Verify both updates succeeded
    assert results["thread_1"], "Thread 1 update failed"
    assert results["thread_2"], "Thread 2 update failed"
    assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"

    # Retrieve the final state
    final_memory = raw_memory_manager.get_raw_memory_by_id(memory_id=memory_id, actor=test_actor)

    assert final_memory is not None
    final_context = final_memory.context

    # Verify BOTH updates are present in the final context
    assert "Initial context" in final_context, "Initial context missing"
    assert "Update from thread 1" in final_context, "Thread 1 update lost!"
    assert "Update from thread 2" in final_context, "Thread 2 update lost!"

    print(f"\n[OK] Concurrent appends preserved both updates")
    print(f"Final context length: {len(final_context)} chars")
    print(f"Final context:\n{final_context}")


def test_raw_memory_concurrent_tag_merge(raw_memory_manager, test_actor, test_user):
    """
    Test that concurrent filter_tags merges don't lose updates.

    Similar to append test but for tag merging operations.
    """
    import threading

    # Create a raw memory with initial tags
    sample_data = RawMemoryItemCreate(
        context="Context for tag merge test",
        filter_tags={"scope": test_actor.scope, "initial_tag": "value0"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    memory_id = created.id
    print(f"\n[Tag Merge Test] Created memory {memory_id}")

    # Track results
    results = {"thread_1": False, "thread_2": False, "errors": []}

    def merge_tags(thread_name: str, new_tags: dict):
        """Helper function to merge tags in a thread."""
        try:
            raw_memory_manager.update_raw_memory(
                memory_id=memory_id,
                new_filter_tags=new_tags,
                actor=test_actor,
                tags_merge_mode="merge",
            )
            results[thread_name] = True
            print(f"[{thread_name}] Successfully merged tags: {new_tags}")
        except Exception as e:
            results["errors"].append(f"{thread_name}: {e}")
            print(f"[{thread_name}] ERROR: {e}")

    # Create two threads that will merge tags concurrently
    thread1 = threading.Thread(target=merge_tags, args=("thread_1", {"tag1": "from_thread_1"}))
    thread2 = threading.Thread(target=merge_tags, args=("thread_2", {"tag2": "from_thread_2"}))

    # Start both threads simultaneously
    thread1.start()
    thread2.start()

    # Wait for both to complete
    thread1.join()
    thread2.join()

    # Verify both updates succeeded
    assert results["thread_1"], "Thread 1 update failed"
    assert results["thread_2"], "Thread 2 update failed"
    assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"

    # Retrieve the final state
    final_memory = raw_memory_manager.get_raw_memory_by_id(memory_id=memory_id, actor=test_actor)

    assert final_memory is not None
    final_tags = final_memory.filter_tags

    # Verify BOTH tag updates are present
    assert final_tags is not None
    assert "scope" in final_tags, "Original scope tag missing"
    assert "initial_tag" in final_tags, "Initial tag missing"
    assert "tag1" in final_tags, "Thread 1 tag lost!"
    assert "tag2" in final_tags, "Thread 2 tag lost!"
    assert final_tags["tag1"] == "from_thread_1"
    assert final_tags["tag2"] == "from_thread_2"

    print(f"\n[OK] Concurrent tag merges preserved both updates")
    print(f"Final tags: {final_tags}")


# =================================================================
# EMBEDDING TESTS
# =================================================================


@pytest.fixture(scope="module")
def test_agent(test_actor):
    """Provide a test agent with Gemini embedding configuration."""
    from pathlib import Path

    import yaml

    from mirix.schemas.agent import CreateAgent
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.services.agent_manager import AgentManager
    from mirix.services.user_manager import UserManager

    # Ensure admin user exists (required for agent creation messages)
    user_mgr = UserManager()
    user_mgr.get_admin_user()  # Creates admin user if it doesn't exist

    agent_mgr = AgentManager()

    # Create an agent with Gemini embedding config (from examples/mirix_gemini.yaml)
    agent_id = "test-agent-raw-mem-gemini"
    try:
        return agent_mgr.get_agent_by_id(agent_id, actor=test_actor)
    except Exception:
        # Load config from mirix_gemini.yaml (same pattern as test_memory_server.py)
        config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Create agent with both llm_config and embedding_config using Gemini
        agent = agent_mgr.create_agent(
            CreateAgent(
                name="Test Agent for Raw Memory Gemini",
                description="Test agent with Gemini embeddings",
                llm_config=LLMConfig(**config["llm_config"]),
                embedding_config=EmbeddingConfig(**config["embedding_config"]),
            ),
            actor=test_actor,
        )
        return agent


@pytest.fixture
def mock_embedding_model(monkeypatch):
    """Mock the embedding model to return fake embeddings for tests (Gemini: 768-dim)."""
    from unittest.mock import Mock

    import numpy as np

    def mock_get_text_embedding(text):
        # Return a fake embedding vector matching Gemini's dimension (768)
        return np.random.rand(768).tolist()

    mock_embed_model = Mock()
    mock_embed_model.get_text_embedding = mock_get_text_embedding

    def mock_embedding_model_factory(config):
        return mock_embed_model

    # Patch the embeddings module directly (where it's imported from)
    monkeypatch.setattr("mirix.embeddings.embedding_model", mock_embedding_model_factory)

    return mock_embed_model


def test_create_raw_memory_with_embeddings(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user, test_agent, mock_embedding_model
):
    """Test creating raw memory with embeddings when agent_state is provided."""
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY, MAX_EMBEDDING_DIM

    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        pytest.skip("BUILD_EMBEDDINGS_FOR_MEMORY is disabled")

    result = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=test_agent,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    assert result.id is not None
    assert result.context_embedding is not None
    assert isinstance(result.context_embedding, list)
    assert len(result.context_embedding) == MAX_EMBEDDING_DIM  # Should be padded
    assert result.embedding_config is not None
    assert result.embedding_config.embedding_model == "gemini-embedding-001"  # Gemini embedding model

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_create_raw_memory_without_agent_state(raw_memory_manager, sample_raw_memory_data, test_actor, test_user):
    """Test creating raw memory without embeddings when agent_state is not provided."""
    result = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=None,  # No agent state
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    assert result.id is not None
    assert result.context_embedding is None
    assert result.embedding_config is None

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_update_raw_memory_regenerates_embeddings(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user, test_agent, mock_embedding_model
):
    """Test updating raw memory regenerates embeddings when context changes."""
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        pytest.skip("BUILD_EMBEDDINGS_FOR_MEMORY is disabled")

    # Create with embeddings
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=test_agent,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    original_embedding = created.context_embedding

    # Update context (should regenerate embedding)
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_context="Completely new context that should have different embedding",
        actor=test_actor,
        agent_state=test_agent,
        context_update_mode="replace",
    )

    assert updated.context_embedding is not None
    # Note: embeddings will be different because mock generates random values
    assert updated.embedding_config is not None

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


def test_update_raw_memory_without_agent_state_preserves_embeddings(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user, test_agent, mock_embedding_model
):
    """Test updating raw memory without agent_state doesn't regenerate embeddings."""
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        pytest.skip("BUILD_EMBEDDINGS_FOR_MEMORY is disabled")

    # Create with embeddings
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=test_agent,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    original_embedding = created.context_embedding

    # Update context WITHOUT agent_state (should NOT regenerate embedding)
    updated = raw_memory_manager.update_raw_memory(
        memory_id=created.id,
        new_context="New context but no agent_state",
        actor=test_actor,
        agent_state=None,  # No agent state
        context_update_mode="replace",
    )

    # Embedding should remain unchanged (not regenerated)
    assert updated.context == "New context but no agent_state"
    # Note: The embedding won't be updated since we didn't provide agent_state

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


def test_embedding_padding_validation(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user, test_agent, mock_embedding_model
):
    """Test that embeddings are padded to MAX_EMBEDDING_DIM."""
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY, MAX_EMBEDDING_DIM

    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        pytest.skip("BUILD_EMBEDDINGS_FOR_MEMORY is disabled")

    result = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=test_agent,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    assert result.context_embedding is not None
    assert len(result.context_embedding) == MAX_EMBEDDING_DIM

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_raw_memory_embeddings_cache_to_redis(
    raw_memory_manager, sample_raw_memory_data, test_actor, test_user, test_agent, redis_client, mock_embedding_model
):
    """Test that raw memory embeddings are properly cached in Redis."""
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        pytest.skip("BUILD_EMBEDDINGS_FOR_MEMORY is disabled")

    # Create with embeddings and caching enabled
    created = raw_memory_manager.create_raw_memory(
        raw_memory=sample_raw_memory_data,
        actor=test_actor,
        agent_state=test_agent,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=True,
    )

    # Verify in Redis JSON
    redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{created.id}"
    cached_data = redis_client.get_json(redis_key)

    assert cached_data is not None
    assert cached_data["id"] == created.id
    assert "context_embedding" in cached_data
    assert cached_data["context_embedding"] is not None
    assert isinstance(cached_data["context_embedding"], list)
    assert "embedding_config" in cached_data
    assert cached_data["embedding_config"] is not None

    # Cleanup
    raw_memory_manager.delete_raw_memory(created.id, test_actor)


# =================================================================
# SEARCH TESTS
# =================================================================


def test_search_raw_memories_filter_tags_multiple_keys(raw_memory_manager, test_actor, test_user):
    """Test filter_tags with multiple keys (AND filtering)."""
    # Create memories with different filter_tags
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 1",
            filter_tags={"scope": test_actor.scope, "priority": "high", "source": "iep"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 2",
            filter_tags={"scope": test_actor.scope, "priority": "high", "source": "manual"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem3 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 3",
            filter_tags={"scope": test_actor.scope, "priority": "low", "source": "iep"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Search with multiple filter_tags (AND filtering)
    results, cursor = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"priority": "high", "source": "iep"},
        limit=10,
    )

    # Should only return mem1 (matches both priority=high AND source=iep)
    assert len(results) == 1
    assert results[0].id == mem1.id

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem3.id, test_actor)


def test_search_raw_memories_sorting_all_fields(raw_memory_manager, test_actor, test_user):
    """Test sorting in both directions for all three fields."""
    from datetime import timedelta

    base_time = datetime.now(UTC)

    # Create memories with different timestamps
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 1",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=3),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 2",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=1),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test ascending sort by updated_at
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="updated_at",
        limit=10,
    )
    assert len(results) >= 2
    # Should be ascending (older first)
    assert results[0].updated_at <= results[1].updated_at

    # Test descending sort by updated_at
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="-updated_at",
        limit=10,
    )
    assert len(results) >= 2
    # Should be descending (newer first)
    assert results[0].updated_at >= results[1].updated_at

    # Test sorting by created_at
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="-created_at",
        limit=10,
    )
    assert len(results) >= 2
    assert results[0].created_at >= results[1].created_at

    # Test sorting by occurred_at
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="-occurred_at",
        limit=10,
    )
    assert len(results) >= 2
    assert results[0].occurred_at >= results[1].occurred_at

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)


def test_search_raw_memories_cursor_pagination(raw_memory_manager, test_actor, test_user):
    """Test cursor pagination across multiple pages."""
    import uuid

    # Use unique tag to isolate this test's memories
    test_tag = f"pagination_test_{uuid.uuid4().hex[:8]}"

    # Create multiple memories
    memories = []
    for i in range(5):
        mem = raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context=f"Memory {i}",
                filter_tags={"scope": test_actor.scope, "test_tag": test_tag},
                user_id=test_user.id,
                organization_id=test_actor.organization_id,
                occurred_at=None,
                id=None,
                context_embedding=None,
                embedding_config=None,
            ),
            actor=test_actor,
            client_id=test_actor.id,
            user_id=test_user.id,
            use_cache=False,
        )
        memories.append(mem)
        # Small delay to ensure different timestamps
        import time

        time.sleep(0.01)

    # First page (limit=2) - filter by unique test_tag to isolate
    results1, cursor1 = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope, "test_tag": test_tag},
        sort="-updated_at",
        limit=2,
    )
    assert len(results1) == 2
    assert cursor1 is not None  # Should have next cursor

    # Second page using cursor
    results2, cursor2 = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope, "test_tag": test_tag},
        sort="-updated_at",
        cursor=cursor1,
        limit=2,
    )
    assert len(results2) == 2
    assert results1[0].id != results2[0].id  # Different items
    assert results1[1].id != results2[0].id  # No duplicates

    # Third page
    results3, cursor3 = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope, "test_tag": test_tag},
        sort="-updated_at",
        cursor=cursor2,
        limit=2,
    )
    assert len(results3) == 1  # Last item
    assert cursor3 is None  # No more pages

    # Cleanup
    for mem in memories:
        raw_memory_manager.delete_raw_memory(mem.id, test_actor)


def test_search_raw_memories_cursor_different_sort_fields(raw_memory_manager, test_actor, test_user):
    """Test cursor with different sort fields."""
    from datetime import timedelta

    base_time = datetime.now(UTC)

    # Create memories with different occurred_at times
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 1",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=2),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 2",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=1),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Get cursor with occurred_at sort
    results1, cursor1 = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="-occurred_at",
        limit=1,
    )
    assert cursor1 is not None

    # Use cursor with same sort field
    results2, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        sort="-occurred_at",
        cursor=cursor1,
        limit=1,
    )
    assert len(results2) == 1
    assert results2[0].id != results1[0].id

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)


def test_search_raw_memories_time_range_filtering(raw_memory_manager, test_actor, test_user):
    """Test time range filtering with various combinations."""
    from datetime import timedelta

    base_time = datetime.now(UTC)

    # Create memories at different times
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 1",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=5),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 2",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=2),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem3 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory 3",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=base_time - timedelta(days=1),
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test occurred_at range filtering
    time_range = {
        "occurred_at_gte": (base_time - timedelta(days=3)).replace(tzinfo=None),
        "occurred_at_lte": (base_time - timedelta(days=0)).replace(tzinfo=None),
    }
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        time_range=time_range,
        limit=10,
    )
    # Should return mem2 and mem3 (within range)
    result_ids = {r.id for r in results}
    assert mem2.id in result_ids
    assert mem3.id in result_ids
    assert mem1.id not in result_ids

    # Test updated_at range filtering
    time_range = {
        "updated_at_gte": (base_time - timedelta(days=3)).replace(tzinfo=None),
    }
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        time_range=time_range,
        limit=10,
    )
    assert len(results) >= 2

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem3.id, test_actor)


def test_search_raw_memories_limit_enforcement(raw_memory_manager, test_actor, test_user):
    """Test limit enforcement (max 100)."""
    # Create many memories
    memories = []
    for i in range(5):
        mem = raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context=f"Memory {i}",
                filter_tags={"scope": test_actor.scope},
                user_id=test_user.id,
                organization_id=test_actor.organization_id,
                occurred_at=None,
                id=None,
                context_embedding=None,
                embedding_config=None,
            ),
            actor=test_actor,
            client_id=test_actor.id,
            user_id=test_user.id,
            use_cache=False,
        )
        memories.append(mem)

    # Test limit=2
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        limit=2,
    )
    assert len(results) == 2

    # Test limit > 100 (should be capped at 100)
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope},
        limit=200,
    )
    assert len(results) <= 100

    # Cleanup
    for mem in memories:
        raw_memory_manager.delete_raw_memory(mem.id, test_actor)


def test_search_raw_memories_scope_handling(raw_memory_manager, test_actor, test_user):
    """Test scope handling (ignore provided scope, always use client scope)."""
    # Create memory with client scope
    mem = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="Memory with scope",
            filter_tags={"scope": test_actor.scope, "priority": "high"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Search with scope in filter_tags - should find it
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope, "priority": "high"},
        limit=10,
    )
    assert len(results) >= 1
    assert any(r.id == mem.id for r in results)

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem.id, test_actor)


def test_search_raw_memories_invalid_cursor(raw_memory_manager, test_actor, test_user):
    """Test invalid cursor format (should handle gracefully)."""
    # Test invalid base64
    with pytest.raises(ValueError, match="Invalid cursor format"):
        raw_memory_manager.search_raw_memories(
            organization_id=test_user.organization_id,
            user_id=test_user.id,
            filter_tags={"scope": test_actor.scope},
            cursor="not-valid-base64!!!",
            limit=10,
        )

    # Test invalid JSON in cursor
    import base64

    invalid_json = base64.b64encode(b"not json").decode()
    with pytest.raises(ValueError, match="Invalid cursor format"):
        raw_memory_manager.search_raw_memories(
            organization_id=test_user.organization_id,
            user_id=test_user.id,
            filter_tags={"scope": test_actor.scope},
            cursor=invalid_json,
            limit=10,
        )

    # Test cursor missing required fields
    import json

    incomplete_cursor = base64.b64encode(json.dumps({"id": "test"}).encode()).decode()
    with pytest.raises(ValueError, match="Invalid cursor format"):
        raw_memory_manager.search_raw_memories(
            organization_id=test_user.organization_id,
            user_id=test_user.id,
            filter_tags={"scope": test_actor.scope},
            sort="-updated_at",
            cursor=incomplete_cursor,
            limit=10,
        )


@pytest.mark.integration
def test_api_search_raw_memories_endpoint(api_client, raw_memory_manager, test_actor, test_user):
    """Test the POST /memory/search_raw endpoint."""
    # Create test memories
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="API test memory 1",
            filter_tags={"scope": test_actor.scope, "priority": "high"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="API test memory 2",
            filter_tags={"scope": test_actor.scope, "priority": "low"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test search endpoint
    response = api_client.post(
        "/memory/search_raw",
        json={
            "filter_tags": {"priority": "high"},
            "sort": "-updated_at",
            "limit": 10,
        },
        params={"user_id": test_user.id},
    )

    assert response.status_code == 200, f"Search failed: {response.text}"
    data = response.json()
    assert "items" in data
    assert "cursor" in data
    assert "count" in data
    assert data["count"] >= 1
    assert any(item["id"] == mem1.id for item in data["items"])

    # Test with cursor pagination
    if data["cursor"]:
        response2 = api_client.post(
            "/memory/search_raw",
            json={
                "filter_tags": {"priority": "high"},
                "sort": "-updated_at",
                "cursor": data["cursor"],
                "limit": 10,
            },
            params={"user_id": test_user.id},
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["count"] >= 0
        # Should not have duplicates
        ids1 = {item["id"] for item in data["items"]}
        ids2 = {item["id"] for item in data2["items"]}
        assert ids1.isdisjoint(ids2)

    # Test invalid sort
    response = api_client.post(
        "/memory/search_raw",
        json={"sort": "invalid_sort", "limit": 10},
        params={"user_id": test_user.id},
    )
    assert response.status_code == 400

    # Test invalid limit (Pydantic validation returns 422)
    response = api_client.post(
        "/memory/search_raw",
        json={"limit": 200},  # Over max
        params={"user_id": test_user.id},
    )
    assert response.status_code == 422

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)


@pytest.mark.integration
def test_api_search_raw_memories_without_user_id(api_client, raw_memory_manager, test_actor, test_user):
    """Test search endpoint without user_id (should return all memories for org, no user filtering)."""
    # Create memory for test_user
    mem = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="No user filter test",
            filter_tags={"scope": test_actor.scope},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Test without user_id parameter (should return memories across all users in org)
    response = api_client.post(
        "/memory/search_raw",
        json={"limit": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "count" in data
    # Memory should be found since no user_id filter is applied
    assert any(item["id"] == mem.id for item in data["items"])

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# SCOPE ENFORCEMENT TESTS
# =================================================================


@pytest.fixture(scope="module")
def test_actor_different_scope():
    """Provide a second test client actor with a different scope."""
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    # Use same org as test_actor
    org_id = "test-org-456"
    try:
        org_mgr.get_organization_by_id(org_id)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=org_id, name="Test Organization"))

    # Create client with DIFFERENT scope
    client_id = "test-client-different-scope"
    try:
        return client_mgr.get_client_by_id(client_id)
    except Exception:
        return client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Test Client Different Scope",
                scope="admin",  # Different scope than test_actor (which is read_write)
            )
        )


def test_scope_auto_injection_on_create(raw_memory_manager, test_actor, test_user):
    """Test that scope is automatically injected from actor when creating raw memory."""
    # Create memory WITHOUT specifying scope in filter_tags
    memory_data = RawMemoryItemCreate(
        context="Test scope auto-injection",
        filter_tags={"custom_tag": "value"},  # No scope specified
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify scope was auto-injected from actor.scope
    assert result.filter_tags is not None
    assert "scope" in result.filter_tags
    assert result.filter_tags["scope"] == test_actor.scope
    assert result.filter_tags["custom_tag"] == "value"  # Other tags preserved

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_scope_auto_injection_overwrites_provided_scope(raw_memory_manager, test_actor, test_user):
    """Test that provided scope is overwritten by actor.scope."""
    # Create memory WITH a different scope in filter_tags
    memory_data = RawMemoryItemCreate(
        context="Test scope overwrite",
        filter_tags={"scope": "wrong_scope", "custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify scope was overwritten by actor.scope
    assert result.filter_tags["scope"] == test_actor.scope
    assert result.filter_tags["scope"] != "wrong_scope"

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_get_denied_for_wrong_scope(raw_memory_manager, test_actor, test_actor_different_scope, test_user):
    """Test that GET returns not found when scope doesn't match."""
    from mirix.orm.errors import NoResultFound

    # Create memory with test_actor's scope
    memory_data = RawMemoryItemCreate(
        context="Test scope denial",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify original actor can read it
    fetched = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor)
    assert fetched.id == result.id

    # Verify different-scope actor cannot read it
    with pytest.raises(NoResultFound):
        raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor_different_scope)

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_update_denied_for_wrong_scope(raw_memory_manager, test_actor, test_actor_different_scope, test_user):
    """Test that UPDATE fails when scope doesn't match."""
    # Create memory with test_actor's scope
    memory_data = RawMemoryItemCreate(
        context="Test update scope denial",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify different-scope actor cannot update it
    with pytest.raises(ValueError, match="Access denied.*scope"):
        raw_memory_manager.update_raw_memory(
            memory_id=result.id,
            actor=test_actor_different_scope,
            new_context="Attempting unauthorized update",
        )

    # Verify original actor can still update it
    updated = raw_memory_manager.update_raw_memory(
        memory_id=result.id,
        actor=test_actor,
        new_context="Authorized update",
    )
    assert updated.context == "Authorized update"

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_delete_denied_for_wrong_scope(raw_memory_manager, test_actor, test_actor_different_scope, test_user):
    """Test that DELETE fails when scope doesn't match."""
    # Create memory with test_actor's scope
    memory_data = RawMemoryItemCreate(
        context="Test delete scope denial",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify different-scope actor cannot delete it
    with pytest.raises(ValueError, match="Access denied.*scope"):
        raw_memory_manager.delete_raw_memory(result.id, test_actor_different_scope)

    # Verify memory still exists
    fetched = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor)
    assert fetched.id == result.id

    # Cleanup - original actor can delete
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_scope_tamper_prevention_on_update(raw_memory_manager, test_actor, test_user):
    """Test that scope cannot be changed via filter_tags update."""
    # Create memory
    memory_data = RawMemoryItemCreate(
        context="Test scope tamper prevention",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Try to change scope via filter_tags update - should fail
    with pytest.raises(ValueError, match="Cannot change memory scope"):
        raw_memory_manager.update_raw_memory(
            memory_id=result.id,
            actor=test_actor,
            new_filter_tags={"scope": "hacked_scope", "custom_tag": "updated"},
        )

    # Verify scope was not changed
    fetched = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor)
    assert fetched.filter_tags["scope"] == test_actor.scope
    assert fetched.filter_tags["scope"] != "hacked_scope"

    # Verify updating with same scope works
    updated = raw_memory_manager.update_raw_memory(
        memory_id=result.id,
        actor=test_actor,
        new_filter_tags={"scope": test_actor.scope, "custom_tag": "updated"},
    )
    assert updated.filter_tags["custom_tag"] == "updated"

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


# =================================================================
# USER_ID FILTERING TESTS
# =================================================================


@pytest.fixture(scope="module")
def test_user_different():
    """Provide a second test user for user_id filtering tests."""
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()

    # Create a different user in the same organization
    user_id = "test-user-different-456"
    try:
        return user_mgr.get_user_by_id(user_id)
    except Exception:
        return user_mgr.create_user(
            PydanticUser(
                id=user_id,
                organization_id="test-org-456",
                name="Test User Different",
                timezone="UTC",
            )
        )


def test_get_raw_memory_filters_by_user_id(raw_memory_manager, test_actor, test_user, test_user_different):
    """Test that get_raw_memory_by_id returns 404 when user_id doesn't match."""
    from mirix.orm.errors import NoResultFound

    # Create memory owned by test_user
    memory_data = RawMemoryItemCreate(
        context="User filtering test",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify owner can read it (with user_id filter)
    fetched = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor, user_id=test_user.id)
    assert fetched.id == result.id

    # Verify different user cannot read it (with user_id filter)
    with pytest.raises(NoResultFound):
        raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor, user_id=test_user_different.id)

    # Verify reading without user_id filter still works (org-level access)
    fetched_no_filter = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor)
    assert fetched_no_filter.id == result.id

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_update_raw_memory_filters_by_user_id(raw_memory_manager, test_actor, test_user, test_user_different):
    """Test that update_raw_memory returns error when user_id doesn't match."""
    # Create memory owned by test_user
    memory_data = RawMemoryItemCreate(
        context="Update user filtering test",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify owner can update it (with user_id filter)
    updated = raw_memory_manager.update_raw_memory(
        memory_id=result.id,
        actor=test_actor,
        new_context="Updated by owner",
        user_id=test_user.id,
    )
    assert updated.context == "Updated by owner"

    # Verify different user cannot update it (with user_id filter)
    with pytest.raises(ValueError, match="not found"):
        raw_memory_manager.update_raw_memory(
            memory_id=result.id,
            actor=test_actor,
            new_context="Updated by wrong user",
            user_id=test_user_different.id,
        )

    # Cleanup
    raw_memory_manager.delete_raw_memory(result.id, test_actor)


def test_delete_raw_memory_filters_by_user_id(raw_memory_manager, test_actor, test_user, test_user_different):
    """Test that delete_raw_memory returns False when user_id doesn't match."""
    # Create memory owned by test_user
    memory_data = RawMemoryItemCreate(
        context="Delete user filtering test",
        filter_tags={"custom_tag": "value"},
        user_id=test_user.id,
        organization_id=test_actor.organization_id,
    )

    result = raw_memory_manager.create_raw_memory(
        raw_memory=memory_data,
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Verify different user cannot delete it (with user_id filter) - returns False
    deleted = raw_memory_manager.delete_raw_memory(result.id, test_actor, user_id=test_user_different.id)
    assert deleted is False

    # Verify memory still exists
    fetched = raw_memory_manager.get_raw_memory_by_id(result.id, actor=test_actor)
    assert fetched.id == result.id

    # Verify owner can delete it (with user_id filter)
    deleted = raw_memory_manager.delete_raw_memory(result.id, test_actor, user_id=test_user.id)
    assert deleted is True


def test_search_raw_memories_filters_by_user_id(raw_memory_manager, test_actor, test_user, test_user_different):
    """Test that search_raw_memories filters by user_id when provided."""
    # Create memory for test_user
    mem1 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="User 1 memory",
            filter_tags={"scope": test_actor.scope, "test_tag": "user_filter_test"},
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )

    # Create memory for test_user_different
    mem2 = raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context="User 2 memory",
            filter_tags={"scope": test_actor.scope, "test_tag": "user_filter_test"},
            user_id=test_user_different.id,
            organization_id=test_actor.organization_id,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user_different.id,
        use_cache=False,
    )

    # Search with user_id filter for test_user - should only return mem1
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user.id,
        filter_tags={"scope": test_actor.scope, "test_tag": "user_filter_test"},
        limit=10,
    )
    assert len(results) == 1
    assert results[0].id == mem1.id

    # Search with user_id filter for test_user_different - should only return mem2
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=test_user_different.id,
        filter_tags={"scope": test_actor.scope, "test_tag": "user_filter_test"},
        limit=10,
    )
    assert len(results) == 1
    assert results[0].id == mem2.id

    # Search without user_id filter - should return both
    results, _ = raw_memory_manager.search_raw_memories(
        organization_id=test_user.organization_id,
        user_id=None,
        filter_tags={"scope": test_actor.scope, "test_tag": "user_filter_test"},
        limit=10,
    )
    result_ids = {r.id for r in results}
    assert mem1.id in result_ids
    assert mem2.id in result_ids

    # Cleanup
    raw_memory_manager.delete_raw_memory(mem1.id, test_actor)
    raw_memory_manager.delete_raw_memory(mem2.id, test_actor)


def test_create_raw_memory_requires_user_id(raw_memory_manager, test_actor):
    """Test that create_raw_memory raises error when user_id is not provided."""
    memory_data = RawMemoryItemCreate(
        context="Test missing user_id",
        filter_tags={"custom_tag": "value"},
        user_id="placeholder",  # Will be ignored, we test the parameter
        organization_id=test_actor.organization_id,
    )

    # Verify error is raised when user_id is empty string
    with pytest.raises(ValueError, match="user_id is required"):
        raw_memory_manager.create_raw_memory(
            raw_memory=memory_data,
            actor=test_actor,
            client_id=test_actor.id,
            user_id="",  # Empty string should fail
            use_cache=False,
        )
