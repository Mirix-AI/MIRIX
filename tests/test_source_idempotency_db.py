"""DB-level integration tests for Layer 1 idempotency (unique constraints on memory_sources).

Requires PostgreSQL + Redis via docker test infrastructure:
    ./scripts/run_tests_with_docker.sh --podman -s -v -k test_source_idempotency_db
"""

import uuid

import pytest
import pytest_asyncio

from mirix.services.memory_source_manager import MemorySourceManager
from mirix.services.source_message_manager import compute_batch_hash

pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.requires_pg]

TEST_ORG_ID = "dedup-test-org"
TEST_CLIENT_ID = "dedup-test-client"
TEST_USER_ID = "dedup-test-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Dedup Test Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Dedup Test User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


@pytest_asyncio.fixture
def manager():
    return MemorySourceManager()


async def _get_actor():
    """Fetch the test client as a PydanticClient for use as actor."""
    from mirix.services.client_manager import ClientManager

    client_mgr = ClientManager()
    return await client_mgr.get_client_by_id(TEST_CLIENT_ID)


async def test_duplicate_external_id_silently_skipped(manager):
    """Same (client_id, user_id, external_id) submitted twice: second INSERT is silently skipped."""
    ext_id = f"ext-{uuid.uuid4().hex[:12]}"
    actor = await _get_actor()
    common = dict(
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        external_id=ext_id,
        source_type="conversation",
        use_cache=False,
    )

    # First insert — should succeed
    src1 = await manager.create(memory_source_id=_unique(), **common)
    assert src1 is not None
    assert src1.external_id == ext_id

    # Second insert with different memory_source_id but same external_id
    src2_id = _unique()
    src2 = await manager.create(memory_source_id=src2_id, **common)

    # The second INSERT is silently skipped by ON CONFLICT DO NOTHING.
    # get_by_id for the new ID returns None because it was never actually inserted.
    direct_lookup = await manager.get_by_id(src2_id, use_cache=False)
    assert direct_lookup is None, (
        "Second insert with duplicate external_id should have been skipped; " f"but a record was found at {src2_id}"
    )

    # Original record should still exist
    original = await manager.get_by_id(src1.id, use_cache=False)
    assert original is not None
    assert original.external_id == ext_id


async def test_same_content_different_thread_produces_different_batch_hash(manager):
    """Same messages but different external_thread_id → different batch_hash → both records inserted."""
    unique_content = f"hello world {_unique('msg')}"
    messages = [{"role": "user", "content": {"text": unique_content}}]

    thread_a = _unique("thread")
    thread_b = _unique("thread")
    hash_a = compute_batch_hash(thread_a, None, messages)
    hash_b = compute_batch_hash(thread_b, None, messages)

    # Hashes must differ
    assert hash_a != hash_b, "batch_hash should differ when thread IDs differ"

    actor = await _get_actor()
    common = dict(
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_type="conversation",
        use_cache=False,
    )

    src_a = await manager.create(
        memory_source_id=_unique(),
        batch_hash=hash_a,
        external_thread_id=thread_a,
        **common,
    )
    src_b = await manager.create(
        memory_source_id=_unique(),
        batch_hash=hash_b,
        external_thread_id=thread_b,
        **common,
    )

    # Both should be inserted (different batch_hash values)
    assert src_a is not None
    assert src_b is not None
    assert src_a.id != src_b.id

    # Verify they are distinct in the DB
    lookup_a = await manager.get_by_id(src_a.id, use_cache=False)
    lookup_b = await manager.get_by_id(src_b.id, use_cache=False)
    assert lookup_a is not None
    assert lookup_b is not None
    assert lookup_a.batch_hash != lookup_b.batch_hash


async def test_duplicate_batch_hash_silently_skipped(manager):
    """Same (client_id, user_id, batch_hash) submitted twice: second INSERT is silently skipped."""
    unique_content = f"duplicate batch content {_unique('msg')}"
    messages = [{"role": "user", "content": {"text": unique_content}}]
    thread_id = _unique("thread")
    batch_hash = compute_batch_hash(thread_id, "2026-01-01T00:00:00Z", messages)

    actor = await _get_actor()
    common = dict(
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        batch_hash=batch_hash,
        source_type="conversation",
        use_cache=False,
    )

    src1 = await manager.create(memory_source_id=_unique(), **common)
    assert src1 is not None

    src2_id = _unique()
    await manager.create(memory_source_id=src2_id, **common)

    direct_lookup = await manager.get_by_id(src2_id, use_cache=False)
    assert direct_lookup is None, "Second insert with duplicate batch_hash should have been skipped"
