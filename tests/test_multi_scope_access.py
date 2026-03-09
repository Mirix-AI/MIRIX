"""
Test multi-scoped client access control.

This test suite verifies the write_scope/read_scopes model for memory access:
- Read-only clients (write_scope=None) cannot create memories
- Clients can read from multiple scopes via read_scopes
- Shared memory pools work correctly
- Private + shared access patterns work correctly
- Empty read_scopes means no read access

These tests cover the proposal:
- write_scope: The single scope this client can write to (null = read-only)
- read_scopes: List of scopes this client can read memories from
"""

import uuid
from datetime import datetime

import pytest
import pytest_asyncio

from mirix.orm.errors import NoResultFound
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.raw_memory import RawMemoryItemCreate
from mirix.schemas.user import User as PydanticUser
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.raw_memory_manager import RawMemoryManager
from mirix.services.user_manager import UserManager

# Use one event loop per module so async fixtures and tests share it.
pytestmark = [pytest.mark.asyncio(loop_scope="module")]

# =============================================================================
# Test Fixtures
# =============================================================================


def generate_test_id(prefix: str) -> str:
    """Generate a unique test ID."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def test_org():
    """Create a test organization for multi-scope tests."""
    org_mgr = OrganizationManager()
    org_id = generate_test_id("multi-scope-org")

    try:
        return await org_mgr.get_organization_by_id(org_id)
    except Exception:
        return await org_mgr.create_organization(
            PydanticOrganization(id=org_id, name="Multi-Scope Test Org")
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def test_user(test_org):
    """Create a test user for multi-scope tests."""
    user_mgr = UserManager()
    user_id = generate_test_id("multi-scope-user")

    try:
        return await user_mgr.get_user_by_id(user_id)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="Multi-Scope Test User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )


@pytest.fixture(scope="module")
def raw_memory_manager():
    """Provide a RawMemoryManager instance."""
    return RawMemoryManager()


@pytest.fixture(scope="module")
def client_manager():
    """Provide a ClientManager instance."""
    return ClientManager()


# =============================================================================
# Client Fixtures for Different Access Patterns
# =============================================================================


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def read_only_client(test_org, client_manager):
    """
    Client with NO write access, can only read from 'shared' scope.

    Example use case: Read-Only Sales Auto-BDR Client
    """
    client_id = generate_test_id("read-only-client")
    try:
        return await client_manager.get_client_by_id(client_id)
    except Exception:
        return await client_manager.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Read-Only Client",
                write_scope=None,  # Cannot write
                read_scopes=["shared"],  # Can only read from shared
            )
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared_writer_client(test_org, client_manager):
    """
    Client that writes to 'shared' scope, doesn't need to read.

    Example use case: Primary Ingestion Pipeline
    """
    client_id = generate_test_id("shared-writer-client")
    try:
        return await client_manager.get_client_by_id(client_id)
    except Exception:
        return await client_manager.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Shared Writer Client",
                write_scope="shared",  # Writes to shared pool
                read_scopes=["shared"],  # Can read what it writes
            )
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def private_client(test_org, client_manager):
    """
    Client with private scope - can read shared AND write to private.

    Example use case: IEP Agent
    """
    client_id = generate_test_id("private-client")
    try:
        return await client_manager.get_client_by_id(client_id)
    except Exception:
        return await client_manager.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Private Client",
                write_scope="private",  # Writes to private scope
                read_scopes=["shared", "private"],  # Can read both shared and private
            )
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def multi_read_client(test_org, client_manager):
    """
    Client that can read from multiple scopes but writes to its own.
    """
    client_id = generate_test_id("multi-read-client")
    try:
        return await client_manager.get_client_by_id(client_id)
    except Exception:
        return await client_manager.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Multi-Read Client",
                write_scope="multi-read-scope",
                read_scopes=["shared", "private", "multi-read-scope"],
            )
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def no_access_client(test_org, client_manager):
    """
    Client with NO read or write access.
    """
    client_id = generate_test_id("no-access-client")
    try:
        return await client_manager.get_client_by_id(client_id)
    except Exception:
        return await client_manager.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="No Access Client",
                write_scope=None,  # Cannot write
                read_scopes=[],  # Cannot read anything
            )
        )


# =============================================================================
# Test: Read-Only Client (write_scope=None)
# =============================================================================


class TestReadOnlyClient:
    """Tests for clients with write_scope=None."""

    async def test_read_only_client_cannot_create_memory(
        self, raw_memory_manager, read_only_client, test_user
    ):
        """Test that a read-only client (write_scope=None) cannot create memories."""
        memory_data = RawMemoryItemCreate(
            context="Attempting to create from read-only client",
            filter_tags={"test": "read_only_create"},
            user_id=test_user.id,
            organization_id=read_only_client.organization_id,
        )

        with pytest.raises(ValueError, match="no write_scope"):
            await raw_memory_manager.create_raw_memory(
                raw_memory=memory_data,
                actor=read_only_client,
                user_id=test_user.id,
                use_cache=False,
            )

    async def test_read_only_client_can_read_from_read_scopes(
        self, raw_memory_manager, read_only_client, shared_writer_client, test_user
    ):
        """Test that a read-only client can read memories from its read_scopes."""
        # First, create a memory using the shared_writer_client
        memory_data = RawMemoryItemCreate(
            context="Shared memory for read-only test",
            filter_tags={"test": "read_only_read"},
            user_id=test_user.id,
            organization_id=shared_writer_client.organization_id,
        )

        created = await raw_memory_manager.create_raw_memory(
            raw_memory=memory_data,
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Verify the memory was created with 'shared' scope
            assert created.filter_tags["scope"] == "shared"

            # Read-only client should be able to read it (has 'shared' in read_scopes)
            fetched = await raw_memory_manager.get_raw_memory_by_id(
                created.id, actor=read_only_client
            )
            assert fetched.id == created.id
            assert fetched.context == memory_data.context
        finally:
            # Cleanup using the writer client
            await raw_memory_manager.delete_raw_memory(created.id, shared_writer_client)


# =============================================================================
# Test: Multi-Scope Read Access
# =============================================================================


class TestMultiScopeRead:
    """Tests for clients reading from multiple scopes."""

    async def test_client_can_read_from_multiple_scopes(
        self, raw_memory_manager, shared_writer_client, private_client, multi_read_client, test_user
    ):
        """Test that a client with multiple read_scopes can read from all of them."""
        # Create memory in 'shared' scope
        shared_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared scope memory",
                filter_tags={"test": "multi_read"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        # Create memory in 'private' scope
        private_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Private scope memory",
                filter_tags={"test": "multi_read"},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Verify scopes
            assert shared_memory.filter_tags["scope"] == "shared"
            assert private_memory.filter_tags["scope"] == "private"

            # multi_read_client has read_scopes=["shared", "private", "multi-read-scope"]
            # It should be able to read both memories
            fetched_shared = await raw_memory_manager.get_raw_memory_by_id(
                shared_memory.id, actor=multi_read_client
            )
            assert fetched_shared.id == shared_memory.id

            fetched_private = await raw_memory_manager.get_raw_memory_by_id(
                private_memory.id, actor=multi_read_client
            )
            assert fetched_private.id == private_memory.id
        finally:
            # Cleanup
            await raw_memory_manager.delete_raw_memory(shared_memory.id, shared_writer_client)
            await raw_memory_manager.delete_raw_memory(private_memory.id, private_client)

    async def test_search_returns_memories_from_multiple_scopes(
        self, raw_memory_manager, shared_writer_client, private_client, multi_read_client, test_user
    ):
        """Test that search returns memories from all read_scopes."""
        test_tag = generate_test_id("search-multi")

        # Create memory in 'shared' scope
        shared_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared for search test",
                filter_tags={"test_tag": test_tag},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        # Create memory in 'private' scope
        private_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Private for search test",
                filter_tags={"test_tag": test_tag},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Search with multi_read_client's read_scopes
            results, _ = await raw_memory_manager.search_raw_memories(
                organization_id=test_user.organization_id,
                user_id=test_user.id,
                filter_tags={"test_tag": test_tag},
                scopes=multi_read_client.read_scopes,
                limit=10,
            )

            # Should find both memories
            result_ids = [r.id for r in results]
            assert shared_memory.id in result_ids, "Should find shared memory"
            assert private_memory.id in result_ids, "Should find private memory"
        finally:
            # Cleanup
            await raw_memory_manager.delete_raw_memory(shared_memory.id, shared_writer_client)
            await raw_memory_manager.delete_raw_memory(private_memory.id, private_client)


# =============================================================================
# Test: Shared Memory Pool
# =============================================================================


class TestSharedMemoryPool:
    """Tests for shared memory pool access patterns."""

    async def test_writer_creates_reader_reads(
        self, raw_memory_manager, shared_writer_client, read_only_client, test_user
    ):
        """Test that one client writes to shared pool and another can read."""
        # Writer creates memory
        memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared pool memory from writer",
                filter_tags={"test": "shared_pool"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Verify scope
            assert memory.filter_tags["scope"] == "shared"

            # Reader can read it
            fetched = await raw_memory_manager.get_raw_memory_by_id(
                memory.id, actor=read_only_client
            )
            assert fetched.id == memory.id
        finally:
            await raw_memory_manager.delete_raw_memory(memory.id, shared_writer_client)

    async def test_reader_cannot_modify_shared_memory(
        self, raw_memory_manager, shared_writer_client, read_only_client, test_user
    ):
        """Test that a read-only client cannot modify shared memories."""
        # Writer creates memory
        memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared memory - should not be modifiable by reader",
                filter_tags={"test": "shared_no_modify"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Read-only client cannot update (write_scope is None)
            with pytest.raises(ValueError, match="write_scope"):
                await raw_memory_manager.update_raw_memory(
                    memory_id=memory.id,
                    actor=read_only_client,
                    new_context="Attempting unauthorized update",
                )

            # Read-only client cannot delete (write_scope is None)
            with pytest.raises(ValueError, match="write_scope"):
                await raw_memory_manager.delete_raw_memory(memory.id, read_only_client)
        finally:
            await raw_memory_manager.delete_raw_memory(memory.id, shared_writer_client)


# =============================================================================
# Test: Private + Shared Access Pattern
# =============================================================================


class TestPrivateAndSharedAccess:
    """Tests for clients with both private and shared access."""

    async def test_private_client_reads_shared_and_private(
        self, raw_memory_manager, shared_writer_client, private_client, test_user
    ):
        """Test that private client can read from both shared and private scopes."""
        # Create shared memory
        shared_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared memory for private client test",
                filter_tags={"test": "private_shared_read"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        # Create private memory
        private_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Private memory for private client test",
                filter_tags={"test": "private_shared_read"},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Private client can read shared memory
            fetched_shared = await raw_memory_manager.get_raw_memory_by_id(
                shared_memory.id, actor=private_client
            )
            assert fetched_shared.id == shared_memory.id

            # Private client can read its own private memory
            fetched_private = await raw_memory_manager.get_raw_memory_by_id(
                private_memory.id, actor=private_client
            )
            assert fetched_private.id == private_memory.id
        finally:
            await raw_memory_manager.delete_raw_memory(shared_memory.id, shared_writer_client)
            await raw_memory_manager.delete_raw_memory(private_memory.id, private_client)

    async def test_private_client_cannot_write_to_shared(
        self, raw_memory_manager, shared_writer_client, private_client, test_user
    ):
        """Test that private client cannot modify memories in shared scope."""
        # Create shared memory
        shared_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Shared memory - private client should not modify",
                filter_tags={"test": "private_no_shared_write"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Private client has write_scope="private", not "shared"
            # So it cannot update shared memories
            with pytest.raises(ValueError, match="Access denied.*scope"):
                await raw_memory_manager.update_raw_memory(
                    memory_id=shared_memory.id,
                    actor=private_client,
                    new_context="Attempting to modify shared from private client",
                )

            # Cannot delete either
            with pytest.raises(ValueError, match="Access denied.*scope"):
                await raw_memory_manager.delete_raw_memory(shared_memory.id, private_client)
        finally:
            await raw_memory_manager.delete_raw_memory(shared_memory.id, shared_writer_client)

    async def test_private_client_can_modify_own_scope(
        self, raw_memory_manager, private_client, test_user
    ):
        """Test that private client can create, update, and delete in its own scope."""
        # Create in private scope
        memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Private memory - should be modifiable",
                filter_tags={"test": "private_modify_own"},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Verify scope
            assert memory.filter_tags["scope"] == "private"

            # Can update
            updated = await raw_memory_manager.update_raw_memory(
                memory_id=memory.id,
                actor=private_client,
                new_context="Updated private memory",
            )
            assert "Updated" in updated.context

            # Can delete
            deleted = await raw_memory_manager.delete_raw_memory(memory.id, private_client)
            assert deleted is True
        except Exception:
            # Cleanup if test fails
            try:
                await raw_memory_manager.delete_raw_memory(memory.id, private_client)
            except Exception:
                pass
            raise


# =============================================================================
# Test: Empty read_scopes
# =============================================================================


class TestEmptyReadScopes:
    """Tests for clients with empty read_scopes."""

    async def test_no_access_client_cannot_read_any_memory(
        self, raw_memory_manager, shared_writer_client, no_access_client, test_user
    ):
        """Test that a client with empty read_scopes cannot read any memories."""
        # Create a memory
        memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Memory that no-access client cannot read",
                filter_tags={"test": "no_access_read"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # no_access_client has read_scopes=[], so it cannot read anything
            with pytest.raises(NoResultFound):
                await raw_memory_manager.get_raw_memory_by_id(
                    memory.id, actor=no_access_client
                )
        finally:
            await raw_memory_manager.delete_raw_memory(memory.id, shared_writer_client)

    async def test_no_access_client_cannot_create_memory(
        self, raw_memory_manager, no_access_client, test_user
    ):
        """Test that a client with no write_scope cannot create memories."""
        memory_data = RawMemoryItemCreate(
            context="Attempting to create from no-access client",
            filter_tags={"test": "no_access_create"},
            user_id=test_user.id,
            organization_id=no_access_client.organization_id,
        )

        with pytest.raises(ValueError, match="no write_scope"):
            await raw_memory_manager.create_raw_memory(
                raw_memory=memory_data,
                actor=no_access_client,
                user_id=test_user.id,
                use_cache=False,
            )

    async def test_search_with_empty_read_scopes_returns_nothing(
        self, raw_memory_manager, shared_writer_client, no_access_client, test_user
    ):
        """Test that search with empty read_scopes returns no results."""
        test_tag = generate_test_id("empty-scope-search")

        # Create a memory
        memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Memory for empty scope search test",
                filter_tags={"test_tag": test_tag},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Search with empty read_scopes
            results, _ = await raw_memory_manager.search_raw_memories(
                organization_id=test_user.organization_id,
                user_id=test_user.id,
                filter_tags={"test_tag": test_tag},
                scopes=no_access_client.read_scopes,  # Empty list
                limit=10,
            )

            # Should return no results
            assert len(results) == 0, "Empty read_scopes should return no results"
        finally:
            await raw_memory_manager.delete_raw_memory(memory.id, shared_writer_client)


# =============================================================================
# Test: Scope Isolation
# =============================================================================


class TestScopeIsolation:
    """Tests for scope isolation between clients."""

    async def test_client_cannot_read_outside_read_scopes(
        self, raw_memory_manager, private_client, shared_writer_client, test_user
    ):
        """Test that a client cannot read memories outside its read_scopes."""
        # shared_writer_client has read_scopes=["shared"]
        # private_client has read_scopes=["shared", "private"]

        # Create memory in 'private' scope
        private_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Private memory - shared_writer should not read",
                filter_tags={"test": "scope_isolation"},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # shared_writer_client only has read_scopes=["shared"]
            # It should NOT be able to read 'private' scope memory
            with pytest.raises(NoResultFound):
                await raw_memory_manager.get_raw_memory_by_id(
                    private_memory.id, actor=shared_writer_client
                )
        finally:
            await raw_memory_manager.delete_raw_memory(private_memory.id, private_client)

    async def test_write_scope_determines_memory_scope(
        self, raw_memory_manager, private_client, shared_writer_client, test_user
    ):
        """Test that memories are tagged with the actor's write_scope."""
        # Create memory with private_client (write_scope="private")
        private_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Memory from private client",
                filter_tags={"test": "write_scope_tag"},
                user_id=test_user.id,
                organization_id=private_client.organization_id,
            ),
            actor=private_client,
            user_id=test_user.id,
            use_cache=False,
        )

        # Create memory with shared_writer_client (write_scope="shared")
        shared_memory = await raw_memory_manager.create_raw_memory(
            raw_memory=RawMemoryItemCreate(
                context="Memory from shared writer client",
                filter_tags={"test": "write_scope_tag"},
                user_id=test_user.id,
                organization_id=shared_writer_client.organization_id,
            ),
            actor=shared_writer_client,
            user_id=test_user.id,
            use_cache=False,
        )

        try:
            # Verify scopes match write_scope
            assert private_memory.filter_tags["scope"] == "private"
            assert private_memory.filter_tags["scope"] == private_client.write_scope

            assert shared_memory.filter_tags["scope"] == "shared"
            assert shared_memory.filter_tags["scope"] == shared_writer_client.write_scope
        finally:
            await raw_memory_manager.delete_raw_memory(private_memory.id, private_client)
            await raw_memory_manager.delete_raw_memory(shared_memory.id, shared_writer_client)
