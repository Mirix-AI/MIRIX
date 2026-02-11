"""
Unit tests for UserManager after client_id removal.

Tests verify that users are organization-scoped, not client-scoped:
1. Users created by different clients in the same org are shared
2. Users in different orgs are isolated
3. list_users() filters by organization_id
4. Deleting a client does NOT cascade-delete users
5. get_or_create_org_default_user() works without client_id

Run tests:
    pytest tests/test_user_manager.py -v
"""

import uuid
from datetime import datetime
from datetime import timezone as dt_timezone

import pytest

from mirix.log import get_logger
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.user import User as PydanticUser
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.user_manager import UserManager

logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_test_id(prefix: str) -> str:
    """Generate a test ID matching Mirix ID pattern (prefix-[8 hex chars])."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def organization_manager():
    """Create organization manager instance."""
    return OrganizationManager()


@pytest.fixture
def user_manager():
    """Create user manager instance."""
    return UserManager()


@pytest.fixture
def client_manager():
    """Create client manager instance."""
    return ClientManager()


@pytest.fixture
def test_org1(organization_manager):
    """Create test organization 1."""
    org = PydanticOrganization(id=generate_test_id("org"), name="Test Organization 1")
    created_org = organization_manager.create_organization(org)
    yield created_org
    # Cleanup
    try:
        organization_manager.delete_organization_by_id(created_org.id)
    except Exception:
        pass


@pytest.fixture
def test_org2(organization_manager):
    """Create test organization 2."""
    org = PydanticOrganization(id=generate_test_id("org"), name="Test Organization 2")
    created_org = organization_manager.create_organization(org)
    yield created_org
    # Cleanup
    try:
        organization_manager.delete_organization_by_id(created_org.id)
    except Exception:
        pass


@pytest.fixture
def client_a(test_org1, client_manager):
    """Create Client A in org1."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client A",
        organization_id=test_org1.id,
        status="active",
        scope="read_write",
    )
    created_client = client_manager.create_client(client)
    yield created_client
    # Cleanup
    try:
        client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


@pytest.fixture
def client_b(test_org1, client_manager):
    """Create Client B in org1 (same org as Client A)."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client B",
        organization_id=test_org1.id,
        status="active",
        scope="read_write",
    )
    created_client = client_manager.create_client(client)
    yield created_client
    # Cleanup
    try:
        client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


@pytest.fixture
def client_c(test_org2, client_manager):
    """Create Client C in org2 (different org)."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client C",
        organization_id=test_org2.id,
        status="active",
        scope="read_write",
    )
    created_client = client_manager.create_client(client)
    yield created_client
    # Cleanup
    try:
        client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


# ============================================================================
# TEST CLASS: User Schema Verification
# ============================================================================


class TestUserSchemaWithoutClientId:
    """Tests verifying User schema has no client_id field."""

    def test_user_schema_has_no_client_id_field(self):
        """Verify User Pydantic schema does not have client_id field."""
        user = PydanticUser(
            id="user-test",
            name="Test User",
            organization_id="org-1",
            timezone="UTC",
        )
        # Check model_fields on the class, not instance
        assert "client_id" not in PydanticUser.model_fields, "client_id should not be in User schema"

    def test_user_creation_without_client_id(self):
        """Verify users can be created without client_id."""
        user = PydanticUser(
            id=generate_test_id("user"),
            name="Test User",
            organization_id="org-test",
            timezone="America/New_York",
            status="active",
        )
        assert user.id is not None
        assert user.organization_id == "org-test"
        assert user.name == "Test User"


# ============================================================================
# TEST CLASS: Organization-Scoped User Creation
# ============================================================================


class TestOrganizationScopedUserCreation:
    """Tests verifying users are organization-scoped, not client-scoped."""

    def test_create_user_is_organization_scoped(self, user_manager, test_org1):
        """
        Verify that create_user creates users at the organization level.
        
        Users should be associated with an organization, not a client.
        """
        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Org Scoped User",
            organization_id=test_org1.id,
            timezone="UTC",
        )
        
        created_user = user_manager.create_user(user)
        
        assert created_user.id == user_id
        assert created_user.organization_id == test_org1.id
        
        # Cleanup
        try:
            user_manager.delete_user_by_id(user_id)
        except Exception:
            pass

    def test_same_user_id_retrieved_by_different_contexts(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that a user created in an org can be retrieved regardless of client context.
        
        - Create user in org1
        - User should be retrievable (users are org-scoped, not client-scoped)
        """
        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Shared User",
            organization_id=test_org1.id,
            timezone="UTC",
        )
        
        # Create user
        created_user = user_manager.create_user(user)
        assert created_user.id == user_id
        
        # Retrieve user - should work since users are org-scoped
        retrieved_user = user_manager.get_user_by_id(user_id)
        assert retrieved_user.id == user_id
        assert retrieved_user.organization_id == test_org1.id
        
        # Cleanup
        try:
            user_manager.delete_user_by_id(user_id)
        except Exception:
            pass


# ============================================================================
# TEST CLASS: Multiple Clients Same Org Share Users
# ============================================================================


class TestMultipleClientsSameOrgShareUsers:
    """Tests verifying that clients in the same org share users."""

    def test_multiple_clients_same_org_see_same_users(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that two clients in the same organization see the same users.
        
        - Create 3 users in org1
        - list_users(organization_id=org1) should return all 3 users
        - Total user count in org1 should be 3 (not duplicated per client)
        """
        created_user_ids = []
        
        try:
            # Create 3 users in org1
            for i in range(3):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"User {i+1}",
                    organization_id=test_org1.id,
                    timezone="UTC",
                )
                user_manager.create_user(user)
                created_user_ids.append(user_id)
            
            # List users for org1
            users = user_manager.list_users(organization_id=test_org1.id)
            
            # Filter to only our test users (there may be other users in the org)
            test_users = [u for u in users if u.id in created_user_ids]
            
            assert len(test_users) == 3, f"Expected 3 users, got {len(test_users)}"
            
            # Verify all created users are in the list
            retrieved_ids = {u.id for u in test_users}
            for user_id in created_user_ids:
                assert user_id in retrieved_ids, f"User {user_id} not found in list"
                
        finally:
            # Cleanup
            for user_id in created_user_ids:
                try:
                    user_manager.delete_user_by_id(user_id)
                except Exception:
                    pass

    def test_user_count_not_multiplied_by_clients(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that having multiple clients doesn't multiply user count.
        
        Before the fix, users were client-scoped, so each client would have
        its own copy. Now users are org-scoped, so count should be consistent.
        """
        user_id = generate_test_id("user")
        
        try:
            # Create one user in org1
            user = PydanticUser(
                id=user_id,
                name="Single User",
                organization_id=test_org1.id,
                timezone="UTC",
            )
            user_manager.create_user(user)
            
            # List users for org1
            users = user_manager.list_users(organization_id=test_org1.id)
            
            # Count how many times our user appears
            user_occurrences = [u for u in users if u.id == user_id]
            
            assert len(user_occurrences) == 1, f"User should appear exactly once, got {len(user_occurrences)}"
            
        finally:
            try:
                user_manager.delete_user_by_id(user_id)
            except Exception:
                pass


# ============================================================================
# TEST CLASS: Users Isolated Across Organizations
# ============================================================================


class TestUsersIsolatedAcrossOrganizations:
    """Tests verifying that users in different orgs are isolated."""

    def test_list_users_filters_by_organization(self, user_manager, test_org1, test_org2):
        """
        Verify list_users filters by organization_id.
        
        - Create users in org1 and org2
        - list_users(org1) should only return org1 users
        - list_users(org2) should only return org2 users
        """
        org1_user_ids = []
        org2_user_ids = []
        
        try:
            # Create 2 users in org1
            for i in range(2):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"Org1 User {i+1}",
                    organization_id=test_org1.id,
                    timezone="UTC",
                )
                user_manager.create_user(user)
                org1_user_ids.append(user_id)
            
            # Create 2 users in org2
            for i in range(2):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"Org2 User {i+1}",
                    organization_id=test_org2.id,
                    timezone="UTC",
                )
                user_manager.create_user(user)
                org2_user_ids.append(user_id)
            
            # List users for org1
            org1_users = user_manager.list_users(organization_id=test_org1.id)
            org1_retrieved_ids = {u.id for u in org1_users}
            
            # List users for org2
            org2_users = user_manager.list_users(organization_id=test_org2.id)
            org2_retrieved_ids = {u.id for u in org2_users}
            
            # Verify org1 users are in org1 list
            for user_id in org1_user_ids:
                assert user_id in org1_retrieved_ids, f"Org1 user {user_id} not in org1 list"
            
            # Verify org2 users are in org2 list
            for user_id in org2_user_ids:
                assert user_id in org2_retrieved_ids, f"Org2 user {user_id} not in org2 list"
            
            # Verify org1 users are NOT in org2 list
            for user_id in org1_user_ids:
                assert user_id not in org2_retrieved_ids, f"Org1 user {user_id} should not be in org2 list"
            
            # Verify org2 users are NOT in org1 list
            for user_id in org2_user_ids:
                assert user_id not in org1_retrieved_ids, f"Org2 user {user_id} should not be in org1 list"
                
        finally:
            # Cleanup
            for user_id in org1_user_ids + org2_user_ids:
                try:
                    user_manager.delete_user_by_id(user_id)
                except Exception:
                    pass


# ============================================================================
# TEST CLASS: Client Deletion Does Not Cascade to Users
# ============================================================================


class TestClientDeletionPreservesUsers:
    """Tests verifying that deleting a client does NOT delete users."""

    def test_delete_client_preserves_users(self, user_manager, client_manager, test_org1):
        """
        Verify deleting a client does NOT cascade-delete users.
        
        Before the fix, users had a FK to clients with CASCADE delete.
        Now users are org-scoped and should persist when clients are deleted.
        """
        # Create a client
        client_id = generate_test_id("client")
        client = PydanticClient(
            id=client_id,
            name="Temporary Client",
            organization_id=test_org1.id,
            status="active",
            scope="read_write",
        )
        created_client = client_manager.create_client(client)
        
        # Create a user in the same org
        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Persistent User",
            organization_id=test_org1.id,
            timezone="UTC",
        )
        created_user = user_manager.create_user(user)
        
        try:
            # Verify user exists
            retrieved_user = user_manager.get_user_by_id(user_id)
            assert retrieved_user.id == user_id
            
            # Delete the client
            client_manager.delete_client_by_id(client_id)
            
            # Verify user STILL exists after client deletion
            user_after_delete = user_manager.get_user_by_id(user_id)
            assert user_after_delete.id == user_id, "User should still exist after client deletion"
            assert user_after_delete.organization_id == test_org1.id
            
        finally:
            # Cleanup user
            try:
                user_manager.delete_user_by_id(user_id)
            except Exception:
                pass


# ============================================================================
# TEST CLASS: get_or_create_org_default_user
# ============================================================================


class TestGetOrCreateOrgDefaultUser:
    """Tests for get_or_create_org_default_user without client_id."""

    def test_get_or_create_org_default_user_creates_user(self, user_manager, test_org1):
        """
        Verify get_or_create_org_default_user creates a default user for the org.
        """
        # Get or create default user
        default_user = user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        
        assert default_user is not None
        assert default_user.organization_id == test_org1.id
        assert default_user.name == user_manager.DEFAULT_USER_NAME
        
        # Cleanup
        try:
            user_manager.delete_user_by_id(default_user.id)
        except Exception:
            pass

    def test_get_or_create_org_default_user_is_idempotent(self, user_manager, test_org1):
        """
        Verify get_or_create_org_default_user returns the same user on repeated calls.
        """
        # First call
        default_user_1 = user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        
        # Second call
        default_user_2 = user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        
        assert default_user_1.id == default_user_2.id, "Should return same user on repeated calls"
        
        # Cleanup
        try:
            user_manager.delete_user_by_id(default_user_1.id)
        except Exception:
            pass

    def test_get_or_create_org_default_user_different_orgs(self, user_manager, test_org1, test_org2):
        """
        Verify get_or_create_org_default_user creates separate users for different orgs.
        """
        # Get default user for org1
        default_user_org1 = user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        
        # Get default user for org2
        default_user_org2 = user_manager.get_or_create_org_default_user(org_id=test_org2.id)
        
        assert default_user_org1.id != default_user_org2.id, "Different orgs should have different default users"
        assert default_user_org1.organization_id == test_org1.id
        assert default_user_org2.organization_id == test_org2.id
        
        # Cleanup
        try:
            user_manager.delete_user_by_id(default_user_org1.id)
        except Exception:
            pass
        try:
            user_manager.delete_user_by_id(default_user_org2.id)
        except Exception:
            pass


# ============================================================================
# TEST CLASS: UserManager API Signature Verification
# ============================================================================


class TestUserManagerApiSignature:
    """Tests verifying UserManager methods have correct signatures."""

    def test_create_user_has_no_client_id_parameter(self):
        """Verify create_user method has no client_id parameter."""
        import inspect
        
        sig = inspect.signature(UserManager.create_user)
        params = list(sig.parameters.keys())
        
        assert "client_id" not in params, "create_user should not have client_id parameter"
        assert "pydantic_user" in params, "create_user should have pydantic_user parameter"

    def test_list_users_has_no_client_id_parameter(self):
        """Verify list_users method has no client_id parameter."""
        import inspect
        
        sig = inspect.signature(UserManager.list_users)
        params = list(sig.parameters.keys())
        
        assert "client_id" not in params, "list_users should not have client_id parameter"
        assert "organization_id" in params, "list_users should have organization_id parameter"

    def test_get_or_create_org_default_user_has_no_client_id_parameter(self):
        """Verify get_or_create_org_default_user method has no client_id parameter."""
        import inspect
        
        sig = inspect.signature(UserManager.get_or_create_org_default_user)
        params = list(sig.parameters.keys())
        
        assert "client_id" not in params, "get_or_create_org_default_user should not have client_id parameter"
        assert "org_id" in params, "get_or_create_org_default_user should have org_id parameter"
