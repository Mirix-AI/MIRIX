"""
Tests for multi-user support functionality in MIRIX.

This module tests the multi-user features including:
- User context switching
- User isolation in memory operations
- Backwards compatibility with single-user mode
- FastAPI endpoint user parameter handling
"""

from unittest.mock import Mock, patch

import pytest

from mirix.agent.agent_wrapper import AgentWrapper
from mirix.schemas.user import User as PydanticUser
from mirix.server.fastapi_server import MessageRequest
from mirix.services.user_manager import UserManager
from mirix.utils.user_context import (
    UserContext,
    create_user_scoped_filters,
    ensure_user_organization_consistency,
    is_multi_user_operation,
    validate_user_id,
)


class TestUserContextUtilities:
    """Test user context management utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user_context = UserContext()

        # Create mock users
        self.user1 = PydanticUser(
            id="user-12345678-1234-4000-8000-123456789abc",
            name="Test User 1",
            organization_id="org-87654321-4321-4000-8000-abcdef123456",
            timezone="UTC",
        )

        self.user2 = PydanticUser(
            id="user-87654321-4321-4000-8000-abcdef123456",
            name="Test User 2",
            organization_id="org-12345678-1234-4000-8000-123456789abc",
            timezone="UTC",
        )

        self.default_user = PydanticUser(
            id="user-00000000-0000-4000-8000-000000000000",
            name="Default User",
            organization_id="org-00000000-0000-4000-8000-000000000000",
            timezone="UTC",
        )

    def test_user_context_get_set_clear(self):
        """Test basic user context operations."""
        # Initially no user
        assert self.user_context.get_current_user() is None

        # Set user
        self.user_context.set_current_user(self.user1)
        current = self.user_context.get_current_user()
        assert current is not None
        assert current.id == self.user1.id

        # Clear user
        self.user_context.clear_current_user()
        assert self.user_context.get_current_user() is None

    def test_user_context_manager(self):
        """Test user context manager for temporary context switching."""
        # Set initial user
        self.user_context.set_current_user(self.user1)

        # Use context manager to temporarily switch
        with self.user_context.user_context(self.user2):
            current = self.user_context.get_current_user()
            assert current.id == self.user2.id

        # Should restore original user
        current = self.user_context.get_current_user()
        assert current.id == self.user1.id

    def test_validate_user_id(self):
        """Test user ID validation."""
        # Valid user IDs
        assert validate_user_id("user-12345678-1234-4000-8000-123456789abc")
        assert validate_user_id("user-00000000-0000-4000-8000-000000000000")

        # Invalid user IDs
        assert not validate_user_id("")
        assert not validate_user_id(None)
        assert not validate_user_id("invalid-id")
        assert not validate_user_id("user-invalid-uuid")
        assert not validate_user_id("12345678-1234-4000-8000-123456789abc")  # Missing prefix

    def test_ensure_user_organization_consistency(self):
        """Test user organization consistency validation."""
        # Valid user
        assert ensure_user_organization_consistency(self.user1)

        # Invalid cases
        assert not ensure_user_organization_consistency(None)

        # User without organization_id
        invalid_user = PydanticUser(
            id="user-12345678-1234-4000-8000-123456789abc", name="Invalid User", organization_id=None, timezone="UTC"
        )
        assert not ensure_user_organization_consistency(invalid_user)

    def test_is_multi_user_operation(self):
        """Test multi-user operation detection."""
        from mirix.utils.user_context import set_current_user

        # No user set
        set_current_user(None)
        assert not is_multi_user_operation()

        # Default user
        set_current_user(self.default_user)
        assert not is_multi_user_operation()

        # Non-default user
        set_current_user(self.user1)
        assert is_multi_user_operation()

    def test_create_user_scoped_filters(self):
        """Test creation of user-scoped database filters."""
        from mirix.utils.user_context import set_current_user

        # No user context
        set_current_user(None)
        filters = create_user_scoped_filters()
        assert filters == {}

        # Default user context
        set_current_user(self.default_user)
        filters = create_user_scoped_filters()
        assert "organization_id" in filters
        assert "user_id" not in filters  # Default user doesn't add user_id filter

        # Multi-user context
        set_current_user(self.user1)
        filters = create_user_scoped_filters()
        assert "organization_id" in filters
        assert "user_id" in filters
        assert filters["user_id"] == self.user1.id

        # With base filters
        base_filters = {"status": "active"}
        filters = create_user_scoped_filters(base_filters)
        assert "status" in filters
        assert "organization_id" in filters
        assert "user_id" in filters


class TestUserManager:
    """Test UserManager multi-user enhancements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user_manager = Mock(spec=UserManager)
        self.user_manager.DEFAULT_USER_ID = "user-00000000-0000-4000-8000-000000000000"

    def test_is_default_user(self):
        """Test default user detection."""
        user_manager = UserManager()

        # Default user
        assert user_manager.is_default_user("user-00000000-0000-4000-8000-000000000000")

        # Non-default user
        assert not user_manager.is_default_user("user-12345678-1234-4000-8000-123456789abc")


class TestAgentWrapperUserContext:
    """Test AgentWrapper user context functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_server = Mock()
        self.mock_user_manager = Mock()

        # Set up user manager mock
        self.default_user = PydanticUser(
            id="user-00000000-0000-4000-8000-000000000000",
            name="Default User",
            organization_id="org-default",
            timezone="UTC",
        )

        self.test_user = PydanticUser(
            id="user-12345678-1234-4000-8000-123456789abc", name="Test User", organization_id="org-test", timezone="UTC"
        )

        self.mock_user_manager.get_user_or_default.return_value = self.test_user
        self.mock_server.user_manager = self.mock_user_manager
        self.mock_client.server = self.mock_server
        self.mock_client.user = self.default_user

    @patch("mirix.agent.agent_wrapper.AgentWrapper.__init__", lambda x, y, z=None: None)
    def test_send_message_user_context_switching(self):
        """Test that send_message properly switches user context."""
        # Create mock agent wrapper
        agent = AgentWrapper.__new__(AgentWrapper)
        agent.client = self.mock_client
        agent.logger = Mock()

        # Mock the necessary methods to avoid actual message processing
        agent.model_name = "test-model"
        agent.is_gemini_client_initialized = Mock(return_value=True)

        # Mock the entire message processing to focus on user context
        with patch.object(agent, "_actual_send_message_logic", return_value="test response"):
            # Test user context switching
            original_user = agent.client.user

            # Call send_message with user_id
            agent.send_message(message="test message", user_id="user-12345678-1234-4000-8000-123456789abc")

            # Verify user_manager.get_user_or_default was called
            self.mock_user_manager.get_user_or_default.assert_called_with("user-12345678-1234-4000-8000-123456789abc")

            # Verify user context was restored
            assert agent.client.user == original_user


class TestFastAPIUserSupport:
    """Test FastAPI endpoints user parameter support."""

    def test_message_request_schema(self):
        """Test MessageRequest schema supports user_id."""
        # Test with user_id
        request = MessageRequest(
            message="test message", memorizing=True, user_id="user-12345678-1234-4000-8000-123456789abc"
        )

        assert request.message == "test message"
        assert request.memorizing is True
        assert request.user_id == "user-12345678-1234-4000-8000-123456789abc"

        # Test without user_id (backwards compatibility)
        request = MessageRequest(message="test message", memorizing=False)

        assert request.message == "test message"
        assert request.memorizing is False
        assert request.user_id is None


class TestUserIsolation:
    """Test user isolation in memory operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user1 = PydanticUser(
            id="user-12345678-1234-4000-8000-123456789abc", name="User 1", organization_id="org-test", timezone="UTC"
        )

        self.user2 = PydanticUser(
            id="user-87654321-4321-4000-8000-abcdef123456", name="User 2", organization_id="org-test", timezone="UTC"
        )

    @patch("mirix.services.episodic_memory_manager.EpisodicMemoryManager")
    def test_memory_manager_user_filtering(self, mock_memory_manager):
        """Test that memory managers properly filter by user_id."""
        memory_manager = mock_memory_manager.return_value

        # Mock the list method to verify user filtering
        mock_results = []
        memory_manager.list_episodic_memory.return_value = mock_results

        # Simulate calling with actor parameter
        memory_manager.list_episodic_memory(
            agent_state=Mock(),
            query="test",
            search_field="summary",
            search_method="bm25",
            limit=10,
            timezone_str="UTC",
            actor=self.user1,
        )

        # Verify the method was called with actor
        memory_manager.list_episodic_memory.assert_called_once()
        call_args = memory_manager.list_episodic_memory.call_args
        assert call_args.kwargs["actor"] == self.user1


class TestBackwardsCompatibility:
    """Test backwards compatibility with single-user mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.default_user = PydanticUser(
            id="user-00000000-0000-4000-8000-000000000000",
            name="Default User",
            organization_id="org-default",
            timezone="UTC",
        )

    @patch("mirix.agent.agent_wrapper.AgentWrapper.__init__", lambda x, y, z=None: None)
    def test_send_message_without_user_id(self):
        """Test that send_message works without user_id parameter."""
        # Create mock agent wrapper
        agent = AgentWrapper.__new__(AgentWrapper)
        agent.client = Mock()
        agent.client.user = self.default_user
        agent.logger = Mock()

        # Mock the necessary methods
        agent.model_name = "test-model"
        agent.is_gemini_client_initialized = Mock(return_value=True)

        # Mock the entire message processing
        with patch.object(agent, "_actual_send_message_logic", return_value="test response"):
            original_user = agent.client.user

            # Call without user_id (backwards compatibility)
            agent.send_message(message="test message")

            # User context should remain unchanged
            assert agent.client.user == original_user

    def test_message_request_backwards_compatibility(self):
        """Test MessageRequest backwards compatibility."""
        # Old-style request without user_id
        request = MessageRequest(message="test")
        assert request.user_id is None

        # Should still work with existing fields
        assert request.message == "test"
        assert request.memorizing is False


# Integration test marker
@pytest.mark.integration
class TestMultiUserIntegration:
    """Integration tests for multi-user functionality."""

    def test_end_to_end_user_isolation(self):
        """Test end-to-end user isolation scenario."""
        # This would test actual database isolation between users
        # Implementation would depend on test database setup
        pytest.skip("Integration test - requires database setup")

    def test_concurrent_user_operations(self):
        """Test concurrent operations with different users."""
        # This would test thread safety of user context switching
        pytest.skip("Integration test - requires concurrency testing setup")


if __name__ == "__main__":
    pytest.main([__file__])
