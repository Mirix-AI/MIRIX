"""
User context management utilities for multi-user support in MIRIX.

This module provides utilities for managing user context in multi-user scenarios,
including thread-local storage for user context and helper functions for user
validation and context switching.
"""

import threading
from typing import Optional, Dict, Any
from contextvars import ContextVar
from contextlib import contextmanager

from mirix.schemas.user import User as PydanticUser


# Thread-local storage for user context
_thread_local = threading.local()

# Context variable for user context (preferred for async environments)
_user_context: ContextVar[Optional[PydanticUser]] = ContextVar('user_context', default=None)


class UserContext:
    """
    User context manager for handling multi-user scenarios.
    
    This class provides a centralized way to manage user context across
    different parts of the application, ensuring proper user isolation
    and context switching.
    """
    
    def __init__(self):
        self._context_stack = []
    
    def get_current_user(self) -> Optional[PydanticUser]:
        """
        Get the current user from context.
        
        Tries context variable first (for async), then thread-local storage.
        
        Returns:
            Optional[PydanticUser]: The current user or None if no user is set
        """
        # Try context variable first (better for async)
        user = _user_context.get()
        if user is not None:
            return user
            
        # Fallback to thread-local storage
        return getattr(_thread_local, 'current_user', None)
    
    def set_current_user(self, user: Optional[PydanticUser]) -> None:
        """
        Set the current user in context.
        
        Sets both context variable and thread-local storage for compatibility.
        
        Args:
            user: The user to set as current, or None to clear
        """
        _user_context.set(user)
        _thread_local.current_user = user
    
    def clear_current_user(self) -> None:
        """Clear the current user from context."""
        self.set_current_user(None)
    
    @contextmanager
    def user_context(self, user: Optional[PydanticUser]):
        """
        Context manager for temporarily switching user context.
        
        Args:
            user: User to switch to temporarily
            
        Yields:
            The user that was set
            
        Example:
            with user_context_manager.user_context(some_user):
                # Operations here run with some_user as the current user
                pass
            # Original user context is restored
        """
        original_user = self.get_current_user()
        self._context_stack.append(original_user)
        
        try:
            self.set_current_user(user)
            yield user
        finally:
            restored_user = self._context_stack.pop() if self._context_stack else None
            self.set_current_user(restored_user)


# Global instance for application-wide use
user_context_manager = UserContext()


def get_current_user() -> Optional[PydanticUser]:
    """
    Get the current user from the global context manager.
    
    Returns:
        Optional[PydanticUser]: The current user or None
    """
    return user_context_manager.get_current_user()


def set_current_user(user: Optional[PydanticUser]) -> None:
    """
    Set the current user in the global context manager.
    
    Args:
        user: The user to set as current
    """
    user_context_manager.set_current_user(user)


def validate_user_id(user_id: str) -> bool:
    """
    Validate that a user ID follows the expected format.
    
    Args:
        user_id: The user ID to validate
        
    Returns:
        bool: True if the user ID is valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False
    
    # Check if it follows the user-{uuid} pattern
    if not user_id.startswith('user-'):
        return False
    
    # Extract the UUID part
    uuid_part = user_id[5:]  # Remove 'user-' prefix
    
    # Basic UUID format validation (8-4-4-4-12 hex digits)
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, uuid_part, re.IGNORECASE))


def ensure_user_organization_consistency(user: PydanticUser) -> bool:
    """
    Ensure that a user's organization context is consistent.
    
    Args:
        user: The user to validate
        
    Returns:
        bool: True if the user's organization context is valid
    """
    if not user:
        return False
    
    # Check that user has an organization_id
    if not hasattr(user, 'organization_id') or not user.organization_id:
        return False
    
    # Check that user ID is valid
    if not validate_user_id(user.id):
        return False
    
    return True


class UserContextError(Exception):
    """Exception raised for user context-related errors."""
    pass


def require_user_context() -> PydanticUser:
    """
    Require that a user context is set, raising an exception if not.
    
    Returns:
        PydanticUser: The current user
        
    Raises:
        UserContextError: If no user context is set
    """
    user = get_current_user()
    if user is None:
        raise UserContextError("No user context is set. This operation requires a valid user context.")
    
    if not ensure_user_organization_consistency(user):
        raise UserContextError(f"Invalid user context: user {user.id} has inconsistent organization data.")
    
    return user


def get_user_organization_id() -> Optional[str]:
    """
    Get the organization ID of the current user.
    
    Returns:
        Optional[str]: The organization ID or None if no user is set
    """
    user = get_current_user()
    return user.organization_id if user else None


def is_multi_user_operation() -> bool:
    """
    Check if we're currently in a multi-user operation context.
    
    Returns:
        bool: True if a specific user context is set (not just default user)
    """
    user = get_current_user()
    if not user:
        return False
    
    # Check if this is the default user (indicating single-user mode)
    DEFAULT_USER_ID = "user-00000000-0000-4000-8000-000000000000"
    return user.id != DEFAULT_USER_ID


def create_user_scoped_filters(base_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create database filters that include user scoping.
    
    Args:
        base_filters: Base filters to extend with user context
        
    Returns:
        Dict[str, Any]: Filters including user and organization scoping
    """
    filters = base_filters.copy() if base_filters else {}
    
    user = get_current_user()
    if user:
        filters['organization_id'] = user.organization_id
        if is_multi_user_operation():
            filters['user_id'] = user.id
    
    return filters
