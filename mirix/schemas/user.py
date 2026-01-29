import uuid
from datetime import datetime
from typing import Optional

from pydantic import Field

from mirix.constants import DEFAULT_ORG_ID
from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.mirix_base import MirixBase


class UserBase(MirixBase):
    __id_prefix__ = "user"


def _generate_user_id() -> str:
    """Generate a random user ID."""
    return f"user-{uuid.uuid4().hex[:8]}"


class User(UserBase):
    """
    Representation of a user.

    Parameters:
        id (str): The unique identifier of the user.
        name (str): The name of the user.
        status (str): Whether the user is active or not.
        client_id (str): The client this user belongs to.
        created_at (datetime): The creation date of the user.
    """

    # ID is now a REQUIRED field (no default_factory) to prevent ID overwrite bugs.
    # Previously, default_factory=_generate_user_id would regenerate IDs when
    # existing users were loaded from DB and re-validated through Pydantic,
    # causing user IDs to be overwritten in the queue worker.
    id: str = Field(
        ...,
        description="The unique identifier of the user.",
    )
    organization_id: Optional[str] = Field(
        DEFAULT_ORG_ID,
        description="The organization id of the user",
    )
    client_id: Optional[str] = Field(
        None,
        description="The client this user belongs to.",
    )
    name: str = Field(..., description="The name of the user.")
    status: str = Field("active", description="Whether the user is active or not.")
    timezone: str = Field(..., description="The timezone of the user.")
    is_admin: bool = Field(False, description="Whether this is an admin user for the client.")
    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the user.")
    updated_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The update date of the user.")
    is_deleted: bool = Field(False, description="Whether this user is deleted or not.")


class UserCreate(UserBase):
    id: Optional[str] = Field(None, description="The unique identifier of the user.")
    name: str = Field(..., description="The name of the user.")
    status: str = Field("active", description="Whether the user is active or not.")
    timezone: str = Field(..., description="The timezone of the user.")
    organization_id: str = Field(..., description="The organization id of the user.")


class UserUpdate(UserBase):
    id: str = Field(..., description="The id of the user to update.")
    name: Optional[str] = Field(None, description="The new name of the user.")
    status: Optional[str] = Field(None, description="The new status of the user.")
    timezone: Optional[str] = Field(None, description="The new timezone of the user.")
    organization_id: Optional[str] = Field(None, description="The new organization id of the user.")
