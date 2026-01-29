import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import Field

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.mirix_base import MirixBase
from mirix.services.organization_manager import OrganizationManager


class ClientBase(MirixBase):
    __id_prefix__ = "client"


def _generate_client_id() -> str:
    """Generate a random client ID."""
    return f"client-{uuid.uuid4().hex[:8]}"


class Client(ClientBase):
    """
    Representation of a client application.

    Parameters:
        id (str): The unique identifier of the client.
        name (str): The name of the client application.
        status (str): Whether the client is active or not.
        scope (str): Scope of client (read, write, read_write, admin).
        email (str): Optional email for dashboard login.
        password_hash (str): Optional password hash for dashboard login.
        created_at (datetime): The creation date of the client.
    """

    id: str = Field(
        default_factory=_generate_client_id,
        description="The unique identifier of the client.",
    )
    organization_id: Optional[str] = Field(
        OrganizationManager.DEFAULT_ORG_ID,
        description="The organization id of the client",
    )
    name: str = Field(..., description="The name of the client application.")
    status: str = Field("active", description="Whether the client is active or not.")
    scope: str = Field("read_write", description="Scope of client.")

    # Dashboard authentication fields
    email: Optional[str] = Field(None, description="Email address for dashboard login.")
    password_hash: Optional[str] = Field(None, description="Hashed password for dashboard login.")
    last_login: Optional[datetime] = Field(None, description="Last dashboard login time.")

    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the client.")
    updated_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The update date of the client.")
    is_deleted: bool = Field(False, description="Whether this client is deleted or not.")


class ClientCreate(ClientBase):
    id: Optional[str] = Field(None, description="The unique identifier of the client.")
    name: str = Field(..., description="The name of the client application.")
    status: str = Field("active", description="Whether the client is active or not.")
    scope: str = Field("read_write", description="Scope of client.")
    organization_id: str = Field(..., description="The organization id of the client.")


class ClientUpdate(ClientBase):
    id: str = Field(..., description="The id of the client to update.")
    name: Optional[str] = Field(None, description="The new name of the client.")
    status: Optional[str] = Field(None, description="The new status of the client.")
    scope: Optional[str] = Field(None, description="The new scope of the client.")
    organization_id: Optional[str] = Field(None, description="The new organization id of the client.")
