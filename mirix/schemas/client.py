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
        write_scope (Optional[str]): Scope for writing memories (null = read-only).
        read_scopes (List[str]): Scopes for reading memories.
        email (str): Optional email for dashboard login.
        password_hash (str): Optional password hash for dashboard login.
        created_at (datetime): The creation date of the client.
    """

    id: str = Field(
        default_factory=_generate_client_id,
        description="The unique identifier of the client.",
    )
    organization_id: Optional[str] = Field(
        default=OrganizationManager.DEFAULT_ORG_ID,
        description="The organization id of the client",
    )
    name: str = Field(..., description="The name of the client application.")
    status: str = Field(default="active", description="Whether the client is active or not.")
    write_scope: Optional[str] = Field(default=None, description="Scope for writing memories (null = read-only).")
    read_scopes: List[str] = Field(default_factory=list, description="Scopes for reading memories.")

    # Message retention
    message_set_retention_count: Optional[int] = Field(
        default=0, description="Number of input message-sets to retain per (agent, user). 0 = no retention."
    )

    # Dashboard authentication fields
    email: Optional[str] = Field(default=None, description="Email address for dashboard login.")
    password_hash: Optional[str] = Field(default=None, description="Hashed password for dashboard login.")
    last_login: Optional[datetime] = Field(default=None, description="Last dashboard login time.")

    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the client.")
    updated_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The update date of the client.")
    is_deleted: bool = Field(default=False, description="Whether this client is deleted or not.")


class ClientCreate(ClientBase):
    id: Optional[str] = Field(default=None, description="The unique identifier of the client.")
    name: str = Field(..., description="The name of the client application.")
    status: str = Field(default="active", description="Whether the client is active or not.")
    write_scope: Optional[str] = Field(default=None, description="Scope for writing memories (null = read-only).")
    read_scopes: List[str] = Field(default_factory=list, description="Scopes for reading memories.")
    organization_id: str = Field(..., description="The organization id of the client.")


class ClientUpdate(ClientBase):
    id: str = Field(..., description="The id of the client to update.")
    name: Optional[str] = Field(default=None, description="The new name of the client.")
    status: Optional[str] = Field(default=None, description="The new status of the client.")
    write_scope: Optional[str] = Field(default=None, description="The new write scope of the client.")
    read_scopes: Optional[List[str]] = Field(default=None, description="The new read scopes of the client.")
    organization_id: Optional[str] = Field(default=None, description="The new organization id of the client.")
    message_set_retention_count: Optional[int] = Field(
        default=None, description="Number of input message-sets to retain per (agent, user). 0 = no retention."
    )
