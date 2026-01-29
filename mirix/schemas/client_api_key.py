from datetime import datetime
from typing import Optional

from pydantic import Field

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.mirix_base import MirixBase


class ClientApiKeyBase(MirixBase):
    __id_prefix__ = "client_api_key"


class ClientApiKey(ClientApiKeyBase):
    """
    Representation of a client API key.

    Parameters:
        id (str): The unique identifier of the API key.
        client_id (str): The client this API key belongs to.
        organization_id (str): The organization id.
        api_key_hash (str): The hashed API key.
        name (Optional[str]): Optional name/label for this API key.
        status (str): Status of the API key (active, revoked, expired).
        permission (str): Permission level (all, restricted, read_only).
        created_at (datetime): The creation date of the API key.
        updated_at (datetime): The last update date of the API key.
    """

    id: str = Field(
        default_factory=lambda: ClientApiKeyBase._generate_id(),
        description="The unique identifier of the API key.",
    )
    client_id: str = Field(..., description="The client this API key belongs to")
    organization_id: str = Field(..., description="The organization id")
    api_key_hash: str = Field(..., description="Hashed API key for authentication")
    name: Optional[str] = Field(None, description="Optional name/label for this API key")
    status: str = Field("active", description="Status: active, revoked, expired")
    permission: str = Field("all", description="Permission level: all, restricted, read_only")
    user_id: Optional[str] = Field(None, description="User ID this API key is associated with")

    created_at: Optional[datetime] = Field(
        default_factory=get_utc_time, description="The creation date of the API key."
    )
    updated_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The update date of the API key.")
    is_deleted: bool = Field(False, description="Whether this API key is deleted or not.")


class ClientApiKeyCreate(ClientApiKeyBase):
    """Schema for creating a new client API key."""

    id: Optional[str] = Field(None, description="The unique identifier of the API key.")
    client_id: str = Field(..., description="The client this API key belongs to")
    organization_id: str = Field(..., description="The organization id")
    api_key_hash: str = Field(..., description="Hashed API key for authentication")
    name: Optional[str] = Field(None, description="Optional name/label for this API key")
    status: str = Field("active", description="Status: active, revoked, expired")
    permission: str = Field("all", description="Permission level: all, restricted, read_only")
    user_id: Optional[str] = Field(None, description="User ID this API key is associated with")


class ClientApiKeyUpdate(ClientApiKeyBase):
    """Schema for updating a client API key."""

    id: str = Field(..., description="The id of the API key to update.")
    name: Optional[str] = Field(None, description="The new name of the API key.")
    status: Optional[str] = Field(None, description="The new status of the API key.")
