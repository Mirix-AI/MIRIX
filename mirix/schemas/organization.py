import uuid
from datetime import datetime
from typing import Optional

from pydantic import Field

from mirix.client.utils import get_utc_time
from mirix.schemas.mirix_base import MirixBase


class OrganizationBase(MirixBase):
    __id_prefix__ = "org"


def _generate_org_id() -> str:
    """Generate a random organization ID."""
    return f"org-{uuid.uuid4().hex[:8]}"


class Organization(OrganizationBase):
    id: str = Field(
        default_factory=_generate_org_id,
        description="The unique identifier of the organization.",
    )
    name: Optional[str] = Field(
        None,
        description="The name of the organization. Server will generate if not provided.",
        json_schema_extra={"default": "SincereYogurt"},
    )
    created_at: Optional[datetime] = Field(
        default_factory=get_utc_time,
        description="The creation date of the organization.",
    )
    # Relational DB provider returns an updated_at timestamp; not present on the PostgreSQL path.
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp (provider path).")


class OrganizationCreate(OrganizationBase):
    id: Optional[str] = Field(None, description="The unique identifier of the organization.")
    name: Optional[str] = Field(None, description="The name of the organization.")
