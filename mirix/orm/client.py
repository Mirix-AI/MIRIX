from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.client import Client as PydanticClient

if TYPE_CHECKING:
    from mirix.orm import Organization
    from mirix.orm.client_api_key import ClientApiKey


class Client(SqlalchemyBase, OrganizationMixin):
    """Client ORM class - represents a client application"""

    __tablename__ = "clients"
    __pydantic_model__ = PydanticClient

    # Basic fields
    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the client application.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the client is active or not.")
    write_scope: Mapped[Optional[str]] = mapped_column(
        nullable=True, default=None, doc="Scope for writing memories (null = read-only)."
    )
    read_scopes: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list, doc="Scopes for reading memories."
    )

    # Dashboard authentication fields
    email: Mapped[Optional[str]] = mapped_column(
        nullable=True, unique=True, index=True, doc="Email address for dashboard login."
    )
    password_hash: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="Hashed password for dashboard login (bcrypt)."
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(nullable=True, doc="Last dashboard login time.")

    # Relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="clients")
    api_keys: Mapped[List["ClientApiKey"]] = relationship(
        "ClientApiKey", back_populates="client", cascade="all, delete-orphan", lazy="selectin"
    )
