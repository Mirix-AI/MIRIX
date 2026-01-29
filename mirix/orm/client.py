from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.client import Client as PydanticClient

if TYPE_CHECKING:
    from mirix.orm import Organization
    from mirix.orm.client_api_key import ClientApiKey
    from mirix.orm.user import User


class Client(SqlalchemyBase, OrganizationMixin):
    """Client ORM class - represents a client application"""

    __tablename__ = "clients"
    __pydantic_model__ = PydanticClient

    # Basic fields
    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the client application.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the client is active or not.")
    scope: Mapped[str] = mapped_column(
        nullable=False, default="read_write", doc="Scope of client: read, write, read_write, admin"
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
    users: Mapped[List["User"]] = relationship(
        "User", back_populates="client", cascade="all, delete-orphan", lazy="selectin"
    )
