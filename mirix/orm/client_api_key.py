from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.client_api_key import ClientApiKey as PydanticClientApiKey

if TYPE_CHECKING:
    from mirix.orm import Client


class ClientApiKey(SqlalchemyBase, OrganizationMixin):
    """ClientApiKey ORM class - represents an API key for a client application"""

    __tablename__ = "client_api_keys"
    __pydantic_model__ = PydanticClientApiKey

    # Foreign key to client
    client_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=False,
        doc="The client this API key belongs to",
    )

    # API key fields
    api_key_hash: Mapped[str] = mapped_column(nullable=False, doc="Hashed API key for authentication")

    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Optional name/label for this API key")

    status: Mapped[str] = mapped_column(
        nullable=False, default="active", doc="Status of the API key: active, revoked, expired"
    )

    permission: Mapped[str] = mapped_column(
        nullable=False, default="all", doc="Permission level: all, restricted, read_only"
    )

    # Optional user association
    user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        doc="The user this API key is associated with (optional)",
    )

    # Relationships
    client: Mapped["Client"] = relationship("Client", back_populates="api_keys")
