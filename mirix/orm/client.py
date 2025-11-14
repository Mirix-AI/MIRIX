from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.client import Client as PydanticClient

if TYPE_CHECKING:
    from mirix.orm import Organization


class Client(SqlalchemyBase, OrganizationMixin):
    """Client ORM class - represents a client application"""

    __tablename__ = "clients"
    __pydantic_model__ = PydanticClient

    # Basic fields
    name: Mapped[str] = mapped_column(
        nullable=False, doc="The display name of the client application."
    )
    status: Mapped[str] = mapped_column(
        nullable=False, doc="Whether the client is active or not."
    )
    scope: Mapped[str] = mapped_column(
        nullable=False,
        default="read_write",
        doc="Scope of client: read, write, read_write, admin"
    )

    # Authentication
    api_key_hash: Mapped[str] = mapped_column(
        nullable=True, doc="Hashed API key for authentication"
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="clients"
    )

