from typing import TYPE_CHECKING, Optional
from datetime import datetime
import datetime as dt

from sqlalchemy import Column, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.schemas.knowledge_vault import KnowledgeVaultItem as PydanticKnowledgeVaultItem

from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class KnowledgeVaultItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores verbatim knowledge vault entries like credentials, bookmarks, addresses,
    or other structured data that needs quick retrieval.

    type:        The category (e.g. 'credential', 'bookmark', 'contact')
    source:      The origin or context (e.g. 'user-provided on 2025-03-01')
    sensitivity: Level of data sensitivity (e.g. 'low', 'high')
    secret_value: The actual data or secret (e.g. password, token)
    metadata_:   Optional JSON for extra fields/notes
    """

    __tablename__ = "knowledge_vault"
    __pydantic_model__ = PydanticKnowledgeVaultItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this knowledge vault entry",
    )

    # Distinguish the type/category of the entry
    entry_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Category (e.g., 'credential', 'bookmark', 'api_key')"
    )

    # Source (where or how it was provided)
    source: Mapped[str] = mapped_column(
        String,
        doc="Information on who/where it was provided (e.g. 'user on 2025-03-01')"
    )

    # Sensitivity level
    sensitivity: Mapped[str] = mapped_column(
        String,
        doc="Data sensitivity (e.g. 'low', 'medium', 'high')"
    )

    # Actual data or secret, e.g. password, API token
    secret_value: Mapped[str] = mapped_column(
        String,
        doc="The actual credential or data value"
    )

    # Description or notes about the entry
    caption: Mapped[str] = mapped_column(
        String,
        doc="Description or notes about the entry"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {"timestamp": datetime.now(dt.timezone.utc).isoformat(), "operation": "created"},
        doc="Last modification info including timestamp and operation type"
    )

    # Optional catch-all for any extra metadata you want to store
    metadata_: Mapped[dict] = mapped_column(
        JSON,
        default={},
        nullable=True,
        doc="Arbitrary additional metadata as a JSON object"
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, 
        nullable=True,
        doc="Embedding configuration"
    )
    
    # Vector embedding field based on database type
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector
        caption_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        caption_embedding = Column(CommonVector, nullable=True)

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        Adjust 'back_populates' to match the collection name in your `Organization` model.
        """
        return relationship(
            "Organization",
            back_populates="knowledge_vault",
            lazy="selectin"
        )

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this knowledge vault item.
        """
        return relationship(
            "User",
            lazy="selectin"
        )
