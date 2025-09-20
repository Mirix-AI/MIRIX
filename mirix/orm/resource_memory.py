from typing import TYPE_CHECKING, Optional
from datetime import datetime
import datetime as dt

from sqlalchemy import Column, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.orm.mixins import OrganizationMixin, UserMixin

from mirix.schemas.resource_memory import ResourceMemoryItem as PydanticResourceMemoryItem
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class ResourceMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores references to user's documents, files, or resources for easy retrieval & linking to tasks.
    
    title:   A short name/title of the resource (e.g. 'MarketingPlan2025')
    summary:        A brief description or summary of the resource.
    metadata_:       JSON for storing tags, creation date, personal notes, etc.
    content:         The text/content of the file (can be partial or full)
    resource_type:   Category or type of the resource (e.g. 'doc', 'text', 'markdown', 'spreadsheet')
    """

    __tablename__ = "resource_memory"
    __pydantic_model__ = PydanticResourceMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this resource memory entry",
    )

    title: Mapped[str] = mapped_column(
        String,
        doc="Short name or title of the resource"
    )

    summary: Mapped[str] = mapped_column(
        String,
        doc="Brief description or summary of the resource"
    )

    resource_type: Mapped[str] = mapped_column(
        String,
        doc="Type or format of the resource (e.g. 'doc', 'markdown', 'pdf_text')"
    )

    content: Mapped[str] = mapped_column(
        String,
        doc="Full text or partial content of this resource"
    )

    # Hierarchical categorization path
    tree_path: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="Hierarchical categorization path as an array of strings"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {"timestamp": datetime.now(dt.timezone.utc).isoformat(), "operation": "created"},
        doc="Last modification info including timestamp and operation type"
    )

    metadata_: Mapped[dict] = mapped_column(
        JSON,
        default={},
        nullable=True,
        doc="Arbitrary additional metadata as JSON (tags, creation date, personal notes, etc.)"
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, 
        nullable=True,
        doc="Embedding configuration"
    )
    
    # Vector embedding field based on database type
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector
        summary_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        summary_embedding = Column(CommonVector, nullable=True)

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        """
        return relationship(
            "Organization",
            back_populates="resource_memory",
            lazy="selectin"
        )

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this resource memory item.
        """
        return relationship(
            "User",
            lazy="selectin"
        )
