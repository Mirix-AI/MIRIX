from typing import TYPE_CHECKING, Optional
from sqlalchemy import Column, JSON, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem
from datetime import datetime
import datetime as dt
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class SemanticMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores semantic memory entries that represent general knowledge,
    concepts, facts, and language elements that can be accessed without 
    relying on specific contextual experiences.

    Attributes:
        id: Unique ID for this semantic memory entry.
        name: The name of the concept or the object (e.g., "MemoryLLM", "Jane").
        summary: A concise summary of the concept or the object.
        details: A more detailed explanation or contextual description.
        source: The reference or origin of the information (e.g., book, article, movie).
        metadata_: Arbitrary additional metadata as a JSON object.
        created_at: Timestamp indicating when the entry was created.
    """

    __tablename__ = "semantic_memory"
    __pydantic_model__ = PydanticSemanticMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this semantic memory entry"
    )

    # The name of the concept or the object
    name: Mapped[str] = mapped_column(
        String,
        doc="The title or main concept for the knowledge entry"
    )

    # A concise summary of the concept
    summary: Mapped[str] = mapped_column(
        String,
        doc="A concise summary of the concept or the object."
    )

    # Detailed explanation or extended context about the concept
    details: Mapped[str] = mapped_column(
        String,
        doc="Detailed explanation or additional context for the concept"
    )

    # Reference or source of the general knowledge (e.g., book, article, or movie)
    source: Mapped[str] = mapped_column(
        String,
        doc="The reference or origin of this information (e.g., book, article, or movie)"
    )

    # Hierarchical tree path for categorization (e.g., ["favorites", "pets", "dog"])
    tree_path: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="Hierarchical categorization path as an array of strings (e.g., ['favorites', 'pets', 'dog'])"
    )

    # Additional arbitrary metadata stored as a JSON object
    metadata_: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=True,
        doc="Additional arbitrary metadata as a JSON object"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {"timestamp": datetime.now(dt.timezone.utc).isoformat(), "operation": "created"},
        doc="Last modification info including timestamp and operation type"
    )

    # Timestamp indicating when this entry was created
    created_at: Mapped[DateTime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(dt.timezone.utc),
        nullable=False,
        doc="Timestamp when this semantic memory entry was created"
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, 
        nullable=True,
        doc="Embedding configuration"
    )
    
    # Vector embedding field based on database type
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector
        details_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        name_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        summary_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        details_embedding = Column(CommonVector, nullable=True)
        name_embedding = Column(CommonVector, nullable=True)
        summary_embedding = Column(CommonVector, nullable=True)

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization, mirroring existing patterns.
        Adjust 'back_populates' to match the collection name in your `Organization` model.
        """
        return relationship(
            "Organization",
            back_populates="semantic_memory",
            lazy="selectin"
        )

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this semantic memory item.
        """
        return relationship(
            "User",
            lazy="selectin"
        )
