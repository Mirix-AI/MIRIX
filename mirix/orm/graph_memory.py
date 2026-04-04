"""
Graph Memory ORM models for MIRIX-V2.

Implements a temporal knowledge graph with:
  - EntityNode: entities (people, places, concepts)
  - EntityEdge: semantic facts between entities (bi-temporal)
  - EpisodeNode: timestamped events
  - InvolvesEdge: cross-links between episodes and entities
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.custom_columns import CommonVector
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.settings import settings

# Graph memory uses flexible-dimension vectors (no fixed MAX_EMBEDDING_DIM).
# text-embedding-3-small = 1536, text-embedding-004 = 768, etc.
GRAPH_EMBEDDING_DIM = 1536

if settings.mirix_pg_uri_no_default:
    from pgvector.sqlalchemy import Vector

    _VectorCol = lambda: mapped_column(Vector(GRAPH_EMBEDDING_DIM), nullable=True)
else:
    _VectorCol = lambda: Column(CommonVector, nullable=True)


class EntityNode(SqlalchemyBase, OrganizationMixin, UserMixin):
    """A named entity extracted from conversations (person, place, concept, etc.)."""

    __tablename__ = "entity_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False, server_default="GENERIC")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding = _VectorCol()
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True, server_default=text("'{}'"))

    __table_args__ = (
        Index("idx_entity_type", "entity_type"),
    )


class EntityEdge(SqlalchemyBase, OrganizationMixin, UserMixin):
    """A semantic fact between two entities with bi-temporal validity."""

    __tablename__ = "entity_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    src_id: Mapped[str] = mapped_column(String, ForeignKey("entity_nodes.id"), nullable=False)
    dst_id: Mapped[str] = mapped_column(String, ForeignKey("entity_nodes.id"), nullable=False)
    rel_type: Mapped[str] = mapped_column(String, nullable=False)
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = _VectorCol()

    # Bi-temporal model
    expired_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    invalid_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Provenance
    source_episode_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True, server_default=text("'{}'"))

    __table_args__ = (
        Index("idx_edge_src", "src_id"),
        Index("idx_edge_dst", "dst_id"),
        Index("idx_edge_rel_type", "rel_type"),
    )


class EpisodeNode(SqlalchemyBase, OrganizationMixin, UserMixin):
    """A timestamped event/episode in the knowledge graph."""

    __tablename__ = "episode_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding = _VectorCol()
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(String, nullable=True, server_default="conversation")
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True, server_default=text("'{}'"))

    __table_args__ = (
        Index("idx_episode_time", "event_time"),
    )


class InvolvesEdge(SqlalchemyBase):
    """Cross-link between an episode and an entity it mentions."""

    __tablename__ = "involves_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    episode_id: Mapped[str] = mapped_column(String, ForeignKey("episode_nodes.id"), nullable=False)
    entity_id: Mapped[str] = mapped_column(String, ForeignKey("entity_nodes.id"), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String, nullable=True, server_default="MENTIONED")

    __table_args__ = (
        UniqueConstraint("episode_id", "entity_id", name="uq_involves_episode_entity"),
        Index("idx_involves_episode", "episode_id"),
        Index("idx_involves_entity", "entity_id"),
    )
