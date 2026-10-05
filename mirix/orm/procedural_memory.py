import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, Computed, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User

# Keyword-retrieval document of a skill: name and triggers at
# weight A, description at weight B.
#
# The triggers are indexed from `triggers::jsonb::text`, not `triggers::text`:
# create_all makes `triggers` a JSON column, whose text is the stored
# `json.dumps` output with every non-ASCII character escaped
# (`["\u201ccool\u201d the apple\u2014then store it"]`), so the words next to a
# curly quote, a dash or any accented / CJK character reached the index as
# 'u201ccool' / 'u2014then' / 'caf' + 'u00e9' and never matched a query.
# The JSONB cast decodes those escapes (and is a no-op for databases whose
# column is JSONB, upgraded by scripts/migrate_procedural_to_skill.sql).
#
# jsonb's own text still escapes a control character, a quote and a
# backslash (`["line1\nline2"]`), which Postgres' parser read as the junk
# lexeme 'nline2' while the SQLite path (a Python tokenizer over the trigger
# strings) found 'line2'. `regexp_replace` (immutable, so allowed in a
# generated column) turns every such escape — `\"`, `\\`, `\/`, `\b`, `\f`,
# `\n`, `\r`, `\t`, `\uXXXX` — into a space first, so both backends index the
# same words.
#
# Must stay byte-identical to the expression in
# scripts/migrate_skill_retrieval_fields.sql.
RETRIEVAL_TSV_TRIGGERS_TEXT = (
    "regexp_replace(triggers::jsonb::text, "
    r"""'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})', ' ', 'g')"""
)
RETRIEVAL_TSV_EXPRESSION = (
    "setweight(to_tsvector('english', coalesce(name, '')), 'A') || "
    f"setweight(to_tsvector('english', coalesce({RETRIEVAL_TSV_TRIGGERS_TEXT}, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(description, '')), 'B')"
)

# Skill names are unique per (organization, user) among LIVE skills only, so a
# soft-deleted skill's name can be reused. Replaces the old full
# UniqueConstraint uq_procedural_memory_org_user_name.
UNIQUE_LIVE_NAME_INDEX = "uq_procedural_memory_org_user_name_active"


class ProceduralMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores procedural knowledge as reusable skills.

    name:         Short skill identifier: lowercase words joined by hyphens
    entry_type:   Category or tag of the skill (e.g. 'workflow', 'guide', 'script')
    description:  Short descriptive text about what this skill accomplishes
    instructions: Step-by-step instructions as plain text
    triggers:     Conditions that indicate this skill is relevant
    examples:     Input/output examples for this skill
    version:      Semantic version of this skill
    """

    __tablename__ = "procedural_memory"
    __pydantic_model__ = PydanticProceduralMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this procedural memory entry",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this procedural memory item belongs to",
    )

    # Foreign key to client (for access control and filtering)
    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this item",
    )

    # Short skill identifier
    name: Mapped[str] = mapped_column(
        String, nullable=False, doc="Short skill identifier: lowercase words joined by hyphens"
    )

    # Distinguish the type/category of the skill
    entry_type: Mapped[str] = mapped_column(String, doc="Category or type (e.g. 'workflow', 'guide', 'script')")

    # A human-friendly description of this skill
    description: Mapped[str] = mapped_column(String, doc="Short descriptive text about what this skill accomplishes")

    # Step-by-step instructions as plain text
    instructions: Mapped[str] = mapped_column(String, doc="Step-by-step instructions as a single string")

    # Conditions that indicate this skill is relevant
    triggers: Mapped[list] = mapped_column(JSON, default=list, doc="Conditions that indicate this skill is relevant")

    # Input/output examples
    examples: Mapped[list] = mapped_column(JSON, default=list, doc="Input/output examples for this skill")

    # Semantic version
    version: Mapped[str] = mapped_column(String, default="0.1.0", doc="Semantic version of this skill")

    # NEW: Filter tags for flexible filtering and categorization
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags for filtering and categorization"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "operation": "created",
        },
        doc="Last modification info including timestamp and operation type",
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, nullable=True, doc="Embedding configuration"
    )

    # The skill's single retrieval vector, padded to MAX_EMBEDDING_DIM.
    # `instructions` deliberately has no vector: an embedding search on
    # it is served by full-text search instead.
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector
        from sqlalchemy.dialects.postgresql import TSVECTOR

        retrieval_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)

        # Postgres-only generated keyword document for BM25 retrieval,
        # GIN-indexed below. Deferred: it is only ever read by ranking SQL, so
        # ORM loads never ship it. Absent on SQLite, whose create_all has no
        # tsvector type (the SQLite path tokenizes in Python instead).
        retrieval_tsv = mapped_column(
            TSVECTOR,
            Computed(RETRIEVAL_TSV_EXPRESSION, persisted=True),
            nullable=True,
            deferred=True,
        )
    else:
        retrieval_embedding = Column(CommonVector, nullable=True)

    # Database indexes for efficient querying.
    # NB: use an explicit ``is not None`` filter rather than ``filter(None, ...)``
    # because an un-attached UniqueConstraint is falsy (it iterates over its
    # columns which are empty until binding), so plain truthiness would drop it.
    __table_args__ = tuple(
        item
        for item in [
                # Organization-level query optimization indexes
                (
                    Index("ix_procedural_memory_organization_id", "organization_id")
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_created_at",
                        "organization_id",
                        "created_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Enforce per-user skill-name uniqueness at the DB level. The
                # application also pre-checks via a search, but that is racy
                # under concurrent creates — this index closes the race. Partial
                # (live rows only) on BOTH backends so a soft-deleted skill never
                # blocks re-creating its name.
                Index(
                    UNIQUE_LIVE_NAME_INDEX,
                    "organization_id",
                    "user_id",
                    "name",
                    unique=True,
                    postgresql_where=text("NOT is_deleted"),
                    sqlite_where=text("NOT is_deleted"),
                ),
                (
                    Index(
                        "ix_procedural_memory_retrieval_tsv",
                        "retrieval_tsv",
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # SQLite indexes
                (
                    Index("ix_procedural_memory_organization_id_sqlite", "organization_id")
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
        ]
        if item is not None
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        """
        Relationship to the Agent that owns this procedural memory item.
        """
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        Adjust 'back_populates' to match the collection name in your `Organization` model.
        """
        return relationship("Organization", back_populates="procedural_memory", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this procedural memory item.
        """
        return relationship("User", lazy="selectin")
