from typing import TYPE_CHECKING, List, Optional, Type

from sqlalchemy import JSON, BigInteger, Index, Integer, UniqueConstraint, event, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import (
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
)

from mirix.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.block import Human, Persona
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm import Organization
    from mirix.orm.user import User


class Block(OrganizationMixin, UserMixin, SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __tablename__ = "block"
    __pydantic_model__ = PydanticBlock
    __table_args__ = tuple(
        filter(
            None,
            [
                UniqueConstraint("id", "label", name="unique_block_id_label"),
                Index("idx_block_id_label", "id", "label", unique=True),
                # GIN index on filter_tags for JSONB containment queries
                (
                    Index(
                        "ix_block_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Btree index on (organization_id, filter_tags->>'scope') for scope queries
                (
                    Index(
                        "ix_block_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    label: Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'")
    value: Mapped[str] = mapped_column(doc="Text content of the block for the respective section of core memory.")
    limit: Mapped[BigInteger] = mapped_column(
        Integer,
        default=CORE_MEMORY_BLOCK_CHAR_LIMIT,
        doc="Character limit of the block.",
    )

    # Filter tags for scope-based access control (mirrors episodic/procedural/resource/semantic/knowledge_vault)
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags including 'scope' for access control"
    )

    # relationships
    organization: Mapped[Optional["Organization"]] = relationship("Organization")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this block.
        """
        return relationship("User", lazy="selectin")

    @classmethod
    async def list_by_scopes(
        cls,
        db_session: AsyncSession,
        user_id: Optional[str],
        organization_id: str,
        scopes: List[str],
        label: Optional[str] = None,
        id: Optional[str] = None,
        limit: int = 50,
        filter_tags: Optional[dict] = None,
    ) -> List["Block"]:
        """
        Query blocks filtered by scope at the SQL level (async).

        Uses filter_tags->>'scope' IN (...) which hits the btree index
        ix_block_org_filter_scope on PostgreSQL.

        Args:
            db_session: SQLAlchemy async session
            user_id: Owner user ID, or None for org-wide (all users in organization).
            organization_id: Organization ID
            scopes: List of scope values to match (block.filter_tags.scope IN scopes)
            label: Optional label filter
            id: Optional block ID filter
            limit: Max results
            filter_tags: Optional dict; when provided, only blocks whose filter_tags
                         contain these keys/values are returned (via apply_filter_tags_sqlalchemy).
        """
        session = db_session
        scope_conditions = [cls.filter_tags["scope"].as_string() == s for s in scopes]
        conditions = [
            cls.organization_id == organization_id,
            or_(*scope_conditions),
        ]
        if user_id is not None:
            conditions.append(cls.user_id == user_id)
        if hasattr(cls, "is_deleted"):
            conditions.append(~cls.is_deleted)
        query = select(cls).where(*conditions).limit(limit)
        if label:
            query = query.where(cls.label == label)
        if id:
            query = query.where(cls.id == id)
        if filter_tags:
            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            query = apply_filter_tags_sqlalchemy(query, cls, filter_tags, scopes=None)
        result = await session.execute(query)
        return list(result.scalars().all())

    def to_pydantic(self) -> Type:
        if self.label == "human":
            Schema = Human
        elif self.label == "persona":
            Schema = Persona
        else:
            Schema = PydanticBlock
        return Schema.model_validate(self)


@event.listens_for(Block, "before_insert")
@event.listens_for(Block, "before_update")
def validate_value_length(mapper, connection, target):
    """Ensure the value length does not exceed the limit."""
    if target.value and len(target.value) > target.limit:
        raise ValueError(
            f"Value length ({len(target.value)}) exceeds the limit ({target.limit}) for block with label '{target.label}' and id '{target.id}'."
        )
