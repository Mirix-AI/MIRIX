from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.user import User as PydanticUser

if TYPE_CHECKING:
    from mirix.orm import Organization


class User(SqlalchemyBase, OrganizationMixin):
    """User ORM class - users are organization-scoped"""

    __tablename__ = "users"
    __pydantic_model__ = PydanticUser

    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the user.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the user is active or not.")
    timezone: Mapped[str] = mapped_column(nullable=False, doc="The timezone of the user.")
    is_admin: Mapped[bool] = mapped_column(nullable=False, default=False, doc="Whether this is an admin user.")
    # Per-user monotonically-increasing counters used by the
    # /memory/add fallback to fill in source_meta.turn_id and
    # source_meta.chunk_id when the client does not provide them. Bumped
    # atomically at /memory/add time. Used by the conflict-resolution
    # path in `SemanticMemoryManager.insert_semantic_item` and by the
    # general source-provenance mechanism documented in
    # docs/mab_conflict_resolution_and_provenance.md.
    turn_counter: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        doc="Next turn_id to hand out for this user's next /memory/add request.",
    )
    chunk_counter: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        doc="Next chunk_id to hand out for this user's next /memory/add request.",
    )

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    # TODO: Add this back later potentially
    # tokens: Mapped[List["Token"]] = relationship("Token", back_populates="user", doc="the tokens associated with this user.")
