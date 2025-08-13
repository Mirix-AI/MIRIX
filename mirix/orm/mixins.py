from typing import Optional
from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.base import Base


def is_valid_uuid4(uuid_string: str) -> bool:
    """Check if a string is a valid UUID4."""
    try:
        uuid_obj = UUID(uuid_string)
        return uuid_obj.version == 4
    except ValueError:
        return False


class OrganizationMixin(Base):
    """Mixin for models that belong to an organization."""

    __abstract__ = True

    organization_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("organizations.id"), nullable=True)


class UserMixin(Base):
    """Mixin for models that belong to a user."""

    __abstract__ = True

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))


class AgentMixin(Base):
    """Mixin for models that belong to an agent."""

    __abstract__ = True

    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id", ondelete="CASCADE"))


class SandboxConfigMixin(Base):
    """Mixin for models that belong to a SandboxConfig."""

    __abstract__ = True

    sandbox_config_id: Mapped[str] = mapped_column(String, ForeignKey("sandbox_configs.id"))
