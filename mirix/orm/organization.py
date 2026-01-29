from typing import TYPE_CHECKING, List

from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.block import Block
    from mirix.orm.client import Client
    from mirix.orm.cloud_file_mapping import CloudFileMapping
    from mirix.orm.episodic_memory import EpisodicEvent
    from mirix.orm.file import FileMetadata
    from mirix.orm.knowledge_vault import KnowledgeVaultItem
    from mirix.orm.message import Message
    from mirix.orm.procedural_memory import ProceduralMemoryItem
    from mirix.orm.provider import Provider
    from mirix.orm.resource_memory import ResourceMemoryItem
    from mirix.orm.semantic_memory import SemanticMemoryItem
    from mirix.orm.tool import Tool
    from mirix.orm.user import User


class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""

    __tablename__ = "organizations"
    __pydantic_model__ = PydanticOrganization

    name: Mapped[str] = mapped_column(doc="The display name of the organization.")

    # relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    clients: Mapped[List["Client"]] = relationship(
        "Client", back_populates="organization", cascade="all, delete-orphan"
    )
    tools: Mapped[List["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    blocks: Mapped[List["Block"]] = relationship("Block", back_populates="organization", cascade="all, delete-orphan")

    # relationships
    agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="organization", cascade="all, delete-orphan"
    )
    providers: Mapped[List["Provider"]] = relationship(
        "Provider", back_populates="organization", cascade="all, delete-orphan"
    )

    # Add knowledge vault relationship
    knowledge_vault: Mapped[List["KnowledgeVaultItem"]] = relationship(
        "KnowledgeVaultItem",
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    episodic_memory: Mapped[List["EpisodicEvent"]] = relationship(
        "EpisodicEvent", back_populates="organization", cascade="all, delete-orphan"
    )

    procedural_memory: Mapped[List["ProceduralMemoryItem"]] = relationship(
        "ProceduralMemoryItem",
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    resource_memory: Mapped[List["ResourceMemoryItem"]] = relationship(
        "ResourceMemoryItem",
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    semantic_memory: Mapped[List["SemanticMemoryItem"]] = relationship(
        "SemanticMemoryItem",
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    # Add files relationship
    files: Mapped[List["FileMetadata"]] = relationship(
        "FileMetadata",
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    cloud_file_mappings: Mapped[List["CloudFileMapping"]] = relationship(
        "CloudFileMapping",
        back_populates="organization",
        cascade="all, delete-orphan",
    )
