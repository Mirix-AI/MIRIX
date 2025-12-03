import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class RawMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    原始记忆存储 - 存储截图的原始数据和OCR提取的元数据
    作为所有其他记忆类型的基础引用源

    This table serves as the foundational layer for all memory types,
    storing raw screenshot data and extracted metadata (OCR text, URLs, etc.).
    Other memory types reference this table to maintain data provenance and trust.
    """

    __tablename__ = "raw_memory"

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for the raw memory item",
    )

    # Screenshot metadata
    screenshot_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="截图在本地文件系统的路径 / Local filesystem path to the screenshot",
    )

    source_app: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="截图来源的应用名称 / Application name where screenshot was captured",
    )

    captured_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="截图捕获时间 / Timestamp when screenshot was captured",
    )

    # OCR extracted data
    ocr_text: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="OCR提取的完整文本 / Full text extracted by OCR",
    )

    source_url: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="OCR提取的URL / URL extracted from OCR (supports formats like google.com)",
    )

    # Cloud storage reference (for Gemini model)
    google_cloud_url: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Google Cloud Storage URI for multimodal processing",
    )

    # Additional metadata
    metadata_: Mapped[dict] = mapped_column(
        JSON,
        default={},
        nullable=True,
        doc="其他元数据信息 / Additional metadata (batch_id, window_title, etc.)",
    )

    # Processing status
    processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="是否已被记忆agents处理 / Whether this has been processed by memory agents",
    )

    processing_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="被处理的次数 / Number of times this has been processed",
    )

    # Last modification tracking
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

    # Vector embedding field - database type dependent
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        ocr_text_embedding = mapped_column(
            Vector(MAX_EMBEDDING_DIM),
            nullable=True,
            doc="OCR文本的向量嵌入 / Vector embedding of OCR text for semantic search",
        )
    else:
        ocr_text_embedding = Column(
            CommonVector,
            nullable=True,
            doc="OCR文本的向量嵌入 / Vector embedding of OCR text for semantic search",
        )

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to the Organization that owns this raw memory.
        """
        return relationship(
            "Organization", back_populates="raw_memory", lazy="selectin"
        )

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this raw memory.
        """
        return relationship("User", lazy="selectin")
