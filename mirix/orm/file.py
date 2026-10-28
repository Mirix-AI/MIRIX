from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.file import FileMetadata as PydanticFileMetadata

if TYPE_CHECKING:
    from mirix.orm.organization import Organization


class FileMetadata(SqlalchemyBase, OrganizationMixin):
    """Represents metadata for an uploaded file."""

    __tablename__ = "files"
    __pydantic_model__ = PydanticFileMetadata

    user_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        doc="The user whose data this file belongs to (e.g. an image stored from "
        "their conversation). NULL for files not attributable to a user. "
        "SET NULL rather than CASCADE: a cascaded row delete would orphan the "
        "file on disk with no record left to find it; user-data cleanup "
        "deletes the disk file and the row explicitly.",
    )
    source_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="The unique identifier of the source associated with the file.",
    )
    file_name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The name of the file.")
    file_path: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The file path on the local filesystem."
    )
    source_url: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="The URL of the remote file (for files not stored locally).",
    )
    google_cloud_url: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="The Google Cloud URI for files stored in Google Cloud (e.g., Google Gemini files).",
    )
    file_type: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The type of the file.")
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="The size of the file in bytes.")
    file_creation_date: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The creation date of the file."
    )
    file_last_modified_date: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The last modified date of the file."
    )

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="files", lazy="selectin")

    # "All images of this user" — orphan cleanup on user-data deletion.
    __table_args__ = (Index("ix_files_user_id", "user_id"),)
