from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Integer, String
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

    source_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The unique identifier of the source associated with the file.")
    file_name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The name of the file.")
    file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The file path on the local filesystem.")
    source_url: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The URL of the remote file (for files not stored locally).")
    google_cloud_url: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The Google Cloud URI for files stored in Google Cloud (e.g., Google Gemini files).")
    file_type: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The type of the file.")
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="The size of the file in bytes.")
    file_creation_date: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The creation date of the file.")
    file_last_modified_date: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The last modified date of the file.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="files", lazy="selectin")