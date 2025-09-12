from typing import TYPE_CHECKING
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.orm.mixins import OrganizationMixin

from sqlalchemy import Column, DateTime, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship

from mirix.schemas.cloud_file_mapping import CloudFileMapping as PydanticCloudFileMapping

if TYPE_CHECKING:
    from mirix.orm.organization import Organization

class CloudFileMapping(SqlalchemyBase, OrganizationMixin):
    """
    Represents a mapping between a cloud file and its metadata.
    """

    __tablename__ = "cloud_file_mapping"
    __pydantic_model__ = PydanticCloudFileMapping

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for the cloud file mapping",
    )

    # The ID of the cloud file
    cloud_file_id: Mapped[str] = mapped_column(
        String,
        doc="ID of the cloud file"
    )

    local_file_id: Mapped[str] = mapped_column(
        String,
        doc="ID of the local file"
    )
    
    status: Mapped[str] = mapped_column(
        String,
        doc="whether it has been processed by our model"
    )

    timestamp: Mapped[str] = mapped_column(
        String,
        doc="timestamp of the screenshot"
    )

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="cloud_file_mappings", lazy="selectin")