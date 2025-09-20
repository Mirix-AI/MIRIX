from datetime import datetime

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase
from mirix.utils import get_utc_time

class CloudFileMappingBase(MirixBase):
    __id_prefix__ = 'cloud_map'


class CloudFileMapping(CloudFileMappingBase):
    """
    Schema for the mapping between cloud file and the local file
    """
    id: str = CloudFileMappingBase.generate_id_field()
    cloud_file_id: str = Field(..., description="The ID of the cloud file")
    local_file_id: str = Field(..., description="The ID of the local file")
    status: str = Field(..., description='whether it has been processed into the memory system.')
    # TODO: change timestamp from str to datetime to accommodate timezone changes.
    timestamp: str = Field(..., description='timestamp of the screenshot')
    created_at: datetime = Field(
        default_factory=get_utc_time,
        description="Timestamp when this memory record was created"
    )