import uuid

from sqlalchemy import select

from mirix.orm.cloud_file_mapping import CloudFileMapping
from mirix.schemas.cloud_file_mapping import CloudFileMapping as PydanticCloudFileMapping


class CloudFileMappingManager:
    """Manage mapping of cloud files to local files."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    async def add_mapping(
        self, cloud_file_id, local_file_id, timestamp, force_add=False
    ):
        """Add a mapping from a cloud file to a local file."""
        async with self.session_maker() as session:
            try:
                existing = await CloudFileMapping.read(
                    db_session=session, cloud_file_id=cloud_file_id
                )
            except Exception:
                existing = None
            if existing:
                if force_add:
                    await existing.hard_delete(session)
                else:
                    raise ValueError(
                        f"Mapping already exists for cloud file {cloud_file_id}"
                    )

            try:
                existing = await CloudFileMapping.read(
                    db_session=session, local_file_id=local_file_id
                )
            except Exception:
                existing = None
            if existing:
                if force_add:
                    await existing.hard_delete(session)
                else:
                    raise ValueError(
                        f"Mapping already exists for local file {local_file_id}"
                    )

            pydantic_mapping_dict = {
                "cloud_file_id": cloud_file_id,
                "local_file_id": local_file_id,
                "status": "uploaded",
                "timestamp": timestamp,
                "id": str(uuid.uuid4()),
            }
            from mirix.services.organization_manager import OrganizationManager

            pydantic_mapping_dict["organization_id"] = (
                OrganizationManager.DEFAULT_ORG_ID
            )

            mapping = CloudFileMapping(**pydantic_mapping_dict)
            await mapping.create(session)
            return mapping.to_pydantic()

    async def get_local_file(self, cloud_file_id):
        """Get the local file ID for a cloud file."""
        async with self.session_maker() as session:
            try:
                mapping = await CloudFileMapping.read(
                    db_session=session, cloud_file_id=cloud_file_id
                )
                return mapping.local_file_id if mapping else None
            except Exception:
                return None

    async def get_cloud_file(self, local_file_id):
        """Get the cloud file ID for a local file."""
        async with self.session_maker() as session:
            try:
                mapping = await CloudFileMapping.read(
                    db_session=session, local_file_id=local_file_id
                )
                return mapping.cloud_file_id if mapping else None
            except Exception:
                return None

    async def delete_mapping(
        self, cloud_file_id=None, local_file_id=None
    ) -> None:
        """Delete a mapping."""
        async with self.session_maker() as session:
            if cloud_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(
                        db_session=session, cloud_file_id=cloud_file_id
                    )
                    await mapping.hard_delete(session)
                except Exception:
                    pass
            if local_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(
                        db_session=session, local_file_id=local_file_id
                    )
                    await mapping.hard_delete(session)
                except Exception:
                    pass

    async def check_if_existing(
        self, cloud_file_id=None, local_file_id=None
    ) -> bool:
        """Check if the file_ids exist in the database."""
        async with self.session_maker() as session:
            if cloud_file_id is not None:
                try:
                    await CloudFileMapping.read(
                        db_session=session, cloud_file_id=cloud_file_id
                    )
                    return True
                except Exception:
                    pass
            elif local_file_id is not None:
                try:
                    await CloudFileMapping.read(
                        db_session=session, local_file_id=local_file_id
                    )
                    return True
                except Exception:
                    pass
        return False

    async def set_processed(
        self, cloud_file_id=None, local_file_id=None
    ) -> PydanticCloudFileMapping:
        """Set status to processed."""
        async with self.session_maker() as session:
            mapping = None
            if cloud_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(
                        db_session=session, cloud_file_id=cloud_file_id
                    )
                except Exception:
                    pass
            elif local_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(
                        db_session=session, local_file_id=local_file_id
                    )
                except Exception:
                    pass
            if mapping is None:
                raise ValueError("File Not Found")
            mapping.status = "processed"
            await mapping.update(session)
            return mapping.to_pydantic()

    async def list_files_with_status(self, status):
        """List all files with the given status."""
        async with self.session_maker() as session:
            stmt = (
                select(CloudFileMapping)
                .where(CloudFileMapping.status == status)
                .order_by(CloudFileMapping.timestamp.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [x.to_pydantic() for x in rows]

    async def list_all_cloud_file_ids(self):
        """List all cloud file IDs."""
        async with self.session_maker() as session:
            result = await session.execute(select(CloudFileMapping))
            rows = result.scalars().all()
            return [x.to_pydantic().cloud_file_id for x in rows]

    async def list_all_local_file_ids(self):
        """List all local file IDs."""
        async with self.session_maker() as session:
            result = await session.execute(select(CloudFileMapping))
            rows = result.scalars().all()
            return [x.to_pydantic().local_file_id for x in rows]
