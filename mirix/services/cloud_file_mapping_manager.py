import uuid

from sqlalchemy import select

from mirix.orm.cloud_file_mapping import CloudFileMapping
from mirix.schemas.cloud_file_mapping import CloudFileMapping as PydanticCloudFileMapping


class CloudFileMappingManager:
    """Manage mapping of cloud files to local files."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    async def add_mapping(self, cloud_file_id, local_file_id, timestamp, force_add=False):
        """Add a mapping from a cloud file to a local file."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            import uuid as _uuid

            from mirix.services.organization_manager import OrganizationManager

            mapping_data = {
                "id": str(_uuid.uuid4()),
                "cloud_file_id": cloud_file_id,
                "local_file_id": local_file_id,
                "status": "uploaded",
                "timestamp": timestamp,
                "organization_id": OrganizationManager.DEFAULT_ORG_ID,
            }
            result = await provider.create("cloud_file_mapping", mapping_data)
            return PydanticCloudFileMapping(**result)
        async with self.session_maker() as session:
            try:
                existing = await CloudFileMapping.read(db_session=session, cloud_file_id=cloud_file_id)
            except Exception:
                existing = None
            if existing:
                if force_add:
                    await existing.hard_delete(session)
                else:
                    raise ValueError(f"Mapping already exists for cloud file {cloud_file_id}")

            try:
                existing = await CloudFileMapping.read(db_session=session, local_file_id=local_file_id)
            except Exception:
                existing = None
            if existing:
                if force_add:
                    await existing.hard_delete(session)
                else:
                    raise ValueError(f"Mapping already exists for local file {local_file_id}")

            pydantic_mapping_dict = {
                "cloud_file_id": cloud_file_id,
                "local_file_id": local_file_id,
                "status": "uploaded",
                "timestamp": timestamp,
                "id": str(uuid.uuid4()),
            }
            from mirix.services.organization_manager import OrganizationManager

            pydantic_mapping_dict["organization_id"] = OrganizationManager.DEFAULT_ORG_ID

            mapping = CloudFileMapping(**pydantic_mapping_dict)
            await mapping.create(session)
            return mapping.to_pydantic()

    async def get_local_file(self, cloud_file_id):
        """Get the local file ID for a cloud file."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            results = await provider.find_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.list_by_cloud_id",
                params={"cloudFileId": cloud_file_id},
                page_size=1,
            )
            if results:
                return results[0].get("local_file_id")
            return None
        async with self.session_maker() as session:
            try:
                mapping = await CloudFileMapping.read(db_session=session, cloud_file_id=cloud_file_id)
                return mapping.local_file_id if mapping else None
            except Exception:
                return None

    async def get_cloud_file(self, local_file_id):
        """Get the cloud file ID for a local file."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            results = await provider.find_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.list_by_local_id",
                params={"localFileId": local_file_id},
                page_size=1,
            )
            if results:
                return results[0].get("cloud_file_id")
            return None
        async with self.session_maker() as session:
            try:
                mapping = await CloudFileMapping.read(db_session=session, local_file_id=local_file_id)
                return mapping.cloud_file_id if mapping else None
            except Exception:
                return None

    async def delete_mapping(self, cloud_file_id=None, local_file_id=None) -> None:
        """Delete a mapping."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            if cloud_file_id is not None:
                results = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_cloud_id",
                    params={"cloudFileId": cloud_file_id},
                    page_size=1,
                )
                if results:
                    await provider.hard_delete("cloud_file_mapping", results[0]["id"])
            if local_file_id is not None:
                results = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_local_id",
                    params={"localFileId": local_file_id},
                    page_size=1,
                )
                if results:
                    await provider.hard_delete("cloud_file_mapping", results[0]["id"])
            return
        async with self.session_maker() as session:
            if cloud_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(db_session=session, cloud_file_id=cloud_file_id)
                    await mapping.hard_delete(session)
                except Exception:
                    pass
            if local_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(db_session=session, local_file_id=local_file_id)
                    await mapping.hard_delete(session)
                except Exception:
                    pass

    async def check_if_existing(self, cloud_file_id=None, local_file_id=None) -> bool:
        """Check if the file_ids exist in the database."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider is not None:
            if cloud_file_id is not None:
                rows = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_cloud_id",
                    params={"cloudFileId": cloud_file_id},
                    page_size=1,
                )
                return bool(rows)
            elif local_file_id is not None:
                rows = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_local_id",
                    params={"localFileId": local_file_id},
                    page_size=1,
                )
                return bool(rows)
            return False

        async with self.session_maker() as session:
            if cloud_file_id is not None:
                try:
                    await CloudFileMapping.read(db_session=session, cloud_file_id=cloud_file_id)
                    return True
                except Exception:
                    pass
            elif local_file_id is not None:
                try:
                    await CloudFileMapping.read(db_session=session, local_file_id=local_file_id)
                    return True
                except Exception:
                    pass
        return False

    async def set_processed(self, cloud_file_id=None, local_file_id=None) -> PydanticCloudFileMapping:
        """Set status to processed."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider is not None:
            # Step 1: resolve the provider entity ID via a lookup NQ
            rows = None
            if cloud_file_id is not None:
                rows = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_cloud_id",
                    params={"cloudFileId": cloud_file_id},
                    page_size=1,
                )
            elif local_file_id is not None:
                rows = await provider.find_using_named_query(
                    "cloud_file_mapping",
                    "cloud_file_mapping_manager.list_by_local_id",
                    params={"localFileId": local_file_id},
                    page_size=1,
                )
            if not rows:
                raise ValueError("File Not Found")
            mapping_id = rows[0].get("id")
            if not mapping_id:
                raise ValueError("File Not Found")
            # Step 2: apply the status mutation
            await provider.mutate_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.set_status_by_id",
                params={"id": mapping_id, "status": "processed"},
            )
            # Step 3: re-read and return the updated row
            updated = await provider.read("cloud_file_mapping", mapping_id)
            if updated is None:
                raise ValueError("File Not Found after update")
            return PydanticCloudFileMapping(**updated)

        async with self.session_maker() as session:
            mapping = None
            if cloud_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(db_session=session, cloud_file_id=cloud_file_id)
                except Exception:
                    pass
            elif local_file_id is not None:
                try:
                    mapping = await CloudFileMapping.read(db_session=session, local_file_id=local_file_id)
                except Exception:
                    pass
            if mapping is None:
                raise ValueError("File Not Found")
            mapping.status = "processed"
            await mapping.update(session)
            return mapping.to_pydantic()

    async def list_files_with_status(self, status):
        """List all files with the given status."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider is not None:
            rows = await provider.find_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.list_files_with_status",
                params={"status": status},
                page_size=1000,
            )
            return [PydanticCloudFileMapping(**r) for r in rows]

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
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider is not None:
            rows = await provider.find_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.list_all_cloud_file_ids",
                page_size=5000,
            )
            return [r.get("cloud_file_id") for r in rows if r.get("cloud_file_id")]

        async with self.session_maker() as session:
            result = await session.execute(select(CloudFileMapping))
            rows = result.scalars().all()
            return [x.to_pydantic().cloud_file_id for x in rows]

    async def list_all_local_file_ids(self):
        """List all local file IDs."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider is not None:
            rows = await provider.find_using_named_query(
                "cloud_file_mapping",
                "cloud_file_mapping_manager.list_all_local_file_ids",
                page_size=5000,
            )
            return [r.get("local_file_id") for r in rows if r.get("local_file_id")]

        async with self.session_maker() as session:
            result = await session.execute(select(CloudFileMapping))
            rows = result.scalars().all()
            return [x.to_pydantic().local_file_id for x in rows]
