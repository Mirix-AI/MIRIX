import datetime as dt
import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func, select

from mirix.orm.file import FileMetadata as FileMetadataModel
from mirix.schemas.file import FileMetadata as PydanticFileMetadata
from mirix.utils import enforce_types


class FileManager:
    """Manager class to handle business logic related to file metadata."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create_file_metadata(
        self, pydantic_file: PydanticFileMetadata
    ) -> PydanticFileMetadata:
        """Create new file metadata."""
        async with self.session_maker() as session:
            file_metadata = FileMetadataModel(**pydantic_file.model_dump())
            await file_metadata.create(session)
            return file_metadata.to_pydantic()

    @enforce_types
    async def get_file_metadata_by_id(self, file_id: str) -> PydanticFileMetadata:
        """Get file metadata by ID."""
        async with self.session_maker() as session:
            file_metadata = await FileMetadataModel.read(
                db_session=session, identifier=file_id
            )
            return file_metadata.to_pydantic()

    @enforce_types
    async def get_files_by_organization_id(
        self,
        organization_id: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticFileMetadata]:
        """Get all files for a specific organization."""
        async with self.session_maker() as session:
            results = await FileMetadataModel.list(
                db_session=session,
                organization_id=organization_id,
                cursor=cursor,
                limit=limit,
            )
            return [f.to_pydantic() for f in results]

    @enforce_types
    async def update_file_metadata(
        self, file_id: str, **kwargs
    ) -> PydanticFileMetadata:
        """Update file metadata."""
        async with self.session_maker() as session:
            file_metadata = await FileMetadataModel.read(
                db_session=session, identifier=file_id
            )
            for key, value in kwargs.items():
                if hasattr(file_metadata, key) and value is not None:
                    setattr(file_metadata, key, value)
            file_metadata.updated_at = datetime.now(dt.UTC)
            await file_metadata.update(session)
            return file_metadata.to_pydantic()

    @enforce_types
    async def delete_file_metadata(self, file_id: str) -> None:
        """Delete file metadata by ID."""
        async with self.session_maker() as session:
            file_metadata = await FileMetadataModel.read(
                db_session=session, identifier=file_id
            )
            await file_metadata.hard_delete(session)

    @enforce_types
    async def list_files(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[PydanticFileMetadata]:
        """List all files with pagination."""
        async with self.session_maker() as session:
            results = await FileMetadataModel.list(
                db_session=session, cursor=cursor, limit=limit
            )
            return [f.to_pydantic() for f in results]

    @enforce_types
    async def create_file_metadata_from_path(
        self, file_path: str, organization_id: str, source_id: Optional[str] = None
    ) -> PydanticFileMetadata:
        """Create file metadata from a file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_stats = os.stat(file_path)
        file_creation_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        file_last_modified_date = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        file_extension = os.path.splitext(file_name)[1].lower()
        file_type_map = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".json": "application/json",
            ".csv": "text/csv",
            ".xml": "application/xml",
            ".html": "text/html",
            ".md": "text/markdown",
        }
        file_type = file_type_map.get(file_extension, "application/octet-stream")

        file_metadata = PydanticFileMetadata(
            organization_id=organization_id,
            source_id=source_id,
            file_name=file_name,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            file_creation_date=file_creation_date,
            file_last_modified_date=file_last_modified_date,
        )
        return await self.create_file_metadata(file_metadata)

    @enforce_types
    async def search_files_by_name(
        self, file_name: str, organization_id: Optional[str] = None
    ) -> List[PydanticFileMetadata]:
        """Search files by name pattern."""
        async with self.session_maker() as session:
            stmt = select(FileMetadataModel).where(
                func.lower(FileMetadataModel.file_name).contains(
                    func.lower(file_name)
                )
            )
            if organization_id:
                stmt = stmt.where(
                    FileMetadataModel.organization_id == organization_id
                )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [f.to_pydantic() for f in rows]

    @enforce_types
    async def get_files_by_type(
        self, file_type: str, organization_id: Optional[str] = None
    ) -> List[PydanticFileMetadata]:
        """Get files by file type."""
        async with self.session_maker() as session:
            stmt = select(FileMetadataModel).where(
                FileMetadataModel.file_type == file_type
            )
            if organization_id:
                stmt = stmt.where(
                    FileMetadataModel.organization_id == organization_id
                )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [f.to_pydantic() for f in rows]

    @enforce_types
    async def check_file_exists(
        self, file_path: str, organization_id: Optional[str] = None
    ) -> bool:
        """Check if a file with the given path exists in the database."""
        async with self.session_maker() as session:
            try:
                stmt = select(FileMetadataModel).where(
                    FileMetadataModel.file_path == file_path
                )
                if organization_id:
                    stmt = stmt.where(
                        FileMetadataModel.organization_id == organization_id
                    )
                result = await session.execute(stmt)
                return result.scalar_one_or_none() is not None
            except Exception:
                return False

    @enforce_types
    async def get_file_stats(
        self, organization_id: Optional[str] = None
    ) -> dict:
        """Get file statistics for an organization or globally."""
        async with self.session_maker() as session:
            stmt = select(
                func.count(FileMetadataModel.id).label("total_files"),
                func.sum(FileMetadataModel.file_size).label("total_size"),
                func.count(func.distinct(FileMetadataModel.file_type)).label(
                    "unique_types"
                ),
            )
            if organization_id:
                stmt = stmt.where(
                    FileMetadataModel.organization_id == organization_id
                )
            result = await session.execute(stmt)
            row = result.one()
            return {
                "total_files": row.total_files or 0,
                "total_size": row.total_size or 0,
                "unique_types": row.unique_types or 0,
            }
