import datetime as dt
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select

from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY
from mirix.embeddings import embedding_model
from mirix.orm.raw_memory import RawMemoryItem
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types


class RawMemoryManager:
    """
    Manager class to handle business logic related to RawMemory items.

    This manager handles CRUD operations for raw memory items, which store
    screenshot metadata and OCR-extracted information as the foundational
    layer for all other memory types.
    """

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def insert_raw_memory(
        self,
        actor: PydanticUser,
        screenshot_path: str,
        source_app: str,
        captured_at: datetime,
        ocr_text: Optional[str] = None,
        source_url: Optional[str] = None,
        google_cloud_url: Optional[str] = None,
        metadata: Optional[dict] = None,
        organization_id: Optional[str] = None,
    ) -> RawMemoryItem:
        """
        Insert a new raw memory item into the database.

        Args:
            actor: The user creating this raw memory
            screenshot_path: Local filesystem path to the screenshot
            source_app: Application name where screenshot was captured
            captured_at: Timestamp when screenshot was captured
            ocr_text: Optional OCR-extracted text from the screenshot
            source_url: Optional URL extracted from OCR (supports formats like google.com)
            google_cloud_url: Optional Google Cloud Storage URI for multimodal processing
            metadata: Optional additional metadata dictionary
            organization_id: Optional organization ID (defaults to actor's organization)

        Returns:
            The created RawMemoryItem instance
        """
        with self.session_maker() as session:
            # Generate embedding for OCR text if available and embeddings are enabled
            ocr_text_embedding = None
            embedding_config = None

            if ocr_text and BUILD_EMBEDDINGS_FOR_MEMORY:
                ocr_text_embedding = embedding_model.get_embedding(ocr_text)
                embedding_config = {
                    "model": embedding_model.model_name,
                    "embedding_dims": len(ocr_text_embedding) if ocr_text_embedding else 0,
                }

            # Create the raw memory item
            raw_memory = RawMemoryItem(
                screenshot_path=screenshot_path,
                source_app=source_app,
                captured_at=captured_at,
                ocr_text=ocr_text,
                source_url=source_url,
                google_cloud_url=google_cloud_url,
                metadata_=metadata or {},
                ocr_text_embedding=ocr_text_embedding,
                embedding_config=embedding_config,
                processed=False,
                processing_count=0,
                user_id=actor.id,
                organization_id=organization_id or actor.organization_id,
            )

            session.add(raw_memory)
            session.commit()
            session.refresh(raw_memory)

            return raw_memory

    @enforce_types
    def get_raw_memory_by_id(
        self,
        raw_memory_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Optional[RawMemoryItem]:
        """
        Retrieve a raw memory item by its ID.

        Args:
            raw_memory_id: The ID of the raw memory item to retrieve
            user_id: Optional user ID filter
            organization_id: Optional organization ID filter

        Returns:
            The RawMemoryItem if found, None otherwise
        """
        with self.session_maker() as session:
            query = select(RawMemoryItem).where(RawMemoryItem.id == raw_memory_id)

            if user_id:
                query = query.where(RawMemoryItem.user_id == user_id)

            if organization_id:
                query = query.where(RawMemoryItem.organization_id == organization_id)

            result = session.execute(query)
            return result.scalar_one_or_none()

    @enforce_types
    def mark_as_processed(self, raw_memory_id: str) -> bool:
        """
        Mark a raw memory item as processed by memory agents.

        Args:
            raw_memory_id: The ID of the raw memory item to mark

        Returns:
            True if the item was found and updated, False otherwise
        """
        with self.session_maker() as session:
            raw_memory = session.get(RawMemoryItem, raw_memory_id)

            if raw_memory:
                raw_memory.processed = True
                raw_memory.processing_count += 1
                raw_memory.last_modify = {
                    "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                    "operation": "marked_processed",
                }
                session.commit()
                return True

            return False

    @enforce_types
    def get_unprocessed_raw_memories(
        self,
        user_id: str,
        organization_id: str,
        limit: int = 100,
    ) -> List[RawMemoryItem]:
        """
        Get unprocessed raw memory items for a user.

        Args:
            user_id: The user ID to filter by
            organization_id: The organization ID to filter by
            limit: Maximum number of items to return (default: 100)

        Returns:
            List of unprocessed RawMemoryItem instances
        """
        with self.session_maker() as session:
            query = (
                select(RawMemoryItem)
                .where(RawMemoryItem.user_id == user_id)
                .where(RawMemoryItem.organization_id == organization_id)
                .where(RawMemoryItem.processed == False)
                .order_by(RawMemoryItem.captured_at.desc())
                .limit(limit)
            )

            result = session.execute(query)
            return list(result.scalars().all())

    @enforce_types
    def get_raw_memories_by_source_app(
        self,
        user_id: str,
        organization_id: str,
        source_app: str,
        limit: int = 100,
    ) -> List[RawMemoryItem]:
        """
        Get raw memory items filtered by source application.

        Args:
            user_id: The user ID to filter by
            organization_id: The organization ID to filter by
            source_app: The source application name to filter by
            limit: Maximum number of items to return (default: 100)

        Returns:
            List of RawMemoryItem instances from the specified app
        """
        with self.session_maker() as session:
            query = (
                select(RawMemoryItem)
                .where(RawMemoryItem.user_id == user_id)
                .where(RawMemoryItem.organization_id == organization_id)
                .where(RawMemoryItem.source_app == source_app)
                .order_by(RawMemoryItem.captured_at.desc())
                .limit(limit)
            )

            result = session.execute(query)
            return list(result.scalars().all())

    @enforce_types
    def get_raw_memories_by_ids(
        self,
        raw_memory_ids: List[str],
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[RawMemoryItem]:
        """
        Get multiple raw memory items by their IDs.

        Args:
            raw_memory_ids: List of raw memory IDs to retrieve
            user_id: Optional user ID filter
            organization_id: Optional organization ID filter

        Returns:
            List of RawMemoryItem instances
        """
        if not raw_memory_ids:
            return []

        with self.session_maker() as session:
            query = select(RawMemoryItem).where(RawMemoryItem.id.in_(raw_memory_ids))

            if user_id:
                query = query.where(RawMemoryItem.user_id == user_id)

            if organization_id:
                query = query.where(RawMemoryItem.organization_id == organization_id)

            result = session.execute(query)
            return list(result.scalars().all())

    @enforce_types
    def delete_raw_memory(self, raw_memory_id: str) -> bool:
        """
        Delete a raw memory item by its ID.

        Args:
            raw_memory_id: The ID of the raw memory item to delete

        Returns:
            True if the item was found and deleted, False otherwise
        """
        with self.session_maker() as session:
            raw_memory = session.get(RawMemoryItem, raw_memory_id)

            if raw_memory:
                session.delete(raw_memory)
                session.commit()
                return True

            return False

    @enforce_types
    def update_raw_memory(
        self,
        raw_memory_id: str,
        ocr_text: Optional[str] = None,
        source_url: Optional[str] = None,
        google_cloud_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[RawMemoryItem]:
        """
        Update an existing raw memory item.

        Args:
            raw_memory_id: The ID of the raw memory item to update
            ocr_text: Optional new OCR text
            source_url: Optional new source URL
            google_cloud_url: Optional new Google Cloud URL
            metadata: Optional metadata to merge with existing

        Returns:
            The updated RawMemoryItem if found, None otherwise
        """
        with self.session_maker() as session:
            raw_memory = session.get(RawMemoryItem, raw_memory_id)

            if not raw_memory:
                return None

            # Update fields if provided
            if ocr_text is not None:
                raw_memory.ocr_text = ocr_text

                # Regenerate embedding if text changed and embeddings are enabled
                if BUILD_EMBEDDINGS_FOR_MEMORY:
                    raw_memory.ocr_text_embedding = embedding_model.get_embedding(ocr_text)
                    raw_memory.embedding_config = {
                        "model": embedding_model.model_name,
                        "embedding_dims": len(raw_memory.ocr_text_embedding) if raw_memory.ocr_text_embedding else 0,
                    }

            if source_url is not None:
                raw_memory.source_url = source_url

            if google_cloud_url is not None:
                raw_memory.google_cloud_url = google_cloud_url

            if metadata is not None:
                # Merge with existing metadata
                raw_memory.metadata_ = {**raw_memory.metadata_, **metadata}

            # Update last_modify timestamp
            raw_memory.last_modify = {
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "operation": "updated",
            }

            session.commit()
            session.refresh(raw_memory)

            return raw_memory
