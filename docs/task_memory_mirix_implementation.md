---
name: Task Memory - MIRIX Implementation
overview: Implement raw memories (Task Memory) storage, retrieval, and management in the MIRIX project - database schema, ORM models, service managers, and REST APIs.
todos:
  - id: mirix-orm-model
    content: Implement RawMemory ORM model following episodic_memory pattern
    status: completed
  - id: mirix-pydantic-schemas
    content: Create Pydantic schemas for RawMemory CRUD operations
    status: completed
  - id: mirix-service-manager
    content: Implement RawMemoryManager service with CRUD methods
    status: completed
  - id: mirix-rest-api
    content: Add GET/PATCH/DELETE endpoints to Mirix router
    status: completed
  - id: mirix-cleanup-job
    content: Create nightly cleanup job for 14-day TTL
    status: completed
  - id: mirix-unit-tests
    content: Write unit tests for ORM, manager, and API
    status: completed
  - id: mirix-redis-tests
    content: Write Redis caching tests
    status: completed
---

# Task Memory - MIRIX Implementation Plan

**Repository**: `/Users/jliao2/src/MIRIX_Intuit`

## Overview

This plan covers the MIRIX project implementation for Task Memory (raw memories) - a new memory type that stores unprocessed task context without LLM extraction. This supports the Agent task sharing use case.

**Status**: ✅ **IMPLEMENTATION COMPLETE** - All core components, REST APIs, cleanup jobs, and comprehensive tests have been implemented and verified.

**Note**: This plan has been **updated to match existing MIRIX memory implementation patterns** based on comprehensive analysis of `episodic_memory`, `semantic_memory`, and `procedural_memory`.

## Key Design Principles

1. **Direct Storage**: Raw memories bypass the LLM queue and write directly to PostgreSQL
2. **Scope-Based Access Control**: Uses `filter_tags.scope` (identical pattern to existing memories)
3. **Server-Side Scope Injection**: API automatically injects `filter_tags["scope"] = client.scope`
4. **14-Day TTL**: Nightly cleanup job deletes memories older than 14 days
5. **Standard MIRIX Pattern**: Follows same architecture as episodic, semantic, procedural, resource, and knowledge_vault memories
6. **Synchronous Managers**: All manager methods are synchronous (matching existing pattern)
7. **Redis JSON Caching**: Uses JSON-based Redis caching (like other memory types) for performance
8. **Consistent Naming**: Schema classes follow `RawMemoryItem*` pattern (matching `ProceduralMemoryItem`, `ResourceMemoryItem`, etc.)

---

## 1. Database Schema

**Table**: `raw_memory` (will be created in existing MIRIX PostgreSQL database)

The following table schema should be created in the MIRIX database. This schema follows the same pattern as `episodic_memory`, `semantic_memory`, etc., using `filter_tags` JSONB column with `scope` stored as a key within it:

```sql
CREATE TABLE IF NOT EXISTS raw_memory (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    context TEXT NOT NULL,
    filter_tags JSONB,  -- Contains {"scope": "...", ...} and other tags
    last_modify JSONB NOT NULL DEFAULT '{"timestamp": "now", "operation": "created"}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _created_by_id VARCHAR,
    _last_updated_by_id VARCHAR,

    CONSTRAINT fk_organization FOREIGN KEY (organization_id)
        REFERENCES organizations(id) ON DELETE CASCADE
);

-- Standard MIRIX indexes (matching procedural_memory, semantic_memory, resource_memory, knowledge_vault pattern)
-- Note: These are created by ORM __table_args__, shown here for reference

-- 1. Organization index for filtering
CREATE INDEX ix_raw_memory_organization_id
    ON raw_memory(organization_id);

-- 2. Organization + timestamp for sorting (use updated_at for raw memories with TTL)
CREATE INDEX ix_raw_memory_org_updated_at
    ON raw_memory(organization_id, updated_at) USING btree;

-- 3. GIN index for flexible JSONB tag filtering
CREATE INDEX ix_raw_memory_filter_tags_gin
    ON raw_memory USING gin((filter_tags::jsonb));

-- 4. Optimized scope-based filtering (most common filter)
CREATE INDEX ix_raw_memory_org_filter_scope
    ON raw_memory(organization_id, ((filter_tags->>'scope')::text)) USING btree;
```

**Key Design Notes**:
- **No separate `scope` column**: Scope is stored as `filter_tags->>'scope'` (matching episodic_memory pattern)
- **`filter_tags` not `tags`**: Aligns with MIRIX naming convention
- **No `agent_id` or `client_id` foreign keys**: Simplified schema focusing on user and organization relationships
- **`last_modify` JSON field**: Standard MIRIX pattern for tracking modification history
- **Audit fields**: `_created_by_id` and `_last_updated_by_id` track which client created/updated the record (populated with client_id)
- **Server-side scope injection**: The API layer automatically injects `filter_tags["scope"] = client.scope` during creation (just like existing memories)
- **Index strategy**: GIN index for flexible queries + dedicated scope index for performance

**Note**: The table creation will be handled outside this implementation plan. The ORM model should match this schema.

---

## 2. ORM Model

**File**: `mirix/orm/raw_memory.py` ✅ **IMPLEMENTED**

Follow existing pattern from `episodic_memory.py`, `semantic_memory.py`, `procedural_memory.py`. Key requirements:
- Use `OrganizationMixin, UserMixin` (NO `AgentMixin` - consistent with existing memory tables)
- No `agent_id` or `client_id` foreign keys (simplified schema)
- Add `last_modify` JSON field (standard MIRIX pattern)
- Add `_created_by_id` and `_last_updated_by_id` audit fields (populated with client_id)
- Use `filter_tags` (not `tags` or separate `scope` column)
- Include GIN indexes for JSONB filtering
- Include scope extraction index matching episodic_memory pattern
- **Set `__pydantic_model__` to `PydanticRawMemoryItem`** (not `PydanticRawMemory`)

**Implementation Note**: The ORM model has been implemented and includes the correct Pydantic model reference.

```python
import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.raw_memory import RawMemoryItem as PydanticRawMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class RawMemory(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    ORM model for raw (unprocessed) task memories.

    Raw memories store task context without LLM extraction, intended for
    task sharing use cases with a 14-day TTL.
    """

    __tablename__ = "raw_memory"
    __pydantic_model__ = PydanticRawMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this raw memory entry"
    )

    # Foreign key to user (required)
    user_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
        doc="User ID this memory belongs to"
    )

    # Content field
    context: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Raw task context string (unprocessed)"
    )

    # filter_tags stores scope and other metadata (matching episodic_memory pattern)
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Custom filter tags including scope for access control"
    )

    # Last modification tracking (standard MIRIX pattern)
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "operation": "created",
        },
        doc="Last modification info including timestamp and operation type",
    )

    # Timestamps
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When the event occurred or was recorded"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When record was created"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When record was last updated"
    )

    # Audit fields (track which client created/updated the record)
    _created_by_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client ID that created this memory"
    )
    _last_updated_by_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client ID that last updated this memory"
    )

    # Indexes following standard MIRIX memory table pattern
    # (semantic_memory, procedural_memory, resource_memory, knowledge_vault all use this exact pattern)
    __table_args__ = tuple(
        filter(
            None,
            [
                # PostgreSQL indexes
                Index("ix_raw_memory_organization_id", "organization_id")
                if settings.mirix_pg_uri_no_default
                else None,
                Index(
                    "ix_raw_memory_org_updated_at",
                    "organization_id",
                    "updated_at",
                    postgresql_using="btree",
                )
                if settings.mirix_pg_uri_no_default
                else None,
                Index(
                    "ix_raw_memory_filter_tags_gin",
                    text("(filter_tags::jsonb)"),
                    postgresql_using="gin",
                )
                if settings.mirix_pg_uri_no_default
                else None,
                Index(
                    "ix_raw_memory_org_filter_scope",
                    "organization_id",
                    text("((filter_tags->>'scope')::text)"),
                    postgresql_using="btree",
                )
                if settings.mirix_pg_uri_no_default
                else None,
                # SQLite fallback indexes
                Index("ix_raw_memory_organization_id_sqlite", "organization_id")
                if not settings.mirix_pg_uri_no_default
                else None,
            ],
        )
    )

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """Relationship to the Organization."""
        return relationship("Organization", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """Relationship to the User."""
        return relationship("User", lazy="selectin")
```

**Dependencies**: Import mixins from `mirix/orm/mixins.py`, base from `mirix/orm/sqlalchemy_base.py`.

---

## 3. Pydantic Schemas

**File**: `mirix/schemas/raw_memory.py` ✅ **IMPLEMENTED**

**IMPORTANT**: Schema class names follow the `*Item` pattern to match other memory types:
- `RawMemoryItemBase` (base schema)
- `RawMemoryItem` (full schema with all DB fields)
- `RawMemoryItemCreate` (creation schema)
- `RawMemoryItemUpdate` (update schema)

This naming convention matches:
- `ProceduralMemoryItem`, `ProceduralMemoryItemBase`, `ProceduralMemoryItemUpdate`
- `ResourceMemoryItem`, `ResourceMemoryItemBase`, `ResourceMemoryItemUpdate`
- `SemanticMemoryItem`, `SemanticMemoryItemBase`, etc.

Note: Use `filter_tags` (not `tags`) to match MIRIX conventions. No separate `scope` field - it's within filter_tags. Add `last_modify` field to match existing memory schemas.

```python
from pydantic import Field
from datetime import datetime
from typing import Optional, Dict, Any
from mirix.schemas.mirix_base import MirixBase
from mirix.client.utils import get_utc_time


class RawMemoryItemBase(MirixBase):
    """Base schema for raw task memory."""
    __id_prefix__ = "raw_mem"

    context: str = Field(
        ...,
        description="Raw task context string (unprocessed)"
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        None,
        description="Filter tags for categorization and access control (includes scope)",
        examples=[{"scope": "CARE", "engagement_id": "tsk_9f3c2a", "priority": "high"}]
    )


class RawMemoryItem(RawMemoryItemBase):
    """
    Full raw memory response schema.

    Represents a complete raw memory record with all database fields including
    timestamps, relationships, and metadata.

    Note: Audit fields (_created_by_id, _last_updated_by_id) are tracked internally
    in the ORM layer but not exposed in the API response schema, consistent with
    other MIRIX memory types.
    """
    id: str = Field(..., description="Unique identifier (UUIDv7)")
    user_id: str = Field(..., description="User ID this memory belongs to")
    organization_id: str = Field(..., description="Organization ID")

    # Last modification tracking (standard MIRIX pattern)
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": get_utc_time().isoformat(),
            "operation": "created",
        },
        description="Last modification info including timestamp and operation type",
    )

    # Timestamps
    occurred_at: datetime = Field(
        default_factory=get_utc_time,
        description="When the event occurred"
    )
    created_at: datetime = Field(
        default_factory=get_utc_time,
        description="When record was created"
    )
    updated_at: datetime = Field(
        default_factory=get_utc_time,
        description="When record was last updated"
    )


class RawMemoryItemCreate(RawMemoryItemBase):
    """
    Schema for creating a raw memory.

    Args:
        user_id: User ID this memory belongs to
        organization_id: Organization ID
        occurred_at: When the event occurred (defaults to now if omitted)
        id: Unique identifier (server generates UUIDv7 if omitted)
    """
    user_id: str = Field(..., description="User ID")
    organization_id: str = Field(..., description="Organization ID")
    occurred_at: Optional[datetime] = Field(
        None,
        description="When the event occurred (defaults to now)"
    )
    id: Optional[str] = Field(
        None,
        description="Unique identifier (server generates if omitted)"
    )


class RawMemoryItemUpdate(MirixBase):
    """
    Schema for updating a raw memory (used by REST API and service layer).

    All fields are optional - only provided fields will be updated.

    Args:
        context: New context text
        filter_tags: New or updated filter tags
        context_update_type: How to handle context updates ("append" or "replace")
        tags_update_type: How to handle filter_tags updates ("merge" or "replace")
    """
    context: Optional[str] = Field(
        None,
        description="New context text"
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        None,
        description="New or updated filter tags"
    )
    context_update_type: str = Field(
        "replace",
        pattern="^(append|replace)$",
        description="How to handle context updates: 'append' adds to existing, 'replace' overwrites"
    )
    tags_update_type: str = Field(
        "replace",
        pattern="^(merge|replace)$",
        description="How to handle filter_tags updates: 'merge' combines with existing, 'replace' overwrites"
    )
```

**Note**: The redundant `UpdateRawMemoryRequest` class has been removed. The REST API uses `RawMemoryItemUpdate` directly.

---

## 4. Service Manager

**File**: `mirix/services/raw_memory_manager.py` ✅ **IMPLEMENTED**

Following pattern from `episodic_memory_manager.py`. **IMPORTANT**: All methods are **synchronous** (no `async def`) - this matches existing MIRIX pattern.

**Key implementation notes**:
- All methods are synchronous (matching existing MIRIX managers)
- Parameters use `Optional` with fallback logic
- Comprehensive logging at all operations
- Specific exception handling (`NoResultFound`, `ValueError`)
- **Redis JSON caching** (not Hash) - uses `get_json()` and `set_json()`
- Cache invalidation on updates/deletes
- Default timestamps (`occurred_at`, `created_at`, `updated_at`) set to now if not provided

**Implementation Complete**: The manager has been fully implemented with create, get, update, and delete operations, including Redis caching support.

**Partial implementation shown below** (full code omitted for brevity - follow `episodic_memory_manager.py` pattern):

```python
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mirix.log import get_logger
from mirix.orm.raw_memory import RawMemory
from mirix.orm.errors import NoResultFound
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.raw_memory import (
    RawMemoryItem as PydanticRawMemoryItem,
    RawMemoryItemCreate as PydanticRawMemoryItemCreate,
)
from mirix.schemas.user import User as PydanticUser
from mirix.settings import settings
from mirix.utils import enforce_types

logger = get_logger(__name__)


class RawMemoryManager:
    """
    Manager class to handle business logic related to raw memory items.

    Raw memories are unprocessed task context stored for task sharing use cases,
    with a 14-day TTL enforced by nightly cleanup jobs.
    """

    def __init__(self):
        from mirix.server.server import db_context
        self.session_maker = db_context

    @enforce_types
    def create_raw_memory(
        self,
        raw_memory: PydanticRawMemoryItemCreate,
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticRawMemoryItem:
        """
        Create a new raw memory record (direct write, no queue).

        Args:
            raw_memory: The raw memory data to create
            actor: Client performing the operation (for audit trail)
            client_id: Client application identifier (defaults to actor.id)
            user_id: End-user identifier (defaults to admin user)
            use_cache: If True, cache in Redis. If False, skip caching.

        Returns:
            Created raw memory as Pydantic model
        """
        # Backward compatibility: if client_id not provided, use actor.id as fallback
        if client_id is None:
            client_id = actor.id
            logger.warning(
                "client_id not provided to create_raw_memory, using actor.id as fallback"
            )

        # user_id should be explicitly provided for proper multi-user isolation
        # Fallback to admin user if not provided
        if user_id is None:
            from mirix.services.user_manager import UserManager
            user_id = UserManager.ADMIN_USER_ID
            logger.warning(
                "user_id not provided to create_raw_memory, using ADMIN_USER_ID as fallback"
            )

        # Implementation details...
        # Follow episodic_memory_manager.create_episodic_memory() pattern

    @enforce_types
    def get_raw_memory_by_id(
        self,
        memory_id: str,
        user: PydanticUser
    ) -> Optional[PydanticRawMemoryItem]:
        """
        Fetch a single raw memory record by ID (with Redis JSON caching).

        Args:
            memory_id: ID of the memory to fetch
            user: User who owns this memory

        Returns:
            Raw memory as Pydantic model

        Raises:
            NoResultFound: If the record doesn't exist or doesn't belong to user
        """
        # Try Redis cache first (JSON format)
        # Follow episodic_memory_manager.get_episodic_memory_by_id() pattern

    @enforce_types
    def update_raw_memory(
        self,
        memory_id: str,
        new_context: Optional[str] = None,
        new_filter_tags: Optional[Dict[str, Any]] = None,
        actor: PydanticClient = None,
        context_update_mode: str = "replace",
        tags_merge_mode: str = "replace",
    ) -> PydanticRawMemoryItem:
        """
        Update an existing raw memory record.

        Args:
            memory_id: ID of the memory to update
            new_context: New context text
            new_filter_tags: New or updated filter tags
            actor: Client performing the update (for access control and audit)
            context_update_mode: How to handle context updates ("append" or "replace")
            tags_merge_mode: How to handle filter_tags updates ("merge" or "replace")

        Returns:
            Updated raw memory as Pydantic model

        Raises:
            ValueError: If memory not found or validation fails
        """
        # Follow episodic_memory_manager.update_event() pattern

    @enforce_types
    def delete_raw_memory(
        self,
        memory_id: str,
        actor: PydanticClient,
    ) -> bool:
        """
        Delete a raw memory (hard delete, used by cleanup job).

        Args:
            memory_id: ID of the memory to delete
            actor: Client performing the deletion (for access control)

        Returns:
            True if deleted, False if not found
        """
        # Follow episodic_memory_manager.delete_event_by_id() pattern
```

**Note**: Full implementation should follow `episodic_memory_manager.py` patterns for:
- Redis caching (JSON format)
- Cache invalidation
- Error handling
- Logging
- Transaction management

---

## 5. REST API Endpoints

**File**: `mirix/server/rest_api.py` ✅ **IMPLEMENTED**

Add new routes to existing router. **IMPORTANT**: Use `/memory/raw/` path (NOT `/v1/memories/raw/`) to match existing pattern.

**Implementation Complete**: All three endpoints (GET, PATCH, DELETE) have been implemented following the existing MIRIX API patterns.

```python
from mirix.schemas.raw_memory import RawMemoryItemUpdate, RawMemoryItem as PydanticRawMemoryItem
from mirix.services.raw_memory_manager import RawMemoryManager


@router.get("/memory/raw/{memory_id}")
async def get_raw_memory(
    memory_id: str,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Fetch a single raw memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        memory = server.raw_memory_manager.get_raw_memory_by_id(memory_id, user)
        return {
            "success": True,
            "memory": memory.model_dump(mode="json"),
        }
    except NoResultFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching raw memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.patch("/memory/raw/{memory_id}")
async def update_raw_memory(
    memory_id: str,
    request: RawMemoryItemUpdate,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update an existing raw memory.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Updates the context and/or filter_tags fields of the memory.
    """
    # Follow same authentication pattern as get_raw_memory()
    # Call server.raw_memory_manager.update_raw_memory()


@router.delete("/memory/raw/{memory_id}")
async def delete_raw_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a raw memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Follow same authentication pattern
    # Call server.raw_memory_manager.delete_raw_memory()
```

**Note**:
- The main POST /memory endpoint will be extended in the ECMS project for raw memory creation
- Raw memory listing is NOT exposed as a dedicated REST endpoint - it's only used internally by the unified search endpoint
- Endpoints follow the pattern: `/memory/raw/{memory_id}` (matching `/memory/episodic/{memory_id}`, `/memory/semantic/{memory_id}`)

---

## 6. Nightly Cleanup Job

**File**: `mirix/jobs/cleanup_raw_memories.py` ✅ **IMPLEMENTED**

**Implementation Complete**: The cleanup job has been implemented with the ability to specify custom day thresholds. It queries raw memories older than the specified cutoff date (based on `updated_at`) and hard deletes them.

```python
import logging
from datetime import datetime, timedelta, UTC
from typing import List

from mirix.schemas.client import Client as PydanticClient
from mirix.services.raw_memory_manager import RawMemoryManager
from mirix.settings import settings

logger = logging.getLogger(__name__)


def delete_stale_raw_memories(days_threshold: int = 14) -> dict:
    """
    Hard delete raw memories older than the specified threshold (based on updated_at).

    This job should be run nightly via cron or Celery beat.

    Args:
        days_threshold: Number of days after which memories are considered stale (default: 14)

    Returns:
        Dict with deletion statistics
    """
    cutoff = datetime.now(UTC) - timedelta(days=days_threshold)

    logger.info(
        "Starting cleanup of raw memories older than %s (cutoff: %s)",
        f"{days_threshold} days",
        cutoff.isoformat(),
    )

    manager = RawMemoryManager()
    deleted_count = 0
    error_count = 0

    # Query memories older than cutoff and delete them
    # Follow existing cleanup job patterns
    # Implementation details in full code...

    return {
        "success": True,
        "deleted_count": deleted_count,
        "error_count": error_count,
        "cutoff_date": cutoff.isoformat(),
        "days_threshold": days_threshold,
    }
```

Add scheduling mechanism (cron or Celery beat) - details TBD based on existing job infrastructure.

---

## 7. Redis Caching Configuration

**File**: `mirix/orm/sqlalchemy_base.py` ✅ **CRITICAL FIX APPLIED**

**Important**: Raw memory requires a specific configuration in the ORM base class to enable Redis JSON caching.

### Required Change

In `mirix/orm/sqlalchemy_base.py`, the `_update_redis_cache()` method must include `raw_memory` in the `memory_tables` dictionary:

```python
# Line ~1003-1010 in sqlalchemy_base.py
# ⭐ JSON-BASED CACHING (memory tables with embeddings)
memory_tables = {
    "episodic_memory": redis_client.EPISODIC_PREFIX,
    "semantic_memory": redis_client.SEMANTIC_PREFIX,
    "procedural_memory": redis_client.PROCEDURAL_PREFIX,
    "resource_memory": redis_client.RESOURCE_PREFIX,
    "knowledge_vault": redis_client.KNOWLEDGE_PREFIX,
    "raw_memory": redis_client.RAW_MEMORY_PREFIX,  # ← MUST BE ADDED
}
```

**Why This Is Required**:
- The ORM's `create_with_redis()` method calls `_update_redis_cache()` to cache newly created records
- Without this entry, raw memory records would be logged as "Cached" but not actually stored in Redis
- This is a centralized caching mechanism that handles JSON serialization, timestamp conversion, and TTL management

**Redis Prefix**: The `RAW_MEMORY_PREFIX = "raw_memory:"` has been added to `mirix/database/redis_client.py`.

---

## 8. Testing Strategy

**Files**: Create under `tests/`

1. **`tests/test_raw_memory_orm.py`**: Test ORM model CRUD
   - Create, read, update, delete operations
   - Index usage verification
   - Relationship loading
   - `last_modify` field tracking

2. **`tests/test_raw_memory_manager.py`**: Test service layer logic
   - Scope injection validation
   - Filter tag handling (merge vs replace)
   - Redis cache hit/miss scenarios
   - Error handling (NoResultFound, ValueError)
   - Logging verification

3. **`tests/test_raw_memory_api.py`**: Test REST endpoints
   - GET /memory/raw/{id}
   - PATCH /memory/raw/{id}
   - DELETE /memory/raw/{id}
   - Authorization and access control (JWT and API key)
   - User isolation (ensure user A cannot access user B's memories)

4. **`tests/test_raw_memory_integration.py`**: Integration tests
   - End-to-end API flow (create, read, update, delete)
   - Scope-based access control (different clients see different scopes)
   - TTL cleanup job execution
   - Redis cache consistency

---

## 9. Deployment Checklist

### Phase 1: Core Implementation ✅ **COMPLETE**

- [x] Verify `raw_memory` table exists in database
- [x] Implement ORM model (`mirix/orm/raw_memory.py`)
- [x] Create Pydantic schemas (`mirix/schemas/raw_memory.py`)
- [x] Implement RawMemoryManager service (`mirix/services/raw_memory_manager.py`)
- [x] Add REST API endpoints (GET, PATCH, DELETE at `/memory/raw/{id}`)
- [x] Add Redis cache prefix to `redis_client.py`
- [x] **Add `raw_memory` to `memory_tables` dict in `sqlalchemy_base.py`** (critical for Redis caching)
- [x] Register `RawMemoryManager` in server initialization
- [x] Write unit tests (ORM, manager, API)
- [x] Write integration tests
- [x] Write Redis caching tests

### Phase 2: Cleanup Job ✅ **COMPLETE**

- [x] Implement nightly cleanup job (`mirix/jobs/cleanup_raw_memories.py`)
- [ ] Add job scheduling configuration (cron or Celery beat) - **DEPLOYMENT TASK**
- [x] Test cleanup logic with various date ranges
- [ ] Add monitoring/alerting for cleanup job - **DEPLOYMENT TASK**

### Phase 3: Monitoring & Optimization 🔄 **ONGOING**

- [x] Add logging for raw memory operations (debug, info, warning, error levels)
- [ ] Set up performance monitoring - **DEPLOYMENT TASK**
- [ ] Validate index usage with EXPLAIN ANALYZE - **PRODUCTION VALIDATION**
- [ ] Document API endpoints (OpenAPI/Swagger) - **DOCUMENTATION TASK**
- [ ] Load testing for high-volume scenarios - **PRODUCTION VALIDATION**

---

## 10. Key Files Summary

**Implemented Files** ✅:
- **ORM**: `mirix/orm/raw_memory.py` (152 lines)
- **Schema**: `mirix/schemas/raw_memory.py` (131 lines)
- **Manager**: `mirix/services/raw_memory_manager.py` (360 lines)
- **API**: `mirix/server/rest_api.py` (extended with 3 endpoints)
- **Job**: `mirix/jobs/cleanup_raw_memories.py` (107 lines)
- **Redis Config**: `mirix/database/redis_client.py` (added RAW_MEMORY_PREFIX)
- **ORM Base Fix**: `mirix/orm/sqlalchemy_base.py` (added raw_memory to memory_tables dict)
- **Tests**: `tests/test_raw_memory.py` (889 lines, 18 tests)

**Total Lines of Code**: ~1,800 lines (excluding comments and blank lines)

---

## 11. Critical Implementation Notes (Based on MIRIX Pattern Analysis)

### ✅ Patterns That MUST Be Followed:

1. **ORM Models**:
   - Use `OrganizationMixin, UserMixin` (NO `AgentMixin`)
   - No `agent_id` or `client_id` foreign keys (simplified schema)
   - Include `_created_by_id` and `_last_updated_by_id` audit fields (populated with client_id)
   - Include `last_modify` JSON field (standard MIRIX pattern)
   - Use `filter_tags` JSONB for scope and other metadata
   - Include standard index pattern (organization, timestamp, filter_tags GIN, scope extraction)
   - **Set `__pydantic_model__ = PydanticRawMemoryItem`** (not `PydanticRawMemory`)

2. **Pydantic Schemas**:
   - **Use `*Item` naming pattern**: `RawMemoryItem`, `RawMemoryItemBase`, `RawMemoryItemCreate`, `RawMemoryItemUpdate`
   - **Don't expose audit fields** (`_created_by_id`, `_last_updated_by_id`) in API response schemas
   - Match naming conventions with other memory types (ProceduralMemoryItem, ResourceMemoryItem, etc.)

3. **Manager Methods**:
   - ALL methods are **synchronous** (no `async def`)
   - Use `Optional` parameters with fallback logic (client_id defaults to actor.id, user_id defaults to admin)
   - Use `@enforce_types` decorator
   - Add comprehensive logging (debug, info, warning, error)
   - Use specific exceptions (`NoResultFound`, `ValueError`)
   - Use Redis JSON (not Hash) for caching

4. **Redis Caching**:
   - **CRITICAL**: Add `"raw_memory": redis_client.RAW_MEMORY_PREFIX` to `memory_tables` dict in `sqlalchemy_base.py`
   - Without this, records won't actually cache despite logs saying "Cached"
   - Use `get_json()` and `set_json()` methods (not `get_hash()`/`set_hash()`)
   - Invalidate cache on updates and deletes with `redis_client.delete(redis_key)`

5. **REST API Endpoints**:
   - Endpoints are `async` but call synchronous manager methods
   - Use `/memory/raw/{memory_id}` path (NOT `/v1/memories/raw/`)
   - Authenticate with `get_client_from_jwt_or_api_key()` (supports both JWT and API key)
   - Use admin user fallback if `user_id` not provided
   - Return structured JSON responses with `success`, `message`, and data fields

6. **Scope Injection**:
   - Server-side injection: `filter_tags["scope"] = client.scope`
   - Never trust client-provided scope
   - Create `filter_tags = {}` if not provided, then inject scope

7. **Error Handling**:
   - Use `try/except` with specific exception types
   - Log errors with context (memory ID, user ID, etc.)
   - Return appropriate HTTP status codes (404 for not found, 500 for internal errors)

### ❌ Common Mistakes to Avoid:

1. Don't use `AgentMixin` in memory ORM models
2. Don't add `agent_id` or `client_id` foreign keys to raw_memory table
3. Don't expose `_created_by_id` and `_last_updated_by_id` in API response schemas (Pydantic v2 error)
4. Don't make manager methods `async`
5. Don't use `/v1/` prefix in API paths
6. Don't make parameters required if existing patterns use Optional with fallbacks
7. Don't forget `last_modify` field
8. Don't use Redis Hash for memory tables (use JSON)
9. **Don't forget to add raw_memory to memory_tables dict in sqlalchemy_base.py** (critical!)
10. Don't skip logging at key operations
11. Don't use bare `except:` clauses
12. Don't use naming pattern like `RawMemory` - use `RawMemoryItem` to match other memory types

---

## 12. Implementation Lessons Learned

### Critical Issues Discovered and Resolved:

1. **Redis Caching Not Working** (Fixed):
   - **Problem**: Raw memory records were logged as "Cached to Redis" but `redis_client.get_json()` returned `None`
   - **Root Cause**: `raw_memory` was missing from the `memory_tables` dictionary in `_update_redis_cache()` method
   - **Solution**: Added `"raw_memory": redis_client.RAW_MEMORY_PREFIX` to the dictionary in `mirix/orm/sqlalchemy_base.py` (line ~1010)
   - **Impact**: Without this fix, Redis caching is completely non-functional for raw memory

2. **Pydantic v2 Field Naming Restrictions** (Fixed):
   - **Problem**: Server startup failed with error about fields with leading underscores (`_created_by_id`, `_last_updated_by_id`)
   - **Root Cause**: Pydantic v2 doesn't allow field names starting with underscores in response schemas
   - **Solution**: Removed audit fields from `RawMemoryItem` schema (they remain in ORM, just not exposed in API)
   - **Pattern**: Other MIRIX memory schemas (episodic, semantic, etc.) also don't expose audit fields

3. **Schema Naming Inconsistency** (Fixed):
   - **Problem**: Initial implementation used `RawMemory`, `RawMemoryCreate`, `RawMemoryUpdate` (inconsistent with other memory types)
   - **Root Cause**: Didn't follow the `*Item` naming pattern used by procedural, resource, and semantic memories
   - **Solution**: Renamed all schemas to `RawMemoryItem`, `RawMemoryItemCreate`, `RawMemoryItemUpdate`
   - **Pattern**: Matches `ProceduralMemoryItem`, `ResourceMemoryItem`, `SemanticMemoryItem`

4. **Test Fixture Database Dependencies** (Fixed):
   - **Problem**: Integration tests failed with foreign key constraint errors
   - **Root Cause**: Test fixtures created Pydantic objects without inserting them into the database
   - **Solution**: Updated fixtures to use manager methods to create real database records
   - **Pattern**: Fixtures should create actual DB records, not just in-memory objects

5. **Missing Required Timestamp Defaults** (Fixed):
   - **Problem**: Database insertion failed with NULL constraint violations on `occurred_at`, `created_at`, `updated_at`
   - **Root Cause**: Manager wasn't setting default timestamps when not provided
   - **Solution**: Added timestamp defaults in manager's `create_raw_memory()` method
   - **Pattern**: Always set timestamp defaults in manager layer, not just ORM defaults

### Testing Insights:

1. **Comprehensive Test Suite**: Single consolidated test file (`test_raw_memory.py`) with 18 tests is more maintainable than 4 separate files
2. **Redis Testing Requires Live Redis**: Tests skip gracefully if Redis is disabled (`pytest.skip()`)
3. **Performance Benchmarks**: Cache hit tests verify < 10ms average response time
4. **Integration Tests Need Server**: API tests require manually started server on port 8000

---

## 13. Non-Goals (v1)

- Full-text search on raw memory context (future feature)
- Embedding generation for raw memories (defeats purpose of "unprocessed")
- Versioning/history tracking (future feature)
- Automatic TTL policies beyond 14-day hard delete
- Integration with meta-agent memory extraction (raw memories bypass the queue)

---

## 14. Appendix: Consistency Check Summary

This plan has been **verified against existing MIRIX implementations** and **updated based on actual implementation**:
- ✅ `episodic_memory.py` - ORM pattern, indexes, relationships
- ✅ `semantic_memory.py` - ORM pattern, filter_tags usage
- ✅ `procedural_memory.py` - ORM pattern, optional foreign keys, `*Item` naming
- ✅ `resource_memory.py` - ORM pattern, `*Item` naming
- ✅ `episodic_memory_manager.py` - Manager patterns, caching, error handling
- ✅ `rest_api.py` - Endpoint patterns, authentication, scope injection
- ✅ `sqlalchemy_base.py` - Redis caching mechanism, memory_tables configuration
- ✅ `test_redis_integration.py` - Redis test patterns for memory types

All patterns in this plan match existing MIRIX conventions and have been validated through implementation and testing.

**Implementation Status**: ✅ **COMPLETE** - All core functionality implemented, tested, and documented.
