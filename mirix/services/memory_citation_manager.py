"""Manager for MemoryCitation CRUD operations."""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, or_, select, tuple_

from mirix.log import get_logger
from mirix.orm.memory_citation import MemoryCitation as MemoryCitationModel
from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation
from mirix.utils import enforce_types

logger = get_logger(__name__)


class MemoryCitationManager:
    """Manager for memory citation persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def _exists_cache_key(self, memory_source_id: str, memory_type: str, memory_id: str) -> str:
        """Cache key for the check_exists lookup (hot-path query for citation-level dedup)."""
        return f"citation_exists:{memory_source_id}:{memory_type}:{memory_id}"

    async def _cache_exists(self, memory_source_id: str, memory_type: str, memory_id: str) -> None:
        """Cache a positive exists result. Never raises."""
        try:
            from mirix.database.cache_provider import get_cache_provider
            from mirix.settings import settings

            cache_provider = get_cache_provider()
            if cache_provider:
                key = self._exists_cache_key(memory_source_id, memory_type, memory_id)
                await cache_provider.set_json(key, {"exists": True}, ttl=settings.redis_ttl_default)
        except Exception as e:
            logger.warning("Cache write failed for citation exists check: %s", e)

    @enforce_types
    async def create(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
        citation_type: str,
        external_thread_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        created_by_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[PydanticMemoryCitation]:
        """Create a citation record using INSERT ON CONFLICT DO NOTHING.

        Returns the created record, or None if it already existed.
        Cache write for the citation record handled by ORM via create_or_ignore_with_redis.
        Also caches the exists check for the (source, type, id) triple.
        """
        citation_id = PydanticMemoryCitation._generate_id()
        now = datetime.now(timezone.utc)
        values = dict(
            id=citation_id,
            memory_source_id=memory_source_id,
            memory_type=memory_type,
            memory_id=memory_id,
            citation_type=citation_type,
            external_thread_id=external_thread_id,
            occurred_at=occurred_at,
            _created_by_id=created_by_id,
            _last_updated_by_id=created_by_id,
            created_at=now,
            updated_at=now,
            is_deleted=False,
        )

        # Relational provider delegation — relies on uq_memory_citations_src_type_id
        # unique constraint for L3 dedup. Conflict → None (matches "already existed").
        # Transient errors are retried; final failure logs WARNING + skip-span and
        # returns None so the caller's memory write stands. Permanent errors propagate.
        from mirix.database.provider_write_retry import (
            is_conflict,
            is_transient,
        )
        from mirix.database.relational_provider import get_relational_provider
        from mirix.observability.skip_spans import emit_idempotency_skip_span

        provider = get_relational_provider()
        if provider:
            payload = {
                **values,
                "occurred_at": occurred_at.isoformat() if occurred_at else None,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            try:
                # The relational provider has its own inner-retry tier
                # (event_retry.retry_with_backoff) — no extra wrapper here.
                await provider.create("memory_citations", payload)
                logger.info(
                    "Created citation %s: %s/%s -> source %s",
                    citation_id,
                    memory_type,
                    memory_id,
                    memory_source_id,
                )
                pydantic_values = {k: v for k, v in values.items() if k != "is_deleted" and not k.startswith("_")}
                return PydanticMemoryCitation(**pydantic_values)
            except Exception as e:
                if is_conflict(e):
                    logger.debug(
                        "memory_citations conflict for %s/%s/%s — treating as no-op",
                        memory_source_id,
                        memory_type,
                        memory_id,
                    )
                    return None
                if is_transient(e):
                    # Exhausted retries on a transient error — log + skip-span,
                    # don't raise (memory write must stand).
                    logger.warning(
                        "memory_citations write failed after retries for %s/%s/%s: %s",
                        memory_source_id,
                        memory_type,
                        memory_id,
                        e,
                    )
                    emit_idempotency_skip_span(
                        name="Citation Write: failed",
                        reason="citation-write-failed",
                        metadata={
                            "memory_source_id": memory_source_id,
                            "memory_type": memory_type,
                            "memory_id": memory_id,
                            "error": str(e),
                        },
                    )
                    return None
                # Permanent error (auth, malformed request, unknown entity) — fail loudly.
                logger.error(
                    "memory_citations write failed permanently for %s/%s/%s: %s",
                    memory_source_id,
                    memory_type,
                    memory_id,
                    e,
                )
                raise

        async with self.session_maker() as session:
            inserted = await MemoryCitationModel.create_or_ignore_with_redis(
                db_session=session,
                use_cache=use_cache,
                **values,
            )

        # Cache the exists result (whether just inserted or already existed)
        if use_cache:
            await self._cache_exists(memory_source_id, memory_type, memory_id)

        if inserted:
            logger.info(
                "Created citation %s: %s/%s -> source %s",
                citation_id,
                memory_type,
                memory_id,
                memory_source_id,
            )
            pydantic_values = {k: v for k, v in values.items() if k != "is_deleted" and not k.startswith("_")}
            return PydanticMemoryCitation(**pydantic_values)
        return None

    @enforce_types
    async def check_exists(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
        use_cache: bool = True,
    ) -> bool:
        """Check if a citation already exists for the given (source, type, id) triple.

        Uses cache-aside pattern since this is the hot-path check for citation-level dedup.
        """
        # Relational provider delegation — provider is source of truth, skip cache
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # route through named query so the response shape
            # matches the Pydantic model (memory_source_id flat column).
            records = await provider.find_using_named_query(
                "memory_citations",
                "memory_citation_manager.find_existing_citation",
                params={
                    "memorySourceId": memory_source_id,
                    "memoryType": memory_type,
                    "memoryId": memory_id,
                },
                page_size=1,
            )
            return bool(records)

        # Try cache first
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    key = self._exists_cache_key(memory_source_id, memory_type, memory_id)
                    cached_data = await cache_provider.get_json(key)
                    if cached_data:
                        logger.debug(
                            "Cache HIT for citation exists: %s/%s/%s",
                            memory_source_id,
                            memory_type,
                            memory_id,
                        )
                        return True
            except Exception as e:
                logger.debug("Cache read failed for citation exists check: %s", e)

        # Fall back to DB
        async with self.session_maker() as session:
            result = await session.execute(
                select(MemoryCitationModel.id).where(
                    MemoryCitationModel.memory_source_id == memory_source_id,
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
            )
            exists = result.scalar_one_or_none() is not None

        # Populate cache on positive result
        if exists and use_cache:
            await self._cache_exists(memory_source_id, memory_type, memory_id)

        return exists

    @enforce_types
    async def get_max_occurred_at(
        self,
        memory_type: str,
        memory_id: str,
    ) -> Optional[datetime]:
        """Get the most recent occurred_at for a given memory record across all citations.

        Used by the temporal guard to prevent backdated overwrites.
        Not cached — called infrequently and the max changes as new citations arrive.
        """
        # Relational provider delegation — fetch matching rows and compute max in memory.
        # The set is small (citations for one memory record) so client-side aggregation is fine.
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # use a dedicated MAX(occurredAt) named query to
            # avoid pulling the full citation set client-side. Returns a
            # single scalar row keyed ``max_occurred_at``.
            #
            # The NQ projects only ``MAX(occurredAt) AS max_occurred_at`` — no
            # ``id`` column — so we pass ``skip_entity_mapping=True`` and a
            # ``MaxOccurredAtResult`` shape. Without that the provider would
            # try to bind the row to the memoryCitations entity schema and
            # reject with "column name id was not found in this ResultSet".
            from mirix.database.named_query_results import MaxOccurredAtResult

            records = await provider.find_using_named_query(
                "memory_citations",
                "memory_citation_manager.max_occurred_at_for_memory",
                params={
                    "memoryType": memory_type,
                    "memoryId": memory_id,
                },
                result_set_entity_class=MaxOccurredAtResult,
                skip_entity_mapping=True,
                page_size=1,
            )
            occurreds = [r.get("max_occurred_at") for r in records if r.get("max_occurred_at")]
            if not occurreds:
                return None
            max_iso = max(occurreds)
            return datetime.fromisoformat(max_iso.replace("Z", "+00:00")) if isinstance(max_iso, str) else max_iso

        async with self.session_maker() as session:
            result = await session.execute(
                select(func.max(MemoryCitationModel.occurred_at)).where(
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
            )
            return result.scalar_one_or_none()

    async def get_citations_for_memory(
        self,
        memory_type: str,
        memory_id: str,
    ) -> List[PydanticMemoryCitation]:
        """Fetch all citations for a single memory."""
        # Relational provider delegation
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # NQ projects memorysource_id so Pydantic
            # construction works.
            # Paginated fetch — the IPSR NQ runner rejects page_size > 1000.
            # The NQ orders by ipsrcreatedon (ties possible), so the helper's
            # cross-page dedup by id guards against boundary repeats.
            from mirix.services.memory_manager_helpers import find_all_using_named_query

            records = await find_all_using_named_query(
                provider,
                "memory_citations",
                "memory_citation_manager.get_citations_for_memory",
                params={
                    "memoryType": memory_type,
                    "memoryId": memory_id,
                },
            )
            # Order by occurred_at desc, nulls last
            records.sort(
                key=lambda r: (
                    r.get("occurred_at") is None,
                    r.get("occurred_at") or "",
                ),
                reverse=False,
            )
            records.sort(
                key=lambda r: r.get("occurred_at") or "",
                reverse=True,
            )
            return [PydanticMemoryCitation(**r) for r in records]

        async with self.session_maker() as session:
            stmt = (
                select(MemoryCitationModel)
                .where(
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
                .order_by(MemoryCitationModel.occurred_at.desc().nulls_last())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            PydanticMemoryCitation(
                id=row.id,
                memory_type=row.memory_type,
                memory_id=row.memory_id,
                memory_source_id=row.memory_source_id,
                external_thread_id=row.external_thread_id,
                occurred_at=row.occurred_at,
                citation_type=row.citation_type,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    async def get_citations_for_memories(
        self,
        memory_keys: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], List[PydanticMemoryCitation]]:
        """Batch-fetch citations for a list of (memory_type, memory_id) pairs.

        Returns a dict keyed by (memory_type, memory_id) -> list of citations.
        Used by search endpoints to attach citation provenance to results.
        """
        if not memory_keys:
            return {}

        # Relational provider delegation — provider.list doesn't support tuple-IN filters,
        # so loop per (memory_type, memory_id) and group. Search hits are typically
        # 10-50 per query, so per-memory call volume is acceptable.
        #
        # route through ``memory_citation_manager.get_citations_for_memory``
        # named query instead of the generic ``provider.list``. The adhoc
        # filter-query endpoint does NOT populate the ``memorySource`` MANY_TO_ONE
        # relationship on returned entities, which causes
        # ``PydanticMemoryCitation(**r)`` to fail with ``memory_source_id Field
        # required`` and ultimately empties the entire search result. The named
        # query's ``SELECT *`` projects ``memorysource_id`` as a flat column, which
        # the SDK ``loads_entity`` helper auto-wraps as ``EntityRef(id=...)`` and
        # the provider's ``from_entity`` flattens back to ``memory_source_id``.
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            grouped: Dict[Tuple[str, str], List[PydanticMemoryCitation]] = defaultdict(list)
            for memory_type, memory_id in memory_keys:
                records = await provider.find_using_named_query(
                    "memory_citations",
                    "memory_citation_manager.get_citations_for_memory",
                    params={
                        "memoryType": memory_type,
                        "memoryId": memory_id,
                    },
                    page_size=1000,
                )
                grouped[(memory_type, memory_id)].extend(PydanticMemoryCitation(**r) for r in records)
            return dict(grouped)

        async with self.session_maker() as session:
            stmt = select(MemoryCitationModel).where(
                tuple_(MemoryCitationModel.memory_type, MemoryCitationModel.memory_id).in_(memory_keys),
                ~MemoryCitationModel.is_deleted,
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        grouped: Dict[Tuple[str, str], List[PydanticMemoryCitation]] = defaultdict(list)
        for row in rows:
            citation = PydanticMemoryCitation(
                id=row.id,
                memory_type=row.memory_type,
                memory_id=row.memory_id,
                memory_source_id=row.memory_source_id,
                external_thread_id=row.external_thread_id,
                occurred_at=row.occurred_at,
                citation_type=row.citation_type,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            grouped[(row.memory_type, row.memory_id)].append(citation)

        return dict(grouped)
