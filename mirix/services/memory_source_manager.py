"""Manager for MemorySource CRUD operations."""

import base64
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.memory_source import MemorySource as MemorySourceModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.memory_source import MemorySource as PydanticMemorySource

if TYPE_CHECKING:
    from mirix.queue.error_policy import SaveOutcome
from mirix.schemas.memory_source import PaginatedResponse
from mirix.utils import enforce_types

logger = get_logger(__name__)


# NOTE: The outcome vocabulary lives in `mirix.queue.error_policy.SaveOutcome`.
# `finalize_source` takes a SaveOutcome value; today (boolean schema) it writes
# `processing_complete=True` for every outcome and records the value in a log
# line. A future status-column migration will diversify the actual column
# writes per outcome by touching only `finalize_source` — upstream callers
# already produce the right value.

# Hard ceiling on the per-request page size we'll forward to IPSR. The IPSR
# named-query runner caps page_size at 1500 server-side; we cap a little
# lower so we never get truncated silently. Callers asking for more than this
# get clamped.
_LIST_SOURCES_MAX_PAGE_SIZE = 1000


def _encode_page_cursor(page_num: int) -> str:
    """Encode an offset-pagination cursor for list_sources.

    The cursor is an opaque base64 JSON blob carrying the *next* page_num to
    fetch. We don't promise stability across param changes — callers who
    change filters mid-pagination get whatever the new query returns at the
    encoded page.
    """
    return base64.urlsafe_b64encode(json.dumps({"p": page_num}).encode("utf-8")).decode("ascii")


def _decode_page_cursor(cursor: Optional[str]) -> int:
    """Decode a list_sources cursor → page_num. None / malformed → 0."""
    if not cursor:
        return 0
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        page_num = int(payload.get("p", 0))
        return page_num if page_num >= 0 else 0
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Malformed list_sources cursor; restarting at page 0")
        return 0


def parse_occurred_at(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse an occurred_at value that may be an ISO 8601 string, datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse occurred_at value: %s", value)
            return None
    return None


class MemorySourceManager:
    """Manager for memory source persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    async def create(
        self,
        memory_source_id: str,
        actor: PydanticClient,
        user_id: str,
        organization_id: str,
        source_type: str = "conversation",
        external_id: Optional[str] = None,
        external_thread_id: Optional[str] = None,
        source_system: Optional[str] = None,
        source_metadata: Optional[dict] = None,
        occurred_at: Union[str, datetime, None] = None,
        summary: Optional[str] = None,
        summary_source: Optional[str] = None,
        batch_hash: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Optional[PydanticMemorySource]:
        """Create (or resolve) the canonical memory source row, idempotently.

        Enforces scope injection: filter_tags["scope"] is always set to
        actor.write_scope, matching the pattern in raw_memory_manager.
        Client-provided filter_tags are preserved but scope cannot be overridden.

        Returns the row to process, or ``None`` to skip:
          * No conflict -> the newly-created row.
          * Conflict whose existing row has the SAME id (the id PK conflicted) ->
            that row. A retry / redelivery of this submission; the caller resumes
            it and finalize (keyed off the unchanged id) completes it.
          * Conflict whose existing row has a DIFFERENT id (external_id /
            batch_hash conflicted) -> ``None``. A separate duplicate submission
            owns the content; the caller skips and that owner finishes its row.

        Both the IPS-Relational and PG ON CONFLICT DO NOTHING paths honor this.
        Cache write handled by the ORM layer via create_or_ignore_with_redis.
        """
        # Enforce scope from actor's write_scope (same pattern as raw_memory_manager)
        if actor.write_scope is None:
            raise ValueError("Client has no write_scope - cannot create memory sources")
        if filter_tags is None:
            filter_tags = {}
        filter_tags["scope"] = actor.write_scope

        occurred_at = parse_occurred_at(occurred_at)

        # Relational provider delegation (create — relies on uq_memory_sources_ext_id /
        # uq_memory_sources_batch unique constraints for dedup).
        # Conflict → look up the pre-existing row (caller wants the existing record back).
        # Transient errors are retried; permanent errors propagate.
        from mirix.database.provider_write_retry import (
            is_conflict,
            is_transient,
        )
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            now = datetime.now(timezone.utc)
            data_dict = {
                "id": memory_source_id,
                "client_id": actor.id,
                "user_id": user_id,
                "organization_id": organization_id,
                "source_type": source_type,
                "external_id": external_id,
                "external_thread_id": external_thread_id,
                "source_system": source_system,
                "source_metadata": source_metadata,
                "occurred_at": occurred_at.isoformat() if occurred_at else None,
                "summary": summary,
                "summary_source": summary_source,
                "batch_hash": batch_hash,
                "filter_tags": filter_tags,
                "processing_complete": False,
                "_created_by_id": actor.id,
                "_last_updated_by_id": actor.id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "is_deleted": False,
            }
            try:
                # The relational provider has its own inner-retry tier
                # (event_retry.retry_with_backoff) — no extra wrapper here.
                result = await provider.create("memory_sources", data_dict)
                return PydanticMemorySource(**result)
            except Exception as e:
                if not is_conflict(e):
                    if is_transient(e):
                        logger.error(
                            "memory_sources create failed after retries for %s: %s",
                            memory_source_id,
                            e,
                        )
                    raise
                # Conflict on a unique index. Resolve the winning row by the key
                # that conflicted, then decide resume vs skip by id identity:
                #
                #   * SAME id (the id PK conflicted) -> a retry / redelivery of
                #     *this* submission. Return the row so the caller resumes it;
                #     finalize keys off the unchanged id, so completion lands on
                #     this row.
                #   * DIFFERENT id (external_id / batch_hash conflicted) -> a
                #     separate duplicate submission owns this content. Return None
                #     so the caller skips: a completed row is a true duplicate, and
                #     an incomplete one is finished by its owner (or that owner's
                #     own same-id redelivery). Resuming it here would race the
                #     owner and mis-target finalize (which keys off this id).
                #
                # The lookup also confirms the row exists (a conflict with no
                # recoverable row is a different failure — re-raise). Named queries
                # project all columns + FK columns so PydanticMemorySource builds.
                if external_id is not None:
                    records = await provider.find_using_named_query(
                        "memory_sources",
                        "memory_source_manager.find_by_external_id",
                        params={
                            # NQ binds :clientId against the client_id FK column
                            # (matches uq_memory_sources_ext_id). The actor IS the
                            # client when MIRIX runs under ECMS, so actor.id is the
                            # client id; we use the parameter name "clientId" so the
                            # provider's APP=-prefixing of :createdById doesn't fire.
                            "clientId": actor.id,
                            "userId": user_id,
                            "externalId": external_id,
                        },
                        page_size=1,
                    )
                elif batch_hash is not None:
                    records = await provider.find_using_named_query(
                        "memory_sources",
                        "memory_source_manager.find_by_batch_hash",
                        params={
                            # See note in find_by_external_id above re: clientId vs createdById.
                            "clientId": actor.id,
                            "userId": user_id,
                            "batchHash": batch_hash,
                        },
                        page_size=1,
                    )
                else:
                    raise
                if not records:
                    raise
                existing = PydanticMemorySource(**records[0])
                if existing.id == memory_source_id:
                    return existing
                logger.debug(
                    "memory_sources conflict for %s — owned by different source %s; skipping",
                    memory_source_id,
                    existing.id,
                )
                return None

        async with self.session_maker() as session:
            now = datetime.now(timezone.utc)

            await MemorySourceModel.create_or_ignore_with_redis(
                db_session=session,
                use_cache=use_cache,
                id=memory_source_id,
                client_id=actor.id,
                user_id=user_id,
                organization_id=organization_id,
                source_type=source_type,
                external_id=external_id,
                external_thread_id=external_thread_id,
                source_system=source_system,
                source_metadata=source_metadata,
                occurred_at=occurred_at,
                summary=summary,
                summary_source=summary_source,
                batch_hash=batch_hash,
                filter_tags=filter_tags,
                processing_complete=False,
                _created_by_id=actor.id,
                _last_updated_by_id=actor.id,
                created_at=now,
                updated_at=now,
                is_deleted=False,
            )

        # Return the record (from cache or DB via get_by_id)
        return await self.get_by_id(memory_source_id, use_cache=use_cache)

    @enforce_types
    async def get_by_id(self, memory_source_id: str, use_cache: bool = True) -> Optional[PydanticMemorySource]:
        """Fetch a memory source by ID with cache-aside pattern."""
        # Relational provider delegation (read by ID) — provider is source of truth, skip cache
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.read("memory_sources", memory_source_id)
            if result is None:
                return None
            return PydanticMemorySource(**result)

        # Try cache first
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    cached_data = await cache_provider.get_json(cache_key)
                    if cached_data:
                        logger.debug("Cache HIT for memory source %s", memory_source_id)
                        # Strip Redis-internal fields (_ts suffixes) that pydantic rejects
                        known_fields = PydanticMemorySource.model_fields
                        clean = {k: v for k, v in cached_data.items() if k in known_fields}
                        return PydanticMemorySource(**clean)
            except Exception as e:
                logger.debug("Cache read failed for memory source %s: %s", memory_source_id, e)

        # Fall back to DB
        async with self.session_maker() as session:
            result = await session.execute(
                select(MemorySourceModel).where(
                    MemorySourceModel.id == memory_source_id,
                    ~MemorySourceModel.is_deleted,
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            pydantic_source = record.to_pydantic()

        # Populate cache on miss
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.settings import settings

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    data = pydantic_source.model_dump(mode="json")
                    await cache_provider.set_json(cache_key, data, ttl=settings.redis_ttl_default)
                    logger.debug("Populated cache for memory source %s", memory_source_id)
            except Exception as e:
                logger.warning("Cache write failed for memory source %s: %s", memory_source_id, e)

        return pydantic_source

    async def list_sources(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[PydanticMemorySource]:
        """List memory sources ordered by occurred_at descending.

        Admin-only listing — no scope-based access control. organization_id
        is required on the IPSR path (the NQ uses it as the access predicate)
        and recommended on the ORM fallback path.

        Pagination is offset-based: ``limit`` is the page size, ``cursor`` is
        an opaque token encoding the next page_num. Page 0 is returned when
        ``cursor`` is None.

        IPSR ``page_size`` is clamped to ``_LIST_SOURCES_MAX_PAGE_SIZE`` (the
        server enforces 1500). Use a smaller ``limit`` if you want snappy
        round-trips.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        page_size = min(limit, _LIST_SOURCES_MAX_PAGE_SIZE)
        page_num = _decode_page_cursor(cursor)

        # Relational provider delegation
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # Real offset pagination against the list_sources_admin named
            # query. The NQ filters and orders server-side (occurredAt DESC
            # NULLS LAST, ipsrcreatedon DESC); page_num + page_size are
            # forwarded to the relational provider's slice-pagination
            # contract.
            if not organization_id:
                raise ValueError(
                    "list_sources requires organization_id when an IPSR "
                    "provider is registered (the NQ uses it as the "
                    "org-scoped access predicate)"
                )
            # Fetch page_size + 1 so we can tell has_more without an extra
            # round-trip. IPSR's slice contract returns up to the requested
            # count, so a full page means "there is at least one more row".
            fetch_size = min(page_size + 1, _LIST_SOURCES_MAX_PAGE_SIZE)
            records = await provider.find_using_named_query(
                "memory_sources",
                "memory_source_manager.list_sources_admin",
                params={
                    "organizationId": organization_id,
                    "userId": user_id,
                    "clientId": client_id,
                    "scope": scope,
                    "since": since.isoformat() if since else None,
                    "until": until.isoformat() if until else None,
                },
                page_size=fetch_size,
                page_num=page_num,
            )
            has_more = len(records) > page_size
            records = records[:page_size]
            items = [PydanticMemorySource(**r) for r in records]
            return PaginatedResponse(
                items=items,
                next_cursor=_encode_page_cursor(page_num + 1) if has_more else None,
                has_more=has_more,
            )

        async with self.session_maker() as session:
            query = (
                select(MemorySourceModel)
                .where(~MemorySourceModel.is_deleted)
                .order_by(
                    MemorySourceModel.occurred_at.desc().nulls_last(),
                    MemorySourceModel.created_at.desc(),
                )
            )

            if organization_id:
                query = query.where(MemorySourceModel.organization_id == organization_id)
            if user_id:
                query = query.where(MemorySourceModel.user_id == user_id)
            if client_id:
                query = query.where(MemorySourceModel.client_id == client_id)
            if scope:
                query = query.where(MemorySourceModel.filter_tags["scope"].astext == scope)
            if since:
                query = query.where(MemorySourceModel.occurred_at >= since)
            if until:
                query = query.where(MemorySourceModel.occurred_at <= until)

            # Mirror the IPSR path's offset-pagination contract: SQL OFFSET
            # for page traversal, LIMIT + 1 for has_more probing.
            query = query.offset(page_num * page_size).limit(page_size + 1)
            result = await session.execute(query)
            records = result.scalars().all()

            has_more = len(records) > page_size
            records = records[:page_size]
            items = [rec.to_pydantic() for rec in records]

            return PaginatedResponse(
                items=items,
                next_cursor=_encode_page_cursor(page_num + 1) if has_more else None,
                has_more=has_more,
            )

    @enforce_types
    async def mark_processing_complete(self, memory_source_id: str) -> None:
        """Set processing_complete = True after all agents finish successfully.

        Uses read-then-update via ORM for consistent cache handling.
        Safe from lost updates because:
        - processing_complete is a one-way flag (False → True), so concurrent
          writers always converge to the same value
        - SQLAlchemy only UPDATEs dirty columns, so this won't overwrite
          concurrent changes to other fields (e.g. summary)

        Tolerant of a missing row: when a source was deduped (external_id /
        batch_hash conflict, either caught up front or having lost an
        INSERT ... ON CONFLICT race) the minted id never became a persisted
        row, so there is nothing to mark — the *winning* row from the prior
        submission marks itself complete. The provider raises
        ProviderNotFoundError for a missing row; we swallow it as a benign
        no-op (DEBUG) so a deduped source doesn't emit a spurious finalize
        ERROR. Real update failures still propagate.
        """
        # Relational provider delegation (partial update)
        from mirix.database.relational_provider import get_relational_provider
        from mirix.errors import ProviderNotFoundError
        from mirix.orm.errors import NoResultFound

        provider = get_relational_provider()
        if provider:
            try:
                await provider.update("memory_sources", memory_source_id, {"processing_complete": True})
            except ProviderNotFoundError:
                logger.debug(
                    "mark_processing_complete: memory source %s not found (deduped) — no-op",
                    memory_source_id,
                )
                return
            logger.info("Marked memory source %s as processing complete", memory_source_id)
            return

        async with self.session_maker() as session:
            try:
                record = await MemorySourceModel.read(db_session=session, identifier=memory_source_id)
            except NoResultFound:
                logger.debug(
                    "mark_processing_complete: memory source %s not found (deduped) — no-op",
                    memory_source_id,
                )
                return
            record.processing_complete = True
            await record.update_with_redis(session)
            logger.info("Marked memory source %s as processing complete", memory_source_id)

    async def finalize_source(
        self,
        memory_source_id: Optional[str],
        outcome: "SaveOutcome",
    ) -> None:
        """Single finalize chokepoint — all save-path "mark done" decisions
        flow through here.

        Called by `dispatch_save` in error_policy.py with the SaveOutcome that
        the policy returned. Every outcome (SUCCESS / PERMANENT_FAILURE /
        TRANSIENT_EXHAUSTED) flows through this one function.

        Today (boolean schema): every outcome writes `processing_complete=True`
        and records the SaveOutcome value in the log line.

        Tomorrow (status-column migration): this function fans out into
        distinct `status` writes per outcome. Upstream callers don't change.
        """
        if not memory_source_id:
            return

        # Best-effort: never mask the original exception by raising from
        # the finalize chokepoint. The caller has already logged the cause.
        try:
            await self.mark_processing_complete(memory_source_id)
            logger.info(
                "Finalized memory_source=%s outcome=%s",
                memory_source_id,
                outcome.value,
            )
        except Exception:
            logger.exception(
                "Failed to finalize memory_source=%s outcome=%s",
                memory_source_id,
                outcome.value,
            )

    @enforce_types
    async def update_summary(self, memory_source_id: str, summary: str, summary_source: str) -> None:
        """Write a summary to an existing memory source.

        Used after processing completes to store a generated summary.
        Safe from lost updates: only touches summary/summary_source columns.
        """
        # Relational provider delegation (partial update)
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            await provider.update(
                "memory_sources",
                memory_source_id,
                {"summary": summary, "summary_source": summary_source},
            )
            logger.info(
                "Updated summary for memory source %s (source=%s)",
                memory_source_id,
                summary_source,
            )
            return

        async with self.session_maker() as session:
            record = await MemorySourceModel.read(db_session=session, identifier=memory_source_id)
            record.summary = summary
            record.summary_source = summary_source
            await record.update_with_redis(session)
            logger.info(
                "Updated summary for memory source %s (source=%s)",
                memory_source_id,
                summary_source,
            )
