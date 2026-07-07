"""Shared helpers for memory-table managers.

This module centralises a few patterns that previously lived as ad-hoc
duplication across the seven memory-table managers (block, raw_memory,
episodic_memory, semantic_memory, procedural_memory, resource_memory,
knowledge_vault):

- :data:`TABLE_TO_CACHE_PREFIX`: maps a memory-table name to the Redis-key
  prefix exposed on the cache provider class (e.g. ``BLOCK_PREFIX``).
- :func:`invalidate_memory_cache`: deletes the cache entries for a list of
  IDs in a memory table, swallowing cache errors with a warning so a Redis
  outage never blocks the underlying provider write.
- :func:`find_most_recently_updated`: provider-agnostic helper that returns
  the most-recently-updated row for a memory table given a set of filters.
  For all 7 memory tables (block, raw_memory, episodic_memory,
  semantic_memory, procedural_memory, resource_memory, knowledge_vault) it
  uses ``find_using_named_query`` with ``ORDER BY COALESCE(ipsrupdatedon,
  ipsrcreatedon) DESC`` and ``page_size=1`` whenever ``user_id`` and
  ``organization_id`` are provided. Falls back to
  ``provider.list(..., sort="updated_at", limit=1)`` for client-scoped or
  extra-filter call shapes that the named query doesn't cover.
- :func:`actor_from_user`: build a duck-typed actor object from a
  ``PydanticUser`` so memory-manager call sites that have only a user
  (no client) can still feed the actor-aware provider methods.

All helpers are pure-Python and fail-soft: cache errors are logged, not
raised; missing dependencies (no provider, no cache) become a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)


# Memory tables -> Redis-key prefix attribute name on the cache provider.
# Kept in sync with mirix.database.redis_cache_provider / redis_client where
# the constants are declared.
TABLE_TO_CACHE_PREFIX: Dict[str, str] = {
    "block": "BLOCK_PREFIX",
    "raw_memory": "RAW_MEMORY_PREFIX",
    "episodic_memory": "EPISODIC_PREFIX",
    "semantic_memory": "SEMANTIC_PREFIX",
    "procedural_memory": "PROCEDURAL_PREFIX",
    "resource_memory": "RESOURCE_PREFIX",
    "knowledge_vault": "KNOWLEDGE_PREFIX",
}


async def invalidate_memory_cache(table: str, ids: Iterable[str]) -> None:
    """Invalidate cache entries for ``ids`` in a memory table.

    Safe to call when:
    - the cache provider is not configured (no-op);
    - a single ``cache_provider.delete`` raises (logged, others still attempted);
    - ``ids`` is empty (no-op).
    """
    id_list = [i for i in ids if i]
    if not id_list:
        return

    prefix_attr = TABLE_TO_CACHE_PREFIX.get(table)
    if not prefix_attr:
        logger.debug(
            "invalidate_memory_cache: unknown table %s, skipping cache invalidation",
            table,
        )
        return

    try:
        from mirix.database.cache_provider import get_cache_provider
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("Cache provider import failed: %s", exc)
        return

    cache_provider = get_cache_provider()
    if cache_provider is None:
        return

    prefix = getattr(cache_provider, prefix_attr, None)
    if prefix is None:
        logger.debug(
            "Cache provider has no %s attribute, skipping invalidation for %s",
            prefix_attr,
            table,
        )
        return

    for entity_id in id_list:
        cache_key = f"{prefix}{entity_id}"
        try:
            await cache_provider.delete(cache_key)
        except Exception as exc:
            logger.warning("Failed to invalidate cache for %s/%s: %s", table, entity_id, exc)


# Tables with a dedicated named query that returns the single most-recently-updated
# row ordered by COALESCE(ipsrupdatedon, ipsrcreatedon) DESC. Only applicable when
# user_id and organization_id are both provided (the common call pattern).
_MOST_RECENTLY_UPDATED_QUERIES: Dict[str, str] = {
    "block": "memory_manager_helpers.get_most_recently_updated_block",
    "raw_memory": "memory_manager_helpers.get_most_recently_updated_raw_memory",
    "episodic_memory": "memory_manager_helpers.get_most_recently_updated_episodic_memory",
    "semantic_memory": "memory_manager_helpers.get_most_recently_updated_semantic_memory",
    "procedural_memory": "memory_manager_helpers.get_most_recently_updated_procedural_memory",
    "resource_memory": "memory_manager_helpers.get_most_recently_updated_resource_memory",
    "knowledge_vault": "memory_manager_helpers.get_most_recently_updated_knowledge_vault",
}


async def find_most_recently_updated(
    provider: Any,
    table: str,
    *,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    client_id: Optional[str] = None,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the most-recently-updated row for ``table`` matching the filters.

    For all memory tables registered in ``_MOST_RECENTLY_UPDATED_QUERIES``
    (block, raw_memory, episodic_memory, semantic_memory, procedural_memory,
    resource_memory, knowledge_vault) this uses a named query with
    server-side ordering so a single round-trip returns the correct row.
    Only active when ``user_id`` and ``organization_id`` are both provided
    and no ``client_id``/``extra_filters`` are supplied; falls back to the
    ``provider.list`` path otherwise.

    For other tables (or unsupported call shapes), uses
    ``provider.list(... sort="updated_at", limit=1)`` and falls back to
    ``sort="created_at"`` when no row has an ``updated_at``.

    Returns ``None`` when no matching row exists.
    """
    query_name = _MOST_RECENTLY_UPDATED_QUERIES.get(table)
    if query_name and user_id is not None and organization_id is not None and client_id is None and not extra_filters:
        rows = await provider.find_using_named_query(
            table,
            query_name,
            params={"userId": user_id, "organizationId": organization_id},
            page_size=1,
        )
        return rows[0] if rows else None

    filter_kwargs: Dict[str, Any] = {}
    if user_id is not None:
        filter_kwargs["user_id"] = user_id
    if organization_id is not None:
        filter_kwargs["organization_id"] = organization_id
    if client_id is not None:
        filter_kwargs["filter_tags"] = {"client_id": client_id}
    if extra_filters:
        filter_kwargs.setdefault("filter_tags", {}).update(extra_filters)

    rows = await provider.list(
        table,
        sort="updated_at",
        limit=1,
        **filter_kwargs,
    )
    if rows:
        return rows[0]

    rows = await provider.list(
        table,
        sort="created_at",
        limit=1,
        **filter_kwargs,
    )
    return rows[0] if rows else None


# The IPSR named-query runner rejects pageSize outside [1, 1000] with
# InvalidPageSizeException (400). Callers that need a complete result set
# must paginate; a single oversized request fails outright rather than
# truncating.
IPSR_NQ_MAX_PAGE_SIZE = 1000

# Safety valve for the pagination loops below: 100 pages x 1000 rows. Hitting
# it means a filter matched >100k rows (or a query stopped making progress) —
# both worth a WARNING rather than an unbounded loop.
_NQ_MAX_PAGES = 100


async def find_all_using_named_query(
    provider: Any,
    table: str,
    query_name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    page_size: int = IPSR_NQ_MAX_PAGE_SIZE,
    **kwargs: Any,
) -> List[Any]:
    """Fetch EVERY row of a named query by looping SLICE pages.

    ``page_size`` is clamped to :data:`IPSR_NQ_MAX_PAGE_SIZE`. Rows are
    deduplicated by ``id`` across pages: offset pagination over a named query
    whose ORDER BY is absent or non-unique can repeat a row on a page
    boundary. (It can also *skip* rows in that situation — callers for whom
    a miss is unacceptable should ensure the NQ orders by a unique column.)

    Stops when a page comes back short of ``page_size``. If ``_NQ_MAX_PAGES``
    pages come back full, logs a WARNING and returns what was collected.
    """
    page_size = min(page_size, IPSR_NQ_MAX_PAGE_SIZE)
    collected: List[Any] = []
    seen_ids: set = set()
    for page_num in range(_NQ_MAX_PAGES):
        rows = await provider.find_using_named_query(
            table,
            query_name,
            params=params,
            page_size=page_size,
            page_num=page_num,
            **kwargs,
        )
        for row in rows:
            row_id = row.get("id") if isinstance(row, dict) else None
            if row_id is not None:
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
            collected.append(row)
        if len(rows) < page_size:
            return collected
    logger.warning(
        "find_all_using_named_query hit max_pages=%s for %s.%s; result may be truncated",
        _NQ_MAX_PAGES,
        table,
        query_name,
    )
    return collected


async def bulk_delete_all_from_named_query(
    provider: Any,
    table: str,
    query_name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    soft: bool = True,
) -> int:
    """Delete every row matched by an id-listing named query, in batches.

    Fetch-delete-repeat: reads the FIRST page (<= 1000 ids), bulk-deletes
    those rows, then re-reads the first page. Because the id-listing NQs
    filter ``NOT isDeleted`` (and hard deletes remove the rows outright),
    each deleted batch leaves the result set, so repeatedly draining page 0
    is stable regardless of the query's ORDER BY — unlike offset pagination,
    it can neither skip nor repeat rows while deleting.

    Guards: stops with a WARNING if a batch makes no progress (identical ids
    twice in a row — e.g. bulk_delete silently failing) or after
    ``_NQ_MAX_PAGES`` batches. Returns the total ``success`` count reported
    by ``provider.bulk_delete``.
    """
    total_deleted = 0
    previous_ids: Optional[List[str]] = None
    for _ in range(_NQ_MAX_PAGES):
        rows = await provider.find_using_named_query(
            table,
            query_name,
            params=params,
            page_size=IPSR_NQ_MAX_PAGE_SIZE,
        )
        ids = [r.get("id") for r in rows if isinstance(r, dict) and r.get("id")]
        if not ids:
            return total_deleted
        if ids == previous_ids:
            logger.warning(
                "bulk_delete_all_from_named_query made no progress on %s.%s "
                "(%d ids unchanged after delete); stopping",
                table,
                query_name,
                len(ids),
            )
            return total_deleted
        previous_ids = ids
        result = await provider.bulk_delete(table, ids, soft=soft)
        total_deleted += int(result.get("success", 0) or 0)
        if len(ids) < IPSR_NQ_MAX_PAGE_SIZE:
            return total_deleted
    logger.warning(
        "bulk_delete_all_from_named_query hit max_pages=%s for %s.%s; rows may remain",
        _NQ_MAX_PAGES,
        table,
        query_name,
    )
    return total_deleted


@dataclass(frozen=True)
class _ActorView:
    """Lightweight, immutable view used to feed actor-aware provider methods.

    Only the attributes the predicate inspects are populated: ``id``,
    ``organization_id``, and ``user_id``.  Mirrors the duck-typed contract
    used by ``PydanticClient`` so the predicate works without importing the
    full schema module.
    """

    id: str
    organization_id: str
    user_id: Optional[str] = None


def actor_from_user(
    user: Any,
    *,
    client_id: Optional[str] = None,
) -> _ActorView:
    """Build a minimal actor from a ``PydanticUser``.

    *user* is expected to expose ``id`` and ``organization_id``; ``client_id``
    fills the actor's ``id`` slot when supplied (it is what the provider
    persists into ``ipsr_entity_owner``).  When ``client_id`` is None, the
    user's own ``id`` is used — appropriate for service-level write paths
    that do not yet know the client.

    The returned object can also be constructed from a ``PydanticClient`` by
    passing the client as ``user`` (since both expose the relevant
    attributes); the wrapper is only here to make user-only call sites work
    against the actor-aware predicate.
    """
    user_id = getattr(user, "id", None)
    organization_id = getattr(user, "organization_id", None)
    if user_id is None or organization_id is None:
        raise ValueError("actor_from_user requires user to expose 'id' and 'organization_id'")
    return _ActorView(
        id=client_id or user_id,
        organization_id=organization_id,
        user_id=user_id,
    )


__all__ = [
    "IPSR_NQ_MAX_PAGE_SIZE",
    "TABLE_TO_CACHE_PREFIX",
    "bulk_delete_all_from_named_query",
    "find_all_using_named_query",
    "invalidate_memory_cache",
    "find_most_recently_updated",
    "actor_from_user",
]
