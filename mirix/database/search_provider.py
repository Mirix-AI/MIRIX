"""
Search provider interface and registry for Mirix.

Search providers implement the interface via duck typing (no base class
required). Same pattern as cache_provider.py and relational_provider.py.
All methods are async; callers must await them.

Expected methods (duck typing, all async):

    async search(
        table: str, *,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        search_method: str = "embedding",
        search_field: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        similarity_threshold: Optional[float] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        time_range: Optional[Dict[str, Optional[datetime]]] = None,
        **kwargs,
    ) -> Tuple[List[Dict], Optional[str]]
        Returns (results, next_cursor). next_cursor is None for non-cursor queries.

    async count(
        table: str, *,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
    ) -> int

    async get_by_id(
        table: str,
        identifier: str, *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict]

Interface design notes:
    - ``search`` has a unified signature covering ALL memory manager
      list/search patterns: derived memories (embedding/bm25/string_match/
      fuzzy_match + search_field + similarity_threshold), raw memory
      (filter_tags + sort + cursor + time_range), blocks (filter_tags +
      scopes + limit), knowledge vault (sensitivity via **kwargs),
      org-wide searches (organization_id instead of user_id).
    - ``count`` mirrors get_total_number_of_items with scoping parameters.
    - ``sort`` and ``cursor`` support keyset-based cursor pagination used by
      raw_memory_manager.search_raw_memories.
    - ``time_range`` supports raw memory's flexible time range filtering.
    - ``get_by_id`` is used for hybrid merge deduplication and internal use;
      individual record reads by ID use the relational provider's read().

Usage:
    # In ECMS startup
    from mirix.database.search_provider import register_search_provider
    provider = IPSSearchProvider(config)
    register_search_provider("ips_search", provider)

    # In Mirix service managers
    from mirix.database.search_provider import get_search_provider
    provider = get_search_provider()
    if provider:
        results, cursor = await provider.search("episodic_memory", query_text=query, ...)
"""

from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

_search_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None


def register_search_provider(name: str, provider: Any) -> None:
    """
    Register a search provider with Mirix.

    Last registered provider becomes the active one.

    Args:
        name: Provider identifier (e.g., "ips_search").
        provider: Provider instance implementing the search interface.
    """
    global _search_providers, _active_provider_name

    _search_providers[name] = provider
    _active_provider_name = name
    logger.info("Registered search provider: %s", name)


def get_search_provider() -> Optional[Any]:
    """
    Get the active search provider.

    Returns None if no provider is registered (graceful fallback to
    existing Redis-as-search / SQLAlchemy path).

    Returns:
        Search provider instance or None.
    """
    if _active_provider_name and _active_provider_name in _search_providers:
        return _search_providers[_active_provider_name]
    return None


def unregister_search_provider(name: str) -> None:
    """
    Unregister a search provider (primarily for test isolation).

    Args:
        name: Provider identifier.
    """
    global _search_providers, _active_provider_name

    if name in _search_providers:
        del _search_providers[name]
        if _active_provider_name == name:
            _active_provider_name = None
        logger.info("Unregistered search provider: %s", name)


def get_registered_search_providers() -> Dict[str, Any]:
    """
    Get all registered search providers (for tests/inspection).

    Returns:
        Dictionary of provider_name -> provider_instance.
    """
    return dict(_search_providers)
