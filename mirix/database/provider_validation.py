"""
Coupled startup validation for relational and search providers.

Relational DB provider and Search provider must be registered together or not at all.
Cache-only registration remains independently valid.
"""

from mirix.log import get_logger

logger = get_logger(__name__)


def validate_provider_pairing_or_raise() -> None:
    """
    Validate that relational and search providers are either both registered
    or neither registered. Raises RuntimeError on invalid pairing (startup
    hard fail).

    Valid states:
        - Both relational and search providers registered.
        - Neither relational nor search provider registered.

    Invalid states (raises RuntimeError):
        - Relational provider registered but search provider is not.
        - Search provider registered but relational provider is not.

    Cache provider state is independent and not checked here.
    """
    from mirix.database.relational_provider import (
        get_registered_relational_providers,
    )
    from mirix.database.search_provider import get_search_provider

    # Use the raw registry for the presence check, not get_relational_provider():
    # the latter raises once provider mode has latched but no provider is
    # currently registered, whereas here we only want "is one registered right
    # now" (a plain membership test that never triggers the guard).
    has_relational = bool(get_registered_relational_providers())
    has_search = get_search_provider() is not None

    if has_relational and not has_search:
        raise RuntimeError(
            "Relational DB provider is registered but Search provider is not. "
            "Both must be registered together or neither registered."
        )
    if has_search and not has_relational:
        raise RuntimeError(
            "Search provider is registered but Relational DB provider is not. "
            "Both must be registered together or neither registered."
        )

    if has_relational and has_search:
        logger.info("Provider pairing validated: both relational and search providers registered")
    else:
        logger.info("Provider pairing validated: no relational/search providers registered")
