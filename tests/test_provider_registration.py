"""
Relational and search provider registry tests for Mirix (providers).

Usage:
    pytest tests/test_provider_registration.py -v
"""

import pytest

from mirix.database.relational_provider import (
    get_registered_relational_providers,
    get_relational_provider,
    register_relational_provider,
    reset_provider_mode_latch,
    unregister_relational_provider,
)
from mirix.database.search_provider import (
    get_registered_search_providers,
    get_search_provider,
    register_search_provider,
    unregister_search_provider,
)
from mirix.errors import RelationalProviderRequiredError


@pytest.fixture(autouse=True)
def cleanup_relational_registry():
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()
    yield
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()


@pytest.fixture(autouse=True)
def cleanup_search_registry():
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)
    yield
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)


class TestRelationalProviderRegistry:
    def test_register_stores_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        assert get_registered_relational_providers()["ips_relational"] is p

    def test_get_returns_active_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        assert get_relational_provider() is p

    def test_unregister_removes_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        unregister_relational_provider("ips_relational")
        assert "ips_relational" not in get_registered_relational_providers()
        # Provider mode has latched (we registered above), so a now-missing
        # provider is a hard error, NOT a silent fall-through to PostgreSQL.
        with pytest.raises(RelationalProviderRequiredError):
            get_relational_provider()

    def test_get_registered_returns_all_providers(self):
        a, b = object(), object()
        register_relational_provider("first", a)
        register_relational_provider("second", b)
        reg = get_registered_relational_providers()
        assert reg == {"first": a, "second": b}
        assert reg is not get_registered_relational_providers()


class TestSearchProviderRegistry:
    def test_register_stores_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        assert get_registered_search_providers()["ips_search"] is p

    def test_get_returns_active_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        assert get_search_provider() is p

    def test_unregister_removes_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        unregister_search_provider("ips_search")
        assert "ips_search" not in get_registered_search_providers()
        assert get_search_provider() is None

    def test_get_registered_returns_all_providers(self):
        a, b = object(), object()
        register_search_provider("first", a)
        register_search_provider("second", b)
        reg = get_registered_search_providers()
        assert reg == {"first": a, "second": b}
        assert reg is not get_registered_search_providers()


class TestPrecomputeEmbeddingProviderMode:
    """In provider mode the search index auto-embeds the query, so Mirix must
    NOT precompute a query embedding (wasted embedding-model call per search)."""

    @pytest.mark.asyncio
    async def test_skips_embedding_when_search_provider_registered(self):
        from mirix.server.rest_api import _precompute_embedding_for_search

        register_search_provider("ips_search", object())

        # agent_state is irrelevant in provider mode — the guard returns before
        # touching it, so a sentinel that would explode if used proves the
        # embedding model is never invoked.
        class _Boom:
            @property
            def embedding_config(self):  # pragma: no cover - must not run
                raise AssertionError("embedding must not be computed in provider mode")

        emb, padded = await _precompute_embedding_for_search("embedding", "some query", _Boom())
        assert emb is None
        assert padded is None

    @pytest.mark.asyncio
    async def test_non_embedding_method_short_circuits(self):
        from mirix.server.rest_api import _precompute_embedding_for_search

        # No provider, but bm25 still returns (None, None) without embedding.
        emb, padded = await _precompute_embedding_for_search("bm25", "some query", object())
        assert emb is None
        assert padded is None


class TestRegistryLastWins:
    def test_multiple_relational_only_last_active(self):
        p1, p2 = object(), object()
        register_relational_provider("first", p1)
        register_relational_provider("second", p2)
        assert get_relational_provider() is p2
        assert get_registered_relational_providers() == {"first": p1, "second": p2}

    def test_multiple_search_only_last_active(self):
        p1, p2 = object(), object()
        register_search_provider("first", p1)
        register_search_provider("second", p2)
        assert get_search_provider() is p2
        assert get_registered_search_providers() == {"first": p1, "second": p2}


class TestEmptyRegistry:
    def test_get_relational_provider_none_when_empty(self):
        # Never latched (no provider ever registered) -> pure-PG fallback.
        assert get_relational_provider() is None

    def test_get_search_provider_none_when_empty(self):
        assert get_search_provider() is None


class TestProviderModeLatch:
    """Once provider mode latches, a missing provider must fail closed.

    Regression coverage for the shutdown teardown race (VEPAGE-1400): the IPS
    providers were unregistered while saves were still in flight, so
    get_relational_provider() returned None and managers silently took the
    PostgreSQL ORM path — reading/writing the wrong store under provider mode
    (surfacing as spurious NoResultFound and "No child memory agents found").
    """

    def test_missing_provider_raises_after_latch(self):
        register_relational_provider("ips_relational", object())
        unregister_relational_provider("ips_relational")
        with pytest.raises(RelationalProviderRequiredError):
            get_relational_provider()

    def test_no_latch_no_raise_pure_pg(self):
        # A process that never registers a provider keeps the ORM fallback.
        assert get_relational_provider() is None

    def test_error_classifies_transient(self):
        from mirix.errors import ProviderTransientError
        from mirix.queue.error_policy import Bucket, classify

        register_relational_provider("ips_relational", object())
        unregister_relational_provider("ips_relational")
        try:
            get_relational_provider()
        except RelationalProviderRequiredError as e:
            assert isinstance(e, ProviderTransientError)
            assert classify(e) is Bucket.TRANSIENT
        else:
            pytest.fail("expected RelationalProviderRequiredError")

    def test_reregister_after_latch_returns_provider(self):
        register_relational_provider("ips_relational", object())
        unregister_relational_provider("ips_relational")
        # Next startup re-registers; the call succeeds again (no permanent break).
        p2 = object()
        register_relational_provider("ips_relational", p2)
        assert get_relational_provider() is p2
