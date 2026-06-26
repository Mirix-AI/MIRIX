"""Read-path organization scoping.

Users are GLOBAL (not org-scoped). On a read, the org used to filter memories
must come from the CALLING CLIENT, not from the global user row — read back
through IPS-R the user's org relationship is not hydrated onto the scalar
field, so it falls to the all-zeros default org and filters out every
document (the LongMemEval "search returns 0 results" bug).

These tests pin the invariant: every per-type memory manager invoked by
``retrieve_memories_by_keywords`` receives a ``user`` whose ``organization_id``
is the client's org, regardless of the org the user row carried.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mirix.server.rest_api import retrieve_memories_by_keywords

pytestmark = pytest.mark.asyncio

# A user row whose org differs from the calling client's — the exact shape the
# bug produced (global user defaulting to the all-zeros org).
STALE_USER_ORG = "org-00000000-0000-4000-8000-000000000000"
CLIENT_ORG = "org-01c87164-4b25-5e69-b59f-66a5fe45888e"
USER_ID = "longmemeval-e47becba"


def _make_user(org_id: str) -> MagicMock:
    user = MagicMock()
    user.id = USER_ID
    user.organization_id = org_id
    user.timezone = "UTC"
    return user


def _make_client(org_id: str) -> MagicMock:
    client = MagicMock()
    client.id = "client-93a58771"
    client.organization_id = org_id
    client.read_scopes = ["customer-oriented-default"]
    return client


def _empty_manager() -> MagicMock:
    """A memory manager whose list/count calls return empty results.

    Each captures the kwargs it was invoked with so the test can assert the
    ``user`` (and therefore its org) handed down to the backend search.
    """
    mgr = MagicMock()
    mgr.list_episodic_memory = AsyncMock(return_value=[])
    mgr.list_semantic_items = AsyncMock(return_value=[])
    mgr.list_resources = AsyncMock(return_value=[])
    mgr.list_procedures = AsyncMock(return_value=[])
    mgr.list_knowledge = AsyncMock(return_value=[])
    mgr.get_total_number_of_items = AsyncMock(return_value=0)
    mgr.get_blocks = AsyncMock(return_value=[])
    return mgr


def _make_server(user: MagicMock) -> MagicMock:
    server = MagicMock()
    server.user_manager.get_user_by_id = AsyncMock(return_value=user)
    shared = _empty_manager()
    server.episodic_memory_manager = shared
    server.semantic_memory_manager = shared
    server.resource_memory_manager = shared
    server.procedural_memory_manager = shared
    server.knowledge_vault_manager = shared
    server.block_manager = shared
    return server, shared


def _captured_user_orgs(manager: MagicMock) -> list:
    """Collect the org of every ``user=`` kwarg passed to the manager."""
    orgs = []
    for call in manager.list_episodic_memory.call_args_list + (
        manager.list_semantic_items.call_args_list
        + manager.list_resources.call_args_list
        + manager.list_procedures.call_args_list
        + manager.list_knowledge.call_args_list
        + manager.get_total_number_of_items.call_args_list
    ):
        user = call.kwargs.get("user")
        if user is not None:
            orgs.append(user.organization_id)
    return orgs


async def test_read_uses_client_org_not_stale_user_org():
    """The bug repro: user row carries the all-zeros org; the read must still
    filter by the client's org so it matches what writes stamped."""
    user = _make_user(STALE_USER_ORG)
    server, shared = _make_server(user)
    client = _make_client(CLIENT_ORG)

    await retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=USER_ID,
        agent_state=MagicMock(),
        key_words="user's graduation degree",
        limit=10,
    )

    orgs = _captured_user_orgs(shared)
    assert orgs, "expected at least one memory-manager list call"
    assert all(o == CLIENT_ORG for o in orgs), f"every read must be scoped to the client's org; saw {set(orgs)}"
    assert STALE_USER_ORG not in orgs


async def test_read_org_override_is_unconditional_when_client_has_org():
    """Even when the user already has a (different, non-zero) org, the client's
    org wins — users are global, the caller's tenancy decides the read scope."""
    user = _make_user("org-some-other-tenant")
    server, shared = _make_server(user)
    client = _make_client(CLIENT_ORG)

    await retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=USER_ID,
        agent_state=MagicMock(),
        key_words="anything",
        limit=10,
    )

    orgs = _captured_user_orgs(shared)
    assert orgs
    assert all(o == CLIENT_ORG for o in orgs)


async def test_read_preserves_user_org_when_client_org_missing():
    """Defensive: a client with no org must not clobber the user's org to None
    (which would re-introduce the empty-filter failure mode)."""
    user = _make_user(CLIENT_ORG)
    server, shared = _make_server(user)
    client = _make_client(None)

    await retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=USER_ID,
        agent_state=MagicMock(),
        key_words="anything",
        limit=10,
    )

    orgs = _captured_user_orgs(shared)
    assert orgs
    assert all(o == CLIENT_ORG for o in orgs)
    assert None not in orgs


async def test_recent_episodic_bucket_capped_at_recent_window():
    """The topic-agnostic 'recent' episodic bucket (the no-query call) is capped
    at settings.conversation_recent_window, while the relevance-ranked 'relevant'
    bucket (the query call) keeps the caller's full limit."""
    from mirix.settings import settings

    user = _make_user(CLIENT_ORG)
    server, shared = _make_server(user)
    client = _make_client(CLIENT_ORG)

    await retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=USER_ID,
        agent_state=MagicMock(),
        key_words="anything",  # non-empty → both recent and relevant calls fire
        limit=10,
    )

    # The recent call is the episodic list with NO query kwarg; the relevant
    # call passes query=. Separate them by that signal.
    recent_limits = [
        c.kwargs.get("limit") for c in shared.list_episodic_memory.call_args_list if not c.kwargs.get("query")
    ]
    relevant_limits = [
        c.kwargs.get("limit") for c in shared.list_episodic_memory.call_args_list if c.kwargs.get("query")
    ]

    assert recent_limits, "expected a recent (no-query) episodic call"
    assert all(lim == settings.conversation_recent_window for lim in recent_limits)
    # The relevant bucket is unaffected — keeps the caller's full limit.
    assert relevant_limits and all(lim == 10 for lim in relevant_limits)
