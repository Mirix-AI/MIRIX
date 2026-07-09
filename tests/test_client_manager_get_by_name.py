"""Tests for ClientManager.get_client_by_name.

The exists-check in ECMS's validate_or_create_client previously scanned the
first page of list_clients() (50 rows, unordered, cross-org) to decide whether
a client already exists — once the clients table grew past one page the scan
missed pre-existing rows and minted duplicates. get_client_by_name is the
replacement: an exact name+organization lookup that is correct regardless of
table size.
"""

import pytest

from mirix.database.relational_provider import (
    get_registered_relational_providers,
    register_relational_provider,
    reset_provider_mode_latch,
    unregister_relational_provider,
)
from mirix.services.client_manager import ClientManager

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def clean_relational_registry():
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()
    yield
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()


class FakeRelationalProvider:
    """Duck-typed relational provider capturing list() calls."""

    def __init__(self, rows):
        self.rows = rows
        self.list_calls = []

    async def list(self, table, **kwargs):
        self.list_calls.append((table, kwargs))
        return self.rows


CLIENT_ROW = {
    "id": "client-abc123",
    "name": "human_events_client",
    "organization_id": "org-1",
    "status": "active",
    "write_scope": "human-events",
    "read_scopes": ["human-events"],
    "message_set_retention_count": 0,
}


async def test_provider_path_returns_matching_client():
    provider = FakeRelationalProvider([CLIENT_ROW])
    register_relational_provider("fake", provider)

    result = await ClientManager().get_client_by_name("human_events_client", organization_id="org-1")

    assert result is not None
    assert result.id == "client-abc123"
    assert result.name == "human_events_client"
    assert result.organization_id == "org-1"


async def test_provider_path_filters_by_name_and_org_server_side():
    """The lookup must push name + organization_id down to the provider as
    filters rather than scanning a page client-side."""
    provider = FakeRelationalProvider([CLIENT_ROW])
    register_relational_provider("fake", provider)

    await ClientManager().get_client_by_name("human_events_client", organization_id="org-1")

    assert len(provider.list_calls) == 1
    table, kwargs = provider.list_calls[0]
    assert table == "clients"
    assert kwargs.get("name") == "human_events_client"
    assert kwargs.get("organization_id") == "org-1"


async def test_provider_path_returns_none_when_absent():
    provider = FakeRelationalProvider([])
    register_relational_provider("fake", provider)

    result = await ClientManager().get_client_by_name("no_such_client", organization_id="org-1")

    assert result is None
