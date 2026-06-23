"""Unit tests for AsyncServer.ensure_defaults(skip_seed_writes=...).

Verifies the read-replica startup path: when skip_seed_writes=True, the seed
WRITES (org/admin-user/default-client/base-tools) are skipped while the
provider-override READ still runs. Defaults preserve the full-seed behavior.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mirix.server.server import AsyncServer


def _server_with_mocked_managers() -> AsyncServer:
    """Build a bare AsyncServer-like object with the managers ensure_defaults
    touches replaced by AsyncMocks, without running the real __init__ (which
    needs a DB)."""
    srv = AsyncServer.__new__(AsyncServer)
    srv._pending_defaults = True
    srv._provider_overrides_loaded = False
    srv.default_org = None
    srv.admin_user = None
    srv.default_client = None
    srv.organization_manager = SimpleNamespace(create_default_organization=AsyncMock(return_value="org"))
    srv.user_manager = SimpleNamespace(create_admin_user=AsyncMock(return_value="user"))
    srv.client_manager = SimpleNamespace(create_default_client=AsyncMock(return_value="client"))
    srv.tool_manager = SimpleNamespace(upsert_base_tools=AsyncMock(return_value=[]))
    srv._load_provider_overrides = AsyncMock()
    return srv


@pytest.mark.asyncio
async def test_skip_seed_writes_skips_all_writes_but_loads_overrides():
    srv = _server_with_mocked_managers()

    await srv.ensure_defaults(skip_seed_writes=True)

    srv.organization_manager.create_default_organization.assert_not_called()
    srv.user_manager.create_admin_user.assert_not_called()
    srv.client_manager.create_default_client.assert_not_called()
    srv.tool_manager.upsert_base_tools.assert_not_called()
    # The provider-override READ must still run so LLM keys resolve on readers.
    srv._load_provider_overrides.assert_awaited_once()
    # _pending_defaults is left untouched (writer still owns seeding).
    assert srv._pending_defaults is True


@pytest.mark.asyncio
async def test_default_runs_full_seed():
    srv = _server_with_mocked_managers()

    await srv.ensure_defaults()  # default skip_seed_writes=False

    srv.organization_manager.create_default_organization.assert_awaited_once()
    srv.user_manager.create_admin_user.assert_awaited_once()
    srv.client_manager.create_default_client.assert_awaited_once()
    srv.tool_manager.upsert_base_tools.assert_awaited_once()
    srv._load_provider_overrides.assert_awaited_once()
    assert srv._pending_defaults is False
