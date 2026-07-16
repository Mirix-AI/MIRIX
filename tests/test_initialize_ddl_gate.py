"""ensure_tables_created must be skipped when a relational provider is registered.

Under IPS (provider registered) the schema is provisioned out-of-band, so the
create_all DDL — the one remaining hard PG dependency at startup — must not run.
"""
from unittest.mock import AsyncMock, patch

import pytest

from mirix.database.relational_provider import (
    get_relational_provider,
    get_registered_relational_providers,
    register_relational_provider,
    reset_provider_mode_latch,
    unregister_relational_provider,
)


@pytest.fixture
def _clear_provider():
    # Save/restore the registry so this test is isolated from others. The
    # registry is a module-global dict + active-provider name + a latch (see
    # mirix/database/relational_provider.py), not a single variable, so we
    # snapshot/restore all three pieces of state directly.
    import mirix.database.relational_provider as rp

    saved_providers = dict(rp._relational_providers)
    saved_active = rp._active_provider_name
    saved_latch = rp._provider_mode_latched

    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()

    yield

    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    rp._relational_providers = saved_providers
    rp._active_provider_name = saved_active
    rp._provider_mode_latched = saved_latch


@pytest.mark.asyncio
async def test_ddl_skipped_when_relational_provider_registered(_clear_provider):
    register_relational_provider("test_provider", object())
    assert get_relational_provider() is not None

    with patch(
        "mirix.server.rest_api.ensure_tables_created", new_callable=AsyncMock
    ) as mock_ddl, patch("mirix.server.rest_api.get_server") as mock_get_server, patch(
        "mirix.server.rest_api.get_redis_client", return_value=None, create=True
    ):
        mock_get_server.return_value.ensure_defaults = AsyncMock()
        from mirix.server.rest_api import initialize

        await initialize()

    mock_ddl.assert_not_called()


@pytest.mark.asyncio
async def test_ddl_runs_when_no_relational_provider(_clear_provider):
    assert get_relational_provider() is None

    with patch(
        "mirix.server.rest_api.ensure_tables_created", new_callable=AsyncMock
    ) as mock_ddl, patch("mirix.server.rest_api.get_server") as mock_get_server, patch(
        "mirix.server.rest_api.get_redis_client", return_value=None, create=True
    ):
        mock_get_server.return_value.ensure_defaults = AsyncMock()
        from mirix.server.rest_api import initialize

        await initialize()

    mock_ddl.assert_called_once()
