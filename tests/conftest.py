"""
Shared test fixtures for Mirix.

Provides:
- Module-scoped engine reset (NullPool) so each test module gets fresh DB connections
- Session-scoped API key tied to a test client for integration tests
"""

import asyncio
import os
from typing import Optional

import pytest
import pytest_asyncio

from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.security.api_keys import generate_api_key
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.settings import settings


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _reset_engine_per_module():
    """Dispose and recreate the async engine with NullPool at the start of
    each test module so every module's event loop gets fresh DB connections.

    NullPool creates a new connection per session and closes it immediately,
    preventing stale connections from a previous module's (now-closed) loop.
    """
    import mirix.server.server as server_module

    if (
        hasattr(server_module, "engine")
        and server_module.engine is not None
        and "asyncpg" in str(server_module.engine.url)
    ):
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlalchemy.pool import NullPool

        await server_module.engine.dispose()
        _pg_uri = settings.mirix_pg_uri.replace("postgresql+pg8000://", "postgresql+asyncpg://").replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        server_module.engine = create_async_engine(_pg_uri, poolclass=NullPool, echo=settings.pg_echo)
        server_module.AsyncSessionLocal = async_sessionmaker(
            bind=server_module.engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    await server_module.ensure_tables_created()
    yield


TEST_ORG_ID = "demo-org"
TEST_CLIENT_ID = "demo-client-id"
TEST_ORG_NAME = "Demo Org"


async def _ensure_org(org_mgr: OrganizationManager, org_id: str, org_name: str):
    try:
        await org_mgr.get_organization_by_id(org_id)
    except Exception:
        await org_mgr.create_organization(PydanticOrganization(id=org_id, name=org_name))


async def _issue_key(client_id: str, org_id: str, client_mgr: ClientManager) -> str:
    api_key = generate_api_key()
    await client_mgr.set_client_api_key(client_id, api_key)
    return api_key


async def _create_client_and_key(client_id: str, org_id: str, org_name: Optional[str] = None) -> dict:
    """
    Create one test client and API key in the current event loop.
    Use this from async fixtures when you need multiple clients in the same loop
    (e.g. call twice for client_a and client_b) to avoid "another operation in progress".
    """
    org_mgr = OrganizationManager()
    client_mgr = ClientManager()
    await _ensure_org(org_mgr, org_id, org_name or TEST_ORG_NAME)
    try:
        await client_mgr.get_client_by_id(client_id)
    except Exception:
        await client_mgr.create_client(
            PydanticClient(
                id=client_id,
                name=f"Test Client {client_id}",
                organization_id=org_id,
                write_scope="test",
                read_scopes=["test"],
            )
        )
    api_key = await _issue_key(client_id, org_id, client_mgr)
    return {"api_key": api_key, "org_id": org_id, "client_id": client_id}


@pytest.fixture(scope="session")
def api_key_factory():
    """
    Factory to provision API keys for test clients.
    """

    def _create(client_id: str = TEST_CLIENT_ID, org_id: str = TEST_ORG_ID):
        result = asyncio.run(_create_client_and_key(client_id, org_id))
        os.environ["MIRIX_API_KEY"] = result["api_key"]
        os.environ.setdefault("MIRIX_API_URL", "http://localhost:8000")
        return result

    return _create


@pytest.fixture(scope="session")
def api_auth(api_key_factory):
    """Default API auth (single client) for tests that need only one key."""
    return api_key_factory()


@pytest.fixture(scope="module")
def isolate_api_key_env():
    """Temporarily clear MIRIX_API_KEY for header-based client tests."""
    previous_api_key = os.environ.pop("MIRIX_API_KEY", None)
    try:
        yield
    finally:
        if previous_api_key is not None:
            os.environ["MIRIX_API_KEY"] = previous_api_key


@pytest_asyncio.fixture(scope="module")
async def server():
    """Shared AsyncServer fixture for tests requiring direct server access."""
    from mirix.server.server import AsyncServer

    srv = AsyncServer()
    await srv.ensure_defaults()
    return srv
