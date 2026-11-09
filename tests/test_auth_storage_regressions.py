import asyncio
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mirix.orm.base import Base
from mirix.orm.organization import Organization
from mirix.schemas.client import Client
from mirix.services.admin_user_manager import ClientAuthManager
from mirix.services.client_manager import ClientManager


@pytest_asyncio.fixture
async def managers(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context():
        async with factory() as session:
            yield session

    async with context() as session:
        session.add(Organization(id="auth-org", name="Auth tests"))
        await session.commit()
    auth, clients = ClientAuthManager(), ClientManager()
    auth.session_maker = clients.session_maker = context
    yield auth, clients
    await engine.dispose()


@pytest.mark.asyncio
async def test_read_only_api_key_removes_write_permission_without_changing_client(managers):
    _, manager = managers
    await manager.create_client(Client(
        id="auth-client", name="Client", organization_id="auth-org",
        write_scope="admin", read_scopes=["admin"],
    ))
    await manager.create_client_api_key(
        "auth-client", "test-only-read-key", permission="read_only",
    )
    assert (await manager.get_client_by_api_key("test-only-read-key")).write_scope is None
    assert (await manager.get_client_by_id("auth-client")).write_scope == "admin"


@pytest.mark.asyncio
async def test_internal_password_hash_is_retained_for_login(managers):
    auth, clients = managers
    await clients.create_client(Client(
        id="auth-client", name="Client", organization_id="auth-org",
        email="login@example.invalid",
        password_hash=auth.hash_password("test-only-password"),
    ))
    client, token, status = await auth.authenticate("login@example.invalid", "test-only-password")
    assert status == "ok" and token and client.password_hash


@pytest.mark.asyncio
async def test_only_one_concurrent_bootstrap_registration_succeeds(managers):
    auth, _ = managers
    results = await asyncio.gather(*[
        auth.register_client_for_dashboard(
            name=f"Bootstrap {n}", email=f"bootstrap{n}@example.invalid",
            password="test-only-password", organization_id="auth-org", bootstrap_only=True,
        )
        for n in range(2)
    ], return_exceptions=True)
    assert sum(isinstance(result, Client) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1, [
        (type(result).__name__, str(result)) for result in results if isinstance(result, Exception)
    ]
    assert not await auth.is_first_dashboard_user()
