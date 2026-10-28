"""Deleting a client API key really revokes it.

``AsyncSession.delete`` is a coroutine. ``ClientManager.delete_client_api_key``
called it without ``await``: the DELETE was never flushed, the commit was a
no-op, ``DELETE /clients/{id}/api-keys/{key}`` answered success and the key
kept authenticating. These tests run on an isolated database, plus a static
scan that no AsyncSession coroutine is left un-awaited anywhere in ``mirix/``.
"""

import ast
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException

from db_isolation import isolated_database
from mirix.orm.errors import NoResultFound
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.security.api_keys import generate_api_key
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager

pytestmark = [pytest.mark.asyncio(loop_scope="module")]

MIRIX_ROOT = Path(__file__).resolve().parent.parent / "mirix"


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def database(tmp_path_factory):
    async with isolated_database(tmp_path_factory.mktemp("api-key-db")) as engine:
        yield engine


async def _client_with_key(org_id: str):
    manager = ClientManager()
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    await manager.create_client(
        PydanticClient(
            id=client_id,
            name=client_id,
            organization_id=org_id,
            write_scope="test",
            read_scopes=["test"],
        )
    )
    raw_key = generate_api_key()
    record = await manager.create_client_api_key(client_id, raw_key)
    return client_id, raw_key, record.id


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def org_id():
    org_id = f"org-{uuid.uuid4().hex[:8]}"
    await OrganizationManager().create_organization(
        PydanticOrganization(id=org_id, name="API key org")
    )
    return org_id


class TestManager:
    async def test_a_deleted_key_no_longer_authenticates(self, org_id):
        manager = ClientManager()
        client_id, raw_key, key_id = await _client_with_key(org_id)
        assert (await manager.get_client_by_api_key(raw_key)).id == client_id

        await manager.delete_client_api_key(key_id)

        assert await manager.get_client_by_api_key(raw_key) is None
        assert key_id not in {
            k.id for k in await manager.list_client_api_keys(client_id)
        }
        with pytest.raises(NoResultFound):
            await manager.delete_client_api_key(key_id)  # really gone

    async def test_another_clients_key_is_not_deleted(self, org_id):
        manager = ClientManager()
        _, raw_key, key_id = await _client_with_key(org_id)
        other_client, _, _ = await _client_with_key(org_id)

        with pytest.raises(NoResultFound):
            await manager.delete_client_api_key(key_id, client_id=other_client)

        assert await manager.get_client_by_api_key(raw_key) is not None

    async def test_the_owning_client_scope_deletes(self, org_id):
        manager = ClientManager()
        client_id, raw_key, key_id = await _client_with_key(org_id)
        await manager.delete_client_api_key(key_id, client_id=client_id)
        assert await manager.get_client_by_api_key(raw_key) is None


def _token(client_id: str, org_id: str) -> str:
    """A real dashboard JWT of ``client_id`` (no monkeypatched auth)."""
    from mirix.services.admin_user_manager import ClientAuthManager

    return "Bearer " + ClientAuthManager.create_access_token(
        PydanticClient(
            id=client_id,
            name=client_id,
            organization_id=org_id,
            write_scope="test",
            read_scopes=["test"],
        )
    )


class TestEndpoint:
    """The api-key endpoints act for the caller's own client only — the
    path's client is chosen by the caller, so a token of another client (of
    the same or another organization) gets a 404 and changes nothing."""

    @pytest.fixture
    def server(self, monkeypatch):
        from types import SimpleNamespace

        from mirix.server import rest_api

        fake = SimpleNamespace(client_manager=ClientManager())
        monkeypatch.setattr(rest_api, "get_server", lambda: fake)
        return rest_api

    async def test_delete_revokes_only_with_the_owners_token(self, org_id, server):
        manager = ClientManager()
        victim, raw_key, key_id = await _client_with_key(org_id)
        attacker, _, _ = await _client_with_key(org_id)

        with pytest.raises(HTTPException) as exc:
            await server.delete_client_api_key(
                victim, key_id, authorization=_token(attacker, org_id)
            )
        assert exc.value.status_code == 404
        assert await manager.get_client_by_api_key(raw_key) is not None

        # The owner's token naming another client's path: still 404.
        with pytest.raises(HTTPException) as exc:
            await server.delete_client_api_key(
                attacker, key_id, authorization=_token(victim, org_id)
            )
        assert exc.value.status_code == 404
        assert await manager.get_client_by_api_key(raw_key) is not None

        body = await server.delete_client_api_key(
            victim, key_id, authorization=_token(victim, org_id)
        )
        assert body["id"] == key_id
        assert await manager.get_client_by_api_key(raw_key) is None

    async def test_create_and_list_need_the_owners_token(self, org_id, server):
        victim, _, key_id = await _client_with_key(org_id)
        attacker, _, _ = await _client_with_key(org_id)
        request = server.CreateApiKeyRequest(name="minted")
        for call in (
            server.create_client_api_key(
                victim, request, authorization=_token(attacker, org_id)
            ),
            server.list_client_api_keys(victim, authorization=_token(attacker, org_id)),
        ):
            with pytest.raises(HTTPException) as exc:
                await call
            assert exc.value.status_code == 404
        listed = await server.list_client_api_keys(
            victim, authorization=_token(victim, org_id)
        )
        assert [k.id for k in listed] == [key_id]  # nothing was minted
        created = await server.create_client_api_key(
            victim, request, authorization=_token(victim, org_id)
        )
        assert created.client_id == victim

    async def test_no_or_a_forged_token_is_401(self, org_id, server):
        victim, _, key_id = await _client_with_key(org_id)
        for authorization in (None, "Bearer not-a-jwt"):
            with pytest.raises(HTTPException) as exc:
                await server.delete_client_api_key(
                    victim, key_id, authorization=authorization
                )
            assert exc.value.status_code == 401


# --------------------------------------------------------------------------- #
# Static scan: AsyncSession coroutines are always awaited                      #
# --------------------------------------------------------------------------- #

# AsyncSession methods that are coroutines (a bare call does nothing).
_SESSION_COROUTINES = {
    "close",
    "commit",
    "connection",
    "delete",
    "execute",
    "flush",
    "get",
    "merge",
    "refresh",
    "rollback",
    "run_sync",
    "scalar",
    "scalars",
    "stream",
}
# Names MIRIX code gives AsyncSession objects.
_SESSION_NAMES = {"session", "db", "db_session", "own_db", "read_db", "async_session"}


def _unawaited_session_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef,)):
            continue
        for stmt in ast.walk(node):
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            func = stmt.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _SESSION_COROUTINES
                and isinstance(func.value, ast.Name)
                and func.value.id in _SESSION_NAMES
            ):
                yield f"{path.relative_to(MIRIX_ROOT.parent)}:{stmt.lineno} {func.value.id}.{func.attr}(...)"


def test_no_async_session_call_is_left_unawaited():
    offenders = [
        hit
        for path in sorted(MIRIX_ROOT.rglob("*.py"))
        for hit in _unawaited_session_calls(path)
    ]
    assert offenders == []
