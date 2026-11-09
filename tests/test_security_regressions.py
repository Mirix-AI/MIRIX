"""Exercise authentication through ASGI, including forged identity headers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from mirix.schemas.client import Client
from mirix.server import rest_api
from mirix.services.admin_user_manager import ClientAuthManager


@pytest_asyncio.fixture
async def api(monkeypatch):
    owner = Client(
        id="client-owner", name="Owner", organization_id="org-owner",
        write_scope="admin", read_scopes=["admin"],
        password_hash="private-password-hash",
    )
    foreign = Client(
        id="client-foreign", name="Foreign", organization_id="org-foreign",
        write_scope="admin", read_scopes=["admin"],
        password_hash="other-password-hash",
    )
    clients = {owner.id: owner, foreign.id: foreign}
    manager = SimpleNamespace(
        get_client_by_api_key=AsyncMock(
            side_effect=lambda key: owner if key == "valid-test-key" else None
        ),
        get_client_by_id=AsyncMock(side_effect=lambda cid: clients.get(cid)),
        list_clients=AsyncMock(return_value=list(clients.values())),
        create_client=AsyncMock(return_value=owner),
        update_client=AsyncMock(return_value=owner),
        delete_client_by_id=AsyncMock(),
    )
    server = SimpleNamespace(
        client_manager=manager,
        organization_manager=SimpleNamespace(DEFAULT_ORG_ID="default-org"),
        agent_manager=SimpleNamespace(list_agents=AsyncMock(return_value=[])),
        user_manager=SimpleNamespace(delete_user_by_id=AsyncMock()),
    )
    monkeypatch.setattr(rest_api, "get_server", lambda: server)
    transport = httpx.ASGITransport(app=rest_api.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, server, owner, foreign


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {},
    {"X-Client-ID": "client-owner", "X-Org-ID": "org-owner"},
    {"Authorization": "Bearer invalid", "X-Client-ID": "client-owner"},
    {"X-API-Key": "invalid", "X-Client-ID": "client-owner"},
])
@pytest.mark.parametrize(("method", "path", "body"), [
    ("GET", "/agents", None),
    ("GET", "/memory/search", None),
    ("PATCH", "/clients/client-owner", {"id": "client-owner", "name": "Changed"}),
    ("DELETE", "/clients/client-owner", None),
    ("POST", "/clients/create_or_get", {"client_id": "client-owner"}),
    ("GET", "/users/user-owner", None),
    ("DELETE", "/users/user-owner", None),
    ("GET", "/organizations", None),
    ("POST", "/debug/token_stats/reset", None),
])
async def test_private_routes_reject_unverified_identity(api, method, path, body, headers):
    http, server, _, _ = api
    response = await http.request(method, path, headers=headers, json=body)
    assert response.status_code == 401, response.text
    assert response.headers.get("www-authenticate")
    server.client_manager.update_client.assert_not_awaited()
    server.client_manager.delete_client_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_verified_key_overrides_spoofed_routing_headers(api):
    http, server, owner, _ = api
    response = await http.get("/agents", headers={
        "X-API-Key": "valid-test-key",
        "X-Client-ID": "client-foreign",
        "X-Org-ID": "org-foreign",
    })
    assert response.status_code == 200, response.text
    assert server.agent_manager.list_agents.await_args.kwargs["actor"].id == owner.id


@pytest.mark.asyncio
async def test_invalid_bearer_does_not_fall_back_to_key(api):
    http, _, _, _ = api
    response = await http.get("/agents", headers={
        "Authorization": "Bearer invalid", "X-API-Key": "valid-test-key",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "PATCH", "DELETE"])
async def test_client_routes_reject_another_clients_token(api, method):
    http, server, _, foreign = api
    token = ClientAuthManager.create_access_token(foreign)
    response = await http.request(
        method, "/clients/client-owner",
        headers={"Authorization": f"Bearer {token}"},
        json={"id": "client-owner", "name": "Changed"} if method == "PATCH" else None,
    )
    assert response.status_code == 404, response.text
    server.client_manager.update_client.assert_not_awaited()
    server.client_manager.delete_client_by_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "PATCH"])
async def test_client_responses_omit_password_hash(api, method):
    http, _, owner, _ = api
    token = ClientAuthManager.create_access_token(owner)
    response = await http.request(
        method, f"/clients/{owner.id}", headers={"Authorization": f"Bearer {token}"},
        json={"id": owner.id, "name": "Changed"} if method == "PATCH" else None,
    )
    assert response.status_code == 200, response.text
    assert "password_hash" not in response.json()
    assert owner.password_hash not in response.text


@pytest.mark.asyncio
async def test_create_or_get_response_omits_password_hash(api):
    http, _, owner, _ = api
    response = await http.post(
        "/clients/create_or_get", headers={"X-API-Key": "valid-test-key"},
        json={"client_id": owner.id},
    )
    assert response.status_code == 200, response.text
    assert "password_hash" not in response.json()


@pytest.mark.asyncio
async def test_inactive_client_cannot_use_existing_jwt(api):
    http, _, owner, _ = api
    token = ClientAuthManager.create_access_token(owner)
    owner.status = "inactive"
    response = await http.get(
        "/clients", headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_header_helper_rejects_identity_without_request(api):
    with pytest.raises(rest_api.HTTPException) as error:
        await rest_api.get_client_and_org("client-owner", "org-owner")
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_health_remains_public(api):
    response = await api[0].get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {"query": "test", "client_id": "client-foreign"},
    {"query": "test", "org_id": "org-foreign"},
])
async def test_search_parameters_cannot_change_authenticated_tenant(api, params):
    response = await api[0].get(
        "/memory/search_all_users", params=params, headers={"X-API-Key": "valid-test-key"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_user_routes_reject_foreign_org(api, method):
    http, server, _, _ = api
    server.user_manager.get_user_by_id = AsyncMock(
        return_value=SimpleNamespace(id="foreign-user", organization_id="org-foreign")
    )
    response = await http.request(
        method, "/users/foreign-user", headers={"X-API-Key": "valid-test-key"},
    )
    assert response.status_code == 404
    server.user_manager.delete_user_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_after_bootstrap_requires_authentication(api, monkeypatch):
    register = AsyncMock()
    monkeypatch.setattr(ClientAuthManager, "is_first_dashboard_user", AsyncMock(return_value=False))
    monkeypatch.setattr(ClientAuthManager, "register_client_for_dashboard", register)
    response = await api[0].post(
        "/admin/auth/register",
        json={"name": "Uninvited", "email": "test@example.invalid", "password": "test-only-password"},
    )
    assert response.status_code == 401
    register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_client_can_read_but_cannot_mutate(api):
    http, server, owner, _ = api
    owner.write_scope = None
    headers = {"X-API-Key": "valid-test-key"}
    assert (await http.get("/agents", headers=headers)).status_code == 200
    for method, path in [
        ("POST", "/agents/meta/initialize"),
        ("POST", "/memory/add"),
        ("POST", "/memory/auto_dream"),
        ("DELETE", "/memory/semantic/one"),
    ]:
        response = await http.request(method, path, json={}, headers=headers)
        assert response.status_code == 403, response.text
    server.client_manager.update_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_listing_is_own_and_never_contains_hashes(api):
    http, _, owner, foreign = api
    token = ClientAuthManager.create_access_token(owner)
    response = await http.get("/clients", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [owner.id]
    assert "password_hash" not in response.text
    assert foreign.id not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("organization_id", [None, ""])
async def test_user_listing_never_drops_tenant_filter_for_legacy_clients(api, organization_id):
    http, server, owner, _ = api
    owner.organization_id = organization_id
    server.admin_user_manager = SimpleNamespace(get_client_by_id=AsyncMock(return_value=owner))
    server.user_manager.list_users = AsyncMock(return_value=[])
    token = ClientAuthManager.create_access_token(owner)
    response = await http.get("/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert server.user_manager.list_users.await_args.kwargs["organization_id"] == "default-org"


@pytest.mark.asyncio
@pytest.mark.parametrize("organization_id", [None, ""])
async def test_dashboard_listing_never_drops_tenant_filter_for_legacy_clients(api, monkeypatch, organization_id):
    http, _, owner, _ = api
    owner.organization_id = organization_id
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(ClientAuthManager, "list_dashboard_clients", listing)
    token = ClientAuthManager.create_access_token(owner)
    response = await http.get("/admin/dashboard-clients", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert listing.await_args.kwargs["organization_id"] == "default-org"
