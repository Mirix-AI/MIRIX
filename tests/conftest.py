"""
Shared test fixtures for Mirix.

Provides a session-scoped API key tied to a test client, so integration tests can
authenticate against the REST API without passing X-Client-ID.
"""

import os

import pytest

from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.security.api_keys import generate_api_key
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager

TEST_ORG_ID = "demo-org"
TEST_CLIENT_ID = "demo-client-id"
TEST_ORG_NAME = "Demo Org"


def _ensure_org(org_mgr: OrganizationManager, org_id: str, org_name: str):
    try:
        org_mgr.get_organization_by_id(org_id)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=org_id, name=org_name))


def _issue_key(client_id: str, org_id: str, client_mgr: ClientManager) -> str:
    api_key = generate_api_key()
    client_mgr.set_client_api_key(client_id, api_key)
    return api_key


@pytest.fixture(scope="session")
def api_key_factory():
    """
    Factory to provision API keys for test clients.
    """
    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    def _create(client_id: str = TEST_CLIENT_ID, org_id: str = TEST_ORG_ID):
        _ensure_org(org_mgr, org_id, TEST_ORG_NAME)
        try:
            client_mgr.get_client_by_id(client_id)
        except Exception:
            client_mgr.create_client(
                PydanticClient(
                    id=client_id,
                    name=f"Test Client {client_id}",
                    organization_id=org_id,
                    scope="read_write",
                )
            )
        api_key = _issue_key(client_id, org_id, client_mgr)
        os.environ["MIRIX_API_KEY"] = api_key
        os.environ.setdefault("MIRIX_API_URL", "http://localhost:8000")
        return {"api_key": api_key, "org_id": org_id, "client_id": client_id}

    return _create


@pytest.fixture(scope="session")
def api_auth(api_key_factory):
    """Default API auth (single client) for tests that need only one key."""
    return api_key_factory()
