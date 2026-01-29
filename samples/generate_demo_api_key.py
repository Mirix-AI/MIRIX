#!/usr/bin/env python
"""
Generate an API key for the demo client used in tests/integration runs.

Creates the org/client if they do not exist, then prints a fresh API key for:
  client_id: demo-client-id
  org_id:    demo-org
"""

import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.security.api_keys import generate_api_key
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager

ORG_ID = "demo-org"
ORG_NAME = "Demo Org"
CLIENT_ID = "demo-client-id"


def main():
    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    # Ensure org exists
    try:
        org_mgr.get_organization_by_id(ORG_ID)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=ORG_ID, name=ORG_NAME))

    # Ensure client exists
    try:
        client_mgr.get_client_by_id(CLIENT_ID)
    except Exception:
        client_mgr.create_client(
            PydanticClient(
                id=CLIENT_ID,
                name="Demo Client",
                organization_id=ORG_ID,
                scope="read_write",
            )
        )

    # Issue fresh API key (creates entry in client_api_keys table)
    api_key = generate_api_key()
    api_key_record = client_mgr.create_client_api_key(CLIENT_ID, api_key, name="Demo API Key")

    print("Client ID:     ", CLIENT_ID)
    print("Org ID:        ", ORG_ID)
    print("API Key:       ", api_key)
    print("API Key ID:    ", api_key_record.id)
    print("API Key Status:", api_key_record.status)
    print("\nExport and use MIRIX_API_KEY with this value in tests.")


if __name__ == "__main__":
    main()
