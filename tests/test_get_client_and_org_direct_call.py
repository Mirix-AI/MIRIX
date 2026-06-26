"""get_client_and_org must tolerate being called as a plain function.

ECMS imports MIRIX route handlers and calls them directly (not through
FastAPI's HTTP layer), so the ``Header(None)`` defaults are never resolved.
A caller that omits ``x_org_id`` passes the ``Header(None)`` FieldInfo
sentinel — which is truthy — so ``x_org_id or DEFAULT_ORG_ID`` would yield
the sentinel instead of the default org, later failing @enforce_types
deep in the call (500 on save).

get_client_and_org normalizes any non-str sentinel back to None so the
default-org fallback fires. Regression test for that normalization.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import Header

from mirix.server.rest_api import get_client_and_org

DEFAULT_ORG = "org-00000000-0000-4000-8000-000000000000"


def _fake_server():
    server = MagicMock()
    server.organization_manager.DEFAULT_ORG_ID = DEFAULT_ORG
    return server


@pytest.mark.asyncio
async def test_omitted_x_org_id_header_sentinel_falls_back_to_default_org():
    # Reproduce the direct-call shape: x_org_id is the unresolved Header(None)
    # FieldInfo object, which is truthy and is NOT a str.
    header_sentinel = Header(None)
    assert not isinstance(header_sentinel, str)
    assert bool(header_sentinel) is True  # this is why the naive `or` failed

    with patch("mirix.server.rest_api.get_server", return_value=_fake_server()):
        client_id, org_id = await get_client_and_org(
            x_client_id="some-client",
            x_org_id=header_sentinel,
        )

    assert client_id == "some-client"
    assert org_id == DEFAULT_ORG  # not the FieldInfo sentinel


@pytest.mark.asyncio
async def test_explicit_org_id_is_honored():
    with patch("mirix.server.rest_api.get_server", return_value=_fake_server()):
        client_id, org_id = await get_client_and_org(
            x_client_id="some-client",
            x_org_id="org-explicit",
        )
    assert client_id == "some-client"
    assert org_id == "org-explicit"


@pytest.mark.asyncio
async def test_all_header_sentinels_normalize_to_unauthorized():
    # If client id, org id, AND api key are all unresolved Header sentinels,
    # the function must treat them as absent and raise the 401 rather than
    # silently using a FieldInfo as one of them.
    from fastapi import HTTPException

    with patch("mirix.server.rest_api.get_server", return_value=_fake_server()):
        with pytest.raises(HTTPException) as exc_info:
            await get_client_and_org(
                x_client_id=Header(None),
                x_org_id=Header(None),
                x_api_key=Header(None),
            )
    assert exc_info.value.status_code == 401
