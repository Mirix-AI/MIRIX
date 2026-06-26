"""Tests that AddMemoryRequest accepts direct_writes and add_memory threads them to put_messages.

The payload shape matches the target LLM-facing tool signature exactly
(e.g. ``{"items": [...]}`` for ``episodic_memory_insert``) so the meta-agent
can route to the tool without a translation shim.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _episodic_payload(**overrides):
    """Build a valid direct-write payload for the episodic handler."""
    item = {
        "event_type": "e",
        "summary": "s",
        "details": "d",
        "actor": "system",
        "occurred_at": "2026-04-17T10:00:00Z",
    }
    item.update(overrides)
    return {"items": [item]}


@pytest.mark.asyncio
async def test_add_memory_request_accepts_direct_writes():
    """AddMemoryRequest constructs with direct_writes=[DirectWriteInput(...)]."""
    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        messages=[],
        direct_writes=[
            DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
        ],
    )
    assert req.direct_writes is not None
    assert len(req.direct_writes) == 1
    assert req.direct_writes[0].memory_type == "episodic"
    assert req.direct_writes[0].payload["items"][0]["event_type"] == "e"
    assert req.direct_writes[0].payload["items"][0]["actor"] == "system"


@pytest.mark.asyncio
async def test_add_memory_request_direct_writes_defaults_to_none():
    """direct_writes is optional and defaults to None when omitted."""
    from mirix.server.rest_api import AddMemoryRequest

    req = AddMemoryRequest(meta_agent_id="agent-1", messages=[])
    assert req.direct_writes is None


@pytest.mark.asyncio
async def test_add_memory_threads_direct_writes_to_put_messages():
    """add_memory(...) with direct_writes set passes them as a list of dicts to put_messages."""
    from mirix.server.rest_api import (
        AddMemoryRequest,
        DirectWriteInput,
        add_memory,
    )

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        user_id="user-1",
        messages=[],  # mutually exclusive with direct_writes — must be empty
        filter_tags={"k": "v"},
        external_id="ext-1",
        source_type="engagement",
        source_system="test-system",
        source_metadata={"display_id": "D1"},
        direct_writes=[
            DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
        ],
    )

    # The save path performs no client or meta-agent lookup before queuing. The
    # server mock's lookup methods raise if called, proving the path stays
    # lookup-free.
    fake_server = MagicMock()
    fake_server.client_manager = MagicMock()
    fake_server.client_manager.get_client_by_id = AsyncMock(
        side_effect=AssertionError("get_client_by_id must not be called on the save path")
    )
    fake_server.agent_manager = MagicMock()
    fake_server.agent_manager.get_agent_by_id = AsyncMock(
        side_effect=AssertionError("get_agent_by_id must not be called on the save path")
    )

    with (
        patch("mirix.server.rest_api.get_server", return_value=fake_server),
        patch(
            "mirix.server.rest_api.get_client_and_org",
            new_callable=AsyncMock,
            return_value=("client-1", "org-1"),
        ),
        patch("mirix.server.rest_api.put_messages", new_callable=AsyncMock) as mock_put,
    ):
        result = await add_memory(req, x_org_id="org-1", x_client_id="client-1")

    assert result["success"] is True
    mock_put.assert_awaited_once()
    fake_server.client_manager.get_client_by_id.assert_not_called()
    fake_server.agent_manager.get_agent_by_id.assert_not_called()
    kwargs = mock_put.call_args.kwargs

    # agent_id comes straight from the request; actor carries only the client id.
    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["actor"].id == "client-1"
    # Scope is owned by the worker now — the API must not bake one in.
    assert "scope" not in (kwargs.get("filter_tags") or {})

    assert "direct_writes" in kwargs
    direct_writes = kwargs["direct_writes"]
    assert len(direct_writes) == 1
    assert direct_writes[0]["memory_type"] == "episodic"
    items = direct_writes[0]["payload"]["items"]
    assert len(items) == 1
    assert items[0]["event_type"] == "e"
    assert items[0]["actor"] == "system"
    assert items[0]["occurred_at"] == "2026-04-17T10:00:00Z"


@pytest.mark.asyncio
async def test_add_memory_with_empty_messages_and_direct_writes_does_not_crash():
    """add_memory must not crash when messages=[] and only direct_writes are provided.

    ECMS sends requests with an empty messages list for pure direct-write flows.
    The flattening block in add_memory historically indexed message[0] without a
    guard, causing IndexError. This test enforces the empty-messages guard.
    """
    from mirix.server.rest_api import (
        AddMemoryRequest,
        DirectWriteInput,
        add_memory,
    )

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        user_id="user-1",
        messages=[],
        direct_writes=[
            DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
        ],
    )

    fake_server = MagicMock()
    fake_server.client_manager = MagicMock()
    fake_server.agent_manager = MagicMock()

    with (
        patch("mirix.server.rest_api.get_server", return_value=fake_server),
        patch(
            "mirix.server.rest_api.get_client_and_org",
            new_callable=AsyncMock,
            return_value=("client-1", "org-1"),
        ),
        patch("mirix.server.rest_api.put_messages", new_callable=AsyncMock) as mock_put,
    ):
        await add_memory(req, x_org_id="org-1", x_client_id="client-1")

    kwargs = mock_put.call_args.kwargs
    assert kwargs.get("direct_writes") is not None


@pytest.mark.asyncio
async def test_direct_write_episodic_payload_missing_item_field_raises_validation():
    """Payload with an item missing a required episodic field fails at
    AddMemoryRequest construction with a ValidationError (→ HTTP 422 via FastAPI),
    not later at the worker.
    """
    from pydantic import ValidationError

    from mirix.server.rest_api import DirectWriteInput

    with pytest.raises(ValidationError):
        DirectWriteInput(
            memory_type="episodic",
            payload={"items": [{"event_type": "e", "summary": "s"}]},  # missing details + actor + occurred_at
        )


@pytest.mark.asyncio
async def test_direct_write_unsupported_memory_type_rejected_at_request_time():
    """A memory_type with no registered item schema is rejected up front."""
    from pydantic import ValidationError

    from mirix.server.rest_api import DirectWriteInput

    with pytest.raises(ValidationError):
        DirectWriteInput(
            memory_type="telepathic",
            payload={"items": [{"anything": "goes"}]},
        )


@pytest.mark.asyncio
async def test_direct_write_payload_missing_items_list_rejected():
    """Payload without an 'items' list (or empty) is rejected at validation."""
    from pydantic import ValidationError

    from mirix.server.rest_api import DirectWriteInput

    with pytest.raises(ValidationError):
        DirectWriteInput(memory_type="episodic", payload={})

    with pytest.raises(ValidationError):
        DirectWriteInput(memory_type="episodic", payload={"items": []})


@pytest.mark.asyncio
async def test_direct_write_episodic_payload_happy_path_coerces_to_dict():
    """Valid payload survives validation and preserves its shape."""
    from mirix.server.rest_api import DirectWriteInput

    dw = DirectWriteInput(memory_type="episodic", payload=_episodic_payload())
    assert isinstance(dw.payload, dict)
    assert "items" in dw.payload
    assert dw.payload["items"][0]["event_type"] == "e"
    assert dw.payload["items"][0]["occurred_at"] == "2026-04-17T10:00:00Z"


@pytest.mark.asyncio
async def test_direct_write_semantic_payload_validated():
    """Semantic memory type accepts SemanticMemoryItemBase items."""
    from mirix.server.rest_api import DirectWriteInput

    dw = DirectWriteInput(
        memory_type="semantic",
        payload={
            "items": [
                {
                    "name": "quarterly revenue target",
                    "summary": "Board set target to $2.5M.",
                    "details": "Q2 revenue target set by board on 2026-04-17.",
                    "source": "board-meeting",
                }
            ]
        },
    )
    assert dw.payload["items"][0]["name"] == "quarterly revenue target"


@pytest.mark.asyncio
async def test_direct_writes_rejects_non_empty_messages():
    """direct_writes + non-empty messages → ValidationError."""
    from pydantic import ValidationError

    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    with pytest.raises(ValidationError, match="mutually exclusive with messages"):
        AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[{"role": "user", "content": "hi"}],
            direct_writes=[
                DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
            ],
        )


@pytest.mark.asyncio
async def test_direct_writes_rejects_summary():
    """direct_writes + summary → ValidationError."""
    from pydantic import ValidationError

    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    with pytest.raises(ValidationError, match="mutually exclusive with summary"):
        AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[],
            summary="a client-provided summary",
            direct_writes=[
                DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
            ],
        )


@pytest.mark.asyncio
async def test_direct_writes_rejects_summarize_true():
    """direct_writes + summarize=True → ValidationError."""
    from pydantic import ValidationError

    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    with pytest.raises(ValidationError, match="mutually exclusive with summarize=True"):
        AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[],
            summarize=True,
            direct_writes=[
                DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
            ],
        )


@pytest.mark.asyncio
async def test_direct_writes_with_empty_messages_summarize_false_no_summary_is_allowed():
    """Positive case: the mutually-exclusive validator does not reject a
    well-formed direct-write request (messages=[], summary=None, summarize=False).
    """
    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        messages=[],
        summarize=False,  # explicit default; equivalent to leaving it off
        direct_writes=[
            DirectWriteInput(memory_type="episodic", payload=_episodic_payload()),
        ],
    )
    assert req.summary is None
    assert req.summarize is False
    assert len(req.direct_writes) == 1


@pytest.mark.asyncio
async def test_derived_path_still_allows_messages_summary_summarize():
    """Regression: when direct_writes is None, the derived path's existing
    combinations (messages + summary, messages + summarize) keep working.
    """
    from mirix.server.rest_api import AddMemoryRequest

    # messages + summarize=True (classic opt-in summary path)
    req1 = AddMemoryRequest(
        meta_agent_id="agent-1",
        messages=[{"role": "user", "content": "hi"}],
        summarize=True,
    )
    assert req1.summarize is True

    # messages + client-provided summary
    req2 = AddMemoryRequest(
        meta_agent_id="agent-1",
        messages=[{"role": "user", "content": "hi"}],
        summary="client wrote this",
    )
    assert req2.summary == "client wrote this"
