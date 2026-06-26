"""The worker is the authority for write scope.

The worker derives "scope" from the client (resolved by client_id on dequeue),
overwrites any scope present on the queue message, and rejects a client that has
no write_scope. These are pure-unit tests that drive
QueueWorker._process_message_async with the DB/user/agent layers mocked.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.protobuf.struct_pb2 import Struct

from mirix.errors import ProviderPermanentError
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import QueueWorker


def _build_message(client_id="client-1", scope_on_message=None):
    msg = QueueMessage()
    msg.client_id = client_id
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    if scope_on_message is not None:
        ft = Struct()
        ft["scope"] = scope_on_message
        msg.filter_tags.CopyFrom(ft)
    return msg


def _make_worker(write_scope):
    """Worker whose client_manager returns a client with the given write_scope
    and whose send_messages is a no-op spy. Returns (worker, send_spy)."""
    actor = SimpleNamespace(id="client-1", organization_id="org-1", write_scope=write_scope)
    user = SimpleNamespace(id="user-1", organization_id="org-1")

    server = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=actor)
    # Return a MagicMock so the worker's post-call usage.model_dump() log works.
    send_spy = AsyncMock(return_value=MagicMock())
    server.send_messages = send_spy

    worker = QueueWorker(queue=MagicMock(), server=server)
    return worker, send_spy, user


async def _run(worker, user, message):
    # Patch the collaborators _process_message_async touches that need real infra.
    with (
        patch("mirix.queue.worker.UserManager") as mock_um,
        patch("mirix.queue.worker.reconcile_user_org_to_actor", side_effect=lambda u, a: u),
        patch("mirix.queue.worker.restore_trace_from_queue_message", return_value=False),
        patch("mirix.queue.worker.get_langfuse_client", return_value=None),
        patch("mirix.queue.worker.get_trace_context", return_value=None),
    ):
        mock_um.return_value.get_user_by_id = AsyncMock(return_value=user)
        mock_um.return_value.get_admin_user = AsyncMock(return_value=user)
        await worker._process_message_async(message)


@pytest.mark.asyncio
async def test_worker_injects_scope_from_authoritative_client():
    """send_messages receives scope == the client's write_scope."""
    worker, send_spy, user = _make_worker(write_scope="client-real-scope")
    await _run(worker, user, _build_message())

    send_spy.assert_awaited_once()
    assert send_spy.call_args.kwargs["filter_tags"]["scope"] == "client-real-scope"


@pytest.mark.asyncio
async def test_worker_overwrites_forged_inbound_scope():
    """A scope that rode in on the queue message is ignored — the worker
    overwrites it with the authoritative client's write_scope."""
    worker, send_spy, user = _make_worker(write_scope="client-real-scope")
    await _run(worker, user, _build_message(scope_on_message="forged-admin-scope"))

    send_spy.assert_awaited_once()
    assert send_spy.call_args.kwargs["filter_tags"]["scope"] == "client-real-scope"


@pytest.mark.asyncio
async def test_worker_rejects_client_without_write_scope():
    """A client with no write_scope is a deterministic misconfiguration — the
    worker raises a permanent error (so it dead-letters, not retries) and never
    calls send_messages."""
    worker, send_spy, user = _make_worker(write_scope=None)

    with pytest.raises(ProviderPermanentError, match="no write_scope"):
        await _run(worker, user, _build_message())

    send_spy.assert_not_called()
