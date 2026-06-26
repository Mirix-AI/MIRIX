"""End-to-end MIRIX test: QueueMessage(direct_writes) → worker → server → agent → real DB.

Exercises the full worker-mediated direct-write pipeline within a single Python
process. The Kafka hop is excluded (we build the QueueMessage directly and call
worker._process_message_async on it). Real DB is used for:
  - memory_sources (written by _persist_memory_source)
  - memory_citations (written by _write_citation inside direct_write_episodic)
  - dedup on second submission with same external_id

episodic_memory_manager.insert_event is stubbed to avoid a real embedding call
(which would need a configured LLM provider) but the stub returns a deterministic
id so we can verify the citation row references it.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_ORG_ID = "direct-writes-worker-org"
TEST_CLIENT_ID = "direct-writes-worker-client"
TEST_USER_ID = "direct-writes-worker-user"


def _unique(prefix: str = "x") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Direct Writes Worker Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Direct Writes Worker User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


async def _get_actor():
    from mirix.services.client_manager import ClientManager

    return await ClientManager().get_client_by_id(TEST_CLIENT_ID)


async def _get_user():
    from mirix.services.user_manager import UserManager

    return await UserManager().get_user_by_id(TEST_USER_ID)


def _build_loaded_agent(actor, user, ep_memory_id: str):
    """Build an Agent instance with real source/citation managers + stub ep_mgr.

    Used by the patched server.load_agent. The agent runs the real _step path:
    _persist_memory_source (real DB) → direct-writes branch → handler
    → episodic_memory_manager.insert_event (stubbed to return ep_memory_id)
    → _write_citation (real DB).
    """
    from mirix.agent.agent import Agent
    from mirix.schemas.agent import AgentType
    from mirix.services.memory_citation_manager import MemoryCitationManager
    from mirix.services.memory_source_manager import MemorySourceManager
    from mirix.services.message_manager import MessageManager
    from mirix.services.source_message_manager import SourceMessageManager

    agent = Agent.__new__(Agent)

    agent_state = MagicMock()
    agent_state.id = "meta-agent-1"
    agent_state.parent_id = None
    agent_state.name = "meta_memory_agent"
    agent_state.agent_type = AgentType.meta_memory_agent
    agent_state.is_type = lambda t: t == AgentType.meta_memory_agent
    agent_state.organization_id = TEST_ORG_ID
    agent_state.llm_config = MagicMock()
    agent_state.llm_config.model = "gpt-4"
    agent.agent_state = agent_state

    # Actor/user context is populated by Agent.step but seed defaults here too.
    agent.actor = actor
    agent.user = user
    agent.user_id = user.id
    agent.filter_tags = {"scope": actor.write_scope or "test"}
    agent.block_filter_tags = None
    agent.use_cache = False
    agent.occurred_at = None
    agent.memory_source_id = None  # Populated by server._step from QueueMessage
    agent.direct_writes = None  # Populated by server._step from QueueMessage
    agent.external_id = None
    agent.external_thread_id = None
    agent.source_type = None
    agent.source_system = None
    agent.source_metadata = None
    agent.source_summary = None
    agent.source_summary_source = None
    agent.summarize = False
    agent.source_messages = None
    agent._block_scopes = [actor.write_scope or "test"]

    # Real source + citation managers — hit the DB
    agent.memory_source_manager = MemorySourceManager()
    agent.source_message_manager = SourceMessageManager()
    agent.message_manager = MessageManager()
    # Stubbed episodic manager — skip embeddings/agent_state setup
    agent.episodic_memory_manager = MagicMock()
    agent.episodic_memory_manager.insert_event = AsyncMock(return_value=SimpleNamespace(id=ep_memory_id))
    agent.episodic_memory_manager.create_episodic_memory = AsyncMock(return_value=SimpleNamespace(id=ep_memory_id))

    # Interface + streaming defaults so server._step doesn't explode on step_yield.
    agent.interface = MagicMock()
    agent.interface.streaming_mode = False
    agent.interface.step_yield = MagicMock(return_value=None)

    return agent


def _build_queue_message(
    *,
    client_id: str,
    user_id: str,
    meta_agent_id: str,
    memory_source_id: str,
    external_id: str,
    direct_write_payload: dict,
    source_system: str = "test-system",
    source_type: str = "engagement",
):
    """Build a QueueMessage with direct_writes set, no input_messages."""
    from google.protobuf.struct_pb2 import Struct

    from mirix.queue.message_pb2 import QueueMessage

    msg = QueueMessage()
    msg.client_id = client_id
    msg.user_id = user_id
    msg.agent_id = meta_agent_id
    msg.chaining = False
    msg.use_cache = False
    msg.memory_source_id = memory_source_id
    msg.external_id = external_id
    msg.source_system = source_system
    msg.source_type = source_type

    # filter_tags as Struct (scope gets injected by the server side)
    filter_tags = Struct()
    filter_tags["scope"] = "test"
    msg.filter_tags.CopyFrom(filter_tags)

    write = msg.direct_writes.add()
    write.memory_type = "episodic"
    write.payload_json = json.dumps(direct_write_payload)

    return msg


async def test_worker_direct_writes_writes_source_and_citation_to_db(monkeypatch):
    """QueueMessage → worker → server → meta-agent direct-write handler.

    End state in real DB:
      - 1 memory_sources row with the external_id
      - 1 memory_citations row pointing at the (stubbed) episodic memory
    """
    from mirix.queue.worker import QueueWorker
    from mirix.services.memory_citation_manager import MemoryCitationManager
    from mirix.services.memory_source_manager import MemorySourceManager
    from mirix.services.source_message_manager import SourceMessageManager

    actor = await _get_actor()
    user = await _get_user()

    memory_source_id = _unique("src")
    external_id = _unique("ext")
    ep_memory_id = _unique("ep")

    # Real server singleton — we patch only load_agent so the agent plumbing
    # uses our test-wired Agent instance with real managers.
    from mirix.server.rest_api import get_server

    server = get_server()

    loaded_agent = _build_loaded_agent(actor, user, ep_memory_id)
    original_load_agent = server.load_agent
    server.load_agent = AsyncMock(return_value=loaded_agent)

    # Worker needs access to the server and client_manager/user_manager lookups
    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

    queue_msg = _build_queue_message(
        client_id=TEST_CLIENT_ID,
        user_id=TEST_USER_ID,
        meta_agent_id="meta-agent-1",
        memory_source_id=memory_source_id,
        external_id=external_id,
        direct_write_payload={
            "items": [
                {
                    "event_type": "engagement_created",
                    "summary": "e2e summary",
                    "details": "e2e details",
                    "actor": "system",
                    "occurred_at": "2026-04-17T10:00:00Z",
                }
            ]
        },
    )

    try:
        # Route through dispatch_save (Option B): step() no longer finalizes
        # internally — the dispatcher calls finalize_source after the save
        # returns. This matches what both the numaflow path
        # (process_external_message) and the in-process _consume_loop do.
        from mirix.queue.error_policy import dispatch_save

        async def _run():
            await worker._process_message_async(queue_msg)

        await dispatch_save(_run, memory_source_id=memory_source_id)

        # 1 memory_source row for this external_id
        source_mgr = MemorySourceManager()
        page = await source_mgr.list_sources(organization_id=TEST_ORG_ID, client_id=TEST_CLIENT_ID, limit=100)
        matching = [s for s in page.items if s.external_id == external_id]
        assert len(matching) == 1, (
            f"Expected exactly 1 memory_source for external_id={external_id}, " f"got {len(matching)}: {matching}"
        )
        created_source = matching[0]
        assert created_source.id == memory_source_id
        assert (
            created_source.processing_complete is True
        ), "dispatch_save should have finalized SUCCESS after direct-write branch"

        # 1 citation row for the stubbed episodic memory id
        citations = await MemoryCitationManager().get_citations_for_memory(
            memory_type="episodic",
            memory_id=ep_memory_id,
        )
        assert len(citations) == 1, f"Expected 1 citation, got {len(citations)}"
        assert citations[0].memory_source_id == memory_source_id
        assert citations[0].memory_type == "episodic"
        assert citations[0].citation_type == "created"

        # Direct writes have no conversation turns — no source_messages rows
        src_page = await SourceMessageManager().get_messages_by_source_id(memory_source_id)
        assert len(src_page.items) == 0, (
            f"direct_writes must not persist placeholder or empty input as source_messages; "
            f"got {len(src_page.items)}"
        )

        # episodic_memory_manager.insert_event called exactly once with expected args
        loaded_agent.episodic_memory_manager.insert_event.assert_awaited_once()
        kwargs = loaded_agent.episodic_memory_manager.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "engagement_created"
        assert kwargs["summary"] == "e2e summary"
        assert kwargs["details"] == "e2e details"
        assert kwargs["event_actor"] == "system"

    finally:
        server.load_agent = original_load_agent


async def test_worker_direct_writes_is_idempotent_on_duplicate_external_id(monkeypatch):
    """Second QueueMessage with same external_id: no duplicate rows.

    Relies on the real DB's partial unique index on (client_id, external_id)
    enforced by memory_source_manager.create ON CONFLICT DO NOTHING.
    """
    from mirix.queue.worker import QueueWorker
    from mirix.server.rest_api import get_server
    from mirix.services.memory_citation_manager import MemoryCitationManager
    from mirix.services.memory_source_manager import MemorySourceManager

    actor = await _get_actor()
    user = await _get_user()

    memory_source_id_1 = _unique("src")
    memory_source_id_2 = _unique("src")  # Different memory_source_id, same external_id
    external_id = _unique("ext-dup")
    ep_memory_id_1 = _unique("ep")
    ep_memory_id_2 = _unique("ep")

    server = get_server()
    original_load_agent = server.load_agent

    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

    # ---------- First submission ----------
    agent_1 = _build_loaded_agent(actor, user, ep_memory_id_1)
    server.load_agent = AsyncMock(return_value=agent_1)

    msg_1 = _build_queue_message(
        client_id=TEST_CLIENT_ID,
        user_id=TEST_USER_ID,
        meta_agent_id="meta-agent-1",
        memory_source_id=memory_source_id_1,
        external_id=external_id,
        direct_write_payload={
            "items": [
                {
                    "event_type": "engagement_created",
                    "summary": "first",
                    "details": "first",
                    "actor": "system",
                    "occurred_at": "2026-04-17T10:00:00Z",
                }
            ]
        },
    )

    try:
        await worker._process_message_async(msg_1)

        # ---------- Second submission (duplicate external_id) ----------
        agent_2 = _build_loaded_agent(actor, user, ep_memory_id_2)
        server.load_agent = AsyncMock(return_value=agent_2)

        msg_2 = _build_queue_message(
            client_id=TEST_CLIENT_ID,
            user_id=TEST_USER_ID,
            meta_agent_id="meta-agent-1",
            memory_source_id=memory_source_id_2,
            external_id=external_id,
            direct_write_payload={
                "items": [
                    {
                        "event_type": "engagement_created",
                        "summary": "second",
                        "details": "second",
                        "actor": "system",
                        "occurred_at": "2026-04-17T10:00:00Z",
                    }
                ]
            },
        )
        await worker._process_message_async(msg_2)

        # Exactly 1 memory_source row for the external_id (ON CONFLICT DO NOTHING)
        source_mgr = MemorySourceManager()
        page = await source_mgr.list_sources(organization_id=TEST_ORG_ID, client_id=TEST_CLIENT_ID, limit=100)
        matching = [s for s in page.items if s.external_id == external_id]
        assert len(matching) == 1, f"Expected exactly 1 memory_source after duplicate submission, got {len(matching)}"

        # Second agent's handler must NOT have run (deduped before direct-write branch)
        agent_2.episodic_memory_manager.insert_event.assert_not_called()

        # First agent's citation is still there; no citation for agent_2's ep id
        citations_1 = await MemoryCitationManager().get_citations_for_memory(
            memory_type="episodic", memory_id=ep_memory_id_1
        )
        citations_2 = await MemoryCitationManager().get_citations_for_memory(
            memory_type="episodic", memory_id=ep_memory_id_2
        )
        assert len(citations_1) == 1
        assert len(citations_2) == 0

    finally:
        server.load_agent = original_load_agent
