"""
Integration test for the top-level session_id field: end-to-end round-trip
through MessageManager against real Postgres.

Verifies:
- A message created with session_id persists that column.
- list_messages_for_agent(session_id=X) returns only messages for session X.
- list_messages_for_agent(session_id=None) returns all messages for that agent.
- Different session_ids correctly isolate messages.

Requires the docker-compose Postgres (port 5433). Run:
    pytest tests/test_session_id_integration.py -v -m integration
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    # Round-trip needs real Postgres; SQLite via aiosqlite throws MissingGreenlet
    # in a fresh sync session. Mirrors tests/test_agent_trigger_state_integration.py.
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="session_id round-trip needs Postgres (set MIRIX_PG_URI)",
    ),
]

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest_asyncio.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.user import User as PydanticUser
from mirix.services.message_manager import MessageManager


@pytest.fixture
def message_manager():
    return MessageManager()


@pytest_asyncio.fixture(scope="module")
async def test_actor():
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    org_id = f"test-session-id-org-{uuid.uuid4().hex[:8]}"
    try:
        await org_mgr.get_organization_by_id(org_id)
    except Exception:
        await org_mgr.create_organization(
            PydanticOrganization(id=org_id, name="Session ID Test Org")
        )

    client_id = f"test-session-id-client-{uuid.uuid4().hex[:8]}"
    try:
        return await client_mgr.get_client_by_id(client_id)
    except Exception:
        return await client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Session ID Test Client",
                write_scope="test-sid",
                read_scopes=["test-sid"],
            )
        )


@pytest_asyncio.fixture(scope="module")
async def test_user(test_actor):
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()
    user_id = f"test-session-id-user-{uuid.uuid4().hex[:8]}"
    try:
        return await user_mgr.get_user_by_id(user_id)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="Session ID Test User",
                organization_id=test_actor.organization_id,
                timezone="UTC",
            )
        )


@pytest_asyncio.fixture(scope="module")
async def test_agent(test_actor):
    """Create a minimal agent we can attach messages to.

    create_agent rejects null llm_config/embedding_config, so supply
    placeholder configs (we never invoke the model — only persist messages).
    Pattern mirrors tests/test_redis_integration.py::test_agent.
    """
    from mirix.schemas.agent import AgentType, CreateAgent
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.services.agent_manager import AgentManager

    agent_mgr = AgentManager()
    agent = await agent_mgr.create_agent(
        agent_create=CreateAgent(
            name=f"test-session-agent-{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.chat_agent,
            description="Test agent for session_id round-trip",
            system=None,
            llm_config=LLMConfig(
                model="gpt-4",
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com",
                context_window=8192,
            ),
            embedding_config=EmbeddingConfig(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
                embedding_dim=1536,
            ),
        ),
        actor=test_actor,
    )
    return agent


async def _create_msg(mgr, actor, agent, text, session_id):
    return await mgr.create_message(
        pydantic_msg=PydanticMessage(
            agent_id=agent.id,
            role=MessageRole.user,
            content=[TextContent(text=text)],
            session_id=session_id,
        ),
        actor=actor,
        use_cache=False,
    )


class TestSessionIdRoundTrip:
    async def test_persists_and_filters_by_session_id(
        self, message_manager, test_actor, test_agent
    ):
        sid_a = f"sess-a-{uuid.uuid4().hex[:6]}"
        sid_b = f"sess-b-{uuid.uuid4().hex[:6]}"

        a1 = await _create_msg(message_manager, test_actor, test_agent, "A1", sid_a)
        a2 = await _create_msg(message_manager, test_actor, test_agent, "A2", sid_a)
        b1 = await _create_msg(message_manager, test_actor, test_agent, "B1", sid_b)
        null_msg = await _create_msg(
            message_manager, test_actor, test_agent, "no-session", None
        )

        # Session A returns only A messages.
        got_a = await message_manager.list_messages_for_agent(
            agent_id=test_agent.id,
            actor=test_actor,
            session_id=sid_a,
            limit=100,
            use_cache=False,
        )
        ids_a = {m.id for m in got_a}
        assert a1.id in ids_a
        assert a2.id in ids_a
        assert b1.id not in ids_a
        assert null_msg.id not in ids_a

        # Session B returns only B.
        got_b = await message_manager.list_messages_for_agent(
            agent_id=test_agent.id,
            actor=test_actor,
            session_id=sid_b,
            limit=100,
            use_cache=False,
        )
        ids_b = {m.id for m in got_b}
        assert b1.id in ids_b
        assert a1.id not in ids_b
        assert null_msg.id not in ids_b

        # No filter returns everything we created (and the session_id column round-trips).
        all_msgs = await message_manager.list_messages_for_agent(
            agent_id=test_agent.id,
            actor=test_actor,
            limit=100,
            use_cache=False,
        )
        by_id = {m.id: m for m in all_msgs}
        assert by_id[a1.id].session_id == sid_a
        assert by_id[b1.id].session_id == sid_b
        assert by_id[null_msg.id].session_id is None
