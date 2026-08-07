"""Unit tests for CreateAgent's optional deterministic `id` field (ECMS-522
task 5 -- race-safety prerequisite for the shared
get_or_create_topic_extraction_agent helper's deterministic-id + conflict-catch
mechanism, task 6).

_create_agent has two backends -- IPS-Relational (when a relational provider
is configured) and a Postgres-fallback ORM path. Both must thread a
caller-supplied `agent_create.id` into the created row instead of always
generating a fresh UUID, while leaving the other 7 agent types (which never
pass this field) unaffected.
"""

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.agent import AgentType, CreateAgent
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager


def _make_actor():
    return Client(
        id="client-1",
        organization_id="org-1",
        name="Test Client",
        status="active",
    )


def _make_llm_config():
    return LLMConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        context_window=8192,
    )


class TestCreateAgentIdFieldIpsRelationalPath:
    @pytest.mark.asyncio
    async def test_supplied_id_is_persisted(self):
        am = AgentManager()
        fake_rp = AsyncMock()
        captured = {}

        async def _fake_create(table, data_dict):
            captured.update(data_dict)
            return {
                "id": data_dict["id"],
                "name": data_dict["name"],
                "system": data_dict["system"],
                "agent_type": data_dict["agent_type"],
                "llm_config": data_dict["llm_config"],
                "embedding_config": data_dict["embedding_config"],
                "organization_id": data_dict["organization_id"],
                "tools": data_dict["tools"],
                "tool_rules": data_dict["tool_rules"],
                "parent_id": data_dict["parent_id"],
            }

        fake_rp.create = AsyncMock(side_effect=_fake_create)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=fake_rp,
        ):
            agent_state = await am.create_agent(
                agent_create=CreateAgent(
                    id="agent-topic-extraction-client-1",
                    name="client-1_topic_extraction_agent",
                    agent_type=AgentType.topic_extraction_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        assert captured["id"] == "agent-topic-extraction-client-1"
        assert agent_state.id == "agent-topic-extraction-client-1"

    @pytest.mark.asyncio
    async def test_omitted_id_still_generates_a_fresh_uuid(self):
        """No behavior change for the other 7 agent types, which never pass
        `id` -- a fresh id is generated exactly as before this field existed.
        """
        am = AgentManager()
        fake_rp = AsyncMock()
        captured = {}

        async def _fake_create(table, data_dict):
            captured.update(data_dict)
            return {
                "id": data_dict["id"],
                "name": data_dict["name"],
                "system": data_dict["system"],
                "agent_type": data_dict["agent_type"],
                "llm_config": data_dict["llm_config"],
                "embedding_config": data_dict["embedding_config"],
                "organization_id": data_dict["organization_id"],
                "tools": data_dict["tools"],
                "tool_rules": data_dict["tool_rules"],
                "parent_id": data_dict["parent_id"],
            }

        fake_rp.create = AsyncMock(side_effect=_fake_create)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=fake_rp,
        ):
            agent_state = await am.create_agent(
                agent_create=CreateAgent(
                    name="core-memory-1",
                    agent_type=AgentType.core_memory_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        # A valid UUID was generated -- no fixed/deterministic id leaked in.
        assert uuid.UUID(captured["id"])
        assert agent_state.id == captured["id"]


class _FakeAsyncSession:
    """Minimal stand-in for the AsyncSession the ORM fallback path uses --
    just enough surface for _create_agent's body to run without a real DB.
    """

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


class TestCreateAgentIdFieldOrmFallbackPath:
    @pytest.mark.asyncio
    async def test_supplied_id_is_passed_to_agent_model(self):
        """On the Postgres-fallback path (no relational provider configured),
        a caller-supplied id must be threaded into AgentModel(...) instead of
        letting the ORM's own `default=lambda: f"agent-{uuid.uuid4()}"` fire.
        """
        am = AgentManager()
        fake_session = _FakeAsyncSession()

        @asynccontextmanager
        async def _fake_session_maker():
            yield fake_session

        am.session_maker = _fake_session_maker

        captured_kwargs = {}
        fake_agent_instance = MagicMock()
        fake_agent_instance.id = "agent-topic-extraction-client-1"
        fake_agent_instance.create_with_redis = AsyncMock(return_value=fake_agent_instance)
        fake_agent_instance.to_pydantic = MagicMock(return_value=MagicMock(id="agent-topic-extraction-client-1"))

        def _fake_agent_model(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_agent_instance

        with (
            patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
            patch("mirix.services.agent_manager.AgentModel", side_effect=_fake_agent_model),
            patch("mirix.services.agent_manager._process_relationship", new=AsyncMock()),
        ):
            await am.create_agent(
                agent_create=CreateAgent(
                    id="agent-topic-extraction-client-1",
                    name="client-1_topic_extraction_agent",
                    agent_type=AgentType.topic_extraction_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        assert captured_kwargs.get("id") == "agent-topic-extraction-client-1"

    @pytest.mark.asyncio
    async def test_omitted_id_lets_orm_default_fire(self):
        """No behavior change for the other 7 agent types: when `id` is not
        supplied, AgentModel(...) is not given an `id` kwarg at all, so the
        ORM column's own `default=lambda: f"agent-{uuid.uuid4()}"` fires,
        exactly as before this field existed.
        """
        am = AgentManager()
        fake_session = _FakeAsyncSession()

        @asynccontextmanager
        async def _fake_session_maker():
            yield fake_session

        am.session_maker = _fake_session_maker

        captured_kwargs = {}
        fake_agent_instance = MagicMock()
        fake_agent_instance.id = "agent-generated-by-orm-default"
        fake_agent_instance.create_with_redis = AsyncMock(return_value=fake_agent_instance)
        fake_agent_instance.to_pydantic = MagicMock(return_value=MagicMock(id="agent-generated-by-orm-default"))

        def _fake_agent_model(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_agent_instance

        with (
            patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
            patch("mirix.services.agent_manager.AgentModel", side_effect=_fake_agent_model),
            patch("mirix.services.agent_manager._process_relationship", new=AsyncMock()),
        ):
            await am.create_agent(
                agent_create=CreateAgent(
                    name="core-memory-1",
                    agent_type=AgentType.core_memory_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        assert "id" not in captured_kwargs
