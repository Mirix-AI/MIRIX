"""Unit tests for AgentType.topic_extraction_agent's participation in the
enum-gated dispatch machinery (ECMS-522).

Covers tasks 1-4 from docs/specs/ECMS-522/tasks.md:
1. AgentType.topic_extraction_agent enum member + round-trip.
2. derive_system_message returns "" without raising for the new type.
3. AgentManager.create_agent creates a tool-less agent for the new type.
4. create_meta_agent's agent_name_to_type map resolves the new type's name.

Plus a regression test caught in PR review (github.com/LiaoJianhe/MIRIX_Intuit#168):
update_meta_agent has its OWN separate agent_name_to_type map (used by
force_update=true), which did not get the topic_extraction_agent entry when
create_meta_agent's map did -- an unmapped name is silently skipped, so a
force_update call for a client whose meta_agent_config.agents list includes
topic_extraction_agent (every client, since ECMS-522 task 11) but doesn't
yet have the row would silently never create it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.agent import AgentType, CreateAgent, CreateMetaAgent, UpdateMetaAgent
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services.agent_manager import AgentManager
from mirix.services.helpers.agent_manager_helper import derive_system_message


class TestAgentTypeTopicExtractionEnum:
    def test_topic_extraction_agent_member_exists(self):
        assert AgentType.topic_extraction_agent == "topic_extraction_agent"

    def test_topic_extraction_agent_round_trips_through_value(self):
        assert AgentType("topic_extraction_agent") == AgentType.topic_extraction_agent


class TestDeriveSystemMessageForTopicExtraction:
    def test_returns_empty_string_without_raising(self):
        assert derive_system_message(AgentType.topic_extraction_agent) == ""


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


class TestCreateAgentToolLessForTopicExtraction:
    @pytest.mark.asyncio
    async def test_create_agent_attaches_no_tools(self):
        am = AgentManager()
        fake_rp = AsyncMock()

        async def _fake_create(table, data_dict):
            return {
                "id": "agent-topic-extraction-client-1",
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
                    name="client-1_topic_extraction_agent",
                    agent_type=AgentType.topic_extraction_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        assert agent_state.tools == []


class TestCreateMetaAgentResolvesTopicExtractionType:
    @pytest.mark.asyncio
    async def test_agent_name_to_type_map_resolves_topic_extraction_agent(self):
        """create_meta_agent's agent_name_to_type map must resolve the string
        "topic_extraction_agent" to AgentType.topic_extraction_agent -- an
        unmapped name is silently skipped (agent_manager.py's `if not
        agent_type: ... continue`), so the map entry is load-bearing, not
        cosmetic.
        """
        am = AgentManager()
        actor = _make_actor()

        created_agent_types = []

        # The real create_agent's return value only needs an `.id` attribute
        # for the parent_id bookkeeping the loop performs -- use a lightweight
        # stand-in rather than a full PydanticAgentState.
        class _FakeAgentState:
            def __init__(self, agent_type):
                self.id = f"agent-{agent_type.value}"
                self.agent_type = agent_type

        async def _fake_create_agent(agent_create, actor):
            created_agent_types.append(agent_create.agent_type)
            return _FakeAgentState(agent_create.agent_type)

        with (
            patch.object(am.tool_manager, "ensure_base_tools_exist", new=AsyncMock(return_value=[])),
            patch(
                "mirix.services.user_manager.UserManager.get_or_create_org_default_user",
                new=AsyncMock(
                    return_value=PydanticUser(
                        id="user-default", organization_id="org-1", name="default", timezone="UTC"
                    )
                ),
            ),
            patch.object(am, "create_agent", new=AsyncMock(side_effect=_fake_create_agent)),
        ):
            await am.create_meta_agent(
                meta_agent_create=CreateMetaAgent(
                    agents=["topic_extraction_agent"],
                    llm_config=_make_llm_config(),
                ),
                actor=actor,
            )

        # First call is always the meta_memory_agent parent; the second is the
        # one sub-agent from `agents=["topic_extraction_agent"]` -- it must
        # have resolved through the map, not been silently skipped.
        assert AgentType.topic_extraction_agent in created_agent_types


class TestUpdateMetaAgentResolvesTopicExtractionType:
    @pytest.mark.asyncio
    async def test_agent_name_to_type_map_resolves_topic_extraction_agent(self):
        """update_meta_agent's OWN agent_name_to_type map (separate from
        create_meta_agent's) must also resolve "topic_extraction_agent" --
        force_update=true reaches this map's `agents_to_create` branch for any
        client whose desired agents list includes a type it doesn't have yet.
        An unmapped name hits `if not agent_type: ... continue` and is
        silently skipped, so this entry is load-bearing exactly like
        create_meta_agent's, not cosmetic (the bug this test pins: the two
        maps had drifted -- create_meta_agent's had the entry, this one
        didn't).
        """
        am = AgentManager()
        actor = _make_actor()

        class _FakeMetaAgentState:
            id = "meta-agent-1"
            name = "meta_memory_agent"
            agent_type = AgentType.meta_memory_agent
            llm_config = _make_llm_config()
            embedding_config = None

        created_agent_types = []

        class _FakeAgentState:
            def __init__(self, agent_type):
                self.id = f"agent-{agent_type.value}"
                self.agent_type = agent_type

        async def _fake_create_agent(agent_create, actor):
            created_agent_types.append(agent_create.agent_type)
            return _FakeAgentState(agent_create.agent_type)

        with (
            patch.object(
                am,
                "get_agent_by_id",
                new=AsyncMock(return_value=_FakeMetaAgentState()),
            ),
            # No existing sub-agents -- topic_extraction_agent lands in
            # agents_to_create (desired - existing), the exact branch that
            # consults this map.
            patch.object(am, "list_agents", new=AsyncMock(return_value=[])),
            patch.object(am, "create_agent", new=AsyncMock(side_effect=_fake_create_agent)),
        ):
            await am.update_meta_agent(
                meta_agent_id="meta-agent-1",
                meta_agent_update=UpdateMetaAgent(agents=["topic_extraction_agent"]),
                actor=actor,
            )

        assert AgentType.topic_extraction_agent in created_agent_types, (
            "topic_extraction_agent was silently skipped by update_meta_agent's "
            "agent_name_to_type map -- it must resolve the same name "
            "create_meta_agent's map does."
        )
