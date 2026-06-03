"""Integration test for skill-based procedural memory lifecycle."""
import pytest
from mirix.schemas.agent import AgentType


async def get_sub_agent(server, client, meta_agent, agent_type):
    """Helper to get a sub-agent by type from meta agent's children."""
    agents = await server.agent_manager.list_agents(parent_id=meta_agent.id, actor=client)
    for agent in agents:
        if agent.agent_type == agent_type:
            return agent
    raise ValueError(f"No agent found with type {agent_type}")


@pytest.mark.asyncio(loop_scope="module")
class TestSkillLifecycle:
    """Test the full skill lifecycle: insert -> search -> update -> search -> delete."""

    async def test_skill_insert_search_update_delete(self, server, client, user, meta_agent):
        """Full CRUD lifecycle for a skill."""
        procedural_agent = await get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)
        manager = server.procedural_memory_manager

        # INSERT
        skill = await manager.insert_procedure(
            agent_state=procedural_agent,
            agent_id=meta_agent.id,
            name="test-skill-lifecycle",
            description="A test skill for lifecycle validation",
            instructions="Step 1: Do something\nStep 2: Do something else",
            entry_type="workflow",
            triggers=["testing lifecycle"],
            examples=[{"input": "test", "output": "result"}],
            actor=client,
            organization_id=user.organization_id,
            user_id=user.id,
        )
        assert skill.id is not None
        assert skill.name == "test-skill-lifecycle"
        assert skill.version == "0.1.0"
        skill_id = skill.id

        # SEARCH by description
        results = await manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="lifecycle validation",
            search_field="description",
            search_method="bm25",
            limit=10,
        )
        assert any(r.id == skill_id for r in results)

        # UPDATE
        from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate
        updated = await manager.update_item(
            item_update=ProceduralMemoryItemUpdate(
                id=skill_id,
                description="Updated test skill description",
                version="0.1.1",
            ),
            user=user,
            actor=client,
        )
        assert updated.description == "Updated test skill description"
        assert updated.version == "0.1.1"

        # DELETE
        await manager.delete_procedure_by_id(procedure_id=skill_id, actor=client)

        # Verify deletion
        results = await manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="lifecycle validation",
            search_field="description",
            search_method="bm25",
            limit=10,
        )
        assert not any(r.id == skill_id for r in results)
        print("[OK] Full skill lifecycle test passed")
