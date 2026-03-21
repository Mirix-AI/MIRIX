"""
Test ORM to_pydantic() conversions to ensure they don't trigger MissingGreenlet errors.

This test ensures that:
1. ORM models can be safely converted to Pydantic even when detached from session
2. Relationship access doesn't trigger lazy loading outside async context
3. Meta agent and memory manager flows work correctly
"""

import pytest
from sqlalchemy import select

from mirix.orm import Agent as AgentModel
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.knowledge_vault import KnowledgeVaultItem
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.orm.resource_memory import ResourceMemoryItem
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.schemas.llm_config import LLMConfig


@pytest.mark.asyncio
async def test_agent_to_pydantic_with_session(server):
    """Test Agent.to_pydantic() inside an async session."""
    from mirix.server.server import db_context

    # Get test client
    actor = server.default_client
    
    # Create an agent with tools
    from mirix.schemas.agent import CreateAgent
    agent_create = CreateAgent(
        name="test_agent_conversion",
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-004"),
        include_base_tools=True,
    )
    
    agent_state = await server.agent_manager.create_agent(
        agent_create=agent_create,
        actor=actor,
    )
    
    # Now fetch it back and convert inside session
    async with db_context() as session:
        agent = await AgentModel.read(
            db_session=session,
            identifier=agent_state.id,
            actor=actor,
        )
        
        # This should work - we're inside the session
        pydantic_agent = agent.to_pydantic()
        
        assert isinstance(pydantic_agent, PydanticAgentState)
        assert pydantic_agent.id == agent_state.id
        assert pydantic_agent.name == "test_agent_conversion"
        # tools should be present (loaded via selectin)
        assert isinstance(pydantic_agent.tools, list)


@pytest.mark.asyncio
async def test_agent_to_pydantic_detached(server):
    """Test Agent.to_pydantic() on a detached instance (session closed)."""
    from mirix.server.server import db_context

    actor = server.default_client
    
    # Create an agent
    from mirix.schemas.agent import CreateAgent
    agent_create = CreateAgent(
        name="test_agent_detached",
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-004"),
        include_base_tools=False,  # No tools to simplify
    )
    
    agent_state = await server.agent_manager.create_agent(
        agent_create=agent_create,
        actor=actor,
    )
    
    # Fetch agent and close session
    async with db_context() as session:
        agent = await AgentModel.read(
            db_session=session,
            identifier=agent_state.id,
            actor=actor,
        )
        # Session closes here
    
    # Now agent is detached - to_pydantic() should still work
    # It should use cached/loaded relationships or empty list
    pydantic_agent = agent.to_pydantic()
    
    assert isinstance(pydantic_agent, PydanticAgentState)
    assert pydantic_agent.id == agent_state.id
    assert pydantic_agent.name == "test_agent_detached"
    # tools might be empty list (not loaded) or loaded collection
    assert isinstance(pydantic_agent.tools, list)


@pytest.mark.asyncio
async def test_episodic_memory_to_pydantic(server):
    """Test EpisodicEvent.to_pydantic() doesn't trigger relationship loading."""
    from datetime import datetime

    from mirix.server.server import db_context

    actor = server.default_client
    user = server.admin_user
    
    # Create an episodic event
    from mirix.schemas.episodic_memory import EpisodicEvent as EpisodicEventCreate
    
    event_data = {
        "event_type": "test_event",
        "actor": "system",
        "summary": "Test event for conversion",
        "details": "Testing to_pydantic conversion safety",
        "occurred_at": datetime.now(),
        "user_id": user.id,
        "organization_id": user.organization_id,
    }
    
    event = await server.episodic_memory_manager.create_episodic_memory(
        agent_state=None,
        event=EpisodicEventCreate(**event_data),
        actor=actor,
    )
    
    # Fetch and convert inside session
    async with db_context() as session:
        result = await session.execute(
            select(EpisodicEvent).where(EpisodicEvent.id == event.id)
        )
        orm_event = result.scalar_one()
        
        # Convert inside session
        pydantic_event = orm_event.to_pydantic()
        assert isinstance(pydantic_event, PydanticEpisodicEvent)
        assert pydantic_event.id == event.id
        assert pydantic_event.summary == "Test event for conversion"
    
    # Now test after session closed
    async with db_context() as session:
        result = await session.execute(
            select(EpisodicEvent).where(EpisodicEvent.id == event.id)
        )
        orm_event = result.scalar_one()
    
    # Session closed - to_pydantic() should still work
    pydantic_event = orm_event.to_pydantic()
    assert isinstance(pydantic_event, PydanticEpisodicEvent)
    assert pydantic_event.id == event.id


@pytest.mark.asyncio
async def test_memory_models_to_pydantic(server):
    """Test all memory models' to_pydantic() methods work safely."""
    from mirix.server.server import db_context

    actor = server.default_client
    user = server.admin_user
    
    # Test semantic memory
    from mirix.schemas.semantic_memory import SemanticMemoryItem as SemanticCreate
    semantic = await server.semantic_memory_manager.create_semantic_item(
        agent_state=None,
        item=SemanticCreate(
            name="test_concept",
            summary="Test summary",
            details="Test details",
            source="test",
            user_id=user.id,
            organization_id=user.organization_id,
        ),
        actor=actor,
    )
    
    async with db_context() as session:
        result = await session.execute(
            select(SemanticMemoryItem).where(SemanticMemoryItem.id == semantic.id)
        )
        orm_semantic = result.scalar_one()
    
    # Detached conversion
    pydantic_semantic = orm_semantic.to_pydantic()
    assert pydantic_semantic.id == semantic.id
    assert pydantic_semantic.name == "test_concept"
    
    # Test procedural memory
    from mirix.schemas.procedural_memory import ProceduralMemoryItem as ProceduralCreate
    procedural = await server.procedural_memory_manager.create_procedure(
        agent_state=None,
        item=ProceduralCreate(
            summary="test_procedure",
            description="Test description",
            steps="Step 1\nStep 2",
            tags=["test"],
            user_id=user.id,
            organization_id=user.organization_id,
        ),
        actor=actor,
    )
    
    async with db_context() as session:
        result = await session.execute(
            select(ProceduralMemoryItem).where(ProceduralMemoryItem.id == procedural.id)
        )
        orm_procedural = result.scalar_one()
    
    pydantic_procedural = orm_procedural.to_pydantic()
    assert pydantic_procedural.id == procedural.id
    assert pydantic_procedural.summary == "test_procedure"
    
    # Test resource memory
    from mirix.schemas.resource_memory import ResourceMemoryItem as ResourceCreate
    resource = await server.resource_memory_manager.create_resource(
        agent_state=None,
        item=ResourceCreate(
            summary="test_resource",
            content="Resource content",
            resource_type="document",
            source="test",
            user_id=user.id,
            organization_id=user.organization_id,
        ),
        actor=actor,
    )
    
    async with db_context() as session:
        result = await session.execute(
            select(ResourceMemoryItem).where(ResourceMemoryItem.id == resource.id)
        )
        orm_resource = result.scalar_one()
    
    pydantic_resource = orm_resource.to_pydantic()
    assert pydantic_resource.id == resource.id
    assert pydantic_resource.summary == "test_resource"
    
    # Test knowledge vault
    from mirix.schemas.knowledge_vault import KnowledgeVaultItem as KnowledgeCreate
    knowledge = await server.knowledge_vault_manager.create_knowledge(
        agent_state=None,
        item=KnowledgeCreate(
            caption="test_knowledge",
            secret_value="Secret data",
            category="test",
            user_id=user.id,
            organization_id=user.organization_id,
        ),
        actor=actor,
    )
    
    async with db_context() as session:
        result = await session.execute(
            select(KnowledgeVaultItem).where(KnowledgeVaultItem.id == knowledge.id)
        )
        orm_knowledge = result.scalar_one()
    
    pydantic_knowledge = orm_knowledge.to_pydantic()
    assert pydantic_knowledge.id == knowledge.id
    assert pydantic_knowledge.caption == "test_knowledge"


@pytest.mark.asyncio
async def test_list_agents_conversion_safety(server):
    """Test list_agents flow (simulating meta agent initialization)."""
    actor = server.default_client
    
    # Create multiple agents
    from mirix.schemas.agent import CreateAgent
    agent_names = ["meta_agent", "episodic_agent", "semantic_agent"]
    
    for name in agent_names:
        await server.agent_manager.create_agent(
            agent_create=CreateAgent(
                name=name,
                llm_config=LLMConfig.default_config("gpt-4"),
                embedding_config=EmbeddingConfig.default_config("text-embedding-004"),
                include_base_tools=True,
            ),
            actor=actor,
        )
    
    # List agents (this is what MetaAgent does)
    agents = await server.agent_manager.list_agents(actor=actor)
    
    # Should have all our test agents
    assert len(agents) >= 3
    
    # All should be Pydantic models
    for agent in agents:
        assert isinstance(agent, PydanticAgentState)
        assert agent.id is not None
        assert isinstance(agent.tools, list)


@pytest.mark.asyncio
async def test_memory_manager_list_conversion(server):
    """Test memory manager list_* methods convert safely."""
    from datetime import datetime

    actor = server.default_client
    user = server.admin_user
    
    # Create test data
    from mirix.schemas.agent import CreateAgent
    agent = await server.agent_manager.create_agent(
        agent_create=CreateAgent(
            name="test_memory_agent",
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config("text-embedding-004"),
        ),
        actor=actor,
    )
    
    from mirix.schemas.episodic_memory import EpisodicEvent as EpisodicCreate
    await server.episodic_memory_manager.create_episodic_memory(
        agent_state=agent,
        event=EpisodicCreate(
            event_type="test",
            actor="system",
            summary="Test event",
            details="Test details",
            occurred_at=datetime.now(),
            user_id=user.id,
            organization_id=user.organization_id,
        ),
        actor=actor,
    )
    
    # List episodic memory (this is what memory tools do)
    events = await server.episodic_memory_manager.list_episodic_memory(
        agent_state=agent,
        user=user,
        query="",
        limit=10,
    )
    
    # Should have at least our test event
    assert len(events) >= 1
    assert all(isinstance(e, PydanticEpisodicEvent) for e in events)
