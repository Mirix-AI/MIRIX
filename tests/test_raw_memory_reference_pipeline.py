"""
Integration test for the complete raw_memory reference pipeline.

This test validates the end-to-end flow:
1. Create raw_memory items (screenshots)
2. Create episodic/semantic memories that reference them
3. Retrieve memories via Agent
4. Verify raw_memory_references are collected and returned
5. Verify memory details are fetched for frontend display
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from mirix.agent.agent import Agent
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.schemas.agent import AgentState
from mirix.schemas.user import User as PydanticUser
from mirix.server.server import db_context
from mirix.services.raw_memory_manager import RawMemoryManager


@pytest.fixture
def setup_test_environment():
    """Set up test environment with user, agent, and raw memories."""
    # Create test user
    test_user = PydanticUser(
        id=f"user-{uuid.uuid4()}",
        name="Test User",
        organization_id=f"org-{uuid.uuid4()}",
        timezone="UTC",
        status="active"
    )

    # Create raw memory manager
    raw_memory_manager = RawMemoryManager()

    # Create 3 raw memory items with different sources
    raw_memory_1 = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/path/screenshot1.png",
        source_app="Chrome",
        captured_at=datetime(2025, 12, 18, 10, 30, 0, tzinfo=timezone.utc),
        ocr_text="GitHub repository page showing MIRIX Phase 1 implementation",
        source_url="https://github.com/user/mirix",
    )

    raw_memory_2 = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/path/screenshot2.png",
        source_app="Safari",
        captured_at=datetime(2025, 12, 17, 15, 45, 0, tzinfo=timezone.utc),
        ocr_text="Python documentation explaining async/await patterns",
        source_url="https://docs.python.org/3/library/asyncio.html",
    )

    raw_memory_3 = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/path/screenshot3.png",
        source_app="Notion",
        captured_at=datetime(2025, 12, 16, 9, 0, 0, tzinfo=timezone.utc),
        ocr_text="Project planning notes for MIRIX memory architecture",
        source_url="notion.so/workspace/mirix-project",
    )

    yield {
        "user": test_user,
        "raw_memory_manager": raw_memory_manager,
        "raw_memories": [raw_memory_1, raw_memory_2, raw_memory_3],
    }

    # Cleanup
    with db_context() as session:
        # Delete test data
        session.query(EpisodicEvent).filter(
            EpisodicEvent.user_id == test_user.id
        ).delete()
        session.query(SemanticMemoryItem).filter(
            SemanticMemoryItem.user_id == test_user.id
        ).delete()
        session.commit()

    # Delete raw memories
    for rm in [raw_memory_1, raw_memory_2, raw_memory_3]:
        raw_memory_manager.delete_raw_memory(rm.id)


def test_raw_memory_creation_and_reference(setup_test_environment):
    """Test 1: Create raw_memory and reference it from episodic memory."""
    env = setup_test_environment
    test_user = env["user"]
    raw_memories = env["raw_memories"]

    # Create episodic memory with raw_memory_references
    with db_context() as session:
        episodic_event = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Reviewed MIRIX implementation on GitHub",
            details="Checked the Phase 1 implementation progress including raw_memory integration",
            event_type="research",
            occurred_at=datetime(2025, 12, 18, 10, 35, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_memories[0].id, raw_memories[1].id],  # Reference first two
            tree_path=["work", "coding", "mirix"]
        )
        session.add(episodic_event)
        session.commit()
        session.refresh(episodic_event)

        # Verify raw_memory_references are stored
        assert episodic_event.raw_memory_references is not None
        assert len(episodic_event.raw_memory_references) == 2
        assert raw_memories[0].id in episodic_event.raw_memory_references
        assert raw_memories[1].id in episodic_event.raw_memory_references

        stored_event_id = episodic_event.id

    # Verify we can retrieve and access the references
    with db_context() as session:
        retrieved_event = session.get(EpisodicEvent, stored_event_id)
        assert retrieved_event is not None
        assert len(retrieved_event.raw_memory_references) == 2

        # Verify we can fetch the referenced raw_memories
        raw_memory_manager = env["raw_memory_manager"]
        for ref_id in retrieved_event.raw_memory_references:
            raw_mem = raw_memory_manager.get_raw_memory_by_id(
                raw_memory_id=ref_id,
                user_id=test_user.id
            )
            assert raw_mem is not None
            assert raw_mem.source_app in ["Chrome", "Safari"]


def test_semantic_memory_with_raw_reference(setup_test_environment):
    """Test 2: Create semantic memory with raw_memory reference."""
    env = setup_test_environment
    test_user = env["user"]
    raw_memories = env["raw_memories"]

    # Create semantic memory with raw_memory_reference
    with db_context() as session:
        semantic_item = SemanticMemoryItem(
            id=f"semantic-{uuid.uuid4()}",
            name="Async/Await Patterns",
            summary="Python async programming patterns and best practices",
            details="Comprehensive guide on using asyncio, async/await syntax, and concurrent programming",
            source="Python Documentation",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_memories[1].id],  # Reference Safari screenshot
            tree_path=["knowledge", "python", "async"]
        )
        session.add(semantic_item)
        session.commit()
        session.refresh(semantic_item)

        # Verify reference
        assert semantic_item.raw_memory_references is not None
        assert len(semantic_item.raw_memory_references) == 1
        assert raw_memories[1].id in semantic_item.raw_memory_references

        stored_semantic_id = semantic_item.id

    # Verify retrieval
    with db_context() as session:
        retrieved_semantic = session.get(SemanticMemoryItem, stored_semantic_id)
        assert retrieved_semantic is not None
        assert len(retrieved_semantic.raw_memory_references) == 1

        # Fetch the raw_memory
        raw_memory_manager = env["raw_memory_manager"]
        raw_mem = raw_memory_manager.get_raw_memory_by_id(
            raw_memory_id=retrieved_semantic.raw_memory_references[0],
            user_id=test_user.id
        )
        assert raw_mem.source_app == "Safari"
        assert "Python" in raw_mem.ocr_text


def test_agent_memory_retrieval_collects_references(setup_test_environment):
    """Test 3: Agent retrieves memories and collects raw_memory_references."""
    env = setup_test_environment
    test_user = env["user"]
    raw_memories = env["raw_memories"]

    # Create episodic memories with different references
    with db_context() as session:
        event1 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="GitHub code review",
            details="Reviewed MIRIX implementation code on GitHub",
            event_type="research",
            occurred_at=datetime(2025, 12, 18, 10, 0, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_memories[0].id],
            tree_path=["work", "coding"]
        )

        event2 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Python async research",
            details="Studied Python asyncio documentation",
            event_type="learning",
            occurred_at=datetime(2025, 12, 17, 16, 0, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_memories[1].id],
            tree_path=["learning", "python"]
        )

        event3 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Project planning",
            details="Planned MIRIX memory architecture in Notion",
            event_type="planning",
            occurred_at=datetime(2025, 12, 16, 9, 30, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_memories[2].id],
            tree_path=["work", "planning"]
        )

        session.add_all([event1, event2, event3])
        session.commit()

        event_ids = [event1.id, event2.id, event3.id]

    # Initialize minimal Agent to test memory retrieval
    try:
        from mirix.agent.agent import Agent
        from mirix.schemas.agent_state import AgentState

        agent_state = AgentState(
            id=f"agent-{uuid.uuid4()}",
            name="chat_agent",
            topic="coding and learning",
            system_prompt="You are a helpful assistant",
            tools=[],
            user_id=test_user.id,
            organization_id=test_user.organization_id
        )

        # Create agent instance
        agent = Agent(
            agent_state=agent_state,
            user=test_user
        )

        # Call build_system_prompt_with_memories to trigger retrieval
        system_prompt, retrieved_memories = agent.build_system_prompt_with_memories(
            raw_system="Test system prompt",
            topics="MIRIX Python coding"
        )

        # Verify that raw_memory_refs were collected
        assert hasattr(agent, 'current_raw_memory_refs'), "Agent should have current_raw_memory_refs attribute"
        assert agent.current_raw_memory_refs is not None, "current_raw_memory_refs should not be None"
        assert len(agent.current_raw_memory_refs) > 0, "Should have collected some raw_memory_references"

        # Verify that the collected references are from our test data
        collected_ref_ids = set(agent.current_raw_memory_refs)
        expected_ref_ids = {rm.id for rm in raw_memories}

        # At least some of our raw_memories should be collected
        assert len(collected_ref_ids & expected_ref_ids) > 0, "Should have collected at least one test raw_memory reference"

        # Verify raw_memory_refs are in retrieved_memories dict
        assert "raw_memory_refs" in retrieved_memories, "retrieved_memories should contain raw_memory_refs key"
        assert retrieved_memories["raw_memory_refs"] == agent.current_raw_memory_refs

        # Verify source info is formatted in system prompt
        assert "[Sources:" in system_prompt or len(agent.current_raw_memory_refs) == 0, \
            "System prompt should include source information when memories have references"

    finally:
        # Cleanup events
        with db_context() as session:
            for event_id in event_ids:
                session.query(EpisodicEvent).filter(EpisodicEvent.id == event_id).delete()
            session.commit()


def test_full_pipeline_response_structure(setup_test_environment):
    """Test 4: Verify the complete pipeline returns proper response structure."""
    env = setup_test_environment
    test_user = env["user"]
    raw_memories = env["raw_memories"]

    # Create episodic memory
    with db_context() as session:
        event = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Full pipeline test",
            details="Testing complete raw_memory reference pipeline",
            event_type="test",
            occurred_at=datetime.now(timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[rm.id for rm in raw_memories],  # All three
            tree_path=["test"]
        )
        session.add(event)
        session.commit()
        event_id = event.id

    try:
        from mirix.agent.agent import Agent
        from mirix.schemas.agent_state import AgentState

        agent_state = AgentState(
            id=f"agent-{uuid.uuid4()}",
            name="chat_agent",
            topic="test",
            system_prompt="You are a helpful assistant",
            tools=[],
            user_id=test_user.id,
            organization_id=test_user.organization_id
        )

        agent = Agent(agent_state=agent_state, user=test_user)

        # Trigger memory retrieval
        agent.build_system_prompt_with_memories(
            raw_system="Test",
            topics="pipeline test"
        )

        # Simulate what AgentWrapper does
        raw_memory_refs = getattr(agent, 'current_raw_memory_refs', [])

        # Verify we got references
        assert len(raw_memory_refs) > 0, "Should have collected references"

        # Simulate the response structure that AgentWrapper.send_message returns
        response = {
            "response": "Test response",
            "memoryReferences": raw_memory_refs
        }

        # Verify structure
        assert isinstance(response, dict)
        assert "response" in response
        assert "memoryReferences" in response
        assert isinstance(response["memoryReferences"], list)

        # Simulate what FastAPI endpoint does: fetch full details
        raw_memory_manager = env["raw_memory_manager"]
        memory_details = []

        for ref_id in response["memoryReferences"]:
            raw_mem = raw_memory_manager.get_raw_memory_by_id(
                raw_memory_id=ref_id,
                user_id=test_user.id
            )
            if raw_mem:
                memory_details.append({
                    "id": raw_mem.id,
                    "source_app": raw_mem.source_app,
                    "source_url": raw_mem.source_url,
                    "captured_at": raw_mem.captured_at.isoformat() if raw_mem.captured_at else None,
                    "ocr_text": raw_mem.ocr_text,
                })

        # Verify we fetched details for all references
        assert len(memory_details) == len(raw_memory_refs)

        # Verify each detail has expected structure (matches frontend ChatBubble expectations)
        for detail in memory_details:
            assert "id" in detail
            assert "source_app" in detail
            assert "source_url" in detail
            assert "captured_at" in detail
            assert "ocr_text" in detail
            assert detail["source_app"] in ["Chrome", "Safari", "Notion"]

        # Verify the final SSE response structure
        sse_response = {
            "type": "final",
            "response": response["response"],
            "memoryReferences": memory_details
        }

        # This is what frontend will receive
        assert sse_response["type"] == "final"
        assert isinstance(sse_response["memoryReferences"], list)
        assert len(sse_response["memoryReferences"]) > 0

        # Verify JSON serialization works (important for SSE)
        json_str = json.dumps(sse_response)
        assert json_str is not None

        # Verify deserialization
        parsed = json.loads(json_str)
        assert parsed["memoryReferences"][0]["source_app"] in ["Chrome", "Safari", "Notion"]

    finally:
        with db_context() as session:
            session.query(EpisodicEvent).filter(EpisodicEvent.id == event_id).delete()
            session.commit()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
