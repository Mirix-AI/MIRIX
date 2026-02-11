"""
Memory System Server-Side Tests for Mirix

Tests memory operations using SyncServer directly (no network, fast).
Automatically initializes users and agents on first run.

Prerequisites:
    export GEMINI_API_KEY=your_api_key_here

Test Coverage:
- Direct memory operations via SyncServer managers
- All 5 memory types (Episodic, Procedural, Resource, Knowledge Vault, Semantic)
- All search methods (bm25, embedding) across relevant fields
- Retrieve with conversation and topic

Usage:
    pytest tests/test_memory_server.py -v
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from dotenv import load_dotenv

# Load .env file (optional - Mirix now loads .env automatically in mirix/settings.py)
# Kept here for backward compatibility
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.schemas.agent import AgentType
from mirix.schemas.client import Client
from mirix.server.server import SyncServer

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")


@pytest.fixture(scope="module")
def server():
    """Create server instance for all tests."""
    return SyncServer()


@pytest.fixture(scope="module")
def user(server):
    """Get or create the demo user."""
    from mirix.schemas.organization import Organization, OrganizationCreate
    from mirix.schemas.user import User as PydanticUser

    org_id = "demo-org"
    user_id = "demo-user"

    # Try to get existing user
    try:
        user = server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass

    # Create organization if it doesn't exist
    try:
        org = server.organization_manager.get_organization_by_id(org_id)
    except Exception:
        org_create = OrganizationCreate(id=org_id, name=org_id)
        org = server.organization_manager.create_organization(pydantic_org=Organization(**org_create.model_dump()))

    # Create user
    user = server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=user_id,
            organization_id=org.id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
        )
    )

    return user


@pytest.fixture(scope="module")
def client(server, user):
    """Get or create the demo client (represents client application)."""
    from mirix.services.client_manager import ClientManager

    client_id = "demo-client-app"
    client_manager = ClientManager()

    # Try to get existing client
    try:
        client = client_manager.get_client_by_id(client_id)
        if client:
            return client
    except Exception:
        pass

    # Create client
    client = client_manager.create_client(
        pydantic_client=Client(
            id=client_id,
            name=client_id,
            organization_id=user.organization_id,
            status="active",
            scope="read_write",
        )
    )

    return client


@pytest.fixture(scope="module")
def meta_agent(server, client):
    """Get or create meta agent with all sub-agents."""
    from mirix import EmbeddingConfig, LLMConfig
    from mirix.schemas.agent import CreateMetaAgent

    # Check if meta agent already exists
    existing_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    for agent in existing_agents:
        if agent.agent_type == AgentType.meta_memory_agent:
            print(f"\n[OK] Using existing meta agent: {agent.id}")
            return agent

    # Meta agent doesn't exist, create it
    print("\n[SETUP] Initializing meta agent and sub-agents...")

    # Load config (same pattern as rest_api.py)
    config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build create_params by flattening meta_agent_config (same as rest_api.py)
    create_params = {
        "llm_config": LLMConfig(**config["llm_config"]),
        "embedding_config": EmbeddingConfig(**config["embedding_config"]),
    }

    # Flatten meta_agent_config fields into create_params
    if "meta_agent_config" in config and config["meta_agent_config"]:
        meta_config = config["meta_agent_config"]
        if "agents" in meta_config:
            create_params["agents"] = meta_config["agents"]
        if "system_prompts" in meta_config:
            create_params["system_prompts"] = meta_config["system_prompts"]

    # Create meta agent using agent_manager (same as rest_api.py)
    meta_agent = server.agent_manager.create_meta_agent(
        meta_agent_create=CreateMetaAgent(**create_params), actor=client
    )

    print(f"[OK] Meta agent created: {meta_agent.id}")
    return meta_agent


def get_sub_agent(server, client, meta_agent, agent_type):
    """Helper to get sub agent by type."""
    sub_agents = server.agent_manager.list_agents(actor=client, parent_id=meta_agent.id)

    for agent in sub_agents:
        if agent.agent_type == agent_type:
            return agent

    raise ValueError(f"Sub agent of type {agent_type} not found")


# =================================================================
# DIRECT MEMORY OPERATIONS TESTS (Using Managers)
# =================================================================


class TestDirectEpisodicMemory:
    """Test direct episodic memory operations using managers."""

    def test_insert_event(self, server, client, user, meta_agent):
        """Test inserting an event directly."""
        episodic_agent = get_sub_agent(server, client, meta_agent, AgentType.episodic_memory_agent)

        event = server.episodic_memory_manager.insert_event(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=episodic_agent,
            event_type="activity",
            timestamp=datetime.now(),
            event_actor="user",
            summary="Attended team meeting",
            details="Weekly standup meeting with the development team",
            organization_id=user.organization_id,
        )

        assert event is not None
        assert event.id is not None
        assert event.summary == "Attended team meeting"
        print(f"[OK] Inserted event: {event.id}")

    def test_search_bm25(self, server, client, user, meta_agent):
        """Test BM25 search on episodic memory."""
        episodic_agent = get_sub_agent(server, client, meta_agent, AgentType.episodic_memory_agent)

        results = server.episodic_memory_manager.list_episodic_memory(
            agent_state=episodic_agent,
            user=user,
            query="meeting",
            search_method="bm25",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one event matching 'meeting'"
        print(f"[OK] BM25 search found {len(results)} events")

    def test_search_embedding(self, server, client, user, meta_agent):
        """Test embedding search on episodic memory."""
        episodic_agent = get_sub_agent(server, client, meta_agent, AgentType.episodic_memory_agent)

        results = server.episodic_memory_manager.list_episodic_memory(
            agent_state=episodic_agent,
            user=user,
            query="team collaboration",
            search_method="embedding",
            search_field="details",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one event matching 'team collaboration'"
        print(f"[OK] Embedding search found {len(results)} events")


class TestDirectProceduralMemory:
    """Test direct procedural memory operations using managers."""

    def test_insert_procedure(self, server, client, user, meta_agent):
        """Test inserting a procedure directly."""
        procedural_agent = get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)

        procedure = server.procedural_memory_manager.insert_procedure(
            agent_state=procedural_agent,
            agent_id=meta_agent.id,
            entry_type="process",
            summary="Deploy application to production",
            steps=[
                "Run all tests",
                "Create release branch",
                "Build production artifacts",
                "Deploy to staging",
                "Verify staging deployment",
                "Deploy to production",
            ],
            actor=client,
            organization_id=user.organization_id,
            user_id=user.id,
        )

        assert procedure is not None
        assert procedure.id is not None
        assert len(procedure.steps) == 6
        print(f"[OK] Inserted procedure: {procedure.id}")

    def test_search_procedures(self, server, client, user, meta_agent):
        """Test searching procedures."""
        procedural_agent = get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)

        results = server.procedural_memory_manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="deploy",
            search_method="embedding",
            search_field="summary",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one procedure matching 'deploy'"
        print(f"[OK] Found {len(results)} procedures")


class TestDirectResourceMemory:
    """Test direct resource memory operations using managers."""

    def test_insert_resource(self, server, client, user, meta_agent):
        """Test inserting a resource directly."""
        resource_agent = get_sub_agent(server, client, meta_agent, AgentType.resource_memory_agent)

        resource = server.resource_memory_manager.insert_resource(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=resource_agent,
            title="Python Best Practices Guide",
            summary="Comprehensive guide on Python coding standards",
            resource_type="documentation",
            content="PEP 8 style guide, type hints, error handling, testing practices",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        assert resource is not None
        assert resource.id is not None
        assert resource.title == "Python Best Practices Guide"
        print(f"[OK] Inserted resource: {resource.id}")

    def test_search_resources_bm25(self, server, client, user, meta_agent):
        """Test BM25 search on resources."""
        resource_agent = get_sub_agent(server, client, meta_agent, AgentType.resource_memory_agent)

        # Ensure test data exists
        server.resource_memory_manager.insert_resource(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=resource_agent,
            title="Python Best Practices Guide",
            summary="Comprehensive guide on Python coding standards",
            resource_type="documentation",
            content="PEP 8 style guide, type hints, error handling, testing practices",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        results = server.resource_memory_manager.list_resources(
            agent_state=resource_agent,
            user=user,
            query="Python",
            search_method="bm25",
            search_field="title",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one resource matching 'Python'"
        print(f"[OK] BM25 search found {len(results)} resources")

    def test_search_resources_embedding(self, server, client, user, meta_agent):
        """Test embedding search on resources."""
        resource_agent = get_sub_agent(server, client, meta_agent, AgentType.resource_memory_agent)

        results = server.resource_memory_manager.list_resources(
            agent_state=resource_agent,
            user=user,
            query="coding standards",
            search_method="embedding",
            search_field="summary",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one resource matching 'coding standards'"
        print(f"[OK] Embedding search found {len(results)} resources")


class TestDirectKnowledgeVault:
    """Test direct knowledge vault operations using managers."""

    def test_insert_knowledge(self, server, client, user, meta_agent):
        """Test inserting knowledge directly."""
        knowledge_vault_memory_agent = get_sub_agent(server, client, meta_agent, AgentType.knowledge_vault_memory_agent)

        knowledge = server.knowledge_vault_manager.insert_knowledge(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=knowledge_vault_memory_agent,
            entry_type="credential",
            source="development_environment",
            sensitivity="medium",
            secret_value="test_api_key_abc123",
            caption="Test API key for development",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        assert knowledge is not None
        assert knowledge.id is not None
        assert knowledge.entry_type == "credential"
        print(f"[OK] Inserted knowledge: {knowledge.id}")

    def test_search_knowledge(self, server, client, user, meta_agent):
        """Test searching knowledge vault."""
        knowledge_vault_memory_agent = get_sub_agent(server, client, meta_agent, AgentType.knowledge_vault_memory_agent)

        # Ensure test data exists
        server.knowledge_vault_manager.insert_knowledge(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=knowledge_vault_memory_agent,
            entry_type="credential",
            source="development_environment",
            sensitivity="medium",
            secret_value="test_api_key_abc123",
            caption="Test API key for development",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        results = server.knowledge_vault_manager.list_knowledge(
            agent_state=knowledge_vault_memory_agent,
            user=user,
            query="api_key",
            search_method="bm25",
            search_field="secret_value",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one knowledge item matching 'api_key'"
        print(f"[OK] Found {len(results)} knowledge items")


class TestDirectSemanticMemory:
    """Test direct semantic memory operations using managers."""

    def test_insert_semantic_item(self, server, client, user, meta_agent):
        """Test inserting semantic item directly."""
        semantic_agent = get_sub_agent(server, client, meta_agent, AgentType.semantic_memory_agent)

        semantic_item = server.semantic_memory_manager.insert_semantic_item(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=semantic_agent,
            name="Machine Learning Concepts",
            summary="Understanding supervised vs unsupervised learning",
            details="Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data",
            source="training_session",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        assert semantic_item is not None
        assert semantic_item.id is not None
        assert semantic_item.name == "Machine Learning Concepts"
        print(f"[OK] Inserted semantic item: {semantic_item.id}")

    def test_search_semantic_embedding(self, server, client, user, meta_agent):
        """Test embedding search on semantic memory."""
        semantic_agent = get_sub_agent(server, client, meta_agent, AgentType.semantic_memory_agent)

        results = server.semantic_memory_manager.list_semantic_items(
            agent_state=semantic_agent,
            user=user,
            query="machine learning",
            search_method="embedding",
            search_field="name",
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0, "Should find at least one semantic item matching 'machine learning'"
        print(f"[OK] Embedding search found {len(results)} semantic items")


# =================================================================
# SEARCH METHOD COMPARISON TESTS
# =================================================================


class TestSearchMethodComparison:
    """Compare different search methods across all memory types."""

    def test_episodic_search_methods_and_fields(self, server, client, user, meta_agent):
        """Test all search methods and fields on episodic memory."""
        episodic_agent = get_sub_agent(server, client, meta_agent, AgentType.episodic_memory_agent)

        # Test different fields: summary, details
        test_cases = [
            ("meeting", "summary", "bm25"),
            ("meeting", "summary", "embedding"),
            ("team", "details", "bm25"),
            ("team", "details", "embedding"),
        ]

        for query, field, method in test_cases:
            results = server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                user=user,
                query=query,
                search_method=method,
                search_field=field,
                limit=10,
            )
            assert isinstance(results, list)
            assert len(results) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"[OK] Episodic {method} on '{field}': {len(results)} results")

    def test_procedural_search_methods_and_fields(self, server, client, user, meta_agent):
        """Test all search methods and fields on procedural memory."""
        procedural_agent = get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)

        # Ensure test data exists
        server.procedural_memory_manager.insert_procedure(
            agent_state=procedural_agent,
            agent_id=meta_agent.id,
            entry_type="process",
            summary="Deploy application to production",
            steps=[
                "Run all tests",
                "Create release branch",
                "Build production artifacts",
                "Deploy to staging",
                "Verify staging deployment",
                "Deploy to production",
            ],
            actor=client,
            organization_id=user.organization_id,
            user_id=user.id,
        )

        # Test different fields: summary, steps
        test_cases = [
            ("deploy", "summary", "bm25"),
            ("deploy", "summary", "embedding"),
            ("production", "steps", "bm25"),
            ("production", "steps", "embedding"),
        ]

        for query, field, method in test_cases:
            results = server.procedural_memory_manager.list_procedures(
                agent_state=procedural_agent,
                user=user,
                query=query,
                search_method=method,
                search_field=field,
                limit=10,
            )
            assert isinstance(results, list)
            assert len(results) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"[OK] Procedural {method} on '{field}': {len(results)} results")

    def test_resource_search_methods_and_fields(self, server, client, user, meta_agent):
        """Test all search methods and fields on resource memory."""
        resource_agent = get_sub_agent(server, client, meta_agent, AgentType.resource_memory_agent)

        # Test different fields: summary (embedding + bm25), content (bm25 only)
        test_cases = [
            ("Python", "summary", "bm25"),
            ("Python", "summary", "embedding"),
            ("PEP", "content", "bm25"),  # embedding NOT supported for content
        ]

        for query, field, method in test_cases:
            results = server.resource_memory_manager.list_resources(
                agent_state=resource_agent,
                user=user,
                query=query,
                search_method=method,
                search_field=field,
                limit=10,
            )
            assert isinstance(results, list)
            assert len(results) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"[OK] Resource {method} on '{field}': {len(results)} results")

    def test_knowledge_vault_search_methods_and_fields(self, server, client, user, meta_agent):
        """Test all search methods and fields on knowledge vault."""
        knowledge_vault_memory_agent = get_sub_agent(server, client, meta_agent, AgentType.knowledge_vault_memory_agent)

        # Ensure test data exists
        server.knowledge_vault_manager.insert_knowledge(
            actor=client,
            agent_id=meta_agent.id,
            agent_state=knowledge_vault_memory_agent,
            entry_type="credential",
            source="development_environment",
            sensitivity="medium",
            secret_value="test_api_key_abc123",
            caption="Test API key for development",
            organization_id=user.organization_id,
            user_id=user.id,
        )

        # Test different fields: secret_value, caption
        test_cases = [
            ("api_key", "secret_value", "bm25"),
            ("api", "caption", "bm25"),
            ("api", "caption", "embedding"),
        ]

        for query, field, method in test_cases:
            results = server.knowledge_vault_manager.list_knowledge(
                agent_state=knowledge_vault_memory_agent,
                user=user,
                query=query,
                search_method=method,
                search_field=field,
                limit=10,
            )
            assert isinstance(results, list)
            assert len(results) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"[OK] Knowledge Vault {method} on '{field}': {len(results)} results")

    def test_semantic_search_methods_and_fields(self, server, client, user, meta_agent):
        """Test all search methods and fields on semantic memory."""
        semantic_agent = get_sub_agent(server, client, meta_agent, AgentType.semantic_memory_agent)

        # Test different fields: name, summary, details
        test_cases = [
            ("Machine Learning", "name", "bm25"),
            ("Machine Learning", "name", "embedding"),
            ("supervised", "summary", "bm25"),
            ("supervised", "summary", "embedding"),
            ("labeled data", "details", "bm25"),
            ("labeled data", "details", "embedding"),
        ]

        for query, field, method in test_cases:
            results = server.semantic_memory_manager.list_semantic_items(
                agent_state=semantic_agent,
                user=user,
                query=query,
                search_method=method,
                search_field=field,
                limit=10,
            )
            assert isinstance(results, list)
            assert len(results) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"[OK] Semantic {method} on '{field}': {len(results)} results")

    def test_all_memory_types_search(self, server, client, user, meta_agent):
        """Test search across all five memory types."""
        # Episodic
        episodic_agent = get_sub_agent(server, client, meta_agent, AgentType.episodic_memory_agent)
        episodic_results = server.episodic_memory_manager.list_episodic_memory(
            agent_state=episodic_agent,
            user=user,
            query="test",
            search_method="bm25",
            limit=5,
        )
        print(f"[OK] Episodic: {len(episodic_results)} results")

        # Procedural
        procedural_agent = get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)
        procedural_results = server.procedural_memory_manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="test",
            search_method="bm25",
            limit=5,
        )
        print(f"[OK] Procedural: {len(procedural_results)} results")

        # Resource
        resource_agent = get_sub_agent(server, client, meta_agent, AgentType.resource_memory_agent)
        resource_results = server.resource_memory_manager.list_resources(
            agent_state=resource_agent,
            user=user,
            query="test",
            search_method="bm25",
            limit=5,
        )
        print(f"[OK] Resource: {len(resource_results)} results")

        # Knowledge Vault
        knowledge_vault_memory_agent = get_sub_agent(server, client, meta_agent, AgentType.knowledge_vault_memory_agent)
        knowledge_results = server.knowledge_vault_manager.list_knowledge(
            agent_state=knowledge_vault_memory_agent,
            user=user,
            query="test",
            search_method="bm25",
            limit=5,
        )
        print(f"[OK] Knowledge: {len(knowledge_results)} results")

        # Semantic
        semantic_agent = get_sub_agent(server, client, meta_agent, AgentType.semantic_memory_agent)
        semantic_results = server.semantic_memory_manager.list_semantic_items(
            agent_state=semantic_agent,
            user=user,
            query="test",
            search_method="bm25",
            limit=5,
        )
        print(f"[OK] Semantic: {len(semantic_results)} results")

        # All should return lists
        for results in [episodic_results, procedural_results, resource_results, knowledge_results, semantic_results]:
            assert isinstance(results, list)


class TestMetaMemoryParallelism:
    def test_trigger_memory_update_runs_in_parallel(self, server, client, user, meta_agent, monkeypatch):
        """Ensure meta memory updates dispatch memory agents concurrently."""

        from mirix.functions.function_sets.memory_tools import trigger_memory_update
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.mirix_message_content import TextContent

        child_agents = server.agent_manager.list_agents(
            actor=client,
            user=user,
            parent_id=meta_agent.id,
        )
        target_types = {
            AgentType.episodic_memory_agent.value,
            AgentType.procedural_memory_agent.value,
        }
        relevant_agents = [agent for agent in child_agents if agent.agent_type in target_types]
        assert len(relevant_agents) == len(target_types), "Expected episodic and procedural memory agents to be present"

        tracker = {"start_times": {}, "thread_ids": {}}
        lock = threading.Lock()

        class MockMemoryAgent:
            def __init__(self, agent_state, interface, actor, user, filter_tags=None, use_cache=True):
                self.agent_state = agent_state

            def step(self, input_messages, chaining, actor=None, user=None):
                memory_type = self.agent_state.agent_type.replace("_memory_agent", "")
                with lock:
                    tracker["start_times"][memory_type] = time.perf_counter()
                    tracker["thread_ids"][memory_type] = threading.get_ident()
                time.sleep(1)

        monkeypatch.setattr("mirix.agent.EpisodicMemoryAgent", MockMemoryAgent)
        monkeypatch.setattr("mirix.agent.ProceduralMemoryAgent", MockMemoryAgent)
        monkeypatch.setattr("mirix.functions.function_sets.memory_tools.os.cpu_count", lambda: 8)

        class StubAgentManager:
            def list_agents(self, *, parent_id, actor, limit=None):
                assert parent_id == meta_agent.id
                return relevant_agents

        stub_meta_agent = SimpleNamespace(
            agent_state=SimpleNamespace(id=meta_agent.id, name=meta_agent.name),
            agent_manager=StubAgentManager(),
            interface=SimpleNamespace(),
            actor=client,
            user=user,
        )

        message = MessageCreate(
            role=MessageRole.user,
            content=[TextContent(text="Trigger memory update")],
        )
        user_message = {"message": message, "chaining": False}

        result = trigger_memory_update(
            stub_meta_agent,
            user_message=user_message,
            memory_types=["episodic", "procedural"],
        )

        assert result is not None

        start_times = tracker["start_times"]
        assert set(start_times.keys()) == {"episodic", "procedural"}
        gap = abs(start_times["episodic"] - start_times["procedural"])

        assert gap < 0.45, f"Expected parallel execution, gap was {gap:.3f}s"
        thread_ids = tracker["thread_ids"]
        assert len(set(thread_ids.values())) == 2, "Updates should run on separate threads"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
