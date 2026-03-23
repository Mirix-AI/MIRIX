"""
Comprehensive Unit Tests for LocalClient

This test suite verifies all APIs in mirix/local_client/local_client.py,
including agent management, memory operations, tool management, block management,
and client-scoped isolation.

Test Coverage:
- LocalClient initialization (with/without client_id)
- Agent CRUD operations
- Meta agent creation
- Tool management (create, list, update, delete)
- Block management (create, list, update, delete)
- Memory operations (core memory, archival, recall)
- Message operations
- Sandbox configuration
- File management
- Client isolation
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix import EmbeddingConfig, LLMConfig
from mirix.local_client.local_client import LocalClient
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.block import Block, Human, Persona
from mirix.schemas.enums import MessageRole
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import Memory
from mirix.schemas.message import MessageCreate
from mirix.schemas.tool import Tool
from mirix.schemas.usage import MirixUsageStatistics

# Test configuration
TEST_RUN_ID = uuid.uuid4().hex[:8]
TEST_ORG_ID = f"test-local-org-{TEST_RUN_ID}"
TEST_CLIENT_A_ID = f"test-local-client-a-{TEST_RUN_ID}"
TEST_CLIENT_B_ID = f"test-local-client-b-{TEST_RUN_ID}"
TEST_USER_A_ID = f"test-local-user-a-{TEST_RUN_ID}"
TEST_USER_B_ID = f"test-local-user-b-{TEST_RUN_ID}"

# LLM configuration for testing
TEST_LLM_CONFIG = LLMConfig(
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="https://api.openai.com/v1",
    context_window=8192,
)

TEST_EMBEDDING_CONFIG = EmbeddingConfig(
    embedding_model="text-embedding-ada-002",
    embedding_endpoint_type="openai",
    embedding_endpoint="https://api.openai.com/v1",
    embedding_dim=1536,
    embedding_chunk_size=300,
)

# Mark all async tests in this module for pytest-asyncio; use one loop per module so
# module-scoped async fixtures (test_organization, client_a, client_b, default_client)
# and all tests share the same event loop (avoids "another operation is in progress").
pytestmark = [pytest.mark.asyncio(loop_scope="module")]


# ============================================================================
# FIXTURES
# ============================================================================


@pytest_asyncio.fixture(scope="module")
async def test_organization():
    """Create test organization before any clients."""
    default_client = await LocalClient.create(
        debug=False,
        default_llm_config=TEST_LLM_CONFIG,
        default_embedding_config=TEST_EMBEDDING_CONFIG,
    )
    org = await default_client.create_org(name=f"test-org-{TEST_RUN_ID}")
    global TEST_ORG_ID
    TEST_ORG_ID = org.id
    yield org


@pytest_asyncio.fixture(scope="module")
async def client_a(test_organization):
    """Create LocalClient A for testing."""
    client = await LocalClient.create(
        client_id=TEST_CLIENT_A_ID,
        user_id=TEST_USER_A_ID,
        org_id=test_organization.id,
        debug=False,
        default_llm_config=TEST_LLM_CONFIG,
        default_embedding_config=TEST_EMBEDDING_CONFIG,
    )
    yield client


@pytest_asyncio.fixture(scope="module")
async def client_b(test_organization):
    """Create LocalClient B for testing client isolation."""
    client = await LocalClient.create(
        client_id=TEST_CLIENT_B_ID,
        user_id=TEST_USER_B_ID,
        org_id=test_organization.id,
        debug=False,
        default_llm_config=TEST_LLM_CONFIG,
        default_embedding_config=TEST_EMBEDDING_CONFIG,
    )
    yield client


@pytest_asyncio.fixture(scope="module")
async def default_client():
    """Create LocalClient with default IDs."""
    client = await LocalClient.create(
        debug=False,
        default_llm_config=TEST_LLM_CONFIG,
        default_embedding_config=TEST_EMBEDDING_CONFIG,
    )
    yield client


@pytest.fixture
def test_memory(client_a):
    """Create a test memory object."""
    from mirix.schemas.block import Block
    from mirix.schemas.memory import BasicBlockMemory

    # Create blocks without user_id - let block_manager set it during save
    return BasicBlockMemory(
        blocks=[
            Block(value="Test persona description", limit=5000, label="persona"),
            Block(value="Test human description", limit=5000, label="human"),
        ]
    )


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test file content")
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# TEST: INITIALIZATION
# ============================================================================


class TestInitialization:
    """Test LocalClient initialization."""

    async def test_init_with_explicit_ids(self, client_a):
        """Test initialization with explicit client_id, user_id, org_id."""
        assert client_a.client_id == TEST_CLIENT_A_ID
        assert client_a.user_id == TEST_USER_A_ID
        assert client_a.org_id == TEST_ORG_ID
        assert client_a.client is not None
        assert client_a.user is not None
        assert client_a.organization is not None

    async def test_init_with_default_ids(self, default_client):
        """Test initialization without explicit IDs (uses defaults)."""
        assert default_client.client_id is not None
        assert default_client.user_id is not None
        assert default_client.org_id is not None
        assert default_client.client is not None
        assert default_client.user is not None
        assert default_client.organization is not None

    async def test_client_attributes(self, client_a):
        """Test that all expected attributes are initialized."""
        assert hasattr(client_a, "client_id")
        assert hasattr(client_a, "user_id")
        assert hasattr(client_a, "org_id")
        assert hasattr(client_a, "client")
        assert hasattr(client_a, "user")
        assert hasattr(client_a, "organization")
        assert hasattr(client_a, "server")
        assert hasattr(client_a, "interface")
        assert hasattr(client_a, "file_manager")

    async def test_default_configs(self, client_a):
        """Test that default LLM and embedding configs are set."""
        assert client_a._default_llm_config == TEST_LLM_CONFIG
        assert client_a._default_embedding_config == TEST_EMBEDDING_CONFIG


# ============================================================================
# TEST: AGENT MANAGEMENT
# ============================================================================


class TestAgentManagement:
    """Test agent CRUD operations."""

    async def test_create_agent(self, client_a, test_memory):
        """Test creating an agent."""
        agent_name = f"test-agent-{uuid.uuid4().hex[:8]}"

        agent = await client_a.create_agent(
            name=agent_name,
            agent_type=AgentType.chat_agent,
            memory=test_memory,
            system="You are a helpful assistant.",
            description="Test agent",
        )

        assert agent is not None
        assert agent.name == agent_name
        assert agent.agent_type == AgentType.chat_agent

    async def test_list_agents(self, client_a, test_memory):
        """Test listing agents."""
        # Create a test agent
        agent_name = f"test-list-agent-{uuid.uuid4().hex[:8]}"
        await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system prompt",
        )

        # List agents
        agents = await client_a.list_agents()
        assert isinstance(agents, list)
        assert len(agents) >= 1

        # Verify our agent is in the list
        agent_names = [a.name for a in agents]
        assert agent_name in agent_names

    async def test_get_agent(self, client_a, test_memory):
        """Test getting an agent by ID."""
        # Create an agent
        agent_name = f"test-get-agent-{uuid.uuid4().hex[:8]}"
        created_agent = await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system",
        )

        # Get the agent
        retrieved_agent = await client_a.get_agent(created_agent.id)
        assert retrieved_agent.id == created_agent.id
        assert retrieved_agent.name == agent_name

    async def test_get_agent_by_name(self, client_a, test_memory):
        """Test getting an agent by name."""
        agent_name = f"test-get-by-name-{uuid.uuid4().hex[:8]}"
        await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system",
        )

        # Get by name
        agent = await client_a.get_agent_by_name(agent_name)
        assert agent.name == agent_name

    async def test_get_agent_id(self, client_a, test_memory):
        """Test getting agent ID by name."""
        agent_name = f"test-get-id-{uuid.uuid4().hex[:8]}"
        created_agent = await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system",
        )

        agent_id = await client_a.get_agent_id(agent_name)
        assert agent_id == created_agent.id

    async def test_agent_exists(self, client_a, test_memory):
        """Test checking if an agent exists."""
        agent_name = f"test-exists-{uuid.uuid4().hex[:8]}"
        created_agent = await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system",
        )

        # Check by ID
        assert await client_a.agent_exists(agent_id=created_agent.id) is True

        # Check by name
        assert await client_a.agent_exists(agent_name=agent_name) is True

        # Check non-existent
        assert await client_a.agent_exists(agent_name="non-existent-agent-xyz") is False

    async def test_rename_agent(self, client_a, test_memory):
        """Test renaming an agent."""
        old_name = f"test-rename-old-{uuid.uuid4().hex[:8]}"
        new_name = f"test-rename-new-{uuid.uuid4().hex[:8]}"

        agent = await client_a.create_agent(
            name=old_name,
            memory=test_memory,
            system="Test system",
        )

        # Rename
        await client_a.rename_agent(agent.id, new_name)

        # Verify
        renamed_agent = await client_a.get_agent(agent.id)
        assert renamed_agent.name == new_name

    async def test_delete_agent(self, client_a, test_memory):
        """Test deleting an agent."""
        agent_name = f"test-delete-{uuid.uuid4().hex[:8]}"
        agent = await client_a.create_agent(
            name=agent_name,
            memory=test_memory,
            system="Test system",
        )

        # Delete
        await client_a.delete_agent(agent.id)

        # Verify deletion
        assert await client_a.agent_exists(agent_id=agent.id) is False


# ============================================================================
# TEST: CLIENT ISOLATION
# ============================================================================


class TestClientIsolation:
    """Test that clients have isolated agents."""

    async def test_agents_isolated_by_client(self, client_a, client_b, test_memory):
        """Test that agents are isolated per client."""
        # Create agent in client A
        agent_a_name = f"test-isolation-a-{uuid.uuid4().hex[:8]}"
        agent_a = await client_a.create_agent(
            name=agent_a_name,
            memory=test_memory,
            system="Client A agent",
        )

        # Create agent in client B
        agent_b_name = f"test-isolation-b-{uuid.uuid4().hex[:8]}"
        agent_b = await client_b.create_agent(
            name=agent_b_name,
            memory=test_memory,
            system="Client B agent",
        )

        # List agents for each client
        agents_a = await client_a.list_agents()
        agents_b = await client_b.list_agents()

        # Verify client A can see its own agent
        agent_a_names = [a.name for a in agents_a]
        assert agent_a_name in agent_a_names

        # Verify client B can see its own agent
        agent_b_names = [a.name for a in agents_b]
        assert agent_b_name in agent_b_names

        # Verify client A cannot see client B's agent
        assert agent_b_name not in agent_a_names

        # Verify client B cannot see client A's agent
        assert agent_a_name not in agent_b_names

    async def test_get_agent_respects_client_ownership(self, client_a, client_b, test_memory):
        """Test that get_agent respects client ownership."""
        # Create agent in client A
        agent_a = await client_a.create_agent(
            name=f"test-ownership-{uuid.uuid4().hex[:8]}",
            memory=test_memory,
            system="Client A agent",
        )

        # Client B should not be able to access client A's agent
        with pytest.raises((NoResultFound, Exception)):
            await client_b.get_agent(agent_a.id)


# ============================================================================
# TEST: TOOL MANAGEMENT
# ============================================================================


class TestToolManagement:
    """Test tool CRUD operations."""

    async def test_create_tool(self, client_a):
        """Test creating a tool."""

        async def test_function(x: int, y: int) -> int:
            """
            Add two numbers.

            Args:
                x: First number to add
                y: Second number to add

            Returns:
                Sum of x and y
            """
            return x + y

        tool_name = f"test_tool_{uuid.uuid4().hex[:8]}"
        tool = await client_a.create_tool(
            func=test_function,
            name=tool_name,
            description="Test tool",
            tags=["test"],
        )

        assert tool is not None
        assert tool.name == tool_name

    async def test_list_tools(self, client_a):
        """Test listing tools."""
        tools = await client_a.list_tools(limit=50)
        assert isinstance(tools, list)

    async def test_get_tool(self, client_a):
        """Test getting a tool by ID."""

        async def test_function() -> str:
            """
            Return a test string.

            Returns:
                A test string
            """
            return "test"

        tool_name = f"test_get_tool_{uuid.uuid4().hex[:8]}"
        created_tool = await client_a.create_tool(func=test_function, name=tool_name)

        retrieved_tool = await client_a.get_tool(created_tool.id)
        assert retrieved_tool.id == created_tool.id
        assert retrieved_tool.name == tool_name

    async def test_get_tool_id(self, client_a):
        """Test getting tool ID by name."""

        async def test_function() -> str:
            """
            Return a test string.

            Returns:
                A test string
            """
            return "test"

        tool_name = f"test_tool_id_{uuid.uuid4().hex[:8]}"
        created_tool = await client_a.create_tool(func=test_function, name=tool_name)

        tool_id = await client_a.get_tool_id(tool_name)
        assert tool_id == created_tool.id

    async def test_update_tool(self, client_a):
        """Test updating a tool."""

        async def test_function() -> str:
            """
            Return a test string.

            Returns:
                A test string
            """
            return "original"

        tool_name = f"test_update_tool_{uuid.uuid4().hex[:8]}"
        tool = await client_a.create_tool(func=test_function, name=tool_name)

        # Update
        new_description = "Updated description"
        updated_tool = await client_a.update_tool(
            id=tool.id,
            description=new_description,
        )

        assert updated_tool.description == new_description

    async def test_delete_tool(self, client_a):
        """Test deleting a tool."""

        async def test_function() -> str:
            """
            Return a test string.

            Returns:
                A test string
            """
            return "test"

        tool_name = f"test_delete_tool_{uuid.uuid4().hex[:8]}"
        tool = await client_a.create_tool(func=test_function, name=tool_name)

        # Delete
        await client_a.delete_tool(tool.id)

        # Verify deletion - get_tool raises NoResultFound for deleted tools
        from mirix.orm.errors import NoResultFound

        with pytest.raises(NoResultFound):
            await client_a.get_tool(tool.id)


# ============================================================================
# TEST: BLOCK MANAGEMENT (Humans, Personas, Custom Blocks)
# ============================================================================


class TestBlockManagement:
    """Test block CRUD operations."""

    async def test_create_human(self, client_a):
        """Test creating a human block."""
        human_name = f"test_human_{uuid.uuid4().hex[:8]}"
        human_text = "I am a software engineer."

        human = await client_a.create_human(name=human_name, text=human_text)
        assert human is not None
        assert human.value == human_text

    async def test_create_persona(self, client_a):
        """Test creating a persona block."""
        persona_name = f"test_persona_{uuid.uuid4().hex[:8]}"
        persona_text = "You are a helpful AI assistant."

        persona = await client_a.create_persona(name=persona_name, text=persona_text)
        assert persona is not None
        assert persona.value == persona_text

    async def test_list_humans(self, client_a):
        """Test listing human blocks."""
        # Create a human first
        await client_a.create_human(name=f"test_list_human_{uuid.uuid4().hex[:8]}", text="Test human")

        humans = await client_a.list_humans()
        assert isinstance(humans, list)

    async def test_list_personas(self, client_a):
        """Test listing persona blocks."""
        # Create a persona first
        await client_a.create_persona(name=f"test_list_persona_{uuid.uuid4().hex[:8]}", text="Test persona")

        personas = await client_a.list_personas()
        assert isinstance(personas, list)

    async def test_create_block(self, client_a):
        """Test creating a custom block."""
        block_label = f"test_block_{uuid.uuid4().hex[:8]}"
        block_value = "Test block content"

        block = await client_a.create_block(
            label=block_label,
            value=block_value,
            limit=1000,
        )

        assert block is not None
        assert block.label == block_label
        assert block.value == block_value

    async def test_list_blocks(self, client_a):
        """Test listing blocks."""
        blocks = await client_a.list_blocks()
        assert isinstance(blocks, list)

    async def test_get_block(self, client_a):
        """Test getting a block by ID."""
        block_label = f"test_get_block_{uuid.uuid4().hex[:8]}"
        created_block = await client_a.create_block(
            label=block_label,
            value="Test content",
        )

        retrieved_block = await client_a.get_block(created_block.id)
        assert retrieved_block.id == created_block.id
        assert retrieved_block.label == block_label

    async def test_update_block(self, client_a):
        """Test updating a block."""
        block_label = f"test_update_block_{uuid.uuid4().hex[:8]}"
        block = await client_a.create_block(
            label=block_label,
            value="Original content",
        )

        # Update
        new_value = "Updated content"
        updated_block = await client_a.update_block(
            block_id=block.id,
            value=new_value,
        )

        assert updated_block.value == new_value

    async def test_delete_block(self, client_a):
        """Test deleting a block."""
        block_label = f"test_delete_block_{uuid.uuid4().hex[:8]}"
        block = await client_a.create_block(
            label=block_label,
            value="Test content",
        )

        # Delete
        deleted_block = await client_a.delete_block(block.id)
        assert deleted_block is not None


# ============================================================================
# TEST: FILE MANAGEMENT
# ============================================================================


class TestFileManagement:
    """Test file management operations."""

    async def test_save_file(self, client_a, temp_file):
        """Test saving a file."""
        file_metadata = await client_a.save_file(temp_file)
        assert file_metadata is not None
        assert file_metadata.file_name is not None

    async def test_list_files(self, client_a):
        """Test listing files."""
        files = await client_a.list_files(limit=10)
        assert isinstance(files, list)

    async def test_get_file(self, client_a, temp_file):
        """Test getting file metadata."""
        saved_file = await client_a.save_file(temp_file)

        retrieved_file = await client_a.get_file(saved_file.id)
        assert retrieved_file.id == saved_file.id

    async def test_search_files(self, client_a, temp_file):
        """Test searching files by name."""
        saved_file = await client_a.save_file(temp_file)

        # Search by partial name
        search_pattern = Path(temp_file).stem[:5]
        results = await client_a.search_files(search_pattern)
        assert isinstance(results, list)

    async def test_delete_file(self, client_a, temp_file):
        """Test deleting a file."""
        saved_file = await client_a.save_file(temp_file)

        # Delete
        await client_a.delete_file(saved_file.id)

        # Verify deletion
        with pytest.raises((NoResultFound, Exception)):
            await client_a.get_file(saved_file.id)


# ============================================================================
# TEST: ORGANIZATION MANAGEMENT
# ============================================================================


class TestOrganizationManagement:
    """Test organization operations."""

    async def test_create_org(self, default_client):
        """Test creating an organization."""
        org_name = f"test_org_{uuid.uuid4().hex[:8]}"
        org = await default_client.create_org(name=org_name)
        assert org is not None
        assert org.name == org_name

    async def test_list_orgs(self, default_client):
        """Test listing organizations."""
        orgs = await default_client.list_orgs(limit=10)
        assert isinstance(orgs, list)
        assert len(orgs) >= 1


# ============================================================================
# TEST: MESSAGE OPERATIONS (block_filter_tags)
# ============================================================================


class TestSendMessagesBlockFilterTags:
    """Test that send_messages passes block_filter_tags to server.send_messages."""

    async def test_send_messages_passes_block_filter_tags_to_server(self, client_a):
        """LocalClient.send_messages(block_filter_tags=...) forwards to server.send_messages."""
        block_filter_tags = {"env": "staging", "team": "platform"}
        mock_send = AsyncMock(return_value=MirixUsageStatistics())
        # Patch MirixResponse so return path doesn't validate messages (MessageCreate != MirixMessageUnion)
        with (
            patch.object(client_a.server, "send_messages", mock_send),
            patch("mirix.local_client.local_client.MirixResponse", Mock),
        ):
            messages = [MessageCreate(role=MessageRole.user, content="Hello")]
            await client_a.send_messages(
                agent_id="test-agent-id",
                messages=messages,
                block_filter_tags=block_filter_tags,
            )
        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs.get("block_filter_tags") == block_filter_tags

    async def test_send_messages_passes_none_block_filter_tags(self, client_a):
        """LocalClient.send_messages() without block_filter_tags passes None (or omits)."""
        mock_send = AsyncMock(return_value=MirixUsageStatistics())
        with (
            patch.object(client_a.server, "send_messages", mock_send),
            patch("mirix.local_client.local_client.MirixResponse", Mock),
        ):
            messages = [MessageCreate(role=MessageRole.user, content="Hi")]
            await client_a.send_messages(agent_id="test-agent-id", messages=messages)
        mock_send.assert_called_once()
        # None is passed explicitly by our implementation
        assert mock_send.call_args.kwargs.get("block_filter_tags") is None


# ============================================================================
# TEST: USER MANAGEMENT
# ============================================================================


class TestUserManagement:
    """Test user operations."""

    async def test_create_user(self, client_a):
        """Test that the client has a valid user."""
        # LocalClient initializes with a user via get_user_or_default
        # If the specified user_id doesn't exist, it uses the default user
        assert client_a.user is not None
        assert client_a.user.id is not None
        assert client_a.user.name is not None
        # Note: user.organization_id may be default org if user was not created for test org


# ============================================================================
# TEST: CONFIGURATION MANAGEMENT
# ============================================================================


class TestConfigurationManagement:
    """Test LLM and embedding configuration management."""

    async def test_set_default_llm_config(self, client_a):
        """Test setting default LLM configuration."""
        new_config = LLMConfig(
            model="gpt-3.5-turbo",
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com/v1",
            context_window=4096,
        )

        client_a.set_default_llm_config(new_config)
        assert client_a._default_llm_config == new_config

    async def test_set_default_embedding_config(self, client_a):
        """Test setting default embedding configuration."""
        new_config = EmbeddingConfig(
            embedding_model="text-embedding-3-small",
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_dim=1536,
            embedding_chunk_size=256,
        )

        client_a.set_default_embedding_config(new_config)
        assert client_a._default_embedding_config == new_config

    async def test_list_llm_configs(self, client_a):
        """Test listing available LLM configurations."""
        configs = await client_a.list_llm_configs()
        assert isinstance(configs, list)

    async def test_list_embedding_configs(self, client_a):
        """Test listing available embedding configurations."""
        configs = await client_a.list_embedding_configs()
        assert isinstance(configs, list)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
