"""
FastAPI REST API server for Mirix.
This provides HTTP endpoints that wrap the SyncServer functionality,
allowing MirixClient instances to communicate with a cloud-hosted server.
"""

import copy
import json
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
)
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.schemas.user import User
from mirix.schemas.client import Client, ClientCreate, ClientUpdate
from mirix.server.server import SyncServer
from mirix.settings import model_settings
from mirix.utils import convert_message_to_mirix_message

logger = get_logger(__name__)

# Import queue components
from mirix.queue import initialize_queue
from mirix.queue.manager import get_manager as get_queue_manager
from mirix.queue.queue_util import put_messages

# Initialize server (single instance shared across all requests)
_server: Optional[SyncServer] = None


def get_server() -> SyncServer:
    """Get or create the singleton SyncServer instance."""
    global _server
    if _server is None:
        logger.info("Creating SyncServer instance")
        _server = SyncServer()
    return _server


async def initialize():
    """
    Initialize the Mirix server and queue services.
    This function can be called by external applications to initialize the server.
    """
    logger.info("Starting Mirix REST API server")

    # Initialize SyncServer (singleton)
    server = get_server()
    logger.info("SyncServer initialized")

    # Initialize queue with server reference
    initialize_queue(server)
    logger.info("Queue service started with SyncServer integration")


async def cleanup():
    """
    Cleanup the Mirix server and queue services.
    This function can be called by external applications to cleanup the server.
    """
    logger.info("Shutting down Mirix REST API server")

    # Cleanup queue
    queue_manager = get_queue_manager()
    queue_manager.cleanup()
    logger.info("Queue service stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    await initialize()

    yield  # Server runs here

    # Shutdown
    await cleanup()


# Create API router for reusable routes
router = APIRouter()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Mirix API",
    description="REST API for Mirix - Memory-augmented AI Agent System",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def get_client_and_org(
    x_client_id: Optional[str] = None,
    x_org_id: Optional[str] = None,
) -> tuple[str, str]:
    """
    Get client_id and org_id from headers or use defaults.
    
    Returns:
        tuple[str, str]: (client_id, org_id)
    """
    server = get_server()
    
    if x_client_id:
        client_id = x_client_id
        org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID
    else:
        client_id = server.client_manager.DEFAULT_CLIENT_ID
        org_id = server.organization_manager.DEFAULT_ORG_ID
    
    return client_id, org_id


def extract_topics_from_messages(messages: List[Dict[str, Any]], llm_config: LLMConfig) -> Optional[str]:
    """
    Extract topics from a list of messages using LLM.

    Args:
        messages: List of message dictionaries (OpenAI format)
        llm_config: LLM configuration to use for topic extraction

    Returns:
        Extracted topics as a string (separated by ';') or None if extraction fails
    """
    try:

        if isinstance(messages, list) and "role" in messages[0].keys():
            # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]

            # We need to convert the message to the format in "content"
            new_messages = []
            for msg in messages:
                new_messages.append({'type': "text", "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]"})
                new_messages.extend(msg["content"])
            messages = new_messages

        temporary_messages = convert_message_to_mirix_message(messages)
        temporary_messages = [prepare_input_message_create(msg, agent_id="topic_extraction", wrap_user_message=False, wrap_system_message=True) for msg in temporary_messages]

        # Add instruction message for topic extraction
        temporary_messages.append(
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.user,
                    content='The above are the inputs from the user, please look at these content and extract the topic (brief description of what the user is focusing on) from these content. If there are multiple focuses in these content, then extract them all and put them into one string separated by ";". Call the function `update_topic` to update the topic with the extracted topics.',
                ),
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            )
        )

        # Prepend system message
        temporary_messages = [
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.system,
                    content="You are a helpful assistant that extracts the topic from the user's input.",
                ),
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            ),
        ] + temporary_messages

        # Define the function for topic extraction
        functions = [
            {
                "name": "update_topic",
                "description": "Update the topic of the conversation/content. The topic will be used for retrieving relevant information from the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": 'The topic of the current conversation/content. If there are multiple topics then separate them with ";".',
                        }
                    },
                    "required": ["topic"],
                },
            }
        ]

        # Use LLMClient to extract topics
        llm_client = LLMClient.create(
            llm_config=llm_config,
        )

        if llm_client:
            response = llm_client.send_llm_request(
                messages=temporary_messages,
                tools=functions,
                stream=False,
                force_tool_call="update_topic",
            )
            
            # Extract topics from the response
            for choice in response.choices:
                if (
                    hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls is not None
                    and len(choice.message.tool_calls) > 0
                ):
                    try:
                        function_args = json.loads(
                            choice.message.tool_calls[0].function.arguments
                        )
                        topics = function_args.get("topic")
                        logger.debug("Extracted topics: %s", topics)
                        return topics
                    except (json.JSONDecodeError, KeyError) as parse_error:
                        logger.warning("Failed to parse topic extraction response: %s", parse_error)
                        continue

    except Exception as e:
        logger.error("Error in extracting topics from messages: %s", e)

    return None


def _flatten_messages_to_plain_text(messages: List[Dict[str, Any]]) -> str:
    """
    Flatten OpenAI-style message payloads into a simple conversation transcript.
    """
    transcript_parts: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        parts: List[str] = []
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if text:
                        parts.append(text.strip())
                elif isinstance(chunk, str):
                    parts.append(chunk.strip())
        elif isinstance(content, str):
            parts.append(content.strip())

        combined = " ".join(filter(None, parts)).strip()
        if combined:
            transcript_parts.append(f"{role.upper()}: {combined}")

    return "\n".join(transcript_parts)


def extract_topics_with_local_model(messages: List[Dict[str, Any]], model_name: str) -> Optional[str]:
    """
    Extract topics using a locally hosted Ollama model via the /api/chat endpoint.

    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    """

    import ipdb; ipdb.set_trace()

    base_url = model_settings.ollama_base_url
    if not base_url:
        logger.warning(
            "local_model_for_retrieval provided (%s) but MIRIX_OLLAMA_BASE_URL is not configured",
            model_name,
        )
        return None

    conversation = _flatten_messages_to_plain_text(messages)
    if not conversation:
        logger.debug("No text content found in messages for local topic extraction")
        return None

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts the topic from the user's input. "
                    "Return a concise list of topics separated by ';' and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Conversation transcript:\n"
                    f"{conversation}\n\n"
                    "Respond ONLY with the topic(s) separated by ';'."
                ),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        import ipdb; ipdb.set_trace()
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=30,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        response_data = response.json()
    except requests.RequestException as exc:
        logger.error("Failed to extract topics with local model %s: %s", model_name, exc)
        return None

    message_payload = response_data.get("message") if isinstance(response_data, dict) else None
    text_response: Optional[str] = None
    if isinstance(message_payload, dict):
        text_response = message_payload.get("content")
    elif isinstance(response_data, dict):
        text_response = response_data.get("content")

    if isinstance(text_response, str):
        topics = text_response.strip()
        logger.debug("Extracted topics via local model %s: %s", model_name, topics)
        return topics or None

    logger.warning(
        "Unexpected response format from Ollama topic extraction: %s",
        response_data,
    )
    return None


# ============================================================================
# Error Handling
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mirix-api"}


# ============================================================================
# Agent Endpoints
# ============================================================================


@router.get("/agents", response_model=List[AgentState])
async def list_agents(
    query_text: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    limit: int = 100,
    cursor: Optional[str] = None,
    parent_id: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all agents for the authenticated user."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    
    tags_list = tags.split(",") if tags else None
    
    return server.agent_manager.list_agents(
        actor=client,
        tags=tags_list,
        query_text=query_text,
        limit=limit,
        cursor=cursor,
        parent_id=parent_id,
    )


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    name: Optional[str] = None
    agent_type: Optional[AgentType] = AgentType.chat_agent
    embedding_config: Optional[EmbeddingConfig] = None
    llm_config: Optional[LLMConfig] = None
    memory: Optional[Memory] = None
    block_ids: Optional[List[str]] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    tool_rules: Optional[List[BaseToolRule]] = None
    include_base_tools: Optional[bool] = True
    include_meta_memory_tools: Optional[bool] = False
    metadata: Optional[Dict] = None
    description: Optional[str] = None
    initial_message_sequence: Optional[List[Message]] = None
    tags: Optional[List[str]] = None


@router.post("/agents", response_model=AgentState)
async def create_agent(
    request: CreateAgentRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new agent."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # Create memory blocks if provided
    if request.memory:
        for block in request.memory.get_blocks():
            server.block_manager.create_or_update_block(block, actor=client)

    # Prepare block IDs
    block_ids = request.block_ids or []
    if request.memory:
        block_ids.extend([b.id for b in request.memory.get_blocks()])

    # Create agent request
    create_params = {
        "description": request.description,
        "metadata_": request.metadata,
        "memory_blocks": [],
        "block_ids": block_ids,
        "tool_ids": request.tool_ids or [],
        "tool_rules": request.tool_rules,
        "include_base_tools": request.include_base_tools,
        "system": request.system,
        "agent_type": request.agent_type,
        "llm_config": request.llm_config,
        "embedding_config": request.embedding_config,
        "initial_message_sequence": request.initial_message_sequence,
        "tags": request.tags,
    }

    if request.name:
        create_params["name"] = request.name

    agent_state = server.create_agent(CreateAgent(**create_params), actor=client)

    return server.agent_manager.get_agent_by_id(agent_state.id, actor=client)


@router.get("/agents/{agent_id}", response_model=AgentState)
async def get_agent(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent by ID."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.agent_manager.get_agent_by_id(agent_id, actor=client)


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete an agent."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    server.agent_manager.delete_agent(agent_id, actor=client)
    return {"status": "success", "message": f"Agent {agent_id} deleted"}


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    llm_config: Optional[LLMConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    message_ids: Optional[List[str]] = None
    memory: Optional[Memory] = None
    tags: Optional[List[str]] = None


@router.patch("/agents/{agent_id}", response_model=AgentState)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Update an agent."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # TODO: Implement update_agent in server
    raise HTTPException(status_code=501, detail="Update agent not yet implemented")

# ============================================================================
# Memory Endpoints
# ============================================================================


@router.get("/agents/{agent_id}/memory", response_model=Memory)
async def get_agent_memory(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent's in-context memory."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.get_agent_memory(agent_id=agent_id, actor=client)


@router.get("/agents/{agent_id}/memory/archival", response_model=ArchivalMemorySummary)
async def get_archival_memory_summary(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get archival memory summary."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.get_archival_memory_summary(agent_id=agent_id, actor=client)


@router.get("/agents/{agent_id}/memory/recall", response_model=RecallMemorySummary)
async def get_recall_memory_summary(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get recall memory summary."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.get_recall_memory_summary(agent_id=agent_id, actor=client)


@router.get("/agents/{agent_id}/messages", response_model=List[Message])
async def get_agent_messages(
    agent_id: str,
    cursor: Optional[str] = None,
    limit: int = 1000,
    use_cache: bool = True,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get messages from an agent.

    Args:
        agent_id: The ID of the agent
        cursor: Cursor for pagination
        limit: Maximum number of messages to return
        use_cache: Control Redis cache behavior (default: True)
        x_user_id: User ID from header
        x_org_id: Organization ID from header
    
    Returns:
        List of messages
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    return server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        before=cursor,
        limit=limit,
        reverse=True,
        use_cache=use_cache,
    )


class SendMessageRequest(BaseModel):
    """Request to send a message to an agent."""
    message: str
    role: str
    name: Optional[str] = None
    stream_steps: bool = False
    stream_tokens: bool = False
    filter_tags: Optional[Dict[str, Any]] = None  # NEW: filter tags support
    use_cache: bool = True  # Control Redis cache behavior


@app.post("/agents/{agent_id}/messages", response_model=MirixResponse)
async def send_message_to_agent(
    agent_id: str,
    request: SendMessageRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Send a message to an agent and get a response.
    
    This endpoint allows sending a single message to an agent for immediate processing.
    The message is processed synchronously through the queue system.
    
    Args:
        agent_id: The ID of the agent to send the message to
        request: The message request containing text, role, and optional filter_tags
        x_user_id: User ID from header
        x_org_id: Organization ID from header
    
    Returns:
        MirixResponse: The agent's response including messages and usage statistics
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    try:
        # Prepare the message
        message_create = MessageCreate(
            role=MessageRole(request.role),
            content=request.message,
            name=request.name,
        )

        # Put message on queue for processing
        put_messages(
            actor=client,
            agent_id=agent_id,
            input_messages=[message_create],
            chaining=True,
            filter_tags=request.filter_tags,  # Pass filter_tags to queue
            use_cache=request.use_cache,  # Pass use_cache to queue
        )

        # For now, return a success response
        # TODO: In the future, this could wait for and return the actual agent response
        return MirixResponse(
            messages=[],
            usage={},
        )

    except Exception as e:
        logger.error("Failed to send message to agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


# ============================================================================
# Tool Endpoints
# ============================================================================


@router.get("/tools", response_model=List[Tool])
async def list_tools(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all tools."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.tool_manager.list_tools(cursor=cursor, limit=limit, actor=client)


@router.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a tool by ID."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.tool_manager.get_tool_by_id(tool_id, actor=client)


@router.post("/tools", response_model=Tool)
async def create_tool(
    tool: Tool,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new tool."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.tool_manager.create_tool(tool, actor=client)


@router.delete("/tools/{tool_id}")
async def delete_tool(
    tool_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a tool."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    server.tool_manager.delete_tool_by_id(tool_id, actor=client)
    return {"status": "success", "message": f"Tool {tool_id} deleted"}


# ============================================================================
# Block Endpoints
# ============================================================================


@router.get("/blocks", response_model=List[Block])
async def list_blocks(
    label: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all blocks."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    # Get default user for block queries (blocks are user-scoped, not client-scoped)
    user = server.user_manager.get_default_user()
    return server.block_manager.get_blocks(user=user, label=label)


@router.get("/blocks/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a block by ID."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    # Get default user for block queries (blocks are user-scoped, not client-scoped)
    user = server.user_manager.get_default_user()
    return server.block_manager.get_block_by_id(block_id, user=user)


@router.post("/blocks", response_model=Block)
async def create_block(
    block: Block,
    user: Optional[User] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a block."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    return server.block_manager.create_or_update_block(block, actor=client, user=user)


@router.delete("/blocks/{block_id}")
async def delete_block(
    block_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a block."""
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    server.block_manager.delete_block(block_id, actor=client)
    return {"status": "success", "message": f"Block {block_id} deleted"}


# ============================================================================
# Configuration Endpoints
# ============================================================================


@router.get("/config/llm", response_model=List[LLMConfig])
async def list_llm_configs(
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available LLM configurations."""
    server = get_server()
    return server.list_llm_models()


@router.get("/config/embedding", response_model=List[EmbeddingConfig])
async def list_embedding_configs(
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available embedding configurations."""
    server = get_server()
    return server.list_embedding_models()


# ============================================================================
# Organization Endpoints
# ============================================================================


@router.get("/organizations", response_model=List[Organization])
async def list_organizations(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List organizations."""
    server = get_server()
    return server.organization_manager.list_organizations(cursor=cursor, limit=limit)


@router.post("/organizations", response_model=Organization)
async def create_organization(
    name: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create an organization."""
    server = get_server()
    return server.organization_manager.create_organization(
        pydantic_org=Organization(name=name)
    )


@router.get("/organizations/{org_id}", response_model=Organization)
async def get_organization(
    org_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an organization by ID."""
    server = get_server()
    try:
        return server.organization_manager.get_organization_by_id(org_id)
    except Exception:
        # If organization doesn't exist, return default or create it
        return server.get_organization_or_default(org_id)


class CreateOrGetOrganizationRequest(BaseModel):
    """Request model for creating or getting an organization."""

    org_id: Optional[str] = None
    name: Optional[str] = None


@router.post("/organizations/create_or_get", response_model=Organization)
async def create_or_get_organization(
    request: CreateOrGetOrganizationRequest,
):
    """
    Create organization if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If org_id is not provided, a random ID will be generated.
    If org_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()
    from mirix.schemas.organization import OrganizationCreate

    # Use provided org_id or generate a new one
    if request.org_id:
        org_id = request.org_id
    else:
        # Generate a random org ID
        import uuid
        org_id = f"org-{uuid.uuid4().hex[:8]}"

    try:
        # Try to get existing organization
        org = server.organization_manager.get_organization_by_id(org_id)
        if org:
            return org
    except Exception:
        pass

    # Create new organization if it doesn't exist
    org_create = OrganizationCreate(
        id=org_id,
        name=request.name or org_id
    )
    org = server.organization_manager.create_organization(
        pydantic_org=Organization(**org_create.model_dump())
    )
    logger.debug("Created new organization: %s", org_id)
    return org


# ============================================================================
# User Endpoints
# ============================================================================


@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a user by ID."""
    server = get_server()
    return server.user_manager.get_user_by_id(user_id)


class CreateOrGetUserRequest(BaseModel):
    """Request model for creating or getting a user."""

    user_id: Optional[str] = None
    name: Optional[str] = None
    org_id: Optional[str] = None


@router.post("/users/create_or_get", response_model=User)
async def create_or_get_user(
    request: CreateOrGetUserRequest,
):
    """
    Create user if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If user_id is not provided, a random ID will be generated.
    If user_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()

    # Use provided user_id or generate a new one
    if request.user_id:
        user_id = request.user_id
    else:
        # Generate a random user ID
        import uuid
        user_id = f"user-{uuid.uuid4().hex[:8]}"

    org_id = request.org_id
    if not org_id:
        org_id = server.organization_manager.DEFAULT_ORG_ID

    try:
        # Try to get existing user
        user = server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass

    from mirix.schemas.user import User as PydanticUser

    # Create a User object with all required fields
    user = server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=request.name or user_id,
            organization_id=org_id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
            status="active"
        )
    )
    logger.debug("Created new user: %s", user_id)
    return user


# ============================================================================
# Client API Endpoints
# ============================================================================


class CreateOrGetClientRequest(BaseModel):
    """Request model for creating or getting a client."""
    client_id: Optional[str] = None
    name: Optional[str] = None
    org_id: Optional[str] = None
    scope: Optional[str] = "read_write"
    status: Optional[str] = "active"


@router.post("/clients/create_or_get", response_model=Client)
async def create_or_get_client(
    request: CreateOrGetClientRequest,
    fail_if_exists: bool = False,
):
    """
    Create client if it doesn't exist, or get existing one.
    
    If client_id is not provided, a random ID will be generated.
    If fail_if_exists is True, return 409 if client already exists.
    """
    server = get_server()

    # Use provided client_id or generate a new one
    if request.client_id:
        client_id = request.client_id
    else:
        import uuid
        client_id = f"client-{uuid.uuid4().hex[:8]}"

    org_id = request.org_id or server.organization_manager.DEFAULT_ORG_ID
    
    try:
        # Try to get existing client
        client = server.client_manager.get_client_by_id(client_id)
        
        if client:
            if fail_if_exists:
                raise HTTPException(
                    status_code=409,
                    detail=f"Client with id '{client_id}' already exists"
                )
            else:
                logger.debug("Client already exists: %s", client_id)
                return JSONResponse(
                    status_code=200,
                    content=client.model_dump(mode='json')
                )
    except Exception as e:
        if fail_if_exists and "already exists" in str(e):
            raise
        pass  # Client doesn't exist, proceed to create

    # Create a Client object with all required fields
    client = server.client_manager.create_client(
        pydantic_client=Client(
            id=client_id,
            name=request.name or client_id,
            organization_id=org_id,
            status=request.status or "active",
            scope=request.scope or "read_write"
        )
    )
    logger.info("Created new client: %s", client_id)
    return JSONResponse(
        status_code=201,
        content=client.model_dump(mode='json')
    )


@router.get("/clients", response_model=List[Client])
async def list_clients(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_org_id: Optional[str] = Header(None),
):
    """
    List all clients with optional pagination.
    """
    server = get_server()
    org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID
    
    clients = server.client_manager.list_clients(
        cursor=cursor,
        limit=limit,
        organization_id=org_id
    )
    return clients


@router.get("/clients/{client_id}", response_model=Client)
async def get_client(client_id: str):
    """
    Get a specific client by ID.
    """
    server = get_server()
    client = server.client_manager.get_client_by_id(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    return client


@router.patch("/clients/{client_id}", response_model=Client)
async def update_client(
    client_id: str,
    update: ClientUpdate,
):
    """
    Update a client's properties.
    """
    server = get_server()
    
    # Ensure the client_id in the path matches the update object
    update.id = client_id
    
    try:
        updated_client = server.client_manager.update_client(update)
        return updated_client
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/clients/{client_id}")
async def delete_client(client_id: str):
    """
    Delete a client by ID.
    """
    server = get_server()
    
    try:
        server.client_manager.delete_client_by_id(client_id)
        return {"message": f"Client {client_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Memory API Endpoints (New)
# ============================================================================


class InitializeMetaAgentRequest(BaseModel):
    """Request model for initializing a meta agent."""

    config: Dict[str, Any]
    project: Optional[str] = None
    update_agents: Optional[bool] = False


@router.post("/agents/meta/initialize", response_model=AgentState)
async def initialize_meta_agent(
    request: InitializeMetaAgentRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Initialize a meta agent with configuration.
    
    This creates a meta memory agent that manages specialized memory agents.
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # Extract config components
    config = request.config
    llm_config = None
    embedding_config = None
    system_prompts = None
    agents_config = None

    # Build create_params by flattening meta_agent_config
    create_params = {
        "llm_config": LLMConfig(**config["llm_config"]),
        "embedding_config": EmbeddingConfig(**config["embedding_config"]),
    }

    # Flatten meta_agent_config fields into create_params
    if "meta_agent_config" in config and config["meta_agent_config"]:
        meta_config = config["meta_agent_config"]
        # Add fields from meta_agent_config directly
        if "agents" in meta_config:
            create_params["agents"] = meta_config["agents"]
        if "system_prompts" in meta_config:
            create_params["system_prompts"] = meta_config["system_prompts"]

    # Check if meta agent already exists for this project
    existing_meta_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    assert len(existing_meta_agents) <= 1, "Only one meta agent can be created for a project"

    if len(existing_meta_agents) == 1:
        meta_agent = existing_meta_agents[0]

        # Only update the meta agent if update_agents is True
        if request.update_agents:
            from mirix.schemas.agent import UpdateMetaAgent

            # Update the existing meta agent
            meta_agent = server.agent_manager.update_meta_agent(
                meta_agent_id=meta_agent.id,
                meta_agent_update=UpdateMetaAgent(**create_params),
                actor=client
            )
    else:
        from mirix.schemas.agent import CreateMetaAgent
        meta_agent = server.agent_manager.create_meta_agent(meta_agent_create=CreateMetaAgent(**create_params), actor=client)

    return meta_agent

class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    user_id: str
    meta_agent_id: str
    messages: List[Dict[str, Any]]
    chaining: bool = True
    verbose: bool = False
    filter_tags: Optional[Dict[str, Any]] = None
    use_cache: bool = True  # Control Redis cache behavior


@router.post("/memory/add")
async def add_memory(
    request: AddMemoryRequest,
    x_org_id: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
):
    """
    Add conversation turns to memory (async via queue).
    
    Messages are queued for asynchronous processing by queue workers.
    Processing happens in the background, allowing for fast API response times.
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)
    
    # If client doesn't exist, create the default client
    if client is None:
        logger.warning("Client %s not found, creating default client", client_id)
        from mirix.services.client_manager import ClientManager
        if client_id == ClientManager.DEFAULT_CLIENT_ID:
            # Create the default client
            client = server.client_manager.create_default_client(org_id)
        else:
            # Client ID was provided but doesn't exist - error
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found. Please create the client first."
            )
    
    # Get the meta agent by ID
    # TODO: need to check if we really need to check if the meta_agent exists here 
    meta_agent = server.agent_manager.get_agent_by_id(request.meta_agent_id, actor=client)

    message = request.messages

    if isinstance(message, list) and "role" in message[0].keys():
        # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]

        # We need to convert the message to the format in "content"
        new_message = []
        for msg in message:
            new_message.append({'type': "text", "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]"})
            new_message.extend(msg["content"])
        message = new_message

    input_messages = convert_message_to_mirix_message(message)

    # Add client scope to filter_tags (create if not provided)
    if request.filter_tags is not None:
        # Create a copy to avoid modifying the original request
        filter_tags = dict(request.filter_tags)
    else:
        # Create new filter_tags if not provided
        filter_tags = {}
    
    # Add or update the "scope" key with the client's scope
    filter_tags["scope"] = client.scope

    # Queue for async processing instead of synchronous execution
    # Note: actor is Client for org-level access control
    #       user_id in request body represents the actual end-user
    put_messages(
        actor=client,
        agent_id=meta_agent.id,
        input_messages=input_messages,
        chaining=request.chaining,
        user_id=request.user_id,  # End-user for data filtering
        verbose=request.verbose,
        filter_tags=filter_tags,
        use_cache=request.use_cache,
    )
    
    logger.debug("Memory queued for processing: %s", meta_agent.id)

    return {
        "success": True,
        "message": "Memory queued for processing",
        "status": "queued",
        "agent_id": meta_agent.id,
        "message_count": len(input_messages),
    }


class RetrieveMemoryRequest(BaseModel):
    """Request model for retrieving memory."""

    user_id: str
    messages: List[Dict[str, Any]]
    limit: int = 10  # Maximum number of items to retrieve per memory type
    local_model_for_retrieval: Optional[str] = None  # Optional local Ollama model for topic extraction
    filter_tags: Optional[Dict[str, Any]] = None  # Optional filter tags for filtering results
    use_cache: bool = True  # Control Redis cache behavior

def retrieve_memories_by_keywords(
    server: SyncServer,
    client: Client,
    user_id: str,
    agent_state: AgentState,
    key_words: str = "",
    limit: int = 10,
    filter_tags: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> dict:
    """
    Helper function to retrieve memories based on keywords using BM25 search.
    
    Args:
        server: The Mirix server instance
        client: The authenticated client application (for authorization)
        user_id: The end-user ID whose memories to retrieve
        agent_state: Agent state (used for configuration)
        key_words: Keywords to search for (empty string returns recent items)
        limit: Maximum number of items to retrieve per memory type
        filter_tags: Tag-based filtering (user_id + filter_tags = complete filter)
        use_cache: Control Redis cache behavior

    Returns:
        Dictionary containing all memory types with their items
    """
    search_method = "bm25"
    
    # Get timezone from user record (if exists)
    try:
        user = server.user_manager.get_user_by_id(user_id)
        timezone_str = user.timezone
    except:
        timezone_str = "UTC"
    memories = {}

    # Get episodic memories (recent + relevant)
    try:
        episodic_manager = server.episodic_memory_manager

        # Get recent episodic memories
        recent_episodic = episodic_manager.list_episodic_memory(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=client,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        # Get relevant episodic memories based on keywords
        relevant_episodic = []
        if key_words:
            relevant_episodic = episodic_manager.list_episodic_memory(
                agent_state=agent_state,  # Not accessed during BM25 search
                actor=client,
                query=key_words,
                search_field="details",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )

        memories["episodic"] = {
            "total_count": episodic_manager.get_total_number_of_items(actor=client),
            "recent": [
                {
                    "id": event.id,
                    "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in recent_episodic
            ],
            "relevant": [
                {
                    "id": event.id,
                    "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in relevant_episodic
            ],
        }
    except Exception as e:
        logger.error("Error retrieving episodic memories: %s", e)
        memories["episodic"] = {"total_count": 0, "recent": [], "relevant": []}

    # Get semantic memories
    try:
        semantic_manager = server.semantic_memory_manager

        semantic_items = semantic_manager.list_semantic_items(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=client,
            query=key_words,
            search_field="details",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["semantic"] = {
            "total_count": semantic_manager.get_total_number_of_items(actor=client),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "details": item.details,
                }
                for item in semantic_items
            ],
        }
    except Exception as e:
        logger.error("Error retrieving semantic memories: %s", e)
        memories["semantic"] = {"total_count": 0, "items": []}

    # Get resource memories
    try:
        resource_manager = server.resource_memory_manager

        resources = resource_manager.list_resources(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=client,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["resource"] = {
            "total_count": resource_manager.get_total_number_of_items(actor=client),
            "items": [
                {
                    "id": resource.id,
                    "title": resource.title,
                    "summary": resource.summary,
                    "resource_type": resource.resource_type,
                }
                for resource in resources
            ],
        }
    except Exception as e:
        logger.error("Error retrieving resource memories: %s", e)
        memories["resource"] = {"total_count": 0, "items": []}

    # Get procedural memories
    try:
        procedural_manager = server.procedural_memory_manager

        procedures = procedural_manager.list_procedures(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=client,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["procedural"] = {
            "total_count": procedural_manager.get_total_number_of_items(actor=client),
            "items": [
                {
                    "id": procedure.id,
                    "entry_type": procedure.entry_type,
                    "summary": procedure.summary,
                }
                for procedure in procedures
            ],
        }
    except Exception as e:
        logger.error("Error retrieving procedural memories: %s", e)
        memories["procedural"] = {"total_count": 0, "items": []}

    # Get knowledge vault items
    try:
        knowledge_vault_manager = server.knowledge_vault_manager

        knowledge_items = knowledge_vault_manager.list_knowledge(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=client,
            query=key_words,
            search_field="caption",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )

        memories["knowledge_vault"] = {
            "total_count": knowledge_vault_manager.get_total_number_of_items(actor=client),
            "items": [
                {
                    "id": item.id,
                    "caption": item.caption,
                }
                for item in knowledge_items
            ],
        }
    except Exception as e:
        logger.error("Error retrieving knowledge vault items: %s", e)
        memories["knowledge_vault"] = {"total_count": 0, "items": []}

    # Get core memory blocks
    try:
        block_manager = server.block_manager

        # Get all blocks for the user (these are the Human and Persona blocks)
        # Note: blocks are user-scoped, not client-scoped
        blocks = block_manager.get_blocks(user=user)

        memories["core"] = {
            "total_count": len(blocks),
            "items": [
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                }
                for block in blocks
            ],
        }
    except Exception as e:
        logger.error("Error retrieving core memory blocks: %s", e)
        memories["core"] = {"total_count": 0, "items": []}

    return memories


@router.post("/memory/retrieve/conversation")
async def retrieve_memory_with_conversation(
    request: RetrieveMemoryRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on conversation context.
    Extracts topics from the conversation messages and uses them to retrieve relevant memories.
    """

    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topics": None,
            "memories": {},
        }

    # Extract topics from the conversation
    # TODO: Consider allowing custom model selection in the future
    llm_config = all_agents[0].llm_config

    # Check if messages have actual content before calling LLM
    has_content = False
    for msg in request.messages:
        if isinstance(msg, dict) and "content" in msg:
            for content_item in msg.get("content", []):
                if isinstance(content_item, dict) and content_item.get("text", "").strip():
                    has_content = True
                    break
            if has_content:
                break

    topics: Optional[str] = None
    if has_content:
        # Prefer local model for topic extraction when explicitly requested
        if request.local_model_for_retrieval:
            topics = extract_topics_with_local_model(
                messages=request.messages,
                model_name=request.local_model_for_retrieval,
            )
            if topics is None:
                logger.warning(
                    "Local topic extraction failed for model %s, falling back to default LLM",
                    request.local_model_for_retrieval,
                )

        if topics is None:
            topics = extract_topics_from_messages(request.messages, llm_config)

        logger.debug("Extracted topics from conversation: %s", topics)
        key_words = topics if topics else ""
    else:
        # No content - skip LLM call and retrieve recent items
        logger.debug("No content in messages - retrieving recent items")
        key_words = ""

    # Retrieve memories using the helper function
    memories = retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=request.user_id,
        agent_state=all_agents[0],
        key_words=key_words,
        limit=request.limit,
        filter_tags=request.filter_tags,
        use_cache=request.use_cache,
    )

    return {
        "success": True,
        "topics": topics,
        "memories": memories,
    }


@router.get("/memory/retrieve/topic")
async def retrieve_memory_with_topic(
    user_id: str,
    topic: str,
    limit: int = 10,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on a topic using BM25 search.
    
    Args:
        user_id: The user ID to retrieve memories for
        topic: The topic/keywords to search for
        limit: Maximum number of items to retrieve per memory type (default: 10)
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topic": topic,
            "memories": {},
        }

    # Retrieve memories using the helper function
    memories = retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=user_id,
        agent_state=all_agents[0],
        key_words=topic,
        limit=limit,
        filter_tags=filter_tags,
        use_cache=use_cache,
    )

    return {
        "success": True,
        "topic": topic,
        "memories": memories,
    }


@router.get("/memory/search")
async def search_memory(
    user_id: str,
    query: str,
    memory_type: str = "all",
    search_field: str = "null",
    search_method: str = "bm25",
    limit: int = 10,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Search for memories using various search methods.
    Similar to the search_in_memory tool function.
    
    Args:
        user_id: The user ID to retrieve memories for
        query: The search query string
        memory_type: Type of memory to search. Options: "episodic", "resource", "procedural", 
                    "knowledge_vault", "semantic", "all" (default: "all")
        search_field: Field to search in. Options vary by memory type:
                     - episodic: "summary", "details"
                     - resource: "summary", "content"
                     - procedural: "summary", "steps"
                     - knowledge_vault: "caption", "secret_value"
                     - semantic: "name", "summary", "details"
                     - For "all": use "null" (default)
        search_method: Search method. Options: "bm25" (default), "embedding"
        limit: Maximum number of results per memory type (default: 10)
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # Get all agents for this client
    all_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this client",
            "query": query,
            "results": [],
            "count": 0,
        }

    agent_state = all_agents[0]
    
    # Get timezone from user record (if exists)
    try:
        user = server.user_manager.get_user_by_id(user_id)
        timezone_str = user.timezone
    except:
        timezone_str = "UTC"

    # Validate search parameters
    if memory_type == "resource" and search_field == "content" and search_method == "embedding":
        return {
            "success": False,
            "error": "embedding is not supported for resource memory's 'content' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if memory_type == "knowledge_vault" and search_field == "secret_value" and search_method == "embedding":
        return {
            "success": False,
            "error": "embedding is not supported for knowledge_vault memory's 'secret_value' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if memory_type == "all":
        search_field = "null"

    # Collect results from requested memory types
    all_results = []

    # Search episodic memories
    if memory_type in ["episodic", "all"]:
        try:
            episodic_memories = server.episodic_memory_manager.list_episodic_memory(
                actor=client,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "episodic",
                    "id": x.id,
                    "timestamp": x.occurred_at.isoformat() if x.occurred_at else None,
                    "event_type": x.event_type,
                    "actor": x.actor,
                    "summary": x.summary,
                    "details": x.details,
                }
                for x in episodic_memories
            ])
        except Exception as e:
            logger.error("Error searching episodic memories: %s", e)

    # Search resource memories
    if memory_type in ["resource", "all"]:
        try:
            resource_memories = server.resource_memory_manager.list_resources(
                actor=client,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else ("summary" if search_method == "embedding" else "content"),
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "resource",
                    "id": x.id,
                    "resource_type": x.resource_type,
                    "title": x.title,
                    "summary": x.summary,
                    "content": x.content[:200] if x.content else None,  # Truncate content for response
                }
                for x in resource_memories
            ])
        except Exception as e:
            logger.error("Error searching resource memories: %s", e)

    # Search procedural memories
    if memory_type in ["procedural", "all"]:
        try:
            procedural_memories = server.procedural_memory_manager.list_procedures(
                actor=client,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "procedural",
                    "id": x.id,
                    "entry_type": x.entry_type,
                    "summary": x.summary,
                    "steps": x.steps,
                }
                for x in procedural_memories
            ])
        except Exception as e:
            logger.error("Error searching procedural memories: %s", e)

    # Search knowledge vault
    if memory_type in ["knowledge_vault", "all"]:
        try:
            knowledge_vault_memories = server.knowledge_vault_manager.list_knowledge(
                actor=client,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "caption",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "knowledge_vault",
                    "id": x.id,
                    "entry_type": x.entry_type,
                    "source": x.source,
                    "sensitivity": x.sensitivity,
                    "secret_value": x.secret_value,
                    "caption": x.caption,
                }
                for x in knowledge_vault_memories
            ])
        except Exception as e:
            logger.error("Error searching knowledge vault: %s", e)

    # Search semantic memories
    if memory_type in ["semantic", "all"]:
        try:
            semantic_memories = server.semantic_memory_manager.list_semantic_items(
                actor=client,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "semantic",
                    "id": x.id,
                    "name": x.name,
                    "summary": x.summary,
                    "details": x.details,
                    "source": x.source,
                }
                for x in semantic_memories
            ])
        except Exception as e:
            logger.error("Error searching semantic memories: %s", e)

    return {
        "success": True,
        "query": query,
        "memory_type": memory_type,
        "search_field": search_field,
        "search_method": search_method,
        "results": all_results,
        "count": len(all_results),
    }


# ============================================================================
# Include Router and Exports
# ============================================================================

app.include_router(router)

# Export both app and router for external use
# - Use 'app' to run the server directly
# - Use 'router' to include routes in another FastAPI application
# - Use 'initialize' and 'cleanup' functions for manual lifecycle management of
#   you're using router and not app.
__all__ = ["app", "router", "initialize", "cleanup"]


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
