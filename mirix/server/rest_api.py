"""
FastAPI REST API server for Mirix.
This provides HTTP endpoints that wrap the SyncServer functionality,
allowing MirixClient instances to communicate with a cloud-hosted server.
"""

import copy
import functools
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.client import Client, ClientCreate, ClientUpdate
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
from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate
from mirix.schemas.raw_memory import (
    RawMemoryItem,
    RawMemoryItemUpdate,
    SearchRawMemoryRequest,
    SearchRawMemoryResponse,
)
from mirix.schemas.resource_memory import ResourceMemoryItemUpdate
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
)
from mirix.schemas.semantic_memory import SemanticMemoryItemUpdate
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.schemas.user import User
from mirix.server.server import SyncServer
from mirix.settings import model_settings, settings
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

    # Flush LangFuse traces before shutdown
    try:
        from mirix.observability import flush_langfuse, shutdown_langfuse

        logger.info("Flushing LangFuse traces...")
        flush_success = flush_langfuse(timeout=10.0)  # Increased timeout for graceful shutdown
        if flush_success:
            logger.info("LangFuse traces flushed successfully")
        else:
            logger.warning("LangFuse trace flush completed with warnings")
        shutdown_langfuse()
        logger.info("LangFuse observability shut down")
    except Exception as e:
        logger.warning(f"Error during LangFuse shutdown: {e}")

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
# Request Context (for accessing request in decorators)
# ============================================================================

from contextvars import ContextVar

# Stores the current request for access by decorators
_current_request: ContextVar[Optional[Request]] = ContextVar("current_request", default=None)


def get_current_request() -> Optional[Request]:
    """Get the current request from context."""
    return _current_request.get()


# Middleware to store request in context (runs for all requests, lightweight)
@app.middleware("http")
async def store_request_context(request: Request, call_next):
    """Store the request in context so decorators can access it."""
    token = _current_request.set(request)
    try:
        return await call_next(request)
    finally:
        _current_request.reset(token)


# ============================================================================
# LangFuse Tracing Decorator (for agentic endpoints only)
# ============================================================================


def with_langfuse_tracing(func):
    """
    Decorator that adds LangFuse tracing to agentic endpoints.

    Usage:
        @router.post("/memory/add")
        @with_langfuse_tracing
        async def add_memory(...):
            ...

    The request is automatically accessed from context - no special
    parameters needed on the decorated function.
    """
    import uuid
    from typing import cast

    from langfuse.types import TraceContext

    from mirix.observability import get_langfuse_client, is_langfuse_enabled
    from mirix.observability.context import clear_trace_context, set_trace_context

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get request from context (set by middleware)
        request = get_current_request()

        # Skip tracing if LangFuse is disabled or no request available
        if not is_langfuse_enabled() or not request:
            return await func(*args, **kwargs)

        langfuse = get_langfuse_client()
        if not langfuse:
            return await func(*args, **kwargs)

        # Extract metadata from request
        method = request.method
        path = request.url.path
        user_id = request.headers.get("x-user-id")
        client_id = request.headers.get("x-client-id")
        org_id = request.headers.get("x-org-id")
        session_id = f"{client_id}-{uuid.uuid4().hex[:8]}"

        # Generate a unique trace_id for each request
        trace_id = uuid.uuid4().hex

        with langfuse.start_as_current_observation(
            name=f"{method} {path}",
            as_type="span",
            trace_context=cast(TraceContext, {"trace_id": trace_id}),
        ) as span:
            try:
                observation_id = getattr(span, "id", None)
                logger.debug(
                    f"LangFuse trace created: trace_id={trace_id}, observation_id={observation_id}, path={path}"
                )

                langfuse.update_current_trace(
                    name=f"{method} {path}",
                    user_id=user_id or client_id,
                    session_id=session_id,
                    metadata={
                        "method": method,
                        "path": path,
                        "client_id": client_id,
                        "org_id": org_id,
                        "user_agent": request.headers.get("user-agent"),
                    },
                )

                set_trace_context(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    user_id=user_id or client_id,
                    session_id=session_id,
                )

                # Execute the actual endpoint function
                result = await func(*args, **kwargs)

                try:
                    span.update(output={"status": "completed"})
                    logger.debug(f"LangFuse trace completed: trace_id={trace_id}")
                except Exception as e:
                    logger.debug(f"Failed to update trace: {e}")

                return result

            except Exception as e:
                try:
                    span.update(output={"error": str(e)}, level="ERROR")
                except Exception as trace_error:
                    logger.debug(f"Failed to update trace on error: {trace_error}")
                raise

            finally:
                clear_trace_context()

    return wrapper


# Middleware to resolve X-API-Key into X-Client-Id/X-Org-Id for all routes
@app.middleware("http")
async def inject_client_org_headers(request: Request, call_next):
    """
    If a request includes X-API-Key, resolve it to client/org and inject
    X-Client-Id and X-Org-Id headers so downstream endpoints don't need to
    handle X-API-Key directly.
    """
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        try:
            client_id, org_id = get_client_and_org(x_api_key=x_api_key)

            # Replace or add headers so FastAPI dependencies can read them
            headers = [
                (name, value)
                for name, value in request.scope.get("headers", [])
                if name.lower() not in {b"x-client-id", b"x-org-id"}
            ]
            headers.append((b"x-client-id", client_id.encode()))
            headers.append((b"x-org-id", org_id.encode()))
            request.scope["headers"] = headers
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return await call_next(request)


# ============================================================================
# Helper Functions
# ============================================================================


def get_client_and_org(
    x_client_id: Optional[str] = None,
    x_org_id: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> tuple[str, str]:
    """
    Get client_id and org_id from headers or use defaults.

    Returns:
        tuple[str, str]: (client_id, org_id)
    """
    server = get_server()

    if x_api_key:
        # get_client_by_api_key already verifies the API key hash internally
        client = server.client_manager.get_client_by_api_key(x_api_key)
        if not client:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if client.is_deleted or client.status != "active":
            raise HTTPException(status_code=403, detail="Client is inactive or deleted")
        client_id = client.id
        org_id = client.organization_id or server.organization_manager.DEFAULT_ORG_ID
    elif x_client_id:
        client_id = x_client_id
        org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID
    else:
        raise HTTPException(
            status_code=401,
            detail="X-API-Key header is required for this endpoint",
        )

    return client_id, org_id


def extract_topics_and_temporal_info(
    messages: List[Dict[str, Any]], llm_config: LLMConfig
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract topics AND temporal expressions from a list of messages using LLM.

    This function analyzes messages to extract both semantic topics and temporal
    expressions (like "today", "yesterday", "last week") for more accurate memory retrieval.

    Args:
        messages: List of message dictionaries (OpenAI format)
        llm_config: LLM configuration to use for extraction

    Returns:
        Tuple of (topics, temporal_expression) where both can be None
        - topics: String with topics separated by ';'
        - temporal_expression: String with temporal phrase or None if not found
    """
    try:
        if isinstance(messages, list) and "role" in messages[0].keys():
            # Convert from OpenAI format to internal format
            new_messages = []
            for msg in messages:
                new_messages.append(
                    {
                        "type": "text",
                        "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]",
                    }
                )
                new_messages.extend(msg["content"])
            messages = new_messages

        temporary_messages = convert_message_to_mirix_message(messages)
        temporary_messages = [
            prepare_input_message_create(
                msg,
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            )
            for msg in temporary_messages
        ]

        # Add instruction message for topic and temporal extraction
        temporary_messages.append(
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.user,
                    content="The above are the inputs from the user. Please extract:\n"
                    '1. Topic(s): Brief description of what the user is focusing on. If multiple topics, separate with ";"\n'
                    '2. Temporal expression: Any time-related phrases like "today", "yesterday", "last week", "this month", etc.\n'
                    "Call the function `update_topic_and_time` with both extracted values. If no temporal expression is found, leave it empty.",
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
                    content="You are a helpful assistant that extracts topics and temporal information from the user's input.",
                ),
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            ),
        ] + temporary_messages

        # Define the function for topic and temporal extraction
        functions = [
            {
                "name": "update_topic_and_time",
                "description": "Update the topic and temporal information of the conversation/content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": 'The topic(s) of the conversation. If multiple topics, separate with ";".',
                        },
                        "temporal_expression": {
                            "type": "string",
                            "description": 'Any temporal/time-related expression found in the input (e.g., "today", "yesterday", "last week", "this month"). Leave empty if no temporal expression found.',
                        },
                    },
                    "required": ["topic"],  # temporal_expression is optional
                },
            }
        ]

        # Use LLMClient to extract topics and temporal info
        llm_client = LLMClient.create(llm_config=llm_config)

        if llm_client:
            response = llm_client.send_llm_request(
                messages=temporary_messages,
                tools=functions,
                stream=False,
                force_tool_call="update_topic_and_time",
            )

            # Extract topics and temporal expression from the response
            for choice in response.choices:
                if (
                    hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls is not None
                    and len(choice.message.tool_calls) > 0
                ):
                    try:
                        function_args = json.loads(choice.message.tool_calls[0].function.arguments)
                        topics = function_args.get("topic")
                        temporal_expr = function_args.get("temporal_expression", "")
                        # Clean up empty strings
                        temporal_expr = temporal_expr.strip() if temporal_expr else None
                        logger.debug("Extracted topics: %s, temporal: %s", topics, temporal_expr)
                        return topics, temporal_expr
                    except (json.JSONDecodeError, KeyError) as parse_error:
                        logger.warning("Failed to parse extraction response: %s", parse_error)
                        continue

    except Exception as e:
        logger.error("Error in extracting topics and temporal info: %s", e)

    return None, None


def extract_topics_from_messages(messages: List[Dict[str, Any]], llm_config: LLMConfig) -> Optional[str]:
    """
    Extract topics from a list of messages using LLM.

    This is a legacy function maintained for backward compatibility.
    New code should use extract_topics_and_temporal_info() for enhanced functionality.

    Args:
        messages: List of message dictionaries (OpenAI format)
        llm_config: LLM configuration to use for topic extraction

    Returns:
        Extracted topics as a string (separated by ';') or None if extraction fails
    """
    topics, _ = extract_topics_and_temporal_info(messages, llm_config)
    return topics


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
    from mirix.orm.errors import NoResultFound

    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    try:
        return server.agent_manager.get_agent_by_id(agent_id, actor=client)
    except NoResultFound as e:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or not accessible")


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


class UpdateSystemPromptRequest(BaseModel):
    """Request model for updating an agent's system prompt."""

    system_prompt: str = Field(..., description="The new system prompt")


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


@router.patch("/agents/by-name/{agent_name}/system", response_model=AgentState)
async def update_agent_system_prompt_by_name(
    agent_name: str,
    request: UpdateSystemPromptRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Update an agent's system prompt by agent name.

    This endpoint accepts an agent name (e.g., "episodic", "semantic", "core",
    "meta_memory_agent") and resolves it to the agent_id for the authenticated client.

    The full agent name pattern is typically:
    - "meta_memory_agent" (the parent meta agent)
    - "meta_memory_agent_episodic_memory_agent" → short name: "episodic"
    - "meta_memory_agent_semantic_memory_agent" → short name: "semantic"
    - "meta_memory_agent_core_memory_agent" → short name: "core"
    - "meta_memory_agent_procedural_memory_agent" → short name: "procedural"
    - "meta_memory_agent_resource_memory_agent" → short name: "resource"
    - "meta_memory_agent_knowledge_vault_memory_agent" → short name: "knowledge_vault"
    - "meta_memory_agent_reflexion_agent" → short name: "reflexion"

    Args:
        agent_name: Short name or full name of the agent
        request: UpdateSystemPromptRequest with the new system prompt

    Returns:
        AgentState: The updated agent state

    Example:
        PATCH /agents/by-name/episodic/system
        {
            "system_prompt": "You are an episodic memory agent for sales..."
        }
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # List all top-level agents for this client
    top_level_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    # Also get sub-agents (children of meta agent)
    all_agents = list(top_level_agents)
    for agent in top_level_agents:
        if agent.name == "meta_memory_agent":
            # Get sub-agents
            sub_agents = server.agent_manager.list_agents(actor=client, parent_id=agent.id, limit=1000)
            all_agents.extend(sub_agents)
            break

    # Try to find the agent by name
    # First try exact match with full name
    matching_agent = None
    for agent in all_agents:
        if agent.name == agent_name:
            matching_agent = agent
            break

    # If not found, try short name match (e.g., "episodic" → "meta_memory_agent_episodic_memory_agent")
    if not matching_agent:
        for agent in all_agents:
            # Extract short name from full name
            # e.g., "meta_memory_agent_episodic_memory_agent" → "episodic"
            if agent.name and "meta_memory_agent_" in agent.name:
                short_name = (
                    agent.name.replace("meta_memory_agent_", "").replace("_memory_agent", "").replace("_agent", "")
                )
                if short_name == agent_name:
                    matching_agent = agent
                    break

    # If still not found, raise error with helpful message
    if not matching_agent:
        # Build helpful error message with available agents
        available_agents = []
        for agent in all_agents:
            full_name = agent.name
            if "meta_memory_agent_" in full_name and full_name != "meta_memory_agent":
                short_name = (
                    full_name.replace("meta_memory_agent_", "").replace("_memory_agent", "").replace("_agent", "")
                )
                available_agents.append(f"'{short_name}' (full: {full_name})")
            else:
                available_agents.append(f"'{full_name}'")

        available_list = ", ".join(available_agents[:5])  # Show first 5
        if len(available_agents) > 5:
            available_list += f", and {len(available_agents) - 5} more"

        error_detail = f"Agent with name '{agent_name}' not found for client {client_id}. "
        if available_agents:
            error_detail += f"Available agents: {available_list}"
        else:
            error_detail += "No agents found for this client. Please initialize agents first."

        raise HTTPException(status_code=404, detail=error_detail)

    # Call the update_agent_system_prompt endpoint logic
    updated_agent = server.agent_manager.update_system_prompt(
        agent_id=matching_agent.id, system_prompt=request.system_prompt, actor=client
    )

    return updated_agent


@router.patch("/agents/{agent_id}/system", response_model=AgentState)
async def update_agent_system_prompt(
    agent_id: str,
    request: UpdateSystemPromptRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Update an agent's system prompt by agent ID.

    This endpoint updates the agent's system prompt and triggers a rebuild
    of the system message in the agent's message history.

    The update process:
    1. Updates the agent.system field in PostgreSQL
    2. Updates the agent.system field in Redis cache
    3. Creates a new system message
    4. Updates message_ids[0] to reference the new system message

    Args:
        agent_id: ID of the agent to update (e.g., "agent-123")
        request: UpdateSystemPromptRequest with the new system prompt

    Returns:
        AgentState: The updated agent state

    Example:
        PATCH /agents/agent-123/system
        {
            "system_prompt": "You are a helpful sales assistant."
        }
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    updated_agent = server.agent_manager.update_system_prompt(
        agent_id=agent_id, system_prompt=request.system_prompt, actor=client
    )

    return updated_agent


# ============================================================================
# Memory Endpoints
# ============================================================================
class SendMessageRequest(BaseModel):
    """Request to send a message to an agent."""

    message: str
    role: str
    user_id: Optional[str] = None  # End-user ID for message attribution
    name: Optional[str] = None
    stream_steps: bool = False
    stream_tokens: bool = False
    filter_tags: Optional[Dict[str, Any]] = None  # Filter tags support
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
        request: The message request containing text, role, user_id, and optional filter_tags
        x_client_id: Client ID from header (identifies the client application)
        x_org_id: Organization ID from header

    Returns:
        MirixResponse: The agent's response including messages and usage statistics

    Note:
        If user_id is not provided in the request, messages will be associated with
        the default user (DEFAULT_USER_ID).
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
            user_id=request.user_id,  # Pass user_id to queue
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
    user = server.user_manager.get_admin_user()
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
    # Get admin user for block queries (blocks are user-scoped, not client-scoped)
    user = server.user_manager.get_admin_user()
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
    return server.organization_manager.create_organization(pydantic_org=Organization(name=name))


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
    org_create = OrganizationCreate(id=org_id, name=request.name or org_id)
    org = server.organization_manager.create_organization(pydantic_org=Organization(**org_create.model_dump()))
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


@router.post("/users/create_or_get", response_model=User)
async def create_or_get_user(
    request: CreateOrGetUserRequest,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Create user if it doesn't exist, or get existing one.

    **Accepts both JWT (dashboard) and Client API Key (programmatic access).**

    The user will be associated with the authenticated client.
    The organization is determined from the API key or JWT token.
    If user_id is not provided, a random ID will be generated.
    """
    server = get_server()

    # Accept JWT or injected headers (from API key middleware)
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Extract client_id and org_id from client object
    client_id = client.id
    org_id = client.organization_id or server.organization_manager.DEFAULT_ORG_ID

    # Use provided user_id or generate a new one
    if request.user_id:
        user_id = request.user_id
    else:
        # Generate a random user ID
        import uuid

        user_id = f"user-{uuid.uuid4().hex[:8]}"

    try:
        # Try to get existing user
        user = server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass

    from mirix.schemas.user import User as PydanticUser

    # Create a User object with all required fields, linked to the client
    user = server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=request.name or user_id,
            organization_id=org_id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
            status="active",
        ),
        client_id=client_id,  # Associate user with client
    )
    logger.debug("Created new user: %s for client: %s", user_id, client_id)
    return user


@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """
    Soft delete a user by ID.

    This marks the user and all associated records as deleted by setting is_deleted=True.
    The records remain in the database but are filtered out from queries.

    Associated records that are soft deleted:
    - Episodic memories for this user
    - Semantic memories for this user
    - Procedural memories for this user
    - Resource memories for this user
    - Knowledge vault items for this user
    - Messages for this user
    - Blocks for this user
    """
    server = get_server()

    try:
        server.user_manager.delete_user_by_id(user_id)
        return {"message": f"User {user_id} soft deleted successfully"}
    except Exception as e:
        error_msg = str(e)
        # Provide a better error message if user not found or already deleted
        if "not found" in error_msg.lower() or "no result" in error_msg.lower():
            raise HTTPException(status_code=404, detail=f"User {user_id} not found or already deleted")
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/users/{user_id}/memories")
async def delete_user_memories(user_id: str):
    """
    Hard delete all memories, messages, and blocks for a user.

    This permanently removes data records while preserving the user record.
    Use this for data cleanup/purging without affecting the user account itself.

    Records that are PERMANENTLY DELETED:
    - Episodic memories for this user
    - Semantic memories for this user
    - Procedural memories for this user
    - Resource memories for this user
    - Knowledge vault items for this user
    - Messages for this user
    - Blocks for this user

    Records that are PRESERVED:
    - User record

    Warning: This operation is irreversible. Deleted data cannot be recovered.
    """
    server = get_server()

    try:
        server.user_manager.delete_memories_by_user_id(user_id)
        return {
            "message": f"All memories for user {user_id} hard deleted successfully",
            "preserved": ["user"],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/users", response_model=List[User])
async def list_users(
    cursor: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
):
    """
    List all users for the authenticated client.

    **Requires JWT authentication (dashboard only).**
    """
    # Require admin JWT authentication
    client_payload = get_current_admin(authorization)
    client_id = client_payload["sub"]

    server = get_server()

    users = server.user_manager.list_users(
        cursor=cursor,
        limit=limit,
        client_id=client_id,  # Filter by client
    )
    return users


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
                    detail=f"Client with id '{client_id}' already exists",
                )
            else:
                logger.debug("Client already exists: %s", client_id)
                return JSONResponse(status_code=200, content=client.model_dump(mode="json"))
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
            scope=request.scope or "read_write",
        )
    )
    logger.info("Created new client: %s", client_id)
    return JSONResponse(status_code=201, content=client.model_dump(mode="json"))


@router.get("/clients", response_model=List[Client])
async def list_clients(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_org_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    List all clients with optional pagination.

    **Requires JWT authentication (dashboard only).**
    """
    # Require admin JWT authentication for listing clients
    get_current_admin(authorization)

    server = get_server()
    org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID

    clients = server.client_manager.list_clients(cursor=cursor, limit=limit, organization_id=org_id)
    return clients


@router.get("/clients/{client_id}", response_model=Client)
async def get_client(
    client_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Get a specific client by ID.

    **Requires JWT authentication (dashboard only).**
    """
    # Require admin JWT authentication
    get_current_admin(authorization)

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
    Soft delete a client by ID.

    This marks the client and all associated records (agents, tools, blocks) as deleted
    by setting is_deleted=True. The records remain in the database but are filtered
    out from queries.

    Associated records that are soft deleted:
    - Agents created by this client
    - Tools created by this client
    - Blocks created by this client

    Memory records (episodic, semantic, etc.) remain but are filtered by client.is_deleted.
    """
    server = get_server()

    try:
        server.client_manager.delete_client_by_id(client_id)
        return {"message": f"Client {client_id} soft deleted successfully"}
    except Exception as e:
        error_msg = str(e)
        # Provide a better error message if client not found or already deleted
        if "not found" in error_msg.lower() or "no result" in error_msg.lower():
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found or already deleted",
            )
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/clients/{client_id}/memories")
async def delete_client_memories(client_id: str):
    """
    Hard delete all memories, messages, and blocks for a client.

    This permanently removes data records while preserving the client configuration.
    Use this for data cleanup/purging without affecting the client, agents, or tools.

    Records that are PERMANENTLY DELETED:
    - Episodic memories for this client
    - Semantic memories for this client
    - Procedural memories for this client
    - Resource memories for this client
    - Knowledge vault items for this client
    - Messages for this client
    - Blocks created by this client

    Records that are PRESERVED:
    - Client record
    - Agents created by this client
    - Tools created by this client

    Warning: This operation is irreversible. Deleted data cannot be recovered.
    """
    server = get_server()

    try:
        server.client_manager.delete_memories_by_client_id(client_id)
        return {
            "message": f"All memories for client {client_id} hard deleted successfully",
            "preserved": ["client", "agents", "tools"],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# API Key Management Endpoints (Dashboard Only - JWT Authentication Required)
# ============================================================================


class CreateApiKeyRequest(BaseModel):
    """Request model for creating an API key."""

    name: Optional[str] = Field(None, description="Optional name/label for the API key")
    permission: Optional[str] = Field("all", description="Permission level: all, restricted, read_only")
    user_id: Optional[str] = Field(None, description="User ID this API key is associated with")


class CreateApiKeyResponse(BaseModel):
    """Response model for API key creation - includes the raw API key (only shown once)."""

    id: str
    client_id: str
    name: Optional[str]
    api_key: str  # Raw API key - only returned at creation time
    status: str
    permission: str  # Permission level: all, restricted, read_only
    created_at: Optional[datetime]


@router.post("/clients/{client_id}/api-keys", response_model=CreateApiKeyResponse)
async def create_client_api_key(
    client_id: str,
    request: CreateApiKeyRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Create a new API key for a client.

    **Requires JWT authentication (dashboard only).**

    The raw API key is only returned once at creation time. Store it securely.
    Subsequent requests will only show the key ID, not the raw key.

    Returns:
        CreateApiKeyResponse with the raw API key (only shown once)
    """
    # Require admin JWT authentication
    get_current_admin(authorization)

    from mirix.security.api_keys import generate_api_key

    server = get_server()

    # Verify client exists
    client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    # Generate new API key
    raw_api_key = generate_api_key()

    # Create API key record (stores hashed version)
    api_key_record = server.client_manager.create_client_api_key(
        client_id=client_id,
        api_key=raw_api_key,
        name=request.name,
        permission=request.permission or "all",
        user_id=request.user_id,
    )

    return CreateApiKeyResponse(
        id=api_key_record.id,
        client_id=api_key_record.client_id,
        name=api_key_record.name,
        api_key=raw_api_key,  # Return raw key only at creation
        status=api_key_record.status,
        permission=api_key_record.permission,
        created_at=api_key_record.created_at,
    )


class ApiKeyInfo(BaseModel):
    """API key information (without the raw key)."""

    id: str
    client_id: str
    name: Optional[str]
    status: str
    created_at: Optional[datetime]


@router.get("/clients/{client_id}/api-keys", response_model=List[ApiKeyInfo])
async def list_client_api_keys(
    client_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    List all API keys for a client.

    **Requires JWT authentication (dashboard only).**

    Note: The raw API key values are never returned after creation.
    Only metadata (id, name, status, created_at) is shown.
    """
    # Require admin JWT authentication
    get_current_admin(authorization)

    server = get_server()

    # Verify client exists
    client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    api_keys = server.client_manager.list_client_api_keys(client_id)

    return [
        ApiKeyInfo(
            id=key.id,
            client_id=key.client_id,
            name=key.name,
            status=key.status,
            created_at=key.created_at,
        )
        for key in api_keys
    ]


@router.delete("/clients/{client_id}/api-keys/{api_key_id}")
async def delete_client_api_key(
    client_id: str,
    api_key_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Delete an API key permanently.

    **Requires JWT authentication (dashboard only).**

    This permanently removes the API key from the database.
    Any applications using this key will immediately stop working.
    """
    # Require admin JWT authentication
    get_current_admin(authorization)

    server = get_server()

    # Verify client exists
    client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    try:
        server.client_manager.delete_client_api_key(api_key_id)
        return {
            "message": f"API key {api_key_id} deleted successfully",
            "id": api_key_id,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"API key {api_key_id} not found")


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

    # Check if meta agent already exists for this client
    # list_agents now automatically filters by client (organization_id + _created_by_id)
    existing_meta_agents = server.agent_manager.list_agents(actor=client, limit=1000)

    assert len(existing_meta_agents) <= 1, "Only one meta agent can be created per client"

    if len(existing_meta_agents) == 1:
        meta_agent = existing_meta_agents[0]

        # Only update the meta agent if update_agents is True
        if request.update_agents:
            from mirix.schemas.agent import UpdateMetaAgent

            # DEBUG: Log what we're passing to update_meta_agent
            logger.debug("[INIT META AGENT] create_params for UpdateMetaAgent: %s", create_params)
            logger.debug(
                "[INIT META AGENT] 'agents' in create_params: %s",
                "agents" in create_params,
            )
            if "agents" in create_params:
                logger.debug("[INIT META AGENT] agents list: %s", create_params["agents"])

            # Update the existing meta agent
            meta_agent = server.agent_manager.update_meta_agent(
                meta_agent_id=meta_agent.id,
                meta_agent_update=UpdateMetaAgent(**create_params),
                actor=client,
            )
    else:
        from mirix.schemas.agent import CreateMetaAgent

        meta_agent = server.agent_manager.create_meta_agent(
            meta_agent_create=CreateMetaAgent(**create_params), actor=client
        )

    return meta_agent


class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    user_id: Optional[str] = None  # Optional - uses admin user if not provided
    meta_agent_id: str
    messages: List[Dict[str, Any]]
    chaining: bool = True
    verbose: bool = False
    filter_tags: Optional[Dict[str, Any]] = None
    use_cache: bool = True  # Control Redis cache behavior
    occurred_at: Optional[str] = None  # Optional ISO 8601 timestamp string for episodic memory


@router.post("/memory/add")
@with_langfuse_tracing
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
                detail=f"Client {client_id} not found. Please create the client first.",
            )

    # Get the meta agent by ID
    # TODO: need to check if we really need to check if the meta_agent exists here
    meta_agent = server.agent_manager.get_agent_by_id(request.meta_agent_id, actor=client)

    # If user_id is not provided, use the admin user for this client
    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    message = request.messages

    if isinstance(message, list) and "role" in message[0].keys():
        # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]
        # OR the simpler format: [{"role": "user", "content": "Hello world"}]

        # We need to convert the message to the format in "content"
        new_message = []
        for msg in message:
            new_message.append(
                {
                    "type": "text",
                    "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]",
                }
            )

            # Handle both string and list content
            content = msg["content"]
            if isinstance(content, str):
                # Content is a string - convert to proper format
                new_message.append({"type": "text", "text": content})
            elif isinstance(content, list):
                # Content is already a list - extend as before
                new_message.extend(content)
            else:
                raise ValueError(f"Invalid content type: {type(content)}")
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
    #       user_id represents the actual end-user (or admin user if not provided)
    put_messages(
        actor=client,
        agent_id=meta_agent.id,
        input_messages=input_messages,
        chaining=request.chaining,
        user_id=user_id,  # End-user for data filtering (or admin user)
        verbose=request.verbose,
        filter_tags=filter_tags,
        use_cache=request.use_cache,
        occurred_at=request.occurred_at,  # Optional timestamp for episodic memory
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

    user_id: Optional[str] = None  # Optional - uses admin user if not provided
    messages: List[Dict[str, Any]]
    limit: int = 10  # Maximum number of items to retrieve per memory type
    local_model_for_retrieval: Optional[str] = None  # Optional local Ollama model for topic extraction
    filter_tags: Optional[Dict[str, Any]] = None  # Optional filter tags for filtering results
    use_cache: bool = True  # Control Redis cache behavior
    # NEW: Optional date range for temporal filtering (ISO 8601 format)
    start_date: Optional[str] = None  # e.g., "2025-11-19T00:00:00" or "2025-11-19T00:00:00+00:00"
    end_date: Optional[str] = None  # e.g., "2025-11-19T23:59:59" or "2025-11-19T23:59:59+00:00"


def retrieve_memories_by_keywords(
    server: SyncServer,
    client: Client,
    user_id: str,
    agent_state: AgentState,
    key_words: str = "",
    limit: int = 10,
    filter_tags: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
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
        start_date: Optional start datetime for filtering episodic memories by occurred_at (inclusive)
        end_date: Optional end datetime for filtering episodic memories by occurred_at (inclusive)

    Returns:
        Dictionary containing all memory types with their items
    """
    search_method = "bm25"

    # Log temporal filtering for monitoring
    if start_date or end_date:
        logger.info(
            "Temporal filtering enabled for episodic memories: start=%s, end=%s",
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )

    # Get timezone from user record (if exists)
    try:
        user = server.user_manager.get_user_by_id(user_id)
        timezone_str = user.timezone
    except:
        timezone_str = "UTC"
    memories = {}

    # Get episodic memories (recent + relevant) with optional temporal filtering
    try:
        episodic_manager = server.episodic_memory_manager

        # Get recent episodic memories with temporal filter
        recent_episodic = episodic_manager.list_episodic_memory(
            agent_state=agent_state,  # Not accessed during BM25 search
            user=user,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
            start_date=start_date,  # NEW: Temporal filtering
            end_date=end_date,  # NEW: Temporal filtering
        )

        # Get relevant episodic memories based on keywords with temporal filter
        relevant_episodic = []
        if key_words:
            relevant_episodic = episodic_manager.list_episodic_memory(
                agent_state=agent_state,  # Not accessed during BM25 search
                user=user,
                query=key_words,
                search_field="details",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,  # Include filter_tags for consistency
                start_date=start_date,  # NEW: Temporal filtering
                end_date=end_date,  # NEW: Temporal filtering
            )

        memories["episodic"] = {
            "total_count": episodic_manager.get_total_number_of_items(user=user),
            "recent": [
                {
                    "id": event.id,
                    "timestamp": (event.occurred_at.isoformat() if event.occurred_at else None),
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in recent_episodic
            ],
            "relevant": [
                {
                    "id": event.id,
                    "timestamp": (event.occurred_at.isoformat() if event.occurred_at else None),
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
            user=user,
            query=key_words,
            search_field="details",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["semantic"] = {
            "total_count": semantic_manager.get_total_number_of_items(user=user),
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
            user=user,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["resource"] = {
            "total_count": resource_manager.get_total_number_of_items(user=user),
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
            user=user,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            use_cache=use_cache,
        )

        memories["procedural"] = {
            "total_count": procedural_manager.get_total_number_of_items(user=user),
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
            user=user,
            query=key_words,
            search_field="caption",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )

        memories["knowledge_vault"] = {
            "total_count": knowledge_vault_manager.get_total_number_of_items(user=user),
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
@with_langfuse_tracing
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

    # If user_id is not provided, use the admin user for this client
    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Add client scope to filter_tags (create if not provided)
    if request.filter_tags is not None:
        # Create a copy to avoid modifying the original request
        filter_tags = dict(request.filter_tags)
    else:
        # Create new filter_tags if not provided
        filter_tags = {}

    # Add or update the "scope" key with the client's scope
    filter_tags["scope"] = client.scope

    # Get all agents for this client (automatically filtered by client via apply_access_predicate)
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
    temporal_expr: Optional[str] = None

    if has_content:
        # Prefer local model for topic extraction when explicitly requested
        if request.local_model_for_retrieval:
            topics = extract_topics_with_local_model(
                messages=request.messages,
                model_name=request.local_model_for_retrieval,
            )
            # Note: Local model extraction doesn't support temporal extraction yet
            if topics is None:
                logger.warning(
                    "Local topic extraction failed for model %s, falling back to default LLM",
                    request.local_model_for_retrieval,
                )

        if topics is None:
            # NEW: Extract both topics and temporal expression
            topics, temporal_expr = extract_topics_and_temporal_info(request.messages, llm_config)

        logger.debug("Extracted topics: %s, temporal: %s", topics, temporal_expr)
        key_words = topics if topics else ""
    else:
        # No content - skip LLM call and retrieve recent items
        logger.debug("No content in messages - retrieving recent items")
        key_words = ""

    # NEW: Parse temporal expression to date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Priority: explicit request parameters > LLM-extracted temporal expression
    if request.start_date or request.end_date:
        # Use explicit date range from request
        try:
            if request.start_date:
                start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
            if request.end_date:
                end_date = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))
            logger.debug("Using explicit date range: %s to %s", start_date, end_date)
        except ValueError as e:
            logger.warning("Invalid date format in request: %s", e)
    elif temporal_expr:
        # Parse LLM-extracted temporal expression
        from mirix.temporal.temporal_parser import parse_temporal_expression

        # Get user's timezone for accurate "today" interpretation
        try:
            user = server.user_manager.get_user_by_id(user_id)
            import pytz

            user_tz = pytz.timezone(user.timezone)
            reference_time = datetime.now(user_tz)
        except Exception:
            # Fallback to UTC if user timezone not available
            reference_time = datetime.now()

        temporal_range = parse_temporal_expression(temporal_expr, reference_time)
        if temporal_range:
            start_date = temporal_range.start
            end_date = temporal_range.end
            # Strip timezone info to match database (which stores naive datetimes)
            if start_date and start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date and end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)
            logger.info(
                "Parsed temporal expression '%s' to range: %s to %s (timezone-naive for DB comparison)",
                temporal_expr,
                start_date,
                end_date,
            )

    # Retrieve memories with temporal filtering
    memories = retrieve_memories_by_keywords(
        server=server,
        client=client,
        user_id=user_id,
        agent_state=all_agents[0],
        key_words=key_words,
        limit=request.limit,
        filter_tags=filter_tags,
        use_cache=request.use_cache,
        start_date=start_date,  # NEW: Temporal filtering
        end_date=end_date,  # NEW: Temporal filtering
    )

    return {
        "success": True,
        "topics": topics,
        "temporal_expression": temporal_expr,  # NEW: Return extracted temporal info
        "date_range": (
            {  # NEW: Return applied date range
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            }
            if (start_date or end_date)
            else None
        ),
        "memories": memories,
    }


@router.get("/memory/retrieve/topic")
@with_langfuse_tracing
async def retrieve_memory_with_topic(
    user_id: Optional[str] = None,
    topic: str = "",
    limit: int = 10,
    filter_tags: Optional[str] = None,
    use_cache: bool = True,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on a topic using BM25 search.

    Args:
        user_id: The user ID to retrieve memories for (uses admin user if not provided)
        topic: The topic/keywords to search for
        limit: Maximum number of items to retrieve per memory type (default: 10)
        filter_tags: Optional JSON string of tags to filter memories (default: None)
        use_cache: Whether to use cached results (default: True)
    """
    server = get_server()
    client_id, org_id = get_client_and_org(x_client_id, x_org_id)
    client = server.client_manager.get_client_by_id(client_id)

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Parse filter_tags from JSON string to dict
    parsed_filter_tags = None
    if filter_tags:
        try:
            parsed_filter_tags = json.loads(filter_tags)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Invalid filter_tags JSON: {filter_tags}",
                "topic": topic,
                "memories": {},
            }

    # Add client scope to filter_tags (create if not provided)
    if parsed_filter_tags is None:
        # Create new filter_tags if not provided
        parsed_filter_tags = {}

    # Add or update the "scope" key with the client's scope
    parsed_filter_tags["scope"] = client.scope

    # Get all agents for this client (automatically filtered by client via apply_access_predicate)
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
        filter_tags=parsed_filter_tags,
        use_cache=use_cache,
    )

    return {
        "success": True,
        "topic": topic,
        "memories": memories,
    }


def _precompute_embedding_for_search(
    search_method: str, query: str, agent_state: AgentState
) -> tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Pre-compute embedding once for memory search to avoid redundant API calls.

    Args:
        search_method: The search method ('embedding', 'bm25', or 'string_match')
        query: The search query text
        agent_state: Agent state containing embedding configuration

    Returns:
        Tuple of (embedded_text, embedded_text_padded):
        - embedded_text: The raw embedding vector (for most memory types)
        - embedded_text_padded: Padded to MAX_EMBEDDING_DIM (for episodic memory)
        Both are None if search_method is not 'embedding' or query is empty.
    """
    if search_method != "embedding" or not query:
        return None, None

    import numpy as np

    from mirix.constants import MAX_EMBEDDING_DIM
    from mirix.embeddings import embedding_model

    # Compute embedding once
    embedded_text = embedding_model(agent_state.embedding_config).get_text_embedding(query)

    # Pad for episodic memory which requires MAX_EMBEDDING_DIM
    embedded_text_padded = np.pad(
        np.array(embedded_text),
        (0, MAX_EMBEDDING_DIM - len(embedded_text)),
        mode="constant",
    ).tolist()

    return embedded_text, embedded_text_padded


@router.get("/memory/search")
@with_langfuse_tracing
async def search_memory(
    user_id: Optional[str] = None,
    query: str = "",
    memory_type: str = "all",
    search_field: str = "null",
    search_method: str = "bm25",
    limit: int = 10,
    authorization: Optional[str] = Header(None),
    filter_tags: Optional[str] = Query(None),
    similarity_threshold: Optional[float] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Search for memories using various search methods with optional temporal filtering.
    Similar to the search_in_memory tool function.

    Args:
        user_id: The user ID to retrieve memories for (uses admin user if not provided)
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
        filter_tags: Optional JSON string of filter tags (scope added automatically)
        similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0).
                             Only results with cosine distance < threshold are returned.
                             Only applies when search_method="embedding"
        start_date: Optional start date/time for episodic memory filtering (ISO 8601 format)
        end_date: Optional end date/time for episodic memory filtering (ISO 8601 format)
    """
    server = get_server()

    client = None

    # Support both dashboard JWTs and programmatic API key access
    if authorization:
        try:
            client, _ = get_client_from_jwt_or_api_key(authorization)
        except HTTPException:
            # If JWT/API key auth fails, fall through to use x_client_id
            pass

    # Fallback to use the client_id and org_id passed in the method parameters.
    if not client and x_client_id:
        client_id = x_client_id
        client = server.client_manager.get_client_by_id(client_id)
    else:
        if not client:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Provide either Authorization (Bearer JWT) or X-API-Key header, or x-client-id and x-org-id headers.",
            )

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get all agents for this client (automatically filtered by client via apply_access_predicate)
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

    # Parse filter_tags from JSON string to dict
    parsed_filter_tags = None
    if filter_tags:
        try:
            parsed_filter_tags = json.loads(filter_tags)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Invalid filter_tags JSON: {filter_tags}",
                "query": query,
                "results": [],
                "count": 0,
            }

    # Add client scope to filter_tags (create if not provided)
    if parsed_filter_tags is None:
        parsed_filter_tags = {}

    # Add or update the "scope" key with the client's scope
    parsed_filter_tags["scope"] = client.scope

    # Parse temporal filtering parameters
    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            # Strip timezone for DB comparison (DB stores naive datetimes)
            if parsed_start_date.tzinfo:
                parsed_start_date = parsed_start_date.replace(tzinfo=None)
        except ValueError as e:
            logger.warning("Invalid start_date format: %s", e)

    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            # Strip timezone for DB comparison
            if parsed_end_date.tzinfo:
                parsed_end_date = parsed_end_date.replace(tzinfo=None)
        except ValueError as e:
            logger.warning("Invalid end_date format: %s", e)

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

    # Pre-compute embedding once if using embedding search (to avoid redundant embeddings)
    embedded_text, embedded_text_padded = _precompute_embedding_for_search(search_method, query, agent_state)

    # Collect results from requested memory types
    all_results = []

    # If searching all memory types, run searches concurrently for better performance
    if memory_type == "all":
        import asyncio

        # Define async wrappers for each manager call
        async def search_episodic():
            try:
                memories = await asyncio.to_thread(
                    server.episodic_memory_manager.list_episodic_memory,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text_padded if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "timestamp": (x.occurred_at.isoformat() if x.occurred_at else None),
                        "event_type": x.event_type,
                        "actor": x.actor,
                        "summary": x.summary,
                        "details": x.details,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching episodic memories: %s", e)
                return []

        async def search_resource():
            try:
                memories = await asyncio.to_thread(
                    server.resource_memory_manager.list_resources,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=(
                        search_field
                        if search_field != "null"
                        else ("summary" if search_method == "embedding" else "content")
                    ),
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "resource",
                        "id": x.id,
                        "resource_type": x.resource_type,
                        "title": x.title,
                        "summary": x.summary,
                        "content": x.content[:200] if x.content else None,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching resource memories: %s", e)
                return []

        async def search_procedural():
            try:
                memories = await asyncio.to_thread(
                    server.procedural_memory_manager.list_procedures,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "procedural",
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "summary": x.summary,
                        "steps": x.steps,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching procedural memories: %s", e)
                return []

        async def search_knowledge():
            try:
                memories = await asyncio.to_thread(
                    server.knowledge_vault_manager.list_knowledge,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "knowledge_vault",
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "source": x.source,
                        "sensitivity": x.sensitivity,
                        "secret_value": x.secret_value,
                        "caption": x.caption,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching knowledge vault: %s", e)
                return []

        async def search_semantic():
            try:
                memories = await asyncio.to_thread(
                    server.semantic_memory_manager.list_semantic_items,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "semantic",
                        "id": x.id,
                        "summary": x.summary,
                        "details": x.details,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching semantic memories: %s", e)
                return []

        # Run all searches concurrently
        results = await asyncio.gather(
            search_episodic(),
            search_resource(),
            search_procedural(),
            search_knowledge(),
            search_semantic(),
        )

        # Flatten results
        for result_list in results:
            all_results.extend(result_list)

    # Single memory type searches (run serially as before)
    elif memory_type == "episodic":
        try:
            episodic_memories = server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(embedded_text_padded if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                start_date=parsed_start_date,
                end_date=parsed_end_date,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "timestamp": (x.occurred_at.isoformat() if x.occurred_at else None),
                        "event_type": x.event_type,
                        "actor": x.actor,
                        "summary": x.summary,
                        "details": x.details,
                    }
                    for x in episodic_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching episodic memories: %s", e)

    # Search resource memories
    elif memory_type == "resource":
        try:
            resource_memories = server.resource_memory_manager.list_resources(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=(
                    search_field
                    if search_field != "null"
                    else ("summary" if search_method == "embedding" else "content")
                ),
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "resource",
                        "id": x.id,
                        "resource_type": x.resource_type,
                        "title": x.title,
                        "summary": x.summary,
                        "content": (x.content[:200] if x.content else None),  # Truncate content for response
                    }
                    for x in resource_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching resource memories: %s", e)

    # Search procedural memories
    elif memory_type == "procedural":
        try:
            procedural_memories = server.procedural_memory_manager.list_procedures(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "procedural",
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "summary": x.summary,
                        "steps": x.steps,
                    }
                    for x in procedural_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching procedural memories: %s", e)

    # Search knowledge vault
    elif memory_type == "knowledge_vault":
        try:
            knowledge_vault_memories = server.knowledge_vault_manager.list_knowledge(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "caption",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
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
                ]
            )
        except Exception as e:
            logger.error("Error searching knowledge vault: %s", e)

    # Search semantic memories
    elif memory_type == "semantic":
        try:
            semantic_memories = server.semantic_memory_manager.list_semantic_items(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "semantic",
                        "id": x.id,
                        "name": x.name,
                        "summary": x.summary,
                        "details": x.details,
                        "source": x.source,
                    }
                    for x in semantic_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching semantic memories: %s", e)

    return {
        "success": True,
        "query": query,
        "memory_type": memory_type,
        "search_field": search_field,
        "search_method": search_method,
        "date_range": (
            {
                "start": parsed_start_date.isoformat() if parsed_start_date else None,
                "end": parsed_end_date.isoformat() if parsed_end_date else None,
            }
            if (parsed_start_date or parsed_end_date)
            else None
        ),
        "results": all_results,
        "count": len(all_results),
    }


@router.get("/memory/search_all_users")
@with_langfuse_tracing
async def search_memory_all_users(
    query: str,
    memory_type: str = "all",
    search_field: str = "null",
    search_method: str = "bm25",
    limit: int = 10,
    client_id: Optional[str] = Query(None),
    org_id: Optional[str] = Query(None),
    filter_tags: Optional[str] = Query(None),
    similarity_threshold: Optional[float] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Search for memories across ALL users in the organization with optional temporal filtering.
    Automatically filters by client scope using filter_tags.

    Organization resolution priority:
    1. If client_id provided: use that client's organization_id
    2. Else: use org_id from query param or x_org_id header

    Args:
        query: The search query string
        memory_type: Type of memory to search. Options: "episodic", "resource",
                    "procedural", "knowledge_vault", "semantic", "all" (default: "all")
        search_field: Field to search in. Options vary by memory type
        search_method: Search method. Options: "bm25" (default), "embedding"
        limit: Maximum number of results (total across all users)
        client_id: Optional client ID (uses its org_id and scope)
        org_id: Optional organization ID (used if client_id not provided)
        filter_tags: Optional JSON string of additional filter tags
        similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0)
        start_date: Optional start date/time for episodic memory filtering (ISO 8601 format)
        end_date: Optional end date/time for episodic memory filtering (ISO 8601 format)
    """
    import json

    server = get_server()

    # Determine which client to use and which org to search
    if client_id:
        # Use the provided client_id - fetch its org_id
        effective_client_id = client_id
        client = server.client_manager.get_client_by_id(effective_client_id)
        effective_org_id = client.organization_id  # Use CLIENT's org_id
        logger.info(
            "Using provided client_id=%s with its organization_id=%s",
            effective_client_id,
            effective_org_id,
        )
    else:
        # Fall back to headers
        effective_client_id, header_org_id = get_client_and_org(x_client_id, x_org_id)
        client = server.client_manager.get_client_by_id(effective_client_id)
        # Use org_id from query param if provided, otherwise use header org_id
        effective_org_id = org_id or header_org_id
        logger.info(
            "Using client_id=%s from header with org_id=%s",
            effective_client_id,
            effective_org_id,
        )

    # Parse filter_tags if provided, otherwise create new dict
    if filter_tags:
        try:
            filter_tags_dict = json.loads(filter_tags)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid filter_tags JSON format")
    else:
        filter_tags_dict = {}

    # Add client scope to filter_tags (same pattern as retrieve_with_conversation)
    # This filters memories where memory.filter_tags["scope"] == client.scope
    filter_tags_dict["scope"] = client.scope

    logger.info(
        "Cross-user search: client=%s, org=%s, client_scope=%s, filter_tags=%s, similarity_threshold=%s",
        effective_client_id,
        effective_org_id,
        client.scope,
        filter_tags_dict,
        similarity_threshold,
    )

    # Parse temporal filtering parameters
    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            # Strip timezone for DB comparison (DB stores naive datetimes)
            if parsed_start_date.tzinfo:
                parsed_start_date = parsed_start_date.replace(tzinfo=None)
        except ValueError as e:
            logger.warning("Invalid start_date format: %s", e)

    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            # Strip timezone for DB comparison
            if parsed_end_date.tzinfo:
                parsed_end_date = parsed_end_date.replace(tzinfo=None)
        except ValueError as e:
            logger.warning("Invalid end_date format: %s", e)

    # Get agents for this client
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

    # Pre-compute embedding once if using embedding search (to avoid redundant embeddings)
    embedded_text, embedded_text_padded = _precompute_embedding_for_search(search_method, query, agent_state)

    # Collect results using organization_id filter
    all_results = []

    # If searching all memory types, run searches concurrently for better performance
    if memory_type == "all":
        import asyncio

        # Define async wrappers for each manager call
        async def search_episodic():
            try:
                memories = await asyncio.to_thread(
                    server.episodic_memory_manager.list_episodic_memory_by_org,
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(embedded_text_padded if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "timestamp": (x.occurred_at.isoformat() if x.occurred_at else None),
                        "event_type": x.event_type,
                        "actor": x.actor,
                        "summary": x.summary,
                        "details": x.details,
                        "user_id": str(x.user_id),
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching episodic memories across org: %s", e)
                return []

        async def search_resource():
            try:
                memories = await asyncio.to_thread(
                    server.resource_memory_manager.list_resources_by_org,
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=(
                        search_field
                        if search_field != "null"
                        else ("summary" if search_method == "embedding" else "content")
                    ),
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "resource",
                        "id": x.id,
                        "resource_type": x.resource_type,
                        "title": x.title,
                        "summary": x.summary,
                        "content": x.content[:200] if x.content else None,
                        "user_id": str(x.user_id),
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching resource memories across org: %s", e)
                return []

        async def search_procedural():
            try:
                memories = await asyncio.to_thread(
                    server.procedural_memory_manager.list_procedures_by_org,
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "procedural",
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "summary": x.summary,
                        "steps": x.steps,
                        "user_id": str(x.user_id),
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching procedural memories across org: %s", e)
                return []

        async def search_knowledge():
            try:
                memories = await asyncio.to_thread(
                    server.knowledge_vault_manager.list_knowledge_by_org,
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "knowledge_vault",
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "source": x.source,
                        "sensitivity": x.sensitivity,
                        "secret_value": x.secret_value,
                        "caption": x.caption,
                        "user_id": str(x.user_id),
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching knowledge vault across org: %s", e)
                return []

        async def search_semantic():
            try:
                memories = await asyncio.to_thread(
                    server.semantic_memory_manager.list_semantic_items_by_org,
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "semantic",
                        "id": x.id,
                        "summary": x.summary,
                        "details": x.details,
                        "user_id": str(x.user_id),
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching semantic memories across org: %s", e)
                return []

        # Run all searches concurrently
        results = await asyncio.gather(
            search_episodic(),
            search_resource(),
            search_procedural(),
            search_knowledge(),
            search_semantic(),
        )

        # Flatten results
        for result_list in results:
            all_results.extend(result_list)

    # Single memory type searches (run serially as before)
    elif memory_type == "episodic":
        try:
            episodic_memories = server.episodic_memory_manager.list_episodic_memory_by_org(
                agent_state=agent_state,
                organization_id=effective_org_id,
                query=query,
                embedded_text=(embedded_text_padded if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str="UTC",
                filter_tags=filter_tags_dict,
                start_date=parsed_start_date,
                end_date=parsed_end_date,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "episodic",
                        "user_id": x.user_id,
                        "id": x.id,
                        "timestamp": (x.occurred_at.isoformat() if x.occurred_at else None),
                        "event_type": x.event_type,
                        "actor": x.actor,
                        "summary": x.summary,
                        "details": x.details,
                    }
                    for x in episodic_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching episodic memories across organization: %s", e)

    # Search resource memories across organization
    elif memory_type == "resource":
        try:
            resource_memories = server.resource_memory_manager.list_resources_by_org(
                agent_state=agent_state,
                organization_id=effective_org_id,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=(
                    search_field
                    if search_field != "null"
                    else ("summary" if search_method == "embedding" else "content")
                ),
                search_method=search_method,
                limit=limit,
                timezone_str="UTC",
                filter_tags=filter_tags_dict,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "resource",
                        "user_id": x.user_id,
                        "id": x.id,
                        "resource_type": x.resource_type,
                        "title": x.title,
                        "summary": x.summary,
                        "content": x.content[:200] if x.content else None,
                    }
                    for x in resource_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching resource memories across organization: %s", e)

    # Search procedural memories across organization
    elif memory_type == "procedural":
        try:
            procedural_memories = server.procedural_memory_manager.list_procedures_by_org(
                agent_state=agent_state,
                organization_id=effective_org_id,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str="UTC",
                filter_tags=filter_tags_dict,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "procedural",
                        "user_id": x.user_id,
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "summary": x.summary,
                        "steps": x.steps,
                    }
                    for x in procedural_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching procedural memories across organization: %s", e)

    # Search knowledge vault across organization
    elif memory_type == "knowledge_vault":
        try:
            knowledge_vault_memories = server.knowledge_vault_manager.list_knowledge_by_org(
                agent_state=agent_state,
                organization_id=effective_org_id,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "caption",
                search_method=search_method,
                limit=limit,
                timezone_str="UTC",
                filter_tags=filter_tags_dict,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "knowledge_vault",
                        "user_id": x.user_id,
                        "id": x.id,
                        "entry_type": x.entry_type,
                        "source": x.source,
                        "sensitivity": x.sensitivity,
                        "secret_value": x.secret_value,
                        "caption": x.caption,
                    }
                    for x in knowledge_vault_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching knowledge vault across organization: %s", e)

    # Search semantic memories across organization
    elif memory_type == "semantic":
        try:
            semantic_memories = server.semantic_memory_manager.list_semantic_items_by_org(
                agent_state=agent_state,
                organization_id=effective_org_id,
                query=query,
                embedded_text=(embedded_text if search_method == "embedding" and query else None),
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str="UTC",
                filter_tags=filter_tags_dict,
                similarity_threshold=similarity_threshold,
            )
            all_results.extend(
                [
                    {
                        "memory_type": "semantic",
                        "user_id": x.user_id,
                        "id": x.id,
                        "name": x.name,
                        "summary": x.summary,
                        "details": x.details,
                        "source": x.source,
                    }
                    for x in semantic_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching semantic memories across organization: %s", e)

    return {
        "success": True,
        "query": query,
        "memory_type": memory_type,
        "search_field": search_field,
        "search_method": search_method,
        "date_range": (
            {
                "start": parsed_start_date.isoformat() if parsed_start_date else None,
                "end": parsed_end_date.isoformat() if parsed_end_date else None,
            }
            if (parsed_start_date or parsed_end_date)
            else None
        ),
        "results": all_results,
        "count": len(all_results),
        "client_id": effective_client_id,
        "organization_id": effective_org_id,
        "client_scope": client.scope,
        "filter_tags": filter_tags_dict,
    }


@router.get("/memory/components")
async def list_memory_components(
    user_id: Optional[str] = None,
    memory_type: str = "all",
    limit: int = 50,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Return memory records grouped by component for the given user.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Args:
        user_id: End-user whose memories should be listed (defaults to the client's admin user)
        memory_type: One of ["episodic","semantic","procedural","resource","knowledge_vault","core","all"]
        limit: Maximum number of items to return per memory type
    """
    valid_memory_types = {
        "episodic",
        "semantic",
        "procedural",
        "resource",
        "knowledge_vault",
        "core",
        "all",
    }
    memory_type = (memory_type or "all").lower()
    if memory_type not in valid_memory_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported memory_type '{memory_type}'. Must be one of {sorted(valid_memory_types)}",
        )

    # Authenticate (JWT or API key)
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    # Default to the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    timezone_str = getattr(user, "timezone", None) or "UTC"
    limit = max(1, min(limit, 200))  # guardrails

    # Need an agent state for memory manager configuration
    agents = server.agent_manager.list_agents(actor=client, limit=1)
    if not agents:
        raise HTTPException(status_code=404, detail="No agents found for this client")
    agent_state = agents[0]

    fetch_all = memory_type == "all"
    memories: Dict[str, Any] = {}

    if fetch_all or memory_type == "episodic":
        episodic_items = server.episodic_memory_manager.list_episodic_memory(
            agent_state=agent_state,
            user=user,
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["episodic"] = {
            "total_count": server.episodic_memory_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": item.id,
                    "occurred_at": (item.occurred_at.isoformat() if item.occurred_at else None),
                    "event_type": item.event_type,
                    "actor": item.actor,
                    "summary": item.summary,
                    "details": item.details,
                    "created_at": (item.created_at.isoformat() if getattr(item, "created_at", None) else None),
                    "updated_at": (item.updated_at.isoformat() if getattr(item, "updated_at", None) else None),
                }
                for item in episodic_items
            ],
        }

    if fetch_all or memory_type == "semantic":
        semantic_items = server.semantic_memory_manager.list_semantic_items(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="summary",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["semantic"] = {
            "total_count": server.semantic_memory_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "details": item.details,
                    "source": item.source,
                    "created_at": (item.created_at.isoformat() if getattr(item, "created_at", None) else None),
                    "updated_at": (item.updated_at.isoformat() if getattr(item, "updated_at", None) else None),
                }
                for item in semantic_items
            ],
        }

    if fetch_all or memory_type == "procedural":
        procedural_items = server.procedural_memory_manager.list_procedures(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="summary",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["procedural"] = {
            "total_count": server.procedural_memory_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": item.id,
                    "entry_type": item.entry_type,
                    "summary": item.summary,
                    "steps": item.steps,
                    "created_at": (item.created_at.isoformat() if getattr(item, "created_at", None) else None),
                    "updated_at": (item.updated_at.isoformat() if getattr(item, "updated_at", None) else None),
                }
                for item in procedural_items
            ],
        }

    if fetch_all or memory_type == "resource":
        resource_items = server.resource_memory_manager.list_resources(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="summary",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["resource"] = {
            "total_count": server.resource_memory_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": item.id,
                    "resource_type": item.resource_type,
                    "title": item.title,
                    "summary": item.summary,
                    "content": item.content,
                    "created_at": (item.created_at.isoformat() if getattr(item, "created_at", None) else None),
                    "updated_at": (item.updated_at.isoformat() if getattr(item, "updated_at", None) else None),
                }
                for item in resource_items
            ],
        }

    if fetch_all or memory_type == "knowledge_vault":
        knowledge_items = server.knowledge_vault_manager.list_knowledge(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="caption",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["knowledge_vault"] = {
            "total_count": server.knowledge_vault_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": item.id,
                    "entry_type": item.entry_type,
                    "source": item.source,
                    "sensitivity": item.sensitivity,
                    "secret_value": item.secret_value,
                    "caption": item.caption,
                    "created_at": (item.created_at.isoformat() if getattr(item, "created_at", None) else None),
                    "updated_at": (item.updated_at.isoformat() if getattr(item, "updated_at", None) else None),
                }
                for item in knowledge_items
            ],
        }

    if fetch_all or memory_type == "core":
        blocks = server.block_manager.get_blocks(user=user)
        memories["core"] = {
            "total_count": len(blocks),
            "items": [
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                }
                for block in blocks[:limit]
            ],
        }

    return {
        "success": True,
        "user_id": user_id,
        "memory_type": memory_type,
        "limit": limit,
        "memories": memories,
    }


@router.get("/memory/fields")
async def list_memory_fields(
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Return the searchable fields for each memory component.

    Useful for UI clients to populate search field dropdowns in sync with
    what the backend supports.
    """
    # Authenticate to keep parity with other memory endpoints
    get_client_from_jwt_or_api_key(authorization, http_request)

    fields_by_type = {
        "episodic": ["summary", "details"],
        "semantic": ["name", "summary", "details"],
        "procedural": ["summary", "steps"],
        "resource": ["summary", "content"],
        "knowledge_vault": ["caption", "secret_value"],
        "core": ["label", "value"],
    }

    return {
        "success": True,
        "fields": fields_by_type,
    }


# ============================================================================
# Memory CRUD Endpoints (Update/Delete individual memories)
# Accepts both JWT (dashboard) and Client API Key (programmatic access)
# ============================================================================


def get_client_from_jwt_or_api_key(
    authorization: Optional[str] = None,
    request: Optional[Request] = None,
) -> tuple:
    """
    Authenticate using either JWT token (dashboard) or Client API Key (programmatic).

    Args:
        authorization: Bearer JWT token from Authorization header
        request: FastAPI request to inspect injected headers (from X-API-Key middleware)

    Returns:
        tuple: (client, auth_type) where auth_type is "jwt" or "api_key"
        Both authentication methods return a valid client object.

    Raises:
        HTTPException: If neither auth method is provided or valid
    """
    server = get_server()

    # Try JWT first (dashboard)
    if authorization:
        try:
            admin_payload = get_current_admin(authorization)
            # Get client from JWT payload (sub contains client_id)
            client_id = admin_payload["sub"]
            client = server.client_manager.get_client_by_id(client_id)
            if not client:
                raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
            return client, "jwt"
        except HTTPException:
            pass  # Try API key next

    # Try injected client headers (programmatic access via middleware)
    if request:
        client_id = request.headers.get("x-client-id")
        org_id = request.headers.get("x-org-id")
        if client_id:
            client_id, org_id = get_client_and_org(client_id, org_id)
            client = server.client_manager.get_client_by_id(client_id)
            if not client:
                raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
            return client, "api_key"

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide either Authorization (Bearer JWT) or X-API-Key header.",
    )


class UpdateEpisodicMemoryRequest(BaseModel):
    """Request model for updating an episodic memory."""

    summary: Optional[str] = None
    details: Optional[str] = None


@router.patch("/memory/episodic/{memory_id}")
async def update_episodic_memory(
    memory_id: str,
    request: UpdateEpisodicMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update an episodic memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Updates the summary and/or details fields of the memory.
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        updated_memory = server.episodic_memory_manager.update_event(
            event_id=memory_id,
            new_summary=request.summary,
            new_details=request.details,
            user=user,
            actor=client,
        )
        return {
            "success": True,
            "message": f"Episodic memory {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "summary": updated_memory.summary,
                "details": updated_memory.details,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/episodic/{memory_id}")
async def delete_episodic_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete an episodic memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    try:
        server.episodic_memory_manager.delete_event_by_id(memory_id, actor=client)
        return {"success": True, "message": f"Episodic memory {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


class UpdateSemanticMemoryRequest(BaseModel):
    """Request model for updating a semantic memory."""

    name: Optional[str] = None
    summary: Optional[str] = None
    details: Optional[str] = None


@router.patch("/memory/semantic/{memory_id}")
async def update_semantic_memory(
    memory_id: str,
    request: UpdateSemanticMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update a semantic memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        semantic_update_data = {"id": memory_id}
        if request.name is not None:
            semantic_update_data["name"] = request.name
        if request.summary is not None:
            semantic_update_data["summary"] = request.summary
        if request.details is not None:
            semantic_update_data["details"] = request.details

        updated_memory = server.semantic_memory_manager.update_item(
            item_update=SemanticMemoryItemUpdate.model_validate(semantic_update_data),
            user=user,
            actor=client,
        )
        return {
            "success": True,
            "message": f"Semantic memory {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "name": updated_memory.name,
                "summary": updated_memory.summary,
                "details": updated_memory.details,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/semantic/{memory_id}")
async def delete_semantic_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a semantic memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    try:
        server.semantic_memory_manager.delete_semantic_item_by_id(memory_id, actor=client)
        return {"success": True, "message": f"Semantic memory {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


class UpdateProceduralMemoryRequest(BaseModel):
    """Request model for updating a procedural memory."""

    summary: Optional[str] = None
    steps: Optional[List[str]] = None


@router.patch("/memory/procedural/{memory_id}")
async def update_procedural_memory(
    memory_id: str,
    request: UpdateProceduralMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update a procedural memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        procedural_update_data = {"id": memory_id}
        if request.summary is not None:
            procedural_update_data["summary"] = request.summary
        if request.steps is not None:
            procedural_update_data["steps"] = request.steps

        updated_memory = server.procedural_memory_manager.update_item(
            item_update=ProceduralMemoryItemUpdate.model_validate(procedural_update_data),
            user=user,
            actor=client,
        )
        return {
            "success": True,
            "message": f"Procedural memory {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "summary": updated_memory.summary,
                "steps": updated_memory.steps,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/procedural/{memory_id}")
async def delete_procedural_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a procedural memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    try:
        server.procedural_memory_manager.delete_procedure_by_id(memory_id, actor=client)
        return {"success": True, "message": f"Procedural memory {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Raw Memory Endpoints
# ============================================================================


@router.get("/memory/raw/{memory_id}")
async def get_raw_memory_handler(
    memory_id: str,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Fetch a single raw memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    response = await get_raw_memory(memory_id, user_id, client)
    return response


async def get_raw_memory(
    memory_id: str,
    user_id: Optional[str] = None,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Fetch a single raw memory by ID.

    Raw memories are unprocessed task context stored for task sharing use cases,
    with a 14-day TTL.
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        from mirix.orm.errors import NoResultFound

        memory = server.raw_memory_manager.get_raw_memory_by_id(memory_id, user)
        return {
            "success": True,
            "memory": memory.model_dump(mode="json"),
        }
    except NoResultFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching raw memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.patch("/memory/raw/{memory_id}")
async def update_raw_memory_handler(
    memory_id: str,
    request: RawMemoryItemUpdate,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update an existing raw memory.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    response = await update_raw_memory(memory_id, request, user_id, client)
    return response


async def update_raw_memory(
    memory_id: str,
    request: RawMemoryItemUpdate,
    user_id: Optional[str] = None,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Update an existing raw memory.

    Updates the context and/or filter_tags fields of the memory.
    Supports append/replace modes for both context and tags.
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get agent_state for embedding generation
    agents = server.agent_manager.list_agents(actor=client, limit=1)
    agent_state = agents[0] if agents else None

    if not agent_state:
        logger.warning("No agents found for client %s, embeddings will not be generated", client.id)

    try:
        updated_memory = server.raw_memory_manager.update_raw_memory(
            memory_id=memory_id,
            new_context=request.context,
            new_filter_tags=request.filter_tags,
            actor=client,
            agent_state=agent_state,
            context_update_mode=request.context_update_type,
            tags_merge_mode=request.tags_update_type,
        )
        return {
            "success": True,
            "message": f"Raw memory {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "context": updated_memory.context,
                "filter_tags": updated_memory.filter_tags,
                "updated_at": updated_memory.updated_at.isoformat(),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating raw memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.delete("/memory/raw/{memory_id}")
async def delete_raw_memory_handler(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a raw memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    response = await delete_raw_memory(memory_id, client)
    return response


async def delete_raw_memory(
    memory_id: str,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Delete a raw memory by ID.

    This performs a hard delete and removes the memory from both PostgreSQL
    and Redis cache.
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    try:
        deleted = server.raw_memory_manager.delete_raw_memory(memory_id, client)
        if deleted:
            return {
                "success": True,
                "message": f"Raw memory {memory_id} deleted",
            }
        else:
            raise HTTPException(status_code=404, detail=f"Raw memory {memory_id} not found")
    except Exception as e:
        logger.error(f"Error deleting raw memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/memory/search_raw")
async def search_raw_memory_handler(
    request: SearchRawMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Search raw memories with filtering, sorting, and pagination.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)
    response = await search_raw_memory(request, user_id, client)
    return response


async def search_raw_memory(
    request: SearchRawMemoryRequest,
    user_id: Optional[str] = None,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Search raw memories with filtering, sorting, cursor-based pagination, and time range filtering.
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    try:
        user = server.user_manager.get_user_by_id(user_id)
    except Exception as e:
        from mirix.orm.errors import NoResultFound

        if isinstance(e, NoResultFound):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        # Re-raise other exceptions
        raise

    # Process filter_tags: always set scope to client scope (overwrites any user-provided scope)
    filter_tags = dict[str, Any](request.filter_tags) if request.filter_tags is not None else {}
    filter_tags["scope"] = client.scope

    # Parse time_range from request, converting to UTC and stripping timezone for DB comparison
    time_range_dict = None
    if request.time_range:

        def to_utc_stripped(dt_val: Optional[datetime]) -> Optional[datetime]:
            """Convert datetime to UTC and strip timezone for DB comparison.

            The database stores naive datetimes assumed to be in UTC.
            """
            if not dt_val:
                return None
            if dt_val.tzinfo:
                # Convert to UTC first, then strip timezone
                from datetime import timezone as dt_timezone

                dt_val = dt_val.astimezone(dt_timezone.utc)
                return dt_val.replace(tzinfo=None)
            # Already naive - assume it's UTC
            return dt_val

        time_range_dict = {}
        if request.time_range.created_at_gte:
            time_range_dict["created_at_gte"] = to_utc_stripped(request.time_range.created_at_gte)
        if request.time_range.created_at_lte:
            time_range_dict["created_at_lte"] = to_utc_stripped(request.time_range.created_at_lte)
        if request.time_range.occurred_at_gte:
            time_range_dict["occurred_at_gte"] = to_utc_stripped(request.time_range.occurred_at_gte)
        if request.time_range.occurred_at_lte:
            time_range_dict["occurred_at_lte"] = to_utc_stripped(request.time_range.occurred_at_lte)
        if request.time_range.updated_at_gte:
            time_range_dict["updated_at_gte"] = to_utc_stripped(request.time_range.updated_at_gte)
        if request.time_range.updated_at_lte:
            time_range_dict["updated_at_lte"] = to_utc_stripped(request.time_range.updated_at_lte)

        if time_range_dict.get("created_at_gte") and time_range_dict.get("created_at_lte"):
            if time_range_dict["created_at_gte"] > time_range_dict["created_at_lte"]:
                raise HTTPException(status_code=400, detail="created_at_gte must be <= created_at_lte")
        if time_range_dict.get("occurred_at_gte") and time_range_dict.get("occurred_at_lte"):
            if time_range_dict["occurred_at_gte"] > time_range_dict["occurred_at_lte"]:
                raise HTTPException(status_code=400, detail="occurred_at_gte must be <= occurred_at_lte")
        if time_range_dict.get("updated_at_gte") and time_range_dict.get("updated_at_lte"):
            if time_range_dict["updated_at_gte"] > time_range_dict["updated_at_lte"]:
                raise HTTPException(status_code=400, detail="updated_at_gte must be <= updated_at_lte")

    # Validate sort string
    valid_sorts = {
        "updated_at",
        "-updated_at",
        "created_at",
        "-created_at",
        "occurred_at",
        "-occurred_at",
    }
    if request.sort not in valid_sorts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort: {request.sort}. Must be one of {valid_sorts}",
        )

    try:
        items, next_cursor = server.raw_memory_manager.search_raw_memories(
            user=user,
            filter_tags=filter_tags,
            sort=request.sort,
            cursor=request.cursor,
            time_range=time_range_dict,
            limit=request.limit,
        )

        return SearchRawMemoryResponse(
            items=items,
            cursor=next_cursor,
            count=len(items),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching raw memories: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/memory/raw/cleanup")
async def cleanup_raw_memories_handler(
    days_threshold: int = Query(14, ge=1, le=365, description="Delete memories older than this many days"),
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Manually trigger the raw memory cleanup job to delete stale memories.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    This endpoint allows manual execution of the cleanup job that
    deletes raw memories older than the specified threshold (based on updated_at).
    Normally this job runs nightly via cron/scheduler, but this endpoint allows
    on-demand execution for maintenance or testing purposes.

    Args:
        days_threshold: Number of days after which memories are considered stale.
                       Must be between 1 and 365 days (default: 14)

    Returns:
        Dict with deletion statistics including deleted_count, error_count, and cutoff_date

    Example:
        POST /memory/raw/cleanup?days_threshold=7

        Response:
        {
            "success": true,
            "deleted_count": 42,
            "error_count": 0,
            "cutoff_date": "2026-01-20T10:30:00+00:00",
            "days_threshold": 7,
            "message": "Deleted 42 stale raw memories (older than 7 days)"
        }
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    logger.info(
        "Manual cleanup job triggered by client %s (auth: %s) with threshold: %d days",
        client.id,
        auth_type,
        days_threshold,
    )

    response = await cleanup_raw_memories(days_threshold, client)
    return response


async def cleanup_raw_memories(
    days_threshold: int = Query(14, ge=1, le=365, description="Delete memories older than this many days"),
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Manually trigger the raw memory cleanup job to delete stale memories.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # Import the cleanup function
    from mirix.jobs.cleanup_raw_memories import delete_stale_raw_memories

    try:
        # Run the cleanup job
        result = delete_stale_raw_memories(days_threshold=days_threshold)

        # Add success message to result
        result["message"] = (
            f"Deleted {result['deleted_count']} stale raw memories " f"(older than {days_threshold} days)"
        )

        logger.info(
            "Cleanup job completed: deleted %d, errors %d",
            result["deleted_count"],
            result["error_count"],
        )

        return result
    except Exception as e:
        logger.error("Cleanup job failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Cleanup job failed: {str(e)}")


class UpdateResourceMemoryRequest(BaseModel):
    """Request model for updating a resource memory."""

    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None


@router.patch("/memory/resource/{memory_id}")
async def update_resource_memory(
    memory_id: str,
    request: UpdateResourceMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Update a resource memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    # Authenticate with either JWT or API key
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        resource_update_data = {"id": memory_id}
        if request.title is not None:
            resource_update_data["title"] = request.title
        if request.summary is not None:
            resource_update_data["summary"] = request.summary
        if request.content is not None:
            resource_update_data["content"] = request.content

        updated_memory = server.resource_memory_manager.update_item(
            item_update=ResourceMemoryItemUpdate.model_validate(resource_update_data),
            user=user,
            actor=client,
        )
        return {
            "success": True,
            "message": f"Resource memory {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "title": updated_memory.title,
                "summary": updated_memory.summary,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/resource/{memory_id}")
async def delete_resource_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a resource memory by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    try:
        server.resource_memory_manager.delete_resource_by_id(memory_id, actor=client)
        return {"success": True, "message": f"Resource memory {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/knowledge_vault/{memory_id}")
async def delete_knowledge_vault_memory(
    memory_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Delete a knowledge vault item by ID.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    """
    client, auth_type = get_client_from_jwt_or_api_key(authorization, http_request)

    server = get_server()

    try:
        server.knowledge_vault_manager.delete_knowledge_by_id(memory_id, actor=client)
        return {"success": True, "message": f"Knowledge vault item {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Dashboard Authentication Endpoints (Client-based)
# ============================================================================


class DashboardLoginRequest(BaseModel):
    """Request model for dashboard login."""

    email: str = Field(..., description="Client email")
    password: str = Field(..., description="Client password")


class DashboardRegisterRequest(BaseModel):
    """Request model for dashboard registration."""

    name: str = Field(..., max_length=100, description="Client name")
    email: str = Field(..., description="Email address for login")
    password: str = Field(..., description="Password for login")


class DashboardClientResponse(BaseModel):
    """Response model for dashboard client (excludes sensitive data)."""

    id: str
    name: str
    email: Optional[str]
    scope: str
    status: str
    admin_user_id: str  # Admin user for memory operations
    created_at: Optional[datetime]
    last_login: Optional[datetime]


class TokenResponse(BaseModel):
    """Response model for authentication token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    client: DashboardClientResponse


def get_current_admin(authorization: Optional[str] = Header(None)) -> dict:
    """
    Dependency to get the current authenticated client from JWT token.

    Args:
        authorization: Bearer token from Authorization header

    Returns:
        Decoded JWT payload with client info

    Raises:
        HTTPException: If token is invalid or missing
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    from mirix.services.admin_user_manager import ClientAuthManager

    payload = ClientAuthManager.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def require_scope(required_scopes: List[str]):
    """
    Dependency factory to require specific client scopes.

    Args:
        required_scopes: List of allowed scopes

    Returns:
        Dependency function that validates scope
    """

    def scope_checker(client_payload: dict = None):
        if client_payload is None:
            raise HTTPException(status_code=401, detail="Not authenticated")

        if client_payload.get("scope") not in required_scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scopes: {required_scopes}",
            )
        return client_payload

    return scope_checker


@router.post("/admin/auth/register", response_model=TokenResponse)
async def dashboard_register(request: DashboardRegisterRequest):
    """
    Register a new client with dashboard access.

    - First registration: Creates an admin client (no auth required)
    - Subsequent registrations: May require authentication (future feature)

    For the first client, no authentication is needed (bootstrap).
    """
    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()

    # Check if this is the first dashboard user (bootstrap mode)
    is_first = auth_manager.is_first_dashboard_user()

    if is_first:
        logger.info("Creating first dashboard client (bootstrap mode)")

    try:
        # Create client with dashboard credentials
        client = auth_manager.register_client_for_dashboard(
            name=request.name,
            email=request.email,
            password=request.password,
            scope="admin",  # First/dashboard users get admin scope
        )

        # Generate token
        access_token = ClientAuthManager.create_access_token(client)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours in seconds
            client=DashboardClientResponse(
                id=client.id,
                name=client.name,
                email=client.email,
                scope=client.scope,
                status=client.status,
                admin_user_id=ClientAuthManager.get_admin_user_id_for_client(client.id),
                created_at=client.created_at,
                last_login=client.last_login,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/admin/auth/login", response_model=TokenResponse)
async def dashboard_login(request: DashboardLoginRequest):
    """
    Authenticate client for dashboard and return JWT token.
    """
    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()

    client, access_token, auth_status = auth_manager.authenticate(request.email, request.password)

    if auth_status == "not_found":
        raise HTTPException(status_code=404, detail="Account does not exist. Please create an account.")
    if auth_status == "wrong_password":
        raise HTTPException(status_code=401, detail="Incorrect password")
    if auth_status != "ok" or not client or not access_token:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=24 * 3600,
        client=DashboardClientResponse(
            id=client.id,
            name=client.name,
            email=client.email,
            scope=client.scope,
            status=client.status,
            admin_user_id=ClientAuthManager.get_admin_user_id_for_client(client.id),
            created_at=client.created_at,
            last_login=client.last_login,
        ),
    )


@router.get("/admin/auth/me", response_model=DashboardClientResponse)
async def dashboard_get_current_client(authorization: Optional[str] = Header(None)):
    """
    Get current authenticated client.
    """
    client_payload = get_current_admin(authorization)

    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()
    client = auth_manager.get_client_by_id(client_payload["sub"])

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return DashboardClientResponse(
        id=client.id,
        name=client.name,
        email=client.email,
        scope=client.scope,
        status=client.status,
        admin_user_id=ClientAuthManager.get_admin_user_id_for_client(client.id),
        created_at=client.created_at,
        last_login=client.last_login,
    )


@router.get("/admin/dashboard-clients", response_model=List[DashboardClientResponse])
async def list_dashboard_clients(
    cursor: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
):
    """
    List all clients with dashboard access. Requires admin scope.
    """
    client_payload = get_current_admin(authorization)

    # Check scope - only admin can list dashboard clients
    if client_payload.get("scope") != "admin":
        raise HTTPException(status_code=403, detail="Only admin clients can list dashboard clients")

    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()
    clients = auth_manager.list_dashboard_clients(cursor=cursor, limit=limit)

    from mirix.services.admin_user_manager import ClientAuthManager

    return [
        DashboardClientResponse(
            id=c.id,
            name=c.name,
            email=c.email,
            scope=c.scope,
            status=c.status,
            admin_user_id=ClientAuthManager.get_admin_user_id_for_client(c.id),
            created_at=c.created_at,
            last_login=c.last_login,
        )
        for c in clients
    ]


class PasswordChangeRequest(BaseModel):
    """Request model for password change."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


@router.post("/admin/auth/change-password")
async def dashboard_change_password(
    request: PasswordChangeRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Change the current client's dashboard password.
    """
    client_payload = get_current_admin(authorization)

    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()

    success = auth_manager.change_password(
        client_id=client_payload["sub"],
        current_password=request.current_password,
        new_password=request.new_password,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Invalid current password")

    return {"message": "Password changed successfully"}


@router.get("/admin/auth/check-setup")
async def dashboard_check_setup():
    """
    Check if dashboard setup is complete (any dashboard clients exist).

    This endpoint is public and used by the dashboard to determine
    whether to show the registration or login page.
    """
    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()
    is_first = auth_manager.is_first_dashboard_user()

    return {
        "setup_required": is_first,
        "message": (
            "No dashboard users exist. Please create the first account." if is_first else "Dashboard setup complete."
        ),
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
