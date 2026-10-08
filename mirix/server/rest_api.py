"""
FastAPI REST API server for Mirix.
This provides HTTP endpoints that wrap the AsyncServer functionality,
allowing MirixClient instances to communicate with a cloud-hosted server.
"""

import asyncio
import copy
import dataclasses
import functools
import json
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.auto_dream import AutoDreamRequest
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
from mirix.schemas.message import Message, MessageCreate, _validate_session_id
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.raw_memory import (
    RawMemoryItem,
    RawMemoryItemCreateRequest,
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
from mirix.server.server import AsyncServer, ensure_tables_created
from mirix.settings import model_settings, settings

logger = get_logger(__name__)

# Import queue components
from mirix.queue import initialize_queue
from mirix.queue.manager import get_manager as get_queue_manager
from mirix.queue.queue_util import put_messages
from mirix.server.constants import MAX_MEMORY_LIMIT
# Initialize server (single instance shared across all requests)
# The singleton accessor lives in mirix.server.server so service-layer modules
# never need to import this REST module; re-exported here for existing callers.
from mirix.server.server import get_server  # noqa: E402


def _isoformat_or_none(value) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _procedural_memory_response(
    item,
    *,
    include_user_id: bool = False,
    retrieval_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize procedural memory as the public skill-shaped read model.

    ``retrieval_scores`` is ``{bm25, cosine, rrf}`` for a skill ranked by the
    skill retrieval pipeline, None for a skill read any other way.
    """
    response = {
        "memory_type": "procedural",
        "id": item.id,
        "entry_type": item.entry_type,
        "name": item.name,
        "description": item.description,
        "instructions": item.instructions,
        "triggers": getattr(item, "triggers", None) or [],
        "examples": getattr(item, "examples", None) or [],
        "version": getattr(item, "version", None),
        "created_at": _isoformat_or_none(getattr(item, "created_at", None)),
        "updated_at": _isoformat_or_none(getattr(item, "updated_at", None)),
        "retrieval_scores": dict(retrieval_scores) if retrieval_scores is not None else None,
    }
    if include_user_id:
        response["user_id"] = str(getattr(item, "user_id", ""))
    return response


async def _search_procedural_memories(
    manager,
    *,
    agent_state: AgentState,
    user,
    query: str,
    search_method: str,
    search_field: str,
    limit: int,
    timezone_str: Optional[str],
    filter_tags: Optional[Dict[str, Any]],
    scopes: Optional[List[str]],
    similarity_threshold: Optional[float],
    embedded_text: Optional[List[float]],
) -> List[Dict[str, Any]]:
    """The procedural part of ``/memory/search``.

    A ranked search over the retrieval text — every ``hybrid`` search, and a
    ``bm25`` or ``embedding`` one on the default field — is the skill ranking
    WITHOUT the gate: true BM25 + cosine + RRF, the top ``limit`` whatever
    their similarity, each with its ``retrieval_scores``. Field-specific
    keyword searches (e.g. ``instructions``), string/fuzzy matching and
    the empty query (the most recent skills) keep ``list_procedures``.
    ``similarity_threshold`` (a cosine distance) keeps the skills at or above
    it out of the vector lane.
    """
    from mirix.services.procedural_memory_manager import uses_skill_ranking

    field = search_field if search_field != "null" else "description"
    if query.strip() and uses_skill_ranking(search_method, field):
        result = await manager.retrieve_skills(
            query,
            user,
            scopes,
            limit,
            apply_gate=False,
            agent_state=agent_state,
            query_embedding=embedded_text,
            filter_tags=filter_tags,
            method=search_method,
            max_cosine_distance=similarity_threshold,
            timezone_str=timezone_str,
        )
        if result.diagnostics.embedding_status == "failed":
            logger.warning(
                "/memory/search: the procedural query vector failed (%s); "
                "ranked by BM25 alone",
                result.diagnostics.embedding_error,
            )
        return [
            _procedural_memory_response(r.skill, retrieval_scores=r.scores.to_dict())
            for r in result.skills
        ]
    items = await manager.list_procedures(
        agent_state=agent_state,
        user=user,
        query=query,
        embedded_text=(
            embedded_text if search_method == "embedding" and query else None
        ),
        search_field=field,
        search_method=search_method,
        limit=limit,
        timezone_str=timezone_str,
        filter_tags=filter_tags,
        scopes=scopes,
        similarity_threshold=similarity_threshold,
    )
    return [_procedural_memory_response(x) for x in items]


async def initialize():
    """
    Initialize the Mirix server and queue services.
    This function can be called by external applications to initialize the server.
    """
    logger.info("Starting Mirix REST API server")

    # Create database tables (async engine) before initializing server
    await ensure_tables_created()

    # Token counting: tiktoken downloads its o200k_base vocabulary
    # synchronously on first use — load it here, off the event loop, into
    # <MIRIX_DIR>/tiktoken. Offline, counting falls back to 3 chars/token.
    from mirix.services.context_budget import (
        TOKENIZER_PRELOAD_TIMEOUT_SECONDS,
        preload_tokenizer,
    )

    await preload_tokenizer(timeout=TOKENIZER_PRELOAD_TIMEOUT_SECONDS)

    # Initialize Redis (async ping, create_indexes, log stats)
    try:
        from mirix.database.redis_client import get_redis_client

        redis_client = get_redis_client()
        if redis_client:
            if await redis_client.ping():
                await redis_client.create_indexes()
                await redis_client.log_connection_stats()
                logger.info("Redis async client ready")
            else:
                logger.warning("Redis ping failed - continuing without Redis")
    except Exception as e:
        logger.warning("Redis async init failed: %s", e)

    # Initialize AsyncServer (singleton) and create default org/user/client
    server = get_server()
    await server.ensure_defaults()
    logger.info("AsyncServer initialized")

    # Initialize queue with server reference
    await initialize_queue(server)
    logger.info("Queue service started with AsyncServer integration")

    # Initialize LangFuse observability (async)
    try:
        from mirix.observability import initialize_langfuse

        logger.info("Initializing LangFuse observability...")
        await initialize_langfuse()
    except Exception as e:
        logger.warning(
            f"LangFuse initialization failed: {e}. Continuing without observability."
        )


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
        flush_success = await flush_langfuse(timeout=10.0)
        if flush_success:
            logger.info("LangFuse traces flushed successfully")
        else:
            logger.warning("LangFuse trace flush completed with warnings")
        await shutdown_langfuse()
        logger.info("LangFuse observability shut down")
    except Exception as e:
        logger.warning(f"Error during LangFuse shutdown: {e}")

    # Cleanup queue
    queue_manager = get_queue_manager()
    await queue_manager.cleanup()
    logger.info("Queue service stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    await initialize()
    sweeper = await start_session_sweeper()
    app.state.session_sweeper = sweeper
    reembedder = start_skill_reembedder()
    app.state.skill_reembedder = reembedder
    log_purger = start_retrieval_log_purger()
    app.state.retrieval_log_purger = log_purger

    yield  # Server runs here

    # Shutdown
    await stop_retrieval_log_purger(log_purger)
    await stop_skill_reembedder(reembedder)
    await stop_session_sweeper(sweeper)
    await cleanup()


def start_retrieval_log_purger():
    """Start the retrieval log's retention purge in the background:
    rows older than MIRIX_RETRIEVAL_LOG_RETENTION_DAYS (default 30; <= 0 keeps
    them) are deleted at startup and every hour. None when it could not start."""
    from mirix.services.retrieval_log_manager import RetrievalLogPurger

    purger = RetrievalLogPurger()
    try:
        purger.start()
    except Exception:  # noqa: BLE001 — never block startup on the purge
        logger.exception("Could not start the retrieval log purge")
        return None
    return purger


async def stop_retrieval_log_purger(purger) -> None:
    """Cancel the retention purge (shutdown)."""
    if purger is None:
        return
    try:
        await purger.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Error stopping the retrieval log purge")


def start_skill_reembedder():
    """Start the skill re-embedding pass in the background.

    Skills whose ``retrieval_embedding`` IS NULL (after the retrieval-field
    migration, or an embedding call that failed for good) are re-embedded
    from their retrieval text, at most ``MIRIX_SKILL_REEMBED_CONCURRENCY`` at
    a time; the server does not wait for it. None when it could not start.
    """
    from mirix.services.skill_retrieval_reembed import SkillRetrievalReembedder

    reembedder = SkillRetrievalReembedder()
    try:
        reembedder.start()
    except Exception:  # noqa: BLE001 — never block startup on the re-embedding
        logger.exception("Could not start the skill re-embedding pass")
        return None
    return reembedder


async def stop_skill_reembedder(reembedder) -> None:
    """Cancel the re-embedding pass if it is still running (shutdown)."""
    if reembedder is None:
        return
    try:
        await reembedder.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Error stopping the skill re-embedding pass")


async def start_session_sweeper():
    """Start the skill-learning session sweeper.

    Every MIRIX_SESSION_SWEEP_INTERVAL_SECONDS (default 3600; <= 0 disables the
    sweep) it reclaims learning runs whose lease expired, then runs the
    skill-learning trigger check for every scope that has pending sessions —
    the only way sessions that ended by idling get counted when their user
    never returns. Lease recovery also runs on its own, at startup and every
    half lease (MIRIX_LEARNING_RUN_LEASE_SECONDS / 2), whatever the sweep
    interval, so a run orphaned by a crash releases its sessions about one
    lease later.
    """
    from mirix.services.skill_learning_trigger import SessionSweeper

    sweeper = SessionSweeper()
    try:
        sweeper.start()
    except Exception:  # noqa: BLE001 — never block startup on the sweeper
        logger.exception("Could not start the session sweeper")
    return sweeper


async def stop_session_sweeper(sweeper) -> None:
    """Cancel the sweeper and any in-flight learning runs (shutdown)."""
    from mirix.services.skill_learning_trigger import cancel_background_learning_runs

    if sweeper is not None:
        try:
            await sweeper.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping the session sweeper")
    try:
        cancelled = await cancel_background_learning_runs()
        if cancelled:
            logger.info("Cancelled %d in-flight skill learning run(s)", cancelled)
    except Exception:  # noqa: BLE001
        logger.exception("Error cancelling in-flight skill learning runs")


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
_current_request: ContextVar[Optional[Request]] = ContextVar(
    "current_request", default=None
)


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

        # Langfuse returns a sync context manager (_AgnosticContextManager); use
        # "with" not "async with" to avoid "does not support the asynchronous
        # context manager protocol".
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
            client_id, org_id = await get_client_and_org(x_api_key=x_api_key)

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
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )

    return await call_next(request)


# ============================================================================
# Helper Functions
# ============================================================================


async def get_client_and_org(
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
        client = await server.client_manager.get_client_by_api_key(x_api_key)
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


async def extract_topics_and_temporal_info(
    messages: List[Dict[str, Any]], llm_config: LLMConfig
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract topics AND a temporal expression from OpenAI-style messages.

    Kept for callers of the old helper: the work is
    :func:`mirix.services.topic_extraction.extract_topics` — ``content`` may be a
    string, a list of parts (images included, sent WITH the text) or None,
    the prompt is generic, and the input is cut to the model's budget.

    Returns:
        ``(topics, temporal_expression)``: topics separated by ';' and the time
        phrase; ``(None, None)`` when the messages are unusable or extraction
        failed.
    """
    from mirix.services.conversation_content import MessageParseError
    from mirix.services.topic_extraction import (
        extract_topics,
        normalize_retrieval_messages,
    )

    try:
        parsed = normalize_retrieval_messages(messages)
    except MessageParseError as exc:
        logger.warning("Topic extraction skipped: %s", exc)
        return None, None
    outcome = await extract_topics(parsed, llm_config)
    return outcome.topics, outcome.temporal_expression


async def extract_topics_from_messages(
    messages: List[Dict[str, Any]], llm_config: LLMConfig
) -> Optional[str]:
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
    topics, _ = await extract_topics_and_temporal_info(messages, llm_config)
    return topics


def _flatten_messages_to_plain_text(messages: List[Dict[str, Any]]) -> str:
    """
    Flatten OpenAI-style message payloads into a simple conversation transcript:
    one ``ROLE: text`` line per message with text (images left out). This is
    the query when topic extraction fails.

    ``content`` may be a string, a list of parts or None; an assistant's
    ``tool_calls`` and a tool result's tool name are kept, as the extractor
    sees them. A payload the message parser rejects is flattened leniently
    (its text parts only).
    """
    from mirix.services.conversation_content import MessageParseError, parse_messages
    from mirix.services.topic_extraction import plain_text

    try:
        return plain_text(parse_messages(messages))
    except MessageParseError:
        pass

    transcript_parts: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content", "")
        if "role" not in msg and msg.get("type") == "text":
            content = [msg]

        parts: List[str] = []
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                elif isinstance(chunk, str):
                    parts.append(chunk.strip())
        elif isinstance(content, str):
            parts.append(content.strip())

        combined = " ".join(filter(None, parts)).strip()
        if combined:
            transcript_parts.append(f"{str(role).upper()}: {combined}")

    return "\n".join(transcript_parts)


async def extract_topics_with_local_model(
    messages: List[Dict[str, Any]],
    model_name: str,
    conversation: Optional[str] = None,
) -> Optional[str]:
    """
    Extract topics using a locally hosted Ollama model via the /api/chat endpoint.

    ``conversation`` is the text to send (the caller cuts it to the local
    model's budget); by default the messages flattened to text.

    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    Uses native async HTTP (httpx).
    """

    base_url = model_settings.ollama_base_url
    if not base_url:
        logger.warning(
            "local_model_for_retrieval provided (%s) but MIRIX_OLLAMA_BASE_URL is not configured",
            model_name,
        )
        return None

    if conversation is None:
        conversation = _flatten_messages_to_plain_text(messages)
    if not conversation:
        logger.debug("No text content found in messages for local topic extraction")
        return None

    from mirix.services.topic_extraction import (
        CONVERSATION_HEADER,
        LOCAL_MODEL_INSTRUCTION,
        LOCAL_MODEL_SYSTEM_PROMPT,
    )

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": LOCAL_MODEL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{CONVERSATION_HEADER}{conversation}\n\n{LOCAL_MODEL_INSTRUCTION}"
                ),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Failed to extract topics with local model %s: %s", model_name, exc
        )
        return None
    except httpx.RequestError as exc:
        logger.error(
            "Failed to extract topics with local model %s: %s", model_name, exc
        )
        return None

    message_payload = (
        response_data.get("message") if isinstance(response_data, dict) else None
    )
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
    """Health check endpoint.

    Also reports the effective skill-trigger config so drivers can fail fast
    when the in-band procedural trigger is unexpectedly enabled. The added
    fields are additive and backward-compatible.
    """
    from mirix.constants import (
        MESSAGE_RETAIN_LAST_N_SESSIONS,
        SKILL_TRIGGER_SESSION_THRESHOLD,
    )

    return {
        "status": "healthy",
        "service": "mirix-api",
        "skill_trigger_session_threshold": SKILL_TRIGGER_SESSION_THRESHOLD,
        "message_retain_last_n_sessions": MESSAGE_RETAIN_LAST_N_SESSIONS,
    }


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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    tags_list = tags.split(",") if tags else None

    return await server.agent_manager.list_agents(
        actor=client,
        query_text=query_text,
        limit=limit,
        cursor=cursor,
        parent_id=parent_id,
        tags=tags_list,
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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    # Create memory blocks if provided
    if request.memory:
        for block in request.memory.get_blocks():
            await server.block_manager.create_or_update_block(block, actor=client)

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

    agent_state = await server.create_agent(CreateAgent(**create_params), client)

    return await server.agent_manager.get_agent_by_id(agent_state.id, client)


@router.get("/agents/{agent_id}", response_model=AgentState)
async def get_agent(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent by ID."""
    from mirix.orm.errors import NoResultFound

    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    try:
        return await server.agent_manager.get_agent_by_id(agent_id, client)
    except NoResultFound as e:
        raise HTTPException(
            status_code=404, detail=f"Agent {agent_id} not found or not accessible"
        )


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete an agent."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    await server.agent_manager.delete_agent(agent_id, client)
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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    # List all top-level agents for this client
    top_level_agents = await server.agent_manager.list_agents(actor=client, limit=1000)
    # Also get sub-agents (children of meta agent)
    all_agents = list(top_level_agents)
    for agent in top_level_agents:
        if agent.name == "meta_memory_agent":
            # Get sub-agents
            sub_agents = await server.agent_manager.list_agents(
                actor=client,
                parent_id=agent.id,
                limit=1000,
            )
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
                    agent.name.replace("meta_memory_agent_", "")
                    .replace("_memory_agent", "")
                    .replace("_agent", "")
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
                    full_name.replace("meta_memory_agent_", "")
                    .replace("_memory_agent", "")
                    .replace("_agent", "")
                )
                available_agents.append(f"'{short_name}' (full: {full_name})")
            else:
                available_agents.append(f"'{full_name}'")

        available_list = ", ".join(available_agents[:5])  # Show first 5
        if len(available_agents) > 5:
            available_list += f", and {len(available_agents) - 5} more"

        error_detail = (
            f"Agent with name '{agent_name}' not found for client {client_id}. "
        )
        if available_agents:
            error_detail += f"Available agents: {available_list}"
        else:
            error_detail += (
                "No agents found for this client. Please initialize agents first."
            )

        raise HTTPException(status_code=404, detail=error_detail)

    # Call the update_agent_system_prompt endpoint logic
    updated_agent = await server.agent_manager.update_system_prompt(
        matching_agent.id,
        request.system_prompt,
        client,
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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    updated_agent = await server.agent_manager.update_system_prompt(
        agent_id,
        request.system_prompt,
        client,
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
    session_id: Optional[str] = None  # Top-level session identifier
    stream_steps: bool = False
    stream_tokens: bool = False
    filter_tags: Optional[Dict[str, Any]] = None  # Filter tags support
    block_filter_tags: Optional[Dict[str, Any]] = (
        None  # Applied only when blocks are created (e.g. from default template)
    )
    block_filter_tags_update_mode: Optional[str] = "merge"  # "merge" or "replace"
    use_cache: bool = True  # Control Redis cache behavior

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, v: Optional[str]) -> Optional[str]:
        return _validate_session_id(v)


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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    if request.block_filter_tags is not None and not isinstance(
        request.block_filter_tags, dict
    ):
        raise HTTPException(
            status_code=400, detail="block_filter_tags must be a dict when provided"
        )
    if request.block_filter_tags is not None:
        request.block_filter_tags.pop("scope", None)
    if request.block_filter_tags_update_mode not in ("merge", "replace"):
        raise HTTPException(
            status_code=400,
            detail="block_filter_tags_update_mode must be 'merge' or 'replace'",
        )

    try:
        # Prepare the message
        message_create = MessageCreate(
            role=MessageRole(request.role),
            content=request.message,
            name=request.name,
            session_id=request.session_id,
        )

        # Put message on queue for processing
        await put_messages(
            actor=client,
            agent_id=agent_id,
            input_messages=[message_create],
            chaining=True,
            user_id=request.user_id,  # Pass user_id to queue
            filter_tags=request.filter_tags,  # Pass filter_tags to queue
            block_filter_tags=request.block_filter_tags,
            block_filter_tags_update_mode=request.block_filter_tags_update_mode,
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
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    return await server.tool_manager.list_tools(
        cursor=cursor, limit=limit, actor=client
    )


@router.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a tool by ID."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    return await server.tool_manager.get_tool_by_id(tool_id, actor=client)


@router.post("/tools", response_model=Tool)
async def create_tool(
    tool: Tool,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new tool."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    return await server.tool_manager.create_tool(tool, actor=client)


@router.delete("/tools/{tool_id}")
async def delete_tool(
    tool_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a tool."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    await server.tool_manager.delete_tool_by_id(tool_id, actor=client)
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
    """List all blocks (filtered by client's read_scopes)."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    user = await server.user_manager.get_admin_user()
    return await server.block_manager.get_blocks(
        user=user,
        label=label,
        any_scopes=client.read_scopes,
        auto_create_from_default=False,
    )


@router.get("/blocks/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a block by ID."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    # Get admin user for block queries (blocks are user-scoped, not client-scoped)
    user = await server.user_manager.get_admin_user()
    return await server.block_manager.get_block_by_id(block_id, user=user)


@router.post("/blocks", response_model=Block)
async def create_block(
    block: Block,
    user: Optional[User] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a block."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    return await server.block_manager.create_or_update_block(
        block, actor=client, user=user
    )


@router.delete("/blocks/{block_id}")
async def delete_block(
    block_id: str,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a block."""
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    await server.block_manager.delete_block(block_id, actor=client)
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
    return await server.list_llm_models()


@router.get("/config/embedding", response_model=List[EmbeddingConfig])
async def list_embedding_configs(
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available embedding configurations."""
    server = get_server()
    return await server.list_embedding_models()


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
    return await server.organization_manager.list_organizations(
        cursor=cursor, limit=limit
    )


@router.post("/organizations", response_model=Organization)
async def create_organization(
    name: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create an organization."""
    server = get_server()
    return await server.organization_manager.create_organization(
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
        return await server.organization_manager.get_organization_by_id(org_id)
    except Exception:
        return await server.get_organization_or_default(org_id)


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
        org = await server.organization_manager.get_organization_by_id(org_id)
        if org:
            return org
    except Exception:
        pass

    # Create new organization if it doesn't exist
    org_create = OrganizationCreate(id=org_id, name=request.name or org_id)
    org = await server.organization_manager.create_organization(
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
    try:
        return await server.user_manager.get_user_by_id(user_id)
    except NoResultFound:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")


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

    The user is organization-scoped (not client-scoped).
    The organization is determined from the API key or JWT token.
    If user_id is not provided, a random ID will be generated.
    """
    server = get_server()

    # Accept JWT or injected headers (from API key middleware)
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Extract org_id from client object
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
        user = await server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass

    from mirix.schemas.user import User as PydanticUser

    # Create a User object with all required fields (organization-scoped)
    user = await server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=request.name or user_id,
            organization_id=org_id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
            status="active",
        ),
    )
    logger.debug("Created new user: %s in organization: %s", user_id, org_id)
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
        await server.user_manager.delete_user_by_id(user_id)
        return {"message": f"User {user_id} soft deleted successfully"}
    except Exception as e:
        error_msg = str(e)
        # Provide a better error message if user not found or already deleted
        if "not found" in error_msg.lower() or "no result" in error_msg.lower():
            raise HTTPException(
                status_code=404, detail=f"User {user_id} not found or already deleted"
            )
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/users/{user_id}/memories")
async def delete_user_memories(
    user_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Hard delete all memories, messages, and blocks for a user.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    The target user must belong to the caller's organization.

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
    - Conversation transcripts recorded for this user
    - Skill experiences distilled from this user's sessions

    Records that are PRESERVED:
    - User record

    Warning: This operation is irreversible. Deleted data cannot be recovered.
    """
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()

    # Tenant guard: an irreversible cross-org erasure must be impossible. A
    # foreign-org target returns the same 404 as a missing user so the endpoint
    # can't be used as a user-id existence oracle.
    default_org = server.organization_manager.DEFAULT_ORG_ID
    try:
        target_user = await server.user_manager.get_user_by_id(user_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    caller_org = client.organization_id or default_org
    target_org = target_user.organization_id or default_org
    if caller_org != target_org:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        await server.user_manager.delete_memories_by_user_id(user_id)
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
    organization_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """
    List all users for the authenticated client's organization.

    **Requires JWT authentication (dashboard only).**

    Users are organization-scoped, so this returns all users in the organization.
    """
    # Require admin JWT authentication
    client_payload = get_current_admin(authorization)

    server = get_server()

    # Get client to determine organization
    client = await server.admin_user_manager.get_client_by_id(client_payload["sub"])
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Use provided organization_id or default to client's organization
    org_id = organization_id or client.organization_id

    users = await server.user_manager.list_users(
        cursor=cursor,
        limit=limit,
        organization_id=org_id,
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
    write_scope: Optional[str] = None
    read_scopes: List[str] = Field(default_factory=list)
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
        client = await server.client_manager.get_client_by_id(client_id)

        if client:
            if fail_if_exists:
                raise HTTPException(
                    status_code=409,
                    detail=f"Client with id '{client_id}' already exists",
                )
            else:
                logger.debug("Client already exists: %s", client_id)
                return JSONResponse(
                    status_code=200, content=client.model_dump(mode="json")
                )
    except Exception as e:
        if fail_if_exists and "already exists" in str(e):
            raise
        pass  # Client doesn't exist, proceed to create

    # Create a Client object with all required fields
    client = await server.client_manager.create_client(
        pydantic_client=Client(
            id=client_id,
            name=request.name or client_id,
            organization_id=org_id,
            status=request.status or "active",
            write_scope=request.write_scope,
            read_scopes=request.read_scopes,
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

    clients = await server.client_manager.list_clients(cursor=cursor, limit=limit)
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
    client = await server.client_manager.get_client_by_id(client_id)

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
        updated_client = await server.client_manager.update_client(update)
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
        await server.client_manager.delete_client_by_id(client_id)
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
async def delete_client_memories(
    client_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Hard delete all memories, messages, and blocks for a client.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**
    The target client must belong to the caller's organization.

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
    - Conversation transcripts recorded by this client
    - Skill experiences created by this client

    Records that are PRESERVED:
    - Client record
    - Agents created by this client
    - Tools created by this client

    Warning: This operation is irreversible. Deleted data cannot be recovered.
    """
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()

    # Tenant guard: same-org only; a missing OR foreign-org target returns the
    # same 404 so the endpoint can't be used to probe client-id existence.
    # get_client_by_id RAISES (NoResultFound) on a miss rather than returning
    # None, so the miss must be caught here, not compared to None.
    default_org = server.organization_manager.DEFAULT_ORG_ID
    try:
        target_client = await server.client_manager.get_client_by_id(client_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    if (target_client.organization_id or default_org) != (
        client.organization_id or default_org
    ):
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    try:
        await server.client_manager.delete_memories_by_client_id(client_id)
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
    permission: Optional[str] = Field(
        "all", description="Permission level: all, restricted, read_only"
    )
    user_id: Optional[str] = Field(
        None, description="User ID this API key is associated with"
    )


class CreateApiKeyResponse(BaseModel):
    """Response model for API key creation - includes the raw API key (only shown once)."""

    id: str
    client_id: str
    name: Optional[str]
    api_key: str  # Raw API key - only returned at creation time
    status: str
    permission: str  # Permission level: all, restricted, read_only
    created_at: Optional[datetime]


def _require_own_client(authorization: Optional[str], client_id: str) -> dict:
    """The dashboard JWT's payload, when it is the path's client's own token.

    The api-key endpoints act on the client named in the path, which the
    caller chooses: a token of another client (of any organization) must not
    mint, list or revoke that client's keys. Such a call is a 404, like an
    unknown client — whether the client exists is not revealed.
    """
    payload = get_current_admin(authorization)
    if payload.get("sub") != client_id:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    return payload


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
    Subsequent requests will only show the key ID, not the raw key. Only the
    client itself (its own dashboard token) can create its keys; any other
    token gets a 404.

    Returns:
        CreateApiKeyResponse with the raw API key (only shown once)
    """
    # Require the client's own dashboard JWT.
    _require_own_client(authorization, client_id)

    from mirix.security.api_keys import generate_api_key

    server = get_server()

    # Verify client exists
    client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    # Generate new API key
    raw_api_key = generate_api_key()

    # Create API key record (stores hashed version)
    api_key_record = await server.client_manager.create_client_api_key(
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
    Only metadata (id, name, status, created_at) is shown. Only the client
    itself (its own dashboard token) can list its keys; any other token gets
    a 404.
    """
    # Require the client's own dashboard JWT.
    _require_own_client(authorization, client_id)

    server = get_server()

    # Verify client exists
    client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    api_keys = await server.client_manager.list_client_api_keys(client_id)

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
    Any applications using this key will immediately stop working. Only the
    client itself (its own dashboard token) can delete its keys; any other
    token gets a 404.
    """
    # Require the client's own dashboard JWT.
    _require_own_client(authorization, client_id)

    server = get_server()

    # Verify client exists
    client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    try:
        # Scoped to the path's client: another client's key id is a 404.
        await server.client_manager.delete_client_api_key(
            api_key_id, client_id=client_id
        )
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


@router.post("/agents/meta/initialize", response_model=Optional[AgentState])
async def initialize_meta_agent(
    request: InitializeMetaAgentRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Initialize a meta agent with configuration.

    This creates a meta memory agent that manages specialized memory agents.
    Returns null if the client has no write_scope (read-only clients don't need agents).
    """
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    # Read-only clients (no write_scope) don't create agents — return null
    if not client.write_scope:
        return None

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
    existing_meta_agents = await server.agent_manager.list_agents(
        actor=client, limit=1000
    )

    assert len(existing_meta_agents) <= 1, (
        "Only one meta agent can be created per client"
    )

    if len(existing_meta_agents) == 1:
        meta_agent = existing_meta_agents[0]

        # Only update the meta agent if update_agents is True
        if request.update_agents:
            from mirix.schemas.agent import UpdateMetaAgent

            # DEBUG: Log what we're passing to update_meta_agent
            logger.debug(
                "[INIT META AGENT] create_params for UpdateMetaAgent: %s", create_params
            )
            logger.debug(
                "[INIT META AGENT] 'agents' in create_params: %s",
                "agents" in create_params,
            )
            if "agents" in create_params:
                logger.debug(
                    "[INIT META AGENT] agents list: %s", create_params["agents"]
                )

            # Update the existing meta agent
            meta_agent = await server.agent_manager.update_meta_agent(
                meta_agent.id,
                UpdateMetaAgent(**create_params),
                client,
            )
    else:
        from mirix.schemas.agent import CreateMetaAgent

        meta_agent = await server.agent_manager.create_meta_agent(
            meta_agent_create=CreateMetaAgent(**create_params), actor=client
        )

    return meta_agent


class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    user_id: Optional[str] = None  # Optional - uses admin user if not provided
    meta_agent_id: str
    # Non-empty: an empty list is a 422, not an IndexError (500).
    messages: List[Dict[str, Any]] = Field(..., min_length=1)
    chaining: bool = True
    verbose: bool = False
    filter_tags: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = (
        None  # Batch-level session id applied to every message in this request
    )
    session_tag: Optional[str] = (
        None  # "task" | "conversation"; routes memory machinery. None -> DEFAULT_SESSION_TAG.
    )
    block_filter_tags: Optional[Dict[str, Any]] = (
        None  # Applied only when blocks are created (e.g. from default template)
    )
    block_filter_tags_update_mode: Optional[str] = "merge"  # "merge" or "replace"
    use_cache: bool = True  # Control Redis cache behavior
    occurred_at: Optional[str] = (
        None  # Optional ISO 8601 timestamp string for episodic memory
    )
    # Optional: how the agent's prompt was built — its prompt
    # template or setup, as the client describes it. Stored on the session
    # (not a turn; the latest one given wins) and shown to the skill
    # distiller beside the session's transcript. Needs a session_id.
    agent_context: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, v: Optional[str]) -> Optional[str]:
        return _validate_session_id(v)

    @field_validator("session_tag")
    @classmethod
    def _check_session_tag(cls, v: Optional[str]) -> Optional[str]:
        from mirix.constants import VALID_SESSION_TAGS

        if v is None:
            return None
        if v not in VALID_SESSION_TAGS:
            raise ValueError(
                f"session_tag must be one of {VALID_SESSION_TAGS}; got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _check_agent_context_has_a_session(self) -> "AddMemoryRequest":
        if (
            self.agent_context is not None
            and self.agent_context.strip()
            and self.session_id is None
        ):
            raise ValueError(
                "agent_context needs a session_id: it is stored with the session"
            )
        return self

    @model_validator(mode="after")
    def _check_session_id_agrees_with_filter_tags(self) -> "AddMemoryRequest":
        # If both the top-level session_id and filter_tags.session_id are
        # provided, they must match. Otherwise message-scope and memory-scope silently diverge.
        if self.session_id is not None and isinstance(self.filter_tags, dict):
            tag_sid = self.filter_tags.get("session_id")
            if tag_sid is not None and tag_sid != self.session_id:
                raise ValueError(
                    "session_id and filter_tags['session_id'] must agree; "
                    f"got {self.session_id!r} vs {tag_sid!r}"
                )
        return self


class SessionIngestResult(BaseModel):
    """What the session ingestion did (returned into the add responses)."""

    effective_tag: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    learning_run_id: Optional[str] = None
    stored_count: int = 0


def _parse_add_messages(messages: Any):
    """Parse a memory-add payload once for both consumers; 422 on a bad shape."""
    from mirix.services.conversation_content import MessageParseError, parse_messages

    try:
        return parse_messages(messages)
    except MessageParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _prepare_payload_images(parsed) -> None:
    """Decode every inline image before anything is written; 422 if one is bad."""
    from mirix.services.conversation_content import MessageParseError, iter_images
    from mirix.services.image_store import prepare_images

    try:
        await prepare_images(iter_images(parsed))
    except MessageParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _check_file_references(parsed, organization_id: str, user_id: str) -> None:
    """422 unless the caller may use every ``files`` row the payload references.

    ``database_image_id`` / ``database_file_id`` / ``database_google_cloud_file_uri``
    parts point at an existing ``files`` row by id; the model clients later read
    that row's bytes into the caller's memory extraction. The row must belong to
    the caller's organization and to this user (a legacy row without a user is
    organization-wide). Missing and foreign ids get the same answer, so the
    check reveals nothing about other tenants' files. Runs before anything is
    written.
    """
    from sqlalchemy import select

    from mirix.orm.file import FileMetadata as FileModel
    from mirix.server.server import db_context
    from mirix.services.conversation_content import referenced_file_ids

    file_ids = referenced_file_ids(parsed)
    if not file_ids:
        return
    async with db_context() as session:
        rows = (
            await session.execute(
                select(FileModel.id, FileModel.organization_id, FileModel.user_id).where(
                    FileModel.id.in_(file_ids)
                )
            )
        ).all()
    accessible = {
        file_id
        for file_id, org_id, owner in rows
        if org_id == organization_id and owner in (None, user_id)
    }
    for file_id in file_ids:
        if file_id not in accessible:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"a content part references file {file_id!r}, which does not "
                    "exist or is not accessible to this caller"
                ),
            )


def _session_conflict(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            f"session {session_id!r} belongs to another meta agent for this user; "
            "nothing was stored (use a different session_id)"
        ),
    )


async def _predict_session_tag(request, organization_id: str, user_id: str) -> str:
    """The tag this write will be processed under.

    An existing session keeps the tag its first write fixed, whatever this
    request says (a mismatch only produces a warning from the store write). A
    new session — or a request without a session_id — takes the requested tag,
    or DEFAULT_SESSION_TAG. The store write reports the tag it actually applied
    (`SessionIngestResult.effective_tag`), which then wins for routing.
    """
    from mirix.constants import DEFAULT_SESSION_TAG
    from mirix.services.conversation_session_manager import ConversationSessionManager

    requested = getattr(request, "session_tag", None)
    if request.session_id is not None:
        sessions = ConversationSessionManager()
        existing = await sessions.get_session(
            organization_id=organization_id,
            user_id=user_id,
            session_id=request.session_id,
        )
        meta_agent_id = getattr(request, "meta_agent_id", None)
        if (
            existing is not None
            and existing.agent_id is not None
            and meta_agent_id is not None
            and existing.agent_id != meta_agent_id
            and await sessions.agent_is_live(existing.agent_id)
        ):
            # Another (live) meta agent's session: neither its tag nor its
            # store may be used by this write. A session whose meta agent
            # was deleted is orphaned; the store write decides whether this
            # writer may adopt it.
            raise _session_conflict(request.session_id)
        if existing is not None and existing.session_tag:
            return existing.session_tag
    return requested or DEFAULT_SESSION_TAG


async def _ensure_user_exists(server, client, user_id: str):
    """create_or_get_user for the add endpoints — BEFORE any write of the user's data.

    The conversation store and the image `files` rows reference `users` by
    foreign key, so a first-contact user of a direct REST caller must exist
    before the store write; the SDK has always created it first.
    """
    from mirix.schemas.user import User as PydanticUser
    from mirix.services.conversation_message_manager import owner_org

    # Tenant guard: an existing user of ANOTHER organization is a 404 —
    # a client never writes sessions, learning runs or skills for another
    # tenant's user.
    existing = await _tenant_user_for_write(server, user_id, client)
    if existing is not None:
        return existing
    try:
        return await server.user_manager.create_user(
            pydantic_user=PydanticUser(
                id=user_id,
                name=user_id,
                organization_id=owner_org(client),
                timezone=getattr(server.user_manager, "DEFAULT_TIME_ZONE", "UTC"),
                status="active",
                is_deleted=False,
                is_admin=False,
            )
        )
    except Exception as create_error:
        # A concurrent first write may have created the user meanwhile.
        existing = await _tenant_user_for_write(server, user_id, client)
        if existing is None:
            raise HTTPException(
                status_code=409,
                detail=f"User {user_id!r} does not exist and could not be created",
            ) from create_error
        return existing


async def _ingest_session_turns(
    request,
    client,
    user_id: str,
    *,
    parsed=None,
    strict: bool = False,
    max_text_chars: Optional[int] = None,
) -> Optional[SessionIngestResult]:
    """Shared /memory/add + /memory/add_sync conversation-store write.

    Persists the request's turns (REAL user/assistant/tool roles, images as
    `content_parts` image refs) into the Conversation Message Store,
    the single source of truth for procedural (skill) distillation, together
    with the session's lifecycle row (created, or reopened if it had ended).
    Without a session_id nothing is written.

    The session's first write fixes its tag; a request naming a different tag
    keeps the stored one and gets a warning. After a write to a learnable
    (task) session the skill-learning trigger check runs for the session's
    scope; it fires only once enough task sessions have ended.

    `strict` is the task-session contract: the store is the ONLY place a
    task session goes, so a storage error propagates to the caller. For a
    conversation session the store write is additive — a failure is logged and
    the meta-agent ingestion proceeds (worst case: one session missing from the
    store). Images stored for a write that then fails are discarded. The
    caller must have created the user already. Returns None when nothing
    was recorded — or, when the request's agent_context could not be stored
    for want of a turn to store, a result with no turn and a warning saying so.
    """
    if request.session_id is None:
        return None

    from mirix.constants import SESSION_TAG_TASK
    from mirix.services.conversation_content import store_turn_drafts
    from mirix.services.conversation_message_manager import (
        ConversationMessageManager,
        owner_org,
    )
    from mirix.services.conversation_session_manager import SessionOwnershipConflict
    from mirix.services.image_store import discard_image_refs, store_image_part

    requested_tag = getattr(request, "session_tag", None)
    meta_agent_id = getattr(request, "meta_agent_id", None)
    organization_id = owner_org(client)

    stored_refs: List[Dict[str, Any]] = []
    try:
        if parsed is None:
            parsed = _parse_add_messages(request.messages)
        drafts = store_turn_drafts(parsed)
        if not drafts:
            if (getattr(request, "agent_context", None) or "").strip():
                # An agent context is stored with the session's turns;
                # with none to store, the caller is told it was not stored.
                from mirix.services.conversation_message_manager import (
                    AGENT_CONTEXT_NOT_STORED_WARNING,
                )

                logger.warning("%s", AGENT_CONTEXT_NOT_STORED_WARNING)
                return SessionIngestResult(warnings=[AGENT_CONTEXT_NOT_STORED_WARNING])
            return None
        turns = []
        for draft in drafts:
            refs = {}
            for image in draft.images:
                ref = await store_image_part(
                    image, organization_id=organization_id, user_id=user_id
                )
                stored_refs.append(ref)
                refs[id(image)] = ref
            turns.append(draft.to_turn(refs))
        recorded = await ConversationMessageManager().record_turns(
            session_id=request.session_id,
            user_id=user_id,
            organization_id=organization_id,
            turns=turns,
            actor=client,
            session_tag=requested_tag,
            agent_id=meta_agent_id,
            max_text_chars=max_text_chars,
            agent_context=getattr(request, "agent_context", None),
        )
    except SessionOwnershipConflict as exc:
        # Never "additive": the write names another meta agent's session.
        await discard_image_refs(stored_refs, organization_id=organization_id)
        raise _session_conflict(request.session_id) from exc
    except Exception:
        await discard_image_refs(stored_refs, organization_id=organization_id)
        if strict:
            raise
        logger.exception(
            "Failed to record conversation turns to the store for "
            "session_id=%s; memory ingestion will still proceed",
            request.session_id,
        )
        return None

    session_row = getattr(recorded, "session", None)
    result = SessionIngestResult(
        effective_tag=getattr(recorded, "effective_tag", None),
        warnings=list(getattr(recorded, "warnings", None) or []),
        stored_count=len(recorded),
    )
    for warning in result.warnings:
        logger.warning("%s", warning)

    if session_row is not None and result.effective_tag in (SESSION_TAG_TASK, None):
        from mirix.services.skill_learning_trigger import trigger_skill_learning

        try:
            claim = await trigger_skill_learning(
                organization_id=organization_id,
                user_id=user_id,
                agent_id=session_row.agent_id,
                trigger="threshold",
                actor=client,
                current_session_id=request.session_id,
            )
            if claim.fired:
                result.learning_run_id = claim.run_id
        except Exception:
            # The write succeeded; the sweeper repeats the check later.
            logger.exception(
                "Skill-learning trigger check failed after writing session %s",
                request.session_id,
            )
    return result


def _resolved_session_tag(
    request, ingest: Optional[SessionIngestResult], predicted: Optional[str] = None
) -> str:
    """The tag routing sees: the one the store write applied, else the prediction."""
    from mirix.constants import DEFAULT_SESSION_TAG

    if ingest is not None and ingest.effective_tag:
        return ingest.effective_tag
    if predicted:
        return predicted
    return getattr(request, "session_tag", None) or DEFAULT_SESSION_TAG


@dataclasses.dataclass
class _AddPlan:
    """Outcome of the shared add pipeline, before the meta-agent dispatch."""

    effective_tag: str
    parsed: Any
    user: Any = None
    ingest: Optional[SessionIngestResult] = None
    stored_response: Optional[Dict[str, Any]] = None


async def _run_add_pipeline(server, request: "AddMemoryRequest", client, meta_agent, user_id: str) -> _AddPlan:
    """Everything /memory/add and /memory/add_sync share, in order:

    1. parse the payload once (422 on a bad shape, and on parts that
       name a file on the server), decode its inline images (422 if one is not
       a supported image or is too large) and check the caller owns every
       stored file the payload references (422) — nothing is written yet, so
       nothing later in the request can still 422;
    2. resolve the session tag — 409 if the session belongs to another
       meta agent; a task session needs a session_id (422) and at least one
       user/assistant/tool turn (422);
    3. create_or_get_user — before any write of the user's data;
    4. write the conversation store (task: errors propagate; conversation:
       additive) with the meta agent's storage cap, then the trigger check.

    A task session ends here with a "stored" response: it never reaches
    `send_messages` or the queue. Anything else continues to the meta agent.
    """
    from mirix.constants import SESSION_TAG_TASK
    from mirix.services.conversation_content import meta_agent_char_cap, store_turn_drafts
    from mirix.services.conversation_message_manager import owner_org

    parsed = _parse_add_messages(request.messages)
    await _prepare_payload_images(parsed)
    await _check_file_references(parsed, owner_org(client), user_id)

    predicted = await _predict_session_tag(request, owner_org(client), user_id)
    is_task = predicted == SESSION_TAG_TASK
    if is_task:
        if request.session_id is None:
            raise HTTPException(
                status_code=422,
                detail="session_tag 'task' requires a session_id: task sessions are "
                "stored in the conversation store for skill learning only",
            )
        if not store_turn_drafts(parsed):
            raise HTTPException(
                status_code=422,
                detail="a task session needs at least one user, assistant or tool "
                "message (role-bearing messages)",
            )

    user = await _ensure_user_exists(server, client, user_id)

    ingest = await _ingest_session_turns(
        request,
        client,
        user_id,
        parsed=parsed,
        strict=is_task,
        max_text_chars=meta_agent_char_cap(meta_agent),
    )
    effective = _resolved_session_tag(request, ingest, predicted)
    plan = _AddPlan(effective_tag=effective, parsed=parsed, user=user, ingest=ingest)
    if effective == SESSION_TAG_TASK:
        if ingest is None:  # unreachable: a task write either stores or raises
            raise HTTPException(status_code=500, detail="task session was not stored")
        plan.stored_response = {
            "success": True,
            "message": "Stored in the session conversation store for skill learning",
            "status": "stored",
            "agent_id": meta_agent.id,
            "session_id": request.session_id,
            "message_count": ingest.stored_count,
            "warnings": ingest.warnings,
            "learning_run_id": ingest.learning_run_id,
        }
    return plan


async def _meta_input_messages(
    plan: _AddPlan, request: "AddMemoryRequest", client, user_id: str, stored_refs: list
):
    """The meta agent's input (its own image copies are appended to ``stored_refs``).

    Everything that could make a payload unacceptable was checked before the
    store write (``_run_add_pipeline``), so a failure here is a server error;
    the image copies stored before it are discarded by the builder.
    """
    from mirix.services.conversation_content import build_meta_input_messages
    from mirix.services.conversation_message_manager import owner_org

    return await build_meta_input_messages(
        plan.parsed,
        organization_id=owner_org(client),
        user_id=user_id,
        session_id=request.session_id,
        stored_refs=stored_refs,
    )


def _require_meta_memory_agent(meta_agent) -> None:
    """The add endpoints feed a META memory agent: its id becomes the session's
    learning scope, so any other agent type is refused (400)."""
    if getattr(meta_agent, "agent_type", None) != AgentType.meta_memory_agent:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Agent {getattr(meta_agent, 'id', None)} is "
                f"{getattr(meta_agent, 'agent_type', None)}, not a meta_memory_agent."
            ),
        )


def _meta_filter_tags(request: "AddMemoryRequest", client, plan: _AddPlan) -> Dict[str, Any]:
    """filter_tags for the meta agent: the request's, plus session id/tag and scope."""
    filter_tags = dict(request.filter_tags) if request.filter_tags is not None else {}
    # Mirror session_id into filter_tags so extracted memories inherit it too.
    # The model_validator on AddMemoryRequest has already ensured agreement if both were set.
    if request.session_id is not None:
        filter_tags["session_id"] = request.session_id
    # The resolved session_tag (the one the session's first write fixed) rides
    # along for extracted-memory provenance.
    filter_tags["session_tag"] = plan.effective_tag
    # Memories are written with the client's write_scope.
    filter_tags["scope"] = client.write_scope
    return filter_tags


def _require_write_scope(client) -> None:
    if client.write_scope is None:
        raise HTTPException(
            status_code=403, detail="Client has no write_scope - cannot create memories"
        )


@router.post("/memory/add")
@with_langfuse_tracing
async def add_memory(
    request: AddMemoryRequest,
    x_org_id: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
):
    """
    Add conversation turns to memory (async via queue).

    A `task` session is stored in the session conversation store for
    skill learning only and returns `status: "stored"` — it is never queued for
    the meta memory agent. Everything else is queued for asynchronous
    processing by queue workers, allowing for fast API response times.
    """
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    # If client doesn't exist, create the default client
    if client is None:
        logger.warning("Client %s not found, creating default client", client_id)
        from mirix.services.client_manager import ClientManager

        if client_id == ClientManager.DEFAULT_CLIENT_ID:
            # Create the default client
            client = await server.client_manager.create_default_client(org_id)
        else:
            # Client ID was provided but doesn't exist - error
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found. Please create the client first.",
            )

    # Get the meta agent by ID
    # TODO: need to check if we really need to check if the meta_agent exists here
    meta_agent = await server.agent_manager.get_agent_by_id(
        request.meta_agent_id,
        client,
    )

    # If user_id is not provided, use the admin user for this client
    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Request-shape checks come before anything is written.
    if request.block_filter_tags is not None and not isinstance(
        request.block_filter_tags, dict
    ):
        raise HTTPException(
            status_code=400, detail="block_filter_tags must be a dict when provided"
        )
    if request.block_filter_tags is not None:
        request.block_filter_tags.pop("scope", None)
    if request.block_filter_tags_update_mode not in ("merge", "replace"):
        raise HTTPException(
            status_code=400,
            detail="block_filter_tags_update_mode must be 'merge' or 'replace'",
        )
    _require_write_scope(client)
    _require_meta_memory_agent(meta_agent)

    plan = await _run_add_pipeline(server, request, client, meta_agent, user_id)
    if plan.stored_response is not None:
        return plan.stored_response

    meta_refs: list = []
    input_messages = await _meta_input_messages(
        plan, request, client, user_id, meta_refs
    )
    filter_tags = _meta_filter_tags(request, client, plan)

    # Queue for async processing instead of synchronous execution
    # Note: actor is Client for org-level access control
    #       user_id represents the actual end-user (or admin user if not provided)
    try:
        await put_messages(
            actor=client,
            agent_id=meta_agent.id,
            input_messages=input_messages,
            chaining=request.chaining,
            user_id=user_id,  # End-user for data filtering (or admin user)
            verbose=request.verbose,
            filter_tags=filter_tags,
            block_filter_tags=request.block_filter_tags,
            block_filter_tags_update_mode=request.block_filter_tags_update_mode,
            use_cache=request.use_cache,
            occurred_at=request.occurred_at,  # Optional timestamp for episodic memory
        )
    except BaseException:
        # Never queued: nothing will ever reference the meta agent's copies.
        from mirix.services.conversation_message_manager import owner_org
        from mirix.services.image_store import discard_image_refs

        await discard_image_refs(meta_refs, organization_id=owner_org(client))
        raise

    logger.debug("Memory queued for processing: %s", meta_agent.id)

    return {
        "success": True,
        "message": "Memory queued for processing",
        "status": "queued",
        "agent_id": meta_agent.id,
        "message_count": len(input_messages),
        "warnings": plan.ingest.warnings if plan.ingest else [],
        "learning_run_id": plan.ingest.learning_run_id if plan.ingest else None,
    }


@router.post("/memory/add_sync")
@with_langfuse_tracing
async def add_memory_sync(
    request: AddMemoryRequest,
    x_org_id: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
):
    """
    Add conversation turns to memory (synchronous processing).

    A `task` session is stored in the session conversation store for
    skill learning only and returns `status: "stored"` without running the meta
    memory agent. Everything else is processed immediately by running the agent
    directly, blocking until all memory extraction is complete.
    """
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)

    if client is None:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} not found. Please create the client first.",
        )

    meta_agent = await server.agent_manager.get_agent_by_id(
        request.meta_agent_id,
        client,
    )

    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    _require_write_scope(client)
    _require_meta_memory_agent(meta_agent)

    plan = await _run_add_pipeline(server, request, client, meta_agent, user_id)
    if plan.stored_response is not None:
        return plan.stored_response

    # The meta agent's image copies stay even if the agent run below fails:
    # it may already have saved the input message that references them (they
    # are released with that message by the message cleanup).
    input_messages = await _meta_input_messages(plan, request, client, user_id, [])
    filter_tags = _meta_filter_tags(request, client, plan)

    await server.send_messages(
        actor=client,
        agent_id=meta_agent.id,
        input_messages=input_messages,
        chaining=request.chaining,
        user=plan.user,
        verbose=request.verbose,
        filter_tags=filter_tags,
        block_filter_tags=request.block_filter_tags,
        block_filter_tags_update_mode=request.block_filter_tags_update_mode,
        use_cache=request.use_cache,
        occurred_at=request.occurred_at,
    )

    logger.debug("Memory processed synchronously: %s", meta_agent.id)

    return {
        "success": True,
        "message": "Memory processed successfully",
        "status": "processed",
        "agent_id": meta_agent.id,
        "message_count": len(input_messages),
        "warnings": plan.ingest.warnings if plan.ingest else [],
        "learning_run_id": plan.ingest.learning_run_id if plan.ingest else None,
    }


class RetrieveMemoryRequest(BaseModel):
    """Request model for retrieving memory."""

    user_id: Optional[str] = None  # Optional - uses admin user if not provided
    # OpenAI-style messages; each content may be a string, a list of parts
    # (text, images) or null. Empty -> 422.
    messages: List[Dict[str, Any]]
    # Maximum number of items to retrieve per memory type.
    limit: int = Field(10, ge=0)
    local_model_for_retrieval: Optional[str] = (
        None  # Optional local Ollama model for topic extraction
    )
    filter_tags: Optional[Dict[str, Any]] = (
        None  # Optional filter tags for filtering results
    )
    use_cache: bool = True  # Control Redis cache behavior
    # NEW: Optional date range for temporal filtering (ISO 8601 format)
    start_date: Optional[str] = (
        None  # e.g., "2025-11-19T00:00:00" or "2025-11-19T00:00:00+00:00"
    )
    end_date: Optional[str] = (
        None  # e.g., "2025-11-19T23:59:59" or "2025-11-19T23:59:59+00:00"
    )
    # The memory types to retrieve (None = all of them). With
    # ["procedural"] only the skills are searched, with the similarity gate.
    memory_types: Optional[List[str]] = None
    # The caller's session (same rules as AddMemoryRequest's); it
    # is recorded on the retrieval log row.
    session_id: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, v: Optional[str]) -> Optional[str]:
        return _validate_session_id(v)

    @field_validator("memory_types")
    @classmethod
    def _check_memory_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        from mirix.schemas.retrieval_log import RETRIEVAL_MEMORY_TYPE_ORDER

        if v is None:
            return None
        if not v:
            raise ValueError(
                "memory_types must name at least one memory type; omit it to "
                "retrieve all of them"
            )
        unknown = [t for t in v if t not in RETRIEVAL_MEMORY_TYPE_ORDER]
        if unknown:
            raise ValueError(
                f"unknown memory_types {unknown}; allowed: "
                f"{list(RETRIEVAL_MEMORY_TYPE_ORDER)}"
            )
        return list(dict.fromkeys(v))


# Memory types whose retrieval uses the query (core blocks are returned whole).
_QUERY_MEMORY_TYPES = ("episodic", "semantic", "resource", "procedural", "knowledge_vault")


def _error_text(exc: BaseException) -> str:
    """An error as the retrieval response and log report it.

    A SQLAlchemy error's ``str()`` carries the SQL statement and its bind
    parameters (the query, user ids, vectors): only its type, the driver
    error's type and the first line of the driver's message are reported;
    the full text stays in the server log."""
    try:
        from sqlalchemy.exc import SQLAlchemyError, StatementError
    except ImportError:  # pragma: no cover - SQLAlchemy is a hard dependency
        SQLAlchemyError = StatementError = ()  # type: ignore[assignment]
    if isinstance(exc, SQLAlchemyError):
        orig = getattr(exc, "orig", None) if isinstance(exc, StatementError) else None
        if orig is not None:
            first_line = (str(orig).strip().splitlines() or [""])[0]
            return f"{type(exc).__name__} ({type(orig).__name__}): {first_line}"[:300]
        return f"{type(exc).__name__}: database error (see the server log)"
    return f"{type(exc).__name__}: {exc}"[:500]


def _query_without_nul(text: Optional[str]) -> str:
    """A search query with every U+0000 as U+FFFD, as MIRIX stores text.

    Postgres rejects U+0000 in a text parameter (``to_tsquery``, ``ILIKE`` …),
    so a query that carries one — the raw text of a conversation whose tool
    printed a NUL, used as the query when topic extraction fails, or a
    caller's query — failed every search, and failed the same way on every
    retry. The conversation store and the meta agent's input write U+0000 as
    U+FFFD too (``conversation_content.replace_nul``), and a Postgres column
    cannot hold it at all, so the query still matches what MIRIX stored; the
    replacement keeps the query's length."""
    from mirix.services.conversation_content import replace_nul

    cleaned, count = replace_nul(text or "")
    if count:
        logger.debug(
            "Replaced %d NUL character(s) (U+0000) with U+FFFD in a search query",
            count,
        )
    return cleaned


def _empty_memories(memory_types: List[str]) -> Dict[str, Any]:
    """What a retrieval returns for each requested type when it finds nothing."""
    empty: Dict[str, Any] = {}
    for memory_type in memory_types:
        if memory_type == "episodic":
            empty[memory_type] = {"total_count": 0, "recent": [], "relevant": []}
        elif memory_type == "core":
            empty[memory_type] = {"total_count": 0, "scopes": {}}
        else:
            empty[memory_type] = {"total_count": 0, "items": []}
    return empty


def _retrieval_agent(agents: List[AgentState]) -> AgentState:
    """The agent whose configs a retrieval uses (the query's embedding, topic
    extraction): the client's meta memory agent — the skills are embedded
    with its (curator's) embedding config — else its first agent."""
    for agent in agents:
        if agent.agent_type == AgentType.meta_memory_agent:
            return agent
    return agents[0]


def _all_requested_types_failed(
    memory_types: List[str], errors: Dict[str, str]
) -> Optional[str]:
    """The error of a retrieval none of whose requested types could be
    searched (their results are empty because they FAILED, not because
    nothing matched), else None."""
    if memory_types and all(t in errors for t in memory_types):
        return "every requested memory type failed: " + "; ".join(
            f"{t}: {errors[t]}" for t in memory_types
        )
    return None


def _episodic_item(event) -> Dict[str, Any]:
    return {
        "id": event.id,
        "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
        "summary": event.summary,
        "details": event.details,
        "retrieval_scores": None,
    }


async def retrieve_memories_by_keywords(
    server: AsyncServer,
    client: Client,
    user_id: str,
    agent_state: AgentState,
    key_words: str = "",
    limit: int = 10,
    filter_tags: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    *,
    memory_types: Optional[List[str]] = None,
    user: Optional[User] = None,
    trace: Optional[Dict[str, Any]] = None,
    errors: Optional[Dict[str, str]] = None,
    apply_skill_gate: bool = True,
    keyword_query: Optional[str] = None,
) -> dict:
    """
    Retrieve memories for ``key_words``, per memory type.

    Procedural memory is ranked by the skill retrieval pipeline
    (``ProceduralMemoryManager.retrieve_skills``: BM25 over the retrieval text,
    cosine, RRF, and — with ``apply_skill_gate``, the retrieval endpoints — the
    similarity gate, so it may return no skill). The other memory types
    keep their BM25 search. Only the types in ``memory_types`` (None =
    all) are retrieved: no other memory manager is called.

    Args:
        server: The Mirix server instance
        client: The authenticated client application (for authorization)
        user_id: The end-user ID whose memories to retrieve
        agent_state: Agent state (its embedding config embeds the skill query)
        key_words: Keywords to search for (empty string returns recent items;
            no skill)
        limit: Maximum number of items to retrieve per memory type
        filter_tags: Tag-based filtering (user_id + filter_tags = complete filter)
        use_cache: Control Redis cache behavior
        start_date: Optional start datetime for filtering episodic memories by occurred_at (inclusive)
        end_date: Optional end datetime for filtering episodic memories by occurred_at (inclusive)
        memory_types: The memory types to retrieve (None = all).
        user: The user (fetched here when not given).
        trace: Filled with each type's retrieval trace (the retrieval log's
            ``per_type``): for procedural memory the ranking diagnostics
            (every candidate's scores and ranks, the gate, the returned ids,
            the skills without a vector), for the others the method, field and
            returned ids.
        errors: Filled with ``{memory_type: error}`` for each type whose
            retrieval failed (its result is then empty).
        apply_skill_gate: Apply the similarity gate to skills.
        keyword_query: The query of the keyword-searched types (episodic,
            semantic, resource, knowledge vault) when it differs from the
            skills' (default ``key_words``): after a failed extraction the
            skills rank the long raw text, the others a short cut of it.

    Returns:
        Dictionary containing the requested memory types with their items;
        every item carries ``retrieval_scores`` ({bm25, cosine, rrf} for a
        ranked skill, None otherwise).
    """
    from mirix import constants as mirix_constants
    from mirix.schemas.retrieval_log import RETRIEVAL_MEMORY_TYPE_ORDER

    wanted = set(memory_types or RETRIEVAL_MEMORY_TYPE_ORDER)
    scopes = client.read_scopes
    search_method = "bm25"
    # No U+0000 reaches a search (Postgres rejects it in any text
    # parameter): the topics, the fallback's raw text and its keyword cut.
    key_words = _query_without_nul(key_words)
    skill_query = key_words
    if keyword_query is not None:
        key_words = _query_without_nul(keyword_query)
    keyword_method = "bm25" if key_words else "recent"

    def failed(memory_type: str, exc: BaseException) -> None:
        logger.exception("Error retrieving %s memories: %s", memory_type, exc)
        if trace is not None:
            trace[memory_type] = {"error": _error_text(exc)}
        if errors is not None:
            errors[memory_type] = _error_text(exc)

    # Log temporal filtering for monitoring
    if (start_date or end_date) and "episodic" in wanted:
        logger.info(
            "Temporal filtering enabled for episodic memories: start=%s, end=%s",
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )

    # Get timezone from user record (if exists)
    if user is None:
        try:
            user = await server.user_manager.get_user_by_id(user_id)
        except Exception:  # noqa: BLE001 — every type below then fails alone
            logger.warning("User %s not found for memory retrieval", user_id)
    timezone_str = getattr(user, "timezone", None) or "UTC"
    memories: Dict[str, Any] = {}

    # Get episodic memories (recent + relevant) with optional temporal filtering
    if "episodic" in wanted:
        try:
            episodic_manager = server.episodic_memory_manager

            # Get recent episodic memories with temporal filter
            recent_episodic = await episodic_manager.list_episodic_memory(
                agent_state=agent_state,  # Not accessed during BM25 search
                user=user,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,
                scopes=scopes,
                use_cache=use_cache,
                start_date=start_date,
                end_date=end_date,
            )

            # Get relevant episodic memories based on keywords with temporal filter
            relevant_episodic = []
            if key_words:
                relevant_episodic = await episodic_manager.list_episodic_memory(
                    agent_state=agent_state,  # Not accessed during BM25 search
                    user=user,
                    query=key_words,
                    search_field="details",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=filter_tags,
                    scopes=scopes,
                    start_date=start_date,
                    end_date=end_date,
                )

            memories["episodic"] = {
                "total_count": await episodic_manager.get_total_number_of_items(
                    user=user
                ),
                "recent": [_episodic_item(event) for event in recent_episodic],
                "relevant": [_episodic_item(event) for event in relevant_episodic],
            }
            if trace is not None:
                trace["episodic"] = {
                    "method": keyword_method,
                    "search_field": "details",
                    "limit": limit,
                    "recent_ids": [event.id for event in recent_episodic],
                    "returned_ids": [event.id for event in relevant_episodic],
                }
        except Exception as e:
            failed("episodic", e)
            memories["episodic"] = {"total_count": 0, "recent": [], "relevant": []}

    # Get semantic memories
    if "semantic" in wanted:
        try:
            semantic_manager = server.semantic_memory_manager

            semantic_items = await semantic_manager.list_semantic_items(
                agent_state=agent_state,  # Not accessed during BM25 search
                user=user,
                query=key_words,
                search_field="details",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,
                scopes=scopes,
                use_cache=use_cache,
            )

            memories["semantic"] = {
                "total_count": await semantic_manager.get_total_number_of_items(
                    user=user
                ),
                "items": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "summary": item.summary,
                        "details": item.details,
                        "retrieval_scores": None,
                    }
                    for item in semantic_items
                ],
            }
            if trace is not None:
                trace["semantic"] = {
                    "method": keyword_method,
                    "search_field": "details",
                    "limit": limit,
                    "returned_ids": [item.id for item in semantic_items],
                }
        except Exception as e:
            failed("semantic", e)
            memories["semantic"] = {"total_count": 0, "items": []}

    # Get resource memories
    if "resource" in wanted:
        try:
            resource_manager = server.resource_memory_manager

            resources = await resource_manager.list_resources(
                agent_state=agent_state,  # Not accessed during BM25 search
                user=user,
                query=key_words,
                search_field="summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,
                scopes=scopes,
                use_cache=use_cache,
            )

            memories["resource"] = {
                "total_count": await resource_manager.get_total_number_of_items(
                    user=user
                ),
                "items": [
                    {
                        "id": resource.id,
                        "title": resource.title,
                        "summary": resource.summary,
                        "resource_type": resource.resource_type,
                        "retrieval_scores": None,
                    }
                    for resource in resources
                ],
            }
            if trace is not None:
                trace["resource"] = {
                    "method": keyword_method,
                    "search_field": "summary",
                    "limit": limit,
                    "returned_ids": [resource.id for resource in resources],
                }
        except Exception as e:
            failed("resource", e)
            memories["resource"] = {"total_count": 0, "items": []}

    # Get procedural memories: the skill ranking, gated when injected.
    if "procedural" in wanted:
        try:
            procedural_manager = server.procedural_memory_manager

            ranked = await procedural_manager.retrieve_skills(
                skill_query,
                user,
                scopes,
                limit,
                apply_gate=apply_skill_gate,
                agent_state=agent_state,
                filter_tags=filter_tags or None,
                timezone_str=timezone_str,
            )

            memories["procedural"] = {
                "total_count": await procedural_manager.get_total_number_of_items(
                    user=user
                ),
                "items": [
                    _procedural_memory_response(
                        r.skill, retrieval_scores=r.scores.to_dict()
                    )
                    for r in ranked.skills
                ],
            }
            if trace is not None:
                # Bounded: the returned skills and the next few in
                # fused order, with the counts of what was left out.
                trace["procedural"] = await asyncio.to_thread(
                    ranked.diagnostics.to_dict,
                    max_candidates=mirix_constants.RETRIEVAL_LOG_TRACE_CANDIDATES,
                )
        except Exception as e:
            failed("procedural", e)
            memories["procedural"] = {"total_count": 0, "items": []}

    # Get knowledge vault items
    if "knowledge_vault" in wanted:
        try:
            knowledge_vault_manager = server.knowledge_vault_manager

            # Scoped like every other type: a client never reads a vault
            # entry of a scope it cannot read (nor one outside filter_tags).
            knowledge_items = await knowledge_vault_manager.list_knowledge(
                agent_state=agent_state,  # Not accessed during BM25 search
                user=user,
                query=key_words,
                search_field="caption",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,
                scopes=scopes,
                use_cache=use_cache,
            )

            memories["knowledge_vault"] = {
                "total_count": await knowledge_vault_manager.get_total_number_of_items(
                    user=user
                ),
                "items": [
                    {
                        "id": item.id,
                        "caption": item.caption,
                        "retrieval_scores": None,
                    }
                    for item in knowledge_items
                ],
            }
            if trace is not None:
                trace["knowledge_vault"] = {
                    "method": keyword_method,
                    "search_field": "caption",
                    "limit": limit,
                    "returned_ids": [item.id for item in knowledge_items],
                }
        except Exception as e:
            failed("knowledge_vault", e)
            memories["knowledge_vault"] = {"total_count": 0, "items": []}

    # Get core memory blocks (filtered by client's read_scopes, grouped by scope)
    if "core" in wanted:
        try:
            block_manager = server.block_manager

            blocks = await block_manager.get_blocks(
                user=user,
                any_scopes=client.read_scopes,
                auto_create_from_default=False,
            )

            # Group blocks by scope
            scopes_dict: Dict[str, list] = {}
            for block in blocks:
                scope = (block.filter_tags or {}).get("scope", "default")
                if scope not in scopes_dict:
                    scopes_dict[scope] = []
                scopes_dict[scope].append(
                    {
                        "id": block.id,
                        "label": block.label,
                        "value": block.value,
                        "retrieval_scores": None,
                    }
                )

            memories["core"] = {
                "total_count": len(blocks),
                "scopes": {
                    scope: {"items": items} for scope, items in scopes_dict.items()
                },
            }
            if trace is not None:
                trace["core"] = {
                    "method": "all_blocks",
                    "returned_ids": [block.id for block in blocks],
                }
        except Exception as e:
            failed("core", e)
            memories["core"] = {"total_count": 0, "scopes": {}}

    return memories


async def _retrieval_user(
    server: AsyncServer, user_id: str, client: Client
) -> Optional[User]:
    """The user whose memories are read, or None when the client's
    organization has no such user.

    A read for a user that does not exist (yet) finds nothing: the retrieval
    and search endpoints answer ``success: true`` with no memories — and no
    retrieval log row, no topic-extraction call — as they did before the
    tenant guard (an integration's first retrieval for a new user usually
    comes before its first ``add``, which creates the user).

    Tenant guard: a user of another organization is exactly the same None —
    a client never reads (or logs a query against) another tenant's memories,
    and a read is no user-id existence oracle.
    """
    from mirix.constants import DEFAULT_ORG_ID
    from mirix.services.conversation_message_manager import owner_org

    try:
        user = await server.user_manager.get_user_by_id(user_id)
    except (NoResultFound, LookupError):
        return None
    if user is None or (user.organization_id or DEFAULT_ORG_ID) != owner_org(client):
        return None
    return user


async def _tenant_user_for_write(
    server: AsyncServer, user_id: str, client: Client
) -> Optional[User]:
    """For an endpoint that WRITES for ``user_id``: the user when it exists
    in the client's organization, None when it does not exist at all; 404
    when it belongs to another organization — a client never writes
    sessions, learning runs or skills for another tenant's user."""
    from mirix.constants import DEFAULT_ORG_ID
    from mirix.services.conversation_message_manager import owner_org

    try:
        user = await server.user_manager.get_user_by_id(user_id)
    except (NoResultFound, LookupError):
        return None
    if user is None:
        return None
    if (user.organization_id or DEFAULT_ORG_ID) != owner_org(client):
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


async def _extract_retrieval_topics(
    parsed, request: RetrieveMemoryRequest, llm_config: LLMConfig, memory_types
):
    """Topic extraction for a conversation retrieval.

    ``skipped`` when no requested memory type searches by query (core only);
    the local model first when one is named (text only, falling back to the
    configured LLM); otherwise the configured LLM with text and images.
    """
    from mirix import constants as mirix_constants
    from mirix.services.conversation_content import iter_images
    from mirix.services.topic_extraction import (
        EXTRACTION_STATUS_OK,
        EXTRACTION_STATUS_SKIPPED,
        TopicExtraction,
        extract_topics,
        fallback_query,
        fallback_query_text,
    )

    if not any(t in _QUERY_MEMORY_TYPES for t in memory_types):
        return TopicExtraction(
            topics=None, temporal_expression=None, status=EXTRACTION_STATUS_SKIPPED
        )
    local_note: Dict[str, Any] = {}
    if request.local_model_for_retrieval:
        if iter_images(parsed):
            # Extraction sees the images; a local text-only model cannot.
            logger.info(
                "Local topic extraction skipped for model %s: the conversation "
                "has images; using the configured LLM",
                request.local_model_for_retrieval,
            )
            local_note = {
                "local_model": request.local_model_for_retrieval,
                "local_model_skipped": "images",
            }
        else:
            # The local model's input keeps its head and tail.
            conversation, cut = fallback_query(
                fallback_query_text(parsed, labeled=True),
                mirix_constants.RETRIEVAL_LOCAL_EXTRACTION_MAX_CHARS,
            )
            topics = await extract_topics_with_local_model(
                messages=request.messages,
                model_name=request.local_model_for_retrieval,
                conversation=conversation,
            )
            # Note: Local model extraction doesn't support temporal extraction yet
            if topics:
                return TopicExtraction(
                    topics=topics,
                    temporal_expression=None,
                    status=EXTRACTION_STATUS_OK,
                    usage={
                        "model": request.local_model_for_retrieval,
                        "calls": 1,
                        "local": True,
                    },
                    input_stats={"text_chars_cut": cut} if cut else {},
                )
            logger.warning(
                "Local topic extraction failed for model %s, falling back to default LLM",
                request.local_model_for_retrieval,
            )
            local_note = {
                "local_model": request.local_model_for_retrieval,
                "local_model_failed": True,
            }
    extraction = await extract_topics(parsed, llm_config)
    if local_note:
        extraction.input_stats = {**extraction.input_stats, **local_note}
    return extraction


def _request_datetime(value: str) -> datetime:
    """A request's ISO 8601 date or time as naive UTC, the form the episodic
    ``occurred_at`` column stores: a value with an offset (``Z``
    included) is converted to UTC before its tzinfo is dropped; one without
    is taken as UTC, as before. A value whose UTC time falls outside the
    datetime range (a min / max sentinel with an offset, e.g.
    ``0001-01-01T00:00:00+01:00``) is that end of the range: it bounds
    nothing, as it meant. ValueError when it is not ISO 8601."""
    from mirix.utils import to_naive_utc

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    try:
        return to_naive_utc(parsed)
    except OverflowError:
        offset = parsed.utcoffset()
        # UTC = local - offset: a positive offset overflows below the range.
        return (
            datetime.min
            if offset is not None and offset > timedelta(0)
            else datetime.max
        )


def _wall_clock_to_naive_utc(value: datetime, zone) -> datetime:
    """``value``'s wall-clock time in ``zone`` (a pytz zone) as naive UTC:
    localized again, so each bound gets its own day's UTC offset (a window
    across a daylight saving change keeps local midnights)."""
    from mirix.utils import to_naive_utc

    return to_naive_utc(zone.localize(value.replace(tzinfo=None)))


def _retrieval_date_range(
    request_start: Optional[str],
    request_end: Optional[str],
    temporal_expr: Optional[str],
    user: Optional[User],
    now: Optional[datetime] = None,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """The episodic date range of a conversation retrieval, as naive UTC.

    Explicit request dates win over the extracted temporal expression (an
    invalid one is logged and leaves its bound open). A temporal expression
    is resolved in the user's time zone (``user.timezone``: MIRIX's
    ``"Region (UTC+hh:mm)"`` form or a zone name; UTC when missing or
    unknown) at ``now`` (default: the current time), and its bounds are
    converted to UTC. An expression naming an absolute date gives no
    range (``parse_temporal_expression``).
    """
    from datetime import timezone as dt_timezone

    from mirix.utils import user_timezone

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    if request_start or request_end:
        # Use explicit date range from request
        try:
            if request_start:
                start_date = _request_datetime(request_start)
            if request_end:
                end_date = _request_datetime(request_end)
            logger.debug("Using explicit date range: %s to %s", start_date, end_date)
        except ValueError as e:
            logger.warning("Invalid date format in request: %s", e)
    elif temporal_expr:
        # Parse LLM-extracted temporal expression
        from mirix.temporal.temporal_parser import parse_temporal_expression

        # "today" is the user's today: the current time in the user's zone.
        zone = user_timezone(getattr(user, "timezone", None))
        current = now or datetime.now(dt_timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt_timezone.utc)
        reference_time = current.astimezone(zone)

        try:
            temporal_range = parse_temporal_expression(temporal_expr, reference_time)
            if temporal_range:
                if temporal_range.start is not None:
                    start_date = _wall_clock_to_naive_utc(temporal_range.start, zone)
                if temporal_range.end is not None:
                    end_date = _wall_clock_to_naive_utc(temporal_range.end, zone)
        except (OverflowError, ValueError) as e:
            # "last 1000000 days" reaches past the datetime range: no filter
            # is the safe degrade, never an error of the request.
            logger.warning(
                "Temporal expression %r gives no date range: %s", temporal_expr, e
            )
            return None, None
        if temporal_range:
            logger.info(
                "Parsed temporal expression '%s' in time zone %s to range: %s to %s "
                "(naive UTC, as occurred_at stores it)",
                temporal_expr,
                zone.zone,
                start_date,
                end_date,
            )
    return start_date, end_date


async def _write_retrieval_log(
    server: AsyncServer,
    *,
    organization_id: str,
    user_id: str,
    client_id: Optional[str],
    session_id: Optional[str],
    memory_types: List[str],
    query_text: Optional[str],
    extraction,
    trace: Dict[str, Any],
    latency_ms: int,
    recorder,
    chunk_counter,
    fallback_query_truncated_chars: int = 0,
    started_monotonic: Optional[float] = None,
) -> Optional[str]:
    """Write the retrieval's log row; its id, or None when the log
    is off (``MIRIX_RETRIEVAL_LOG_ENABLED=false``), the user's data was
    erased while the retrieval ran (``started_monotonic``: its start on
    ``time.monotonic()``), or the write failed (the retrieval itself is still
    answered)."""
    from mirix.services.retrieval_log_manager import (
        RetrievalLogManager,
        retrieval_log_enabled,
    )

    if not retrieval_log_enabled():
        return None
    entry = extraction.log_entry()
    if fallback_query_truncated_chars:
        entry["fallback_query_truncated_chars"] = fallback_query_truncated_chars
    procedural = trace.get("procedural") if isinstance(trace, dict) else None
    embedding_status = (
        procedural.get("embedding_status")
        if isinstance(procedural, dict) and "embedding_status" in procedural
        else ("skipped" if "procedural" not in memory_types else None)
    )
    manager = getattr(server, "retrieval_log_manager", None) or RetrievalLogManager()
    try:
        return await manager.create_log(
            organization_id=organization_id,
            user_id=user_id,
            client_id=client_id,
            session_id=session_id,
            memory_types=memory_types,
            query_text=query_text,
            extracted_topics=extraction.topics,
            extraction_status=extraction.status,
            embedding_status=embedding_status,
            per_type=trace,
            llm_usage={"topic_extraction": entry},
            latency_ms=latency_ms,
            retry_events=recorder.events,
            embedding_chunked_calls=chunk_counter.chunked_calls,
            started_monotonic=started_monotonic,
        )
    except Exception:  # noqa: BLE001 — the log must not fail the retrieval
        logger.exception("Could not write the retrieval log row")
        return None


@router.post("/memory/retrieve/conversation")
@with_langfuse_tracing
async def retrieve_memory_with_conversation(
    request: RetrieveMemoryRequest,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on conversation context.

    1. The messages are parsed (``content``: string, list of parts or null;
       images kept) — no text and no image is a 422.
    2. An LLM extracts the topics and a temporal expression from the text AND
       the images, within the model's input budget (head and tail kept,
       images whole). If that fails, the conversation's own text is the
       query (``extraction_status: "fallback"``).
    3. Each requested memory type (``memory_types``, default all) is searched;
       skills with the skill retrieval pipeline and its similarity gate, so possibly none.
       Skills rank the task-level part of the topics (``procedural_query``:
       the goal and the operations) when the extraction gave one; the
       other types search all the topics.
    4. One retrieval log row records the whole retrieval; its id is returned
       as ``retrieval_id`` (null when the log is disabled).

    Every returned item carries ``retrieval_scores`` ({bm25, cosine, rrf} for
    a skill, null for the other types). A type whose search failed is listed
    in ``errors`` (its result is empty); when EVERY requested type failed the
    answer is ``success: false`` with ``error`` — never mistaken for "nothing
    matched". A user the client's organization does not have finds nothing:
    ``success: true``, empty memories, no log row.
    """
    from mirix import constants as mirix_constants
    from mirix.embedding_guard import EmbeddingChunkCounter, counting_embedding_chunks
    from mirix.observability.context import conversation_session_scope
    from mirix.schemas.retrieval_log import RETRIEVAL_MEMORY_TYPE_ORDER
    from mirix.services.conversation_content import MessageParseError, iter_images
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.image_store import prepare_images
    from mirix.services.retry_policy import RetryEventRecorder, recording_retries
    from mirix.services.topic_extraction import (
        EXTRACTION_STATUS_FALLBACK,
        EXTRACTION_STATUS_OK,
        EXTRACTION_STATUS_SKIPPED,
        fallback_query,
        fallback_query_text,
        normalize_retrieval_messages,
        plain_text,
    )

    started = time.perf_counter()
    started_mono = time.monotonic()  # the erasure guard's clock (retrieval log)
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    # If user_id is not provided, use the admin user for this client
    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    filter_tags = dict(request.filter_tags) if request.filter_tags is not None else {}
    memory_types = list(request.memory_types or RETRIEVAL_MEMORY_TYPE_ORDER)
    organization_id = owner_org(client)

    # Parse the messages (string, parts or null content), decode
    # and check the inline images, and check any stored file they reference.
    try:
        parsed = normalize_retrieval_messages(request.messages)
        await prepare_images(iter_images(parsed))
    except MessageParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await _check_file_references(parsed, organization_id, user_id)

    # Get all agents for this client (automatically filtered by client via apply_access_predicate)
    all_agents = await server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topics": None,
            "memories": {},
        }
    agent_state = _retrieval_agent(all_agents)
    user = await _retrieval_user(server, user_id, client)
    if user is None:
        # Nothing to find (and no user to log the query against).
        return {
            "success": True,
            "retrieval_id": None,
            "topics": None,
            "procedural_query": None,
            "temporal_expression": None,
            "extraction_status": EXTRACTION_STATUS_SKIPPED,
            "date_range": None,
            "memories": _empty_memories(memory_types),
            "errors": {},
        }

    recorder = RetryEventRecorder()
    chunk_counter = EmbeddingChunkCounter()
    trace: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    query_text = plain_text(parsed)
    fallback_cut = 0
    keyword_query: Optional[str] = None
    # The topic extraction and the query embedding serve the request's session
    # (sent as MIRIX_LLM_SESSION_HEADER when that is set; off by default).
    with (
        conversation_session_scope(request.session_id),
        recording_retries(recorder, stage="retrieval"),
        counting_embedding_chunks(chunk_counter),
    ):
        extraction = await _extract_retrieval_topics(
            parsed, request, agent_state.llm_config, memory_types
        )
        if extraction.status == EXTRACTION_STATUS_OK:
            # Skills rank the task-level part of the topics (goal and
            # operations: a skill's retrieval text describes a kind of task);
            # the keyword-searched types search all of them (the instance's
            # entities and values are what a fact or an event matches).
            key_words = extraction.procedural_query or extraction.topics or ""
            if extraction.topics and extraction.topics != key_words:
                keyword_query = extraction.topics
        elif extraction.status == EXTRACTION_STATUS_FALLBACK:
            # Extraction failed -> the conversation's own text, without
            # the role labels (they would be search terms). Skills rank the
            # long text (BM25 on its first terms, the cosine on all of it);
            # the keyword-searched types get a short head-and-tail cut.
            raw_query = fallback_query_text(parsed)
            key_words, fallback_cut = fallback_query(
                raw_query, mirix_constants.RETRIEVAL_FALLBACK_QUERY_MAX_CHARS
            )
            keyword_query, _ = fallback_query(
                raw_query, mirix_constants.RETRIEVAL_FALLBACK_KEYWORD_QUERY_MAX_CHARS
            )
        else:
            key_words = ""
        logger.debug(
            "Retrieval query (%s): %s", extraction.status, key_words[:200]
        )

        start_date, end_date = _retrieval_date_range(
            request.start_date,
            request.end_date,
            extraction.temporal_expression,
            user,
        )

        # Retrieve memories with temporal filtering
        memories = await retrieve_memories_by_keywords(
            server=server,
            client=client,
            user_id=user_id,
            agent_state=agent_state,
            key_words=key_words,
            limit=request.limit,
            filter_tags=filter_tags,
            use_cache=request.use_cache,
            start_date=start_date,
            end_date=end_date,
            memory_types=memory_types,
            user=user,
            trace=trace,
            errors=errors,
            apply_skill_gate=True,
            keyword_query=keyword_query,
        )

    retrieval_id = await _write_retrieval_log(
        server,
        organization_id=organization_id,
        user_id=user.id,
        client_id=client.id,
        session_id=request.session_id,
        memory_types=memory_types,
        query_text=query_text,
        extraction=extraction,
        trace=trace,
        latency_ms=int((time.perf_counter() - started) * 1000),
        recorder=recorder,
        chunk_counter=chunk_counter,
        fallback_query_truncated_chars=fallback_cut,
        started_monotonic=started_mono,
    )

    failure = _all_requested_types_failed(memory_types, errors)
    return {
        "success": failure is None,
        **({"error": failure} if failure else {}),
        "retrieval_id": retrieval_id,
        "topics": extraction.topics,
        "procedural_query": extraction.procedural_query,
        "temporal_expression": extraction.temporal_expression,
        "extraction_status": extraction.status,
        "date_range": (
            {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            }
            if (start_date or end_date)
            else None
        ),
        "memories": memories,
        "errors": errors,
    }


@router.get("/memory/retrieve/topic")
@with_langfuse_tracing
async def retrieve_memory_with_topic(
    user_id: Optional[str] = None,
    topic: str = "",
    limit: int = Query(10, ge=0),
    filter_tags: Optional[str] = None,
    use_cache: bool = True,
    x_client_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on a topic (the topic is the query; no
    extraction). Skills are ranked by the skill retrieval pipeline with its gate; the
    other types by BM25. One retrieval log row is written (``retrieval_id``).
    Every type failing is ``success: false`` with ``error``; a user the
    client's organization does not have finds nothing (``success: true``).

    Args:
        user_id: The user ID to retrieve memories for (uses admin user if not provided)
        topic: The topic/keywords to search for
        limit: Maximum number of items to retrieve per memory type (default: 10)
        filter_tags: Optional JSON string of tags to filter memories (default: None)
        use_cache: Whether to use cached results (default: True)
    """
    from mirix.embedding_guard import EmbeddingChunkCounter, counting_embedding_chunks
    from mirix.schemas.retrieval_log import RETRIEVAL_MEMORY_TYPE_ORDER
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.retry_policy import RetryEventRecorder, recording_retries
    from mirix.services.topic_extraction import (
        EXTRACTION_STATUS_SKIPPED,
        TopicExtraction,
    )

    started = time.perf_counter()
    started_mono = time.monotonic()  # the erasure guard's clock (retrieval log)
    server = get_server()
    client_id, org_id = await get_client_and_org(x_client_id, x_org_id)
    client = await server.client_manager.get_client_by_id(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

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

    if parsed_filter_tags is None:
        parsed_filter_tags = {}

    # Get all agents for this client (automatically filtered by client via apply_access_predicate)
    all_agents = await server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topic": topic,
            "memories": {},
        }
    memory_types = list(RETRIEVAL_MEMORY_TYPE_ORDER)
    user = await _retrieval_user(server, user_id, client)
    if user is None:
        return {
            "success": True,
            "retrieval_id": None,
            "topic": topic,
            "memories": _empty_memories(memory_types),
            "errors": {},
        }

    recorder = RetryEventRecorder()
    chunk_counter = EmbeddingChunkCounter()
    trace: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    with recording_retries(recorder, stage="retrieval"), counting_embedding_chunks(
        chunk_counter
    ):
        # Retrieve memories using the helper function
        memories = await retrieve_memories_by_keywords(
            server=server,
            client=client,
            user_id=user_id,
            agent_state=_retrieval_agent(all_agents),
            key_words=topic,
            limit=limit,
            filter_tags=parsed_filter_tags,
            use_cache=use_cache,
            memory_types=memory_types,
            user=user,
            trace=trace,
            errors=errors,
            apply_skill_gate=True,
        )

    retrieval_id = await _write_retrieval_log(
        server,
        organization_id=owner_org(client),
        user_id=user.id,
        client_id=client.id,
        session_id=None,
        memory_types=memory_types,
        query_text=topic,
        extraction=TopicExtraction(
            topics=None, temporal_expression=None, status=EXTRACTION_STATUS_SKIPPED
        ),
        trace=trace,
        latency_ms=int((time.perf_counter() - started) * 1000),
        recorder=recorder,
        chunk_counter=chunk_counter,
        started_monotonic=started_mono,
    )

    failure = _all_requested_types_failed(memory_types, errors)
    return {
        "success": failure is None,
        **({"error": failure} if failure else {}),
        "retrieval_id": retrieval_id,
        "topic": topic,
        "memories": memories,
        "errors": errors,
    }


async def _precompute_embedding_for_search(
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

    Raises whatever the embedding call raised; ``/memory/search`` then falls
    back to keyword search (the "no query vector" rule).
    """
    if search_method != "embedding" or not query:
        return None, None

    import numpy as np

    from mirix.constants import MAX_EMBEDDING_DIM
    from mirix.embeddings import embedding_model

    # Compute embedding once
    embedded_text = await (
        await embedding_model(agent_state.embedding_config)
    ).get_text_embedding(query)

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
    search_method: str = "",  # "" = per-memory-type default (see resolution below)
    limit: int = 10,
    authorization: Optional[str] = Header(None),
    filter_tags: Optional[str] = Query(None),
    similarity_threshold: Optional[float] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    include_core_memory: bool = Query(
        False, description="When True, include core (block) memory in the response."
    ),
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
                     - procedural: "description", "instructions"
                     - knowledge_vault: "caption", "secret_value"
                     - semantic: "name", "summary", "details"
                     - For "all": use "null" (default)
        search_method: Search method. Options: "bm25", "embedding", and (procedural
                    only) "hybrid" (BM25 + embedding fused via Reciprocal Rank Fusion).
                    When omitted, defaults per memory type: procedural -> "hybrid"
                    (override via MIRIX_SKILL_SEARCH_METHOD), all others -> "embedding".
                    "hybrid" requested for a non-procedural type falls back to embedding.
                    Procedural memory (also within "all") ranks a hybrid search, and a
                    bm25/embedding search on the default field, with the skill retrieval
                    pipeline over the skill's retrieval text (name, triggers,
                    description): true BM25, cosine and RRF, WITHOUT the retrieval
                    gate — the top `limit` skills, each with
                    `retrieval_scores` {bm25, cosine, rrf}.
        limit: Maximum number of results per memory type (default: 10)
        filter_tags: Optional JSON string of filter tags (scope added automatically)
        similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0).
                             Only results with cosine distance < threshold are returned.
                             Only applies when search_method="embedding"
        start_date: Optional start date/time for episodic memory filtering (ISO 8601 format)
        end_date: Optional end date/time for episodic memory filtering (ISO 8601 format)
    """
    query = _query_without_nul(query)  # Postgres rejects U+0000
    server = get_server()

    client = None

    # Support both dashboard JWTs and programmatic API key access
    if authorization:
        try:
            client, _ = await get_client_from_jwt_or_api_key(authorization)
        except HTTPException:
            # If JWT/API key auth fails, fall through to use x_client_id
            pass

    # Fallback to use the client_id and org_id passed in the method parameters.
    if not client and x_client_id:
        client_id = x_client_id
        client = await server.client_manager.get_client_by_id(client_id)
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
    all_agents = await server.agent_manager.list_agents(actor=client, limit=1000)

    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this client",
            "query": query,
            "results": [],
            "count": 0,
        }

    agent_state = _retrieval_agent(all_agents)

    # The user, in the client's organization only (tenant guard): a user
    # the organization does not have — missing, or another tenant's — finds
    # nothing, and no manager is asked for another organization's memories.
    user = await _retrieval_user(server, user_id, client)
    timezone_str = (user.timezone if user is not None else None) or "UTC"

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

    if parsed_filter_tags is None:
        parsed_filter_tags = {}
    scopes = client.read_scopes

    # Parse temporal filtering parameters
    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    # Naive UTC for the DB comparison (it stores naive UTC): an offset is
    # converted first, never just dropped.
    if start_date:
        try:
            parsed_start_date = _request_datetime(start_date)
        except ValueError as e:
            logger.warning("Invalid start_date format: %s", e)

    if end_date:
        try:
            parsed_end_date = _request_datetime(end_date)
        except ValueError as e:
            logger.warning("Invalid end_date format: %s", e)

    # Resolve the search method per memory type. FastAPI passes "" when the
    # caller omits the query param, so "" is the "use the default" sentinel:
    #   * procedural -> EverOS-aligned hybrid (BM25 + embedding fused via RRF),
    #     env-overridable via MIRIX_SKILL_SEARCH_METHOD;
    #   * every other type (incl. the cross-type "all" sweep) -> embedding.
    from mirix.constants import PROCEDURAL_DEFAULT_SEARCH_METHOD

    from mirix.services.procedural_memory_manager import (
        effective_procedural_search_method,
    )

    # Procedural memory resolves its own method, also inside the "all" sweep:
    # the caller's explicit choice, else PROCEDURAL_DEFAULT_SEARCH_METHOD
    # ("hybrid": the skill ranking, never gated here). Procedural
    # `instructions` has no vector, so an embedding or hybrid search on it
    # runs — and is reported in the response's search_method — as bm25.
    procedural_search_method = effective_procedural_search_method(
        search_method or PROCEDURAL_DEFAULT_SEARCH_METHOD,
        # The "all" sweep searches every type's default field.
        "" if search_field == "null" or memory_type == "all" else search_field,
    )
    if not search_method:
        search_method = (
            PROCEDURAL_DEFAULT_SEARCH_METHOD
            if memory_type == "procedural"
            else "embedding"
        )
    # "hybrid" is procedural-only — the other managers have no hybrid branch.
    # For any non-procedural memory_type (including "all") fall back to
    # embedding so those managers receive a method they support.
    if search_method == "hybrid" and memory_type != "procedural":
        search_method = "embedding"
    if memory_type == "procedural":
        search_method = procedural_search_method

    # Validate search parameters
    if (
        memory_type == "resource"
        and search_field == "content"
        and search_method == "embedding"
    ):
        return {
            "success": False,
            "error": "embedding is not supported for resource memory's 'content' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if (
        memory_type == "knowledge_vault"
        and search_field == "secret_value"
        and search_method == "embedding"
    ):
        return {
            "success": False,
            "error": "embedding is not supported for knowledge_vault memory's 'secret_value' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if memory_type == "all":
        search_field = "null"

    if user is None:
        return {
            "success": True,
            "query": query,
            "memory_type": memory_type,
            "search_field": search_field,
            "search_method": search_method,
            "date_range": None,
            "results": [],
            "count": 0,
        }

    # Pre-compute embedding once if using embedding search (to avoid redundant embeddings)
    warnings: List[str] = []
    try:
        embedded_text, embedded_text_padded = await _precompute_embedding_for_search(
            search_method, query, agent_state
        )
    except Exception as exc:  # noqa: BLE001 — fall back to keywords
        # No query vector (an embedding outage): every type searches by
        # keywords instead of the whole request failing with a 500 — the
        # procedural ranking by BM25 alone, like M7.3's own fallback.
        logger.warning(
            "/memory/search: the query embedding failed (%s); searching by bm25",
            _error_text(exc),
        )
        embedded_text = embedded_text_padded = None
        search_method = "bm25"
        if procedural_search_method in ("embedding", "hybrid"):
            procedural_search_method = "bm25"
        warnings.append(
            f"the query embedding failed ({_error_text(exc)}); searched by bm25"
        )

    # Collect results from requested memory types
    all_results = []
    # For a single procedural search, surface the user's total procedural count
    # alongside the page-sized result set.
    procedural_total_count = None

    # If searching all memory types, run searches concurrently for better performance
    if memory_type == "all":
        import asyncio

        # Define async wrappers for each manager call
        async def search_episodic():
            try:
                memories = await server.episodic_memory_manager.list_episodic_memory(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "details",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "occurred_at": (
                            x.occurred_at.isoformat() if x.occurred_at else None
                        ),
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
                memories = await server.resource_memory_manager.list_resources(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=(
                        search_field
                        if search_field != "null"
                        else ("summary" if search_method == "embedding" else "content")
                    ),
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
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
                return await _search_procedural_memories(
                    server.procedural_memory_manager,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    search_method=procedural_search_method,
                    search_field=search_field,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                    embedded_text=embedded_text,
                )
            except Exception as e:
                logger.error("Error searching procedural memories: %s", e)
                return []

        async def search_knowledge():
            try:
                memories = await server.knowledge_vault_manager.list_knowledge(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
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
                memories = await server.semantic_memory_manager.list_semantic_items(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "details",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
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

        async def search_core():
            try:
                blocks = await server.block_manager.get_blocks(
                    user=user,
                    any_scopes=client.read_scopes,
                    auto_create_from_default=False,
                    limit=limit or 50,
                )
                return [
                    {
                        "memory_type": "core",
                        "id": block.id,
                        "user_id": block.user_id,
                        "label": block.label,
                        "value": block.value,
                        "scope": (block.filter_tags or {}).get("scope", "default"),
                    }
                    for block in blocks
                ]
            except Exception as e:
                logger.error(
                    "Error retrieving core memory blocks: %s", e, exc_info=True
                )
                return []

        # Run all searches concurrently
        tasks = [
            search_episodic(),
            search_resource(),
            search_procedural(),
            search_knowledge(),
            search_semantic(),
        ]
        if include_core_memory:
            tasks.append(search_core())
        results = await asyncio.gather(*tasks)

        # Flatten results
        for result_list in results:
            all_results.extend(result_list)

    # Single memory type searches (run serially)
    elif memory_type == "episodic":
        try:
            episodic_memories = (
                await server.episodic_memory_manager.list_episodic_memory(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    similarity_threshold=similarity_threshold,
                )
            )
            all_results.extend(
                [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "timestamp": (
                            x.occurred_at.isoformat() if x.occurred_at else None
                        ),
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
            resource_memories = await server.resource_memory_manager.list_resources(
                agent_state=agent_state,
                user=user,
                query=query,
                embedded_text=(
                    embedded_text if search_method == "embedding" and query else None
                ),
                search_field=(
                    search_field
                    if search_field != "null"
                    else ("summary" if search_method == "embedding" else "content")
                ),
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=parsed_filter_tags,
                scopes=scopes,
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
                        "content": (
                            x.content[:200] if x.content else None
                        ),  # Truncate content for response
                    }
                    for x in resource_memories
                ]
            )
        except Exception as e:
            logger.error("Error searching resource memories: %s", e)

    # Search procedural memories
    elif memory_type == "procedural":
        try:
            all_results.extend(
                await _search_procedural_memories(
                    server.procedural_memory_manager,
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    search_method=procedural_search_method,
                    search_field=search_field,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                    embedded_text=embedded_text,
                )
            )
            # Report the user's total procedural count; the page's `count` is
            # only len(results), not the global total.
            procedural_total_count = (
                await server.procedural_memory_manager.get_total_number_of_items(
                    user=user
                )
            )
        except Exception as e:
            logger.error("Error searching procedural memories: %s", e)

    # Search knowledge vault
    elif memory_type == "knowledge_vault":
        try:
            knowledge_vault_memories = (
                await server.knowledge_vault_manager.list_knowledge(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
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
            semantic_memories = (
                await server.semantic_memory_manager.list_semantic_items(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
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

    # For single memory type searches, fetch core memory sequentially if requested
    if include_core_memory and memory_type != "all":
        try:
            blocks = await server.block_manager.get_blocks(
                user=user,
                any_scopes=client.read_scopes,
                auto_create_from_default=False,
                limit=limit or 50,
            )
            for block in blocks:
                all_results.append(
                    {
                        "memory_type": "core",
                        "id": block.id,
                        "user_id": block.user_id,
                        "label": block.label,
                        "value": block.value,
                        "scope": (block.filter_tags or {}).get("scope", "default"),
                    }
                )
        except Exception as e:
            logger.error(
                "Error retrieving core memory blocks for single-user search: %s",
                e,
                exc_info=True,
            )

    response = {
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
    if warnings:
        response["warnings"] = warnings
    # A single procedural search also reports the global per-user procedural
    # total alongside the page count.
    if procedural_total_count is not None:
        response["total_count"] = procedural_total_count
    return response


@router.get("/memory/search_all_users")
@with_langfuse_tracing
async def search_memory_all_users(
    query: str,
    memory_type: str = "all",
    search_field: str = "null",
    search_method: str = "",  # "" = per-memory-type default (see resolution below)
    limit: int = 10,
    client_id: Optional[str] = Query(None),
    org_id: Optional[str] = Query(None),
    filter_tags: Optional[str] = Query(None),
    include_core_memory: bool = Query(
        False,
        description="When True, include a 'core' section with block memory in the response.",
    ),
    block_filter_tags: Optional[str] = Query(
        None,
        description="Optional JSON string of filter tags applied when including core memory; only blocks whose filter_tags contain these are returned. Scope is auto-injected.",
    ),
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
        include_core_memory: When True, include core (block) memory in the response.
        block_filter_tags: Optional JSON string; when including core memory, only blocks
                          whose filter_tags contain these keys/values. Scope is auto-injected.
        similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0)
        start_date: Optional start date/time for episodic memory filtering (ISO 8601 format)
        end_date: Optional end date/time for episodic memory filtering (ISO 8601 format)
    """
    import json

    query = _query_without_nul(query)  # Postgres rejects U+0000
    server = get_server()

    # Determine which client to use and which org to search
    if client_id:
        # Use the provided client_id - fetch its org_id
        effective_client_id = client_id
        client = await server.client_manager.get_client_by_id(effective_client_id)
        effective_org_id = client.organization_id  # Use CLIENT's org_id
        logger.info(
            "Using provided client_id=%s with its organization_id=%s",
            effective_client_id,
            effective_org_id,
        )
    else:
        # Fall back to headers
        effective_client_id, header_org_id = await get_client_and_org(
            x_client_id, x_org_id
        )
        client = await server.client_manager.get_client_by_id(effective_client_id)
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
            raise HTTPException(
                status_code=400, detail="Invalid filter_tags JSON format"
            )
    else:
        filter_tags_dict = {}

    scopes = client.read_scopes

    logger.info(
        "Cross-user search: client=%s, org=%s, scopes=%s, filter_tags=%s, similarity_threshold=%s, include_core_memory=%s, block_filter_tags=%s",
        effective_client_id,
        effective_org_id,
        scopes,
        filter_tags_dict,
        similarity_threshold,
        include_core_memory,
        block_filter_tags,
    )

    # Parse temporal filtering parameters
    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    # Naive UTC for the DB comparison (it stores naive UTC): an offset is
    # converted first, never just dropped.
    if start_date:
        try:
            parsed_start_date = _request_datetime(start_date)
        except ValueError as e:
            logger.warning("Invalid start_date format: %s", e)

    if end_date:
        try:
            parsed_end_date = _request_datetime(end_date)
        except ValueError as e:
            logger.warning("Invalid end_date format: %s", e)

    # Get agents for this client
    all_agents = await server.agent_manager.list_agents(actor=client, limit=1000)
    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this client",
            "query": query,
            "results": [],
            "count": 0,
        }

    agent_state = _retrieval_agent(all_agents)

    # Resolve the search method per memory type. FastAPI passes "" when the
    # caller omits the query param, so "" is the "use the default" sentinel:
    #   * procedural -> EverOS-aligned hybrid (BM25 + embedding fused via RRF),
    #     env-overridable via MIRIX_SKILL_SEARCH_METHOD;
    #   * every other type (incl. the cross-type "all" sweep) -> embedding.
    from mirix.constants import PROCEDURAL_DEFAULT_SEARCH_METHOD

    if not search_method:
        search_method = (
            PROCEDURAL_DEFAULT_SEARCH_METHOD
            if memory_type == "procedural"
            else "embedding"
        )
    # "hybrid" is procedural-only — the other managers have no hybrid branch.
    # For any non-procedural memory_type (including "all") fall back to
    # embedding so those managers receive a method they support.
    if search_method == "hybrid" and memory_type != "procedural":
        search_method = "embedding"
    # Procedural `instructions` has no vector, so an embedding search on
    # it runs — and is reported in the response's search_method — as bm25.
    if memory_type == "procedural":
        from mirix.services.procedural_memory_manager import (
            effective_procedural_search_method,
        )

        search_method = effective_procedural_search_method(search_method, search_field)

    # Validate search parameters
    if (
        memory_type == "resource"
        and search_field == "content"
        and search_method == "embedding"
    ):
        return {
            "success": False,
            "error": "embedding is not supported for resource memory's 'content' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if (
        memory_type == "knowledge_vault"
        and search_field == "secret_value"
        and search_method == "embedding"
    ):
        return {
            "success": False,
            "error": "embedding is not supported for knowledge_vault memory's 'secret_value' field.",
            "query": query,
            "results": [],
            "count": 0,
        }

    if memory_type == "all":
        search_field = "null"

    # Pre-compute embedding once if using embedding search (to avoid redundant
    # embeddings). An embedding outage searches by keywords instead of failing
    # the request with a 500 (as /memory/search).
    warnings: List[str] = []
    try:
        embedded_text, embedded_text_padded = await _precompute_embedding_for_search(
            search_method, query, agent_state
        )
    except Exception as exc:  # noqa: BLE001 — fall back to keywords
        logger.warning(
            "/memory/search_all_users: the query embedding failed (%s); "
            "searching by bm25",
            _error_text(exc),
        )
        embedded_text = embedded_text_padded = None
        search_method = "bm25"
        warnings.append(
            f"the query embedding failed ({_error_text(exc)}); searched by bm25"
        )

    # Collect results using organization_id filter
    all_results = []

    # If searching all memory types, run searches concurrently for better performance
    if memory_type == "all":
        import asyncio

        # Define async wrappers for each manager call
        async def search_episodic():
            try:
                memories = (
                    await server.episodic_memory_manager.list_episodic_memory_by_org(
                        agent_state=agent_state,
                        organization_id=effective_org_id,
                        query=query,
                        embedded_text=(
                            embedded_text_padded
                            if search_method == "embedding" and query
                            else None
                        ),
                        search_field=search_field
                        if search_field != "null"
                        else "summary",
                        search_method=search_method,
                        limit=limit,
                        timezone_str="UTC",
                        filter_tags=filter_tags_dict,
                        scopes=scopes,
                        start_date=parsed_start_date,
                        end_date=parsed_end_date,
                        similarity_threshold=similarity_threshold,
                    )
                )
                return [
                    {
                        "memory_type": "episodic",
                        "id": x.id,
                        "timestamp": (
                            x.occurred_at.isoformat() if x.occurred_at else None
                        ),
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
                memories = await server.resource_memory_manager.list_resources_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=(
                        search_field
                        if search_field != "null"
                        else ("summary" if search_method == "embedding" else "content")
                    ),
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
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
                memories = (
                    await server.procedural_memory_manager.list_procedures_by_org(
                        agent_state=agent_state,
                        organization_id=effective_org_id,
                        query=query,
                        embedded_text=(
                            embedded_text
                            if search_method == "embedding" and query
                            else None
                        ),
                        search_field=search_field
                        if search_field != "null"
                        else "description",
                        search_method=search_method,
                        limit=limit,
                        timezone_str="UTC",
                        filter_tags=filter_tags_dict,
                        scopes=scopes,
                        similarity_threshold=similarity_threshold,
                    )
                )
                return [
                    _procedural_memory_response(x, include_user_id=True)
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching procedural memories across org: %s", e)
                return []

        async def search_knowledge():
            try:
                memories = await server.knowledge_vault_manager.list_knowledge_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
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
                memories = (
                    await server.semantic_memory_manager.list_semantic_items_by_org(
                        agent_state=agent_state,
                        organization_id=effective_org_id,
                        query=query,
                        embedded_text=(
                            embedded_text_padded
                            if search_method == "embedding" and query
                            else None
                        ),
                        search_field=search_field
                        if search_field != "null"
                        else "summary",
                        search_method=search_method,
                        limit=limit,
                        timezone_str="UTC",
                        filter_tags=filter_tags_dict,
                        scopes=scopes,
                        similarity_threshold=similarity_threshold,
                    )
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
            episodic_memories = (
                await server.episodic_memory_manager.list_episodic_memory_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
                    start_date=parsed_start_date,
                    end_date=parsed_end_date,
                    similarity_threshold=similarity_threshold,
                )
            )
            all_results.extend(
                [
                    {
                        "memory_type": "episodic",
                        "user_id": x.user_id,
                        "id": x.id,
                        "timestamp": (
                            x.occurred_at.isoformat() if x.occurred_at else None
                        ),
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
            resource_memories = (
                await server.resource_memory_manager.list_resources_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=(
                        search_field
                        if search_field != "null"
                        else ("summary" if search_method == "embedding" else "content")
                    ),
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
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
            procedural_memories = (
                await server.procedural_memory_manager.list_procedures_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field
                    if search_field != "null"
                    else "description",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
            )
            all_results.extend(
                [
                    _procedural_memory_response(x, include_user_id=True)
                    for x in procedural_memories
                ]
            )
        except Exception as e:
            logger.error(
                "Error searching procedural memories across organization: %s", e
            )

    # Search knowledge vault across organization
    elif memory_type == "knowledge_vault":
        try:
            knowledge_vault_memories = (
                await server.knowledge_vault_manager.list_knowledge_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "caption",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
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
            semantic_memories = (
                await server.semantic_memory_manager.list_semantic_items_by_org(
                    agent_state=agent_state,
                    organization_id=effective_org_id,
                    query=query,
                    embedded_text=(
                        embedded_text_padded
                        if search_method == "embedding" and query
                        else None
                    ),
                    search_field=search_field if search_field != "null" else "summary",
                    search_method=search_method,
                    limit=limit,
                    timezone_str="UTC",
                    filter_tags=filter_tags_dict,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
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

    # Optionally include core memory (blocks) in the results array
    if include_core_memory:
        try:
            block_filter_tags_parsed: Optional[Dict[str, Any]] = None
            if block_filter_tags:
                try:
                    block_filter_tags_parsed = json.loads(block_filter_tags)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400, detail="Invalid block_filter_tags JSON format"
                    )
            blocks = await server.block_manager.get_blocks(
                user=None,
                organization_id=effective_org_id,
                any_scopes=client.read_scopes,
                filter_tags=block_filter_tags_parsed,
                auto_create_from_default=False,
                limit=limit,
            )
            logger.info(
                "Cross-user search core memory: found %d blocks for org=%s, scopes=%s, block_filter_tags=%s",
                len(blocks),
                effective_org_id,
                client.read_scopes,
                block_filter_tags_parsed,
            )
            for block in blocks:
                all_results.append(
                    {
                        "memory_type": "core",
                        "id": block.id,
                        "user_id": block.user_id,
                        "label": block.label,
                        "value": block.value,
                        "scope": (block.filter_tags or {}).get("scope", "default"),
                    }
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Error retrieving core memory blocks for cross-user search: %s",
                e,
                exc_info=True,
            )

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
        "read_scopes": client.read_scopes,
        "filter_tags": filter_tags_dict,
        **({"warnings": warnings} if warnings else {}),
    }


@router.get("/memory/components")
async def list_memory_components(
    user_id: Optional[str] = None,
    memory_type: str = "all",
    limit: Optional[int] = None,
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()

    # Default to the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    timezone_str = getattr(user, "timezone", None) or "UTC"
    limit = max(1, min(limit if limit is not None else MAX_MEMORY_LIMIT, MAX_MEMORY_LIMIT))

    # Need an agent state for memory manager configuration
    agents = await server.agent_manager.list_agents(actor=client, limit=1)
    if not agents:
        raise HTTPException(status_code=404, detail="No agents found for this client")
    agent_state = agents[0]

    fetch_all = memory_type == "all"
    memories: Dict[str, Any] = {}

    if fetch_all or memory_type == "episodic":
        episodic_items = await server.episodic_memory_manager.list_episodic_memory(
            agent_state=agent_state,
            user=user,
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["episodic"] = {
            "total_count": await server.episodic_memory_manager.get_total_number_of_items(
                user=user
            ),
            "items": [
                {
                    "id": item.id,
                    "occurred_at": (
                        item.occurred_at.isoformat() if item.occurred_at else None
                    ),
                    "event_type": item.event_type,
                    "actor": item.actor,
                    "summary": item.summary,
                    "details": item.details,
                    "created_at": (
                        item.created_at.isoformat()
                        if getattr(item, "created_at", None)
                        else None
                    ),
                    "updated_at": (
                        item.updated_at.isoformat()
                        if getattr(item, "updated_at", None)
                        else None
                    ),
                }
                for item in episodic_items
            ],
        }

    if fetch_all or memory_type == "semantic":
        semantic_items = await server.semantic_memory_manager.list_semantic_items(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="summary",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["semantic"] = {
            "total_count": await server.semantic_memory_manager.get_total_number_of_items(
                user=user
            ),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "details": item.details,
                    "source": item.source,
                    "created_at": (
                        item.created_at.isoformat()
                        if getattr(item, "created_at", None)
                        else None
                    ),
                    "updated_at": (
                        item.updated_at.isoformat()
                        if getattr(item, "updated_at", None)
                        else None
                    ),
                }
                for item in semantic_items
            ],
        }

    if fetch_all or memory_type == "procedural":
        procedural_items = await server.procedural_memory_manager.list_procedures(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="description",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["procedural"] = {
            "total_count": await server.procedural_memory_manager.get_total_number_of_items(
                user=user
            ),
            "items": [
                _procedural_memory_response(item)
                for item in procedural_items
            ],
        }

    if fetch_all or memory_type == "resource":
        resource_items = await server.resource_memory_manager.list_resources(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="summary",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["resource"] = {
            "total_count": await server.resource_memory_manager.get_total_number_of_items(
                user=user
            ),
            "items": [
                {
                    "id": item.id,
                    "resource_type": item.resource_type,
                    "title": item.title,
                    "summary": item.summary,
                    "content": item.content,
                    "created_at": (
                        item.created_at.isoformat()
                        if getattr(item, "created_at", None)
                        else None
                    ),
                    "updated_at": (
                        item.updated_at.isoformat()
                        if getattr(item, "updated_at", None)
                        else None
                    ),
                }
                for item in resource_items
            ],
        }

    if fetch_all or memory_type == "knowledge_vault":
        knowledge_items = await server.knowledge_vault_manager.list_knowledge(
            agent_state=agent_state,
            user=user,
            query="",
            search_field="caption",
            search_method="bm25",
            limit=limit,
            timezone_str=timezone_str,
        )
        memories["knowledge_vault"] = {
            "total_count": await server.knowledge_vault_manager.get_total_number_of_items(
                user=user
            ),
            "items": [
                {
                    "id": item.id,
                    "entry_type": item.entry_type,
                    "source": item.source,
                    "sensitivity": item.sensitivity,
                    "secret_value": item.secret_value,
                    "caption": item.caption,
                    "created_at": (
                        item.created_at.isoformat()
                        if getattr(item, "created_at", None)
                        else None
                    ),
                    "updated_at": (
                        item.updated_at.isoformat()
                        if getattr(item, "updated_at", None)
                        else None
                    ),
                }
                for item in knowledge_items
            ],
        }

    if fetch_all or memory_type == "core":
        blocks = await server.block_manager.get_blocks(
            user=user,
            any_scopes=client.read_scopes,
            auto_create_from_default=False,
        )

        # Group blocks by scope
        scopes_dict: Dict[str, list] = {}
        for block in blocks[:limit]:
            scope = (block.filter_tags or {}).get("scope", "default")
            if scope not in scopes_dict:
                scopes_dict[scope] = []
            scopes_dict[scope].append(
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                }
            )

        memories["core"] = {
            "total_count": len(blocks),
            "scopes": {scope: {"items": items} for scope, items in scopes_dict.items()},
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
    await get_client_from_jwt_or_api_key(authorization, http_request)

    fields_by_type = {
        "episodic": ["summary", "details"],
        "semantic": ["name", "summary", "details"],
        "procedural": ["description", "instructions"],
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


async def get_client_from_jwt_or_api_key(
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
            client = await server.client_manager.get_client_by_id(client_id)
            if not client:
                raise HTTPException(
                    status_code=404, detail=f"Client {client_id} not found"
                )
            return client, "jwt"
        except HTTPException:
            pass  # Try API key next

    # Try injected client headers (programmatic access via middleware)
    if request:
        client_id = request.headers.get("x-client-id")
        org_id = request.headers.get("x-org-id")
        if client_id:
            client_id, org_id = await get_client_and_org(client_id, org_id)
            client = await server.client_manager.get_client_by_id(client_id)
            if not client:
                raise HTTPException(
                    status_code=404, detail=f"Client {client_id} not found"
                )
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        updated_memory = await server.episodic_memory_manager.update_event(
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    try:
        await server.episodic_memory_manager.delete_event_by_id(memory_id, actor=client)
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = await server.user_manager.get_user_by_id(user_id)
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

        updated_memory = await server.semantic_memory_manager.update_item(
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    try:
        await server.semantic_memory_manager.delete_semantic_item_by_id(
            memory_id, actor=client
        )
        return {"success": True, "message": f"Semantic memory {memory_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/memory/procedural/{memory_id}")
async def delete_procedural_memory(
    memory_id: str,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Erase a procedural memory (skill) by ID, with its whole version history.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Deletion stays on the public surface like every other memory type — a user
    must be able to remove a skill that is wrong or unwanted. It is a HARD
    delete: the skill (live or soft-deleted), every version of it, and
    the stored contents no other version references are removed in one
    transaction. The skill must be in the calling client's organization,
    belong to ``user_id`` (default: the client's admin user) and be of the
    client's ``write_scope`` — a scope the client can also read (an
    erasure is a write, by an id the client must be able to see; the scope is
    checked inside the purge transaction); otherwise 404 — a skill of another
    scope, readable or not, is not found. 403 for a read-only client (no
    ``write_scope``). Skill WRITES have no public endpoint: skills are
    created/edited only through the session-distillation evolution flow; a
    past version can be brought back with ``POST /memory/skills/{id}/restore``.
    """
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    _require_write_scope(client)

    server = get_server()
    owner = _default_user_id(client, user_id)

    # The write scope, when the client can read it too; else nothing ([]).
    erasable = (
        [client.write_scope] if client.write_scope in (client.read_scopes or []) else []
    )
    try:
        await server.procedural_memory_manager.delete_procedure_by_id(
            memory_id, actor=client, user_id=owner, scopes=erasable
        )
        return {"success": True, "message": f"Procedural memory {memory_id} deleted"}
    except NoResultFound as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Skill version history
# ============================================================================


class RestoreSkillRequest(BaseModel):
    """Body of POST /memory/skills/{skill_id}/restore."""

    version_no: int = Field(..., ge=1, description="The version to restore.")
    user_id: Optional[str] = None  # defaults to the client's admin user


def _skill_scope(filter_tags: Any) -> Optional[str]:
    if isinstance(filter_tags, dict):
        scope = filter_tags.get("scope")
        return scope if isinstance(scope, str) else None
    return None


def _require_readable_skill_scope(client, *filter_tags_values: Any) -> None:
    """404 unless every given filter_tags' scope is one the client may read.

    Skill history exposes skill content, so it follows the same read-scope
    rule as memory search (``filter_tags.scope IN client.read_scopes``). A
    skill outside it is reported as not found — its existence is not leaked.
    """
    readable = set(client.read_scopes or [])
    for filter_tags in filter_tags_values:
        if _skill_scope(filter_tags) not in readable:
            raise HTTPException(status_code=404, detail="Skill not found")


def _require_writable_skill_scope(client, *filter_tags_values: Any) -> None:
    """403 unless every given filter_tags' scope is the client's
    ``write_scope``: a skill the client can read (checked first) but whose
    scope it cannot write is not changed by it."""
    for filter_tags in filter_tags_values:
        if _skill_scope(filter_tags) != client.write_scope:
            raise HTTPException(
                status_code=403,
                detail="The skill is in a scope this client can read but not write",
            )


def _skill_version_response(version: Any) -> Dict[str, Any]:
    if hasattr(version, "model_dump"):
        return version.model_dump(mode="json")
    return dict(version)


@router.get("/memory/skills/{skill_id}/history")
async def get_skill_history_endpoint(
    skill_id: str,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    List every version of a skill, oldest first.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    ``has_retrieval_embedding`` says whether the skill has its retrieval vector
    now: one without it is invisible to every gated retrieval until a
    re-embedding pass gives it one (the eval harness refuses to freeze such a
    memory, and its frozen snapshot notices a vector added later).

    Works for soft-deleted skills too (so they can be restored). Each version
    has ``version_no``, ``change_type`` (baseline, create, edit, delete,
    restore), provenance (``run_id``, ``actor_kind``, ``reason``,
    ``created_at``), the sha256 of its seven fields and ``changed_fields``
    (versus its parent). 404 unless the skill is in the client's organization
    and its scope is one the client can read — and, when ``user_id`` is given,
    belongs to that user. Without ``user_id`` the skill may belong to any user
    of the organization: a skill id names one skill, and the organization and
    the client's read scopes are the tenant boundary (the eval harness exports
    a skill's history by its id alone).
    """
    from mirix.services.conversation_message_manager import owner_org

    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()
    owner = user_id or None  # None: any user of the organization (see above)
    try:
        result = await server.procedural_memory_manager.get_skill_history(
            skill_id, organization_id=owner_org(client), user_id=owner
        )
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    skill = result["skill"]
    _require_readable_skill_scope(client, skill.filter_tags)
    deleted_at = result["deleted_at"]
    return {
        "skill_id": skill.id,
        "name": skill.name,
        "is_deleted": result["is_deleted"],
        "deleted_at": _isoformat_or_none(deleted_at),
        "delete_reason": result["delete_reason"],
        "has_retrieval_embedding": getattr(skill, "retrieval_embedding", None)
        is not None,
        "skill": _procedural_memory_response(skill, include_user_id=True),
        "versions": result["versions"],
    }


@router.get("/memory/skills/{skill_id}/versions/{version_no}")
async def get_skill_version_endpoint(
    skill_id: str,
    version_no: int,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    One version of a skill, rebuilt from the content store.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Returns ``{skill_id, version, content}``: the version record and the
    seven versioned fields (name, entry_type, description, instructions,
    triggers, examples, filter_tags) exactly as they were stored. 404 when the
    skill or the version does not exist for this client (and ``user_id``, when
    given) — the same rules as the history.
    """
    from mirix.services.conversation_message_manager import owner_org

    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()
    owner = user_id or None  # None: any user of the organization (the history's rule)
    manager = server.procedural_memory_manager
    try:
        history = await manager.get_skill_history(
            skill_id, organization_id=owner_org(client), user_id=owner
        )
        _require_readable_skill_scope(client, history["skill"].filter_tags)
        result = await manager.get_skill_version(
            skill_id, version_no, organization_id=owner_org(client), user_id=owner
        )
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _require_readable_skill_scope(client, result["content"].get("filter_tags"))
    return {
        "skill_id": skill_id,
        "version": _skill_version_response(result["version"]),
        "content": result["content"],
    }


@router.post("/memory/skills/{skill_id}/restore")
async def restore_skill_endpoint(
    skill_id: str,
    body: RestoreSkillRequest,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Restore a skill to one of its versions.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Writes a ``restore`` version whose content is version ``version_no``'s. A
    soft-deleted skill is un-deleted. 409 when another live skill of the user
    already has the restored name (the holder is never named: it may be in a
    scope the client cannot read); 404 when the skill or version does not
    exist for this client and user (``user_id``, default: the client's admin
    user — a write, unlike the history, always names its owner); 403 for a
    read-only client (no ``write_scope``) and when the skill or the restored
    version is of a scope the client can read but not write — a restore
    writes, so both must be the client's ``write_scope``, checked again
    on the locked row. Returns ``{skill, version, restored_version_no,
    changed}`` — ``changed`` is false (and ``version`` null) when the skill
    already had exactly that content.
    """
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.procedural_memory_manager import SkillNameConflictError

    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    _require_write_scope(client)
    server = get_server()
    owner = _default_user_id(client, body.user_id)
    manager = server.procedural_memory_manager
    try:
        history = await manager.get_skill_history(
            skill_id, organization_id=owner_org(client), user_id=owner
        )
        _require_readable_skill_scope(client, history["skill"].filter_tags)
        _require_writable_skill_scope(client, history["skill"].filter_tags)
        target = await manager.get_skill_version(
            skill_id, body.version_no, organization_id=owner_org(client), user_id=owner
        )
        _require_readable_skill_scope(client, target["content"].get("filter_tags"))
        _require_writable_skill_scope(client, target["content"].get("filter_tags"))
        result = await manager.restore_skill(
            skill_id,
            body.version_no,
            actor=client,
            user_id=owner,
            scopes=[client.write_scope],
        )
    except SkillNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "skill": _procedural_memory_response(result.skill, include_user_id=True),
        "version": (
            _skill_version_response(result.version) if result.version is not None else None
        ),
        "restored_version_no": body.version_no,
        "changed": result.changed,
    }


# ============================================================================
# Raw Memory Endpoints
# ============================================================================


@router.post("/memory/raw")
async def create_raw_memory_handler(
    request: RawMemoryItemCreateRequest,
    user_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Create a new raw memory.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Args:
        request: Create request with context, filter_tags, etc.
        user_id: User ID this memory belongs to (required query parameter)
    """
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    response = await create_raw_memory(request, user_id, client)
    return response


async def create_raw_memory(
    request: RawMemoryItemCreateRequest,
    user_id: str,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
):
    """
    Create a new raw memory.

    Args:
        request: Create request with context, filter_tags, etc.
        user_id: User ID this memory belongs to (required)
        client: Client performing the operation
        client_id: Alternative to client - will be resolved to client
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # Get agent_state for embedding generation (required)
    agents = await server.agent_manager.list_agents(actor=client, limit=1)
    agent_state = agents[0] if agents else None

    if not agent_state:
        raise HTTPException(
            status_code=400,
            detail=f"No agents found for client {client.id}. An agent is required for embedding generation.",
        )

    # Build PydanticRawMemoryItemCreate from request
    from mirix.schemas.raw_memory import (
        RawMemoryItemCreate as PydanticRawMemoryItemCreate,
    )

    assert client.organization_id is not None
    raw_memory_create = PydanticRawMemoryItemCreate(
        context=request.context,
        filter_tags=request.filter_tags,
        occurred_at=request.occurred_at,
        id=request.id,
        user_id=user_id,
        organization_id=client.organization_id,
        context_embedding=None,
        embedding_config=None,
    )

    try:
        created_memory = await server.raw_memory_manager.create_raw_memory(
            raw_memory=raw_memory_create,
            actor=client,
            user_id=user_id,
            agent_state=agent_state,
            client_id=client.id,
        )
        return {
            "success": True,
            "message": f"Raw memory {created_memory.id} created",
            "memory": {
                "id": created_memory.id,
                "context": created_memory.context,
                "filter_tags": created_memory.filter_tags,
                "user_id": created_memory.user_id,
                "created_at": created_memory.created_at.isoformat(),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating raw memory: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
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

    Args:
        memory_id: ID of the memory to fetch
        user_id: Optional user ID - if provided, validates user exists and filters by user
        client: Client performing the operation
        client_id: Alternative to client - will be resolved to client
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id provided, validate user exists
    if user_id:
        try:
            from mirix.orm.errors import NoResultFound

            await server.user_manager.get_user_by_id(user_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        from mirix.orm.errors import NoResultFound

        memory = await server.raw_memory_manager.get_raw_memory_by_id(
            memory_id, actor=client, user_id=user_id
        )
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
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

    Args:
        memory_id: ID of the memory to update
        request: Update request with new context/filter_tags
        user_id: Optional user ID - if provided, validates user exists and filters by user
        client: Client performing the operation
        client_id: Alternative to client - will be resolved to client
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id provided, validate user exists
    if user_id:
        try:
            from mirix.orm.errors import NoResultFound

            await server.user_manager.get_user_by_id(user_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get agent_state for embedding generation (required)
    agents = await server.agent_manager.list_agents(actor=client, limit=1)
    agent_state = agents[0] if agents else None

    if not agent_state:
        raise HTTPException(
            status_code=400,
            detail=f"No agents found for client {client.id}. An agent is required for embedding generation.",
        )

    try:
        updated_memory = await server.raw_memory_manager.update_raw_memory(
            memory_id=memory_id,
            new_context=request.context,
            new_filter_tags=request.filter_tags,
            actor=client,
            agent_state=agent_state,
            context_update_mode=request.context_update_type,
            tags_merge_mode=request.tags_update_type,
            user_id=user_id,
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    response = await delete_raw_memory(memory_id, client)
    return response


async def delete_raw_memory(
    memory_id: str,
    client: Optional[Client] = None,
    client_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    Delete a raw memory by ID.

    This performs a hard delete and removes the memory from both PostgreSQL
    and Redis cache.

    Args:
        memory_id: ID of the memory to delete
        client: Client performing the operation
        client_id: Alternative to client - will be resolved to client
        user_id: Optional user ID - if provided, validates user exists and filters by user
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id provided, validate user exists
    if user_id:
        try:
            from mirix.orm.errors import NoResultFound

            await server.user_manager.get_user_by_id(user_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        deleted = await server.raw_memory_manager.delete_raw_memory(
            memory_id, client, user_id=user_id
        )
        if deleted:
            return {
                "success": True,
                "message": f"Raw memory {memory_id} deleted",
            }
        else:
            raise HTTPException(
                status_code=404, detail=f"Raw memory {memory_id} not found"
            )
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
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

    Args:
        request: Search request with filter_tags, sort, cursor, time_range, limit
        user_id: Optional user ID - if provided, validates user exists and filters by user
        client: Client performing the operation
        client_id: Alternative to client - will be resolved to client
    """
    server = get_server()

    # Resolve client: use provided client or fetch from client_id
    if not client and client_id:
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # If user_id provided, validate user exists
    if user_id:
        try:
            from mirix.orm.errors import NoResultFound

            await server.user_manager.get_user_by_id(user_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    filter_tags = (
        dict[str, Any](request.filter_tags) if request.filter_tags is not None else {}
    )
    scopes = client.read_scopes

    # Parse time_range from request, converting to UTC and stripping timezone for DB comparison
    time_range_dict = None
    if request.time_range:
        # The database stores naive datetimes in UTC: an aware bound is
        # converted to UTC first, a naive one is taken as UTC.
        from mirix.utils import to_naive_utc

        time_range_dict = {}
        if request.time_range.created_at_gte:
            time_range_dict["created_at_gte"] = to_naive_utc(
                request.time_range.created_at_gte
            )
        if request.time_range.created_at_lte:
            time_range_dict["created_at_lte"] = to_naive_utc(
                request.time_range.created_at_lte
            )
        if request.time_range.occurred_at_gte:
            time_range_dict["occurred_at_gte"] = to_naive_utc(
                request.time_range.occurred_at_gte
            )
        if request.time_range.occurred_at_lte:
            time_range_dict["occurred_at_lte"] = to_naive_utc(
                request.time_range.occurred_at_lte
            )
        if request.time_range.updated_at_gte:
            time_range_dict["updated_at_gte"] = to_naive_utc(
                request.time_range.updated_at_gte
            )
        if request.time_range.updated_at_lte:
            time_range_dict["updated_at_lte"] = to_naive_utc(
                request.time_range.updated_at_lte
            )

        if time_range_dict.get("created_at_gte") and time_range_dict.get(
            "created_at_lte"
        ):
            if time_range_dict["created_at_gte"] > time_range_dict["created_at_lte"]:
                raise HTTPException(
                    status_code=400, detail="created_at_gte must be <= created_at_lte"
                )
        if time_range_dict.get("occurred_at_gte") and time_range_dict.get(
            "occurred_at_lte"
        ):
            if time_range_dict["occurred_at_gte"] > time_range_dict["occurred_at_lte"]:
                raise HTTPException(
                    status_code=400, detail="occurred_at_gte must be <= occurred_at_lte"
                )
        if time_range_dict.get("updated_at_gte") and time_range_dict.get(
            "updated_at_lte"
        ):
            if time_range_dict["updated_at_gte"] > time_range_dict["updated_at_lte"]:
                raise HTTPException(
                    status_code=400, detail="updated_at_gte must be <= updated_at_lte"
                )

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

    # Validate client has organization_id
    if not client.organization_id:
        raise HTTPException(
            status_code=401, detail="Client must have an organization_id"
        )

    try:
        items, next_cursor = await server.raw_memory_manager.search_raw_memories(
            organization_id=client.organization_id,
            user_id=user_id,
            filter_tags=filter_tags,
            scopes=scopes,
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
    days_threshold: int = Query(
        14, ge=1, le=365, description="Delete memories older than this many days"
    ),
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    logger.info(
        "Manual cleanup job triggered by client %s (auth: %s) with threshold: %d days",
        client.id,
        auth_type,
        days_threshold,
    )

    response = await cleanup_raw_memories(days_threshold, client)
    return response


async def cleanup_raw_memories(
    days_threshold: int = Query(
        14, ge=1, le=365, description="Delete memories older than this many days"
    ),
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
        client = await server.client_manager.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Client or client_id required")

    # Import the cleanup function
    from mirix.jobs.cleanup_raw_memories import delete_stale_raw_memories

    try:
        # Run the cleanup job
        result = delete_stale_raw_memories(days_threshold=days_threshold)

        # Add success message to result
        result["message"] = (
            f"Deleted {result['deleted_count']} stale raw memories "
            f"(older than {days_threshold} days)"
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


async def _resolve_meta_agent_for_client(
    server, client, meta_agent_id: Optional[str], *, purpose: str
):
    """The meta memory agent a skill-learning endpoint acts on.

    An explicit ``meta_agent_id`` must be one of the client's agents (404) and a
    meta memory agent (400). Without one, the client's first meta memory agent
    is used (backward compatible); none at all is a 400.
    """
    if meta_agent_id:
        try:
            meta_agent_state = await server.agent_manager.get_agent_by_id(
                meta_agent_id,
                client,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Meta agent {meta_agent_id} not found.",
            ) from exc
        if meta_agent_state.agent_type != AgentType.meta_memory_agent:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Agent {meta_agent_id} is "
                    f"{meta_agent_state.agent_type}, not a meta_memory_agent."
                ),
            )
        return meta_agent_state
    meta_agents = await server.agent_manager.list_agents(actor=client)
    matching_meta_agents = [
        a for a in meta_agents if a.agent_type == AgentType.meta_memory_agent
    ]
    if len(matching_meta_agents) > 1:
        logger.warning(
            "%s called without meta_agent_id; using first of %d meta agents for client %s",
            purpose,
            len(matching_meta_agents),
            client.id,
        )
    if not matching_meta_agents:
        raise HTTPException(
            status_code=400,
            detail="No meta agent found for this client. Call /agents/meta/initialize first.",
        )
    return matching_meta_agents[0]


@router.post("/memory/auto_dream")
async def auto_dream_handler(
    request_body: AutoDreamRequest,
    user_id: Optional[str] = Query(None, description="User ID to run auto dream for"),
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Run the auto dream self-reflection pipeline for a user.

    Fetches memories in the specified time window (default: since last dream → now),
    then invokes the AutoDreamAgent to remove redundancies and resolve conflicts.

    Args:
        start_date: Start of time window (default: last auto dream time, or 30 days ago)
        end_date: End of time window (default: now)
        mode: Which auto-dream mode to process. One of: core, episodic, semantic,
            resource, procedural, knowledge, experience. experience processes
            episodic, semantic, and knowledge together in one agent pass.
        dry_run: If true, return counts without applying any changes
        model: Override the LLM model (e.g. "gpt-4.1-mini" for testing); for
            mode=procedural it applies to both the distiller and the curator.
        wait: mode=procedural only. The call claims every pending task session
            of the meta agent's scope (like POST /memory/skills/flush, at most
            last_n_sessions when set) into a skill learning run with trigger
            'manual' — recorded even when nothing is pending. true (default)
            runs it here and returns its result; false returns at once with the
            run id in run_ids (poll GET /memory/learning_runs/{id}).
    """
    from mirix.services.auto_dream_manager import AutoDreamManager

    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    if user_id:
        user = await server.user_manager.get_user_by_id(user_id=user_id)
    else:
        from mirix.services.admin_user_manager import ClientAuthManager
        user = await server.user_manager.get_user_by_id(
            user_id=ClientAuthManager.get_admin_user_id_for_client(client.id)
        )

    meta_agent_state = await _resolve_meta_agent_for_client(
        server, client, request_body.meta_agent_id, purpose="Auto dream"
    )

    try:
        mgr = AutoDreamManager()
        return await mgr.run(
            request=request_body,
            user=user,
            actor=client,
            meta_agent_state=meta_agent_state,
        )
    except Exception as e:
        logger.error("Auto dream failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Auto dream failed: {str(e)}")


# ============================================================================
# Conversation sessions and skill learning runs
# ============================================================================


class CloseSessionRequest(BaseModel):
    """Body of POST /memory/sessions/{session_id}/close."""

    user_id: Optional[str] = None  # defaults to the client's admin user


class FlushSkillsRequest(BaseModel):
    """Body of POST /memory/skills/flush."""

    user_id: Optional[str] = None  # defaults to the client's admin user
    close_open_sessions: bool = False
    meta_agent_id: Optional[str] = None  # defaults to the client's first meta agent


_LEARNING_RUN_STATUS_FILTERS = ("queued", "running", "succeeded", "failed", "active")


def _default_user_id(client, user_id: Optional[str]) -> str:
    if user_id:
        return user_id
    from mirix.services.admin_user_manager import ClientAuthManager

    return ClientAuthManager.get_admin_user_id_for_client(client.id)


async def _agent_owned_by_client(agent_id: str, client_id: str) -> bool:
    """True iff ``agent_id`` was created by ``client_id`` (client-level isolation)."""
    from sqlalchemy import select

    from mirix.orm.agent import Agent as AgentModel
    from mirix.server.server import db_context

    async with db_context() as session:
        row = (
            await session.execute(
                select(AgentModel.id).where(
                    AgentModel.id == agent_id,
                    AgentModel._created_by_id == client_id,
                )
            )
        ).first()
    return row is not None


def _learning_run_response(run) -> Dict[str, Any]:
    return run.model_dump(mode="json")


@router.post("/memory/sessions/{session_id}/close")
async def close_conversation_session(
    session_id: str,
    body: Optional[CloseSessionRequest] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Explicitly end a conversation session.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Sets ``closed_at = now`` / ``close_reason = 'explicit'`` and then runs the
    skill-learning trigger check for the session's scope, which starts a
    learning run once enough task sessions have ended. The check runs whenever
    this call is the one that records the session's end — also for a session
    that had idled past the timeout without any check noticing, which counts
    toward N at once. Idempotent: closing a session whose end was already
    recorded changes nothing, reports ``already_closed: true`` and never
    triggers again. 404 when the session does not exist for this user, or
    belongs to another client's meta agent (a legacy session without a
    recorded agent must be attributable to one of this client's).

    Returns ``{session_id, closed_at, close_reason, already_closed,
    learning_run_id}``.
    """
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.conversation_session_manager import (
        ConversationSessionManager,
    )
    from mirix.services.skill_learning_trigger import trigger_skill_learning

    # Authenticate first: an anonymous caller learns nothing, not even whether
    # its session_id is well-formed.
    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    try:
        _validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    user_id = _default_user_id(client, body.user_id if body else None)
    organization_id = owner_org(client)
    sessions = ConversationSessionManager()
    not_found = HTTPException(
        status_code=404,
        detail=f"Session {session_id} not found for user {user_id}",
    )
    # Tenant guard: no session of another organization's user is
    # closed (nor a learning run started for it) — the same 404.
    try:
        await _tenant_user_for_write(get_server(), user_id, client)
    except HTTPException as exc:
        raise not_found from exc

    existing = await sessions.get_session(
        organization_id=organization_id, user_id=user_id, session_id=session_id
    )
    if existing is None:
        raise not_found
    if existing.agent_id is None or not await sessions.agent_is_live(
        existing.agent_id
    ):
        # A legacy session, or an orphan whose meta agent was deleted: only a
        # client it can be attributed to may close it.
        await sessions.adopt_unattributed_sessions(
            organization_id=organization_id, user_id=user_id, session_id=session_id
        )
        existing = await sessions.get_session(
            organization_id=organization_id, user_id=user_id, session_id=session_id
        )
        if existing is None or existing.agent_id is None:
            raise not_found
    if not await _agent_owned_by_client(existing.agent_id, client.id):
        raise not_found

    outcome = await sessions.close_session(
        organization_id=organization_id, user_id=user_id, session_id=session_id
    )
    if outcome is None:  # deleted between the two calls
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found for user {user_id}",
        )

    learning_run_id = None
    if outcome.newly_ended:
        try:
            claim = await trigger_skill_learning(
                organization_id=organization_id,
                user_id=user_id,
                agent_id=outcome.session.agent_id,
                trigger="close",
                actor=client,
                current_session_id=session_id,
            )
            learning_run_id = claim.run_id if claim.fired else None
        except Exception:
            # The close is durable; the sweeper repeats the check later.
            logger.exception(
                "Skill-learning trigger check failed after closing session %s",
                session_id,
            )

    closed_at = outcome.session.closed_at
    return {
        "session_id": session_id,
        "closed_at": closed_at.isoformat() if closed_at else None,
        "close_reason": outcome.session.close_reason,
        "already_closed": outcome.already_closed,
        "learning_run_id": learning_run_id,
    }


@router.post("/memory/skills/flush")
async def flush_skills_endpoint(
    body: Optional[FlushSkillsRequest] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    Learn skills from every pending task session now, however few.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Claims all pending (ended, not yet distilled, not claimed by a live run)
    task sessions of the (user, meta agent) scope into one skill learning run
    with trigger 'flush' and runs it in the background — no threshold. With
    ``close_open_sessions: true`` the scope's still-open task sessions are
    closed explicitly first. With no pending session, experiences an earlier
    run left waiting for the curator (it crashed after distilling, or its
    curator stopped on an infrastructure error) still start a run without
    sessions, unless a queued or running run of the scope curates them.
    Returns ``{learning_run_id, session_count, closed_session_ids,
    pending_experiences}``; ``learning_run_id`` is null when there was nothing
    to learn, ``pending_experiences`` counts the experiences that were waiting
    for the curator.
    """
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.skill_learning_trigger import flush_skills

    body = body or FlushSkillsRequest()
    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    server = get_server()
    meta_agent_state = await _resolve_meta_agent_for_client(
        server, client, body.meta_agent_id, purpose="Skill flush"
    )
    user_id = _default_user_id(client, body.user_id)
    not_found = HTTPException(status_code=404, detail=f"User {user_id} not found")
    # Tenant guard: another organization's user is the same 404.
    user = await _tenant_user_for_write(server, user_id, client)
    if user is None:
        raise not_found
    try:
        result = await flush_skills(
            organization_id=owner_org(client),
            user_id=user_id,
            agent_id=meta_agent_state.id,
            close_open_sessions=body.close_open_sessions,
            actor=client,
            user=user,
            meta_agent_state=meta_agent_state,
        )
    except LookupError as exc:
        # The scope cannot be installed (e.g. the user or agent vanished).
        raise not_found from exc
    return {
        "learning_run_id": result.learning_run_id,
        "session_count": result.session_count,
        "closed_session_ids": result.closed_session_ids,
        "pending_experiences": result.pending_experiences,
    }


@router.get("/memory/learning_runs")
async def list_learning_runs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    meta_agent_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    List skill learning runs, newest first.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    Only runs of the calling client's meta agents in its organization are
    visible. Filters: ``user_id``, ``meta_agent_id``, ``status`` (queued,
    running, succeeded, failed, or ``active`` = queued + running — what a caller
    waiting for learning to go idle polls).
    """
    from sqlalchemy import select

    from mirix.orm.agent import Agent as AgentModel
    from mirix.server.server import db_context
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    if status is not None and status not in _LEARNING_RUN_STATUS_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {list(_LEARNING_RUN_STATUS_FILTERS)}",
        )
    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    async with db_context() as session:
        agent_ids = [
            row[0]
            for row in (
                await session.execute(
                    select(AgentModel.id).where(AgentModel._created_by_id == client.id)
                )
            ).all()
        ]
    if meta_agent_id is not None:
        agent_ids = [a for a in agent_ids if a == meta_agent_id]
    runs = await SkillLearningRunManager().list_runs(
        organization_id=owner_org(client),
        user_id=user_id,
        agent_ids=agent_ids,
        status=status,
        limit=limit,
    )
    return {"learning_runs": [_learning_run_response(run) for run in runs]}


@router.get("/memory/learning_runs/{run_id}")
async def get_learning_run(
    run_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    One skill learning run: status, claimed sessions and watermarks, lease
    heartbeat, error, distill/curator metrics.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    404 unless the run belongs to one of the calling client's meta agents.
    """
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    run = await SkillLearningRunManager().get_run(
        run_id, organization_id=owner_org(client)
    )
    if run is None or not await _agent_owned_by_client(run.agent_id, client.id):
        raise HTTPException(status_code=404, detail=f"Learning run {run_id} not found")
    return _learning_run_response(run)


@router.get("/memory/retrieval_logs")
async def list_retrieval_logs(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    include_trace: bool = True,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """
    List retrieval log rows, newest first.

    **Accepts both JWT (dashboard) and Client API Key (programmatic).**

    One row per memory retrieval (``/memory/retrieve/conversation`` and
    ``/memory/retrieve/topic``); its ``id`` is the ``retrieval_id`` the
    retrieval returned. A row holds the raw query, the extracted topics and
    ``extraction_status`` (ok / fallback / skipped), ``embedding_status``, the
    per-memory-type trace (``per_type``; for procedural memory every
    candidate's bm25 / cosine / rrf scores and ranks, the gate and the
    returned ids), ``llm_usage``, ``latency_ms``, ``retry_events`` and
    ``embedding_chunked_calls``.

    Only the calling client's own retrievals in its organization are visible.
    Filters: ``session_id`` (the retrieval request's), ``user_id``.
    ``include_trace=false`` leaves ``per_type`` out (null) — a cheap listing.
    The procedural trace of a row lists the returned skills and the next
    candidates in fused order (``candidates_logged`` / ``candidates_omitted``
    count them), not every candidate; the raw query keeps its head and tail
    (``llm_usage.query_text_truncated_chars`` counts the cut).
    """
    from mirix.services.conversation_message_manager import owner_org
    from mirix.services.retrieval_log_manager import RetrievalLogManager

    client, _auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )
    if session_id is not None:
        try:
            _validate_session_id(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    server = get_server()
    manager = getattr(server, "retrieval_log_manager", None) or RetrievalLogManager()
    logs = await manager.list_logs(
        organization_id=owner_org(client),
        client_id=client.id,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        include_trace=include_trace,
    )
    dumped = await asyncio.to_thread(
        lambda: [log.model_dump(mode="json") for log in logs]
    )
    return {"retrieval_logs": dumped}


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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    # If user_id is not provided, use the admin user for this client
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager

        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)
        logger.debug("No user_id provided, using admin user: %s", user_id)

    # Get user
    user = await server.user_manager.get_user_by_id(user_id)
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

        updated_memory = await server.resource_memory_manager.update_item(
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    try:
        await server.resource_memory_manager.delete_resource_by_id(
            memory_id, actor=client
        )
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
    client, auth_type = await get_client_from_jwt_or_api_key(
        authorization, http_request
    )

    server = get_server()

    try:
        await server.knowledge_vault_manager.delete_knowledge_by_id(
            memory_id, actor=client
        )
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
    write_scope: Optional[str]
    read_scopes: List[str]
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
    is_first = await auth_manager.is_first_dashboard_user()

    if is_first:
        logger.info("Creating first dashboard client (bootstrap mode)")

    try:
        # Create client with dashboard credentials
        client = await auth_manager.register_client_for_dashboard(
            name=request.name,
            email=request.email,
            password=request.password,
            write_scope="admin",  # First/dashboard users get admin scope
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
                write_scope=client.write_scope,
                read_scopes=client.read_scopes,
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

    client, access_token, auth_status = await auth_manager.authenticate(
        request.email, request.password
    )

    if auth_status == "not_found":
        raise HTTPException(
            status_code=404, detail="Account does not exist. Please create an account."
        )
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
            write_scope=client.write_scope,
            read_scopes=client.read_scopes,
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
    client = await auth_manager.get_client_by_id(client_payload["sub"])

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return DashboardClientResponse(
        id=client.id,
        name=client.name,
        email=client.email,
        write_scope=client.write_scope,
        read_scopes=client.read_scopes,
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
        raise HTTPException(
            status_code=403, detail="Only admin clients can list dashboard clients"
        )

    from mirix.services.admin_user_manager import ClientAuthManager

    auth_manager = ClientAuthManager()
    clients = await auth_manager.list_dashboard_clients(cursor=cursor, limit=limit)

    from mirix.services.admin_user_manager import ClientAuthManager

    return [
        DashboardClientResponse(
            id=c.id,
            name=c.name,
            email=c.email,
            write_scope=c.write_scope,
            read_scopes=c.read_scopes,
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

    success = await auth_manager.change_password(
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
    is_first = await auth_manager.is_first_dashboard_user()

    return {
        "setup_required": is_first,
        "message": (
            "No dashboard users exist. Please create the first account."
            if is_first
            else "Dashboard setup complete."
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
