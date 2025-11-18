import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


logger = logging.getLogger(__name__)


app = FastAPI(title="Mirix Memory API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global agent instance (initialized on startup)
agent = None


class Message(BaseModel):
    role: Optional[str] = None
    content: str


class WrapSystemPromptRequest(BaseModel):
    system_prompt: str
    conversation_history: List[Message] = []


class WrapSystemPromptResponse(BaseModel):
    system_prompt: str
    extracted_memory: Optional[str] = None


class RetrieveFromMemoryRequest(BaseModel):
    conversation_history: List[Message] = []
    memory_types: Optional[List[str]] = None  # defaults to all types if None
    search_method: Optional[str] = "bm25"
    limit: int = 10


class RetrieveFromMemoryResponse(BaseModel):
    memory: Dict[str, Any]
    topic: Optional[str] = None


class WriteToMemoryRequest(BaseModel):
    # Intentionally minimal; implementation to be provided later
    memory_type: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class WriteToMemoryResponse(BaseModel):
    success: bool
    message: str


class RawMemoryResponse(BaseModel):
    id: str
    screenshot_path: str
    source_app: str
    captured_at: str
    ocr_text: Optional[str] = None
    source_url: Optional[str] = None
    google_cloud_url: Optional[str] = None
    metadata_: Dict[str, Any] = {}
    processed: bool
    processing_count: int
    user_id: str
    organization_id: str
    created_at: Optional[str] = None
    has_embedding: bool = False


def _extract_topic_from_history(messages: List[Message]) -> str:
    """Very simple topic extraction: use the last non-empty user message, otherwise last message."""
    if not messages:
        return ""
    # Prefer last user message
    for msg in reversed(messages):
        if (msg.role or "").lower() == "user" and msg.content and msg.content.strip():
            return msg.content.strip()
    # Fallback to last message content
    return messages[-1].content.strip() if messages[-1].content else ""


@app.on_event("startup")
async def startup_event():
    """Initialize the lightweight AgentWrapper so we can access memory managers."""
    global agent
    try:
        import sys

        if getattr(sys, "frozen", False):
            bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            config_path = bundle_dir / "mirix" / "configs" / "mirix_monitor.yaml"
        else:
            config_path = Path("mirix/configs/mirix_monitor.yaml")

        from mirix.agent.agent_wrapper import AgentWrapper

        agent = AgentWrapper(str(config_path))
        logger.info("Memory server AgentWrapper initialized")
    except Exception as exc:
        logger.exception("Failed to initialize AgentWrapper: %s", exc)
        raise


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
    }


@app.post("/wrap_system_prompt", response_model=WrapSystemPromptResponse)
async def wrap_system_prompt(request: WrapSystemPromptRequest):
    """
    Takes a system_prompt and conversation_history, extracts a topic, retrieves memory,
    and returns a constructed system_prompt. Stubbed for now; implementation will be added later.
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # TODO: maybe need to convert request.conversation_history to string
    extracted_memory = agent.extract_memory_for_system_prompt(request.conversation_history)
    system_prompt = request.system_prompt + "\n\n" + extracted_memory

    # Placeholder: no wrapping logic yet; return the input prompt and empty memories
    return WrapSystemPromptResponse(system_prompt=system_prompt, extracted_memory=extracted_memory)


@app.post("/retrieve_from_memory", response_model=RetrieveFromMemoryResponse)
async def retrieve_from_memory(request: RetrieveFromMemoryRequest):
    """Extract a topic from conversation history and retrieve related memories."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    topic = _extract_topic_from_history(request.conversation_history)
    if not topic:
        return RetrieveFromMemoryResponse(memory={}, topic=topic)

    try:
        # Resolve current user
        users = agent.client.server.user_manager.list_users()
        active_user = next((u for u in users if u.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)
        if not target_user:
            raise HTTPException(status_code=404, detail="No user found")

        types = request.memory_types or [
            "episodic",
            "semantic",
            "procedural",
            "resource",
            "knowledge_vault",
        ]

        results: Dict[str, Any] = {}

        if "episodic" in types:
            try:
                items = agent.client.server.episodic_memory_manager.list_episodic_memory(
                    agent_state=agent.agent_states.episodic_memory_agent_state,
                    actor=target_user,
                    query=topic,
                    search_field="details",
                    search_method=request.search_method or "bm25",
                    limit=request.limit,
                    timezone_str=target_user.timezone,
                )
                results["episodic"] = [
                    {
                        "timestamp": it.occurred_at.isoformat() if getattr(it, "occurred_at", None) else None,
                        "summary": getattr(it, "summary", None),
                        "details": getattr(it, "details", None),
                        "event_type": getattr(it, "event_type", None),
                        "tree_path": getattr(it, "tree_path", []) if hasattr(it, "tree_path") else [],
                    }
                    for it in items
                ]
            except Exception as e:
                logger.debug("episodic retrieval failed: %s", e)
                results["episodic"] = []

        if "semantic" in types:
            try:
                items = agent.client.server.semantic_memory_manager.list_semantic_items(
                    agent_state=agent.agent_states.semantic_memory_agent_state,
                    actor=target_user,
                    query=topic,
                    search_field="summary",
                    search_method=request.search_method or "bm25",
                    limit=request.limit,
                    timezone_str=target_user.timezone,
                )
                results["semantic"] = [
                    {
                        "name": getattr(it, "name", None),
                        "summary": getattr(it, "summary", None),
                        "details": getattr(it, "details", None),
                        "source": getattr(it, "source", None),
                        "tree_path": getattr(it, "tree_path", []) if hasattr(it, "tree_path") else [],
                    }
                    for it in items
                ]
            except Exception as e:
                logger.debug("semantic retrieval failed: %s", e)
                results["semantic"] = []

        if "procedural" in types:
            try:
                items = agent.client.server.procedural_memory_manager.list_procedures(
                    agent_state=agent.agent_states.procedural_memory_agent_state,
                    actor=target_user,
                    query=topic,
                    search_field="summary",
                    search_method=request.search_method or "bm25",
                    limit=request.limit,
                    timezone_str=target_user.timezone,
                )
                formatted = []
                for it in items:
                    steps = getattr(it, "steps", [])
                    if isinstance(steps, str):
                        # Best-effort simple split for display
                        steps = [s.strip() for s in steps.replace("\n", "|").split("|") if s.strip()]
                    formatted.append(
                        {
                            "entry_type": getattr(it, "entry_type", None),
                            "summary": getattr(it, "summary", None),
                            "steps": steps if isinstance(steps, list) else [],
                            "tree_path": getattr(it, "tree_path", []) if hasattr(it, "tree_path") else [],
                        }
                    )
                results["procedural"] = formatted
            except Exception as e:
                logger.debug("procedural retrieval failed: %s", e)
                results["procedural"] = []

        if "resource" in types:
            try:
                items = agent.client.server.resource_memory_manager.list_resources(
                    agent_state=agent.agent_states.resource_memory_agent_state,
                    actor=target_user,
                    query=topic,
                    search_field="content",
                    search_method=request.search_method or "bm25",
                    limit=request.limit,
                    timezone_str=target_user.timezone,
                )
                results["resource"] = [
                    {
                        "title": getattr(it, "title", None),
                        "summary": getattr(it, "summary", None),
                        "content_preview": (getattr(it, "content", "")[:200] + "...") if getattr(it, "content", "") else None,
                        "resource_type": getattr(it, "resource_type", None),
                        "tree_path": getattr(it, "tree_path", []) if hasattr(it, "tree_path") else [],
                    }
                    for it in items
                ]
            except Exception as e:
                logger.debug("resource retrieval failed: %s", e)
                results["resource"] = []

        if "knowledge_vault" in types:
            try:
                items = agent.client.server.knowledge_vault_manager.list_knowledge(
                    agent_state=agent.agent_states.knowledge_vault_agent_state,
                    actor=target_user,
                    query=topic,
                    search_field="caption",
                    search_method=request.search_method or "bm25",
                    limit=request.limit,
                    timezone_str=target_user.timezone,
                )
                results["knowledge_vault"] = [
                    {
                        "caption": getattr(it, "caption", None),
                        "entry_type": getattr(it, "entry_type", None),
                        "source": getattr(it, "source", None),
                        "sensitivity": getattr(it, "sensitivity", None),
                    }
                    for it in items
                ]
            except Exception as e:
                logger.debug("knowledge_vault retrieval failed: %s", e)
                results["knowledge_vault"] = []

        return RetrieveFromMemoryResponse(memory=results, topic=topic)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error retrieving from memory: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error retrieving from memory: {exc}")


@app.post("/write_to_memory", response_model=WriteToMemoryResponse)
async def write_to_memory(request: WriteToMemoryRequest):
    """Stub for writing into memory; implementation to be provided later."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return WriteToMemoryResponse(success=False, message="write_to_memory not implemented yet")


@app.get("/memory/raw/{raw_memory_id}", response_model=RawMemoryResponse)
async def get_raw_memory(raw_memory_id: str):
    """
    Get detailed information about a raw memory item by its ID.

    This endpoint returns the complete raw memory data including:
    - Screenshot metadata (path, source app, capture time)
    - OCR extracted text and URLs
    - Processing status
    - Vector embedding availability

    Args:
        raw_memory_id: The unique ID of the raw memory item (format: rawmem-{uuid})

    Returns:
        RawMemoryResponse with complete raw memory details

    Raises:
        404: If raw memory item not found
        500: If server error occurs
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Get current user
        users = agent.client.server.user_manager.list_users()
        active_user = next((u for u in users if u.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)
        if not target_user:
            raise HTTPException(status_code=404, detail="No user found")

        # Get raw memory manager
        from mirix.services.raw_memory_manager import RawMemoryManager
        raw_memory_manager = RawMemoryManager()

        # Retrieve the raw memory item
        raw_memory = raw_memory_manager.get_raw_memory_by_id(
            raw_memory_id=raw_memory_id,
            user_id=target_user.id
        )

        if not raw_memory:
            raise HTTPException(
                status_code=404,
                detail=f"Raw memory item '{raw_memory_id}' not found"
            )

        # Convert to response model
        return RawMemoryResponse(
            id=raw_memory.id,
            screenshot_path=raw_memory.screenshot_path,
            source_app=raw_memory.source_app,
            captured_at=raw_memory.captured_at.isoformat() if raw_memory.captured_at else "",
            ocr_text=raw_memory.ocr_text,
            source_url=raw_memory.source_url,
            google_cloud_url=raw_memory.google_cloud_url,
            metadata_=raw_memory.metadata_ or {},
            processed=raw_memory.processed,
            processing_count=raw_memory.processing_count,
            user_id=raw_memory.user_id,
            organization_id=raw_memory.organization_id,
            created_at=raw_memory.created_at.isoformat() if raw_memory.created_at else None,
            has_embedding=raw_memory.ocr_text_embedding is not None
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error retrieving raw memory: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving raw memory: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=47284)



