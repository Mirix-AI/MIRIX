"""
Shared helpers for v4 graph managers (episodic + semantic).

Both managers do roughly the same things but write into disjoint Neo4j labels:
  - episodic: (:Episode), (:EpisodicEntity), [:EP_RELATES], [:MENTIONS], [:NEXT]
  - semantic: (:Concept), (:SemanticEntity), [:SEM_RELATES], [:MENTIONS],
              [:CONCEPT_RELATES]

This module hosts the parts that don't care which label set is in play:
helpers for id generation, name normalization, embedding batching, and the
LLM model resolution from an AgentState.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.schemas.agent import AgentState

logger = get_logger(__name__)


def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def iso(ts: datetime) -> str:
    """Neo4j datetime properties want ISO-8601 strings (with tz)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def llm_model_from_agent(agent_state: AgentState, default: str = "gpt-4.1-mini") -> str:
    """Pull LLM model name from agent_state, falling back to default."""
    try:
        cfg = getattr(agent_state, "llm_config", None)
        if cfg is not None and getattr(cfg, "model", None):
            return cfg.model
    except Exception:
        pass
    return default


async def embed_batch(
    texts: list[str], agent_state: AgentState, *, max_concurrency: int = 8
) -> list[Optional[list[float]]]:
    """
    Compute embeddings for many short strings via the agent's configured model.

    MIRIX's embedding adapter is single-text only, so we fan out with bounded
    concurrency. Returns ``None`` for failed entries so callers can decide
    whether to drop the row or store without a vector.
    """
    if not texts:
        return []
    try:
        model = await embedding_model(agent_state.embedding_config)
    except Exception as e:
        logger.warning("Embedding model init failed: %s", e)
        return [None] * len(texts)

    sem = asyncio.Semaphore(max_concurrency)

    async def one(t: str) -> Optional[list[float]]:
        async with sem:
            try:
                return await model.get_text_embedding(t)
            except Exception as e:
                logger.debug("Embed failed for '%s...': %s", (t or "")[:40], e)
                return None

    return await asyncio.gather(*(one(t) for t in texts))
