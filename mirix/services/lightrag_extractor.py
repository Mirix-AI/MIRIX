"""
Shared helpers for the graph extractors: the ExtractedEntity record and a bare
OpenAI-compatible chat call (honours OPENAI_API_BASE).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from mirix.log import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str


async def call_openai_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: float = 180.0,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Bare-metal OpenAI chat completion. Mirrors v2 graph_memory_manager."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    endpoint = f"{api_base.rstrip('/')}/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Record token usage if a phase is active (no-op outside instrumented evals)
    try:
        from mirix.database.token_tracker import record as _record_tokens
        usage = (data.get("usage") or {})
        _record_tokens(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
        )
    except Exception:
        pass

    return data["choices"][0]["message"]["content"]
