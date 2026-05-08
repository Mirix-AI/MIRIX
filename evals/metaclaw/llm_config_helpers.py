"""LLMConfig / EmbeddingConfig builders for the OpenRouter setup used by
this eval harness. Both chat and embedding go through OpenRouter; the
MIRIX OpenAI client (`mirix/llm_api/openai_client.py`) accepts arbitrary
`base_url`, so the same client class serves both endpoints.
"""
from __future__ import annotations

import os

from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-5.2"
DEFAULT_EMBED_MODEL = "google/gemini-embedding-001"
DEFAULT_EMBED_DIM = 1536


def openrouter_chat_config(model: str | None = None) -> LLMConfig:
    return LLMConfig(
        model=model or os.environ.get("EVAL_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        model_endpoint_type="openai",
        model_endpoint=OPENROUTER_BASE_URL,
        context_window=128_000,
    )


def openrouter_embedding_config(
    model: str | None = None,
    dim: int | None = None,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_model=model or os.environ.get("EVAL_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embedding_endpoint_type="openai",
        embedding_endpoint=OPENROUTER_BASE_URL,
        embedding_dim=int(dim or os.environ.get("EVAL_EMBED_DIM", DEFAULT_EMBED_DIM)),
        embedding_chunk_size=300,
    )


def assert_openrouter_env() -> None:
    """Fail fast if env is not configured."""
    missing = [k for k in ("OPENAI_API_KEY",) if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing env vars: {missing}. "
            f"Set OPENAI_API_KEY to your OpenRouter key, "
            f"OPENAI_API_BASE to {OPENROUTER_BASE_URL}."
        )
