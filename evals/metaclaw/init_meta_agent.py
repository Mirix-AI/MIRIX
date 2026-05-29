"""One-shot helper: initialize the MIRIX meta agent on a fresh DB.

The MetaClaw eval assumes a MIRIX server with a pre-initialized meta agent
(which owns the procedural-memory sub-agent used by /v1/skills/evolve).  A
freshly-rebuilt DB has none, so /v1/skills/evolve returns
"No procedural memory agent found".  This script POSTs /agents/meta/initialize
with an OpenRouter-backed LLM + embedding config matching the eval.

Usage:
    python -m evals.metaclaw.init_meta_agent            # uses defaults
    MIRIX_URL=http://127.0.0.1:8531 python -m evals.metaclaw.init_meta_agent
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

MIRIX_URL = os.environ.get("MIRIX_URL", "http://127.0.0.1:8531")
CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
ORG_ID = "org-00000000-0000-4000-8000-000000000000"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in env or .env")


def main() -> int:
    api_key = _load_openrouter_key()
    config = {
        "llm_config": {
            "model": "openai/gpt-5.2",
            # OpenRouter's /chat/completions is OpenAI-compatible, and this
            # MIRIX branch has no dedicated `openrouter_client` for the chat
            # side — so use the "openai" endpoint type pointed at OpenRouter
            # (same trick the eval's BENCHMARK_BASE_URL uses for the agent).
            "model_endpoint_type": "openai",
            "model_endpoint": OPENROUTER_BASE,
            "context_window": 128000,
            "api_key": api_key,
        },
        "embedding_config": {
            "embedding_model": "gemini-embedding-001",
            "embedding_endpoint_type": "openrouter",
            "embedding_endpoint": OPENROUTER_BASE,
            # Pad/truncate to MAX_EMBEDDING_DIM so it matches the VECTOR(4096)
            # ORM columns (mirix/constants.py: MAX_EMBEDDING_DIM = 4096).
            "embedding_dim": 4096,
            "embedding_chunk_size": 2048,
            "api_key": api_key,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Client-Id": CLIENT_ID,
        "X-Org-Id": ORG_ID,
    }
    body = {"config": config, "update_agents": True}

    print(f"POST {MIRIX_URL}/agents/meta/initialize")
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(
            f"{MIRIX_URL}/agents/meta/initialize", json=body, headers=headers
        )
        print(f"HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError:
            print(resp.text[:1000])
            return 1
        print(json.dumps(payload, indent=2)[:1500])
        return 0 if resp.status_code < 400 else 1


if __name__ == "__main__":
    sys.exit(main())
