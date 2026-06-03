import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path to allow importing mirix package
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirix import MirixClient
import uuid
import os
import yaml
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

_PLACEHOLDER = "your-api-key"


def _resolve_api_keys(config: dict) -> dict:
    """Replace placeholder api_key values with the matching env var."""
    for section in config.values():
        if not isinstance(section, dict) or section.get("api_key") != _PLACEHOLDER:
            continue

        # Check if endpoint points to OpenRouter
        endpoint = (
            section.get("model_endpoint")
            or section.get("embedding_endpoint")
            or ""
        ).lower()
        if "openrouter.ai" in endpoint:
            section["api_key"] = os.environ.get("OPENROUTER_API_KEY", _PLACEHOLDER)
            continue

        provider = (
            section.get("model_endpoint_type")
            or section.get("embedding_endpoint_type")
            or ""
        ).lower()
        env_var = _PROVIDER_ENV_VARS.get(provider)
        if env_var:
            section["api_key"] = os.environ.get(env_var, _PLACEHOLDER)
    return config


class MirixMemorySystem:

    def __init__(self, user_id: Optional[str] = None, mirix_config_path: Optional[str] = None, client_id: Optional[str] = None, org_id: Optional[str] = None, client: Optional[MirixClient] = None):
        if client is None:
            self.client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write", timeout=600)
            config_path = Path(mirix_config_path) if mirix_config_path else Path(__file__).with_name("mirix_openai.yaml")
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            config = _resolve_api_keys(config)
            asyncio.run(self.client.initialize_meta_agent(
                config=config,
            ))
        else:
            self.client = client
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())


    def add_chunk(self, chunk: str, raw_input: Optional[str] = None, async_add: bool = False):
        response = asyncio.run(self.client.add(
            user_id=self.user_id,
            messages=[
                {"role": "user", "content": chunk}
            ],
            # LoCoMo ingestion: run full agent chain to extract memories.
            chaining=True,
            filter_tags={"scope": "read_write", "kind": "conversation_session"},
            async_add=async_add,
        ))
        return response

    def wrap_user_prompt(self, prompt: str):
        memories = asyncio.run(self.client.retrieve_with_conversation(
            user_id=self.user_id,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        ))

        memory_context_lines = ["<episodic_memory>"]
        memories_found = False

        if memories.get("memories"):
            for memory_type, data in memories["memories"].items():
                if not data or data.get("total_count", 0) == 0:
                    continue

                # Prefer items, but fall back to recent/relevant shapes Mirix may return
                items = data.get("items", [])
                if memory_type == "episodic" and not items:
                    seen_ids = set()
                    items = []
                    for item in data.get("recent", []) + data.get("relevant", []):
                        item_id = item.get("id")
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                        items.append(item)
                if not items and "recent" in data:
                    items = data.get("recent", [])
                if not items:
                    continue

                for item in items:
                    memories_found = True

                    if memory_type == "core":
                        label = item.get("label", "")
                        value = item.get("value", "")
                        content = f"{label}: {value}".strip(": ").strip()
                        if not content:
                            content = item.get("summary", "") or str(item)
                    else:
                        content = (
                            item.get("summary")
                            or item.get("caption")
                            or item.get("name")
                            or item.get("title")
                            or item.get("description")
                            or item.get("value", "")
                        )
                        if not content:
                            content = str(item)

                        if item.get("timestamp"):
                            content = f"[{item.get('timestamp')}] {content}"

                    memory_context_lines.append(content)

        if not memories_found:
            memory_context_lines.append("None")

        memory_context_lines.append("</episodic_memory>")
        return [
            {
                'role': "system",
                'content': "These are the high-level memories retrieved automatically according to the user's query:\n" + "\n".join(memory_context_lines)
            },
            {
                'role': "user",
                'content': prompt
            }
        ]

    def list_all_memories(self, memory_type: str = "all", limit: int = 0):
        return asyncio.run(self.client.list_memory_components(
            user_id=self.user_id,
            memory_type=memory_type,
            limit=limit,
        ))
