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

# Load .env file from project root (parent directory of evals/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class MirixMemorySystem:
    
    def __init__(self, user_id: Optional[str] = None, mirix_config_path: Optional[str] = None, mirix_api_key: Optional[str] = None):
        self.client = MirixClient(api_key=mirix_api_key, base_url="http://127.0.0.1:8531")
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        config_path = Path(mirix_config_path) if mirix_config_path else Path(__file__).with_name("mirix_openai.yaml")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        self.client.initialize_meta_agent(
            config=config
        )

    def add_chunk(self, chunk: str, raw_input: Optional[str] = None):
        response = self.client.add(
            user_id=self.user_id,
            messages=[
                {"role": "user", "content": chunk}
            ],
            raw_input=raw_input,
            save_raw_inputs=True if raw_input is not None else None,
            # LoCoMo ingestion: avoid running the full multi-agent chain on every session.
            chaining=False,
            filter_tags={"scope": "read_write", "kind": "conversation_session"},
            async_add=False,
        )
        return response
    
    def wrap_user_prompt(self, prompt: str):
        memories = self.client.retrieve_with_conversation(
            user_id=self.user_id,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

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
        return self.client.list_memory_components(
            user_id=self.user_id,
            memory_type=memory_type,
            limit=limit,
        )
