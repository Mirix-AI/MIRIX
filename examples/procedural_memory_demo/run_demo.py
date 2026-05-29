"""
Procedural Memory Demo
======================
Tests MIRIX's ability to extract procedural knowledge (skills) from
conversations that contain step-by-step workflows, recipes, and routines.

Usage:
    # 1. Start MIRIX server first:
    #    python scripts/start_server.py --port 8531
    #
    # 2. Run the demo:
    #    python examples/procedural_memory_demo/run_demo.py \
    #        --config evals/configs/skill_evolve_openrouter.yaml

Data and pipeline are separated:
    data/conversations.json  — test conversations with procedural content
    run_demo.py              — ingestion + verification pipeline
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Allow importing mirix from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "evals"))

from mirix import MirixClient
from mirix_memory_system import _resolve_api_keys
import yaml


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA = DATA_DIR / "conversations.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_session(session: Dict, idx: int) -> str:
    header = f"Session {idx}"
    if session.get("date_time"):
        header += f" ({session['date_time']})"
    lines = [header]
    for turn in session.get("turns", []):
        lines.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


def count_messages(session: Dict) -> int:
    return len(session.get("turns", []))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ProceduralMemoryDemo:
    def __init__(self, config_path: str, client_id: str = "proc-demo", org_id: str = "proc-demo-org"):
        self.client = MirixClient(
            client_id=client_id,
            org_id=org_id,
            base_url="http://127.0.0.1:8531",
            write_scope="read_write",
            timeout=600,
        )
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self.config = _resolve_api_keys(config)

    def initialize(self, user_id: str):
        """Initialize meta agent for a user."""
        asyncio.run(self.client.initialize_meta_agent(config=self.config))
        self.user_id = user_id

    def ingest_session(self, chunk: str) -> Dict:
        """Ingest one conversation session synchronously (waits for processing)."""
        return asyncio.run(self.client.add(
            user_id=self.user_id,
            messages=[{"role": "user", "content": chunk}],
            chaining=True,
            filter_tags={"scope": "read_write", "kind": "conversation_session"},
            async_add=False,
        ))

    def search_memories(self, query: str, memory_type: str = "all",
                        method: str = "bm25", limit: int = 20) -> List[Dict]:
        """Search memories."""
        results = asyncio.run(self.client.search(
            user_id=self.user_id,
            query=query,
            memory_type=memory_type,
            search_method=method,
            limit=limit,
        ))
        return results.get("results", []) if results.get("success") else []

    def get_procedural_memories(self) -> List[Dict]:
        """Get all procedural memories (skills)."""
        items = self.search_memories("*", memory_type="procedural", method="bm25", limit=50)
        return items


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_skill(skill: Dict, idx: int):
    name = skill.get("name", "Unnamed")
    desc = skill.get("description", "")
    instructions = skill.get("instructions", "")
    version = skill.get("version", "?")
    triggers = skill.get("triggers", [])
    entry_type = skill.get("entry_type", "")

    print(f"\n  [{idx}] {name}  (v{version})")
    print(f"      Type: {entry_type}")
    if desc:
        print(f"      Description: {desc[:120]}")
    if triggers:
        print(f"      Triggers: {triggers}")
    if instructions:
        preview = instructions[:200].replace("\n", "\n      ")
        print(f"      Instructions:\n      {preview}")
        if len(instructions) > 200:
            print(f"      ... ({len(instructions)} chars total)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Procedural Memory Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to MIRIX config YAML.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to conversations JSON.")
    parser.add_argument("--user-id", type=str, default="proc-demo-user", help="User ID for the demo.")
    args = parser.parse_args()

    conversations = load_conversations(args.data)
    demo = ProceduralMemoryDemo(config_path=args.config)

    print_header("Procedural Memory Demo")
    print(f"  Config: {args.config}")
    print(f"  Data: {args.data}")
    print(f"  Conversations: {len(conversations)}")

    for conv in conversations:
        conv_id = conv.get("id", "unknown")
        title = conv.get("title", "")
        sessions = conv.get("sessions", [])
        user_id = args.user_id

        print_header(f"Conversation: {title}")
        print(f"  ID: {conv_id}, Sessions: {len(sessions)}, User: {user_id}")

        # Initialize
        print("\n  Initializing MIRIX agent...")
        demo.initialize(user_id)

        # Ingest each session
        for idx, session in enumerate(sessions, start=1):
            msg_count = count_messages(session)
            chunk = format_session(session, idx)
            date_time = session.get("date_time", "")
            if date_time:
                chunk = f"The conversation is timestamped at {date_time}.\n\n{chunk}"

            print(f"\n  Ingesting session {idx}/{len(sessions)} ({msg_count} messages)...")
            start = time.perf_counter()
            response = demo.ingest_session(chunk)
            elapsed = time.perf_counter() - start
            status = response.get("status", "unknown")
            print(f"    Status: {status} ({elapsed:.1f}s)")

        # Check all memory types
        print_header("Memory Summary")
        for mem_type in ["episodic", "semantic", "core", "knowledge", "procedural"]:
            items = demo.search_memories("*", memory_type=mem_type, method="bm25", limit=50)
            print(f"  {mem_type}: {len(items)} items")

        # Show procedural memories in detail
        print_header("Procedural Memories (Skills)")
        skills = demo.get_procedural_memories()
        if not skills:
            print("  No procedural memories found.")
            print("  This may indicate the procedural trigger did not fire.")
            # Try different search
            skills_embed = demo.search_memories(
                "routine recipe debugging workflow",
                memory_type="procedural", method="bm25", limit=10
            )
            if skills_embed:
                print(f"  (Found {len(skills_embed)} via targeted search)")
                skills = skills_embed
        else:
            print(f"  Found {len(skills)} skills:")

        for i, skill in enumerate(skills, 1):
            print_skill(skill, i)

        # Show episodic for reference
        print_header("Episodic Memories (for reference)")
        episodic = demo.search_memories(
            "morning routine pasta debugging",
            memory_type="episodic", method="bm25", limit=10
        )
        for item in episodic[:5]:
            summary = item.get("summary", "")
            ts = item.get("occurred_at", "")
            prefix = f"[{ts}] " if ts else ""
            print(f"  {prefix}{summary[:100]}")

    print_header("Demo Complete")


if __name__ == "__main__":
    main()
