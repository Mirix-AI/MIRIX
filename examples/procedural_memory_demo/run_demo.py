"""
Procedural Memory Demo
======================
Tests MIRIX's ability to extract procedural knowledge (skills) from
conversations that contain step-by-step workflows, recipes, and routines.

Procedural memory is learned from session-id'd conversations: each session is
ingested with a distinct ``session_id`` (no session id → no skill learning),
and skills are distilled from SEALED sessions — a session only seals once a
newer session exists, so the demo writes one tiny boundary session at the end
and then drives consolidation explicitly via auto_dream(mode="procedural").

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
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

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


def build_session_id(*parts: str) -> str:
    """Mint a server-valid session id: [A-Za-z0-9_-]+, max 64 chars.

    The last part is the distinguishing suffix (s01/s02/boundary) and is kept
    intact. When the head overflows its budget, the overflow is replaced with
    a stable hash of the FULL head so two long heads sharing a prefix (e.g.
    two long conversation ids) can never collapse into one session id.
    """
    cleaned = [
        re.sub(r"[^A-Za-z0-9_-]+", "-", str(p)).strip("-") for p in parts if str(p)
    ]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return "session"
    suffix = cleaned[-1][:32]
    head = "-".join(cleaned[:-1])
    if not head:
        return suffix
    head_budget = 64 - len(suffix) - 1
    if len(head) > head_budget:
        # 16 hex chars (64 bits): birthday-safe for any realistic id count,
        # unlike an 8-char slice which collides after ~2^16 distinct heads.
        digest = hashlib.sha256(head.encode("utf-8")).hexdigest()[:16]
        keep = head_budget - len(digest) - 1
        head = f"{head[:keep].rstrip('-')}-{digest}" if keep >= 1 else digest
    return f"{head}-{suffix}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ProceduralMemoryDemo:
    def __init__(
        self,
        config_path: str,
        client_id: str = "proc-demo",
        org_id: str = "proc-demo-org",
    ):
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

    def ingest_session(self, chunk: str, session_id: str) -> Dict:
        """Ingest one conversation session synchronously (waits for processing).

        The top-level session_id is what routes the turns into the Conversation
        Message Store — without it no procedural skill can be distilled.
        """
        return asyncio.run(
            self.client.add(
                user_id=self.user_id,
                messages=[{"role": "user", "content": chunk}],
                chaining=True,
                session_id=session_id,
                async_add=False,
            )
        )

    def consolidate(self, last_n_sessions: int, boundary_session_id: str) -> Dict:
        """Distill the sealed sessions into procedural skills.

        Sessions only seal when a newer session exists, so we first write a
        tiny boundary session to seal the last real one, then run auto_dream
        in procedural mode instead of waiting for the in-band trigger. The
        boundary id must be fresh per consolidation (sealing goes by a
        session's first appearance), and the previous consolidation's boundary
        occupies one slot in this batch — hence the +1 slack.
        """
        asyncio.run(
            self.client.add(
                user_id=self.user_id,
                messages=[{"role": "user", "content": "Consolidation boundary."}],
                chaining=False,
                session_id=boundary_session_id,
                async_add=False,
            )
        )
        return asyncio.run(
            self.client.auto_dream(
                user_id=self.user_id,
                mode="procedural",
                last_n_sessions=last_n_sessions + 1,
            )
        )

    def search_memories(
        self,
        query: str,
        memory_type: str = "all",
        method: str = "bm25",
        limit: int = 20,
    ) -> List[Dict]:
        """Search memories."""
        results = asyncio.run(
            self.client.search(
                user_id=self.user_id,
                query=query,
                memory_type=memory_type,
                search_method=method,
                limit=limit,
            )
        )
        return results.get("results", []) if results.get("success") else []

    def get_procedural_memories(self) -> List[Dict]:
        """Get all procedural memories (skills)."""
        items = self.search_memories(
            "*", memory_type="procedural", method="bm25", limit=50
        )
        return items


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


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
    parser.add_argument(
        "--config", type=str, required=True, help="Path to MIRIX config YAML."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Path to conversations JSON."
    )
    parser.add_argument(
        "--user-id", type=str, default="proc-demo-user", help="User ID for the demo."
    )
    args = parser.parse_args()

    conversations = load_conversations(args.data)
    demo = ProceduralMemoryDemo(config_path=args.config)
    # Session ids must be run-unique: reruns reusing an id would append turns
    # to an already-distilled session, which the distiller never revisits.
    # The random tail keeps two runs started in the same second distinct.
    run_token = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

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

        # Ingest each session under its own session_id (required for skills)
        for idx, session in enumerate(sessions, start=1):
            msg_count = count_messages(session)
            chunk = format_session(session, idx)
            date_time = session.get("date_time", "")
            if date_time:
                chunk = f"The conversation is timestamped at {date_time}.\n\n{chunk}"

            session_id = build_session_id(
                "proc-demo", run_token, conv_id, f"s{idx:02d}"
            )
            print(
                f"\n  Ingesting session {idx}/{len(sessions)} "
                f"({msg_count} messages, session_id={session_id})..."
            )
            start = time.perf_counter()
            response = demo.ingest_session(chunk, session_id)
            elapsed = time.perf_counter() - start
            status = response.get("status", "unknown")
            print(f"    Status: {status} ({elapsed:.1f}s)")

        # Distill the ingested sessions into procedural skills
        print_header("Consolidation (auto_dream mode=procedural)")
        start = time.perf_counter()
        dream = demo.consolidate(
            last_n_sessions=len(sessions),
            boundary_session_id=build_session_id(
                "proc-demo", run_token, conv_id, "boundary"
            ),
        )
        elapsed = time.perf_counter() - start
        print(f"  skills_changed: {dream.get('skills_changed', 0)} ({elapsed:.1f}s)")
        if dream.get("message"):
            print(f"  message: {dream['message']}")

        # Check all memory types
        print_header("Memory Summary")
        for mem_type in ["episodic", "semantic", "core", "knowledge", "procedural"]:
            items = demo.search_memories(
                "*", memory_type=mem_type, method="bm25", limit=50
            )
            print(f"  {mem_type}: {len(items)} items")

        # Show procedural memories in detail
        print_header("Procedural Memories (Skills)")
        skills = demo.get_procedural_memories()
        if not skills:
            print("  No procedural memories found.")
            print("  Check the consolidation step above: skills_changed should be > 0.")
            # Try different search
            skills_embed = demo.search_memories(
                "routine recipe debugging workflow",
                memory_type="procedural",
                method="bm25",
                limit=10,
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
            memory_type="episodic",
            method="bm25",
            limit=10,
        )
        for item in episodic[:5]:
            summary = item.get("summary", "")
            ts = item.get("occurred_at", "")
            prefix = f"[{ts}] " if ts else ""
            print(f"  {prefix}{summary[:100]}")

    print_header("Demo Complete")


if __name__ == "__main__":
    main()
