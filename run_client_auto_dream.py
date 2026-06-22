#!/usr/bin/env python3
"""
Smoke test for auto-dream.

Prerequisites:
- Start the server first: python scripts/start_server.py
- Ensure MIRIX_API_KEY is set if your server requires it.

Examples:
    python run_client_auto_dream.py --mode experience --dry-run
    python run_client_auto_dream.py --mode procedural
"""

import argparse
import asyncio
import logging
from typing import Any, Dict, Iterable, List

from mirix import MirixClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODE_TO_COMPONENTS = {
    "core": ["core"],
    "episodic": ["episodic"],
    "semantic": ["semantic"],
    "resource": ["resource"],
    "procedural": ["procedural"],
    "knowledge": ["knowledge_vault"],
    "experience": ["episodic", "semantic", "knowledge_vault"],
}


def _sample_messages() -> List[Dict[str, Any]]:
    """Return intentionally overlapping memories for auto-dream to inspect."""
    first = (
        "Please update memory from this project onboarding note. "
        "User David is a senior software engineer at TechCorp. David prefers Python over JavaScript "
        "and uses VS Code as his main IDE. Yesterday David attended the Q4 AI roadmap planning meeting "
        "about retrieval quality, memory consolidation, and database migration cleanup. "
        "The deployment workflow is: run tests, build the Docker image, push it to the registry, "
        "then apply Kubernetes manifests to staging before production. "
        "The Q4 Architecture Brief is a resource document about memory services and background workers. "
        "The staging database password is stage_db_pw_2026, source: onboarding note, sensitivity: secret."
    )
    second = (
        "Please update memory from this follow-up note. "
        "David works as a senior software engineer at TechCorp and strongly prefers Python. "
        "David uses VS Code for most coding. Yesterday he joined the AI features roadmap meeting, "
        "also called the Q4 AI roadmap planning meeting, covering retrieval quality and memory consolidation. "
        "Deployment process reminder: run the test suite, build the Docker image, push to the registry, "
        "apply Kubernetes manifests in staging, then promote to production. "
        "The Q4 Architecture Brief resource describes memory services, background workers, and deployment checks. "
        "The staging database password is stage_db_pw_2026_ROTATED, source: follow-up note, sensitivity: secret."
    )
    return [
        {
            "role": "user",
            "content": first,
        },
        {
            "role": "assistant",
            "content": "I will update the relevant memories from the onboarding note.",
        },
        {
            "role": "user",
            "content": second,
        },
        {
            "role": "assistant",
            "content": "I will update the relevant memories from the follow-up note.",
        },
    ]


def _component_total(component_data: Dict[str, Any]) -> int:
    if "total_count" in component_data:
        return component_data["total_count"]
    if "scopes" in component_data:
        return sum(len(scope_data.get("items", [])) for scope_data in component_data["scopes"].values())
    return 0


def _print_component_summary(title: str, response: Dict[str, Any], components: Iterable[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    memories = response.get("memories", {})
    for component in components:
        data = memories.get(component, {})
        print(f"{component}: {_component_total(data)}")

        items = data.get("items", [])
        if not items and "scopes" in data:
            for scope_data in data["scopes"].values():
                items.extend(scope_data.get("items", []))

        for item in items[:3]:
            label = (
                item.get("summary")
                or item.get("name")
                or item.get("title")
                or item.get("caption")
                or item.get("label")
                or item.get("id")
            )
            print(f"  - {label}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sample memories and run auto-dream.")
    parser.add_argument("--mode", default="experience", choices=sorted(MODE_TO_COMPONENTS.keys()))
    parser.add_argument("--user-id", default="auto-dream-test-user")
    parser.add_argument("--client-id", default="auto-dream-test-client")
    parser.add_argument("--client-scope", default="AutoDreamTest")
    parser.add_argument("--org-id", default="demo-org")
    parser.add_argument("--config-path", default="mirix/configs/examples/mirix_openai_with_auto_dram.yaml")
    parser.add_argument("--skip-seed", action="store_true", help="Do not insert sample memories before auto-dream.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch/count memories without invoking the agent.")
    parser.add_argument("--model", default=None, help="Optional model override for auto-dream.")
    args = parser.parse_args()

    client = await MirixClient.create(
        client_id=args.client_id,
        client_scope=args.client_scope,
        org_id=args.org_id,
        debug=True,
    )

    meta_agent = await client.initialize_meta_agent(
        config_path=args.config_path,
        update_agents=True,
    )
    if meta_agent is None:
        raise RuntimeError("Meta agent was not initialized. Check that the client has a write_scope.")

    if not args.skip_seed:
        print("Seeding sample memories with /memory/add_sync ...")
        seed_result = await client.add(
            user_id=args.user_id,
            messages=_sample_messages(),
            chaining=True,
            verbose=True,
            filter_tags={"source": "run_client_auto_dream"},
            block_filter_tags={"source": "run_client_auto_dream"},
            async_add=False,
        )
        print(f"Seed status: {seed_result.get('status')} ({seed_result.get('message_count')} messages)")

    components = MODE_TO_COMPONENTS[args.mode]
    before = await client.list_memory_components(args.user_id, memory_type="all", limit=10)
    _print_component_summary("Before auto-dream", before, components)

    print(f"\nRunning auto-dream: mode={args.mode}, dry_run={args.dry_run}")
    result = await client.auto_dream(
        user_id=args.user_id,
        mode=args.mode,
        dry_run=args.dry_run,
        model=args.model,
    )
    print("Auto-dream result:")
    print(result)

    after = await client.list_memory_components(args.user_id, memory_type="all", limit=10)
    _print_component_summary("After auto-dream", after, components)


if __name__ == "__main__":
    asyncio.run(main())
