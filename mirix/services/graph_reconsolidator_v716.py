"""v7.12 AutoDream with a current-batch semantic source frontier.

This is intentionally a very small experiment.  Ingest, candidate scoring and
fan-out, LLM verification, global Fact collapse, conflict reporting, and graph
maintenance all remain v7.12 behavior.  The only algorithmic change is that the
left side of anchor candidate generation starts from anchors cited by semantic
memories created in the current Dream batch.  The right side remains the same
v7.12 user-owned anchor registry.
"""

from __future__ import annotations

from typing import Any, Optional

from mirix.services.graph_reconsolidator import reconsolidate_graph
from mirix.settings import settings


async def _semantic_delta_anchor_ids(
    driver, user_id: str, semantic_memory_ids: set[str]
) -> list[str]:
    """Return anchors touched by at least one current-batch semantic memory."""
    if not semantic_memory_ids:
        return []
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE any(mid IN coalesce(a.semantic_ids, []) WHERE mid IN $batch)
               RETURN DISTINCT a.id AS id""",
            u=user_id,
            batch=sorted(semantic_memory_ids),
        )
        return [row["id"] async for row in result if row["id"]]


async def reconsolidate_semantic_frontier(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
    dry_run: bool = False,
    every_n_memories: Optional[int] = None,
) -> dict[str, Any]:
    """Run the v7.12 pipeline with only the candidate source frontier narrowed."""
    semantic_memory_ids = {mid for mid in semantic_memory_ids if mid}
    source_anchor_ids = await _semantic_delta_anchor_ids(
        driver, user_id, semantic_memory_ids
    )
    stats = await reconsolidate_graph(
        driver,
        user_id=user_id,
        agent_state=agent_state,
        dry_run=dry_run,
        every_n_memories=every_n_memories,
        source_anchor_ids=source_anchor_ids,
    )
    return {
        **stats,
        "batch_semantic_memories": len(semantic_memory_ids),
        "source_semantic_anchors": len(source_anchor_ids),
    }
