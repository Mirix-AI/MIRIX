"""
AutoDreamManager: orchestrates the auto_dream consolidation cycle.

Default flow (``graph_only``, the only path that touches nothing but the graph):
  1. Semantic reconsolidation — merge near-duplicate anchors (an LLM decides
     identity; the merge itself is a value union plus edge rewiring, no text
     generation) and flag apparent conflicts without resolving them.
  2. Structural maintenance — orphan refs, tautologies, duplicate triples, zombie
     facts, dead-weight anchors, role-noise anchors, temporal-chain rebuild. It
     runs AFTER the merge on purpose; see _refine_graph for why the reverse order
     destroys memory links.
  3. Write a checkpoint so the next cycle knows where it resumed from.

The flat PostgreSQL memories are never read or written on this path: consolidation
belongs on the graph, where merging is a structural operation. The legacy path
(``graph_only=false``) instead hands the memory rows to an LLM agent that rewrites
and hard-deletes them; it measured -7.7 QA points on LongMemEval-S before the
union-coverage gate was added and break-even after, so it is opt-in only.
"""

import datetime as dt
import json
import os
from typing import List, Optional

from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.auto_dream import AutoDreamRequest, AutoDreamResponse, MemoryTypeStats
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.user import User as PydanticUser
from mirix.log import get_logger
from mirix.settings import settings

# Use Mirix's configured logger, not stdlib logging.getLogger — the latter's records
# never reach the server log, so batch progress, the graph-maintenance hook results and
# any batch failures were all invisible when this ran.
logger = get_logger(__name__)

# event_type used to tag auto_dream checkpoint records in episodic memory
_CHECKPOINT_EVENT_TYPE = "auto_dream_checkpoint"

_MODE_COMPONENTS = {
    "core": ["core"],
    "episodic": ["episodic"],
    "semantic": ["semantic"],
    "resource": ["resource"],
    "procedural": ["procedural"],
    "knowledge": ["knowledge"],
    "experience": ["episodic", "semantic", "knowledge"],
}


def _load_mode_system_prompt(mode: str) -> str:
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts",
        "system",
        "base",
        "auto_dream_agent",
        f"{mode}.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _dream_fetch_limit() -> int:
    """Per-component fetch cap for the dream input. The old hardcoded 500 silently
    truncated any store with more than 500 rows of one type — measured on the
    962-memory LongMemEval store, 132 of 632 episodic rows (21%) never entered the
    dream's view: not merged, not even seen. Batching is scale-independent, so a
    high default just means more batches. Override with MIRIX_DREAM_FETCH_LIMIT."""
    try:
        return max(100, int(os.environ.get("MIRIX_DREAM_FETCH_LIMIT", "10000")))
    except ValueError:
        return 10000


class AutoDreamManager:
    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                   #
    # ------------------------------------------------------------------ #

    async def get_last_dream_time(
        self,
        user: PydanticUser,
        actor: PydanticClient,
        agent_state: AgentState,
    ) -> Optional[dt.datetime]:
        """Return occurred_at of the most recent auto_dream checkpoint, or None.

        Direct MAX() over the table, not a list_* search: the previous
        limit=1 string-match could return an arbitrary (not the latest)
        checkpoint, and any incremental window derived from it needs the LATEST
        one — a wrong answer silently widens the window back to "everything".
        """
        try:
            from sqlalchemy import text as sa_text

            from mirix.server.server import db_context

            async with db_context() as session:
                res = await session.execute(
                    sa_text("SELECT MAX(occurred_at) FROM episodic_memory "
                            "WHERE user_id = :u AND event_type = :t AND NOT is_deleted"),
                    {"u": user.id, "t": _CHECKPOINT_EVENT_TYPE})
                return res.scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto dream: last-dream-time lookup failed (%s)", exc)
            return None

    async def write_checkpoint(
        self,
        user: PydanticUser,
        actor: PydanticClient,
        agent_state: AgentState,
        dream_time: dt.datetime,
    ) -> None:
        from mirix.schemas.episodic_memory import EpisodicEvent
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

        import uuid

        # Column is TIMESTAMP WITHOUT TIME ZONE — store naive UTC so comparisons
        # against created_at (also naive UTC) are apples to apples.
        if dream_time.tzinfo is not None:
            dream_time = dream_time.astimezone(dt.timezone.utc).replace(tzinfo=None)

        mgr = EpisodicMemoryManager()
        checkpoint = EpisodicEvent(
            id=f"ep_{uuid.uuid4().hex[:12]}",
            occurred_at=dream_time,
            actor="system",
            event_type=_CHECKPOINT_EVENT_TYPE,
            summary="Auto dream completed",
            details=f"Auto dream run finished at {dream_time.isoformat()}",
            filter_tags={"type": "system", "source": "auto_dream"},
            user_id=user.id,
            organization_id=actor.organization_id,
        )
        # user_id/client_id MUST be passed explicitly: create_episodic_memory
        # ignores the model's own user_id field and falls back to ADMIN_USER_ID
        # when the kwarg is absent — which silently filed every checkpoint under
        # the admin user, so get_last_dream_time (scoped to the real user) never
        # found one and every incremental window stayed fully open.
        await mgr.create_episodic_memory(
            episodic_memory=checkpoint, actor=actor,
            client_id=actor.id, user_id=user.id,
        )

    # ------------------------------------------------------------------ #
    # Memory fetching                                                      #
    # ------------------------------------------------------------------ #

    async def _graph_memory_ids(self, user: PydanticUser) -> Optional[set]:
        """Complete id set of the memories that can own a graph ref.

        Only episodic and semantic memories call ``V7GraphManager.process_memory``,
        so only those two can have a ``V7MemoryRef``.

        This set MUST be complete: ``maintain_graph`` deletes every ref NOT in it, so
        a truncated list would destroy live refs. That is why this queries the tables
        directly instead of reusing the ``list_*`` helpers, which cap at limit=500 and
        would silently truncate any store larger than that.

        Returns ``None`` (meaning "skip the orphan sweep") on any failure or if the
        result is empty — an empty set is far more likely a query bug than a real
        store with graph refs but no memories, and acting on it would wipe the graph.
        """
        try:
            from sqlalchemy import text as sa_text

            from mirix.server.server import db_context

            ids: set = set()
            async with db_context() as session:
                for table in ("episodic_memory", "semantic_memory"):
                    res = await session.execute(
                        sa_text(f"SELECT id FROM {table} "
                                f"WHERE user_id = :uid AND NOT is_deleted"),
                        {"uid": user.id},
                    )
                    ids.update(row[0] for row in res.fetchall())
            if not ids:
                logger.warning(
                    "Auto dream: memory-id enumeration came back empty for user=%s; "
                    "skipping orphan sweep rather than deleting every graph ref", user.id)
                return None
            return ids
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto dream: could not enumerate memory ids (%s); "
                           "skipping orphan sweep", exc)
            return None

    async def _semantic_ids_for_source_chunks(
        self, user: PydanticUser, source_chunk_ids: List[int]
    ) -> set[str]:
        """Resolve a v7.14+ dream batch to the semantic PG rows it created.

        Conversation ingest persists the zero-based chunk id in
        ``filter_tags.source_meta.chunk_id`` on every semantic row. Querying that
        provenance is restart-safe and avoids changing the public add-chunk response
        merely to shuttle ids into AutoDream.
        """
        if not source_chunk_ids:
            return set()
        from sqlalchemy import text as sa_text

        from mirix.server.server import db_context

        chunk_ids = [str(i) for i in sorted(set(source_chunk_ids))]
        async with db_context() as session:
            result = await session.execute(
                sa_text(
                    "SELECT id FROM semantic_memory "
                    "WHERE user_id = :uid AND NOT is_deleted "
                    "AND (filter_tags::jsonb #>> '{source_meta,chunk_id}') "
                    "= ANY(CAST(:chunk_ids AS text[]))"
                ),
                {"uid": user.id, "chunk_ids": chunk_ids},
            )
            return {row[0] for row in result.fetchall()}

    async def _fetch_episodic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

        mgr = EpisodicMemoryManager()
        events = await mgr.list_episodic_memory(
            user=user,
            agent_state=agent_state,
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
            search_method="string_match",
            query="",
            limit=_dream_fetch_limit(),
            use_cache=False,
        )
        # exclude system checkpoints
        return [e for e in events if e.event_type != _CHECKPOINT_EVENT_TYPE]

    async def _fetch_core(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
        actor: PydanticClient,
    ) -> list:
        from mirix.services.block_manager import BlockManager

        mgr = BlockManager()
        return await mgr.get_blocks(
            user=user,
            any_scopes=actor.read_scopes,
            limit=_dream_fetch_limit(),
            auto_create_from_default=False,
        )

    async def _fetch_semantic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        mgr = SemanticMemoryManager()
        return await mgr.list_semantic_items(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=_dream_fetch_limit(),
            use_cache=False,
        )

    async def _fetch_procedural(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        mgr = ProceduralMemoryManager()
        return await mgr.list_procedures(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=_dream_fetch_limit(),
            use_cache=False,
        )

    async def _fetch_resource(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.resource_memory_manager import ResourceMemoryManager

        mgr = ResourceMemoryManager()
        return await mgr.list_resources(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=_dream_fetch_limit(),
            use_cache=False,
        )

    async def _fetch_knowledge_vault(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager

        mgr = KnowledgeVaultManager()
        return await mgr.list_knowledge(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=_dream_fetch_limit(),
            use_cache=False,
        )

    # ------------------------------------------------------------------ #
    # Agent management                                                     #
    # ------------------------------------------------------------------ #

    async def get_or_create_dream_agent_state(
        self,
        actor: PydanticClient,
        meta_agent_state: AgentState,
    ) -> AgentState:
        """Return the auto_dream_agent state for this client, creating it if needed."""
        from mirix.server.rest_api import get_server

        server = get_server()
        children = await server.agent_manager.list_agents(
            actor=actor,
            parent_id=meta_agent_state.id,
        )
        for child in children:
            if child.agent_type == AgentType.auto_dream_agent:
                await server.agent_manager.update_agent_tools_and_system_prompts(child.id, actor=actor)
                child = await server.agent_manager.get_agent_by_id(agent_id=child.id, actor=actor)
                return child

        agent_create = CreateAgent(
            name=f"{meta_agent_state.name}_auto_dream_agent",
            agent_type=AgentType.auto_dream_agent,
            llm_config=meta_agent_state.llm_config,
            embedding_config=meta_agent_state.embedding_config,
            parent_id=meta_agent_state.id,
        )
        return await server.agent_manager.create_agent(agent_create=agent_create, actor=actor)

    # ------------------------------------------------------------------ #
    # Graph refinement                                                     #
    # ------------------------------------------------------------------ #

    async def _refine_graph(
        self,
        user: PydanticUser,
        dream_agent_state: AgentState,
        *,
        semantic_memory_ids: Optional[set[str]] = None,
        final_full_graph: bool = False,
    ) -> dict:
        """Refine the hypergraph only — never touches the flat PG memories.

        Merge first, then sweep. The order is load-bearing, and the obvious
        alternative (sweep first, so the merge step is not asked about anchors that
        are about to be pruned) is wrong: the dead-weight rule deletes any anchor
        with no facts and at most one memory link, and it cannot see that such an
        anchor is often just another wording of a live one. Measured on a fresh
        unmaintained graph, 139 of 663 anchors matched the dead-weight rule and
        **63 of those (45%) had a live twin at cosine >= 0.93** — "Train companies
        in Japan" beside "trains in Japan", "Phone GPS" beside "Phone GPS use".
        Sweeping first destroys those 63 memory links permanently; merging first
        transfers them onto the survivor. The asymmetry decides it: irreversible
        information loss on one side, a bounded amount of LLM budget on the other —
        and that budget is not even at risk, since an anchor with no near neighbour
        never enters candidate generation in the first place.

        1. ``reconsolidate_graph`` — merge anchors that mean the same thing in
           different words (LLM-verified, because cosine alone would merge "5-10% of
           budget" with "10-20% of budget"). The merge itself is a value union plus
           edge rewiring, no text generation. Also flags — never resolves — apparent
           contradictions. Self-gated to every N new memories since it costs LLM calls.
        2. ``maintain_graph`` — structural cleanup, now running against the merged
           graph so it collects the merge's own debris in the same cycle: a fact whose
           subject and object anchors both folded into the survivor is a tautology, and
           two facts that differed only in which merged anchor they named are duplicate
           triples. Under the old ordering that debris waited a full cycle to be
           collected (~31 pieces across a 23-cycle run), and the final cycle's was
           never collected at all. The pass also sweeps orphaned refs, zombie facts,
           genuinely dead anchors, role-noise anchors, and rebuilds the temporal chain.

        Because merging only redirects edges onto surviving anchors, every
        anchor→memory_id link is preserved, so the graph→memory_id→PG retrieval path
        keeps reaching the same memories. Every step is fail-soft: a graph error must
        never fail the dream cycle. Returns the merged stats dict.
        """
        stats: dict = {}

        if settings.graph_version in (
            "v7.19", "v7.20", "v7.21", "v7.22", "v7.23", "v7.24"
        ):
            from mirix.database.neo4j_client import get_neo4j_driver
            from mirix.services.graph_reconsolidator_v719 import (
                reconsolidate_bounded_frontier,
            )

            result = await reconsolidate_bounded_frontier(
                get_neo4j_driver(),
                user_id=user.id,
                agent_state=dream_agent_state,
                semantic_memory_ids=semantic_memory_ids or set(),
                final_full_graph=final_full_graph,
            )
            logger.info("Auto dream %s bounded frontier: %s", settings.graph_version, result)
            return {"bounded_frontier": result}

        if settings.graph_version == "v7.18":
            from mirix.database.neo4j_client import get_neo4j_driver
            from mirix.services.graph_reconsolidator_v718 import (
                reconsolidate_iterative_semantic,
            )

            result = await reconsolidate_iterative_semantic(
                get_neo4j_driver(),
                user_id=user.id,
                agent_state=dream_agent_state,
                semantic_memory_ids=semantic_memory_ids or set(),
                final_full_graph=final_full_graph,
            )
            logger.info("Auto dream v7.18 iterative semantic: %s", result)
            return {"iterative_semantic": result}

        if settings.graph_version == "v7.17":
            from mirix.database.neo4j_client import get_neo4j_driver
            from mirix.services.graph_reconsolidator_v717 import (
                reconsolidate_hybrid_semantic,
            )

            result = await reconsolidate_hybrid_semantic(
                get_neo4j_driver(),
                user_id=user.id,
                agent_state=dream_agent_state,
                semantic_memory_ids=semantic_memory_ids or set(),
                final_full_graph=final_full_graph,
            )
            logger.info("Auto dream v7.17 hybrid semantic: %s", result)
            return {"hybrid_semantic": result}

        # v7.14/v7.15 are deliberately incremental and semantic-only. Their candidate source
        # is the current batch, while its comparison target is the existing semantic
        # graph. It owns its own scoped Fact cleanup, so running the legacy full-graph
        # maintenance afterwards would violate episodic isolation and erase the cost
        # benefit this version exists to measure.
        if settings.graph_version in ("v7.14", "v7.15"):
            if not semantic_memory_ids:
                return {"incremental_semantic": {"skipped": "empty_semantic_delta"}}
            from mirix.database.neo4j_client import get_neo4j_driver

            if settings.graph_version == "v7.15":
                from mirix.services.graph_reconsolidator_v715 import (
                    reconsolidate_semantic_delta,
                )
            else:
                from mirix.services.graph_reconsolidator_v714 import (
                    reconsolidate_semantic_delta,
                )

            result = await reconsolidate_semantic_delta(
                get_neo4j_driver(),
                user_id=user.id,
                agent_state=dream_agent_state,
                semantic_memory_ids=semantic_memory_ids,
            )
            logger.info(
                "Auto dream %s incremental semantic: %s",
                settings.graph_version,
                result,
            )
            return {"incremental_semantic": result}

        try:
            from mirix.database.neo4j_client import get_neo4j_driver

            driver = get_neo4j_driver()
            every_n_memories = int(
                os.environ.get("MIRIX_GRAPH_RECONSOLIDATE_EVERY", "10")
            )
            if settings.graph_version == "v7.16":
                from mirix.services.graph_reconsolidator_v716 import (
                    reconsolidate_semantic_frontier,
                )

                recon_stats = await reconsolidate_semantic_frontier(
                    driver,
                    user_id=user.id,
                    agent_state=dream_agent_state,
                    semantic_memory_ids=semantic_memory_ids or set(),
                    every_n_memories=every_n_memories,
                )
            else:
                from mirix.services.graph_reconsolidator import reconsolidate_graph

                recon_stats = await reconsolidate_graph(
                    driver,
                    user_id=user.id,
                    agent_state=dream_agent_state,
                    every_n_memories=every_n_memories,
                )
            stats["reconsolidation"] = recon_stats
            logger.info("Auto dream: graph reconsolidation %s",
                        {k: v for k, v in recon_stats.items() if k != "conflict_samples"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto dream: graph reconsolidation skipped (%s)", exc)

        try:
            from mirix.services.graph_memory_manager_v7 import V7GraphManager

            graph_stats = await V7GraphManager().maintain_graph(
                user.id, valid_memory_ids=await self._graph_memory_ids(user)
            )
            stats["maintenance"] = graph_stats
            logger.info("Auto dream: graph maintenance %s", graph_stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto dream: graph maintenance skipped (%s)", exc)

        return stats

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
    ) -> AutoDreamResponse:
        from mirix.agent.auto_dream_agent import AutoDreamAgent
        from mirix.server.rest_api import get_server

        server = get_server()
        now = dt.datetime.now(dt.timezone.utc)

        # -- resolve time window --
        end_date = request.end_date or now
        if request.start_date:
            start_date = request.start_date
        else:
            last = await self.get_last_dream_time(user, actor, meta_agent_state)
            start_date = last or (now - dt.timedelta(days=30))

        # DB columns are TIMESTAMP WITHOUT TIME ZONE; strip tzinfo
        if start_date.tzinfo is not None:
            start_date = start_date.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.astimezone(dt.timezone.utc).replace(tzinfo=None)

        # -- graph-only dream: refine the graph, leave the flat PG memories alone --
        # The LLM memory-merge pass (below) is what degrades fact-recall QA: it blends
        # peripheral specifics into coarser merged memories whose embeddings no longer
        # retrieve them. Retrieval is graph→memory_id→PG *and* flat pgvector, both keyed
        # on those per-memory rows/embeddings; mutating them is what hurts. This mode
        # skips the merge entirely and only consolidates the hypergraph structure —
        # break-even QA, cleaner graph. reconsolidate still needs the dream agent state
        # (for its LLM verification + embeddings), so we resolve it here too.
        if request.graph_only:
            # dry_run must resolve NOTHING and mutate NOTHING — return before
            # get_or_create_dream_agent_state, which is not read-only (it can insert or
            # update an agent row). The normal path likewise returns its dry_run before
            # resolving the agent state.
            if request.dry_run:
                return AutoDreamResponse(
                    start_date=None, end_date=None, processed={}, last_dream_at=now,
                    dry_run=True,
                    message="Dry run (graph_only) — would refine the graph; no memory changes.",
                )
            dream_agent_state = await self.get_or_create_dream_agent_state(actor, meta_agent_state)
            if request.model:
                from copy import deepcopy

                dream_agent_state = deepcopy(dream_agent_state)
                dream_agent_state.llm_config.model = request.model
            # No checkpoint write here: the checkpoint only seeds the default window for
            # the merge path, and that path fetches ALL memories regardless of window
            # anyway — so in graph_only it is a pure no-op PG insert (plus a stray graph
            # write for its own row). Skipping it makes the invariant exact: graph_only
            # issues ZERO writes to the flat memory store, only graph mutations.
            semantic_memory_ids: Optional[set[str]] = None
            if settings.graph_version in (
                "v7.14", "v7.15", "v7.16", "v7.17", "v7.18", "v7.19", "v7.20", "v7.21", "v7.22", "v7.23", "v7.24"
            ):
                if request.semantic_memory_ids is not None:
                    semantic_memory_ids = set(request.semantic_memory_ids)
                elif request.source_chunk_ids is not None:
                    semantic_memory_ids = await self._semantic_ids_for_source_chunks(
                        user, request.source_chunk_ids
                    )
                else:
                    semantic_memory_ids = set()
                logger.info(
                    "Auto dream %s batch chunks=%s semantic_ids=%d",
                    settings.graph_version,
                    request.source_chunk_ids,
                    len(semantic_memory_ids),
                )
            graph_stats = await self._refine_graph(
                user,
                dream_agent_state,
                semantic_memory_ids=semantic_memory_ids,
                final_full_graph=request.final_full_graph,
            )
            return AutoDreamResponse(
                start_date=None, end_date=None, processed={}, last_dream_at=now,
                dry_run=False,
                message="Auto dream (graph_only) completed — graph refined, flat memories untouched.",
                graph_stats=graph_stats,
            )

        components = _MODE_COMPONENTS[request.mode]
        logger.info("Auto dream: window %s → %s, mode=%s, components=%s", start_date, end_date, request.mode, components)

        # -- fetch memories --
        fetch_map = {
            "core": self._fetch_core,
            "episodic": self._fetch_episodic,
            "semantic": self._fetch_semantic,
            "procedural": self._fetch_procedural,
            "resource": self._fetch_resource,
            "knowledge": self._fetch_knowledge_vault,
        }
        memories: dict = {}
        for component in components:
            fetcher = fetch_map[component]
            if component == "core":
                items = await fetcher(user, meta_agent_state, start_date, end_date, actor)
            else:
                items = await fetcher(user, meta_agent_state, start_date, end_date)
            memories[component] = items
            logger.info("  %s: fetched %d items", component, len(items))

        # -- dry_run: just return counts without invoking agent --
        if request.dry_run:
            processed = {t: MemoryTypeStats(total=len(items)) for t, items in memories.items()}
            return AutoDreamResponse(
                start_date=start_date,
                end_date=end_date,
                processed=processed,
                last_dream_at=now,
                dry_run=True,
                message="Dry run — no changes applied.",
            )

        # -- split into batches --
        # A single payload does not fit: this store's 962 memories serialise to ~569k
        # chars (~142k tokens) against a 128k window, and the summariser cannot rescue
        # it because there is only one message to compress (num_candidate_messages=0),
        # so the whole run used to die with CONTEXT_WINDOW_EXCEEDED before touching
        # anything. Batching makes the pass scale-independent.
        batches = _batch_memories(memories, _batch_char_budget())
        logger.info("Auto dream: %d batch(es) over %d items",
                    len(batches), sum(len(v) for v in memories.values()))

        # -- get or create agent state --
        dream_agent_state = await self.get_or_create_dream_agent_state(actor, meta_agent_state)
        mode_system_prompt = _load_mode_system_prompt(request.mode)
        if dream_agent_state.system != mode_system_prompt:
            dream_agent_state = await server.agent_manager.update_system_prompt(
                agent_id=dream_agent_state.id,
                system_prompt=mode_system_prompt,
                actor=actor,
            )

        # override model if caller requested it (e.g. for testing)
        if request.model:
            from copy import deepcopy

            dream_agent_state = deepcopy(dream_agent_state)
            dream_agent_state.llm_config.model = request.model

        # -- run the agent once per batch --
        # Each batch gets a freshly loaded agent so context does not accumulate across
        # batches (which would reintroduce the overflow). A failing batch is logged and
        # skipped rather than aborting the cycle — partial consolidation beats none.
        batches_ok = batches_failed = 0
        for idx, batch in enumerate(batches, start=1):
            try:
                dream_agent = await server.load_agent(
                    agent_id=dream_agent_state.id,
                    actor=actor,
                    user=user,
                    use_cache=False,
                )
                payload = _format_memories_as_message(batch, start_date, end_date, request.mode)
                await dream_agent.step(
                    input_messages=MessageCreate(role=MessageRole.user, content=payload),
                    actor=actor, user=user,
                )
                batches_ok += 1
                logger.info("Auto dream: batch %d/%d ok (%d items)",
                            idx, len(batches), sum(len(v) for v in batch.values()))
            except Exception as exc:  # noqa: BLE001
                batches_failed += 1
                logger.warning("Auto dream: batch %d/%d failed (%s)", idx, len(batches), exc)
        logger.info("Auto dream: %d batch(es) ok, %d failed", batches_ok, batches_failed)

        # -- write checkpoint --
        await self.write_checkpoint(user, actor, meta_agent_state, now)

        # -- refine the graph (maintenance + semantic reconsolidation) --
        await self._refine_graph(user, dream_agent_state)

        # -- build response (stats are approximate: we report totals fetched) --
        processed = {t: MemoryTypeStats(total=len(items)) for t, items in memories.items()}
        return AutoDreamResponse(
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
            processed=processed,
            last_dream_at=now,
            dry_run=False,
            message="Auto dream completed.",
        )


# ------------------------------------------------------------------ #
# Formatting helper                                                    #
# ------------------------------------------------------------------ #

def _serialize_item(item) -> dict:
    """Convert a Pydantic memory item to a compact dict for the LLM."""
    data = item.model_dump(exclude_none=True)
    # drop heavy embedding vectors
    for key in list(data.keys()):
        if key.endswith("_embedding"):
            del data[key]
    return data


def _batch_char_budget() -> int:
    """Serialised chars allowed per batch. ~80k chars ≈ 20k tokens, leaving the rest of
    a 128k window for the system prompt, tool schemas, the agent's reasoning and its
    tool results. Override with MIRIX_AUTO_DREAM_BATCH_CHARS."""
    try:
        return max(5_000, int(os.environ.get("MIRIX_AUTO_DREAM_BATCH_CHARS", "80000")))
    except ValueError:
        return 80_000


def _item_chars(item) -> int:
    d = _serialize_item(item)
    return sum(len(str(v)) for v in d.values()) if isinstance(d, dict) else len(str(d))


def _batch_memories(memories: dict, budget_chars: int) -> list[dict]:
    """Split memories into batches that each hold a slice of EVERY component.

    Not split by component: `experience` mode exists to review episodic, semantic and
    knowledge *together*, so a batch that contained only one type could never merge
    across types. Slicing every component proportionally keeps that cross-type view
    inside each batch.

    Order within a component is preserved (episodic arrives time-ordered, and duplicate
    events tend to sit near each other in time, so they usually land in the same batch).

    Known limitation: duplicates that fall in *different* batches are not merged in this
    cycle — the agent only ever sees one batch. Successive runs will still converge, and
    a smaller batch count (a larger budget) widens the window.
    """
    total = sum(_item_chars(it) for items in memories.values() for it in items)
    if total <= budget_chars:
        return [memories] if total else []
    n = max(1, -(-total // budget_chars))  # ceil
    batches: list[dict] = []
    for b in range(n):
        batch = {}
        for comp, items in memories.items():
            lo = (len(items) * b) // n
            hi = (len(items) * (b + 1)) // n
            if items[lo:hi]:
                batch[comp] = items[lo:hi]
        if batch:
            batches.append(batch)
    return batches


def _format_memories_as_message(
    memories: dict,
    start_date: dt.datetime,
    end_date: dt.datetime,
    mode: str,
) -> str:
    component_labels = {
        "core": "CORE MEMORY",
        "episodic": "EPISODIC MEMORY",
        "semantic": "SEMANTIC MEMORY",
        "resource": "RESOURCE MEMORY",
        "procedural": "PROCEDURAL MEMORY",
        "knowledge": "KNOWLEDGE VAULT",
    }
    component_order = " → ".join(component_labels[mem_type] for mem_type in memories.keys())
    lines = [
        f"Mode: {mode}",
        f"Time window: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        f"Provided component(s), in order: {component_order}.",
        "",
    ]
    for mem_type, items in memories.items():
        label = component_labels[mem_type]
        lines.append(f"=== {label} ({len(items)} items) ===")
        if not items:
            lines.append("(empty)")
        else:
            serialized = [_serialize_item(item) for item in items]
            lines.append(json.dumps(serialized, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    return "\n".join(lines)
