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

import asyncio
import contextlib
import datetime as dt
import json
import os
from typing import Dict, List, Optional

from mirix.helpers.keyed_locks import KeyedLocks
from mirix.services.conversation_message_manager import owner_org as _owner_org
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.auto_dream import AutoDreamRequest, AutoDreamResponse, MemoryTypeStats
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.user import User as PydanticUser
from mirix.errors import LearningRunInternalError
from mirix.log import get_logger
from mirix.services.retry_policy import RetryEventRecorder
from mirix.services.session_experience_distiller import internal_failure_sessions
from mirix.settings import settings

# Use Mirix's configured logger, not stdlib logging.getLogger — the latter's records
# never reach the server log, so batch progress, the graph-maintenance hook results and
# any batch failures were all invisible when this ran.
logger = get_logger(__name__)

# event_type used to tag auto_dream checkpoint records in episodic memory
_CHECKPOINT_EVENT_TYPE = "auto_dream_checkpoint"
# The newest episodic events an episodic dream reads (checkpoints excluded).
_EPISODIC_DREAM_WINDOW = 500

# Per-(user, org) serialization of skill learning runs. Runs claim DISJOINT
# sessions (a live run's sessions are excluded from every other claim), but
# their curator stages edit the same skill bank and curate the same `pending`
# experiences of the scope; running two at once would curate the same
# experiences twice. So a run executes only while holding this guard.
#
# Two layers:
#   * a process-local asyncio.Lock (serializes coroutines within ONE event loop), AND
#   * a Postgres SESSION-level ADVISORY lock keyed by hash(user, org), held across
#     the whole run, so concurrent gunicorn/uvicorn WORKER PROCESSES also serialize.
# On non-Postgres backends (SQLite / PGlite — single-process test setups with no
# advisory locks) the asyncio.Lock alone is sufficient, so the advisory layer is
# skipped gracefully.
# Self-evicting per-(user, org) registry; see _procedural_dream_guard.
_PROCEDURAL_DREAM_LOCKS = KeyedLocks()

# Stable namespace salt so this feature's advisory-lock keys never collide with
# another feature's pg_advisory_lock keyspace.
_PROCEDURAL_ADVISORY_NAMESPACE = "mirix.procedural_dream"


def _distill_usage(entries: List[dict]) -> dict:
    """The run's ``llm_usage`` delta from its distill metric entries:
    totals, and the same under ``by_stage.distill``."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for entry in entries:
        totals["calls"] += int(entry.get("llm_calls") or 0)
        usage = entry.get("llm_usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)
    if not totals["calls"]:
        return {}
    return {**totals, "by_stage": {"distill": dict(totals)}}


def _dream_lock_key(user_id: str, organization_id: Optional[str]) -> str:
    """Registry key for the process-local procedural auto-dream lock — keyed
    per (user, organization) so different owners never block each other."""
    return f"{organization_id or '-'}::{user_id}"


def _advisory_lock_key(user_id: str, organization_id: Optional[str]) -> int:
    """Map (user, org) to a stable signed 64-bit int for pg_advisory_lock.

    Postgres advisory locks key on a bigint; we hash the namespaced owner string
    with blake2b (stable across processes, unlike Python's salted hash()) and fold
    it into the signed 64-bit range Postgres expects.
    """
    import hashlib

    raw = f"{_PROCEDURAL_ADVISORY_NAMESPACE}:{organization_id or '-'}:{user_id}".encode()
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    val = int.from_bytes(digest, "big", signed=False)
    # Fold unsigned 64-bit into signed 64-bit range expected by bigint.
    return val - (1 << 64) if val >= (1 << 63) else val


async def _invalidate_guard_connection(session) -> None:
    """Close the guard's DBAPI connection instead of returning it to the pool.

    Closing a Postgres connection releases every session-level advisory lock it
    holds, so a lock whose acquisition or release was interrupted can never
    stay held by a pooled connection (where it would block this owner's runs
    in every process until the pool recycles it).
    """
    try:
        connection = await session.connection()
        await connection.invalidate()
    except BaseException:  # noqa: BLE001 — best effort; the caller re-raises
        logger.debug("Could not invalidate the guard connection", exc_info=True)


async def _exit_guard_session(session_cm) -> None:
    try:
        await session_cm.__aexit__(None, None, None)
    except BaseException:  # noqa: BLE001 — best effort
        logger.debug("Could not close the guard session", exc_info=True)


@contextlib.asynccontextmanager
async def _procedural_dream_guard(user_id: str, organization_id: Optional[str]):
    """Serialize the procedural auto-dream for one (user, org) across coroutines
    AND across processes.

    Holds the process-local asyncio.Lock for the whole run, and — on Postgres —
    additionally takes a SESSION-level advisory lock on a dedicated connection
    held open for the run's duration, releasing it in a finally. On non-Postgres
    backends (SQLite/PGlite — single-process setups the asyncio.Lock already
    covers) the advisory step is skipped. Lock-plumbing failures degrade to the
    asyncio.Lock rather than crashing the run.

    Cancellation-safe: a run cancelled while it waits for (or right after it was
    granted) the advisory lock — shutdown, a lost lease, an erasure — closes the
    dedicated connection rather than pooling it, and so does a release that
    fails or is interrupted; the lock can never outlive its holder.
    """
    from sqlalchemy import text

    from mirix.server.server import db_context

    key = _advisory_lock_key(user_id, organization_id)
    async with _PROCEDURAL_DREAM_LOCKS.acquire(_dream_lock_key(user_id, organization_id)):
        # Open a dedicated session and try to hold a session-level advisory lock
        # for the whole run. session_cm/session are None when we are not on
        # Postgres or acquisition failed — then the asyncio.Lock alone serializes.
        session_cm = None
        session = None
        try:
            session_cm = db_context()
            session = await session_cm.__aenter__()
            dialect = session.bind.dialect.name if session.bind is not None else ""
            if dialect == "postgresql":
                # SESSION-level (NOT xact) so it spans the run's many transactions.
                await session.execute(
                    text("SELECT pg_advisory_lock(:k)"), {"k": key}
                )
            else:
                # No advisory-lock primitive on SQLite/PGlite — release the unused
                # session; the asyncio.Lock already covers these single-process setups.
                await session_cm.__aexit__(None, None, None)
                session_cm = None
                session = None
        except asyncio.CancelledError:
            # The lock may have been granted to this connection just before the
            # cancellation landed: drop the connection, never pool it.
            if session is not None:
                await _invalidate_guard_connection(session)
            if session_cm is not None:
                await _exit_guard_session(session_cm)
            raise
        except Exception:  # noqa: BLE001 — lock plumbing must never crash the run
            logger.warning(
                "Procedural dream advisory lock unavailable; relying on the "
                "process-local lock only.",
                exc_info=True,
            )
            if session is not None:
                await _invalidate_guard_connection(session)
            if session_cm is not None:
                await _exit_guard_session(session_cm)
            session_cm = None
            session = None

        try:
            yield
        finally:
            if session is not None:
                released = False
                try:
                    await session.execute(
                        text("SELECT pg_advisory_unlock(:k)"), {"k": key}
                    )
                    released = True
                except Exception:  # noqa: BLE001 — the connection is dropped below
                    logger.debug(
                        "pg_advisory_unlock failed; dropping the connection releases it."
                    )
                finally:
                    if not released:
                        await _invalidate_guard_connection(session)
                    await _exit_guard_session(session_cm)

# Generic "fetch current memories of these components -> one consolidation agent"
# modes. mode='procedural' is intentionally NOT here: it is handled by the
# skill-learning path (_run_procedural), which returns before this mapping is
# ever consulted.
_MODE_COMPONENTS = {
    "core": ["core"],
    "episodic": ["episodic"],
    "semantic": ["semantic"],
    "resource": ["resource"],
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
        text = f.read()
    # The graph pipeline adds its merge rule (a merge drops wording, never facts)
    # to the modes that merge records.
    if settings.graph_pipeline and mode in ("episodic", "semantic", "experience"):
        with open(
            os.path.join(os.path.dirname(prompt_path), "graph_pipeline_merge_rule.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            text += f.read()
    return text


def _dream_fetch_limit() -> int:
    """Per-component fetch cap for the dream input. The old hardcoded 500 silently
    truncated any store with more than 500 rows of one type — measured on the
    962-memory LongMemEval store, 132 of 632 episodic rows (21%) never entered the
    dream's view: not merged, not even seen. Batching is scale-independent, so a
    high default just means more batches. Override with MIRIX_DREAM_FETCH_LIMIT.
    Outside the graph pipeline (settings.graph_pipeline) the default stays 500."""
    default = 10000 if settings.graph_pipeline else 500
    try:
        return max(100, int(os.environ.get("MIRIX_DREAM_FETCH_LIMIT", str(default))))
    except ValueError:
        return default


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
        Outside the graph pipeline (settings.graph_pipeline) this keeps the
        string-match lookup.
        """
        if not settings.graph_pipeline:
            from mirix.services.episodic_memory_manager import EpisodicMemoryManager

            mgr = EpisodicMemoryManager()
            events = await mgr.list_episodic_memory(
                user=user,
                agent_state=agent_state,
                query=_CHECKPOINT_EVENT_TYPE,
                search_method="string_match",
                search_field="event_type",
                limit=1,
                use_cache=False,
            )
            for ev in events:
                if ev.event_type == _CHECKPOINT_EVENT_TYPE:
                    return ev.occurred_at
            return None
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
        # (graph pipeline only; outside it the time is stored unchanged)
        if settings.graph_pipeline and dream_time.tzinfo is not None:
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
        # The owner is passed explicitly: create_episodic_memory stamps its own
        # user_id / client_id arguments over the event's, and without them the
        # checkpoint would land in the ADMIN user's memory — where
        # get_last_dream_time(user) never finds it.
        await mgr.create_episodic_memory(
            episodic_memory=checkpoint,
            actor=actor,
            client_id=actor.id,
            user_id=user.id,
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

        if settings.graph_pipeline:
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

        # Auto-dream fetches the newest events regardless of date (the passed
        # window is only recorded in the response for reference). The dream
        # checkpoints (one per learning run) are left out in the query itself,
        # before the window's limit: they never crowd out the real events.
        return await EpisodicMemoryManager().list_recent_events(
            user,
            limit=_EPISODIC_DREAM_WINDOW,
            exclude_event_types=(_CHECKPOINT_EVENT_TYPE,),
        )

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
        from mirix.server.server import get_server

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
    # Skill learning — procedural path                                    #
    # ------------------------------------------------------------------ #

    async def _run_procedural(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
        run_id: Optional[str],
    ) -> AutoDreamResponse:
        """mode='procedural': session distillation + skill evolution.

        * ``run_id`` given — execute that claimed learning run. This is what the
          trigger's background task calls (``skill_learning_trigger``).
        * ``dry_run`` — report how many pending sessions a run would claim; no
          side effects at all.
        * otherwise (a manual ``POST /memory/auto_dream``) — flush semantics:
          claim every pending session of this (organization, user, meta agent)
          scope, at most ``request.last_n_sessions`` when set, into a new run
          with trigger 'manual'. The run is recorded even when nothing is
          pending (it still drains experiences a previous run left pending).
          ``request.wait`` (default) executes it here and returns its result;
          ``wait=False`` schedules it in the background and returns the run id
          at once.
        """
        from mirix.services.agent_trigger_state_manager import (
            AgentTriggerStateManager,
        )
        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )

        organization_id = _owner_org(actor)
        if run_id is None and request.dry_run:
            pending = await ConversationSessionManager().list_pending_distill_sessions(
                organization_id=organization_id,
                user_id=user.id,
                agent_id=meta_agent_state.id,
            )
            if request.last_n_sessions and request.last_n_sessions > 0:
                pending = pending[: request.last_n_sessions]
            return AutoDreamResponse(
                start_date=None,
                end_date=None,
                processed={"procedural": MemoryTypeStats(total=0)},
                last_dream_at=now,
                dry_run=True,
                message=(
                    f"Dry run — {len(pending)} pending session(s) would be claimed "
                    "and distilled; nothing was changed."
                ),
            )

        if run_id is None:
            claim = await AgentTriggerStateManager().check_and_claim_fire(
                agent_id=meta_agent_state.id,
                user_id=user.id,
                organization_id=organization_id,
                trigger="manual",
                force=True,
                allow_empty=True,
                limit=request.last_n_sessions,
                actor_id=getattr(actor, "id", None),
            )
            run_id = claim.run_id
            logger.info(
                "Auto dream (procedural): manual run %s claimed %d session(s) "
                "(meta_agent=%s, user=%s, wait=%s)",
                run_id,
                len(claim.session_ids),
                meta_agent_state.id,
                user.id,
                request.wait,
            )
            if not request.wait:
                from mirix.services.skill_learning_trigger import schedule_learning_run

                # The request travels with the run: its model override must
                # apply in the background exactly as with wait=true.
                schedule_learning_run(
                    run_id,
                    actor=actor,
                    user=user,
                    meta_agent_state=meta_agent_state,
                    request=request,
                )
                return AutoDreamResponse(
                    start_date=None,
                    end_date=None,
                    processed={"procedural": MemoryTypeStats(total=0)},
                    last_dream_at=now,
                    dry_run=False,
                    run_ids=[run_id],
                    message=(
                        f"Skill learning run {run_id} queued with "
                        f"{len(claim.session_ids)} session(s); poll "
                        f"GET /memory/learning_runs/{run_id} for its outcome."
                    ),
                )

        return await self._execute_learning_run(
            run_id=run_id,
            request=request,
            user=user,
            actor=actor,
            meta_agent_state=meta_agent_state,
            now=now,
        )

    async def _execute_learning_run(
        self,
        *,
        run_id: str,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Execute one claimed run under its lease (heartbeat, outcome recording).

        Runs of the same (user, organization) are serialized by
        ``_procedural_dream_guard``: their sessions are disjoint, but their
        curator passes edit the same skill bank and drain the same leftover
        experiences. A run waiting for the guard stays ``queued`` (its heartbeat
        keeps the lease) and becomes ``running`` once it holds the guard.
        """
        from mirix.services.skill_learning_run_manager import (
            LearningRunNotActive,
            SkillLearningRunManager,
        )

        run_mgr = SkillLearningRunManager()
        organization_id = _owner_org(actor)
        run = await run_mgr.get_run(run_id)
        if run is None:
            raise LearningRunNotActive(run_id, None)
        if (
            run.agent_id != meta_agent_state.id
            or run.user_id != user.id
            or run.organization_id != organization_id
        ):
            raise ValueError(
                f"learning run {run_id} belongs to (org={run.organization_id}, "
                f"user={run.user_id}, agent={run.agent_id}), not to this request's scope"
            )

        async def work() -> AutoDreamResponse:
            async with _procedural_dream_guard(user.id, organization_id):
                if not await run_mgr.mark_running(run_id):
                    current = await run_mgr.get_run(run_id)
                    raise LearningRunNotActive(run_id, getattr(current, "status", None))
                return await self._process_learning_run(
                    run=run,
                    request=request,
                    user=user,
                    actor=actor,
                    meta_agent_state=meta_agent_state,
                    now=now,
                )

        return await run_mgr.execute(run_id, work)

    def _llm_config_for(self, meta_agent_state: AgentState, request: AutoDreamRequest):
        """The meta agent's llm_config, with ``request.model`` applied (testing)."""
        llm_config = meta_agent_state.llm_config
        if request.model:
            from copy import deepcopy

            llm_config = deepcopy(llm_config)
            llm_config.model = request.model
        return llm_config

    def _make_distiller(self, *, llm_config, conversation_manager):
        from mirix.services.distill_session_context import SessionContextLoader
        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )

        return SessionExperienceDistiller(
            llm_config=llm_config,
            conversation_manager=conversation_manager,
            # The skills served in each session and its agent context.
            session_context_loader=SessionContextLoader(),
        )

    async def _process_learning_run(
        self,
        *,
        run,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Distill exactly the run's claimed sessions, then evolve skills.

        Steps:
          1. For each claimed session, read only its rows with
             ``created_at <= watermark`` and ``distilled_at IS NULL`` and
             distill them (sessions in parallel) into pending experiences.
          2. Each successfully distilled session (an empty/low-signal one
             included) is committed ATOMICALLY and only while this run is still
             queued/running (``commit_distilled_session``): its experiences are
             inserted and exactly its rows up to the watermark are marked
             distilled in one transaction. Rows written after the claim — a
             reopened session — stay pending; a session that failed, or was not
             committed because the run was cancelled / reclaimed / erased,
             keeps its rows pending with no experiences written, and is
             claimable again once this run ends. So stopping a run never
             duplicates a session's experiences.
          3. A session that could not be distilled is settled: an
             infrastructure error backs it off (``distill_retry_after``, not
             counted); any other failure left after the in-run retries
             (halvings, repair prompts, a timeout retry) counts, and the 3rd
             count sets ``distill_failed_at``. Every retry is appended to the
             run's ``retry_events``.
          4. Record the per-session distill metrics and the LLM usage
             on the run; prune distilled turns past the retention window.
          5. Skill curation: every pending experience of the scope —
             this run's and any an earlier run left pending — oldest first, in
             batches of at most half the curator's input budget; each batch is
             settled and its curator metrics recorded atomically with a check
             that the run is still active. Then the dream checkpoint (again
             only while active).
        """
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
            parse_watermark,
        )
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        run_mgr = SkillLearningRunManager()
        organization_id = _owner_org(actor)
        llm_config = self._llm_config_for(meta_agent_state, request)
        conversation_manager = ConversationMessageManager()
        distiller = self._make_distiller(
            llm_config=llm_config, conversation_manager=conversation_manager
        )

        watermarks: dict = {}
        session_ids: List[str] = []
        for sid in run.session_ids:
            raw = (run.session_watermarks or {}).get(sid)
            if raw is None:
                # Every claim records a watermark; without one we cannot bound
                # the read, so the session is left pending for a later run.
                logger.error(
                    "Learning run %s has no watermark for session %s; skipped",
                    run.id,
                    sid,
                )
                continue
            watermarks[sid] = parse_watermark(raw)
            session_ids.append(sid)

        logger.info(
            "Learning run %s (%s): distilling %d session(s) for meta_agent=%s, user=%s",
            run.id,
            run.trigger,
            len(session_ids),
            meta_agent_state.id,
            user.id,
        )
        session_metrics: dict = {}
        marks: Dict[str, int] = {}
        recorder = RetryEventRecorder.for_learning_run(run.id, run_manager=run_mgr)

        async def commit_session(
            session_id: str,
            payloads: List[dict],
            skill_feedback: Optional[List[dict]] = None,
        ):
            committed, rows_marked = await run_mgr.commit_distilled_session(
                run.id,
                organization_id=organization_id,
                user_id=user.id,
                session_id=session_id,
                watermark=watermarks[session_id],
                experiences=payloads,
                created_by_id=getattr(actor, "id", None),
                skill_feedback=skill_feedback,
            )
            marks[session_id] = rows_marked
            return committed, rows_marked

        async def record_failure(session_id: str, failure) -> Optional[dict]:
            # An infrastructure error backs the session off (never
            # counted); any other failure left after the in-run retries counts,
            # and the 3rd sets distill_failed_at. Both only while the run is
            # still active.
            if failure.deferred:
                retry_after = await run_mgr.defer_distilled_session(
                    run.id,
                    organization_id=organization_id,
                    user_id=user.id,
                    session_id=session_id,
                )
                return {
                    "distill_retry_after": retry_after.isoformat()
                    if retry_after
                    else None
                }
            if failure.counted:
                failures, failed_at = await run_mgr.record_distill_failure(
                    run.id,
                    organization_id=organization_id,
                    user_id=user.id,
                    session_id=session_id,
                )
                return {
                    "distill_failures": failures,
                    "failed_permanently": failed_at is not None,
                }
            return None

        experiences, processed_session_ids = await distiller.distill_sessions(
            meta_agent_state=meta_agent_state,
            user=user,
            actor=actor,
            session_ids=session_ids,
            session_watermarks=watermarks,
            metrics=session_metrics,
            commit_session=commit_session,
            record_failure=record_failure,
            recorder=recorder,
        )

        marked = sum(marks.values())
        logger.info(
            "Learning run %s: marked %d row(s) across %d/%d session(s) distilled "
            "(%d not distilled, left pending)",
            run.id,
            marked,
            len(processed_session_ids),
            len(session_ids),
            len(session_ids) - len(processed_session_ids),
        )
        distill_entries = [
            session_metrics.get(sid, {"session_id": sid}) for sid in session_ids
        ]
        await run_mgr.append_distill_metrics(run.id, distill_entries)
        await run_mgr.add_llm_usage(run.id, _distill_usage(distill_entries))

        # Retention: distilled turns past the window have already yielded their
        # learning into skill_experience rows; drop the verbatim transcripts so
        # the store doesn't accumulate raw conversation content forever.
        from mirix.constants import CONVERSATION_RETENTION_DAYS

        try:
            await conversation_manager.prune_distilled_turns(
                user_id=user.id,
                organization_id=organization_id,
                actor=actor,
                retention_days=CONVERSATION_RETENTION_DAYS,
            )
        except Exception:
            # Best-effort: a failed prune must not fail the run; the next run
            # retries. Logged so a persistent failure is visible.
            logger.exception("Learning run %s: retention prune failed", run.id)

        # -- Skill curation --
        # Every pending experience of the scope — this run's fresh ones and
        # any an earlier run left pending — oldest first, in batches; each
        # batch settles (consumed / returned / kept pending) atomically with a
        # check that this run is still active, and every skill write re-checks
        # it too, so a run that was reclaimed or erased stops changing skills.
        # The sessions stay distilled either way: the experiences are the unit
        # retried, not the sessions.
        await run_mgr.ensure_active(run.id)
        curate_started = dt.datetime.now(dt.timezone.utc).isoformat()
        curation = await self._curate(
            run_id=run.id,
            user=user,
            actor=actor,
            meta_agent_state=meta_agent_state,
            model=request.model,
        )
        await self._record_curation_skip(run.id, curation, curate_started, run_mgr)
        skills_changed = curation.skills_changed
        evolution_changes = {k: list(v) for k, v in curation.changes.items()}

        # A skill the curator (or anything else) stored without
        # a retrieval vector — its embedding still failing after the backoff
        # retries — is invisible to every gated retrieval until it gets one.
        # Re-embed the user's such skills now, in this process, instead of
        # only at the next server start.
        await run_mgr.ensure_active(run.id)
        await self._reembed_missing_vectors(
            run_id=run.id,
            user=user,
            organization_id=organization_id,
            run_mgr=run_mgr,
        )

        # A session MIRIX itself could not distill (a request it could
        # not build) was neither counted nor backed off; the run is recorded
        # failed with it — after the other sessions and the curation — so the
        # failure is loud instead of a silent "succeeded".
        internal = internal_failure_sessions(session_metrics)
        if internal:
            errors = "; ".join(
                f"{sid}: {session_metrics[sid].get('error', '')}"[:300]
                for sid in internal
            )
            raise LearningRunInternalError(
                f"{len(internal)} session(s) not distilled because of MIRIX's own "
                f"failure (left pending, uncounted): {errors}",
                session_ids=internal,
            )

        await run_mgr.ensure_active(run.id)
        await self.write_checkpoint(user, actor, meta_agent_state, now)

        return AutoDreamResponse(
            start_date=None,
            end_date=None,
            processed={"procedural": MemoryTypeStats(total=len(experiences))},
            last_dream_at=now,
            dry_run=False,
            skills_changed=skills_changed,
            changes=evolution_changes,
            run_ids=[run.id],
            message=(
                f"Auto dream (procedural) completed: run {run.id}, "
                f"{len(experiences)} experience(s) from {len(session_ids)} session(s); "
                f"{skills_changed} skill(s) changed."
            ),
        )

    async def _reembed_missing_vectors(
        self,
        *,
        run_id: str,
        user: PydanticUser,
        organization_id: Optional[str],
        run_mgr,
    ) -> None:
        """Give every live skill of ``user`` that has no retrieval vector one,
        with each skill's own embedding config; record the pass on
        the run when it found any. A skill still without one afterwards (the
        embedding calls failed after their retries, or the row has no
        embedding config) is reported as a ``curator_metrics`` entry with
        ``stage: reembed`` — with ``error_kind: embedding`` when embedding
        calls failed: the run's user then has a skill no gated retrieval can
        return, an outage cost learning, visible to whoever reads the run.
        During a sustained outage the pass stops after
        ``MIRIX_SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES`` skills in a row failed
        (``stopped_early``; the others are ``skipped``), so the run is not
        held for minutes of doomed calls. Never raises except cancellation:
        the vectors are retried by the next run and at startup."""
        from mirix import constants
        from mirix.services.retry_policy import recording_retries
        from mirix.services.skill_retrieval_reembed import (
            reembed_missing_retrieval_embeddings,
        )

        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        recorder = RetryEventRecorder.for_learning_run(run_id, run_manager=run_mgr)
        report = None
        error: Optional[BaseException] = None
        try:
            with recording_retries(recorder, stage="reembed"):
                report = await reembed_missing_retrieval_embeddings(
                    organization_id=organization_id,
                    user_id=user.id,
                    max_consecutive_failures=(
                        constants.SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — logged and recorded below
            logger.exception("Learning run %s: re-embedding pass failed", run_id)
            error = exc
        if error is None and (report is None or not report.scanned):
            return  # every skill of the user has its vector
        entry: dict = {
            "batch_id": f"{run_id}:reembed",
            "stage": "reembed",
            "status": "settled",
            "started_at": started_at,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if report is not None:
            entry.update(
                scanned=report.scanned,
                outcomes=dict(report.outcomes),
                filled=report.filled,
                still_missing=report.still_missing,
                stopped_early=report.stopped_early,
            )
            if report.failed:
                entry["error_kind"] = "embedding"
                entry["error"] = (
                    f"{report.failed + report.skipped} skill(s) of the user still "
                    "have no retrieval vector (the embedding calls failed after "
                    "their retries): no gated retrieval returns them until they "
                    "are re-embedded"
                )
                if report.stopped_early:
                    entry["error"] += (
                        "; the pass stopped after "
                        f"{constants.SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES} "
                        f"failures in a row and did not try {report.skipped} "
                        "skill(s) (nor any it had not listed yet)"
                    )
                logger.warning("Learning run %s: %s", run_id, entry["error"])
            if report.no_config:
                logger.warning(
                    "Learning run %s: %d skill(s) of the user have no embedding "
                    "config and can never get a retrieval vector",
                    run_id,
                    report.no_config,
                )
            if not report.still_missing:
                logger.info(
                    "Learning run %s: re-embedded %d skill(s) left without a "
                    "retrieval vector",
                    run_id,
                    report.filled,
                )
        if error is not None:
            entry["error_kind"] = "embedding"
            entry["error"] = f"{type(error).__name__}: {error}"[:500]
        try:
            await run_mgr.append_curator_metrics(run_id, [entry])
        except Exception:  # noqa: BLE001 — bookkeeping must not fail the run
            logger.exception("Learning run %s: could not record the re-embed pass", run_id)

    async def _record_curation_skip(
        self, run_id: str, curation, started_at: str, run_mgr
    ) -> None:
        """Record a curator stage a configuration stopped before any batch
        (the procedural agent lacks a curator tool, the owner has no write
        scope, ...) as a ``curator_metrics`` entry with ``error_kind:
        config``: the pending experiences wait, uncounted, so without it the
        run would read as a normal run that changed nothing."""
        from mirix.services.skill_experience_curator import curation_skip_entry

        entry = curation_skip_entry(run_id, curation, started_at=started_at)
        if entry is None:
            return
        logger.warning("Learning run %s: %s", run_id, entry["error"])
        try:
            await run_mgr.append_curator_metrics(run_id, [entry])
        except Exception:  # noqa: BLE001 — bookkeeping must not fail the run
            logger.exception(
                "Learning run %s: could not record the curation skip", run_id
            )

    async def _curate(
        self,
        *,
        run_id: str,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        model: Optional[str] = None,
    ):
        """The curator stage of learning run ``run_id``: every pending
        experience of the scope, in batches (see
        :func:`mirix.services.skill_experience_curator.curate_pending_experiences`).
        ``model`` overrides the curator's model like it overrides the
        distiller's. Returns a ``CuratorOutcome``; its ``changes`` merge
        every batch's."""
        from mirix.services.skill_experience_curator import (
            curate_pending_experiences,
        )

        return await curate_pending_experiences(
            run_id=run_id,
            user=user,
            actor=actor,
            meta_agent_state=meta_agent_state,
            model_override=model,
        )

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
        run_id: Optional[str] = None,
    ) -> AutoDreamResponse:
        now = dt.datetime.now(dt.timezone.utc)

        # ----------------------------------------------------------------- #
        # mode='procedural' is skill learning: distill a learning run's     #
        # claimed task sessions (only their rows up to the claim-time       #
        # watermark) into pending SkillExperience rows, then drive skill    #
        # evolution from them. `run_id` executes an already-claimed run     #
        # (the trigger's background task); without it, this call claims    #
        # every pending session of the scope itself (flush semantics).      #
        # Other modes run the generic fetch → format → single-step path,    #
        # recorded as a learning run of their mode unless dry_run.          #
        # ----------------------------------------------------------------- #
        if request.mode == "procedural":
            return await self._run_procedural(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
                run_id=run_id,
            )
        if request.dry_run:
            # A dry run changes nothing and is not recorded.
            return await self._run_generic(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
            )
        return await self._run_recorded_generic(
            request=request,
            user=user,
            actor=actor,
            meta_agent_state=meta_agent_state,
            now=now,
            run_id=run_id,
        )

    async def _run_recorded_generic(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
        run_id: Optional[str],
    ) -> AutoDreamResponse:
        """A non-procedural auto_dream, recorded as a ``skill_learning_run``.

        Without ``run_id`` a ``queued`` run (``mode=request.mode``, trigger
        'manual', no sessions) is recorded first; ``request.wait`` (default)
        executes it here under its lease and returns its result with
        ``run_ids``, ``wait=False`` schedules it in the background and returns
        its id at once. With ``run_id`` (the background task) that recorded run
        is executed.
        """
        from mirix.services.skill_learning_run_manager import (
            LearningRunNotActive,
            SkillLearningRunManager,
        )

        run_mgr = SkillLearningRunManager()
        organization_id = _owner_org(actor)
        if run_id is None:
            run = await run_mgr.create_standalone_run(
                organization_id=organization_id,
                user_id=user.id,
                agent_id=meta_agent_state.id,
                mode=request.mode,
                trigger="manual",
                actor_id=getattr(actor, "id", None),
            )
            run_id = run.id
            if not request.wait:
                from mirix.services.skill_learning_trigger import schedule_learning_run

                schedule_learning_run(
                    run_id,
                    actor=actor,
                    user=user,
                    meta_agent_state=meta_agent_state,
                    request=request,
                )
                return AutoDreamResponse(
                    start_date=None,
                    end_date=None,
                    processed={},
                    last_dream_at=now,
                    dry_run=False,
                    run_ids=[run_id],
                    message=(
                        f"Auto dream ({request.mode}) run {run_id} queued; poll "
                        f"GET /memory/learning_runs/{run_id} for its outcome."
                    ),
                )
        else:
            run = await run_mgr.get_run(run_id)
            if run is None:
                raise LearningRunNotActive(run_id, None)
            if (
                run.mode != request.mode
                or run.agent_id != meta_agent_state.id
                or run.user_id != user.id
                or run.organization_id != organization_id
            ):
                raise ValueError(
                    f"learning run {run_id} is a {run.mode!r} run of (org="
                    f"{run.organization_id}, user={run.user_id}, agent={run.agent_id}), "
                    f"not this request's {request.mode!r} run"
                )

        recorded_run_id = run_id

        async def work() -> AutoDreamResponse:
            if not await run_mgr.mark_running(recorded_run_id):
                current = await run_mgr.get_run(recorded_run_id)
                raise LearningRunNotActive(
                    recorded_run_id, getattr(current, "status", None)
                )
            return await self._run_generic(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
            )

        response = await run_mgr.execute(recorded_run_id, work)
        response.run_ids = [recorded_run_id]
        return response

    async def _run_generic(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Fetch the mode's memories -> one consolidation agent step -> checkpoint."""
        from mirix.server.server import get_server

        server = get_server()

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
        # graph_only unset: graph-only refinement in the graph pipeline
        # (settings.graph_pipeline), the LLM memory-merge pass outside it.
        graph_only = (
            settings.graph_pipeline if request.graph_only is None else request.graph_only
        )
        if graph_only:
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
