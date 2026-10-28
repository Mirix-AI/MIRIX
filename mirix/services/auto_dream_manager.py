"""
AutoDreamManager: orchestrates the auto_dream self-reflection pipeline.

Flow:
  1. Resolve time window (last dream checkpoint → now)
  2. Fetch memories for each requested type
  3. Format memories into a structured message
  4. Invoke AutoDreamAgent via step()
  5. Write a checkpoint episodic memory entry
  6. Return stats
"""

import asyncio
import contextlib
import datetime as dt
import json
import logging
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
from mirix.services.retry_policy import RetryEventRecorder
from mirix.services.session_experience_distiller import internal_failure_sessions

logger = logging.getLogger(__name__)

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
        return f.read()


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
        """Return occurred_at of the most recent auto_dream checkpoint, or None."""
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

    async def _fetch_episodic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

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
            limit=500,
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
            limit=500,
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
            limit=500,
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
            limit=500,
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
            limit=500,
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

        # -- build input message for the agent --
        payload = _format_memories_as_message(memories, start_date, end_date, request.mode)

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

        # -- load and run agent --
        dream_agent = await server.load_agent(
            agent_id=dream_agent_state.id,
            actor=actor,
            user=user,
            use_cache=False,
        )
        input_msg = MessageCreate(role=MessageRole.user, content=payload)
        await dream_agent.step(input_messages=input_msg, actor=actor, user=user)

        # -- write checkpoint --
        await self.write_checkpoint(user, actor, meta_agent_state, now)

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
