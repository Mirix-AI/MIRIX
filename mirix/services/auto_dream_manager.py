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

from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.auto_dream import AutoDreamRequest, AutoDreamResponse, MemoryTypeStats
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.user import User as PydanticUser

logger = logging.getLogger(__name__)

# event_type used to tag auto_dream checkpoint records in episodic memory
_CHECKPOINT_EVENT_TYPE = "auto_dream_checkpoint"

# Per-(user, org) serialization for the procedural auto-dream. Two fires can be
# scheduled (fire-and-forget) before the first has marked its sessions distilled;
# if both ran concurrently they would each enumerate the SAME oldest sealed
# sessions and distill duplicate experiences. Serializing per owner makes the
# second run wait until the first has marked its batch distilled, so the second
# then enumerates the NEXT (disjoint) batch.
#
# Two layers:
#   * a process-local asyncio.Lock (serializes coroutines within ONE event loop), AND
#   * a Postgres SESSION-level ADVISORY lock keyed by hash(user, org), held across
#     the whole run, so concurrent gunicorn/uvicorn WORKER PROCESSES also serialize.
# On non-Postgres backends (SQLite / PGlite — single-process test setups with no
# advisory locks) the asyncio.Lock alone is sufficient, so the advisory layer is
# skipped gracefully.
_PROCEDURAL_DREAM_LOCKS: Dict[str, asyncio.Lock] = {}

# Stable namespace salt so this feature's advisory-lock keys never collide with
# another feature's pg_advisory_lock keyspace.
_PROCEDURAL_ADVISORY_NAMESPACE = "mirix.procedural_dream"


def _owner_org(actor: PydanticClient) -> str:
    """Resolve the organization the Conversation Message Store was written under.

    Ingestion (rest_api ``/memory/add(_sync)``) records turns under
    ``client.organization_id or DEFAULT_ORG_ID`` (the store column is NOT NULL).
    Every read/mark of that store MUST mirror the same fallback, otherwise a
    NULL-org client would write to DEFAULT_ORG_ID but read from ``None`` and its
    sessions would never distill.
    """
    from mirix.constants import DEFAULT_ORG_ID

    return actor.organization_id or DEFAULT_ORG_ID


def _procedural_dream_lock(user_id: str, organization_id: Optional[str]) -> asyncio.Lock:
    """Return the process-local lock that serializes procedural auto-dream runs
    for one (user, organization). Created lazily; keyed so different owners never
    block each other."""
    key = f"{organization_id or '-'}::{user_id}"
    lock = _PROCEDURAL_DREAM_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _PROCEDURAL_DREAM_LOCKS[key] = lock
    return lock


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
    """
    from sqlalchemy import text

    from mirix.server.server import db_context

    key = _advisory_lock_key(user_id, organization_id)
    async with _procedural_dream_lock(user_id, organization_id):
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
        except Exception:  # noqa: BLE001 — lock plumbing must never crash the run
            logger.warning(
                "Procedural dream advisory lock unavailable; relying on the "
                "process-local lock only.",
                exc_info=True,
            )
            if session_cm is not None:
                try:
                    await session_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            session_cm = None
            session = None

        try:
            yield
        finally:
            if session is not None:
                try:
                    await session.execute(
                        text("SELECT pg_advisory_unlock(:k)"), {"k": key}
                    )
                except Exception:  # noqa: BLE001 — close releases it anyway
                    logger.debug(
                        "pg_advisory_unlock failed; session close will release it."
                    )
                finally:
                    try:
                        await session_cm.__aexit__(None, None, None)
                    except Exception:  # noqa: BLE001
                        pass

# Generic "fetch current memories of these components -> one consolidation agent"
# modes. mode='procedural' is intentionally NOT here: it is handled by the
# dedicated session-experience path (_run_procedural_experience), which returns
# before this mapping is ever consulted.
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
        await mgr.create_episodic_memory(episodic_memory=checkpoint, actor=actor)

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
            limit=500,
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
    # Goal 2/3 — general session-experience procedural path                #
    # ------------------------------------------------------------------ #

    async def _run_procedural_experience(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Serialize per (user, org), then run the procedural distillation.

        The guard makes the enumerate → distill → mark sequence atomic against
        another procedural run for the SAME owner — within the process (asyncio
        lock) AND across worker processes (Postgres advisory lock). A second fire
        scheduled before this one marks its batch distilled waits here, then
        enumerates the NEXT (disjoint) sealed batch — preventing duplicate
        distillation of the oldest sessions. See ``_procedural_dream_guard``.
        """
        async with _procedural_dream_guard(user.id, _owner_org(actor)):
            return await self._run_procedural_experience_locked(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
            )

    async def _run_procedural_experience_locked(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Goal-2 distillation (+ Goal-3 evolution unless dry_run).

        This is the explicit "program call" entry point for general
        session-experience distillation: calling
        ``AutoDreamManager().run(AutoDreamRequest(mode='procedural', ...))``
        runs it directly. It is ALSO fired every N sessions by the
        ``trigger_memory_update`` claim path (see
        ``functions/function_sets/memory_tools.py``).

        Steps:
          1. Resolve N (``request.last_n_sessions`` or
             ``MESSAGE_RETAIN_LAST_N_SESSIONS``) — the max number of sealed
             sessions to process this round.
          2. Enumerate up to N SEALED, not-yet-distilled sessions from the
             Conversation Message Store (oldest-first; the open head of the
             window is never included).
          3. Distill each session IN PARALLEL into pending SkillExperience rows.
          4. Mark the SUCCESSFULLY-processed sessions distilled so the rolling
             barrier advances and they are never re-distilled (failed sessions
             are left for retry).
          5. dry_run → return counts; else run Goal-3 skill evolution over the
             freshly-pending experiences, then write the dream checkpoint.

        Always invoked under the per-owner lock acquired by
        ``_run_procedural_experience``.
        """
        from mirix.constants import MESSAGE_RETAIN_LAST_N_SESSIONS
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )
        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )

        n = request.last_n_sessions or MESSAGE_RETAIN_LAST_N_SESSIONS

        # The dream agent inherits the meta agent's llm_config; allow a model
        # override for testing (same convention as the generic path).
        llm_config = meta_agent_state.llm_config
        if request.model:
            from copy import deepcopy

            llm_config = deepcopy(llm_config)
            llm_config.model = request.model

        conversation_manager = ConversationMessageManager()
        distiller = SessionExperienceDistiller(
            llm_config=llm_config,
            conversation_manager=conversation_manager,
        )
        # Source of truth is the Conversation Message Store, scoped per
        # (user, organization). Pull at most N sealed, not-yet-distilled
        # sessions (oldest-first); the in-progress head is left for the next
        # window.
        session_ids = await distiller.enumerate_sealed_sessions(
            user_id=user.id,
            organization_id=_owner_org(actor),
            actor=actor,
            limit=n,
        )
        logger.info(
            "Auto dream (procedural): meta_agent=%s, user=%s, last_n=%d, sealed_sessions=%s",
            meta_agent_state.id,
            user.id,
            n,
            session_ids,
        )

        if request.dry_run:
            # A dry run has NO side effects: it does NOT distill (no LLM call, no
            # SkillExperience writes), does NOT mark sessions distilled, does NOT
            # evolve skills, and does NOT checkpoint. It reports how many sealed
            # sessions WOULD be processed, leaving the rolling barrier untouched
            # so a subsequent real run processes the same window.
            return AutoDreamResponse(
                start_date=None,
                end_date=None,
                processed={"procedural": MemoryTypeStats(total=0)},
                last_dream_at=now,
                dry_run=True,
                message=(
                    f"Dry run — {len(session_ids)} sealed session(s) would be "
                    f"distilled; no experiences persisted, barrier not advanced."
                ),
            )

        existing_skills = await self._existing_skill_summaries(user, meta_agent_state)
        # distill_sessions returns (experiences, processed_session_ids): only the
        # SUCCESSFULLY-processed sessions (including legitimately-empty ones) are
        # in processed_session_ids; a session that hit an operational failure is
        # omitted so we never advance the barrier past a failed conversation.
        experiences, processed_session_ids = await distiller.distill_sessions(
            meta_agent_state=meta_agent_state,
            user=user,
            actor=actor,
            session_ids=session_ids,
            existing_skills=existing_skills,
        )

        stats = {
            "procedural": MemoryTypeStats(total=len(experiences)),
        }

        # Advance the rolling barrier: mark every SUCCESSFULLY-processed session
        # distilled (idempotent; scoped to this (user, org)) so a later round —
        # automatic or explicit — never re-distills them. Sessions that yielded
        # zero experiences but were processed cleanly are still marked (an
        # empty/low-signal conversation is "consumed" and must not block the
        # window). Sessions that FAILED are intentionally left undistilled so a
        # later round retries them rather than losing their learning.
        if processed_session_ids:
            marked = await conversation_manager.mark_sessions_distilled(
                session_ids=processed_session_ids,
                user_id=user.id,
                organization_id=_owner_org(actor),
                actor=actor,
            )
            logger.info(
                "Auto dream (procedural): marked %d turn(s) across %d/%d session(s) "
                "distilled (%d failed, left for retry)",
                marked,
                len(processed_session_ids),
                len(session_ids),
                len(session_ids) - len(processed_session_ids),
            )

        # -- Goal 3: evolve skills from the pending experiences. --
        #
        # Two independent steps, so a prior round's stranded experiences are
        # drained even under active traffic (closing the "evolution failure
        # strands distilled experiences" gap):
        #   (1) FRESH round: evolve THIS round's freshly-distilled batch, scoped to
        #       its own ids (the deliberate optimization — earlier rounds'
        #       low-signal experiences would dilute the curator prompt).
        #   (2) RECOVERY: drain any LEFTOVER pending experiences from PRIOR rounds
        #       whose evolution failed (leftover pending minus this round's fresh
        #       ids), in its own scoped run. Always attempted, regardless of
        #       whether this round produced fresh experiences.
        # The sessions stay distilled either way — re-distilling would duplicate;
        # the experiences are the unit retried, not the sessions.
        skills_changed = 0
        evolution_changes: dict = {}
        fresh_ids = [e.id for e in experiences]
        leftover_ids = await self._leftover_pending_experience_ids(
            meta_agent_state=meta_agent_state, user=user
        )
        # Exclude this round's fresh ids from the recovery scope: they are handled
        # by step (1), so the recovery step only sees PRIOR rounds' leftovers.
        fresh_set = set(fresh_ids)
        recovery_ids = [eid for eid in leftover_ids if eid not in fresh_set]

        if fresh_ids:
            sc, ch = await self._evolve_experiences(
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                experience_ids=fresh_ids,
                label="fresh",
            )
            skills_changed += sc
            evolution_changes.update(ch)
        if recovery_ids:
            sc, ch = await self._evolve_experiences(
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                experience_ids=recovery_ids,
                label="recovery",
            )
            skills_changed += sc
            # Don't clobber fresh changes with recovery ones — merge keys.
            for k, v in ch.items():
                evolution_changes.setdefault(k, v)

        await self.write_checkpoint(user, actor, meta_agent_state, now)

        return AutoDreamResponse(
            start_date=None,
            end_date=None,
            processed=stats,
            last_dream_at=now,
            dry_run=False,
            skills_changed=skills_changed,
            changes=evolution_changes,
            message=(
                f"Auto dream (procedural) completed: {len(experiences)} experience(s) "
                f"from {len(session_ids)} session(s); {skills_changed} skill(s) changed."
            ),
        )

    async def _evolve_experiences(
        self,
        *,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        experience_ids: List[str],
        label: str,
    ) -> tuple:
        """Run Goal-3 skill evolution over an explicit, scoped experience batch.

        Returns ``(skills_changed, changes)``; on any failure returns ``(0, {})``
        and logs — the experiences are durably `pending`, so a failed evolution
        loses no learning and is retried by the RECOVERY step on a later run.
        ``label`` ("fresh" / "recovery") only colours the log.
        """
        if not experience_ids:
            return 0, {}
        try:
            from mirix.services.skill_experience_curator import (
                run_experience_evolution,
            )

            evolution = await run_experience_evolution(
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                experience_ids=experience_ids,
            )
            skills_changed = (evolution or {}).get("skills_changed", 0)
            changes = (evolution or {}).get("changes", {}) or {}
            if label == "recovery":
                logger.info(
                    "Auto dream (procedural): recovered %d stranded pending "
                    "experience(s); %d skill(s) changed.",
                    len(experience_ids),
                    skills_changed,
                )
            return skills_changed, changes
        except ImportError:
            # Goal-3 curator not yet present in this build — distillation alone is
            # still valuable, so do not fail the run.
            logger.info(
                "Goal-3 experience curator unavailable; experiences persisted as pending."
            )
            return 0, {}
        except Exception as e:  # noqa: BLE001 — evolution failure mustn't lose experiences
            # Experiences are durably `pending`, so NO learning is lost — only the
            # evolution STEP failed. The sessions stay distilled (re-distilling
            # would duplicate); these pending experiences are drained by the
            # RECOVERY step on a later procedural run. Logged LOUD for observability.
            logger.error(
                "Goal-3 experience evolution (%s) FAILED for meta_agent=%s "
                "(%d experience(s) left pending; auto-recovered on a later procedural "
                "run, or via POST /memory/auto_dream mode=procedural): %s",
                label,
                meta_agent_state.id,
                len(experience_ids),
                e,
                exc_info=True,
            )
            return 0, {}

    async def _leftover_pending_experience_ids(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
    ) -> List[str]:
        """Return ids of this (agent, user)'s still-`pending` experiences.

        Used by the RECOVERY path: when a procedural run distilled nothing fresh,
        these are the experiences a PRIOR round's failed evolution left behind
        (``list_experiences(status='pending')`` already excludes consumed /
        superseded, so this is exactly the un-evolved leftover pool). Best-effort:
        any failure yields ``[]`` so recovery is simply skipped this round.
        """
        try:
            from mirix.services.skill_experience_manager import SkillExperienceManager

            pending = await SkillExperienceManager().list_experiences(
                agent_id=meta_agent_state.id,
                user_id=user.id,
                status="pending",
            )
        except Exception as e:  # noqa: BLE001 — recovery is best-effort
            logger.debug("Could not list leftover pending experiences: %s", e)
            return []
        return [e.id for e in pending]

    async def _existing_skill_summaries(
        self,
        user: PydanticUser,
        meta_agent_state: AgentState,
    ) -> list:
        """Return [{name, description}] of existing procedural skills (dedup context).

        Best-effort: a failure here just yields an empty list (the distiller
        prompt treats it as "(none)").
        """
        try:
            procedures = await self._fetch_procedural(user, meta_agent_state, None, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not fetch existing skills for dedup context: %s", e)
            return []
        summaries = []
        for p in procedures:
            summaries.append(
                {
                    "name": getattr(p, "name", "") or "",
                    "description": getattr(p, "description", "") or "",
                }
            )
        return summaries

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

        # ----------------------------------------------------------------- #
        # Goal 2/3 — mode='procedural' is general per-session experience    #
        # distillation (NOT the legacy procedure-consolidation pass). It    #
        # reads the meta agent's last-N RETAINED sessions, distills each    #
        # session's transcript IN PARALLEL into pending SkillExperience     #
        # rows, then (Goal 3, unless dry_run) drives skill self-evolution   #
        # from those experiences. Other modes fall through to the generic   #
        # fetch → format → single-step path below, unchanged.               #
        # ----------------------------------------------------------------- #
        if request.mode == "procedural":
            return await self._run_procedural_experience(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
            )

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
