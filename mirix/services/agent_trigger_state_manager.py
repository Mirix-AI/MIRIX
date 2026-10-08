"""Skill-learning trigger claim.

``AgentTriggerStateManager.check_and_claim_fire`` decides, for one
(organization, user, meta agent) scope, whether a skill learning run starts
now, and if so claims its sessions — atomically, so concurrent checks (several
task writes, a close, the sweeper, a flush; in one process or several) start at
most one run per batch of sessions and two live runs never share a session.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from mirix.helpers.keyed_locks import KeyedLocks
from mirix.log import get_logger
from mirix.orm.agent_trigger_state import AgentTriggerState as AgentTriggerStateModel
from mirix.schemas.agent_trigger_state import (
    TRIGGER_TYPE_PROCEDURAL_SKILL,
    AgentTriggerState as PydanticAgentTriggerState,
    _validate_trigger_type,
)
from mirix.services.conversation_session_manager import (
    Clock,
    ConversationSessionManager,
    as_utc,
    dialect_name,
    utc_now,
)

logger = get_logger(__name__)

# Process-local serialization of claims per scope (saves database lock waits).
# Correctness does not depend on it: on Postgres the scope's cursor row
# (SELECT ... FOR UPDATE) serializes claims across processes and the session
# rows (FOR UPDATE) serialize a claim with writes; on SQLite, which ignores FOR
# UPDATE, the claim opens its transaction with a write, so it holds the
# database write lock for its whole duration and every concurrent write (a
# session write, another claim) waits for it.
_SCOPE_LOCKS = KeyedLocks()


@dataclass
class ClaimFireResult:
    """Outcome of a `check_and_claim_fire` call.

    fired          -- True iff a learning run was created (status 'queued'); the
                      caller schedules its execution.
    run_id         -- The created run's id (None when not fired).
    session_ids    -- The sessions the run claimed, oldest first.
    pending_count  -- Claimable pending sessions seen at decision time (pending
                      task sessions of the scope minus those claimed by any
                      queued/running run).
    pending_experiences -- With ``force`` (a flush): the scope's experiences
                      still waiting for the curator at decision time.
    state          -- The scope's cursor row after the call (diagnostics).
    """

    fired: bool
    run_id: Optional[str] = None
    session_ids: List[str] = field(default_factory=list)
    pending_count: int = 0
    pending_experiences: int = 0
    state: Optional[PydanticAgentTriggerState] = None


def scope_lock_key(organization_id: str, user_id: str, agent_id: str) -> str:
    return f"{organization_id}::{user_id}::{agent_id}"


# A claim transaction that hits a transient database conflict is retried from
# scratch ("database: deadlock / serialization failure").
_CLAIM_ATTEMPTS = 5
_RETRYABLE_DB_MESSAGES = (
    "deadlock detected",
    "could not serialize access",
    "lock not available",
    "database is locked",
)


def _is_retryable_db_error(exc: BaseException) -> bool:
    from sqlalchemy.exc import DBAPIError

    if not isinstance(exc, DBAPIError):
        return False
    message = str(exc).lower()
    return any(fragment in message for fragment in _RETRYABLE_DB_MESSAGES)


class AgentTriggerStateManager:
    """Owns the per-(meta agent, user) cursor row and the trigger claim.

    The cursor row (``agent_trigger_state``, one per agent/user/trigger type)
    is the scope's lock: every claim first locks it ``SELECT ... FOR UPDATE``.
    Its fields are diagnostics only (when the scope last fired and which
    sessions that run claimed); the count is always derived live from
    ``conversation_session`` / ``conversation_message`` / ``skill_learning_run``.

    The claim (one transaction, under the lock):

    0. reclaim the user's learning runs whose lease expired (a crashed run's
       sessions must not stay excluded until the next sweep), and attribute
       the user's legacy agent-less sessions this scope owns;
    1. materialize idle sessions of the scope as ended (conditional UPDATEs);
    2. select the scope's pending task sessions (``SELECT ... FOR UPDATE`` on
       their session rows, so a concurrent write to one of them waits for the
       claim and then reopens it with rows newer than the watermark) and drop
       those claimed by ANY queued/running run — the union of all live claims,
       not just the last one;
    3. fire iff the count reaches the threshold (strict — no
       wait-time rule), or on a flush (``force``) iff anything is pending;
    4. on fire, record each claimed session's watermark (its newest row's
       ``created_at``) and insert the ``queued`` ``skill_learning_run`` row.

    The caller schedules the run's execution after the commit (see
    ``skill_learning_trigger``).
    """

    def __init__(
        self,
        *,
        session_maker=None,
        clock: Optional[Clock] = None,
        scope_locks: Optional[KeyedLocks] = None,
    ):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        # None = real time (the run manager then uses the database clock for
        # lease timestamps); an injected clock is passed through as-is.
        self._injected_clock = clock
        self.clock: Clock = clock or utc_now
        self._scope_locks = scope_locks if scope_locks is not None else _SCOPE_LOCKS

    def _session_manager(self) -> ConversationSessionManager:
        return ConversationSessionManager(
            session_maker=self.session_maker, clock=self.clock
        )

    async def check_and_claim_fire(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        trigger: str = "threshold",
        threshold: Optional[int] = None,
        force: bool = False,
        allow_empty: bool = False,
        limit: Optional[int] = None,
        current_session_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        trigger_type: str = TRIGGER_TYPE_PROCEDURAL_SKILL,
    ) -> ClaimFireResult:
        """Decide and claim, atomically (see the class docstring).

        Args:
            agent_id / user_id / organization_id: the scope (meta memory agent).
            trigger: recorded on the run — 'threshold' (after a task write),
                'close', 'sweep', 'flush' or 'manual'.
            threshold: sessions needed to fire; defaults to
                ``SKILL_TRIGGER_SESSION_THRESHOLD``. Ignored with ``force``.
            force: flush semantics — claim every claimable pending session,
                however few. With no pending session, a run is still
                created (without sessions) when the scope has experiences
                waiting for the curator — left by a run that crashed after its
                distill commit, stopped on an infrastructure error or failed —
                and no queued/running run of the scope would curate them
                anyway (the curator's leftovers are retried at the next
                trigger, and a flush is one).
            allow_empty: with ``force``, create the run even when nothing is
                pending (a manual ``/memory/auto_dream`` records every run).
            limit: claim at most this many sessions (oldest first).
            current_session_id: diagnostics only.
            actor_id: stamped as the run's creator.
        """
        _validate_trigger_type(trigger_type)
        if threshold is None:
            from mirix.constants import SKILL_TRIGGER_SESSION_THRESHOLD

            threshold = SKILL_TRIGGER_SESSION_THRESHOLD
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if limit is not None and limit <= 0:
            limit = None

        async with self._scope_locks.acquire(
            scope_lock_key(organization_id, user_id, agent_id)
        ):
            await self._ensure_cursor(
                agent_id=agent_id,
                user_id=user_id,
                trigger_type=trigger_type,
                organization_id=organization_id,
            )
            for attempt in range(1, _CLAIM_ATTEMPTS + 1):
                try:
                    return await self._claim_once(
                        agent_id=agent_id,
                        user_id=user_id,
                        organization_id=organization_id,
                        trigger=trigger,
                        threshold=threshold,
                        force=force,
                        allow_empty=allow_empty,
                        limit=limit,
                        current_session_id=current_session_id,
                        actor_id=actor_id,
                        trigger_type=trigger_type,
                    )
                except Exception as exc:
                    if attempt == _CLAIM_ATTEMPTS or not _is_retryable_db_error(exc):
                        raise
                    delay = min(0.05 * (2 ** (attempt - 1)), 1.0)
                    logger.warning(
                        "Skill-learning claim for (agent=%s, user=%s) hit a transient "
                        "database error (attempt %d/%d), retrying in %.2fs: %s",
                        agent_id,
                        user_id,
                        attempt,
                        _CLAIM_ATTEMPTS,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            raise AssertionError("unreachable")  # pragma: no cover

    async def _claim_once(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        trigger: str,
        threshold: int,
        force: bool,
        allow_empty: bool,
        limit: Optional[int],
        current_session_id: Optional[str],
        actor_id: Optional[str],
        trigger_type: str,
    ) -> ClaimFireResult:
        """One claim transaction (see the class docstring); rolled back on error."""
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        session_mgr = self._session_manager()
        message_mgr = ConversationMessageManager(
            session_maker=self.session_maker, clock=self.clock
        )
        run_mgr = SkillLearningRunManager(
            session_maker=self.session_maker, clock=self._injected_clock
        )

        async with self.session_maker() as db:
            if dialect_name(db) == "sqlite":
                # SQLite ignores FOR UPDATE and pysqlite begins the transaction
                # lazily at the first write, so every read below would see its
                # own snapshot and take no lock. A no-op write first makes this
                # transaction hold the database write lock throughout: a
                # concurrent session write waits for the claim (and then
                # reopens the session with rows newer than the watermark).
                await db.execute(
                    update(AgentTriggerStateModel)
                    .where(
                        AgentTriggerStateModel.agent_id == agent_id,
                        AgentTriggerStateModel.user_id == user_id,
                        AgentTriggerStateModel.trigger_type == trigger_type,
                    )
                    .values(updated_at=AgentTriggerStateModel.updated_at)
                    .execution_options(synchronize_session=False)
                )
            cursor = (
                await db.execute(
                    select(AgentTriggerStateModel)
                    .where(
                        AgentTriggerStateModel.agent_id == agent_id,
                        AgentTriggerStateModel.user_id == user_id,
                        AgentTriggerStateModel.trigger_type == trigger_type,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()

            now = as_utc(self.clock())
            await run_mgr.recover_expired_leases(
                db=db, organization_id=organization_id, user_id=user_id
            )
            await session_mgr.adopt_unattributed_sessions(
                db=db,
                organization_id=organization_id,
                user_id=user_id,
                agent_id=agent_id,
            )
            await session_mgr.materialize_idle(
                db=db,
                organization_id=organization_id,
                user_id=user_id,
                agent_id=agent_id,
                now=now,
            )
            pending = await session_mgr.list_pending_distill_sessions(
                organization_id=organization_id,
                user_id=user_id,
                agent_id=agent_id,
                db=db,
                now=now,
                exclude_claimed=True,
                for_update=True,
            )
            pending_count = len(pending)
            pending_experiences = 0
            curate_only = False
            if force:
                pending_experiences = await self._count_pending_experiences(
                    db,
                    organization_id=organization_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                if (
                    pending_count == 0
                    and not allow_empty
                    and pending_experiences > 0
                    and not await self._has_active_run(
                        db,
                        organization_id=organization_id,
                        user_id=user_id,
                        agent_id=agent_id,
                    )
                ):
                    curate_only = True
                fire = pending_count > 0 or allow_empty or curate_only
            else:
                fire = pending_count >= threshold

            if not fire:
                await db.commit()
                return ClaimFireResult(
                    fired=False,
                    pending_count=pending_count,
                    pending_experiences=pending_experiences,
                    state=cursor.to_pydantic(),
                )

            claimed = pending[:limit] if limit else list(pending)
            watermarks = await message_mgr.session_watermarks(
                db,
                user_id=user_id,
                organization_id=organization_id,
                session_ids=claimed,
            )
            # A session always has a live row here (pending requires one);
            # keep only sessions with a watermark so run and marks agree.
            claimed = [sid for sid in claimed if sid in watermarks]
            if not claimed and not (force and (allow_empty or curate_only)):
                await db.commit()
                return ClaimFireResult(
                    fired=False,
                    pending_count=pending_count,
                    pending_experiences=pending_experiences,
                    state=cursor.to_pydantic(),
                )
            run = await run_mgr.create_run(
                db,
                organization_id=organization_id,
                user_id=user_id,
                agent_id=agent_id,
                trigger=trigger,
                session_ids=claimed,
                session_watermarks={sid: watermarks[sid] for sid in claimed},
                actor_id=actor_id,
            )
            cursor.last_fired_at = now
            cursor.last_fired_session_id = current_session_id
            cursor.last_fired_tied_session_ids = list(claimed)
            cursor.set_updated_at(now)
            run_id = run.id
            state = cursor.to_pydantic()
            # Nothing runs after the commit, so a retry can never re-run a claim
            # that already committed.
            await db.commit()
            logger.info(
                "Skill learning run %s claimed %d session(s) (trigger=%s, "
                "pending=%d, threshold=%d, agent=%s, user=%s)",
                run_id,
                len(claimed),
                trigger,
                pending_count,
                threshold,
                agent_id,
                user_id,
            )
            return ClaimFireResult(
                fired=True,
                run_id=run_id,
                session_ids=list(claimed),
                pending_count=pending_count,
                pending_experiences=pending_experiences,
                state=state,
            )

    @staticmethod
    async def _count_pending_experiences(
        db, *, organization_id: str, user_id: str, agent_id: str
    ) -> int:
        """The scope's experiences waiting for the curator (in ``db``)."""
        from mirix.orm.skill_experience import SkillExperience as SkillExperienceModel

        return int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(SkillExperienceModel)
                    .where(
                        SkillExperienceModel.organization_id == organization_id,
                        SkillExperienceModel.user_id == user_id,
                        SkillExperienceModel.agent_id == agent_id,
                        SkillExperienceModel.status == "pending",
                        SkillExperienceModel.is_deleted.is_(False),
                    )
                )
            ).scalar_one()
        )

    @staticmethod
    async def _has_active_run(
        db, *, organization_id: str, user_id: str, agent_id: str
    ) -> bool:
        """Whether a queued/running learning run of the scope exists (its
        curator will drain the scope's pending experiences anyway)."""
        from mirix.orm.skill_learning_run import SkillLearningRun as RunModel

        return (
            await db.execute(
                select(RunModel.id)
                .where(
                    RunModel.organization_id == organization_id,
                    RunModel.user_id == user_id,
                    RunModel.agent_id == agent_id,
                    RunModel.status.in_(("queued", "running")),
                    RunModel.is_deleted.is_(False),
                )
                .limit(1)
            )
        ).first() is not None

    async def count_pending_sessions(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
    ) -> int:
        """Diagnostic: claimable pending sessions of the scope right now (read-only)."""
        pending = await self._session_manager().list_pending_distill_sessions(
            organization_id=organization_id,
            user_id=user_id,
            agent_id=agent_id,
            now=as_utc(self.clock()),
        )
        return len(pending)

    async def _ensure_cursor(
        self,
        *,
        agent_id: str,
        user_id: str,
        trigger_type: str,
        organization_id: Optional[str],
    ) -> None:
        """Insert the scope's cursor row if missing (idempotent under races)."""
        async with self.session_maker() as session:
            existing = await self._fetch(
                session, agent_id, user_id, trigger_type, include_deleted=True
            )
            if existing is not None:
                if existing.is_deleted:
                    # The unique key covers soft-deleted rows too: revive it.
                    existing.is_deleted = False
                    await session.commit()
                return
            session.add(
                AgentTriggerStateModel(
                    agent_id=agent_id,
                    user_id=user_id,
                    trigger_type=trigger_type,
                    organization_id=organization_id,
                    last_fired_at=None,
                    last_fired_session_id=None,
                    last_fired_tied_session_ids=[],
                )
            )
            try:
                await session.commit()
            except IntegrityError as exc:
                # Usually another worker installed it first and its row serves
                # as the lock. Otherwise (e.g. the meta agent or user no longer
                # exists) there is no scope to claim for: say so plainly.
                await session.rollback()
                if await self._fetch(session, agent_id, user_id, trigger_type) is None:
                    raise LookupError(
                        f"cannot install the trigger cursor for agent {agent_id} / "
                        f"user {user_id}: {exc.orig}"
                    ) from exc

    async def _fetch(
        self,
        session,
        agent_id: str,
        user_id: str,
        trigger_type: str,
        *,
        include_deleted: bool = False,
    ) -> Optional[AgentTriggerStateModel]:
        filters = [
            AgentTriggerStateModel.agent_id == agent_id,
            AgentTriggerStateModel.user_id == user_id,
            AgentTriggerStateModel.trigger_type == trigger_type,
        ]
        if not include_deleted:
            filters.append(AgentTriggerStateModel.is_deleted.is_(False))
        stmt = select(AgentTriggerStateModel).where(*filters)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
