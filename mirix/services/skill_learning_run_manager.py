"""Skill learning runs: records, lease heartbeat, recovery, metrics.

A learning run is one execution of the skill learner (distill the claimed
sessions -> curate skills) for one (organization, user, meta agent) scope.

Lifecycle::

    queued --(mark_running)--> running --(succeed)--> succeeded
       |                          |
       +------(fail / lease)------+--> failed

* The row is created ``queued`` by the trigger claim, in the SAME transaction
  that selects its sessions and records their watermarks (``create_run`` takes
  the caller's session). ``heartbeat_at`` starts at creation, so a run whose
  process dies before it even starts is still reclaimed.
* ``execute`` drives a run: it refreshes ``heartbeat_at`` every
  ``LEARNING_RUN_HEARTBEAT_SECONDS`` (30 s) from a background task while the
  work runs, then records ``succeeded`` or ``failed`` (with the error). Every
  status change is a conditional UPDATE on a live status, so a run that was
  already reclaimed is never resurrected; a run that finds its lease gone stops.
* ``recover_expired_leases`` marks queued/running runs whose heartbeat is older
  than ``MIRIX_LEARNING_RUN_LEASE_SECONDS`` (300 s) as ``failed`` with error
  ``lease_expired`` — at startup, every half lease (the sweeper's lease loop)
  and, for one user, at the start of every trigger claim. Because "pending
  distill" only excludes sessions claimed by queued/running runs, that releases
  the claims; the next trigger check picks the sessions up again.
* ``commit_distilled_session`` is the only way a run writes a session's
  outcome: its experiences and the ``distilled_at`` marks of its rows up to the
  watermark land in ONE transaction that first checks (and locks) the run is
  still queued/running. A run that was cancelled, reclaimed or erased therefore
  never half-writes a session: either the session is distilled with its
  experiences, or nothing of it was written and it is claimable again.
* ``schedule`` starts a run's execution as a background asyncio task and keeps
  a reference to it until it finishes (an unreferenced task can be garbage
  collected mid-flight). ``abort_local_runs`` stops this process's executions
  of the given runs (user/client erasure).

Lease timestamps (``heartbeat_at`` and the recovery cutoff) come from the
DATABASE clock on Postgres, so clock skew between app hosts can never make one
host reclaim another host's live run. The clock and heartbeat interval are
injectable for tests; an injected clock is used for the lease too.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set

from sqlalchemy import func, literal, or_, select, update
from sqlalchemy.types import Interval

from mirix.constants import LEARNING_RUN_HEARTBEAT_SECONDS, LEARNING_RUN_LEASE_SECONDS
from mirix.errors import AbortAgentStep
from mirix.log import get_logger
from mirix.orm.skill_learning_run import SkillLearningRun as SkillLearningRunModel
from mirix.schemas.skill_learning_run import (
    ACTIVE_LEARNING_RUN_STATUSES,
    SkillLearningRun as PydanticSkillLearningRun,
    SkillLearningRunCreate,
)
from mirix.services.conversation_session_manager import (
    Clock,
    as_utc,
    dialect_name,
    utc_now,
)

if TYPE_CHECKING:
    from mirix.services.skill_experience_manager import PoolSettlement

logger = get_logger(__name__)

LEASE_EXPIRED_ERROR = "lease_expired"
ERASED_ERROR = "erased"
TERMINAL_LEARNING_RUN_STATUSES = ("succeeded", "failed")

# Strong references to in-flight learning-run tasks. A task removes itself
# when it finishes.
_BACKGROUND_TASKS: Set[asyncio.Task] = set()


class _Execution:
    """This process's execution of one run (see ``execute``)."""

    def __init__(self, work_task: asyncio.Future):
        self.work_task = work_task
        self.stop_reason: Optional[str] = None


# run_id -> its live execution in THIS process (``abort_local_runs``).
_EXECUTIONS: Dict[str, _Execution] = {}


class LearningRunNotActive(AbortAgentStep):
    """The run is no longer queued/running (e.g. its lease was reclaimed).

    An :class:`~mirix.errors.AbortAgentStep`: raised by the curator's write
    guard inside a skill tool, it ends the whole agent step instead of becoming
    a tool result the model could answer."""

    def __init__(self, run_id: str, status: Optional[str] = None):
        self.run_id = run_id
        self.status = status
        super().__init__(f"learning run {run_id} is not active (status={status})")


class LearningRunAborted(LearningRunNotActive):
    """The run was stopped from outside while executing (e.g. user erasure)."""

    def __init__(self, run_id: str, reason: str):
        super().__init__(run_id, "failed")
        self.reason = reason
        self.args = (f"learning run {run_id} was stopped: {reason}",)


class LearningRunLeaseLost(Exception):
    """A heartbeat found the run no longer active while its work was running."""


def background_learning_tasks() -> Set[asyncio.Task]:
    """The live background learning-run tasks (read-only view for tests/shutdown)."""
    return set(_BACKGROUND_TASKS)


def local_run_ids() -> Set[str]:
    """Runs this process is executing right now."""
    return {rid for rid, ex in _EXECUTIONS.items() if not ex.work_task.done()}


async def abort_local_runs(
    run_ids, *, reason: str = ERASED_ERROR, timeout: float = 30.0
) -> List[str]:
    """Stop this process's executions of ``run_ids`` and wait for them to end.

    Each execution's work is cancelled and ``execute`` raises
    :class:`LearningRunAborted` (the caller is expected to have recorded the
    runs as failed already). Waits up to ``timeout`` seconds for the work to
    unwind — releasing, among others, the per-owner procedural dream guard.
    Returns the run ids that were executing here.
    """
    stopped: List[str] = []
    waits = []
    for run_id in run_ids:
        execution = _EXECUTIONS.get(run_id)
        if execution is None or execution.work_task.done():
            continue
        execution.stop_reason = reason
        execution.work_task.cancel()
        stopped.append(run_id)
        waits.append(execution.work_task)
    if waits:
        await asyncio.wait(waits, timeout=timeout)
    return stopped


def _merge_usage(total: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Sum numeric leaves of two usage dicts (nested dicts merge recursively)."""
    merged = dict(total)
    for key, value in (delta or {}).items():
        current = merged.get(key)
        if isinstance(value, dict):
            merged[key] = _merge_usage(
                current if isinstance(current, dict) else {}, value
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            base = current if isinstance(current, (int, float)) else 0
            merged[key] = base + value
        else:
            merged[key] = value
    return merged


def _find_batch_entry(entries, batch_id: str) -> Optional[Dict[str, Any]]:
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("batch_id") == batch_id:
            return entry
    return None


def _upsert_batch_entry(entries, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``curator_metrics`` with ``entry`` replacing the one of its batch (or
    appended). A new list, so the JSON column is seen as changed."""
    batch_id = entry.get("batch_id")
    result: List[Dict[str, Any]] = []
    replaced = False
    for current in entries or []:
        if isinstance(current, dict) and batch_id and current.get("batch_id") == batch_id:
            result.append(dict(entry))
            replaced = True
        else:
            result.append(current)
    if not replaced:
        result.append(dict(entry))
    return result


def _parse_time(value: Any) -> Optional[dt.datetime]:
    """A stored ISO timestamp as an aware UTC datetime (None stays None)."""
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return as_utc(value)
    return as_utc(dt.datetime.fromisoformat(str(value)))


def _find_session_entry(entries, session_id: str) -> Optional[Dict[str, Any]]:
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("session_id") == session_id:
            return entry
    return None


def _merge_session_entries(
    entries, updates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """``distill_metrics`` with each update merged into its session's entry
    (one entry per session), or appended when the session has none yet
    (or the update names no session). A new list of new dicts, so the JSON
    column is seen as changed."""
    result: List[Dict[str, Any]] = [
        dict(e) if isinstance(e, dict) else e for e in (entries or [])
    ]
    for change in updates:
        session_id = change.get("session_id")
        target = (
            _find_session_entry(result, session_id) if session_id is not None else None
        )
        if target is None:
            result.append(dict(change))
        else:
            target.update(change)
    return result


# Key of a session's distill_metrics entry recording how THIS run settled the
# session's failure — written in the settle's own transaction, so a repeat of
# that transaction (a database retry after an ambiguous COMMIT) finds it
# and does not count the failure (or back the session off) twice.
_SETTLED_KEY = "settled"


class SkillLearningRunManager:
    """CRUD, lease and metrics for ``skill_learning_run`` rows."""

    def __init__(
        self,
        *,
        session_maker=None,
        clock: Optional[Clock] = None,
        lease_seconds: Optional[int] = None,
        heartbeat_seconds: Optional[float] = None,
    ):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        # Lease timestamps use the database clock unless a clock is injected.
        self._db_clock = clock is None
        self.clock: Clock = clock or utc_now
        self.lease_seconds = (
            LEARNING_RUN_LEASE_SECONDS if lease_seconds is None else int(lease_seconds)
        )
        self.heartbeat_seconds = (
            LEARNING_RUN_HEARTBEAT_SECONDS
            if heartbeat_seconds is None
            else float(heartbeat_seconds)
        )

    def now(self) -> dt.datetime:
        return as_utc(self.clock())

    async def _lease_now(self, db) -> dt.datetime:
        """Now, for lease bookkeeping: the database clock on Postgres (so every
        app host compares heartbeats against the same clock), else ``now()``."""
        if self._db_clock and dialect_name(db) == "postgresql":
            value = (await db.execute(select(func.clock_timestamp()))).scalar_one()
            return as_utc(value)
        return self.now()

    # ------------------------------------------------------------------ #
    # Create / read                                                        #
    # ------------------------------------------------------------------ #

    async def create_run(
        self,
        db,
        *,
        organization_id: str,
        user_id: str,
        agent_id: str,
        trigger: str,
        session_ids: List[str],
        session_watermarks: Dict[str, Any],
        actor_id: Optional[str] = None,
        mode: str = "procedural",
    ) -> SkillLearningRunModel:
        """Insert a ``queued`` run in the caller's transaction (the caller commits).

        Validated through ``SkillLearningRunCreate`` (trigger / mode value
        space, unique session ids, watermarks only for claimed sessions, no
        sessions for a non-procedural mode); watermarks are stored as ISO-8601
        strings.
        """
        payload = SkillLearningRunCreate(
            organization_id=organization_id,
            user_id=user_id,
            agent_id=agent_id,
            trigger=trigger,
            mode=mode,
            session_ids=list(session_ids),
            session_watermarks=dict(session_watermarks),
        )
        now = await self._lease_now(db)
        row = SkillLearningRunModel(
            organization_id=payload.organization_id,
            user_id=payload.user_id,
            agent_id=payload.agent_id,
            trigger=payload.trigger,
            mode=payload.mode,
            status="queued",
            session_ids=list(payload.session_ids),
            session_watermarks=dict(payload.session_watermarks),
            heartbeat_at=now,
            created_at=now,
            updated_at=now,
            embedding_chunked_calls=0,
        )
        if actor_id:
            row._set_created_and_updated_by_fields(actor_id)
        db.add(row)
        await db.flush()
        return row

    async def create_standalone_run(
        self,
        *,
        organization_id: str,
        user_id: str,
        agent_id: str,
        mode: str,
        trigger: str = "manual",
        actor_id: Optional[str] = None,
    ) -> PydanticSkillLearningRun:
        """Record a ``queued`` run that claims no sessions, in its own transaction.

        Every auto_dream run is recorded; a non-procedural mode (core,
        episodic, ...) has no sessions to claim, so it needs no trigger claim.
        """
        async with self.session_maker() as db:
            row = await self.create_run(
                db,
                organization_id=organization_id,
                user_id=user_id,
                agent_id=agent_id,
                trigger=trigger,
                session_ids=[],
                session_watermarks={},
                actor_id=actor_id,
                mode=mode,
            )
            await db.commit()
            await db.refresh(row)
            return row.to_pydantic()

    async def get_run(
        self,
        run_id: str,
        *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[PydanticSkillLearningRun]:
        """One run, or None (also None when it is outside the given org/user)."""
        async with self.session_maker() as db:
            filters = [
                SkillLearningRunModel.id == run_id,
                SkillLearningRunModel.is_deleted.is_(False),
            ]
            if organization_id is not None:
                filters.append(SkillLearningRunModel.organization_id == organization_id)
            if user_id is not None:
                filters.append(SkillLearningRunModel.user_id == user_id)
            row = (
                await db.execute(select(SkillLearningRunModel).where(*filters))
            ).scalar_one_or_none()
            return row.to_pydantic() if row is not None else None

    async def list_runs(
        self,
        *,
        organization_id: str,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[PydanticSkillLearningRun]:
        """Runs of an organization, newest first.

        ``status`` filters by one status, or ``"active"`` for queued + running.
        ``agent_ids`` restricts to runs of those meta agents (tenant scoping by
        client); an empty list matches nothing.
        """
        async with self.session_maker() as db:
            filters = [
                SkillLearningRunModel.organization_id == organization_id,
                SkillLearningRunModel.is_deleted.is_(False),
            ]
            if user_id is not None:
                filters.append(SkillLearningRunModel.user_id == user_id)
            if agent_ids is not None:
                if not agent_ids:
                    return []
                filters.append(SkillLearningRunModel.agent_id.in_(agent_ids))
            if status == "active":
                filters.append(
                    SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES)
                )
            elif status is not None:
                filters.append(SkillLearningRunModel.status == status)
            rows = (
                await db.execute(
                    select(SkillLearningRunModel)
                    .where(*filters)
                    .order_by(
                        SkillLearningRunModel.created_at.desc(),
                        SkillLearningRunModel.id.desc(),
                    )
                    .limit(max(1, int(limit)))
                )
            ).scalars()
            return [row.to_pydantic() for row in rows]

    # ------------------------------------------------------------------ #
    # Status transitions (all conditional on a live status)                #
    # ------------------------------------------------------------------ #

    async def _conditional_update(
        self,
        run_id: str,
        from_statuses,
        values: Dict[str, Any],
        *,
        stamp_heartbeat: bool = False,
    ) -> bool:
        async with self.session_maker() as db:
            if stamp_heartbeat:
                values = {**values, "heartbeat_at": await self._lease_now(db)}
            result = await db.execute(
                update(SkillLearningRunModel)
                .where(
                    SkillLearningRunModel.id == run_id,
                    SkillLearningRunModel.status.in_(tuple(from_statuses)),
                )
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            return int(result.rowcount or 0) == 1

    async def mark_running(self, run_id: str) -> bool:
        """queued -> running. False when the run is not queued any more."""
        now = self.now()
        return await self._conditional_update(
            run_id,
            ("queued",),
            {"status": "running", "started_at": now, "updated_at": now},
            stamp_heartbeat=True,
        )

    async def heartbeat(self, run_id: str) -> bool:
        """Refresh the lease. False when the run is no longer queued/running."""
        return await self._conditional_update(
            run_id,
            ACTIVE_LEARNING_RUN_STATUSES,
            {"updated_at": self.now()},
            stamp_heartbeat=True,
        )

    async def succeed(self, run_id: str) -> bool:
        now = self.now()
        return await self._conditional_update(
            run_id,
            ACTIVE_LEARNING_RUN_STATUSES,
            {"status": "succeeded", "finished_at": now, "updated_at": now},
            stamp_heartbeat=True,
        )

    async def fail(self, run_id: str, error: str) -> bool:
        now = self.now()
        return await self._conditional_update(
            run_id,
            ACTIVE_LEARNING_RUN_STATUSES,
            {
                "status": "failed",
                "error": (error or "unknown error")[:4000],
                "finished_at": now,
                "updated_at": now,
            },
        )

    async def recover_expired_leases(
        self,
        *,
        db=None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Fail every queued/running run whose heartbeat is older than the lease.

        Runs with no heartbeat at all fall back to their ``created_at``. Returns
        the ids of the reclaimed runs; their sessions are claimable again.
        ``organization_id`` / ``user_id`` restrict recovery to one owner (a
        trigger claim reclaims its user's dead runs before counting). With
        ``db`` the updates join the caller's transaction and the caller
        commits.
        """
        if db is None:
            async with self.session_maker() as own_db:
                reclaimed = await self.recover_expired_leases(
                    db=own_db, organization_id=organization_id, user_id=user_id
                )
                await own_db.commit()
                return reclaimed

        now = self.now()
        lease = dt.timedelta(seconds=self.lease_seconds)
        if self._db_clock and dialect_name(db) == "postgresql":
            # Compared in SQL against the database clock (see _lease_now).
            cutoff = func.clock_timestamp() - literal(lease, Interval())
        else:
            cutoff = now - lease
        stale = or_(
            SkillLearningRunModel.heartbeat_at < cutoff,
            (SkillLearningRunModel.heartbeat_at.is_(None))
            & (SkillLearningRunModel.created_at < cutoff),
        )
        owner = []
        if organization_id is not None:
            owner.append(SkillLearningRunModel.organization_id == organization_id)
        if user_id is not None:
            owner.append(SkillLearningRunModel.user_id == user_id)
        ids = [
            row[0]
            for row in (
                await db.execute(
                    select(SkillLearningRunModel.id).where(
                        SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES),
                        stale,
                        *owner,
                    )
                )
            ).all()
        ]
        reclaimed: List[str] = []
        for run_id in ids:
            # Conditional per run: a heartbeat that lands meanwhile wins.
            result = await db.execute(
                update(SkillLearningRunModel)
                .where(
                    SkillLearningRunModel.id == run_id,
                    SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES),
                    stale,
                )
                .values(
                    status="failed",
                    error=LEASE_EXPIRED_ERROR,
                    finished_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if int(result.rowcount or 0) == 1:
                reclaimed.append(run_id)
        if reclaimed:
            logger.warning(
                "Reclaimed %d learning run(s) with an expired lease: %s",
                len(reclaimed),
                reclaimed,
            )
        return reclaimed

    async def fail_active_runs(
        self,
        *,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        error: str = ERASED_ERROR,
    ) -> List[PydanticSkillLearningRun]:
        """Fail every queued/running run of a user and/or of some meta agents.

        Used by erasure before it purges: a failed run can no longer commit a
        session (``commit_distilled_session`` checks the status under a lock)
        and its heartbeat stops its work at the next beat in whatever process
        runs it. Each update is conditional on a live status. Returns the runs
        that were failed here (as they were before the update).
        """
        if user_id is None and agent_ids is None:
            raise ValueError("fail_active_runs needs a user_id or agent_ids")
        if agent_ids is not None and not agent_ids:
            return []
        now = self.now()
        filters = [SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES)]
        if user_id is not None:
            filters.append(SkillLearningRunModel.user_id == user_id)
        if agent_ids is not None:
            filters.append(SkillLearningRunModel.agent_id.in_(list(agent_ids)))
        failed: List[PydanticSkillLearningRun] = []
        async with self.session_maker() as db:
            rows = (
                (await db.execute(select(SkillLearningRunModel).where(*filters)))
                .scalars()
                .all()
            )
            for row in rows:
                snapshot = row.to_pydantic()
                result = await db.execute(
                    update(SkillLearningRunModel)
                    .where(
                        SkillLearningRunModel.id == row.id,
                        SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES),
                    )
                    .values(
                        status="failed", error=error, finished_at=now, updated_at=now
                    )
                    .execution_options(synchronize_session=False)
                )
                if int(result.rowcount or 0) == 1:
                    failed.append(snapshot)
            await db.commit()
        if failed:
            logger.warning(
                "Failed %d active learning run(s) (%s): %s",
                len(failed),
                error,
                [r.id for r in failed],
            )
        return failed

    async def _lock_active_run(self, db, run_id: str) -> None:
        """In ``db``'s transaction: lock the run row and require a live status.

        Postgres: ``SELECT ... FOR UPDATE`` (a concurrent reclaim, erasure or
        status change waits for this transaction, or this one sees it).
        SQLite: a no-op UPDATE first, so the transaction holds the database
        write lock before it reads. Raises :class:`LearningRunNotActive`.
        """
        if dialect_name(db) == "sqlite":
            await db.execute(
                update(SkillLearningRunModel)
                .where(SkillLearningRunModel.id == run_id)
                .values(updated_at=SkillLearningRunModel.updated_at)
                .execution_options(synchronize_session=False)
            )
        status = (
            await db.execute(
                select(SkillLearningRunModel.status)
                .where(SkillLearningRunModel.id == run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if status not in ACTIVE_LEARNING_RUN_STATUSES:
            raise LearningRunNotActive(run_id, status)

    async def lock_active_run(self, db, run_id: str) -> None:
        """In ``db``'s transaction: lock the run and require it to be queued or
        running, else :class:`LearningRunNotActive`. The curator's write guard:
        every skill write of a learning run calls it inside the write's own
        transaction, so a run that was reclaimed (lost lease), erased or
        cancelled cannot change a skill any more."""
        await self._lock_active_run(db, run_id)

    async def _locked_run_row(self, db, run_id: str) -> SkillLearningRunModel:
        row = (
            await db.execute(
                select(SkillLearningRunModel)
                .where(SkillLearningRunModel.id == run_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
        return row

    async def start_curator_batch(self, run_id: str, entry: Dict[str, Any]) -> None:
        """Record that a curator batch starts (``curator_metrics`` entry with
        ``status: running``), atomically with a check that the run is still
        active (:class:`LearningRunNotActive` otherwise, nothing written). An
        entry left ``running`` is a batch that never settled — the run died or
        was stopped in the middle of it. Safe to repeat (a database
        retry): the entry replaces the batch's earlier one."""
        from mirix.services.retry_policy import retry_call

        async def _once() -> None:
            async with self.session_maker() as db:
                await self._lock_active_run(db, run_id)
                row = await self._locked_run_row(db, run_id)
                row.curator_metrics = _upsert_batch_entry(row.curator_metrics, entry)
                row.updated_at = self.now()
                await db.commit()

        await retry_call(_once, stage="curator_start")

    async def settle_curator_batch(
        self,
        run_id: str,
        *,
        organization_id: Optional[str],
        user_id: str,
        agent_id: str,
        batch_id: str,
        experience_ids: List[str],
        outcome: str,
        entry: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
        max_attempts: Optional[int] = None,
        pool: Optional["PoolSettlement"] = None,
    ) -> Dict[str, Any]:
        """A curator batch's outcome, in ONE transaction:
        require the run to still be active (locked), move the batch's
        experiences and update the experience pool with ``pool`` (what the
        batch showed and triaged; see
        :meth:`SkillExperienceManager.settle_curation_batch`), write the
        batch's ``curator_metrics`` entry — with the pool's summary — and add
        its LLM usage.

        Safe to repeat (a database retry): a batch whose entry is already
        settled is not applied twice (the pool is not decayed twice either).
        Raises :class:`LearningRunNotActive` (nothing written) when the run is
        gone.
        """
        from mirix.constants import SKILL_CURATION_MAX_ATTEMPTS
        from mirix.services.retry_policy import retry_call
        from mirix.services.skill_experience_manager import (
            POOL_SUMMARY_KEYS,
            SkillExperienceManager,
        )

        attempts = SKILL_CURATION_MAX_ATTEMPTS if max_attempts is None else max_attempts

        async def _once() -> Dict[str, Any]:
            async with self.session_maker() as db:
                await self._lock_active_run(db, run_id)
                row = await self._locked_run_row(db, run_id)
                previous = _find_batch_entry(row.curator_metrics, batch_id)
                if previous is not None and previous.get("status") == "settled":
                    await db.rollback()
                    repeated: Dict[str, Any] = {
                        "consumed": list(previous.get("consumed_ids") or []),
                        "returned": list(previous.get("returned_ids") or []),
                        "unaddressed": list(previous.get("unaddressed_ids") or []),
                    }
                    if any(key in previous for key in POOL_SUMMARY_KEYS):
                        repeated["pool"] = {
                            key: previous.get(key) for key in POOL_SUMMARY_KEYS
                        }
                    return repeated
                now = self.now()
                settled = await SkillExperienceManager.settle_curation_batch(
                    db,
                    ids=list(experience_ids),
                    agent_id=agent_id,
                    user_id=user_id,
                    outcome=outcome,
                    batch_id=batch_id,
                    max_attempts=attempts,
                    now=now,
                    organization_id=organization_id,
                    pool=pool,
                )
                final = dict(entry)
                final.update(
                    {
                        "batch_id": batch_id,
                        "status": "settled",
                        "outcome": outcome,
                        "consumed": len(settled["consumed"]),
                        "returned": len(settled["returned"]),
                        "unaddressed": len(settled["unaddressed"]),
                        "consumed_ids": settled["consumed"],
                        "returned_ids": settled["returned"],
                        "unaddressed_ids": settled["unaddressed"],
                        **(settled.get("pool") or {}),
                    }
                )
                row.curator_metrics = _upsert_batch_entry(row.curator_metrics, final)
                if usage:
                    row.llm_usage = _merge_usage(dict(row.llm_usage or {}), usage)
                row.updated_at = now
                await db.commit()
                return settled

        return await retry_call(_once, stage="curator_settle")

    async def ensure_active(self, run_id: str) -> None:
        """Raise :class:`LearningRunNotActive` unless the run is queued/running.

        A run checks this before each stage that writes outside
        ``commit_distilled_session`` (curation, the dream checkpoint), so a run
        that was reclaimed or erased stops instead of writing on."""
        run = await self.get_run(run_id)
        if run is None or run.status not in ACTIVE_LEARNING_RUN_STATUSES:
            raise LearningRunNotActive(run_id, getattr(run, "status", None))

    async def commit_distilled_session(
        self,
        run_id: str,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        watermark: Any,
        experiences: List[Dict[str, Any]],
        created_by_id: Optional[str] = None,
        skill_feedback: Optional[List[Dict[str, Any]]] = None,
    ):
        """Persist one session's experiences and mark its rows, atomically.

        ``skill_feedback``: the session's feedback rows on the skills
        served in it (keyword arguments of
        ``SkillSessionFeedbackManager.build_row``), inserted in the same
        transaction with this run's id — and, like the experiences, never
        twice when the commit is repeated.

        In ONE transaction: require the run to still be queued/running (row
        lock, see :meth:`_lock_active_run`), insert ``experiences`` (keyword
        arguments of ``SkillExperienceManager.build_row``, status ``pending``)
        and stamp ``distilled_at`` on the session's live rows up to
        ``watermark`` (clearing the session's infrastructure-error backoff).
        Cancellation, a lost lease or an erasure between two sessions
        therefore never leaves a session with experiences but undistilled rows
        (which the next run would distill again, duplicating them). The commit
        is safe to repeat (a database retry): when none of the session's
        rows up to the watermark is left undistilled, an earlier attempt
        already committed and nothing is inserted again. Returns
        ``(experiences, rows_marked)``; raises :class:`LearningRunNotActive`
        (nothing written) when the run is gone.
        """
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )
        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )
        from mirix.services.skill_experience_manager import SkillExperienceManager
        from mirix.services.skill_session_feedback import SkillSessionFeedbackManager

        rows = [
            SkillExperienceManager.build_row(**dict(item, created_by_id=created_by_id))
            for item in experiences
        ]
        feedback = [
            SkillSessionFeedbackManager.build_row(
                **dict(item, created_by_id=created_by_id, learning_run_id=run_id)
            )
            for item in skill_feedback or ()
        ]
        async with self.session_maker() as db:
            await self._lock_active_run(db, run_id)
            if (rows or feedback) and not await self._has_rows_to_mark(
                db,
                organization_id=organization_id,
                user_id=user_id,
                session_id=session_id,
                watermark=watermark,
            ):
                # Nothing up to the watermark is left undistilled, yet only
                # this (live) run holds the claim: an earlier attempt of this
                # very commit went through (e.g. the connection dropped after
                # COMMIT and the transaction was retried). Inserting again
                # would duplicate the session's experiences.
                logger.warning(
                    "Learning run %s: session %s is already committed; the "
                    "repeated commit writes nothing",
                    run_id,
                    session_id,
                )
                rows = []
                feedback = []
            for row in rows:
                db.add(row)
            for row in feedback:
                db.add(row)
            if rows or feedback:
                await db.flush()
            result = await db.execute(
                ConversationMessageManager.mark_distilled_stmt(
                    organization_id=organization_id,
                    user_id=user_id,
                    session_id=session_id,
                    watermark=watermark,
                    now=self.now(),
                )
            )
            marked = int(result.rowcount or 0)
            # A success ends the session's infrastructure-failure streak: the
            # next failure backs off from the minimum again.
            await db.execute(
                ConversationSessionManager.clear_distill_backoff_stmt(
                    organization_id=organization_id,
                    user_id=user_id,
                    session_ids=[session_id],
                )
            )
            await db.commit()
            for row in rows:
                await db.refresh(row)
            return [row.to_pydantic() for row in rows], marked

    @staticmethod
    async def _has_rows_to_mark(
        db, *, organization_id: str, user_id: str, session_id: str, watermark: Any
    ) -> bool:
        """Whether the session has live, undistilled rows up to ``watermark``."""
        from mirix.orm.conversation_message import (
            ConversationMessage as ConversationMessageModel,
        )
        from mirix.services.conversation_message_manager import parse_watermark

        found = (
            await db.execute(
                select(ConversationMessageModel.id)
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id == session_id,
                    ConversationMessageModel.created_at <= parse_watermark(watermark),
                    ConversationMessageModel.distilled_at.is_(None),
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .limit(1)
            )
        ).first()
        return found is not None

    async def defer_distilled_session(
        self,
        run_id: str,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
    ) -> Optional[dt.datetime]:
        """An infrastructure failure: back the session off, atomically
        with a check that the run is still queued/running.

        The session stays pending and undistilled; it is neither counted nor
        claimed before the returned ``distill_retry_after``. Never counted as
        a distill failure. Raises :class:`LearningRunNotActive` (nothing
        written) when the run is gone.
        """
        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )

        async with self.session_maker() as db:
            await self._lock_active_run(db, run_id)
            run = await self._locked_run_row(db, run_id)
            settled = self._settled(run, session_id, "deferred")
            if settled is not None:
                # This very settle already committed (a repeated transaction).
                await db.rollback()
                return _parse_time(settled.get("distill_retry_after"))
            deferred = await ConversationSessionManager(
                session_maker=self.session_maker, clock=self.clock
            ).defer_distill(
                organization_id=organization_id,
                user_id=user_id,
                session_ids=[session_id],
                now=self.now(),
                db=db,
            )
            retry_after = as_utc(deferred.get(session_id))
            self._mark_settled(
                run,
                session_id,
                {
                    "action": "deferred",
                    "distill_retry_after": retry_after.isoformat()
                    if retry_after
                    else None,
                },
            )
            await db.commit()
        return retry_after

    @staticmethod
    def _settled(
        run: SkillLearningRunModel, session_id: str, action: str
    ) -> Optional[Dict[str, Any]]:
        entry = _find_session_entry(run.distill_metrics, session_id)
        marker = (entry or {}).get(_SETTLED_KEY)
        if isinstance(marker, dict) and marker.get("action") == action:
            return marker
        return None

    def _mark_settled(
        self, run: SkillLearningRunModel, session_id: str, marker: Dict[str, Any]
    ) -> None:
        run.distill_metrics = _merge_session_entries(
            run.distill_metrics,
            [{"session_id": session_id, _SETTLED_KEY: dict(marker)}],
        )
        run.updated_at = self.now()

    async def record_distill_failure(
        self,
        run_id: str,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        max_failures: Optional[int] = None,
    ):
        """A counted distill failure, atomically with a check that the run
        is still queued/running.

        Returns ``(distill_failures, distill_failed_at)``; ``distill_failed_at``
        is set once the session reached ``DISTILL_MAX_FAILURES`` (3), after
        which it is never selected again. Raises :class:`LearningRunNotActive`
        (nothing written) when the run is gone.

        Idempotent per run and session: the transaction also records the
        settle on the run's ``distill_metrics`` entry of the session, so
        repeating it after an ambiguous COMMIT (a database retry) never
        counts the same failure twice. The same holds for
        :meth:`defer_distilled_session`.
        """
        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )

        async with self.session_maker() as db:
            await self._lock_active_run(db, run_id)
            run = await self._locked_run_row(db, run_id)
            settled = self._settled(run, session_id, "counted")
            if settled is not None:
                # This very settle already committed (a repeated transaction).
                await db.rollback()
                return (
                    int(settled.get("distill_failures") or 0),
                    _parse_time(settled.get("distill_failed_at")),
                )
            recorded = await ConversationSessionManager(
                session_maker=self.session_maker, clock=self.clock
            ).record_distill_failure(
                organization_id=organization_id,
                user_id=user_id,
                session_ids=[session_id],
                now=self.now(),
                max_failures=max_failures,
                db=db,
            )
            failures, failed_at = recorded.get(session_id, (0, None))
            self._mark_settled(
                run,
                session_id,
                {
                    "action": "counted",
                    "distill_failures": failures,
                    "distill_failed_at": failed_at.isoformat() if failed_at else None,
                },
            )
            await db.commit()
        return failures, failed_at

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    async def _update_locked(
        self, run_id: str, mutate: Callable[[SkillLearningRunModel], None]
    ) -> bool:
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    select(SkillLearningRunModel)
                    .where(SkillLearningRunModel.id == run_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            mutate(row)
            row.updated_at = self.now()
            await db.commit()
            return True

    async def append_distill_metrics(
        self, run_id: str, entries: List[Dict[str, Any]]
    ) -> bool:
        """Record per-session distill metric entries (one entry per
        session): an entry is merged into the session's existing entry (e.g.
        the one a failure settle started), else appended."""
        if not entries:
            return True

        def mutate(row):
            row.distill_metrics = _merge_session_entries(
                row.distill_metrics, [dict(e) for e in entries]
            )

        return await self._update_locked(run_id, mutate)

    async def append_curator_metrics(
        self, run_id: str, entries: List[Dict[str, Any]]
    ) -> bool:
        """Append per-batch curator metric entries."""
        if not entries:
            return True

        def mutate(row):
            row.curator_metrics = list(row.curator_metrics or []) + [
                dict(e) for e in entries
            ]

        return await self._update_locked(run_id, mutate)

    async def add_llm_usage(self, run_id: str, usage: Dict[str, Any]) -> bool:
        """Add token usage into the run's aggregate (numeric leaves are summed)."""
        if not usage:
            return True

        def mutate(row):
            row.llm_usage = _merge_usage(dict(row.llm_usage or {}), usage)

        return await self._update_locked(run_id, mutate)

    async def append_retry_events(
        self, run_id: str, events: List[Dict[str, Any]]
    ) -> bool:
        """Append retry events (kind, attempt, waited_s, ...) to the run.

        The sink of ``RetryEventRecorder.for_learning_run``; False when the run
        does not exist.
        """
        if not events:
            return True

        def mutate(row):
            row.retry_events = list(row.retry_events or []) + [dict(e) for e in events]

        return await self._update_locked(run_id, mutate)

    async def add_embedding_chunked_calls(self, run_id: str, count: int) -> bool:
        if count <= 0:
            return True

        def mutate(row):
            row.embedding_chunked_calls = int(row.embedding_chunked_calls or 0) + int(
                count
            )

        return await self._update_locked(run_id, mutate)

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    async def execute(self, run_id: str, work: Callable[[], Awaitable[Any]]) -> Any:
        """Run ``work()`` for ``run_id`` under a heartbeat, then record the outcome.

        The heartbeat runs every ``heartbeat_seconds`` until the work finishes;
        if one finds the run no longer queued/running (its lease was
        reclaimed), the work is cancelled and ``LearningRunLeaseLost`` raised —
        the sessions were released and may already belong to another run. On
        success the run becomes ``succeeded``; on any error ``failed`` with the
        error, and the error propagates. ``work`` is expected to call
        :meth:`mark_running` once it actually starts (it may first wait for the
        per-owner serialization lock while the heartbeat keeps the lease).
        :func:`abort_local_runs` stops the work from outside; ``execute`` then
        raises :class:`LearningRunAborted`.

        Every embedding call of the work whose input had to be chunked is
        counted and added to the run's ``embedding_chunked_calls`` when
        the work ends, whatever its outcome.
        """
        from mirix.embedding_guard import (
            EmbeddingChunkCounter,
            counting_embedding_chunks,
        )

        chunk_counter = EmbeddingChunkCounter()
        # The task copies the context: the work, and every task it starts,
        # counts into chunk_counter.
        with counting_embedding_chunks(chunk_counter):
            work_task = asyncio.ensure_future(work())
        lease_lost = asyncio.Event()
        execution = _Execution(work_task)
        _EXECUTIONS[run_id] = execution

        async def _beat() -> None:
            while True:
                await asyncio.sleep(self.heartbeat_seconds)
                try:
                    alive = await self.heartbeat(run_id)
                except Exception as exc:  # noqa: BLE001 — a DB hiccup must not kill the run
                    logger.warning(
                        "Heartbeat for learning run %s failed: %s", run_id, exc
                    )
                    continue
                if not alive:
                    lease_lost.set()
                    work_task.cancel()
                    return

        beat_task = asyncio.ensure_future(_beat())
        try:
            result = await work_task
        except asyncio.CancelledError:
            if lease_lost.is_set():
                logger.error(
                    "Learning run %s lost its lease and was stopped; its sessions "
                    "were released for another run.",
                    run_id,
                )
                raise LearningRunLeaseLost(run_id) from None
            if execution.stop_reason is not None:
                # Stopped from outside (abort_local_runs): the run was already
                # recorded failed by whoever stopped it.
                logger.warning(
                    "Learning run %s was stopped (%s)", run_id, execution.stop_reason
                )
                raise LearningRunAborted(run_id, execution.stop_reason) from None
            work_task.cancel()
            await self._safe_fail(run_id, "cancelled")
            raise
        except LearningRunNotActive:
            raise
        except Exception as exc:
            await self._safe_fail(run_id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if _EXECUTIONS.get(run_id) is execution:
                del _EXECUTIONS[run_id]
            beat_task.cancel()
            try:
                await beat_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            await self._record_embedding_chunks(run_id, chunk_counter)
        if not await self.succeed(run_id):
            logger.warning(
                "Learning run %s finished but was no longer active; outcome not recorded.",
                run_id,
            )
        return result

    async def _record_embedding_chunks(self, run_id: str, counter: Any) -> None:
        """Add the work's chunked embedding calls to the run. A
        failed write is logged, never raised: a metric must not change the
        run's outcome."""
        if not counter.chunked_calls:
            return
        try:
            await self.add_embedding_chunked_calls(run_id, counter.chunked_calls)
        except Exception:  # noqa: BLE001 — see the docstring
            logger.warning(
                "Could not record %d chunked embedding call(s) on learning run %s",
                counter.chunked_calls,
                run_id,
                exc_info=True,
            )

    async def _safe_fail(self, run_id: str, error: str) -> None:
        try:
            await self.fail(run_id, error)
        except Exception:  # noqa: BLE001 — recording the failure must not mask it
            logger.exception("Could not record failure of learning run %s", run_id)

    @staticmethod
    def schedule(coro: Awaitable[Any], *, name: Optional[str] = None) -> asyncio.Task:
        """Run ``coro`` as a background task and keep a reference until it ends."""
        task = asyncio.ensure_future(coro)
        if name:
            try:
                task.set_name(name)
            except Exception:  # noqa: BLE001
                pass
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        return task
