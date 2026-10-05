"""Skill-learning orchestration: trigger checks, flush, run scheduling, sweeper.

The skill-learning trigger lives outside the meta memory agent's
``trigger_memory_update`` tool: the check now runs deterministically

1. after a task-session write (``/memory/add`` / ``/memory/add_sync``),
2. after an explicit session close (``POST /memory/sessions/{id}/close``),
3. from the background sweeper (``SessionSweeper``, started in the FastAPI
   lifespan): it reclaims expired learning-run leases, then checks every scope
   that has pending sessions — the only way idle-ended sessions get counted
   when the user never comes back.

A check fires only when ``SKILL_TRIGGER_SESSION_THRESHOLD`` pending task
sessions are claimable. ``flush_skills`` (``POST /memory/skills/flush``)
claims whatever is pending, however few. Either way the claim creates a
``queued`` ``skill_learning_run`` and this module schedules its execution —
``AutoDreamManager.run(mode="procedural", run_id=...)`` — as a background task
whose reference is retained until it finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from mirix.log import get_logger
from mirix.services.agent_trigger_state_manager import (
    AgentTriggerStateManager,
    ClaimFireResult,
)
from mirix.services.conversation_session_manager import (
    Clock,
    ConversationSessionManager,
)
from mirix.services.skill_learning_run_manager import (
    LearningRunLeaseLost,
    LearningRunNotActive,
    SkillLearningRunManager,
    background_learning_tasks,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Run execution                                                                #
# --------------------------------------------------------------------------- #


async def resolve_run_context(run, *, actor=None, user=None, meta_agent_state=None):
    """Return ``(user, actor, meta_agent_state)`` to execute ``run`` with.

    The run belongs to its meta agent's scope, so the actor is the client that
    owns that meta agent (its write scope stamps the evolved skills) — the
    given ``actor`` is used as-is when it is that owner. Missing pieces are
    loaded, so the sweeper (which has no request context) can run anything.
    """
    from sqlalchemy import select

    from mirix.orm.agent import Agent as AgentModel
    from mirix.server.server import db_context, get_server

    server = get_server()
    async with db_context() as db:
        owner = (
            await db.execute(
                select(AgentModel._created_by_id).where(AgentModel.id == run.agent_id)
            )
        ).first()
    owner_id = owner[0] if owner is not None else None
    if owner_id and (actor is None or getattr(actor, "id", None) != owner_id):
        actor = await server.client_manager.get_client_by_id(owner_id)
    if actor is None:
        raise LookupError(
            f"cannot resolve the client owning meta agent {run.agent_id} of run {run.id}"
        )
    if user is None or getattr(user, "id", None) != run.user_id:
        user = await server.user_manager.get_user_by_id(run.user_id)
    if (
        meta_agent_state is None
        or getattr(meta_agent_state, "id", None) != run.agent_id
    ):
        meta_agent_state = await server.agent_manager.get_agent_by_id(
            agent_id=run.agent_id, actor=actor
        )
    return user, actor, meta_agent_state


async def execute_learning_run(
    run_id: str,
    *,
    actor=None,
    user=None,
    meta_agent_state=None,
    request=None,
):
    """Execute one claimed run to completion (``AutoDreamManager`` procedural path).

    Returns the ``AutoDreamResponse``. Raises ``LearningRunNotActive`` if the
    run is gone or no longer queued/running.
    """
    from mirix.schemas.auto_dream import AutoDreamRequest
    from mirix.services.auto_dream_manager import AutoDreamManager

    run_mgr = SkillLearningRunManager()
    run = await run_mgr.get_run(run_id)
    if run is None:
        raise LearningRunNotActive(run_id, None)
    try:
        user, actor, meta_agent_state = await resolve_run_context(
            run, actor=actor, user=user, meta_agent_state=meta_agent_state
        )
    except Exception as exc:
        await run_mgr.fail(run_id, f"context: {type(exc).__name__}: {exc}")
        raise
    return await AutoDreamManager().run(
        request=request or AutoDreamRequest(mode=getattr(run, "mode", "procedural")),
        user=user,
        actor=actor,
        meta_agent_state=meta_agent_state,
        run_id=run_id,
    )


async def _run_in_background(run_id: str, **context) -> None:
    try:
        # Module-level lookup so tests can substitute the executor.
        await execute_learning_run(run_id, **context)
    except LearningRunNotActive as exc:
        logger.info("Skipped learning run %s: %s", run_id, exc)
    except LearningRunLeaseLost:
        pass  # already logged by the run manager
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a background run must never crash the loop
        # Most failures were recorded by the run's execution already. Some are
        # raised before it (a scope mismatch, a failed read of the run): record
        # them now — fail() is conditional, so this never overwrites an outcome
        # — so the run does not sit `queued` (claiming its sessions) until its
        # lease expires.
        try:
            recorded_now = await SkillLearningRunManager().fail(
                run_id, f"{type(exc).__name__}: {exc}"
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Skill learning run %s FAILED and the failure could not be "
                "recorded; its sessions are released when its lease expires",
                run_id,
            )
            return
        logger.exception(
            "Skill learning run %s FAILED (%s; its sessions are claimable again)",
            run_id,
            "recorded now" if recorded_now else "already recorded on the run",
        )


def schedule_learning_run(
    run_id: str, *, actor=None, user=None, meta_agent_state=None, request=None
) -> asyncio.Task:
    """Execute ``run_id`` in the background, keeping a reference to the task.

    ``request`` (the ``AutoDreamRequest`` that queued the run, e.g. an
    ``auto_dream`` call with ``wait=false``) travels with it, so its options —
    the model override among them — apply in the background too.
    """
    context = {"actor": actor, "user": user, "meta_agent_state": meta_agent_state}
    if request is not None:
        context["request"] = request
    return SkillLearningRunManager.schedule(
        _run_in_background(run_id, **context),
        name=f"skill-learning-run:{run_id}",
    )


@contextlib.asynccontextmanager
async def quiesce_learning_for_erasure(
    *,
    user_id: Optional[str] = None,
    agent_ids: Optional[List[str]] = None,
    extra_owners: Optional[List[Tuple[str, Optional[str]]]] = None,
    timeout: Optional[float] = None,
):
    """Stop the learning runs an erasure is about to purge, and keep them out.

    An erasure (``DELETE /users/{id}/memories``, ``DELETE /clients/{id}/memories``)
    must not race a learning run that already holds the transcript in memory
    and would write experiences, skills or a checkpoint back after the purge.
    Around the purge this:

    1. fails every queued/running run of the user (or of the client's meta
       agents) — error 'erased'. A failed run cannot commit a session
       (``commit_distilled_session`` checks under a lock) and its heartbeat
       stops it within 30 s in whatever process executes it;
    2. stops this process's executions of those runs and waits for them;
    3. holds the procedural dream guard (process lock + Postgres advisory lock)
       of every affected (user, organization) — plus ``extra_owners`` — while
       the purge runs, so a run of the same owner in another process has
       finished (or is still waiting) before anything is deleted, and whatever
       it wrote before it stopped is purged too.

    Waiting for the guards is bounded by ``timeout`` (default: one learning-run
    lease); on timeout the purge proceeds anyway and the event is logged —
    distillation still cannot write after its run was failed.
    """
    from mirix.constants import LEARNING_RUN_LEASE_SECONDS
    from mirix.services.auto_dream_manager import _procedural_dream_guard
    from mirix.services.skill_learning_run_manager import (
        ERASED_ERROR,
        abort_local_runs,
    )

    if timeout is None:
        timeout = float(LEARNING_RUN_LEASE_SECONDS)
    failed = []
    if user_id is not None or agent_ids is not None:
        failed = await SkillLearningRunManager().fail_active_runs(
            user_id=user_id, agent_ids=agent_ids, error=ERASED_ERROR
        )
    if failed:
        await abort_local_runs(
            [run.id for run in failed], reason=ERASED_ERROR, timeout=timeout
        )
    owners = {(run.user_id, run.organization_id) for run in failed}
    owners.update(extra_owners or [])
    async with contextlib.AsyncExitStack() as stack:

        async def _acquire_all() -> None:
            for owner_user, owner_org in sorted(owners, key=lambda o: (o[0], o[1] or "")):
                await stack.enter_async_context(
                    _procedural_dream_guard(owner_user, owner_org)
                )

        try:
            await asyncio.wait_for(_acquire_all(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "Erasure proceeds without the learning guards of %s: a learning "
                "run did not stop within %.0f s",
                sorted(owners, key=lambda o: (o[0], o[1] or "")),
                timeout,
            )
        yield [run.id for run in failed]


async def cancel_background_learning_runs(timeout: float = 10.0) -> int:
    """Cancel in-flight learning-run tasks (server shutdown).

    A cancelled run is recorded ``failed`` ('cancelled'), which releases its
    sessions right away instead of after the lease expires. Returns the number
    of tasks cancelled.
    """
    tasks = [task for task in background_learning_tasks() if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)
    return len(tasks)


# --------------------------------------------------------------------------- #
# Trigger check and flush                                                      #
# --------------------------------------------------------------------------- #


async def trigger_skill_learning(
    *,
    organization_id: Optional[str],
    user_id: Optional[str],
    agent_id: Optional[str],
    trigger: str = "threshold",
    actor=None,
    user=None,
    meta_agent_state=None,
    force: bool = False,
    allow_empty: bool = False,
    limit: Optional[int] = None,
    threshold: Optional[int] = None,
    schedule: bool = True,
    current_session_id: Optional[str] = None,
    trigger_manager: Optional[AgentTriggerStateManager] = None,
) -> ClaimFireResult:
    """Run the trigger check for one scope; on fire, schedule the run.

    Without a complete scope (no meta agent — e.g. a legacy session) nothing
    can be claimed and nothing fires.
    """
    if not (organization_id and user_id and agent_id):
        return ClaimFireResult(fired=False)
    manager = trigger_manager or AgentTriggerStateManager()
    claim = await manager.check_and_claim_fire(
        agent_id=agent_id,
        user_id=user_id,
        organization_id=organization_id,
        trigger=trigger,
        threshold=threshold,
        force=force,
        allow_empty=allow_empty,
        limit=limit,
        current_session_id=current_session_id,
        actor_id=getattr(actor, "id", None),
    )
    if claim.fired and schedule:
        schedule_learning_run(
            claim.run_id, actor=actor, user=user, meta_agent_state=meta_agent_state
        )
    return claim


@dataclass
class FlushResult:
    learning_run_id: Optional[str]
    session_count: int
    closed_session_ids: List[str] = field(default_factory=list)
    #: Experiences of the scope waiting for the curator when the flush
    #: decided (a run it started curates them; a run already queued or
    #: running does otherwise).
    pending_experiences: int = 0


async def flush_skills(
    *,
    organization_id: str,
    user_id: str,
    agent_id: str,
    close_open_sessions: bool = False,
    actor=None,
    user=None,
    meta_agent_state=None,
    schedule: bool = True,
    session_manager: Optional[ConversationSessionManager] = None,
    trigger_manager: Optional[AgentTriggerStateManager] = None,
) -> FlushResult:
    """Learn from every pending session of the scope now, however few.

    ``close_open_sessions`` first closes the scope's still-open task sessions
    explicitly, so "learn from everything so far" includes them. With no
    pending session, experiences an earlier run left for the curator
    still start a run — without sessions — unless a queued or running run of
    the scope will curate them. Returns the run id (None when there was
    nothing to learn), how many sessions it claimed and how many experiences
    were waiting for the curator.
    """
    closed: List[str] = []
    if close_open_sessions:
        closed = await (
            session_manager or ConversationSessionManager()
        ).close_open_sessions(
            organization_id=organization_id,
            user_id=user_id,
            agent_id=agent_id,
        )
    claim = await trigger_skill_learning(
        organization_id=organization_id,
        user_id=user_id,
        agent_id=agent_id,
        trigger="flush",
        actor=actor,
        user=user,
        meta_agent_state=meta_agent_state,
        force=True,
        schedule=schedule,
        trigger_manager=trigger_manager,
    )
    return FlushResult(
        learning_run_id=claim.run_id if claim.fired else None,
        session_count=len(claim.session_ids) if claim.fired else 0,
        closed_session_ids=closed,
        pending_experiences=claim.pending_experiences,
    )


# --------------------------------------------------------------------------- #
# Sweeper                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class SweepResult:
    reclaimed_run_ids: List[str] = field(default_factory=list)
    materialized_sessions: int = 0
    attributed_sessions: int = 0
    scopes_checked: int = 0
    fired_run_ids: List[str] = field(default_factory=list)


TriggerFn = Callable[..., Awaitable[ClaimFireResult]]


class SessionSweeper:
    """Background lease recovery and trigger checks.

    Two loops, both started by :meth:`start`:

    * the **sweep** — one pass immediately, then every ``interval_seconds``
      (``MIRIX_SESSION_SWEEP_INTERVAL_SECONDS``, default 3600; <= 0 disables
      it): reclaim expired leases, record idle ends, attribute legacy
      sessions, then run the trigger check for every scope with pending
      sessions;
    * **lease recovery** — one pass immediately (startup recovery, even with
      the sweep disabled), then every ``lease_interval_seconds`` (default half
      of ``MIRIX_LEARNING_RUN_LEASE_SECONDS``), so a run orphaned by a crash or
      a restart releases its sessions about one lease after it died instead of
      after the next hourly sweep.

    A failing pass or scope is logged and the loops keep going; :meth:`stop`
    cancels both cleanly.
    """

    def __init__(
        self,
        *,
        interval_seconds: Optional[float] = None,
        lease_interval_seconds: Optional[float] = None,
        session_maker=None,
        clock: Optional[Clock] = None,
        trigger_fn: Optional[TriggerFn] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ):
        if interval_seconds is None:
            from mirix.constants import SESSION_SWEEP_INTERVAL_SECONDS

            interval_seconds = SESSION_SWEEP_INTERVAL_SECONDS
        if lease_interval_seconds is None:
            from mirix.constants import LEARNING_RUN_LEASE_SECONDS

            lease_interval_seconds = max(1.0, LEARNING_RUN_LEASE_SECONDS / 2)
        self.interval_seconds = float(interval_seconds)
        self.lease_interval_seconds = float(lease_interval_seconds)
        self.session_maker = session_maker
        self.clock = clock
        self._trigger_fn = trigger_fn
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None
        self._lease_task: Optional[asyncio.Task] = None
        self.passes = 0
        self.lease_passes = 0

    def _managers_kwargs(self) -> dict:
        kwargs = {}
        if self.session_maker is not None:
            kwargs["session_maker"] = self.session_maker
        if self.clock is not None:
            kwargs["clock"] = self.clock
        return kwargs

    async def _trigger(self, **kwargs) -> ClaimFireResult:
        if self._trigger_fn is not None:
            return await self._trigger_fn(**kwargs)
        return await trigger_skill_learning(
            trigger_manager=AgentTriggerStateManager(**self._managers_kwargs()),
            **kwargs,
        )

    async def run_once(self) -> SweepResult:
        """One pass: reclaim expired leases, record idle ends, check each scope."""
        result = SweepResult()
        kwargs = self._managers_kwargs()
        result.reclaimed_run_ids = await SkillLearningRunManager(
            **kwargs
        ).recover_expired_leases()
        session_mgr = ConversationSessionManager(**kwargs)
        result.materialized_sessions = await session_mgr.materialize_idle()
        result.attributed_sessions = await session_mgr.adopt_unattributed_sessions()
        scopes = await session_mgr.list_pending_scopes()
        result.scopes_checked = len(scopes)
        for organization_id, user_id, agent_id in scopes:
            try:
                claim = await self._trigger(
                    organization_id=organization_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    trigger="sweep",
                )
            except Exception:  # noqa: BLE001 — one bad scope must not stop the sweep
                logger.exception(
                    "Sweep trigger check failed for scope (org=%s, user=%s, agent=%s)",
                    organization_id,
                    user_id,
                    agent_id,
                )
                continue
            if claim.fired and claim.run_id:
                result.fired_run_ids.append(claim.run_id)
        self.passes += 1
        if (
            result.reclaimed_run_ids
            or result.fired_run_ids
            or result.materialized_sessions
        ):
            logger.info(
                "Session sweep: reclaimed %d run(s), ended %d idle session(s), "
                "checked %d scope(s), fired %d run(s)",
                len(result.reclaimed_run_ids),
                result.materialized_sessions,
                result.scopes_checked,
                len(result.fired_run_ids),
            )
        return result

    async def recover_leases_once(self) -> List[str]:
        """One lease-recovery pass (the lease loop's body)."""
        reclaimed = await SkillLearningRunManager(
            **self._managers_kwargs()
        ).recover_expired_leases()
        self.lease_passes += 1
        return reclaimed

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — keep sweeping after a failed pass
                logger.exception("Session sweep pass failed")
            await self._sleep(self.interval_seconds)

    async def _lease_loop(self) -> None:
        while True:
            try:
                await self.recover_leases_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — keep recovering after a failed pass
                logger.exception("Learning-run lease recovery pass failed")
            await self._sleep(self.lease_interval_seconds)

    def start(self) -> Optional[asyncio.Task]:
        """Start the loops on the running event loop (idempotent).

        Returns the sweep task, or None when the sweep is disabled (interval
        <= 0) — lease recovery runs either way.
        """
        loop = asyncio.get_running_loop()
        if self.lease_interval_seconds > 0 and (
            self._lease_task is None or self._lease_task.done()
        ):
            self._lease_task = loop.create_task(
                self._lease_loop(), name="mirix-learning-run-lease-recovery"
            )
        if self.interval_seconds <= 0:
            logger.info(
                "Session sweep disabled (interval <= 0); lease recovery still runs"
            )
            return None
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._loop(), name="mirix-session-sweeper")
        return self._task

    @property
    def running(self) -> bool:
        """True while either loop runs."""
        return any(t is not None and not t.done() for t in (self._task, self._lease_task))

    @property
    def sweeping(self) -> bool:
        """True while the periodic sweep runs."""
        return self._task is not None and not self._task.done()

    async def stop(self) -> None:
        """Cancel both loops and wait for them to finish."""
        tasks = [t for t in (self._task, self._lease_task) if t is not None]
        self._task = self._lease_task = None
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("Session sweeper ended with an error")
