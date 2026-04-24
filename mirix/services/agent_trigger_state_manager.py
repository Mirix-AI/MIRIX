import datetime as dt
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError

from mirix.log import get_logger
from mirix.orm.agent_trigger_state import AgentTriggerState as AgentTriggerStateModel
from mirix.orm.message import Message as MessageModel
from mirix.schemas.agent_trigger_state import (
    AgentTriggerState as PydanticAgentTriggerState,
    _validate_trigger_type,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)


@dataclass
class ClaimFireResult:
    """Outcome of a `check_and_claim_fire` call.

    fired            -- True if the cursor was advanced and the caller should
                        run its downstream action (e.g. kick off procedural
                        extraction). Exactly one worker gets `fired=True` per
                        qualifying window, enforced by `SELECT ... FOR UPDATE`.
    sessions_since   -- Distinct message sessions observed in this window.
                        Useful for logging / metrics.
    just_installed   -- True iff this call initialized the cursor for the
                        first time. Callers typically treat this as "don't
                        fire, just start counting from here".
    state            -- The persisted cursor after the call, for logging.
    """

    fired: bool
    sessions_since: int
    just_installed: bool
    state: PydanticAgentTriggerState


class AgentTriggerStateManager:
    """
    Manages per-(agent, user, trigger_type) bookkeeping rows used by
    interval-driven memory triggers.

    The running session counter is derived at read time from the messages
    table, so there is exactly one source of truth. The only persisted state
    is a cursor that bounds the derivation:

      * last_fired_at                -- timestamp watermark
      * last_fired_tied_session_ids  -- session_ids whose MAX(created_at) in
                                        the counted window equalled the
                                        watermark; used to break ties when
                                        multiple messages share the exact
                                        watermark timestamp.
      * last_fired_session_id        -- the in-progress session at fire
                                        time; excluded from the next window
                                        so a single continuing session does
                                        not keep re-qualifying.

    Concurrency: `check_and_claim_fire` is the serialization point. It locks
    the cursor row with `SELECT ... FOR UPDATE`, counts new sessions inside
    the same transaction, and advances the cursor to the observed high
    watermark before committing. Two concurrent workers on the same
    (agent, user, trigger_type) therefore execute the fire decision
    strictly in order and cannot double-fire for the same window.

    Window semantics are MIN-based: each session_id has exactly one first-
    appearance timestamp (the MIN(created_at) of its messages), which is
    immutable once the first message is inserted. Filtering "MIN > cursor"
    therefore guarantees each session_id contributes to exactly one window
    — a session that was counted in window W cannot be re-counted in W+1,
    even if it keeps producing messages after the fire. That is the only
    way to avoid the double-count bug that MAX-based windowing had: with
    MAX semantics, any session producing new messages after the watermark
    got silently re-counted.

    Tie-at-watermark safety: because Postgres TIMESTAMPTZ collapses to
    microseconds, two different sessions can have identical first-
    appearance timestamps. When that timestamp happens to be the new
    watermark, the tie-breaker `(MIN = cursor AND session_id NOT IN
    tied_session_ids)` lets sessions we did not see at fire time qualify
    next time while sessions we did see stay excluded.

    Not solved (known rare race): a concurrent transaction whose
    `created_at` (= transaction start time) is strictly less than the
    watermark, but which only commits after our SELECT, is invisible to
    us and permanently falls outside the window. In practice this
    requires an older transaction to still be running when a newer one
    commits, which is rare under typical single-writer workloads. For
    procedural memory extraction — an eventually-consistent soft trigger
    — this tradeoff is acceptable; the alternative would be an
    unboundedly growing list of already-counted session_ids or a global
    serialization point on message inserts, neither of which is worth it.
    """

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def get_state(
        self,
        *,
        agent_id: str,
        user_id: str,
        trigger_type: str,
    ) -> Optional[PydanticAgentTriggerState]:
        """Return the trigger-state row for (agent, user, trigger_type), or None."""
        _validate_trigger_type(trigger_type)
        async with self.session_maker() as session:
            row = await self._fetch(session, agent_id, user_id, trigger_type)
            return row.to_pydantic() if row else None

    @enforce_types
    async def check_and_claim_fire(
        self,
        *,
        agent_id: str,
        user_id: str,
        trigger_type: str,
        threshold: int,
        organization_id: Optional[str] = None,
        current_session_id: Optional[str] = None,
    ) -> ClaimFireResult:
        """
        Atomically decide whether to fire, and if so, claim the fire by
        advancing the cursor.

        Semantics:
        - No cursor row: install one at wall-clock now and return
          fired=False, just_installed=True. Legacy sessions older than now
          are intentionally ignored so the feature does not sweep a backlog
          on first use.
        - Cursor exists: inside one transaction, lock it, count DISTINCT
          non-null session_ids in the "new since cursor" window (with the
          watermark-tie-break filter described on the class docstring),
          and if count >= threshold, advance `last_fired_at` to the window's
          MAX(created_at), record the set of session_ids tied at that
          watermark, and record the current in-progress session. Return
          fired=True.
        - Cursor exists but count < threshold: no writes, release the lock.
        """
        _validate_trigger_type(trigger_type)
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        # First-install is its own short path: never fires, and racing two
        # installers is fine because the unique key + IntegrityError fallback
        # makes INSERT idempotent.
        async with self.session_maker() as session:
            existing = await self._fetch(session, agent_id, user_id, trigger_type)
            if existing is None:
                installed = await self._install_cursor(
                    session=session,
                    agent_id=agent_id,
                    user_id=user_id,
                    trigger_type=trigger_type,
                    organization_id=organization_id,
                    session_id=current_session_id,
                )
                return ClaimFireResult(
                    fired=False,
                    sessions_since=0,
                    just_installed=True,
                    state=installed.to_pydantic(),
                )

        # Row exists; do threshold check and cursor advance under a
        # pessimistic lock so concurrent workers serialize on this cursor.
        async with self.session_maker() as session:
            locked_stmt = (
                select(AgentTriggerStateModel)
                .where(
                    AgentTriggerStateModel.agent_id == agent_id,
                    AgentTriggerStateModel.user_id == user_id,
                    AgentTriggerStateModel.trigger_type == trigger_type,
                    AgentTriggerStateModel.is_deleted.is_(False),
                )
                .with_for_update()
            )
            result = await session.execute(locked_stmt)
            row = result.scalar_one_or_none()
            if row is None:
                # Lost a race with a deletion. Treat like a re-install.
                installed = await self._install_cursor(
                    session=session,
                    agent_id=agent_id,
                    user_id=user_id,
                    trigger_type=trigger_type,
                    organization_id=organization_id,
                    session_id=current_session_id,
                )
                await session.commit()
                return ClaimFireResult(
                    fired=False,
                    sessions_since=0,
                    just_installed=True,
                    state=installed.to_pydantic(),
                )

            count, watermark, tied_ids = await self._aggregate_window(
                session,
                agent_id=agent_id,
                user_id=user_id,
                since=row.last_fired_at,
                tied_session_ids=row.last_fired_tied_session_ids or [],
            )

            if count < threshold:
                # No write; the FOR UPDATE lock releases at commit.
                await session.commit()
                return ClaimFireResult(
                    fired=False,
                    sessions_since=count,
                    just_installed=False,
                    state=row.to_pydantic(),
                )

            # count, watermark, and tied_ids all come from a single aggregate
            # query above, so they are mutually consistent by construction:
            # we can never store a session_id in tied_ids that was not also
            # part of the count. Any message that commits at `watermark`
            # after this query runs falls into the NEXT window via the
            # tie-breaker predicate rather than being silently dropped.
            new_watermark = watermark or row.last_fired_at
            if new_watermark is None:
                new_watermark = datetime.now(dt.timezone.utc)

            row.last_fired_at = new_watermark
            row.last_fired_session_id = current_session_id
            row.last_fired_tied_session_ids = tied_ids
            row.set_updated_at()
            await session.commit()
            await session.refresh(row)
            return ClaimFireResult(
                fired=True,
                sessions_since=count,
                just_installed=False,
                state=row.to_pydantic(),
            )

    @enforce_types
    async def count_distinct_sessions_since(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        tied_session_ids: Optional[List[str]] = None,
    ) -> int:
        """Diagnostic helper: count DISTINCT non-null session_ids whose
        first-appearance timestamp falls inside the "new since cursor"
        window, using the same MIN filter and tie-break as the fire path.
        Not on the hot path; exposed for tests/metrics/admin.
        """
        async with self.session_maker() as session:
            count, _, _ = await self._aggregate_window(
                session,
                agent_id=agent_id,
                user_id=user_id,
                since=since,
                tied_session_ids=tied_session_ids or [],
            )
            return count

    async def _aggregate_window(
        self,
        session,
        *,
        agent_id: str,
        user_id: Optional[str],
        since: Optional[datetime],
        tied_session_ids: List[str],
    ) -> Tuple[int, Optional[datetime], List[str]]:
        """
        Return (count of new sessions, watermark, tied_session_ids) from
        ONE aggregate query so the three results are mutually consistent
        by construction.

        Shape: `GROUP BY session_id` with `MIN(created_at)` per session,
        filtered by a HAVING clause that implements the "new since cursor"
        rule on the first-appearance timestamp:

            (MIN(created_at) > since)
            OR
            (MIN(created_at) = since AND session_id NOT IN tied_session_ids)

        When `since` is None (no cursor installed yet) the window is
        "all sessions for this agent/user" — diagnostic only; the fire
        path installs a cursor before ever reaching this branch.

        Why MIN and not MAX: each session's MIN is fixed once its first
        message is inserted, so a session can satisfy "MIN > cursor" in
        exactly one window, then fail it forever. MAX-based windowing, in
        contrast, lets a session continue producing messages after a fire
        and land in a later window's MAX, double-counting it.

        Why one query: under READ COMMITTED, splitting count and tied-set
        across two statements lets a concurrent commit show up in the
        second but not the first, landing its session_id in tied_ids
        without ever being counted — which would exclude that session from
        the next window forever. A single `GROUP BY` ensures tied_ids is
        always a subset of the counted sessions.
        """
        preds = [
            MessageModel.agent_id == agent_id,
            MessageModel.session_id.isnot(None),
            # Skip soft-deleted messages: if a user wipes conversation
            # history, the sessions those messages belonged to should not
            # keep contributing to the procedural fire threshold. Mirrors
            # the `is_deleted == False` filter in message_manager.
            MessageModel.is_deleted.is_(False),
        ]
        if user_id is not None:
            preds.append(MessageModel.user_id == user_id)

        per_session_min = func.min(MessageModel.created_at).label("first_ts")
        stmt = (
            select(MessageModel.session_id, per_session_min)
            .where(*preds)
            .group_by(MessageModel.session_id)
        )
        if since is not None:
            if tied_session_ids:
                stmt = stmt.having(
                    or_(
                        per_session_min > since,
                        and_(
                            per_session_min == since,
                            MessageModel.session_id.notin_(tied_session_ids),
                        ),
                    )
                )
            else:
                # No tied set to dedup — `>=` is safe and also captures
                # late-visible commits at exactly `since` that were not
                # present when the cursor was last installed.
                stmt = stmt.having(per_session_min >= since)

        result = await session.execute(stmt)
        rows = result.all()
        if not rows:
            return 0, None, []

        count = len(rows)
        watermark = max(row.first_ts for row in rows if row.first_ts is not None)
        tied = [row.session_id for row in rows if row.first_ts == watermark]
        return count, watermark, tied

    async def _install_cursor(
        self,
        *,
        session,
        agent_id: str,
        user_id: str,
        trigger_type: str,
        organization_id: Optional[str],
        session_id: Optional[str],
    ) -> AgentTriggerStateModel:
        """
        Insert a fresh cursor row at `now`. Idempotent: if two workers both
        try to install concurrently, one wins and the other reads back the
        winner's row via the unique constraint.
        """
        now = datetime.now(dt.timezone.utc)
        row = AgentTriggerStateModel(
            agent_id=agent_id,
            user_id=user_id,
            trigger_type=trigger_type,
            organization_id=organization_id,
            last_fired_at=now,
            last_fired_session_id=session_id,
            last_fired_tied_session_ids=[session_id] if session_id else [],
        )
        session.add(row)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = await self._fetch(session, agent_id, user_id, trigger_type)
            if existing is None:
                raise
            return existing
        await session.refresh(row)
        return row

    async def _fetch(
        self,
        session,
        agent_id: str,
        user_id: str,
        trigger_type: str,
    ) -> Optional[AgentTriggerStateModel]:
        stmt = select(AgentTriggerStateModel).where(
            AgentTriggerStateModel.agent_id == agent_id,
            AgentTriggerStateModel.user_id == user_id,
            AgentTriggerStateModel.trigger_type == trigger_type,
            AgentTriggerStateModel.is_deleted.is_(False),
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
