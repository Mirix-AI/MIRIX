import datetime as dt
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from mirix.log import get_logger
from mirix.orm.agent_trigger_state import AgentTriggerState as AgentTriggerStateModel
from mirix.orm.conversation_message import (
    ConversationMessage as ConversationMessageModel,
)
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
                        qualifying batch, enforced by `SELECT ... FOR UPDATE`.
    sessions_since   -- Sealed, not-yet-distilled conversation sessions observed
                        for this (user, org) at decision time (excluding the
                        previous fire's still-unmarked claim). Useful for
                        logging / metrics.
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

    Source of truth is the **Conversation Message Store**, not the agent-loop
    `messages` table: the running counter is the number of SEALED,
    not-yet-distilled sessions for this `(user, organization)`, derived at read
    time. "Sealed" = a strictly-newer distinct session exists (by
    `MIN(created_at)` per session), so the open head of the window never counts;
    "not-yet-distilled" = none of the session's turns has a `distilled_at`.
    Because distillation stamps `distilled_at` on a session's turns, that session
    drops out of the count permanently — the marker IS the dedup, replacing the
    old `messages`-table watermark/tie machinery entirely.

    The fire is intentionally a SOFT, eventually-consistent trigger. It does not
    try to make each fire claim a unique batch; instead the DOWNSTREAM procedural
    auto-dream is serialized per `(user, organization)` (see
    `auto_dream_manager._procedural_dream_lock`) and is idempotent via
    `distilled_at` marking. So a redundant fire (two `trigger_memory_update`
    calls before the first auto-dream has marked its batch) is harmless: the
    second auto-dream run waits on the lock, then re-enumerates the OLDEST
    sealed-undistilled sessions — which is exactly the right next batch, and
    which naturally RETRIES any session a previous run failed to distill (a failed
    session keeps `distilled_at` NULL, stays sealed-undistilled, and is counted
    and re-enumerated until it succeeds).

    The only persisted state is the cursor row (audit/log only — it does NOT gate
    the count, which is derived live from the store):

      * last_fired_at                -- wall-clock of the last fire.
      * last_fired_session_id        -- the in-progress session at fire time.
      * last_fired_tied_session_ids  -- session_ids of the most recent fire's
                                        batch (diagnostics only). NOT used to
                                        exclude from the next count — excluding a
                                        still-undistilled (e.g. failed) session
                                        from its own retry count would strand it,
                                        so the count is always the raw live
                                        sealed-undistilled set.

    Concurrency: `check_and_claim_fire` is the serialization point. It locks the
    cursor row with `SELECT ... FOR UPDATE`, counts sealed-undistilled sessions
    inside the same transaction, and — when the count reaches the threshold —
    advances the cursor before committing. Two concurrent workers on the same
    (agent, user, trigger_type) execute the fire decision strictly in order; a
    redundant fire is absorbed by the downstream lock + idempotent marking.
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
        recording the claimed sessions on the cursor.

        Semantics:
        - No cursor row: install one and return fired=False,
          just_installed=True. The first call only starts the cursor; the
          threshold is evaluated on subsequent calls. (The Conversation Message
          Store holds only real session'd conversations — there is no scaffolding
          backlog to sweep — but the install-then-count step keeps the first
          call cheap and the fire decision uniform.)
        - Cursor exists: inside one transaction, lock it, count SEALED,
          not-yet-distilled sessions for this (user, org) — excluding the
          session_ids claimed by the previous fire that the background distill
          has not yet marked distilled. If count >= threshold, claim the oldest
          `threshold` of them, store that set on the cursor, advance
          `last_fired_at` to wall-clock now, and return fired=True.
        - Cursor exists but count < threshold: no writes, release the lock.
        - organization_id cannot be resolved (neither the call nor the cursor
          carries one): cannot scope the store query, so never fire.
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

            # Scope the store query by (user, org). Prefer the call's org, fall
            # back to the cursor's stored org. Without one we cannot isolate this
            # owner's sessions, so refuse to fire rather than count across orgs.
            effective_org_id = organization_id or row.organization_id
            if effective_org_id is None:
                await session.commit()
                logger.debug(
                    "Skipping procedural trigger: no organization_id to scope the "
                    "conversation store (agent=%s, user=%s).",
                    agent_id,
                    user_id,
                )
                return ClaimFireResult(
                    fired=False,
                    sessions_since=0,
                    just_installed=False,
                    state=row.to_pydantic(),
                )

            # The full set of sealed, not-yet-distilled sessions for this
            # (user, org), OLDEST first. distilled_at is the dedup, so anything an
            # earlier distill already marked is gone; a session that FAILED to
            # distill stays here and is therefore counted again (auto-retry).
            all_sealed = await self._aggregate_sealed_undistilled(
                session,
                user_id=user_id,
                organization_id=effective_org_id,
            )
            count = len(all_sealed)

            if count < threshold:
                # No write; the FOR UPDATE lock releases at commit.
                await session.commit()
                return ClaimFireResult(
                    fired=False,
                    sessions_since=count,
                    just_installed=False,
                    state=row.to_pydantic(),
                )

            # Fire. Record the oldest `threshold` sealed sessions on the cursor for
            # diagnostics (NOT to exclude from future counts — the downstream
            # auto-dream is serialized per owner and idempotent, so a redundant
            # fire is harmless and excluding would strand a failed session from its
            # own retry).
            fired_batch = all_sealed[:threshold]
            row.last_fired_at = datetime.now(dt.timezone.utc)
            row.last_fired_session_id = current_session_id
            row.last_fired_tied_session_ids = fired_batch
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
    async def count_sealed_undistilled_sessions(
        self,
        *,
        user_id: str,
        organization_id: str,
    ) -> int:
        """Diagnostic helper: count SEALED, not-yet-distilled sessions for this
        `(user, organization)` in the Conversation Message Store, using the same
        seal/distill aggregate as the fire path. Not on the hot path; exposed for
        tests/metrics/admin. (Does NOT subtract any in-flight claims — that is a
        fire-path concern; this is the raw sealed-undistilled count.)
        """
        async with self.session_maker() as session:
            sealed = await self._aggregate_sealed_undistilled(
                session,
                user_id=user_id,
                organization_id=organization_id,
            )
            return len(sealed)

    async def _aggregate_sealed_undistilled(
        self,
        session,
        *,
        user_id: str,
        organization_id: str,
    ) -> List[str]:
        """
        Return ALL SEALED, not-yet-distilled session_ids for this
        `(user, organization)` in the Conversation Message Store, OLDEST first,
        from ONE aggregate query.

        Shape mirrors
        `ConversationMessageManager.list_sealed_undistilled_sessions`: one
        `GROUP BY session_id` with `MIN(created_at)` (first-appearance order) and
        `MAX(distilled_at)` (a session is undistilled iff this is NULL on every
        turn). The single newest session by `MIN(created_at)` is the OPEN HEAD of
        the window and is dropped — "sealed" means a strictly-newer distinct
        session exists, so the in-progress head is never counted. Among the
        remaining sealed sessions, the undistilled ones are returned in oldest-
        first order.

        Exclusion of in-flight claims and the threshold cap are deliberately the
        CALLER's job (it needs the full set to prune its stale-claim list against),
        so this aggregate stays a pure "what is sealed and undistilled right now".
        """
        first_ts = func.min(ConversationMessageModel.created_at).label("first_ts")
        # MAX(distilled_at) IS NULL <=> no turn of this session has been distilled
        # (any non-null turn makes the MAX non-null), so the session is "open".
        last_distilled = func.max(ConversationMessageModel.distilled_at).label(
            "last_distilled"
        )
        stmt = (
            select(
                ConversationMessageModel.session_id,
                first_ts,
                last_distilled,
            )
            .where(
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                # Skip soft-deleted turns: if a conversation is wiped, the
                # sessions it belonged to should not keep contributing to the
                # procedural fire threshold. Mirrors the manager's filter.
                ConversationMessageModel.is_deleted.is_(False),
            )
            .group_by(ConversationMessageModel.session_id)
            .order_by(first_ts.asc())
        )
        result = await session.execute(stmt)
        grouped = result.all()

        if len(grouped) <= 1:
            # 0 or 1 session: nothing is sealed (no strictly-newer session).
            return []

        # The last row (newest MIN(created_at)) is the open head — drop it. Among
        # the remaining sealed sessions, keep the undistilled ones, oldest-first.
        sealed = grouped[:-1]
        return [grp.session_id for grp in sealed if grp.last_distilled is None]

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
            # Start the claimed-set EMPTY. Under the distilled_at-based scheme this
            # list holds session_ids CLAIMED by a fire (excluded from the next
            # count until the background distill marks them distilled) — seeding it
            # with the in-progress session_id would wrongly exclude the very first
            # real session from the first threshold window, shifting the cadence by
            # one (the 6th distinct session would only see 4 countable sealed
            # sessions). No fire has happened yet, so nothing is claimed.
            last_fired_tied_session_ids=[],
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
