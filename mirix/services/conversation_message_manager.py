"""Manager for the Conversation Message Store.

Async-only, mirroring `SkillExperienceManager` / `MessageManager`: it grabs
sessions via `self.session_maker` (the server's `db_context`) and never calls
`asyncio.run()` (the server event loop is already running).

This store is the SINGLE source of truth for procedural-memory (skill)
distillation. It holds only external conversation turns that arrived with a
`session_id`, with their REAL `user`/`assistant` roles preserved. The five
methods below are the stable contract the ingestion seam, the trigger/cadence
logic, and the distiller all depend on:

  - record_turns                      -- append a batch of turns to a session
  - count_distinct_sessions           -- how many distinct sessions exist
  - list_sealed_undistilled_sessions  -- oldest sealed, not-yet-distilled ids
  - list_turns_for_session            -- one session's turns, ascending
  - mark_sessions_distilled           -- advance the rolling barrier

"Sealed" = a strictly newer distinct `session_id` exists (by MIN(created_at)),
so the open head of the window is never distilled while it may still grow.
Everything is scoped per `(user, organization)` so one user's sessions never
count toward or leak into another's learning window.
"""

from __future__ import annotations

import datetime as dt
from datetime import timedelta
from typing import List

from sqlalchemy import func, select, update

from mirix.client.utils import get_utc_time
from mirix.log import get_logger
from mirix.orm.conversation_message import (
    ConversationMessage as ConversationMessageModel,
)
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.conversation_message import (
    ConversationMessage as PydanticConversationMessage,
    ConversationMessageCreate,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)


class ConversationMessageManager:
    """Persist and query external conversation turns for skill distillation."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def record_turns(
        self,
        *,
        session_id: str,
        user_id: str,
        organization_id: str,
        turns: List[dict],
        actor: PydanticClient,
    ) -> List[PydanticConversationMessage]:
        """Append `turns` to `session_id`, preserving their given order.

        `turns` is a list of `{"role": "user"|"assistant", "content": str}`.
        Each turn is validated through `ConversationMessageCreate` first so an
        invalid `role` (or over-length content / malformed session_id) is
        rejected up front — the DB columns are plain String/Text, so without
        this an invalid value would commit and only blow up later in
        `to_pydantic()` / list.

        created_at ordering is preserved EXPLICITLY: a single batch shares one
        transaction, and the server-side `func.now()` default would collapse all
        rows in that transaction to one timestamp, destroying intra-batch order.
        We instead stamp a strictly increasing `created_at` per turn — anchored
        at the current MAX(created_at) for this session so a later call's turns
        always sort after an earlier call's, making multi-call sessions a single
        ordered unit. Returns the persisted rows in insertion order.
        """
        if not turns:
            return []

        validated = [
            ConversationMessageCreate(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                role=turn.get("role"),
                content=turn.get("content", ""),
            )
            for turn in turns
        ]

        async with self.session_maker() as session:
            # Anchor after the latest existing turn for this session so turns
            # from successive record_turns calls accumulate in order. Scope the
            # MAX by (org, user, session) for the same isolation as everything
            # else here.
            max_stmt = select(func.max(ConversationMessageModel.created_at)).where(
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.session_id == session_id,
                ConversationMessageModel.is_deleted.is_(False),
            )
            result = await session.execute(max_stmt)
            last_ts = result.scalar_one_or_none()

            base = get_utc_time()
            if last_ts is not None:
                # Normalize to tz-aware UTC for a safe comparison with `base`.
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=dt.timezone.utc)
                if last_ts >= base:
                    base = last_ts + timedelta(microseconds=1)

            rows = []
            for offset, payload in enumerate(validated):
                row = ConversationMessageModel(
                    session_id=payload.session_id,
                    user_id=payload.user_id,
                    organization_id=payload.organization_id,
                    role=payload.role,
                    content=payload.content,
                    # +offset microseconds keeps a strict, stable order within
                    # the batch even though they share one transaction.
                    created_at=base + timedelta(microseconds=offset),
                )
                if actor is not None:
                    row._set_created_and_updated_by_fields(actor.id)
                session.add(row)
                rows.append(row)

            await session.commit()
            for row in rows:
                await session.refresh(row)
            return [row.to_pydantic() for row in rows]

    @enforce_types
    async def count_distinct_sessions(
        self,
        *,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
        only_undistilled: bool = False,
    ) -> int:
        """Count DISTINCT `session_id`s for this `(user, organization)`.

        With `only_undistilled=True`, counts only sessions that have NOT been
        distilled. "Undistilled" is a SESSION-level property: a session counts
        iff NONE of its turns have a `distilled_at` (i.e. `MAX(distilled_at) IS
        NULL`). A row-level `distilled_at IS NULL` filter would over-count a
        previously-distilled session that later gained a fresh turn — and would
        disagree with `list_sealed_undistilled_sessions`, which already seals on
        `MAX(distilled_at)`. Soft-deleted rows are excluded.
        """
        async with self.session_maker() as session:
            preds = [
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.is_deleted.is_(False),
            ]
            if only_undistilled:
                # Group by session and keep only those whose MAX(distilled_at) is
                # NULL, then count the surviving groups. Wrapping the grouped
                # query in a COUNT subquery keeps this a single round-trip and
                # dialect-safe (no DISTINCT-on-HAVING gymnastics).
                grouped = (
                    select(ConversationMessageModel.session_id)
                    .where(*preds)
                    .group_by(ConversationMessageModel.session_id)
                    .having(func.max(ConversationMessageModel.distilled_at).is_(None))
                    .subquery()
                )
                stmt = select(func.count()).select_from(grouped)
            else:
                stmt = select(
                    func.count(func.distinct(ConversationMessageModel.session_id))
                ).where(*preds)
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

    @enforce_types
    async def list_sealed_undistilled_sessions(
        self,
        *,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
        limit: int,
    ) -> List[str]:
        """Return up to `limit` sealed, not-yet-distilled session_ids, OLDEST first.

        "Sealed" = a strictly newer distinct session exists, by first-appearance
        time (`MIN(created_at)` per session). The newest session is the open head
        of the window and is never returned (it may still grow). Of the sealed
        sessions, only those that have NOT been distilled (`distilled_at IS NULL`
        on their turns) are returned.

        Shape mirrors `agent_trigger_state_manager._aggregate_window`: one
        `GROUP BY session_id` with `MIN(created_at)` per session, so ordering and
        sealing come from a single dialect-safe aggregate. Soft-deleted rows are
        excluded.
        """
        if limit <= 0:
            return []

        async with self.session_maker() as session:
            first_ts = func.min(ConversationMessageModel.created_at).label("first_ts")
            # NULL distilled_at on EVERY turn of a session <=> the session is
            # undistilled. MAX(distilled_at) IS NULL captures exactly that (any
            # non-null turn makes the MAX non-null), so a session is "open" iff
            # its MAX(distilled_at) is NULL.
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

        # The last row (newest MIN(created_at)) is the open head — drop it.
        # Among the remaining sealed sessions, keep the undistilled ones,
        # oldest-first, capped at `limit`.
        sealed = grouped[:-1]
        out: List[str] = []
        for row in sealed:
            if row.last_distilled is None:
                out.append(row.session_id)
                if len(out) >= limit:
                    break
        return out

    @enforce_types
    async def list_turns_for_session(
        self,
        *,
        session_id: str,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
    ) -> List[PydanticConversationMessage]:
        """Return one session's turns for this `(user, org)`, ascending by created_at."""
        async with self.session_maker() as session:
            stmt = (
                select(ConversationMessageModel)
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id == session_id,
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .order_by(ConversationMessageModel.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_pydantic() for row in rows]

    @enforce_types
    async def mark_sessions_distilled(
        self,
        *,
        session_ids: List[str],
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
    ) -> int:
        """Stamp `distilled_at` on all turns of the given sessions; idempotent.

        Only turns still un-distilled (`distilled_at IS NULL`) are stamped, so a
        re-run leaves already-distilled rows untouched and the rolling barrier
        advances without reprocessing history. Scoped to this `(user, org)` so a
        caller can never mark another owner's session. Returns the number of
        rows updated.

        A bulk UPDATE (not load-modify-save) is correct here: `distilled_at` is a
        write-only barrier marker with no Redis cache and no `updated_at`
        semantics the distiller depends on, so the simple set is both faster and
        avoids loading potentially large transcripts into memory.
        """
        if not session_ids:
            return 0

        now = get_utc_time()
        async with self.session_maker() as session:
            stmt = (
                update(ConversationMessageModel)
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id.in_(session_ids),
                    ConversationMessageModel.distilled_at.is_(None),
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .values(distilled_at=now)
            )
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)
