"""Conversation-session lifecycle.

One ``conversation_session`` row per (organization, user, session_id) tracks a
session of the Conversation Message Store: first/last message time, message
count, whether it has ENDED, how often it was reopened, and the skill learner's
per-session bookkeeping. This manager owns every state transition of that row:

* **write** (``lock_for_write`` + ``apply_write``, called by
  ``ConversationMessageManager.record_turns`` inside the message transaction):
  upsert the row, lock it ``SELECT ... FOR UPDATE``, fix the tag on the first
  write (a later mismatching tag is kept out and reported as a warning),
  reopen an ended session (``closed_at = NULL``, ``reopen_count + 1``), then
  advance ``last_message_at`` / ``message_count``.
* **explicit close** (``close_session``): conditional UPDATE, idempotent.
* **idle materialization** (``materialize_idle``): a session whose last message
  is older than ``MIRIX_SESSION_IDLE_TIMEOUT_SECONDS`` has ended; the first check
  that sees it writes ``closed_at = last_message_at + timeout`` with a
  conditional UPDATE, so a concurrent write (which moves ``last_message_at``)
  always wins.
* **pending selection** (``list_pending_distill_sessions``): the sessions a
  learning run may claim.

A session ends ONLY by an explicit close or by idling past the timeout. There
is no ordering rule between sessions: several sessions of one user may be open
at the same time.

"Pending distill" = all of:
  1. the session has ended;
  2. it has message rows with ``distilled_at IS NULL`` (distillation state is
     per row, so a reopened session's new rows are pending again while its
     earlier rows stay distilled);
  3. ``distill_failed_at IS NULL`` and ``distill_retry_after`` is unset or past;
  4. no queued/running learning run has claimed it;
  and it is a task session: ``session_tag = 'task'``, or NULL for legacy rows
  backfilled from before session tags existed (fail-open, as before).

Ownership: a session belongs to ONE meta agent (``agent_id``), the scope that
learns from it. A write that names another meta agent is refused
(:class:`SessionOwnershipConflict`, HTTP 409) — merging it would feed one
client's conversation into another client's skills. Legacy rows backfilled
without an agent are attributed (:meth:`adopt_unattributed_sessions`) to the
only live meta agent of the client that wrote the session's messages (or, when
no writer is recorded, to the organization's only live meta agent); a legacy
row nobody can be shown to own stays unattributed and is never learned from.
A session whose meta agent no longer exists (deleted — e.g. a client that
re-created its meta agent) is ORPHANED: the writing client's new meta agent
adopts it on its next write, and the sweep / a claim / a close re-attribute it
by the same writer rule, instead of the session answering 409 and never being
learned from again. An orphan is never attributed by the organization-wide
fallback of legacy rows (it would hand one client's transcript to another
client's scope): with no recorded writer it goes only to the client that owned
it — the deleted agent's creator, else the client whose write created the
session row. Its learning moves with it: the session's still-pending
experiences are re-attributed to the adopting meta agent in the same
transaction (a soft-deleted agent keeps them), and a meta agent's hard delete
first marks the rows of sessions with pending experiences undistilled (the
experiences go with the agent), so the adopting scope distills them again.

The clock is injectable (``clock=``) so every transition is testable without
sleeping. All I/O is async; nothing here blocks the event loop.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

from sqlalchemy import delete, literal, or_, select, update
from sqlalchemy.types import Interval

from mirix.constants import (
    DEFAULT_SESSION_TAG,
    SESSION_IDLE_TIMEOUT_SECONDS,
    SESSION_TAG_TASK,
)
from mirix.log import get_logger
from mirix.orm.conversation_message import (
    ConversationMessage as ConversationMessageModel,
)
from mirix.orm.conversation_session import (
    ConversationSession as ConversationSessionModel,
)
from mirix.orm.skill_learning_run import SkillLearningRun as SkillLearningRunModel
from mirix.schemas.conversation_session import (
    ConversationSession as PydanticConversationSession,
)
from mirix.schemas.skill_learning_run import ACTIVE_LEARNING_RUN_STATUSES

logger = get_logger(__name__)

Clock = Callable[[], dt.datetime]

CLOSE_REASON_EXPLICIT = "explicit"
CLOSE_REASON_IDLE = "idle"


def utc_now() -> dt.datetime:
    """Timezone-aware UTC now (the default clock)."""
    return dt.datetime.now(dt.timezone.utc)


def as_utc(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """Normalize a datetime to tz-aware UTC.

    SQLite hands DateTime(timezone=True) columns back as naive values (they are
    stored as UTC text); Postgres returns aware ones. Everything this module
    compares is UTC, so a naive value is UTC.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def dialect_name(db) -> str:
    """Dialect of an AsyncSession ('postgresql', 'sqlite', ...), '' if unknown."""
    bind = getattr(db, "bind", None)
    if bind is None:
        try:
            bind = db.get_bind()
        except Exception:  # noqa: BLE001 — unknown binds just skip dialect features
            return ""
    dialect = getattr(bind, "dialect", None)
    return getattr(dialect, "name", "") or ""


def learnable_tag_clause():
    """Task sessions, plus legacy untagged (NULL) sessions (fail-open).

    Sessions tagged 'conversation' never feed skill learning.
    """
    return or_(
        ConversationSessionModel.session_tag == SESSION_TAG_TASK,
        ConversationSessionModel.session_tag.is_(None),
    )


def undistilled_rows_exist_clause():
    """EXISTS a not-yet-distilled, live message row of the correlated session."""
    return (
        select(ConversationMessageModel.id)
        .where(
            ConversationMessageModel.organization_id
            == ConversationSessionModel.organization_id,
            ConversationMessageModel.user_id == ConversationSessionModel.user_id,
            ConversationMessageModel.session_id == ConversationSessionModel.session_id,
            ConversationMessageModel.distilled_at.is_(None),
            ConversationMessageModel.is_deleted.is_(False),
        )
        .exists()
    )


def live_owner_clause():
    """EXISTS the correlated session's meta agent, not deleted."""
    from mirix.orm.agent import Agent as AgentModel

    return (
        select(AgentModel.id)
        .where(
            AgentModel.id == ConversationSessionModel.agent_id,
            AgentModel.is_deleted.is_(False),
        )
        .exists()
    )


def ownerless_clause():
    """No live owner: never attributed (legacy NULL) or orphaned (its meta
    agent was deleted). Both are adopted by the same attribution rule."""
    return or_(
        ConversationSessionModel.agent_id.is_(None),
        ~live_owner_clause(),
    )


class SessionOwnershipConflict(Exception):
    """A write names a meta agent other than the one that owns the session."""

    def __init__(
        self, session_id: str, owner_agent_id: Optional[str], writer_agent_id: Optional[str]
    ):
        self.session_id = session_id
        self.owner_agent_id = owner_agent_id
        self.writer_agent_id = writer_agent_id
        super().__init__(
            f"session {session_id!r} belongs to another meta agent; it cannot be "
            "written through this one (use a different session_id)"
        )


@dataclass
class SessionWriteState:
    """What a write did to its session row (returned to the ingestion caller)."""

    created: bool
    reopened: bool
    effective_tag: Optional[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CloseOutcome:
    """Result of an explicit close.

    ``already_closed`` — the session had ended before this call (explicitly,
    or by idling). ``newly_ended`` — this call is the one that recorded its end:
    it closed it, or it materialized an idle end nobody had recorded yet. A
    trigger check is due exactly when ``newly_ended`` is true, so an idle
    session the user closes explicitly still counts toward N at once, while a
    repeated close never re-triggers.
    """

    session: PydanticConversationSession
    already_closed: bool
    newly_ended: bool = False


class ConversationSessionManager:
    """State transitions of ``conversation_session`` rows (see module docstring)."""

    def __init__(
        self,
        *,
        session_maker=None,
        clock: Optional[Clock] = None,
        idle_timeout_seconds: Optional[int] = None,
    ):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        self.clock: Clock = clock or utc_now
        self.idle_timeout_seconds = (
            SESSION_IDLE_TIMEOUT_SECONDS
            if idle_timeout_seconds is None
            else int(idle_timeout_seconds)
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def now(self) -> dt.datetime:
        return as_utc(self.clock())

    @property
    def idle_timeout(self) -> dt.timedelta:
        return dt.timedelta(seconds=self.idle_timeout_seconds)

    def idle_cutoff(self, now: Optional[dt.datetime] = None) -> dt.datetime:
        """A still-open session whose last message is before this has ended."""
        return (as_utc(now) if now is not None else self.now()) - self.idle_timeout

    def is_ended(self, row, now: Optional[dt.datetime] = None) -> bool:
        """Explicitly/idle-closed, or idle past the timeout but not yet materialized."""
        if row.closed_at is not None:
            return True
        last = as_utc(row.last_message_at)
        return last is not None and last < self.idle_cutoff(now)

    @staticmethod
    def _scope_filters(
        organization_id: Optional[str],
        user_id: Optional[str],
        agent_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> list:
        filters = []
        if organization_id is not None:
            filters.append(ConversationSessionModel.organization_id == organization_id)
        if user_id is not None:
            filters.append(ConversationSessionModel.user_id == user_id)
        if agent_id is not None:
            filters.append(ConversationSessionModel.agent_id == agent_id)
        if session_id is not None:
            filters.append(ConversationSessionModel.session_id == session_id)
        return filters

    # ------------------------------------------------------------------ #
    # Write path (runs inside the caller's message transaction)            #
    # ------------------------------------------------------------------ #

    async def lock_for_write(
        self,
        db,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        agent_id: Optional[str],
        requested_tag: Optional[str],
        actor_id: Optional[str],
        now: dt.datetime,
    ) -> Tuple[ConversationSessionModel, SessionWriteState]:
        """Upsert and lock the session row for a write, in ``db``'s transaction.

        The row is created with ``INSERT ... ON CONFLICT DO NOTHING`` (so two
        concurrent first writes never fail on the unique key) and then locked
        with ``SELECT ... FOR UPDATE``: a concurrent learning-run claim or idle
        materialization of this session waits for the write to commit, and the
        write sees their committed state. The caller inserts its messages and
        calls :meth:`apply_write` before committing.

        Tag rule: the first write fixes the tag (``requested_tag`` or
        ``DEFAULT_SESSION_TAG`` when the first write names none). A later write
        naming a different tag is NOT an error — the stored tag is kept and a
        warning is returned. A later write naming no tag keeps the stored tag.
        A legacy row with a NULL tag adopts the first tag a write names.

        Reopen rule: a write to an ended session (explicitly closed, idle
        closed, or idle but not yet materialized) reopens it: ``closed_at`` and
        ``close_reason`` are cleared and ``reopen_count`` goes up by one. Its
        already-distilled rows stay distilled; the new rows are pending once the
        session ends again.
        """
        now = as_utc(now)
        first_tag = requested_tag or DEFAULT_SESSION_TAG
        created = await self._insert_if_missing(
            db,
            organization_id=organization_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            session_tag=first_tag,
            actor_id=actor_id,
            now=now,
        )
        stmt = (
            select(ConversationSessionModel)
            .where(
                *self._scope_filters(organization_id, user_id, None, session_id),
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        row = (await db.execute(stmt)).scalar_one()

        warnings: List[str] = []
        stored_tag = row.session_tag
        if created:
            effective_tag = stored_tag
        elif stored_tag is None:
            effective_tag = requested_tag
            if requested_tag is not None:
                row.session_tag = requested_tag
        else:
            effective_tag = stored_tag
            if requested_tag is not None and requested_tag != stored_tag:
                warnings.append(
                    f"session_tag mismatch: session {session_id!r} was first "
                    f"written as {stored_tag!r}; the request's {requested_tag!r} "
                    f"is ignored and the session stays {stored_tag!r}."
                )

        if (
            agent_id is not None
            and row.agent_id is not None
            and row.agent_id != agent_id
            and await self.agent_is_live(row.agent_id, db=db)
        ):
            # The session is another (live) meta agent's learning scope:
            # storing this write there would feed it into that scope's skills.
            raise SessionOwnershipConflict(session_id, row.agent_id, agent_id)
        if agent_id is not None and row.agent_id != agent_id:
            # No live owner: a legacy (backfilled) row that never knew its meta
            # agent, or an orphan whose meta agent was deleted (e.g. the client
            # re-created it). The writer adopts it only if nobody else wrote
            # the session's rows — so a client's new meta agent inherits its
            # own sessions, and never another client's.
            if not created:
                writers = await self._session_writers(
                    db,
                    organization_id=organization_id,
                    user_id=user_id,
                    session_ids=[session_id],
                )
                known = {w for w in writers.get(session_id, set()) if w}
                if known and actor_id not in known:
                    raise SessionOwnershipConflict(session_id, row.agent_id, agent_id)
                if not known and row.agent_id is not None:
                    # An orphan no row names a writer of: only its owning
                    # client may adopt it.
                    owner = await self.orphan_owner(
                        db,
                        previous_agent_id=row.agent_id,
                        session_creator=row._created_by_id,
                    )
                    if owner is None or owner != actor_id:
                        raise SessionOwnershipConflict(
                            session_id, row.agent_id, agent_id
                        )
            previous = row.agent_id
            row.agent_id = agent_id
            if previous is not None:
                logger.info(
                    "Session %s of user %s: meta agent %s no longer exists; "
                    "adopted by %s",
                    session_id,
                    user_id,
                    previous,
                    agent_id,
                )
                await self.move_pending_experiences(
                    db,
                    organization_id=organization_id,
                    user_id=user_id,
                    session_id=session_id,
                    from_agent_id=previous,
                    to_agent_id=agent_id,
                )

        reopened = False
        if not created and self.is_ended(row, now):
            row.closed_at = None
            row.close_reason = None
            row.reopen_count = int(row.reopen_count or 0) + 1
            # The new rows are a new segment: an earlier segment's distill
            # failures / backoff must not keep them from being learned.
            row.distill_failures = 0
            row.distill_failed_at = None
            row.distill_retry_after = None
            row.distill_infra_failures = 0
            reopened = True

        return row, SessionWriteState(
            created=created,
            reopened=reopened,
            effective_tag=effective_tag,
            warnings=warnings,
        )

    @staticmethod
    def apply_write(
        row: ConversationSessionModel,
        *,
        first_message_at: dt.datetime,
        last_message_at: dt.datetime,
        message_count: int,
        created: bool,
        actor_id: Optional[str],
        now: dt.datetime,
        truncated_messages: int = 0,
        truncated_chars: int = 0,
    ) -> None:
        """Advance the locked row after the caller inserted ``message_count`` rows.

        ``truncated_messages`` / ``truncated_chars`` are what the storage cap
        cut from those rows; they accumulate on the session.
        """
        if created:
            row.first_message_at = as_utc(first_message_at)
        else:
            current_first = as_utc(row.first_message_at)
            if current_first is None or as_utc(first_message_at) < current_first:
                row.first_message_at = as_utc(first_message_at)
        current_last = as_utc(row.last_message_at)
        new_last = as_utc(last_message_at)
        if current_last is None or new_last > current_last:
            row.last_message_at = new_last
        row.message_count = int(row.message_count or 0) + int(message_count)
        if truncated_messages:
            row.truncated_message_count = int(row.truncated_message_count or 0) + int(
                truncated_messages
            )
            row.truncated_chars = int(row.truncated_chars or 0) + int(truncated_chars)
        row.updated_at = as_utc(now)
        if actor_id:
            row._last_updated_by_id = actor_id

    async def _insert_if_missing(
        self,
        db,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        agent_id: Optional[str],
        session_tag: Optional[str],
        actor_id: Optional[str],
        now: dt.datetime,
    ) -> bool:
        """INSERT the row unless (org, user, session_id) exists. True if inserted."""
        values = dict(
            id=f"convsess-{uuid.uuid4()}",
            organization_id=organization_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            session_tag=session_tag,
            first_message_at=now,
            last_message_at=now,
            message_count=0,
            reopen_count=0,
            distill_failures=0,
            distill_infra_failures=0,
            truncated_message_count=0,
            truncated_chars=0,
            created_at=now,
            updated_at=now,
            is_deleted=False,
            _created_by_id=actor_id,
            _last_updated_by_id=actor_id,
        )
        name = dialect_name(db)
        if name in ("postgresql", "sqlite"):
            if name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as dialect_insert
            else:
                from sqlalchemy.dialects.sqlite import insert as dialect_insert
            stmt = (
                dialect_insert(ConversationSessionModel)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=["organization_id", "user_id", "session_id"]
                )
                .returning(ConversationSessionModel.id)
            )
            result = await db.execute(stmt)
            return result.first() is not None

        # Other dialects: look first; a racing insert surfaces as IntegrityError.
        existing = await db.execute(
            select(ConversationSessionModel.id).where(
                *self._scope_filters(organization_id, user_id, None, session_id)
            )
        )
        if existing.first() is not None:
            return False
        db.add(ConversationSessionModel(**values))
        await db.flush()
        return True

    # ------------------------------------------------------------------ #
    # Reads                                                                #
    # ------------------------------------------------------------------ #

    async def get_session(
        self, *, organization_id: str, user_id: str, session_id: str
    ) -> Optional[PydanticConversationSession]:
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    select(ConversationSessionModel).where(
                        *self._scope_filters(organization_id, user_id, None, session_id)
                    )
                )
            ).scalar_one_or_none()
            return row.to_pydantic() if row is not None else None

    async def agent_is_live(self, agent_id: Optional[str], *, db=None) -> bool:
        """True iff ``agent_id`` names an agent that exists and is not deleted.

        A session whose owner is not live is orphaned (see the module
        docstring). With ``db`` the check joins the caller's transaction.
        """
        if agent_id is None:
            return False
        from mirix.orm.agent import Agent as AgentModel

        stmt = select(AgentModel.id).where(
            AgentModel.id == agent_id, AgentModel.is_deleted.is_(False)
        )
        if db is not None:
            return (await db.execute(stmt)).first() is not None
        async with self.session_maker() as own_db:
            return (await own_db.execute(stmt)).first() is not None

    # ------------------------------------------------------------------ #
    # Ending a session                                                     #
    # ------------------------------------------------------------------ #

    async def close_session(
        self,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
    ) -> Optional[CloseOutcome]:
        """Explicitly end a session. ``None`` when it does not exist.

        ``closed_at = now``, ``close_reason = 'explicit'`` via a conditional
        UPDATE (``WHERE closed_at IS NULL``), so closing twice — or closing a
        session that already ended by idling — changes nothing and reports
        ``already_closed=True``. Idempotence is what keeps a repeated close from
        firing a second learning run.
        """
        now = self.now()
        async with self.session_maker() as db:
            filters = self._scope_filters(organization_id, user_id, None, session_id)
            exists_row = (
                await db.execute(select(ConversationSessionModel.id).where(*filters))
            ).first()
            if exists_row is None:
                return None
            # An idle session has already ended: record that first so the explicit
            # close below is a no-op and reports already_closed.
            materialized = await self.materialize_idle(
                db=db,
                organization_id=organization_id,
                user_id=user_id,
                session_id=session_id,
                now=now,
            )
            result = await db.execute(
                update(ConversationSessionModel)
                .where(
                    ConversationSessionModel.id == exists_row[0],
                    ConversationSessionModel.closed_at.is_(None),
                )
                .values(
                    closed_at=now,
                    close_reason=CLOSE_REASON_EXPLICIT,
                    updated_at=now,
                )
            )
            closed_now = int(result.rowcount or 0) == 1
            await db.commit()
            row = (
                await db.execute(
                    select(ConversationSessionModel)
                    .where(ConversationSessionModel.id == exists_row[0])
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()
            return CloseOutcome(
                session=row.to_pydantic(),
                already_closed=not closed_now,
                newly_ended=closed_now or materialized > 0,
            )

    async def close_open_sessions(
        self,
        *,
        organization_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """Explicitly close every open learnable (task / legacy) session of a scope.

        Used by ``flush(close_open_sessions=True)``. Each session is
        closed in its own short transaction (conditional UPDATE), so this never
        holds several session-row locks at once and cannot deadlock with a
        concurrent claim or write. Returns the session_ids this call closed.
        """
        now = self.now()
        if agent_id is not None:
            # Legacy sessions this scope can be shown to own join it first.
            await self.adopt_unattributed_sessions(
                organization_id=organization_id, user_id=user_id, agent_id=agent_id
            )
        async with self.session_maker() as db:
            rows = (
                await db.execute(
                    select(
                        ConversationSessionModel.id, ConversationSessionModel.session_id
                    )
                    .where(
                        *self._scope_filters(organization_id, user_id, agent_id),
                        ConversationSessionModel.closed_at.is_(None),
                        ConversationSessionModel.is_deleted.is_(False),
                        learnable_tag_clause(),
                    )
                    .order_by(ConversationSessionModel.first_message_at.asc())
                )
            ).all()
        closed: List[str] = []
        for row_id, sid in rows:
            async with self.session_maker() as db:
                result = await db.execute(
                    update(ConversationSessionModel)
                    .where(
                        ConversationSessionModel.id == row_id,
                        ConversationSessionModel.closed_at.is_(None),
                    )
                    .values(
                        closed_at=now,
                        close_reason=CLOSE_REASON_EXPLICIT,
                        updated_at=now,
                    )
                )
                await db.commit()
                if int(result.rowcount or 0) == 1:
                    closed.append(sid)
        return closed

    async def materialize_idle(
        self,
        *,
        db=None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        now: Optional[dt.datetime] = None,
    ) -> int:
        """Record ``closed_at`` for open sessions that idled past the timeout.

        ``closed_at = last_message_at + timeout``, ``close_reason = 'idle'``,
        written by a conditional UPDATE ``WHERE closed_at IS NULL AND
        last_message_at < cutoff``: a write that lands first moves
        ``last_message_at`` forward and the UPDATE matches nothing; a write that
        lands after sees an ended session and reopens it. Returns the number of
        sessions materialized.

        With ``db`` (a trigger claim, serialized per scope by the cursor lock)
        the updates join the caller's transaction and the caller commits.
        Without it (the sweeper, across all scopes) every session is updated in
        its own short transaction, so a global pass never holds several
        session-row locks at once and cannot deadlock with a claim.
        """
        now = as_utc(now) if now is not None else self.now()
        cutoff = self.idle_cutoff(now)
        filters = [
            *self._scope_filters(organization_id, user_id, agent_id, session_id),
            ConversationSessionModel.closed_at.is_(None),
            ConversationSessionModel.last_message_at < cutoff,
        ]

        if db is None:
            async with self.session_maker() as read_db:
                rows = (
                    await read_db.execute(
                        select(
                            ConversationSessionModel.id,
                            ConversationSessionModel.last_message_at,
                        ).where(*filters)
                    )
                ).all()
            count = 0
            for row_id, last_message_at in rows:
                async with self.session_maker() as own_db:
                    count += await self._materialize_one(
                        own_db, row_id, last_message_at, cutoff, now
                    )
                    await own_db.commit()
            return count

        if dialect_name(db) == "postgresql":
            # Set-based; Postgres re-checks the WHERE clause on rows a concurrent
            # write updated, so a write that moved last_message_at wins.
            result = await db.execute(
                update(ConversationSessionModel)
                .where(*filters)
                .values(
                    closed_at=ConversationSessionModel.last_message_at
                    + literal(self.idle_timeout, Interval()),
                    close_reason=CLOSE_REASON_IDLE,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            return int(result.rowcount or 0)

        rows = (
            await db.execute(
                select(
                    ConversationSessionModel.id,
                    ConversationSessionModel.last_message_at,
                ).where(*filters)
            )
        ).all()
        count = 0
        for row_id, last_message_at in rows:
            count += await self._materialize_one(
                db, row_id, last_message_at, cutoff, now
            )
        return count

    async def _materialize_one(
        self, db, row_id: str, last_message_at, cutoff: dt.datetime, now: dt.datetime
    ) -> int:
        """Conditional per-row idle close. The observed ``last_message_at`` is
        part of the condition, so ``closed_at`` is exactly observed + timeout or
        nothing is written."""
        result = await db.execute(
            update(ConversationSessionModel)
            .where(
                ConversationSessionModel.id == row_id,
                ConversationSessionModel.closed_at.is_(None),
                ConversationSessionModel.last_message_at == last_message_at,
                ConversationSessionModel.last_message_at < cutoff,
            )
            .values(
                closed_at=as_utc(last_message_at) + self.idle_timeout,
                close_reason=CLOSE_REASON_IDLE,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Pending selection                                                    #
    # ------------------------------------------------------------------ #

    def pending_filters(self, now: dt.datetime) -> list:
        """Conditions 1-3 of "pending distill" plus the task-tag rule.

        Condition 4 (not claimed by a queued/running run) is applied by
        :meth:`list_pending_distill_sessions` from the runs' claim lists.
        """
        return [
            ConversationSessionModel.is_deleted.is_(False),
            learnable_tag_clause(),
            or_(
                ConversationSessionModel.closed_at.is_not(None),
                ConversationSessionModel.last_message_at < self.idle_cutoff(now),
            ),
            ConversationSessionModel.distill_failed_at.is_(None),
            or_(
                ConversationSessionModel.distill_retry_after.is_(None),
                ConversationSessionModel.distill_retry_after <= now,
            ),
            undistilled_rows_exist_clause(),
        ]

    async def claimed_session_ids(
        self, db, *, organization_id: str, user_id: str
    ) -> Set[str]:
        """Union of the sessions claimed by every queued/running run of the user.

        Deliberately wider than one meta agent: a session belongs to exactly one
        scope, so the union over the user's runs never hides a claimable session,
        and it can never let two live runs share one.
        """
        rows = (
            await db.execute(
                select(SkillLearningRunModel.session_ids).where(
                    SkillLearningRunModel.organization_id == organization_id,
                    SkillLearningRunModel.user_id == user_id,
                    SkillLearningRunModel.status.in_(ACTIVE_LEARNING_RUN_STATUSES),
                    SkillLearningRunModel.is_deleted.is_(False),
                )
            )
        ).all()
        claimed: Set[str] = set()
        for (session_ids,) in rows:
            claimed.update(session_ids or [])
        return claimed

    async def list_pending_distill_sessions(
        self,
        *,
        organization_id: str,
        user_id: str,
        agent_id: str,
        db=None,
        now: Optional[dt.datetime] = None,
        exclude_claimed: bool = True,
        for_update: bool = False,
    ) -> List[str]:
        """Pending-distill session_ids of one (org, user, meta agent) scope.

        Oldest first (by ``first_message_at``, then session_id). Idle sessions
        count as ended here even before they are materialized, so this is also a
        read-only answer (``/memory/auto_dream`` dry runs). ``for_update`` locks
        the returned session rows (Postgres) so a claim and a concurrent write
        to the same session serialize; with ``db`` the query joins the caller's
        transaction.
        """
        if db is None:
            async with self.session_maker() as own_db:
                return await self.list_pending_distill_sessions(
                    organization_id=organization_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    db=own_db,
                    now=now,
                    exclude_claimed=exclude_claimed,
                    for_update=False,
                )

        now = as_utc(now) if now is not None else self.now()
        stmt = (
            select(ConversationSessionModel.session_id)
            .where(
                *self._scope_filters(organization_id, user_id, agent_id),
                *self.pending_filters(now),
            )
            .order_by(
                ConversationSessionModel.first_message_at.asc(),
                ConversationSessionModel.session_id.asc(),
            )
        )
        if for_update:
            stmt = stmt.with_for_update(of=ConversationSessionModel)
        session_ids = [row[0] for row in (await db.execute(stmt)).all()]
        if not exclude_claimed or not session_ids:
            return session_ids
        claimed = await self.claimed_session_ids(
            db, organization_id=organization_id, user_id=user_id
        )
        return [sid for sid in session_ids if sid not in claimed]

    async def list_pending_scopes(
        self, *, now: Optional[dt.datetime] = None
    ) -> List[Tuple[str, str, str]]:
        """Every (organization_id, user_id, agent_id) with pending sessions.

        Claims are not subtracted here — the per-scope trigger check does that
        under its lock. Skipped, because nothing could run their learning (a
        run would only fail, on every sweep, forever):

        * sessions with no known meta agent (legacy rows the sweep could not
          attribute — see :meth:`adopt_unattributed_sessions`);
        * sessions whose meta agent is deleted (orphans the sweep could not
          re-attribute), or whose owning client is deleted or gone;
        * sessions of a soft-deleted user.
        """
        from mirix.orm.agent import Agent as AgentModel
        from mirix.orm.client import Client as ClientModel
        from mirix.orm.user import User as UserModel

        now = as_utc(now) if now is not None else self.now()
        live_agent = (
            select(AgentModel.id)
            .where(
                AgentModel.id == ConversationSessionModel.agent_id,
                AgentModel.is_deleted.is_(False),
                select(ClientModel.id)
                .where(
                    ClientModel.id == AgentModel._created_by_id,
                    ClientModel.is_deleted.is_(False),
                )
                .exists(),
            )
            .exists()
        )
        live_user = (
            select(UserModel.id)
            .where(
                UserModel.id == ConversationSessionModel.user_id,
                UserModel.is_deleted.is_(False),
            )
            .exists()
        )
        async with self.session_maker() as db:
            rows = (
                await db.execute(
                    select(
                        ConversationSessionModel.organization_id,
                        ConversationSessionModel.user_id,
                        ConversationSessionModel.agent_id,
                    )
                    .where(
                        ConversationSessionModel.agent_id.is_not(None),
                        live_agent,
                        live_user,
                        *self.pending_filters(now),
                    )
                    .distinct()
                )
            ).all()
            return sorted((org, user, agent) for org, user, agent in rows)

    # ------------------------------------------------------------------ #
    # Cross-run backoff after infrastructure failures                      #
    # ------------------------------------------------------------------ #

    async def defer_distill(
        self,
        *,
        organization_id: str,
        user_id: str,
        session_ids: List[str],
        now: Optional[dt.datetime] = None,
        db=None,
    ) -> dict:
        """Back sessions off after an infrastructure failure.

        Each session's consecutive-infrastructure-failure count goes up by one
        and ``distill_retry_after = now + 1 min * 2**(count - 1)`` (capped at
        1 h; ``MIRIX_DISTILL_RETRY_BACKOFF_MIN_SECONDS`` / ``_MAX_SECONDS``).
        Until then the session is neither counted by the trigger nor claimed.
        It stays pending: ``distill_failures`` is NOT touched — infrastructure
        errors never count toward the 3 failures. Returns ``{session_id:
        retry_after}``. With ``db`` the update joins the caller's transaction.
        """
        from mirix.services.retry_policy import next_distill_retry_after

        if not session_ids:
            return {}
        if db is None:
            async with self.session_maker() as own_db:
                deferred = await self.defer_distill(
                    organization_id=organization_id,
                    user_id=user_id,
                    session_ids=session_ids,
                    now=now,
                    db=own_db,
                )
                await own_db.commit()
                return deferred

        at = self.now() if now is None else as_utc(now) or self.now()
        rows = (
            (
                await db.execute(
                    select(ConversationSessionModel)
                    .where(
                        *self._scope_filters(organization_id, user_id, None),
                        ConversationSessionModel.session_id.in_(list(session_ids)),
                        ConversationSessionModel.is_deleted.is_(False),
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        deferred = {}
        for row in rows:
            row.distill_infra_failures = int(row.distill_infra_failures or 0) + 1
            row.distill_retry_after = next_distill_retry_after(
                at, row.distill_infra_failures
            )
            row.updated_at = at
            deferred[row.session_id] = row.distill_retry_after
        await db.flush()
        return deferred

    async def record_distill_failure(
        self,
        *,
        organization_id: str,
        user_id: str,
        session_ids: List[str],
        now: Optional[dt.datetime] = None,
        max_failures: Optional[int] = None,
        db=None,
    ) -> dict:
        """Count one distillation failure against each session.

        A counted failure is one the current run could not recover from: the
        answer stayed truncated after every chunk-budget halving, stayed
        unparseable after every repair prompt, timed out again after its
        retry, or another non-infrastructure error. ``distill_failures`` goes
        up by one; once it reaches ``max_failures`` (``DISTILL_MAX_FAILURES``,
        3) ``distill_failed_at`` is set and the session is never counted or
        claimed again (until a new write reopens it, which resets the count).
        The session's infrastructure-failure streak ends (its backoff is
        cleared): this attempt reached the model. Returns ``{session_id:
        (distill_failures, distill_failed_at)}``. With ``db`` the update joins
        the caller's transaction.
        """
        from mirix.constants import DISTILL_MAX_FAILURES

        if not session_ids:
            return {}
        if db is None:
            async with self.session_maker() as own_db:
                recorded = await self.record_distill_failure(
                    organization_id=organization_id,
                    user_id=user_id,
                    session_ids=session_ids,
                    now=now,
                    max_failures=max_failures,
                    db=own_db,
                )
                await own_db.commit()
                return recorded

        limit = max(1, int(DISTILL_MAX_FAILURES if max_failures is None else max_failures))
        at = self.now() if now is None else as_utc(now) or self.now()
        rows = (
            (
                await db.execute(
                    select(ConversationSessionModel)
                    .where(
                        *self._scope_filters(organization_id, user_id, None),
                        ConversationSessionModel.session_id.in_(list(session_ids)),
                        ConversationSessionModel.is_deleted.is_(False),
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        recorded = {}
        for row in rows:
            row.distill_failures = int(row.distill_failures or 0) + 1
            row.distill_infra_failures = 0
            row.distill_retry_after = None
            if row.distill_failures >= limit and row.distill_failed_at is None:
                row.distill_failed_at = at
            row.updated_at = at
            recorded[row.session_id] = (
                row.distill_failures,
                as_utc(row.distill_failed_at),
            )
        await db.flush()
        return recorded

    @staticmethod
    def clear_distill_backoff_stmt(
        *, organization_id: str, user_id: str, session_ids: List[str]
    ):
        """UPDATE ending the sessions' infrastructure-failure streak (a
        distill succeeded): counter 0, no ``distill_retry_after``."""
        return (
            update(ConversationSessionModel)
            .where(
                ConversationSessionModel.organization_id == organization_id,
                ConversationSessionModel.user_id == user_id,
                ConversationSessionModel.session_id.in_(list(session_ids)),
            )
            .values(distill_infra_failures=0, distill_retry_after=None)
            .execution_options(synchronize_session=False)
        )

    async def clear_distill_backoff(
        self, *, organization_id: str, user_id: str, session_ids: List[str]
    ) -> int:
        """:meth:`clear_distill_backoff_stmt` in its own transaction."""
        if not session_ids:
            return 0
        async with self.session_maker() as db:
            result = await db.execute(
                self.clear_distill_backoff_stmt(
                    organization_id=organization_id,
                    user_id=user_id,
                    session_ids=session_ids,
                )
            )
            await db.commit()
            return int(result.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Attribution of legacy sessions (no meta agent recorded)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _session_writers(
        db, *, organization_id: str, user_id: str, session_ids: List[str]
    ) -> dict:
        """session_id -> the set of ``_created_by_id`` (writing clients) of its
        live message rows (None = a row with no recorded writer)."""
        if not session_ids:
            return {}
        rows = (
            await db.execute(
                select(
                    ConversationMessageModel.session_id,
                    ConversationMessageModel._created_by_id,
                )
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id.in_(list(session_ids)),
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .distinct()
            )
        ).all()
        writers: dict = {}
        for sid, writer in rows:
            writers.setdefault(sid, set()).add(writer)
        return writers

    @staticmethod
    async def _agent_creators(db, agent_ids) -> dict:
        """agent_id -> the client that created it (``_created_by_id``), for
        the agent rows that still exist (a soft-deleted agent included)."""
        from mirix.orm.agent import Agent as AgentModel

        ids = sorted({a for a in agent_ids if a})
        if not ids:
            return {}
        return {
            agent_id: creator
            for agent_id, creator in (
                await db.execute(
                    select(AgentModel.id, AgentModel._created_by_id).where(
                        AgentModel.id.in_(ids)
                    )
                )
            ).all()
            if creator
        }

    async def orphan_owner(
        self, db, *, previous_agent_id: str, session_creator: Optional[str]
    ) -> Optional[str]:
        """The client an orphaned session belongs to when none of its rows
        records a writer: the deleted meta agent's creator (its row survives a
        soft delete), else the client whose write created the session row."""
        creators = await self._agent_creators(db, [previous_agent_id])
        return creators.get(previous_agent_id) or session_creator

    async def resolve_session_agents(
        self,
        db,
        *,
        organization_id: str,
        user_id: str,
        session_ids: List[str],
        previous_agents: Optional[dict] = None,
        session_creators: Optional[dict] = None,
    ) -> dict:
        """session_id -> the meta agent an ownerless session belongs to.

        The session's live rows name the client(s) that wrote them. With exactly
        one known writer, the session belongs to that client's only live meta
        agent. With no recorded writer at all, a LEGACY session (no agent,
        ``previous_agents[sid]`` None) goes to the organization's only live
        meta agent; an ORPHAN (``previous_agents[sid]`` names its deleted meta
        agent) only to its owning client's only live meta agent — see
        :meth:`orphan_owner`, with ``session_creators[sid]`` the session row's
        creator — never to another client's by the organization-wide fallback.
        Anything else (several writers, a client with zero or several meta
        agents, an orphan whose owner is unknown) is ambiguous and the session
        is left out.
        """
        from mirix.orm.agent import Agent as AgentModel

        previous_agents = previous_agents or {}
        session_creators = session_creators or {}
        writers = await self._session_writers(
            db,
            organization_id=organization_id,
            user_id=user_id,
            session_ids=session_ids,
        )
        orphan_owners: dict = {}
        orphan_agents = [
            previous_agents[sid]
            for sid in session_ids
            if previous_agents.get(sid) is not None
            and not {w for w in writers.get(sid, set()) if w}
        ]
        agent_creators = await self._agent_creators(db, orphan_agents)
        for sid in session_ids:
            previous = previous_agents.get(sid)
            if previous is not None:
                orphan_owners[sid] = agent_creators.get(
                    previous
                ) or session_creators.get(sid)
        owner_ids = {w for ws in writers.values() for w in ws if w} | {
            owner for owner in orphan_owners.values() if owner
        }
        agents_by_owner: dict = {}
        if owner_ids:
            for agent_id, owner in (
                await db.execute(
                    select(AgentModel.id, AgentModel._created_by_id).where(
                        AgentModel._created_by_id.in_(sorted(owner_ids)),
                        AgentModel.agent_type == "meta_memory_agent",
                        AgentModel.is_deleted.is_(False),
                    )
                )
            ).all():
                agents_by_owner.setdefault(owner, []).append(agent_id)
        org_agents: Optional[List[str]] = None
        resolved: dict = {}
        for sid in session_ids:
            known = {w for w in writers.get(sid, set()) if w}
            if len(known) == 1:
                candidates = agents_by_owner.get(next(iter(known)), [])
            elif not known and previous_agents.get(sid) is not None:
                # An orphan: only its owning client's meta agent.
                owner = orphan_owners.get(sid)
                candidates = agents_by_owner.get(owner, []) if owner else []
            elif not known and sid in writers:
                if org_agents is None:
                    org_agents = [
                        row[0]
                        for row in (
                            await db.execute(
                                select(AgentModel.id)
                                .where(
                                    AgentModel.organization_id == organization_id,
                                    AgentModel.agent_type == "meta_memory_agent",
                                    AgentModel.is_deleted.is_(False),
                                )
                                .limit(2)
                            )
                        ).all()
                    ]
                candidates = org_agents
            else:
                candidates = []
            if len(candidates) == 1:
                resolved[sid] = candidates[0]
        return resolved

    async def adopt_unattributed_sessions(
        self,
        *,
        db=None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Attribute sessions that have no live meta agent.

        Rows backfilled by ``scripts/migrate_add_conversation_session.sql``
        whose agent-loop history was already gone carry ``agent_id NULL``, and
        a session whose meta agent was deleted is orphaned; no scope could ever
        claim either. Each is attributed as described in
        :meth:`resolve_session_agents` (the writing client's only live meta
        agent — so a client that re-created its meta agent gets its sessions
        back; an orphan never goes to another client) with a conditional
        UPDATE on the ``agent_id`` it was read with; an adopted orphan's
        pending experiences move to the new meta agent in the same transaction
        (:meth:`move_pending_experiences`). ``agent_id`` restricts adoption to
        that meta agent (a claim adopts only into its own scope). Returns the
        number of sessions attributed. With ``db`` the updates join the
        caller's transaction; without it the call commits its own.
        """
        if db is None:
            async with self.session_maker() as own_db:
                count = await self.adopt_unattributed_sessions(
                    db=own_db,
                    organization_id=organization_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                )
                await own_db.commit()
                return count

        filters = [
            *self._scope_filters(organization_id, user_id, None, session_id),
            ownerless_clause(),
            ConversationSessionModel.is_deleted.is_(False),
        ]
        rows = (
            await db.execute(
                select(
                    ConversationSessionModel.id,
                    ConversationSessionModel.organization_id,
                    ConversationSessionModel.user_id,
                    ConversationSessionModel.session_id,
                    ConversationSessionModel.agent_id,
                    ConversationSessionModel._created_by_id,
                ).where(*filters)
            )
        ).all()
        by_owner: dict = {}
        for row_id, org, user, sid, previous, creator in rows:
            by_owner.setdefault((org, user), []).append(
                (row_id, sid, previous, creator)
            )
        adopted = 0
        for (org, user), items in by_owner.items():
            resolved = await self.resolve_session_agents(
                db,
                organization_id=org,
                user_id=user,
                session_ids=[sid for _, sid, _, _ in items],
                previous_agents={sid: previous for _, sid, previous, _ in items},
                session_creators={sid: creator for _, sid, _, creator in items},
            )
            for row_id, sid, previous, _creator in items:
                target = resolved.get(sid)
                if target is None or (agent_id is not None and target != agent_id):
                    continue
                result = await db.execute(
                    update(ConversationSessionModel)
                    .where(
                        ConversationSessionModel.id == row_id,
                        # Only if nobody attributed it meanwhile.
                        ConversationSessionModel.agent_id.is_not_distinct_from(
                            previous
                        ),
                    )
                    .values(agent_id=target)
                    .execution_options(synchronize_session=False)
                )
                if int(result.rowcount or 0) == 1:
                    adopted += 1
                    if previous is not None:
                        await self.move_pending_experiences(
                            db,
                            organization_id=org,
                            user_id=user,
                            session_id=sid,
                            from_agent_id=previous,
                            to_agent_id=target,
                        )
                    logger.info(
                        "Attributed %s session %s (user %s) to meta agent %s",
                        "legacy" if previous is None else f"orphaned ({previous})",
                        sid,
                        user,
                        target,
                    )
        return adopted

    @staticmethod
    async def move_pending_experiences(
        db,
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        from_agent_id: str,
        to_agent_id: str,
    ) -> int:
        """In ``db``'s transaction: re-attribute an adopted orphan's still
        PENDING experiences from its deleted meta agent to the adopting one,
        so its already-distilled learning reaches the new scope's curator.
        Consumed / closed-out experiences stay as they were. Returns how
        many moved."""
        from mirix.orm.skill_experience import SkillExperience as ExperienceModel

        result = await db.execute(
            update(ExperienceModel)
            .where(
                ExperienceModel.organization_id == organization_id,
                ExperienceModel.user_id == user_id,
                ExperienceModel.session_id == session_id,
                ExperienceModel.agent_id == from_agent_id,
                ExperienceModel.status == "pending",
                ExperienceModel.is_deleted.is_(False),
            )
            .values(agent_id=to_agent_id, updated_at=utc_now())
            .execution_options(synchronize_session=False)
        )
        moved = int(result.rowcount or 0)
        if moved:
            logger.info(
                "Moved %d pending experience(s) of session %s from meta agent %s "
                "to %s",
                moved,
                session_id,
                from_agent_id,
                to_agent_id,
            )
        return moved

    @staticmethod
    async def release_pending_learning(db, *, agent_id: str) -> int:
        """In ``db``'s transaction, before meta agent ``agent_id`` is HARD
        deleted (its experiences go with it, ``ON DELETE CASCADE``): mark the
        rows of every session with a still-pending experience of it
        undistilled, so the meta agent that adopts the orphaned session
        distills them again instead of their learning being lost.
        Returns the number of message rows released."""
        from mirix.orm.skill_experience import SkillExperience as ExperienceModel

        sessions = (
            await db.execute(
                select(
                    ExperienceModel.organization_id,
                    ExperienceModel.user_id,
                    ExperienceModel.session_id,
                )
                .where(
                    ExperienceModel.agent_id == agent_id,
                    ExperienceModel.status == "pending",
                    ExperienceModel.is_deleted.is_(False),
                )
                .distinct()
            )
        ).all()
        released = 0
        for organization_id, user_id, session_id in sessions:
            result = await db.execute(
                update(ConversationMessageModel)
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id == session_id,
                    ConversationMessageModel.distilled_at.is_not(None),
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .values(distilled_at=None)
                .execution_options(synchronize_session=False)
            )
            released += int(result.rowcount or 0)
        if released:
            logger.info(
                "Meta agent %s is deleted with pending experiences: %d message "
                "row(s) of %d session(s) will be distilled again by their next "
                "owner",
                agent_id,
                released,
                len(sessions),
            )
        return released

    # ------------------------------------------------------------------ #
    # Erasure                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def delete_for_user_stmt(user_id: str):
        return delete(ConversationSessionModel).where(
            ConversationSessionModel.user_id == user_id
        )
