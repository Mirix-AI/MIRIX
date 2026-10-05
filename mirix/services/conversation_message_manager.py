"""Manager for the Conversation Message Store.

Async-only, mirroring `SkillExperienceManager` / `MessageManager`: it grabs
sessions via `self.session_maker` (the server's `db_context`) and never calls
`asyncio.run()` (the server event loop is already running).

This store is the SINGLE source of truth for procedural-memory (skill)
distillation. It holds only external conversation turns that arrived with a
`session_id`, with their REAL `user`/`assistant`/`tool` roles preserved. The
methods below are the stable contract the ingestion seam, the trigger, the
learning run, and the data-lifecycle paths depend on:

  - record_turns             -- append a batch of turns to a session and, in the
                                same transaction, upsert/lock/reopen its
                                `conversation_session` row
  - count_distinct_sessions  -- how many distinct sessions exist (optionally only
                                those with not-yet-distilled rows)
  - session_watermarks       -- per-session newest `created_at` (claim time)
  - list_turns_for_session   -- one session's turns, ascending; optionally only
                                the not-yet-distilled rows up to a watermark
  - mark_sessions_distilled  -- stamp `distilled_at` on rows up to each watermark
  - prune_distilled_turns    -- retention: drop old distilled turns and the
                                images only they referenced
  - delete_by_user_id        -- erasure: purge one user's turns, sessions, runs,
                                and the images the turns referenced
  - delete_by_client_id      -- erasure: purge one client's turns, sessions,
                                runs, and the images the turns referenced

A turn is text (`content`) or, when it carries images, ordered `content_parts`
(text + `image_ref` parts) with `content` holding `[image: <file_id>]`
placeholders. `record_turns` applies the storage cap: an
over-long message keeps its head and tail with an explicit truncation
marker, records `original_text_length`, and the session's
`truncated_message_count` / `truncated_chars` count what was cut.

Distillation state is tracked PER ROW (`distilled_at`), not per session: a
session that is reopened after it was learned from keeps its old rows marked and
gets new, unmarked rows. A learning run reads and marks only the rows
at or before the watermark it recorded when it claimed the session, so
rows written after the claim are never marked by that run. Which sessions a
run may claim is decided by `ConversationSessionManager`.

Because this table stores VERBATIM user conversation content, it must be
covered by the same erasure guarantees as the memory tables: the user/client
"delete memories" purge paths call the two delete_* methods, and the
distillation flow prunes distilled turns past the retention window.
Everything is scoped per `(user, organization)` so one user's sessions never
count toward or leak into another's learning.
"""

from __future__ import annotations

import datetime as dt
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import delete, func, select, update

from mirix.log import get_logger
from mirix.orm.conversation_message import (
    ConversationMessage as ConversationMessageModel,
)
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.conversation_message import (
    CONTENT_PART_TEXT,
    ConversationMessage as PydanticConversationMessage,
    ConversationMessageCreate,
    content_parts_have_images,
    render_content_parts_text,
)
from mirix.services.conversation_content import (
    cap_text_parts,
    effective_char_cap,
    nul_replacement_warning,
    replace_nul,
)
from mirix.services.conversation_session_manager import (
    Clock,
    ConversationSessionManager,
    as_utc,
    utc_now,
)
from mirix.services.image_store import (
    file_ids_referenced_by_agent_messages,
    release_image_files,
    unlink_image_paths,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)


def owner_org(actor: PydanticClient) -> str:
    """Resolve the organization the Conversation Message Store is scoped under.

    Ingestion records turns under ``client.organization_id or DEFAULT_ORG_ID``
    (the store's organization_id column is NOT NULL). Every read/mark/prune of
    the store MUST mirror the same fallback, otherwise a NULL-org client would
    write under DEFAULT_ORG_ID but read under ``None`` and its sessions would
    never distill. Single definition here — the distiller and the auto-dream
    manager import it rather than re-deriving the fallback.
    """
    from mirix.constants import DEFAULT_ORG_ID

    return actor.organization_id or DEFAULT_ORG_ID


def parse_watermark(value: Any) -> dt.datetime:
    """A watermark from a learning run (ISO string or datetime) as aware UTC."""
    if isinstance(value, dt.datetime):
        return as_utc(value)
    if isinstance(value, str):
        return as_utc(dt.datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ValueError(f"not a watermark: {value!r}")


#: The warning of a write that gave an agent context but stored no turn: the
#: context is stored with a session's turns, so it was not stored.
AGENT_CONTEXT_NOT_STORED_WARNING = (
    "agent_context was not stored: this write stored no turn of the session "
    "(an agent context is stored with the session's turns); send it again with "
    "the session's messages"
)


class RecordedTurns(list):
    """The persisted turns of one `record_turns` call (a plain list of
    `ConversationMessage`), plus what the write did to the session row:

    * ``session``       -- the `ConversationSession` after the write (or None);
    * ``created``       -- the write created the session;
    * ``reopened``      -- the write reopened an ended session;
    * ``effective_tag`` -- the session's tag, fixed by its first write;
    * ``warnings``      -- e.g. a session_tag mismatch, for the caller's
                           response.
    """

    def __init__(self, turns=(), *, session=None, state=None):
        super().__init__(turns)
        self.session = session
        self.created = bool(getattr(state, "created", False))
        self.reopened = bool(getattr(state, "reopened", False))
        self.effective_tag = getattr(state, "effective_tag", None)
        self.warnings = list(getattr(state, "warnings", []) or [])


class ConversationMessageManager:
    """Persist and query external conversation turns for skill distillation."""

    def __init__(self, *, session_maker=None, clock: Optional[Clock] = None):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        self.clock: Clock = clock or utc_now

    def _session_manager(self) -> ConversationSessionManager:
        return ConversationSessionManager(
            session_maker=self.session_maker, clock=self.clock
        )

    @enforce_types
    async def record_turns(
        self,
        *,
        session_id: str,
        user_id: str,
        organization_id: str,
        turns: List[dict],
        actor: PydanticClient,
        session_tag: Optional[str] = None,
        agent_id: Optional[str] = None,
        max_text_chars: Optional[int] = None,
        agent_context: Optional[str] = None,
    ) -> List[PydanticConversationMessage]:
        """Append `turns` to `session_id`, preserving their given order.

        `turns` is a list of `{"role": "user"|"assistant"|"tool", "content": str}`
        or, for a turn with images, `{"role", "content_parts": [...]}` (text
        parts and `image_ref` parts whose images are already stored); its
        `content` is rendered from the parts with `[image: <file_id>]`
        placeholders. `content_parts` without an image collapse to plain
        `content` (the column stays NULL for text-only turns).

        Storage cap: `max_text_chars` is the writing meta agent's cap
        (`floor(context_window * 0.8) * 4`); `MIRIX_CONVERSATION_MESSAGE_MAX_CHARS`
        lowers it further (None and unset = no cap). A turn whose text exceeds
        the cap is truncated head + tail from its longest text part on, with an
        explicit `…[truncated N chars]…` marker; its `original_text_length` is
        recorded and the session's `truncated_message_count` / `truncated_chars`
        grow in the same transaction.

        NUL characters: Postgres cannot store U+0000, so every one in a turn's
        text (its `content` and its text parts) is stored as U+FFFD, on every
        backend; the returned `warnings` say how many. Without this
        a tool result carrying a NUL would fail the write on every retry.

        Each turn is validated through `ConversationMessageCreate` first so an
        invalid `role` (or malformed `content_parts` / session_id) is rejected
        up front — before any row or session state is written.

        One transaction: lock the `conversation_session` row
        (`SELECT ... FOR UPDATE`, creating it on the first write), reopen it if
        it had ended, insert the turns, advance `last_message_at` /
        `message_count`. `session_tag` is the tag the caller asked for (None =
        not stated); the session's first write fixes it, and the stored tag is
        what the turns are written with. `agent_id` is the meta memory
        agent that owns the session (the learning scope).

        created_at ordering is preserved EXPLICITLY: a single batch shares one
        transaction, and the server-side `func.now()` default would collapse all
        rows in that transaction to one timestamp, destroying intra-batch order.
        We instead stamp a strictly increasing `created_at` per turn — anchored
        after the session's current newest row, read while the session row is
        locked — so a later call's turns always sort after an earlier call's and
        after any learning-run watermark taken before this write.

        `agent_context` (optional): how the agent's prompt was built,
        as the client describes it (its prompt template / setup). It is not a
        turn: it is stored on the session row (`agent_context`) in the same
        transaction — the latest non-blank one wins, NUL characters as
        U+FFFD, cut head + tail to the same storage cap as a turn (with the
        same marker; a warning says so) — and the distiller shows it beside
        the transcript. None or blank leaves the stored one as it is. A call
        with no turns stores nothing — its agent context neither — and says
        so in a warning.

        Returns a `RecordedTurns` (a list of the persisted rows in insertion
        order that also carries the session outcome and warnings).
        """
        if not turns:
            recorded = RecordedTurns([])
            if isinstance(agent_context, str) and agent_context.strip():
                recorded.warnings.append(AGENT_CONTEXT_NOT_STORED_WARNING)
            return recorded

        cap = effective_char_cap(max_text_chars)
        context_text: Optional[str] = None
        context_nul = context_cut = 0
        if isinstance(agent_context, str) and agent_context.strip():
            context_text, context_nul = replace_nul(agent_context)
            [context_text], context_cut = cap_text_parts([context_text], cap)
        validated = []
        truncated_messages = 0
        truncated_chars = 0
        nul_messages = 0
        nul_chars = 0
        for turn in turns:
            turn, replaced = _turn_without_nul(turn)
            if replaced:
                nul_messages += 1
                nul_chars += replaced
            content, parts, original_length, removed = _capped_turn_content(turn, cap)
            if removed:
                truncated_messages += 1
                truncated_chars += removed
            validated.append(
                ConversationMessageCreate(
                    session_id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    role=turn.get("role"),
                    content=content,
                    content_parts=parts,
                    original_text_length=original_length,
                )
            )

        session_mgr = self._session_manager()
        actor_id = getattr(actor, "id", None) if actor is not None else None
        async with self.session_maker() as session:
            now = session_mgr.now()
            session_row, state = await session_mgr.lock_for_write(
                session,
                organization_id=organization_id,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                requested_tag=session_tag,
                actor_id=actor_id,
                now=now,
            )

            # Anchor after the latest existing turn for this session (read under
            # the session-row lock) so turns from successive calls accumulate in
            # order. Scoped by (org, user, session) like everything else here.
            bounds = await session.execute(
                select(
                    func.min(ConversationMessageModel.created_at),
                    func.max(ConversationMessageModel.created_at),
                    func.count(ConversationMessageModel.id),
                ).where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id == session_id,
                    ConversationMessageModel.is_deleted.is_(False),
                )
            )
            first_existing, last_existing, existing_count = bounds.one()
            first_existing = as_utc(first_existing)
            last_ts = as_utc(last_existing)
            if not state.created:
                stored_last = as_utc(session_row.last_message_at)
                if stored_last is not None and (last_ts is None or stored_last > last_ts):
                    last_ts = stored_last

            base = now
            if last_ts is not None and last_ts >= base:
                base = last_ts + timedelta(microseconds=1)

            rows = []
            for offset, payload in enumerate(validated):
                row = ConversationMessageModel(
                    session_id=payload.session_id,
                    user_id=payload.user_id,
                    organization_id=payload.organization_id,
                    role=payload.role,
                    content=payload.content,
                    content_parts=payload.content_parts,
                    original_text_length=payload.original_text_length,
                    session_tag=state.effective_tag,
                    # +offset microseconds keeps a strict, stable order within
                    # the batch even though they share one transaction.
                    created_at=base + timedelta(microseconds=offset),
                )
                if actor_id:
                    row._set_created_and_updated_by_fields(actor_id)
                session.add(row)
                rows.append(row)

            first_ts = rows[0].created_at
            if state.created:
                # A session row created for turns that predate it (written before
                # the conversation_session migration ran) still spans them.
                if first_existing is not None and first_existing < first_ts:
                    first_ts = first_existing
                session_row.message_count = int(existing_count or 0)
            session_mgr.apply_write(
                session_row,
                first_message_at=first_ts,
                last_message_at=rows[-1].created_at,
                message_count=len(rows),
                created=state.created,
                actor_id=actor_id,
                now=now,
                truncated_messages=truncated_messages,
                truncated_chars=truncated_chars,
            )
            if context_text is not None:
                session_row.agent_context = context_text

            await session.commit()
            for row in rows:
                await session.refresh(row)
            await session.refresh(session_row)
            if state.reopened:
                logger.info(
                    "Session %s (user %s) reopened by a new write (reopen #%d)",
                    session_id,
                    user_id,
                    session_row.reopen_count,
                )
            if truncated_messages:
                logger.warning(
                    "Session %s (user %s): %d message(s) over the %d-char storage "
                    "cap were truncated (%d chars removed)",
                    session_id,
                    user_id,
                    truncated_messages,
                    cap,
                    truncated_chars,
                )
            recorded = RecordedTurns(
                [row.to_pydantic() for row in rows],
                session=session_row.to_pydantic(),
                state=state,
            )
            if nul_chars:
                recorded.warnings.append(
                    nul_replacement_warning(nul_chars, nul_messages)
                )
            if context_nul:
                recorded.warnings.append(
                    f"agent_context: {context_nul} NUL character(s) were stored as "
                    "U+FFFD"
                )
            if context_cut:
                recorded.warnings.append(
                    f"agent_context is over the {cap}-character storage cap: "
                    f"{context_cut} characters were cut from its middle"
                )
            return recorded

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

        With `only_undistilled=True`, counts only sessions that still have at
        least one row with `distilled_at IS NULL`. Distillation state is per row:
        a session that was learned from and then reopened counts again
        once it has new, undistilled rows. Soft-deleted rows are excluded.
        """
        async with self.session_maker() as session:
            preds = [
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.is_deleted.is_(False),
            ]
            if only_undistilled:
                preds.append(ConversationMessageModel.distilled_at.is_(None))
            stmt = select(
                func.count(func.distinct(ConversationMessageModel.session_id))
            ).where(*preds)
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

    async def session_watermarks(
        self,
        db,
        *,
        user_id: str,
        organization_id: str,
        session_ids: List[str],
    ) -> Dict[str, dt.datetime]:
        """Newest live `created_at` per session, in the caller's transaction.

        A learning run records these when it claims its sessions and then
        only reads and marks rows at or before them. Sessions with no live rows
        are absent from the result.
        """
        if not session_ids:
            return {}
        rows = (
            await db.execute(
                select(
                    ConversationMessageModel.session_id,
                    func.max(ConversationMessageModel.created_at),
                )
                .where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.session_id.in_(session_ids),
                    ConversationMessageModel.is_deleted.is_(False),
                )
                .group_by(ConversationMessageModel.session_id)
            )
        ).all()
        return {sid: as_utc(max_ts) for sid, max_ts in rows if max_ts is not None}

    @enforce_types
    async def list_turns_for_session(
        self,
        *,
        session_id: str,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
        watermark: Optional[Union[dt.datetime, str]] = None,
        undistilled_only: bool = False,
    ) -> List[PydanticConversationMessage]:
        """Return one session's turns for this `(user, org)`, ascending by created_at.

        `watermark` (datetime or ISO string) keeps only rows with
        `created_at <= watermark`; `undistilled_only` keeps only rows with
        `distilled_at IS NULL`. A learning run passes both.
        """
        async with self.session_maker() as session:
            preds = [
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.session_id == session_id,
                ConversationMessageModel.is_deleted.is_(False),
            ]
            if watermark is not None:
                preds.append(
                    ConversationMessageModel.created_at <= parse_watermark(watermark)
                )
            if undistilled_only:
                preds.append(ConversationMessageModel.distilled_at.is_(None))
            stmt = (
                select(ConversationMessageModel)
                .where(*preds)
                .order_by(ConversationMessageModel.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_pydantic() for row in rows]

    @enforce_types
    async def mark_sessions_distilled(
        self,
        *,
        session_watermarks: Dict[str, Any],
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
    ) -> int:
        """Stamp `distilled_at` on each session's rows up to its watermark.

        `session_watermarks` maps session_id -> watermark (datetime or ISO
        string), normally the claim-time watermarks of a learning run. Only rows
        with `created_at <= watermark` and `distilled_at IS NULL` are stamped,
        so rows written after the claim (a reopened session) stay pending and a
        re-run is a no-op. Scoped to this `(user, org)` so a caller can never
        mark another owner's session. Returns the number of rows updated.

        A bulk UPDATE (not load-modify-save) is correct here: `distilled_at` is a
        write-only marker with no cache, and loading potentially large
        transcripts just to stamp them would be wasteful.
        """
        if not session_watermarks:
            return 0

        now = as_utc(self.clock())
        total = 0
        async with self.session_maker() as session:
            for session_id, watermark in session_watermarks.items():
                result = await session.execute(
                    self.mark_distilled_stmt(
                        organization_id=organization_id,
                        user_id=user_id,
                        session_id=session_id,
                        watermark=watermark,
                        now=now,
                    )
                )
                total += int(result.rowcount or 0)
            await session.commit()
        return total

    @staticmethod
    def mark_distilled_stmt(
        *,
        organization_id: str,
        user_id: str,
        session_id: str,
        watermark: Any,
        now: dt.datetime,
    ):
        """UPDATE stamping ``distilled_at`` on a session's live, undistilled rows
        up to ``watermark`` (for callers that mark inside their own transaction)."""
        return (
            update(ConversationMessageModel)
            .where(
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.session_id == session_id,
                ConversationMessageModel.created_at <= parse_watermark(watermark),
                ConversationMessageModel.distilled_at.is_(None),
                ConversationMessageModel.is_deleted.is_(False),
            )
            .values(distilled_at=now)
            .execution_options(synchronize_session=False)
        )

    @enforce_types
    async def prune_distilled_turns(
        self,
        *,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
        retention_days: int,
    ) -> int:
        """Hard-delete turns distilled more than `retention_days` ago.

        Retention policy for the verbatim conversation store: once a turn has
        been distilled its learning value has been extracted into
        skill_experience rows, so keeping the raw transcript indefinitely is a
        pure PII liability. Row-level `distilled_at <= cutoff` is deliberately
        conservative: undistilled turns are NEVER pruned (their learning hasn't
        been extracted yet), even when they share a session with pruned rows.

        The images the pruned rows referenced (`content_parts` image refs) that
        no remaining turn of this user references are deleted with them:
        their `files` rows in the same transaction, their disk files after the
        commit.

        `retention_days <= 0` disables pruning (keep forever). Returns the
        number of rows deleted.
        """
        if retention_days <= 0:
            return 0

        cutoff = as_utc(self.clock()) - timedelta(days=retention_days)
        async with self.session_maker() as session:
            where = [
                ConversationMessageModel.organization_id == organization_id,
                ConversationMessageModel.user_id == user_id,
                ConversationMessageModel.distilled_at.is_not(None),
                ConversationMessageModel.distilled_at <= cutoff,
            ]
            parts_rows = await session.execute(
                select(ConversationMessageModel.content_parts).where(
                    *where, ConversationMessageModel.content_parts.is_not(None)
                )
            )
            file_ids = _image_file_ids(parts for (parts,) in parts_rows.all())
            result = await session.execute(delete(ConversationMessageModel).where(*where))
            deleted = int(result.rowcount or 0)
            released_paths: List[str] = []
            if file_ids:
                released_paths = await self._release_image_refs(
                    session,
                    organization_id=organization_id,
                    user_id=user_id,
                    file_ids=file_ids,
                )
            await session.commit()
        await unlink_image_paths(released_paths, session_maker=self.session_maker)
        if deleted:
            logger.info(
                "Pruned %d distilled conversation turn(s) older than %d day(s) for "
                "user %s (%d image reference(s) released)",
                deleted,
                retention_days,
                user_id,
                len(file_ids),
            )
        return deleted

    async def _release_image_refs(
        self,
        db,
        *,
        organization_id: str,
        user_id: Optional[str],
        file_ids: List[str],
    ) -> List[str]:
        """Release the images of conversation turns just deleted in ``db``.

        Runs inside the deleting transaction. An image still referenced by a
        remaining turn of the same (organization, user) is kept, and so is one
        an agent message of that user still references (a caller may pass a
        stored image to the meta agent by id); the others lose their `files`
        rows here. Returns the disk paths that no `files`
        row references any more — the caller unlinks them after it commits.
        ``user_id=None`` skips the remaining-reference checks (the user's whole
        data is being erased).
        """
        if not file_ids:
            return []
        if user_id is not None:
            remaining = await db.execute(
                select(ConversationMessageModel.content_parts).where(
                    ConversationMessageModel.organization_id == organization_id,
                    ConversationMessageModel.user_id == user_id,
                    ConversationMessageModel.content_parts.is_not(None),
                )
            )
            still_referenced = set(
                _image_file_ids(parts for (parts,) in remaining.all())
            )
            file_ids = [fid for fid in file_ids if fid not in still_referenced]
            if file_ids:
                in_messages = await file_ids_referenced_by_agent_messages(
                    db,
                    organization_id=organization_id,
                    user_id=user_id,
                    file_ids=file_ids,
                )
                file_ids = [fid for fid in file_ids if fid not in in_messages]
        return await release_image_files(
            db, organization_id=organization_id, file_ids=file_ids
        )

    @enforce_types
    async def delete_by_user_id(self, user_id: str) -> int:
        """Hard delete ALL conversation data for a user (erasure path).

        Called from `UserManager.delete_memories_by_user_id`, which backs the
        irreversible DELETE /users/{user_id}/memories purge. Deletes the user's
        turns, the images they referenced (`files` rows and disk files),
        their `conversation_session` rows and their skill learning runs, across
        all organizations and regardless of distillation state — erasure trumps
        the learning window. Returns the number of turns deleted.
        """
        from mirix.orm.skill_learning_run import SkillLearningRun as RunModel

        released_paths: List[str] = []
        async with self.session_maker() as session:
            refs = await self._image_refs_by_org(
                session, ConversationMessageModel.user_id == user_id
            )
            stmt = delete(ConversationMessageModel).where(
                ConversationMessageModel.user_id == user_id
            )
            result = await session.execute(stmt)
            for org_id, file_ids in refs.items():
                released_paths += await self._release_image_refs(
                    session, organization_id=org_id, user_id=None, file_ids=file_ids
                )
            await session.execute(
                ConversationSessionManager.delete_for_user_stmt(user_id)
            )
            await session.execute(delete(RunModel).where(RunModel.user_id == user_id))
            await session.commit()
        await unlink_image_paths(released_paths, session_maker=self.session_maker)
        return int(result.rowcount or 0)

    @enforce_types
    async def delete_by_client_id(self, actor: PydanticClient) -> int:
        """Hard delete all conversation data recorded by a client (erasure path).

        Attribution is via the `_created_by_id` audit column, which
        `record_turns` stamps with the ingesting client's id — the tables have
        no dedicated client_id column. Also deletes the images those turns
        referenced (unless a remaining turn of the same user still does) and
        the learning runs of the client's meta agents.

        Sessions: every session that held one of the client's turns, or whose
        row the client created, is re-derived from the turns that REMAIN:
        a session with no live turn left is deleted; one that still has another
        client's turns keeps its row — its tag and owner stay as the first
        write fixed them — with ``message_count`` / first / last message
        time recomputed. The session rows are locked first (``FOR UPDATE``) so
        a concurrent write to one of them waits for the erasure. Called from
        `ClientManager.delete_memories_by_client_id`. Returns the number of
        turns deleted.
        """
        from mirix.orm.agent import Agent as AgentModel
        from mirix.orm.conversation_session import (
            ConversationSession as SessionModel,
        )
        from mirix.orm.skill_learning_run import SkillLearningRun as RunModel

        released_paths: List[str] = []
        async with self.session_maker() as session:
            touched = {
                tuple(row)
                for row in (
                    await session.execute(
                        select(
                            ConversationMessageModel.organization_id,
                            ConversationMessageModel.user_id,
                            ConversationMessageModel.session_id,
                        )
                        .where(ConversationMessageModel._created_by_id == actor.id)
                        .distinct()
                    )
                ).all()
            }
            touched.update(
                tuple(row)
                for row in (
                    await session.execute(
                        select(
                            SessionModel.organization_id,
                            SessionModel.user_id,
                            SessionModel.session_id,
                        ).where(SessionModel._created_by_id == actor.id)
                    )
                ).all()
            )
            session_rows = await self._lock_sessions(session, touched)

            by_owner = await self._image_refs_by_owner(
                session, ConversationMessageModel._created_by_id == actor.id
            )
            stmt = delete(ConversationMessageModel).where(
                ConversationMessageModel._created_by_id == actor.id
            )
            result = await session.execute(stmt)
            for (org_id, owner_user_id), file_ids in by_owner.items():
                released_paths += await self._release_image_refs(
                    session,
                    organization_id=org_id,
                    user_id=owner_user_id,
                    file_ids=file_ids,
                )
            await self._rederive_sessions(session, session_rows)
            await session.execute(
                delete(RunModel).where(
                    RunModel.agent_id.in_(
                        select(AgentModel.id).where(AgentModel._created_by_id == actor.id)
                    )
                )
            )
            await session.commit()
        await unlink_image_paths(released_paths, session_maker=self.session_maker)
        return int(result.rowcount or 0)

    @staticmethod
    async def _lock_sessions(db, keys) -> List[Any]:
        """Lock the ``conversation_session`` rows of (org, user, session_id) keys.

        Postgres: ``SELECT ... FOR UPDATE`` (a write to one of them — which
        locks its row first — waits). SQLite ignores FOR UPDATE; the caller's
        deletes that follow take the database write lock.
        """
        from mirix.orm.conversation_session import (
            ConversationSession as SessionModel,
        )
        from sqlalchemy import tuple_

        keys = sorted(keys, key=lambda k: tuple(str(v) for v in k))
        rows: List[Any] = []
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            rows.extend(
                (
                    await db.execute(
                        select(SessionModel)
                        .where(
                            tuple_(
                                SessionModel.organization_id,
                                SessionModel.user_id,
                                SessionModel.session_id,
                            ).in_(chunk)
                        )
                        .order_by(SessionModel.id)
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
        return rows

    @staticmethod
    async def _rederive_sessions(db, session_rows) -> None:
        """Delete sessions with no live turn left; recount the others."""
        for row in session_rows:
            count, first_at, last_at = (
                await db.execute(
                    select(
                        func.count(ConversationMessageModel.id),
                        func.min(ConversationMessageModel.created_at),
                        func.max(ConversationMessageModel.created_at),
                    ).where(
                        ConversationMessageModel.organization_id == row.organization_id,
                        ConversationMessageModel.user_id == row.user_id,
                        ConversationMessageModel.session_id == row.session_id,
                        ConversationMessageModel.is_deleted.is_(False),
                    )
                )
            ).one()
            if not count:
                await db.delete(row)
                continue
            row.message_count = int(count)
            row.first_message_at = as_utc(first_at)
            row.last_message_at = as_utc(last_at)
        await db.flush()

    @staticmethod
    async def _image_refs_by_owner(db, *predicates) -> Dict[tuple, List[str]]:
        """Image file ids of the matching turns, grouped by (organization, user)."""
        rows = await db.execute(
            select(
                ConversationMessageModel.organization_id,
                ConversationMessageModel.user_id,
                ConversationMessageModel.content_parts,
            ).where(*predicates, ConversationMessageModel.content_parts.is_not(None))
        )
        grouped: Dict[tuple, List[list]] = {}
        for org_id, owner_user_id, parts in rows.all():
            grouped.setdefault((org_id, owner_user_id), []).append(parts)
        return {
            key: ids
            for key, lists in grouped.items()
            if (ids := _image_file_ids(lists))
        }

    @classmethod
    async def _image_refs_by_org(cls, db, *predicates) -> Dict[str, List[str]]:
        """Image file ids of the matching turns, grouped by organization."""
        by_org: Dict[str, Dict[str, None]] = {}
        for (org_id, _), ids in (await cls._image_refs_by_owner(db, *predicates)).items():
            by_org.setdefault(org_id, {}).update(dict.fromkeys(ids))
        return {org_id: list(ids) for org_id, ids in by_org.items()}


def _turn_without_nul(turn: dict):
    """(turn, replaced): ``turn`` with every U+0000 in its text stored as U+FFFD.

    Covers the turn's ``content`` and the ``text`` of its ``content_parts``;
    image refs are MIRIX's own values. The input turn is not modified.
    """
    replaced = 0
    out = dict(turn)
    content = turn.get("content")
    if isinstance(content, str):
        out["content"], replaced = replace_nul(content)
    raw_parts = turn.get("content_parts")
    if raw_parts:
        parts = []
        for part in raw_parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text, count = replace_nul(part["text"])
                if count:
                    part = {**part, "text": text}
                    replaced += count
            parts.append(part)
        out["content_parts"] = parts
    return out, replaced


def _capped_turn_content(turn: dict, cap: Optional[int]):
    """(content, content_parts, original_text_length, removed_chars) of a turn.

    Applies the storage cap to the turn's text parts (a text-only turn has exactly
    one: its `content`). `content_parts` is kept only if the turn has an image;
    its `content` is then the placeholder rendering of the (capped) parts.
    """
    raw_parts = turn.get("content_parts")
    parts = [dict(p) for p in raw_parts] if raw_parts else None
    if parts is not None and not content_parts_have_images(parts):
        # content_parts is written ONLY for messages that carry an image.
        turn = {"content": render_content_parts_text(parts)}
        parts = None

    if parts is None:
        texts = [turn.get("content") or ""]
    else:
        text_indexes = [i for i, p in enumerate(parts) if p.get("type") == CONTENT_PART_TEXT]
        texts = [str(parts[i].get("text", "")) for i in text_indexes]

    capped, removed = cap_text_parts(texts, cap)
    original_length = sum(len(t) for t in texts) if removed else None

    if parts is None:
        return capped[0], None, original_length, removed
    for i, text in zip(text_indexes, capped):
        parts[i]["text"] = text
    return render_content_parts_text(parts), parts, original_length, removed


def _image_file_ids(parts_lists) -> List[str]:
    """File ids of the `image_ref` parts in a sequence of `content_parts` lists."""
    seen: Dict[str, None] = {}
    for parts in parts_lists:
        for part in parts or []:
            if isinstance(part, dict) and part.get("type") == "image_ref":
                file_id = part.get("file_id")
                if isinstance(file_id, str) and file_id:
                    seen.setdefault(file_id, None)
    return list(seen)
