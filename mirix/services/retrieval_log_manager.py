"""The retrieval log.

One ``retrieval_log`` row per memory retrieval (``/memory/retrieve/conversation``
and ``/memory/retrieve/topic``); its id is returned to the caller as
``retrieval_id``. A row holds the raw query, the extracted topics and how
extraction went, the query-embedding status, a per-memory-type trace (for
procedural memory every candidate's bm25 / cosine / rrf scores and ranks, the
gate parameters, the returned ids with the version each was served at
(``returned_versions``) and the skills missing a vector), the
extraction's token usage, the latency, every retry and the chunked
embedding calls.

* ``MIRIX_RETRIEVAL_LOG_ENABLED=false`` writes no rows (``retrieval_id`` is
  then null).
* Rows are kept ``MIRIX_RETRIEVAL_LOG_RETENTION_DAYS`` (default 30; <= 0 keeps
  them): :class:`RetrievalLogPurger` deletes older ones at startup and every
  hour.
* Erasing a user's memories (``DELETE /users/{id}/memories``) or a client's
  (``DELETE /clients/{id}/memories``), and deleting a user
  (``DELETE /users/{id}``), deletes their rows at once. A retrieval still in
  flight while its user (or its client) is erased or deleted writes no row
  afterwards: :meth:`RetrievalLogManager.create_log` skips a user that is
  deleted, or whose (or whose client's) erasure in this process ran after
  the retrieval started (:func:`erasing_retrieval_logs`).
* A row stays small whatever the user holds: the raw query keeps its head
  and tail up to ``RETRIEVAL_LOG_QUERY_MAX_CHARS`` (an explicit marker says
  how much was cut) and the procedural trace is bounded by the retrieval
  (``SkillRetrievalDiagnostics.to_dict(max_candidates=...)``).
* ``GET /memory/retrieval_logs`` lists a client's own rows, newest first
  (``include_trace=false``: without ``per_type``).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import time
import uuid
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import defer

from mirix import constants
from mirix.log import get_logger
from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogModel
from mirix.schemas.retrieval_log import RetrievalLog as PydanticRetrievalLog
from mirix.services.conversation_session_manager import Clock, as_utc, utc_now
from mirix.services.retry_policy import retry_call

logger = get_logger(__name__)

#: GET /memory/retrieval_logs returns at most this many rows.
MAX_LIST_LIMIT = 500


def new_retrieval_log_id() -> str:
    """A fresh ``retrieval_log`` id (the ORM's own format)."""
    return f"rlog-{uuid.uuid4()}"


def retrieval_log_enabled() -> bool:
    """``MIRIX_RETRIEVAL_LOG_ENABLED`` (read at call time)."""
    return bool(constants.RETRIEVAL_LOG_ENABLED)


def cut_query_text(
    text: Optional[str], max_chars: Optional[int] = None
) -> Tuple[Optional[str], int]:
    """``text`` for the log's ``query_text``: its head and tail up to
    ``max_chars`` (``RETRIEVAL_LOG_QUERY_MAX_CHARS``), joined by an explicit
    marker naming how many characters were cut. Returns the text and the
    number of characters cut."""
    limit = constants.RETRIEVAL_LOG_QUERY_MAX_CHARS if max_chars is None else max_chars
    if text is None or limit <= 0 or len(text) <= limit:
        return text, 0
    cut = len(text) - limit
    head = limit // 2 + limit % 2
    tail = limit // 2
    marker = f"\n[... {cut} characters cut ...]\n"
    return text[:head] + marker + (text[-tail:] if tail else ""), cut


def _without_nul(text: Optional[str]) -> Tuple[Optional[str], int]:
    """``text`` with every U+0000 stored as U+FFFD (Postgres cannot hold it)."""
    if text is None:
        return None, 0
    from mirix.services.conversation_content import replace_nul

    return replace_nul(text)


# --------------------------------------------------------------------------- #
# Erasures in progress (this process)                                          #
# --------------------------------------------------------------------------- #

#: ("user" | "client", id) -> erasures of it running now in this process.
_ERASURES_RUNNING: Dict[Tuple[str, str], int] = {}
#: ("user" | "client", id) -> time.monotonic() when its last erasure ended.
_ERASURE_ENDED_AT: Dict[Tuple[str, str], float] = {}


def _erasure_keys(
    user_id: Optional[str], client_id: Optional[str]
) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    if user_id:
        keys.append(("user", user_id))
    if client_id:
        keys.append(("client", client_id))
    return keys


@asynccontextmanager
async def erasing_retrieval_logs(
    user_id: Optional[str] = None, *, client_id: Optional[str] = None
) -> AsyncIterator[None]:
    """Mark a user's data (``user_id``: its erasure, or the user's deletion)
    or a client's (``client_id``: its erasure) as being erased for the
    duration of the block: a retrieval of that user / by that client that
    started before the block ended writes no log row afterwards
    (:meth:`RetrievalLogManager.create_log`), so the erasure's deletion of the
    rows cannot be undone by a retrieval still in flight. In-process only:
    another worker process relies on the user check and on the next erasure /
    the retention purge."""
    keys = _erasure_keys(user_id, client_id)
    for key in keys:
        _ERASURES_RUNNING[key] = _ERASURES_RUNNING.get(key, 0) + 1
    try:
        yield
    finally:
        ended = time.monotonic()
        for key in keys:
            _ERASURE_ENDED_AT[key] = ended
            left = _ERASURES_RUNNING.get(key, 1) - 1
            if left > 0:
                _ERASURES_RUNNING[key] = left
            else:
                _ERASURES_RUNNING.pop(key, None)


def erased_since(
    user_id: Optional[str],
    started_monotonic: Optional[float],
    client_id: Optional[str] = None,
) -> bool:
    """Whether an erasure of ``user_id`` (or of ``client_id``) runs now, or
    ended after ``started_monotonic`` (``time.monotonic()`` at the
    retrieval's start): its deletion of the log rows may then have run while
    the retrieval was in flight."""
    for key in _erasure_keys(user_id, client_id):
        if _ERASURES_RUNNING.get(key):
            return True
        ended = _ERASURE_ENDED_AT.get(key)
        if ended is not None and started_monotonic is not None:
            if ended >= started_monotonic:
                return True
    return False


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON data: non-finite floats become None, unknown
    objects their ``str()`` (Postgres rejects NaN in a JSON column)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        try:
            return json_safe(value.item())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


async def _user_is_live(db, user_id: str) -> bool:
    """Whether ``user_id`` exists and is not soft-deleted (in ``db``)."""
    from mirix.orm.user import User as UserModel

    deleted = (
        await db.execute(select(UserModel.is_deleted).where(UserModel.id == user_id))
    ).first()
    return deleted is not None and not deleted[0]


def _without_trace(row: RetrievalLogModel) -> PydanticRetrievalLog:
    """A row as its schema, ``per_type`` left out (never loaded)."""
    return PydanticRetrievalLog(
        id=row.id,
        organization_id=row.organization_id,
        user_id=row.user_id,
        client_id=row.client_id,
        session_id=row.session_id,
        memory_types=row.memory_types,
        query_text=row.query_text,
        extracted_topics=row.extracted_topics,
        extraction_status=row.extraction_status,  # type: ignore[arg-type]
        embedding_status=row.embedding_status,  # type: ignore[arg-type]
        per_type=None,
        llm_usage=row.llm_usage,
        latency_ms=row.latency_ms,
        retry_events=list(row.retry_events or []),
        embedding_chunked_calls=int(row.embedding_chunked_calls or 0),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class RetrievalLogManager:
    """Writes, lists, purges and erases ``retrieval_log`` rows."""

    def __init__(self, *, session_maker=None, clock: Optional[Clock] = None):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        self.clock: Clock = clock or utc_now

    def now(self) -> dt.datetime:
        return as_utc(self.clock())  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Write                                                              #
    # ------------------------------------------------------------------ #

    async def create_log(
        self,
        *,
        organization_id: str,
        user_id: str,
        client_id: Optional[str],
        session_id: Optional[str],
        memory_types: Optional[Sequence[str]],
        query_text: Optional[str],
        extracted_topics: Optional[str],
        extraction_status: Optional[str],
        embedding_status: Optional[str],
        per_type: Optional[Dict[str, Any]],
        llm_usage: Optional[Dict[str, Any]],
        latency_ms: Optional[int],
        retry_events: Optional[List[Dict[str, Any]]] = None,
        embedding_chunked_calls: int = 0,
        log_id: Optional[str] = None,
        started_monotonic: Optional[float] = None,
    ) -> Optional[str]:
        """Insert one row (transient database errors retried); returns
        its id — or None, writing nothing, when the user was deleted or its
        erasure began after the retrieval started (``started_monotonic``,
        :func:`erasing_retrieval_logs`). The values are validated by the
        schema first (in a worker thread: a trace can be large)."""
        if erased_since(user_id, started_monotonic, client_id):
            logger.info(
                "Retrieval log row of user %s (client %s) not written: the data "
                "is being (or was just) erased",
                user_id,
                client_id,
            )
            return None
        query_text, cut = cut_query_text(query_text)
        if cut:
            llm_usage = dict(llm_usage or {})
            llm_usage["query_text_truncated_chars"] = cut
        # Postgres cannot store U+0000: a query carrying one (a tool result in the
        # conversation) would lose the whole row.
        query_text, nul = _without_nul(query_text)
        extracted_topics, topics_nul = _without_nul(extracted_topics)
        if nul or topics_nul:
            llm_usage = dict(llm_usage or {})
            llm_usage["nul_chars_replaced"] = nul + topics_nul

        def _record() -> PydanticRetrievalLog:
            return PydanticRetrievalLog(
                organization_id=organization_id,
                user_id=user_id,
                client_id=client_id,
                session_id=session_id,
                memory_types=list(memory_types) if memory_types is not None else None,
                query_text=query_text,
                extracted_topics=extracted_topics,
                extraction_status=extraction_status,  # type: ignore[arg-type]
                embedding_status=embedding_status,  # type: ignore[arg-type]
                per_type=json_safe(per_type) if per_type is not None else None,
                llm_usage=json_safe(llm_usage) if llm_usage is not None else None,
                latency_ms=max(0, int(latency_ms)) if latency_ms is not None else None,
                retry_events=json_safe(list(retry_events or [])),
                embedding_chunked_calls=max(0, int(embedding_chunked_calls or 0)),
            )

        record = await asyncio.to_thread(_record)
        row_id = log_id or new_retrieval_log_id()
        created_at = self.now()

        async def _insert() -> Optional[str]:
            async with self.session_maker() as db:
                if not await _user_is_live(db, user_id):
                    return None
                db.add(
                    RetrievalLogModel(
                        id=row_id,
                        organization_id=record.organization_id,
                        user_id=record.user_id,
                        client_id=record.client_id,
                        session_id=record.session_id,
                        memory_types=record.memory_types,
                        query_text=record.query_text,
                        extracted_topics=record.extracted_topics,
                        extraction_status=record.extraction_status,
                        embedding_status=record.embedding_status,
                        per_type=record.per_type,
                        llm_usage=record.llm_usage,
                        latency_ms=record.latency_ms,
                        retry_events=record.retry_events,
                        embedding_chunked_calls=record.embedding_chunked_calls,
                        created_at=created_at,
                        updated_at=created_at,
                    )
                )
                try:
                    await db.commit()
                except IntegrityError:
                    # A retry after a COMMIT whose reply was lost: the row is
                    # there (same id). Anything else is a real failure.
                    await db.rollback()
                    if await self._exists(row_id):
                        return row_id
                    raise
            if erased_since(user_id, started_monotonic, client_id):
                # An erasure began while this row was written: its
                # deletion may already have run — remove the row ourselves.
                await self._delete_where(RetrievalLogModel.id == row_id)
                return None
            return row_id

        return await retry_call(_insert, stage="retrieval_log")

    async def _exists(self, row_id: str) -> bool:
        async with self.session_maker() as db:
            return (
                await db.execute(
                    select(RetrievalLogModel.id).where(RetrievalLogModel.id == row_id)
                )
            ).first() is not None

    # ------------------------------------------------------------------ #
    # Read                                                               #
    # ------------------------------------------------------------------ #

    async def list_logs(
        self,
        *,
        organization_id: str,
        client_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        include_trace: bool = True,
    ) -> List[PydanticRetrievalLog]:
        """A client's own rows in its organization, newest first;
        ``include_trace=False`` does not read ``per_type`` (None)."""
        filters = [
            RetrievalLogModel.organization_id == organization_id,
            RetrievalLogModel.client_id == client_id,
            RetrievalLogModel.is_deleted.is_(False),
        ]
        if user_id is not None:
            filters.append(RetrievalLogModel.user_id == user_id)
        if session_id is not None:
            filters.append(RetrievalLogModel.session_id == session_id)
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        stmt = (
            select(RetrievalLogModel)
            .where(*filters)
            .order_by(
                RetrievalLogModel.created_at.desc(),
                RetrievalLogModel.id.desc(),
            )
            .limit(limit)
        )
        if not include_trace:
            stmt = stmt.options(defer(RetrievalLogModel.per_type, raiseload=True))
        async with self.session_maker() as db:
            rows = list((await db.execute(stmt)).scalars())
            if include_trace:
                return [cast(PydanticRetrievalLog, row.to_pydantic()) for row in rows]
            return [_without_trace(row) for row in rows]

    async def get_log(
        self, log_id: str, *, organization_id: str
    ) -> Optional[PydanticRetrievalLog]:
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    select(RetrievalLogModel).where(
                        RetrievalLogModel.id == log_id,
                        RetrievalLogModel.organization_id == organization_id,
                        RetrievalLogModel.is_deleted.is_(False),
                    )
                )
            ).scalar_one_or_none()
            return (
                cast(PydanticRetrievalLog, row.to_pydantic())
                if row is not None
                else None
            )

    # ------------------------------------------------------------------ #
    # Retention and erasure                                              #
    # ------------------------------------------------------------------ #

    async def _delete_where(self, *conditions: Any, batch: Optional[int] = None) -> int:
        """Delete matching rows in batches (short transactions); returns the count."""
        batch = int(batch or constants.RETRIEVAL_LOG_PURGE_BATCH)
        total = 0
        while True:
            async with self.session_maker() as db:
                ids = (
                    (
                        await db.execute(
                            select(RetrievalLogModel.id).where(*conditions).limit(batch)
                        )
                    )
                    .scalars()
                    .all()
                )
                if not ids:
                    return total
                await db.execute(
                    delete(RetrievalLogModel).where(RetrievalLogModel.id.in_(ids))
                )
                await db.commit()
            total += len(ids)
            if len(ids) < batch:
                return total

    async def purge_expired(self, retention_days: Optional[int] = None) -> int:
        """Delete rows older than the retention window (default
        ``MIRIX_RETRIEVAL_LOG_RETENTION_DAYS``); <= 0 days keeps every row."""
        days = (
            constants.RETRIEVAL_LOG_RETENTION_DAYS
            if retention_days is None
            else int(retention_days)
        )
        if days <= 0:
            return 0
        cutoff = self.now() - dt.timedelta(days=days)
        purged = await self._delete_where(RetrievalLogModel.created_at < cutoff)
        if purged:
            logger.info(
                "Purged %d retrieval log row(s) older than %d day(s)", purged, days
            )
        return purged

    async def delete_by_user_id(self, user_id: str) -> int:
        """Erasure: every row of ``user_id``, in every organization."""
        return await self._delete_where(RetrievalLogModel.user_id == user_id)

    async def delete_by_client_id(self, client_id: str) -> int:
        """Erasure: every row the client's retrievals wrote."""
        return await self._delete_where(RetrievalLogModel.client_id == client_id)


class RetrievalLogPurger:
    """Background retention purge: one pass at startup, then every
    ``interval_seconds`` (``RETRIEVAL_LOG_PURGE_INTERVAL_SECONDS``, 3600).
    A failing pass is logged and the loop goes on; :meth:`stop` cancels it."""

    def __init__(
        self,
        *,
        interval_seconds: Optional[float] = None,
        manager: Optional[RetrievalLogManager] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ):
        self.interval_seconds = float(
            constants.RETRIEVAL_LOG_PURGE_INTERVAL_SECONDS
            if interval_seconds is None
            else interval_seconds
        )
        self._manager = manager
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None
        self.passes = 0

    @property
    def manager(self) -> RetrievalLogManager:
        if self._manager is None:
            self._manager = RetrievalLogManager()
        return self._manager

    async def run_once(self) -> int:
        purged = await self.manager.purge_expired()
        self.passes += 1
        return purged

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — keep purging after a failed pass
                logger.exception("Retrieval log retention purge failed")
            await self._sleep(self.interval_seconds)

    def start(self) -> Optional[asyncio.Task]:
        """Start the loop on the running event loop (idempotent). None when
        the interval is <= 0 (no periodic purge)."""
        if self.interval_seconds <= 0:
            return None
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._loop(), name="mirix-retrieval-log-purge"
            )
        return self._task

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def task(self) -> Optional[asyncio.Task]:
        return self._task

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("Retrieval log purge ended with an error")
