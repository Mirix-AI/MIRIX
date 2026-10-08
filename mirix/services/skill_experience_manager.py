"""Manager for the general session-experience store.

Async-only: it grabs sessions via `self.session_maker` (the server's
`db_context`) and never calls `asyncio.run()` (the server event loop is already
running).

The store is the durable hand-off between session distillation (one or more
experiences per session) and skill evolution (consumes pending experiences,
prioritized by importance*credibility, to create/edit skills). Experiences flow
pending -> consumed; the curator never marks one superseded (that transition is
reserved for pending lessons later evidence replaced, :meth:`mark_superseded`).

The experience pool: a curator batch that finishes records each
experience's ``disposition`` — ``used`` when a committed skill write cited it
(``cited_skill_ids``), else ``covered`` / ``not_worth`` / ``deferred`` as the
curator triaged it (``deferred`` when it did not) — and one closed out after
its third attempt is ``unaddressed`` (``used`` when a write cited it all the
same). Consumed ``deferred`` / ``not_worth`` experiences stay in the pool of
their (organization, user, meta agent) and write scope (``pool_scope``) until
they are retired (``pool_retired_at``, ``pool_retire_reason``): used, covered,
ignored by too many batches that showed them, merged into the earlier lesson
they recurred, or unembeddable (:meth:`settle_curation_batch`). ``status``
stays ``consumed``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import delete, false, func, select, type_coerce
from sqlalchemy.types import NullType

from mirix.client.utils import get_utc_time
from mirix.log import get_logger
from mirix.orm.skill_experience import SkillExperience as SkillExperienceModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.skill_experience import (
    SkillExperience as PydanticSkillExperience,
    SkillExperienceCreate,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)

#: ``consumed_by`` of an experience closed out unaddressed: the batch
#: that gave up on it, plus this suffix. Its status is ``consumed`` so it is
#: never curated again; the suffix tells it apart from one a batch finished.
UNADDRESSED_SUFFIX = ":unaddressed"

# Batch outcomes (see SkillExperienceManager.settle_curation_batch).
BATCH_CONSUME = "consume"  # the curator finished the batch
BATCH_RETURN = "return"  # back to pending, one more curation attempt counted
BATCH_KEEP = "keep"  # left pending as is (an infrastructure failure)

# What became of a consumed experience (``disposition``).
DISPOSITION_USED = "used"  # a committed skill write cited it
DISPOSITION_COVERED = "covered"  # an existing skill already says it
DISPOSITION_NOT_WORTH = "not_worth"  # too narrow, wrong, not worth a skill
DISPOSITION_DEFERRED = "deferred"  # may matter later (also: not triaged)
DISPOSITION_UNADDRESSED = "unaddressed"  # closed out after its third attempt
#: Dispositions of the experiences that form the pool.
POOL_DISPOSITIONS = (DISPOSITION_DEFERRED, DISPOSITION_NOT_WORTH)
#: The lists of ``experience_triage`` (an id's triage is one of them).
TRIAGE_LISTS = (DISPOSITION_COVERED, DISPOSITION_NOT_WORTH, DISPOSITION_DEFERRED)
# Why a pool experience was retired (``pool_retire_reason``).
RETIRE_USED = "used"
RETIRE_COVERED = "covered"
RETIRE_IGNORED = "ignored"
#: A new experience that recurred a lesson still in the pool: that lesson
#: counts it (``recurrence_count``), so it does not stay as a second row.
RETIRE_MERGED = "merged"
#: Its embedding kept failing (MIRIX_SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES).
RETIRE_UNEMBEDDABLE = "unembeddable"
RETIRE_REASONS = (
    RETIRE_USED,
    RETIRE_COVERED,
    RETIRE_IGNORED,
    RETIRE_MERGED,
    RETIRE_UNEMBEDDABLE,
)

#: The pool keys of a settled batch's ``curator_metrics`` entry.
POOL_SUMMARY_KEYS = (
    "dispositions",
    "history_cited",
    "history_covered",
    "history_not_worth",
    "history_ignored",
    "recurrences",
    "embedding_failures",
    "retired",
    "retired_ignored",
    "pool_active_after",
)


@dataclass(frozen=True)
class PoolSettlement:
    """What a curator batch showed and decided, for its settle:
    its write ``scope``, the earlier lessons it showed (``pushed_ids``), the
    pool experiences that recurred in it (``recurred_ids``) and which of them
    each new experience recurred (``recurred_by``, new id -> pool ids), the
    pool experiences whose own embedding failed while the batch's new ones
    embedded (``embedding_failed_ids``), the curator's triage (id ->
    covered | not_worth | deferred) and the ids a write cited without a
    quote showing where it applies them (``unquoted_ids``: not
    recorded as citations, they stay pending)."""

    scope: Optional[str]
    pushed_ids: Tuple[str, ...] = ()
    recurred_ids: Tuple[str, ...] = ()
    triage: Mapping[str, str] = field(default_factory=dict)
    recurred_by: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    embedding_failed_ids: Tuple[str, ...] = ()
    unquoted_ids: Tuple[str, ...] = ()


def _empty_pool_summary() -> Dict[str, Any]:
    return {
        "dispositions": {
            DISPOSITION_USED: 0,
            DISPOSITION_COVERED: 0,
            DISPOSITION_NOT_WORTH: 0,
            DISPOSITION_DEFERRED: 0,
            "defaulted": 0,
            DISPOSITION_UNADDRESSED: 0,
        },
        "history_cited": 0,
        "history_covered": 0,
        "history_not_worth": 0,
        "history_ignored": 0,
        "recurrences": 0,
        "embedding_failures": 0,
        "retired": {reason: 0 for reason in RETIRE_REASONS},
        "retired_ignored": 0,
        "pool_active_after": 0,
    }


def _retire(row, reason: str, now, summary: Optional[Dict[str, Any]]) -> None:
    row.pool_retired_at = now
    row.pool_retire_reason = reason
    if summary is not None:
        summary["retired"][reason] += 1
        if reason == RETIRE_IGNORED:
            summary["retired_ignored"] += 1


def _vector_or_none(value):
    """A stored pool vector as a float32 array (pgvector and the SQLite
    column both return numpy arrays; a list is converted), or None."""
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


# Above this many raw pool vectors, parsing them (pgvector's text form: about
# a third of a millisecond per 4,096-dimension vector) runs off the event loop.
_VECTOR_PARSE_THREAD_THRESHOLD = 64


def _parse_vectors(raw_values: Sequence[Any], process) -> List[Any]:
    """Raw driver values of the pool vector column through the column type's
    own result processor (``process``; None when the driver value is already
    the result), as float32 arrays. Pure CPU."""
    parsed = []
    for raw in raw_values:
        value = process(raw) if process is not None and raw is not None else raw
        parsed.append(_vector_or_none(value))
    return parsed


def unaddressed_consumer(batch_id: str) -> str:
    """The ``consumed_by`` of an experience a batch closed out unaddressed."""
    return f"{batch_id}{UNADDRESSED_SUFFIX}"


def is_unaddressed(experience) -> bool:
    """Whether an experience was closed out unaddressed."""
    consumed_by = getattr(experience, "consumed_by", None)
    if isinstance(experience, dict):
        consumed_by = experience.get("consumed_by")
    return bool(consumed_by) and str(consumed_by).endswith(UNADDRESSED_SUFFIX)


class SkillExperienceManager:
    """Persist and query distilled per-session transferable experiences."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create_experience(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        session_id: str,
        experience_type: str,
        title: str,
        content: str = "",
        importance: float = 0.0,
        credibility: float = 0.0,
        evidence: str = "",
        status: str = "pending",
        created_by_id: Optional[str] = None,
    ) -> PydanticSkillExperience:
        """Insert one distilled experience in status 'pending' (default).

        The payload is validated through `SkillExperienceCreate` first so an
        invalid `experience_type` (or over-length field) is rejected up front,
        and importance/credibility are clamped into [0,1]. The DB column is a
        plain String, so without this an invalid value would commit and only
        blow up later in `to_pydantic()` / `list_experiences`.
        """
        row = self.build_row(
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            session_id=session_id,
            experience_type=experience_type,
            title=title,
            content=content,
            importance=importance,
            credibility=credibility,
            evidence=evidence,
            status=status,
            created_by_id=created_by_id,
        )
        async with self.session_maker() as session:
            await row.create(session)
            return row.to_pydantic()

    @staticmethod
    def build_row(
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        session_id: str,
        experience_type: str,
        title: str,
        content: str = "",
        importance: float = 0.0,
        credibility: float = 0.0,
        evidence: str = "",
        status: str = "pending",
        created_by_id: Optional[str] = None,
    ) -> SkillExperienceModel:
        """A validated, not-yet-added experience row.

        For callers that insert experiences inside their own transaction (a
        learning run commits a session's experiences together with marking its
        rows distilled). Validation is the same as :meth:`create_experience`.
        """
        validated = SkillExperienceCreate(
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            session_id=session_id,
            experience_type=experience_type,
            title=title,
            content=content,
            importance=importance,
            credibility=credibility,
            evidence=evidence,
            status=status,
        )
        row = SkillExperienceModel(
            agent_id=validated.agent_id,
            user_id=validated.user_id,
            organization_id=validated.organization_id,
            session_id=validated.session_id,
            experience_type=validated.experience_type,
            title=validated.title,
            content=validated.content,
            importance=validated.importance,
            credibility=validated.credibility,
            evidence=validated.evidence,
            status=validated.status,
        )
        if created_by_id is not None:
            # Client attribution via the _created_by_id audit column — this is
            # what delete_by_client_id (the client erasure path) filters on.
            row._set_created_and_updated_by_fields(created_by_id)
        return row

    @enforce_types
    async def list_experiences(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        status: Optional[str] = "pending",
        limit: int = 100,
        ids: Optional[List[str]] = None,
    ) -> List[PydanticSkillExperience]:
        """Return this agent's experiences, prioritized for evolution consumption.

        Filters by agent (always), optionally by user and status (default
        'pending'); `status=None` returns all statuses. Excludes soft-deleted
        rows.

        `ids` scopes the result to an explicit id set (the experience curator passes
        THIS evolution round's freshly-distilled experiences so a run only ever
        sees its own batch, never the accumulated cross-round pending pool).
        `ids=None` (default) applies no id filter; `ids=[]` is an explicit
        "scope to nothing" and short-circuits to `[]` (so a round that distilled
        zero experiences evolves over nothing rather than the whole pool).

        Ordering: `(importance * credibility) DESC` computed in SQL (the
        priority), then `created_at DESC` as a stable tiebreak.
        """
        # Explicit empty scope -> nothing (distinct from ids=None == "no filter").
        # Also avoids emitting a degenerate `IN ()` predicate.
        if ids is not None and len(ids) == 0:
            return []

        priority = (SkillExperienceModel.importance * SkillExperienceModel.credibility)
        preds = [
            SkillExperienceModel.agent_id == agent_id,
            SkillExperienceModel.is_deleted.is_(False),
        ]
        if user_id is not None:
            preds.append(SkillExperienceModel.user_id == user_id)
        if status is not None:
            preds.append(SkillExperienceModel.status == status)
        if ids is not None:
            preds.append(SkillExperienceModel.id.in_(ids))

        stmt = (
            select(SkillExperienceModel)
            .where(*preds)
            .order_by(priority.desc(), SkillExperienceModel.created_at.desc())
            .limit(limit)
        )
        async with self.session_maker() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_pydantic() for row in rows]

    async def list_pending_for_curation(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> List[PydanticSkillExperience]:
        """Every pending experience of the (organization, user, meta agent)
        scope, oldest first — ordered by ``(created_at, id)`` — with no limit:
        the curator batches all of them, none is dropped."""
        preds = [
            SkillExperienceModel.agent_id == agent_id,
            SkillExperienceModel.user_id == user_id,
            SkillExperienceModel.status == "pending",
            SkillExperienceModel.is_deleted.is_(False),
        ]
        if organization_id is not None:
            preds.append(SkillExperienceModel.organization_id == organization_id)
        stmt = (
            select(SkillExperienceModel)
            .where(*preds)
            .order_by(SkillExperienceModel.created_at, SkillExperienceModel.id)
        )
        async with self.session_maker() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [row.to_pydantic() for row in rows]

    # ------------------------------------------------------------------ #
    # The experience pool                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _owner_preds(
        *, agent_id: str, user_id: str, organization_id: Optional[str]
    ) -> list:
        preds = [
            SkillExperienceModel.agent_id == agent_id,
            SkillExperienceModel.user_id == user_id,
        ]
        if organization_id is not None:
            preds.append(SkillExperienceModel.organization_id == organization_id)
        return preds

    @staticmethod
    def _pool_preds(
        *,
        agent_id: str,
        user_id: str,
        organization_id: Optional[str],
        scope: Optional[str],
    ) -> list:
        """The active pool of the (organization, user, meta agent) and write
        ``scope``: consumed, deferred or not worth a skill, not retired, not
        deleted. A ``None`` scope matches nothing (a scope-less owner never
        pools). Rows a skill write cited are left to the caller."""
        if not scope:
            return [false()]
        return SkillExperienceManager._owner_preds(
            agent_id=agent_id, user_id=user_id, organization_id=organization_id
        ) + [
            SkillExperienceModel.status == "consumed",
            SkillExperienceModel.disposition.in_(POOL_DISPOSITIONS),
            SkillExperienceModel.pool_retired_at.is_(None),
            SkillExperienceModel.is_deleted.is_(False),
            SkillExperienceModel.pool_scope == scope,
        ]

    async def list_active_pool(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: Optional[str],
        scope: Optional[str],
        limit: int,
    ) -> Tuple[List[Any], bool]:
        """The active pool experiences of the owner and write ``scope`` that
        no skill write cited, as :class:`PoolCandidate` values (with their
        stored vectors) in ``(created_at, id)`` order, and whether the scan
        was cut: at most ``limit`` rows are read, newest first. The vectors
        are read as the driver returns them and parsed by the column type's
        own processor — off the event loop for a large scan (up to
        ``MIRIX_SKILL_EXPERIENCE_POOL_SCAN_LIMIT`` vectors of 4,096
        dimensions). A transient database failure is retried."""
        from mirix.services.retry_policy import retry_call
        from mirix.services.skill_experience_pool import PoolCandidate

        preds = self._pool_preds(
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            scope=scope,
        )
        limit = max(0, int(limit))
        # The raw column (no result processing on the loop), parsed below.
        raw_vector = type_coerce(SkillExperienceModel.pool_embedding, NullType())

        async def _once():
            async with self.session_maker() as session:
                dialect = (await session.connection()).dialect
                rows = (
                    await session.execute(
                        select(SkillExperienceModel, raw_vector)
                        .where(*preds)
                        .order_by(
                            SkillExperienceModel.created_at.desc(),
                            SkillExperienceModel.id.desc(),
                        )
                        .limit(limit + 1)
                    )
                ).all()
                truncated = len(rows) > limit
                # Already used: a committed write cited it.
                kept = [(row, raw) for row, raw in rows[:limit] if not row.cited_skill_ids]
                candidates = [
                    PoolCandidate(
                        id=row.id,
                        experience_type=row.experience_type,
                        title=row.title,
                        content=row.content or "",
                        importance=float(row.importance or 0.0),
                        credibility=float(row.credibility or 0.0),
                        recurrence_count=int(row.recurrence_count or 1),
                        pushed_count=int(row.pushed_count or 0),
                        ignored_pushes=int(row.ignored_pushes or 0),
                        created_at=row.created_at,
                        vector_model=row.pool_embedding_model,
                    )
                    for row, _raw in kept
                ]
            column_type = SkillExperienceModel.pool_embedding.type
            process = column_type.dialect_impl(dialect).result_processor(dialect, None)
            raw_values = [raw for _row, raw in kept]
            present = sum(1 for raw in raw_values if raw is not None)
            if present > _VECTOR_PARSE_THREAD_THRESHOLD:
                vectors = await asyncio.to_thread(_parse_vectors, raw_values, process)
            else:
                vectors = _parse_vectors(raw_values, process)
            candidates = [
                replace(candidate, vector=vector)
                for candidate, vector in zip(candidates, vectors)
            ]
            candidates.reverse()
            return candidates, truncated

        return await retry_call(_once, stage="experience_pool_read")

    async def load_pool_vectors(
        self,
        *,
        ids: Sequence[str],
        agent_id: str,
        user_id: str,
        organization_id: Optional[str],
    ) -> Dict[str, Tuple[Optional[Any], Optional[str]]]:
        """``id -> (pool vector, its model)`` of the owner's experiences
        ``ids`` (any status); another owner's ids are absent."""
        from mirix.services.retry_policy import retry_call

        if not ids:
            return {}
        preds = self._owner_preds(
            agent_id=agent_id, user_id=user_id, organization_id=organization_id
        ) + [
            SkillExperienceModel.id.in_(list(ids)),
            SkillExperienceModel.is_deleted.is_(False),
        ]

        async def _once():
            async with self.session_maker() as session:
                rows = (
                    await session.execute(
                        select(
                            SkillExperienceModel.id,
                            SkillExperienceModel.pool_embedding,
                            SkillExperienceModel.pool_embedding_model,
                        ).where(*preds)
                    )
                ).all()
            return {row[0]: (_vector_or_none(row[1]), row[2]) for row in rows}

        return await retry_call(_once, stage="experience_pool_read")

    async def store_pool_vectors(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        model: Optional[str],
        agent_id: str,
        user_id: str,
        organization_id: Optional[str],
        write_guard=None,
    ) -> int:
        """Store pool vectors (zero-padded to MAX_EMBEDDING_DIM) and their
        embedding ``model`` on the owner's experiences, in ONE transaction
        that runs ``write_guard(db)`` first (the curator's check that its
        learning run is still active: a reclaimed run writes nothing). Another
        owner's ids are never touched. Returns the rows updated; a transient
        database failure repeats the transaction."""
        from mirix.services.procedural_memory_manager import _pad_to_max
        from mirix.services.retry_policy import retry_call

        if not vectors:
            return 0
        preds = self._owner_preds(
            agent_id=agent_id, user_id=user_id, organization_id=organization_id
        ) + [
            SkillExperienceModel.id.in_(list(vectors)),
            SkillExperienceModel.is_deleted.is_(False),
        ]
        padded = {key: _pad_to_max(list(value)) for key, value in vectors.items()}

        async def _once() -> int:
            async with self.session_maker() as session:
                if write_guard is not None:
                    await write_guard(session)
                rows = (
                    (
                        await session.execute(
                            select(SkillExperienceModel)
                            .where(*preds)
                            .order_by(SkillExperienceModel.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                now = get_utc_time()
                for row in rows:
                    row.pool_embedding = padded[row.id]
                    row.pool_embedding_model = model
                    row.set_updated_at(now)
                await session.commit()
                return len(rows)

        return await retry_call(_once, stage="experience_pool_vectors")

    @staticmethod
    async def count_active_pool(
        db,
        *,
        agent_id: str,
        user_id: str,
        organization_id: Optional[str],
        scope: Optional[str],
    ) -> int:
        """The size of the owner's active pool in ``scope`` (in the caller's
        transaction): pool rows no skill write cited."""
        preds = SkillExperienceManager._pool_preds(
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            scope=scope,
        )
        total = int(
            (await db.execute(select(func.count(SkillExperienceModel.id)).where(*preds)))
            .scalar_one()
        )
        if not total:
            return 0
        # JSON emptiness is awkward across dialects: subtract the cited rows.
        cited = (
            await db.execute(
                select(SkillExperienceModel.id, SkillExperienceModel.cited_skill_ids).where(
                    *preds, SkillExperienceModel.cited_skill_ids.is_not(None)
                )
            )
        ).all()
        return total - sum(1 for _id, skills in cited if skills)

    @staticmethod
    async def settle_curation_batch(
        db,
        *,
        ids: List[str],
        agent_id: str,
        user_id: str,
        outcome: str,
        batch_id: str,
        max_attempts: int,
        now,
        organization_id: Optional[str] = None,
        pool: Optional[PoolSettlement] = None,
    ) -> Dict[str, Any]:
        """Record a curator batch's outcome on its experiences and on the
        experience pool, in the caller's transaction. Only
        the owner's (``agent_id``, ``user_id`` and, when given,
        ``organization_id``) rows are touched — the batch's still-pending
        experiences, and the pool rows of ``pool.scope``:

        * ``consume`` (the curator called finish_memory_update): every one
          becomes ``consumed`` with ``consumed_by = batch_id`` and its
          ``disposition``: ``used`` when a committed skill write cited it
          (``cited_skill_ids``, an earlier returned attempt's citations
          included), else ``covered`` / ``not_worth`` / ``deferred`` from
          ``pool.triage`` (``deferred`` — and counted as defaulted — when not
          triaged). ``not_worth`` and ``deferred`` enter the pool of
          ``pool.scope`` (``not_worth`` already decayed once). An experience
          neither cited nor triaged that a write cited without a verifiable
          quote (``pool.unquoted_ids``) is NOT consumed: it stays
          ``pending`` like a returned one (``curation_attempts + 1``, closed
          out ``unaddressed`` at ``max_attempts``). Then, in this
          order: the pool experiences in ``pool.recurred_ids`` recurred
          (``recurrence_count + 1``, ``ignored_pushes = 0``); every pool
          experience in ``pool.pushed_ids`` counts one more push and is
          retired ``used`` when cited, ``covered`` when triaged so, left as
          it was (not ignored) when untriaged and cited only without a
          verifiable quote (``pool.unquoted_ids``), and otherwise ignored (+2
          when triaged not worth, else +1 — listed deferred included),
          retired ``ignored`` at 3; every pool experience
          in ``pool.embedding_failed_ids`` counts one more failed embedding
          and is retired ``unembeddable`` at
          ``MIRIX_SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES``; last, a new
          experience that entered the pool and recurred a pool experience
          still active (``pool.recurred_by``) is retired ``merged``: that
          lesson already counts it, so one lesson stays one pool row;
        * ``return`` (context_limit, step_cap, a non-infrastructure curator
          failure): ``curation_attempts + 1``, and one that reached
          ``max_attempts`` is closed out — ``consumed`` with ``consumed_by =
          batch_id + ":unaddressed"`` and disposition ``unaddressed``, or
          ``used`` when a committed write cited it all the same (it never
          enters the pool); the others stay ``pending``;
        * ``keep`` (an infrastructure failure): the experiences do not change.

        On every outcome, a shown pool experience that a committed write cited
        is retired ``used`` (the citation happened). A returned or kept batch
        leaves the pool otherwise as it was, so a batch's recurrence, decay and
        failed embeddings apply once, when it finishes.

        Returns ``{"consumed": [...], "returned": [...], "unaddressed": [...]}``,
        plus ``"pool"`` (the :data:`POOL_SUMMARY_KEYS` of the batch's metrics
        entry) when ``pool`` is given.
        """
        from mirix.constants import (
            SKILL_EXPERIENCE_NOT_WORTH_PENALTY,
            SKILL_EXPERIENCE_NOT_WORTH_START,
            SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES,
            SKILL_EXPERIENCE_POOL_RETIRE_IGNORED,
        )

        if outcome not in (BATCH_CONSUME, BATCH_RETURN, BATCH_KEEP):
            raise ValueError(f"unknown curator batch outcome {outcome!r}")
        settled: Dict[str, Any] = {"consumed": [], "returned": [], "unaddressed": []}
        summary = _empty_pool_summary() if pool is not None else None
        triage = dict(pool.triage) if pool is not None else {}
        unquoted = set(pool.unquoted_ids) if pool is not None else set()

        def kept_pending(row) -> bool:
            # Cited only without a quote that shows where a write applies
            # it, and not triaged: the batch did not address it.
            return (
                row.id in unquoted
                and not row.cited_skill_ids
                and triage.get(row.id) is None
            )

        def count_attempt(row) -> None:
            """A curation attempt that did not address ``row``: it stays
            pending, or is closed out at ``max_attempts``."""
            row.curation_attempts = int(row.curation_attempts or 0) + 1
            if row.curation_attempts >= max_attempts:
                row.status = "consumed"
                row.consumed_by = unaddressed_consumer(batch_id)
                # A committed write of an attempt cited it: that is work
                # done on it, closed out or not.
                row.disposition = (
                    DISPOSITION_USED if row.cited_skill_ids else DISPOSITION_UNADDRESSED
                )
                if summary is not None:
                    summary["dispositions"][row.disposition] += 1
                settled["unaddressed"].append(row.id)
            else:
                settled["returned"].append(row.id)

        async def pool_rows(pool_ids: Sequence[str]):
            if pool is None or not pool_ids:
                return []
            preds = SkillExperienceManager._pool_preds(
                agent_id=agent_id,
                user_id=user_id,
                organization_id=organization_id,
                scope=pool.scope,
            )
            return (
                (
                    await db.execute(
                        select(SkillExperienceModel)
                        .where(SkillExperienceModel.id.in_(list(pool_ids)), *preds)
                        .order_by(SkillExperienceModel.created_at, SkillExperienceModel.id)
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )

        async def finish() -> Dict[str, Any]:
            if summary is not None and pool is not None:
                await db.flush()
                summary["pool_active_after"] = await SkillExperienceManager.count_active_pool(
                    db,
                    agent_id=agent_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    scope=pool.scope,
                )
                settled["pool"] = summary
            await db.flush()
            return settled

        if pool is not None and outcome in (BATCH_RETURN, BATCH_KEEP):
            # A citation is a fact whatever the batch's outcome.
            for row in await pool_rows(pool.pushed_ids):
                if row.cited_skill_ids:
                    _retire(row, RETIRE_USED, now, summary)
                    summary["history_cited"] += 1
                    row.set_updated_at(now)
        if not ids or outcome == BATCH_KEEP:
            return await finish()

        preds = SkillExperienceManager._owner_preds(
            agent_id=agent_id, user_id=user_id, organization_id=organization_id
        ) + [
            SkillExperienceModel.id.in_(list(ids)),
            SkillExperienceModel.status == "pending",
            SkillExperienceModel.is_deleted.is_(False),
        ]
        rows = (
            (
                await db.execute(
                    select(SkillExperienceModel)
                    .where(*preds)
                    .order_by(SkillExperienceModel.created_at, SkillExperienceModel.id)
                    .with_for_update()
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            if outcome == BATCH_CONSUME and kept_pending(row):
                count_attempt(row)
            elif outcome == BATCH_CONSUME:
                row.status = "consumed"
                row.consumed_by = batch_id
                listed = triage.get(row.id)
                if row.cited_skill_ids:
                    row.disposition = DISPOSITION_USED
                elif listed == DISPOSITION_COVERED:
                    row.disposition = DISPOSITION_COVERED
                elif listed == DISPOSITION_NOT_WORTH:
                    row.disposition = DISPOSITION_NOT_WORTH
                    row.ignored_pushes = SKILL_EXPERIENCE_NOT_WORTH_START
                    row.pool_scope = pool.scope if pool is not None else None
                else:
                    row.disposition = DISPOSITION_DEFERRED
                    row.ignored_pushes = 0
                    row.pool_scope = pool.scope if pool is not None else None
                    if summary is not None and listed is None:
                        summary["dispositions"]["defaulted"] += 1
                if summary is not None:
                    summary["dispositions"][row.disposition] += 1
                settled["consumed"].append(row.id)
            else:
                count_attempt(row)
            row.set_updated_at(now)

        if outcome == BATCH_CONSUME and pool is not None:
            # (1) Recurrence first: a lesson that came up again starts afresh.
            recurred_rows = await pool_rows(pool.recurred_ids)
            for row in recurred_rows:
                row.recurrence_count = int(row.recurrence_count or 1) + 1
                row.ignored_pushes = 0
                row.set_updated_at(now)
                summary["recurrences"] += 1
            # (2) Then every earlier lesson the batch showed (the same
            # identity map: a recurred row is the same object).
            for row in await pool_rows(pool.pushed_ids):
                row.pushed_count = int(row.pushed_count or 0) + 1
                row.last_pushed_at = now
                listed = triage.get(row.id)
                if row.cited_skill_ids:
                    _retire(row, RETIRE_USED, now, summary)
                    summary["history_cited"] += 1
                elif listed == DISPOSITION_COVERED:
                    _retire(row, RETIRE_COVERED, now, summary)
                    summary["history_covered"] += 1
                elif kept_pending(row):
                    pass  # Cited without a verifiable quote: as it was
                else:
                    if listed == DISPOSITION_NOT_WORTH:
                        row.ignored_pushes = (
                            int(row.ignored_pushes or 0) + SKILL_EXPERIENCE_NOT_WORTH_PENALTY
                        )
                        summary["history_not_worth"] += 1
                    else:
                        row.ignored_pushes = int(row.ignored_pushes or 0) + 1
                        summary["history_ignored"] += 1
                    if row.ignored_pushes >= SKILL_EXPERIENCE_POOL_RETIRE_IGNORED:
                        _retire(row, RETIRE_IGNORED, now, summary)
                row.set_updated_at(now)
            # (3) A pool experience the embedding model refused again while
            # this batch's own experiences embedded.
            for row in await pool_rows(pool.embedding_failed_ids):
                row.pool_embedding_failures = int(row.pool_embedding_failures or 0) + 1
                summary["embedding_failures"] += 1
                if row.pool_embedding_failures >= SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES:
                    _retire(row, RETIRE_UNEMBEDDABLE, now, summary)
                row.set_updated_at(now)
            # (4) A new experience that recurred a lesson still in the pool is
            # that lesson again: it counted there (1), so it does not stay as a
            # second row that would take a second top-K slot in every later
            # batch and split the "seen N times" count.
            active = {r.id for r in recurred_rows if r.pool_retired_at is None}
            for row in rows:
                if (
                    row.status == "consumed"
                    and row.consumed_by == batch_id
                    and row.disposition in POOL_DISPOSITIONS
                    and row.pool_retired_at is None
                    and any(i in active for i in pool.recurred_by.get(row.id, ()))
                ):
                    _retire(row, RETIRE_MERGED, now, summary)
        return await finish()

    @enforce_types
    async def mark_consumed(
        self,
        *,
        ids: List[str],
        run_id: str,
        influenced_skill_ids: Optional[List[str]] = None,
    ) -> int:
        """Flip the given experiences to `status='consumed'`, stamping
        `consumed_by` (and optional `influenced_skill_ids` lineage).

        Only flips rows still pending (idempotent; a second consumer can't
        clobber an existing `consumed_by`). Returns the number updated.
        """
        if not ids:
            return 0
        return await self._set_status(
            ids=ids,
            status="consumed",
            consumed_by=run_id,
            influenced_skill_ids=influenced_skill_ids,
            require_pending=True,
        )

    @enforce_types
    async def mark_superseded(
        self,
        *,
        ids: List[str],
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Soft-delete via status transition to 'superseded' (NOT is_deleted).

        Only rows still `pending` are transitioned (`require_pending`): an
        experience that already did its job (`consumed`) must never be flipped to
        `superseded`, so passing a mixed id set (e.g. a curator's "scoped minus
        consumed" overflow) can only ever retire the still-pending leftovers.

        `agent_id` / `user_id`, when given, additionally constrain the update to
        that owner — so an id set that (defensively) contains a foreign owner's id
        can never supersede a DIFFERENT (agent, user)'s experience.
        `list_experiences(status='pending')` already excludes the result. Returns
        the number updated.
        """
        if not ids:
            return 0
        return await self._set_status(
            ids=ids,
            status="superseded",
            require_pending=True,
            agent_id=agent_id,
            user_id=user_id,
        )

    async def _set_status(
        self,
        *,
        ids: List[str],
        status: str,
        consumed_by: Optional[str] = None,
        influenced_skill_ids: Optional[List[str]] = None,
        require_pending: bool = False,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Load-modify-save each row so updated_at / Redis cache stay consistent
        with the rest of the ORM (instead of a bulk UPDATE that bypasses both).

        When `require_pending` is set, only rows still in `status='pending'` are
        transitioned (the rest are left untouched). `agent_id` / `user_id`, when
        given, additionally scope the update to that owner.
        """
        updated = 0
        async with self.session_maker() as session:
            preds = [
                SkillExperienceModel.id.in_(ids),
                SkillExperienceModel.is_deleted.is_(False),
            ]
            if require_pending:
                preds.append(SkillExperienceModel.status == "pending")
            if agent_id is not None:
                preds.append(SkillExperienceModel.agent_id == agent_id)
            if user_id is not None:
                preds.append(SkillExperienceModel.user_id == user_id)
            stmt = select(SkillExperienceModel).where(*preds)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            for row in rows:
                row.status = status
                if consumed_by is not None:
                    row.consumed_by = consumed_by
                if influenced_skill_ids is not None:
                    row.influenced_skill_ids = influenced_skill_ids
                # Pass an explicit UTC timestamp: the no-arg path in
                # CommonSqlalchemyMetaMixins.set_updated_at uses datetime.UTC,
                # which only exists on Python 3.11+ (this repo targets 3.10+).
                row.set_updated_at(get_utc_time())
                session.add(row)
                updated += 1
            await session.commit()
        return updated

    @enforce_types
    async def aggregate(self, *, ids: List[str]) -> Dict[str, float]:
        """Summarize a set of experiences for the skill-evolution budget driver.

        Returns:
            n                 -- total experiences in the set
            n_worth_learning  -- experiences of type 'worth_learning'
            n_worth_avoiding  -- experiences of type 'worth_avoiding'
            sum_priority      -- sum of importance*credibility over the set

        The Goal-3 curator maps `n_worth_avoiding -> n_high_fail` and
        `n_worth_learning -> n_high_succ` so the existing C4 edit-budget formula
        (avoid weighted heavier than learn) applies unchanged.
        """
        if not ids:
            return {
                "n": 0,
                "n_worth_learning": 0,
                "n_worth_avoiding": 0,
                "sum_priority": 0.0,
            }

        async with self.session_maker() as session:
            stmt = select(SkillExperienceModel).where(
                SkillExperienceModel.id.in_(ids),
                SkillExperienceModel.is_deleted.is_(False),
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        n = len(rows)
        n_worth_learning = 0
        n_worth_avoiding = 0
        sum_priority = 0.0
        for row in rows:
            sum_priority += (row.importance or 0.0) * (row.credibility or 0.0)
            if row.experience_type == "worth_learning":
                n_worth_learning += 1
            elif row.experience_type == "worth_avoiding":
                n_worth_avoiding += 1

        return {
            "n": n,
            "n_worth_learning": n_worth_learning,
            "n_worth_avoiding": n_worth_avoiding,
            "sum_priority": sum_priority,
        }

    @enforce_types
    async def delete_by_user_id(self, user_id: str) -> int:
        """Hard delete ALL experiences distilled from a user's sessions (erasure path).

        Experiences carry titles/content/evidence quotes distilled from the
        user's verbatim conversations, so the irreversible
        DELETE /users/{user_id}/memories purge must cover them like every other
        memory table. Deletes regardless of status (pending/consumed/superseded).
        """
        async with self.session_maker() as session:
            stmt = delete(SkillExperienceModel).where(
                SkillExperienceModel.user_id == user_id
            )
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)

    @enforce_types
    async def delete_by_client_id(self, actor: PydanticClient) -> int:
        """Hard delete all experiences created by a client (erasure path).

        Attribution is via the `_created_by_id` audit column — the table has no
        dedicated client_id column. Called from
        `ClientManager.delete_memories_by_client_id`.
        """
        async with self.session_maker() as session:
            stmt = delete(SkillExperienceModel).where(
                SkillExperienceModel._created_by_id == actor.id
            )
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)
