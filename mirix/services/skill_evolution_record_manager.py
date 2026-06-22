"""Manager for the skill-evolution record store (C2).

Async-only, mirroring `AgentTriggerStateManager`: it grabs sessions via
`self.session_maker` (the server's `db_context`) and never calls
`asyncio.run()` (the server event loop is already running).

The record store is the durable hand-off between the C1 distiller (one record
per graded round) and the C3 curator (consumes a window of records every N
rounds). Records flow pending -> consumed | superseded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import case, select

from mirix.client.utils import get_utc_time
from mirix.log import get_logger
from mirix.orm.skill_evolution_record import (
    SkillEvolutionRecord as SkillEvolutionRecordModel,
)
from mirix.schemas.skill_evolution_record import (
    SkillEvolutionRecord as PydanticSkillEvolutionRecord,
    SkillEvolutionRecordCreate,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)


class SkillEvolutionRecordManager:
    """Persist and query distilled per-round success/failure records."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @staticmethod
    def _is_structurally_gated(record) -> bool:
        """A record counts toward the C4 budget (`n_high_*`) ONLY if it is
        structurally gated: it cites at least one evidence round AND carries a
        non-empty `detail` (the root_cause / what_worked payload).

        Defined here so C4 reuses the exact same gate instead of re-deriving it.
        Accepts either an ORM row or a pydantic record (duck-typed on the two
        attributes).
        """
        evidence = getattr(record, "evidence_round_ids", None) or []
        detail = getattr(record, "detail", None) or ""
        return len(evidence) >= 1 and bool(detail.strip())

    @enforce_types
    async def record_round_result(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        day: str,
        round_id: str,
        round_index: int,
        record_type: str,
        title: str,
        description: str,
        detail: str,
        evidence_round_ids: Optional[List[str]] = None,
        quality_score: float = 0.0,
        generality: float = 0.0,
    ) -> PydanticSkillEvolutionRecord:
        """Insert one distilled record (success or failure) in status 'pending'.

        The payload is validated through `SkillEvolutionRecordCreate` first so an
        invalid `record_type` (or over-length field) is rejected up front. The DB
        column is a plain String, so without this an invalid value would commit
        and only blow up later in `to_pydantic()` / `list_pending`.
        """
        validated = SkillEvolutionRecordCreate(
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            day=day,
            round_id=round_id,
            round_index=round_index,
            record_type=record_type,
            title=title,
            description=description,
            detail=detail,
            evidence_round_ids=evidence_round_ids or [],
            quality_score=quality_score,
            generality=generality,
            status="pending",
        )
        row = SkillEvolutionRecordModel(
            agent_id=validated.agent_id,
            user_id=validated.user_id,
            organization_id=validated.organization_id,
            day=validated.day,
            round_id=validated.round_id,
            round_index=validated.round_index,
            record_type=validated.record_type,
            title=validated.title,
            description=validated.description,
            detail=validated.detail,
            evidence_round_ids=validated.evidence_round_ids,
            quality_score=validated.quality_score,
            generality=validated.generality,
            status=validated.status,
        )
        async with self.session_maker() as session:
            await row.create(session)
            return row.to_pydantic()

    @enforce_types
    async def list_pending(
        self,
        *,
        agent_id: str,
        before_round_index: Optional[int] = None,
        limit: int = 50,
    ) -> List[PydanticSkillEvolutionRecord]:
        """Return this agent's `status='pending'` records.

        Watermark: if `before_round_index` is given, only rounds strictly below
        it are returned (`round_index < before_round_index`) — this prevents
        consuming a record whose own round is still being graded.

        Ordering: failures first (so the curator prioritizes them), then by
        ascending `round_index`.
        """
        # failures (0) sort before successes (1).
        type_rank = case(
            (SkillEvolutionRecordModel.record_type == "failure", 0),
            else_=1,
        )
        stmt = (
            select(SkillEvolutionRecordModel)
            .where(
                SkillEvolutionRecordModel.agent_id == agent_id,
                SkillEvolutionRecordModel.status == "pending",
                SkillEvolutionRecordModel.is_deleted.is_(False),
            )
            .order_by(type_rank, SkillEvolutionRecordModel.round_index)
            .limit(limit)
        )
        if before_round_index is not None:
            stmt = stmt.where(
                SkillEvolutionRecordModel.round_index < before_round_index
            )

        async with self.session_maker() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_pydantic() for row in rows]

    @enforce_types
    async def mark_consumed(self, *, ids: List[str], run_id: str) -> int:
        """Flip the given records to `status='consumed'`, stamping `consumed_by`.

        Returns the number of records updated.
        """
        if not ids:
            return 0
        # Only flip rows that are still pending. This makes consume idempotent
        # and prevents a second consumer from clobbering an existing
        # `consumed_by`. Cross-process exclusion of two evolution runs on the
        # same agent is the curator's job (C3 per-agent lock), per DESIGN §C3.
        return await self._set_status(
            ids=ids, status="consumed", consumed_by=run_id, require_pending=True
        )

    @enforce_types
    async def mark_superseded(self, *, ids: List[str]) -> int:
        """Soft-delete the given records by flipping to `status='superseded'`.

        Soft-delete here means a status transition, NOT `is_deleted=True`: the
        anti-thrash buffer (C4) later reads superseded records' signatures, so
        they must stay queryable. `list_pending` already excludes them by
        filtering `status='pending'`.

        Returns the number of records updated.
        """
        if not ids:
            return 0
        return await self._set_status(ids=ids, status="superseded")

    async def _set_status(
        self,
        *,
        ids: List[str],
        status: str,
        consumed_by: Optional[str] = None,
        require_pending: bool = False,
    ) -> int:
        """Load-modify-save each row so updated_at / Redis cache stay consistent
        with the rest of the ORM (instead of a bulk UPDATE that bypasses both).

        When `require_pending` is set, only rows still in `status='pending'` are
        transitioned (the rest are left untouched).
        """
        updated = 0
        async with self.session_maker() as session:
            preds = [
                SkillEvolutionRecordModel.id.in_(ids),
                SkillEvolutionRecordModel.is_deleted.is_(False),
            ]
            if require_pending:
                preds.append(SkillEvolutionRecordModel.status == "pending")
            stmt = select(SkillEvolutionRecordModel).where(*preds)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            for row in rows:
                row.status = status
                if consumed_by is not None:
                    row.consumed_by = consumed_by
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
        """Summarize a set of records for the C4 budget driver.

        Returns:
            n            -- total records in the set
            n_high_fail  -- structurally-gated failures (see `_is_structurally_gated`)
            n_high_succ  -- structurally-gated successes
            mean_q       -- mean quality_score over the set (0.0 if empty)

        Only structurally-gated records (evidence >= 1 AND non-empty detail)
        count toward `n_high_*`; that gate is the driver of the count-driven
        edit budget, so it lives here for C4 to reuse.
        """
        if not ids:
            return {"n": 0, "n_high_fail": 0, "n_high_succ": 0, "mean_q": 0.0}

        async with self.session_maker() as session:
            stmt = select(SkillEvolutionRecordModel).where(
                SkillEvolutionRecordModel.id.in_(ids),
                SkillEvolutionRecordModel.is_deleted.is_(False),
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        n = len(rows)
        n_high_fail = 0
        n_high_succ = 0
        q_sum = 0.0
        for row in rows:
            q_sum += row.quality_score or 0.0
            if self._is_structurally_gated(row):
                if row.record_type == "failure":
                    n_high_fail += 1
                elif row.record_type == "success":
                    n_high_succ += 1

        mean_q = q_sum / n if n else 0.0
        return {
            "n": n,
            "n_high_fail": n_high_fail,
            "n_high_succ": n_high_succ,
            "mean_q": mean_q,
        }
