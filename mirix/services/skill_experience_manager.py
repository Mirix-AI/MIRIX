"""Manager for the general session-experience store (Goal 2).

Async-only, mirroring `SkillEvolutionRecordManager`: it grabs sessions via
`self.session_maker` (the server's `db_context`) and never calls
`asyncio.run()` (the server event loop is already running).

The store is the durable hand-off between Goal-2 distillation (one or more
experiences per session) and Goal-3 evolution (consumes pending experiences,
prioritized by importance*credibility, to create/edit skills). Experiences flow
pending -> consumed | superseded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import select

from mirix.client.utils import get_utc_time
from mirix.log import get_logger
from mirix.orm.skill_experience import SkillExperience as SkillExperienceModel
from mirix.schemas.skill_experience import (
    SkillExperience as PydanticSkillExperience,
    SkillExperienceCreate,
)
from mirix.utils import enforce_types

logger = get_logger(__name__)


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
    ) -> PydanticSkillExperience:
        """Insert one distilled experience in status 'pending' (default).

        The payload is validated through `SkillExperienceCreate` first so an
        invalid `experience_type` (or over-length field) is rejected up front,
        and importance/credibility are clamped into [0,1]. The DB column is a
        plain String, so without this an invalid value would commit and only
        blow up later in `to_pydantic()` / `list_experiences`.
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
        async with self.session_maker() as session:
            await row.create(session)
            return row.to_pydantic()

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
        """Return this agent's experiences, prioritized for Goal-3 consumption.

        Filters by agent (always), optionally by user and status (default
        'pending'); `status=None` returns all statuses. Excludes soft-deleted
        rows.

        `ids` scopes the result to an explicit id set (the Goal-3 curator passes
        THIS evolution round's freshly-distilled experiences so a run only ever
        sees its own batch, never the accumulated cross-round pending pool).
        `ids=None` (default) applies no id filter; `ids=[]` is an explicit
        "scope to nothing" and short-circuits to `[]` (so a round that distilled
        zero experiences evolves over nothing rather than the whole pool).

        Ordering: `(importance * credibility) DESC` computed in SQL (the Goal-3
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
        """Summarize a set of experiences for the Goal-3 budget driver.

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
