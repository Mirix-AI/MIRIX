"""Per-session feedback on served skills, and each skill's usage record.

Feedback — a learning run's distiller judges every skill MIRIX served to the agent
in a session (was it followed, what effect did it have, each with a checked
quote of the transcript); one ``skill_session_feedback`` row per (distilled
session, served skill) keeps that, with the version served and the session's
outcome. The rows are written in the transaction that commits the session's
experiences (``SkillLearningRunManager.commit_distilled_session``).

Usage records — :meth:`SkillSessionFeedbackManager.usage_records` aggregates them per
skill across every session distilled so far: the sessions it was served in
(how many, the most recent ids), how often it was followed, its effects, the
outcomes of those sessions, the same per version served, and its provenance
(how many distinct sessions the experiences cited by its writes came from; only
citations with a verified quote are recorded, so only those count).
:func:`render_usage_record` / :func:`render_usage_line` turn a record into the
plain text the curator sees — in a batch's header, in ``skill_read`` and in
``skill_list``. Information only: nothing here decides anything.

Every value is MIRIX's own judgement from a transcript (there are no labels).
All I/O is async.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import Text, cast, delete, func, or_, select

from mirix.log import get_logger
from mirix.orm.skill_session_feedback import (
    SkillSessionFeedback as SkillSessionFeedbackModel,
)
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.skill_session_feedback import EFFECT_VALUES, FOLLOWED_VALUES

logger = get_logger(__name__)

#: The session outcomes a usage record counts (``none``: the distiller stated
#: no outcome — a legacy answer).
OUTCOME_VALUES = ("succeeded", "failed", "unclear", "none")
#: Session ids a usage record names (the most recent first).
RECENT_SESSIONS_SHOWN = 3


def _counts(values: Sequence[str]) -> Dict[str, int]:
    return {value: 0 for value in values}


@dataclass
class UsageCounts:
    """How often a skill was judged, followed, with which effect, and the
    outcomes of those sessions (one count per feedback row: a session
    reopened after it was distilled is judged once per distilled part)."""

    served: int = 0
    followed: Dict[str, int] = field(default_factory=lambda: _counts(FOLLOWED_VALUES))
    effect: Dict[str, int] = field(default_factory=lambda: _counts(EFFECT_VALUES))
    outcomes: Dict[str, int] = field(default_factory=lambda: _counts(OUTCOME_VALUES))

    def add(self, row: Any) -> None:
        self.served += 1
        followed = row.followed if row.followed in self.followed else "unclear"
        effect = row.effect if row.effect in self.effect else "unclear"
        outcome = (
            row.session_outcome if row.session_outcome in self.outcomes else "none"
        )
        self.followed[followed] += 1
        self.effect[effect] += 1
        self.outcomes[outcome] += 1


@dataclass
class SkillUsageRecord:
    """A skill's usage record (see the module docstring)."""

    skill_id: str
    total: UsageCounts = field(default_factory=UsageCounts)
    #: Distinct sessions it was served in, the most recently distilled first.
    sessions: List[str] = field(default_factory=list)
    #: Per history version served (None: not recorded at serve time).
    by_version: Dict[Optional[int], UsageCounts] = field(default_factory=dict)
    #: The newest version of its history (None: no history).
    current_version: Optional[int] = None
    #: Distinct sessions of the experiences its writes cited with a verified quote
    #: (provenance; a citation without one stays pending and is not counted).
    source_sessions: int = 0

    @property
    def session_count(self) -> int:
        return len(self.sessions)


class SkillSessionFeedbackManager:
    """Writes, aggregates and erases ``skill_session_feedback`` rows."""

    def __init__(self, *, session_maker=None):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker

    # ------------------------------------------------------------------ #
    # Write                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_row(
        *,
        created_by_id: Optional[str] = None,
        learning_run_id: Optional[str] = None,
        **values: Any,
    ) -> SkillSessionFeedbackModel:
        """An unsaved, validated row from the distiller's feedback
        (``feedback_rows``)."""
        from mirix.schemas.skill_session_feedback import SkillSessionFeedback

        SkillSessionFeedback(learning_run_id=learning_run_id, **values)
        row = SkillSessionFeedbackModel(learning_run_id=learning_run_id, **values)
        if created_by_id:
            row._set_created_and_updated_by_fields(created_by_id)
        return row

    async def create_rows(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        created_by_id: Optional[str] = None,
        learning_run_id: Optional[str] = None,
    ) -> int:
        """Insert feedback rows in a transaction of their own (the legacy
        distill path; a learning run writes them in its commit)."""
        built = [
            self.build_row(
                created_by_id=created_by_id, learning_run_id=learning_run_id, **row
            )
            for row in rows
        ]
        if not built:
            return 0
        async with self.session_maker() as db:
            for row in built:
                db.add(row)
            await db.commit()
        return len(built)

    # ------------------------------------------------------------------ #
    # Read                                                               #
    # ------------------------------------------------------------------ #

    async def skills_served_in_sessions(
        self,
        *,
        organization_id: Optional[str],
        user_id: str,
        session_ids: Sequence[str],
    ) -> List[str]:
        """The skills served in ``session_ids`` (per their feedback rows), in
        first-served order."""
        from mirix.services.retry_policy import retry_call

        ids = [sid for sid in dict.fromkeys(session_ids or []) if sid]
        if not ids:
            return []
        stmt = (
            select(SkillSessionFeedbackModel.skill_id)
            .where(
                SkillSessionFeedbackModel.organization_id == organization_id,
                SkillSessionFeedbackModel.user_id == user_id,
                SkillSessionFeedbackModel.session_id.in_(ids),
                SkillSessionFeedbackModel.is_deleted.is_(False),
            )
            .order_by(
                SkillSessionFeedbackModel.created_at,
                SkillSessionFeedbackModel.id,
            )
        )

        async def _once():
            async with self.session_maker() as db:
                return (await db.execute(stmt)).all()

        rows = await retry_call(_once, stage="skill_usage_read")
        return list(dict.fromkeys(row[0] for row in rows))

    async def usage_records(
        self,
        *,
        organization_id: Optional[str],
        user_id: str,
        skill_ids: Sequence[str],
    ) -> Dict[str, SkillUsageRecord]:
        """The usage record of each of ``skill_ids`` (the user's; one for every
        id, empty when it was never served). A transient database failure is
        retried."""
        from mirix.services.retry_policy import retry_call

        ids = [sid for sid in dict.fromkeys(skill_ids or []) if sid]
        records = {sid: SkillUsageRecord(skill_id=sid) for sid in ids}
        if not ids:
            return records
        rows, versions, cited = await retry_call(
            lambda: self._read_usage(
                organization_id=organization_id, user_id=user_id, ids=ids
            ),
            stage="skill_usage_read",
        )
        for row in rows:
            record = records[row.skill_id]
            record.total.add(row)
            record.by_version.setdefault(row.skill_version_no, UsageCounts()).add(row)
            if row.session_id not in record.sessions:
                record.sessions.append(row.session_id)
        for skill_id, version_no in versions:
            if skill_id in records and version_no is not None:
                records[skill_id].current_version = int(version_no)
        sources: Dict[str, set] = {sid: set() for sid in ids}
        for session_id, skill_list in cited:
            for skill_id in skill_list or ():
                if skill_id in sources and session_id:
                    sources[skill_id].add(session_id)
        for skill_id, sessions in sources.items():
            records[skill_id].source_sessions = len(sessions)
        return records

    async def _read_usage(
        self, *, organization_id: Optional[str], user_id: str, ids: List[str]
    ):
        """One read of what :meth:`usage_records` aggregates: the feedback
        rows (newest first), each skill's newest history version, and the
        (session, cited skills) of the owner's experiences a write cited."""
        from mirix.orm.skill_experience import SkillExperience as SkillExperienceModel
        from mirix.orm.skill_version import SkillVersion as SkillVersionModel

        async with self.session_maker() as db:
            rows = (
                (
                    await db.execute(
                        select(SkillSessionFeedbackModel)
                        .where(
                            SkillSessionFeedbackModel.organization_id
                            == organization_id,
                            SkillSessionFeedbackModel.user_id == user_id,
                            SkillSessionFeedbackModel.skill_id.in_(ids),
                            SkillSessionFeedbackModel.is_deleted.is_(False),
                        )
                        .order_by(
                            SkillSessionFeedbackModel.created_at.desc(),
                            SkillSessionFeedbackModel.id.desc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            versions = (
                await db.execute(
                    select(
                        SkillVersionModel.skill_id,
                        func.max(SkillVersionModel.version_no),
                    )
                    .where(SkillVersionModel.skill_id.in_(ids))
                    .group_by(SkillVersionModel.skill_id)
                )
            ).all()
            # Provenance: the experiences a write of the skill cited. Only
            # those whose cited ids (JSON text) mention one of ``ids`` are read
            # — a superset the caller narrows exactly.
            cited_text = cast(SkillExperienceModel.cited_skill_ids, Text)
            cited = (
                await db.execute(
                    select(
                        SkillExperienceModel.session_id,
                        SkillExperienceModel.cited_skill_ids,
                    ).where(
                        SkillExperienceModel.user_id == user_id,
                        SkillExperienceModel.organization_id == organization_id,
                        SkillExperienceModel.cited_skill_ids.is_not(None),
                        SkillExperienceModel.is_deleted.is_(False),
                        or_(*(cited_text.like(f'%"{sid}"%') for sid in ids)),
                    )
                )
            ).all()
        return rows, versions, cited

    # ------------------------------------------------------------------ #
    # Erasure                                                            #
    # ------------------------------------------------------------------ #

    async def delete_by_user_id(self, user_id: str) -> int:
        """Erasure: every feedback row of ``user_id`` (its quotes come from
        the user's transcripts)."""
        async with self.session_maker() as db:
            result = await db.execute(
                delete(SkillSessionFeedbackModel).where(
                    SkillSessionFeedbackModel.user_id == user_id
                )
            )
            await db.commit()
            return int(result.rowcount or 0)

    async def delete_by_client_id(self, actor: PydanticClient) -> int:
        """Erasure: every feedback row a client's learning runs wrote
        (attributed by ``_created_by_id``, like its experiences)."""
        async with self.session_maker() as db:
            result = await db.execute(
                delete(SkillSessionFeedbackModel).where(
                    SkillSessionFeedbackModel._created_by_id == actor.id
                )
            )
            await db.commit()
            return int(result.rowcount or 0)


# --------------------------------------------------------------------------- #
# Which skills a text names (a curator batch's experiences)                    #
# --------------------------------------------------------------------------- #


#: A one-word skill name says which skill a text means only with at least
#: this many letters and digits (a CJK character counting three, as a
#: citation quote's): a shorter one ("search", "a") is an ordinary word any
#: text may hold, so such a skill is found by its id (or lineage) only. A name
#: of two or more words ("check-folder", "Folder Sync") is specific enough.
NAME_MATCH_MIN_LETTERS = 12
_NAME_WORDS = re.compile(r"[^\s_\-]+")


def _specific_name(name: str) -> bool:
    from mirix.services.skill_text_checks import quote_weight

    if len(_NAME_WORDS.findall(name)) >= 2:
        return True
    return quote_weight(name) >= NAME_MATCH_MIN_LETTERS


def skills_named_in(
    texts: Iterable[Optional[str]], skills: Sequence[Tuple[str, str]]
) -> List[str]:
    """The ids of ``skills`` (``(id, name)``) whose id or whole name (case
    ignored; not part of a longer word or hyphenated name; a name only when
    it is specific enough, :data:`NAME_MATCH_MIN_LETTERS`) appears in any of
    ``texts``, in ``skills`` order. Pure."""
    haystack = "\n".join(t for t in texts if isinstance(t, str) and t).lower()
    if not haystack:
        return []
    found: List[str] = []
    for skill_id, name in skills:
        named = name if name and _specific_name(name) else ""
        keys = [k.strip().lower() for k in (skill_id, named) if k and k.strip()]
        for key in keys:
            if re.search(_whole_key_pattern(key), haystack):
                found.append(skill_id)
                break
    return found


def _whole_key_pattern(key: str) -> str:
    """``key`` as a whole word: not part of a longer word or hyphenated
    name — at an end in a CJK character (scripts written without spaces)
    any neighbour will do."""
    from mirix.services.skill_text_checks import _CJK_CHAR

    before = "" if _CJK_CHAR.match(key[0]) else r"(?<![\w-])"
    after = "" if _CJK_CHAR.match(key[-1]) else r"(?![\w-])"
    return before + re.escape(key) + after


# --------------------------------------------------------------------------- #
# Rendering (plain text for the curator)                                       #
# --------------------------------------------------------------------------- #

#: The heading of a usage record, which says what it is and how to take it.
USAGE_RECORD_HEADING = (
    "Usage record (from the sessions distilled so far; MIRIX's own judgement "
    "of each transcript, use it as evidence):"
)


def _slashed(counts: Dict[str, int], labels: Dict[str, str]) -> str:
    return "/".join(f"{labels.get(k, k)} {v}" for k, v in counts.items())


def _outcomes(counts: Dict[str, int]) -> Dict[str, int]:
    """The outcome counts without an empty ``none`` (legacy answers)."""
    return {k: v for k, v in counts.items() if k != "none" or v}


_EFFECT_LABELS = {"no_effect": "no effect"}


def _served_phrase(record: SkillUsageRecord) -> str:
    sessions = record.session_count
    judged = record.total.served
    phrase = f"served in {sessions} distilled session(s)"
    if judged != sessions:
        phrase += f" ({judged} judgements: a reopened session is judged per part)"
    return phrase


def _provenance_phrase(record: SkillUsageRecord, *, distinct: str = "distinct ") -> str:
    """The provenance as what it counts: only a citation the curator quoted the
    applying text of is recorded, so experiences cited without a verified quote
    are not in it ("built from experiences of 0 sessions" would read as a
    skill without any grounding)."""
    return (
        f"cited, with a verified quote, by experiences of {record.source_sessions} "
        f"{distinct}session(s)"
    )


def _counts_phrase(counts: UsageCounts) -> str:
    return (
        f"followed {_slashed(counts.followed, {})}; "
        f"effect {_slashed(counts.effect, _EFFECT_LABELS)}; "
        f"session outcomes {_slashed(_outcomes(counts.outcomes), {})}"
    )


def render_usage_line(record: Optional[SkillUsageRecord]) -> str:
    """One compact line of a skill's usage record (``skill_list`` and a
    batch's header when the full records do not fit)."""
    if record is None:
        return "usage: none recorded"
    built = _provenance_phrase(record, distinct="")
    if not record.total.served:
        return f"usage: not served in any distilled session yet; {built}"
    return f"usage: {_served_phrase(record)}; {_counts_phrase(record.total)}; {built}"


def render_usage_record(record: Optional[SkillUsageRecord]) -> str:
    """A skill's usage record as a few plain lines (``skill_read`` and a
    curator batch's header)."""
    lines = [USAGE_RECORD_HEADING]
    if record is None:
        lines.append("- none recorded")
        return "\n".join(lines)
    total = record.total
    if not total.served:
        lines.append("- not served in any distilled session yet")
    else:
        recent = ", ".join(record.sessions[:RECENT_SESSIONS_SHOWN])
        more = len(record.sessions) - RECENT_SESSIONS_SHOWN
        if more > 0:
            recent += f", … {more} more"
        lines.append(f"- {_served_phrase(record)}; most recent: {recent}")
        lines.append(f"- {_counts_phrase(total)}")
        if len(record.by_version) > 1 or None not in record.by_version:
            for version in sorted(
                record.by_version, key=lambda v: (v is None, -(v or 0))
            ):
                counts = record.by_version[version]
                label = (
                    "version not recorded"
                    if version is None
                    else f"version {version}"
                    + (" (current)" if version == record.current_version else "")
                )
                lines.append(
                    f"- served at {label}: {counts.served} judgement(s); "
                    f"{_counts_phrase(counts)}"
                )
    lines.append(f"- {_provenance_phrase(record)}")
    return "\n".join(lines)


__all__ = [
    "USAGE_RECORD_HEADING",
    "SkillSessionFeedbackManager",
    "SkillUsageRecord",
    "UsageCounts",
    "render_usage_line",
    "render_usage_record",
    "skills_named_in",
]
