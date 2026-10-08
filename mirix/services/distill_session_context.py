"""What the distiller is told about a session beside its transcript.

Two things a transcript does not show, both optional:

* **the skills MIRIX served to the agent in the session** — every retrieval of
  the session is logged (``retrieval_log``, :mod:`retrieval_log_manager`) with
  the skills it returned and the version each was served at
  (``per_type.procedural.returned_versions``); the text of that version is
  rebuilt from the skill's history. A skill served before it had a history
  (logged with no version) shows the text it was served with: its current
  text while it still has none, else its ``baseline`` version 1, which its
  first recorded change saved. A log row written before versions were
  logged, and a version that no longer rebuilds, show the current text,
  saying so;
* **how the agent's prompt was built** — the ``agent_context`` the client
  attached to the session (``conversation_session.agent_context``), verbatim.

:class:`SessionContextLoader` reads both for the rows a learning run distills:
retrievals logged at or before the run's watermark and — for a reopened
session, whose earlier rows were distilled already — after the newest of
those rows, so a segment is never shown the skills served for another one.
Only skills of the session's owner in the scopes the run's client may read
are shown: a retrieval by another client never reveals another scope's skill.

The texts are untrusted data (a skill's text was written by a model from
user- and tool-controlled transcripts; the agent context comes from the
client): :mod:`distill_transcript` puts each in a delimited block that
nothing inside can close, and the distiller is told they are context, never
instructions and never evidence (quotes are checked against the
conversation's messages only).

All I/O is async.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sqlalchemy import func, select

from mirix.log import get_logger

logger = get_logger(__name__)


#: How a served skill's text relates to what was served
#: (``ServedSkill.version_status``).
VERSION_EXACT = "exact"  # the served version, rebuilt from history
VERSION_UNVERSIONED = "unversioned"  # served before its history: as served
VERSION_UNKNOWN = "unknown"  # the version served was not logged: current text
VERSION_UNREBUILT = "unrebuilt"  # the served version does not rebuild: current text
VERSION_STATUSES = (
    VERSION_EXACT,
    VERSION_UNVERSIONED,
    VERSION_UNKNOWN,
    VERSION_UNREBUILT,
)


@dataclass(frozen=True)
class ServedSkill:
    """One skill MIRIX served to the agent in a session, as it was served."""

    skill_id: str
    name: str
    description: str
    instructions: str
    #: The history version served (None: unknown, or the skill had no
    #: history when it was served and still has none).
    version_no: Optional[int] = None
    #: How many of the session's retrievals served it.
    served_count: int = 1
    #: Other versions of it an earlier retrieval of the session served.
    other_versions: Tuple[int, ...] = ()
    #: The skill was deleted since it was served.
    deleted: bool = False
    #: Its triggers and examples as served (the agent was shown them too).
    triggers: Tuple[str, ...] = ()
    examples: Tuple[Any, ...] = ()
    #: How the text relates to what was served (:data:`VERSION_STATUSES`;
    #: None: ``exact`` when ``version_no`` is given, else ``unknown``).
    version_status: Optional[str] = None

    @property
    def status(self) -> str:
        if self.version_status in VERSION_STATUSES:
            return self.version_status
        return VERSION_EXACT if self.version_no is not None else VERSION_UNKNOWN

    @property
    def exact(self) -> bool:
        """The text shown is the text served."""
        return self.status in (VERSION_EXACT, VERSION_UNVERSIONED)


@dataclass(frozen=True)
class SessionContext:
    """What the distiller is told about a session beside its transcript."""

    served: Tuple[ServedSkill, ...] = ()
    agent_context: Optional[str] = None
    #: Served skill ids not shown: gone (hard deleted), another owner's, or
    #: outside the scopes the run's client may read.
    served_hidden: int = 0
    #: Retrieval log rows of the session that were read.
    retrievals: int = 0

    @property
    def empty(self) -> bool:
        return not self.served and not self.agent_context


EMPTY_CONTEXT = SessionContext()


class ServedTrace(NamedTuple):
    """One skill a session's retrievals served (:func:`served_from_traces`)."""

    skill_id: str
    #: The version its latest retrieval served (None: not logged, or none).
    version_no: Optional[int]
    served_count: int
    other_versions: Tuple[int, ...]
    #: Its latest retrieval logged that it had no history yet (a version
    #: logged as None) — unlike a row written before versions were logged.
    unversioned: bool = False


def served_from_traces(traces: Sequence[Mapping[str, Any]]) -> List[ServedTrace]:
    """The skills the procedural traces of a session's retrievals served,
    oldest first: each in the order first served, at the version its latest
    retrieval served (None when that retrieval logged none: a row written
    before versions were logged — or, ``unversioned``, a skill logged with
    no version because it had no history yet), how many retrievals served
    it, and the other versions earlier ones served. Pure."""
    order: List[str] = []
    latest: Dict[str, Optional[int]] = {}
    unversioned: Dict[str, bool] = {}
    counts: Dict[str, int] = {}
    seen_versions: Dict[str, List[int]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        returned = trace.get("returned_ids")
        if not isinstance(returned, list):
            continue
        versions = trace.get("returned_versions")
        logged = isinstance(versions, Mapping)
        versions = versions if logged else {}
        for skill_id in returned:
            if not isinstance(skill_id, str) or not skill_id:
                continue
            raw = versions.get(skill_id)
            version = (
                int(raw)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                else None
            )
            if skill_id not in counts:
                order.append(skill_id)
                counts[skill_id] = 0
            counts[skill_id] += 1
            latest[skill_id] = version
            # Logged, as None: the skill had no history when it was served.
            unversioned[skill_id] = (
                version is None and logged and skill_id in versions and raw is None
            )
            if version is not None and version not in seen_versions.setdefault(
                skill_id, []
            ):
                seen_versions[skill_id].append(version)
    out = []
    for skill_id in order:
        version = latest[skill_id]
        others = tuple(v for v in seen_versions.get(skill_id, []) if v != version)
        out.append(
            ServedTrace(
                skill_id, version, counts[skill_id], others, unversioned[skill_id]
            )
        )
    return out


def _as_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


class SessionContextLoader:
    """Reads a session's :class:`SessionContext` (see the module docstring)."""

    def __init__(self, *, session_maker=None, procedural_manager=None):
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        self._procedural_manager = procedural_manager

    def _procedural(self):
        if self._procedural_manager is None:
            from mirix.services.procedural_memory_manager import (
                ProceduralMemoryManager,
            )

            self._procedural_manager = ProceduralMemoryManager(
                session_maker=self.session_maker
            )
        return self._procedural_manager

    async def load(
        self,
        *,
        organization_id: Optional[str],
        user_id: str,
        session_id: str,
        watermark: Optional[Any],
        scopes: Optional[List[str]],
    ) -> SessionContext:
        """The session's served skills and agent context for the rows a
        learning run distills (at or before ``watermark``); ``scopes`` are
        the scopes whose skills may be shown (the run's client's read
        scopes; [] shows none)."""
        from mirix.orm.conversation_message import (
            ConversationMessage as ConversationMessageModel,
        )
        from mirix.orm.conversation_session import (
            ConversationSession as ConversationSessionModel,
        )
        from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogModel
        from mirix.services.conversation_message_manager import parse_watermark
        from mirix.services.conversation_session_manager import as_utc

        upper: Optional[dt.datetime] = (
            parse_watermark(watermark) if watermark is not None else None
        )
        async with self.session_maker() as db:
            agent_context = (
                await db.execute(
                    select(ConversationSessionModel.agent_context).where(
                        ConversationSessionModel.organization_id == organization_id,
                        ConversationSessionModel.user_id == user_id,
                        ConversationSessionModel.session_id == session_id,
                    )
                )
            ).scalar_one_or_none()
            # A reopened session: the retrievals of its earlier, already
            # distilled segment are not this segment's.
            lower = as_utc(
                (
                    await db.execute(
                        select(func.max(ConversationMessageModel.created_at)).where(
                            ConversationMessageModel.organization_id == organization_id,
                            ConversationMessageModel.user_id == user_id,
                            ConversationMessageModel.session_id == session_id,
                            ConversationMessageModel.distilled_at.is_not(None),
                            ConversationMessageModel.is_deleted.is_(False),
                        )
                    )
                ).scalar_one_or_none()
            )
            preds = [
                RetrievalLogModel.organization_id == organization_id,
                RetrievalLogModel.user_id == user_id,
                RetrievalLogModel.session_id == session_id,
                RetrievalLogModel.is_deleted.is_(False),
            ]
            if upper is not None:
                preds.append(RetrievalLogModel.created_at <= upper)
            if lower is not None:
                preds.append(RetrievalLogModel.created_at > lower)
            rows = (
                await db.execute(
                    select(RetrievalLogModel.per_type)
                    .where(*preds)
                    .order_by(RetrievalLogModel.created_at, RetrievalLogModel.id)
                )
            ).all()
        traces = [
            row[0].get("procedural")
            for row in rows
            if isinstance(row[0], dict) and isinstance(row[0].get("procedural"), dict)
        ]
        served = served_from_traces(traces)
        texts: Dict[str, Dict[str, Any]] = {}
        if served:
            texts = await self._procedural().get_served_skill_texts(
                organization_id=organization_id,
                user_id=user_id,
                served={trace.skill_id: trace.version_no for trace in served},
                scopes=list(scopes) if scopes is not None else None,
                unversioned=[trace.skill_id for trace in served if trace.unversioned],
            )
        skills: List[ServedSkill] = []
        for trace in served:
            text = texts.get(trace.skill_id)
            if text is None:
                continue
            status = text.get("version_status")
            skills.append(
                ServedSkill(
                    skill_id=trace.skill_id,
                    name=str(text.get("name") or ""),
                    description=str(text.get("description") or ""),
                    instructions=str(text.get("instructions") or ""),
                    version_no=text.get("version_no"),
                    served_count=trace.served_count,
                    other_versions=tuple(trace.other_versions),
                    deleted=bool(text.get("is_deleted")),
                    triggers=_as_tuple(text.get("triggers")),
                    examples=_as_tuple(text.get("examples")),
                    version_status=(
                        status if status in VERSION_STATUSES else VERSION_UNKNOWN
                    ),
                )
            )
        context = agent_context if isinstance(agent_context, str) else None
        return SessionContext(
            served=tuple(skills),
            agent_context=context if context and context.strip() else None,
            served_hidden=len(served) - len(skills),
            retrievals=len(rows),
        )


__all__ = [
    "EMPTY_CONTEXT",
    "VERSION_EXACT",
    "VERSION_STATUSES",
    "VERSION_UNKNOWN",
    "VERSION_UNREBUILT",
    "VERSION_UNVERSIONED",
    "ServedSkill",
    "ServedTrace",
    "SessionContext",
    "SessionContextLoader",
    "served_from_traces",
]
