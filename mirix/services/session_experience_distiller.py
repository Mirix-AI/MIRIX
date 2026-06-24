"""Goal 2 — General per-session experience distillation.

Given the sealed, not-yet-distilled sessions in the Conversation Message Store,
this distills each session's transcript, IN PARALLEL, into zero or more
transferable :class:`SkillExperience` rows (status ``pending``). A single
session can yield MULTIPLE experiences and a MIX of ``worth_learning`` /
``worth_avoiding`` — there is no external success/failure oracle; the lessons
are derived purely from the conversation content (user critique/confirmation,
tool errors/retries, the agent's own self-corrections, or weak inference).

The transcript source is the Conversation Message Store (the SINGLE source of
truth for skill learning): a dedicated store of external conversation turns with
their REAL ``user`` / ``assistant`` roles preserved, written only when an add
request carried a ``session_id``. The distiller therefore can NEVER see the meta
agent's own memory-management scaffolding (the "[System Message] As the meta
memory manager …" instruction, its trigger/finish_memory_update tool calls and
results, or the continue_chaining control replies) — those never enter this
store. Correctness comes from structure (which table we read), not from a string
heuristic, so the legacy ``_is_mirix_scaffolding`` filter is gone.

Design points (CLAUDE.md async rules):
* Async-only; never starts a nested event loop (the server loop is running).
* Per-session fan-out via ``asyncio.gather`` — each coroutine does ONE transcript
  fetch + ONE LLM call + N inserts, wrapped in try/except so one bad session can
  never crash the whole run.
* The LLM completion reuses MIRIX's async :class:`LLMClient` exactly like
  :class:`SkillSessionDistiller` (``send_llm_request`` → ``choices[0].message``).
* The system prompt is the rewritten general Experience-Distiller prompt at
  ``prompts/system/base/auto_dream_agent/procedural.txt``.

Session enumeration delegates to
``ConversationMessageManager.list_sealed_undistilled_sessions`` (sealed =
a strictly-newer distinct session exists; oldest-first), so the rolling barrier
never distills the in-progress head of the window.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Tuple

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.skill_experience import (
    SKILL_EXPERIENCE_MAX_CONTENT_LEN as _CONTENT_CAP,
    SKILL_EXPERIENCE_MAX_EVIDENCE_LEN as _EVIDENCE_CAP,
    SKILL_EXPERIENCE_MAX_TITLE_LEN as _TITLE_CAP,
    SkillExperience as PydanticSkillExperience,
    _clamp01,
)
from mirix.schemas.user import User as PydanticUser
from mirix.helpers.json_parsing import parse_distiller_json_array
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.skill_experience_manager import SkillExperienceManager

logger = get_logger(__name__)

# Per-transcript char budget. We keep the HEAD and TAIL (where the task framing
# and the final user verdict / confirmation usually live) and elide the middle.
_MAX_TRANSCRIPT_CHARS = 16000

# Per-turn content cap so one giant turn can't dominate the transcript.
_MAX_MESSAGE_CHARS = 2000


def _owner_org(actor: PydanticClient) -> str:
    """Resolve the organization the Conversation Message Store was written under.

    Ingestion records turns under ``client.organization_id or DEFAULT_ORG_ID``
    (the column is nullable, but the write always resolves the fallback). Every
    read/mark of that store MUST mirror it, otherwise a NULL-org client would
    read/persist under ``None`` and its sessions would never distill (and, having
    never been marked, would retry forever).
    """
    from mirix.constants import DEFAULT_ORG_ID

    return actor.organization_id or DEFAULT_ORG_ID


class _LLMCallError(Exception):
    """Raised on an OPERATIONAL LLM failure (no client, failed request, or
    malformed response) — as opposed to a legitimately empty distillation. Lets
    the per-session path mark the session as NOT processed so the barrier does
    not advance and a later round retries it."""


class _DistillFailed(Exception):
    """Raised by ``_distill_one`` when a session could NOT be processed due to an
    operational failure (DB read, LLM call, or persistence). ``distill_sessions``
    catches it to EXCLUDE that session from ``processed_session_ids`` so the
    barrier is not advanced past a failed conversation and a later round retries
    it. Carries the offending ``session_id`` for diagnostics."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(session_id)


def _load_distiller_prompt() -> str:
    """Load the general Experience-Distiller system prompt (rewritten for Goal 2).

    Uses the codebase-standard cached system-prompt loader (the same one the other
    distillers use) instead of a raw ``open()`` in the async path — it resolves to
    prompts/system/base/auto_dream_agent/procedural.txt, the single on-disk source
    of truth shared with the auto_dream procedural mode.
    """
    from mirix.prompts.gpt_system import get_system_text

    return get_system_text("base/auto_dream_agent/procedural")


class SessionExperienceDistiller:
    """Distill the sealed, not-yet-distilled conversation sessions into experiences.

    Args:
        llm_client: an instance exposing ``async send_llm_request(messages)``
            (an ``LLMClientBase``). Injected directly in tests; in production
            pass ``llm_config`` and one is created lazily.
        llm_config: an ``LLMConfig`` used to build the client when ``llm_client``
            is not supplied. Required for a real completion.
        experience_manager: a :class:`SkillExperienceManager`; defaults to a
            fresh instance (constructed lazily so import never touches the DB).
        conversation_manager: a :class:`ConversationMessageManager` — the source
            of session enumeration and per-session transcripts. Defaults to a
            fresh instance (constructed lazily so import never touches the DB).
    """

    def __init__(
        self,
        *,
        llm_client=None,
        llm_config=None,
        experience_manager: Optional[SkillExperienceManager] = None,
        conversation_manager: Optional[ConversationMessageManager] = None,
    ):
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._experience_manager = experience_manager
        self._conversation_manager = conversation_manager
        self._system_prompt = _load_distiller_prompt()

    # ------------------------------------------------------------------ #
    # Session enumeration                                                 #
    # ------------------------------------------------------------------ #

    async def enumerate_sealed_sessions(
        self,
        *,
        user_id: str,
        organization_id: str,
        actor: PydanticClient,
        limit: int,
    ) -> List[str]:
        """Return up to ``limit`` sealed, not-yet-distilled session_ids, OLDEST first.

        Delegates to
        :meth:`ConversationMessageManager.list_sealed_undistilled_sessions`.
        "Sealed" = a strictly-newer distinct session exists, so the open head of
        the window is never returned; only sessions whose turns have NULL
        ``distilled_at`` survive. Scoped per ``(user, organization)``.
        """
        if limit <= 0:
            return []
        return await self._get_conversation_manager().list_sealed_undistilled_sessions(
            user_id=user_id,
            organization_id=organization_id,
            actor=actor,
            limit=limit,
        )

    # ------------------------------------------------------------------ #
    # Per-session distillation (parallel fan-out)                         #
    # ------------------------------------------------------------------ #

    async def distill_sessions(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_ids: List[str],
        existing_skills: Optional[List[Dict]] = None,
    ) -> Tuple[List[PydanticSkillExperience], List[str]]:
        """Distill each session IN PARALLEL into persisted experiences.

        Each session → one coroutine (fetch transcript → one LLM call → parse
        the JSON array → persist each element as a ``pending`` SkillExperience).
        Per-session try/except isolates failures: a single bad session yields an
        empty outcome rather than crashing the whole gather.

        Returns ``(experiences, processed_session_ids)``:
          * ``experiences``          — the flat list of all created experiences.
          * ``processed_session_ids``— the session_ids that were SUCCESSFULLY
            processed (including those that legitimately yielded zero
            experiences). A session that hit an OPERATIONAL failure (bad
            transcript fetch / LLM call / persistence) is OMITTED here, so the
            caller leaves it undistilled and a later round retries it instead of
            silently advancing the barrier past a failed conversation.
        """
        if not session_ids:
            return [], []

        skills_block = self._format_existing_skills(existing_skills or [])
        # return_exceptions=True so one operationally-failed session surfaces as a
        # _DistillFailed in `results` rather than crashing the whole gather; we map
        # exceptions -> "not processed" and any returned list (even []) ->
        # "processed".
        results = await asyncio.gather(
            *[
                self._distill_one(
                    meta_agent_state=meta_agent_state,
                    user=user,
                    actor=actor,
                    session_id=sid,
                    skills_block=skills_block,
                )
                for sid in session_ids
            ],
            return_exceptions=True,
        )
        created: List[PydanticSkillExperience] = []
        processed_session_ids: List[str] = []
        for sid, result in zip(session_ids, results):
            if isinstance(result, BaseException):
                # Operational failure (logged in _distill_one). Leave the session
                # undistilled so a later round retries it.
                continue
            created.extend(result)
            processed_session_ids.append(sid)
        logger.info(
            "Distilled %d session(s) → %d experience(s); %d processed, %d failed",
            len(session_ids),
            len(created),
            len(processed_session_ids),
            len(session_ids) - len(processed_session_ids),
        )
        return created, processed_session_ids

    async def _distill_one(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        skills_block: str,
    ) -> List[PydanticSkillExperience]:
        """Distill ONE session into persisted experiences.

        Returns the (possibly empty) list of created experiences on success — a
        legitimately empty session (no turns / no usable content / the model
        returned ``[]``) is a SUCCESS that returns ``[]``. On an OPERATIONAL
        failure (a failed transcript fetch / LLM call / persistence) it raises
        :class:`_DistillFailed`, which :meth:`distill_sessions` catches to mark
        the session as NOT processed (so its barrier is not advanced and a later
        round retries it). It never lets any OTHER exception escape.
        """
        try:
            turns = await self._get_conversation_manager().list_turns_for_session(
                session_id=session_id,
                user_id=user.id,
                organization_id=_owner_org(actor),
                actor=actor,
            )
            if not turns:
                # Empty session: nothing to learn, but successfully processed.
                return []

            transcript = self._render_transcript(turns)
            if not transcript.strip():
                # No usable content — successfully processed, nothing to learn.
                return []
            parsed = await self._call_llm(
                agent_id=meta_agent_state.id,
                session_id=session_id,
                transcript=transcript,
                skills_block=skills_block,
            )
            if not parsed:
                # The model returned an empty array — a legitimate "nothing worth
                # remembering" result. Processed successfully.
                return []

            return await self._persist_experiences(
                meta_agent_state=meta_agent_state,
                user=user,
                actor=actor,
                session_id=session_id,
                parsed=parsed,
            )
        except _DistillFailed:
            # Already classified (e.g. total persistence failure) — propagate as-is
            # so distill_sessions excludes this session from processed.
            logger.warning(
                "Session experience distill failed for session %s (operational)",
                session_id,
            )
            raise
        except Exception as e:  # noqa: BLE001 — classify as an operational failure
            # DB read, LLM call (raised as _LLMCallError), or another operational
            # error. Re-raise as _DistillFailed so distill_sessions leaves the
            # session undistilled for a later retry instead of advancing the barrier.
            logger.warning(
                "Session experience distill failed for session %s: %s", session_id, e
            )
            raise _DistillFailed(session_id) from e

    async def _persist_experiences(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        parsed: List[Dict],
    ) -> List[PydanticSkillExperience]:
        """Persist each valid parsed item as a pending SkillExperience.

        Per-row try/except means one bad row can't drop the rest. But if there
        were valid rows to persist and EVERY persistence attempt raised (i.e. the
        store is down, not just one malformed row), that is an operational failure
        — raise :class:`_DistillFailed` so the session is NOT marked distilled and
        a later round retries it, rather than silently advancing the barrier past
        a conversation whose experiences never landed. Pure VALIDATION skips
        (bad type / empty title) are not failures and never raise.
        """
        mgr = self._get_experience_manager()
        created: List[PydanticSkillExperience] = []
        attempted = 0  # valid rows we actually tried to persist
        persist_errors = 0
        for item in parsed:
            exp_type = item.get("experience_type")
            if exp_type not in ("worth_learning", "worth_avoiding"):
                # Unknown/missing type → skip (create_experience would also
                # reject it, but skip quietly here so one bad element does not
                # abort the whole session's persistence).
                logger.debug(
                    "Skipping experience with bad type %r in session %s",
                    exp_type,
                    session_id,
                )
                continue
            title = (item.get("title") or "").strip()
            if not title:
                continue
            content = item.get("content") or ""
            importance = _clamp01(item.get("importance"))
            credibility = _clamp01(item.get("credibility"))
            evidence = self._normalize_evidence(item.get("evidence"))
            attempted += 1
            try:
                exp = await mgr.create_experience(
                    agent_id=meta_agent_state.id,
                    user_id=user.id,
                    organization_id=_owner_org(actor),
                    session_id=session_id,
                    experience_type=exp_type,
                    title=title[: _TITLE_CAP],
                    content=content[: _CONTENT_CAP],
                    importance=importance,
                    credibility=credibility,
                    evidence=evidence[: _EVIDENCE_CAP],
                    status="pending",
                )
                created.append(exp)
            except Exception as e:  # noqa: BLE001 — one bad row mustn't drop the rest
                persist_errors += 1
                logger.warning(
                    "Failed to persist experience %r (session %s): %s",
                    title,
                    session_id,
                    e,
                )

        # Total persistence failure: we had valid rows but NONE landed and every
        # attempt errored → the store is down. Signal an operational failure so
        # the session is left undistilled for retry. (A partial success — at least
        # one row landed — returns what persisted; a session with only validation
        # skips has attempted==0 and is a clean empty result.)
        if attempted > 0 and not created and persist_errors == attempted:
            raise _DistillFailed(session_id)
        return created

    # ------------------------------------------------------------------ #
    # LLM plumbing (mirrors SkillSessionDistiller)                        #
    # ------------------------------------------------------------------ #

    async def _call_llm(
        self,
        *,
        agent_id: str,
        session_id: str,
        transcript: str,
        skills_block: str,
    ) -> List[Dict]:
        """Run ONE completion and parse a JSON array of experiences.

        An empty ``[]`` is a legitimate "nothing worth remembering" result and is
        returned normally. An OPERATIONAL failure (no client/config, a failed
        request, or a malformed response) raises ``_LLMCallError`` instead of
        returning ``[]`` — so the caller can tell a transient failure apart from a
        genuinely empty session and NOT advance the barrier on the former.
        """
        client = self._get_client()
        if client is None:
            raise _LLMCallError(
                f"no LLM client/config available for session {session_id}"
            )

        user_payload = (
            f"agent_id: {agent_id}\n"
            f"session_id: {session_id}\n\n"
            f"existing_skills:\n{skills_block}\n\n"
            "session_transcript (chronological):\n"
            f"{transcript}\n\n"
            "Distill the transferable experiences from this session as a JSON array."
        )
        messages = [
            Message(
                agent_id=agent_id,
                role=MessageRole.system,
                content=[TextContent(text=self._system_prompt)],
            ),
            Message(
                agent_id=agent_id,
                role=MessageRole.user,
                content=[TextContent(text=user_payload)],
            ),
        ]
        try:
            response = await client.send_llm_request(messages=messages)
        except Exception as e:  # noqa: BLE001 — surface as an operational failure
            raise _LLMCallError(f"LLM request failed: {e}") from e
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            raise _LLMCallError(f"malformed LLM response: {e}") from e

        parsed = parse_distiller_json_array(content)
        if not parsed and not self._is_explicit_empty_result(content):
            # The shared parser returns [] both for a real empty array AND for
            # unparseable output. A non-empty content that parsed to [] is most
            # likely a TRANSIENT bad completion (truncation, a wrapped error), so
            # we log it for observability. We do NOT raise here on purpose: this
            # barrier has no durable per-session retry counter, so retrying a
            # session the model *consistently* renders unparseable (e.g. it states
            # "no experiences" in prose with no JSON) would peg the rolling window
            # forever. Treating an unparseable empty as "nothing learned" advances
            # the window at the cost of at most one session's learning on a one-off
            # bad completion — the safer tradeoff than a permanently stuck barrier.
            logger.warning(
                "session distiller: non-empty LLM output for session %s parsed to "
                "[] (treating as empty; possible transient bad completion)",
                session_id,
            )
        return parsed

    @staticmethod
    def _is_explicit_empty_result(content) -> bool:
        """True iff `content` legitimately represents "no experiences" rather than
        unparseable garbage.

        A model that means "nothing worth remembering" emits an empty array. We
        treat empty/whitespace OR a content whose only JSON-array token is an empty
        `[]` (optionally fenced / surrounded by prose) as a legitimate empty
        result. Anything else that parsed to `[]` is malformed output, which
        `_call_llm` raises on. Conservative by design: a false "explicit empty"
        only costs one skipped (already-empty) session, never a wrongful retry.
        """
        if content is None:
            return True
        if not isinstance(content, str):
            return False
        text = content.strip()
        if not text:
            return True
        # Strip a single ```json / ``` fence if present, then look at the payload.
        import re as _re

        fence = _re.search(r"```(?:json)?\s*(.*?)```", text, _re.DOTALL | _re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        # An explicit empty array (the only well-formed "nothing" the model emits).
        # `[]`, `[ ]`, or those wrapped in a tiny bit of prose all qualify; a
        # non-empty `[...]` would have parsed to a non-empty list and never reached
        # here, so the only empty-array token we accept is a literal empty pair.
        return _re.fullmatch(r"\[\s*\]", text) is not None

    def _get_client(self):
        if self._llm_client is not None:
            return self._llm_client
        if self._llm_config is not None:
            from mirix.llm_api.llm_client import LLMClient

            self._llm_client = LLMClient.create(
                llm_config=self._llm_config.model_copy(deep=True)
            )
            return self._llm_client
        return None

    def _get_experience_manager(self) -> SkillExperienceManager:
        if self._experience_manager is None:
            self._experience_manager = SkillExperienceManager()
        return self._experience_manager

    def _get_conversation_manager(self) -> ConversationMessageManager:
        if self._conversation_manager is None:
            self._conversation_manager = ConversationMessageManager()
        return self._conversation_manager

    # ------------------------------------------------------------------ #
    # Rendering helpers (pure)                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_existing_skills(existing_skills: List[Dict]) -> str:
        if not existing_skills:
            return "(none)"
        lines = []
        for s in existing_skills:
            name = (s.get("name") or "").strip()
            desc = (s.get("description") or "").strip()
            if not name and not desc:
                continue
            lines.append(f"- {name}: {desc}"[:300])
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _normalize_evidence(evidence) -> str:
        """Coerce the model's `evidence` into a JSON string {quote, signal_type}."""
        valid_signals = {
            "user_critique",
            "user_confirmation",
            "tool_error",
            "self_correction",
            "inferred",
        }
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "")[:512]
            signal = evidence.get("signal_type")
            if signal not in valid_signals:
                signal = "inferred"
            return json.dumps({"quote": quote, "signal_type": signal}, ensure_ascii=False)
        if isinstance(evidence, str) and evidence.strip():
            return json.dumps(
                {"quote": evidence[:512], "signal_type": "inferred"},
                ensure_ascii=False,
            )
        return json.dumps({"quote": "", "signal_type": "inferred"}, ensure_ascii=False)

    @classmethod
    def _render_transcript(cls, turns) -> str:
        """Render conversation turns compactly as ``role: content`` lines.

        The source is the Conversation Message Store, so each turn already has a
        REAL ``role`` ('user' | 'assistant') and a plain-text ``content`` — no
        scaffolding, no tool-call flattening, no embeddings. We simply cap each
        turn and bound the whole transcript by keeping the HEAD and TAIL (task
        framing + final verdict) and eliding the middle.
        """
        lines: List[str] = []
        for t in turns:
            role = cls._role_of(t)
            text = cls._content_text(t)
            if not text:
                continue
            if len(text) > _MAX_MESSAGE_CHARS:
                text = text[:_MAX_MESSAGE_CHARS] + " …[truncated]"
            lines.append(f"{role}: {text}")
        transcript = "\n".join(lines)
        if len(transcript) <= _MAX_TRANSCRIPT_CHARS:
            return transcript
        head = transcript[: _MAX_TRANSCRIPT_CHARS // 2]
        tail = transcript[-(_MAX_TRANSCRIPT_CHARS // 2):]
        return f"{head}\n…[elided middle of session]…\n{tail}"

    @staticmethod
    def _role_of(t) -> str:
        """The real turn role of a ConversationMessage ('user' | 'assistant')."""
        role = getattr(t, "role", None)
        return getattr(role, "value", None) or str(role) or "unknown"

    @staticmethod
    def _content_text(t) -> str:
        """Plain text of a ConversationMessage turn (content is already a str)."""
        content = getattr(t, "content", None)
        if isinstance(content, str):
            return content.strip()
        return ""
