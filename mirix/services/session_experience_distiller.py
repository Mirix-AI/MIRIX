"""Goal 2 — General per-session experience distillation.

Given the meta agent's last-N RETAINED sessions (Goal-1 keeps their raw
messages), this distills each session's transcript, IN PARALLEL, into zero or
more transferable :class:`SkillExperience` rows (status ``pending``). A single
session can yield MULTIPLE experiences and a MIX of ``worth_learning`` /
``worth_avoiding`` — there is no external success/failure oracle; the lessons
are derived purely from the conversation content (user critique/confirmation,
tool errors/retries, the agent's own self-corrections, or weak inference).

Design points (CLAUDE.md async rules):
* Async-only; never starts a nested event loop (the server loop is running).
* Per-session fan-out via ``asyncio.gather`` — each coroutine does ONE transcript
  fetch + ONE LLM call + N inserts, wrapped in try/except so one bad session can
  never crash the whole run.
* The LLM completion reuses MIRIX's async :class:`LLMClient` exactly like
  :class:`SkillSessionDistiller` (``send_llm_request`` → ``choices[0].message``).
* The system prompt is the rewritten general Experience-Distiller prompt at
  ``prompts/system/base/auto_dream_agent/procedural.txt``.

Session enumeration mirrors ``AgentTriggerStateManager._aggregate_window``:
``GROUP BY session_id`` with ``MIN(created_at)`` per session, ordered MIN DESC,
take the most-recent N — these are the same sessions Goal-1 retained.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Optional

from sqlalchemy import func, select

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
from mirix.services.skill_experience_manager import SkillExperienceManager
from mirix.constants import META_MEMORY_TOOLS, UNIVERSAL_MEMORY_TOOLS

logger = get_logger(__name__)

# How many transcript messages to pull per session (high — we want the whole
# retained session, but bounded so a runaway session can't blow up the prompt).
_MAX_MESSAGES_PER_SESSION = 400

# Per-transcript char budget. We keep the HEAD and TAIL (where the task framing
# and the final user verdict / confirmation usually live) and elide the middle.
_MAX_TRANSCRIPT_CHARS = 16000

# Per-message content cap so one giant tool dump can't dominate the transcript.
_MAX_MESSAGE_CHARS = 2000

# MIRIX's OWN memory-management scaffolding must NEVER be distilled as if it were
# the external conversation. The meta agent's session transcript comingles the
# ingested external turns with the meta agent's own memory-update actions: the
# "[System Message] As the meta memory manager …" instruction, its
# trigger_memory_update / finish_memory_update tool calls, those tools' results,
# and the automated continue/finish-chaining replies. Learning from the latter
# yields meta-noise experiences like "trigger episodic memory update when the
# system requests meta memory management" — i.e. the distiller learning to operate
# MIRIX itself instead of the task. We drop those in _render_transcript (see
# _is_mirix_scaffolding). Ingested external content is always [USER]/[ASSISTANT]-
# tagged (the /memory/add role-collapse); MIRIX scaffolding never is.
# Sourced from the agent tool registry (not hardcoded) so the filter tracks the
# meta agent's tools as they change: trigger_memory_update + search_in_memory +
# finish_memory_update + list_memory_within_timerange. The meta agent operates
# ONLY these memory tools — it never makes external task tool calls — so a tool
# call to ANY of them marks the message as MIRIX scaffolding.
_MIRIX_MEMORY_TOOLS = frozenset(META_MEMORY_TOOLS) | frozenset(UNIVERSAL_MEMORY_TOOLS)
_META_INSTRUCTION_MARKER = "As the meta memory manager, analyze the provided content"
_CHAINING_MARKERS = (
    '"contine_chaining"',
    '"continue_chaining"',
    "automated system message hidden from the user",
)


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
    """Distill the meta agent's last-N retained sessions into experiences.

    Args:
        llm_client: an instance exposing ``async send_llm_request(messages)``
            (an ``LLMClientBase``). Injected directly in tests; in production
            pass ``llm_config`` and one is created lazily.
        llm_config: an ``LLMConfig`` used to build the client when ``llm_client``
            is not supplied. Required for a real completion.
        experience_manager: a :class:`SkillExperienceManager`; defaults to a
            fresh instance (constructed lazily so import never touches the DB).
    """

    def __init__(
        self,
        *,
        llm_client=None,
        llm_config=None,
        experience_manager: Optional[SkillExperienceManager] = None,
    ):
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._experience_manager = experience_manager
        self._system_prompt = _load_distiller_prompt()

    # ------------------------------------------------------------------ #
    # Session enumeration                                                 #
    # ------------------------------------------------------------------ #

    async def enumerate_last_n_sessions(
        self,
        *,
        agent_id: str,
        user_id: Optional[str],
        n: int,
    ) -> List[str]:
        """Return the most-recent ``n`` session_ids for (agent[, user]).

        Mirrors ``AgentTriggerStateManager._aggregate_window``: ``GROUP BY
        session_id`` with ``MIN(created_at)`` per session, ordered MIN DESC,
        skipping NULL session_ids and soft-deleted rows. Returns at most ``n``
        session_ids, ordered most-recent-first.
        """
        if n <= 0:
            return []
        from mirix.orm.message import Message as MessageModel
        from mirix.server.server import db_context

        per_session_min = func.min(MessageModel.created_at).label("first_ts")
        preds = [
            MessageModel.agent_id == agent_id,
            MessageModel.session_id.isnot(None),
            MessageModel.is_deleted.is_(False),
        ]
        if user_id is not None:
            preds.append(MessageModel.user_id == user_id)

        stmt = (
            select(MessageModel.session_id, per_session_min)
            .where(*preds)
            .group_by(MessageModel.session_id)
            .order_by(per_session_min.desc())
            .limit(n)
        )
        async with db_context() as session:
            result = await session.execute(stmt)
            rows = result.all()
        return [row.session_id for row in rows if row.session_id is not None]

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
    ) -> List[PydanticSkillExperience]:
        """Distill each session IN PARALLEL into persisted experiences.

        Each session → one coroutine (fetch transcript → one LLM call → parse
        the JSON array → persist each element as a ``pending`` SkillExperience).
        Per-session try/except isolates failures: a single bad session yields an
        empty list rather than crashing the whole gather. Returns the flat list
        of all created experiences across sessions.
        """
        if not session_ids:
            return []

        skills_block = self._format_existing_skills(existing_skills or [])
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
            return_exceptions=False,  # _distill_one never raises (own try/except)
        )
        created: List[PydanticSkillExperience] = []
        for items in results:
            created.extend(items)
        logger.info(
            "Distilled %d session(s) → %d experience(s)", len(session_ids), len(created)
        )
        return created

    async def _distill_one(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        skills_block: str,
    ) -> List[PydanticSkillExperience]:
        """Distill ONE session; never raises (returns [] on any failure)."""
        try:
            from mirix.services.message_manager import MessageManager

            msgs = await MessageManager().list_messages_for_agent(
                agent_id=meta_agent_state.id,
                session_id=session_id,
                ascending=True,
                actor=actor,
                limit=_MAX_MESSAGES_PER_SESSION,
                use_cache=False,
            )
            if not msgs:
                return []

            transcript = self._render_transcript(msgs)
            if not transcript.strip():
                # Whole session was MIRIX scaffolding (no ingested external
                # content) — nothing to learn; skip the LLM call entirely.
                return []
            parsed = await self._call_llm(
                agent_id=meta_agent_state.id,
                session_id=session_id,
                transcript=transcript,
                skills_block=skills_block,
            )
            if not parsed:
                return []

            return await self._persist_experiences(
                meta_agent_state=meta_agent_state,
                user=user,
                actor=actor,
                session_id=session_id,
                parsed=parsed,
            )
        except Exception as e:  # noqa: BLE001 — isolate per-session failures
            logger.warning(
                "Session experience distill failed for session %s: %s", session_id, e
            )
            return []

    async def _persist_experiences(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        parsed: List[Dict],
    ) -> List[PydanticSkillExperience]:
        mgr = self._get_experience_manager()
        created: List[PydanticSkillExperience] = []
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
            try:
                exp = await mgr.create_experience(
                    agent_id=meta_agent_state.id,
                    user_id=user.id,
                    organization_id=actor.organization_id,
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
                logger.warning(
                    "Failed to persist experience %r (session %s): %s",
                    title,
                    session_id,
                    e,
                )
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
        client = self._get_client()
        if client is None:
            logger.warning(
                "session distiller: no LLM client/config available; skipping session %s",
                session_id,
            )
            return []

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
        except Exception as e:  # noqa: BLE001 — never crash the per-session path
            logger.warning("session distiller: LLM request failed: %s", e)
            return []
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return []
        return parse_distiller_json_array(content)

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
    def _render_transcript(cls, msgs) -> str:
        """Render messages compactly as ``role: content`` lines.

        Drops embeddings/heavy fields; flattens tool calls/returns to readable
        text; caps each message; and bounds the whole transcript by keeping the
        HEAD and TAIL (task framing + final verdict) and eliding the middle.
        """
        lines: List[str] = []
        for m in msgs:
            # Never learn from MIRIX's own memory-management scaffolding — only
            # from the ingested external conversation (see _is_mirix_scaffolding).
            if cls._is_mirix_scaffolding(m):
                continue
            role = cls._role_of(m)
            text = cls._content_text(m)
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
    def _plain_content_text(m) -> str:
        """Plain text of a message's content (no tool-call surfacing)."""
        content = getattr(m, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = []
            for c in content:
                txt = getattr(c, "text", None)
                if txt and isinstance(txt, str):
                    out.append(txt)
            return " ".join(out)
        return ""

    @classmethod
    def _is_mirix_scaffolding(cls, m) -> bool:
        """True iff this message is MIRIX's OWN memory-management process output
        rather than the ingested external conversation.

        The distiller must learn ONLY from the external user/agent/tool messages
        fed INTO MIRIX, never from the memory-producing (meta) agent's own
        scaffolding — otherwise it distills meta-noise about operating MIRIX
        itself. Three shapes of scaffolding appear on the meta agent's transcript:
          1. memory-subagent tool RESULTS (role=tool from a memory-mgmt tool),
          2. the meta agent's OWN tool CALLS (trigger/finish_memory_update),
          3. MIRIX control messages (the meta-manager instruction + the automated
             continue/finish-chaining replies).
        Ingested external content is always [USER]/[ASSISTANT]-tagged and matches
        none of these.
        """
        role = getattr(m, "role", None)
        name = getattr(m, "name", None)
        # (1) Memory tool RESULTS.
        if role == MessageRole.tool and name in _MIRIX_MEMORY_TOOLS:
            return True
        # (2) The meta agent's own memory tool CALLS. The meta agent operates ONLY
        #     memory tools and never makes external task tool calls (external tool
        #     output is collapsed into [USER]/[ASSISTANT] text, never assistant
        #     tool_calls on this transcript), so ANY memory tool_call — even mixed
        #     with another memory tool like search_in_memory — is scaffolding.
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                n = getattr(getattr(tc, "function", None), "name", None)
                if n in _MIRIX_MEMORY_TOOLS:
                    return True
        # (3) Synthesized control messages (the meta-manager instruction + the
        #     automated chaining replies). Gate on NON-external text: ingested
        #     external content is always [USER]/[ASSISTANT]-tagged, so a turn that
        #     merely QUOTES one of these markers is never dropped.
        text = cls._plain_content_text(m)
        if text and "[USER]" not in text and "[ASSISTANT]" not in text:
            if text.lstrip().startswith("[System Message]"):
                return True
            if _META_INSTRUCTION_MARKER in text:
                return True
            if any(marker in text for marker in _CHAINING_MARKERS):
                return True
        return False

    @staticmethod
    def _role_of(m) -> str:
        role = getattr(m, "role", None)
        return getattr(role, "value", None) or str(role) or "unknown"

    @staticmethod
    def _content_text(m) -> str:
        """Extract readable text from a PydanticMessage, including tool calls."""
        parts: List[str] = []

        content = getattr(m, "content", None)
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for c in content:
                txt = getattr(c, "text", None)
                if txt and isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())

        # Tool calls (assistant invoking a tool) — surface name + args so a
        # tool error / retry pattern is visible to the distiller.
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                args = getattr(fn, "arguments", None) if fn else None
                if name:
                    snippet = f"[tool_call {name}]"
                    if args:
                        snippet += f" {str(args)[:400]}"
                    parts.append(snippet)

        # Tool return content lives in `content` for role=tool; the loop above
        # already captured it. Also surface a tool name when present.
        name = getattr(m, "name", None)
        if name and getattr(m, "role", None) == MessageRole.tool:
            parts.append(f"[tool_result {name}]")

        return " ".join(p for p in parts if p).strip()
