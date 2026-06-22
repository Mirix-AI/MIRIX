"""C1 — Per-round experience distiller.

Distills exactly ONE completed MetaClaw round into a single success/failure
record, persisted via the C2 :class:`SkillEvolutionRecordManager`. It is the
narrow waist between per-round ingestion (C5) and the every-N-rounds curator
(C3).

Two design invariants are enforced here:

* **Leakage filter (§3 guard #1).** :func:`sanitize_turn` is a pure, LLM-free
  function that takes ONLY the buffered ``{prompt_text, response_text}`` turn and
  surfaces ONLY ``{question, answer, prev_feedback}``. It can never accept or emit
  an oracle-derived field (``inline_score``, ``reward``, ``eval.*``,
  ``feedback.options`` …). The round's pass/fail signal comes solely from the
  round-(t-1) ``[Previous Feedback]`` block already embedded verbatim in
  ``prompt_text`` — never from a benchmark score. We deliberately do NOT reuse the
  legacy evolver adapter's prompt-tail truncation (it can drop the leading
  feedback, which is the only legitimate outcome source); when a prompt must be
  bounded we keep the head and elide the MIDDLE.

* **One-round lag (§3 guard #3).** Round t's outcome is only known when round t+1
  arrives, because t+1's prompt carries t's correctness feedback. So
  :meth:`SkillSessionDistiller.distill_round` BUFFERS the incoming round and
  distills the PREVIOUS buffered round using the new round's feedback prefix as
  the pass/fail source. The last round of a session has no successor, so
  :meth:`flush_session` drops it (no graded record) by default.

Async-only and never starts a nested event loop (the server loop is already
running), per CLAUDE.md. The LLM completion reuses MIRIX's async
:class:`LLMClient` exactly like ``memory.summarize_messages``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional

from mirix.log import get_logger
from mirix.prompts.gpt_system import get_system_text
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent

logger = get_logger(__name__)

# The exact MetaClaw injection marker (benchmark/src/infer/prompts.py).
_FEEDBACK_MARKER = "[Previous Feedback]"

# When a prompt must be bounded, keep this much of the HEAD verbatim (the head is
# where the legitimate [Previous Feedback] + question live) and elide the rest.
DEFAULT_MAX_PROMPT_CHARS = 6000

# Substrings that, when present in a feedback string, mark the round as a FAILURE.
# These mirror the dataset/harness-authored correction templates
# (benchmark/src/infer/prompts.py): per-option corrections, the missing-\bbox
# format error, and the continue-reminder suffix appended to every incorrect
# feedback (file_check + multi_choice). We match the SIGNAL of correction, never
# the oracle answer set itself.
_FAILURE_MARKERS = (
    "you missed option",
    "you incorrectly selected option",
    "did not include a \\bbox",
    "keep this in mind as you continue with the next task",
)

# The system prompt key resolved by get_system_text -> prompts/system/base/...
_DISTILLER_PROMPT_KEY = "base/session_distiller"


def sanitize_turn(turn: Dict, *, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> Dict:
    """Leakage filter (§3 guard #1): the ONLY way a round's text reaches the LLM.

    Input is the clean buffered turn ``{prompt_text, response_text}`` (the
    role-flattened message list from ``api_server.py``). Any other keys on the
    dict — including oracle-derived ones (``inline_score``, ``reward``, ``eval``,
    ``feedback`` …) — are IGNORED and never surfaced.

    Returns a dict with EXACTLY the allow-listed keys::

        {"question": str, "answer": str, "prev_feedback": str}

    Where:
      * ``prev_feedback`` is the round-(t-1) correctness feedback embedded in the
        ``[Previous Feedback] … `` head of ``prompt_text`` (empty for round 1).
      * ``question`` is this round's task text (the prompt head with the feedback
        block and the ``role:`` prefix stripped).
      * ``answer`` is the agent's own ``response_text``.

    The head of ``prompt_text`` is always preserved; only the MIDDLE/TAIL is
    bounded when the prompt exceeds ``max_chars`` (never the leading feedback —
    that is the sole legitimate pass/fail source).
    """
    if not isinstance(turn, dict):
        return {"question": "", "answer": "", "prev_feedback": ""}

    prompt_text = turn.get("prompt_text") or ""
    answer = turn.get("response_text") or ""

    if not isinstance(prompt_text, str):
        prompt_text = ""
    if not isinstance(answer, str):
        answer = ""

    # Bound the prompt by keeping the HEAD (where feedback + question live) and
    # eliding the middle/tail. Never tail-truncate (that drops the feedback).
    if len(prompt_text) > max_chars:
        prompt_text = prompt_text[:max_chars] + "\n…[truncated middle/tail]…"

    prev_feedback, question = _split_feedback_and_question(prompt_text)

    # The output is constructed field-by-field from the allow-list ONLY, so it is
    # structurally impossible for a forbidden input key to appear in the result.
    return {
        "question": question.strip(),
        "answer": answer.strip(),
        "prev_feedback": prev_feedback.strip(),
    }


# A flattened "<role>:" turn boundary in the buffered prompt_text. api_server.py
# flattens each message as "{role}: {content}" with role drawn from the chat
# roles, so we match ONLY those known roles (not an arbitrary "Word:" line, which
# could appear inside a question's option list). The current round's injected
# user message is the FIRST turn; everything from the next such boundary onward
# is accumulated history that must NOT contaminate the single-round question.
_KNOWN_ROLES = ("user", "assistant", "system", "tool", "function", "developer")
_NEXT_TURN_BOUNDARY = re.compile(
    r"\n(?:" + "|".join(_KNOWN_ROLES) + r"):\s", re.IGNORECASE
)


def _strip_role_prefix(text: str) -> str:
    """Drop a leading flattened ``role:`` prefix (e.g. ``user: ``) if present."""
    # The buffered prompt_text is "<role>: <content>\n<role>: <content>…".
    # We only care about the FIRST turn's content, which holds the head.
    head_line = text.lstrip()
    m = re.match(r"^[A-Za-z_]+:\s", head_line)
    if m:
        return head_line[m.end() :]
    return head_line


def _trim_to_first_turn(text: str) -> str:
    """Keep only the FIRST flattened turn — drop accumulated history that follows.

    The buffered ``prompt_text`` flattens the whole message list, so after the
    round's injected user message come prior assistant/user/tool turns. Those are
    accumulated history (not this round's question) and must not be sent to the
    distiller LLM. Cut at the next ``\\n<role>: `` boundary.
    """
    m = _NEXT_TURN_BOUNDARY.search(text)
    return text[: m.start()] if m else text


def _split_feedback_and_question(prompt_text: str) -> tuple[str, str]:
    """Split the prompt head into (prev_feedback, question).

    The MetaClaw head is ``[Previous Feedback] <feedback>\\n\\n<question>`` for
    rounds > 1, or just ``<question>`` for round 1. The feedback may itself span
    multiple lines (it does in the dataset), so we split on the FIRST blank line
    after the marker, which the harness inserts between feedback and question
    (``with_feedback`` joins them with ``\\n\\n``).

    The injected user message is the FIRST flattened turn; we trim BOTH the
    feedback and the question to that first turn so accumulated history (later
    ``role:`` turns) never leaks into either field.
    """
    # Only the first flattened turn is this round's injected message; the rest is
    # accumulated openclaw history. Trim before doing anything else.
    head = _strip_role_prefix(_trim_to_first_turn(prompt_text.lstrip()))

    marker_pos = head.find(_FEEDBACK_MARKER)
    if marker_pos == -1:
        # Round 1: no feedback prefix. The (already-trimmed) head is the question.
        return "", head

    # Drop ONLY the marker + a single following space, NOT the rest of the leading
    # whitespace: the "\n\n" that separates feedback from question must survive so
    # the split below works even when the feedback is empty.
    after_marker = head[marker_pos + len(_FEEDBACK_MARKER) :]
    if after_marker.startswith(" "):
        after_marker = after_marker[1:]
    # Feedback and question are joined by a blank line ("\n\n"). Split on the
    # first blank line; everything before is feedback, after is the question.
    parts = re.split(r"\n[ \t]*\n", after_marker, maxsplit=1)
    if len(parts) == 2:
        feedback, question = parts[0], parts[1]
    else:
        # No blank-line separator found; treat it all as feedback (defensive —
        # the question head is unrecoverable, so we keep the legitimate feedback).
        feedback, question = parts[0], ""
    return feedback.strip(), question.strip()


def derive_record_type(prev_feedback: Optional[str]) -> Optional[str]:
    """Derive the round's outcome from its correctness-feedback string.

    Returns ``"failure"`` if the feedback carries any correction marker,
    ``"success"`` if it is a non-empty congratulatory string, or ``None`` if there
    is no feedback (outcome unknown -> no graded record).

    This is the ONLY pass/fail source (§2 PINNED): never ``inline_score`` /
    ``reward`` / ``eval``. We match the SIGNAL of correction (per-option
    correction lines, the missing-\\bbox format error, the continue-reminder
    suffix appended to every incorrect feedback), not the oracle answer set.
    """
    if not prev_feedback:
        return None
    lowered = prev_feedback.lower()
    if any(marker in lowered for marker in _FAILURE_MARKERS):
        return "failure"
    return "success"


def parse_distiller_json(raw: Optional[str]) -> Optional[Dict]:
    """Robustly parse the distiller LLM reply into a dict.

    Tolerates: a bare JSON object, a ```json fenced block, a plain ``` fenced
    block, or a single object embedded in surrounding prose. Returns ``None`` if
    no JSON object can be recovered.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    # 1) Strip a fenced code block if present (```json … ``` or ``` … ```).
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        parsed = _try_load_object(candidate)
        if parsed is not None:
            return parsed

    # 2) Try the whole text as JSON.
    parsed = _try_load_object(text)
    if parsed is not None:
        return parsed

    # 3) Last resort: try each balanced {...} span until one parses. An earlier
    # non-JSON brace pair in surrounding prose must not block a later real object.
    for span in _balanced_objects(text):
        parsed = _try_load_object(span)
        if parsed is not None:
            return parsed

    return None


def _try_load_object(text: str) -> Optional[Dict]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _balanced_objects(text: str):
    """Yield each top-level balanced ``{...}`` substring (respecting strings)."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
        # Advance past this opening brace and keep scanning for the next object.
        i += 1


def _clamp01(value, default: float = 0.0) -> float:
    """Coerce a distiller-reported score into [0.0, 1.0]; default on garbage."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


class SkillSessionDistiller:
    """Distill one graded round into a success/failure record, lagged one round.

    State: a per-(agent, session) in-memory buffer of the single last un-distilled
    round turn (``self._buffers[(agent_id, session_id)]``). Keying by session (the
    MetaClaw day) — not just agent — prevents a new session's round 1 from
    finalizing the previous session's dangling final round. The buffer holds the
    FULL ingest context of the previous round (owner ids, day, round id/index, and
    the sanitized turn) so that when the next round arrives we can distill the
    previous one with the new round's feedback as the pass/fail source.

    Concurrency: a per-key ``asyncio.Lock`` serializes ``distill_round`` /
    ``flush_session`` for the same (agent, session) so two concurrent ingests
    cannot read-modify-write the same buffer and duplicate/drop a record. The eval
    driver ingests serially, but the REST endpoint does not enforce that, so the
    lock makes the buffer correct regardless.
    """

    def __init__(self, *, record_manager, llm_client=None, llm_config=None):
        """
        Args:
            record_manager: a :class:`SkillEvolutionRecordManager` (C2) used to
                persist the distilled records.
            llm_client: an instance exposing ``async send_llm_request(messages)``
                (an ``LLMClientBase``). Injected directly in tests; in production
                pass ``llm_config`` instead and one is created lazily.
            llm_config: an ``LLMConfig`` used to build the client when
                ``llm_client`` is not supplied. Required for a real completion.
        """
        self.record_manager = record_manager
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._system_prompt = get_system_text(_DISTILLER_PROMPT_KEY)
        # (agent_id, session_id) -> buffered previous round context (one-round lag).
        self._buffers: Dict[tuple, Dict] = {}
        # (agent_id, session_id) -> lock guarding that buffer slot.
        self._locks: Dict[tuple, asyncio.Lock] = {}

    def _lock_for(self, key: tuple) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def distill_round(
        self,
        *,
        agent_id: str,
        user_id: str,
        organization_id: str,
        day: str,
        round_id: str,
        round_index: int,
        turn: Dict,
        session_id: Optional[str] = None,
    ):
        """Ingest one graded round; distill the PREVIOUS buffered round (if any).

        Implements the one-round lag: the incoming round is sanitized and
        buffered. The feedback embedded in THIS round's prompt
        (``sanitized["prev_feedback"]``) is the outcome of the PREVIOUSLY buffered
        round, so it is both (a) the pass/fail source for that previous round and
        (b) the feedback string the distiller LLM reasons over when writing the
        previous round's record.

        ``session_id`` keys the lag buffer; it defaults to ``day`` (a MetaClaw day
        == one session). Two sessions for the same agent never share a buffer.

        Returns the persisted :class:`SkillEvolutionRecord` for the previous round,
        or ``None`` if there was no previous round in this session, the previous
        round had no feedback (cross-session boundary / round 1), or the LLM
        produced no parseable record.
        """
        sid = session_id if session_id is not None else day
        key = (agent_id, sid)
        sanitized = sanitize_turn(turn)

        async with self._lock_for(key):
            previous = self._buffers.get(key)
            # Buffer the incoming round (it becomes the next "previous").
            self._buffers[key] = {
                "agent_id": agent_id,
                "user_id": user_id,
                "organization_id": organization_id,
                "day": day,
                "round_id": round_id,
                "round_index": round_index,
                "sanitized": sanitized,
            }

            if previous is None:
                # First round of this (agent, session) -> nothing to distill yet.
                return None

            # The outcome feedback of the PREVIOUS round is embedded in THIS
            # round's prompt head — it is the prev_feedback we just sanitized.
            outcome_feedback = sanitized.get("prev_feedback", "")
            return await self._finalize(previous, outcome_feedback)

    async def flush_session(self, *, agent_id: str, session_id: Optional[str] = None):
        """End a session: the last buffered round has no successor.

        Per DESIGN §C1 default: DROP the final buffered round (it has no feedback
        successor, so no legitimate pass/fail source) and produce NO record. This
        preserves the one-round-lag leakage guard (we never fabricate an outcome).

        When ``session_id`` is omitted, every buffered session for the agent is
        dropped (whole-agent flush); otherwise only that session's buffer is
        cleared. Returns ``None`` (the dropped round yields no record).

        Locks are acquired (never deleted) so a flush cannot race a concurrent
        ingest into the same slot; retaining the lock object is what guarantees a
        single mutex per key (deleting it would let a concurrent caller mint a
        second lock and defeat the exclusion).
        """
        if session_id is not None:
            keys = [(agent_id, session_id)]
        else:
            # Whole-agent flush: every (agent, *) buffer slot currently present.
            keys = [k for k in self._buffers if k[0] == agent_id]
        for key in keys:
            async with self._lock_for(key):
                self._buffers.pop(key, None)
        return None

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _finalize(self, previous: Dict, outcome_feedback: str):
        """Distill + persist a single buffered round given its outcome feedback.

        ``outcome_feedback`` is the correctness feedback for ``previous`` (it
        arrived embedded in the SUCCESSOR round's prompt). It is BOTH the pass/fail
        source AND the feedback string handed to the distiller LLM, so the LLM
        reasons over the right round's outcome (never a stale predecessor's).
        """
        record_type = derive_record_type(outcome_feedback)
        if record_type is None:
            # No legitimate pass/fail source for the previous round -> drop it.
            logger.debug(
                "distiller: no feedback for round %s; skipping (no graded record)",
                previous.get("round_id"),
            )
            return None

        parsed = await self._call_llm(
            previous["sanitized"], record_type, previous, outcome_feedback
        )
        if parsed is None:
            logger.warning(
                "distiller: unparseable LLM reply for round %s; no record persisted",
                previous.get("round_id"),
            )
            return None

        # record_type is authoritatively DERIVED from the feedback string, NOT the
        # LLM's self-report (§2: record_type is derived from feedback).
        evidence = parsed.get("evidence_round_ids") or [previous["round_id"]]
        if not isinstance(evidence, list):
            evidence = [previous["round_id"]]
        evidence = [str(e) for e in evidence if e]
        if not evidence:
            evidence = [previous["round_id"]]

        title = (parsed.get("title") or f"round {previous['round_id']} {record_type}")[
            :256
        ]
        description = parsed.get("description") or ""
        detail = parsed.get("detail") or ""

        return await self.record_manager.record_round_result(
            agent_id=previous["agent_id"],
            user_id=previous["user_id"],
            organization_id=previous["organization_id"],
            day=previous["day"],
            round_id=previous["round_id"],
            round_index=previous["round_index"],
            record_type=record_type,
            title=title,
            description=description,
            detail=detail,
            evidence_round_ids=evidence,
            quality_score=_clamp01(parsed.get("quality_score")),
            generality=_clamp01(parsed.get("generality")),
        )

    async def _call_llm(
        self,
        sanitized: Dict,
        record_type: str,
        previous: Dict,
        outcome_feedback: str,
    ) -> Optional[Dict]:
        """One async LLM completion -> parsed record dict (or None)."""
        client = self._get_client()
        if client is None:
            logger.warning("distiller: no LLM client/config available; skipping")
            return None

        user_payload = self._build_user_payload(
            sanitized, record_type, previous, outcome_feedback
        )
        messages = self._build_messages(user_payload, previous["agent_id"])

        try:
            response = await client.send_llm_request(messages=messages)
        except Exception as e:  # noqa: BLE001 — never crash the per-round path
            logger.warning("distiller: LLM request failed: %s", e)
            return None

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return None
        return parse_distiller_json(content)

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

    @staticmethod
    def _build_user_payload(
        sanitized: Dict, record_type: str, previous: Dict, outcome_feedback: str
    ) -> str:
        """Render the sanitized, leakage-filtered round into the LLM user turn.

        Only the allow-listed fields + the pre-derived outcome are passed; no
        oracle field can reach the model. ``prev_feedback`` here is the
        CORRECTNESS FEEDBACK FOR THE ROUND BEING DISTILLED (``outcome_feedback``),
        which arrived embedded in the successor round's prompt — NOT the buffered
        round's own (stale) predecessor feedback.
        """
        payload = {
            "round_id": previous["round_id"],
            "day": previous["day"],
            "outcome": record_type,
            "question": sanitized.get("question", ""),
            "answer": sanitized.get("answer", ""),
            "prev_feedback": outcome_feedback or "",
        }
        return (
            "Distill this single round into one "
            f"{record_type} record.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _build_messages(self, user_payload: str, agent_id: str) -> List[Message]:
        return [
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
