"""Per-session experience distillation.

Given the sessions a skill learning run claimed, this distills each session's
not-yet-distilled rows (those at or before the run's claim-time watermark)
into zero or more transferable :class:`SkillExperience` rows
(status ``pending``). A single session can yield MULTIPLE experiences and a
MIX of ``worth_learning`` / ``worth_avoiding``. There is no external
success/failure oracle — MIRIX never receives labels, gold answers or scores —
so in the same call, BEFORE its lessons, the distiller states from the
transcript alone whether the session achieved its task (``succeeded`` /
``failed`` / ``unclear``) and the evidence for it: the reply is ONE JSON object
``{outcome_evidence, outcome, experiences}``. The lessons follow from it —
the path that worked (its actions in the form the environment accepted),
where a failure's turns went and the step to take instead, or (unclear) the
procedure followed, at low credibility — and are grounded
in the conversation content (user critique/confirmation, tool errors/retries,
the evidenced outcome, the agent's own self-corrections, or weak inference).
The outcome and its evidence are stored with every experience of the session
(``session_outcome`` / ``outcome_evidence`` in its ``evidence`` JSON), where
the curator sees them. A ``succeeded`` / ``failed`` outcome is checked against
the transcript (:func:`check_reply_outcome`): EVERY substantial quote of its
evidence and of its ``task_outcome`` lessons (:data:`OUTCOME_QUOTE_MIN_CHARS`)
must occur in a message the chunk was shown, and one must occur beyond the
task statement (the first user message, written before the agent acted);
otherwise it is stored as ``unclear`` (``stated_outcome`` keeps what was
stated). One that only the agent's own words show — in its messages, or
repeated by a tool or the user after it wrote them (``outcome_source``
``agent_only``) — grounds ``task_outcome`` lessons at MEDIUM credibility at
most. A quote is compared squashed (no spaces, punctuation or case, Unicode
NFKC, JSON escapes either kept or decoded) with the ``role: text`` lines the
distiller was shown.
A long session's chunks are each checked against what they were shown
(:func:`chunk_verdicts`) and reconciled into one outcome
(:func:`reconcile_verdicts`). The legacy answer — a bare array of experiences —
is still taken, with no outcome.

A session's ``failed`` outcome needs a failure the transcript shows beyond
the agent's own words: one that only the agent's messages show
(``outcome_source`` ``agent_only``) is stored ``unclear`` like an unverified
one, ``stated_outcome`` keeping what was stated — the absence of a success
message is no failure.

Session context and skill feedback
    With a ``session_context_loader`` (a learning run passes one), the
    distiller is also shown, before the conversation, the skills MIRIX served
    to the agent in the session (as served: name, version, description,
    instructions) and how the agent's prompt was built (the client's
    ``agent_context``), as delimited untrusted blocks
    (:mod:`mirix.services.distill_session_context`,
    :class:`~mirix.services.distill_transcript.TranscriptContext`). It then
    gives, per served skill, a ``skill_feedback`` entry: ``followed``
    (yes / partly / no / unclear) and ``effect`` (helped / no_effect / hurt /
    unclear), each with a quote. The quotes are checked like an outcome's
    (:func:`check_feedback_entry`): a ``followed`` other than unclear needs a
    quote found in a message the chunk was shown (the agent's own words may
    show it); an ``effect`` other than unclear needs one from a message of
    the user or a tool — otherwise it is stored ``unclear`` (the stated value
    kept). Every served skill gets one ``skill_session_feedback`` row (unclear
    when no entry was given), committed with the session's experiences.

The transcript source is the Conversation Message Store (the SINGLE source of
truth for skill learning): a dedicated store of external conversation turns with
their REAL ``user`` / ``assistant`` / ``tool`` roles preserved, written only
when an add request carried a ``session_id``. The distiller therefore can NEVER
see the meta agent's own memory-management scaffolding — those rows never enter
this store. Correctness comes from structure (which table we read), not from a
string heuristic.

Input (:mod:`mirix.services.distill_transcript`)
    The rows are rendered into ONE multimodal user message: header metadata,
    ``role: text`` lines, and every stored image as ``ImageContent`` at its
    original position (the LLM client expands it to base64 before sending).
    Nothing is cut by character counts; the session is sized by the context
    budget: whole when it fits ``input_budget(system prompt, planning
    reserve)``, else chunked on round boundaries. A long session's chunks
    are distilled independently and their experiences are all kept — no dedupe;
    the curator handles duplicates. The distiller does NOT see the
    existing skills.

Concurrency
    At most ``MIRIX_DISTILLER_MAX_CONCURRENCY`` (4) distiller LLM calls run at
    once in the whole process — across every learning run, session and chunk
    (:func:`distiller_call_gate`) — and at most that many of a run's sessions
    are in progress at once.

Failures — every retry is recorded as a retry event of the run
    * ``finish_reason=length`` or ``ContextBudgetExceeded`` (or the provider's
      context-length refusal): the session's chunk budget is halved and the
      session re-chunked and distilled again, at most 3 halvings; a truncated
      answer is never used. Still failing after the 3rd: one counted failure.
    * unparseable output (neither the reply object with an ``experiences``
      array nor the legacy array of experiences): a repair prompt is appended
      to the SAME dialogue, at most twice; still unparseable: one counted
      failure. Unparseable output is never taken as "no experiences" — only
      an explicit ``[]`` is.
    * a timed-out call: retried once as is; timing out again: one counted
      failure.
    * infrastructure errors (network, 408/429/5xx, provider unavailable) and
      configuration errors — among them a context window too small for the
      distiller's own prompt — : the session stays pending and backs off
      (``distill_retry_after``); never counted. A deferral the distiller's own
      configuration causes, which every run meets again, is named in the
      session's metrics (``config_deferral``,
      :data:`DISTILL_CONFIGURATION_DEFERRALS`).
    * a transient database error: the transaction is retried (up to 5 times);
      still failing: the whole run fails and its sessions are released.
    * MIRIX's own failure — a distill request it could not build
      (``LLMRequestBuildError``, whatever it wraps: a ``KeyError``, a
      ``ValueError``, an image file missing from disk …; kind ``internal``):
      never the session's fault, so never counted and never backed off. The
      session stays pending (status ``internal_error`` in its metrics), the
      run's other sessions and its curation go on, and the run is then
      recorded ``failed`` with the error (``LearningRunInternalError``) —
      loud, instead of silently retiring or endlessly deferring a transcript.
    * after 3 counted failures the session gets ``distill_failed_at`` and is
      never selected again.

Metrics: one entry per session — messages, rounds, raw tokens, images,
chunks, truncated messages / tokens, LLM calls and usage, failure kinds,
experiences produced, the outcome each chunk stated (``outcomes``, in chunk
order; None where it stated none), the outcome stored (``session_outcome``),
the chunks whose outcome the transcript did not show or only the agent's own
messages did (``outcomes_unverified`` / ``outcomes_self_reported``) and the
credibilities capped — written to ``skill_learning_run.distill_metrics``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import weakref
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from mirix.constants import DISTILLER_MAX_CONCURRENCY
from mirix.errors import (
    ContextBudgetExceeded,
    LLMJSONParsingError,
    LLMOutputTruncatedError,
    LLMRequestBuildError,
)
from mirix.helpers.json_parsing import (
    parse_distiller_json,
    parse_json_array_or_none,
    parse_json_object_with_key,
)
from mirix.log import get_logger
from mirix.observability.context import conversation_session_scope
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.skill_experience import (
    SESSION_OUTCOMES,
    SKILL_EXPERIENCE_MAX_CONTENT_LEN as _CONTENT_CAP,
    SKILL_EXPERIENCE_MAX_EVIDENCE_LEN as _EVIDENCE_CAP,
    SKILL_EXPERIENCE_MAX_TITLE_LEN as _TITLE_CAP,
    SessionOutcome,
    SkillExperience as PydanticSkillExperience,
    _clamp01,
)
from mirix.schemas.user import User as PydanticUser
from mirix.services.context_budget import (
    LARGE_TEXT_CHARS,
    MESSAGE_OVERHEAD_TOKENS,
    ModelLimits,
    TokenCounter,
    ensure_tokenizer,
    get_token_counter,
)
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_message_manager import owner_org as _owner_org
from mirix.schemas.skill_session_feedback import EFFECT_VALUES, FOLLOWED_VALUES
from mirix.services.skill_text_checks import squash_letters
from mirix.services.distill_transcript import (
    MIN_CHUNK_BUDGET_TOKENS,
    TranscriptChunk,
    TranscriptContext,
    TranscriptPlan,
    TranscriptPlanner,
    build_messages,
    plan_text_size,
    row_image_ids,
)
from mirix.services.retry_policy import (
    RETRY_POLICIES,
    ErrorKind,
    RetryEventRecorder,
    classify_error,
    raise_if_truncated,
    recording_retries,
    retry_call,
)
from mirix.services.skill_experience_manager import SkillExperienceManager
from mirix.services.skill_learning_run_manager import LearningRunNotActive

logger = get_logger(__name__)

# At most this many halvings of a session's chunk budget.
MAX_BUDGET_HALVINGS = 3
# At most this many repair prompts for an unparseable answer.
MAX_REPAIR_PROMPTS = 2
# The previous answer is echoed back with the repair prompt, cut head + tail
# past this many characters (it only shows the model what it did wrong).
REPAIR_ECHO_MAX_CHARS = 8000

# The credibility an ``inferred`` lesson has at most: the top of the LOW band
# (0.1-0.3) the distiller prompt gives inference. A lesson no explicit signal
# grounds — among them what an ``unclear`` session teaches — never outranks
# one that a signal grounds, whatever score the model gave it.
INFERRED_CREDIBILITY_MAX = 0.3
# ``outcome_evidence`` past this many characters keeps its head and tail. The
# prompt asks for a brief statement with short quotes (a few hundred
# characters); it is repeated on every experience of the session, so it stays
# bounded.
OUTCOME_EVIDENCE_MAX_CHARS = 2000

# A ``task_outcome`` lesson of a session whose outcome only the agent's own
# messages show (a check it states in its own words: "I double-checked the
# table") has at most the top of the MEDIUM band (0.4-0.7) the distiller prompt
# gives such grounding: HIGH needs the environment, a tool or the user.
SELF_REPORTED_OUTCOME_CREDIBILITY_MAX = 0.7
# A quote (each part of an elided one) with fewer letters and digits than this
# says too little to check an outcome against the transcript, and is ignored.
# The quote and the transcript are compared squashed (:func:`_squash`: no
# spaces, punctuation or case), so what makes a coincidental match improbable
# is the length of the string compared, not its word count ("it is done"
# squashes to 8 characters found inside "commit is done"). Twelve admits the
# shortest real environment and tool reports ("Nothing happens." 14,
# "file not found" 12, "permission denied" 16, "task completed" 13) and
# refuses the single words and stock fragments any transcript holds ("answer",
# "success", "correct", "done", "you won", "wrong answer").
OUTCOME_QUOTE_MIN_CHARS = 12
# A CJK character counts as this many letters toward the minimum (a character
# is a syllable or a word, as much as a short Latin word): four are enough.
CJK_QUOTE_CHAR_WEIGHT = 3
# Where the quotes behind a succeeded / failed outcome were found in the
# session's transcript (``outcome_source`` in the evidence JSON): in a message
# of the user or a tool (the environment), only in the agent's own words (its
# messages, or a tool's or the user's that repeat them after it wrote them),
# or nowhere — then the stated outcome is kept as ``stated_outcome`` and the
# session's outcome is ``unclear``: an outcome the transcript does not show
# grounds no lesson. Every quote must be in the transcript, and the task
# statement (the transcript's first user message) shows no outcome: a quote
# found only there grounds nothing (:func:`outcome_source`).
OUTCOME_SOURCE_EXTERNAL = "user_or_tool"
OUTCOME_SOURCE_AGENT = "agent_only"
OUTCOME_SOURCE_NOT_FOUND = "not_in_transcript"
_OUTCOME_SOURCES = (
    OUTCOME_SOURCE_EXTERNAL,
    OUTCOME_SOURCE_AGENT,
    OUTCOME_SOURCE_NOT_FOUND,
)

_VALID_EXPERIENCE_TYPES = ("worth_learning", "worth_avoiding")
_VALID_SIGNALS = {
    "user_critique",
    "user_confirmation",
    "tool_error",
    "task_outcome",
    "self_correction",
    "inferred",
}
# Other spellings a model may use for a signal (after case, spaces and
# hyphens are normalized, :func:`normalize_signal_type`).
_SIGNAL_ALIASES = {
    "user_correction": "user_critique",
    "critique": "user_critique",
    "confirmation": "user_confirmation",
    "tool_failure": "tool_error",
    "error": "tool_error",
    "outcome": "task_outcome",
    "self_corrected": "self_correction",
    "inference": "inferred",
}
# The outcomes that can ground a ``task_outcome`` lesson (an unclear session's
# cannot: its lessons are inferred).
_EVIDENCED_OUTCOMES = ("succeeded", "failed")
# The session-level keys of an experience's evidence JSON.
_OUTCOME_KEYS = (
    "session_outcome",
    "outcome_evidence",
    "outcome_source",
    "stated_outcome",
)
# Other words a model may use for an outcome.
_OUTCOME_ALIASES = {
    "success": "succeeded",
    "successful": "succeeded",
    "succeed": "succeeded",
    "failure": "failed",
    "fail": "failed",
    "unknown": "unclear",
    "uncertain": "unclear",
}

REPAIR_PROMPT = (
    "Your previous reply could not be used: {reason}. Reply again with ONLY one "
    'JSON object: "outcome_evidence" (what in the transcript shows whether the '
    'session achieved its task, quoted briefly), "outcome" ("succeeded", '
    '"failed" or "unclear"), "skill_feedback" (when skills were served: one '
    'object per served skill with "skill", "followed", "followed_evidence", '
    '"effect" and "effect_evidence"; [] otherwise) and "experiences" — an array '
    'of experience objects, each with "experience_type" ("worth_learning" or '
    '"worth_avoiding"), a non-empty "title", "content", "importance", '
    '"credibility" and "evidence"; [] if nothing in the transcript is worth '
    "remembering. No prose, no markdown fences."
)
_REASON_UNUSABLE = 'it is not one JSON object with an "experiences" array'
_REASON_NO_VALID_ITEM = (
    "none of its elements is a valid experience (each needs an experience_type "
    "of worth_learning or worth_avoiding and a non-empty title)"
)


def normalize_signal_type(value: Any) -> Optional[str]:
    """``value`` as one of the signal types, with case, surrounding spaces,
    inner spaces and hyphens forgiven ("Tool_Error", "tool error") and a few
    synonyms mapped; None when it is none of them."""
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    key = _SIGNAL_ALIASES.get(key, key)
    return key if key in _VALID_SIGNALS else None


def normalize_session_outcome(value: Any) -> Optional[SessionOutcome]:
    """``value`` as one of :data:`SESSION_OUTCOMES` (case, spaces and a few
    synonyms forgiven), or None when it is none of them."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    key = _OUTCOME_ALIASES.get(key, key)
    return cast(SessionOutcome, key) if key in SESSION_OUTCOMES else None


def _as_text(value: Any) -> str:
    """A model-written free-text field as one string ("" for none): a list of
    quotes is joined, anything else structured is JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class DistillerReply:
    """A usable distiller answer: the session's outcome as the model stated it
    (None when it stated none — the legacy array answer, or a value that is not
    an outcome), the evidence it gave for it, and its experience objects (not
    yet validated)."""

    outcome: Optional[SessionOutcome]
    outcome_evidence: str
    items: List[Dict[str, Any]]
    #: The reply's ``skill_feedback`` entries (objects; not yet matched to the
    #: served skills nor checked).
    skill_feedback: List[Dict[str, Any]] = field(default_factory=list)


# The reply object's own key, as JSON writes it: a legacy bare array of
# experiences never has it (an escaped \"experiences\" inside a string is no key).
_EXPERIENCES_KEY = re.compile(r'(?<!\\)"experiences"\s*:')


def _is_experience_like(obj: Dict[str, Any]) -> bool:
    """An object that is (meant as) one experience, not the reply object."""
    return "experience_type" in obj or "title" in obj


def parse_distiller_reply(content: Optional[str]) -> Optional[DistillerReply]:
    """The distiller's answer — ONE JSON object ``{outcome_evidence, outcome,
    experiences}`` (fenced, wrapped in prose or in an array is forgiven) — or
    None when it is unusable. ``experiences`` must be an array of objects (a
    single object is wrapped; other elements are dropped); ``[]`` is an
    explicit "nothing worth remembering". The legacy answer — a bare array of
    experiences, or one experience object (a per-lesson ``outcome`` field
    included) — is taken with no outcome. Unusable, so the repair prompt asks
    again: a reply object that does not parse (an unescaped quote in its
    ``outcome_evidence``, a trailing comma, a missing brace: taking the array
    inside it as a legacy answer would silently drop the session's outcome),
    and an outcome object without its ``experiences``."""
    reply = parse_json_object_with_key(content, "experiences")
    if reply is None:
        text = content if isinstance(content, str) else ""
        if _EXPERIENCES_KEY.search(text):
            return None  # the reply object is there, but broken
        first = parse_distiller_json(text)
        if first is not None and "outcome" in first and not _is_experience_like(first):
            return None  # an outcome object without its experiences
        items = parse_json_array_or_none(content)
        if items is None:
            return None
        return DistillerReply(outcome=None, outcome_evidence="", items=items)
    raw = reply.get("experiences")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    items = [item for item in raw if isinstance(item, dict)]
    if raw and not items:
        return None
    feedback = reply.get("skill_feedback")
    if isinstance(feedback, dict):
        feedback = [feedback]
    return DistillerReply(
        outcome=normalize_session_outcome(reply.get("outcome")),
        outcome_evidence=_as_text(reply.get("outcome_evidence")),
        items=items,
        skill_feedback=[
            entry
            for entry in (feedback if isinstance(feedback, list) else [])
            if isinstance(entry, dict)
        ],
    )


@dataclass(frozen=True)
class SessionVerdict:
    """The outcome stored with every experience of a session: ``outcome``
    (``session_outcome``; None when no chunk stated one — the legacy answer),
    ``evidence`` (``outcome_evidence``), ``source`` (``outcome_source``: where
    the transcript shows a succeeded / failed outcome, :data:`_OUTCOME_SOURCES`;
    None when it was not checked) and ``stated`` (``stated_outcome``: the
    outcome the distiller stated when the transcript did not show it and it
    was replaced by ``unclear``)."""

    outcome: Optional[SessionOutcome]
    evidence: str = ""
    source: Optional[str] = None
    stated: Optional[SessionOutcome] = None


# Quotation marks (:func:`_quoted_spans`). A span in double quotes, curly
# double quotes or CJK corner brackets runs to the first closing mark. It is
# found by a linear scan: an opening mark whose closing mark the rest of the
# text lacks is not sought again from every later one (as a regex,
# ``\u201c([^\u201d]+)\u201d`` is quadratic on a reply full of unclosed marks).
_PAIRED_QUOTES = {
    '"': '"',
    "\u201c": "\u201d",
    "\u300c": "\u300d",
    "\u300e": "\u300f",
}
# A single-quoted span. A single quote followed by a letter inside it is an
# apostrophe ('that's it', \u2018that\u2019s it\u2019), not its end: a model
# quoting in single quotes to save escaping inside its JSON string must have
# its whole quote checked, not the head before the first apostrophe. (Curly
# quotes: an ASCII letter, so a CJK text right after the closing quote still
# ends it.) Such a span is at most _SINGLE_QUOTED_MAX_CHARS long: a span that
# may run past apostrophes would otherwise be sought to the end of the text
# from every opening quote (quadratic on a reply full of them); a longer one
# is no short quote.
_SINGLE_QUOTED_MAX_CHARS = 300
_SINGLE_QUOTED = {
    "\u2018": re.compile(
        r"\u2018((?:[^\u2019]|\u2019(?=[A-Za-z])){1,%d})\u2019(?![A-Za-z])"
        % _SINGLE_QUOTED_MAX_CHARS
    ),
    "'": re.compile(
        r"(?<![\w'])'((?:[^']|'(?=\w)){1,%d})'(?!\w)" % _SINGLE_QUOTED_MAX_CHARS
    ),
}
_OPENING_MARK = re.compile(
    "[%s]" % re.escape("".join(_PAIRED_QUOTES) + "".join(_SINGLE_QUOTED))
)
_ELLIPSIS = re.compile(r"\.{3,}|\u2026")
_CJK_CHAR = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
# A JSON string escape. A tool's result or a call's arguments shown as JSON
# text hold them ("Table 3.\nTotal", "caf\u00e9"), and a quote may give the
# text they stand for or keep them as shown (:func:`_squashed_forms`).
_JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|["\\/bfnrt])')


def _squash(text: Any) -> str:
    """``text`` in Unicode NFKC form, casefolded, with everything but letters,
    digits and combining marks removed (:func:`skill_text_checks.squash_letters`):
    a quote matches its source whatever the spacing, punctuation, case or
    Unicode form ("coffee machine" = "coffeemachine"; a composed or a
    decomposed accent, full-width digits; a vowel sign of a script that
    writes vowels as marks counts as a letter)."""
    return squash_letters(text)


def _unescape(match: "re.Match[str]") -> str:
    code = match.group(1)
    if code is None:
        return " "  # a quote, a slash, a backslash or a control: no letter
    value = int(code, 16)
    return " " if 0xD800 <= value <= 0xDFFF else chr(value)  # half a pair: none


def _squashed_forms(text: str) -> Tuple[str, ...]:
    """``text`` squashed (:func:`_squash`), and a second form with its JSON
    string escapes decoded first when that differs: the same words shown as
    JSON text or as plain text, and quoted either way, still match
    (:func:`_occurs`)."""
    squashed = _squash(text)
    if "\\" not in text:
        return (squashed,)
    decoded = _squash(_JSON_ESCAPE.sub(_unescape, text))
    return (squashed,) if decoded == squashed else (squashed, decoded)


def _quoted_spans(text: str) -> List[str]:
    """The spans of ``text`` in quotation marks, left to right, or ``[text]``
    when it quotes nothing (then the whole text must be verbatim). Linear in
    the size of ``text``: each closing mark is sought once past the text
    already read, and a single-quoted span at most
    ``_SINGLE_QUOTED_MAX_CHARS`` ahead."""
    spans: List[str] = []
    unclosed: Set[str] = set()  # closing marks the rest of the text lacks
    pos = 0
    while True:
        opening = _OPENING_MARK.search(text, pos)
        if opening is None:
            break
        start = opening.start()
        pos = start + 1
        closing = _PAIRED_QUOTES.get(opening.group())
        if closing is None:
            single = _SINGLE_QUOTED[opening.group()].match(text, start)
            if single is not None:
                spans.append(single.group(1))
                pos = single.end()
            continue
        if closing in unclosed:
            continue
        end = text.find(closing, pos)
        if end < 0:
            unclosed.add(closing)
        elif end > pos:  # an empty pair quotes nothing
            spans.append(text[pos:end])
            pos = end + 1
    return spans or ([text] if text.strip() else [])


def _quote_weight(squashed: str) -> int:
    """How much a squashed quote part says: its letters and digits, a CJK
    character counting :data:`CJK_QUOTE_CHAR_WEIGHT`."""
    return len(squashed) + (CJK_QUOTE_CHAR_WEIGHT - 1) * len(
        _CJK_CHAR.findall(squashed)
    )


# One part of a quote as :func:`outcome_source` looks for it: squashed, or its
# squashed forms (:func:`_squashed_forms`) when decoding its JSON escapes
# changes it; any form found is the part found.
QuotePart = Union[str, Tuple[str, ...]]


def _checkable_parts(span: str) -> List[QuotePart]:
    """A quote's squashed parts (split at an ellipsis), each substantial
    enough to check (:data:`OUTCOME_QUOTE_MIN_CHARS`); [] when none is."""
    parts: List[QuotePart] = []
    for part in _ELLIPSIS.split(span):
        forms = _squashed_forms(part)
        if max(map(_quote_weight, forms)) >= OUTCOME_QUOTE_MIN_CHARS:
            parts.append(forms[0] if len(forms) == 1 else forms)
    return parts


def outcome_quotes(
    outcome_evidence: str, items: Sequence[Dict[str, Any]]
) -> List[List[QuotePart]]:
    """The quotes that should show a session's stated outcome in its
    transcript, each as its checkable parts: the quoted spans of
    ``outcome_evidence`` and the quote of every ``task_outcome`` lesson of
    ``items`` (a text without quotation marks counts whole). A quote too short
    to check (:data:`OUTCOME_QUOTE_MIN_CHARS`) is left out."""
    spans = _quoted_spans(outcome_evidence or "")
    for item in items:
        evidence = item.get("evidence") if isinstance(item, dict) else None
        if isinstance(evidence, dict) and (
            normalize_signal_type(evidence.get("signal_type")) == "task_outcome"
        ):
            spans.extend(_quoted_spans(_as_text(evidence.get("quote"))))
    return [parts for parts in (_checkable_parts(span) for span in spans) if parts]


# The role of the agent's own messages in a transcript.
_AGENT_ROLE = "assistant"


@dataclass(frozen=True)
class IndexedMessage:
    """One rendered transcript message as :func:`outcome_source` searches it:
    its 1-based ``position``, its ``role``, its ``text`` squashed
    (:func:`_squash`) as the distiller is shown it — its ``role: `` label
    included, so a quote of the whole line is verbatim and a quote without
    the label still occurs in it — with ``unescaped`` its second squashed
    form, JSON escapes decoded (:func:`_squashed_forms`; "" when the same),
    and whether it is the ``task_statement`` — the transcript's first user
    message, which states the task before the agent did anything (the chunk
    frames show its round as "task statement"). A reopened session is
    distilled from its new rows only, so there the first user message may
    answer the agent's earlier reply ("that's wrong"); it is still taken as
    the task statement, as the frame names it: such a reply grounds nothing,
    and an outcome only it shows is stored ``unclear`` — lost, never wrongly
    grounded."""

    position: int
    role: str
    text: str
    task_statement: bool = False
    unescaped: str = ""


def transcript_index(messages: Sequence[Any]) -> List[IndexedMessage]:
    """Each rendered transcript message (``TranscriptMessage``) as an
    :class:`IndexedMessage`, in transcript order, for :func:`outcome_source`."""
    index: List[IndexedMessage] = []
    task_seen = False
    for message in messages:
        role = str(message.role)
        forms = _squashed_forms(message.plain_text())
        task = role == "user" and not task_seen
        task_seen = task_seen or task
        index.append(
            IndexedMessage(
                position=message.position,
                role=role,
                text=forms[0],
                task_statement=task,
                unescaped=forms[1] if len(forms) > 1 else "",
            )
        )
    return index


def _occurs(parts: Sequence[QuotePart], message: IndexedMessage) -> bool:
    """Whether every part of a quote occurs in ``message`` (a part in any of
    its forms, in either form of the message's text)."""
    texts = (message.text, message.unescaped) if message.unescaped else (message.text,)
    return all(
        any(
            form in text
            for form in ((part,) if isinstance(part, str) else part)
            for text in texts
        )
        for part in parts
    )


def outcome_source(
    quotes: Sequence[Sequence[QuotePart]], index: Sequence[IndexedMessage]
) -> str:
    """Where ``quotes`` show a stated outcome in the transcript ``index`` (in
    transcript order; a quote occurs in a message when all its parts do):

    * :data:`OUTCOME_SOURCE_NOT_FOUND` when there is no quote, when one of
      them occurs in no message (it was not copied from the transcript, so
      what the evidence claims is not shown), or when they occur only in the
      task statement (it was written before the agent acted: it restates the
      task and shows no outcome);
    * else :data:`OUTCOME_SOURCE_EXTERNAL` when one occurs in a message of the
      user or a tool other than the task statement, before any of the
      agent's own messages holds it;
    * else :data:`OUTCOME_SOURCE_AGENT`: beyond the task statement, they are
      the agent's own words — in its (``assistant``) messages, or in a tool's
      or the user's message only after the agent wrote them (a tool printing
      back the verdict the agent put in its call, the user citing the agent's
      reply): what they show is the agent's claim."""
    if not quotes:
        return OUTCOME_SOURCE_NOT_FOUND
    grounding: Set[str] = set()
    for parts in quotes:
        found = False
        agent_wrote = False  # an earlier agent message holds the quote
        for message in index:
            if not _occurs(parts, message):
                continue
            found = True
            if message.task_statement:
                continue
            if agent_wrote or message.role == _AGENT_ROLE:
                agent_wrote = True
                grounding.add(_AGENT_ROLE)
                continue
            grounding.add(message.role)
            break  # nothing this quote can add
        if not found:
            return OUTCOME_SOURCE_NOT_FOUND
    if not grounding:
        return OUTCOME_SOURCE_NOT_FOUND
    if grounding - {_AGENT_ROLE}:
        return OUTCOME_SOURCE_EXTERNAL
    return OUTCOME_SOURCE_AGENT


def check_reply_outcome(
    reply: DistillerReply, index: Optional[Sequence[IndexedMessage]]
) -> SessionVerdict:
    """One chunk's verdict: a stated ``succeeded`` / ``failed`` outcome is
    kept only when the quotes of its evidence and of its ``task_outcome``
    lessons show it in ``index`` — the messages the chunk was shown
    (:func:`outcome_source`); otherwise it becomes ``unclear`` (the stated one
    kept as ``stated``). A ``failed`` outcome that only the agent's own words
    show (``agent_only``) becomes ``unclear`` too (a failure needs a
    signal the transcript shows beyond the agent's claim). ``index`` None:
    nothing to check against, the stated outcome is kept unchecked."""
    if reply.outcome not in _EVIDENCED_OUTCOMES or index is None:
        return SessionVerdict(reply.outcome, reply.outcome_evidence)
    source = outcome_source(outcome_quotes(reply.outcome_evidence, reply.items), index)
    if source == OUTCOME_SOURCE_NOT_FOUND or (
        reply.outcome == "failed" and source == OUTCOME_SOURCE_AGENT
    ):
        return SessionVerdict("unclear", reply.outcome_evidence, source, reply.outcome)
    return SessionVerdict(reply.outcome, reply.outcome_evidence, source)


def frame_positions(planner: TranscriptPlanner) -> Set[int]:
    """The positions of the messages a chunk's full frame shows beside its
    own rounds: the task statement (round 1) and the final rounds. A section
    cut head + tail to fit shows only part of them; a quote still comes from
    them."""
    rounds = planner.rounds[:1] + planner._final_rounds()
    return {message.position for round_ in rounds for message in round_.messages}


def chunk_verdicts(
    replies: Sequence[DistillerReply],
    plan: TranscriptPlan,
    planner: Optional[TranscriptPlanner],
) -> List[SessionVerdict]:
    """Each chunk's verdict (:func:`check_reply_outcome`, ``replies`` in the
    order of ``plan.chunks``), its quotes checked against the messages THAT
    chunk was shown: the whole session for a whole chunk, else the chunk's own
    rounds (overlap included) and — unless its frame is compact — the task
    statement and the final rounds of its frame (:func:`frame_positions`). A
    chunk cannot quote what it was never shown. ``planner`` None: nothing to
    check against. Pure CPU."""
    if planner is None or not any(
        reply.outcome in _EVIDENCED_OUTCOMES for reply in replies
    ):
        return [check_reply_outcome(reply, None) for reply in replies]
    index = transcript_index(planner.messages)
    frame: Set[int] = set() if plan.compact_frame else frame_positions(planner)
    verdicts: List[SessionVerdict] = []
    for reply, chunk in zip(replies, plan.chunks, strict=True):
        shown: Sequence[IndexedMessage] = index
        if not chunk.whole:
            positions = frame.union(chunk.positions)
            shown = [message for message in index if message.position in positions]
        verdicts.append(check_reply_outcome(reply, shown))
    return verdicts


def reconcile_verdicts(verdicts: Sequence[SessionVerdict]) -> SessionVerdict:
    """The one outcome of a session distilled in chunks (each chunk judges
    it from what it is shown, the final rounds included): the last chunk's
    when it is succeeded or failed (it covers the end); else the other
    chunks' succeeded / failed outcome when they agree; ``unclear`` when they
    disagree; else the last outcome any chunk stated (None: none did)."""
    if not verdicts:
        return SessionVerdict(None)
    last = verdicts[-1]
    if last.outcome in _EVIDENCED_OUTCOMES:
        return last
    evidenced = [
        (number, v)
        for number, v in enumerate(verdicts, 1)
        if v.outcome in _EVIDENCED_OUTCOMES
    ]
    if evidenced:
        if len({v.outcome for _, v in evidenced}) == 1:
            return evidenced[-1][1]
        return SessionVerdict(
            "unclear",
            "The chunks of this session disagree: "
            + "; ".join(
                f"chunk {number} stated {v.outcome} ({v.evidence})"
                for number, v in evidenced
            ),
        )
    stated = [v for v in verdicts if v.outcome is not None]
    return stated[-1] if stated else last


def _with_session_outcome(
    item: Dict[str, Any], verdict: SessionVerdict
) -> Dict[str, Any]:
    """``item`` with its session's outcome in its ``evidence`` — the
    session's, which replace any the item states itself (the outcome belongs
    to the session, not to one lesson)."""
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        evidence = {k: v for k, v in evidence.items() if k not in _OUTCOME_KEYS}
    elif isinstance(evidence, str) and evidence.strip():
        evidence = {"quote": evidence, "signal_type": "inferred"}
    else:
        evidence = {"quote": "", "signal_type": "inferred"}
    if verdict.outcome is not None:
        evidence["session_outcome"] = verdict.outcome
    if verdict.evidence:
        evidence["outcome_evidence"] = verdict.evidence
    if verdict.source is not None:
        evidence["outcome_source"] = verdict.source
    if verdict.stated is not None:
        evidence["stated_outcome"] = verdict.stated
    return {**item, "evidence": evidence}


# --------------------------------------------------------------------------- #
# Feedback on the skills served in the session                                  #
# --------------------------------------------------------------------------- #

# Other words a model may use for ``followed`` / ``effect`` (after case,
# spaces and hyphens are normalized to underscores).
_FOLLOWED_ALIASES = {
    "true": "yes",
    "followed": "yes",
    "fully": "yes",
    "full": "yes",
    "partially": "partly",
    "partial": "partly",
    "partly_followed": "partly",
    "false": "no",
    "not_followed": "no",
    "ignored": "no",
    "none": "no",
    "unknown": "unclear",
    "uncertain": "unclear",
    "n/a": "unclear",
}
_EFFECT_ALIASES = {
    "help": "helped",
    "helpful": "helped",
    "positive": "helped",
    "none": "no_effect",
    "no": "no_effect",
    "neutral": "no_effect",
    "noeffect": "no_effect",
    "hurt_it": "hurt",
    "harmful": "hurt",
    "harmed": "hurt",
    "negative": "hurt",
    "unknown": "unclear",
    "uncertain": "unclear",
    "n/a": "unclear",
}
# A stored feedback quote keeps its head and tail past this many characters.
FEEDBACK_EVIDENCE_MAX_CHARS = 1000


def _normalized(
    value: Any, aliases: Dict[str, str], values: Sequence[str]
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    key = aliases.get(key, key)
    return key if key in values else None


def normalize_followed(value: Any) -> Optional[str]:
    """``value`` as one of yes / partly / no / unclear (case, spaces and a
    few synonyms forgiven), or None."""
    return _normalized(value, _FOLLOWED_ALIASES, FOLLOWED_VALUES)


def normalize_effect(value: Any) -> Optional[str]:
    """``value`` as one of helped / no_effect / hurt / unclear (case, spaces
    and a few synonyms forgiven), or None."""
    return _normalized(value, _EFFECT_ALIASES, EFFECT_VALUES)


@dataclass(frozen=True)
class FeedbackVerdict:
    """One served skill's feedback as stored: ``followed`` / ``effect``
    after the quote checks (:func:`check_feedback_entry`), their quotes,
    where each quote was found (``*_source``: :data:`_OUTCOME_SOURCES`; None
    when nothing was checked) and the value the distiller stated when it was
    stored ``unclear`` (``stated_*``)."""

    followed: str = "unclear"
    followed_evidence: str = ""
    followed_source: Optional[str] = None
    stated_followed: Optional[str] = None
    effect: str = "unclear"
    effect_evidence: str = ""
    effect_source: Optional[str] = None
    stated_effect: Optional[str] = None


NO_FEEDBACK = FeedbackVerdict()


def evidence_source(text: str, index: Sequence[IndexedMessage]) -> str:
    """Where the quotes of ``text`` show in the transcript ``index``
    (:func:`outcome_source`; a text without quotation marks counts whole, a
    quote too short to check is left out)."""
    quotes = [
        parts
        for parts in (_checkable_parts(span) for span in _quoted_spans(text or ""))
        if parts
    ]
    return outcome_source(quotes, index)


def check_feedback_entry(
    entry: Mapping[str, Any], index: Optional[Sequence[IndexedMessage]]
) -> FeedbackVerdict:
    """One ``skill_feedback`` entry checked against ``index`` — the messages
    the chunk was shown (None: kept unchecked): ``followed`` yes / partly /
    no stands only when a quote of ``followed_evidence`` occurs in them (the
    agent's own messages count: following is what it did); ``effect``
    helped / no_effect / hurt only when one of ``effect_evidence`` occurs in
    a message of the user or a tool (``user_or_tool``) — the agent's own
    claim of an effect is no evidence of it. Otherwise the value is stored
    ``unclear`` and the stated one kept."""
    followed = normalize_followed(entry.get("followed")) or "unclear"
    followed_evidence = _as_text(entry.get("followed_evidence"))
    effect = normalize_effect(entry.get("effect")) or "unclear"
    effect_evidence = _as_text(entry.get("effect_evidence"))
    followed_source = effect_source = None
    stated_followed = stated_effect = None
    if followed != "unclear" and index is not None:
        followed_source = evidence_source(followed_evidence, index)
        if followed_source == OUTCOME_SOURCE_NOT_FOUND:
            stated_followed, followed = followed, "unclear"
    if effect != "unclear" and index is not None:
        effect_source = evidence_source(effect_evidence, index)
        if effect_source != OUTCOME_SOURCE_EXTERNAL:
            stated_effect, effect = effect, "unclear"
    return FeedbackVerdict(
        followed=followed,
        followed_evidence=_cap_text(followed_evidence, FEEDBACK_EVIDENCE_MAX_CHARS)[0],
        followed_source=followed_source,
        stated_followed=stated_followed,
        effect=effect,
        effect_evidence=_cap_text(effect_evidence, FEEDBACK_EVIDENCE_MAX_CHARS)[0],
        effect_source=effect_source,
        stated_effect=stated_effect,
    )


def _skill_key(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def match_feedback(
    entries: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> Dict[int, Mapping[str, Any]]:
    """``{served position: entry}``: each entry matched to a served skill by
    its ``skill`` (or ``name``) — the skill's name, or its number in the
    ``<skills_served>`` block ("1", "[1]"); the first entry for a skill wins
    (two served skills of the same name take that name's entries in served
    order), an entry that names no served skill is left out."""
    keys: Dict[str, List[int]] = {}
    for position, name in enumerate(names):
        keys.setdefault(_skill_key(name), []).append(position)
    matched: Dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        raw = entry.get("skill", entry.get("name", entry.get("skill_name")))
        key = _skill_key(raw)
        position = next((p for p in keys.get(key, ()) if p not in matched), None)
        if position is None and key not in keys:
            number = key.strip("[]#")
            if number.isdigit() and 1 <= int(number) <= len(names):
                position = int(number) - 1
        if position is not None and position not in matched:
            matched[position] = entry
    return matched


def chunk_feedback(
    replies: Sequence[DistillerReply],
    plan: TranscriptPlan,
    planner: Optional[TranscriptPlanner],
    names: Sequence[str],
) -> List[Dict[int, FeedbackVerdict]]:
    """Each chunk's checked feedback (``replies`` in the order of
    ``plan.chunks``), its quotes checked against the messages THAT chunk was
    shown, as :func:`chunk_verdicts` does. ``planner`` None: kept unchecked.
    An entry about a served skill the chunks did not show (the context cut
    left its header out: ``plan.context_visible_skills``) is no judgement of
    it and is left out. Pure CPU."""
    visible = plan.context_visible_skills
    matched = [
        {
            pos: entry
            for pos, entry in match_feedback(reply.skill_feedback, names).items()
            if visible is None or pos in visible
        }
        for reply in replies
    ]
    if not any(matched):
        return [{} for _ in replies]
    if planner is None:
        return [
            {pos: check_feedback_entry(entry, None) for pos, entry in m.items()}
            for m in matched
        ]
    index = transcript_index(planner.messages)
    frame: Set[int] = set() if plan.compact_frame else frame_positions(planner)
    out: List[Dict[int, FeedbackVerdict]] = []
    for entries, chunk in zip(matched, plan.chunks, strict=True):
        shown: Sequence[IndexedMessage] = index
        if not chunk.whole:
            positions = frame.union(chunk.positions)
            shown = [message for message in index if message.position in positions]
        out.append({pos: check_feedback_entry(e, shown) for pos, e in entries.items()})
    return out


def _mixed_followed(values: Set[str]) -> str:
    """Chunks that disagree on ``followed``: followed in part (each chunk
    saw a part of the session)."""
    return "partly"


def _mixed_effect(values: Set[str]) -> str:
    """Chunks that disagree on ``effect``: ``helped`` against ``hurt`` is
    no clear effect; one chunk seeing an effect where another saw none
    keeps the effect (the effect showed somewhere in the session)."""
    if "helped" in values and "hurt" in values:
        return "unclear"
    effects = values - {"no_effect"}
    return next(iter(effects)) if len(effects) == 1 else "unclear"


def reconcile_feedback(verdicts: Sequence[FeedbackVerdict]) -> FeedbackVerdict:
    """One served skill's feedback from the chunks that gave one (in chunk
    order), per field: the chunks' clear (not unclear) values when they all
    agree; when they disagree ``partly`` for ``followed`` (each chunk saw a
    part of the session) and for ``effect`` ``unclear`` when one chunk saw
    it help and another hurt, else the effect a chunk saw over another's
    ``no_effect`` — the evidence naming each chunk's; with no clear value,
    the last stated one (stored unclear)."""
    if not verdicts:
        return NO_FEEDBACK
    if len(verdicts) == 1:
        return verdicts[0]

    def pick(name: str, mixed: Callable[[Set[str]], str]) -> Dict[str, Any]:
        last = verdicts[-1]
        keys = (name, f"{name}_evidence", f"{name}_source", f"stated_{name}")
        clear = [
            (number, v)
            for number, v in enumerate(verdicts, 1)
            if getattr(v, name) != "unclear"
        ]
        if not clear:
            stated = [v for v in verdicts if getattr(v, f"stated_{name}")]
            source = stated[-1] if stated else last
            return {key: getattr(source, key) for key in keys}
        values = {getattr(v, name) for _, v in clear}
        if len(values) == 1:
            _, chosen = clear[-1]
            return {key: getattr(chosen, key) for key in keys}
        evidence = "; ".join(
            f"chunk {number}: {getattr(v, name)} ({getattr(v, f'{name}_evidence')})"
            for number, v in clear
        )
        value = mixed(values)
        sources = {
            getattr(v, f"{name}_source") for _, v in clear if getattr(v, name) == value
        }
        return {
            name: value,
            f"{name}_evidence": _cap_text(evidence, FEEDBACK_EVIDENCE_MAX_CHARS)[0],
            # The quotes behind the value kept, when they agree on where.
            f"{name}_source": next(iter(sources)) if len(sources) == 1 else None,
            # Reconciled from the chunks' checked values: none was downgraded.
            f"stated_{name}": None,
        }

    return FeedbackVerdict(
        **pick("followed", _mixed_followed), **pick("effect", _mixed_effect)
    )


def feedback_rows(
    served: Sequence[Any],
    per_chunk: Sequence[Mapping[int, FeedbackVerdict]],
    *,
    session: SessionVerdict,
    organization_id: Optional[str],
    user_id: str,
    agent_id: Optional[str],
    session_id: str,
) -> List[Dict[str, Any]]:
    """One ``skill_session_feedback`` row (keyword arguments of
    ``SkillSessionFeedbackManager.build_row``) per served skill, in served
    order: its chunks' feedback reconciled (:func:`reconcile_feedback`;
    unclear / unclear when none gave one) with the session's outcome."""
    rows = []
    for position, skill in enumerate(served):
        verdict = reconcile_feedback(
            [chunk[position] for chunk in per_chunk if position in chunk]
        )
        rows.append(
            {
                "organization_id": organization_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "skill_id": skill.skill_id,
                "skill_name": skill.name,
                "skill_version_no": skill.version_no,
                "served_count": int(skill.served_count or 1),
                "session_outcome": session.outcome,
                "followed": verdict.followed,
                "followed_evidence": verdict.followed_evidence or None,
                "followed_source": verdict.followed_source,
                "stated_followed": verdict.stated_followed,
                "effect": verdict.effect,
                "effect_evidence": verdict.effect_evidence or None,
                "effect_source": verdict.effect_source,
                "stated_effect": verdict.stated_effect,
            }
        )
    return rows


# commit_session(session_id, experience payloads) -> (experiences, rows marked)
# (and ``skill_feedback=`` the session's feedback rows when it has any)
CommitSession = Callable[
    [str, List[Dict[str, Any]]],
    Awaitable[Tuple[List[PydanticSkillExperience], int]],
]


@dataclass(frozen=True)
class SessionFailure:
    """Why a session could not be distilled in this run, and what it means.

    ``counted`` — a failure toward the session's 3. ``deferred`` — an
    infrastructure / configuration error: the session backs off instead
    (``distill_retry_after``) and the failure is never counted.
    ``config_deferral`` — for a deferral the distiller's own configuration
    causes (:data:`DISTILL_CONFIGURATION_DEFERRALS`), which every later run
    meets again until a person changes it: its name, recorded in the
    session's metrics (``config_deferral``); None for any other failure (a
    misconfigured deployment's 401/403/404 is kind ``config`` too, but a fix
    of the deployment ends it).
    """

    kind: ErrorKind
    counted: bool
    deferred: bool
    error: str
    config_deferral: Optional[str] = None


#: The distiller's input budget (its context window less its own prompt and
#: the output reserve) is below ``MIN_CHUNK_BUDGET_TOKENS``: no chunk of the
#: session fits.
DISTILL_DEFERRAL_INPUT_BUDGET_TOO_SMALL = "input_budget_too_small"
#: The distiller has no LLM client and no LLM config to make one from.
DISTILL_DEFERRAL_NO_LLM_CLIENT = "no_llm_client"
#: Every ``config_deferral`` name (:class:`SessionFailure`).
DISTILL_CONFIGURATION_DEFERRALS = frozenset(
    {DISTILL_DEFERRAL_INPUT_BUDGET_TOO_SMALL, DISTILL_DEFERRAL_NO_LLM_CLIENT}
)


#: A session's ``distill_metrics`` status after MIRIX's own failure (kind
#: ``internal``): not counted, not deferred; the run fails.
STATUS_INTERNAL_ERROR = "internal_error"


def internal_failure_sessions(metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """The sessions of a run's per-session ``metrics`` that MIRIX's own
    failure (kind ``internal``) left undistilled."""
    return [
        sid
        for sid, entry in metrics.items()
        if isinstance(entry, dict) and entry.get("status") == STATUS_INTERNAL_ERROR
    ]


# record_failure(session_id, failure) -> extra metric fields (or None)
RecordFailure = Callable[[str, SessionFailure], Awaitable[Optional[Dict[str, Any]]]]


class _SessionFailed(Exception):
    """The session cannot be distilled in this run (see ``failure``).

    ``noted`` — the failed attempt behind it is already in the session's
    ``failure_kinds`` (every failed LLM attempt is noted where it happens)."""

    def __init__(self, failure: SessionFailure, *, noted: bool = False):
        super().__init__(failure.error)
        self.failure = failure
        self.noted = noted


class _HalveBudget(Exception):
    """A truncated answer or an over-budget input: shrink the chunks."""

    def __init__(self, kind: ErrorKind, error: BaseException):
        super().__init__(f"{kind.value}: {error}")
        self.kind = kind
        self.error = error


class _DistillFailed(Exception):
    """Every experience of a session failed to persist (legacy persistence
    path without ``commit_session``): the session is left undistilled."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(session_id)


@dataclass
class _Outcome:
    processed: bool
    experiences: List[PydanticSkillExperience] = field(default_factory=list)


@dataclass
class _RunContext:
    meta_agent_state: AgentState
    user: PydanticUser
    actor: PydanticClient
    call_gate: asyncio.Semaphore
    recorder: Optional[RetryEventRecorder]
    commit_session: Optional[CommitSession]
    record_failure: Optional[RecordFailure]


def _failure_for(
    kind: ErrorKind, error: BaseException, *, note: str = ""
) -> SessionFailure:
    """The retry-policy consequence of a failure left after the in-run retries."""
    policy = RETRY_POLICIES.get(kind)
    deferred = bool(policy and policy.defer_across_runs)
    counted = bool(policy and policy.counts_as_failure) and not deferred
    text = f"{type(error).__name__}: {error}"
    return SessionFailure(
        kind=kind,
        counted=counted,
        deferred=deferred,
        error=(f"{note}: {text}" if note else text)[:1000],
    )


def _new_metrics(session_id: str) -> Dict[str, Any]:
    """One ``distill_metrics`` entry."""
    return {
        "session_id": session_id,
        "status": "pending",
        # size
        "messages": 0,
        "rounds": 0,
        "raw_tokens": 0,
        "images": 0,
        "images_unavailable": 0,
        "tool_errors": 0,
        "empty_results": 0,
        "chunks": 0,
        "chunk_budget_tokens": None,
        "compact_frame": False,
        "halvings": 0,
        # truncation
        "truncated_messages": 0,
        "truncated_tokens": 0,
        "dropped_images": 0,
        # calls
        "llm_calls": 0,
        "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "repair_prompts": 0,
        "timeouts": 0,
        "failure_kinds": [],
        # output
        "experiences": 0,
        # the outcome each chunk's answer stated, in chunk order (None: none)
        "outcomes": [],
        # the one outcome stored with the session's experiences
        # (SessionVerdict: checked against the transcript, chunks reconciled)
        "session_outcome": None,
        # chunks whose succeeded / failed outcome no quote in the transcript
        # showed (stored as unclear), or only the agent's own messages did
        "outcomes_unverified": 0,
        "outcomes_self_reported": 0,
        "invalid_items": 0,
        # The session context shown and the served skills' feedback
        "served_skills": 0,
        "served_skills_hidden": 0,
        "agent_context_chars": 0,
        "context_tokens": 0,
        "context_truncated_tokens": 0,
        "skill_feedback": None,
        "fields_truncated": 0,
        "field_chars_truncated": 0,
        "credibility_capped": 0,
        "error": None,
    }


def _note_kind(metrics: Dict[str, Any], kind: ErrorKind) -> None:
    metrics["failure_kinds"].append(kind.value)


def _usage_numbers(response: Any) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        get = usage.get
    else:

        def get(key, default=None):
            return getattr(usage, key, default)

    def number(key: str) -> int:
        value = get(key, 0)
        return (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else 0
        )

    prompt, completion = number("prompt_tokens"), number("completion_tokens")
    total = number("total_tokens") or prompt + completion
    return prompt, completion, total


def _response_text(response: Any) -> Optional[str]:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError, KeyError):
        return None
    return content if isinstance(content, str) else None


def _cap_text(text: str, cap: int) -> Tuple[str, int]:
    """``text`` within ``cap`` characters: head + tail around an explicit
    marker. Returns ``(text, characters removed)``."""
    if len(text) <= cap:
        return text, 0
    removed = len(text) - cap
    while True:
        marker = f"…[truncated {removed} chars]…"
        room = cap - len(marker)
        if room <= 0:
            return text[:cap], len(text) - cap
        head = room - room // 2
        tail = room // 2
        actual_removed = len(text) - head - tail
        if actual_removed == removed:
            return text[:head] + marker + (
                text[len(text) - tail :] if tail else ""
            ), removed
        removed = actual_removed


def _echo(content: Optional[str]) -> str:
    if not content or not content.strip():
        return "(empty reply)"
    text, _ = _cap_text(content, REPAIR_ECHO_MAX_CHARS)
    return text


async def _gather_or_cancel(coros: Sequence[Awaitable[Any]]) -> List[Any]:
    """Await all ``coros`` concurrently; on the first failure cancel the rest
    and raise it (in task order among the failed). Cancelling the caller
    cancels them all."""
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    if not tasks:
        return []
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        return [task.result() for task in tasks]
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in tasks:  # retrieve every exception (no "never retrieved")
            if task.done() and not task.cancelled():
                task.exception()


# One gate per event loop and limit: an asyncio.Semaphore belongs to the loop
# it is used on.
_CALL_GATES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[int, asyncio.Semaphore]]" = weakref.WeakKeyDictionary()


def distiller_call_gate(limit: Optional[int] = None) -> asyncio.Semaphore:
    """The process-wide gate of distiller LLM calls: at most
    ``limit`` (default ``MIRIX_DISTILLER_MAX_CONCURRENCY``, 4) calls in
    flight in the whole process, whichever learning runs, sessions and chunks
    they belong to — concurrent runs of different users share it. A distiller
    constructed with another ``max_concurrency`` shares the gate of that
    limit."""
    size = max(1, int(limit or DISTILLER_MAX_CONCURRENCY))
    loop = asyncio.get_running_loop()
    gates = _CALL_GATES.get(loop)
    if gates is None:
        gates = {}
        _CALL_GATES[loop] = gates
    gate = gates.get(size)
    if gate is None:
        gate = gates[size] = asyncio.Semaphore(size)
    return gate


def _load_distiller_prompt() -> str:
    """The general Experience-Distiller system prompt
    (``prompts/system/base/auto_dream_agent/procedural.txt``), via the cached
    system-prompt loader."""
    from mirix.prompts.gpt_system import get_system_text

    return get_system_text("base/auto_dream_agent/procedural")


class SessionExperienceDistiller:
    """Distill a learning run's claimed conversation sessions into experiences.

    Args:
        llm_client: an instance exposing ``async send_llm_request(messages=)``
            (an ``LLMClientBase``). Injected directly in tests; in production
            pass ``llm_config`` and one is created lazily.
        llm_config: an ``LLMConfig`` used to build the client when
            ``llm_client`` is not supplied; its ``context_window`` /
            ``max_output_tokens`` size the chunks.
        experience_manager: a :class:`SkillExperienceManager` (legacy
            persistence without ``commit_session``).
        conversation_manager: a :class:`ConversationMessageManager` — the
            source of the transcripts.
        max_concurrency: LLM calls (and sessions) in flight at once; defaults
            to ``MIRIX_DISTILLER_MAX_CONCURRENCY`` (4).
        token_counter: the :class:`TokenCounter` (default: the shared one).
        model_limits: the :class:`ModelLimits` to size chunks with (default:
            from ``llm_config`` or the client's config; none = no chunking).
        image_resolver: ``async (file_ids) -> available file ids`` (default:
            the ``files`` table and the image files on disk).
        session_context_loader: reads a session's context — the skills
            served to the agent and its agent context — with an
            ``async load(organization_id=, user_id=, session_id=,
            watermark=, scopes=)`` (a
            :class:`~mirix.services.distill_session_context.SessionContextLoader`;
            a learning run passes one). None: no session context is shown
            and no skill feedback asked for.
    """

    def __init__(
        self,
        *,
        llm_client=None,
        llm_config=None,
        experience_manager: Optional[SkillExperienceManager] = None,
        conversation_manager: Optional[ConversationMessageManager] = None,
        max_concurrency: Optional[int] = None,
        token_counter: Optional[TokenCounter] = None,
        model_limits: Optional[ModelLimits] = None,
        image_resolver: Optional[Callable[[List[str]], Awaitable[Set[str]]]] = None,
        session_context_loader: Optional[Any] = None,
    ):
        self._session_context_loader = session_context_loader
        self._max_concurrency = max(
            1, int(max_concurrency or DISTILLER_MAX_CONCURRENCY)
        )
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._experience_manager = experience_manager
        self._conversation_manager = conversation_manager
        self._counter = token_counter or get_token_counter()
        self._model_limits = model_limits
        self._image_resolver = image_resolver
        self._system_prompt = _load_distiller_prompt()

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    async def distill_sessions(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_ids: List[str],
        session_watermarks: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        commit_session: Optional[CommitSession] = None,
        record_failure: Optional[RecordFailure] = None,
        recorder: Optional[RetryEventRecorder] = None,
    ) -> Tuple[List[PydanticSkillExperience], List[str]]:
        """Distill each session (concurrently) into persisted experiences.

        ``commit_session(session_id, experiences)`` — when given (a learning
        run passes ``SkillLearningRunManager.commit_distilled_session``) —
        persists a session's experience payloads AND marks its rows distilled
        in one transaction; a session is "processed" exactly when its commit
        succeeded. Without it the experiences are persisted one by one and the
        caller marks the rows.

        ``record_failure(session_id, failure)`` settles a session that could
        not be distilled: back it off (``failure.deferred``) or count the
        failure (``failure.counted``); what it returns is added to the
        session's metrics. ``recorder`` receives every retry event.

        ``session_watermarks`` bounds each session's transcript to its
        not-yet-distilled rows at or before the watermark; a session
        missing from it reads all of its not-yet-distilled rows. ``metrics`` is
        filled with one metrics entry per session.

        A :class:`LearningRunNotActive` (the run was reclaimed or erased), a
        database failure left after its retries, or cancellation stops the
        whole call: the other sessions are cancelled and the error is raised.

        Returns ``(experiences, processed_session_ids)``.
        """
        if not session_ids:
            return [], []
        if self._counter is get_token_counter():
            # Exact counts for the chunk plans when the vocabulary can be
            # loaded (a failed startup preload is retried here).
            await ensure_tokenizer()
        watermarks = session_watermarks or {}
        per_session = {sid: _new_metrics(sid) for sid in session_ids}
        ctx = _RunContext(
            meta_agent_state=meta_agent_state,
            user=user,
            actor=actor,
            call_gate=distiller_call_gate(self._max_concurrency),
            recorder=recorder,
            commit_session=commit_session,
            record_failure=record_failure,
        )
        session_gate = asyncio.Semaphore(self._max_concurrency)

        async def _bounded(sid: str) -> _Outcome:
            async with session_gate:
                # Retries made below without an explicit recorder (the SDK's
                # transport retries) are recorded with this session, and
                # its model calls serve it (MIRIX_LLM_SESSION_HEADER; each
                # session's task has its own context).
                with (
                    conversation_session_scope(sid),
                    recording_retries(recorder, stage="distill", session_id=sid),
                ):
                    return await self._distill_one(
                        ctx, sid, watermarks.get(sid), per_session[sid]
                    )

        outcomes = await _gather_or_cancel([_bounded(sid) for sid in session_ids])
        created: List[PydanticSkillExperience] = []
        processed_session_ids: List[str] = []
        for sid, outcome in zip(session_ids, outcomes):
            if outcome.processed:
                processed_session_ids.append(sid)
                created.extend(outcome.experiences)
        if metrics is not None:
            metrics.update(per_session)
        logger.info(
            "Distilled %d session(s) → %d experience(s); %d processed, %d not",
            len(session_ids),
            len(created),
            len(processed_session_ids),
            len(session_ids) - len(processed_session_ids),
        )
        return created, processed_session_ids

    # ------------------------------------------------------------------ #
    # One session                                                          #
    # ------------------------------------------------------------------ #

    async def _distill_one(
        self,
        ctx: _RunContext,
        session_id: str,
        watermark: Optional[Any],
        metrics: Dict[str, Any],
    ) -> _Outcome:
        """Read, distill and commit one session; settle it on failure.

        Returns the outcome; raises only what must stop the run
        (:class:`LearningRunNotActive`, a database failure left after its
        retries, cancellation)."""
        feedback: List[Dict[str, Any]] = []
        try:
            rows = await retry_call(
                lambda: self._get_conversation_manager().list_turns_for_session(
                    session_id=session_id,
                    user_id=ctx.user.id,
                    organization_id=_owner_org(ctx.actor),
                    actor=ctx.actor,
                    watermark=watermark,
                    undistilled_only=True,
                ),
                stage="distill_read",
                session_id=session_id,
                recorder=ctx.recorder,
            )
            metrics["messages"] = len(rows)
            items = await self._distill_rows(
                ctx,
                session_id,
                rows,
                metrics,
                watermark=watermark,
                feedback_out=feedback,
            )
        except LearningRunNotActive:
            raise
        except _SessionFailed as failed:
            return await self._settle_failure(
                ctx, session_id, failed.failure, metrics, noted=failed.noted
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            kind = classify_error(exc)
            if kind is ErrorKind.DB:
                raise  # the run fails; its sessions are released
            return await self._settle_failure(
                ctx, session_id, _failure_for(kind, exc), metrics
            )

        payloads = self._experience_payloads(
            meta_agent_state=ctx.meta_agent_state,
            user=ctx.user,
            actor=ctx.actor,
            session_id=session_id,
            parsed=items,
            metrics=metrics,
        )
        try:
            if ctx.commit_session is not None:
                commit = ctx.commit_session
                experiences, _marked = await retry_call(
                    # The feedback rows go with the same commit.
                    (lambda: commit(session_id, payloads, skill_feedback=feedback))
                    if feedback
                    else (lambda: commit(session_id, payloads)),
                    stage="distill_commit",
                    session_id=session_id,
                    recorder=ctx.recorder,
                )
            else:
                experiences = await self._persist_payloads(
                    payloads, session_id, ctx.actor
                )
                if feedback:
                    await self._persist_feedback(feedback, ctx.actor)
        except LearningRunNotActive:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            kind = classify_error(exc)
            if kind is ErrorKind.DB:
                raise
            return await self._settle_failure(
                ctx, session_id, _failure_for(kind, exc, note="commit"), metrics
            )
        metrics["status"] = "ok"
        metrics["experiences"] = len(experiences)
        return _Outcome(processed=True, experiences=list(experiences))

    async def _settle_failure(
        self,
        ctx: _RunContext,
        session_id: str,
        failure: SessionFailure,
        metrics: Dict[str, Any],
        *,
        noted: bool = False,
    ) -> _Outcome:
        """Record why the session was not distilled (metrics, and — through
        ``record_failure`` — its backoff or its counted failure)."""
        if not noted:
            _note_kind(metrics, failure.kind)
        if failure.deferred:
            status = "deferred"
        elif failure.kind is ErrorKind.INTERNAL:
            status = STATUS_INTERNAL_ERROR
        else:
            status = "failed"
        metrics.update(
            status=status,
            failure_kind=failure.kind.value,
            counted_failure=failure.counted,
            error=failure.error[:500],
        )
        if failure.config_deferral is not None:
            metrics["config_deferral"] = failure.config_deferral
        logger.warning(
            "Session %s not distilled (%s, %s): %s",
            session_id,
            failure.kind.value,
            "deferred"
            if failure.deferred
            else ("counted" if failure.counted else "not counted"),
            failure.error,
        )
        if ctx.record_failure is not None:
            settle = ctx.record_failure
            extra = await retry_call(
                lambda: settle(session_id, failure),
                stage="distill_settle",
                session_id=session_id,
                recorder=ctx.recorder,
            )
            if extra:
                metrics.update(extra)
                if extra.get("failed_permanently"):
                    metrics["status"] = "failed_permanently"
        return _Outcome(processed=False)

    async def _session_context(
        self,
        ctx: _RunContext,
        session_id: str,
        watermark: Optional[Any],
        metrics: Dict[str, Any],
    ) -> Optional[Any]:
        """The session's context: the skills served to the agent and
        its agent context, read with the session context loader (None without
        one). A database failure left after its retries stops the run like
        any other; any other failure is MIRIX's own: the session stays
        pending, uncounted, and the run fails with it."""
        loader = self._session_context_loader
        if loader is None:
            return None
        read_scopes = getattr(ctx.actor, "read_scopes", None)
        try:
            context = await retry_call(
                lambda: loader.load(
                    organization_id=_owner_org(ctx.actor),
                    user_id=ctx.user.id,
                    session_id=session_id,
                    watermark=watermark,
                    # The skills shown are those the run's client may read.
                    scopes=list(read_scopes or []),
                ),
                stage="distill_context",
                session_id=session_id,
                recorder=ctx.recorder,
            )
        except LearningRunNotActive:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            kind = classify_error(exc)
            if kind is ErrorKind.DB:
                raise
            raise _SessionFailed(
                SessionFailure(
                    kind=ErrorKind.INTERNAL,
                    counted=False,
                    deferred=False,
                    error=(
                        "could not read the session's context: "
                        f"{type(exc).__name__}: {exc}"
                    )[:1000],
                )
            ) from exc
        served = tuple(getattr(context, "served", ()) or ())
        metrics.update(
            served_skills=len(served),
            served_skills_hidden=int(getattr(context, "served_hidden", 0) or 0),
            agent_context_chars=len(getattr(context, "agent_context", None) or ""),
        )
        return context

    async def _distill_rows(
        self,
        ctx: _RunContext,
        session_id: str,
        rows: Sequence[Any],
        metrics: Dict[str, Any],
        *,
        watermark: Optional[Any] = None,
        feedback_out: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """The parsed experience items of a session's rows ([] for no rows);
        the served skills' feedback rows go to ``feedback_out``."""
        context = (
            await self._session_context(ctx, session_id, watermark, metrics)
            if rows
            else None
        )
        transcript_context = TranscriptContext.of(context)
        served = tuple(getattr(context, "served", ()) or ())
        image_ids = row_image_ids(rows)
        available: Optional[Set[str]] = None
        if image_ids:
            available = await retry_call(
                lambda: self._available_images(image_ids),
                stage="distill_images",
                session_id=session_id,
                recorder=ctx.recorder,
            )
        large = plan_text_size(rows) > LARGE_TEXT_CHARS

        def build() -> TranscriptPlanner:
            messages, unavailable = build_messages(
                rows, counter=self._counter, available_images=available
            )
            return TranscriptPlanner(
                messages,
                agent_id=ctx.meta_agent_state.id,
                session_id=session_id,
                counter=self._counter,
                images_unavailable=unavailable,
                context=transcript_context,
            )

        planner = await asyncio.to_thread(build) if large else build()
        stats = planner.stats
        metrics.update(
            rounds=stats.rounds,
            raw_tokens=planner.raw_tokens(),
            images=stats.images,
            images_unavailable=stats.images_unavailable,
            tool_errors=stats.tool_errors,
            empty_results=stats.empty_results,
        )
        if planner.empty:
            return []
        return await self._distill_with_halving(
            ctx,
            session_id,
            planner,
            large,
            metrics,
            served=served,
            feedback_out=feedback_out,
        )

    async def _distill_with_halving(
        self,
        ctx: _RunContext,
        session_id: str,
        planner: TranscriptPlanner,
        large: bool,
        metrics: Dict[str, Any],
        *,
        served: Sequence[Any] = (),
        feedback_out: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Distill with the session's chunk budget, halving it (at most
        :data:`MAX_BUDGET_HALVINGS` times) on a truncated answer or an
        over-budget input."""
        whole = planner.whole_tokens()
        base = self._input_budget()
        if base is not None and base < MIN_CHUNK_BUDGET_TOKENS and whole > base:
            # The distiller's own prompt and the output reserve (nearly) fill
            # the usable window: no chunk of this session can be sent with
            # this model configuration. A configuration error, not the
            # session's fault: it backs off, never counted — like the
            # curator's context_window_too_small skip.
            raise _SessionFailed(
                SessionFailure(
                    kind=ErrorKind.CONFIG,
                    counted=False,
                    deferred=True,
                    error=(
                        f"the distiller's input budget is {base} tokens "
                        f"(< {MIN_CHUNK_BUDGET_TOKENS}): the context window "
                        "cannot hold the distiller prompt, the output reserve "
                        "and any chunk of the session"
                    ),
                    config_deferral=DISTILL_DEFERRAL_INPUT_BUDGET_TOO_SMALL,
                )
            )
        start = whole if base is None else min(base, whole)
        last: Optional[_HalveBudget] = None
        for halving in range(MAX_BUDGET_HALVINGS + 1):
            budget = start // (2**halving)
            if halving:
                metrics["halvings"] = halving
                assert last is not None
                await self._record_retry(
                    ctx, last.kind, halving, session_id, last.error
                )
            try:
                plan = (
                    await asyncio.to_thread(planner.plan, budget)
                    if large
                    else planner.plan(budget)
                )
            except Exception as exc:  # noqa: BLE001 — classified below
                kind = classify_error(exc)
                if kind is not ErrorKind.BUDGET:
                    raise
                last = _HalveBudget(kind, exc)
                _note_kind(metrics, kind)
                continue
            self._note_plan(metrics, plan, budget if plan.chunked else None)
            try:
                return await self._distill_plan(
                    ctx,
                    session_id,
                    plan,
                    metrics,
                    planner=planner,
                    large=large,
                    served=served,
                    feedback_out=feedback_out,
                )
            except _HalveBudget as halve:
                last = halve  # noted where the attempt failed
        assert last is not None
        raise _SessionFailed(
            SessionFailure(
                kind=last.kind,
                counted=True,
                deferred=False,
                error=(
                    f"still {last.kind.value} after {MAX_BUDGET_HALVINGS} halvings of "
                    f"the chunk budget: {type(last.error).__name__}: {last.error}"
                )[:1000],
            ),
            noted=True,
        )

    @staticmethod
    def _note_plan(
        metrics: Dict[str, Any], plan: TranscriptPlan, budget: Optional[int]
    ) -> None:
        metrics.update(
            chunks=len(plan.chunks),
            chunk_budget_tokens=budget,
            compact_frame=plan.compact_frame,
            truncated_messages=plan.truncated_messages,
            truncated_tokens=plan.truncated_tokens,
            dropped_images=plan.dropped_images,
            context_tokens=plan.context_tokens,
            context_truncated_tokens=plan.context_truncated_tokens,
        )

    async def _distill_plan(
        self,
        ctx: _RunContext,
        session_id: str,
        plan: TranscriptPlan,
        metrics: Dict[str, Any],
        *,
        planner: Optional[TranscriptPlanner] = None,
        large: bool = False,
        served: Sequence[Any] = (),
        feedback_out: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Every chunk's items, in chunk order (no dedupe), each carrying
        the session's one outcome. With ``served`` skills, their
        feedback — each chunk's entries checked against what it was shown
        (:func:`chunk_feedback`), then reconciled — goes to ``feedback_out``
        as one row per served skill (:func:`feedback_rows`). Each chunk's stated outcome is checked
        against the messages of ``planner``'s transcript that chunk was shown
        (:func:`chunk_verdicts`; None: not checked) and the chunks' verdicts
        are reconciled (:func:`reconcile_verdicts`). The stated outcomes go to
        ``metrics["outcomes"]`` in chunk order, the stored one to
        ``metrics["session_outcome"]``. The first failing chunk cancels its
        siblings: the whole plan is redone (smaller) or the session fails, so
        partial results are never kept."""
        if len(plan.chunks) > 1:
            logger.info(
                "Session %s is distilled in %d chunks of <= %s tokens",
                session_id,
                len(plan.chunks),
                plan.budget,
            )
        replies = await _gather_or_cancel(
            [
                self._distill_chunk(ctx, session_id, chunk, metrics)
                for chunk in plan.chunks
            ]
        )
        metrics["outcomes"] = [reply.outcome for reply in replies]
        verdicts = (
            await asyncio.to_thread(chunk_verdicts, replies, plan, planner)
            if large
            else chunk_verdicts(replies, plan, planner)
        )
        for verdict in verdicts:
            if verdict.source == OUTCOME_SOURCE_NOT_FOUND:
                metrics["outcomes_unverified"] = (
                    metrics.get("outcomes_unverified", 0) + 1
                )
            elif verdict.source == OUTCOME_SOURCE_AGENT:
                metrics["outcomes_self_reported"] = (
                    metrics.get("outcomes_self_reported", 0) + 1
                )
        session = reconcile_verdicts(verdicts)
        metrics["session_outcome"] = session.outcome
        if served and feedback_out is not None:
            names = [skill.name for skill in served]
            per_chunk = (
                await asyncio.to_thread(chunk_feedback, replies, plan, planner, names)
                if large
                else chunk_feedback(replies, plan, planner, names)
            )
            rows = feedback_rows(
                served,
                per_chunk,
                session=session,
                organization_id=_owner_org(ctx.actor),
                user_id=ctx.user.id,
                agent_id=ctx.meta_agent_state.id,
                session_id=session_id,
            )
            feedback_out[:] = rows
            visible = plan.context_visible_skills
            metrics["skill_feedback"] = {
                "served": len(served),
                # Served skills whose header the context cut left out (their
                # rows are unclear: the distiller never saw them).
                "not_shown": 0 if visible is None else len(served) - len(visible),
                "given": len({pos for chunk in per_chunk for pos in chunk}),
                "followed_unverified": sum(1 for r in rows if r["stated_followed"]),
                "effect_unverified": sum(1 for r in rows if r["stated_effect"]),
                # Each served skill's stored values (the rows' own; their evidence
                # stays in skill_session_feedback): a run's record shows them
                # without a database read.
                "skills": [
                    {
                        "skill_id": r["skill_id"],
                        "name": r["skill_name"],
                        "version_no": r["skill_version_no"],
                        "followed": r["followed"],
                        "effect": r["effect"],
                    }
                    for r in rows
                ],
            }
        return [
            _with_session_outcome(item, session)
            for reply in replies
            for item in reply.items
        ]

    # ------------------------------------------------------------------ #
    # One chunk: the LLM dialogue                                          #
    # ------------------------------------------------------------------ #

    async def _distill_chunk(
        self,
        ctx: _RunContext,
        session_id: str,
        chunk: TranscriptChunk,
        metrics: Dict[str, Any],
    ) -> DistillerReply:
        """One chunk's answer: the outcome it stated and its valid experience
        items (the session's outcome is stamped on them by
        :meth:`_distill_plan`, once every chunk answered). An unparseable
        answer gets a repair prompt appended to the same dialogue, at most
        :data:`MAX_REPAIR_PROMPTS` times."""
        client = self._get_client()
        if client is None:
            raise _SessionFailed(
                SessionFailure(
                    kind=ErrorKind.CONFIG,
                    counted=False,
                    deferred=True,
                    error="no LLM client or LLM config for the distiller",
                    config_deferral=DISTILL_DEFERRAL_NO_LLM_CLIENT,
                )
            )
        agent_id = ctx.meta_agent_state.id
        dialogue: List[Message] = [
            Message(
                agent_id=agent_id,
                role=MessageRole.system,
                content=[TextContent(text=self._system_prompt)],
            ),
            Message(
                agent_id=agent_id, role=MessageRole.user, content=list(chunk.content)
            ),
        ]
        for repair in range(MAX_REPAIR_PROMPTS + 1):
            response = await self._send(ctx, client, dialogue, session_id, metrics)
            content = _response_text(response)
            reply = parse_distiller_reply(content)
            reason = _REASON_UNUSABLE
            if reply is not None:
                items = reply.items
                valid = [item for item in items if self._is_valid_item(item)]
                if valid or not items:
                    metrics["invalid_items"] += len(items) - len(valid)
                    return replace(reply, items=valid)
                reason = _REASON_NO_VALID_ITEM
            _note_kind(metrics, ErrorKind.PARSE)
            if repair == MAX_REPAIR_PROMPTS:
                raise _SessionFailed(
                    SessionFailure(
                        kind=ErrorKind.PARSE,
                        counted=True,
                        deferred=False,
                        error=(
                            f"unusable answer after {MAX_REPAIR_PROMPTS} repair "
                            f"prompts ({reason}): {(content or '')[:300]!r}"
                        ),
                    ),
                    noted=True,
                )
            metrics["repair_prompts"] += 1
            await self._record_retry(
                ctx,
                ErrorKind.PARSE,
                repair + 1,
                session_id,
                LLMJSONParsingError(reason),
            )
            dialogue = dialogue + [
                Message(
                    agent_id=agent_id,
                    role=MessageRole.assistant,
                    content=[TextContent(text=_echo(content))],
                ),
                Message(
                    agent_id=agent_id,
                    role=MessageRole.user,
                    content=[TextContent(text=REPAIR_PROMPT.format(reason=reason))],
                ),
            ]
        raise AssertionError("unreachable")  # pragma: no cover

    async def _send(
        self,
        ctx: _RunContext,
        client: Any,
        dialogue: List[Message],
        session_id: str,
        metrics: Dict[str, Any],
    ) -> Any:
        """One LLM call through the run's concurrency gate. A timed-out call is
        repeated once (``retry_call``); every other failure is mapped to its
        retry-policy consequence."""

        async def attempt() -> Any:
            async with ctx.call_gate:
                try:
                    response = await client.send_llm_request(messages=dialogue)
                except ContextBudgetExceeded:
                    # Refused before sending: a failed attempt, not a call.
                    _note_kind(metrics, ErrorKind.BUDGET)
                    raise
                except Exception as exc:  # noqa: BLE001 — noted, then re-raised
                    # The metrics count the calls actually made: a request MIRIX
                    # could not build, or one the SDK refused before sending
                    # it (LLMRequestBuildError), never reached the model.
                    if not isinstance(exc, LLMRequestBuildError):
                        metrics["llm_calls"] += 1
                    kind = classify_error(exc)
                    _note_kind(metrics, kind)
                    if kind is ErrorKind.TIMEOUT:
                        metrics["timeouts"] += 1
                    raise
                metrics["llm_calls"] += 1
                return response

        try:
            response = await retry_call(
                attempt, stage="distill", session_id=session_id, recorder=ctx.recorder
            )
        except LearningRunNotActive:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            kind = classify_error(exc)
            if kind in (ErrorKind.TRUNCATED, ErrorKind.BUDGET):
                raise _HalveBudget(kind, exc) from exc
            if kind is ErrorKind.DB:
                raise
            raise _SessionFailed(_failure_for(kind, exc), noted=True) from exc
        prompt, completion, total = _usage_numbers(response)
        usage = metrics["llm_usage"]
        usage["prompt_tokens"] += prompt
        usage["completion_tokens"] += completion
        usage["total_tokens"] += total
        try:
            raise_if_truncated(response)
        except LLMOutputTruncatedError as exc:
            # The truncated answer is never used: shrink and retry.
            _note_kind(metrics, ErrorKind.TRUNCATED)
            raise _HalveBudget(ErrorKind.TRUNCATED, exc) from exc
        return response

    async def _record_retry(
        self,
        ctx: _RunContext,
        kind: ErrorKind,
        attempt: int,
        session_id: str,
        error: BaseException,
    ) -> None:
        logger.info(
            "Session %s: %s, retry %d (%s)", session_id, kind.value, attempt, error
        )
        if ctx.recorder is not None:
            await ctx.recorder.record(
                kind, attempt, 0.0, stage="distill", session_id=session_id, error=error
            )

    # ------------------------------------------------------------------ #
    # Budget, client, images                                               #
    # ------------------------------------------------------------------ #

    def _limits(self) -> Optional[ModelLimits]:
        if self._model_limits is not None:
            return self._model_limits
        config = self._llm_config or getattr(self._llm_client, "llm_config", None)
        window = getattr(config, "context_window", None)
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            return None
        try:
            return ModelLimits.from_llm_config(config)
        except (AttributeError, TypeError, ValueError):
            return None

    def _input_budget(self) -> Optional[int]:
        """``input_budget(system prompt, planning reserve)`` for the user
        message, or None when the model's limits are unknown."""
        limits = self._limits()
        if limits is None:
            return None
        system_tokens = self._counter.count_text(self._system_prompt) + (
            2 * MESSAGE_OVERHEAD_TOKENS
        )
        return max(0, limits.input_budget(system_tokens))

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

    async def _available_images(self, file_ids: List[str]) -> Set[str]:
        """The ids whose image can be sent: a ``files`` row with a remote URL,
        or with a local file that exists. A missing one is rendered as an
        ``[image unavailable: …]`` text instead of failing the session."""
        if self._image_resolver is not None:
            return set(await self._image_resolver(list(file_ids)))
        from sqlalchemy.exc import NoResultFound

        from mirix.services.file_manager import FileManager

        manager = FileManager()
        available: Set[str] = set()
        for file_id in file_ids:
            try:
                meta = await manager.get_file_metadata_by_id(file_id)
            except NoResultFound:
                continue
            if meta.source_url:
                available.add(file_id)
            elif meta.file_path and await asyncio.to_thread(
                os.path.exists, meta.file_path
            ):
                available.add(file_id)
        return available

    def _get_experience_manager(self) -> SkillExperienceManager:
        if self._experience_manager is None:
            self._experience_manager = SkillExperienceManager()
        return self._experience_manager

    def _get_conversation_manager(self) -> ConversationMessageManager:
        if self._conversation_manager is None:
            self._conversation_manager = ConversationMessageManager()
        return self._conversation_manager

    # ------------------------------------------------------------------ #
    # Experience payloads                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_valid_item(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and item.get("experience_type") in _VALID_EXPERIENCE_TYPES
            and bool(str(item.get("title") or "").strip())
        )

    def _experience_payloads(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        parsed: List[Dict],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """The valid parsed items as ``SkillExperienceManager.build_row``
        kwargs. An unknown type or an empty title is skipped; a field past its
        cap keeps its head and tail with an explicit marker, counted in
        ``metrics`` (``fields_truncated`` / ``field_chars_truncated``). The
        credibility of an ``inferred`` lesson is capped at
        :data:`INFERRED_CREDIBILITY_MAX`, that of a ``task_outcome`` lesson whose
        outcome only the agent's own messages show at
        :data:`SELF_REPORTED_OUTCOME_CREDIBILITY_MAX` (both counted as
        ``credibility_capped``)."""
        payloads: List[Dict[str, Any]] = []
        for item in parsed:
            if not self._is_valid_item(item):
                continue
            title, title_cut = _cap_text(
                str(item.get("title") or "").strip(), _TITLE_CAP
            )
            raw_content = item.get("content")
            content_text = (
                raw_content
                if isinstance(raw_content, str)
                else (
                    ""
                    if raw_content is None
                    else json.dumps(raw_content, ensure_ascii=False, default=str)
                )
            )
            content, content_cut = _cap_text(content_text, _CONTENT_CAP)
            evidence, evidence_cut, signal, source = self._evidence_json(
                item.get("evidence")
            )
            cuts = [c for c in (title_cut, content_cut, evidence_cut) if c]
            if metrics is not None and cuts:
                metrics["fields_truncated"] += len(cuts)
                metrics["field_chars_truncated"] += sum(cuts)
            credibility = _clamp01(item.get("credibility"))
            cap = None
            if signal == "inferred":
                cap = INFERRED_CREDIBILITY_MAX
            elif signal == "task_outcome" and source == OUTCOME_SOURCE_AGENT:
                cap = SELF_REPORTED_OUTCOME_CREDIBILITY_MAX
            if cap is not None and credibility > cap:
                credibility = cap
                if metrics is not None:
                    metrics["credibility_capped"] = (
                        metrics.get("credibility_capped", 0) + 1
                    )
            payloads.append(
                {
                    "agent_id": meta_agent_state.id,
                    "user_id": user.id,
                    "organization_id": _owner_org(actor),
                    "session_id": session_id,
                    "experience_type": item["experience_type"],
                    "title": title,
                    "content": content,
                    "importance": _clamp01(item.get("importance")),
                    "credibility": credibility,
                    "evidence": evidence,
                    "status": "pending",
                }
            )
        return payloads

    async def _persist_payloads(
        self, payloads: List[Dict[str, Any]], session_id: str, actor: PydanticClient
    ) -> List[PydanticSkillExperience]:
        """Persist payloads one by one (no ``commit_session``). One bad row
        does not drop the rest; if EVERY attempt failed the store is down and
        :class:`_DistillFailed` leaves the session undistilled."""
        mgr = self._get_experience_manager()
        created: List[PydanticSkillExperience] = []
        errors = 0
        for payload in payloads:
            try:
                created.append(
                    await mgr.create_experience(
                        **payload, created_by_id=getattr(actor, "id", None)
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one bad row mustn't drop the rest
                errors += 1
                logger.warning(
                    "Failed to persist experience %r (session %s): %s",
                    payload.get("title"),
                    session_id,
                    exc,
                )
        if payloads and not created and errors == len(payloads):
            raise _DistillFailed(session_id)
        return created

    async def _persist_feedback(
        self, rows: List[Dict[str, Any]], actor: PydanticClient
    ) -> None:
        """Persist the served skills' feedback rows (no ``commit_session``:
        the legacy path, best effort like its experiences)."""
        from mirix.services.skill_session_feedback import SkillSessionFeedbackManager

        try:
            await SkillSessionFeedbackManager().create_rows(
                rows, created_by_id=getattr(actor, "id", None)
            )
        except Exception as exc:  # noqa: BLE001 — the experiences are kept
            logger.warning("Failed to persist the skill feedback rows: %s", exc)

    async def _persist_experiences(
        self,
        *,
        meta_agent_state: AgentState,
        user: PydanticUser,
        actor: PydanticClient,
        session_id: str,
        parsed: List[Dict],
    ) -> List[PydanticSkillExperience]:
        """Validate ``parsed`` items and persist them one by one."""
        payloads = self._experience_payloads(
            meta_agent_state=meta_agent_state,
            user=user,
            actor=actor,
            session_id=session_id,
            parsed=parsed,
        )
        return await self._persist_payloads(payloads, session_id, actor)

    @classmethod
    def _evidence_json(cls, evidence: Any) -> Tuple[str, int, str, Optional[str]]:
        """``{quote, signal_type}`` — plus the session's ``session_outcome``,
        ``outcome_evidence``, ``outcome_source`` and ``stated_outcome`` when the
        evidence carries them — as JSON within the evidence cap. The outcome
        evidence keeps its head and tail past :data:`OUTCOME_EVIDENCE_MAX_CHARS`,
        the quote past the room left. The signal type is normalized
        (:func:`normalize_signal_type`; unknown: ``inferred``). A
        ``task_outcome`` signal needs a succeeded or failed session; otherwise
        it is ``inferred``. Returns ``(json, characters removed, signal,
        outcome_source)``."""
        extra: Dict[str, str] = {}
        extra_removed = 0
        if isinstance(evidence, dict):
            quote = evidence.get("quote")
            quote = "" if quote is None else str(quote)
            signal = normalize_signal_type(evidence.get("signal_type")) or "inferred"
            outcome = normalize_session_outcome(evidence.get("session_outcome"))
            if outcome is not None:
                extra["session_outcome"] = outcome
            why = _as_text(evidence.get("outcome_evidence"))
            if why:
                extra["outcome_evidence"], extra_removed = _cap_text(
                    why, OUTCOME_EVIDENCE_MAX_CHARS
                )
            if evidence.get("outcome_source") in _OUTCOME_SOURCES:
                extra["outcome_source"] = evidence["outcome_source"]
            stated = normalize_session_outcome(evidence.get("stated_outcome"))
            if stated is not None:
                extra["stated_outcome"] = stated
        elif isinstance(evidence, str) and evidence.strip():
            quote, signal = evidence, "inferred"
        else:
            quote, signal = "", "inferred"
        if signal == "task_outcome" and (
            extra.get("session_outcome") not in _EVIDENCED_OUTCOMES
        ):
            signal = "inferred"
        source = extra.get("outcome_source")

        def dump(text: str) -> str:
            return json.dumps(
                {"quote": text, "signal_type": signal, **extra}, ensure_ascii=False
            )

        encoded = dump(quote)
        if len(encoded) <= _EVIDENCE_CAP:
            return encoded, extra_removed, signal, source
        room = _EVIDENCE_CAP - len(dump(""))
        while room > 0:
            cut, removed = _cap_text(quote, room)
            encoded = dump(cut)
            if len(encoded) <= _EVIDENCE_CAP:
                return encoded, removed + extra_removed, signal, source
            room -= max(1, len(encoded) - _EVIDENCE_CAP)
        return dump(""), len(quote) + extra_removed, signal, source

    @classmethod
    def _normalize_evidence(cls, evidence: Any) -> str:
        """The model's ``evidence`` as a JSON string ``{quote, signal_type}``."""
        return cls._evidence_json(evidence)[0]
