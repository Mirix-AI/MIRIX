"""The retrieval input and its topic extraction.

``/memory/retrieve/conversation`` receives OpenAI-style messages. Before any
memory is searched, an LLM reads them once and extracts the search keywords
("topics") and a temporal expression. This module owns that first step.

Messages
    :func:`normalize_retrieval_messages` reads ``messages`` with the same parser
    as ``/memory/add`` (:func:`mirix.services.conversation_content.parse_messages`):
    a message's ``content`` may be a string, a list of parts or ``None`` (an
    assistant turn that only carries ``tool_calls``). Before, a string counted
    as "no text" and a list of parts raised ``TypeError`` inside the extractor
    (swallowed), so extraction never ran. Messages with neither text nor
    an image are a :class:`RetrievalInputError` (HTTP 422), never "the most
    recent memories".

Images
    Image parts stay images: the extraction LLM sees them WITH the text, in
    place. A request's own images are never stored — the extraction call reads
    them from temporary files that are removed as soon as it returns
    (:class:`ExtractionImageFiles`), so a retrieval writes nothing but its log
    row. A ``database_image_id`` part refers to an image MIRIX already stores
    (the caller's ownership is checked by the REST layer first).

Budget
    The extraction input — text and images (``85 + 170 x tiles`` tokens each) —
    must fit ``input_budget`` of the extraction model (``floor(window x 0.8)``
    minus the prompts, the tool schema and the planning output reserve). A
    larger input keeps its head and tail with an explicit marker; an image is
    never split (it is kept whole or left out). A small margin is kept free,
    and a client that still refuses the request before sending it
    (``ContextBudgetExceeded``) gets it cut again by the shortfall. What was
    cut is reported in :attr:`TopicExtraction.input_stats` and lands in the
    retrieval log.

Prompt
    The system prompt and the instruction are generic: what the user or the
    agent wants to accomplish, the key objects, the operations needed, and any
    temporal expression. Nothing in them names a benchmark or a task domain.

Answer
    The forced tool call answers each part in its own field: ``goal`` (the
    kind of task, without the conversation's names, numbers or other values),
    ``operations``, ``key_objects`` and ``temporal_expression``. The
    ``topics`` join goal, key objects and operations (the keyword-searched
    memory types look for the instance, its entities and values); the
    ``procedural_query`` joins goal and operations only: a skill's retrieval
    text (name, triggers, description) describes a kind of task, so the
    entities and values of one page or one request only pull its query
    vector away from the skill that fits. A legacy one-string ``topic``
    answer is still accepted (no procedural query: the caller then ranks
    skills with the topics). The temporal expression is when the things to
    recall happened, never a date the conversation is about.

Failure
    The LLM client retries transport failures itself. When extraction still
    fails — an error, no tool call, an empty topic — :func:`extract_topics`
    returns status ``fallback`` and the caller searches with the
    conversation's own text (:func:`plain_text`).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import string
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mirix.errors import ContextBudgetExceeded
from mirix.log import get_logger
from mirix.services.context_budget import (
    LARGE_TEXT_CHARS,
    MESSAGE_OVERHEAD_TOKENS,
    MIN_OUTPUT_TOKENS,
    UNKNOWN_IMAGE_TOKENS,
    ModelLimits,
    TokenCounter,
    get_token_counter,
    image_tokens,
)
from mirix.services.conversation_content import (
    ImagePart,
    MessageParseError,
    ParsedMessage,
    PassthroughPart,
    TextPart,
    _decorate,
    parse_messages,
)
from mirix.services.distill_transcript import (
    MIN_CHUNK_BUDGET_TOKENS,
    ImagePiece,
    Piece,
    TextPiece,
    _coalesce,
    _pieces_tokens,
    head_tail_pieces,
)

logger = get_logger(__name__)

EXTRACTION_AGENT_ID = "topic_extraction"
EXTRACTION_FUNCTION_NAME = "update_topic_and_time"
TOPIC_SEPARATOR = ";"

EXTRACTION_STATUS_OK = "ok"
EXTRACTION_STATUS_FALLBACK = "fallback"
EXTRACTION_STATUS_SKIPPED = "skipped"

# Generic — no benchmark, dataset or task-domain vocabulary.
EXTRACTION_SYSTEM_PROMPT = (
    "You read a conversation between a user and an AI agent and extract the "
    "keywords a memory search needs to find knowledge relevant to it. You do "
    "not answer, continue or judge the conversation."
)
CONVERSATION_HEADER = "Conversation (in order):\n"
EXTRACTION_INSTRUCTION = (
    "Extract what a memory search needs from the conversation above, its text "
    "and any images:\n"
    "1. Goal: what the user or the agent wants to accomplish, stated as the "
    "kind of task: general enough to match instructions for any task of that "
    "kind, without the names, numbers, dates or other values of this "
    "conversation or its images.\n"
    "2. Key objects: the specific entities, items, names, values or places "
    "involved.\n"
    "3. Operations: the kinds of action, step or tool needed to accomplish "
    "the goal, without the names, numbers, dates or other values of this "
    "conversation (those are key objects).\n"
    "4. Temporal expression: a phrase saying when the things to recall "
    'happened, relative to now, such as "today", "yesterday" or "last week"; '
    "empty when the conversation only mentions dates as part of its subject.\n"
    f"Call `{EXTRACTION_FUNCTION_NAME}` with `goal`, `key_objects` and "
    f'`operations` (short phrases separated by "{TOPIC_SEPARATOR}"), and '
    "`temporal_expression` (empty if there is none)."
)
# The same extraction for a local text-only model (Ollama), answered in text.
LOCAL_MODEL_SYSTEM_PROMPT = EXTRACTION_SYSTEM_PROMPT
LOCAL_MODEL_INSTRUCTION = (
    "Extract what a memory search needs from the conversation above:\n"
    "1. Goal: what the user or the agent wants to accomplish.\n"
    "2. Key objects: the entities, items, names, values or places involved.\n"
    "3. Operations: the actions or steps needed to accomplish the goal.\n"
    f'Respond ONLY with these as short phrases separated by "{TOPIC_SEPARATOR}".'
)

# The answer's fields (all required: a strict schema requires every one).
EXTRACTION_FIELDS = ("goal", "operations", "key_objects", "temporal_expression")
EXTRACTION_TOOL: Dict[str, Any] = {
    "name": EXTRACTION_FUNCTION_NAME,
    "description": (
        "Record the search keywords extracted from the conversation and any "
        "temporal expression in it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the user or the agent wants to accomplish, stated as "
                    "the kind of task: general enough to match instructions for "
                    "any task of that kind, with no names, numbers, dates or "
                    "other values from the conversation or its images."
                ),
            },
            "operations": {
                "type": "string",
                "description": (
                    "The kinds of action, step or tool needed to accomplish the "
                    "goal, with no names, numbers, dates or other values from "
                    "the conversation (those are key objects), as short phrases "
                    f'separated by "{TOPIC_SEPARATOR}".'
                ),
            },
            "key_objects": {
                "type": "string",
                "description": (
                    "The specific entities, items, names, values or places "
                    f'involved, as short phrases separated by "{TOPIC_SEPARATOR}".'
                ),
            },
            "temporal_expression": {
                "type": "string",
                "description": (
                    "A phrase saying when the things to recall happened, "
                    'relative to now (e.g. "today", "yesterday", "last week"); '
                    "empty when the conversation only mentions dates as part of "
                    "its subject, or has no such phrase."
                ),
            },
        },
        "required": list(EXTRACTION_FIELDS),
    },
}

_EPHEMERAL_PREFIX = "retrieval-image-"
# Kept free below the extraction budget: a client serializes the request its
# own way (a strict tool schema, one part per text run), so its estimate runs
# a few tokens above ours.
EXTRACTION_BUDGET_MARGIN_TOKENS = 64
# A client that still refuses the request before sending
# (ContextBudgetExceeded) gets the input cut by the shortfall, at most this
# many times.
MAX_BUDGET_SHRINKS = 2


class RetrievalInputError(MessageParseError):
    """A retrieval ``messages`` payload the server cannot use (HTTP 422)."""


class TopicExtractionError(Exception):
    """Topic extraction could not produce topics (the caller falls back)."""


# --------------------------------------------------------------------------- #
# Messages                                                                     #
# --------------------------------------------------------------------------- #


def normalize_retrieval_messages(messages: Any) -> List[ParsedMessage]:
    """Parse a retrieval ``messages`` payload.

    Each message's ``content`` may be a string, a list of parts (text, inline
    or remote images, stored-file references) or ``None``; a legacy payload of
    bare parts without roles is one message. Raises
    :class:`~mirix.services.conversation_content.MessageParseError` for a
    payload the parser rejects and :class:`RetrievalInputError` when there is
    nothing to retrieve with (no message, or no text and no image anywhere).
    """
    if messages is None or (isinstance(messages, (list, tuple)) and not messages):
        raise RetrievalInputError("messages must be a non-empty list")
    parsed = parse_messages(messages)
    if not has_retrieval_content(parsed):
        raise RetrievalInputError(
            "messages contain no text and no image to retrieve memories with"
        )
    return parsed


def _message_texts(message: ParsedMessage) -> List[str]:
    """The texts of a message as the extractor sees them (tool calls and a
    tool result's tool name folded in)."""
    parts = _decorate(message, list(message.parts)) if message.role else message.parts
    return [p.text for p in parts if isinstance(p, TextPart)]


def has_retrieval_content(parsed: Sequence[ParsedMessage]) -> bool:
    """True when some message has non-blank text or an image."""
    for message in parsed:
        if any(isinstance(p, (ImagePart, PassthroughPart)) for p in message.parts):
            return True
        if any(text.strip() for text in _message_texts(message)):
            return True
    return False


def plain_text(parsed: Sequence[ParsedMessage]) -> str:
    """The conversation as plain text, one ``ROLE: text`` line per message
    with text (whitespace collapsed, images left out): the raw query when
    extraction fails."""
    lines: List[str] = []
    for message in parsed:
        text = " ".join(" ".join(_message_texts(message)).split())
        if text:
            role = (message.role or "user").upper()
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def fallback_query_text(parsed: Sequence[ParsedMessage], labeled: bool = False) -> str:
    """The conversation as the text of a search query when extraction fails:
    one line per message with text (whitespace collapsed, images left out).

    Without ``labeled`` the ``ROLE:`` labels of :func:`plain_text` are left
    out — they are not what the user asked about, and as words they would
    match every skill or memory mentioning a user or an assistant.
    ``labeled=True`` is :func:`plain_text` (the input of a local extraction
    model, which reads a transcript)."""
    if labeled:
        return plain_text(parsed)
    lines: List[str] = []
    for message in parsed:
        text = " ".join(" ".join(_message_texts(message)).split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def fallback_query(text: str, max_chars: int) -> Tuple[str, int]:
    """``text`` as a search query of at most ``max_chars`` characters: its head
    and tail (no marker — it would become a search term). Returns the query
    and the number of characters removed."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, 0
    head = max_chars // 2 + max_chars % 2
    tail = max_chars // 2
    return text[:head] + "\n" + (text[-tail:] if tail else ""), len(text) - head - tail


# --------------------------------------------------------------------------- #
# The extraction input (with its budget)                                       #
# --------------------------------------------------------------------------- #


@dataclass
class ExtractionImage:
    """An image of the extraction input: one of the request's own parts
    (sent from a temporary file) or a ``files`` row MIRIX already stores."""

    file_id: str
    part: Optional[ImagePart] = None

    @property
    def stored(self) -> bool:
        return self.part is None


@dataclass
class ExtractionInput:
    """The conversation as extraction content, cut to the budget."""

    pieces: List[Piece]
    images: Dict[str, ExtractionImage]
    stats: Dict[str, Any]


def _image_piece(file_id: str, part: Optional[ImagePart], detail: Any) -> ImagePiece:
    prepared = getattr(part, "prepared", None) if part is not None else None
    tokens = (
        image_tokens(prepared.width, prepared.height)
        if prepared is not None
        else UNKNOWN_IMAGE_TOKENS
    )
    return ImagePiece(
        file_id=file_id,
        detail=detail if isinstance(detail, str) and detail else None,
        tokens=tokens,
    )


_ROLE_LABELS = {
    "user": "[USER]",
    "assistant": "[ASSISTANT]",
    "tool": "[TOOL]",
    "function": "[TOOL]",
    "system": "[SYSTEM]",
    "developer": "[SYSTEM]",
}


def role_label(role: str) -> str:
    """The label a message's text starts with in the extraction input."""
    return _ROLE_LABELS.get(role, f"[{role.upper()}]")


def conversation_pieces(
    parsed: Sequence[ParsedMessage],
) -> Tuple[List[Piece], Dict[str, ExtractionImage]]:
    """The conversation as ordered text and image pieces.

    Each message is ``[USER]`` / ``[ASSISTANT]`` / ``[TOOL]`` plus its parts
    in their order (a role-less legacy payload has no prefix); an image stays
    where it was. A stored-file reference that is not an image becomes an
    ``[attachment: …]`` note.
    """
    pieces: List[Piece] = []
    images: Dict[str, ExtractionImage] = {}
    for index, message in enumerate(parsed):
        if index:
            pieces.append(TextPiece("\n"))
        if message.role:
            parts = _decorate(message, list(message.parts))
            pieces.append(TextPiece(role_label(message.role) + " "))
        else:
            parts = list(message.parts)
        for position, part in enumerate(parts):
            if position:
                pieces.append(TextPiece("\n"))
            if isinstance(part, TextPart):
                pieces.append(TextPiece(part.text))
            elif isinstance(part, ImagePart):
                file_id = f"{_EPHEMERAL_PREFIX}{uuid.uuid4().hex}"
                images[file_id] = ExtractionImage(file_id=file_id, part=part)
                pieces.append(_image_piece(file_id, part, part.detail))
            elif isinstance(part, PassthroughPart):
                image_id = part.raw.get("image_id")
                if part.part_type == "database_image_id" and isinstance(image_id, str):
                    images[image_id] = ExtractionImage(file_id=image_id)
                    pieces.append(_image_piece(image_id, None, part.raw.get("detail")))
                else:
                    pieces.append(TextPiece(f"[attachment: {part.part_type}]"))
    return _coalesce(pieces), images


def fixed_prompt_tokens(counter: TokenCounter) -> int:
    """Tokens of everything but the conversation: the system prompt, the
    header and instruction around it, the tool schema, message overheads."""
    return (
        counter.count_text(EXTRACTION_SYSTEM_PROMPT)
        + counter.count_text(CONVERSATION_HEADER)
        + counter.count_text("\n\n" + EXTRACTION_INSTRUCTION)
        + counter.count_tools([EXTRACTION_TOOL])
        + 2 * MESSAGE_OVERHEAD_TOKENS
    )


def extraction_budget(llm_config: Any, counter: TokenCounter) -> int:
    """The most tokens the conversation may take in the extraction call:
    ``input_budget(prompts + tool schema, reserve)``, where the
    reserve is the planning output reserve, but never less than the
    ``MIN_OUTPUT_TOKENS`` a client insists on leaving for the answer (a
    small window's planning reserve can be below it)."""
    limits = ModelLimits.from_llm_config(llm_config)
    reserve = max(limits.planning_output_reserve, MIN_OUTPUT_TOKENS)
    return limits.input_budget(fixed_prompt_tokens(counter), reserve)


def build_extraction_input(
    parsed: Sequence[ParsedMessage], budget: int, counter: TokenCounter
) -> ExtractionInput:
    """The conversation pieces within ``budget`` tokens (pure CPU).

    Too large an input keeps its head and tail (:func:`head_tail_pieces`):
    images are kept whole or left out, never split. ``stats`` records the
    estimate, the budget and what was cut, for the retrieval log.
    """
    pieces, images = conversation_pieces(parsed)
    estimated = _pieces_tokens(pieces, counter)
    image_count = sum(1 for p in pieces if isinstance(p, ImagePiece))
    image_cost = sum(p.cost for p in pieces if isinstance(p, ImagePiece))
    stats: Dict[str, Any] = {
        "estimated_tokens": estimated,
        "budget_tokens": budget,
        "images": image_count,
        "image_tokens": image_cost,
        "truncated": False,
        "truncated_chars": 0,
        "truncated_tokens": 0,
        "dropped_images": 0,
        "sent_tokens": estimated,
    }
    if estimated > budget:
        cut = head_tail_pieces(pieces, budget, counter)
        pieces = cut.pieces
        stats.update(
            truncated=cut.truncated,
            truncated_chars=cut.removed_chars,
            truncated_tokens=cut.removed_tokens,
            dropped_images=cut.dropped_images,
            sent_tokens=cut.tokens,
        )
        kept = {p.file_id for p in pieces if isinstance(p, ImagePiece)}
        images = {k: v for k, v in images.items() if k in kept}
    return ExtractionInput(pieces=pieces, images=images, stats=stats)


def _text_chars(parsed: Sequence[ParsedMessage]) -> int:
    return sum(len(p.text) for m in parsed for p in m.parts if isinstance(p, TextPart))


# --------------------------------------------------------------------------- #
# The request's images, for the extraction call only                           #
# --------------------------------------------------------------------------- #


def _write_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)


@dataclass(frozen=True)
class EphemeralFile:
    """The metadata a model client reads for an image (``FileMetadata``'s
    fields it uses), for an image that is never stored."""

    id: str
    file_name: str
    file_path: Optional[str] = None
    source_url: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    google_cloud_url: Optional[str] = None


class ExtractionImageFiles:
    """The ``file_manager`` the extraction call's LLM client reads images from.

    The model clients resolve an image part by ``files`` id
    (``get_file_metadata_by_id``) and read its ``file_path`` or send its
    ``source_url``. For the request's own images this serves in-memory
    metadata instead: an inline image is written to a private temporary
    directory (off the event loop), a remote one is its URL. Nothing is
    recorded in ``files`` and :meth:`aclose` removes the directory. Any other
    id (a ``database_image_id`` reference) is looked up in the real
    ``FileManager``.
    """

    def __init__(self, images: Dict[str, ExtractionImage], fallback: Any = None):
        self._images = {k: v for k, v in images.items() if not v.stored}
        self._fallback = fallback
        self._metadata: Dict[str, Any] = {}
        self._dir: Optional[Path] = None

    async def __aenter__(self) -> "ExtractionImageFiles":
        try:
            for file_id, image in self._images.items():
                part = image.part
                assert part is not None
                if part.is_remote:
                    self._metadata[file_id] = EphemeralFile(
                        id=file_id, file_name="remote_image", source_url=part.url
                    )
                    continue
                prepared = part.prepared
                if prepared is None:
                    from mirix.services.image_store import prepare_image

                    prepared = await prepare_image(part)
                assert prepared is not None
                if self._dir is None:
                    self._dir = Path(
                        await asyncio.to_thread(
                            tempfile.mkdtemp, prefix="mirix-retrieval-"
                        )
                    )
                path = self._dir / f"{file_id}.{prepared.extension}"
                await asyncio.to_thread(_write_file, path, prepared.data)
                self._metadata[file_id] = EphemeralFile(
                    id=file_id,
                    file_name=path.name,
                    file_path=str(path),
                    file_type=prepared.mime_type,
                    file_size=prepared.size,
                )
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    @property
    def directory(self) -> Optional[Path]:
        return self._dir

    async def aclose(self) -> None:
        directory, self._dir = self._dir, None
        if directory is not None:
            await asyncio.to_thread(shutil.rmtree, directory, True)

    async def get_file_metadata_by_id(self, file_id: str, *args: Any, **kwargs: Any):
        if file_id in self._metadata:
            return self._metadata[file_id]
        if self._fallback is None:
            from mirix.services.file_manager import FileManager

            self._fallback = FileManager()
        return await self._fallback.get_file_metadata_by_id(file_id, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Anything else a client may call goes to the real FileManager.
        if name.startswith("_"):
            raise AttributeError(name)
        if self._fallback is None:
            from mirix.services.file_manager import FileManager

            self._fallback = FileManager()
        return getattr(self._fallback, name)


# --------------------------------------------------------------------------- #
# Extraction                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class TopicExtraction:
    """The outcome of one extraction.

    ``status`` is ``ok`` (topics extracted) or ``fallback`` (it failed:
    ``error`` says why and the caller searches with the raw text).
    ``usage`` is the call's token usage; ``input_stats`` the budget
    bookkeeping of :func:`build_extraction_input`. ``procedural_query`` is
    the task-level part of the topics (goal and operations) that skills are
    ranked with; None when the answer had none (a legacy one-string answer,
    or key objects only): skills then rank the topics.
    """

    topics: Optional[str]
    temporal_expression: Optional[str]
    status: str
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    input_stats: Dict[str, Any] = field(default_factory=dict)
    procedural_query: Optional[str] = None

    def log_entry(self) -> Dict[str, Any]:
        """The ``llm_usage["topic_extraction"]`` entry of the retrieval log."""
        entry: Dict[str, Any] = dict(self.usage)
        entry["status"] = self.status
        if self.procedural_query is not None:
            entry["procedural_query"] = self.procedural_query
        if self.input_stats:
            entry["input"] = dict(self.input_stats)
        if self.error:
            entry["error"] = self.error
        return entry


def build_extraction_messages(pieces: Sequence[Piece]) -> List[Any]:
    """The extraction dialogue: the system prompt, then ONE user message with
    the conversation (images in place) followed by the instruction."""
    from mirix.schemas.enums import MessageRole
    from mirix.schemas.message import Message
    from mirix.schemas.mirix_message_content import ImageContent, TextContent

    content: List[Any] = []
    for piece in _coalesce(
        [
            TextPiece(CONVERSATION_HEADER),
            *pieces,
            TextPiece("\n\n" + EXTRACTION_INSTRUCTION),
        ]
    ):
        if isinstance(piece, TextPiece):
            content.append(TextContent(text=piece.text))
        else:
            content.append(
                ImageContent(image_id=piece.file_id, detail=piece.detail or "auto")
            )
    return [
        Message(
            agent_id=EXTRACTION_AGENT_ID,
            role=MessageRole.system,
            content=[TextContent(text=EXTRACTION_SYSTEM_PROMPT)],
        ),
        Message(agent_id=EXTRACTION_AGENT_ID, role=MessageRole.user, content=content),
    ]


def _clean_topics(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = f"{TOPIC_SEPARATOR} ".join(
            str(v).strip() for v in value if str(v).strip()
        )
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


# Whitespace and separators a part may start or end with ("goal;").
_PART_EDGES = string.whitespace + TOPIC_SEPARATOR


def _clean_part(value: Any) -> Optional[str]:
    """One field of a structured answer as text (a list joined like a topic
    list), without stray separators at its edges; None when empty."""
    value = _clean_topics(value)
    if value is None:
        return None
    return value.strip(_PART_EDGES) or None


def _structured_topics(
    arguments: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """``(topics, procedural_query)`` of a structured answer: the topics
    join goal, key objects and operations in that order (the one-string
    topic's order); the procedural query joins goal and operations."""
    goal = _clean_part(arguments.get("goal"))
    key_objects = _clean_part(arguments.get("key_objects"))
    operations = _clean_part(arguments.get("operations"))
    joiner = f"{TOPIC_SEPARATOR} "
    topics = joiner.join(p for p in (goal, key_objects, operations) if p) or None
    procedural = joiner.join(p for p in (goal, operations) if p) or None
    return topics, procedural


def parse_extraction_response(
    response: Any,
) -> Tuple[str, Optional[str], Optional[str]]:
    """``(topics, temporal_expression, procedural_query)`` from the forced
    tool call; raises :class:`TopicExtractionError` when the answer has no
    usable topic.

    A structured answer (``goal``, ``operations``, ``key_objects``) gives
    both the topics and the task-level procedural query; a legacy
    ``{"topic": ...}`` answer gives the topics alone (procedural query None).
    """
    problems: List[str] = []
    for choice in getattr(response, "choices", None) or []:
        tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None)
        if not tool_calls:
            problems.append("no tool call")
            continue
        for call in tool_calls:
            function = getattr(call, "function", None)
            if (
                getattr(function, "name", EXTRACTION_FUNCTION_NAME)
                != EXTRACTION_FUNCTION_NAME
            ):
                problems.append(f"unexpected tool {getattr(function, 'name', None)!r}")
                continue
            try:
                arguments = json.loads(getattr(function, "arguments", None) or "{}")
            except (TypeError, ValueError) as exc:
                problems.append(f"arguments are not JSON ({exc})")
                continue
            if not isinstance(arguments, dict):
                problems.append("arguments are not an object")
                continue
            topics, procedural = _structured_topics(arguments)
            if topics is None:
                topics = _clean_topics(arguments.get("topic"))
                procedural = None
            if topics is None:
                problems.append("empty topic")
                continue
            temporal = arguments.get("temporal_expression")
            temporal = temporal.strip() if isinstance(temporal, str) else None
            return topics, temporal or None, procedural
    raise TopicExtractionError(
        "the extraction answer has no usable topic: "
        + ("; ".join(problems) or "no choices")
    )


def _usage_dict(response: Any, llm_config: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "model": getattr(llm_config, "model", None),
        "calls": 1,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


ClientFactory = Callable[[Any], Any]


async def extract_topics(
    parsed: Sequence[ParsedMessage],
    llm_config: Any,
    *,
    counter: Optional[TokenCounter] = None,
    client_factory: Optional[ClientFactory] = None,
    file_manager: Any = None,
) -> TopicExtraction:
    """Extract topics and a temporal expression from ``parsed``.

    One forced tool call to ``llm_config``'s model with the conversation's
    text and images, cut to the extraction budget. Never raises except
    on cancellation: a failure is status ``fallback`` with its ``error``.
    ``client_factory`` (default ``LLMClient.create``) and ``file_manager``
    (default ``FileManager()`` for stored-image references) are injectable.
    """
    counter = counter or get_token_counter()
    usage: Dict[str, Any] = {"model": getattr(llm_config, "model", None), "calls": 0}
    stats: Dict[str, Any] = {}
    try:
        if llm_config is None:
            raise TopicExtractionError("no LLM configured for topic extraction")
        if client_factory is None:
            from mirix.llm_api.llm_client import LLMClient

            client_factory = LLMClient.create
        client = client_factory(llm_config)
        if client is None:
            raise TopicExtractionError(
                "no LLM client for endpoint type "
                f"{getattr(llm_config, 'model_endpoint_type', None)!r}"
            )
        budget = (
            extraction_budget(llm_config, counter) - EXTRACTION_BUDGET_MARGIN_TOKENS
        )
        shrinks = 0
        while True:
            if budget < MIN_CHUNK_BUDGET_TOKENS:
                raise TopicExtractionError(
                    f"the model's context window leaves {budget} tokens for the "
                    "conversation; too small for topic extraction"
                )
            if _text_chars(parsed) > LARGE_TEXT_CHARS:
                built = await asyncio.to_thread(
                    build_extraction_input, parsed, budget, counter
                )
            else:
                built = build_extraction_input(parsed, budget, counter)
            stats = built.stats
            stats["budget_shrinks"] = shrinks
            messages = build_extraction_messages(built.pieces)
            try:
                async with ExtractionImageFiles(
                    built.images, fallback=file_manager
                ) as files:
                    client.file_manager = files
                    response = await client.send_llm_request(
                        messages=messages,
                        tools=[EXTRACTION_TOOL],
                        stream=False,
                        force_tool_call=EXTRACTION_FUNCTION_NAME,
                    )
                break
            except ContextBudgetExceeded as exc:
                # Refused before sending: the client's own estimate of
                # the request (its tool schema, part boundaries) is above ours.
                # Shrink by the shortfall and cut again.
                if shrinks >= MAX_BUDGET_SHRINKS:
                    raise
                shrinks += 1
                shortfall = (exc.minimum_output_tokens or MIN_OUTPUT_TOKENS) - (
                    exc.output_cap or 0
                )
                budget -= max(1, shortfall) + EXTRACTION_BUDGET_MARGIN_TOKENS
        usage = _usage_dict(response, llm_config)
        topics, temporal, procedural = parse_extraction_response(response)
        logger.debug(
            "Extracted topics: %s, procedural query: %s, temporal: %s",
            topics,
            procedural,
            temporal,
        )
        return TopicExtraction(
            topics=topics,
            temporal_expression=temporal,
            status=EXTRACTION_STATUS_OK,
            usage=usage,
            input_stats=stats,
            procedural_query=procedural,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — fall back to the raw text
        logger.warning(
            "Topic extraction failed (%s); retrieving with the conversation's text",
            _error_text(exc),
        )
        return TopicExtraction(
            topics=None,
            temporal_expression=None,
            status=EXTRACTION_STATUS_FALLBACK,
            error=_error_text(exc),
            usage=usage,
            input_stats=stats,
        )
