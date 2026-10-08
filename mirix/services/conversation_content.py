"""One parser for memory-add message payloads, and the conversation-store text cap.

``/memory/add`` and ``/memory/add_sync`` accept OpenAI-style messages whose
``content`` is a string, a list of content parts, or ``None`` (an assistant
turn that only carries ``tool_calls``). Both consumers of a request read it
through :func:`parse_messages`:

* the **conversation store** (every request with a ``session_id``; the only
  consumer of a ``task`` session), via :func:`store_turn_drafts`;
* the **meta memory agent** (``conversation`` sessions), via
  :func:`build_meta_input_messages`.

Before this module the meta path collapsed roles with ``prefix + " " + part``,
which raised ``TypeError`` (HTTP 500) for the SDK's own recommended
``[{"type": "text", "text": ...}]`` parts, and the store flattened parts with
``str()``. Now both share one parse, so a payload is either accepted by both or
rejected up front with :class:`MessageParseError` (HTTP 422) before anything is
written.

Part types:

* text: ``"..."`` or ``{"type": "text", "text": "..."}``;
* inline image: ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}``
  or ``{"type": "image_data", "image_data": {"data": "<base64 or data URL>"}}``;
* remote image: ``{"type": "image_url", "image_url": {"url": "https://..."}}`` —
  recorded by address, never downloaded;
* references to files already stored by MIRIX (``database_image_id``,
  ``database_file_id``, ``database_google_cloud_file_uri``) pass through to the
  meta path unchanged once the caller is shown to own them (the REST layer
  checks that before anything is written); the conversation store records them
  as an ``[attachment: <type>]`` placeholder;
* ``file_uri`` / ``google_cloud_file_uri`` name a file on the SERVER's own
  filesystem or cloud storage. They are in-process (local client) features and
  are rejected here, before anything is read: over the API they would let any
  caller read, or probe for, files on the server.

An image part's ``detail`` must be one of ``auto``, ``low``, ``high``.

Tool payloads are data, not parts: a JSON object or number in a tool result is
stored as JSON text. A typed part of an unknown type in a user/assistant
message is rejected rather than silently mangled.

Parsing is pure CPU work on the request body. Decoding inline images (base64,
Pillow) happens later, in :mod:`mirix.services.image_store`, off the event loop.

A NUL character (U+0000) cannot be stored in a Postgres ``text`` or ``jsonb``
value, and a tool result or a file read can carry one. Both consumers write
message text with every U+0000 replaced by U+FFFD (:func:`replace_nul`), on
every backend, so the write never fails on it; the conversation store reports
how many it replaced in the add response's ``warnings``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mirix.schemas.conversation_message import (
    CONTENT_PART_IMAGE_REF,
    CONTENT_PART_TEXT,
)

TOOL_ROLES = ("tool", "function")

# Real turn roles kept by the conversation store; everything else (system
# scaffolding, a legacy content-only payload) is not a learnable turn.
_STORE_ROLE = {
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}

# Parts that reference a file MIRIX already stored (a ``files`` row id). The
# meta path passes them through (mirix.utils.convert_message_to_mirix_message);
# the conversation store only notes their presence. The REST layer checks the
# caller owns every referenced file before anything is written.
LEGACY_PASSTHROUGH_PART_TYPES = (
    "database_image_id",
    "database_file_id",
    "database_google_cloud_file_uri",
)
# The string field each file-reference part must carry: the ``files`` row id.
_PASSTHROUGH_REQUIRED_KEYS = {
    "database_image_id": ("image_id",),
    "database_file_id": ("file_id",),
    "database_google_cloud_file_uri": ("cloud_file_uri",),
}
# Parts that name a file on the server's own filesystem / cloud storage. Only
# an in-process caller may use them (convert_message_to_mirix_message); an API
# payload carrying one is rejected before anything is read (never a disk
# access, and the same answer whether or not the path exists).
SERVER_LOCAL_PART_TYPES = ("file_uri", "google_cloud_file_uri")

REMOTE_IMAGE_SCHEMES = ("http://", "https://")

# The storage-cap marker spliced into a truncated text part: head + marker + tail.
TRUNCATION_MARKER_TEMPLATE = "…[truncated {n} chars]…"

CONVERSATION_MESSAGE_MAX_CHARS_ENV = "MIRIX_CONVERSATION_MESSAGE_MAX_CHARS"


class MessageParseError(ValueError):
    """A memory-add payload the server cannot accept (maps to HTTP 422)."""


# --------------------------------------------------------------------------- #
# NUL characters                                                               #
# --------------------------------------------------------------------------- #

NUL = "\x00"
#: What a NUL in stored message text becomes: the Unicode replacement character.
NUL_REPLACEMENT = "\ufffd"


def replace_nul(text: str) -> Tuple[str, int]:
    """``text`` with every U+0000 replaced by U+FFFD, and how many there were.

    Postgres rejects U+0000 in ``text`` and ``jsonb`` values, so a message that
    carries one (a tool printing ``\\x00``, a binary file read) would fail the
    whole write, and it would fail the same way on every retry. The replacement
    keeps the text's length, so the storage cap is unaffected.
    """
    count = text.count(NUL)
    if not count:
        return text, 0
    return text.replace(NUL, NUL_REPLACEMENT), count


def nul_replacement_warning(chars: int, messages: int) -> str:
    """The add-response warning for NUL characters replaced in stored turns."""
    return (
        f"{chars} NUL character(s) (U+0000) in {messages} message(s) were stored "
        "as U+FFFD: the conversation store cannot hold U+0000"
    )


# --------------------------------------------------------------------------- #
# Parsed shapes                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class TextPart:
    text: str


@dataclass(eq=False)
class ImagePart:
    """An image content part.

    Inline images carry their base64 payload (``data_b64``) until
    :func:`mirix.services.image_store.prepare_images` decodes them into
    ``prepared``; remote images carry only ``url``. ``eq=False`` keeps identity
    semantics so one part can key the image references stored for it.
    """

    detail: Optional[str] = None
    data_b64: Optional[str] = None
    declared_mime_type: Optional[str] = None
    url: Optional[str] = None
    prepared: Any = None

    @property
    def is_remote(self) -> bool:
        return self.url is not None


@dataclass
class PassthroughPart:
    """A legacy file part for the meta path (see LEGACY_PASSTHROUGH_PART_TYPES)."""

    raw: Dict[str, Any]

    @property
    def part_type(self) -> str:
        return str(self.raw.get("type"))


Part = Union[TextPart, ImagePart, PassthroughPart]


@dataclass
class ParsedMessage:
    """One message of a memory-add payload.

    ``role`` is None for a legacy content-only payload (a bare list of parts
    with no roles, e.g. a screenshot stream): it goes to the meta agent as one
    user message and is not a conversation turn.
    """

    role: Optional[str]
    parts: List[Part] = field(default_factory=list)
    tool_calls: Any = None
    name: Optional[str] = None
    # A tool result's tool_call_id: the assistant tool call it answers.
    tool_call_id: Optional[str] = None

    @property
    def images(self) -> List[ImagePart]:
        return [p for p in self.parts if isinstance(p, ImagePart)]


def iter_images(parsed: Sequence[ParsedMessage]) -> List[ImagePart]:
    """Every image part of a parsed payload, in order."""
    return [img for message in parsed for img in message.images]


def referenced_file_ids(parsed: Sequence[ParsedMessage]) -> List[str]:
    """The ``files`` ids a parsed payload's file-reference parts point at."""
    ids: Dict[str, None] = {}
    for message in parsed:
        for part in message.parts:
            if isinstance(part, PassthroughPart):
                for key in _PASSTHROUGH_REQUIRED_KEYS.get(part.part_type, ()):
                    value = part.raw.get(key)
                    if isinstance(value, str) and value:
                        ids.setdefault(value, None)
    return list(ids)


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #


def parse_messages(messages: Any) -> List[ParsedMessage]:
    """Parse a memory-add ``messages`` payload; raise MessageParseError if invalid.

    Either every item carries a ``role`` (OpenAI-style messages) or none does
    (a legacy content-only payload, parsed as one role-less message).
    """
    if not isinstance(messages, list) or not messages:
        raise MessageParseError("messages must be a non-empty list")
    if not all(isinstance(item, dict) for item in messages):
        raise MessageParseError("every item of messages must be an object")
    with_role = [("role" in item) for item in messages]
    if all(with_role):
        return _name_tool_results(
            [_parse_role_message(i, item) for i, item in enumerate(messages)]
        )
    if any(with_role):
        raise MessageParseError(
            "messages mixes role-bearing messages with bare content parts; "
            "send either [{'role': ..., 'content': ...}, ...] or a list of parts"
        )
    for i, item in enumerate(messages):
        if "type" not in item:
            raise MessageParseError(
                f"messages[{i}] has neither a 'role' (a message) nor a 'type' (a part)"
            )
    return [
        ParsedMessage(role=None, parts=_parse_parts(messages, "messages", tool=False))
    ]


def _parse_role_message(index: int, message: Dict[str, Any]) -> ParsedMessage:
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise MessageParseError(f"messages[{index}].role must be a non-empty string")
    tool = role in TOOL_ROLES
    parts = _parse_content(
        message.get("content"), f"messages[{index}].content", tool=tool
    )
    name = message.get("name") or message.get("tool_name")
    call_id = message.get("tool_call_id")
    return ParsedMessage(
        role=role,
        parts=parts,
        tool_calls=message.get("tool_calls") or None,
        name=name if isinstance(name, str) and name else None,
        tool_call_id=call_id if isinstance(call_id, str) and call_id else None,
    )


def _name_tool_results(parsed: List[ParsedMessage]) -> List[ParsedMessage]:
    """A tool result without a name takes the name of the assistant tool call
    its ``tool_call_id`` answers, in the same payload (OpenAI's tool messages
    carry only the id)."""
    names: Dict[str, str] = {}
    for message in parsed:
        if message.role != "assistant" or not isinstance(message.tool_calls, list):
            continue
        for call in message.tool_calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            function = (
                call.get("function") if isinstance(call.get("function"), dict) else call
            )
            name = function.get("name") or call.get("name")
            if isinstance(name, str) and name:
                names[call["id"]] = name
    for message in parsed:
        if message.role in TOOL_ROLES and message.name is None and message.tool_call_id:
            message.name = names.get(message.tool_call_id)
    return parsed


def _parse_content(content: Any, where: str, *, tool: bool) -> List[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(content)]
    if isinstance(content, list):
        return _parse_parts(content, where, tool=tool)
    if isinstance(content, dict):
        return _parse_parts([content], where, tool=tool)
    if isinstance(content, (bool, int, float)):
        return [TextPart(_json_text(content))]
    raise MessageParseError(f"{where} has an unsupported type {type(content).__name__}")


def _parse_parts(items: List[Any], where: str, *, tool: bool) -> List[Part]:
    parts: List[Part] = []
    for i, item in enumerate(items):
        part = _parse_part(item, f"{where}[{i}]", tool=tool)
        if part is not None:
            parts.append(part)
    return parts


def _parse_part(item: Any, where: str, *, tool: bool) -> Optional[Part]:
    if item is None:
        return None
    if isinstance(item, str):
        return TextPart(item)
    if isinstance(item, (bool, int, float)):
        return TextPart(_json_text(item))
    if isinstance(item, list):
        if tool:
            return TextPart(_json_text(item))
        raise MessageParseError(f"{where} is a nested list, not a content part")
    if not isinstance(item, dict):
        raise MessageParseError(
            f"{where} has an unsupported type {type(item).__name__}"
        )

    part_type = item.get("type")
    if part_type == "text":
        text = item.get("text")
        if isinstance(text, str):
            return TextPart(text)
        if tool:
            return TextPart(_json_text(item))
        raise MessageParseError(f"{where} is a text part without a string 'text'")
    if part_type == "image_url":
        return parse_image_url(item.get("image_url"), where)
    if part_type == "image_data":
        return parse_image_data(item.get("image_data"), where)
    if part_type in SERVER_LOCAL_PART_TYPES:
        raise MessageParseError(
            f"{where} has content part type {part_type!r}, which is not accepted "
            "over the API: it names a file on the server. Send the image inline "
            "('image_data', or an 'image_url' with a base64 data URL) or as an "
            "https URL."
        )
    if part_type in LEGACY_PASSTHROUGH_PART_TYPES:
        keys = _PASSTHROUGH_REQUIRED_KEYS[part_type]
        if not any(isinstance(item.get(k), str) and item.get(k) for k in keys):
            raise MessageParseError(
                f"{where} ({part_type}) needs a non-empty string {' or '.join(map(repr, keys))}"
            )
        return PassthroughPart(dict(item))
    if part_type is None or tool:
        # Untyped objects are data (e.g. a JSON tool result), not parts.
        return TextPart(_json_text(item))
    raise MessageParseError(
        f"{where} has unsupported content part type {part_type!r}; supported: "
        "'text', 'image_url', 'image_data'"
    )


def _parse_detail(detail: Any, where: str) -> Optional[str]:
    """An image part's ``detail``: None, or one of auto / low / high."""
    from mirix.constants import IMAGE_DETAIL_VALUES

    if detail is None or detail == "":
        return None
    if not isinstance(detail, str) or detail not in IMAGE_DETAIL_VALUES:
        raise MessageParseError(
            f"{where}.detail must be one of {list(IMAGE_DETAIL_VALUES)}; got {detail!r}"
        )
    return detail


def parse_image_url(spec: Any, where: str) -> ImagePart:
    detail = None
    if isinstance(spec, dict):
        url = spec.get("url")
        detail = spec.get("detail")
    else:
        url = spec
    if not isinstance(url, str) or not url.strip():
        raise MessageParseError(f"{where}.image_url needs a non-empty 'url'")
    url = url.strip()
    detail = _parse_detail(detail, f"{where}.image_url")
    if url.startswith("data:"):
        mime, data = _split_data_url(url, where)
        return ImagePart(detail=detail, data_b64=data, declared_mime_type=mime)
    if url.lower().startswith(REMOTE_IMAGE_SCHEMES):
        return ImagePart(detail=detail, url=url)
    raise MessageParseError(
        f"{where}.image_url.url must be a base64 data URL or an http(s) URL"
    )


def parse_image_data(spec: Any, where: str) -> ImagePart:
    if not isinstance(spec, dict) or not isinstance(spec.get("data"), str):
        raise MessageParseError(f"{where}.image_data needs a string 'data'")
    data = spec["data"].strip()
    if not data:
        raise MessageParseError(f"{where}.image_data.data is empty")
    detail = _parse_detail(spec.get("detail"), f"{where}.image_data")
    mime = spec.get("mime_type") if isinstance(spec.get("mime_type"), str) else None
    if data.startswith("data:"):
        mime, data = _split_data_url(data, where)
    return ImagePart(detail=detail, data_b64=data, declared_mime_type=mime)


def _split_data_url(url: str, where: str) -> Tuple[Optional[str], str]:
    header, sep, data = url.partition(",")
    if not sep:
        raise MessageParseError(f"{where} is a data URL without a ',' separator")
    params = header[len("data:") :].split(";")
    if "base64" not in (p.strip().lower() for p in params[1:]):
        raise MessageParseError(f"{where} must be a base64 data URL (';base64,')")
    if not data.strip():
        raise MessageParseError(f"{where} is a data URL with no data")
    mime = params[0].strip().lower() or None
    return mime, data


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def serialize_tool_calls(tool_calls: Any) -> str:
    """Render an assistant message's tool_calls into a compact text form.

    Accepts the OpenAI shape (`[{"function": {"name", "arguments"}, ...}]`) as
    well as simpler `[{"name", "arguments"}]` dicts; anything unrecognized falls
    back to `str()`. The distiller only needs to SEE which tool was called with
    what arguments — a readable line beats a lossless JSON blob.
    """
    if not isinstance(tool_calls, list):
        return str(tool_calls)
    lines = []
    for call in tool_calls:
        if isinstance(call, dict):
            function = (
                call.get("function") if isinstance(call.get("function"), dict) else call
            )
            name = function.get("name") or call.get("name") or "unknown_tool"
            arguments = function.get("arguments", call.get("arguments", ""))
            if not isinstance(arguments, str):
                arguments = _json_text(arguments)
            lines.append(f"[tool_call] {name}({arguments})")
        else:
            lines.append(f"[tool_call] {call}")
    return "\n".join(lines)


def _decorate(message: ParsedMessage, parts: List[Part]) -> List[Part]:
    """Fold a message's tool activity into its parts.

    An assistant message's ``tool_calls`` are appended as text; a tool result's
    tool ``name`` prefixes it. A message without media collapses to one text
    part (its text parts joined by newlines) — exactly the text the store has
    always kept for it.
    """
    serialized = None
    if message.role == "assistant" and message.tool_calls:
        serialized = serialize_tool_calls(message.tool_calls)
    tool_name = message.name if message.role in TOOL_ROLES else None

    if all(isinstance(p, TextPart) for p in parts):
        text = "\n".join(p.text for p in parts)  # type: ignore[union-attr]
        if serialized is not None:
            text = f"{text}\n{serialized}".strip() if text else serialized
        if tool_name:
            text = f"[{tool_name}] {text}"
        return [TextPart(text)]

    parts = list(parts)
    if serialized is not None:
        parts.append(TextPart(serialized))
    if tool_name:
        if parts and isinstance(parts[0], TextPart):
            parts[0] = TextPart(f"[{tool_name}] {parts[0].text}")
        else:
            parts.insert(0, TextPart(f"[{tool_name}]"))
    return parts


# --------------------------------------------------------------------------- #
# Conversation store                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class StoreTurnDraft:
    """A turn for the conversation store before its images are stored."""

    role: str
    parts: List[Union[TextPart, ImagePart]]

    @property
    def images(self) -> List[ImagePart]:
        return [p for p in self.parts if isinstance(p, ImagePart)]

    def to_turn(self, image_refs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """The ``record_turns`` payload, given the stored ref of each image part.

        A text-only turn is ``{"role", "content"}`` (``content_parts`` stays
        NULL, so the text is never stored twice); a turn with images is
        ``{"role", "content_parts"}`` and the manager renders its placeholder
        ``content``.
        """
        if not self.images:
            text = "\n".join(p.text for p in self.parts if isinstance(p, TextPart))
            return {"role": self.role, "content": text}
        content_parts: List[Dict[str, Any]] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                content_parts.append({"type": CONTENT_PART_TEXT, "text": part.text})
            else:
                ref = image_refs[id(part)]
                if ref.get("type") != CONTENT_PART_IMAGE_REF:
                    raise ValueError("image refs must be content_parts image_ref parts")
                content_parts.append(dict(ref))
        return {"role": self.role, "content_parts": content_parts}


def store_turn_drafts(parsed: Sequence[ParsedMessage]) -> List[StoreTurnDraft]:
    """The conversation-store turns of a parsed payload.

    Keeps the REAL roles (user / assistant / tool, with OpenAI's legacy
    'function' role stored as 'tool'); drops system scaffolding and role-less
    content-only payloads. Tool activity is kept deliberately: work-process
    lessons (a tool error, a retry, the fix that worked) live in tool calls and
    tool results.
    """
    drafts: List[StoreTurnDraft] = []
    for message in parsed:
        role = _STORE_ROLE.get(message.role or "")
        if role is None:
            continue
        parts: List[Part] = [
            TextPart(f"[attachment: {p.part_type}]")
            if isinstance(p, PassthroughPart)
            else p
            for p in message.parts
        ]
        decorated = _decorate(message, parts)
        drafts.append(
            StoreTurnDraft(
                role=role,
                parts=[p for p in decorated if isinstance(p, (TextPart, ImagePart))],
            )
        )
    return drafts


# --------------------------------------------------------------------------- #
# Meta memory agent                                                            #
# --------------------------------------------------------------------------- #


def _role_prefix(role: str) -> str:
    if role == "user":
        return "[USER]"
    if role in TOOL_ROLES:
        return "[TOOL]"
    return "[ASSISTANT]"


async def build_meta_input_messages(
    parsed: Sequence[ParsedMessage],
    *,
    organization_id: str,
    user_id: Optional[str],
    session_id: Optional[str] = None,
    file_manager=None,
    images_dir=None,
    stored_refs: Optional[list] = None,
) -> list:
    """The meta agent's input for a payload: ONE user ``MessageCreate``.

    Role-bearing messages are role-collapsed as before — each message's text is
    prefixed ``[USER]`` / ``[ASSISTANT]`` / ``[TOOL]`` — but image parts now stay
    images (stored through ``store_image_part`` and referenced by file id) and
    list-of-parts content no longer crashes. A content-only payload keeps its
    parts unprefixed. The meta path stores its own copy of each image reference
    (its own ``files`` row), so pruning the conversation store never removes an
    image the agent's message history still points to.

    ``stored_refs``, when given, collects the image refs stored here so a
    caller whose dispatch then fails can discard them
    (``image_store.discard_image_refs``). If building itself fails, the images
    it already stored are discarded before the error propagates.
    """
    from mirix.schemas.enums import MessageRole
    from mirix.schemas.message import MessageCreate
    from mirix.schemas.mirix_message_content import ImageContent, TextContent
    from mirix.services.image_store import discard_image_refs, store_image_part

    refs: list = stored_refs if stored_refs is not None else []
    replaced_nul = 0

    async def convert(part: Part):
        nonlocal replaced_nul
        if isinstance(part, TextPart):
            # The agent's message history is stored too: no U+0000 (see replace_nul).
            text, count = replace_nul(part.text)
            replaced_nul += count
            return TextContent(text=text)
        if isinstance(part, ImagePart):
            ref = await store_image_part(
                part,
                organization_id=organization_id,
                user_id=user_id,
                file_manager=file_manager,
                images_dir=images_dir,
            )
            refs.append(ref)
            return ImageContent(image_id=ref["file_id"], detail=part.detail or "auto")
        from mirix.utils import convert_message_to_mirix_message

        converted = await convert_message_to_mirix_message(
            [part.raw],
            org_id=organization_id,
            user_id=user_id,
            file_manager=file_manager,
            images_dir=images_dir,
        )
        return converted[0].content[0]

    content: list = []
    try:
        for message in parsed:
            if message.role is None:
                for part in message.parts:
                    content.append(await convert(part))
                continue
            prefix = _role_prefix(message.role)
            parts = _decorate(message, list(message.parts))
            if parts and isinstance(parts[0], TextPart):
                parts[0] = TextPart(f"{prefix} {parts[0].text}")
            else:
                parts.insert(0, TextPart(prefix))
            for part in parts:
                content.append(await convert(part))
    except BaseException:
        await discard_image_refs(refs, organization_id=organization_id)
        refs.clear()
        raise

    if replaced_nul:
        from mirix.log import get_logger

        get_logger(__name__).warning(
            "Replaced %d NUL character(s) (U+0000) with U+FFFD in the meta agent's "
            "input (session_id=%s)",
            replaced_nul,
            session_id,
        )
    return [
        MessageCreate(role=MessageRole.user, content=content, session_id=session_id)
    ]


# --------------------------------------------------------------------------- #
# Storage cap                                                                  #
# --------------------------------------------------------------------------- #


def context_window_char_cap(context_window: Any) -> Optional[int]:
    """``floor(context_window * 0.8) * 4`` characters; None if unknown."""
    if isinstance(context_window, bool) or not isinstance(context_window, int):
        return None
    if context_window <= 0:
        return None
    return (context_window * 8 // 10) * 4


def meta_agent_char_cap(meta_agent_state: Any) -> Optional[int]:
    """The per-message text cap derived from the writing meta agent's model."""
    llm_config = getattr(meta_agent_state, "llm_config", None)
    return context_window_char_cap(getattr(llm_config, "context_window", None))


def env_char_cap() -> Optional[int]:
    """``MIRIX_CONVERSATION_MESSAGE_MAX_CHARS`` if set to a positive integer."""
    raw = os.getenv(CONVERSATION_MESSAGE_MAX_CHARS_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def effective_char_cap(context_cap: Optional[int]) -> Optional[int]:
    """The cap to apply: the smaller of the model-derived cap and the env override."""
    caps = [c for c in (context_cap, env_char_cap()) if c is not None and c > 0]
    return min(caps) if caps else None


def truncation_marker(removed: int) -> str:
    return TRUNCATION_MARKER_TEMPLATE.format(n=removed)


def _removal_for(excess: int) -> int:
    """Chars to remove so that, marker included, a text shrinks by ``excess``."""
    removed = excess + len(truncation_marker(excess))
    while True:
        needed = excess + len(truncation_marker(removed))
        if needed <= removed:
            return removed
        removed = needed


def cap_text_parts(texts: Sequence[str], cap: Optional[int]) -> Tuple[List[str], int]:
    """Truncate a message's text parts to ``cap`` total characters.

    Starting from the longest part, each part keeps its head and tail (half
    each) with ``…[truncated N chars]…`` in between, where N is exactly the
    number of characters removed from it; the next longest part is cut only if
    the total is still over the cap. Markers count toward the cap. Returns the
    new texts and the total number of characters removed (0 = untouched).
    """
    result = list(texts)
    if cap is None or sum(len(t) for t in result) <= cap:
        return result, 0
    removed_total = 0
    order = sorted(range(len(result)), key=lambda i: (-len(result[i]), i))
    for i in order:
        excess = sum(len(t) for t in result) - cap
        if excess <= 0:
            break
        text = result[i]
        removed = _removal_for(excess)
        if removed >= len(text):
            # Nothing of this part fits: keep only the marker (if that shrinks it).
            marker = truncation_marker(len(text))
            if len(marker) >= len(text):
                continue
            result[i] = marker
            removed_total += len(text)
            continue
        keep = len(text) - removed
        head = keep - keep // 2
        tail = keep // 2
        result[i] = text[:head] + truncation_marker(removed) + text[len(text) - tail :]
        removed_total += removed
    return result, removed_total
