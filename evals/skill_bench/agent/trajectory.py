"""Trajectories and the method-visible view.

Two objects per episode:

* :class:`Trajectory` is the **private** per-episode record, kept by the harness and the
  scorer: every actor call (:class:`~evals.skill_bench.agent.actor.ActorCall`, with the
  full request, its sha256, the response, usage and latency), the injection and its token
  count, each step's executed action and scorer-only ``private`` data, the final answer,
  the :class:`~evals.skill_bench.envs.base.EpisodeResult` and any failure.
* :class:`~evals.skill_bench.methods.base.TaskView` and
  :class:`~evals.skill_bench.methods.base.TrajectoryView` are what memory methods see:
  the opaque task uid, the Q28 ``instruction`` and the Q20 raw messages, and in the
  ``TaskView`` the benchmark's static ``agent_context`` (the actor's prompt template,
  the same for every item; recorded in the trajectory for the audit), nothing else.

Q20 messages, per benchmark:

* ALFWorld: per step, the model's raw output (``assistant``) and the observation with the
  admissible actions (``user``). ``first_messages`` = the initial observation message.
* WebShop: the same, with the shop's page and available actions as the observation.

Every message handed to a method goes through :func:`visible_message`, an allowlist of
keys per role: provider extras (``reasoning``, ``refusal``, annotations, ...) never reach a
method, and each method call gets its own deep copy.

The opaque task uid is ``"t"`` + the first 20 hex chars of HMAC-SHA256(cell salt, raw id)
(H5, :func:`opaque_task_uid`): it satisfies MIRIX's ``session_id`` rule, and it hides the
raw id, whose manifest position would leak the ALFWorld task type.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from evals.skill_bench.envs.base import (
    EpisodeResult,
    Message,
    Split,
    new_cell_salt,
    opaque_task_uid,
)
from evals.skill_bench.methods.base import Injection, TaskView, TrajectoryView

__all__ = [
    "FORBIDDEN_LABEL_KEYS",
    "StepRecord",
    "Trajectory",
    "canonical_json",
    "externalize_blobs",
    "find_label_leaks",
    "internalize_blobs",
    "new_cell_salt",
    "opaque_task_uid",
    "read_trajectory",
    "trajectory_view_from_record",
    "unreadable_blobs",
    "visible_message",
    "visible_messages",
    "write_trajectory",
]

# ---------------------------------------------------------------------------
# Method-visible messages (Q20)
# ---------------------------------------------------------------------------

#: Keys a method-visible message may carry, per role.
_ROLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "user": ("role", "content"),
    "assistant": ("role", "content"),
}


def _visible_part(part: Any) -> dict[str, Any]:
    if not isinstance(part, Mapping):
        raise ValueError(f"content part must be an object, got {type(part).__name__}")
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": str(part.get("text", ""))}
    raise ValueError(f"content part type {kind!r} is not allowed in a trajectory")


def _visible_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return [_visible_part(p) for p in content]
    raise ValueError(
        f"message content must be text or parts, got {type(content).__name__}"
    )


def visible_message(message: Mapping[str, Any]) -> Message:
    """A deep, allowlisted copy of one trajectory message (Q20).

    ``user`` and ``assistant`` keep role and content (text parts only). Anything else
    (``system`` and ``tool`` messages, tool calls, images, reasoning, provider extras) is
    refused or dropped.
    """
    role = message.get("role")
    if not isinstance(role, str) or role not in _ROLE_KEYS:
        raise ValueError(
            f"role {role!r} does not belong in a method-visible trajectory"
        )
    out: dict[str, Any] = {
        "role": role,
        "content": _visible_content(message.get("content")),
    }
    return {k: out[k] for k in _ROLE_KEYS[role] if k in out}


def visible_messages(messages: Iterable[Mapping[str, Any]]) -> tuple[Message, ...]:
    return tuple(visible_message(m) for m in messages)


# ---------------------------------------------------------------------------
# Label scan (H10.1)
# ---------------------------------------------------------------------------

#: Keys that must never appear anywhere in what a method receives.
FORBIDDEN_LABEL_KEYS: frozenset[str] = frozenset(
    {
        "won",
        "score",
        "scores",
        "reward",
        "gold",
        "gold_answer",
        "gold_answers",
        "answer",
        "answers",
        "ground_truth",
        "label",
        "labels",
        "task_type",
        "question_types",
        "category",
        "topic",
        "difficulty",
        "source",
        "source_files",
        "source_docs",
        "source_split",
        "page",
        "pages",
        "page_no",
        "ucsf_document_page_no",
        "ucsf_document_id",
        "info",
        "infos",
        "facts",
        "expert_plan",
        "extra.expert_plan",
        "extra.gamefile",
        "gamefile",
        "raw_id",
        "uid",
        "questionId",
        "docId",
        "split",
        "private",
        "final_answer",
        "stop_reason",
    }
)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def find_label_leaks(
    value: Any,
    *,
    forbidden_strings: Iterable[str] = (),
    skip_roles: Sequence[str] = ("assistant", "tool"),
    path: str = "$",
) -> list[str]:
    """Describe every label leak in a method-visible object (empty list = clean).

    * any key in :data:`FORBIDDEN_LABEL_KEYS`, at any depth;
    * any ``forbidden_strings`` (gold answers, raw ids, task types, page references ...)
      inside harness-authored text, compared case- and whitespace-insensitively.

    Text the model wrote (``assistant``) and tool outputs (``tool``: document excerpts,
    which may legitimately contain the answer) are skipped for the string check, as are
    image data URLs (base64 would match short strings by chance). Keys are always checked.
    """
    needles = sorted({_normalise(s) for s in forbidden_strings if s and s.strip()})
    leaks: list[str] = []

    def visit(node: Any, where: str, check_text: bool) -> None:
        if is_dataclass(node) and not isinstance(node, type):
            for f in fields(node):
                visit(getattr(node, f.name), f"{where}.{f.name}", check_text)
            return
        if isinstance(node, Mapping):
            role = node.get("role")
            text_ok = check_text and not (isinstance(role, str) and role in skip_roles)
            for key, child in node.items():
                if str(key) in FORBIDDEN_LABEL_KEYS:
                    leaks.append(f"{where}: forbidden key {key!r}")
                visit(child, f"{where}.{key}", text_ok)
            return
        if isinstance(node, (list, tuple)):
            for i, child in enumerate(node):
                visit(child, f"{where}[{i}]", check_text)
            return
        if isinstance(node, str) and check_text and not node.startswith("data:"):
            normal = _normalise(node)
            leaks.extend(
                f"{where}: contains {needle!r}"
                for needle in needles
                if needle in normal
            )

    visit(value, path, True)
    return leaks


# ---------------------------------------------------------------------------
# Private trajectory
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no spaces) used for hashes in records."""
    return json.dumps(
        to_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def to_jsonable(value: Any) -> Any:
    """Plain JSON types from dataclasses, mapping proxies, tuples and paths."""
    if is_dataclass(value) and not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_jsonable(to_dict())
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value != value:  # NaN is not JSON
        return None
    return value


@dataclass
class StepRecord:
    """One env step: the model's message, the environment's reply, and scorer-side data.

    ``assistant`` and ``env_messages`` are the method-visible messages of this step;
    ``reasoning`` (provider-separated model reasoning) and ``private`` (e.g. ALFWorld
    ``won``) are audit/scorer-only. ``fields`` keeps the step's agent-visible facts worth
    auditing (the executed action and format error).
    """

    index: int
    call_index: int
    assistant: Message
    reasoning: str | None
    env_messages: tuple[Message, ...]
    done: bool
    fields: Mapping[str, Any] = field(default_factory=dict)
    private: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """The full private record of one episode (H5). Never handed to a method."""

    benchmark: str
    split: Split
    raw_id: str
    task_uid: str
    phase: str
    method: str
    model: str
    run_id: str
    instruction: str = ""
    first_messages: tuple[Message, ...] = ()
    #: The ``TaskView.agent_context`` (the environment's static prompt template; empty
    #: for a benchmark that offers none).
    agent_context: str = ""
    messages: list[Message] = field(default_factory=list)
    injection: Injection | None = None
    injection_encoding: str | None = None
    first_prompt_sha256: str | None = None
    first_prompt_no_memory_sha256: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    calls: list[Any] = field(default_factory=list)  # ActorCall records
    final_answer: str | None = None
    final_private: Mapping[str, Any] = field(default_factory=dict)
    done: bool = False
    end_reason: str | None = None
    #: MODEL failures of single calls (truncated, filtered, timed out), each with its
    #: ``call_index``; the episode went on unless ``failure`` / ``end_reason`` say otherwise.
    model_failures: list[Mapping[str, Any]] = field(default_factory=list)
    failure: Mapping[str, Any] | None = None
    failure_stage: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    started_at: str | None = None
    result: EpisodeResult | None = None

    # --- method-visible views -----------------------------------------------------

    def task_view(self) -> TaskView:
        """``TaskView`` for ``before_task`` (a fresh deep copy each time)."""
        return TaskView(
            task_uid=self.task_uid,
            instruction=self.instruction,
            first_messages=copy.deepcopy(self.first_messages),
            agent_context=self.agent_context,
        )

    def view(self) -> TrajectoryView:
        """``TrajectoryView`` for ``after_task`` (a fresh deep copy each time)."""
        return TrajectoryView(
            task_uid=self.task_uid,
            instruction=self.instruction,
            first_messages=copy.deepcopy(self.first_messages),
            messages=copy.deepcopy(tuple(self.messages)),
        )

    # --- harness side ---------------------------------------------------------------

    @property
    def injected_tokens(self) -> int:
        return (
            self.injection.tokens
            if self.injection and self.injection.tokens is not None
            else 0
        )

    @property
    def injected_tokens_total(self) -> int:
        """Injection tokens times the actor calls that carried them (ALFWorld injects per step)."""
        return self.injected_tokens * len(self.calls)

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["injected_tokens"] = self.injected_tokens
        data["injected_tokens_total"] = self.injected_tokens_total
        return to_jsonable(data)


# ---------------------------------------------------------------------------
# Serialization: content-addressed blobs for image data URLs
# ---------------------------------------------------------------------------

BLOB_MIN_CHARS = 4096
BLOB_KEY = "$blob"


def _fsync_dir(directory: Path) -> None:
    """fsync a directory, so the entries renamed or created in it survive a crash (a
    platform that cannot open a directory skips it). A local copy of
    ``cell_records._fsync_dir``: ``cell_records`` imports this module."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _mkdir_durable(directory: Path) -> None:
    """``mkdir -p`` whose new directories survive a crash: the parent of each directory
    it creates is fsync'd."""
    created: list[Path] = []
    probe = directory
    while not probe.exists() and probe.parent != probe:
        created.append(probe)
        probe = probe.parent
    directory.mkdir(parents=True, exist_ok=True)
    for new in reversed(created):
        _fsync_dir(new.parent)


def _blob_holds(path: Path, data: bytes) -> bool:
    """Whether ``path`` holds exactly ``data`` (a zero-length or torn blob does not)."""
    try:
        if path.stat().st_size != len(data):
            return False
        return path.read_bytes() == data
    except OSError:
        return False


def _write_blob(path: Path, data: bytes) -> None:
    """Durably store one blob: a temporary file in the same directory,
    flushed and fsync'd, renamed over ``path``, then the directory fsync'd. A played
    checkpoint's resume reads it back, so it must be on disk before the trajectory and
    the checkpoint that reference it: ext4's delayed allocation leaves a file written
    with a plain ``write`` + ``rename`` zero bytes long after a power loss."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def externalize_blobs(
    value: Any, blob_dir: Path, *, min_chars: int = BLOB_MIN_CHARS
) -> Any:
    """Replace every ``data:`` URL of at least ``min_chars`` by ``{"$blob": sha256}``.

    The string itself is written once to ``blob_dir/<sha256>``, so the same data shared
    by every method and cell is stored once. Lossless: :func:`internalize_blobs`
    restores the exact string. Durable: the blob is fsync'd, renamed into
    place and its directory fsync'd before this returns, and a blob already there is
    checked byte for byte (one a crash left empty or torn is written again) and fsync'd,
    so the record that references it never outlives it. Blocking.
    """
    durable: set[str] = set()

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {k: visit(v) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(v) for v in node]
        if (
            isinstance(node, str)
            and len(node) >= min_chars
            and node.startswith("data:")
        ):
            digest = sha256_hex(node)
            if digest not in durable:
                path = blob_dir / digest
                data = node.encode("utf-8")
                _mkdir_durable(blob_dir)
                if _blob_holds(path, data):
                    # Another writer's copy: make sure it is on disk too.
                    try:
                        _fsync_file(path)
                    except FileNotFoundError:
                        _write_blob(path, data)
                    _fsync_dir(blob_dir)
                else:
                    _write_blob(path, data)
                durable.add(digest)
            return {BLOB_KEY: digest}
        return node

    return visit(to_jsonable(value))


def internalize_blobs(value: Any, blob_dir: Path) -> Any:
    """Inverse of :func:`externalize_blobs`; checks each blob's sha256 (a missing blob
    raises ``FileNotFoundError``, a corrupt one ``ValueError``)."""

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            if set(node) == {BLOB_KEY}:
                digest = str(node[BLOB_KEY])
                text = (blob_dir / digest).read_text(encoding="utf-8")
                if sha256_hex(text) != digest:
                    raise ValueError(f"blob {digest} is corrupt")
                return text
            return {k: visit(v) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(v) for v in node]
        return node

    return visit(value)


def unreadable_blobs(value: Any, blob_dir: Path) -> list[str]:
    """The sha256 of every blob ``value`` references that :func:`internalize_blobs`
    could not restore: missing, unreadable, or not hashing to its name (sorted, each
    once; blocking). ``run.py`` re-creates them from the benchmark data before it
    rebuilds a played item's view."""
    wanted: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if set(node) == {BLOB_KEY}:
                wanted.add(str(node[BLOB_KEY]))
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    lost = []
    for digest in sorted(wanted):
        try:
            data = (blob_dir / digest).read_bytes()
        except OSError:
            lost.append(digest)
            continue
        if sha256_hex(data) != digest:
            lost.append(digest)
    return lost


def write_trajectory(
    path: Path, trajectory: Trajectory, *, blob_dir: Path | None = None
) -> None:
    """Write one trajectory as JSON (atomic rename, fsync'd); images to ``blob_dir`` if given.

    Durable in order: the images first (:func:`externalize_blobs`), then
    the JSON (fsync'd, renamed, its directory fsync'd), so a played checkpoint written
    after this returns never references a record or an image a crash can lose.

    Blocking file I/O: async callers run it with ``asyncio.to_thread``. Safe to call from
    several threads at once (unique temporary names, atomic renames).
    """
    data: Any = trajectory.to_dict()
    if blob_dir is not None:
        data = externalize_blobs(data, blob_dir)
    _mkdir_durable(path.parent)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def read_trajectory(path: Path, *, blob_dir: Path | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return internalize_blobs(data, blob_dir) if blob_dir is not None else data


def trajectory_view_from_record(
    record: Mapping[str, Any], blob_dir: Path | None
) -> TrajectoryView:
    """The ``TrajectoryView`` a recorded trajectory's ``after_task`` received, rebuilt from
    its private record (:func:`write_trajectory`'s JSON; blocking when images are read
    from ``blob_dir``). Its canonical JSON equals that of the original
    :meth:`Trajectory.view`: ``run.py`` hands it to ``after_task`` when a learning phase
    resumes from an item's ``played`` checkpoint. A missing or corrupt image blob raises
    (``ValueError`` / ``OSError``), never a view with another image."""
    data = internalize_blobs(record, blob_dir) if blob_dir is not None else record
    if not isinstance(data, Mapping) or not isinstance(data.get("task_uid"), str):
        raise ValueError("a trajectory record needs its task_uid")
    first = data.get("first_messages") or []
    messages = data.get("messages") or []
    if not isinstance(first, list) or not isinstance(messages, list):
        raise ValueError("a trajectory record's messages must be lists")
    return TrajectoryView(
        task_uid=data["task_uid"],
        instruction=str(data.get("instruction") or ""),
        first_messages=tuple(copy.deepcopy(m) for m in first),
        messages=tuple(copy.deepcopy(m) for m in messages),
    )
