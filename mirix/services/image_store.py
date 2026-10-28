"""Image storage for memory-add payloads.

An inline image (base64 or a base64 data URL) is decoded and inspected with
Pillow, written to disk content-addressed, and recorded as a ``files`` row that
carries the owning ``user_id``. A remote image URL is recorded by address
only — never downloaded. Every store returns the ``image_ref`` part that
``conversation_message.content_parts`` keeps::

    {"type": "image_ref", "file_id": "file-…", "mime_type": "image/png",
     "width": 1700, "height": 2200, "sha256": "…", "bytes": 912345}

Layout: ``<images_dir>/<scope>/<sha256>.<ext>``, where ``<scope>`` is a hash of
(organization, user). Identical bytes of one user share a disk file; different
users never do, so erasing one user's images can never delete another's.

Every ``store_image_part`` call creates its OWN ``files`` row: a row is owned by
exactly one reference (one conversation-store turn, or one meta-agent message),
so releasing a reference deletes its row, and the disk file goes once no row
points at it any more (:func:`release_image_files`). Only files this module
wrote are ever unlinked (:func:`is_managed_image_path`).

All disk I/O and all decoding run in ``asyncio.to_thread``; the event loop never
blocks on an image.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import datetime as dt
import hashlib
import io
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse

from sqlalchemy import delete, or_, select

from mirix.log import get_logger
from mirix.schemas.conversation_message import CONTENT_PART_IMAGE_REF
from mirix.services.conversation_content import ImagePart, MessageParseError

logger = get_logger(__name__)

# <sha256>.<ext> inside a <32-hex scope> directory: the only shape this module
# writes, and (with the legacy shape below) the only shape release unlinks.
_IMAGE_FILE_NAME_RE = re.compile(r"^[0-9a-f]{64}\.[a-z0-9]{1,8}$")
_SCOPE_DIR_RE = re.compile(r"^[0-9a-f]{32}$")
# Images saved before the per-user layout: <images root>/image_<16 hex>.<ext>
# (shared by content hash across users, so released only once no files row of
# any owner points at them).
_LEGACY_IMAGE_FILE_NAME_RE = re.compile(r"^image_[0-9a-f]{16}\.[A-Za-z0-9]{1,8}$")

_EXTENSION_BY_MIME = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}

# IN-list chunk for bulk release queries.
_CHUNK = 500


class ImageDecodeError(MessageParseError):
    """An inline image that is not valid base64 or not a decodable image (422)."""


@dataclass(frozen=True)
class PreparedImage:
    """A decoded, inspected inline image, ready to be written."""

    data: bytes
    sha256: str
    mime_type: str
    width: int
    height: int

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def extension(self) -> str:
        ext = _EXTENSION_BY_MIME.get(self.mime_type)
        if ext:
            return ext
        subtype = self.mime_type.split("/")[-1].lower()
        return re.sub(r"[^a-z0-9]", "", subtype)[:8] or "img"


# --------------------------------------------------------------------------- #
# Decoding (CPU; always called through asyncio.to_thread)                      #
# --------------------------------------------------------------------------- #


def _max_image_bytes() -> int:
    from mirix.constants import MAX_IMAGE_BYTES

    return MAX_IMAGE_BYTES


def _too_large(size: int) -> ImageDecodeError:
    return ImageDecodeError(
        f"image is {size} bytes; the limit is {_max_image_bytes()} bytes per image"
    )


def decode_image_payload(
    data_b64: str, declared_mime_type: Optional[str] = None
) -> PreparedImage:
    """Decode base64 image data and read its format and size with Pillow.

    The decoded size is bounded before decoding (from the base64 length), so an
    oversized payload is rejected without materializing it.
    """
    compact = "".join(data_b64.split())
    compact += "=" * (-len(compact) % 4)
    estimated = len(compact) // 4 * 3
    if estimated > _max_image_bytes() + 2:
        raise _too_large(estimated)
    try:
        raw = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("image data is not valid base64") from exc
    return inspect_image_bytes(raw, declared_mime_type)


def inspect_image_bytes(
    raw: bytes, declared_mime_type: Optional[str] = None
) -> PreparedImage:
    """Read an image's format and size with Pillow (header only, no full decode).

    Only the formats every supported model provider accepts
    (``ALLOWED_IMAGE_FORMATS``: PNG, JPEG, GIF, WEBP) pass, within
    ``MAX_IMAGE_BYTES`` bytes and ``MAX_IMAGE_PIXELS`` pixels; anything else is
    an :class:`ImageDecodeError` (a 422 at ingest) rather than a provider error
    later, in the worker or the distiller.
    """
    from mirix.constants import ALLOWED_IMAGE_FORMATS, MAX_IMAGE_PIXELS

    if not raw:
        raise ImageDecodeError("image data is empty")
    if len(raw) > _max_image_bytes():
        raise _too_large(len(raw))

    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            image_format = img.format
    except Image.DecompressionBombError as exc:
        raise ImageDecodeError(
            "image dimensions exceed the decoder's safety limit"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — any decoder failure is a bad payload
        raise ImageDecodeError(
            f"image data is not a decodable image ({type(exc).__name__})"
        ) from exc

    if (image_format or "").upper() not in ALLOWED_IMAGE_FORMATS:
        raise ImageDecodeError(
            f"unsupported image format {image_format or 'unknown'!s}; supported: "
            + ", ".join(ALLOWED_IMAGE_FORMATS)
        )
    if int(width) * int(height) > MAX_IMAGE_PIXELS:
        raise ImageDecodeError(
            f"image is {width}x{height} pixels; the limit is {MAX_IMAGE_PIXELS} pixels"
        )

    mime_type = Image.MIME.get(image_format or "") or declared_mime_type
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "application/octet-stream"
    return PreparedImage(
        data=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type=mime_type,
        width=int(width),
        height=int(height),
    )


async def prepare_image(part: ImagePart) -> Optional[PreparedImage]:
    """Decode an inline part once (cached on the part); None for a remote one."""
    if part.is_remote:
        return None
    if part.prepared is None:
        if not part.data_b64:
            raise ImageDecodeError("image part has no data")
        part.prepared = await asyncio.to_thread(
            decode_image_payload, part.data_b64, part.declared_mime_type
        )
    return part.prepared


async def prepare_images(parts: Iterable[ImagePart]) -> None:
    """Decode every inline image up front so a bad one fails before any write.

    The request's inline images together may not exceed
    ``MAX_IMAGE_BYTES_PER_REQUEST`` decoded bytes.
    """
    from mirix.constants import MAX_IMAGE_BYTES_PER_REQUEST

    total = 0
    for part in parts:
        prepared = await prepare_image(part)
        if prepared is None:
            continue
        total += prepared.size
        if total > MAX_IMAGE_BYTES_PER_REQUEST:
            raise ImageDecodeError(
                f"the request's images exceed {MAX_IMAGE_BYTES_PER_REQUEST} bytes in total"
            )


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #


def images_root(images_dir: Union[str, Path, None] = None) -> Path:
    if images_dir is not None:
        return Path(images_dir)
    from mirix.settings import settings

    return Path(settings.images_dir or (Path.home() / ".mirix" / "images"))


def scope_dir_name(organization_id: str, user_id: Optional[str]) -> str:
    """A path-safe directory name for one (organization, user)."""
    key = f"{organization_id}\x00{user_id or ''}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def image_path(
    prepared: PreparedImage,
    *,
    organization_id: str,
    user_id: Optional[str],
    images_dir: Union[str, Path, None] = None,
) -> Path:
    return (
        images_root(images_dir)
        / scope_dir_name(organization_id, user_id)
        / f"{prepared.sha256}.{prepared.extension}"
    )


def is_managed_image_path(path: Optional[str]) -> bool:
    """True only for image files MIRIX itself wrote.

    That is ``<32-hex scope>/<sha256>.<ext>`` (this module's layout), or a
    legacy ``image_<16 hex>.<ext>`` directly inside the images root (the
    pre-scope layout). Anything else a ``files`` row may point at — a caller's
    own file recorded by path, say — is never unlinked.
    """
    if not path:
        return False
    candidate = Path(path)
    if _IMAGE_FILE_NAME_RE.match(candidate.name) and _SCOPE_DIR_RE.match(
        candidate.parent.name
    ):
        managed_dir = candidate.parent.parent
    elif _LEGACY_IMAGE_FILE_NAME_RE.match(candidate.name):
        managed_dir = candidate.parent
    else:
        return False
    try:
        return os.path.realpath(managed_dir) == os.path.realpath(images_root())
    except (OSError, ValueError):
        return False


def _write_image_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically unless an identical file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size == len(data):
            return
    except FileNotFoundError:
        pass
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _tombstone_paths(paths: Sequence[str]) -> Dict[str, str]:
    """Atomically move each managed file aside; path -> its tombstone."""
    moved: Dict[str, str] = {}
    for raw in paths:
        if not is_managed_image_path(raw):
            continue
        tomb = f"{raw}.released-{os.urandom(6).hex()}"
        try:
            os.replace(raw, tomb)
            moved[raw] = tomb
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not release image file %s: %s", raw, exc)
    return moved


def _settle_tombstones(moved: Dict[str, str], restore: Iterable[str]) -> int:
    """Put back the tombstones in ``restore``; delete the others."""
    restore = set(restore)
    removed = 0
    for raw, tomb in moved.items():
        try:
            if raw in restore:
                # A store of the same bytes created a new reference meanwhile.
                os.replace(tomb, raw)
            else:
                os.unlink(tomb)
                removed += 1
        except OSError as exc:
            logger.warning("Could not settle released image file %s: %s", raw, exc)
    return removed


async def _referenced_paths(paths: Sequence[str], session_maker) -> set:
    from mirix.orm.file import FileMetadata as FileModel

    if session_maker is None:
        from mirix.server.server import db_context

        session_maker = db_context
    referenced: set = set()
    async with session_maker() as db:
        for chunk in _chunks(sorted(paths)):
            result = await db.execute(
                select(FileModel.file_path).where(FileModel.file_path.in_(chunk))
            )
            referenced.update(result.scalars().all())
    return referenced


async def unlink_image_paths(paths: Sequence[str], *, session_maker=None) -> int:
    """Delete released image files from disk (never anything unmanaged).

    Race-safe against a concurrent store of the same bytes, which writes the
    file, commits a new ``files`` row and then writes the file again if it is
    missing: each file is first moved aside atomically, the ``files`` table is
    re-checked, and a file that gained a reference meanwhile is moved back.
    Whatever the interleaving, a file with a live ``files`` row ends up on disk
    (the store's own second write restores it if it ran while the file was
    aside).
    """
    candidates = [p for p in dict.fromkeys(paths or []) if is_managed_image_path(p)]
    if not candidates:
        return 0
    moved = await asyncio.to_thread(_tombstone_paths, candidates)
    if not moved:
        return 0
    try:
        referenced = await _referenced_paths(list(moved), session_maker)
    except Exception:  # noqa: BLE001 — unsure: keep every file
        logger.exception("Could not re-check released image files; keeping them")
        referenced = set(moved)
    return await asyncio.to_thread(_settle_tombstones, moved, referenced)


# --------------------------------------------------------------------------- #
# Store                                                                        #
# --------------------------------------------------------------------------- #


def _file_manager(file_manager=None):
    if file_manager is not None:
        return file_manager
    from mirix.services.file_manager import FileManager

    return FileManager()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


async def store_image_part(
    part: Union[ImagePart, PreparedImage],
    *,
    organization_id: str,
    user_id: Optional[str],
    file_manager=None,
    images_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Store one image part and return its ``content_parts`` image_ref.

    Inline data is decoded (if not yet prepared), written under the user's
    scope directory and recorded in ``files`` with ``user_id``; a remote URL is
    only recorded. The caller must have created the user first (``files.user_id``
    is a foreign key). Raises :class:`ImageDecodeError` for undecodable data.
    """
    detail = part.detail if isinstance(part, ImagePart) else None
    if isinstance(part, ImagePart) and part.is_remote:
        return await store_remote_image_ref(
            part.url,
            organization_id=organization_id,
            user_id=user_id,
            detail=detail,
            file_manager=file_manager,
        )
    prepared = part if isinstance(part, PreparedImage) else await prepare_image(part)
    assert prepared is not None

    path = image_path(
        prepared,
        organization_id=organization_id,
        user_id=user_id,
        images_dir=images_dir,
    )
    await asyncio.to_thread(_write_image_file, path, prepared.data)

    from mirix.schemas.file import FileMetadata as PydanticFileMetadata

    now = _now_iso()
    metadata = await _file_manager(file_manager).create_file_metadata(
        PydanticFileMetadata(
            organization_id=organization_id,
            user_id=user_id,
            file_name=path.name,
            file_path=str(path),
            file_type=prepared.mime_type,
            file_size=prepared.size,
            file_creation_date=now,
            file_last_modified_date=now,
        )
    )
    # A release of the same bytes (same user) may have unlinked the file between
    # our write and our row becoming visible; the row now exists, so restore it.
    await asyncio.to_thread(_write_image_file, path, prepared.data)

    ref: Dict[str, Any] = {
        "type": CONTENT_PART_IMAGE_REF,
        "file_id": metadata.id,
        "mime_type": prepared.mime_type,
        "width": prepared.width,
        "height": prepared.height,
        "sha256": prepared.sha256,
        "bytes": prepared.size,
    }
    if detail:
        ref["detail"] = detail
    return ref


async def store_remote_image_ref(
    url: str,
    *,
    organization_id: str,
    user_id: Optional[str],
    detail: Optional[str] = None,
    file_manager=None,
) -> Dict[str, Any]:
    """Record a remote image by address (never downloaded) and return its ref."""
    from mirix.schemas.file import FileMetadata as PydanticFileMetadata

    name = PurePosixPath(urlparse(url).path).name or "remote_image"
    guessed = mimetypes.guess_type(name)[0]
    mime_type = guessed if guessed and guessed.startswith("image/") else None
    now = _now_iso()
    metadata = await _file_manager(file_manager).create_file_metadata(
        PydanticFileMetadata(
            organization_id=organization_id,
            user_id=user_id,
            file_name=name[:255],
            file_path=None,
            source_url=url,
            file_type=mime_type,
            file_size=None,
            file_creation_date=now,
            file_last_modified_date=now,
        )
    )
    ref: Dict[str, Any] = {
        "type": CONTENT_PART_IMAGE_REF,
        "file_id": metadata.id,
        "url": url,
    }
    if mime_type:
        ref["mime_type"] = mime_type
    if detail:
        ref["detail"] = detail
    return ref


# --------------------------------------------------------------------------- #
# Release                                                                      #
# --------------------------------------------------------------------------- #


def _chunks(values: Sequence[str]) -> List[List[str]]:
    return [list(values[i : i + _CHUNK]) for i in range(0, len(values), _CHUNK)]


async def release_image_files(
    db,
    *,
    organization_id: Optional[str],
    file_ids: Sequence[str],
) -> List[str]:
    """Delete these ``files`` rows in ``db``'s transaction.

    Returns the disk paths that no remaining ``files`` row references; the
    caller unlinks them with :func:`unlink_image_paths` AFTER committing (a
    rolled-back delete must not have lost the bytes). ``organization_id``
    scopes the delete (None = any organization, for cross-org erasure).
    """
    from mirix.orm.file import FileMetadata as FileModel

    ids = list(dict.fromkeys(fid for fid in file_ids if fid))
    if not ids:
        return []
    paths: set = set()
    owners: set = set()
    for chunk in _chunks(ids):
        stmt = select(FileModel.id, FileModel.file_path, FileModel.user_id).where(
            FileModel.id.in_(chunk)
        )
        if organization_id is not None:
            stmt = stmt.where(FileModel.organization_id == organization_id)
        rows = (await db.execute(stmt)).all()
        if not rows:
            continue
        for _, path, owner in rows:
            if path and is_managed_image_path(path):
                paths.add(path)
                owners.add(owner)
        await db.execute(
            delete(FileModel)
            .where(FileModel.id.in_([file_id for file_id, _, _ in rows]))
            .execution_options(synchronize_session=False)
        )
    if not paths:
        return []
    # A managed path lives under its (organization, user) scope directory, so
    # only rows of the same user can still point at it: filtering by user keeps
    # this on the files.user_id index instead of scanning every file.
    still_used: set = set()
    ordered = sorted(paths)
    owner_ids = [o for o in owners if o is not None]
    owner_filters = []
    if owner_ids:
        owner_filters.append(FileModel.user_id.in_(owner_ids))
    if None in owners:
        owner_filters.append(FileModel.user_id.is_(None))
    for chunk in _chunks(ordered):
        result = await db.execute(
            select(FileModel.file_path).where(
                FileModel.file_path.in_(chunk), or_(*owner_filters)
            )
        )
        still_used.update(result.scalars().all())
    return [path for path in ordered if path not in still_used]


async def discard_image_refs(
    refs: Sequence[Dict[str, Any]],
    *,
    organization_id: str,
    session_maker=None,
) -> None:
    """Best-effort undo of images stored for a write that then failed."""
    file_ids = [
        r.get("file_id") for r in refs if isinstance(r, dict) and r.get("file_id")
    ]
    if not file_ids:
        return
    if session_maker is None:
        from mirix.server.server import db_context

        session_maker = db_context
    try:
        async with session_maker() as db:
            paths = await release_image_files(
                db, organization_id=organization_id, file_ids=file_ids
            )
            await db.commit()
        await unlink_image_paths(paths, session_maker=session_maker)
    except Exception:  # noqa: BLE001 — cleanup must never mask the real error
        logger.exception(
            "Could not discard %d image(s) of a failed write", len(file_ids)
        )


def message_image_ids(contents: Iterable[Any]) -> List[str]:
    """Image ``files`` ids referenced by agent messages' content parts.

    ``contents`` holds ``messages.content`` values: lists of MessageContent
    objects or of their JSON dicts (``{"type": "image_url", "image_id": ...}``).
    """
    seen: Dict[str, None] = {}
    for content in contents:
        for part in content or []:
            if isinstance(part, dict):
                part_type, image_id = part.get("type"), part.get("image_id")
            else:
                part_type = getattr(part, "type", None)
                image_id = getattr(part, "image_id", None)
            part_type = getattr(part_type, "value", part_type)
            if part_type == "image_url" and isinstance(image_id, str) and image_id:
                seen.setdefault(image_id, None)
    return list(seen)


async def _ids_mentioned(db, column, filters, file_ids: Sequence[str]) -> set:
    """The ``file_ids`` whose id appears in ``column`` (JSON cast to text) of a
    row matching ``filters``. A LIKE prefilter narrows the scan; each hit is
    confirmed on the text itself (ids never contain LIKE wildcards that could
    hide one, and a false positive only keeps an image)."""
    from sqlalchemy import String, cast

    ids = list(dict.fromkeys(fid for fid in file_ids if fid))
    text_column = cast(column, String)
    found: set = set()
    for start in range(0, len(ids), 100):
        chunk = ids[start : start + 100]
        stmt = select(text_column).where(
            *filters,
            column.is_not(None),
            or_(*[text_column.like(f"%{fid}%") for fid in chunk]),
        )
        for (text,) in (await db.execute(stmt)).all():
            for fid in chunk:
                if text and fid in text:
                    found.add(fid)
    return found


async def file_ids_referenced_by_agent_messages(
    db,
    *,
    organization_id: Optional[str],
    file_ids: Sequence[str],
    user_id: Optional[str] = None,
) -> set:
    """The ``file_ids`` an agent ``messages`` row (of this organization, and
    user when given) still references in its content."""
    from mirix.orm.message import Message as MessageModel

    filters = []
    if organization_id is not None:
        filters.append(MessageModel.organization_id == organization_id)
    if user_id is not None:
        filters.append(MessageModel.user_id == user_id)
    return await _ids_mentioned(db, MessageModel.content, filters, file_ids)


async def file_ids_referenced_by_turns(
    db, *, organization_id: Optional[str], file_ids: Sequence[str]
) -> set:
    """The ``file_ids`` a conversation-store turn of this organization still
    references in its ``content_parts``."""
    from mirix.orm.conversation_message import (
        ConversationMessage as ConversationMessageModel,
    )

    filters = []
    if organization_id is not None:
        filters.append(ConversationMessageModel.organization_id == organization_id)
    return await _ids_mentioned(
        db, ConversationMessageModel.content_parts, filters, file_ids
    )


async def release_message_images(db, *, file_ids: Sequence[str]) -> List[str]:
    """Release the images of agent messages just deleted in ``db``.

    Runs inside the deleting transaction. An image still referenced elsewhere
    is kept: by a remaining ``messages`` row of the same organization (the meta
    agent forwards its input's images to the memory sub-agents by id), or by a
    conversation-store turn (a caller may reference a stored image by id). The
    others lose their ``files`` rows here, but only rows MIRIX wrote for an
    image (a managed path, see :func:`is_managed_image_path`, or a remote
    image recorded by address); other rows a message referenced by id are
    never deleted. Returns the disk paths no ``files``
    row references any more — unlink them with :func:`unlink_image_paths`
    AFTER committing.
    """
    from mirix.orm.file import FileMetadata as FileModel

    ids = list(dict.fromkeys(fid for fid in file_ids if fid))
    if not ids:
        return []
    org_by_id: Dict[str, Optional[str]] = {}
    for chunk in _chunks(ids):
        for file_id, org_id, path, source_url in (
            await db.execute(
                select(
                    FileModel.id,
                    FileModel.organization_id,
                    FileModel.file_path,
                    FileModel.source_url,
                ).where(FileModel.id.in_(chunk))
            )
        ).all():
            # A message may reference any files row by id (database_image_id);
            # only rows MIRIX itself wrote for an image (a managed file, or a
            # remote image recorded by address) are ours to drop.
            if is_managed_image_path(path) or (path is None and source_url):
                org_by_id[file_id] = org_id
    by_org: Dict[Optional[str], List[str]] = {}
    for file_id, org_id in org_by_id.items():
        by_org.setdefault(org_id, []).append(file_id)
    still_referenced: set = set()
    for org_id, org_ids in by_org.items():
        still_referenced |= await file_ids_referenced_by_agent_messages(
            db, organization_id=org_id, file_ids=org_ids
        )
        still_referenced |= await file_ids_referenced_by_turns(
            db, organization_id=org_id, file_ids=org_ids
        )
    releasable = [fid for fid in org_by_id if fid not in still_referenced]
    return await release_image_files(db, organization_id=None, file_ids=releasable)


async def delete_user_image_files(user_id: str, *, session_maker=None) -> int:
    """Erasure: delete every image ``files`` row attributed to ``user_id``.

    Covers the meta-agent copies too (whose ``messages`` rows the same erasure
    deletes). Rows of other users and unattributed legacy rows are untouched;
    disk files are removed once unreferenced. Returns the rows deleted.
    """
    from mirix.orm.file import FileMetadata as FileModel

    if session_maker is None:
        from mirix.server.server import db_context

        session_maker = db_context
    async with session_maker() as db:
        file_ids = list(
            (await db.execute(select(FileModel.id).where(FileModel.user_id == user_id)))
            .scalars()
            .all()
        )
        paths = await release_image_files(db, organization_id=None, file_ids=file_ids)
        await db.commit()
    await unlink_image_paths(paths, session_maker=session_maker)
    return len(file_ids)
