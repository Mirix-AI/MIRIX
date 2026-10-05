"""Skill version history: a git-style content-addressed store.

Every change of a procedural-memory skill is one "commit" in ``skill_version``;
the seven versioned field values live, deduplicated and zlib-compressed, in
``skill_object``, keyed by ``(organization_id, sha256)``. This module holds the
pieces :meth:`ProceduralMemoryManager.apply_skill_change` is built from:

* canonical encoding — text fields are their UTF-8 bytes; JSON fields are
  ``json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))``;
  the sha256 is taken over those bytes BEFORE compression, so equal content is
  stored once per organization whatever skill or version it belongs to;
* object upsert (``INSERT ... ON CONFLICT DO UPDATE SET last_referenced_at``),
  object load with an integrity check, and garbage collection of objects no
  version references any more (hard delete);
* version rows: latest version, insert, content reconstruction;
* the retrieval text of a skill — the text its one vector embeds;
* advisory-lock keys and the lock statements (Postgres), and the SQLite
  write-lock equivalent.

Locking protocol (Postgres; see ``ProceduralMemoryManager``):

* every WRITER takes ``pg_advisory_xact_lock_shared(org key)``, then — only
  when it claims a skill name (create, rename, restore) — the exclusive
  ``pg_advisory_xact_lock((org, user, name) key)``, then ``SELECT ... FOR
  UPDATE`` on the skill row;
* a hard delete (erasure) takes ``pg_advisory_xact_lock(org key)``
  EXCLUSIVELY before it garbage-collects objects, so no writer can reference an
  object between the collector's "is it referenced?" check and its delete.

All helpers are pure CPU work or awaited SQL on the caller's session: nothing
here blocks the event loop (compression of a skill field — at most tens of KB —
is an allowed pure-CPU helper).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
import zlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import delete, select, text, update

from mirix.orm.errors import NoResultFound
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralMemoryModel
from mirix.orm.skill_object import SkillObject as SkillObjectModel
from mirix.orm.skill_version import SkillVersion as SkillVersionModel
from mirix.schemas.skill_version import SKILL_VERSIONED_FIELDS
from mirix.services.conversation_session_manager import dialect_name

# The versioned fields split by canonical encoding.
TEXT_FIELDS: Tuple[str, ...] = ("name", "entry_type", "description", "instructions")
JSON_FIELDS: Tuple[str, ...] = ("triggers", "examples", "filter_tags")
assert set(TEXT_FIELDS) | set(JSON_FIELDS) == set(SKILL_VERSIONED_FIELDS)

# change_type values.
CHANGE_BASELINE = "baseline"
CHANGE_CREATE = "create"
CHANGE_EDIT = "edit"
CHANGE_DELETE = "delete"
CHANGE_RESTORE = "restore"
CHANGE_TYPES = frozenset(
    {CHANGE_BASELINE, CHANGE_CREATE, CHANGE_EDIT, CHANGE_DELETE, CHANGE_RESTORE}
)

# zlib level: skills are small; 6 is zlib's own default speed/ratio balance.
_ZLIB_LEVEL = 6
# Bound IN (...) lists (SQLite's host-parameter limit is 32766 on modern
# builds, 999 on old ones).
_IN_CHUNK = 500

# Advisory-lock namespaces: stable salts so these keys never collide with
# another feature's pg_advisory_lock keyspace (e.g. the procedural dream guard).
_ORG_LOCK_NAMESPACE = "mirix.skill_store.org"
_NAME_LOCK_NAMESPACE = "mirix.skill_store.name"


class SkillHistoryError(Exception):
    """Base class for skill-history failures."""


class SkillVersionNotFound(SkillHistoryError, NoResultFound):
    """The requested version does not exist for this skill."""


class SkillObjectCorrupt(SkillHistoryError):
    """A stored object is missing or does not hash to its key."""


# --------------------------------------------------------------------------- #
# Canonical encoding                                                           #
# --------------------------------------------------------------------------- #


def _get(source: Any, field: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(field)
    return getattr(source, field, None)


def canonical_bytes(field: str, value: Any) -> bytes:
    """The canonical bytes of one versioned field value.

    Text fields: their UTF-8 bytes (``None`` — never valid for a skill's text
    fields, which are NOT NULL — encodes as the empty string). JSON fields:
    ``json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))``.
    """
    if field in TEXT_FIELDS:
        return ("" if value is None else str(value)).encode("utf-8")
    if field in JSON_FIELDS:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    raise ValueError(f"{field!r} is not a versioned skill field")


def decode_field(field: str, raw: bytes) -> Any:
    """Inverse of :func:`canonical_bytes`."""
    decoded = raw.decode("utf-8")
    if field in TEXT_FIELDS:
        return decoded
    if field in JSON_FIELDS:
        return json.loads(decoded)
    raise ValueError(f"{field!r} is not a versioned skill field")


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def content_of(source: Any) -> Dict[str, Any]:
    """The seven versioned field values of a skill row, schema or mapping."""
    return {field: _get(source, field) for field in SKILL_VERSIONED_FIELDS}


def encode_content(content: Mapping[str, Any]) -> Dict[str, Tuple[str, bytes]]:
    """``{field: (sha256, canonical bytes)}`` for the seven versioned fields."""
    encoded = {}
    for field in SKILL_VERSIONED_FIELDS:
        raw = canonical_bytes(field, content.get(field))
        encoded[field] = (sha256_hex(raw), raw)
    return encoded


def hashes_of(encoded: Mapping[str, Tuple[str, bytes]]) -> Dict[str, str]:
    return {field: encoded[field][0] for field in SKILL_VERSIONED_FIELDS}


def version_hashes(version: Any) -> Dict[str, str]:
    """``{field: sha256}`` of a skill_version row (ORM or schema)."""
    return {
        field: getattr(version, f"{field}_sha256") for field in SKILL_VERSIONED_FIELDS
    }


def changed_fields(
    before: Optional[Mapping[str, str]], after: Mapping[str, str]
) -> List[str]:
    """Versioned fields whose hash differs (all of them when there is no parent)."""
    if before is None:
        return list(SKILL_VERSIONED_FIELDS)
    return [f for f in SKILL_VERSIONED_FIELDS if before.get(f) != after.get(f)]


def skill_retrieval_text(item: Any) -> str:
    """The text a skill's single retrieval vector embeds.

    ``name``, then each trigger on its own line, then ``description``. The
    ``instructions`` never enter it. Empty parts are skipped.
    """
    parts: List[str] = []
    name = _get(item, "name")
    if name:
        parts.append(str(name))
    triggers = _get(item, "triggers") or []
    if isinstance(triggers, str):
        triggers = [triggers]
    parts.extend(str(t) for t in triggers if t)
    description = _get(item, "description")
    if description:
        parts.append(str(description))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Locks                                                                        #
# --------------------------------------------------------------------------- #


def _advisory_key(namespace: str, *parts: Optional[str]) -> int:
    """A stable signed 64-bit advisory-lock key (blake2b, not salted hash())."""
    raw = ":".join([namespace, *((p or "-") for p in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return value - (1 << 64) if value >= (1 << 63) else value


def org_lock_key(organization_id: Optional[str]) -> int:
    return _advisory_key(_ORG_LOCK_NAMESPACE, organization_id)


def name_lock_key(
    organization_id: Optional[str], user_id: Optional[str], name: str
) -> int:
    return _advisory_key(_NAME_LOCK_NAMESPACE, organization_id, user_id, name)


async def lock_for_write(
    db,
    *,
    organization_ids: Iterable[Optional[str]],
    name_keys: Iterable[int] = (),
) -> None:
    """Take a writer's locks in the transaction ``db`` has open.

    Postgres: the SHARED org lock of each organization (sorted), then the
    exclusive name locks (sorted, deduplicated) — one global order, so two
    writers can never deadlock on these. SQLite: a zero-row UPDATE, which takes
    the database write lock up front (SQLite has one writer at a time; taking
    it before any read makes the version-number read and the write atomic).
    """
    dialect = dialect_name(db)
    if dialect == "postgresql":
        for key in sorted({org_lock_key(o) for o in organization_ids}):
            await db.execute(
                text("SELECT pg_advisory_xact_lock_shared(:k)"), {"k": key}
            )
        for key in sorted(set(name_keys)):
            await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})
    elif dialect == "sqlite":
        await acquire_sqlite_write_lock(db)


async def lock_for_purge(db, *, organization_ids: Iterable[Optional[str]]) -> None:
    """Take the EXCLUSIVE org lock(s) a hard delete needs before collecting
    objects (Postgres); the SQLite write lock otherwise."""
    dialect = dialect_name(db)
    if dialect == "postgresql":
        for key in sorted({org_lock_key(o) for o in organization_ids}):
            await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})
    elif dialect == "sqlite":
        await acquire_sqlite_write_lock(db)


async def acquire_sqlite_write_lock(db) -> None:
    """Begin SQLite's write transaction now: a zero-row UPDATE takes the
    database write (RESERVED) lock, so every read that follows sees a state no
    other writer can change before this transaction commits."""
    await db.execute(
        update(SkillObjectModel)
        .where(SkillObjectModel.sha256 == "")
        .values(raw_length=SkillObjectModel.raw_length)
        .execution_options(synchronize_session=False)
    )


# --------------------------------------------------------------------------- #
# Objects                                                                      #
# --------------------------------------------------------------------------- #


def _chunks(values: Sequence[str], size: int = _IN_CHUNK):
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def upsert_objects(
    db,
    *,
    organization_id: str,
    encoded: Mapping[str, Tuple[str, bytes]],
    now: dt.datetime,
) -> None:
    """Write the objects of ``encoded`` (``{field: (sha, raw)}``) once each.

    ``INSERT ... ON CONFLICT (organization_id, sha256) DO UPDATE SET
    last_referenced_at`` — an object that already exists is only touched. Rows
    go in sha order, so two writers upserting overlapping sets lock the object
    rows in the same order and cannot deadlock.
    """
    by_sha: Dict[str, bytes] = {}
    for sha, raw in encoded.values():
        by_sha[sha] = raw
    if not by_sha:
        return
    rows = [
        {
            "organization_id": organization_id,
            "sha256": sha,
            "content_zlib": zlib.compress(raw, _ZLIB_LEVEL),
            "raw_length": len(raw),
            "created_at": now,
            "last_referenced_at": now,
        }
        for sha, raw in sorted(by_sha.items())
    ]
    dialect = dialect_name(db)
    if dialect in ("postgresql", "sqlite"):
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        stmt = dialect_insert(SkillObjectModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["organization_id", "sha256"],
            set_={"last_referenced_at": stmt.excluded.last_referenced_at},
        )
        await db.execute(stmt)
        return
    # Other dialects: look first, insert the missing, touch the rest.
    existing = set(
        (
            await db.execute(
                select(SkillObjectModel.sha256).where(
                    SkillObjectModel.organization_id == organization_id,
                    SkillObjectModel.sha256.in_(list(by_sha)),
                )
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if row["sha256"] in existing:
            await db.execute(
                update(SkillObjectModel)
                .where(
                    SkillObjectModel.organization_id == organization_id,
                    SkillObjectModel.sha256 == row["sha256"],
                )
                .values(last_referenced_at=now)
            )
        else:
            db.add(SkillObjectModel(**row))
    await db.flush()


async def load_objects(
    db, *, organization_id: str, shas: Iterable[str]
) -> Dict[str, bytes]:
    """``{sha: canonical bytes}``, each decompressed and verified against its key."""
    wanted = sorted(set(shas))
    found: Dict[str, bytes] = {}
    for chunk in _chunks(wanted):
        rows = (
            await db.execute(
                select(
                    SkillObjectModel.sha256,
                    SkillObjectModel.content_zlib,
                    SkillObjectModel.raw_length,
                ).where(
                    SkillObjectModel.organization_id == organization_id,
                    SkillObjectModel.sha256.in_(chunk),
                )
            )
        ).all()
        for sha, blob, raw_length in rows:
            raw = zlib.decompress(blob)
            if len(raw) != raw_length or sha256_hex(raw) != sha:
                raise SkillObjectCorrupt(
                    f"skill object {sha} of organization {organization_id} "
                    "does not match its hash"
                )
            found[sha] = raw
    missing = [sha for sha in wanted if sha not in found]
    if missing:
        raise SkillObjectCorrupt(
            f"skill objects missing for organization {organization_id}: {missing}"
        )
    return found


async def referenced_hashes(
    db, *, organization_id: str, candidates: Iterable[str]
) -> set:
    """The subset of ``candidates`` some version of the organization references."""
    wanted = sorted(set(candidates))
    referenced: set = set()
    for chunk in _chunks(wanted):
        for field in SKILL_VERSIONED_FIELDS:
            column = getattr(SkillVersionModel, f"{field}_sha256")
            rows = await db.execute(
                select(column)
                .where(
                    SkillVersionModel.organization_id == organization_id,
                    column.in_(chunk),
                )
                .distinct()
            )
            referenced.update(rows.scalars().all())
    return referenced


async def purge_unreferenced_objects(
    db, *, organization_id: str, candidates: Iterable[str]
) -> int:
    """Delete the ``candidates`` no version of the organization references.

    Caller must hold the EXCLUSIVE org lock (:func:`lock_for_purge`) — then no
    writer can add a reference between the check and the delete.
    """
    wanted = sorted(set(candidates))
    if not wanted:
        return 0
    orphans = sorted(
        set(wanted)
        - await referenced_hashes(
            db, organization_id=organization_id, candidates=wanted
        )
    )
    deleted = 0
    for chunk in _chunks(orphans):
        result = await db.execute(
            delete(SkillObjectModel)
            .where(
                SkillObjectModel.organization_id == organization_id,
                SkillObjectModel.sha256.in_(chunk),
            )
            .execution_options(synchronize_session=False)
        )
        deleted += result.rowcount or 0
    return deleted


# --------------------------------------------------------------------------- #
# Versions                                                                     #
# --------------------------------------------------------------------------- #


async def latest_version(db, skill_id: str) -> Optional[SkillVersionModel]:
    return (
        await db.execute(
            select(SkillVersionModel)
            .where(SkillVersionModel.skill_id == skill_id)
            .order_by(SkillVersionModel.version_no.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def list_versions(db, skill_id: str) -> List[SkillVersionModel]:
    return list(
        (
            await db.execute(
                select(SkillVersionModel)
                .where(SkillVersionModel.skill_id == skill_id)
                .order_by(SkillVersionModel.version_no.asc())
            )
        )
        .scalars()
        .all()
    )


async def get_version(db, skill_id: str, version_no: int) -> SkillVersionModel:
    row = (
        await db.execute(
            select(SkillVersionModel).where(
                SkillVersionModel.skill_id == skill_id,
                SkillVersionModel.version_no == version_no,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise SkillVersionNotFound(f"Skill {skill_id} has no version {version_no}.")
    return row


async def insert_version(
    db,
    *,
    skill_id: str,
    organization_id: str,
    user_id: str,
    version_no: int,
    parent_version_id: Optional[str],
    change_type: str,
    hashes: Mapping[str, str],
    now: dt.datetime,
    run_id: Optional[str] = None,
    actor_kind: Optional[str] = None,
    reason: Optional[str] = None,
    actor_id: Optional[str] = None,
    version_id: Optional[str] = None,
) -> SkillVersionModel:
    """Insert one version row. ``version_id`` fixes its id (a write that may
    be repeated after an ambiguous COMMIT looks its own version up by it)."""
    if change_type not in CHANGE_TYPES:
        raise ValueError(f"unknown change_type {change_type!r}")
    row = SkillVersionModel(
        id=version_id or f"skver-{uuid.uuid4()}",
        organization_id=organization_id,
        user_id=user_id,
        skill_id=skill_id,
        version_no=version_no,
        parent_version_id=parent_version_id,
        change_type=change_type,
        run_id=run_id,
        actor_kind=actor_kind,
        reason=reason,
        created_at=now,
        updated_at=now,
        is_deleted=False,
        **{f"{field}_sha256": hashes[field] for field in SKILL_VERSIONED_FIELDS},
    )
    if actor_id:
        row._created_by_id = actor_id
        row._last_updated_by_id = actor_id
    db.add(row)
    await db.flush()
    return row


async def version_content(
    db, *, organization_id: str, version: Any
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Rebuild a version: ``({field: value}, {field: canonical bytes})``."""
    hashes = version_hashes(version)
    objects = await load_objects(
        db, organization_id=organization_id, shas=hashes.values()
    )
    raws = {field: objects[sha] for field, sha in hashes.items()}
    return {field: decode_field(field, raw) for field, raw in raws.items()}, raws


async def skill_ids_without_history(
    db,
    *,
    organization_id: Optional[str] = None,
    after_id: Optional[str] = None,
    limit: int = 500,
) -> List[str]:
    """Skills (live AND soft-deleted) that have no version yet, by id."""
    has_version = (
        select(SkillVersionModel.id)
        .where(SkillVersionModel.skill_id == ProceduralMemoryModel.id)
        .exists()
    )
    stmt = select(ProceduralMemoryModel.id).where(~has_version)
    if organization_id is not None:
        stmt = stmt.where(ProceduralMemoryModel.organization_id == organization_id)
    if after_id is not None:
        stmt = stmt.where(ProceduralMemoryModel.id > after_id)
    stmt = stmt.order_by(ProceduralMemoryModel.id.asc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())
