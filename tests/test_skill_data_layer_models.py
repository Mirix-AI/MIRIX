"""Unit tests for the skill-learning data layer.

Covers the new tables (conversation_session, skill_object, skill_version,
skill_learning_run, retrieval_log) and the changed ones (conversation_message,
files, skill_experience, procedural_memory):

* ORM <-> Pydantic field-set parity, so `to_pydantic()` can never silently drop
  a column and every schema is `extra="forbid"`;
* the constraints and indexes the data layer requires;
* SQLite `create_all` builds everything and every new model round-trips;
* the schema validators.

Postgres-specific DDL (generated tsvector column, GIN index, partial unique
index, CHECK constraints) is covered by tests/test_skill_data_layer_pg.py.
"""

import datetime as dt
import uuid
import zlib

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import mirix.orm  # noqa: F401 -- register every ORM class on Base.metadata
from mirix.orm.base import Base
from mirix.orm.conversation_message import ConversationMessage as ConversationMessageORM
from mirix.orm.conversation_session import ConversationSession as ConversationSessionORM
from mirix.orm.file import FileMetadata as FileMetadataORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import (
    RETRIEVAL_TSV_EXPRESSION,
    UNIQUE_LIVE_NAME_INDEX,
)
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogORM
from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
from mirix.orm.skill_learning_run import SkillLearningRun as SkillLearningRunORM
from mirix.orm.skill_object import SkillObject as SkillObjectORM
from mirix.orm.skill_version import SkillVersion as SkillVersionORM
from mirix.orm.user import User as UserORM
from mirix.schemas.conversation_message import (
    ConversationMessage,
    ConversationMessageCreate,
)
from mirix.schemas.conversation_session import (
    ConversationSession,
    ConversationSessionCreate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.procedural_memory import ProceduralMemoryItem
from mirix.schemas.retrieval_log import RetrievalLog
from mirix.schemas.skill_experience import SkillExperience, SkillExperienceUpdate
from mirix.schemas.skill_learning_run import (
    SkillLearningRun,
    SkillLearningRunCreate,
)
from mirix.schemas.skill_version import (
    SKILL_VERSIONED_FIELDS,
    SkillObject,
    SkillVersion,
)
from mirix.settings import settings

# Columns present on every SqlalchemyBase table that the read schemas never
# expose (audit ids) — plus is_deleted, which only some schemas surface.
_AUDIT_COLUMNS = {"_created_by_id", "_last_updated_by_id"}

PARITY_CASES = [
    pytest.param(
        ConversationSessionORM, ConversationSession, id="conversation_session"
    ),
    pytest.param(SkillObjectORM, SkillObject, id="skill_object"),
    pytest.param(SkillVersionORM, SkillVersion, id="skill_version"),
    pytest.param(SkillLearningRunORM, SkillLearningRun, id="skill_learning_run"),
    pytest.param(RetrievalLogORM, RetrievalLog, id="retrieval_log"),
    pytest.param(
        ConversationMessageORM, ConversationMessage, id="conversation_message"
    ),
    pytest.param(FileMetadataORM, FileMetadata, id="files"),
    pytest.param(SkillExperienceORM, SkillExperience, id="skill_experience"),
    pytest.param(ProceduralORM, ProceduralMemoryItem, id="procedural_memory"),
]


def _orm_columns(model) -> set:
    return {column.name for column in model.__table__.columns}


# --------------------------------------------------------------------------- #
# ORM <-> Pydantic parity                                                     #
# --------------------------------------------------------------------------- #


# Deferred vector columns no schema carries: reading one in to_pydantic would
# lazy-load it on an async session (the experience pool vector).
_UNSCHEMATIZED_COLUMNS = {"pool_embedding"}


@pytest.mark.parametrize("orm_model, pydantic_model", PARITY_CASES)
def test_orm_and_pydantic_field_sets_match(orm_model, pydantic_model):
    orm = (
        _orm_columns(orm_model)
        - _AUDIT_COLUMNS
        - {"is_deleted", "retrieval_tsv"}
        - _UNSCHEMATIZED_COLUMNS
    )
    schema = set(pydantic_model.model_fields) - {"is_deleted"}
    assert orm == schema, (
        f"ORM-only: {sorted(orm - schema)}; schema-only: {sorted(schema - orm)}"
    )


@pytest.mark.parametrize("orm_model, pydantic_model", PARITY_CASES)
def test_schemas_forbid_unknown_fields(orm_model, pydantic_model):
    assert pydantic_model.model_config.get("extra") == "forbid"
    assert pydantic_model.model_config.get("from_attributes") is True


# --------------------------------------------------------------------------- #
# Required columns, keys and indexes                                          #
# --------------------------------------------------------------------------- #


def _unique_sets(model) -> set:
    table = model.__table__
    sets = {
        tuple(c.name for c in con.columns)
        for con in table.constraints
        if con.__class__.__name__ == "UniqueConstraint"
    }
    sets |= {tuple(c.name for c in ix.columns) for ix in table.indexes if ix.unique}
    return sets


def test_conversation_session_columns_match_spec_m1_1():
    expected = {
        "id",
        "organization_id",
        "user_id",
        "agent_id",
        "session_id",
        "session_tag",
        "first_message_at",
        "last_message_at",
        "message_count",
        "closed_at",
        "close_reason",
        "reopen_count",
        "distill_failures",
        "distill_failed_at",
        "distill_retry_after",
        # The backoff doubles per consecutive infrastructure failure.
        "distill_infra_failures",
        "truncated_message_count",
        "truncated_chars",
        # How the agent's prompt was built, as its client attached it.
        "agent_context",
        "created_at",
        "updated_at",
        "is_deleted",
    }
    assert _orm_columns(ConversationSessionORM) - _AUDIT_COLUMNS == expected
    assert ("organization_id", "user_id", "session_id") in _unique_sets(
        ConversationSessionORM
    )
    table = ConversationSessionORM.__table__
    for nullable in (
        "closed_at",
        "close_reason",
        "distill_failed_at",
        "distill_retry_after",
        "agent_id",
    ):
        assert table.c[nullable].nullable, nullable
    for counter in (
        "message_count",
        "reopen_count",
        "distill_failures",
        "distill_infra_failures",
        "truncated_message_count",
        "truncated_chars",
    ):
        assert not table.c[counter].nullable, counter
        assert table.c[counter].server_default is not None, counter


def test_skill_object_primary_key_is_org_and_sha():
    table = SkillObjectORM.__table__
    assert [c.name for c in table.primary_key.columns] == ["organization_id", "sha256"]
    assert _orm_columns(SkillObjectORM) == {
        "organization_id",
        "sha256",
        "content_zlib",
        "raw_length",
        "created_at",
        "last_referenced_at",
    }


def test_skill_version_columns_and_unique_version_number():
    cols = _orm_columns(SkillVersionORM)
    for field in SKILL_VERSIONED_FIELDS:
        assert f"{field}_sha256" in cols
    assert len(SKILL_VERSIONED_FIELDS) == 7
    for column in (
        "skill_id",
        "version_no",
        "parent_version_id",
        "change_type",
        "run_id",
        "actor_kind",
        "reason",
    ):
        assert column in cols
    # v4 removed per-version experience lineage (reachable via the learning run).
    assert "experience_ids" not in cols
    assert ("skill_id", "version_no") in _unique_sets(SkillVersionORM)
    fk = next(iter(SkillVersionORM.__table__.c.skill_id.foreign_keys))
    assert fk.target_fullname == "procedural_memory.id"
    assert fk.ondelete == "CASCADE"


def test_skill_learning_run_columns_match_spec_m9():
    expected = {
        "id",
        "organization_id",
        "user_id",
        "agent_id",
        "trigger",
        # Every auto_dream run is recorded, so the mode is part of the run.
        "mode",
        "status",
        "session_ids",
        "session_watermarks",
        "heartbeat_at",
        "started_at",
        "finished_at",
        "error",
        "distill_metrics",
        "curator_metrics",
        "llm_usage",
        "embedding_chunked_calls",
        # Every retry is recorded on the run.
        "retry_events",
        "created_at",
        "updated_at",
        "is_deleted",
    }
    assert _orm_columns(SkillLearningRunORM) - _AUDIT_COLUMNS == expected


def test_retrieval_log_columns_match_spec_m7_4():
    expected = {
        "id",
        "organization_id",
        "user_id",
        "client_id",
        "session_id",
        "memory_types",
        "query_text",
        "extracted_topics",
        "extraction_status",
        "embedding_status",
        "per_type",
        "llm_usage",
        "latency_ms",
        # Every retry is recorded on the retrieval's log row.
        "retry_events",
        # The retrieval's chunked embedding calls.
        "embedding_chunked_calls",
        "created_at",
        "updated_at",
        "is_deleted",
    }
    assert _orm_columns(RetrievalLogORM) - _AUDIT_COLUMNS == expected


def test_changed_columns_on_existing_tables():
    cm = ConversationMessageORM.__table__
    assert cm.c.content_parts.nullable and cm.c.original_text_length.nullable
    files = FileMetadataORM.__table__
    assert files.c.user_id.nullable
    fk = next(iter(files.c.user_id.foreign_keys))
    assert fk.target_fullname == "users.id" and fk.ondelete == "SET NULL"
    exp = SkillExperienceORM.__table__
    assert not exp.c.curation_attempts.nullable
    assert exp.c.curation_attempts.server_default is not None
    # No grouping by vector. The only experience vector is the pool's:
    # deferred, never part of the schema, never used for batching.
    assert [c.name for c in exp.columns if "embedding" in c.name] == [
        "pool_embedding",
        "pool_embedding_model",
        "pool_embedding_failures",
    ]
    assert exp.c.pool_embedding.nullable
    assert SkillExperienceORM.pool_embedding.property.deferred
    # Existing rows get the pool's counters from server defaults (no rewrite).
    for column, default in (
        ("recurrence_count", "1"),
        ("ignored_pushes", "0"),
        ("pushed_count", "0"),
        ("pool_embedding_failures", "0"),
    ):
        assert not exp.c[column].nullable
        assert exp.c[column].server_default.arg.text == default
    for column in (
        "disposition",
        "cited_skill_ids",
        "pool_scope",
        "pool_embedding_model",
        "last_pushed_at",
        "pool_retired_at",
        "pool_retire_reason",
    ):
        assert exp.c[column].nullable, column
    assert exp.c.last_pushed_at.type.timezone and exp.c.pool_retired_at.type.timezone


def test_procedural_retrieval_fields():
    table = ProceduralORM.__table__
    assert "retrieval_embedding" in table.c
    assert "description_embedding" not in table.c
    assert "instructions_embedding" not in table.c
    # The generated tsvector column (and its GIN index) exist only in Postgres
    # mode; the SQLite ORM must not declare them or create_all would break.
    if settings.mirix_pg_uri_no_default:
        assert table.c.retrieval_tsv.computed is not None
    else:
        assert "retrieval_tsv" not in table.c
        assert not any(
            ix.name == "ix_procedural_memory_retrieval_tsv" for ix in table.indexes
        )
    # Live-name uniqueness is a partial index on BOTH backends.
    index = next(ix for ix in table.indexes if ix.name == UNIQUE_LIVE_NAME_INDEX)
    assert index.unique
    assert [c.name for c in index.columns] == ["organization_id", "user_id", "name"]
    assert str(index.dialect_options["postgresql"]["where"]) == "NOT is_deleted"
    assert str(index.dialect_options["sqlite"]["where"]) == "NOT is_deleted"
    assert not any(
        con.name == "uq_procedural_memory_org_user_name" for con in table.constraints
    )


def test_retrieval_tsv_expression_matches_the_spec():
    # The expression, with the escapes of the triggers' JSON text
    # removed: `triggers::jsonb::text` for its `triggers::text` (the text of a
    # JSON column keeps json.dumps' \uXXXX escapes, which would index
    # 'u201ccool' instead of 'cool'), then every escape jsonb's
    # text keeps (\n, \t, \", \\, \u00XX ...) replaced by a space (else
    # 'line1\nline2' indexes 'nline2').
    assert RETRIEVAL_TSV_EXPRESSION == (
        "setweight(to_tsvector('english', coalesce(name, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(regexp_replace("
        "triggers::jsonb::text, "
        """'\\\\(["\\\\/bfnrt]|u[0-9a-fA-F]{4})', ' ', 'g'), '')), 'A') || """
        "setweight(to_tsvector('english', coalesce(description, '')), 'B')"
    )


def test_the_triggers_escape_pattern_matches_what_jsonb_text_escapes():
    """The pattern (as Postgres' regex engine reads it) matches each escape of
    Postgres' jsonb text output as one unit, and nothing else. Python's re
    reads this pattern the same way as Postgres' ARE."""
    import json
    import re

    from mirix.orm.procedural_memory import RETRIEVAL_TSV_TRIGGERS_TEXT

    [pattern] = re.findall(r"'([^']*)', ' ', 'g'", RETRIEVAL_TSV_TRIGGERS_TEXT)
    assert pattern == r"""\\(["\\/bfnrt]|u[0-9a-fA-F]{4})"""
    escape = re.compile(pattern)
    triggers = [
        "line1\nline2",
        "tab\there",
        'say "cool"',
        "C:\\new",
        "bell\u0007ring",
        "a/b",
        "café",
    ]
    # json.dumps(ensure_ascii=False) escapes like jsonb's text output.
    as_text = json.dumps(triggers, ensure_ascii=False)
    assert "\\n" in as_text and "\\u0007" in as_text and "\\\\" in as_text
    cleaned = escape.sub(" ", as_text)
    assert "\\" not in cleaned
    assert cleaned == (
        '["line1 line2", "tab here", "say  cool ", "C: new", "bell ring", '
        '"a/b", "café"]'
    )


@pytest.mark.parametrize(
    "orm_model, prefix",
    [
        (ConversationSessionORM, "convsess-"),
        (SkillVersionORM, "skver-"),
        (SkillLearningRunORM, "slrun-"),
        (RetrievalLogORM, "rlog-"),
    ],
)
def test_id_prefixes(orm_model, prefix):
    generated = orm_model.__table__.c.id.default.arg(None)
    assert generated.startswith(prefix)
    # The Pydantic id pattern convention (`<prefix>-<8 hex>...`) accepts it.
    assert uuid.UUID(generated[len(prefix) :])


# --------------------------------------------------------------------------- #
# SQLite create_all + round trips                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(OrganizationORM(id="org-1", name="org-1"))
        session.add(
            UserORM(
                id="user-1",
                name="u",
                organization_id="org-1",
                status="active",
                timezone="UTC",
            )
        )
        session.add(mirix.orm.Agent(id="agent-1", organization_id="org-1"))
        session.commit()
        yield session
    engine.dispose()


def test_sqlite_create_all_builds_every_new_table():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    tables = set(inspect(engine).get_table_names())
    assert {
        "conversation_session",
        "skill_object",
        "skill_version",
        "skill_learning_run",
        "retrieval_log",
    } <= tables
    engine.dispose()


def _now():
    return dt.datetime(2026, 9, 28, 12, 0, tzinfo=dt.timezone.utc)


def test_conversation_session_round_trip_and_unique_key(sqlite_session):
    row = ConversationSessionORM(
        organization_id="org-1",
        user_id="user-1",
        agent_id="agent-1",
        session_id="sess_1",
        session_tag="task",
        first_message_at=_now(),
        last_message_at=_now(),
        message_count=2,
    )
    sqlite_session.add(row)
    sqlite_session.commit()
    model = row.to_pydantic()
    assert model.id.startswith("convsess-")
    assert (model.reopen_count, model.distill_failures, model.truncated_chars) == (
        0,
        0,
        0,
    )
    assert model.closed_at is None and model.close_reason is None

    sqlite_session.add(
        ConversationSessionORM(
            organization_id="org-1",
            user_id="user-1",
            session_id="sess_1",
            first_message_at=_now(),
            last_message_at=_now(),
        )
    )
    with pytest.raises(IntegrityError):
        sqlite_session.commit()


def test_skill_history_round_trip(sqlite_session):
    skill = ProceduralORM(
        id="proc-1",
        organization_id="org-1",
        user_id="user-1",
        name="wash-mug",
        entry_type="workflow",
        description="d",
        instructions="i",
    )
    sqlite_session.add(skill)
    raw = b"wash-mug"
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    sqlite_session.add(
        SkillObjectORM(
            organization_id="org-1",
            sha256=sha,
            content_zlib=zlib.compress(raw),
            raw_length=len(raw),
        )
    )
    version = SkillVersionORM(
        organization_id="org-1",
        user_id="user-1",
        skill_id="proc-1",
        version_no=1,
        change_type="baseline",
        **{f"{f}_sha256": sha for f in SKILL_VERSIONED_FIELDS},
    )
    sqlite_session.add(version)
    sqlite_session.commit()

    obj = sqlite_session.get(SkillObjectORM, ("org-1", sha)).to_pydantic()
    assert zlib.decompress(obj.content_zlib) == raw
    assert obj.raw_length == len(raw)
    assert version.to_pydantic().change_type == "baseline"

    sqlite_session.add(
        SkillVersionORM(
            organization_id="org-1",
            user_id="user-1",
            skill_id="proc-1",
            version_no=1,
            change_type="edit",
            **{f"{f}_sha256": sha for f in SKILL_VERSIONED_FIELDS},
        )
    )
    with pytest.raises(IntegrityError):
        sqlite_session.commit()


def test_learning_run_and_retrieval_log_round_trip(sqlite_session):
    run = SkillLearningRunORM(
        organization_id="org-1",
        user_id="user-1",
        agent_id="agent-1",
        trigger="flush",
        session_ids=["s1"],
        session_watermarks={"s1": _now().isoformat()},
    )
    log = RetrievalLogORM(
        organization_id="org-1",
        user_id="user-1",
        session_id="s1",
        memory_types=["procedural"],
        query_text="how to wash a mug",
        extraction_status="fallback",
        embedding_status="failed",
        per_type={"procedural": {"method": "bm25", "returned": []}},
        latency_ms=12,
    )
    sqlite_session.add_all([run, log])
    sqlite_session.commit()
    run_model = run.to_pydantic()
    assert run_model.status == "queued"
    assert run_model.embedding_chunked_calls == 0
    assert run_model.session_watermarks == {"s1": _now().isoformat()}
    log_model = log.to_pydantic()
    assert log_model.id.startswith("rlog-")
    assert log_model.per_type["procedural"]["method"] == "bm25"


def test_changed_tables_round_trip(sqlite_session):
    parts = [
        {"type": "text", "text": "look:"},
        {"type": "image_ref", "file_id": "file-1", "mime_type": "image/png"},
    ]
    msg = ConversationMessageORM(
        organization_id="org-1",
        user_id="user-1",
        session_id="s1",
        role="user",
        content="look: [image: file-1]",
        content_parts=parts,
        original_text_length=123,
        session_tag="task",
    )
    file_row = FileMetadataORM(
        id="file-0123abcd-0000-4000-8000-000000000001",
        organization_id="org-1",
        user_id="user-1",
        file_type="image/png",
    )
    exp = SkillExperienceORM(
        organization_id="org-1",
        user_id="user-1",
        agent_id="agent-1",
        session_id="s1",
        experience_type="worth_learning",
        title="t",
    )
    sqlite_session.add_all([msg, file_row, exp])
    sqlite_session.commit()
    assert msg.to_pydantic().content_parts == parts
    assert msg.to_pydantic().original_text_length == 123
    assert msg.to_pydantic().session_tag == "task"
    assert file_row.to_pydantic().user_id == "user-1"
    assert exp.to_pydantic().curation_attempts == 0


def test_sqlite_partial_unique_index_allows_reusing_a_deleted_name(sqlite_session):
    def skill(skill_id, deleted=False):
        return ProceduralORM(
            id=skill_id,
            organization_id="org-1",
            user_id="user-1",
            name="same-name",
            entry_type="workflow",
            description="d",
            instructions="i",
            is_deleted=deleted,
        )

    sqlite_session.add(skill("proc-a", deleted=True))
    sqlite_session.add(skill("proc-b"))
    sqlite_session.commit()  # a soft-deleted twin does not block the live name
    sqlite_session.add(skill("proc-c"))
    with pytest.raises(IntegrityError):
        sqlite_session.commit()
    sqlite_session.rollback()
    names = (
        sqlite_session.execute(
            select(ProceduralORM.id).where(ProceduralORM.name == "same-name")
        )
        .scalars()
        .all()
    )
    assert sorted(names) == ["proc-a", "proc-b"]


# --------------------------------------------------------------------------- #
# Validators                                                                  #
# --------------------------------------------------------------------------- #


def _session_kwargs(**overrides):
    base = {
        "session_id": "sess_1",
        "user_id": "user-1",
        "organization_id": "org-1",
        "first_message_at": _now(),
        "last_message_at": _now(),
    }
    base.update(overrides)
    return base


def test_conversation_session_validators():
    assert (
        ConversationSessionCreate(**_session_kwargs(session_tag="task")).session_tag
        == "task"
    )
    assert ConversationSession(**_session_kwargs(session_tag=None)).session_tag is None
    with pytest.raises(ValidationError):
        ConversationSessionCreate(**_session_kwargs(session_tag="chat"))
    with pytest.raises(ValidationError):
        ConversationSessionCreate(**_session_kwargs(session_id="bad id!"))
    with pytest.raises(ValidationError):
        ConversationSession(**_session_kwargs(close_reason="timeout"))
    with pytest.raises(ValidationError):
        ConversationSession(**_session_kwargs(reopen_count=-1))
    with pytest.raises(ValidationError):
        ConversationSession(**_session_kwargs(unknown_field=1))


def test_learning_run_validators():
    created = SkillLearningRunCreate(
        organization_id="org-1",
        user_id="user-1",
        agent_id="agent-1",
        trigger="threshold",
        session_ids=["s1", "s2"],
        session_watermarks={
            "s1": dt.datetime(2026, 9, 28, 1, 2, 3, 456789),
            "s2": "2026-09-28T01:02:03Z",
        },
    )
    assert created.status == "queued"
    # Naive datetimes are taken as UTC; microseconds survive; strings normalize.
    assert created.session_watermarks["s1"] == "2026-09-28T01:02:03.456789+00:00"
    assert created.session_watermarks["s2"] == "2026-09-28T01:02:03+00:00"
    base = {"organization_id": "o", "user_id": "u", "agent_id": "a"}
    with pytest.raises(ValidationError):
        SkillLearningRunCreate(**base, trigger="timer")
    with pytest.raises(ValidationError):
        SkillLearningRun(**base, trigger="flush", status="done")
    with pytest.raises(ValidationError):
        SkillLearningRunCreate(
            **base,
            trigger="flush",
            session_ids=["s1"],
            session_watermarks={"s9": "2026-01-01T00:00:00"},
        )
    with pytest.raises(ValidationError):
        SkillLearningRunCreate(**base, trigger="flush", session_ids=["s1", "s1"])
    with pytest.raises(ValidationError):
        SkillLearningRunCreate(
            **base,
            trigger="flush",
            session_ids=["s1"],
            session_watermarks={"s1": "yesterday"},
        )
    for trigger in ("threshold", "close", "sweep", "flush", "manual"):
        SkillLearningRunCreate(**base, trigger=trigger)


def test_retrieval_log_validators():
    ok = RetrievalLog(
        organization_id="o",
        user_id="u",
        memory_types=["procedural", "episodic"],
        extraction_status="ok",
    )
    assert ok.memory_types == ["procedural", "episodic"]
    with pytest.raises(ValidationError):
        RetrievalLog(organization_id="o", user_id="u", memory_types=["skills"])
    with pytest.raises(ValidationError):
        RetrievalLog(organization_id="o", user_id="u", extraction_status="maybe")
    with pytest.raises(ValidationError):
        RetrievalLog(organization_id="o", user_id="u", embedding_status="partial")
    with pytest.raises(ValidationError):
        RetrievalLog(organization_id="o", user_id="u", session_id="has space")


def test_skill_version_validators():
    sha = "a" * 64
    fields = {f"{f}_sha256": sha for f in SKILL_VERSIONED_FIELDS}
    version = SkillVersion(
        organization_id="o",
        user_id="u",
        skill_id="proc-1",
        version_no=1,
        change_type="create",
        **fields,
    )
    assert version.change_type == "create"
    with pytest.raises(ValidationError):
        SkillVersion(
            organization_id="o",
            user_id="u",
            skill_id="p",
            version_no=0,
            change_type="create",
            **fields,
        )
    with pytest.raises(ValidationError):
        SkillVersion(
            organization_id="o",
            user_id="u",
            skill_id="p",
            version_no=1,
            change_type="rename",
            **fields,
        )
    with pytest.raises(ValidationError):
        SkillVersion(
            organization_id="o",
            user_id="u",
            skill_id="p",
            version_no=1,
            change_type="edit",
            **{**fields, "name_sha256": "XYZ"},
        )
    with pytest.raises(ValidationError):
        SkillObject(
            organization_id="o", sha256="A" * 64, content_zlib=b"", raw_length=0
        )


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "text", "text": "hi"}],
        [{"type": "image_ref", "file_id": "file-1", "width": 10}],
        [{"type": "image_ref", "url": "https://example.com/a.png"}],
        None,
    ],
)
def test_content_parts_accepts_valid_shapes(parts):
    msg = ConversationMessageCreate(
        session_id="s1",
        role="user",
        user_id="u",
        organization_id="o",
        content_parts=parts,
    )
    assert msg.content_parts == parts


@pytest.mark.parametrize(
    "parts",
    [
        ["text"],
        [{"type": "image_url", "image_url": "x"}],
        [{"type": "text"}],
        [{"type": "image_ref"}],
        [{"type": "image_ref", "file_id": ""}],
    ],
)
def test_content_parts_rejects_invalid_shapes(parts):
    with pytest.raises(ValidationError):
        ConversationMessageCreate(
            session_id="s1",
            role="user",
            user_id="u",
            organization_id="o",
            content_parts=parts,
        )


def test_original_text_length_must_be_non_negative():
    with pytest.raises(ValidationError):
        ConversationMessageCreate(
            session_id="s1",
            role="user",
            user_id="u",
            organization_id="o",
            original_text_length=-1,
        )


def test_skill_experience_curation_attempts():
    exp = SkillExperience(
        session_id="s",
        experience_type="worth_learning",
        title="t",
        user_id="u",
        organization_id="o",
    )
    assert exp.curation_attempts == 0
    assert (
        SkillExperienceUpdate(id="sexp-1", curation_attempts=2).curation_attempts == 2
    )
    with pytest.raises(ValidationError):
        SkillExperienceUpdate(id="sexp-1", curation_attempts=-1)
