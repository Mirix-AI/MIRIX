"""
Unit tests for mirix.database.filter_tags_query shared utility.

No database required -- tests pure logic, SQL compilation, and Redis query building.
"""

import json

import pytest
from sqlalchemy import JSON, Column, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from mirix.database.filter_tags_query import (
    apply_filter_tags_sqlalchemy,
    build_filter_tags_raw_sql,
    build_filter_tags_redis,
    can_redis_handle,
)


# ---------------------------------------------------------------------------
# Minimal ORM model for testing SQLAlchemy compilation (no real DB needed)
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class _FakeMemory(_Base):
    __tablename__ = "fake_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filter_tags = Column(JSON, nullable=True)
    organization_id: Mapped[str] = mapped_column(String, nullable=True)


def _compile_query(query) -> str:
    """Compile a SQLAlchemy query to a SQL string for inspection."""
    from sqlalchemy.dialects import postgresql

    return str(query.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}))


# ===================================================================
# can_redis_handle
# ===================================================================

class TestCanRedisHandle:
    def test_none_filter_tags(self):
        assert can_redis_handle(None) is True

    def test_empty_filter_tags(self):
        assert can_redis_handle({}) is True

    def test_plain_scalars(self):
        assert can_redis_handle({"env": "prod", "expert_id": "e-123"}) is True

    def test_ignored_keys_skipped(self):
        assert can_redis_handle({"read_scopes": "anything", "scope": "CARE"}) is True

    def test_list_value_rejected(self):
        assert can_redis_handle({"account_ids": ["a", "b"]}) is False

    def test_contains_operator(self):
        assert can_redis_handle({"account_ids": {"$contains": "ABC"}}) is False

    def test_exists_operator(self):
        assert can_redis_handle({"account_ids": {"$exists": True}}) is False

    def test_in_operator(self):
        assert can_redis_handle({"status": {"$in": ["active", "pending"]}}) is False

    def test_mixed_scalar_and_operator(self):
        assert can_redis_handle({"env": "prod", "account_ids": {"$contains": "X"}}) is False


# ===================================================================
# build_filter_tags_redis
# ===================================================================

class TestBuildFilterTagsRedis:
    def test_none_no_scopes(self):
        assert build_filter_tags_redis(None) == ""

    def test_empty_no_scopes(self):
        assert build_filter_tags_redis({}) == ""

    def test_single_scalar(self):
        result = build_filter_tags_redis({"env": "prod"})
        assert "@filter_tags_env:{prod}" == result

    def test_multiple_scalars(self):
        result = build_filter_tags_redis({"env": "prod", "expert_id": "e123"})
        assert "@filter_tags_env:" in result
        assert "@filter_tags_expert_id:" in result

    def test_scopes_single(self):
        result = build_filter_tags_redis(None, scopes=["CARE"])
        assert "@filter_tags_scope:{CARE}" == result

    def test_scopes_multiple(self):
        result = build_filter_tags_redis(None, scopes=["A", "B"])
        assert "@filter_tags_scope:{A|B}" == result

    def test_scopes_empty_list(self):
        result = build_filter_tags_redis(None, scopes=[])
        assert result == ""

    def test_scopes_with_filter_tags(self):
        result = build_filter_tags_redis({"expert_id": "e1"}, scopes=["A"])
        assert "@filter_tags_scope:{A}" in result
        assert "@filter_tags_expert_id:" in result

    def test_ignored_keys_excluded(self):
        result = build_filter_tags_redis({"read_scopes": ["X"], "scope": "Y", "env": "prod"})
        assert "read_scopes" not in result
        assert "@filter_tags_scope:" not in result
        assert "@filter_tags_env:{prod}" in result

    def test_special_chars_escaped(self):
        result = build_filter_tags_redis({"env": "my-scope:v1"})
        assert "my\\-scope\\:v1" in result


# ===================================================================
# build_filter_tags_raw_sql
# ===================================================================

class TestBuildFilterTagsRawSql:
    def test_none(self):
        clauses, params = build_filter_tags_raw_sql(None)
        assert clauses == []
        assert params == {}

    def test_empty(self):
        clauses, params = build_filter_tags_raw_sql({})
        assert clauses == []
        assert params == {}

    def test_plain_scalar(self):
        clauses, params = build_filter_tags_raw_sql({"env": "prod"})
        assert len(clauses) == 1
        assert "filter_tags->>'env' = :filter_tag_env" == clauses[0]
        assert params["filter_tag_env"] == "prod"

    def test_multiple_scalars(self):
        clauses, params = build_filter_tags_raw_sql({"env": "prod", "expert_id": "e1"})
        assert len(clauses) == 2
        assert "filter_tag_env" in params
        assert "filter_tag_expert_id" in params

    def test_scopes(self):
        clauses, params = build_filter_tags_raw_sql(None, scopes=["A", "B"])
        assert len(clauses) == 1
        assert "filter_tags->>'scope' IN" in clauses[0]
        assert params["scope_0"] == "A"
        assert params["scope_1"] == "B"

    def test_scopes_empty(self):
        clauses, params = build_filter_tags_raw_sql(None, scopes=[])
        assert clauses == ["1 = 0"]

    def test_scopes_with_filter_tags(self):
        clauses, params = build_filter_tags_raw_sql(
            {"env": "prod"}, scopes=["A"]
        )
        assert len(clauses) == 2
        assert any("filter_tags->>'scope' IN" in c for c in clauses)
        assert any("filter_tags->>'env'" in c for c in clauses)

    def test_ignored_keys_excluded(self):
        clauses, params = build_filter_tags_raw_sql(
            {"read_scopes": ["X"], "scope": "Y", "env": "prod"}
        )
        assert len(clauses) == 1
        assert "filter_tags->>'env'" in clauses[0]

    def test_contains_operator(self):
        clauses, params = build_filter_tags_raw_sql(
            {"account_ids": {"$contains": "ABC"}}
        )
        assert len(clauses) == 1
        assert "filter_tags::jsonb @>" in clauses[0]
        param_val = json.loads(params["filter_contains_account_ids"])
        assert param_val == {"account_ids": ["ABC"]}

    def test_exists_true(self):
        clauses, params = build_filter_tags_raw_sql(
            {"account_ids": {"$exists": True}}
        )
        assert len(clauses) == 1
        assert "filter_tags::jsonb ? 'account_ids'" == clauses[0]

    def test_exists_false(self):
        clauses, params = build_filter_tags_raw_sql(
            {"account_ids": {"$exists": False}}
        )
        assert len(clauses) == 1
        assert "NOT (filter_tags::jsonb ? 'account_ids')" == clauses[0]

    def test_in_operator(self):
        clauses, params = build_filter_tags_raw_sql(
            {"status": {"$in": ["active", "pending"]}}
        )
        assert len(clauses) == 1
        assert "filter_tags->>'status' IN" in clauses[0]
        assert params["filter_in_status_0"] == "active"
        assert params["filter_in_status_1"] == "pending"

    def test_in_operator_empty_list(self):
        clauses, params = build_filter_tags_raw_sql(
            {"status": {"$in": []}}
        )
        assert clauses == ["1 = 0"]

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown filter_tags operator"):
            build_filter_tags_raw_sql({"x": {"$foo": "bar"}})

    def test_multiple_operators_in_one_dict_raises(self):
        with pytest.raises(ValueError, match="multiple operator keys"):
            build_filter_tags_raw_sql({"x": {"$contains": "a", "$in": ["b"]}})

    def test_mixed_scalar_and_operator(self):
        clauses, params = build_filter_tags_raw_sql(
            {"env": "prod", "account_ids": {"$contains": "ABC"}}
        )
        assert len(clauses) == 2

    def test_scopes_with_operator(self):
        clauses, params = build_filter_tags_raw_sql(
            {"account_ids": {"$contains": "X"}}, scopes=["A"]
        )
        assert len(clauses) == 2


# ===================================================================
# apply_filter_tags_sqlalchemy
# ===================================================================

class TestApplyFilterTagsSqlalchemy:
    def _base_query(self):
        return select(_FakeMemory)

    def test_none_filter_tags(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(q, _FakeMemory, None)
        assert _compile_query(result) == _compile_query(q)

    def test_empty_filter_tags(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(q, _FakeMemory, {})
        assert _compile_query(result) == _compile_query(q)

    def test_plain_scalar(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(q, _FakeMemory, {"env": "prod"})
        sql = _compile_query(result)
        assert "filter_tags" in sql
        assert "WHERE" in sql.upper()

    def test_scopes(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(q, _FakeMemory, None, scopes=["A", "B"])
        sql = _compile_query(result)
        assert "filter_tags" in sql
        assert " OR " in sql.upper()

    def test_scopes_empty(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(q, _FakeMemory, None, scopes=[])
        sql = _compile_query(result)
        assert "1 = 0" in sql

    def test_ignored_keys_excluded(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"read_scopes": ["X"], "scope": "Y"}
        )
        sql = _compile_query(result)
        assert _compile_query(result) == _compile_query(q)

    def test_contains_operator(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"account_ids": {"$contains": "ABC"}}
        )
        sql = _compile_query(result)
        assert "CAST" in sql or "cast" in sql.lower() or "@>" in sql

    def test_exists_true(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"account_ids": {"$exists": True}}
        )
        sql = _compile_query(result)
        assert "JSONB" in sql.upper() or "jsonb" in sql
        assert "?" in sql

    def test_exists_false(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"account_ids": {"$exists": False}}
        )
        sql = _compile_query(result)
        assert "?" in sql
        assert "NOT" in sql.upper()

    def test_in_operator(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"status": {"$in": ["active", "pending"]}}
        )
        sql = _compile_query(result)
        assert "IN" in sql.upper()

    def test_in_operator_empty(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q, _FakeMemory, {"status": {"$in": []}}
        )
        sql = _compile_query(result)
        assert "1 = 0" in sql

    def test_unknown_operator_raises(self):
        q = self._base_query()
        with pytest.raises(ValueError, match="Unknown filter_tags operator"):
            apply_filter_tags_sqlalchemy(q, _FakeMemory, {"x": {"$foo": "bar"}})

    def test_mixed_all_types(self):
        q = self._base_query()
        result = apply_filter_tags_sqlalchemy(
            q,
            _FakeMemory,
            {
                "account_ids": {"$contains": "ABC"},
                "status": {"$in": ["active"]},
                "extra": {"$exists": True},
            },
            scopes=["A"],
        )
        sql = _compile_query(result)
        assert "filter_tags" in sql
        assert "@>" in sql
        assert "IN" in sql.upper()
        assert "?" in sql
