"""
Shared filter_tags query building for PostgreSQL (SQLAlchemy ORM + raw SQL) and Redis.

Supports OpenSearch-inspired operator syntax:
  - Plain scalar: exact match (backward compatible)
  - {"$contains": val}: value exists in stored JSON array
  - {"$exists": true/false}: key exists (or does not) in filter_tags
  - {"$in": [val1, val2]}: stored scalar is one of the provided values

Scope filtering:
  Scopes are passed as a separate `scopes` parameter (a list of scope strings
  the caller is authorized to read). This translates to:
    WHERE filter_tags->>'scope' IN (:scope_0, :scope_1, ...)
  The "read_scopes" and "scope" keys in filter_tags are explicitly ignored
  to prevent accidental scope bypass or double-filtering.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy import cast, or_, text, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Query

SUPPORTED_OPERATORS = frozenset({"$contains", "$exists", "$in"})


def _validate_operator(value: dict) -> str:
    """Return the single operator key from a value dict, or raise ValueError."""
    ops = [k for k in value if k.startswith("$")]
    if not ops:
        raise ValueError(
            f"filter_tags value dict has no operator key: {value!r}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_OPERATORS))}"
        )
    if len(ops) > 1:
        raise ValueError(
            f"filter_tags value dict has multiple operator keys: {ops!r}. "
            f"Only one operator per key is supported."
        )
    op = ops[0]
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unknown filter_tags operator '{op}'. "
            f"Supported operators: {', '.join(sorted(SUPPORTED_OPERATORS))}"
        )
    return op


def _is_operator_dict(value: Any) -> bool:
    """Check if a value is a dict containing a $ operator."""
    return isinstance(value, dict) and any(k.startswith("$") for k in value)


# ---------------------------------------------------------------------------
# SQLAlchemy ORM builder
# ---------------------------------------------------------------------------

_IGNORED_FILTER_KEYS = frozenset({"read_scopes", "scope"})


def apply_filter_tags_sqlalchemy(
    query,
    model_class,
    filter_tags: Optional[Dict[str, Any]],
    scopes: Optional[List[str]] = None,
):
    """
    Apply filter_tags conditions and scope authorization to a SQLAlchemy ORM query.

    Args:
        query: SQLAlchemy Select statement to augment.
        model_class: ORM model class with a `filter_tags` column (JSON type).
        filter_tags: Dict of filter conditions. May contain plain scalars or
                     operator dicts ($contains, $exists, $in). The keys
                     "read_scopes" and "scope" are ignored here — use the
                     `scopes` parameter instead.
        scopes: Optional list of scope strings the caller is authorized to read.
                Translates to: filter_tags->>'scope' IN (:s0, :s1, ...)

    Returns:
        The query with additional WHERE clauses appended.
    """
    if scopes is not None:
        query = _apply_scopes_sqla(query, model_class, scopes)

    if not filter_tags:
        return query

    for key, value in filter_tags.items():
        if key in _IGNORED_FILTER_KEYS:
            continue
        if _is_operator_dict(value):
            query = query.where(_resolve_operator_sqla(key, value, model_class))
        else:
            query = query.where(
                model_class.filter_tags[key].as_string() == str(value)
            )

    return query


def _apply_scopes_sqla(query, model_class, scopes: List[str]):
    """Apply scope authorization filter for SQLAlchemy."""
    if scopes:
        scope_conditions = [
            model_class.filter_tags["scope"].as_string() == scope
            for scope in scopes
        ]
        return query.where(or_(*scope_conditions))
    return query.where(text("1 = 0"))


def _resolve_operator_sqla(key: str, value: dict, model_class):
    """Resolve a single $ operator into a SQLAlchemy WHERE clause element."""
    op = _validate_operator(value)

    if op == "$contains":
        # Pass the dict directly — type_coerce lets psycopg2 serialize it once.
        # Using json.dumps + cast would double-encode the string.
        return cast(model_class.filter_tags, JSONB).contains(
            type_coerce({key: [value["$contains"]]}, JSONB)
        )
    elif op == "$exists":
        condition = cast(model_class.filter_tags, JSONB).has_key(key)  # noqa: W601
        if not value["$exists"]:
            condition = ~condition
        return condition
    elif op == "$in":
        vals = value["$in"]
        if not isinstance(vals, list) or not vals:
            return text("1 = 0")
        return model_class.filter_tags[key].as_string().in_(
            [str(v) for v in vals]
        )


# ---------------------------------------------------------------------------
# Raw SQL builder (for BM25 full-text search paths)
# ---------------------------------------------------------------------------

def build_filter_tags_raw_sql(
    filter_tags: Optional[Dict[str, Any]],
    scopes: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Build raw SQL WHERE clause fragments from filter_tags and scopes.

    Args:
        filter_tags: Dict of filter conditions. Keys "read_scopes" and "scope"
                     are ignored — use the `scopes` parameter instead.
        scopes: Optional list of scope strings the caller is authorized to read.

    Returns:
        (where_clauses, params) where where_clauses is a list of SQL strings
        and params is a dict of bind parameters.
    """
    where_clauses: List[str] = []
    params: Dict[str, Any] = {}

    if scopes is not None:
        clauses, p = _build_scopes_raw_sql(scopes)
        where_clauses.extend(clauses)
        params.update(p)

    if not filter_tags:
        return where_clauses, params

    for key, value in filter_tags.items():
        if key in _IGNORED_FILTER_KEYS:
            continue
        if _is_operator_dict(value):
            clause, p = _resolve_operator_raw_sql(key, value)
            where_clauses.append(clause)
            params.update(p)
        else:
            param_name = f"filter_tag_{key}"
            where_clauses.append(f"filter_tags->>'{key}' = :{param_name}")
            params[param_name] = str(value)

    return where_clauses, params


def _build_scopes_raw_sql(scopes: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Build raw SQL scope authorization clause."""
    clauses: List[str] = []
    params: Dict[str, Any] = {}

    if scopes:
        placeholders = [f":scope_{i}" for i in range(len(scopes))]
        clauses.append(
            f"filter_tags->>'scope' IN ({', '.join(placeholders)})"
        )
        for i, scope in enumerate(scopes):
            params[f"scope_{i}"] = scope
    else:
        clauses.append("1 = 0")

    return clauses, params


def _resolve_operator_raw_sql(
    key: str, value: dict
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a single $ operator into a raw SQL clause + params."""
    op = _validate_operator(value)
    params: Dict[str, Any] = {}

    if op == "$contains":
        param_name = f"filter_contains_{key}"
        clause = f"filter_tags::jsonb @> :{param_name}::jsonb"
        params[param_name] = json.dumps({key: [value["$contains"]]})
        return clause, params
    elif op == "$exists":
        if value["$exists"]:
            clause = f"filter_tags::jsonb ? '{key}'"
        else:
            clause = f"NOT (filter_tags::jsonb ? '{key}')"
        return clause, params
    elif op == "$in":
        vals = value["$in"]
        if not isinstance(vals, list) or not vals:
            return "1 = 0", params
        placeholders = []
        for i, v in enumerate(vals):
            param_name = f"filter_in_{key}_{i}"
            placeholders.append(f":{param_name}")
            params[param_name] = str(v)
        clause = f"filter_tags->>'{key}' IN ({', '.join(placeholders)})"
        return clause, params

    raise ValueError(f"Unhandled operator: {op}")


# ---------------------------------------------------------------------------
# Redis support
# ---------------------------------------------------------------------------

def can_redis_handle(filter_tags: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether all filter_tags values can be handled by Redis TAG queries.

    Returns False if any value is:
      - A dict with $ operators (Redis doesn't support these)
      - A list

    Returns True when all values are plain scalars that map to
    @filter_tags_{key}:{value} Redis TAG queries.
    """
    if not filter_tags:
        return True

    for key, value in filter_tags.items():
        if key in _IGNORED_FILTER_KEYS:
            continue
        if isinstance(value, (dict, list)):
            return False

    return True


def build_filter_tags_redis(
    filter_tags: Optional[Dict[str, Any]],
    scopes: Optional[List[str]] = None,
) -> str:
    """
    Build a Redis Search query string from filter_tags and scopes.

    Handles:
      - scopes: translated to OR query on filter_tags_scope
      - Plain scalars: @filter_tags_{key}:{escaped_value}

    Callers should check can_redis_handle() first; this function only
    handles scalar values and scopes.
    """
    def escape_tag_value(val: str) -> str:
        special_chars = r'[\-:.()\[\]{}"\',<>;!@#$%^&*+=~]'
        return re.sub(special_chars, lambda m: f"\\{m.group(0)}", str(val))

    query_parts: List[str] = []

    if scopes:
        escaped_scopes = [escape_tag_value(s) for s in scopes]
        joined = "|".join(escaped_scopes)
        query_parts.append(f"@filter_tags_scope:{{{joined}}}")

    if filter_tags:
        for key, value in filter_tags.items():
            if key in _IGNORED_FILTER_KEYS:
                continue
            escaped_value = escape_tag_value(value)
            field_name = f"filter_tags_{key}"
            query_parts.append(f"@{field_name}:{{{escaped_value}}}")

    return " ".join(query_parts)
