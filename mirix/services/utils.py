from functools import lru_cache, wraps
from typing import List, Optional

import numpy as np
import pytz
from sqlalchemy import func

from mirix.constants import (
    MAX_EMBEDDING_DIM,
)
from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.orm.sqlite_functions import adapt_array
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.settings import settings

logger = get_logger(__name__)


async def build_query(
    base_query,
    search_field,
    query_text: Optional[str] = None,
    embedded_text: Optional[List[float]] = None,
    embed_query: bool = True,
    embedding_config: Optional[EmbeddingConfig] = None,
    ascending: bool = True,
    target_class: object = None,
    similarity_threshold: Optional[float] = None,
):
    """
    Build a query based on the query text

    Args:
        similarity_threshold: Maximum cosine distance (0.0=identical, 2.0=opposite).
                             Results with distance >= threshold are excluded.
    """

    if embed_query:
        if embedded_text is None:
            assert embedding_config is not None, "embedding_config must be specified for vector search"
            assert query_text is not None, "query_text must be specified for vector search"
            embedded_text = await (await embedding_model(embedding_config)).get_text_embedding(query_text)
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(
                embedded_text,
                (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                mode="constant",
            ).tolist()
        else:
            # Normalize to MAX_EMBEDDING_DIM so query matches DB column (pgvector requirement)
            embedded_text = np.asarray(embedded_text, dtype=float)
            if embedded_text.shape[0] != MAX_EMBEDDING_DIM:
                embedded_text = np.pad(
                    embedded_text,
                    (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                    mode="constant",
                ).tolist()
            else:
                embedded_text = embedded_text.tolist()

    main_query = base_query.order_by(None)

    if embedded_text:
        # Check which database type we're using
        if settings.mirix_pg_uri_no_default:
            # PostgreSQL with pgvector - use direct cosine_distance method
            distance_field = search_field.cosine_distance(embedded_text)

            # Apply similarity threshold filter if provided
            if similarity_threshold is not None:
                main_query = main_query.where(distance_field < similarity_threshold)

            if ascending:
                main_query = main_query.order_by(
                    distance_field.asc(),
                    target_class.created_at.asc(),
                    target_class.id.asc(),
                )
            else:
                main_query = main_query.order_by(
                    distance_field.asc(),
                    target_class.created_at.desc(),
                    target_class.id.asc(),
                )
        else:
            # SQLite with custom vector type
            query_embedding_binary = adapt_array(embedded_text)
            distance_field = func.cosine_distance(search_field, query_embedding_binary)

            # Apply similarity threshold filter if provided
            if similarity_threshold is not None:
                main_query = main_query.where(distance_field < similarity_threshold)

            if ascending:
                main_query = main_query.order_by(
                    distance_field.asc(),
                    target_class.created_at.asc(),
                    target_class.id.asc(),
                )
            else:
                main_query = main_query.order_by(
                    distance_field.asc(),
                    target_class.created_at.desc(),
                    target_class.id.asc(),
                )

    else:
        # TODO: add other kinds of search
        raise NotImplementedError

    return main_query


def update_timezone(func):
    """Decorator that applies timezone conversion to datetime fields on returned results.
    Only supports async functions (MIRIX is async-native).
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        timezone_str = (
            kwargs.get("timezone_str")
            or (getattr(kwargs.get("actor"), "timezone", "UTC") if kwargs.get("actor") else None)
        )
        results = await func(*args, **kwargs)
        return localize_timestamps(results, timezone_str)

    return wrapper


@lru_cache(maxsize=256)
def display_timezone(timezone_str: str):
    """The pytz zone a user's timestamps are shown in: the region of MIRIX's
    ``"Region (UTC+hh:mm)"`` form (``"Asia/Shanghai (UTC+08:00)"``) or a bare
    zone name. A string pytz cannot name ("Beijing (UTC+08:00)", "UTC+8")
    is UTC, logged once per string: the same fallback as
    :func:`mirix.utils.parse_datetime_arg` (so a time range the user gave is
    read and answered on the same clock), never an ``UnknownTimeZoneError``
    that fails the read."""
    region = str(timezone_str).split(" (")[0].strip()
    try:
        return pytz.timezone(region)
    except pytz.UnknownTimeZoneError:
        logger.warning(
            "Unknown user timezone %r: its timestamps are shown in UTC", timezone_str
        )
        return pytz.utc


def _in_zone(value, zone):
    """``value`` (naive UTC, or aware) in ``zone``; in UTC (or as it is) when
    that would pass the end of the datetime range (a bound stored as
    ``datetime.min`` / ``max``, :func:`mirix.utils.parse_datetime_arg`), never
    an OverflowError that fails the read."""
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    for target in (zone, pytz.utc):
        try:
            return value.astimezone(target)
        except OverflowError:
            continue
    return value


def localize_timestamps(results, timezone_str: Optional[str]):
    """Convert the datetime fields of ``results`` (occurred_at, created_at,
    updated_at, last_modify.timestamp) to ``timezone_str`` in place, as
    :func:`update_timezone` does for a method's return value; returns them.
    A timezone pytz cannot name is UTC (:func:`display_timezone`), and so is a
    time the zone's offset would carry past the datetime range."""
    if results is None or not timezone_str:
        return results
    target_tz = display_timezone(timezone_str)
    for result in results:
        if hasattr(result, "occurred_at"):
            result.occurred_at = _in_zone(result.occurred_at, target_tz)
        if hasattr(result, "created_at"):
            result.created_at = _in_zone(result.created_at, target_tz)
        if hasattr(result, "updated_at") and result.updated_at is not None:
            result.updated_at = _in_zone(result.updated_at, target_tz)
        if (
            hasattr(result, "last_modify")
            and result.last_modify
            and "timestamp" in result.last_modify
        ):
            timestamp = result.last_modify["timestamp"]
            if isinstance(timestamp, str):
                from datetime import datetime

                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            result.last_modify["timestamp"] = _in_zone(timestamp, target_tz)
    return results
