import asyncio
import datetime as dt
import json
import re
import string
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import lazyload

from mirix.constants import (
    BUILD_EMBEDDINGS_FOR_MEMORY,
    DEFAULT_ORG_ID,
    MAX_EMBEDDING_DIM,
    SKILL_HYBRID_RECALL_MULTIPLIER,
    SKILL_HYBRID_RRF_K,
)
from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.orm.errors import NoResultFound, UniqueConstraintViolationError
from mirix.orm.procedural_memory import UNIQUE_LIVE_NAME_INDEX, ProceduralMemoryItem
from mirix.orm.skill_experience import SkillExperience as SkillExperienceModel
from mirix.orm.skill_object import SkillObject as SkillObjectModel
from mirix.orm.skill_version import SkillVersion as SkillVersionModel
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.procedural_memory import (
    ProceduralMemoryItem as PydanticProceduralMemoryItem,
)
from mirix.schemas.procedural_memory import (
    ProceduralMemoryItemUpdate,
    skill_nul_problem,
)
from mirix.schemas.skill_version import SKILL_VERSIONED_FIELDS
from mirix.schemas.skill_version import SkillVersion as PydanticSkillVersion
from mirix.schemas.user import User as PydanticUser
from mirix.services import skill_history as history
from mirix.services.retry_policy import retry_call
from mirix.services.utils import build_query, localize_timestamps, update_timezone
from mirix.settings import settings
from mirix.utils import enforce_types

if TYPE_CHECKING:
    from mirix.services.skill_retrieval import SkillRetrievalResult

logger = get_logger(__name__)


def _pad_to_max(vec):
    """Pad a raw embedding to MAX_EMBEDDING_DIM with trailing zeros.

    Every write path pads here (and the query path pads the query vector), so
    stored and query vectors are always MAX_EMBEDDING_DIM wide: a raw (e.g.
    3072-dim) vector written onto the Vector(4096) column would make
    SQLAlchemy's compare_values raise "operands could not be broadcast together
    with shapes (4096,) (N,)" on the next UPDATE. Zero-padding is
    cosine-preserving (the appended dims contribute nothing to dot product or
    norms).
    """
    if vec is None:
        return None
    arr = np.array(vec)
    if arr.shape[0] == MAX_EMBEDDING_DIM:
        return list(vec)
    return np.pad(arr, (0, MAX_EMBEDDING_DIM - arr.shape[0]), mode="constant").tolist()


# Whitelisted searchable text fields and their associated ORM columns.
# Kept as a constant so every search path uses the same mapping — no eval(),
# no silent column fallbacks, and unknown fields can be rejected early.
_SEARCHABLE_TEXT_FIELDS: Dict[str, str] = {
    "description": "description",
    "instructions": "instructions",
    "entry_type": "entry_type",
    "name": "name",
}

# A skill has ONE vector, `retrieval_embedding`: the embedding of
# its retrieval text (name + triggers + description, see
# skill_history.skill_retrieval_text). Embedding searches on "description" (the
# default field) use it.
_RETRIEVAL_EMBEDDING_COLUMN = "retrieval_embedding"
_EMBEDDING_FIELDS: Dict[str, str] = {
    "description": _RETRIEVAL_EMBEDDING_COLUMN,
}

# Fields that no longer carry a vector: an embedding (or hybrid) search on
# them is served by full-text keyword search on that same field instead.
_FULLTEXT_FALLBACK_FIELDS = frozenset({"instructions"})

# The search fields whose ranked search is the skill's retrieval text (name,
# triggers, description): /memory/search sends "null" for "no field",
# the manager's default is "".
RETRIEVAL_TEXT_SEARCH_FIELDS = frozenset({"", "null", "description"})


def effective_procedural_search_method(search_method: str, search_field: str) -> str:
    """The search method a procedural search actually runs.

    `instructions` has no vector any more, so an "embedding" or "hybrid"
    search on it runs as "bm25" full-text search on `instructions`. Every other
    combination is unchanged. The REST layer reports this value back to the
    caller.
    """
    if (
        search_method in ("embedding", "hybrid")
        and search_field in _FULLTEXT_FALLBACK_FIELDS
    ):
        return "bm25"
    return search_method


def uses_skill_ranking(search_method: str, search_field: str) -> bool:
    """Whether a procedural search is the skill ranking (:meth:`ProceduralMemoryManager.retrieve_skills`):
    every "hybrid" search, and a "bm25" or "embedding" search over the
    retrieval text. The other combinations (a field-specific keyword search,
    string or fuzzy matching) keep :meth:`ProceduralMemoryManager.list_procedures`'
    per-field behavior."""
    if search_method == "hybrid":
        return search_field not in _FULLTEXT_FALLBACK_FIELDS
    return search_method in ("bm25", "embedding") and (
        search_field in RETRIEVAL_TEXT_SEARCH_FIELDS
    )


# --------------------------------------------------------------------------- #
# Skill writes: every change goes through apply_skill_change                   #
# --------------------------------------------------------------------------- #

# Who made a change (skill_version.actor_kind).
ACTOR_CURATOR = "curator"  # a skill tool inside a skill learning (curator) run
ACTOR_AGENT = "agent"  # a skill tool outside a learning run (in-flow agent)
ACTOR_API = "api"  # a REST / client call (restore, delete)
ACTOR_SYSTEM = "system"  # bulk erasure, lazily written baselines
ACTOR_BACKFILL = "backfill"  # scripts/backfill_skill_baseline_versions.py

# Row columns an edit may set besides the seven versioned fields. Never id,
# organization_id, user_id, client_id or the audit timestamps: a skill cannot
# be moved to another tenant or owner by an edit.
_EDITABLE_META_FIELDS = frozenset(
    {"version", "agent_id", "embedding_config", _RETRIEVAL_EMBEDDING_COLUMN}
)
_REQUIRED_CREATE_FIELDS = ("name", "entry_type", "description", "instructions")

# Prepare/commit attempts when a concurrent change made the precomputed
# retrieval vector (or the claimed name) stale; the last attempt stores a NULL
# vector instead of failing (re-embedded later).
_MAX_PLAN_ATTEMPTS = 3

# The fields of a skill's retrieval text (skill_history.skill_retrieval_text).
_RETRIEVAL_TEXT_FIELDS = ("name", "triggers", "description")

# ProceduralMemoryManager.reembed_skill outcomes.
REEMBED_FILLED = "filled"
REEMBED_PRESENT = "present"
REEMBED_GONE = "gone"
REEMBED_CHANGED = "changed"
REEMBED_NO_CONFIG = "no_config"
REEMBED_FAILED = "failed"
REEMBED_DISABLED = "disabled"

# Skills being re-embedded in this process right now, with the future of the
# outcome: a second pass reaching one of them (the startup pass and a learning
# run's end-of-run pass meet right after the retrieval-field migration, when
# every vector is NULL) awaits that outcome instead of paying for the same
# embedding again. Process-wide on purpose: each pass
# builds its own manager.
_REEMBED_IN_FLIGHT: Dict[str, "asyncio.Future[Optional[str]]"] = {}


class SkillFieldInvalid(ValueError):
    """A skill write whose fields cannot be stored (a NUL character, U+0000),
    refused before anything is read, embedded or written."""


class SkillNameConflictError(UniqueConstraintViolationError):
    """Another live skill of the same (organization, user) has this name.

    ``holder_id`` is that skill's id when it is known. The message never names
    it: the holder may be in a scope the caller cannot read, and its id is all
    a by-id call needs."""

    def __init__(self, message: str, *, holder_id: Optional[str] = None):
        super().__init__(message)
        self.holder_id = holder_id


class SkillChangedMeanwhile(Exception):
    """An edit's ``expected_fields`` no longer match the skill: another writer
    changed those fields after the caller read the skill and computed its edit
    from them. Nothing was written."""

    def __init__(self, skill_id: str, fields: Sequence[str]):
        self.skill_id = skill_id
        self.fields = list(fields)
        super().__init__(
            f"Skill {skill_id} changed meanwhile ({', '.join(self.fields)}); "
            "nothing was written."
        )


class _PurgeReachedUnlockedOrgs(Exception):
    """A purge's locked re-read reached organizations it had not locked."""

    def __init__(self, organization_ids: set):
        super().__init__(sorted(organization_ids))
        self.organization_ids = set(organization_ids)


# Rounds of a purge whose organization set grew under the lock.
_MAX_PURGE_LOCK_ROUNDS = 5


class _StalePlan(Exception):
    """The skill changed between the out-of-transaction preparation (retrieval
    vector, claimed name) and the locked commit; prepare again."""


@dataclass
class SkillChangeResult:
    """What :meth:`ProceduralMemoryManager.apply_skill_change` did."""

    skill: PydanticProceduralMemoryItem
    #: The version this change wrote; None when the content did not change.
    version: Optional[PydanticSkillVersion] = None
    #: The baseline written first because the skill had no history yet.
    baseline: Optional[PydanticSkillVersion] = None

    @property
    def changed(self) -> bool:
        return self.version is not None


@dataclass
class _ChangePlan:
    change_type: str
    skill_id: str
    organization_id: str  # the skill's organization (objects/versions/locks)
    lock_user_id: Optional[str]  # the skill owner (name-lock key)
    access_org_id: Optional[str]  # row filter: organization (None = none)
    access_user_id: Optional[str]  # row filter: owner (None = none)
    access_scopes: Optional[List[str]] = None  # row filter: filter_tags.scope
    fields: Dict[str, Any] = field(default_factory=dict)
    create_content: Optional[Dict[str, Any]] = None
    restore_version_no: Optional[int] = None
    claimed_name: Optional[str] = None
    explicit_embedding: bool = False
    embedding: Optional[List[float]] = None
    embedded_text: Optional[str] = None
    embedding_config: Any = None
    allow_stale_embedding: bool = False
    # edit: the values the caller computed the edit from (optimistic check).
    expected_fields: Optional[Dict[str, Any]] = None
    # edit: bump the row's version string (patch + 1) under the row lock.
    bump_version: bool = False
    # The id of the version this plan writes, fixed before its first commit:
    # a repeat of the transaction finds it when an ambiguous COMMIT did go
    # through, and reports that success instead of acting twice.
    version_id: str = field(default_factory=lambda: f"skver-{uuid.uuid4()}")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _resolve_org(*candidates: Optional[str]) -> str:
    for candidate in candidates:
        if candidate:
            return candidate
    return DEFAULT_ORG_ID


def _bump_patch(version: Optional[str]) -> str:
    try:
        parts = (version or "0.1.0").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return "0.1.1"


def _version_schema(row: Optional[SkillVersionModel]) -> Optional[PydanticSkillVersion]:
    return None if row is None else PydanticSkillVersion.model_validate(row)


def _tags_in_scopes(tags: Any, scopes: Optional[Sequence[str]]) -> bool:
    """Whether ``filter_tags`` has a ``scope`` that is one of ``scopes``
    (None = no restriction; ``[]`` matches nothing), like the SQL filter."""
    if scopes is None:
        return True
    scope = tags.get("scope") if isinstance(tags, dict) else None
    return scope is not None and str(scope) in {str(s) for s in scopes}


def _in_scopes(item: Any, scopes: Optional[Sequence[str]]) -> bool:
    """Whether a skill row's ``filter_tags.scope`` is one of ``scopes``."""
    return _tags_in_scopes(getattr(item, "filter_tags", None), scopes)


def _row_values(row: Any, keys: Sequence[str]) -> Dict[str, Any]:
    """A skill row's current values of ``keys`` (versioned or row fields)."""
    content = history.content_of(row)
    return {
        key: content[key] if key in SKILL_VERSIONED_FIELDS else getattr(row, key, None)
        for key in keys
    }


def _fields_differ(current: Mapping[str, Any], expected: Mapping[str, Any]) -> List[str]:
    """The fields of ``expected`` whose value ``current`` no longer holds —
    versioned fields compared by their canonical bytes (what the history
    hashes), the others by equality."""
    differ = []
    for key, value in expected.items():
        now = current.get(key)
        if key in SKILL_VERSIONED_FIELDS:
            same = history.canonical_bytes(key, now) == history.canonical_bytes(
                key, value
            )
        else:
            same = now == value
        if not same:
            differ.append(key)
    return differ


def _check_target_scope(plan: "_ChangePlan", target: Mapping[str, Any]) -> None:
    """A change confined to ``plan.access_scopes`` may not give the
    skill content of another scope: a restored version of another scope is
    not found (like the history endpoints report it); an edit that moves the
    skill out is invalid."""
    if _tags_in_scopes(target.get("filter_tags"), plan.access_scopes):
        return
    if plan.change_type == history.CHANGE_RESTORE:
        raise history.SkillVersionNotFound(
            f"Skill {plan.skill_id} has no version {plan.restore_version_no}."
        )
    raise ValueError("an edit cannot move a skill out of the caller's scopes")


def _is_live_name_violation(exc: BaseException) -> bool:
    message = str(getattr(exc, "orig", exc))
    # The partial unique index, or the pre-K12 full constraint on databases
    # that have not run scripts/migrate_skill_retrieval_fields.sql yet.
    return UNIQUE_LIVE_NAME_INDEX in message or (
        "uq_procedural_memory_org_user_name" in message
    ) or (
        "UNIQUE constraint failed: procedural_memory.organization_id" in message
    )


def _resolve_text_column(search_field: str):
    """Return the ORM column for a text search field, or None if not whitelisted."""
    column_name = _SEARCHABLE_TEXT_FIELDS.get(search_field)
    if column_name is None:
        return None
    return getattr(ProceduralMemoryItem, column_name)


def _resolve_embedding_column(search_field: str):
    """Return the ORM embedding column for a text search field."""
    column_name = _EMBEDDING_FIELDS.get(search_field)
    if column_name is None:
        return None
    return getattr(ProceduralMemoryItem, column_name)


def _rrf_fuse(
    ranked_lists: List[List[PydanticProceduralMemoryItem]],
    *,
    k: int,
    limit: Optional[int],
) -> List[PydanticProceduralMemoryItem]:
    """Fuse several ranked result lists with Reciprocal Rank Fusion.

    Bit-for-bit aligned with EverOS/everalgo (``everalgo.rank.fusion.rrf``):
    the fusion is RANK-BASED (raw BM25 ts_rank_cd and cosine scores are
    intentionally discarded — only a result's POSITION in each lane feeds the
    sum), 1-BASED (the top item in a lane has rank 1), and UNWEIGHTED (every
    lane contributes ``1 / (k + rank)`` with no per-lane weight). ``k`` defaults
    to the canonical 60.

    Items are keyed by their stable ``id`` so the same skill surfaced by both
    the lexical and the dense lane accumulates both contributions (the
    hallmark of hybrid retrieval: agreement across lanes ranks higher). Items
    with no ``id`` cannot be deduplicated across lanes and are skipped. The
    fused list is sliced to ``limit`` (no slice when ``limit`` is falsy).
    """
    scores: Dict[str, float] = {}
    items_by_id: Dict[str, PydanticProceduralMemoryItem] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            item_id = getattr(item, "id", None)
            if item_id is None:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items_by_id.setdefault(item_id, item)
    ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    fused = [items_by_id[i] for i in ordered_ids]
    return fused[:limit] if limit else fused


class ProceduralMemoryManager:
    """Manager class to handle business logic related to Procedural Memory Items."""

    def __init__(self, session_maker=None, sleep=None):
        """``session_maker`` defaults to the server's ``db_context``; ``sleep``
        (default ``asyncio.sleep``) is the wait between embedding retries."""
        if session_maker is None:
            from mirix.server.server import db_context

            session_maker = db_context
        self.session_maker = session_maker
        self._sleep = sleep or asyncio.sleep

    def _clean_text_for_search(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text with punctuation removed and normalized whitespace
        """
        if not text:
            return ""

        # Remove punctuation using string.punctuation
        # Create translation table that maps each punctuation character to space
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)

        # Convert to lowercase and normalize whitespace
        text = re.sub(r"\s+", " ", text.lower().strip())

        return text

    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 search by tokenizing and cleaning.

        Args:
            text: Input text to preprocess

        Returns:
            List of cleaned tokens
        """
        if not text:
            return []

        # Clean text first
        cleaned_text = self._clean_text_for_search(text)

        # Split into tokens and filter out empty strings and very short tokens
        tokens = [
            token for token in cleaned_text.split() if token.strip() and len(token) > 1
        ]
        return tokens

    def _parse_embedding_field(self, embedding_value):
        """
        Helper method to parse embedding field from different PostgreSQL return formats.

        Args:
            embedding_value: The raw embedding value from PostgreSQL query

        Returns:
            List of floats or None if parsing fails
        """
        if embedding_value is None:
            return None

        try:
            # If it's already a list or tuple, convert to list
            if isinstance(embedding_value, (list, tuple)):
                return list(embedding_value)

            # If it's a string, try different parsing approaches
            if isinstance(embedding_value, str):
                # Remove any whitespace
                embedding_value = embedding_value.strip()

                # Check if it's a JSON array string: "[-0.006639634,-0.0114432...]"
                if embedding_value.startswith("[") and embedding_value.endswith("]"):
                    try:
                        return json.loads(embedding_value)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try manual parsing
                        # Remove brackets and split by comma
                        inner = embedding_value[1:-1]  # Remove [ and ]
                        return [float(x.strip()) for x in inner.split(",") if x.strip()]

                # Try comma-separated values
                if "," in embedding_value:
                    return [
                        float(x.strip())
                        for x in embedding_value.split(",")
                        if x.strip()
                    ]

                # Try space-separated values
                if " " in embedding_value:
                    return [
                        float(x.strip()) for x in embedding_value.split() if x.strip()
                    ]

            # Try using the original deserialize_vector approach for binary data
            try:
                from mirix.helpers.converters import deserialize_vector

                class MockDialect:
                    name = "postgresql"

                return deserialize_vector(embedding_value, MockDialect())
            except Exception:
                pass

            # If all else fails, return None to avoid validation errors
            return None

        except Exception as e:
            logger.error("Warning: Failed to parse embedding field: %s", e)
            return None

    def _count_word_matches(
        self, item_data: Dict[str, Any], query_words: List[str], search_field: str = ""
    ) -> int:
        """
        Count how many of the query words are present in the procedural memory item data.

        Args:
            item_data: Dictionary containing procedural memory item data
            query_words: List of query words to search for
            search_field: Specific field to search in, or empty string to search all text fields

        Returns:
            Number of query words found in the item
        """
        if not query_words:
            return 0

        # Determine which text fields to search in
        if search_field == "description":
            search_texts = [item_data.get("description", "")]
        elif search_field == "instructions":
            search_texts = [item_data.get("instructions", "")]
        elif search_field == "entry_type":
            search_texts = [item_data.get("entry_type", "")]
        else:
            # Search across all relevant text fields
            search_texts = [
                item_data.get("description", ""),
                item_data.get("instructions", ""),
                item_data.get("entry_type", ""),
            ]

        # Combine all search texts and clean them (remove punctuation)
        combined_text = " ".join(text for text in search_texts if text)
        cleaned_combined_text = self._clean_text_for_search(combined_text)

        # Count how many query words are present
        word_matches = 0
        for word in query_words:
            # Query words are already cleaned, so we can do direct comparison
            if word in cleaned_combined_text:
                word_matches += 1

        return word_matches

    async def _postgresql_fulltext_search(
        self,
        session,
        base_query,
        query_text,
        search_field,
        limit,
        user,
        filter_tags=None,
        scopes=None,
    ):
        """
        Efficient PostgreSQL-native full-text search using ts_rank_cd for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.

        Args:
            session: Database session
            base_query: Base SQLAlchemy query (not used, kept for API compatibility)
            query_text: Search query string
            search_field: Field to search in ('description', 'instructions', 'entry_type', etc.)
            limit: Maximum number of results to return
            user: User object to filter by
            filter_tags: Optional dict of tag key-value pairs to filter by (e.g., {"scope": "CARE"})
            scopes: Optional list of scope strings the caller is authorized to read.

        Returns:
            List of ProceduralMemoryItem objects ranked by relevance
        """
        from sqlalchemy import func

        from mirix.database.filter_tags_query import build_filter_tags_raw_sql

        # Clean and prepare the search query
        cleaned_query = self._clean_text_for_search(query_text)
        if not cleaned_query.strip():
            return []

        # Split into words and create a tsquery - PostgreSQL will handle the ranking
        query_words = [word.strip() for word in cleaned_query.split() if word.strip()]
        if not query_words:
            return []

        # Create tsquery string with improved logic
        tsquery_parts = []
        for word in query_words:
            # Escape special characters for tsquery
            escaped_word = (
                word.replace("'", "''")
                .replace("&", "")
                .replace("|", "")
                .replace("!", "")
                .replace(":", "")
            )
            if escaped_word and len(escaped_word) > 1:  # Skip very short words
                # Add both exact and prefix matching for better results
                if len(escaped_word) >= 3:
                    tsquery_parts.append(f"('{escaped_word}' | '{escaped_word}':*)")
                else:
                    tsquery_parts.append(f"'{escaped_word}'")

        if not tsquery_parts:
            return []

        # One OR query, ranked by ts_rank_cd. (The old "AND first, OR when AND
        # finds fewer than 10" pass is gone: skill RETRIEVAL ranks
        # with true BM25 in retrieve_skills; this per-field keyword search
        # serves the remaining field-specific searches, e.g. the
        # instructions search.)
        tsquery_string = " | ".join(tsquery_parts)

        # Determine which field to search based on search_field
        if search_field == "description":
            tsvector_sql = "to_tsvector('english', coalesce(description, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(description, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "instructions":
            tsvector_sql = "to_tsvector('english', coalesce(instructions, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(instructions, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "entry_type":
            tsvector_sql = "to_tsvector('english', coalesce(entry_type, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(entry_type, '')), to_tsquery('english', :tsquery), 32)"
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(description, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(instructions, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(entry_type, '')), 'C')"""
            rank_sql = """ts_rank_cd(
                setweight(to_tsvector('english', coalesce(description, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(instructions, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(entry_type, '')), 'C'),
                to_tsquery('english', :tsquery), 32)"""

        # Build WHERE clauses dynamically
        where_clauses = [
            f"{tsvector_sql} @@ to_tsquery('english', :tsquery)",
            "user_id = :user_id",
            # Exclude soft-deleted skills from the native PG full-text path too,
            # so a curator soft-delete is excluded from retrieval everywhere.
            "is_deleted = false",
        ]
        query_params = {
            "tsquery": tsquery_string,
            "user_id": user.id,
            "limit_val": limit or 50,
        }

        ft_clauses, ft_params = build_filter_tags_raw_sql(filter_tags, scopes=scopes)
        where_clauses.extend(ft_clauses)
        query_params.update(ft_params)

        where_clause = " AND ".join(where_clauses)

        async def _fetch_ordered_items(tsquery_value):
            # Rank via raw SQL for ts_rank_cd, but fetch only IDs to avoid
            # hydrating partial ORM instances (which silently drops triggers,
            # examples, version, filter_tags, etc.). Full skill rows are then
            # loaded through SQLAlchemy so every schema field is present.
            params = {**query_params, "tsquery": tsquery_value}
            id_query_sql = text(
                f"""
                SELECT id, {rank_sql} as rank_score
                FROM procedural_memory
                WHERE {where_clause}
                ORDER BY rank_score DESC, created_at DESC, id
                LIMIT :limit_val
                """
            )
            rows = (await session.execute(id_query_sql, params)).all()
            ordered_ids = [row.id for row in rows]
            if not ordered_ids:
                return []
            items_result = await session.execute(
                select(ProceduralMemoryItem).where(
                    ProceduralMemoryItem.id.in_(ordered_ids)
                )
            )
            items_by_id = {item.id: item for item in items_result.scalars().all()}
            return [
                items_by_id[item_id]
                for item_id in ordered_ids
                if item_id in items_by_id
            ]

        try:
            procedures = await _fetch_ordered_items(tsquery_string)
            return [procedure.to_pydantic() for procedure in procedures]
        except Exception as e:
            # If there's an error with the tsquery, fall back to simpler search
            # (after a rollback: Postgres refuses every further statement of
            # an aborted transaction).
            logger.error("PostgreSQL full-text search error: %s", e)
            await session.rollback()
            # Fall back to simple ILIKE search on a whitelisted field
            fallback_column = (
                _resolve_text_column(search_field) or ProceduralMemoryItem.description
            )
            fallback_query = (
                select(ProceduralMemoryItem)
                .where(ProceduralMemoryItem.user_id == user.id)
                # Exclude soft-deleted skills on the PG FTS error fallback too.
                .where(~ProceduralMemoryItem.is_deleted)
                .where(func.lower(fallback_column).contains(query_text.lower()))
                .order_by(ProceduralMemoryItem.created_at.desc())
            )

            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            fallback_query = apply_filter_tags_sqlalchemy(
                fallback_query, ProceduralMemoryItem, filter_tags, scopes=scopes
            )

            if limit:
                fallback_query = fallback_query.limit(limit)

            result = await session.execute(fallback_query)
            procedures = result.scalars().all()
            return [procedure.to_pydantic() for procedure in procedures]

    @update_timezone
    @enforce_types
    async def get_item_by_id(
        self,
        item_id: str,
        user: PydanticUser,
        timezone_str: str,
        actor: Optional[PydanticClient] = None,
        scopes: Optional[List[str]] = None,
    ) -> Optional[PydanticProceduralMemoryItem]:
        """Fetch a live (not soft-deleted) skill by ID, straight from the
        database — procedural memory has no cache.

        Access control (see ``delete_procedure_by_id`` for the same pattern):

        * ``actor`` → organization scope via ``apply_access_predicate``.
        * ``user``  → explicit post-load ``user_id`` equality check.
        * ``scopes`` → the skill's ``filter_tags.scope`` must be one of them
          (None = no scope restriction, ``[]`` = nothing).

        Passing only ``user`` is not enough, because ``SqlalchemyBase.read``
        only runs the access predicate when ``actor`` is truthy. A skill
        outside any of them is :class:`NoResultFound` (existence never leaks).
        """
        async with self.session_maker() as session:
            try:
                item = await ProceduralMemoryItem.read(
                    db_session=session, identifier=item_id, actor=actor
                )
            except NoResultFound:
                raise NoResultFound(
                    f"Procedural memory item with id {item_id} not found."
                )

            if getattr(item, "user_id", None) != user.id or not _in_scopes(
                item, scopes
            ):
                raise NoResultFound(
                    f"Procedural memory item with id {item_id} not found."
                )

            return item.to_pydantic()

    async def get_live_skill_by_name(
        self,
        *,
        user: PydanticUser,
        name: str,
        scopes: Optional[List[str]] = None,
    ) -> Optional[PydanticProceduralMemoryItem]:
        """The user's live skill named exactly ``name`` (in ``scopes`` when
        given), or None."""
        stmt = (
            select(ProceduralMemoryItem)
            .where(
                ProceduralMemoryItem.organization_id
                == _resolve_org(user.organization_id),
                ProceduralMemoryItem.user_id == user.id,
                ProceduralMemoryItem.name == name,
                ~ProceduralMemoryItem.is_deleted,
            )
            .options(lazyload("*"))
            .limit(1)
        )
        if scopes is not None:
            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            stmt = apply_filter_tags_sqlalchemy(
                stmt, ProceduralMemoryItem, None, scopes=list(scopes)
            )
        async with self.session_maker() as session:
            row = (await session.execute(stmt)).scalar_one_or_none()
            return row.to_pydantic() if row is not None else None

    async def list_skills_page(
        self,
        *,
        user: PydanticUser,
        scopes: Optional[List[str]] = None,
        query: str = "",
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[PydanticProceduralMemoryItem], int]:
        """One page of the user's live skills, sorted by name.

        ``query`` (optional) keeps the skills whose name, description or
        triggers contain every whitespace-separated term (case-insensitive).
        Returns ``(skills, total)`` where ``total`` counts every matching
        skill, so a caller can page through all of them with ``offset``.
        """
        from sqlalchemy import String, and_, cast

        from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

        base = select(ProceduralMemoryItem).where(
            ProceduralMemoryItem.organization_id == _resolve_org(user.organization_id),
            ProceduralMemoryItem.user_id == user.id,
            ~ProceduralMemoryItem.is_deleted,
        )
        if scopes is not None:
            base = apply_filter_tags_sqlalchemy(
                base, ProceduralMemoryItem, None, scopes=list(scopes)
            )
        terms = [t for t in (query or "").lower().split() if t]
        if terms:
            base = base.where(
                and_(
                    *[
                        or_(
                            func.lower(ProceduralMemoryItem.name).contains(
                                term, autoescape=True
                            ),
                            func.lower(ProceduralMemoryItem.description).contains(
                                term, autoescape=True
                            ),
                            func.lower(
                                cast(ProceduralMemoryItem.triggers, String)
                            ).contains(term, autoescape=True),
                        )
                        for term in terms
                    ]
                )
            )
        offset = max(0, int(offset or 0))
        limit = max(1, int(limit))
        async with self.session_maker() as session:
            total = (
                await session.execute(
                    select(func.count()).select_from(base.subquery())
                )
            ).scalar_one()
            rows = (
                await session.execute(
                    base.options(lazyload("*"))
                    .order_by(ProceduralMemoryItem.name, ProceduralMemoryItem.id)
                    .offset(offset)
                    .limit(limit)
                )
            ).scalars().all()
            return [row.to_pydantic() for row in rows], int(total or 0)

    async def list_live_skill_names(
        self,
        *,
        user: PydanticUser,
        scopes: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """``[(id, name)]`` of the user's live skills (in ``scopes`` when
        given), sorted by name — how a curator batch finds the skills its
        experiences name. A transient database failure is retried."""
        from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy
        from mirix.services.retry_policy import retry_call

        stmt = select(ProceduralMemoryItem.id, ProceduralMemoryItem.name).where(
            ProceduralMemoryItem.organization_id == _resolve_org(user.organization_id),
            ProceduralMemoryItem.user_id == user.id,
            ~ProceduralMemoryItem.is_deleted,
        )
        if scopes is not None:
            stmt = apply_filter_tags_sqlalchemy(
                stmt, ProceduralMemoryItem, None, scopes=list(scopes)
            )
        stmt = stmt.order_by(ProceduralMemoryItem.name, ProceduralMemoryItem.id)

        async def _once():
            async with self.session_maker() as session:
                return [
                    (str(row[0]), str(row[1] or ""))
                    for row in (await session.execute(stmt)).all()
                ]

        return await retry_call(_once, stage="skill_names_read")

    async def get_skill_labels(
        self,
        *,
        user: PydanticUser,
        skill_ids: Sequence[str],
        scopes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """``{skill_id: {"name", "version", "is_deleted",
        "instructions_changed_at"}}`` for the user's skills among
        ``skill_ids`` (deleted ones included) — how a curator batch names the
        skills an earlier attempt already changed. ``instructions_changed_at``
        is when a curator batch last created the skill or changed its
        instructions (a version with a ``run_id`` whose instructions differ
        from its parent's), None when none did: a retried batch's free
        retrieval-text follow-up counts that change. With ``scopes``
        only skills in those scopes are named."""
        ids = [sid for sid in dict.fromkeys(skill_ids or []) if sid]
        if not ids:
            return {}
        stmt = select(
            ProceduralMemoryItem.id,
            ProceduralMemoryItem.name,
            ProceduralMemoryItem.version,
            ProceduralMemoryItem.is_deleted,
        ).where(
            ProceduralMemoryItem.id.in_(ids),
            ProceduralMemoryItem.organization_id == _resolve_org(user.organization_id),
            ProceduralMemoryItem.user_id == user.id,
        )
        if scopes is not None:
            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            stmt = apply_filter_tags_sqlalchemy(
                stmt, ProceduralMemoryItem, None, scopes=list(scopes)
            )
        async with self.session_maker() as session:
            rows = (await session.execute(stmt)).all()
            found = [row[0] for row in rows]
            changed: Dict[str, Any] = {}
            if found:
                from sqlalchemy.orm import aliased

                parent = aliased(SkillVersionModel)
                versions = (
                    await session.execute(
                        select(
                            SkillVersionModel.skill_id,
                            SkillVersionModel.created_at,
                            SkillVersionModel.instructions_sha256,
                            parent.instructions_sha256,
                        )
                        .outerjoin(
                            parent, parent.id == SkillVersionModel.parent_version_id
                        )
                        .where(
                            SkillVersionModel.skill_id.in_(found),
                            SkillVersionModel.run_id.is_not(None),
                            SkillVersionModel.change_type.in_(
                                (history.CHANGE_CREATE, history.CHANGE_EDIT)
                            ),
                        )
                    )
                ).all()
                for skill_id, created_at, sha, parent_sha in versions:
                    if created_at is None or sha == parent_sha:
                        continue
                    if skill_id not in changed or created_at > changed[skill_id]:
                        changed[skill_id] = created_at
        return {
            row[0]: {
                "name": row[1],
                "version": row[2],
                "is_deleted": bool(row[3]),
                "instructions_changed_at": changed.get(row[0]),
            }
            for row in rows
        }

    @update_timezone
    @enforce_types
    async def get_most_recently_updated_item(
        self, user: PydanticUser, timezone_str: str = None
    ) -> Optional[PydanticProceduralMemoryItem]:
        """
        Fetch the most recently updated procedural memory item based on last_modify timestamp.
        Filter by user_id from actor.
        Returns None if no items exist.
        """
        async with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import DateTime, cast, text

            query = select(ProceduralMemoryItem).order_by(
                cast(
                    text("procedural_memory.last_modify ->> 'timestamp'"), DateTime
                ).desc()
            )

            # Filter by user_id for multi-user support; never a soft-deleted
            # skill.
            query = query.where(ProceduralMemoryItem.user_id == user.id).where(
                ~ProceduralMemoryItem.is_deleted
            )

            result = await session.execute(query.limit(1))
            item = result.scalar_one_or_none()

            return [item.to_pydantic()] if item else None

    # ------------------------------------------------------------------ #
    # The one write path: apply_skill_change                               #
    # ------------------------------------------------------------------ #

    async def apply_skill_change(
        self,
        change_type: str,
        *,
        actor: Optional[PydanticClient],
        user_id: Optional[str],
        skill_id: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        version_no: Optional[int] = None,
        agent_state: Optional[AgentState] = None,
        organization_id: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
        reason: Optional[str] = None,
        consume_experience_ids: Optional[Sequence[str]] = None,
        scopes: Optional[Sequence[str]] = None,
        write_guard: Optional[Callable[[Any], Awaitable[None]]] = None,
        lineage_experience_ids: Optional[Sequence[str]] = None,
        expected_fields: Optional[Mapping[str, Any]] = None,
        bump_version: bool = False,
        cited_experience_ids: Optional[Sequence[str]] = None,
    ) -> SkillChangeResult:
        """Create, edit, soft-delete or restore a skill, with its history.

        Every path that changes a skill goes through here:

        1. OUTSIDE any transaction, read the skill, work out its new content
           and — when its retrieval text (name, triggers, description)
           changes — compute the new retrieval vector (embedding failures are
           retried with backoff, then leave the vector NULL for
           re-embedding).
        2. In ONE transaction: the writer's locks (Postgres: shared org
           advisory lock, then the (org, user, name) advisory lock when a name
           is claimed; SQLite: the database write lock), ``SELECT ... FOR
           UPDATE`` of the skill row, a ``baseline`` version first if the skill
           has no history yet, the content-addressed objects (``INSERT ... ON
           CONFLICT DO UPDATE SET last_referenced_at``), the new version
           (``version_no = max + 1``), the skill row itself and — when
           ``consume_experience_ids`` is given — those pending experiences
           marked consumed. A failure anywhere rolls all of it back; a
           transient database failure repeats the whole transaction.

        ``change_type``:

        * ``create`` — ``fields`` holds the new skill (the four text fields are
          required; ``id``, ``agent_id``, ``client_id``, ``filter_tags``,
          ``version``, ``organization_id``, ``embedding_config`` and an
          explicit ``retrieval_embedding`` are optional). The owner is
          ``user_id``.
        * ``edit`` — ``fields`` holds the changed versioned fields and/or
          ``version``, ``agent_id``, ``embedding_config``,
          ``retrieval_embedding``. Content identical to the current one writes
          no version (``result.version is None``).
        * ``delete`` — soft delete: ``is_deleted`` and a ``delete``
          version carrying ``reason``; the name becomes free for a new skill
          (partial unique index).
        * ``restore`` — content of version ``version_no``; un-deletes a
          deleted skill. Raises :class:`SkillNameConflictError` when another
          live skill of the owner holds the name.

        Access: ``actor`` limits the skill to its organization and ``user_id``
        to its owner; both None is reserved for system-initiated bulk work
        (erasure). ``scopes`` additionally limits an existing skill to those
        ``filter_tags.scope`` values (a caller confined to a scope cannot
        change another scope's skill by id). A skill outside them is
        :class:`NoResultFound`; for a ``restore`` the target version must be of
        one of them too (otherwise :class:`SkillVersionNotFound`), and an
        ``edit`` cannot move the skill out of them.

        A read-modify-write (a skill tool's patch, computed from a skill it
        read before this call): ``expected_fields`` holds the values the edit
        was computed from, and the change is refused with
        :class:`SkillChangedMeanwhile` — nothing written — when the LOCKED row
        holds anything else, so a concurrent change is never silently
        overwritten. ``bump_version`` sets the row's version string to the
        locked row's one plus a patch (not a string computed from the stale
        read) when the edit writes a version.

        A transient database failure repeats the transaction; when the
        failure hid a COMMIT that did go through, the repeat finds the version
        it wrote (its id is fixed before the first attempt) and returns that
        result: a create is not reported as a duplicate and a delete not as
        "not found".

        Skill learning (the curator): ``write_guard(db)`` runs inside
        the transaction, right after the locks — the curator passes a check
        that locks its learning run and requires it to be still active, so a
        run that lost its lease (or was erased, or cancelled) cannot change a
        skill any more; whatever it raises rolls the change back.
        ``lineage_experience_ids`` stamps the skill on those experiences of the
        owner (``influenced_skill_ids``) in the same transaction when a version
        is written. ``cited_experience_ids`` stamps the skill on those
        experiences of the owner (``cited_skill_ids``), in the same
        transaction, when a version is written: the experiences the change
        itself cites, not the whole batch.
        """
        if change_type not in (
            history.CHANGE_CREATE,
            history.CHANGE_EDIT,
            history.CHANGE_DELETE,
            history.CHANGE_RESTORE,
        ):
            raise ValueError(f"unknown skill change type {change_type!r}")
        if change_type != history.CHANGE_CREATE and not skill_id:
            raise ValueError(f"a {change_type} needs the skill_id")
        if change_type == history.CHANGE_RESTORE and version_no is None:
            raise ValueError("a restore needs the version_no to restore")
        if change_type == history.CHANGE_CREATE and not user_id:
            raise ValueError("a create needs the owning user_id")
        if expected_fields is not None and change_type != history.CHANGE_EDIT:
            raise ValueError("expected_fields is only for an edit")
        fields = dict(fields or {})
        if change_type in (history.CHANGE_CREATE, history.CHANGE_EDIT):
            # Before any read, embedding or transaction.
            problem = skill_nul_problem(fields)
            if problem:
                raise SkillFieldInvalid(problem)

        for attempt in range(_MAX_PLAN_ATTEMPTS):
            plan = await self._prepare_change(
                change_type,
                actor=actor,
                user_id=user_id,
                skill_id=skill_id,
                fields=fields,
                version_no=version_no,
                agent_state=agent_state,
                organization_id=organization_id,
                allow_stale_embedding=attempt == _MAX_PLAN_ATTEMPTS - 1,
                scopes=list(scopes) if scopes is not None else None,
                expected_fields=(
                    dict(expected_fields) if expected_fields is not None else None
                ),
                bump_version=bump_version,
            )

            async def _commit(plan=plan):
                return await self._commit_change(
                    plan,
                    actor=actor,
                    run_id=run_id,
                    actor_kind=actor_kind,
                    reason=reason,
                    consume_experience_ids=consume_experience_ids,
                    write_guard=write_guard,
                    lineage_experience_ids=lineage_experience_ids,
                    cited_experience_ids=cited_experience_ids,
                )

            try:
                return await retry_call(_commit, stage="skill_write")
            except _StalePlan:
                logger.info(
                    "Skill %s changed while its %s was being prepared; preparing again",
                    plan.skill_id,
                    change_type,
                )
                continue
        raise RuntimeError("unreachable: the last plan attempt accepts a stale vector")

    # -- step 1: outside the transaction ---------------------------------- #

    async def _prepare_change(
        self,
        change_type: str,
        *,
        actor: Optional[PydanticClient],
        user_id: Optional[str],
        skill_id: Optional[str],
        fields: Dict[str, Any],
        version_no: Optional[int],
        agent_state: Optional[AgentState],
        organization_id: Optional[str],
        allow_stale_embedding: bool,
        scopes: Optional[List[str]] = None,
        expected_fields: Optional[Dict[str, Any]] = None,
        bump_version: bool = False,
    ) -> _ChangePlan:
        access_org = organization_id or (
            _resolve_org(actor.organization_id) if actor is not None else None
        )
        agent_config = getattr(agent_state, "embedding_config", None)
        explicit = fields.get(_RETRIEVAL_EMBEDDING_COLUMN) is not None

        if change_type == history.CHANGE_CREATE:
            missing = [f for f in _REQUIRED_CREATE_FIELDS if fields.get(f) is None]
            if missing:
                raise ValueError(
                    f"Required field(s) {missing} missing from procedural memory data"
                )
            content = {f: fields.get(f) for f in SKILL_VERSIONED_FIELDS}
            if content["triggers"] is None:
                content["triggers"] = []
            if content["examples"] is None:
                content["examples"] = []
            new_id = fields.get("id") or skill_id
            if not new_id:
                from mirix.utils import generate_unique_short_id_async

                new_id = await generate_unique_short_id_async(
                    self.session_maker, ProceduralMemoryItem, "proc"
                )
            skill_org = _resolve_org(
                fields.get("organization_id"), organization_id,
                getattr(actor, "organization_id", None),
            )
            plan = _ChangePlan(
                change_type=change_type,
                skill_id=new_id,
                organization_id=skill_org,
                lock_user_id=user_id,
                access_org_id=None,
                access_user_id=user_id,
                fields=fields,
                create_content=content,
                claimed_name=content["name"],
                allow_stale_embedding=allow_stale_embedding,
            )
            if explicit:
                plan.explicit_embedding = True
                plan.embedding = _pad_to_max(fields[_RETRIEVAL_EMBEDDING_COLUMN])
                plan.embedding_config = fields.get("embedding_config")
            else:
                config = agent_config or fields.get("embedding_config")
                plan.embedded_text = history.skill_retrieval_text(content)
                plan.embedding = await self._embed(plan.embedded_text, config)
                # Kept even when the call failed: a NULL vector is re-embedded
                # later with the row's own config.
                plan.embedding_config = config
            return plan

        row = await self._read_skill_row(
            skill_id,
            organization_id=access_org,
            user_id=user_id,
            include_deleted=change_type == history.CHANGE_RESTORE,
            scopes=scopes,
        )
        before = history.content_of(row)
        plan = _ChangePlan(
            change_type=change_type,
            skill_id=row.id,
            organization_id=_resolve_org(row.organization_id),
            lock_user_id=row.user_id,
            access_org_id=access_org,
            access_user_id=user_id,
            access_scopes=scopes,
            fields=fields,
            restore_version_no=version_no,
            allow_stale_embedding=allow_stale_embedding,
            expected_fields=expected_fields,
            bump_version=bump_version,
        )
        if change_type == history.CHANGE_DELETE:
            return plan
        if expected_fields is not None:
            # Early out (the locked row is checked again in the commit).
            differ = _fields_differ(_row_values(row, list(expected_fields)), expected_fields)
            if differ:
                raise SkillChangedMeanwhile(row.id, differ)

        # The name this change may claim is locked whether or not it differs
        # from the row's name right now: a concurrent rename or delete cannot
        # then leave the transaction holding the wrong name's lock.
        if change_type == history.CHANGE_EDIT:
            target = {
                f: (fields[f] if f in fields else before[f])
                for f in SKILL_VERSIONED_FIELDS
            }
            if "name" in fields:
                plan.claimed_name = fields["name"]
        else:  # restore
            async with self.session_maker() as db:
                version = await history.get_version(db, row.id, version_no)
                target, _ = await history.version_content(
                    db, organization_id=plan.organization_id, version=version
                )
            plan.claimed_name = target["name"]
        _check_target_scope(plan, target)

        if explicit:
            plan.explicit_embedding = True
            plan.embedding = _pad_to_max(fields[_RETRIEVAL_EMBEDDING_COLUMN])
            plan.embedding_config = fields.get("embedding_config")
            return plan
        target_text = history.skill_retrieval_text(target)
        if target_text != history.skill_retrieval_text(before):
            # Name, triggers or description changed -> a new vector. With
            # no agent_state, the row's own embedding_config is used.
            config = agent_config or row.embedding_config
            plan.embedded_text = target_text
            plan.embedding = await self._embed(target_text, config)
            plan.embedding_config = config
        elif row.retrieval_embedding is None:
            # A skill left without a vector (an embedding outage when it was
            # written) gets one from any edit or restore that touches
            # it — not only from the startup / end-of-run re-embedding pass.
            config = agent_config or row.embedding_config
            if config is not None:
                plan.embedded_text = target_text
                plan.embedding = await self._embed(target_text, config)
                plan.embedding_config = config
        return plan

    async def _embed(self, text_value: str, config) -> Optional[List[float]]:
        """The padded retrieval vector of ``text_value``, or None.

        None when embeddings are off or there is no embedding config, and when
        the embedding call still fails after its backoff retries — the skill
        is then stored without a vector (kept out of vector ranking until it is
        re-embedded) instead of failing the write.
        """
        if not BUILD_EMBEDDINGS_FOR_MEMORY or config is None:
            return None

        async def _call():
            model = await embedding_model(config)
            return await model.get_text_embedding(text_value)

        try:
            vector = await retry_call(
                _call, operation="embedding", stage="skill_embedding", sleep=self._sleep
            )
        except Exception as exc:  # noqa: BLE001 — store without a vector
            logger.warning(
                "Skill retrieval embedding failed after retries (%s: %s); "
                "storing the skill without a vector",
                type(exc).__name__,
                exc,
            )
            return None
        return _pad_to_max(vector)

    async def _read_skill_row(
        self,
        skill_id: str,
        *,
        organization_id: Optional[str],
        user_id: Optional[str],
        include_deleted: bool,
        scopes: Optional[List[str]] = None,
    ) -> ProceduralMemoryItem:
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    self._skill_row_query(
                        skill_id,
                        organization_id=organization_id,
                        user_id=user_id,
                        include_deleted=include_deleted,
                        scopes=scopes,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            raise NoResultFound(f"Procedural memory item with id {skill_id} not found.")
        return row

    @staticmethod
    def _skill_row_query(
        skill_id: str,
        *,
        organization_id: Optional[str],
        user_id: Optional[str],
        include_deleted: bool,
        scopes: Optional[List[str]] = None,
    ):
        stmt = (
            select(ProceduralMemoryItem)
            .where(ProceduralMemoryItem.id == skill_id)
            .options(lazyload("*"))
        )
        if organization_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.organization_id == organization_id)
        if user_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(~ProceduralMemoryItem.is_deleted)
        if scopes is not None:
            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            stmt = apply_filter_tags_sqlalchemy(
                stmt, ProceduralMemoryItem, None, scopes=list(scopes)
            )
        return stmt

    # -- step 2: the transaction ------------------------------------------- #

    async def _commit_change(
        self,
        plan: _ChangePlan,
        *,
        actor: Optional[PydanticClient],
        run_id: Optional[str],
        actor_kind: Optional[str],
        reason: Optional[str],
        consume_experience_ids: Optional[Sequence[str]],
        write_guard: Optional[Callable[[Any], Awaitable[None]]] = None,
        lineage_experience_ids: Optional[Sequence[str]] = None,
        cited_experience_ids: Optional[Sequence[str]] = None,
    ) -> SkillChangeResult:
        now = _utcnow()
        actor_id = getattr(actor, "id", None)
        name_keys = (
            [
                history.name_lock_key(
                    plan.organization_id, plan.lock_user_id, plan.claimed_name
                )
            ]
            if plan.claimed_name is not None
            else []
        )
        try:
            async with self.session_maker() as db:
                async with db.begin():
                    await history.lock_for_write(
                        db, organization_ids=[plan.organization_id], name_keys=name_keys
                    )
                    done = await self._committed_earlier(db, plan)
                    if done is not None:
                        return done
                    if write_guard is not None:
                        # e.g. "my learning run is still active", checked under
                        # a lock in THIS transaction (see apply_skill_change).
                        await write_guard(db)
                    if plan.change_type == history.CHANGE_CREATE:
                        result = await self._commit_create(
                            db, plan, now=now, actor_id=actor_id, run_id=run_id,
                            actor_kind=actor_kind, reason=reason,
                        )
                    else:
                        result = await self._commit_existing(
                            db, plan, now=now, actor_id=actor_id, run_id=run_id,
                            actor_kind=actor_kind, reason=reason,
                        )
                    if lineage_experience_ids and result.version is not None:
                        await self._stamp_lineage(
                            db,
                            ids=lineage_experience_ids,
                            owner_user_id=result.skill.user_id,
                            skill_id=result.skill.id,
                            now=now,
                        )
                    if cited_experience_ids and result.version is not None:
                        await self._stamp_citations(
                            db,
                            ids=cited_experience_ids,
                            owner_user_id=result.skill.user_id,
                            skill_id=result.skill.id,
                            now=now,
                        )
                    if consume_experience_ids and result.version is not None:
                        await self._consume_experiences(
                            db,
                            ids=consume_experience_ids,
                            owner_user_id=result.skill.user_id,
                            consumed_by=run_id or result.version.id,
                            skill_id=result.skill.id,
                            now=now,
                        )
                return result
        except IntegrityError as exc:
            if _is_live_name_violation(exc):
                raise SkillNameConflictError(
                    f"Skill with name '{plan.claimed_name}' already exists for this user."
                ) from exc
            raise

    async def _committed_earlier(
        self, db, plan: _ChangePlan
    ) -> Optional[SkillChangeResult]:
        """The result of ``plan`` when an earlier attempt of its transaction
        already committed — the version it writes exists (a transient
        error on COMMIT is ambiguous, and the transaction is repeated); None
        otherwise. Runs under the writer's locks."""
        version = (
            await db.execute(
                select(SkillVersionModel).where(SkillVersionModel.id == plan.version_id)
            )
        ).scalar_one_or_none()
        if version is None:
            return None
        row = (
            await db.execute(
                self._skill_row_query(
                    version.skill_id,
                    organization_id=None,
                    user_id=None,
                    include_deleted=True,
                )
            )
        ).scalar_one()
        logger.info(
            "Skill %s: the %s (version %s) had already been committed by an "
            "earlier attempt of its transaction",
            row.id,
            plan.change_type,
            version.version_no,
        )
        return SkillChangeResult(skill=row.to_pydantic(), version=_version_schema(version))

    async def _check_name_free(
        self, db, *, plan: _ChangePlan, name: str, skill_id: Optional[str]
    ) -> None:
        stmt = select(ProceduralMemoryItem.id).where(
            ProceduralMemoryItem.organization_id == plan.organization_id,
            ProceduralMemoryItem.user_id == plan.lock_user_id,
            ProceduralMemoryItem.name == name,
            ~ProceduralMemoryItem.is_deleted,
        )
        if skill_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.id != skill_id)
        taken = (await db.execute(stmt.limit(1))).scalar_one_or_none()
        if taken is not None:
            raise SkillNameConflictError(
                f"Skill with name '{name}' already exists for this user.",
                holder_id=taken,
            )

    async def _commit_create(
        self, db, plan: _ChangePlan, *, now, actor_id, run_id, actor_kind, reason
    ) -> SkillChangeResult:
        content = plan.create_content
        actor_client_id = actor_id  # the creating client, unless given
        await self._check_name_free(db, plan=plan, name=content["name"], skill_id=None)
        data = plan.fields
        row = ProceduralMemoryItem(
            id=plan.skill_id,
            organization_id=plan.organization_id,
            user_id=plan.lock_user_id,
            agent_id=data.get("agent_id"),
            client_id=data.get("client_id") or actor_client_id,
            version=data.get("version") or "0.1.0",
            last_modify={"timestamp": now.isoformat(), "operation": "created"},
            embedding_config=plan.embedding_config,
            retrieval_embedding=plan.embedding,
            created_at=now,
            updated_at=now,
            is_deleted=False,
            **content,
        )
        if actor_id:
            row._created_by_id = actor_id
            row._last_updated_by_id = actor_id
        db.add(row)
        await db.flush()
        encoded = history.encode_content(content)
        await history.upsert_objects(
            db, organization_id=plan.organization_id, encoded=encoded, now=now
        )
        version = await history.insert_version(
            db,
            skill_id=row.id,
            organization_id=plan.organization_id,
            user_id=row.user_id,
            version_no=1,
            parent_version_id=None,
            change_type=history.CHANGE_CREATE,
            hashes=history.hashes_of(encoded),
            now=now,
            run_id=run_id,
            actor_kind=actor_kind,
            reason=reason,
            actor_id=actor_id,
            version_id=plan.version_id,
        )
        return SkillChangeResult(skill=row.to_pydantic(), version=_version_schema(version))

    async def _commit_existing(
        self, db, plan: _ChangePlan, *, now, actor_id, run_id, actor_kind, reason
    ) -> SkillChangeResult:
        change_type = plan.change_type
        row = (
            await db.execute(
                self._skill_row_query(
                    plan.skill_id,
                    organization_id=plan.access_org_id,
                    user_id=plan.access_user_id,
                    include_deleted=change_type == history.CHANGE_RESTORE,
                    scopes=plan.access_scopes,
                ).with_for_update()
            )
        ).scalar_one_or_none()
        if row is None:
            raise NoResultFound(
                f"Procedural memory item with id {plan.skill_id} not found."
            )
        before = history.content_of(row)
        if plan.expected_fields is not None:
            differ = _fields_differ(
                _row_values(row, list(plan.expected_fields)), plan.expected_fields
            )
            if differ:
                raise SkillChangedMeanwhile(row.id, differ)
        before_encoded = history.encode_content(before)
        before_hashes = history.hashes_of(before_encoded)

        # A skill without history gets its baseline first.
        latest = await history.latest_version(db, row.id)
        baseline = None
        if latest is None:
            await history.upsert_objects(
                db, organization_id=plan.organization_id, encoded=before_encoded, now=now
            )
            baseline = await history.insert_version(
                db,
                skill_id=row.id,
                organization_id=plan.organization_id,
                user_id=row.user_id,
                version_no=1,
                parent_version_id=None,
                change_type=history.CHANGE_BASELINE,
                hashes=before_hashes,
                now=now,
                actor_kind=ACTOR_SYSTEM,
                actor_id=actor_id,
            )
            latest = baseline

        if change_type == history.CHANGE_EDIT:
            target = {
                f: (plan.fields[f] if f in plan.fields else before[f])
                for f in SKILL_VERSIONED_FIELDS
            }
        elif change_type == history.CHANGE_DELETE:
            target = dict(before)
        else:
            version = await history.get_version(db, row.id, plan.restore_version_no)
            target, _ = await history.version_content(
                db, organization_id=plan.organization_id, version=version
            )
        if change_type != history.CHANGE_DELETE:
            _check_target_scope(plan, target)

        # Claiming a name: renaming, or bringing a deleted skill back.
        claims_name = (
            change_type in (history.CHANGE_EDIT, history.CHANGE_RESTORE)
            and (target["name"] != row.name or row.is_deleted)
        )
        if claims_name:
            if target["name"] != plan.claimed_name:
                raise _StalePlan()  # we hold another name's lock
            await self._check_name_free(
                db, plan=plan, name=target["name"], skill_id=row.id
            )

        # The retrieval vector, computed outside the transaction for the text
        # the prepare step saw; stale if the locked row says otherwise.
        new_vector_needed = False
        if not plan.explicit_embedding and change_type != history.CHANGE_DELETE:
            target_text = history.skill_retrieval_text(target)
            if target_text != history.skill_retrieval_text(before):
                new_vector_needed = True
                if plan.embedded_text != target_text:
                    if not plan.allow_stale_embedding:
                        raise _StalePlan()
                    plan.embedding = None  # re-embedded later
                    plan.embedded_text = target_text
            elif (
                row.retrieval_embedding is None
                and plan.embedding is not None
                and plan.embedded_text == target_text
            ):
                # Fill a missing vector computed for exactly this text (the
                # prepare step saw the vector missing).
                new_vector_needed = True

        encoded = history.encode_content(target)
        hashes = history.hashes_of(encoded)
        content_changed = hashes != before_hashes
        state_change = change_type == history.CHANGE_DELETE or (
            change_type == history.CHANGE_RESTORE and row.is_deleted
        )

        version = None
        if content_changed or state_change:
            await history.upsert_objects(
                db, organization_id=plan.organization_id, encoded=encoded, now=now
            )
            version_reason = reason
            if change_type == history.CHANGE_RESTORE and version_reason is None:
                version_reason = f"restored version {plan.restore_version_no}"
            version = await history.insert_version(
                db,
                skill_id=row.id,
                organization_id=plan.organization_id,
                user_id=row.user_id,
                version_no=latest.version_no + 1,
                parent_version_id=latest.id,
                change_type=change_type,
                hashes=hashes,
                now=now,
                run_id=run_id,
                actor_kind=actor_kind,
                reason=version_reason,
                actor_id=actor_id,
                version_id=plan.version_id,
            )

        meta_changed = False
        if change_type == history.CHANGE_EDIT:
            for key in _EDITABLE_META_FIELDS - {_RETRIEVAL_EMBEDDING_COLUMN}:
                if key in plan.fields and getattr(row, key) != plan.fields[key]:
                    setattr(row, key, plan.fields[key])
                    meta_changed = True
        if plan.explicit_embedding:
            # The vector is derived from the retrieval text, not part of the
            # skill: set on its own (e.g. re-embedding a skill left without
            # one) it leaves the row's modification stamp alone.
            row.retrieval_embedding = plan.embedding
            if (
                plan.embedding_config is not None
                and row.embedding_config != plan.embedding_config
            ):
                row.embedding_config = plan.embedding_config
                meta_changed = True
        elif new_vector_needed:
            row.retrieval_embedding = plan.embedding
            if plan.embedding_config is not None:
                row.embedding_config = plan.embedding_config

        if version is None and not meta_changed:
            return SkillChangeResult(
                skill=row.to_pydantic(), baseline=_version_schema(baseline)
            )

        if content_changed:
            for key in SKILL_VERSIONED_FIELDS:
                setattr(row, key, target[key])
        if (
            plan.bump_version
            and change_type == history.CHANGE_EDIT
            and version is not None
            and "version" not in plan.fields
        ):
            row.version = _bump_patch(row.version)
        operation = {
            history.CHANGE_EDIT: "updated",
            history.CHANGE_DELETE: "deleted",
            history.CHANGE_RESTORE: "restored",
        }[change_type]
        last_modify: Dict[str, Any] = {"timestamp": now.isoformat(), "operation": operation}
        if change_type == history.CHANGE_DELETE:
            row.is_deleted = True
            if reason:
                last_modify["reason"] = reason
        elif change_type == history.CHANGE_RESTORE:
            row.is_deleted = False
            row.version = _bump_patch(row.version)
            last_modify["restored_version_no"] = plan.restore_version_no
        row.last_modify = last_modify
        row.updated_at = now
        if actor_id:
            row._last_updated_by_id = actor_id
        await db.flush()
        return SkillChangeResult(
            skill=row.to_pydantic(),
            version=_version_schema(version),
            baseline=_version_schema(baseline),
        )

    async def _consume_experiences(
        self,
        db,
        *,
        ids: Sequence[str],
        owner_user_id: str,
        consumed_by: str,
        skill_id: str,
        now: dt.datetime,
    ) -> int:
        """Mark the owner's still-pending experiences consumed, in the write's
        transaction. Other owners' ids are never touched."""
        rows = (
            await db.execute(
                select(SkillExperienceModel).where(
                    SkillExperienceModel.id.in_(list(ids)),
                    SkillExperienceModel.user_id == owner_user_id,
                    SkillExperienceModel.status == "pending",
                    ~SkillExperienceModel.is_deleted,
                )
            )
        ).scalars().all()
        for row in rows:
            row.status = "consumed"
            row.consumed_by = consumed_by
            lineage = list(row.influenced_skill_ids or [])
            if skill_id not in lineage:
                lineage.append(skill_id)
            row.influenced_skill_ids = lineage
            row.updated_at = now
        await db.flush()
        return len(rows)

    async def _stamp_lineage(
        self,
        db,
        *,
        ids: Sequence[str],
        owner_user_id: str,
        skill_id: str,
        now: dt.datetime,
    ) -> int:
        """Add ``skill_id`` to ``influenced_skill_ids`` of the owner's
        experiences ``ids`` (any status), in the write's transaction: the
        lineage of a curator batch, kept even when the batch never finishes so
        a later attempt on the same experiences sees what was already changed.
        Other owners' ids are never touched."""
        rows = (
            await db.execute(
                select(SkillExperienceModel).where(
                    SkillExperienceModel.id.in_(list(ids)),
                    SkillExperienceModel.user_id == owner_user_id,
                    ~SkillExperienceModel.is_deleted,
                )
            )
        ).scalars().all()
        stamped = 0
        for row in rows:
            lineage = list(row.influenced_skill_ids or [])
            if skill_id in lineage:
                continue
            lineage.append(skill_id)
            row.influenced_skill_ids = lineage
            row.updated_at = now
            stamped += 1
        await db.flush()
        return stamped

    async def _stamp_citations(
        self,
        db,
        *,
        ids: Sequence[str],
        owner_user_id: str,
        skill_id: str,
        now: dt.datetime,
    ) -> int:
        """Add ``skill_id`` to ``cited_skill_ids`` of the owner's experiences
        ``ids`` (any status: a consumed pool experience too), in the write's
        transaction: the per-experience lineage of the change —
        the experiences it came from, not the whole batch. The skill tools
        pass only ids the batch showed (the owner's, of its agent and scope);
        other owners' ids are never touched."""
        rows = (
            await db.execute(
                select(SkillExperienceModel).where(
                    SkillExperienceModel.id.in_(list(ids)),
                    SkillExperienceModel.user_id == owner_user_id,
                    ~SkillExperienceModel.is_deleted,
                )
            )
        ).scalars().all()
        stamped = 0
        for row in rows:
            cited = list(row.cited_skill_ids or [])
            if skill_id in cited:
                continue
            cited.append(skill_id)
            row.cited_skill_ids = cited
            row.updated_at = now
            stamped += 1
        await db.flush()
        return stamped

    # ------------------------------------------------------------------ #
    # Write entry points — all of them go through apply_skill_change      #
    # ------------------------------------------------------------------ #

    @enforce_types
    async def create_item(
        self,
        item_data: PydanticProceduralMemoryItem,
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
        agent_state: Optional[AgentState] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
    ) -> PydanticProceduralMemoryItem:
        """Create a new skill (version 1, ``create``) via :meth:`apply_skill_change`.

        Args:
            item_data: The skill to create. An explicit ``retrieval_embedding``
                is stored as given; otherwise the retrieval text is embedded
                (``agent_state``'s embedding config, else ``item_data``'s).
            actor: Client performing the operation (audit trail, organization).
            client_id: Client application identifier (defaults to actor.id).
            user_id: The owning user (defaults to ``item_data.user_id``).
            use_cache: Ignored — procedural memory is never cached.
        """
        del use_cache
        data = item_data.model_dump()
        data["client_id"] = client_id if client_id is not None else actor.id
        owner = user_id or data.get("user_id")
        logger.debug("create_item: client_id=%s, user_id=%s", data["client_id"], owner)
        result = await self.apply_skill_change(
            history.CHANGE_CREATE,
            actor=actor,
            user_id=owner,
            fields=data,
            agent_state=agent_state,
            run_id=run_id,
            actor_kind=actor_kind or ACTOR_API,
        )
        return result.skill

    @enforce_types
    async def update_item(
        self,
        item_update: ProceduralMemoryItemUpdate,
        user: PydanticUser,
        actor: PydanticClient,
        agent_state: Optional[AgentState] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> PydanticProceduralMemoryItem:
        """Edit a live skill via :meth:`apply_skill_change` (an ``edit`` version).

        When ``name``, ``triggers`` or ``description`` change, the retrieval
        vector is recomputed with ``agent_state``'s embedding config, or
        the skill's own when there is no agent_state. An explicit
        ``retrieval_embedding`` on ``item_update`` is stored as given.
        ``organization_id`` on the update is ignored: an edit never moves a
        skill to another tenant. Access: the actor's organization and ``user``
        must own the skill.
        """
        data = item_update.model_dump(exclude_unset=True)
        fields = {
            key: value
            for key, value in data.items()
            if key in SKILL_VERSIONED_FIELDS or key in _EDITABLE_META_FIELDS
        }
        # An explicit None is "not given" for every column that cannot hold it
        # (only filter_tags may be cleared).
        for key in list(fields):
            if fields[key] is None and key != "filter_tags":
                del fields[key]
        result = await self.apply_skill_change(
            history.CHANGE_EDIT,
            actor=actor,
            user_id=user.id,
            skill_id=item_update.id,
            fields=fields,
            agent_state=agent_state,
            run_id=run_id,
            actor_kind=actor_kind or ACTOR_API,
            reason=reason,
        )
        return result.skill

    @enforce_types
    async def create_many_items(
        self,
        items: List[PydanticProceduralMemoryItem],
        user: PydanticUser,
        actor: Optional[PydanticClient] = None,
    ) -> List[PydanticProceduralMemoryItem]:
        """Create several skills, each a ``create`` through
        :meth:`apply_skill_change` (one transaction per skill). ``actor`` is
        the creating client (its id becomes ``client_id``); the legacy call
        shape ``(items, user)`` — which used to pass the user as the actor and
        fail type checking — creates them without a client."""
        created = []
        for item in items:
            data = item.model_dump()
            if actor is not None and not data.get("client_id"):
                data["client_id"] = actor.id
            result = await self.apply_skill_change(
                history.CHANGE_CREATE,
                actor=actor,
                user_id=item.user_id or user.id,
                fields=data,
                organization_id=item.organization_id or user.organization_id,
                actor_kind=ACTOR_API,
            )
            created.append(result.skill)
        return created

    async def restore_skill(
        self,
        skill_id: str,
        version_no: int,
        *,
        actor: PydanticClient,
        user_id: str,
        agent_state: Optional[AgentState] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
        reason: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> SkillChangeResult:
        """Restore a skill to version ``version_no`` (a ``restore`` version);
        un-deletes a soft-deleted skill. :class:`SkillNameConflictError` when
        another live skill of the user has the restored name;
        :class:`~mirix.services.skill_history.SkillVersionNotFound` /
        :class:`NoResultFound` when the version / skill does not exist — or,
        with ``scopes``, when the skill or the version is of another scope
        (checked on the locked row)."""
        return await self.apply_skill_change(
            history.CHANGE_RESTORE,
            actor=actor,
            user_id=user_id,
            skill_id=skill_id,
            version_no=version_no,
            agent_state=agent_state,
            run_id=run_id,
            actor_kind=actor_kind or ACTOR_API,
            reason=reason,
            scopes=scopes,
        )

    # ------------------------------------------------------------------ #
    # History reads                                                       #
    # ------------------------------------------------------------------ #

    async def get_skill_history(
        self,
        skill_id: str,
        *,
        organization_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """A skill (live or soft-deleted) and all its versions, oldest first.

        Returns ``{skill, is_deleted, deleted_at, delete_reason, versions}``;
        each version carries ``changed_fields`` (versus its parent).
        :class:`NoResultFound` outside the organization / owner.
        """
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    self._skill_row_query(
                        skill_id,
                        organization_id=organization_id,
                        user_id=user_id,
                        include_deleted=True,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise NoResultFound(
                    f"Procedural memory item with id {skill_id} not found."
                )
            versions = await history.list_versions(db, row.id)
        by_id = {v.id: v for v in versions}
        entries = []
        for version in versions:
            parent = by_id.get(version.parent_version_id)
            entry = PydanticSkillVersion.model_validate(version).model_dump(mode="json")
            entry["changed_fields"] = history.changed_fields(
                history.version_hashes(parent) if parent is not None else None,
                history.version_hashes(version),
            )
            entries.append(entry)
        deleted_at = delete_reason = None
        if row.is_deleted:
            deletes = [v for v in versions if v.change_type == history.CHANGE_DELETE]
            if deletes:
                deleted_at = deletes[-1].created_at
                delete_reason = deletes[-1].reason
        return {
            "skill": row.to_pydantic(),
            "is_deleted": bool(row.is_deleted),
            "deleted_at": deleted_at,
            "delete_reason": delete_reason,
            "versions": entries,
        }

    async def get_skill_version(
        self,
        skill_id: str,
        version_no: int,
        *,
        organization_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Version ``version_no`` of a skill rebuilt from its objects:
        ``{version, content, canonical}`` — ``canonical`` holds each field's
        canonical bytes (what the sha256 keys cover)."""
        async with self.session_maker() as db:
            row = (
                await db.execute(
                    self._skill_row_query(
                        skill_id,
                        organization_id=organization_id,
                        user_id=user_id,
                        include_deleted=True,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise NoResultFound(
                    f"Procedural memory item with id {skill_id} not found."
                )
            version = await history.get_version(db, row.id, version_no)
            content, raws = await history.version_content(
                db, organization_id=_resolve_org(row.organization_id), version=version
            )
        return {
            "version": PydanticSkillVersion.model_validate(version),
            "content": content,
            "canonical": raws,
        }

    async def get_served_skill_texts(
        self,
        *,
        organization_id: Optional[str],
        user_id: str,
        served: Mapping[str, Optional[int]],
        scopes: Optional[List[str]],
        unversioned: Sequence[str] = (),
    ) -> Dict[str, Dict[str, Any]]:
        """The text each skill had in the version it was served at:
        ``{skill_id: {"name", "description", "instructions", "triggers",
        "examples", "version_no", "version_status", "exact", "is_deleted"}}``
        for the user's skills among ``served`` (``skill_id -> version_no``),
        deleted ones included, in ``scopes`` (None: any scope; [] none). A
        skill of another owner or scope is left out.

        ``version_status`` (:data:`distill_session_context.VERSION_STATUSES`):

        * ``exact`` — the served version, rebuilt from history. A skill in
          ``unversioned`` (served while it had no history, so logged with no
          version) that has a history now was served with its ``baseline``
          version 1 — the text its first recorded change saved (every change
          of a skill records a version) — so that is its served version;
        * ``unversioned`` — a skill in ``unversioned`` that still has no
          history: unchanged since, its current text is the text served;
        * ``unrebuilt`` — the served version does not rebuild
          (``version_no`` still names it): its current text;
        * ``unknown`` — no version was logged (a row written before versions
          were): its current text.

        ``exact`` is True when the text given is the text served."""
        from mirix.services.distill_session_context import (
            VERSION_EXACT,
            VERSION_UNKNOWN,
            VERSION_UNREBUILT,
            VERSION_UNVERSIONED,
        )

        found: Dict[str, Dict[str, Any]] = {}
        if not served:
            return found
        before_history = set(unversioned or ())
        async with self.session_maker() as db:
            for skill_id, version_no in served.items():
                row = (
                    await db.execute(
                        self._skill_row_query(
                            skill_id,
                            organization_id=organization_id,
                            user_id=user_id,
                            include_deleted=True,
                            scopes=scopes,
                        )
                    )
                ).scalar_one_or_none()
                if row is None:
                    continue
                content: Dict[str, Any] = history.content_of(row)
                status, shown = VERSION_UNKNOWN, None
                wanted: Optional[int] = None
                if version_no is not None:
                    status, shown = VERSION_UNREBUILT, int(version_no)
                    wanted = shown
                elif skill_id in before_history:
                    if await history.latest_version(db, row.id) is None:
                        status = VERSION_UNVERSIONED
                    else:
                        wanted = 1  # its baseline, if version 1 is one
                if wanted is not None:
                    try:
                        version = await history.get_version(db, row.id, wanted)
                        if version_no is None and (
                            version.change_type != history.CHANGE_BASELINE
                        ):
                            raise history.SkillVersionNotFound(
                                f"Skill {row.id} has no baseline version."
                            )
                        content, _raws = await history.version_content(
                            db,
                            organization_id=_resolve_org(row.organization_id),
                            version=version,
                        )
                        status, shown = VERSION_EXACT, wanted
                    except (history.SkillVersionNotFound, history.SkillObjectCorrupt):
                        logger.warning(
                            "Skill %s: served version %s does not rebuild; its "
                            "current text is shown instead",
                            row.id,
                            wanted,
                        )
                found[row.id] = {
                    "name": content.get("name") or row.name,
                    "description": content.get("description") or "",
                    "instructions": content.get("instructions") or "",
                    "triggers": list(content.get("triggers") or []),
                    "examples": list(content.get("examples") or []),
                    "version_no": shown,
                    "version_status": status,
                    "exact": status in (VERSION_EXACT, VERSION_UNVERSIONED),
                    "is_deleted": bool(row.is_deleted),
                }
        return found

    # ------------------------------------------------------------------ #
    # Baselines (scripts/backfill_skill_baseline_versions.py)             #
    # ------------------------------------------------------------------ #

    async def ensure_baseline_version(
        self, skill_id: str, *, actor_kind: str = ACTOR_BACKFILL
    ) -> Optional[PydanticSkillVersion]:
        """Write the ``baseline`` version of a skill that has no history yet
        (live or soft-deleted); None when it already has one or is gone.
        Idempotent: runs under the same locks as every write."""

        async def _once():
            async with self.session_maker() as db:
                async with db.begin():
                    org = (
                        await db.execute(
                            select(ProceduralMemoryItem.organization_id).where(
                                ProceduralMemoryItem.id == skill_id
                            )
                        )
                    ).first()
                    if org is None:
                        return None
                    organization_id = _resolve_org(org[0])
                    await history.lock_for_write(db, organization_ids=[organization_id])
                    row = (
                        await db.execute(
                            self._skill_row_query(
                                skill_id,
                                organization_id=None,
                                user_id=None,
                                include_deleted=True,
                            ).with_for_update()
                        )
                    ).scalar_one_or_none()
                    if row is None or await history.latest_version(db, row.id):
                        return None
                    now = _utcnow()
                    encoded = history.encode_content(history.content_of(row))
                    await history.upsert_objects(
                        db, organization_id=organization_id, encoded=encoded, now=now
                    )
                    version = await history.insert_version(
                        db,
                        skill_id=row.id,
                        organization_id=organization_id,
                        user_id=row.user_id,
                        version_no=1,
                        parent_version_id=None,
                        change_type=history.CHANGE_BASELINE,
                        hashes=history.hashes_of(encoded),
                        now=now,
                        actor_kind=actor_kind,
                    )
                    return _version_schema(version)

        return await retry_call(_once, stage="skill_baseline")

    # ------------------------------------------------------------------ #
    # Missing retrieval vectors (startup task, re-embed script)            #
    # ------------------------------------------------------------------ #

    async def list_skill_ids_missing_retrieval_embedding(
        self,
        *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        """Ids of the live skills whose ``retrieval_embedding`` IS NULL (after
        the migration, or an embedding call that failed for good), in id
        order after ``after_id`` — of one organization and / or one user when
        given: pages for :meth:`reembed_skill`. Until re-embedded such a skill
        fails every vector gate."""
        stmt = select(ProceduralMemoryItem.id).where(
            ProceduralMemoryItem.retrieval_embedding.is_(None),
            ~ProceduralMemoryItem.is_deleted,
        )
        if organization_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.organization_id == organization_id)
        if user_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.user_id == user_id)
        if after_id is not None:
            stmt = stmt.where(ProceduralMemoryItem.id > after_id)
        stmt = stmt.order_by(ProceduralMemoryItem.id).limit(max(1, int(limit)))
        async with self.session_maker() as db:
            return [str(skill_id) for skill_id in (await db.execute(stmt)).scalars()]

    async def reembed_skill(self, skill_id: str) -> str:
        """Compute the retrieval vector of a live skill that has none.

        The vector of the skill's retrieval text (name, triggers,
        description), with the row's OWN ``embedding_config`` (there is no
        agent here), written through :meth:`apply_skill_change` as an explicit
        vector under the optimistic check that the retrieval text is still the
        one embedded — a concurrent change wins and brings its own vector. No
        version is written (a vector is not part of the history) and the
        row's modification stamp is left alone. Safe to repeat. Returns what
        happened: ``filled``; ``present`` (it has a vector now); ``gone``
        (deleted meanwhile); ``changed`` (its retrieval text changed
        meanwhile); ``no_config`` (the row has no embedding config);
        ``failed`` (the embedding call failed after its retries; retried on a
        later pass); ``disabled`` (``BUILD_EMBEDDINGS_FOR_MEMORY`` is off).

        One call per skill at a time in this process: while another pass is
        re-embedding the same skill, this one awaits and returns that outcome
        (no second provider call); if that pass is cancelled or fails
        unexpectedly, this one does the work itself.
        """
        if not BUILD_EMBEDDINGS_FOR_MEMORY:
            return REEMBED_DISABLED
        loop = asyncio.get_running_loop()
        while True:
            running = _REEMBED_IN_FLIGHT.get(skill_id)
            if running is None or running.get_loop() is not loop:
                break
            # shield: cancelling THIS caller must not cancel the other pass.
            shared = await asyncio.shield(running)
            if shared is not None:
                return shared
            # The other pass was cancelled or failed: take the skill over.
        future: "asyncio.Future[Optional[str]]" = loop.create_future()
        _REEMBED_IN_FLIGHT[skill_id] = future
        outcome: Optional[str] = None
        try:
            outcome = await self._reembed_skill_once(skill_id)
            return outcome
        finally:
            # None tells a waiter to do the work itself (this call raised or
            # was cancelled).
            if not future.done():
                future.set_result(outcome)
            if _REEMBED_IN_FLIGHT.get(skill_id) is future:
                del _REEMBED_IN_FLIGHT[skill_id]

    async def _reembed_skill_once(self, skill_id: str) -> str:
        """:meth:`reembed_skill`'s work, for the one caller in flight."""
        try:
            row = await self._read_skill_row(
                skill_id, organization_id=None, user_id=None, include_deleted=False
            )
        except NoResultFound:
            return REEMBED_GONE
        if row.retrieval_embedding is not None:
            return REEMBED_PRESENT
        if row.embedding_config is None:
            return REEMBED_NO_CONFIG
        expected = _row_values(row, list(_RETRIEVAL_TEXT_FIELDS))
        vector = await self._embed(history.skill_retrieval_text(expected), row.embedding_config)
        if vector is None:
            return REEMBED_FAILED
        try:
            await self.apply_skill_change(
                history.CHANGE_EDIT,
                actor=None,
                user_id=None,
                skill_id=skill_id,
                fields={_RETRIEVAL_EMBEDDING_COLUMN: vector},
                expected_fields=expected,
                actor_kind=ACTOR_SYSTEM,
            )
        except SkillChangedMeanwhile:
            return REEMBED_CHANGED
        except NoResultFound:
            return REEMBED_GONE
        return REEMBED_FILLED

    async def get_total_number_of_items(self, user: PydanticUser) -> int:
        """Get the total number of items in the procedural memory for the user."""
        async with self.session_maker() as session:
            # Soft-deleted skills are not counted.
            query = select(func.count(ProceduralMemoryItem.id)).where(
                ProceduralMemoryItem.user_id == user.id,
                ~ProceduralMemoryItem.is_deleted,
            )
            result = await session.execute(query)
            return result.scalar_one()

    async def _embed_query(self, query: str, config) -> Optional[List[float]]:
        """The padded query vector for skill retrieval, or None when embeddings
        are off or there is no embedding config. A failure after the
        embedding retries raises: the ranking then falls back to BM25."""
        if not BUILD_EMBEDDINGS_FOR_MEMORY or config is None:
            return None

        async def _call():
            model = await embedding_model(config)
            return await model.get_text_embedding(query)

        vector = await retry_call(
            _call,
            operation="embedding",
            stage="skill_retrieval_embedding",
            sleep=self._sleep,
        )
        return _pad_to_max(vector)

    async def retrieve_skills(
        self,
        query: str,
        user: PydanticUser,
        scopes: Optional[Sequence[str]],
        limit: Optional[int] = None,
        apply_gate: bool = False,
        *,
        agent_state: Optional[AgentState] = None,
        embedding_config=None,
        query_embedding: Optional[Sequence[float]] = None,
        filter_tags: Optional[Mapping[str, Any]] = None,
        method: str = "hybrid",
        max_cosine_distance: Optional[float] = None,
        timezone_str: Optional[str] = None,
    ) -> "SkillRetrievalResult":
        """Rank ``user``'s live skills in ``scopes`` for ``query``.

        Candidates straight from the database, true BM25 over
        ``retrieval_tsv`` (Postgres) or Python tokens (SQLite) with N, df and
        avgdl over all candidates, cosine on ``retrieval_embedding``, RRF
        (k = 60), the similarity gate when ``apply_gate`` (``/memory/retrieve/
        conversation``; never ``/memory/search``), then the first
        ``limit`` (default 10, possibly none). See
        :func:`mirix.services.skill_retrieval.retrieve_skills` for every
        argument.

        The query is embedded with ``embedding_config`` (default: the
        ``agent_state``'s), unless ``query_embedding`` is given; when that
        fails the ranking is BM25 alone, ungated
        (``diagnostics.embedding_status == "failed"``). ``timezone_str``
        converts the returned skills' timestamps like :meth:`list_procedures`.
        A gated ranking uses the gate of that embedding config's model
        (:func:`~mirix.services.skill_retrieval.skill_retrieval_gate`: its
        calibration, or the defaults as ``uncalibrated``), recorded in the
        diagnostics' gate trace.

        Returns a :class:`~mirix.services.skill_retrieval.SkillRetrievalResult`:
        the skills with their ``retrieval_scores`` ``{bm25, cosine, rrf}``, and
        the diagnostics the retrieval log records.
        """
        from mirix.services import skill_retrieval

        config = embedding_config
        if config is None and agent_state is not None:
            config = agent_state.embedding_config

        async def embed(text_value: str) -> Optional[List[float]]:
            return await self._embed_query(text_value, config)

        gate: Dict[str, Any] = {}
        model = getattr(config, "embedding_model", None) if config is not None else None
        if apply_gate and model:
            settings = skill_retrieval.skill_retrieval_gate(model)
            gate = {
                "min_similarity": settings.min_similarity,
                "relative_ratio": settings.relative_ratio,
                "gate_calibration": settings.source,
            }

        result = await skill_retrieval.retrieve_skills(
            query,
            user,
            scopes,
            limit,
            apply_gate,
            session_maker=self.session_maker,
            embed_query=embed,
            query_embedding=query_embedding,
            filter_tags=filter_tags,
            method=method,
            max_cosine_distance=max_cosine_distance,
            **gate,
        )
        localize_timestamps(result.items, timezone_str)
        return result

    async def _hybrid_search_for_org(
        self,
        *,
        agent_state: AgentState,
        organization_id: str,
        query: str,
        embedded_text: Optional[List[float]],
        search_field: str,
        limit: Optional[int],
        timezone_str: Optional[str],
        filter_tags: Optional[dict],
        scopes: Optional[List[str]],
        use_cache: bool,
        similarity_threshold: Optional[float],
    ) -> List[PydanticProceduralMemoryItem]:
        """Org-wide counterpart of :meth:`_hybrid_search_for_user`.

        Same RRF fusion, but each lane reuses :meth:`list_procedures_by_org`.
        """
        effective_limit = limit or 50
        recall_limit = max(
            effective_limit * SKILL_HYBRID_RECALL_MULTIPLIER, effective_limit
        )
        sparse = await self.list_procedures_by_org(
            agent_state=agent_state,
            organization_id=organization_id,
            query=query,
            search_field=search_field,
            search_method="bm25",
            limit=recall_limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            scopes=scopes,
            use_cache=use_cache,
        )
        dense_field = (
            search_field if search_field in _EMBEDDING_FIELDS else "description"
        )
        dense = await self.list_procedures_by_org(
            agent_state=agent_state,
            organization_id=organization_id,
            query=query,
            embedded_text=embedded_text,
            search_field=dense_field,
            search_method="embedding",
            limit=recall_limit,
            timezone_str=timezone_str,
            filter_tags=filter_tags,
            scopes=scopes,
            use_cache=use_cache,
            similarity_threshold=similarity_threshold,
        )
        return _rrf_fuse(
            [sparse, dense], k=SKILL_HYBRID_RRF_K, limit=effective_limit
        )

    @update_timezone
    @enforce_types
    async def list_procedures(
        self,
        agent_state: AgentState,
        user: PydanticUser,
        query: str = "",
        embedded_text: Optional[List[float]] = None,
        search_field: str = "",
        search_method: str = "embedding",
        limit: Optional[int] = 50,
        timezone_str: str = None,
        filter_tags: Optional[dict] = None,
        scopes: Optional[List[str]] = None,
        use_cache: bool = True,
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticProceduralMemoryItem]:
        """
        List procedural memory items with various search methods.

        Args:
            agent_state: The agent state containing embedding configuration
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('description', 'instructions', 'entry_type')
            search_method: Search method to use:
                - 'hybrid': **RECOMMENDED for skill retrieval** - the skill
                               ranking (:meth:`retrieve_skills`, ungated): true
                               BM25 over the retrieval text (name, triggers,
                               description) and cosine on the retrieval vector,
                               fused with Reciprocal Rank Fusion (k=60).
                               Self-embeds the query, so no embedded_text need
                               be supplied; on `instructions` it runs as 'bm25'.
                - 'embedding': Vector similarity search using embeddings
                - 'string_match': Simple string containment search
                - 'bm25': PostgreSQL native full-text search (ts_rank_cd) when using PostgreSQL,
                               falls back to in-memory BM25 for SQLite
                - 'fuzzy_match': Fuzzy string matching (legacy, kept for compatibility)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            use_cache: Ignored — procedural memory is read from the database only.

        Returns:
            List of procedural memory items matching the search criteria

        Note:
            **For PostgreSQL users**: 'bm25' is now the recommended method for text-based searches as it uses
            PostgreSQL's native full-text search with ts_rank_cd for BM25-like scoring. This is much more efficient
            than loading all documents into memory and leverages your existing GIN indexes.

            **For SQLite users**: 'bm25' now has fallback support that uses in-memory BM25 processing.

            Performance comparison:
            - PostgreSQL 'bm25': Native DB search, very fast, scales well
            - Fallback 'bm25' (SQLite): In-memory processing, slower for large datasets but still provides
              proper BM25 ranking
        """
        query = query.strip() if query else ""
        is_empty_query = not query or query == ""

        # Pad any caller-supplied embedding to MAX_EMBEDDING_DIM at the point of
        # CONSUMPTION (not just where we recompute it), so a raw provider vector
        # (e.g. 3072-dim) passed in by REST callers never reaches the Vector(4096)
        # PG column or the DIM=4096 Redis index unpadded.
        embedded_text = _pad_to_max(embedded_text)

        # `instructions` has no vector; an embedding search on it runs as
        # full-text search on `instructions` (see effective_procedural_search_method).
        search_method = effective_procedural_search_method(search_method, search_field)

        # Extract organization_id from user for multi-tenant isolation
        organization_id = user.organization_id

        # Hybrid retrieval is the M7.3 ranking over the retrieval text (BM25 +
        # cosine fused with RRF, never gated here; `instructions` was rerouted
        # to keyword search above). Empty queries fall through to the
        # recent-items path, where the method is moot.
        if search_method == "hybrid" and not is_empty_query:
            ranked = await self.retrieve_skills(
                query,
                user,
                scopes,
                limit or 50,
                apply_gate=False,
                agent_state=agent_state,
                query_embedding=embedded_text,
                filter_tags=filter_tags,
                max_cosine_distance=similarity_threshold,
            )
            return ranked.items

        async with self.session_maker() as session:
            if query == "":
                query_stmt = (
                    select(ProceduralMemoryItem)
                    .where(ProceduralMemoryItem.user_id == user.id)
                    .where(ProceduralMemoryItem.organization_id == organization_id)
                    # Exclude soft-deleted skills (curator soft-delete sets
                    # is_deleted=True; they must NOT surface in retrieval or the
                    # evolve before/after snapshots). The base SqlalchemyBase
                    # reads filter ~is_deleted, but these raw fallback queries
                    # build select() directly and would otherwise leak them.
                    .where(~ProceduralMemoryItem.is_deleted)
                    .order_by(ProceduralMemoryItem.created_at.desc())
                )

                from mirix.database.filter_tags_query import (
                    apply_filter_tags_sqlalchemy,
                )

                query_stmt = apply_filter_tags_sqlalchemy(
                    query_stmt, ProceduralMemoryItem, filter_tags, scopes=scopes
                )

                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = await session.execute(query_stmt)
                procedural_memory = result.scalars().all()
                return [event.to_pydantic() for event in procedural_memory]

            else:
                # Select full ORM entities so every schema field (triggers,
                # examples, version, filter_tags, updated_at, client_id, ...)
                # survives the round-trip. Projecting a subset and re-hydrating
                # silently loses columns.
                base_query = (
                    select(ProceduralMemoryItem)
                    .where(ProceduralMemoryItem.user_id == user.id)
                    .where(ProceduralMemoryItem.organization_id == organization_id)
                    # Exclude soft-deleted skills from every search method too
                    # (same reason as the empty-query branch above).
                    .where(~ProceduralMemoryItem.is_deleted)
                )

                from mirix.database.filter_tags_query import (
                    apply_filter_tags_sqlalchemy,
                )

                base_query = apply_filter_tags_sqlalchemy(
                    base_query, ProceduralMemoryItem, filter_tags, scopes=scopes
                )

                if search_method == "embedding":
                    embedding_column = _resolve_embedding_column(search_field)
                    if embedding_column is None:
                        raise ValueError(
                            f"Invalid search_field '{search_field}' for embedding search. "
                            f"Allowed: {sorted(_EMBEDDING_FIELDS)}."
                        )
                    main_query = await build_query(
                        base_query=base_query,
                        query_text=query,
                        embedded_text=embedded_text,
                        embed_query=True,
                        embedding_config=agent_state.embedding_config,
                        search_field=embedding_column,
                        target_class=ProceduralMemoryItem,
                        similarity_threshold=similarity_threshold,
                    )

                elif search_method == "string_match":
                    search_field_obj = _resolve_text_column(search_field)
                    if search_field_obj is None:
                        raise ValueError(
                            f"Invalid search_field '{search_field}' for string_match. "
                            f"Allowed: {sorted(_SEARCHABLE_TEXT_FIELDS)}."
                        )
                    main_query = base_query.where(
                        func.lower(search_field_obj).contains(query.lower())
                    )

                elif search_method == "bm25":
                    # Check if we're using PostgreSQL - use native full-text search if available
                    if settings.mirix_pg_uri_no_default:
                        # Use PostgreSQL native full-text search
                        return await self._postgresql_fulltext_search(
                            session,
                            base_query,
                            query,
                            search_field,
                            limit,
                            user,
                            filter_tags=filter_tags,
                            scopes=scopes,
                        )
                    else:
                        # Fallback to in-memory BM25 for SQLite (legacy method)
                        # Load all candidate items (memory-intensive, kept for
                        # compatibility). Mirror base_query's full WHERE set
                        # (org + filter_tags + scopes) so the SQLite fallback
                        # enforces the SAME scope filtering as the PG/other
                        # paths — otherwise scoped reads leak NULL-scope rows.
                        candidate_query = (
                            select(ProceduralMemoryItem)
                            .where(ProceduralMemoryItem.user_id == user.id)
                            .where(
                                ProceduralMemoryItem.organization_id
                                == organization_id
                            )
                            # Exclude soft-deleted skills from the SQLite BM25
                            # fallback too (this is the path the validity run's
                            # before/after evolve snapshot uses).
                            .where(~ProceduralMemoryItem.is_deleted)
                        )
                        candidate_query = apply_filter_tags_sqlalchemy(
                            candidate_query,
                            ProceduralMemoryItem,
                            filter_tags,
                            scopes=scopes,
                        )
                        result = await session.execute(candidate_query)
                        all_items = result.scalars().all()

                        if not all_items:
                            return []

                        # Prepare documents for BM25
                        documents = []
                        valid_items = []

                        for item in all_items:
                            # Determine which field to use for search
                            if search_field == "description":
                                text_to_search = item.description or ""
                            elif search_field == "instructions":
                                text_to_search = item.instructions or ""
                            elif search_field == "entry_type":
                                text_to_search = item.entry_type or ""
                            else:
                                # Default to searching across all text fields
                                texts = [
                                    item.description or "",
                                    item.entry_type or "",
                                    item.instructions or "",
                                ]
                                text_to_search = " ".join(texts)

                            # Preprocess the text into tokens
                            tokens = self._preprocess_text_for_bm25(text_to_search)

                            # Only include items that have tokens after preprocessing
                            if tokens:
                                documents.append(tokens)
                                valid_items.append(item)

                        if not documents:
                            return []

                        # Initialize BM25 with the documents
                        bm25 = BM25Okapi(documents)

                        # Preprocess the query
                        query_tokens = self._preprocess_text_for_bm25(query)

                        if not query_tokens:
                            # If query has no valid tokens, return most recent items
                            return [item.to_pydantic() for item in valid_items[:limit]]

                        # Get BM25 scores for all documents
                        scores = bm25.get_scores(query_tokens)

                        # Create scored items list
                        scored_items = list(zip(scores, valid_items))

                        # Sort by BM25 score in descending order
                        scored_items.sort(key=lambda x: x[0], reverse=True)

                        # Get top items based on limit
                        top_items = [item for score, item in scored_items[:limit]]

                        # Return the list after converting to Pydantic
                        return [item.to_pydantic() for item in top_items]

                elif search_method == "fuzzy_match":
                    # For fuzzy matching, load all candidate items into memory.
                    # Mirror base_query's full WHERE set (org + filter_tags +
                    # scopes) so fuzzy reads enforce the SAME scope filtering.
                    candidate_query = (
                        select(ProceduralMemoryItem)
                        .where(ProceduralMemoryItem.user_id == user.id)
                        .where(
                            ProceduralMemoryItem.organization_id == organization_id
                        )
                        # Exclude soft-deleted skills from the fuzzy path too.
                        .where(~ProceduralMemoryItem.is_deleted)
                    )
                    candidate_query = apply_filter_tags_sqlalchemy(
                        candidate_query,
                        ProceduralMemoryItem,
                        filter_tags,
                        scopes=scopes,
                    )
                    result = await session.execute(candidate_query)
                    all_items = result.scalars().all()
                    scored_items = []
                    for item in all_items:
                        # Use the provided search_field if available; default to 'description'
                        if search_field and hasattr(item, search_field):
                            text_to_search = getattr(item, search_field)
                        else:
                            text_to_search = item.description

                        # Compute a fuzzy matching score using partial_ratio,
                        # which is suited for comparing a short query to longer text.
                        score = fuzz.partial_ratio(
                            query.lower(), text_to_search.lower()
                        )
                        scored_items.append((score, item))

                    # Sort items by score in descending order and select the top ones.
                    scored_items.sort(key=lambda x: x[0], reverse=True)
                    top_items = [item for score, item in scored_items[:limit]]
                    return [item.to_pydantic() for item in top_items]

                if limit:
                    main_query = main_query.limit(limit)

                result = await session.execute(main_query)
                procedures = result.scalars().all()

                return [procedure.to_pydantic() for procedure in procedures]

    @enforce_types
    async def insert_procedure(
        self,
        agent_state: AgentState,
        agent_id: str,
        name: str,
        description: str,
        instructions: str,
        entry_type: str,
        actor: PydanticClient,
        organization_id: str,
        triggers: Optional[List[str]] = None,
        examples: Optional[List[dict]] = None,
        version: str = "0.1.0",
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
    ) -> PydanticProceduralMemoryItem:
        """Create a skill from its fields (the skill tools' create path).

        The retrieval vector (name + triggers + description) is computed
        with ``agent_state``'s embedding config by :meth:`apply_skill_change`.
        ``user_id`` defaults to the admin user; ``client_id`` is the actor's.
        ``use_cache`` is ignored.
        """
        del use_cache
        from mirix.services.user_manager import UserManager

        if user_id is None:
            user_id = UserManager.ADMIN_USER_ID
        result = await self.apply_skill_change(
            history.CHANGE_CREATE,
            actor=actor,
            user_id=user_id,
            fields={
                "name": name,
                "entry_type": entry_type,
                "description": description,
                "instructions": instructions,
                "triggers": triggers or [],
                "examples": examples or [],
                "version": version,
                "agent_id": agent_id,
                "client_id": actor.id,
                "organization_id": organization_id,
                "filter_tags": filter_tags,
            },
            agent_state=agent_state,
            run_id=run_id,
            actor_kind=actor_kind or ACTOR_AGENT,
        )
        return result.skill

    async def soft_delete_procedure_by_id(
        self,
        procedure_id: str,
        actor: PydanticClient,
        user: Optional[PydanticUser] = None,
        reason: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_kind: Optional[str] = None,
    ) -> SkillChangeResult:
        """Soft-delete a skill: ``is_deleted`` plus a ``delete`` version
        carrying ``reason``, via :meth:`apply_skill_change`. It leaves retrieval
        and every read, its name becomes free again, and it can be
        restored from its history. Access: the actor's organization, and
        ``user`` (when given) must own it."""
        return await self.apply_skill_change(
            history.CHANGE_DELETE,
            actor=actor,
            user_id=user.id if user is not None else None,
            skill_id=procedure_id,
            run_id=run_id,
            actor_kind=actor_kind or ACTOR_API,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Hard delete = erasure: the skill, its whole history, and the        #
    # objects no other version references, in one transaction under the   #
    # EXCLUSIVE org lock.                                                  #
    # ------------------------------------------------------------------ #

    async def delete_procedure_by_id(
        self,
        procedure_id: str,
        actor: PydanticClient,
        user: Optional[PydanticUser] = None,
        user_id: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> None:
        """Erase a skill (live or soft-deleted) with its whole version history:
        the skill row, every version, and the objects no other version
        of the organization references — one transaction, under the exclusive
        org advisory lock the object collector needs.

        Ownership: ``actor`` → the skill must be in its organization;
        ``user`` / ``user_id`` → the skill must belong to that user; ``scopes``
        → its ``filter_tags.scope`` must be one of them (checked in the
        purge transaction itself). A skill outside any of them is
        :class:`NoResultFound` (existence is never leaked).
        """
        org = _resolve_org(actor.organization_id)
        owner = user.id if user is not None else user_id
        deleted = await self._purge_skills(
            filters=[
                ProceduralMemoryItem.id == procedure_id,
                ProceduralMemoryItem.organization_id == org,
                *([ProceduralMemoryItem.user_id == owner] if owner is not None else []),
            ],
            scopes=scopes,
        )
        if not deleted:
            raise NoResultFound(
                f"Procedural memory item with id {procedure_id} not found."
            )

    @enforce_types
    async def delete_by_client_id(self, actor: PydanticClient) -> int:
        """Erase every skill a client created, with their histories.

        Returns the number of skills deleted.
        """
        return await self._purge_skills(
            filters=[ProceduralMemoryItem.client_id == actor.id]
        )

    async def delete_by_user_id(self, user_id: str) -> int:
        """Erase every skill of a user and the user's whole skill history.

        Returns the number of skills deleted.
        """
        return await self._purge_skills(
            filters=[ProceduralMemoryItem.user_id == user_id],
            version_user_id=user_id,
        )

    async def _purge_skills(
        self,
        *,
        filters: List[Any],
        version_user_id: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> int:
        """Hard-delete the skills matching ``filters`` (and ``scopes``)
        and, with ``version_user_id``, every version of that user, plus the
        objects left unreferenced. Returns the number of skill rows deleted.

        The organizations to lock come from a first, unlocked read; the rows
        and versions are then read again under their exclusive locks. When the
        second read reaches an organization the first did not (a skill written
        meanwhile in another organization), the transaction is rolled back
        and repeated with that organization locked too: objects are never
        collected in an organization whose exclusive lock is not held.
        """

        def _targets():
            stmt = select(
                ProceduralMemoryItem.id, ProceduralMemoryItem.organization_id
            ).where(*filters)
            if scopes is not None:
                from mirix.database.filter_tags_query import (
                    apply_filter_tags_sqlalchemy,
                )

                stmt = apply_filter_tags_sqlalchemy(
                    stmt, ProceduralMemoryItem, None, scopes=list(scopes)
                )
            return stmt

        async def _attempt(extra_orgs: set) -> int:
            async with self.session_maker() as db:
                async with db.begin():
                    rows = (await db.execute(_targets())).all()
                    orgs = {_resolve_org(org) for _, org in rows} | extra_orgs
                    if version_user_id is not None:
                        orgs |= {
                            _resolve_org(org)
                            for (org,) in (
                                await db.execute(
                                    select(SkillVersionModel.organization_id)
                                    .where(SkillVersionModel.user_id == version_user_id)
                                    .distinct()
                                )
                            ).all()
                        }
                    if not rows and not orgs:
                        return 0
                    await history.lock_for_purge(db, organization_ids=orgs)
                    # Re-read under the lock: the set may have changed.
                    rows = (await db.execute(_targets())).all()
                    skill_ids = sorted({skill_id for skill_id, _ in rows})
                    version_filter = []
                    if skill_ids:
                        version_filter.append(SkillVersionModel.skill_id.in_(skill_ids))
                    if version_user_id is not None:
                        version_filter.append(SkillVersionModel.user_id == version_user_id)
                    candidates: Dict[str, set] = {}
                    versions = []
                    if version_filter:
                        versions = (
                            await db.execute(
                                select(SkillVersionModel).where(or_(*version_filter))
                            )
                        ).scalars().all()
                        for version in versions:
                            candidates.setdefault(
                                _resolve_org(version.organization_id), set()
                            ).update(history.version_hashes(version).values())
                    unlocked = (
                        {_resolve_org(org) for _, org in rows} | set(candidates)
                    ) - orgs
                    if unlocked:
                        raise _PurgeReachedUnlockedOrgs(unlocked)
                    if versions:
                        # Explicit, not only through ON DELETE CASCADE: the
                        # filter also reaches a user's versions whatever skill
                        # row they hang off, and a database whose foreign key
                        # lacks the cascade still ends up without history.
                        await db.execute(
                            delete(SkillVersionModel)
                            .where(or_(*version_filter))
                            .execution_options(synchronize_session=False)
                        )
                    if skill_ids:
                        await db.execute(
                            delete(ProceduralMemoryItem)
                            .where(ProceduralMemoryItem.id.in_(skill_ids))
                            .execution_options(synchronize_session=False)
                        )
                    for organization_id, shas in sorted(candidates.items()):
                        await history.purge_unreferenced_objects(
                            db, organization_id=organization_id, candidates=shas
                        )
                    return len(skill_ids)

        async def _once() -> int:
            extra: set = set()
            for _ in range(_MAX_PURGE_LOCK_ROUNDS):
                try:
                    return await _attempt(extra)
                except _PurgeReachedUnlockedOrgs as grew:
                    logger.info(
                        "Skill purge reached organization(s) %s it had not "
                        "locked; repeating with them locked",
                        sorted(grew.organization_ids),
                    )
                    extra |= grew.organization_ids
            raise RuntimeError(
                "skill purge: the set of organizations kept growing under the lock"
            )

        return await retry_call(_once, stage="skill_purge")

    async def purge_organization_history(self, db, organization_id: str) -> None:
        """In ``db``'s open transaction: delete every skill version and object
        of an organization, under its exclusive org lock — the organization
        itself is about to be deleted (its skills go with it)."""
        await history.lock_for_purge(db, organization_ids=[organization_id])
        skill_ids = select(ProceduralMemoryItem.id).where(
            ProceduralMemoryItem.organization_id == organization_id
        )
        await db.execute(
            delete(SkillVersionModel)
            .where(
                or_(
                    SkillVersionModel.organization_id == organization_id,
                    SkillVersionModel.skill_id.in_(skill_ids),
                )
            )
            .execution_options(synchronize_session=False)
        )
        await db.execute(
            delete(SkillObjectModel)
            .where(SkillObjectModel.organization_id == organization_id)
            .execution_options(synchronize_session=False)
        )

    # ------------------------------------------------------------------ #
    # Bulk soft deletes (user / client soft deletion)                     #
    # ------------------------------------------------------------------ #

    async def soft_delete_by_client_id(self, actor: PydanticClient) -> int:
        """Soft-delete every live skill a client created — each one a ``delete``
        change through :meth:`apply_skill_change`. Returns how many."""
        return await self._soft_delete_all(
            filters=[ProceduralMemoryItem.client_id == actor.id],
            actor=actor,
            reason="client deleted",
        )

    async def soft_delete_by_user_id(self, user_id: str) -> int:
        """Soft-delete every live skill of a user — each one a ``delete``
        change through :meth:`apply_skill_change`. Returns how many."""
        return await self._soft_delete_all(
            filters=[ProceduralMemoryItem.user_id == user_id],
            actor=None,
            reason="user deleted",
        )

    async def _soft_delete_all(
        self, *, filters: List[Any], actor: Optional[PydanticClient], reason: str
    ) -> int:
        async with self.session_maker() as db:
            skill_ids = (
                await db.execute(
                    select(ProceduralMemoryItem.id)
                    .where(*filters, ~ProceduralMemoryItem.is_deleted)
                    .order_by(ProceduralMemoryItem.id)
                )
            ).scalars().all()
        count = 0
        for skill_id in skill_ids:
            try:
                await self.apply_skill_change(
                    history.CHANGE_DELETE,
                    actor=None,
                    user_id=None,
                    skill_id=skill_id,
                    actor_kind=ACTOR_SYSTEM,
                    reason=reason,
                )
            except NoResultFound:
                continue  # deleted meanwhile
            count += 1
        return count

    @update_timezone
    @enforce_types
    async def list_procedures_by_org(
        self,
        agent_state: AgentState,
        organization_id: str,
        query: str = "",
        embedded_text: Optional[List[float]] = None,
        search_field: str = "",
        search_method: str = "embedding",
        limit: Optional[int] = 50,
        timezone_str: str = None,
        filter_tags: Optional[dict] = None,
        scopes: Optional[List[str]] = None,
        use_cache: bool = True,
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticProceduralMemoryItem]:
        """
        List procedural memories across ALL users in an organization.
        """
        # Pad any caller-supplied embedding to MAX_EMBEDDING_DIM at the point of
        # CONSUMPTION (REST callers may pass a raw 3072-dim vector) so it never
        # reaches the Vector(4096) PG column or DIM=4096 Redis index unpadded.
        embedded_text = _pad_to_max(embedded_text)

        # An embedding search on `instructions` runs as full-text search.
        search_method = effective_procedural_search_method(search_method, search_field)

        # Hybrid retrieval (EverOS-aligned) intercepted at the TOP. See
        # _hybrid_search_for_org. Empty queries fall through to recent-items.
        if search_method == "hybrid" and query and query.strip():
            return await self._hybrid_search_for_org(
                agent_state=agent_state,
                organization_id=organization_id,
                query=query,
                embedded_text=embedded_text,
                search_field=search_field,
                limit=limit,
                timezone_str=timezone_str,
                filter_tags=filter_tags,
                scopes=scopes,
                use_cache=use_cache,
                similarity_threshold=similarity_threshold,
            )

        async with self.session_maker() as session:
            base_query = (
                select(ProceduralMemoryItem)
                .where(ProceduralMemoryItem.organization_id == organization_id)
                # Exclude soft-deleted skills (curator soft-delete sets
                # is_deleted=True; they must NOT surface via org-wide retrieval).
                # The base SqlalchemyBase reads filter ~is_deleted, but this raw
                # fallback query builds select() directly and would otherwise
                # leak them. Mirrors list_procedures' ~is_deleted predicate.
                .where(~ProceduralMemoryItem.is_deleted)
            )

            from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

            base_query = apply_filter_tags_sqlalchemy(
                base_query, ProceduralMemoryItem, filter_tags, scopes=scopes
            )

            # Handle empty query - fall back to recent sort
            if not query or query == "":
                base_query = base_query.order_by(ProceduralMemoryItem.created_at.desc())
                if limit:
                    base_query = base_query.limit(limit)
                result = await session.execute(base_query)
                items = result.scalars().all()
                return [item.to_pydantic() for item in items]

            # Embedding search
            if search_method == "embedding":
                embedding_config = agent_state.embedding_config
                if embedded_text is None:
                    # Pad to MAX_EMBEDDING_DIM so a raw (e.g. 3072-dim) query
                    # vector matches the Vector(4096) column. (Caller-supplied
                    # embeddings are already padded at function entry above.)
                    embedded_text = _pad_to_max(
                        await (
                            await embedding_model(embedding_config)
                        ).get_text_embedding(query)
                    )

                # The single retrieval vector (instructions searches were
                # already rerouted to full-text at the top).
                embedding_field = ProceduralMemoryItem.retrieval_embedding

                embedding_query_field = embedding_field.cosine_distance(
                    embedded_text
                ).label("distance")
                base_query = base_query.add_columns(embedding_query_field)

                # Apply similarity threshold if provided
                if similarity_threshold is not None:
                    base_query = base_query.where(
                        embedding_query_field < similarity_threshold
                    )

                base_query = base_query.order_by(embedding_query_field)

            # BM25 search
            elif search_method == "bm25":
                from sqlalchemy import func

                # Determine search field
                if search_field == "description":
                    text_field = ProceduralMemoryItem.description
                elif search_field == "instructions":
                    text_field = ProceduralMemoryItem.instructions
                else:
                    text_field = ProceduralMemoryItem.description

                tsquery = func.plainto_tsquery("english", query)
                tsvector = func.to_tsvector("english", text_field)
                rank = func.ts_rank_cd(tsvector, tsquery).label("rank")

                base_query = (
                    base_query.add_columns(rank)
                    .where(tsvector.op("@@")(tsquery))
                    .order_by(rank.desc())
                )

            if limit:
                base_query = base_query.limit(limit)

            result = await session.execute(base_query)
            items = result.scalars().all()
            return [item.to_pydantic() for item in items]
