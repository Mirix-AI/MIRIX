"""Result dataclasses for relational-provider named queries.

When a named query's SELECT projection doesn't correspond to a full entity row
(scalar aggregates, single-column projections, custom joined shapes), callers
must pass a ``result_set_entity_class`` to ``find_using_named_query`` so the
provider knows how to hydrate the response. The classes below cover the common
shapes used by the manager layer.

Lives in MIRIX (not the host application) because every caller is a MIRIX
manager. Host applications that wrap MIRIX register their own relational
provider implementation; that provider consumes these classes via the same
public ``result_set_entity_class`` kwarg.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IdOnlyResult:
    """For ``SELECT id FROM ...`` named queries (e.g. resolve-id lookups)."""

    id: Optional[str] = None


@dataclass
class CountResult:
    """For ``SELECT COUNT(*) AS count FROM ...`` named queries (size, dashboard counters)."""

    count: Optional[int] = 0


@dataclass
class FileStatsResult:
    """For ``file_manager.get_file_stats``.

    Projection: ``COUNT(id) AS total_files, SUM(file_size) AS total_size,
    COUNT(DISTINCT file_type) AS unique_types``.
    """

    total_files: Optional[int] = 0
    total_size: Optional[int] = 0
    unique_types: Optional[int] = 0


@dataclass
class MaxOccurredAtResult:
    """For ``memory_citation_manager.max_occurred_at_for_memory``.

    Projection: ``MAX(occurredAt) AS max_occurred_at``. Single-column aggregate
    with no ``id`` field — must be paired with ``skip_entity_mapping=True`` so
    the underlying provider doesn't try to bind the row to the memoryCitations
    entity schema. Used by the temporal guard to detect out-of-order writes.
    """

    max_occurred_at: Optional[object] = None


@dataclass
class AgentToolRow:
    """For ``agent_manager.list_agents_with_tools_by_parent`` — one flat
    ``(agent, tool)`` join row (LEFT JOIN, so tool columns are NULL for an agent
    with no tools).

    Field order is contractual with the NQ SELECT order: the relational
    provider returns positional rows under ``skip_entity_mapping=True`` and zips
    each row against these field names to produce a uniform dict. All fields
    default to ``None`` so tool-less rows (and any short positional rows)
    hydrate cleanly. If the NQ projection order changes, this must change in
    lockstep.
    """

    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_name: Optional[str] = None
    agent_description: Optional[str] = None
    agent_llm_config: Optional[str] = None
    agent_embedding_config: Optional[str] = None
    agent_system: Optional[str] = None
    agent_tool_rules: Optional[str] = None
    agent_mcp_tools: Optional[str] = None
    agent_parent_id: Optional[str] = None
    agent_organization_id: Optional[str] = None
    agent_owner: Optional[str] = None
    agent_created_at: Optional[str] = None
    agent_updated_at: Optional[str] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    tool_json_schema: Optional[str] = None
    tool_type: Optional[str] = None
    tool_organization_id: Optional[str] = None
