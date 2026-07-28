"""DB-level integration tests for citation writing in memory tool functions.

Verifies that:
1. MemoryCitationManager CRUD works against a real PostgreSQL database
2. Memory tool functions (episodic_memory_insert, semantic_memory_insert, etc.)
   produce real citation rows in the memory_citations table when memory_source_id
   is set on the agent

Requires PostgreSQL + Redis via docker test infrastructure:
    ./scripts/run_tests_with_docker.sh --podman -s -v -k test_citation_writes_db
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from mirix.functions.function_sets.memory_tools import (
    episodic_memory_insert,
    knowledge_vault_insert,
    procedural_memory_insert,
    resource_memory_insert,
    semantic_memory_insert,
)
from mirix.services.memory_citation_manager import MemoryCitationManager
from mirix.services.memory_source_manager import MemorySourceManager

pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.requires_pg]

TEST_ORG_ID = "citation-test-org"
TEST_CLIENT_ID = "citation-test-client"
TEST_USER_ID = "citation-test-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Citation Test Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Citation Test User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


@pytest_asyncio.fixture
def citation_mgr():
    return MemoryCitationManager()


@pytest_asyncio.fixture
def source_mgr():
    return MemorySourceManager()


async def _get_actor():
    """Fetch the test client as a PydanticClient for use as actor."""
    from mirix.services.client_manager import ClientManager

    client_mgr = ClientManager()
    return await client_mgr.get_client_by_id(TEST_CLIENT_ID)


async def _create_source(source_mgr, source_id=None):
    """Create a memory_source record to use as FK target for citations."""
    sid = source_id or _unique("src")
    actor = await _get_actor()
    await source_mgr.create(
        memory_source_id=sid,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_type="conversation",
        use_cache=False,
    )
    return sid


# ---------------------------------------------------------------------------
# Part 1: MemoryCitationManager CRUD against real DB
# ---------------------------------------------------------------------------


async def test_create_citation_persists(citation_mgr, source_mgr):
    """A created citation should be retrievable via check_exists."""
    source_id = await _create_source(source_mgr)
    memory_id = _unique("mem")

    result = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        citation_type="created",
        use_cache=False,
    )

    assert result is not None
    assert result.memory_type == "episodic"
    assert result.memory_id == memory_id
    assert result.citation_type == "created"

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True


async def test_duplicate_citation_silently_skipped(citation_mgr, source_mgr):
    """Same (memory_source_id, memory_type, memory_id) twice: second INSERT returns None."""
    source_id = await _create_source(source_mgr)
    memory_id = _unique("mem")

    first = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="semantic",
        memory_id=memory_id,
        citation_type="created",
        use_cache=False,
    )
    assert first is not None

    second = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="semantic",
        memory_id=memory_id,
        citation_type="updated",
        use_cache=False,
    )
    assert second is None, "Duplicate citation should be silently skipped"


async def test_different_memory_types_are_independent(citation_mgr, source_mgr):
    """Same source + same memory_id but different memory_type should both succeed."""
    source_id = await _create_source(source_mgr)
    memory_id = _unique("mem")

    c1 = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        citation_type="created",
        use_cache=False,
    )
    c2 = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="semantic",
        memory_id=memory_id,
        citation_type="created",
        use_cache=False,
    )

    assert c1 is not None
    assert c2 is not None


async def test_check_exists_returns_false_for_missing(citation_mgr, source_mgr):
    """check_exists returns False when no citation matches."""
    source_id = await _create_source(source_mgr)

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="procedural",
        memory_id=_unique("nonexistent"),
        use_cache=False,
    )
    assert exists is False


async def test_citation_preserves_metadata(citation_mgr, source_mgr):
    """external_thread_id and occurred_at are stored on the citation record."""
    source_id = await _create_source(source_mgr)
    memory_id = _unique("mem")
    thread_id = "thread-test-123"
    ts = datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc)

    result = await citation_mgr.create(
        memory_source_id=source_id,
        memory_type="core",
        memory_id=memory_id,
        citation_type="updated",
        external_thread_id=thread_id,
        occurred_at=ts,
        use_cache=False,
    )

    assert result is not None
    assert result.external_thread_id == thread_id
    assert result.occurred_at == ts
    assert result.citation_type == "updated"


async def test_get_max_occurred_at_returns_latest(citation_mgr, source_mgr):
    """get_max_occurred_at returns the most recent occurred_at across citations."""
    memory_id = _unique("mem")
    ts_old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts_new = datetime(2026, 6, 1, tzinfo=timezone.utc)

    source1 = await _create_source(source_mgr)
    source2 = await _create_source(source_mgr)

    await citation_mgr.create(
        memory_source_id=source1,
        memory_type="semantic",
        memory_id=memory_id,
        citation_type="created",
        occurred_at=ts_old,
        use_cache=False,
    )
    await citation_mgr.create(
        memory_source_id=source2,
        memory_type="semantic",
        memory_id=memory_id,
        citation_type="updated",
        occurred_at=ts_new,
        use_cache=False,
    )

    max_ts = await citation_mgr.get_max_occurred_at("semantic", memory_id)
    assert max_ts == ts_new


async def test_get_max_occurred_at_returns_none_for_missing(citation_mgr):
    """get_max_occurred_at returns None when no citations exist for the memory."""
    result = await citation_mgr.get_max_occurred_at("resource", _unique("nonexistent"))
    assert result is None


# ---------------------------------------------------------------------------
# Part 2: Memory tool functions produce real citation rows
# ---------------------------------------------------------------------------
#
# These tests call the actual tool functions with:
# - Mocked memory managers (so we don't need real agent/LLM setup)
# - Real memory_source_id pointing to a real memory_sources row
# - The _write_citation helper creates a real MemoryCitationManager
#   that writes to the real DB
# After the tool call, we query the DB to verify citation rows exist.
# ---------------------------------------------------------------------------


def _make_tool_agent(source_id, *, manager_attr, insert_method, return_id=None):
    """Build an Agent-like stub with a mocked memory manager and real source fields."""
    mock_return = SimpleNamespace(id=return_id or _unique("mem"))
    manager = SimpleNamespace(**{insert_method: AsyncMock(return_value=mock_return)})
    agent = SimpleNamespace(
        agent_state=SimpleNamespace(id="agent-1", parent_id="meta-1", name="test-agent"),
        actor=SimpleNamespace(id=TEST_CLIENT_ID, organization_id=TEST_ORG_ID),
        user=SimpleNamespace(id=TEST_USER_ID, organization_id=TEST_ORG_ID),
        filter_tags=None,
        use_cache=False,
        client_id=TEST_CLIENT_ID,
        user_id=TEST_USER_ID,
        occurred_at=None,
        memory_source_id=source_id,
        external_thread_id=None,
        **{manager_attr: manager},
    )
    return agent, mock_return.id


async def test_episodic_insert_creates_citation_row(source_mgr, citation_mgr):
    """episodic_memory_insert with memory_source_id produces a citation in the DB."""
    source_id = await _create_source(source_mgr)
    agent, memory_id = _make_tool_agent(
        source_id,
        manager_attr="episodic_memory_manager",
        insert_method="insert_event",
    )

    await episodic_memory_insert(
        agent,
        [
            {
                "occurred_at": "2026-01-01T00:00:00Z",
                "event_type": "test",
                "actor": "user",
                "summary": "s",
                "details": "d",
            }
        ],
    )

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True, "episodic_memory_insert should have created a citation row"


async def test_semantic_insert_creates_citation_row(source_mgr, citation_mgr):
    """semantic_memory_insert with memory_source_id produces a citation in the DB."""
    source_id = await _create_source(source_mgr)
    agent, memory_id = _make_tool_agent(
        source_id,
        manager_attr="semantic_memory_manager",
        insert_method="insert_semantic_item",
    )

    await semantic_memory_insert(
        agent,
        [{"name": "fact", "summary": "s", "details": "d", "source": "conv"}],
    )

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="semantic",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True, "semantic_memory_insert should have created a citation row"


async def test_resource_insert_creates_citation_row(source_mgr, citation_mgr):
    """resource_memory_insert with memory_source_id produces a citation in the DB."""
    source_id = await _create_source(source_mgr)
    agent, memory_id = _make_tool_agent(
        source_id,
        manager_attr="resource_memory_manager",
        insert_method="insert_resource",
    )

    await resource_memory_insert(
        agent,
        [{"title": "t", "summary": "s", "resource_type": "note", "content": "c"}],
    )

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="resource",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True, "resource_memory_insert should have created a citation row"


async def test_procedural_insert_creates_citation_row(source_mgr, citation_mgr):
    """procedural_memory_insert with memory_source_id produces a citation in the DB."""
    source_id = await _create_source(source_mgr)
    agent, memory_id = _make_tool_agent(
        source_id,
        manager_attr="procedural_memory_manager",
        insert_method="insert_procedure",
    )

    await procedural_memory_insert(
        agent,
        [{"entry_type": "workflow", "summary": "s", "steps": ["1", "2"]}],
    )

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="procedural",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True, "procedural_memory_insert should have created a citation row"


async def test_knowledge_vault_insert_creates_citation_row(source_mgr, citation_mgr):
    """knowledge_vault_insert with memory_source_id produces a citation in the DB."""
    source_id = await _create_source(source_mgr)
    agent, memory_id = _make_tool_agent(
        source_id,
        manager_attr="knowledge_vault_manager",
        insert_method="insert_knowledge",
    )

    await knowledge_vault_insert(
        agent,
        [{"entry_type": "credential", "source": "s", "sensitivity": "low", "secret_value": "v", "caption": "c"}],
    )

    exists = await citation_mgr.check_exists(
        memory_source_id=source_id,
        memory_type="knowledge_vault",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True, "knowledge_vault_insert should have created a citation row"


async def test_get_citations_for_memories_batch(source_mgr, citation_mgr):
    """get_citations_for_memories returns citations grouped by (memory_type, memory_id)."""
    source_id_a = await _create_source(source_mgr)
    source_id_b = await _create_source(source_mgr)

    mem_1 = _unique("mem")
    mem_2 = _unique("mem")

    # Two citations for mem_1 (from different sources)
    await citation_mgr.create(
        memory_source_id=source_id_a,
        memory_type="episodic",
        memory_id=mem_1,
        citation_type="created",
        use_cache=False,
    )
    await citation_mgr.create(
        memory_source_id=source_id_b,
        memory_type="episodic",
        memory_id=mem_1,
        citation_type="updated",
        use_cache=False,
    )
    # One citation for mem_2
    await citation_mgr.create(
        memory_source_id=source_id_a,
        memory_type="semantic",
        memory_id=mem_2,
        citation_type="created",
        use_cache=False,
    )

    result = await citation_mgr.get_citations_for_memories(
        [
            ("episodic", mem_1),
            ("semantic", mem_2),
            ("procedural", "nonexistent"),
        ]
    )

    assert ("episodic", mem_1) in result
    assert len(result[("episodic", mem_1)]) == 2
    assert ("semantic", mem_2) in result
    assert len(result[("semantic", mem_2)]) == 1
    assert ("procedural", "nonexistent") not in result

    # Verify source IDs are correct
    source_ids = {c.memory_source_id for c in result[("episodic", mem_1)]}
    assert source_ids == {source_id_a, source_id_b}


async def test_get_citations_for_memories_empty_list(citation_mgr):
    """get_citations_for_memories with empty list returns empty dict."""
    result = await citation_mgr.get_citations_for_memories([])
    assert result == {}


async def test_no_citation_without_memory_source_id(source_mgr, citation_mgr):
    """When memory_source_id is None, no citation row should be created."""
    agent, memory_id = _make_tool_agent(
        None,  # no memory_source_id
        manager_attr="semantic_memory_manager",
        insert_method="insert_semantic_item",
    )

    await semantic_memory_insert(
        agent,
        [{"name": "fact", "summary": "s", "details": "d", "source": "conv"}],
    )

    # Can't check_exists without a source_id, but we can verify the manager was still called
    agent.semantic_memory_manager.insert_semantic_item.assert_awaited_once()


async def test_multiple_items_create_multiple_citations(source_mgr, citation_mgr):
    """Inserting 3 items should produce 3 distinct citation rows."""
    source_id = await _create_source(source_mgr)

    mem_ids = [_unique("mem") for _ in range(3)]
    call_count = 0

    async def _mock_insert(**kwargs):
        nonlocal call_count
        result = SimpleNamespace(id=mem_ids[call_count])
        call_count += 1
        return result

    agent = SimpleNamespace(
        agent_state=SimpleNamespace(id="agent-1", parent_id="meta-1", name="test-agent"),
        actor=SimpleNamespace(id=TEST_CLIENT_ID, organization_id=TEST_ORG_ID),
        user=SimpleNamespace(id=TEST_USER_ID, organization_id=TEST_ORG_ID),
        filter_tags=None,
        use_cache=False,
        client_id=TEST_CLIENT_ID,
        user_id=TEST_USER_ID,
        occurred_at=None,
        memory_source_id=source_id,
        external_thread_id=None,
        semantic_memory_manager=SimpleNamespace(insert_semantic_item=_mock_insert),
    )

    await semantic_memory_insert(
        agent,
        [{"name": f"fact-{i}", "summary": f"s-{i}", "details": f"d-{i}", "source": "conv"} for i in range(3)],
    )

    for mid in mem_ids:
        exists = await citation_mgr.check_exists(
            memory_source_id=source_id,
            memory_type="semantic",
            memory_id=mid,
            use_cache=False,
        )
        assert exists is True, f"Citation for memory_id={mid} should exist"
