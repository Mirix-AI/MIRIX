"""
Tests for memory tool insert functions (memory_tools.py).

Verifies that insert functions call the underlying manager for every item.

These are fast unit tests — they mock manager methods and require no
running server, database, or API key.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mirix.functions.function_sets.memory_tools import (
    episodic_memory_insert,
    knowledge_vault_insert,
    procedural_memory_insert,
    resource_memory_insert,
    semantic_memory_insert,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_stub(*, manager_attr: str, insert_method: str, memory_source_id=None):
    """Build a minimal Agent-like object with a mocked manager."""
    mock_return = SimpleNamespace(id="inserted-id-1")
    manager = SimpleNamespace(**{insert_method: AsyncMock(return_value=mock_return)})
    agent = SimpleNamespace(
        agent_state=SimpleNamespace(id="agent-1", parent_id="meta-1", name="test-agent"),
        actor=SimpleNamespace(organization_id="org-1"),
        user=SimpleNamespace(organization_id="org-1"),
        filter_tags=["tag-a"],
        use_cache=False,
        client_id=None,
        user_id="user-1",
        occurred_at=None,
        memory_source_id=memory_source_id,
        external_thread_id=None,
        **{f"{manager_attr}": manager},
    )
    return agent, getattr(manager, insert_method)


# ---------------------------------------------------------------------------
# Semantic Memory Insert
# ---------------------------------------------------------------------------


class TestSemanticMemoryInsert:
    async def test_inserts_all_items(self):
        """Every item should be passed to the manager — no skipping."""
        agent, mock_insert = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
        )
        items = [
            {"name": f"item-{i}", "summary": f"summary-{i}", "details": f"details-{i}", "source": "test"}
            for i in range(3)
        ]

        result = await semantic_memory_insert(agent, items)

        assert mock_insert.await_count == 3
        assert "3" in result

    async def test_inserts_duplicate_items(self):
        """Identical items must still all be inserted (no dedup)."""
        agent, mock_insert = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
        )
        item = {"name": "dup", "summary": "same", "details": "same", "source": "test"}
        items = [item, item, item]

        result = await semantic_memory_insert(agent, items)

        assert mock_insert.await_count == 3
        assert "3" in result

    async def test_passes_correct_args(self):
        """Verify the manager is called with the right kwargs."""
        agent, mock_insert = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
        )
        items = [{"name": "n", "summary": "s", "details": "d", "source": "src"}]

        await semantic_memory_insert(agent, items)

        mock_insert.assert_awaited_once()
        call_kwargs = mock_insert.call_args[1]
        assert call_kwargs["name"] == "n"
        assert call_kwargs["summary"] == "s"
        assert call_kwargs["details"] == "d"
        assert call_kwargs["source"] == "src"
        assert call_kwargs["agent_id"] == "meta-1"
        assert call_kwargs["filter_tags"] == ["tag-a"]
        assert call_kwargs["use_cache"] is False
        assert call_kwargs["user_id"] == "user-1"

    async def test_empty_items(self):
        """An empty list should result in zero inserts."""
        agent, mock_insert = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
        )

        result = await semantic_memory_insert(agent, [])

        assert mock_insert.await_count == 0
        assert "0" in result


# ---------------------------------------------------------------------------
# Resource Memory Insert
# ---------------------------------------------------------------------------


class TestResourceMemoryInsert:
    async def test_inserts_all_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="resource_memory_manager",
            insert_method="insert_resource",
        )
        items = [
            {"title": f"res-{i}", "summary": f"s-{i}", "resource_type": "doc", "content": f"c-{i}"} for i in range(4)
        ]

        result = await resource_memory_insert(agent, items)

        assert mock_insert.await_count == 4
        assert "4" in result

    async def test_inserts_duplicate_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="resource_memory_manager",
            insert_method="insert_resource",
        )
        item = {"title": "dup", "summary": "same", "resource_type": "doc", "content": "same"}

        result = await resource_memory_insert(agent, [item, item])

        assert mock_insert.await_count == 2
        assert "2" in result

    async def test_passes_correct_args(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="resource_memory_manager",
            insert_method="insert_resource",
        )
        items = [{"title": "t", "summary": "s", "resource_type": "link", "content": "c"}]

        await resource_memory_insert(agent, items)

        call_kwargs = mock_insert.call_args[1]
        assert call_kwargs["title"] == "t"
        assert call_kwargs["resource_type"] == "link"
        assert call_kwargs["agent_id"] == "meta-1"


# ---------------------------------------------------------------------------
# Procedural Memory Insert
# ---------------------------------------------------------------------------


class TestProceduralMemoryInsert:
    async def test_inserts_all_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="procedural_memory_manager",
            insert_method="insert_procedure",
        )
        items = [{"entry_type": "process", "summary": f"proc-{i}", "steps": ["a", "b"]} for i in range(2)]

        result = await procedural_memory_insert(agent, items)

        assert mock_insert.await_count == 2
        assert "2" in result

    async def test_inserts_duplicate_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="procedural_memory_manager",
            insert_method="insert_procedure",
        )
        item = {"entry_type": "process", "summary": "same", "steps": ["x"]}

        result = await procedural_memory_insert(agent, [item, item, item])

        assert mock_insert.await_count == 3
        assert "3" in result

    async def test_passes_correct_args(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="procedural_memory_manager",
            insert_method="insert_procedure",
        )
        items = [{"entry_type": "workflow", "summary": "s", "steps": ["1", "2"]}]

        await procedural_memory_insert(agent, items)

        call_kwargs = mock_insert.call_args[1]
        assert call_kwargs["entry_type"] == "workflow"
        assert call_kwargs["steps"] == ["1", "2"]
        assert call_kwargs["agent_id"] == "meta-1"


# ---------------------------------------------------------------------------
# Knowledge Vault Insert
# ---------------------------------------------------------------------------


class TestKnowledgeVaultInsert:
    async def test_inserts_all_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="knowledge_vault_manager",
            insert_method="insert_knowledge",
        )
        items = [
            {
                "entry_type": "secret",
                "source": "vault",
                "sensitivity": "high",
                "secret_value": f"val-{i}",
                "caption": f"cap-{i}",
            }
            for i in range(3)
        ]

        result = await knowledge_vault_insert(agent, items)

        assert mock_insert.await_count == 3
        assert "3" in result

    async def test_inserts_duplicate_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="knowledge_vault_manager",
            insert_method="insert_knowledge",
        )
        item = {"entry_type": "secret", "source": "v", "sensitivity": "low", "secret_value": "same", "caption": "same"}

        result = await knowledge_vault_insert(agent, [item, item])

        assert mock_insert.await_count == 2
        assert "2" in result

    async def test_passes_correct_args(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="knowledge_vault_manager",
            insert_method="insert_knowledge",
        )
        items = [
            {"entry_type": "credential", "source": "s", "sensitivity": "high", "secret_value": "pw", "caption": "c"}
        ]

        await knowledge_vault_insert(agent, items)

        call_kwargs = mock_insert.call_args[1]
        assert call_kwargs["entry_type"] == "credential"
        assert call_kwargs["secret_value"] == "pw"
        assert call_kwargs["agent_id"] == "meta-1"


# ---------------------------------------------------------------------------
# Episodic Memory Insert
# ---------------------------------------------------------------------------


class TestEpisodicMemoryInsert:
    async def test_inserts_all_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="episodic_memory_manager",
            insert_method="insert_event",
        )
        agent.occurred_at = None
        items = [
            {
                "event_type": "activity",
                "actor": "user",
                "summary": f"ev-{i}",
                "details": f"d-{i}",
                "occurred_at": datetime.now().isoformat(),
            }
            for i in range(2)
        ]

        result = await episodic_memory_insert(agent, items)

        assert mock_insert.await_count == 2
        assert "Events inserted" in result

    async def test_inserts_duplicate_items(self):
        agent, mock_insert = _make_agent_stub(
            manager_attr="episodic_memory_manager",
            insert_method="insert_event",
        )
        agent.occurred_at = None
        item = {
            "event_type": "activity",
            "actor": "user",
            "summary": "same",
            "details": "same",
            "occurred_at": datetime.now().isoformat(),
        }

        result = await episodic_memory_insert(agent, [item, item])

        assert mock_insert.await_count == 2

    async def test_occurred_at_override(self):
        """When agent has occurred_at set, it overrides item timestamps."""
        agent, mock_insert = _make_agent_stub(
            manager_attr="episodic_memory_manager",
            insert_method="insert_event",
        )
        override_time = datetime(2026, 1, 1, 12, 0, 0)
        agent.occurred_at = override_time
        items = [
            {
                "event_type": "activity",
                "actor": "user",
                "summary": "s",
                "details": "d",
                "occurred_at": "2025-06-15T00:00:00",
            }
        ]

        await episodic_memory_insert(agent, items)

        call_kwargs = mock_insert.call_args[1]
        assert call_kwargs["timestamp"] == override_time


# ---------------------------------------------------------------------------
# Citation writes — verify MemoryCitationManager.create is called
# ---------------------------------------------------------------------------

CITATION_PATCH = "mirix.services.memory_citation_manager.MemoryCitationManager"


class TestSemanticMemoryInsertCitation:
    async def test_writes_citation_when_source_id_present(self):
        agent, _ = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
            memory_source_id="src-123",
        )
        items = [{"name": "n", "summary": "s", "details": "d", "source": "src"}]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await semantic_memory_insert(agent, items)
            MockMgr.return_value.create.assert_awaited_once()
            kw = MockMgr.return_value.create.call_args.kwargs
            assert kw["memory_type"] == "semantic"
            assert kw["citation_type"] == "created"

    async def test_no_citation_when_source_id_absent(self):
        agent, _ = _make_agent_stub(
            manager_attr="semantic_memory_manager",
            insert_method="insert_semantic_item",
        )
        items = [{"name": "n", "summary": "s", "details": "d", "source": "src"}]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await semantic_memory_insert(agent, items)
            MockMgr.return_value.create.assert_not_awaited()


class TestResourceMemoryInsertCitation:
    async def test_writes_citation_when_source_id_present(self):
        agent, _ = _make_agent_stub(
            manager_attr="resource_memory_manager",
            insert_method="insert_resource",
            memory_source_id="src-123",
        )
        items = [{"title": "t", "summary": "s", "resource_type": "doc", "content": "c"}]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await resource_memory_insert(agent, items)
            MockMgr.return_value.create.assert_awaited_once()
            assert MockMgr.return_value.create.call_args.kwargs["memory_type"] == "resource"


class TestProceduralMemoryInsertCitation:
    async def test_writes_citation_when_source_id_present(self):
        agent, _ = _make_agent_stub(
            manager_attr="procedural_memory_manager",
            insert_method="insert_procedure",
            memory_source_id="src-123",
        )
        items = [{"entry_type": "workflow", "summary": "s", "steps": ["1"]}]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await procedural_memory_insert(agent, items)
            MockMgr.return_value.create.assert_awaited_once()
            assert MockMgr.return_value.create.call_args.kwargs["memory_type"] == "procedural"


class TestKnowledgeVaultInsertCitation:
    async def test_writes_citation_when_source_id_present(self):
        agent, _ = _make_agent_stub(
            manager_attr="knowledge_vault_manager",
            insert_method="insert_knowledge",
            memory_source_id="src-123",
        )
        items = [{"entry_type": "secret", "source": "v", "sensitivity": "low", "secret_value": "pw", "caption": "c"}]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await knowledge_vault_insert(agent, items)
            MockMgr.return_value.create.assert_awaited_once()
            assert MockMgr.return_value.create.call_args.kwargs["memory_type"] == "knowledge_vault"


class TestEpisodicMemoryInsertCitation:
    async def test_writes_citation_when_source_id_present(self):
        agent, _ = _make_agent_stub(
            manager_attr="episodic_memory_manager",
            insert_method="insert_event",
            memory_source_id="src-123",
        )
        items = [
            {
                "event_type": "activity",
                "actor": "user",
                "summary": "s",
                "details": "d",
                "occurred_at": datetime.now().isoformat(),
            }
        ]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await episodic_memory_insert(agent, items)
            MockMgr.return_value.create.assert_awaited_once()
            kw = MockMgr.return_value.create.call_args.kwargs
            assert kw["memory_type"] == "episodic"
            assert kw["citation_type"] == "created"

    async def test_no_citation_when_source_id_absent(self):
        agent, _ = _make_agent_stub(
            manager_attr="episodic_memory_manager",
            insert_method="insert_event",
        )
        items = [
            {
                "event_type": "activity",
                "actor": "user",
                "summary": "s",
                "details": "d",
                "occurred_at": datetime.now().isoformat(),
            }
        ]

        with patch(CITATION_PATCH) as MockMgr:
            MockMgr.return_value.create = AsyncMock(return_value=None)
            await episodic_memory_insert(agent, items)
            MockMgr.return_value.create.assert_not_awaited()
