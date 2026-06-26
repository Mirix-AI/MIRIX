"""Unit tests for citation writing in memory tool functions and memory_source_id propagation.

Verifies that each memory tool function writes a citation record (via MemoryCitationManager)
after a successful memory operation when memory_source_id is present, and skips citation
writes when it is absent (backward compat).

These are fast unit tests — they mock the citation manager and require no running server or DB.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.functions.function_sets.memory_tools import (
    _write_citation,
    core_memory_append,
    core_memory_rewrite,
    episodic_memory_insert,
    episodic_memory_merge,
    episodic_memory_replace,
    knowledge_vault_insert,
    knowledge_vault_update,
    procedural_memory_insert,
    procedural_memory_update,
    resource_memory_insert,
    resource_memory_update,
    semantic_memory_insert,
    semantic_memory_update,
)

# --- Helpers ---


def _make_agent(memory_source_id=None, use_cache=True, external_thread_id=None, occurred_at=None):
    """Create a minimal mock agent with memory source fields."""
    agent = MagicMock()
    agent.memory_source_id = memory_source_id
    agent.use_cache = use_cache
    agent.external_thread_id = external_thread_id
    agent.occurred_at = occurred_at
    agent.filter_tags = None
    agent.client_id = "client-1"
    agent.user_id = "user-1"
    agent.actor = MagicMock()
    agent.actor.id = "client-1"
    agent.actor.organization_id = "org-1"
    agent.user = MagicMock()
    agent.user.id = "user-1"
    agent.user.organization_id = "org-1"
    agent.agent_state = MagicMock()
    agent.agent_state.parent_id = None
    agent.agent_state.id = "agent-1"
    agent.interface = MagicMock()
    return agent


# --- _write_citation ---


class TestWriteCitation:
    """Tests for the _write_citation helper."""

    @pytest.mark.asyncio
    async def test_no_op_without_memory_source_id(self):
        """When no memory_source_id, _write_citation does nothing."""
        agent = _make_agent(memory_source_id=None)
        await _write_citation(agent, "episodic", "mem-1", "created")

    @pytest.mark.asyncio
    async def test_creates_citation(self):
        agent = _make_agent(
            memory_source_id="src-789",
            external_thread_id="thread-1",
            occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await _write_citation(agent, "semantic", "sem-5", "updated")

            mock_instance.create.assert_awaited_once_with(
                memory_source_id="src-789",
                memory_type="semantic",
                memory_id="sem-5",
                citation_type="updated",
                external_thread_id="thread-1",
                occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                created_by_id="client-1",
                use_cache=True,
            )


# --- Citation writes per memory type ---


class TestEpisodicInsertCitationWrite:
    """episodic_memory_insert writes a citation for each inserted event."""

    @pytest.mark.asyncio
    async def test_writes_citation_per_event(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_event = MagicMock()
        mock_event.id = "event-new-1"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.insert_event = AsyncMock(return_value=mock_event)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

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

            mock_instance.create.assert_awaited_once()
            call_kwargs = mock_instance.create.call_args.kwargs
            assert call_kwargs["memory_type"] == "episodic"
            assert call_kwargs["memory_id"] == "event-new-1"
            assert call_kwargs["citation_type"] == "created"

    @pytest.mark.asyncio
    async def test_no_citation_without_source_id(self):
        """When no memory_source_id, insert runs normally without citation writes."""
        agent = _make_agent(memory_source_id=None)
        mock_event = MagicMock()
        mock_event.id = "event-1"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.insert_event = AsyncMock(return_value=mock_event)

        result = await episodic_memory_insert(
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

        assert "inserted" in result.lower()
        agent.episodic_memory_manager.insert_event.assert_awaited_once()


class TestEpisodicMergeCitationWrite:
    """episodic_memory_merge writes an 'updated' citation."""

    @pytest.mark.asyncio
    async def test_writes_updated_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_event = MagicMock()
        mock_event.id = "event-42"
        mock_event.summary = "merged"
        mock_event.details = "details"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.update_event = AsyncMock(return_value=mock_event)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await episodic_memory_merge(agent, "event-42", "new summary", "new details")

            mock_instance.create.assert_awaited_once()
            call_kwargs = mock_instance.create.call_args.kwargs
            assert call_kwargs["memory_type"] == "episodic"
            assert call_kwargs["memory_id"] == "event-42"
            assert call_kwargs["citation_type"] == "updated"


class TestEpisodicReplaceCitationWrite:
    """episodic_memory_replace writes 'created' citations for new events."""

    @pytest.mark.asyncio
    async def test_writes_citation_per_new_event(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_event = MagicMock()
        mock_event.id = "event-new-1"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.get_episodic_memory_by_id = AsyncMock()
        agent.episodic_memory_manager.delete_event_by_id = AsyncMock()
        agent.episodic_memory_manager.insert_event = AsyncMock(return_value=mock_event)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await episodic_memory_replace(
                agent,
                ["old-1"],
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

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["citation_type"] == "created"


class TestCoreMemoryCitationWrite:
    """core_memory_append and core_memory_rewrite write 'updated' citations."""

    def _make_memory(self, label="human", value="existing", limit=1000, block_id="block-1"):
        block = MagicMock()
        block.label = label
        block.value = value
        block.limit = limit
        block.id = block_id
        memory = MagicMock()
        memory.get_block.return_value = block
        return memory

    @pytest.mark.asyncio
    async def test_append_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        memory = self._make_memory()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await core_memory_append(agent, memory, "human", "new content")

            mock_instance.create.assert_awaited_once()
            call_kwargs = mock_instance.create.call_args.kwargs
            assert call_kwargs["memory_type"] == "core"
            assert call_kwargs["memory_id"] == "block-1"
            assert call_kwargs["citation_type"] == "updated"

    @pytest.mark.asyncio
    async def test_rewrite_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        memory = self._make_memory(value="old content")

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await core_memory_rewrite(agent, memory, "human", "new content")

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["citation_type"] == "updated"

    @pytest.mark.asyncio
    async def test_rewrite_no_citation_when_unchanged(self):
        """No citation written if the content didn't actually change."""
        agent = _make_agent(memory_source_id="src-1")
        memory = self._make_memory(value="same content")

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await core_memory_rewrite(agent, memory, "human", "same content")

            mock_instance.create.assert_not_awaited()


class TestSemanticCitationWrite:
    """semantic_memory_insert and semantic_memory_update write citations."""

    @pytest.mark.asyncio
    async def test_insert_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_item = MagicMock()
        mock_item.id = "sem-1"
        agent.semantic_memory_manager = MagicMock()
        agent.semantic_memory_manager.insert_semantic_item = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await semantic_memory_insert(
                agent,
                [{"name": "fact", "summary": "s", "details": "d", "source": "conv"}],
            )

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["memory_type"] == "semantic"

    @pytest.mark.asyncio
    async def test_update_writes_citation_per_new_item(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_item = MagicMock()
        mock_item.id = "sem-new-1"
        agent.semantic_memory_manager = MagicMock()
        agent.semantic_memory_manager.delete_semantic_item_by_id = AsyncMock()
        agent.semantic_memory_manager.insert_semantic_item = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await semantic_memory_update(
                agent,
                ["old-1"],
                [{"name": "fact", "summary": "s", "details": "d", "source": "conv"}],
            )

            mock_instance.create.assert_awaited_once()


class TestResourceCitationWrite:
    @pytest.mark.asyncio
    async def test_insert_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_item = MagicMock()
        mock_item.id = "res-1"
        agent.resource_memory_manager = MagicMock()
        agent.resource_memory_manager.insert_resource = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await resource_memory_insert(
                agent,
                [{"title": "t", "summary": "s", "resource_type": "note", "content": "c"}],
            )

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["memory_type"] == "resource"


class TestProceduralCitationWrite:
    @pytest.mark.asyncio
    async def test_insert_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_item = MagicMock()
        mock_item.id = "proc-1"
        agent.procedural_memory_manager = MagicMock()
        agent.procedural_memory_manager.insert_procedure = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await procedural_memory_insert(
                agent,
                [{"entry_type": "workflow", "summary": "s", "steps": ["1", "2"]}],
            )

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["memory_type"] == "procedural"


class TestKnowledgeVaultCitationWrite:
    @pytest.mark.asyncio
    async def test_insert_writes_citation(self):
        agent = _make_agent(memory_source_id="src-1")
        mock_item = MagicMock()
        mock_item.id = "kv-1"
        agent.knowledge_vault_manager = MagicMock()
        agent.knowledge_vault_manager.insert_knowledge = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.create = AsyncMock(return_value=None)

            await knowledge_vault_insert(
                agent,
                [
                    {
                        "entry_type": "credential",
                        "source": "s",
                        "sensitivity": "low",
                        "secret_value": "v",
                        "caption": "c",
                    }
                ],
            )

            mock_instance.create.assert_awaited_once()
            assert mock_instance.create.call_args.kwargs["memory_type"] == "knowledge_vault"


# --- Propagation in trigger_memory_update ---


class TestMemorySourceIdPropagation:
    """Verify memory_source_id is propagated to child agents in trigger_memory_update."""

    @pytest.mark.asyncio
    async def test_propagates_memory_source_id_to_child(self):
        from mirix.functions.function_sets.memory_tools import trigger_memory_update

        agent = _make_agent(memory_source_id="src-propagate", external_thread_id="thread-t")
        agent.agent_manager = MagicMock()

        child_state = MagicMock()
        child_state.name = "episodic_memory_agent"
        child_state.agent_type = "episodic_memory_agent"
        # VEPAGE-1228: trigger_memory_update resolves children via the joined NQ.
        agent.agent_manager.list_agents_with_tools = AsyncMock(return_value=[child_state])

        captured_agent = {}

        class FakeEpisodicAgent:
            def __init__(self, **kwargs):
                self.memory_source_id = None
                self.external_thread_id = None
                self.occurred_at = None
                for k, v in kwargs.items():
                    setattr(self, k, v)
                captured_agent["instance"] = self

            async def step(self, **kwargs):
                pass

        user_message = {
            "message": MagicMock(),
            "chaining": False,
        }
        user_message["message"].content = "test"
        user_message["message"].model_copy = MagicMock(return_value=user_message["message"])

        with (
            patch("mirix.agent.EpisodicMemoryAgent", FakeEpisodicAgent),
            patch("mirix.functions.function_sets.memory_tools.get_trace_context", return_value={}),
            patch("mirix.functions.function_sets.memory_tools.get_langfuse_client", return_value=None),
            patch("mirix.functions.function_sets.memory_tools.clear_trace_context"),
        ):
            await trigger_memory_update(agent, user_message, ["episodic"])

        child = captured_agent["instance"]
        assert child.memory_source_id == "src-propagate"
        assert child.external_thread_id == "thread-t"
