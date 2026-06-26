"""
Tests that memory managers delegate reads/writes to relational and search providers
when registered, and use the ORM/cache path when providers are absent.

Run:
    pytest tests/test_manager_delegation.py -v
"""

import sys
import types

# episodic/semantic managers import embedding_model at module load; that pulls llm/observability
# and optional Langfuse (pydantic v1), which breaks on Python 3.14+. Stub for unit collection.
if "mirix.embeddings" not in sys.modules:
    from unittest.mock import MagicMock as _MagicMock

    _emb_mod = types.ModuleType("mirix.embeddings")

    async def _stub_embedding_model(*_args, **_kwargs):
        return _MagicMock()

    _emb_mod.embedding_model = _stub_embedding_model
    sys.modules["mirix.embeddings"] = _emb_mod

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.block import Block, BlockUpdate
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import ToolType
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem
from mirix.schemas.semantic_memory import SemanticMemoryItemUpdate
from mirix.schemas.tool import Tool as PydanticTool
from mirix.schemas.user import User as PydanticUser
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.tool_manager import ToolManager
from mirix.services.user_manager import UserManager


def _episodic_mgr() -> EpisodicMemoryManager:
    m = EpisodicMemoryManager.__new__(EpisodicMemoryManager)
    m.session_maker = MagicMock()
    return m


def _semantic_mgr() -> SemanticMemoryManager:
    m = SemanticMemoryManager.__new__(SemanticMemoryManager)
    m.session_maker = MagicMock()
    return m


def _block_mgr() -> BlockManager:
    m = BlockManager.__new__(BlockManager)
    m.session_maker = MagicMock()
    return m


def _naive_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _mock_user() -> MagicMock:
    u = MagicMock(spec=PydanticUser)
    u.id = "user-1"
    u.organization_id = "org-1"
    return u


def _mock_actor() -> MagicMock:
    c = MagicMock(spec=PydanticClient)
    c.id = "client-1"
    c.organization_id = "org-1"
    c.name = "test-client"
    c.write_scope = "scope-a"
    return c


def _minimal_agent_state() -> AgentState:
    return AgentState.model_construct(
        id="agent-a1b2c3d4",
        name="delegation-test-agent",
        system="sys",
        agent_type=AgentType.episodic_memory_agent,
        llm_config=LLMConfig(model="m", model_endpoint_type="openai", context_window=8000),
        embedding_config=EmbeddingConfig.default_config("text-embedding-3-small"),
        organization_id="org-1",
        tools=[],
    )


def _session_maker_cm(mock_session: MagicMock):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def _episodic_row_dict() -> dict:
    ts = _naive_utc_now()
    return {
        "id": "ep-123",
        "event_type": "test",
        "summary": "sum",
        "details": "det",
        "actor": "actor-1",
        "user_id": "user-1",
        "organization_id": "org-1",
        "client_id": "client-1",
        "occurred_at": ts,
        "created_at": ts,
        "last_modify": {"timestamp": ts.isoformat(), "operation": "created"},
        "filter_tags": None,
        "agent_id": None,
        "summary_embedding": None,
        "details_embedding": None,
        "embedding_config": None,
        "updated_at": None,
    }


def _semantic_row_dict() -> dict:
    ts = _naive_utc_now()
    return {
        "id": "sem-123",
        "name": "n",
        "summary": "s",
        "details": "d",
        "source": "src",
        "user_id": "user-1",
        "organization_id": "org-1",
        "client_id": "client-1",
        "created_at": ts,
        "last_modify": {"timestamp": ts.isoformat(), "operation": "created"},
        "filter_tags": None,
        "agent_id": None,
        "name_embedding": None,
        "summary_embedding": None,
        "details_embedding": None,
        "embedding_config": None,
        "updated_at": None,
    }


def _block_row_dict(bid: str = "block-c0ffee00") -> dict:
    return {
        "id": bid,
        "label": "human",
        "value": "hello",
        "limit": 8000,
        "user_id": "user-1",
        "organization_id": "org-1",
        "filter_tags": {"scope": "scope-a"},
        "created_by_id": "user-1",
        "last_updated_by_id": "client-1",
    }


def _organization_row_dict(oid: str = "org-1") -> dict:
    ts = _naive_utc_now()
    return {"id": oid, "name": "test-org", "created_at": ts}


def _user_row_dict(uid: str = "user-1") -> dict:
    ts = _naive_utc_now()
    return {
        "id": uid,
        "name": "test-user",
        "status": "active",
        "timezone": "UTC (UTC+00:00)",
        "organization_id": "org-1",
        "is_admin": False,
        "created_at": ts,
        "updated_at": ts,
        "is_deleted": False,
    }


def _tool_source_with_docstring() -> str:
    return '''def test_delegation_tool():
    """Minimal tool for manager delegation tests."""
    pass
'''


def _tool_row_dict(tid: str = "tool-c0ffee00") -> dict:
    return {
        "id": tid,
        "name": "test_tool",
        "tool_type": ToolType.CUSTOM,
        "description": "d",
        "source_type": None,
        "organization_id": "org-1",
        "tags": [],
        "source_code": _tool_source_with_docstring(),
        "json_schema": {"type": "object", "description": "d"},
        "return_char_limit": 10000,
        "created_by_id": "client-1",
        "last_updated_by_id": None,
    }


def _agent_row_dict(aid: str = "agent-1") -> dict:
    emb = EmbeddingConfig.default_config("text-embedding-3-small")
    return {
        "id": aid,
        "name": "test-agent",
        "system": "sys",
        "agent_type": AgentType.meta_memory_agent,
        "created_by_id": "client-1",
        "organization_id": "org-1",
        "llm_config": {"model": "gpt-4", "model_endpoint_type": "openai", "context_window": 8000},
        "embedding_config": emb.model_dump(),
        "tools": [],
        "parent_id": None,
        "tool_rules": None,
        "mcp_tools": [],
        "description": None,
        "children": None,
    }


def _org_mgr() -> OrganizationManager:
    m = OrganizationManager.__new__(OrganizationManager)
    m.session_maker = MagicMock()
    return m


def _user_mgr() -> UserManager:
    m = UserManager.__new__(UserManager)
    m.session_maker = MagicMock()
    return m


def _tool_mgr() -> ToolManager:
    m = ToolManager.__new__(ToolManager)
    m.session_maker = MagicMock()
    return m


def _agent_mgr() -> AgentManager:
    m = AgentManager.__new__(AgentManager)
    m.session_maker = MagicMock()
    return m


class TestEpisodicMemoryManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_by_id_delegates_to_relational_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            out = await mgr.get_episodic_memory_by_id("ep-123", _mock_user())
            mock_provider.read.assert_awaited_once()
            args, _ = mock_provider.read.call_args
            assert args[0] == "episodic_memory"
            assert args[1] == "ep-123"
            assert out.id == "ep-123"
            assert out.summary == "sum"

    @pytest.mark.asyncio
    async def test_get_by_id_raises_when_provider_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            with pytest.raises(NoResultFound):
                await mgr.get_episodic_memory_by_id("missing", _mock_user())

    @pytest.mark.asyncio
    async def test_get_by_id_falls_back_when_no_relational_provider(self):
        pyd = PydanticEpisodicEvent(**_episodic_row_dict())
        mock_orm = MagicMock()
        mock_orm.to_pydantic = MagicMock(return_value=pyd)
        mock_session = MagicMock()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=None):
            with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
                with patch(
                    "mirix.services.episodic_memory_manager.EpisodicEvent.read",
                    new_callable=AsyncMock,
                    return_value=mock_orm,
                ):
                    mgr = _episodic_mgr()
                    mgr.session_maker = _session_maker_cm(mock_session)
                    out = await mgr.get_episodic_memory_by_id("ep-123", _mock_user())
                    assert out.id == "ep-123"

    @pytest.mark.asyncio
    async def test_create_episodic_memory_delegates_to_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)

        ev = PydanticEpisodicEvent(**{**row, "id": "ep-new"})
        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            out = await mgr.create_episodic_memory(ev, _mock_actor(), client_id="client-1", user_id="user-1")
            mock_provider.create.assert_awaited_once()
            call_kw = mock_provider.create.await_args
            assert call_kw[0][0] == "episodic_memory"
            assert call_kw[0][1]["user_id"] == "user-1"
            assert out.id == "ep-123"

    @pytest.mark.asyncio
    async def test_delete_event_by_id_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            await mgr.delete_event_by_id("ep-123", _mock_actor())
            mock_provider.delete.assert_awaited_once()
            args, _ = mock_provider.delete.call_args
            assert args[0] == "episodic_memory"
            assert args[1] == "ep-123"

    @pytest.mark.asyncio
    async def test_insert_event_delegates_to_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        ts = _naive_utc_now()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            await mgr.insert_event(
                actor=_mock_actor(),
                agent_state=_minimal_agent_state(),
                agent_id="ag-1",
                event_type="test",
                timestamp=ts,
                event_actor="bot",
                details="d",
                summary="s",
                organization_id="org-1",
                client_id="client-1",
                user_id="user-1",
            )
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "episodic_memory"

    @pytest.mark.asyncio
    async def test_list_episodic_memory_delegates_to_search_provider(self):
        """After the labeled-bucket refactor, the manager surface is
        Search-only — no contextvar branching, no hybrid_search call. The
        recent (5s indexing-lag) bucket is fetched separately from the
        save-flow prompt builder's _fetch_recent_indexing_lag_window."""
        row = _episodic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _episodic_mgr()
            out = await mgr.list_episodic_memory(
                _minimal_agent_state(),
                _mock_user(),
                query="q",
                search_method="bm25",
                limit=10,
            )
            mock_search.search.assert_awaited_once()
            assert mock_search.search.await_args[0][0] == "episodic_memory"
            assert len(out) == 1
            assert out[0].id == "ep-123"

    @pytest.mark.asyncio
    async def test_get_total_number_delegates_to_search_count(self):
        mock_search = MagicMock()
        mock_search.count = AsyncMock(return_value=42)

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _episodic_mgr()
            n = await mgr.get_total_number_of_items(_mock_user())
            mock_search.count.assert_awaited_once_with(
                "episodic_memory",
                user_id="user-1",
                organization_id="org-1",
            )
            assert n == 42


class TestSemanticMemoryManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_semantic_item_by_id_delegates(self):
        row = _semantic_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.get_semantic_item_by_id("sem-123", _mock_user(), timezone_str="UTC")
            mock_provider.read.assert_awaited_once()
            args, _ = mock_provider.read.call_args
            assert args[0] == "semantic_memory"
            assert args[1] == "sem-123"
            assert out.id == "sem-123"

    @pytest.mark.asyncio
    async def test_get_semantic_item_by_id_raises_when_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            with pytest.raises(NoResultFound):
                await mgr.get_semantic_item_by_id("x", _mock_user(), timezone_str="UTC")

    @pytest.mark.asyncio
    async def test_get_by_id_falls_back_when_no_relational_provider(self):
        row = _semantic_row_dict()
        pyd = PydanticSemanticMemoryItem(**row)
        mock_orm = MagicMock()
        mock_orm.to_pydantic = MagicMock(return_value=pyd)
        mock_session = MagicMock()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=None):
            with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
                with patch(
                    "mirix.services.semantic_memory_manager.SemanticMemoryItem.read",
                    new_callable=AsyncMock,
                    return_value=mock_orm,
                ):
                    mgr = _semantic_mgr()
                    mgr.session_maker = _session_maker_cm(mock_session)
                    out = await mgr.get_semantic_item_by_id("sem-123", _mock_user(), timezone_str="UTC")
                    assert out.name == "n"

    @pytest.mark.asyncio
    async def test_create_item_delegates(self):
        row = _semantic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        item = PydanticSemanticMemoryItem(**row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.create_item(item, _mock_actor(), client_id="client-1", user_id="user-1")
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "semantic_memory"
            assert out.id == "sem-123"

    @pytest.mark.asyncio
    async def test_update_item_delegates(self):
        row = _semantic_row_dict()
        row["summary"] = "updated"
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=row)
        upd = SemanticMemoryItemUpdate(id="sem-123", summary="updated")

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.update_item(upd, _mock_user(), _mock_actor())
            mock_provider.update.assert_awaited_once()
            args, _ = mock_provider.update.call_args
            assert args[0] == "semantic_memory"
            assert args[1] == "sem-123"
            assert args[2] == {"summary": "updated"}
            assert out.summary == "updated"

    @pytest.mark.asyncio
    async def test_delete_semantic_item_by_id_delegates(self):
        mock_provider = MagicMock()
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            await mgr.delete_semantic_item_by_id("sem-123", _mock_actor())
            mock_provider.delete.assert_awaited_once()
            args, _ = mock_provider.delete.call_args
            assert args[0] == "semantic_memory"
            assert args[1] == "sem-123"

    @pytest.mark.asyncio
    async def test_list_semantic_items_delegates_to_search_provider(self):
        row = _semantic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _semantic_mgr()
            out = await mgr.list_semantic_items(
                _minimal_agent_state(),
                _mock_user(),
                query="q",
                limit=3,
            )
            mock_search.search.assert_awaited_once()
            args, _kwargs = mock_search.search.await_args
            assert args[0] == "semantic_memory"
            assert len(out) == 1

    @pytest.mark.asyncio
    async def test_get_total_number_delegates_to_search_count(self):
        mock_search = MagicMock()
        mock_search.count = AsyncMock(return_value=7)

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _semantic_mgr()
            n = await mgr.get_total_number_of_items(_mock_user())
            mock_search.count.assert_awaited_once_with(
                "semantic_memory",
                user_id="user-1",
                organization_id="org-1",
            )
            assert n == 7


class TestBlockManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_block_by_id_delegates(self):
        row = _block_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.get_block_by_id("block-c0ffee00", user=_mock_user())
            mock_provider.read.assert_awaited_once_with("block", "block-c0ffee00")
            assert out.id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_get_block_by_id_returns_none_when_provider_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            assert await mgr.get_block_by_id("missing") is None

    @pytest.mark.asyncio
    async def test_create_or_update_block_creates_via_provider(self):
        row = _block_row_dict("block-deadb33f")
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)
        mock_provider.create = AsyncMock(return_value=row)

        blk = Block(
            id="block-deadb33f",
            label="human",
            value="v",
            limit=8000,
            user_id="user-1",
            organization_id="org-1",
            filter_tags={"scope": "scope-a"},
            created_by_id="user-1",
            last_updated_by_id="client-1",
        )

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.create_or_update_block(blk, _mock_actor(), user=_mock_user())
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "block"
            assert out.id == "block-deadb33f"

    @pytest.mark.asyncio
    async def test_update_block_delegates(self):
        row = _block_row_dict()
        row["value"] = "patched"
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=_block_row_dict())
        mock_provider.update = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.update_block(
                "block-c0ffee00",
                BlockUpdate(value="patched"),
                _mock_actor(),
                user=_mock_user(),
            )
            mock_provider.update.assert_awaited_once()
            assert mock_provider.update.await_args[0][0] == "block"
            assert mock_provider.update.await_args[0][1] == "block-c0ffee00"
            assert out.value == "patched"

    @pytest.mark.asyncio
    async def test_delete_block_delegates(self):
        existing = _block_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            with patch.object(BlockManager, "_invalidate_block_cache", new_callable=AsyncMock):
                mgr = _block_mgr()
                out = await mgr.delete_block("block-c0ffee00", _mock_actor())
                mock_provider.delete.assert_awaited_once()
                args, _ = mock_provider.delete.call_args
                assert args[0] == "block"
                assert args[1] == "block-c0ffee00"
                assert out.id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_get_blocks_delegates_to_relational_provider(self):
        row = _block_row_dict()
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[row])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _block_mgr()
            out = await mgr.get_blocks(
                user=_mock_user(),
                any_scopes=["scope-a"],
                auto_create_from_default=False,
            )
            mock_provider.list.assert_awaited_once()
            args, kwargs = mock_provider.list.await_args
            assert args[0] == "block"
            assert kwargs["user_id"] == "user-1"
            assert kwargs["scopes"] == ["scope-a"]
            assert len(out) == 1

    @pytest.mark.asyncio
    async def test_search_blocks_delegates_to_search_provider_single_user(self):
        """The query path reads core blocks from the search provider so that
        filter_tags are honoured (the relational list() drops them)."""
        row = _block_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _block_mgr()
            out = await mgr.search_blocks(
                user_id="user-1",
                organization_id="org-1",
                query="q",
                search_method="bm25",
                scopes=["scope-a"],
                filter_tags={"seller_id": "tom"},
                limit=10,
            )
            mock_search.search.assert_awaited_once()
            args, kwargs = mock_search.search.await_args
            assert args[0] == "block"
            assert kwargs["user_id"] == "user-1"
            assert kwargs["organization_id"] == "org-1"
            assert kwargs["scopes"] == ["scope-a"]
            assert kwargs["filter_tags"] == {"seller_id": "tom"}
            assert kwargs["search_method"] == "bm25"
            assert len(out) == 1
            assert out[0].id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_search_blocks_delegates_org_wide_when_user_id_none(self):
        """Cross-user (org-wide) core search passes user_id=None and an org id."""
        row = _block_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _block_mgr()
            out = await mgr.search_blocks(
                user_id=None,
                organization_id="org-1",
                query="",
                search_method="bm25",
                scopes=["scope-a"],
                filter_tags={"fettuchini": "noodles"},
                limit=10,
            )
            mock_search.search.assert_awaited_once()
            _, kwargs = mock_search.search.await_args
            assert kwargs["user_id"] is None
            assert kwargs["organization_id"] == "org-1"
            assert kwargs["filter_tags"] == {"fettuchini": "noodles"}
            assert len(out) == 1

    @pytest.mark.asyncio
    async def test_search_blocks_falls_back_to_postgres_when_no_search_provider(self):
        """With no search provider (PG-fallback / provider mode off), block search
        reads from the relational store via Block.list_by_scopes, honouring
        scope + filter_tags so block_filter_tags still work."""
        pyd = MagicMock(spec=Block)
        pyd.id = "block-c0ffee00"
        orm_block = MagicMock()
        orm_block.value = "pizza party at acme"
        orm_block.to_pydantic = MagicMock(return_value=pyd)

        mock_session = MagicMock()
        mgr = _block_mgr()
        mgr.session_maker = _session_maker_cm(mock_session)

        with patch("mirix.database.search_provider.get_search_provider", return_value=None):
            with patch(
                "mirix.orm.block.Block.list_by_scopes",
                new=AsyncMock(return_value=[orm_block]),
            ) as mock_list:
                out = await mgr.search_blocks(
                    user_id=None,
                    organization_id="org-1",
                    query="pizza",
                    scopes=["scope-a"],
                    filter_tags={"account_ids": {"$contains": "acct-1999"}},
                    limit=50,
                )
                mock_list.assert_awaited_once()
                _, kwargs = mock_list.await_args
                assert kwargs["organization_id"] == "org-1"
                assert kwargs["user_id"] is None
                assert kwargs["scopes"] == ["scope-a"]
                assert kwargs["filter_tags"] == {"account_ids": {"$contains": "acct-1999"}}
                assert len(out) == 1
                assert out[0].id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_search_blocks_fallback_query_filters_by_value_substring(self):
        """In PG-fallback a non-empty query does a case-insensitive substring
        filter on the block value (no BM25 ranking available)."""
        match = MagicMock()
        match.value = "PIZZA toppings"
        match.to_pydantic = MagicMock(return_value=MagicMock(spec=Block))
        miss = MagicMock()
        miss.value = "sushi menu"
        miss.to_pydantic = MagicMock(return_value=MagicMock(spec=Block))

        mgr = _block_mgr()
        mgr.session_maker = _session_maker_cm(MagicMock())
        with patch("mirix.database.search_provider.get_search_provider", return_value=None):
            with patch(
                "mirix.orm.block.Block.list_by_scopes",
                new=AsyncMock(return_value=[match, miss]),
            ):
                out = await mgr.search_blocks(
                    user_id=None,
                    organization_id="org-1",
                    query="pizza",
                    scopes=["scope-a"],
                    limit=50,
                )
                assert len(out) == 1
                match.to_pydantic.assert_called_once()
                miss.to_pydantic.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_blocks_fallback_returns_empty_without_org(self):
        """Fallback needs an organization_id; without one it returns []."""
        with patch("mirix.database.search_provider.get_search_provider", return_value=None):
            mgr = _block_mgr()
            out = await mgr.search_blocks(user_id="user-1", organization_id=None)
            assert out == []


@pytest.mark.asyncio
class TestOrganizationManagerDelegation:
    async def test_get_org_by_id_delegates_to_provider(self):
        row = _organization_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _org_mgr()
                out = await mgr.get_organization_by_id("org-1")
                mock_provider.find_using_named_query.assert_awaited_once()
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "organizations"
                assert args[1] == "organization_manager.get_organization_by_id"
                assert kwargs["params"] == {"id": "org-1"}
                assert isinstance(out, PydanticOrganization)
                assert out.id == "org-1"
                assert out.name == "test-org"

    async def test_create_organization_delegates(self):
        row = _organization_row_dict("org-new")
        mock_provider = MagicMock()
        # _create_organization reads the row by id first; provide None to force create.
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.read = AsyncMock(return_value=None)
        mock_provider.create = AsyncMock(return_value=row)
        pyd = PydanticOrganization(id="org-new", name="new-org")

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _org_mgr()
                out = await mgr.create_organization(pyd)
                mock_provider.create.assert_awaited_once()
                assert mock_provider.create.await_args[0][0] == "organizations"
                assert mock_provider.create.await_args[0][1]["id"] == "org-new"
                assert out.id == "org-new"

    async def test_list_organizations_delegates(self):
        row = _organization_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _org_mgr()
            out = await mgr.list_organizations(limit=25)
            mock_provider.find_using_named_query.assert_awaited_once()
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "organizations"
            assert args[1] == "organization_manager.list_organizations"
            assert kwargs["params"]["cursor"] is None
            assert kwargs["page_size"] == 25
            assert len(out) == 1
            assert out[0].id == "org-1"


@pytest.mark.asyncio
class TestUserManagerDelegation:
    async def test_get_user_by_id_delegates_to_provider(self):
        row = _user_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _user_mgr()
                out = await mgr.get_user_by_id("user-1")
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "users"
                assert args[1] == "user_manager.get_user_by_id"
                assert kwargs["params"] == {"id": "user-1"}
                assert out.id == "user-1"
                assert out.name == "test-user"

    async def test_create_user_delegates(self):
        row = _user_row_dict("user-new")
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        pyd = PydanticUser(
            id="user-new",
            name="new-user",
            status="active",
            timezone="UTC (UTC+00:00)",
            organization_id="org-1",
        )

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _user_mgr()
            out = await mgr.create_user(pyd)
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "users"
            assert mock_provider.create.await_args[0][1]["id"] == "user-new"
            assert out.id == "user-new"

    async def test_list_users_delegates(self):
        row = _user_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _user_mgr()
            out = await mgr.list_users(organization_id="org-1", limit=10)
            mock_provider.find_using_named_query.assert_awaited_once()
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "users"
            assert args[1] == "user_manager.list_users"
            assert kwargs["page_size"] == 10
            assert len(out) == 1
            assert out[0].id == "user-1"


@pytest.mark.asyncio
class TestToolManagerDelegation:
    async def test_get_tool_by_id_delegates_to_provider(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "tools"
                assert args[1] == "tool_manager.get_tool_by_id"
                assert kwargs["params"]["id"] == "tool-c0ffee00"
                assert out.id == "tool-c0ffee00"
                assert out.name == "test_tool"

    async def test_create_tool_delegates(self):
        row = _tool_row_dict("tool-deadb33f")
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        tool = PydanticTool(
            id="tool-deadb33f",
            name="new_tool",
            json_schema={"type": "object", "description": "x"},
            source_code='''def new_tool_fn():
    """Create-tool delegation test."""
    pass
''',
        )
        actor = _mock_actor()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _tool_mgr()
            out = await mgr.create_tool(tool, actor=actor)
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "tools"
            sent = mock_provider.create.await_args[0][1]
            assert sent["_created_by_id"] == actor.id
            assert sent["organization_id"] == actor.organization_id
            assert out.id == "tool-deadb33f"

    async def test_list_tools_delegates(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])
        actor = _mock_actor()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _tool_mgr()
            out = await mgr.list_tools(actor, limit=30)
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "tools"
            assert args[1] == "tool_manager.list_tools"
            assert kwargs["params"] == {"organizationId": actor.organization_id, "cursor": None}
            assert kwargs["page_size"] == 30
            assert len(out) == 1
            assert out[0].id == "tool-c0ffee00"


@pytest.mark.asyncio
class TestAgentManagerDelegation:
    async def test_get_agent_by_id_delegates_to_provider(self):
        row = _agent_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                actor = _mock_actor()
                out = await mgr.get_agent_by_id("agent-1", actor)
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "agents"
                assert args[1] == "agent_manager.get_agent_by_id"
                assert kwargs["params"] == {
                    "id": "agent-1",
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                }
                assert kwargs["include_relationships"] == ["tools"]
                assert out.id == "agent-1"
                assert out.name == "test-agent"
                assert out.created_by_id == "client-1"

    async def test_delete_agent_delegates(self):
        existing = {**_agent_row_dict(), "parent_id": None}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.hard_delete = AsyncMock(return_value=None)

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                await mgr.delete_agent("agent-1", _mock_actor())
                mock_provider.read.assert_awaited_once_with("agents", "agent-1")
                mock_provider.hard_delete.assert_awaited_once_with("agents", "agent-1")

    async def test_get_agent_by_id_populates_cache_on_provider_path(self):
        """GAP J: read-through cache should be populated after provider read."""
        row = _agent_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=None)
        mock_cache.set_hash = AsyncMock()
        mock_cache.AGENT_PREFIX = "agent:"
        mock_cache.TOOL_PREFIX = "tool:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                out = await mgr.get_agent_by_id("agent-1", _mock_actor())
                assert out.id == "agent-1"
                mock_cache.set_hash.assert_awaited()


# ---------------------------------------------------------------------------
# GAP C: update_event delegates to relational provider + skips embedding
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicUpdateEventDelegation:
    async def test_update_event_delegates_to_provider(self):
        existing = _episodic_row_dict()
        updated = {**existing, "summary": "new summary"}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.update = AsyncMock(return_value=updated)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            result = await mgr.update_event(
                event_id="ep-123",
                new_summary="new summary",
                actor=_mock_actor(),
            )
            mock_provider.update.assert_awaited_once()
            assert "episodic_memory" == mock_provider.update.await_args[0][0]
            assert "ep-123" == mock_provider.update.await_args[0][1]
            assert result.id == "ep-123"

    async def test_update_event_appends_details(self):
        existing = _episodic_row_dict()
        expected_details = existing["details"] + "\n" + "more info"
        updated = {**existing, "details": expected_details}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.update = AsyncMock(return_value=updated)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            result = await mgr.update_event(
                event_id="ep-123",
                new_details="more info",
                actor=_mock_actor(),
                update_mode="append",
            )
            call_data = mock_provider.update.await_args[0][2]
            assert "more info" in call_data["details"]


# ---------------------------------------------------------------------------
# GAP D: list_episodic_memory_by_org delegates to search provider
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicListByOrgDelegation:
    async def test_list_by_org_delegates_to_search_provider(self):
        row = _episodic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            mgr = _episodic_mgr()
            results = await mgr.list_episodic_memory_by_org(
                agent_state=agent_state,
                organization_id="org-1",
                query="test",
            )
            mock_search.search.assert_awaited_once()
            assert len(results) == 1
            assert results[0].id == "ep-123"


# ---------------------------------------------------------------------------
# GAP E: list_episodic_memory_around_timestamp uses named query
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicAroundTimestampDelegation:
    async def test_delegates_to_named_query(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            results = await mgr.list_episodic_memory_around_timestamp(
                agent_state=agent_state,
                start_time=start,
                end_time=end,
                user=_mock_user(),
            )
            mock_provider.find_using_named_query.assert_awaited_once()
            call_args = mock_provider.find_using_named_query.await_args
            assert call_args[0][0] == "episodic_memory"
            assert call_args[0][1] == "episodic_memory_manager.list_by_occurred_at_range"
            params = call_args[1]["params"]
            assert params["userId"] == "user-1"
            assert params["since"] == start.isoformat()
            assert params["until"] == end.isoformat()
            assert len(results) == 1
            assert results[0].id == "ep-123"

    async def test_delegates_with_open_ended_range(self):
        """Open-ended range (distant past/future) passes None for since/until params."""
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        # Use sentinel datetimes that represent "no start" / "no end"
        distant_past = datetime(1970, 1, 1, tzinfo=timezone.utc)
        distant_future = datetime(2999, 12, 31, tzinfo=timezone.utc)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            results = await mgr.list_episodic_memory_around_timestamp(
                agent_state=agent_state,
                start_time=distant_past,
                end_time=distant_future,
                user=_mock_user(),
            )
            call_args = mock_provider.find_using_named_query.await_args
            params = call_args[1]["params"]
            assert params["since"] == distant_past.isoformat()
            assert params["until"] == distant_future.isoformat()
            assert results == []


# ---------------------------------------------------------------------------
# GAP J: tool_manager read-through cache on provider path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestToolManagerCacheDelegation:
    async def test_get_tool_by_id_populates_cache_on_provider_path(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=None)
        mock_cache.set_hash = AsyncMock()
        mock_cache.TOOL_PREFIX = "tool:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                assert out.id == "tool-c0ffee00"
                mock_cache.set_hash.assert_awaited()

    async def test_get_tool_by_id_returns_from_cache_hit(self):
        cached_data = _tool_row_dict()
        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=cached_data)
        mock_cache.TOOL_PREFIX = "tool:"

        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock()

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                assert out.id == "tool-c0ffee00"
                mock_provider.find_using_named_query.assert_not_awaited()


# ---------------------------------------------------------------------------
# Memory table cascade deletes use named queries in client_manager / user_manager
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestClientManagerCascadeNamedQuery:
    """delete_client_by_id and delete_memories_by_client_id use find_using_named_query."""

    def _client_mgr(self):
        from mirix.services.client_manager import ClientManager

        m = ClientManager.__new__(ClientManager)
        m.session_maker = MagicMock()
        return m

    async def test_delete_client_by_id_uses_named_query_for_memory_tables(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.bulk_delete = AsyncMock(return_value={"success": 0, "failed": 0})
        mock_provider.mutate_using_named_query = AsyncMock(return_value=0)
        mock_provider.delete = AsyncMock(return_value=True)

        _MEMORY_TABLES = {
            "episodic_memory",
            "semantic_memory",
            "procedural_memory",
            "resource_memory",
            "knowledge_vault",
            "raw_memory",
            "block",
        }

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = self._client_mgr()
            await mgr.delete_client_by_id("client-1")

        nq_calls = mock_provider.find_using_named_query.await_args_list
        # Filter to only memory-table named query calls (exclude agents/tools engine-table calls)
        memory_nq_calls = [c for c in nq_calls if c[0][0] in _MEMORY_TABLES]
        tables_queried = [c[0][0] for c in memory_nq_calls]
        query_names = [c[0][1] for c in memory_nq_calls]

        assert "episodic_memory" in tables_queried
        assert "raw_memory" in tables_queried
        assert "block" in tables_queried
        assert "messages" not in tables_queried  # messages uses mutate_using_named_query
        for name in query_names:
            assert name.startswith("client_manager.list_ids_")

    async def test_delete_memories_by_client_id_includes_raw_memory(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[{"id": "m-1"}])
        mock_provider.bulk_delete = AsyncMock(return_value={"success": 1, "failed": 0})
        mock_provider.mutate_using_named_query = AsyncMock(return_value=0)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = self._client_mgr()
            await mgr.delete_memories_by_client_id("client-1")

        nq_calls = mock_provider.find_using_named_query.await_args_list
        tables_queried = [c[0][0] for c in nq_calls]
        assert "raw_memory" in tables_queried

    async def test_messages_soft_deleted_via_mutate(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.bulk_delete = AsyncMock(return_value={"success": 0, "failed": 0})
        mock_provider.mutate_using_named_query = AsyncMock(return_value=0)
        mock_provider.delete = AsyncMock(return_value=True)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = self._client_mgr()
            await mgr.delete_client_by_id("client-1")

        mutate_calls = mock_provider.mutate_using_named_query.await_args_list
        mutate_tables = [c[0][0] for c in mutate_calls]
        assert "messages" in mutate_tables


@pytest.mark.asyncio
class TestUserManagerCascadeNamedQuery:
    """delete_user_by_id and delete_memories_by_user_id use find_using_named_query."""

    def _user_mgr(self):
        m = UserManager.__new__(UserManager)
        m.session_maker = MagicMock()
        return m

    async def test_delete_user_by_id_uses_named_query_for_memory_tables(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.bulk_delete = AsyncMock(return_value={"success": 0, "failed": 0})
        mock_provider.mutate_using_named_query = AsyncMock(return_value=0)
        mock_provider.delete = AsyncMock(return_value=True)

        mock_cache = MagicMock()
        mock_cache.delete = AsyncMock()
        mock_cache.USER_PREFIX = "user:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = self._user_mgr()
                await mgr.delete_user_by_id("user-1")

        nq_calls = mock_provider.find_using_named_query.await_args_list
        tables_queried = [c[0][0] for c in nq_calls]
        query_names = [c[0][1] for c in nq_calls]

        assert "raw_memory" in tables_queried
        assert "episodic_memory" in tables_queried
        assert "messages" not in tables_queried
        for name in query_names:
            assert name.startswith("user_manager.list_ids_")

    async def test_messages_mutated_via_named_query_in_delete_user(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.bulk_delete = AsyncMock(return_value={"success": 0, "failed": 0})
        mock_provider.mutate_using_named_query = AsyncMock(return_value=0)
        mock_provider.delete = AsyncMock(return_value=True)

        mock_cache = MagicMock()
        mock_cache.delete = AsyncMock()
        mock_cache.USER_PREFIX = "user:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = self._user_mgr()
                await mgr.delete_user_by_id("user-1")

        mutate_calls = mock_provider.mutate_using_named_query.await_args_list
        mutate_tables = [c[0][0] for c in mutate_calls]
        assert "messages" in mutate_tables


# ---------------------------------------------------------------------------
# find_most_recently_updated uses named query for registered tables
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestFindMostRecentlyUpdatedNamedQuery:
    """find_most_recently_updated prefers named query for semantic/knowledge_vault."""

    async def test_semantic_uses_named_query(self):
        from mirix.services.memory_manager_helpers import find_most_recently_updated

        expected = {"id": "s-1", "name": "foo", "summary": "bar"}
        mock_provider = AsyncMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[expected])

        result = await find_most_recently_updated(
            mock_provider, "semantic_memory", user_id="user-1", organization_id="org-1"
        )

        assert result == expected
        mock_provider.find_using_named_query.assert_awaited_once()
        args = mock_provider.find_using_named_query.await_args[0]
        assert args[0] == "semantic_memory"
        assert args[1] == "memory_manager_helpers.get_most_recently_updated_semantic_memory"
        kw = mock_provider.find_using_named_query.await_args[1]
        assert kw["params"] == {"userId": "user-1", "organizationId": "org-1"}
        assert kw["page_size"] == 1

    async def test_knowledge_vault_uses_named_query(self):
        from mirix.services.memory_manager_helpers import find_most_recently_updated

        mock_provider = AsyncMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        result = await find_most_recently_updated(mock_provider, "knowledge_vault", user_id="u", organization_id="o")

        assert result is None
        args = mock_provider.find_using_named_query.await_args[0]
        assert args[1] == "memory_manager_helpers.get_most_recently_updated_knowledge_vault"

    @pytest.mark.parametrize(
        "table,expected_query",
        [
            (
                "block",
                "memory_manager_helpers.get_most_recently_updated_block",
            ),
            (
                "raw_memory",
                "memory_manager_helpers.get_most_recently_updated_raw_memory",
            ),
            (
                "episodic_memory",
                "memory_manager_helpers.get_most_recently_updated_episodic_memory",
            ),
            (
                "procedural_memory",
                "memory_manager_helpers.get_most_recently_updated_procedural_memory",
            ),
            (
                "resource_memory",
                "memory_manager_helpers.get_most_recently_updated_resource_memory",
            ),
        ],
    )
    async def test_all_memory_tables_use_named_query(self, table, expected_query):
        """Every memory table now resolves to its dedicated named query."""
        from mirix.services.memory_manager_helpers import find_most_recently_updated

        mock_provider = AsyncMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[{"id": f"{table}-1"}])

        result = await find_most_recently_updated(mock_provider, table, user_id="u", organization_id="o")

        assert result == {"id": f"{table}-1"}
        mock_provider.find_using_named_query.assert_awaited_once()
        args = mock_provider.find_using_named_query.await_args[0]
        assert args[0] == table
        assert args[1] == expected_query
        mock_provider.list.assert_not_awaited()

    async def test_falls_back_to_list_when_no_org_id(self):
        """Named query not used when organization_id is missing."""
        from mirix.services.memory_manager_helpers import find_most_recently_updated

        mock_provider = AsyncMock()
        mock_provider.list = AsyncMock(return_value=[])

        await find_most_recently_updated(mock_provider, "semantic_memory", user_id="u", organization_id=None)

        mock_provider.list.assert_awaited()
        mock_provider.find_using_named_query.assert_not_awaited()

    async def test_falls_back_to_list_when_client_id_provided(self):
        """Named query not used when client_id adds extra filter complexity."""
        from mirix.services.memory_manager_helpers import find_most_recently_updated

        mock_provider = AsyncMock()
        mock_provider.list = AsyncMock(return_value=[])

        await find_most_recently_updated(
            mock_provider, "semantic_memory", user_id="u", organization_id="o", client_id="c-1"
        )

        mock_provider.list.assert_awaited()
        mock_provider.find_using_named_query.assert_not_awaited()


# ---------------------------------------------------------------------------
# Batch tool lookup helpers (list_tools_by_names / list_tools_by_ids) and the
# ensure_base_tools_exist init-path fast path. Validates that the request path
# avoids the historical N+1 IPSR get_tool_by_name loop.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestToolManagerBatchLookups:
    async def test_list_tools_by_names_single_batch_call(self):
        row_a = {**_tool_row_dict("tool-aaaaaaaa"), "name": "name_a"}
        row_b = {**_tool_row_dict("tool-bbbbbbbb"), "name": "name_b"}
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row_a, row_b])
        actor = _mock_actor()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            out = await mgr.list_tools_by_names(["name_a", "name_b"], actor)

        mock_provider.find_using_named_query.assert_awaited_once()
        args, kwargs = mock_provider.find_using_named_query.await_args
        assert args[0] == "tools"
        assert args[1] == "tool_manager.list_tools_by_names"
        # NQ uses ``string_to_array(CAST(:names AS varchar), ',')``, so the
        # bind value is a comma-separated string, not a list.
        names_param = kwargs["params"]["names"]
        assert isinstance(names_param, str)
        assert set(names_param.split(",")) == {"name_a", "name_b"}
        assert kwargs["params"]["organizationId"] == actor.organization_id
        assert {t.name for t in out} == {"name_a", "name_b"}

    async def test_list_tools_by_names_dedupes_and_drops_empties(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        actor = _mock_actor()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            await mgr.list_tools_by_names(["a", "a", "", "b"], actor)

        sent = mock_provider.find_using_named_query.await_args[1]["params"]["names"]
        assert isinstance(sent, str)
        parts = sent.split(",")
        assert set(parts) == {"a", "b"}
        assert len(parts) == 2

    async def test_list_tools_by_names_empty_short_circuits(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            out = await mgr.list_tools_by_names([], _mock_actor())

        assert out == []
        mock_provider.find_using_named_query.assert_not_called()

    async def test_list_tools_by_names_chunks_above_200(self):
        from mirix.services.tool_manager import _BATCH_CHUNK_SIZE

        assert _BATCH_CHUNK_SIZE == 200
        names = [f"name_{i}" for i in range(250)]

        async def _fake_nq(_table, _nq_name, **kwargs):
            # NQ receives a comma-separated string; split to recover the chunk.
            chunk_names = kwargs["params"]["names"].split(",")
            return [{**_tool_row_dict(f"tool-{i:08x}"), "name": n} for i, n in enumerate(chunk_names)]

        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(side_effect=_fake_nq)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            out = await mgr.list_tools_by_names(names, _mock_actor())

        # 250 unique names => 2 chunks (200 + 50)
        assert mock_provider.find_using_named_query.await_count == 2
        chunk_sizes = sorted(
            len(call.kwargs["params"]["names"].split(","))
            for call in mock_provider.find_using_named_query.await_args_list
        )
        assert chunk_sizes == [50, 200]
        assert len(out) == 250

    async def test_list_tools_by_ids_single_batch_call(self):
        row = _tool_row_dict("tool-abcdef01")
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])
        actor = _mock_actor()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            out = await mgr.list_tools_by_ids(["tool-abcdef01", "tool-abcdef01"], actor)

        mock_provider.find_using_named_query.assert_awaited_once()
        args, kwargs = mock_provider.find_using_named_query.await_args
        assert args[1] == "tool_manager.list_tools_by_ids"
        # Comma-separated string (one id after dedup).
        assert kwargs["params"]["ids"] == "tool-abcdef01"
        assert kwargs["params"]["organizationId"] == actor.organization_id
        assert len(out) == 1


@pytest.mark.asyncio
class TestEnsureBaseToolsExist:
    async def test_first_call_does_one_batch_read_then_creates_missing(self):
        from mirix.constants import ALL_TOOLS

        # Pretend half of the expected base tools are already present.
        existing_names = list(ALL_TOOLS)[: len(ALL_TOOLS) // 2]
        existing_rows = [{**_tool_row_dict(f"tool-{i:08x}"), "name": n} for i, n in enumerate(existing_names)]

        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=existing_rows)

        # create_tool would normally hit the provider too; stub it.
        created_calls = []

        async def _fake_create_tool(self, pydantic_tool, actor):
            created_calls.append(pydantic_tool.name)
            return pydantic_tool

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            with patch.object(ToolManager, "create_tool", _fake_create_tool):
                mgr = _tool_mgr()
                mgr._base_tools_verified_org_ids = set()
                actor = _mock_actor()
                out = await mgr.ensure_base_tools_exist(actor=actor)

        # Exactly one batched read against the IPSR provider.
        assert mock_provider.find_using_named_query.await_count == 1
        nq_args = mock_provider.find_using_named_query.await_args
        assert nq_args[0][1] == "tool_manager.list_tools_by_names"

        # Only the missing tools were created (no updates for existing rows).
        expected_missing = [n for n in ALL_TOOLS if n not in existing_names]
        assert set(created_calls) == set(expected_missing)
        assert len(out) == len(expected_missing)

    async def test_second_call_same_org_is_zero_roundtrip(self):
        from mirix.constants import ALL_TOOLS

        all_rows = [{**_tool_row_dict(f"tool-{i:08x}"), "name": n} for i, n in enumerate(ALL_TOOLS)]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=all_rows)

        async def _no_create(self, *_a, **_kw):  # pragma: no cover
            raise AssertionError("create_tool must not be called when nothing is missing")

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            with patch.object(ToolManager, "create_tool", _no_create):
                mgr = _tool_mgr()
                mgr._base_tools_verified_org_ids = set()
                actor = _mock_actor()
                await mgr.ensure_base_tools_exist(actor=actor)
                first_call_count = mock_provider.find_using_named_query.await_count

                # Second invocation for the same org_id must be a no-op.
                out = await mgr.ensure_base_tools_exist(actor=actor)

        assert out == []
        assert mock_provider.find_using_named_query.await_count == first_call_count

    async def test_upsert_base_tools_memoized_per_org(self):
        from mirix.constants import ALL_TOOLS

        existing_rows = [{**_tool_row_dict(f"tool-{i:08x}"), "name": n} for i, n in enumerate(ALL_TOOLS)]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=existing_rows)
        mock_provider.update = AsyncMock(side_effect=lambda *_a, **_kw: existing_rows[0])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _tool_mgr()
            mgr._base_tools_verified_org_ids = set()
            actor = _mock_actor()
            # First call performs full per-tool create_or_update_tool walk.
            await mgr.upsert_base_tools(actor=actor)
            calls_after_first = mock_provider.find_using_named_query.await_count
            # Second call hits the memoization guard.
            out_second = await mgr.upsert_base_tools(actor=actor)

        assert out_second == []
        assert mock_provider.find_using_named_query.await_count == calls_after_first


@pytest.mark.asyncio
class TestAgentManagerBatchedToolResolution:
    """Asserts agent_manager paths use the batched ToolManager helpers and
    issue zero per-name get_tool_by_name calls (the original N+1 shape)."""

    async def test_create_agent_uses_list_tools_by_names(self):
        from mirix.schemas.agent import CreateAgent

        mgr = _agent_mgr()
        # Inject a mocked tool_manager that records calls.
        mgr.tool_manager = MagicMock()
        tool_a = PydanticTool.model_construct(id="tool-aaaaaaaa", name="send_message")
        mgr.tool_manager.list_tools_by_names = AsyncMock(return_value=[tool_a])
        mgr.tool_manager.get_tool_by_name = AsyncMock()

        # Short-circuit the underlying _create_agent (DB-touching).
        mgr._create_agent = AsyncMock(return_value=_minimal_agent_state())

        actor = _mock_actor()
        await mgr.create_agent(
            agent_create=CreateAgent(
                name="x",
                agent_type=AgentType.chat_agent,
                llm_config=LLMConfig(model="m", model_endpoint_type="openai", context_window=8000),
                embedding_config=EmbeddingConfig.default_config("text-embedding-3-small"),
                include_base_tools=False,
                tools=["send_message"],
            ),
            actor=actor,
        )

        mgr.tool_manager.list_tools_by_names.assert_awaited_once()
        mgr.tool_manager.get_tool_by_name.assert_not_awaited()

    async def test_update_agent_tools_uses_one_batched_lookup(self):
        from mirix.schemas.agent import AgentState
        from mirix.schemas.tool import Tool

        mgr = _agent_mgr()
        mgr.tool_manager = MagicMock()
        mgr.tool_manager.list_tools_by_names = AsyncMock(
            return_value=[Tool.model_construct(id="tool-11111111", name="new_tool")]
        )
        mgr.tool_manager.get_tool_by_name = AsyncMock()
        mgr.get_agent_by_id = AsyncMock(
            return_value=AgentState.model_construct(
                id="agent-1",
                name="a",
                system="sys",
                agent_type=AgentType.chat_agent,
                llm_config=LLMConfig(model="m", model_endpoint_type="openai", context_window=8000),
                embedding_config=EmbeddingConfig.default_config("text-embedding-3-small"),
                organization_id="org-1",
                tools=[Tool.model_construct(id="tool-22222222", name="old_tool")],
            )
        )
        mgr.update_agent = AsyncMock()
        mgr.update_system_prompt = AsyncMock()

        await mgr.update_agent_tools_and_system_prompts(agent_id="agent-1", actor=_mock_actor())

        # Exactly one batched lookup covering union(new, removed); the
        # per-name N+1 path must not be used.
        assert mgr.tool_manager.list_tools_by_names.await_count == 1
        mgr.tool_manager.get_tool_by_name.assert_not_awaited()
