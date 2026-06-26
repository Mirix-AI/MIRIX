"""Unit tests for the provenance manager NQ routing.

Covers:
- ``get_citations_for_memories`` routes through the
  ``memory_citation_manager.get_citations_for_memory`` named query.
- Well-formed rows hydrate into ``PydanticMemoryCitation`` directly.
- Rows missing ``memory_source_id`` now fail loudly (no defensive empty-string
  synthesis) so contract violations surface immediately instead of producing
  unclickable citations downstream.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation
from mirix.services.memory_citation_manager import MemoryCitationManager


def _well_formed_row() -> Dict[str, Any]:
    """A row as the named-query path produces it — fully populated."""
    return {
        "id": "cit-abcdef01-2345-6789-abcd-ef0123456789",
        "memory_source_id": "src-abcdef01-2345-6789-abcd-ef0123456789",
        "memory_type": "episodic",
        "memory_id": "ep-abcdef01",
        "citation_type": "created",
        "occurred_at": None,
        "external_thread_id": None,
    }


class TestRowHydration:
    def test_well_formed_row_constructs_directly(self):
        citation = PydanticMemoryCitation(**_well_formed_row())
        assert citation.memory_source_id == "src-abcdef01-2345-6789-abcd-ef0123456789"
        assert citation.memory_type == "episodic"

    def test_missing_memory_source_id_raises(self):
        row = _well_formed_row()
        del row["memory_source_id"]
        with pytest.raises(Exception):
            PydanticMemoryCitation(**row)

    def test_explicit_none_memory_source_id_raises(self):
        row = _well_formed_row()
        row["memory_source_id"] = None
        with pytest.raises(Exception):
            PydanticMemoryCitation(**row)


@pytest.mark.asyncio
class TestGetCitationsForMemoriesNQ:
    async def test_routes_through_named_query_not_provider_list(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[_well_formed_row()])
        mock_provider.list = AsyncMock(
            side_effect=AssertionError("get_citations_for_memories must use find_using_named_query")
        )

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            result = await mgr.get_citations_for_memories([("episodic", "ep-abcdef01")])

        mock_provider.find_using_named_query.assert_awaited_once()
        call = mock_provider.find_using_named_query.await_args
        assert call.args[0] == "memory_citations"
        assert call.args[1] == "memory_citation_manager.get_citations_for_memory"
        assert call.kwargs["params"] == {
            "memoryType": "episodic",
            "memoryId": "ep-abcdef01",
        }
        assert ("episodic", "ep-abcdef01") in result
        assert len(result[("episodic", "ep-abcdef01")]) == 1
        assert result[("episodic", "ep-abcdef01")][0].memory_source_id == "src-abcdef01-2345-6789-abcd-ef0123456789"

    async def test_fans_out_per_memory_key(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[_well_formed_row()])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            await mgr.get_citations_for_memories([("episodic", "ep-1"), ("semantic", "sem-1"), ("core", "block-1")])

        assert mock_provider.find_using_named_query.await_count == 3

    async def test_row_missing_fk_now_raises(self):
        """The defensive empty-string synthesis was removed — rows without
        memory_source_id should fail loudly so contract violations are
        immediately visible instead of producing unclickable citations."""
        bad_row = _well_formed_row()
        del bad_row["memory_source_id"]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[bad_row])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            with pytest.raises(Exception):
                await mgr.get_citations_for_memories([("episodic", "ep-abcdef01")])

    async def test_empty_memory_keys_returns_empty(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            result = await mgr.get_citations_for_memories([])

        assert result == {}
        mock_provider.find_using_named_query.assert_not_awaited()
