"""
Tests for the recent/relevant union in the save-flow prompt builder
(VEPAGE-1178): ``Agent._merge_recent_into_relevant`` and
``Agent._fetch_recent_indexing_lag_window``.

The owning sub-agent unions just-written ("recent", from the Relational
recent-window) rows into the ranked ("relevant", from Search) list, de-duped
by id, in a single list.

Usage:
    pytest tests/test_recent_relevant_union.py -v
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.agent.agent import Agent


def _item(id_):
    return SimpleNamespace(id=id_)


class TestMergeRecentIntoRelevant:
    def test_appends_recent_only_rows_after_relevant(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("c"), _item("d")]

        merged = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["a", "b", "c", "d"]

    def test_dedups_recent_rows_already_in_relevant(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("b"), _item("c")]  # "b" overlaps

        merged = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["a", "b", "c"]

    def test_preserves_relevant_ranking_order(self):
        relevant = [_item("c"), _item("a"), _item("b")]  # ranked, not sorted
        recent = []

        merged = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["c", "a", "b"]

    def test_empty_recent_returns_relevant_unchanged(self):
        relevant = [_item("a")]
        merged = Agent._merge_recent_into_relevant(relevant, [])
        assert [m.id for m in merged] == ["a"]

    def test_empty_relevant_returns_all_recent(self):
        recent = [_item("x"), _item("y")]
        merged = Agent._merge_recent_into_relevant([], recent)
        assert [m.id for m in merged] == ["x", "y"]

    def test_all_recent_overlap_appends_nothing(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("a"), _item("b")]
        merged = Agent._merge_recent_into_relevant(relevant, recent)
        assert [m.id for m in merged] == ["a", "b"]


def _row(id_):
    return {"id": id_, "name": "n", "summary": "s"}


@pytest.mark.asyncio
class TestFetchRecentIndexingLagWindow:
    """The recent-window fetch queries the Relational provider within
    HYBRID_READ_WINDOW_SECONDS, and short-circuits to [] when running PG-only
    (either provider unregistered)."""

    def _agent(self):
        agent = Agent.__new__(Agent)
        agent.user = SimpleNamespace(id="u1", organization_id="org1")
        return agent

    async def test_returns_empty_when_search_provider_missing(self):
        agent = self._agent()
        with (
            patch("mirix.database.search_provider.get_search_provider", return_value=None),
            patch("mirix.database.relational_provider.get_relational_provider", return_value=MagicMock()),
        ):
            out = await agent._fetch_recent_indexing_lag_window("semantic_memory", SimpleNamespace)
        assert out == []

    async def test_returns_empty_when_relational_provider_missing(self):
        agent = self._agent()
        with (
            patch("mirix.database.search_provider.get_search_provider", return_value=MagicMock()),
            patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
        ):
            out = await agent._fetch_recent_indexing_lag_window("semantic_memory", SimpleNamespace)
        assert out == []

    async def test_queries_relational_recent_window_and_wraps_rows(self):
        agent = self._agent()
        rp = MagicMock()
        rp.list = AsyncMock(return_value=[_row("a"), _row("b")])
        with (
            patch("mirix.database.search_provider.get_search_provider", return_value=MagicMock()),
            patch("mirix.database.relational_provider.get_relational_provider", return_value=rp),
        ):
            out = await agent._fetch_recent_indexing_lag_window("semantic_memory", SimpleNamespace)

        rp.list.assert_awaited_once()
        call = rp.list.await_args
        assert call[0][0] == "semantic_memory"
        assert call[1]["user_id"] == "u1"
        assert call[1]["organization_id"] == "org1"
        tr = call[1]["time_range"]
        assert "updated_at__gte" in tr and "created_at__gte" in tr
        assert call[1]["time_range_or_null_updated"] is True
        assert [o.id for o in out] == ["a", "b"]

    async def test_relational_error_propagates(self):
        agent = self._agent()
        rp = MagicMock()
        rp.list = AsyncMock(side_effect=RuntimeError("relational down"))
        with (
            patch("mirix.database.search_provider.get_search_provider", return_value=MagicMock()),
            patch("mirix.database.relational_provider.get_relational_provider", return_value=rp),
        ):
            with pytest.raises(RuntimeError, match="relational down"):
                await agent._fetch_recent_indexing_lag_window("semantic_memory", SimpleNamespace)
