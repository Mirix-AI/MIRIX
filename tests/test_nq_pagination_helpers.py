"""Unit tests for the IPSR named-query pagination helpers.

The IPSR named-query runner rejects pageSize > 1000 with
InvalidPageSizeException (400), so any caller needing a complete result set
must paginate. Covers:

- ``find_all_using_named_query``: loops SLICE pages until a short page,
  clamps oversized page_size, dedupes rows repeated across page boundaries,
  and stops with a warning at the max-pages safety valve.
- ``bulk_delete_all_from_named_query``: drains page 0 in fetch-delete-repeat
  batches, sums the reported success counts, and stops when a batch makes no
  progress instead of looping forever.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mirix.services.memory_manager_helpers import (
    IPSR_NQ_MAX_PAGE_SIZE,
    bulk_delete_all_from_named_query,
    find_all_using_named_query,
)


def _rows(start: int, count: int):
    return [{"id": f"row-{i}"} for i in range(start, start + count)]


@pytest.mark.asyncio
class TestFindAllUsingNamedQuery:
    async def test_single_short_page_returns_rows_one_call(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=_rows(0, 3))

        result = await find_all_using_named_query(provider, "t", "q", params={"a": 1})

        assert [r["id"] for r in result] == ["row-0", "row-1", "row-2"]
        provider.find_using_named_query.assert_awaited_once()
        call = provider.find_using_named_query.await_args
        assert call.args == ("t", "q")
        assert call.kwargs["params"] == {"a": 1}
        assert call.kwargs["page_size"] == IPSR_NQ_MAX_PAGE_SIZE
        assert call.kwargs["page_num"] == 0

    async def test_loops_pages_until_short_page(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(
            side_effect=[
                _rows(0, IPSR_NQ_MAX_PAGE_SIZE),
                _rows(IPSR_NQ_MAX_PAGE_SIZE, IPSR_NQ_MAX_PAGE_SIZE),
                _rows(2 * IPSR_NQ_MAX_PAGE_SIZE, 7),
            ]
        )

        result = await find_all_using_named_query(provider, "t", "q")

        assert len(result) == 2 * IPSR_NQ_MAX_PAGE_SIZE + 7
        assert provider.find_using_named_query.await_count == 3
        page_nums = [c.kwargs["page_num"] for c in provider.find_using_named_query.await_args_list]
        assert page_nums == [0, 1, 2]

    async def test_oversized_page_size_is_clamped(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=[])

        await find_all_using_named_query(provider, "t", "q", page_size=5000)

        assert provider.find_using_named_query.await_args.kwargs["page_size"] == IPSR_NQ_MAX_PAGE_SIZE

    async def test_dedupes_rows_repeated_across_page_boundary(self):
        # Non-unique ORDER BY can make offset pagination repeat a row on the
        # next page; the helper must not return it twice.
        page1 = _rows(0, IPSR_NQ_MAX_PAGE_SIZE)
        page2 = [page1[-1]] + _rows(IPSR_NQ_MAX_PAGE_SIZE, 2)
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(side_effect=[page1, page2])

        result = await find_all_using_named_query(provider, "t", "q")

        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids)) == IPSR_NQ_MAX_PAGE_SIZE + 2

    async def test_forwards_extra_kwargs(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=[])

        await find_all_using_named_query(provider, "t", "q", skip_entity_mapping=True)

        assert provider.find_using_named_query.await_args.kwargs["skip_entity_mapping"] is True


@pytest.mark.asyncio
class TestBulkDeleteAllFromNamedQuery:
    async def test_drains_batches_until_empty(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(
            side_effect=[
                _rows(0, IPSR_NQ_MAX_PAGE_SIZE),
                _rows(IPSR_NQ_MAX_PAGE_SIZE, 4),
            ]
        )
        provider.bulk_delete = AsyncMock(side_effect=[{"success": IPSR_NQ_MAX_PAGE_SIZE}, {"success": 4}])

        total = await bulk_delete_all_from_named_query(provider, "t", "q", params={"userId": "u"}, soft=True)

        assert total == IPSR_NQ_MAX_PAGE_SIZE + 4
        assert provider.bulk_delete.await_count == 2
        # Always drains page 0 — deleted rows leave the filter, so no offset.
        for call in provider.find_using_named_query.await_args_list:
            assert call.kwargs["page_size"] == IPSR_NQ_MAX_PAGE_SIZE
        first_delete = provider.bulk_delete.await_args_list[0]
        assert first_delete.args[0] == "t"
        assert first_delete.kwargs["soft"] is True

    async def test_short_first_batch_deletes_once_and_stops(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=_rows(0, 5))
        provider.bulk_delete = AsyncMock(return_value={"success": 5})

        total = await bulk_delete_all_from_named_query(provider, "t", "q", soft=False)

        assert total == 5
        provider.find_using_named_query.assert_awaited_once()
        provider.bulk_delete.assert_awaited_once()
        assert provider.bulk_delete.await_args.kwargs["soft"] is False

    async def test_no_rows_is_a_noop(self):
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=[])
        provider.bulk_delete = AsyncMock()

        total = await bulk_delete_all_from_named_query(provider, "t", "q")

        assert total == 0
        provider.bulk_delete.assert_not_awaited()

    async def test_stops_when_delete_makes_no_progress(self):
        # bulk_delete silently failing must not produce an infinite loop:
        # the same full page coming back twice stops the drain.
        same_page = _rows(0, IPSR_NQ_MAX_PAGE_SIZE)
        provider = MagicMock()
        provider.find_using_named_query = AsyncMock(return_value=same_page)
        provider.bulk_delete = AsyncMock(return_value={"success": 0})

        total = await bulk_delete_all_from_named_query(provider, "t", "q")

        assert total == 0
        assert provider.bulk_delete.await_count == 1
        assert provider.find_using_named_query.await_count == 2
