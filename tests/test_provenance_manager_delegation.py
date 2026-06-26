"""
Tests that the 3 provenance managers (memory_source, source_message, memory_citation)
delegate to the relational provider when registered, and fall back to ORM/cache
when it is absent.

Run:
    pytest tests/test_provenance_manager_delegation.py -v
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.errors import ProviderConflictError as _ProviderConflictError
from mirix.errors import ProviderPermanentError as _ProviderPermanentError
from mirix.errors import ProviderTransientError as _ProviderTransientError
from mirix.schemas.client import Client as PydanticClient
from mirix.services.memory_citation_manager import MemoryCitationManager
from mirix.services.memory_source_manager import MemorySourceManager
from mirix.services.source_message_manager import SourceMessageManager


def _memory_source_mgr() -> MemorySourceManager:
    m = MemorySourceManager.__new__(MemorySourceManager)
    m.session_maker = MagicMock()
    return m


def _source_message_mgr() -> SourceMessageManager:
    m = SourceMessageManager.__new__(SourceMessageManager)
    m.session_maker = MagicMock()
    return m


def _memory_citation_mgr() -> MemoryCitationManager:
    m = MemoryCitationManager.__new__(MemoryCitationManager)
    m.session_maker = MagicMock()
    return m


def _mock_actor() -> MagicMock:
    a = MagicMock(spec=PydanticClient)
    a.id = "client-1"
    a.write_scope = "scope-a"
    a.read_scopes = ["scope-a"]
    return a


def _memory_source_row(**overrides) -> dict:
    base = {
        "id": "src-aaaaaaaa",
        "client_id": "client-1",
        "user_id": "user-1",
        "organization_id": "org-1",
        "source_type": "conversation",
        "external_id": "ext-1",
        "external_thread_id": "thr-1",
        "source_system": None,
        "source_metadata": None,
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "summary": None,
        "summary_source": None,
        "processing_complete": False,
        "batch_hash": None,
        "filter_tags": {"scope": "scope-a"},
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------- MemorySourceManager ----------


class TestMemorySourceManagerDelegation:
    @pytest.mark.asyncio
    async def test_create_delegates_to_provider(self):
        row = _memory_source_row()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                actor=_mock_actor(),
                user_id="user-1",
                organization_id="org-1",
                external_id="ext-1",
            )
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "memory_sources"
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_create_same_id_conflict_returns_the_row(self):
        """A conflict whose existing row has the SAME id as the requested id is a
        retry / redelivery of *this* submission (the id PK is what conflicted).

        The manager returns that row so the caller resumes it. Finalize keys off
        the unchanged queue id, so marking it complete later lands on this row.
        Completion is the caller's decision (it reads processing_complete); the
        manager just hands back the canonical row.
        """
        existing = _memory_source_row(id="src-aaaaaaaa", processing_complete=False)
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_ProviderConflictError("uq_test"))
        mock_provider.find_using_named_query = AsyncMock(return_value=[existing])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.create(
                memory_source_id="src-aaaaaaaa",
                actor=_mock_actor(),
                user_id="user-1",
                organization_id="org-1",
                external_id="ext-1",
            )
            assert out is not None
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_create_different_id_conflict_returns_none(self):
        """A conflict whose existing row has a DIFFERENT id is NOT a retry — it is
        a separate duplicate submission that another worker owns (the conflict is
        on the external_id / batch_hash index, not the id PK).

        The manager returns None ("skip") regardless of the existing row's
        completion state: a completed row is a genuine duplicate, and an
        incomplete row is owned by the other submission, which finishes it on its
        own (or via same-id redelivery). Resuming a different-id row here would
        race the owner and mis-target finalize (which keys off this submission's
        id), so we never do it.
        """
        for complete in (True, False):
            existing = _memory_source_row(id="src-bbbbbbbb", processing_complete=complete)
            mock_provider = MagicMock()
            mock_provider.create = AsyncMock(side_effect=_ProviderConflictError("uq_test"))
            mock_provider.find_using_named_query = AsyncMock(return_value=[existing])

            with patch(
                "mirix.database.relational_provider.get_relational_provider",
                return_value=mock_provider,
            ):
                mgr = _memory_source_mgr()
                out = await mgr.create(
                    memory_source_id="src-aaaaaaaa",
                    actor=_mock_actor(),
                    user_id="user-1",
                    organization_id="org-1",
                    external_id="ext-1",
                )
                assert out is None, f"different-id conflict (complete={complete})"

    @pytest.mark.asyncio
    async def test_get_by_id_delegates_to_provider(self):
        row = _memory_source_row()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.get_by_id("src-aaaaaaaa")
            mock_provider.read.assert_awaited_once_with("memory_sources", "src-aaaaaaaa")
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.get_by_id("missing")
            assert out is None

    @pytest.mark.asyncio
    async def test_mark_processing_complete_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            await mgr.mark_processing_complete("src-aaaaaaaa")
            mock_provider.update.assert_awaited_once_with(
                "memory_sources", "src-aaaaaaaa", {"processing_complete": True}
            )

    @pytest.mark.asyncio
    async def test_update_summary_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            await mgr.update_summary("src-aaaaaaaa", "a summary", "generated")
            mock_provider.update.assert_awaited_once_with(
                "memory_sources",
                "src-aaaaaaaa",
                {"summary": "a summary", "summary_source": "generated"},
            )

    @pytest.mark.asyncio
    async def test_list_sources_delegates_to_provider(self):
        # VEPAGE-1144: list_sources_admin NQ orders server-side
        # (occurredAt DESC NULLS LAST, ipsrcreatedon DESC), so the manager
        # forwards rows without any client-side sorting.
        rows = [
            _memory_source_row(id="src-aaaaaaa2", occurred_at="2026-05-02T00:00:00+00:00"),
            _memory_source_row(id="src-aaaaaaa1", occurred_at="2026-05-01T00:00:00+00:00"),
        ]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            page = await mgr.list_sources(organization_id="org-1", user_id="user-1", limit=10)
            mock_provider.find_using_named_query.assert_awaited_once()
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "memory_sources"
            assert args[1] == "memory_source_manager.list_sources_admin"
            assert kwargs["params"]["organizationId"] == "org-1"
            assert kwargs["params"]["userId"] == "user-1"
            # First call → page 0; fetch_size = limit + 1 = 11.
            assert kwargs["page_num"] == 0
            assert kwargs["page_size"] == 11
            # Order preserved as returned by the NQ.
            assert [item.id for item in page.items] == ["src-aaaaaaa2", "src-aaaaaaa1"]
            # Only 2 rows back, page size 10 → no more pages.
            assert page.has_more is False
            assert page.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sources_requires_organization_id_on_ipsr_path(self):
        """VEPAGE-1144: the NQ uses organizationId as its access predicate;
        calling without one when an IPSR provider is registered should fail
        loudly rather than silently issue an unscoped query."""
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            with pytest.raises(ValueError, match="organization_id"):
                await mgr.list_sources(user_id="user-1", limit=10)
            mock_provider.find_using_named_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_list_sources_pagination_full_page_advances_cursor(self):
        """When the provider returns a full page + 1 (the sentinel row
        signalling more pages exist), has_more=True and next_cursor
        decodes to the next page_num."""
        # limit=2 → fetch_size=3. Provider returns 3 rows → full page +
        # sentinel → has_more=True, items trimmed to 2.
        rows = [
            _memory_source_row(id="src-aaaaaaa1", occurred_at="2026-05-03T00:00:00+00:00"),
            _memory_source_row(id="src-aaaaaaa2", occurred_at="2026-05-02T00:00:00+00:00"),
            _memory_source_row(id="src-aaaaaaa3", occurred_at="2026-05-01T00:00:00+00:00"),
        ]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            page = await mgr.list_sources(organization_id="org-1", limit=2)
            assert len(page.items) == 2
            assert page.has_more is True
            assert page.next_cursor is not None
            # Round-trip the cursor — second call must request page_num=1.
            await mgr.list_sources(organization_id="org-1", limit=2, cursor=page.next_cursor)
            second_call_kwargs = mock_provider.find_using_named_query.call_args.kwargs
            assert second_call_kwargs["page_num"] == 1

    @pytest.mark.asyncio
    async def test_list_sources_cursor_malformed_resets_to_page_zero(self):
        """A garbage cursor should not 500 the request; it should start
        over at page 0."""
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            await mgr.list_sources(organization_id="org-1", limit=10, cursor="not-base64-at-all!!!")
            kwargs = mock_provider.find_using_named_query.call_args.kwargs
            assert kwargs["page_num"] == 0


# ---------- SourceMessageManager ----------


class TestSourceMessageManagerDelegation:
    @pytest.mark.asyncio
    async def test_bulk_insert_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value={})

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            inserted = await mgr.bulk_insert(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                memory_source_id="src-1",
            )
            assert inserted == 2
            assert mock_provider.create.await_count == 2

    @pytest.mark.asyncio
    async def test_bulk_insert_treats_conflicts_as_no_op(self):
        # First create succeeds, second raises a "conflict" — total inserted = 1.
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=[{}, _ProviderConflictError("uq_test")])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            inserted = await mgr.bulk_insert(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                memory_source_id="src-1",
            )
            assert inserted == 1

    @pytest.mark.asyncio
    async def test_get_messages_by_source_id_delegates_to_provider(self):
        rows = [
            {
                "id": "smsg-aaaaaaa1",
                "memory_source_id": "src-aaaaaaaa",
                "role": "user",
                "content": {"text": "hi"},
                "sequence_num": 0,
                "content_hash": "h0",
            },
            {
                "id": "smsg-aaaaaaa2",
                "memory_source_id": "src-aaaaaaaa",
                "role": "assistant",
                "content": {"text": "hello"},
                "sequence_num": 1,
                "content_hash": "h1",
            },
        ]
        mock_provider = MagicMock()
        # VEPAGE-1107: rewritten to use a named query (ordering / projection).
        mock_provider.find_using_named_query = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            page = await mgr.get_messages_by_source_id("src-aaaaaaaa")
            assert [m.id for m in page.items] == ["smsg-aaaaaaa1", "smsg-aaaaaaa2"]


# ---------- MemoryCitationManager ----------


def _citation_row(**overrides) -> dict:
    base = {
        "id": "cit-aaaaaaaa",
        "memory_source_id": "src-aaaaaaaa",
        "memory_type": "episodic",
        "memory_id": "ep-1",
        "external_thread_id": None,
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "citation_type": "created",
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestMemoryCitationManagerDelegation:
    @pytest.mark.asyncio
    async def test_create_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=_citation_row())

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            mock_provider.create.assert_awaited_once()
            assert out is not None
            assert out.memory_type == "episodic"

    @pytest.mark.asyncio
    async def test_create_returns_none_on_conflict(self):
        # L3 dedup: provider raises on duplicate (memory_source_id, memory_type, memory_id).
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_ProviderConflictError("uq_test"))

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            assert out is None

    @pytest.mark.asyncio
    async def test_check_exists_returns_true_when_provider_returns_row(self):
        # VEPAGE-1107: routes through find_existing_citation named query.
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[_citation_row()])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.check_exists("src-1", "episodic", "ep-1") is True

    @pytest.mark.asyncio
    async def test_check_exists_returns_false_when_provider_returns_empty(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.check_exists("src-1", "episodic", "ep-1") is False

    @pytest.mark.asyncio
    async def test_get_max_occurred_at_takes_max_across_rows(self):
        # VEPAGE-1107: server-side MAX via max_occurred_at_for_memory NQ.
        # The NQ projects a single scalar column ``max_occurred_at``.
        rows = [
            {"max_occurred_at": "2026-05-01T00:00:00+00:00"},
            {"max_occurred_at": "2026-05-03T00:00:00+00:00"},
            {"max_occurred_at": "2026-05-02T00:00:00+00:00"},
        ]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_max_occurred_at("episodic", "ep-1")
            assert out is not None
            assert out == datetime(2026, 5, 3, 0, 0, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_get_max_occurred_at_returns_none_when_no_rows(self):
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.get_max_occurred_at("episodic", "ep-1") is None

    @pytest.mark.asyncio
    async def test_get_citations_for_memory_delegates_to_provider(self):
        # VEPAGE-1107: routes through get_citations_for_memory NQ.
        rows = [_citation_row()]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_citations_for_memory("episodic", "ep-1")
            assert len(out) == 1
            assert out[0].memory_id == "ep-1"

    @pytest.mark.asyncio
    async def test_get_citations_for_memories_groups_by_key(self):
        # VEPAGE-1107: per-key fan-out via get_citations_for_memory NQ.
        ep_row = _citation_row(id="cit-aaaaaaa1", memory_type="episodic", memory_id="ep-1")
        sem_row = _citation_row(id="cit-aaaaaaa2", memory_type="semantic", memory_id="sem-1")
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(side_effect=[[ep_row], [sem_row]])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_citations_for_memories([("episodic", "ep-1"), ("semantic", "sem-1")])
            assert ("episodic", "ep-1") in out
            assert ("semantic", "sem-1") in out
            assert len(out[("episodic", "ep-1")]) == 1
            assert len(out[("semantic", "sem-1")]) == 1


# ---------------------------------------------------------------------------
# T6 — best-effort citation/source-message write semantics
# ---------------------------------------------------------------------------


class _FakeServerError(_ProviderTransientError):
    """Stand-in for an IPSR ServerError (5xx).

    Post-VEPAGE-1251 §5.6, transient failures arriving at the MIRIX retry
    helper are already translated to ProviderTransientError by the ECMS
    provider boundary. The fake subclasses that type so the helper's
    isinstance check classifies it as transient.
    """

    def __init__(self, status_code: int = 503):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeUnauthorizedError(_ProviderPermanentError):
    """Stand-in for an IPSR UnauthorizedError (401/403). Permanent — must propagate."""

    def __init__(self):
        super().__init__("HTTP 401")
        self.status_code = 401


class TestBestEffortCitationWrite:
    """T6 (VEPAGE-1026): MemoryCitationManager.create classifies provider errors."""

    @pytest.mark.asyncio
    async def test_returns_none_silently_on_conflict(self):
        # A conflict (unique constraint hint in message) → return None, no retry.
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_ProviderConflictError("uq_test"))
        with (
            patch(
                "mirix.database.relational_provider.get_relational_provider",
                return_value=mock_provider,
            ),
            patch("mirix.observability.skip_spans.emit_idempotency_skip_span") as skip_span,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            assert out is None
            assert mock_provider.create.await_count == 1  # no retry on conflict
            skip_span.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_with_skip_span_on_transient(self):
        """An exhausted ProviderTransientError (from the provider's own
        inner-retry tier exhausting) is classified transient at the
        manager; the manager logs + skip-spans + returns None so the
        caller's memory write still stands. The manager itself doesn't
        retry — that's the provider's job."""
        mock_provider = MagicMock()
        # ProviderTransientError stands in for "provider's retry budget
        # exhausted." `_FakeServerError` is a subclass of it in this
        # test file.
        mock_provider.create = AsyncMock(side_effect=_FakeServerError(503))
        with (
            patch(
                "mirix.database.relational_provider.get_relational_provider",
                return_value=mock_provider,
            ),
            patch("mirix.observability.skip_spans.emit_idempotency_skip_span") as skip_span,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            assert out is None
            assert mock_provider.create.await_count == 1  # manager doesn't retry
            skip_span.assert_called_once()
            kwargs = skip_span.call_args.kwargs
            assert kwargs["reason"] == "citation-write-failed"

    @pytest.mark.asyncio
    async def test_propagates_permanent_error(self):
        # UnauthorizedError → no retry, exception propagates.
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_FakeUnauthorizedError())
        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            with pytest.raises(_FakeUnauthorizedError):
                await mgr.create(
                    memory_source_id="src-1",
                    memory_type="episodic",
                    memory_id="ep-1",
                    citation_type="created",
                )
            assert mock_provider.create.await_count == 1  # no retry


class TestBestEffortSourceMessageWrite:
    """T6 (VEPAGE-1026): SourceMessageManager.bulk_insert classifies per-row errors."""

    @pytest.mark.asyncio
    async def test_per_row_conflict_skipped_transient_retried_permanent_raises(self):
        # Row 0: success.  Row 1: conflict (silent skip).  Row 2: permanent (raise).
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(
            side_effect=[
                {},  # row 0 success
                _ProviderConflictError("uq_test"),  # row 1 conflict
                _FakeUnauthorizedError(),  # row 2 permanent
            ]
        )
        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            with pytest.raises(_FakeUnauthorizedError):
                await mgr.bulk_insert(
                    messages=[
                        {"role": "user", "content": "a"},
                        {"role": "user", "content": "b"},
                        {"role": "user", "content": "c"},
                    ],
                    memory_source_id="src-1",
                )

    @pytest.mark.asyncio
    async def test_transient_row_emits_skip_span(self):
        """Row 0 fails with ProviderTransientError (provider's retry tier
        already exhausted); manager logs + skip-spans + continues with
        row 1. Manager itself doesn't retry — that's the provider's job."""
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(
            side_effect=[
                _FakeServerError(503),  # row 0: provider exhausted -> skip
                {},  # row 1: success
            ]
        )
        with (
            patch(
                "mirix.database.relational_provider.get_relational_provider",
                return_value=mock_provider,
            ),
            patch("mirix.observability.skip_spans.emit_idempotency_skip_span") as skip_span,
        ):
            mgr = _source_message_mgr()
            inserted = await mgr.bulk_insert(
                messages=[
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ],
                memory_source_id="src-1",
            )
            assert inserted == 1  # only row 1 succeeded
            assert mock_provider.create.await_count == 2  # one call per row, no retries
            assert skip_span.call_count == 1
            assert skip_span.call_args.kwargs["reason"] == "source-message-write-failed"


class TestBestEffortMemorySourceWrite:
    """T6 (VEPAGE-1026): MemorySourceManager.create retries transient + propagates permanent."""

    @pytest.mark.asyncio
    async def test_raises_on_exhausted_transient(self):
        """An exhausted ProviderTransientError from the provider's retry
        tier propagates out of the memory_source manager (unlike citation
        / source-message writes which best-effort skip). Manager itself
        doesn't retry."""
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_FakeServerError(503))
        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            with pytest.raises(_FakeServerError):
                await mgr.create(
                    memory_source_id="src-new",
                    actor=_mock_actor(),
                    user_id="user-1",
                    organization_id="org-1",
                    external_id="ext-1",
                )
            assert mock_provider.create.await_count == 1  # manager doesn't retry

    @pytest.mark.asyncio
    async def test_propagates_permanent_error_immediately(self):
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=_FakeUnauthorizedError())
        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            with pytest.raises(_FakeUnauthorizedError):
                await mgr.create(
                    memory_source_id="src-new",
                    actor=_mock_actor(),
                    user_id="user-1",
                    organization_id="org-1",
                    external_id="ext-1",
                )
            assert mock_provider.create.await_count == 1
