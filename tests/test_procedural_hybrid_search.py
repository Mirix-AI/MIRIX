"""Unit tests for the EverOS-aligned procedural HYBRID retrieval path.

Covers the two pieces most prone to silent divergence from EverOS/everalgo:

  1. `_rrf_fuse` — the Reciprocal Rank Fusion math. Must be rank-based (raw
     lane scores discarded), 1-based, unweighted, k=60, dedup-by-id, and the
     cross-lane-agreement property (an item ranked in BOTH lanes outranks an
     item that tops only one lane).
  2. `_hybrid_search_for_user` orchestration — runs exactly the bm25 + embedding
     lanes, over-fetches `limit * SKILL_HYBRID_RECALL_MULTIPLIER` per lane,
     applies the similarity_threshold quality floor to the DENSE lane only, and
     falls the dense lane's search_field back to "description" when the caller's
     field is not embeddable.

Both are exercised WITHOUT a database: `_rrf_fuse` is pure, and the
orchestration test monkeypatches the recursive `list_procedures` lane calls, so
the manager's session_maker is never touched.
"""

from types import SimpleNamespace

import pytest

from mirix.constants import SKILL_HYBRID_RECALL_MULTIPLIER, SKILL_HYBRID_RRF_K
from mirix.services.procedural_memory_manager import (
    ProceduralMemoryManager,
    _rrf_fuse,
)


def _item(item_id):
    """Minimal stand-in: _rrf_fuse only reads `.id`."""
    return SimpleNamespace(id=item_id)


class TestRrfFuse:
    def test_k_is_canonical_60(self):
        assert SKILL_HYBRID_RRF_K == 60

    def test_rank_based_unweighted_order(self):
        # k=60, 1-based: score(d) = sum 1/(60+rank).
        sparse = [_item("a"), _item("b"), _item("c")]
        dense = [_item("b"), _item("a"), _item("d")]
        fused = _rrf_fuse([sparse, dense], k=60, limit=10)
        # a: 1/61 + 1/62 ; b: 1/62 + 1/61 (tie with a); c: 1/63 ; d: 1/63.
        # a,b (in both) >> c,d (one lane each). Ties broken by first-seen order.
        assert [i.id for i in fused] == ["a", "b", "c", "d"]

    def test_cross_lane_agreement_beats_single_lane_top(self):
        # `shared` is rank 2 in BOTH lanes; x/y are rank 1 in one lane only.
        # Agreement (2/62) must outrank a single rank-1 hit (1/61).
        sparse = [_item("x"), _item("shared")]
        dense = [_item("y"), _item("shared")]
        fused = _rrf_fuse([sparse, dense], k=60, limit=10)
        assert fused[0].id == "shared"
        assert set(i.id for i in fused) == {"shared", "x", "y"}

    def test_scores_are_rank_based_not_value_based(self):
        # Even if one lane "wanted" to dominate, only POSITION matters: an item
        # at rank 1 in lane A and absent from lane B scores exactly 1/(k+1).
        only_a = _rrf_fuse([[_item("solo")], []], k=60, limit=10)
        assert [i.id for i in only_a] == ["solo"]

    def test_limit_slices_after_fusion(self):
        sparse = [_item("a"), _item("b"), _item("c")]
        dense = [_item("a"), _item("b"), _item("c")]
        fused = _rrf_fuse([sparse, dense], k=60, limit=2)
        assert [i.id for i in fused] == ["a", "b"]

    def test_falsy_limit_returns_all(self):
        sparse = [_item("a"), _item("b")]
        fused = _rrf_fuse([sparse, []], k=60, limit=0)
        assert {i.id for i in fused} == {"a", "b"}

    def test_none_id_items_are_skipped(self):
        # Items without a stable id cannot be deduped across lanes -> skipped.
        fused = _rrf_fuse([[_item(None), _item("a")], [_item("a")]], k=60, limit=10)
        assert [i.id for i in fused] == ["a"]

    def test_empty_lanes(self):
        assert _rrf_fuse([[], []], k=60, limit=10) == []

    def test_dedup_keeps_first_seen_object(self):
        # The object returned for a shared id is the one first encountered.
        first = SimpleNamespace(id="dup", lane="sparse")
        second = SimpleNamespace(id="dup", lane="dense")
        fused = _rrf_fuse([[first], [second]], k=60, limit=10)
        assert len(fused) == 1
        assert fused[0].lane == "sparse"


class TestHybridOrchestration:
    @pytest.mark.asyncio
    async def test_runs_both_lanes_overfetches_and_fuses(self, monkeypatch):
        mgr = ProceduralMemoryManager()
        calls = []

        async def fake_list_procedures(**kwargs):
            calls.append(kwargs)
            if kwargs["search_method"] == "bm25":
                return [_item("a"), _item("b")]
            return [_item("b"), _item("c")]  # embedding lane

        monkeypatch.setattr(mgr, "list_procedures", fake_list_procedures)

        out = await mgr._hybrid_search_for_user(
            agent_state=None,
            user=None,
            query="fix dates",
            embedded_text=None,
            search_field="description",
            limit=5,
            timezone_str=None,
            filter_tags=None,
            scopes=None,
            use_cache=True,
            similarity_threshold=None,
        )

        # Exactly the two lanes ran.
        assert sorted(c["search_method"] for c in calls) == ["bm25", "embedding"]
        # Each lane over-fetched limit * multiplier.
        for c in calls:
            assert c["limit"] == 5 * SKILL_HYBRID_RECALL_MULTIPLIER
        # `b` appears in both lanes -> fused to the top.
        assert out[0].id == "b"
        assert {i.id for i in out} == {"a", "b", "c"}
        # Final result sliced back to the requested limit.
        assert len(out) <= 5

    @pytest.mark.asyncio
    async def test_similarity_threshold_is_dense_only(self, monkeypatch):
        mgr = ProceduralMemoryManager()
        by_method = {}

        async def fake_list_procedures(**kwargs):
            by_method[kwargs["search_method"]] = kwargs
            return []

        monkeypatch.setattr(mgr, "list_procedures", fake_list_procedures)

        await mgr._hybrid_search_for_user(
            agent_state=None,
            user=None,
            query="q",
            embedded_text=None,
            search_field="description",
            limit=3,
            timezone_str=None,
            filter_tags=None,
            scopes=None,
            use_cache=True,
            similarity_threshold=0.4,
        )

        # The BM25 lane must NOT receive the cosine quality floor...
        assert "similarity_threshold" not in by_method["bm25"]
        # ...but the dense lane must, applied BEFORE fusion.
        assert by_method["embedding"]["similarity_threshold"] == 0.4

    @pytest.mark.asyncio
    async def test_dense_lane_field_falls_back_when_not_embeddable(self, monkeypatch):
        mgr = ProceduralMemoryManager()
        by_method = {}

        async def fake_list_procedures(**kwargs):
            by_method[kwargs["search_method"]] = kwargs
            return []

        monkeypatch.setattr(mgr, "list_procedures", fake_list_procedures)

        # "entry_type" is searchable lexically but has no embedding column.
        await mgr._hybrid_search_for_user(
            agent_state=None,
            user=None,
            query="q",
            embedded_text=None,
            search_field="entry_type",
            limit=3,
            timezone_str=None,
            filter_tags=None,
            scopes=None,
            use_cache=True,
            similarity_threshold=None,
        )

        # Lexical lane keeps the requested field; dense lane falls back to
        # "description" (the only sensible embedding column).
        assert by_method["bm25"]["search_field"] == "entry_type"
        assert by_method["embedding"]["search_field"] == "description"
