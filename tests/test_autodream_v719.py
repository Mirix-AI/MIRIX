from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator_v719 import (
    _FINAL_FIRST_MAX_VERIFY,
    _FINAL_REPAIR_MAX_VERIFY,
    _INTERMEDIATE_MAX_VERIFY,
    _MERGE_SAFE_TYPES,
    _pair_key,
    _run_candidate_round,
    reconsolidate_bounded_frontier,
)
from mirix.settings import settings


def _round_result(*, merged: int, submitted: int = 4) -> dict:
    return {
        "candidate_pairs": submitted,
        "unseen_candidate_pairs": submitted,
        "pairs_submitted": submitted,
        "explicit_rejections_cached": submitted - merged,
        "llm_confirmed_merges": merged,
        "anchors_merged": merged,
    }


def test_v719_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.19")
    v719 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)

    assert v719.frames[0].args == v712.frames[0].args


def test_v719_pair_cache_key_is_unordered_type_and_policy_scoped():
    assert _pair_key(" Alpha ", "BETA", "Person") == _pair_key(
        "beta", "alpha", "person"
    )
    assert _pair_key("alpha", "beta", "person") != _pair_key(
        "alpha", "beta", "concept"
    )


def test_v719_merge_safe_types_exclude_temporal_and_occurrence_types():
    assert {"person", "organization", "location", "object", "concept"} == (
        _MERGE_SAFE_TYPES
    )
    assert not ({"event", "date", "time", "state", "duration"} & _MERGE_SAFE_TYPES)


@pytest.mark.asyncio
async def test_v719_round_uses_requested_cap_and_type_scoped_cache(monkeypatch):
    pairs = [
        (f"a-{i}", f"b-{i}", 0.99 - i / 10000, "person")
        for i in range(130)
    ]
    cached = {_pair_key("a-0", "b-0", "person")}
    candidate_fn = AsyncMock(return_value=pairs)
    verify = AsyncMock(
        return_value=(
            [("a-1", "b-1", "b-1")],
            {_pair_key("a-2", "b-2", "person")},
        )
    )
    cache = AsyncMock()
    merge = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._safe_semantic_candidate_pairs",
        candidate_fn,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._verify_with_rejections", verify
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._cache_rejected_pair_keys", cache
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._merge_confirmed", merge
    )

    result, confirmed = await _run_candidate_round(
        object(),
        user_id="user-1",
        agent_state=object(),
        source_anchor_ids=["source"],
        rejected_keys=cached,
        max_verify=80,
    )

    submitted = verify.await_args.args[0]
    assert len(submitted) == 80
    assert pairs[0] not in submitted
    assert result["unseen_candidate_pairs"] == 129
    assert result["pairs_submitted"] == 80
    assert result["anchors_merged"] == 1
    assert confirmed == [("a-1", "b-1", "b-1")]
    assert _pair_key("a-2", "b-2", "person") in cached


@pytest.mark.asyncio
async def test_v719_intermediate_is_delta_only_max80_and_no_online_cleanup(
    monkeypatch,
):
    driver = object()
    rejected = AsyncMock(return_value=set())
    delta = AsyncMock(return_value=["safe-delta"])
    undreamed = AsyncMock(side_effect=AssertionError("intermediate is delta-only"))
    run_round = AsyncMock(return_value=(_round_result(merged=1), [("a", "b", "b")]))
    mark = AsyncMock()
    representatives = AsyncMock(side_effect=AssertionError("no intermediate repair"))
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._rejected_pair_keys", rejected
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._safe_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._undreamed_safe_anchor_ids",
        undreamed,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._run_candidate_round", run_round
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._mark_dreamed_anchor_ids", mark
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._representative_anchor_ids",
        representatives,
    )

    result = await reconsolidate_bounded_frontier(
        driver,
        user_id="user-1",
        agent_state=object(),
        semantic_memory_ids={"sem-new"},
        final_full_graph=False,
    )

    assert run_round.await_count == 1
    assert run_round.await_args.kwargs["source_anchor_ids"] == ["safe-delta"]
    assert run_round.await_args.kwargs["max_verify"] == _INTERMEDIATE_MAX_VERIFY
    mark.assert_awaited_once_with(driver, "user-1", ["safe-delta"])
    assert result["round_count"] == 1
    assert result["online_fact_cleanup"] == "disabled"
    assert result["semantic_facts_in_scope"] == 0
    assert result["episodic_only_facts_in_scope"] == 0


@pytest.mark.asyncio
async def test_v719_final_unions_delta_and_undreamed_then_repairs_dirty_only(
    monkeypatch,
):
    driver = object()
    rejected = AsyncMock(return_value=set())
    delta = AsyncMock(return_value=["delta", "shared"])
    undreamed = AsyncMock(return_value=["old", "shared"])
    run_round = AsyncMock(
        side_effect=[
            (_round_result(merged=2), [("a", "b", "b"), ("c", "d", "d")]),
            (_round_result(merged=1, submitted=3), [("e", "f", "f")]),
        ]
    )
    representatives = AsyncMock(return_value=["rep-b", "rep-d"])
    mark = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._rejected_pair_keys", rejected
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._safe_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._undreamed_safe_anchor_ids",
        undreamed,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._run_candidate_round", run_round
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._representative_anchor_ids",
        representatives,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._mark_dreamed_anchor_ids", mark
    )

    result = await reconsolidate_bounded_frontier(
        driver,
        user_id="user-1",
        agent_state=object(),
        semantic_memory_ids={"sem-tail"},
        final_full_graph=True,
    )

    first, repair = run_round.await_args_list
    assert first.kwargs["source_anchor_ids"] == ["delta", "old", "shared"]
    assert first.kwargs["max_verify"] == _FINAL_FIRST_MAX_VERIFY
    assert repair.kwargs["source_anchor_ids"] == ["rep-b", "rep-d"]
    assert repair.kwargs["max_verify"] == _FINAL_REPAIR_MAX_VERIFY
    assert result["round_count"] == 2
    assert result["pairs_submitted"] == 7
    assert result["anchors_merged"] == 3
    assert result["rounds"][1]["frontier"] == "dirty_representatives"
    assert mark.await_count == 2


@pytest.mark.asyncio
async def test_v719_final_zero_first_merge_does_not_run_repair(monkeypatch):
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._rejected_pair_keys",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._safe_delta_anchor_ids",
        AsyncMock(return_value=["tail"]),
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._undreamed_safe_anchor_ids",
        AsyncMock(return_value=[]),
    )
    run_round = AsyncMock(return_value=(_round_result(merged=0), []))
    representatives = AsyncMock(side_effect=AssertionError("zero merge stops repair"))
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._run_candidate_round", run_round
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._representative_anchor_ids",
        representatives,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719._mark_dreamed_anchor_ids",
        AsyncMock(),
    )

    result = await reconsolidate_bounded_frontier(
        object(),
        user_id="user-1",
        agent_state=object(),
        semantic_memory_ids={"sem-tail"},
        final_full_graph=True,
    )

    assert run_round.await_count == 1
    assert result["round_count"] == 1


@pytest.mark.asyncio
async def test_v719_manager_routes_without_legacy_maintenance(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.19")
    dream = AsyncMock(return_value={"round_count": 2})
    maintain = AsyncMock(side_effect=AssertionError("v7.19 has no online cleanup"))
    driver = object()
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719.reconsolidate_bounded_frontier",
        dream,
    )
    monkeypatch.setattr(
        "mirix.database.neo4j_client.get_neo4j_driver", lambda: driver
    )
    monkeypatch.setattr(
        "mirix.services.graph_memory_manager_v7.V7GraphManager.maintain_graph",
        maintain,
    )

    result = await AutoDreamManager()._refine_graph(
        SimpleNamespace(id="user-1"),
        SimpleNamespace(),
        semantic_memory_ids={"sem-new"},
        final_full_graph=True,
    )

    dream.assert_awaited_once_with(
        driver,
        user_id="user-1",
        agent_state=ANY,
        semantic_memory_ids={"sem-new"},
        final_full_graph=True,
    )
    maintain.assert_not_awaited()
    assert result == {"bounded_frontier": {"round_count": 2}}
