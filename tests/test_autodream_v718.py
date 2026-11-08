from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator_v718 import (
    _fact_signature,
    _is_semantic_only,
    _pair_key,
    _run_candidate_round,
    reconsolidate_iterative_semantic,
)
from mirix.settings import settings


def test_v718_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.18")
    v718 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)

    assert v718.frames[0].args == v712.frames[0].args


def test_v718_pair_cache_key_is_unordered_and_casefolded():
    assert _pair_key(" Alpha ", "BETA") == _pair_key("beta", "alpha")
    assert _pair_key("alpha", "beta") != _pair_key("alpha", "gamma")


def test_v718_fact_scope_requires_every_citation_to_be_semantic():
    assert _is_semantic_only(["sem_a", "sem_b"])
    assert not _is_semantic_only(["sem_a", "ep_b"])
    assert not _is_semantic_only(["ep_a"])
    assert not _is_semantic_only([])


def test_v718_fact_signature_preserves_timestamp_literals_roles_and_anchors():
    base = {
        "predicate": "visit",
        "args": [
            {"role": "place", "anchor_id": "beach"},
            {"role": "agent", "anchor_id": "melanie"},
        ],
        "lit_keys": ["weather"],
        "lit_vals": ["sunny"],
        "timestamp": "2023-07-04T00:00:00Z",
    }
    reordered = {**base, "args": list(reversed(base["args"]))}
    another_day = {**base, "timestamp": "2023-07-20T00:00:00Z"}

    assert _fact_signature(base) == _fact_signature(reordered)
    assert _fact_signature(base) != _fact_signature(another_day)


@pytest.mark.asyncio
async def test_v718_round_skips_cached_rejections_and_caps_at_120(monkeypatch):
    pairs = [(f"a-{i}", f"b-{i}", 0.99 - i / 10000) for i in range(130)]
    cached = {_pair_key("a-0", "b-0")}
    candidate_fn = AsyncMock(return_value=pairs)
    verify = AsyncMock(return_value=([("a-1", "b-1", "b-1")], {_pair_key("a-2", "b-2")}))
    cache = AsyncMock()
    merge = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._semantic_candidate_pairs",
        candidate_fn,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._verify_with_rejections", verify
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._cache_rejected_pair_keys", cache
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._merge_confirmed", merge
    )

    result = await _run_candidate_round(
        object(),
        user_id="user-1",
        agent_state=object(),
        source_anchor_ids=["source"],
        rejected_keys=cached,
    )

    submitted = verify.await_args.args[0]
    assert len(submitted) == 120
    assert pairs[0] not in submitted
    assert result == {
        "candidate_pairs": 130,
        "unseen_candidate_pairs": 129,
        "pairs_submitted": 120,
        "explicit_rejections_cached": 1,
        "llm_confirmed_merges": 1,
        "anchors_merged": 1,
    }
    assert _pair_key("a-2", "b-2") in cached


@pytest.mark.asyncio
async def test_v718_intermediate_is_one_delta_round_and_no_fact_cleanup(monkeypatch):
    rejected = AsyncMock(return_value=set())
    delta = AsyncMock(return_value=["new-sem-anchor"])
    run_round = AsyncMock(
        return_value={
            "candidate_pairs": 10,
            "unseen_candidate_pairs": 10,
            "pairs_submitted": 10,
            "explicit_rejections_cached": 8,
            "llm_confirmed_merges": 2,
            "anchors_merged": 2,
        }
    )
    fact_rows = AsyncMock(side_effect=AssertionError("facts wait for final dream"))
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._rejected_pair_keys", rejected
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._semantic_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._run_candidate_round", run_round
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._semantic_only_fact_rows", fact_rows
    )

    result = await reconsolidate_iterative_semantic(
        object(),
        user_id="user-1",
        agent_state=object(),
        semantic_memory_ids={"sem-new"},
        final_full_graph=False,
    )

    delta.assert_awaited_once()
    assert run_round.await_count == 1
    fact_rows.assert_not_awaited()
    assert result["round_count"] == 1
    assert result["global_cleanup"] == "deferred_to_final_dream"
    assert result["mixed_facts_in_scope"] == 0
    assert result["episodic_only_facts_in_scope"] == 0


@pytest.mark.asyncio
async def test_v718_final_recomputes_then_stops_on_zero_merge_and_cleans_sem_only(
    monkeypatch,
):
    rejected = AsyncMock(return_value=set())
    all_anchors = AsyncMock(side_effect=[["a", "b", "c"], ["b", "c"]])
    run_round = AsyncMock(
        side_effect=[
            {
                "candidate_pairs": 4,
                "unseen_candidate_pairs": 4,
                "pairs_submitted": 4,
                "explicit_rejections_cached": 2,
                "llm_confirmed_merges": 1,
                "anchors_merged": 1,
            },
            {
                "candidate_pairs": 2,
                "unseen_candidate_pairs": 2,
                "pairs_submitted": 2,
                "explicit_rejections_cached": 2,
                "llm_confirmed_merges": 0,
                "anchors_merged": 0,
            },
        ]
    )
    fact_rows = AsyncMock(
        return_value=[
            {
                "id": "f-sem",
                "memory_ids": ["sem-1"],
                "predicate": "p",
                "args": [],
                "lit_keys": [],
                "lit_vals": [],
                "timestamp": "",
            }
        ]
    )
    collapse = AsyncMock(return_value=1)
    degenerate = AsyncMock(return_value=0)
    prune = AsyncMock(return_value=2)
    conflict_candidates = AsyncMock(return_value=[])
    confirm_conflicts = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._rejected_pair_keys", rejected
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._all_semantic_anchor_ids", all_anchors
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._run_candidate_round", run_round
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._semantic_only_fact_rows", fact_rows
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._merge_exact_semantic_only_facts",
        collapse,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._remove_semantic_only_degenerate_facts",
        degenerate,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._prune_dead_semantic_only_anchors",
        prune,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._semantic_only_conflict_candidates",
        conflict_candidates,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718._confirm_conflicts",
        confirm_conflicts,
    )

    result = await reconsolidate_iterative_semantic(
        object(),
        user_id="user-1",
        agent_state=object(),
        semantic_memory_ids={"sem-tail"},
        final_full_graph=True,
    )

    assert all_anchors.await_count == 2
    assert run_round.await_count == 2
    assert result["round_count"] == 2
    assert result["anchors_merged"] == 1
    assert result["pairs_submitted"] == 6
    assert result["global_cleanup"] == "completed_semantic_only_strict"
    assert result["semantic_only_facts_in_scope"] == 1
    assert result["mixed_facts_in_scope"] == 0
    assert result["episodic_only_facts_in_scope"] == 0


@pytest.mark.asyncio
async def test_v718_manager_routes_without_legacy_maintenance(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.18")
    dream = AsyncMock(return_value={"round_count": 2})
    maintain = AsyncMock(side_effect=AssertionError("v7.18 owns strict cleanup"))
    driver = object()
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v718.reconsolidate_iterative_semantic",
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
    assert result == {"iterative_semantic": {"round_count": 2}}
