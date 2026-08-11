from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator_v717 import (
    _semantic_candidate_pairs,
    reconsolidate_hybrid_semantic,
)
from mirix.settings import settings


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def iterator():
            for row in self.rows:
                yield row

        return iterator()


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _Rows(self.rows)


class _Driver:
    def __init__(self, rows):
        self.session_obj = _Session(rows)

    def session(self, **_):
        return self.session_obj


def test_v717_request_marks_only_explicit_final_dream():
    intermediate = AutoDreamRequest(
        graph_only=True, source_chunk_ids=[10, 11, 12, 13, 14]
    )
    final = AutoDreamRequest(
        graph_only=True,
        source_chunk_ids=[15, 16, 17, 18],
        final_full_graph=True,
    )

    assert intermediate.final_full_graph is False
    assert final.final_full_graph is True


def test_v717_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.17")
    v717 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)

    assert v717.frames[0].args == v712.frames[0].args
    assert v717.frames[0].args == [
        ("attendee", "Caroline"),
        ("event", "Support group"),
    ]


@pytest.mark.asyncio
async def test_v717_candidates_are_v712_top4_but_semantic_on_both_sides():
    driver = _Driver(
        [
            {"a": "New semantic name", "b": "Old semantic name", "sc": 0.97},
            {"a": "Old semantic name", "b": "New semantic name", "sc": 0.96},
        ]
    )

    pairs = await _semantic_candidate_pairs(driver, "user-1", ["delta-anchor"])

    assert pairs == [("New semantic name", "Old semantic name", 0.97)]
    query, params = driver.session_obj.calls[0]
    assert "UNWIND $source_ids" in query
    assert query.count("size(coalesce(") == 2
    assert "a.semantic_ids" in query
    assert "b.semantic_ids" in query
    assert params == {
        "source_ids": ["delta-anchor"],
        "u": "user-1",
        "neighbours": 4,
        "threshold": 0.93,
    }


@pytest.mark.asyncio
async def test_v717_intermediate_cycle_defers_all_global_cleanup(monkeypatch):
    delta = AsyncMock(return_value=["new-a"])
    all_anchors = AsyncMock(return_value=["old-a", "new-a"])
    candidates = AsyncMock(return_value=[("new", "old", 0.97)])
    verify = AsyncMock(return_value=[("old", "new", "new")])
    merge = AsyncMock(return_value=1)
    all_scope = AsyncMock(side_effect=AssertionError("must wait for final dream"))
    collapse = AsyncMock(side_effect=AssertionError("must wait for final dream"))

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._semantic_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._all_semantic_anchor_ids", all_anchors
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._semantic_candidate_pairs", candidates
    )
    monkeypatch.setattr("mirix.services.graph_reconsolidator_v717._verify", verify)
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._merge_confirmed", merge
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._all_semantic_scope", all_scope
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._collapse_affected_frames", collapse
    )

    driver = object()
    agent = object()
    result = await reconsolidate_hybrid_semantic(
        driver,
        user_id="user-1",
        agent_state=agent,
        semantic_memory_ids={"sem-new"},
        final_full_graph=False,
    )

    delta.assert_awaited_once_with(driver, "user-1", {"sem-new"})
    all_anchors.assert_not_awaited()
    candidates.assert_awaited_once_with(driver, "user-1", ["new-a"])
    verify.assert_awaited_once_with([("new", "old", 0.97)], agent)
    merge.assert_awaited_once()
    all_scope.assert_not_awaited()
    collapse.assert_not_awaited()
    assert result["global_cleanup"] == "deferred_to_final_dream"
    assert result["episodic_only_candidates"] == 0


@pytest.mark.asyncio
async def test_v717_final_cycle_scans_all_semantic_then_cleans_semantic_only(
    monkeypatch,
):
    delta = AsyncMock(side_effect=AssertionError("final must use full semantic source"))
    all_anchors = AsyncMock(return_value=["old-a", "new-a"])
    candidates = AsyncMock(return_value=[])
    merge = AsyncMock(return_value=0)
    all_scope = AsyncMock(return_value=(["sem-fact-1", "sem-fact-2"], ["a-1"]))
    collapse = AsyncMock(return_value=2)
    degenerate = AsyncMock(return_value=1)
    prune = AsyncMock(return_value=3)
    conflicts = AsyncMock(return_value=[{"subject": "s"}])
    confirm_conflicts = AsyncMock(return_value=[{"subject": "s"}])

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._semantic_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._all_semantic_anchor_ids", all_anchors
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._semantic_candidate_pairs", candidates
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._merge_confirmed", merge
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._all_semantic_scope", all_scope
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._collapse_affected_frames", collapse
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._remove_affected_degenerate_frames",
        degenerate,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._prune_touched_dead_semantic_anchors",
        prune,
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._nary_conflict_candidates", conflicts
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717._confirm_conflicts",
        confirm_conflicts,
    )

    driver = object()
    agent = object()
    result = await reconsolidate_hybrid_semantic(
        driver,
        user_id="user-1",
        agent_state=agent,
        semantic_memory_ids={"sem-tail"},
        final_full_graph=True,
    )

    delta.assert_not_awaited()
    all_anchors.assert_awaited_once_with(driver, "user-1")
    candidates.assert_awaited_once_with(driver, "user-1", ["old-a", "new-a"])
    all_scope.assert_awaited_once_with(driver, "user-1")
    collapse.assert_awaited_once_with(driver, "user-1", ["sem-fact-1", "sem-fact-2"])
    degenerate.assert_awaited_once()
    prune.assert_awaited_once_with(driver, "user-1", ["a-1"])
    conflicts.assert_awaited_once_with(
        driver, "user-1", ["sem-fact-1", "sem-fact-2"]
    )
    confirm_conflicts.assert_awaited_once_with([{"subject": "s"}], agent)
    assert result["global_cleanup"] == "completed_semantic_only"
    assert result["semantic_facts_in_scope"] == 2
    assert result["duplicate_semantic_frames_merged"] == 2
    assert result["episodic_only_candidates"] == 0


@pytest.mark.asyncio
async def test_v717_manager_routes_final_flag_without_legacy_maintenance(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.17")
    dream = AsyncMock(return_value={"candidate_pairs": 2})
    maintain = AsyncMock(side_effect=AssertionError("v7.17 owns its cleanup"))
    driver = object()

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v717.reconsolidate_hybrid_semantic",
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
    assert result == {"hybrid_semantic": {"candidate_pairs": 2}}
