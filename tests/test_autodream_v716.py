from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator import _anchor_pairs
from mirix.services.graph_reconsolidator_v716 import (
    _semantic_delta_anchor_ids,
    reconsolidate_semantic_frontier,
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


def test_v716_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.16")
    v716 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)

    assert v716.frames[0].args == v712.frames[0].args
    assert v716.frames[0].args == [
        ("attendee", "Caroline"),
        ("event", "Support group"),
    ]


@pytest.mark.asyncio
async def test_v716_resolves_only_current_batch_semantic_anchors():
    driver = _Driver([{"id": "anchor-b"}, {"id": "anchor-a"}])

    ids = await _semantic_delta_anchor_ids(
        driver, "user-1", {"sem-new-2", "sem-new-1"}
    )

    assert ids == ["anchor-b", "anchor-a"]
    query, params = driver.session_obj.calls[0]
    assert "mid IN $batch" in query
    assert "episodic_ids" not in query
    assert params["batch"] == ["sem-new-1", "sem-new-2"]


@pytest.mark.asyncio
async def test_v716_v712_pair_query_starts_only_from_source_ids():
    driver = _Driver(
        [
            {"a": "New semantic name", "b": "Old name", "sc": 0.97},
            {"a": "Old name", "b": "New semantic name", "sc": 0.96},
        ]
    )

    pairs = await _anchor_pairs(driver, "user-1", ["delta-anchor"])

    assert pairs == [("New semantic name", "Old name", 0.97)]
    query, params = driver.session_obj.calls[0]
    assert "UNWIND $source_anchor_ids AS source_id" in query
    assert "queryNodes('v7_anchor_name_emb', 4" in query
    assert "size(coalesce(b.semantic_ids, []))" not in query
    assert params["source_anchor_ids"] == ["delta-anchor"]
    assert params["th"] == 0.93


@pytest.mark.asyncio
async def test_v716_empty_source_never_falls_back_to_full_graph():
    driver = _Driver([])

    assert await _anchor_pairs(driver, "user-1", []) == []
    assert driver.session_obj.calls == []


@pytest.mark.asyncio
async def test_v716_wraps_unchanged_v712_reconsolidator(monkeypatch):
    delta = AsyncMock(return_value=["anchor-new"])
    legacy = AsyncMock(return_value={"candidate_pairs": 3, "anchors_merged": 1})
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v716._semantic_delta_anchor_ids", delta
    )
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v716.reconsolidate_graph", legacy
    )
    driver = object()
    agent = object()

    result = await reconsolidate_semantic_frontier(
        driver,
        user_id="user-1",
        agent_state=agent,
        semantic_memory_ids={"sem-new"},
        every_n_memories=10,
    )

    delta.assert_awaited_once_with(driver, "user-1", {"sem-new"})
    legacy.assert_awaited_once_with(
        driver,
        user_id="user-1",
        agent_state=agent,
        dry_run=False,
        every_n_memories=10,
        source_anchor_ids=["anchor-new"],
    )
    assert result == {
        "candidate_pairs": 3,
        "anchors_merged": 1,
        "batch_semantic_memories": 1,
        "source_semantic_anchors": 1,
    }


@pytest.mark.asyncio
async def test_v716_manager_routes_delta_but_keeps_legacy_maintenance(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.16")
    dream = AsyncMock(return_value={"candidate_pairs": 2})
    maintain = AsyncMock(return_value={"dead_anchors_pruned": 1})
    driver = object()

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v716.reconsolidate_semantic_frontier",
        dream,
    )
    monkeypatch.setattr(
        "mirix.database.neo4j_client.get_neo4j_driver", lambda: driver
    )
    monkeypatch.setattr(
        "mirix.services.graph_memory_manager_v7.V7GraphManager.maintain_graph",
        maintain,
    )
    monkeypatch.setattr(
        AutoDreamManager, "_graph_memory_ids", AsyncMock(return_value={"sem-new"})
    )

    result = await AutoDreamManager()._refine_graph(
        SimpleNamespace(id="user-1"),
        SimpleNamespace(),
        semantic_memory_ids={"sem-new"},
    )

    dream.assert_awaited_once_with(
        driver,
        user_id="user-1",
        agent_state=ANY,
        semantic_memory_ids={"sem-new"},
        every_n_memories=10,
    )
    maintain.assert_awaited_once()
    assert result == {
        "reconsolidation": {"candidate_pairs": 2},
        "maintenance": {"dead_anchors_pruned": 1},
    }
