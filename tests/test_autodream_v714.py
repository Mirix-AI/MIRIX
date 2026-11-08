import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator_v714 import _batch_digest, _candidate_pairs
from mirix.settings import settings


def test_v714_request_and_tail_batch_scope():
    request = AutoDreamRequest(graph_only=True, source_chunk_ids=[15, 16, 17, 18])
    assert request.source_chunk_ids == [15, 16, 17, 18]

    eval_dir = Path(__file__).parents[1] / "evals"
    sys.path.insert(0, str(eval_dir))
    try:
        from main_eval import dream_source_chunk_ids

        assert dream_source_chunk_ids(5, 5) == [0, 1, 2, 3, 4]
        assert dream_source_chunk_ids(10, 5) == [5, 6, 7, 8, 9]
        assert dream_source_chunk_ids(15, 5) == [10, 11, 12, 13, 14]
        assert dream_source_chunk_ids(19, 5) == [15, 16, 17, 18]
    finally:
        sys.path.remove(str(eval_dir))


def test_v714_batch_digest_is_order_independent():
    assert _batch_digest({"sem_a", "sem_b"}) == _batch_digest({"sem_b", "sem_a"})


def test_v714_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.14")
    v714 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.13")
    v713 = _parse(raw)

    assert v714.frames[0].args == v712.frames[0].args
    assert v714.frames[0].args == [("attendee", "Caroline"), ("event", "Support group")]
    assert v713.frames[0].args == [("agent", "Caroline"), ("theme", "Support group")]


@pytest.mark.asyncio
async def test_v714_manager_routes_only_explicit_semantic_delta(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.14")
    dream = AsyncMock(return_value={"candidate_pairs": 2, "episodic_only_candidates": 0})
    driver = object()

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v714.reconsolidate_semantic_delta",
        dream,
    )
    monkeypatch.setattr(
        "mirix.database.neo4j_client.get_neo4j_driver", lambda: driver
    )

    result = await AutoDreamManager()._refine_graph(
        SimpleNamespace(id="user-1"),
        SimpleNamespace(),
        semantic_memory_ids={"sem_new"},
    )

    dream.assert_awaited_once_with(
        driver,
        user_id="user-1",
        agent_state=ANY,
        semantic_memory_ids={"sem_new"},
    )
    assert result["incremental_semantic"]["episodic_only_candidates"] == 0


@pytest.mark.asyncio
async def test_v714_empty_delta_never_falls_back_to_full_graph(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.14")
    result = await AutoDreamManager()._refine_graph(
        SimpleNamespace(id="user-1"), SimpleNamespace(), semantic_memory_ids=set()
    )
    assert result == {"incremental_semantic": {"skipped": "empty_semantic_delta"}}


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def iterator():
            for row in self.rows:
                yield row

        return iterator()


class _Session:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _Rows(
            [
                {"a": "New semantic name", "b": "Old semantic name", "sc": 0.97},
                {"a": "Old semantic name", "b": "New semantic name", "sc": 0.96},
            ]
        )


class _Driver:
    def __init__(self):
        self.session_obj = _Session()

    def session(self, **_):
        return self.session_obj


@pytest.mark.asyncio
async def test_v714_pair_query_is_delta_started_and_semantic_targeted():
    driver = _Driver()
    pairs = await _candidate_pairs(driver, "user-1", ["delta-anchor"])

    assert pairs == [("New semantic name", "Old semantic name", 0.97)]
    query, params = driver.session_obj.calls[0]
    assert "UNWIND $delta_ids" in query
    assert "size(coalesce(a.semantic_ids, [])) > 0" in query
    assert "size(coalesce(b.semantic_ids, [])) > 0" in query
    assert params["delta_ids"] == ["delta-anchor"]
