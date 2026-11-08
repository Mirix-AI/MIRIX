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
