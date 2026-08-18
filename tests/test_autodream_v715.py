from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_reconsolidator_v715 import (
    _ANN_FETCH_K,
    _MAX_VERIFY,
    _PER_DELTA_TOP_K,
    _candidate_pairs,
    _fact_signature,
)
from mirix.settings import settings


def test_v715_keeps_v712_ingest_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.15")
    v715 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.12")
    v712 = _parse(raw)

    assert v715.frames[0].args == v712.frames[0].args
    assert v715.frames[0].args == [
        ("attendee", "Caroline"),
        ("event", "Support group"),
    ]


@pytest.mark.asyncio
async def test_v715_manager_routes_only_explicit_semantic_delta(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.15")
    dream = AsyncMock(return_value={"selected_for_verification": 2})
    driver = object()

    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v715.reconsolidate_semantic_delta",
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
    assert result["incremental_semantic"]["selected_for_verification"] == 2


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
                {
                    "a_id": "new-id",
                    "a_name": "New semantic name",
                    "b_id": "old-id",
                    "b_name": "Old semantic name",
                    "sc": 0.97,
                },
                {
                    "a_id": "old-id",
                    "a_name": "Old semantic name",
                    "b_id": "new-id",
                    "b_name": "New semantic name",
                    "sc": 0.96,
                },
            ]
        )


class _Driver:
    def __init__(self):
        self.session_obj = _Session()

    def session(self, **_):
        return self.session_obj


@pytest.mark.asyncio
async def test_v715_pair_funnel_filters_then_limits_and_dedups_by_id():
    driver = _Driver()
    pairs, returned_rows = await _candidate_pairs(
        driver, "user-1", ["new-id", "old-id"]
    )

    assert returned_rows == 2
    assert len(pairs) == 1
    assert pairs[0]["score"] == 0.97
    assert pairs[0]["delta_ids"] == {"new-id", "old-id"}

    query, params = driver.session_obj.calls[0]
    assert "UNWIND $delta_ids" in query
    assert "b.user_id = $u" in query
    assert "size(coalesce(b.semantic_ids, [])) > 0" in query
    assert "ORDER BY sc DESC LIMIT $per_delta_k" in query
    assert params["fetch_k"] == _ANN_FETCH_K == 64
    assert params["per_delta_k"] == _PER_DELTA_TOP_K == 4
    assert _MAX_VERIFY == 60


def test_v715_fact_signature_is_order_independent_but_time_exact():
    base = {
        "predicate": "attend",
        "args": [
            {"role": "event", "anchor_id": "event-1"},
            {"role": "agent", "anchor_id": "person-1"},
        ],
        "lit_keys": ["venue", "count"],
        "lit_vals": ["hall", 3],
        "timestamp": "2023-07-07T00:00:00Z",
    }
    reordered = {
        **base,
        "args": list(reversed(base["args"])),
        "lit_keys": ["count", "venue"],
        "lit_vals": [3, "hall"],
    }
    later = {**reordered, "timestamp": "2023-07-08T00:00:00Z"}

    assert _fact_signature(base) == _fact_signature(reordered)
    assert _fact_signature(base) != _fact_signature(later)
