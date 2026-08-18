import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.retrieval_policy_v722 import (
    merge_memory_rows,
    plan_query,
    predicate_match_score,
    rank_relation_facts,
)
from mirix.settings import settings


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from v720_policy import is_v720  # noqa: E402
from v722_policy import question_operator  # noqa: E402


def _fact(fid, predicate, obj, *, timestamp=None, literal=None):
    return SimpleNamespace(
        id=fid,
        predicate=predicate,
        args=[
            {"id": "jon", "name": "Jon", "anchor_type": "person", "role": "agent"},
            {"id": f"obj-{fid}", "name": obj, "anchor_type": "book", "role": "theme"},
        ],
        memory_ids=[f"sem-{fid}"],
        timestamp=timestamp,
        lit_keys=["status"] if literal else [],
        lit_vals=[literal] if literal else [],
        policy_score=None,
    )


def _row(rid, text):
    return SimpleNamespace(
        id=rid, kind="semantic", summary=text, details="", extra={}
    )


def test_v722_inherits_v720_ingest_and_open_roles(monkeypatch):
    assert is_v720("v7.22")
    raw = '{"frames":[{"predicate":"buy","args":[' \
          '{"role":"buyer","name":"Melanie","type":"person"},' \
          '{"role":"item","name":"figurines","type":"object"}]}]}'
    monkeypatch.setattr(settings, "graph_version", "v7.22")
    parsed = _parse(raw)
    assert [arg[0] for arg in parsed.frames[0].args] == ["buyer", "item"]


@pytest.mark.asyncio
async def test_v722_reuses_v719_bounded_frontier(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.22")
    dream = AsyncMock(return_value={"round_count": 1})
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719.reconsolidate_bounded_frontier",
        dream,
    )
    monkeypatch.setattr(
        "mirix.database.neo4j_client.get_neo4j_driver", lambda: object(),
    )
    result = await AutoDreamManager()._refine_graph(
        SimpleNamespace(id="user-1"),
        SimpleNamespace(),
        semantic_memory_ids={"sem-new"},
        final_full_graph=False,
    )
    dream.assert_awaited_once_with(
        ANY,
        user_id="user-1",
        agent_state=ANY,
        semantic_memory_ids={"sem-new"},
        final_full_graph=False,
    )
    assert result == {"bounded_frontier": {"round_count": 1}}


def test_v722_query_plan_is_general_and_normalizes_inflection():
    located = plan_query("Which country was Jolene located in during August 2023?")
    assert located.relations == ("locate",)
    assert "located" in located.predicate_forms
    assert located.years == (2023,)
    assert located.months == (8,)

    interested = plan_query("What hobby did James become interested in?")
    assert interested.relations == ("interest",)
    assert predicate_match_score("What hobby was James interested in?", "show_interest") >= 0.72

    listened = plan_query("Which artists did Calvin listen to as a child?")
    assert listened.relations == ("listen",)
    assert plan_query("Which activity was James pursuing?").relations == ("engage",)
    assert question_operator("How many cities did Calvin travel to?") == "count"
    assert not plan_query("What were the new shoes used for?").preserve_raw_query


def test_v722_does_not_open_relation_lane_for_auxiliary_or_generic_make():
    assert not plan_query("What traits has Caroline shown?").is_relation_query
    assert not plan_query("What kind of place does Caroline want to make?").is_relation_query
    assert plan_query("What pets does Caroline have?").relations == ("possess",)


def test_v722_fact_rank_uses_object_and_explicit_date():
    facts = [
        _fact("wrong", "read", "Another Book", timestamp="2023-05-27T00:00:00"),
        _fact("right", "reading", "The Lean Startup", timestamp="2023-06-12T00:00:00"),
    ]
    ranked = rank_relation_facts(
        "What did Jon read about the Lean Startup in June 2023?",
        facts,
        exact_anchor_ids={"jon"},
        limit=2,
    )
    assert ranked[0].id == "right"


def test_v722_suppresses_relation_facts_that_miss_concrete_object():
    facts = [
        _fact("pottery", "use for", "pottery", literal="self expression"),
        _fact("painting", "used", "paintings", literal="creative outlet"),
    ]
    ranked = rank_relation_facts(
        "What were Jon's new shoes used for?",
        facts,
        exact_anchor_ids={"jon"},
        limit=8,
    )
    assert ranked == []


def test_v722_fact_rank_uses_earliest_and_latest_qualifiers():
    facts = [
        _fact("old", "visit", "Rome", timestamp="2022-03-10T00:00:00"),
        _fact("new", "visited", "Paris", timestamp="2023-07-10T00:00:00"),
    ]
    first = rank_relation_facts(
        "Which place did Jon visit first?", facts, exact_anchor_ids={"jon"}, limit=2
    )
    latest = rank_relation_facts(
        "Which place did Jon visit most recently?", facts,
        exact_anchor_ids={"jon"}, limit=2,
    )
    assert first[0].id == "old"
    assert latest[0].id == "new"


def test_v722_fact_rank_normalizes_mixed_timestamp_timezones():
    facts = [
        _fact("naive", "visit", "Rome", timestamp="2022-03-10T00:00:00"),
        _fact("aware", "visited", "Paris", timestamp="2023-07-10T00:00:00+00:00"),
    ]
    ranked = rank_relation_facts(
        "Which place did Jon visit most recently?", facts,
        exact_anchor_ids={"jon"}, limit=2,
    )
    assert ranked[0].id == "aware"


def test_v722_quota_protects_base_rows_from_relation_crowding():
    base = [_row(f"base-{idx}", f"base evidence {idx}") for idx in range(12)]
    relation = [_row(f"rel-{idx}", f"relation evidence {idx}") for idx in range(12)]
    merged = merge_memory_rows(
        "What book did Jon read?", base, relation, limit=15
    )
    assert len(merged) == 15
    assert sum(row.id.startswith("base-") for row in merged) == 10
    assert sum(row.id.startswith("rel-") for row in merged) == 5


def test_v722_aggregation_budgets_expand_without_changing_single_questions():
    single = plan_query("What book did Jon read?")
    count = plan_query("How many books did Jon read?")
    assert single.fact_limit == 8
    assert count.fact_limit > single.fact_limit
    assert count.relation_quota > single.relation_quota

