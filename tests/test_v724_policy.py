import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.graph_memory_manager_v7 import frame_identity_for_source
from mirix.services.graph_retriever_v7 import V7Retriever
from mirix.services.retrieval_policy_v724 import (
    is_v724,
    merge_memory_rows,
    plan_query,
    rank_relation_facts,
    role_alignment_score,
    temporal_state_score,
    verification_decision,
)
from mirix.settings import settings


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from v720_policy import is_v720  # noqa: E402
from v724_policy import deduplicate_results, question_operator  # noqa: E402


def _fact(fid, predicate, args, *, timestamp=None, literals=None):
    literals = literals or {}
    return SimpleNamespace(
        id=fid,
        predicate=predicate,
        args=args,
        memory_ids=[f"ep-{fid}"],
        timestamp=timestamp,
        lit_keys=list(literals),
        lit_vals=[literals[key] for key in literals],
        policy_score=None,
    )


def _row(rid, text, *, timestamp=None, kind="episodic"):
    return SimpleNamespace(
        id=rid,
        kind=kind,
        summary=text,
        details="",
        timestamp=timestamp,
        extra={},
    )


def test_v724_isolated_version_inherits_v720_and_general_operators():
    assert is_v724("v7.24")
    assert is_v720("v7.24")
    assert question_operator("How many trips did Kim complete?") == "count"

    show = plan_query("Which photo did Dave show Calvin?")
    assert "show" in show.relations
    assert {"showed", "shared", "displayed"} <= set(show.predicate_forms)
    assert show.base_quota < 10
    assert show.relation_quota >= 4

    enroll = plan_query("What course did Jolene sign up for?")
    assert "enroll" in enroll.relations

    assert "learn" in plan_query("Which song motivates Kim?").relations
    assert "suffer" in plan_query("When did Kim get hurt?").relations
    assert "suffer" in plan_query("What setback did Kim face?").relations


def test_v724_role_direction_prefers_matching_agent_and_recipient():
    right = _fact(
        "right",
        "show",
        [
            {"id": "dave", "name": "Dave", "anchor_type": "person", "role": "agent"},
            {"id": "calvin", "name": "Calvin", "anchor_type": "person", "role": "recipient"},
            {"id": "photo-right", "name": "photograph", "anchor_type": "object", "role": "theme"},
            {"id": "boston", "name": "Boston", "anchor_type": "location", "role": "content"},
        ],
    )
    reversed_fact = _fact(
        "reversed",
        "show",
        [
            {"id": "dave", "name": "Dave", "anchor_type": "person", "role": "recipient"},
            {"id": "calvin", "name": "Calvin", "anchor_type": "person", "role": "agent"},
            {"id": "photo-wrong", "name": "photograph", "anchor_type": "object", "role": "theme"},
            {"id": "tokyo", "name": "Tokyo", "anchor_type": "location", "role": "content"},
        ],
    )
    query = "Which city was in the photograph Dave showed Calvin?"

    assert role_alignment_score(query, right, {"dave", "calvin"}) > 0
    assert role_alignment_score(query, reversed_fact, {"dave", "calvin"}) < 0
    ranked = rank_relation_facts(
        query, [reversed_fact, right], exact_anchor_ids={"dave", "calvin"}, limit=2
    )
    assert ranked[0].id == "right"


def test_v724_intersection_constraint_survives_soft_reranking():
    caroline_sunset = _fact(
        "caroline-sunset",
        "paint",
        [
            {"id": "caroline", "name": "Caroline", "anchor_type": "person", "role": "agent"},
            {"id": "sunset-a", "name": "beach sunset", "anchor_type": "object", "role": "theme"},
        ],
        timestamp="2023-08-18",
    )
    melanie_sunset = _fact(
        "melanie-sunset",
        "paint",
        [
            {"id": "melanie", "name": "Melanie", "anchor_type": "person", "role": "agent"},
            {"id": "sunset-b", "name": "sunset painting", "anchor_type": "object", "role": "theme"},
        ],
        timestamp="2023-10-13",
    )
    melanie_horse = _fact(
        "melanie-horse",
        "paint",
        [
            {"id": "melanie", "name": "Melanie", "anchor_type": "person", "role": "agent"},
            {"id": "horse", "name": "horse", "anchor_type": "object", "role": "theme"},
        ],
        # A recent timestamp must not turn a one-sided item into an intersection.
        timestamp="2024-01-01",
    )
    ranked = rank_relation_facts(
        "What subject have Caroline and Melanie both painted?",
        [melanie_horse, caroline_sunset, melanie_sunset],
        exact_anchor_ids={"caroline", "melanie"},
        limit=3,
    )
    assert {fact.id for fact in ranked[:2]} == {
        "caroline-sunset", "melanie-sunset",
    }


def test_v724_typed_chain_ignores_recommendations_of_the_wrong_answer_type():
    read_generic = _fact(
        "read-generic", "read", [
            {"id": "melanie", "name": "Melanie", "anchor_type": "person", "role": "agent"},
            {"id": "books", "name": "books", "anchor_type": "content", "role": "theme"},
        ],
    )
    recommend_book = _fact(
        "recommend-book", "recommend", [
            {"id": "caroline", "name": "Caroline", "anchor_type": "person", "role": "agent"},
            {"id": "title", "name": "Becoming Nicole", "anchor_type": "content", "role": "theme"},
        ],
    )
    recommend_method = _fact(
        "recommend-method", "recommend", [
            {"id": "caroline", "name": "Caroline", "anchor_type": "person", "role": "agent"},
            {"id": "documents", "name": "gathering documents", "anchor_type": "method", "role": "theme"},
        ],
    )
    rows = V7Retriever._relation_chain_rows(
        "What book did Melanie read from Caroline's suggestion?",
        [read_generic, recommend_method, recommend_book],
        exact_anchor_ids={"melanie", "caroline"},
    )
    assert len(rows) == 1
    assert "requested book=Becoming Nicole" in rows[0].summary


def test_v724_event_literal_beats_mention_time_and_plans_do_not_count():
    exact_event = _fact(
        "exact",
        "attend",
        [
            {"id": "kim", "name": "Kim", "anchor_type": "person", "role": "agent"},
            {"id": "race", "name": "race", "anchor_type": "event", "role": "theme"},
        ],
        timestamp="2023-09-10T00:00:00",
        literals={"event_date": "21 July 2023", "status": "completed"},
    )
    wrong_mention = _fact(
        "wrong",
        "attend",
        [
            {"id": "kim", "name": "Kim", "anchor_type": "person", "role": "agent"},
            {"id": "race2", "name": "race", "anchor_type": "event", "role": "theme"},
        ],
        timestamp="2023-07-21T00:00:00",
        literals={"event_date": "10 September 2023", "status": "completed"},
    )
    assert temporal_state_score("When did Kim attend the race in July 2023?", exact_event) > 0
    assert temporal_state_score("When did Kim attend the race in July 2023?", wrong_mention) < 0

    planned = _fact(
        "planned",
        "plan visit",
        [
            {"id": "kim", "name": "Kim", "anchor_type": "person", "role": "agent"},
            {"id": "rome", "name": "Rome", "anchor_type": "location", "role": "theme"},
        ],
        literals={"status": "planned"},
    )
    assert temporal_state_score("How many cities did Kim visit?", planned) < 0
    assert temporal_state_score("Which cities did Kim plan to visit?", planned) > 0


def test_v724_episodic_frame_identity_preserves_occurrences():
    args = ["agent:kim", "theme:race"]
    first = frame_identity_for_source(
        "user", "attend", args, literals={}, source_kind="episodic",
        memory_id="episode-1", version="v7.24",
    )
    second = frame_identity_for_source(
        "user", "attend", args, literals={}, source_kind="episodic",
        memory_id="episode-2", version="v7.24",
    )
    assert first != second

    # Multiple semantic citations still converge on one durable claim.
    semantic_first = frame_identity_for_source(
        "user", "likes", args, literals={}, source_kind="semantic",
        memory_id="semantic-1", version="v7.24",
    )
    semantic_second = frame_identity_for_source(
        "user", "likes", args, literals={}, source_kind="semantic",
        memory_id="semantic-2", version="v7.24",
    )
    assert semantic_first == semantic_second

    # Repeat mentions of the exact same dated occurrence may share citations.
    dated_first = frame_identity_for_source(
        "user", "attend", args, literals={"event_date": "2023-07-21"},
        source_kind="episodic", memory_id="episode-3", version="v7.24",
    )
    dated_second = frame_identity_for_source(
        "user", "attend", args, literals={"event_date": "2023-07-21"},
        source_kind="episodic", memory_id="episode-4", version="v7.24",
    )
    assert dated_first == dated_second


def test_v724_merge_reserves_citations_and_deduplicates_same_event():
    base = [_row(f"base-{i}", f"base evidence {i}") for i in range(12)]
    relation = [
        _row("cite-1", "Kim completed the charity race with friends", timestamp="2023-07-21"),
        _row("cite-2", "Kim completed the charity race together with friends", timestamp="2023-07-21"),
        *[_row(f"cite-{i}", f"cited detail {i}") for i in range(3, 8)],
    ]
    merged = merge_memory_rows("What did Kim show Lee?", base, relation, 10)

    assert len(merged) == 10
    assert merged[0].id.startswith("cite-")
    assert sum(row.id.startswith("cite-") for row in merged) == 4
    assert not ({"cite-1", "cite-2"} <= {row.id for row in merged})


def test_v724_cross_search_event_dedupe_preserves_separate_dates():
    first = {
        "id": "ep-1",
        "memory_type": "episodic",
        "timestamp": "2023-07-21T10:00:00",
        "summary": "Kim completed the charity race with close friends downtown",
    }
    paraphrase = {
        "id": "ep-2",
        "memory_type": "episodic",
        "timestamp": "2023-07-21T12:00:00",
        "summary": "Kim completed the downtown charity race together with close friends",
    }
    later = {**paraphrase, "id": "ep-3", "timestamp": "2023-08-21T12:00:00"}

    out, seen, occurrences = deduplicate_results([first], set(), [])
    assert len(out) == 1
    out, seen, occurrences = deduplicate_results([paraphrase, later], seen, occurrences)
    assert [item["evidence_id"] for item in out] == ["ep-3"]


def test_v724_verifier_reviews_direction_state_and_unsupported_literals():
    directional = verification_decision(
        "Which city did Dave show Calvin?",
        "Tokyo",
        [{
            "summary": "Calvin showed Dave a Tokyo photograph.",
            "role_alignment": -0.56,
        }],
    )
    assert "role_direction_review" in directional.reasons

    unsupported = verification_decision(
        "When did Kim leave?",
        "Kim left on 9 August 2024.",
        [{"summary": "Kim left on 7 May 2023.", "timestamp": "2023-05-07"}],
    )
    assert "unsupported_answer_literal" in unsupported.reasons

    count = verification_decision(
        "How many races did Kim attend?",
        "2 races",
        [{"summary": "Kim planned to attend a race."}],
    )
    assert "planned_occurrence_review" in count.reasons


@pytest.mark.asyncio
async def test_v724_reuses_v719_bounded_semantic_frontier(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.24")
    dream = AsyncMock(return_value={"round_count": 1, "episodic_only_candidates": 0})
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
    assert result["bounded_frontier"]["episodic_only_candidates"] == 0
