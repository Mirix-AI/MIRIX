import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.graph_retriever_v7 import V7FactCandidate, V7Retriever
from mirix.services.retrieval_policy_v721 import (
    exact_anchor_terms,
    predicate_hints,
    predicate_match_score,
    rank_anchor_hits,
    rank_relation_facts,
)
from mirix.settings import settings


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from v720_policy import is_v720  # noqa: E402
from v721_policy import normalize_count_answer, question_operator  # noqa: E402
from task_agent import TaskAgent  # noqa: E402


def test_v721_inherits_v720_ingest_and_keeps_open_roles(monkeypatch):
    assert is_v720("v7.21")
    raw = '{"frames":[{"predicate":"buy","args":[' \
          '{"role":"buyer","name":"Melanie","type":"person"},' \
          '{"role":"item","name":"figurines","type":"object"}]}]}'
    monkeypatch.setattr(settings, "graph_version", "v7.21")
    v721 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.20")
    v720 = _parse(raw)
    assert v721.frames[0].args == v720.frames[0].args


@pytest.mark.asyncio
async def test_v721_reuses_v719_bounded_frontier(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.21")
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


def test_v721_exact_anchor_terms_and_general_relation_families():
    terms = exact_anchor_terms("What items has Melanie bought?")
    assert "melanie" in terms
    buy_hints = predicate_hints("What items has Melanie bought?")
    assert "purchase" in buy_hints
    assert "own" not in buy_hints
    assert predicate_match_score("What items has Melanie bought?", "purchase") >= 0.48
    assert "create" in predicate_hints(
        "What subject have Caroline and Melanie both painted?"
    )
    assert "research" in predicate_hints("What are Caroline's plans for summer?")
    assert predicate_hints("What kind of place does Caroline want to create?") == []
    assert "perform" in predicate_hints("What instruments does Melanie play?")
    assert "go" in predicate_hints("How many times has Melanie gone to the beach?")
    assert "have" not in predicate_hints("How many times has Melanie gone to the beach?")
    assert predicate_hints("What personality traits might Melanie say Caroline has?") == []
    assert "have" in predicate_hints("What pets does Melanie have?")


def test_v721_explicit_person_anchor_is_not_removed_by_hub_penalty():
    person = SimpleNamespace(
        id="melanie", name="Melanie", anchor_type="person", cosine=0.88,
        degree=500, predicates=["buy"], policy_score=None,
    )
    distractor = SimpleNamespace(
        id="pottery", name="pottery", anchor_type="object", cosine=0.94,
        degree=2, predicates=["create"], policy_score=None,
    )
    ranked = rank_anchor_hits("What items has Melanie bought?", [distractor, person], 2)
    assert ranked[0].id == "melanie"


def test_v721_relation_facts_prioritize_matching_predicate_and_named_entity():
    facts = [
        SimpleNamespace(
            id="horse", predicate="paint",
            args=[{"id": "melanie", "name": "Melanie", "role": "agent"},
                  {"id": "horse-a", "name": "horse", "role": "theme"}],
            policy_score=None,
        ),
        SimpleNamespace(
            id="figurines", predicate="buy",
            args=[{"id": "melanie", "name": "Melanie", "role": "agent"},
                  {"id": "fig-a", "name": "figurines", "role": "patient"}],
            policy_score=None,
        ),
    ]
    ranked = rank_relation_facts(
        "What items has Melanie bought?", facts,
        exact_anchor_ids={"melanie"}, limit=4,
    )
    assert [fact.id for fact in ranked] == ["figurines"]


def test_v721_operator_detects_intersection_before_list():
    assert question_operator(
        "What subject have Caroline and Melanie both painted?"
    ) == "intersection"


def test_v721_count_instruction_forbids_numbering_duplicate_mentions():
    from v721_policy import aggregation_instruction

    instruction = aggregation_instruction("count")
    assert "another mention" in instruction
    assert "final list length" in instruction


def test_v721_count_answer_is_forced_to_its_numbered_ledger_length():
    answer = (
        "Melanie went 3 times in 2023:\n\n"
        "1. Early July - camping at the beach.\n"
        "2. July 19 - beach with her kids."
    )
    normalized = normalize_count_answer(answer)
    assert "went 2 times" in normalized
    assert normalized.count("\n1. ") == 1
    assert normalized.count("\n2. ") == 1


def test_v721_count_answer_drops_frequency_and_duplicate_rows():
    answer = (
        "Melanie went 4 times:\n"
        "1. Early July camping; three mentions describe the same trip.\n"
        "2. July 19 beach visit.\n"
        "3. General frequency is once or twice a year.\n"
        "4. Another mention of the same trip, counted once."
    )
    normalized = normalize_count_answer(answer)
    assert "went 2 times" in normalized
    assert "same trip" in normalized
    assert "General frequency" not in normalized
    assert "Another mention" not in normalized


def test_v721_tool_search_preserves_raw_relation_question():
    calls = []

    class FakeClient:
        async def search(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "results": []}

    agent = TaskAgent.__new__(TaskAgent)
    agent.mirix_client = FakeClient()
    agent.user_id = "conv-26"
    agent._v720_enabled = True
    agent._v721_enabled = True
    agent._v721_relation_query = True
    agent._v721_question = "What instruments does Melanie play?"
    agent._v720_seen_evidence = set()
    agent._v720_ledger = []
    agent._retrieve_coldfacts = lambda *args, **kwargs: []

    result = agent._search_memory(
        "conv-26",
        {"query": "Melanie instruments", "memory_type": "semantic", "limit": 5},
    )

    assert result == []
    assert calls[0]["query"] == "What instruments does Melanie play?"


def _relation_fact(fid, predicate, actor_id, actor_name, object_id, object_name):
    return V7FactCandidate(
        id=fid,
        predicate=predicate,
        args=[
            {"id": actor_id, "name": actor_name, "role": "agent"},
            {"id": object_id, "name": object_name, "role": "theme"},
        ],
        memory_ids=[f"sem-{fid}"],
    )


def test_v721_composes_one_unambiguous_typed_recommendation_chain():
    facts = [
        _relation_fact("read-books", "read", "melanie", "Melanie", "books", "Books"),
        _relation_fact(
            "rec-nicole", "recommend", "caroline", "Caroline",
            "nicole", "Becoming Nicole",
        ),
    ]
    rows = V7Retriever._relation_chain_rows(
        "What book did Melanie read from Caroline's suggestion?",
        facts,
        exact_anchor_ids={"melanie", "caroline"},
    )
    assert len(rows) == 1
    assert "requested book=Becoming Nicole" in rows[0].summary
    assert rows[0].extra["citation_memory_ids"] == ["sem-read-books", "sem-rec-nicole"]


def test_v721_does_not_guess_when_recommendation_chain_is_ambiguous():
    facts = [
        _relation_fact("read-books", "read", "melanie", "Melanie", "books", "Books"),
        _relation_fact(
            "rec-nicole", "recommend", "caroline", "Caroline",
            "nicole", "Becoming Nicole",
        ),
        _relation_fact(
            "rec-other", "recommend", "caroline", "Caroline",
            "other", "Another Book",
        ),
    ]
    rows = V7Retriever._relation_chain_rows(
        "What book did Melanie read from Caroline's suggestion?",
        facts,
        exact_anchor_ids={"melanie", "caroline"},
    )
    assert rows == []
