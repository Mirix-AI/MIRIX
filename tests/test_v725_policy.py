import types

import pytest

from mirix.services import retrieval_policy_v724 as v724
from mirix.services import retrieval_policy_v725 as v725


def row(rid, summary, extra=None):
    return types.SimpleNamespace(
        id=rid, kind="episodic", summary=summary, details="",
        timestamp=None, extra=extra or {})


QUERY = "What painting did Melanie show to Caroline on October 13, 2023?"


@pytest.fixture
def rows():
    return [
        row("ep_A", "Caroline shared a self-portrait with a blue face."),
        row("ep_B", "Melanie showed a painting inspired by sunsets with a pink sky."),
        row("ep_C", "Caroline and Melanie discussed painting techniques."),
    ]


def test_squash_is_neutral_at_zero_and_never_rewards_a_negative_score():
    # "no structural evidence" must contribute exactly nothing, and a fact the policy
    # scored NEGATIVELY (a role violation) must not become a bonus.
    assert v725._squash(0.0) == 0.0
    assert v725._squash(-1.0) == 0.0
    assert 0.0 < v725._squash(0.5) < v725._squash(2.0) < 1.0


def test_structural_evidence_lifts_the_cited_row(rows):
    relation = [row("f1", "Graph fact: show(agent=Melanie, recipient=Caroline)",
                    {"source": "graph_fact", "v722_fact_score": 0.9,
                     "citation_memory_ids": ["ep_B"]})]
    scores = v725.fact_scores_from_rows(relation)
    assert scores == {"ep_B": 0.9}

    before = [r.id for r in v724.merge_memory_rows(QUERY, rows, [], 3)]
    after = [r.id for r in v725.merge_memory_rows(QUERY, rows, [], 3, fact_scores=scores)]
    assert before.index("ep_B") > 0, "fixture should start with ep_B off the top"
    assert after[0] == "ep_B"


def test_empty_fact_scores_reproduce_v724_exactly(rows):
    # 41.4% of questions produce no relation span at all. Those must not merely behave
    # similarly to v7.24 — they must be identical, or the arm measures two changes.
    a = [r.id for r in v724.merge_memory_rows(QUERY, list(rows), [], 3)]
    b = [r.id for r in v725.merge_memory_rows(QUERY, list(rows), [], 3, fact_scores={})]
    assert a == b
    c = [r.id for r in v725.merge_memory_rows(QUERY, list(rows), [], 3, fact_scores=None)]
    assert a == c


def test_fact_scores_take_the_best_citing_fact():
    relation = [
        row("f1", "", {"v722_fact_score": 0.2, "citation_memory_ids": ["ep_X"]}),
        row("f2", "", {"v722_fact_score": 0.7, "citation_memory_ids": ["ep_X", "ep_Y"]}),
        row("f3", "", {"v722_fact_score": None, "citation_memory_ids": ["ep_Z"]}),
    ]
    assert v725.fact_scores_from_rows(relation) == {"ep_X": 0.7, "ep_Y": 0.7}


def test_structural_bonus_cannot_float_a_row_with_no_textual_support(rows):
    # A weak structural match on an unrelated row must not outrank a strong textual
    # match; the term is additive and bounded by STRUCTURAL_WEIGHT.
    unrelated = row("ep_Z", "The weather was cold that winter in another city.")
    scores = {"ep_Z": 0.05}
    ordered = v725.merge_memory_rows(
        QUERY, rows + [unrelated], [], 4, fact_scores=scores)
    assert ordered[0].id != "ep_Z"


def test_limit_zero_returns_nothing(rows):
    assert v725.merge_memory_rows(QUERY, rows, [], 0, fact_scores={"ep_B": 1.0}) == []
