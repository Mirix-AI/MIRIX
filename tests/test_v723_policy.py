import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import (
    Frame,
    FrameResult,
    _coverage_gaps,
    _merge_frame_results,
)
from mirix.services.ingest_policy_v723 import apply_source_fidelity_prompt
from mirix.services.lightrag_extractor import ExtractedEntity
from mirix.services.retrieval_policy_v723 import (
    is_v723,
    plan_query,
    rank_relation_facts,
    verification_decision,
    verification_query,
)
from mirix.settings import settings


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from v720_policy import is_v720  # noqa: E402
from v723_policy import question_operator  # noqa: E402


def test_v723_inherits_multimodal_adapter_and_query_planner():
    assert is_v723("v7.23")
    assert is_v720("v7.23")
    assert question_operator("How many cities did Kim visit?") == "count"
    retry = verification_query("What books did Kim read?")
    assert retry.startswith("What books did Kim read?")


def test_v723_source_fidelity_prompt_is_general_and_idempotent():
    state = SimpleNamespace(system="Base prompt")
    assert apply_source_fidelity_prompt(state, "episodic")
    assert not apply_source_fidelity_prompt(state, "episodic")
    assert "quoted" in state.system
    assert "separate episodic items" in state.system
    assert "LoCoMo" not in state.system


def test_v723_verifier_retries_only_observable_answer_defects():
    uncertain = verification_decision(
        "What sign did Kim see?", "The exact wording is not mentioned."
    )
    assert uncertain.retry
    assert "explicit_uncertainty" in uncertain.reasons

    temporal = verification_decision("When did Kim leave?", "Kim left.")
    assert temporal.retry
    assert "temporal_answer_without_time" in temporal.reasons

    count = verification_decision("How many trips did Kim take?", "Several trips.")
    assert count.retry
    assert "count_without_number" in count.reasons

    grounded = verification_decision(
        "When did Kim leave?", "Kim left on 7 May 2023."
    )
    assert not grounded.retry

    intersection = verification_decision(
        "What subject have Ada and Bea both painted?", "horses"
    )
    assert intersection.retry
    assert "intersection_constraint_review" in intersection.reasons


def test_v723_opens_safe_music_exposure_and_symbol_list_lanes():
    music = plan_query("What musical artists/bands has Melanie seen?")
    assert music.operator == "list_union"
    assert {"attend", "listen", "watch"} <= set(music.relations)

    symbols = plan_query("What symbols are important to Caroline?")
    assert symbols.operator == "list_union"
    assert {"represent", "symbolize", "meaningful"} <= set(symbols.relations)

    intersection = plan_query("What subject have Ada and Bea both painted?")
    assert intersection.operator == "intersection"
    assert "create" in intersection.relations
    assert "possess" not in intersection.relations


def test_v723_intersection_ranks_concrete_shared_subject_above_generic_painting():
    def fact(fid, owner, subject):
        return SimpleNamespace(
            id=fid,
            predicate="paint",
            args=[
                {"id": owner.lower(), "name": owner, "role": "agent"},
                {"id": fid + "-subject", "name": subject, "role": "patient"},
            ],
            timestamp=None,
            lit_keys=[],
            lit_vals=[],
        )

    facts = [
        fact("a-generic", "Ada", "colorful painting"),
        fact("b-generic", "Bea", "animal painting"),
        fact("a-sunset", "Ada", "sunset painting"),
        fact("b-sunset", "Bea", "calming sunset painting"),
    ]
    ranked = rank_relation_facts(
        "What subject have Ada and Bea both painted?",
        facts,
        exact_anchor_ids={"ada", "bea"},
        limit=2,
    )

    assert {item.id for item in ranked} == {"a-sunset", "b-sunset"}


def test_v723_frame_coverage_detects_missing_visual_detail_and_merges_repair():
    primary = FrameResult(
        entities=[ExtractedEntity(name="Kim", entity_type="Person", description="")],
        frames=[Frame(predicate="share", args=[("agent", "Kim"), ("theme", "Photo")])],
    )
    source = 'Kim shared a photo of a purple sign saying "Keep Going" in 2024.'
    gaps = _coverage_gaps(source, primary)
    assert "Keep Going" in gaps
    assert "2024" in gaps

    repair = FrameResult(
        entities=[
            ExtractedEntity(name="Kim", entity_type="Person", description=""),
            ExtractedEntity(name="Keep Going sign", entity_type="Object", description=""),
        ],
        frames=[
            Frame(
                predicate="share",
                args=[("agent", "Kim"), ("theme", "Keep Going sign")],
                literals={"year": "2024", "color": "purple"},
            )
        ],
    )
    merged = _merge_frame_results(primary, repair)
    assert len(merged.entities) == 2
    assert len(merged.frames) == 2


def test_v723_coverage_does_not_split_dates_or_clock_times():
    gaps = _coverage_gaps(
        "mentioned_at=2023-07-17T14:31:00 and the event was on 17 July 2023",
        FrameResult(),
    )

    assert gaps == []


def test_v723_coverage_keeps_standalone_quantity_without_clock_components():
    result = FrameResult(
        frames=[Frame(predicate="visit", args=[("agent", "Jon"), ("location", "Paris")])]
    )

    assert _coverage_gaps("Jon visited Paris at 14:31 with 5 friends.", result) == ["5"]


def test_v723_coverage_includes_date_when_quantity_already_triggers_repair():
    assert _coverage_gaps(
        "Jon visited on 17 July 2023 with 5 friends.", FrameResult()
    ) == ["17 July 2023", "5"]


@pytest.mark.asyncio
async def test_v723_reuses_v719_bounded_semantic_frontier(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.23")
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
