import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from mirix.services.auto_dream_manager import AutoDreamManager
from mirix.services.frame_extractor import _parse
from mirix.services.retrieval_policy_v720 import (
    anchor_policy_score,
    rank_memory_rows,
)
from mirix.settings import settings


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from main_eval import extract_selective_visual_evidence, format_session_chunk  # noqa: E402
from v720_policy import (  # noqa: E402
    deduplicate_results,
    question_operator,
    relative_time_annotations,
)


def test_v720_keeps_v719_open_roles(monkeypatch):
    raw = '{"frames":[{"predicate":"attend","args":[' \
          '{"role":"attendee","name":"Caroline","type":"person"},' \
          '{"role":"event","name":"Support group","type":"event"}]}]}'

    monkeypatch.setattr(settings, "graph_version", "v7.20")
    v720 = _parse(raw)
    monkeypatch.setattr(settings, "graph_version", "v7.19")
    v719 = _parse(raw)

    assert v720.frames[0].args == v719.frames[0].args


@pytest.mark.asyncio
async def test_v720_reuses_v719_bounded_frontier_without_maintenance(monkeypatch):
    monkeypatch.setattr(settings, "graph_version", "v7.20")
    dream = AsyncMock(return_value={"round_count": 2})
    maintain = AsyncMock(side_effect=AssertionError("v7.20 must retain v7.19 Dream"))
    driver = object()
    monkeypatch.setattr(
        "mirix.services.graph_reconsolidator_v719.reconsolidate_bounded_frontier",
        dream,
    )
    monkeypatch.setattr("mirix.database.neo4j_client.get_neo4j_driver", lambda: driver)
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
    assert result == {"bounded_frontier": {"round_count": 2}}


def test_v720_specific_anchor_can_beat_high_degree_person_hub():
    query = "What kind of art does Caroline make?"
    hub = anchor_policy_score(
        query=query,
        name="Caroline",
        anchor_type="person",
        cosine=0.96,
        degree=493,
        predicates=(),
    )
    specific = anchor_policy_score(
        query=query,
        name="abstract art",
        anchor_type="concept",
        cosine=0.91,
        degree=2,
        predicates=("make",),
    )
    assert specific > hub


def test_v720_memory_row_selection_promotes_exact_concrete_evidence():
    rows = [
        SimpleNamespace(
            id="broad",
            summary="Caroline enjoys many creative activities.",
            details="Painting, community work, friends, and family.",
            extra={"name": "Caroline"},
        ),
        SimpleNamespace(
            id="exact",
            summary="Caroline experiments with abstract art.",
            details="She makes abstract art with different techniques.",
            extra={"name": "abstract art"},
        ),
    ]
    ranked = rank_memory_rows("What kind of art does Caroline make?", rows, 1)
    assert ranked[0].id == "exact"


def test_v720_session_ingest_keeps_visual_and_temporal_provenance_separate():
    session = {
        "number": 3,
        "date_time": "1:00 pm on 9 June, 2023",
        "turns": [{
            "speaker": "Caroline",
            "dia_id": "D3:11",
            "text": "I met up with them last week.",
            "img_url": ["https://example.test/sign.jpg"],
            "blip_caption": "a sign on a door",
            "query": "door warning sign",
        }],
    }
    chunk = format_session_chunk(
        session,
        date_time=session["date_time"],
        v720=True,
        visual_evidence={"D3:11": "VISIBLE TEXT: DO NOT LEAVE"},
    )

    assert "mentioned_at=2023-06-09T13:00:00" in chunk
    assert "relative_phrase='last week'" in chunk
    assert "event_time_hint=2023-06-02..2023-06-08" in chunk
    assert "[Visual evidence D3:11; BLIP caption]" in chunk
    assert "selective OCR/vision" in chunk
    assert "use only as a search hint, not direct evidence" in chunk


def test_selective_ocr_can_be_fully_disabled_without_calling_image_url(monkeypatch):
    monkeypatch.setenv("MIRIX_DISABLE_SELECTIVE_OCR", "1")
    agent = SimpleNamespace(
        model="unused",
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        AssertionError("image URL must not be called")
                    )
                )
            )
        ),
    )
    turn = {
        "dia_id": "D1:1",
        "img_url": ["https://example.test/sign.jpg"],
        "blip_caption": "a warning sign with visible text",
        "query": "warning sign wording",
    }
    assert extract_selective_visual_evidence(turn, agent) == ""


def test_v720_relative_time_preserves_phrase_and_resolves_last_year():
    annotations = relative_time_annotations(
        "I read this book last year.", "4:33 pm on 12 July, 2023"
    )
    assert annotations == [{"phrase": "last year", "event_time_hint": "2022"}]


def test_v720_operator_and_evidence_ledger_deduplicate_rows_not_occurrences():
    assert question_operator("How many times did Melanie go to the beach?") == "count"
    rows = [
        {"id": "ep-1", "memory_type": "episodic", "timestamp": "2023-06-01", "summary": "Beach trip"},
        {"id": "ep-1", "memory_type": "episodic", "timestamp": "2023-06-01", "summary": "Beach trip"},
        {"id": "ep-2", "memory_type": "episodic", "timestamp": "2023-08-01", "summary": "Beach trip"},
    ]
    unique, _ = deduplicate_results(rows)
    assert [row["evidence_id"] for row in unique] == ["ep-1", "ep-2"]
    assert unique[0]["occurrence_key"] != unique[1]["occurrence_key"]
