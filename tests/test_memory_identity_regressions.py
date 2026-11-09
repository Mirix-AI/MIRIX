import pytest

from mirix.services.graph_memory_manager_v7 import (
    anchor_canonical_key,
    frame_identity_for_source,
)
from mirix.services.semantic_memory_manager import _dedup_safe


@pytest.mark.parametrize(("a", "b"), [
    ("The temperature is -5 degrees", "The temperature is 5 degrees"),
    ("The price is 1.2 dollars", "The price is 2.1 dollars"),
    ("The rate is 10%", "The rate is 10"),
    ("The date is 2025-01-02", "The date is 2025-02-01"),
    ("The dimensions are 10 by 20", "The dimensions are 20 by 10"),
    ("We bought 2 boxes with 2 items", "We bought 2 boxes with items"),
    ("The price is $10", "The price is €10"),
    ("The temperature is −5 degrees", "The temperature is 5 degrees"),
    ("The weight is 10kg", "The weight is 100kg"),
    ("The weight is 10 kg", "The weight is 10 lb"),
    ("Form I-601A", "Form I-601"),
])
def test_different_numeric_facts_are_not_duplicates(a, b):
    assert not _dedup_safe(a, b)


@pytest.mark.parametrize(("a", "b"), [
    ("-5 degrees", "5 degrees"),
    ("1.2 dollars", "2.1 dollars"),
    ("10 percent", "10%"),
    ("10 by 20", "20 by 10"),
    ("2025-01-02", "2025-02-01"),
])
def test_anchor_keys_preserve_numeric_identity(a, b):
    assert anchor_canonical_key(a) != anchor_canonical_key(b)


def test_anchor_key_keeps_ordinary_aliases():
    assert anchor_canonical_key("The Mobile Apps") == anchor_canonical_key("mobile app")


def test_same_numeric_fact_is_safe():
    assert _dedup_safe("The price is $1.25", "The listed price is $1.25")


def test_semantic_frames_with_different_literals_are_distinct():
    common = dict(
        user_id="u", predicate="cost", role_keys=["item:phone", "vendor:shop"],
        source_kind="semantic",
    )
    a = frame_identity_for_source(**common, memory_id="m1", literals={"price": "10"})
    b = frame_identity_for_source(**common, memory_id="m2", literals={"price": "20"})
    assert a != b
    assert a == frame_identity_for_source(
        **common, memory_id="m3", literals={"price": "10"},
    )


@pytest.mark.asyncio
async def test_dream_cannot_merge_different_values_even_if_the_llm_approves(monkeypatch):
    from unittest.mock import AsyncMock
    from mirix.services import lightrag_extractor
    from mirix.services.graph_reconsolidator import _merge_anchor, _verify

    monkeypatch.setattr(
        lightrag_extractor, "call_openai_chat",
        AsyncMock(return_value='{"results":[{"a":"1.2 dollars","b":"2.1 dollars","same":true}]}'),
    )
    assert await _verify([("1.2 dollars", "2.1 dollars", 0.99)], None) == []
    session = AsyncMock()
    assert await _merge_anchor(session, "u", "1.2 dollars", "2.1 dollars") is False
    session.run.assert_not_awaited()


def test_orphaned_graph_anchors_are_not_rendered_as_memory():
    from mirix.services.graph_retriever_v7 import V7AnchorHit, V7MemoryRow, V7Retriever
    retriever = V7Retriever()
    anchors = [V7AnchorHit("stale", "Deleted private name", "Person", 1.0)]
    assert retriever._format_context(anchors, [], []) == ""
    context = retriever._format_context(
        anchors, [], [V7MemoryRow("live", "semantic", "Visible fact", "Visible details")],
    )
    assert "Visible fact" in context
    assert "Deleted private name" not in context
