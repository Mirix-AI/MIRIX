"""Unit tests for LegacyMirixManager."""
import pytest

from evals.metaclaw.mirix_legacy_manager import (
    DEFAULT_TOP_K,
    LegacyMirixManager,
)


class FakeMirix:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    async def search_procedural(self, query: str, limit: int = 6):
        self.calls.append((query, limit))
        return self.rows


@pytest.mark.asyncio
async def test_retrieve_async_maps_rows_to_metaclaw_shape():
    rows = [
        {
            "summary": "Format dates as ISO 8601",
            "steps": "Use YYYY-MM-DDTHH:MM:SSZ",
            "entry_type": "guide",
        },
        {
            "summary": "Save under workspace/answers/",
            "steps": "Append answer.txt",
            "entry_type": "workflow",
        },
    ]
    mgr = LegacyMirixManager(mirix=FakeMirix(rows))
    out = await mgr.retrieve_async("how should I format dates?")
    assert out == [
        {
            "name": "guide",
            "description": "Format dates as ISO 8601",
            "content": "Use YYYY-MM-DDTHH:MM:SSZ",
            "category": "guide",
        },
        {
            "name": "workflow",
            "description": "Save under workspace/answers/",
            "content": "Append answer.txt",
            "category": "workflow",
        },
    ]


@pytest.mark.asyncio
async def test_retrieve_async_passes_query_and_default_top_k():
    fake = FakeMirix([])
    mgr = LegacyMirixManager(mirix=fake)
    await mgr.retrieve_async("Q")
    assert fake.calls == [("Q", DEFAULT_TOP_K)]


@pytest.mark.asyncio
async def test_retrieve_async_passes_custom_top_k():
    fake = FakeMirix([])
    mgr = LegacyMirixManager(mirix=fake)
    await mgr.retrieve_async("Q", top_k=3)
    assert fake.calls == [("Q", 3)]


@pytest.mark.asyncio
async def test_retrieve_async_empty_rows_yields_empty_list():
    mgr = LegacyMirixManager(mirix=FakeMirix([]))
    out = await mgr.retrieve_async("anything")
    assert out == []


@pytest.mark.asyncio
async def test_retrieve_async_handles_list_shape_steps_from_real_mirix():
    """Real MIRIX procedural_memory.steps is List[str]. Verify end-to-end
    that the manager flattens it into a `.strip()`-safe string, since
    round_runner.build_system_prompt() calls `.strip()` on `content`.
    """
    rows = [
        {
            "summary": "Convert dates",
            "steps": ["Identify date", "Emit ISO 8601"],
            "entry_type": "guide",
        }
    ]
    mgr = LegacyMirixManager(mirix=FakeMirix(rows))
    out = await mgr.retrieve_async("dates")
    assert out == [
        {
            "name": "guide",
            "description": "Convert dates",
            "content": "Identify date\nEmit ISO 8601",
            "category": "guide",
        }
    ]
    # Round_runner downstream contract: must be str.
    assert isinstance(out[0]["content"], str)
    out[0]["content"].strip()  # would raise AttributeError if list leaked through


def test_no_op_state_fields_present():
    """Parent class / dashboard code may read self.skills / self.generation."""
    mgr = LegacyMirixManager(mirix=FakeMirix([]))
    assert mgr.skills == {
        "general_skills": [], "task_specific_skills": {}, "common_mistakes": []
    }
    assert mgr.generation == 0
