"""Unit tests for LegacyMirixEvolver."""
import pytest

from evals.metaclaw.format_adapter import RoundResult
from evals.metaclaw.mirix_legacy_evolver import LegacyMirixEvolver


class FakeMirix:
    def __init__(self):
        self.calls: list[str] = []

    async def add_memory(self, message_text: str) -> dict:
        self.calls.append(message_text)
        return {"status": "processed"}


def _round(rid: str = "r1") -> RoundResult:
    return RoundResult(
        round_id=rid,
        round_type="multi_choice",
        question=f"Question for {rid}?",
        final_answer="A",
        reward=1,
        eval_outcome="correct",
        feedback="",
        transcript="(transcript)",
        error=None,
    )


@pytest.mark.asyncio
async def test_should_evolve_always_true():
    evo = LegacyMirixEvolver(mirix=FakeMirix())
    assert evo.should_evolve() is True
    assert evo.should_evolve(some_kwarg=123) is True


@pytest.mark.asyncio
async def test_evolve_async_posts_one_message_per_round():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve_async([_round("r1"), _round("r2"), _round("r3")])
    assert len(mirix.calls) == 3
    assert "r1" in mirix.calls[0]
    assert "r3" in mirix.calls[2]
    assert out == {"sent": 3}


@pytest.mark.asyncio
async def test_evolve_async_empty_list_is_noop():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve_async([])
    assert mirix.calls == []
    assert out == {"sent": 0}
