"""Unit tests for LegacyMirixEvolver."""
import pytest

from evals.metaclaw.format_adapter import RoundResult
from evals.metaclaw.mirix_legacy_evolver import LegacyMirixEvolver


class FakeMirix:
    def __init__(self, fail_on: set[str] | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    async def add_memory(self, message_text: str) -> dict:
        for token in self.fail_on:
            if token in message_text:
                raise RuntimeError(f"simulated failure for {token}")
        self.calls.append(message_text)
        return {"status": "processed"}


def _round(rid: str = "r1") -> RoundResult:
    return RoundResult(
        round_id=rid,
        round_type="multi_choice",
        question=f"Question for {rid}?",
        final_answer="A",
        reward=1.0,
        eval_outcome="correct",
        feedback="",
        transcript="(transcript)",
        error=None,
    )


def test_should_evolve_always_true():
    evo = LegacyMirixEvolver(mirix=FakeMirix())
    assert evo.should_evolve(batch=None) is True
    assert evo.should_evolve(batch=None, threshold=0.5) is True


@pytest.mark.asyncio
async def test_evolve_posts_one_message_per_round_and_returns_empty_list():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve([_round("r1"), _round("r2"), _round("r3")])
    assert len(mirix.calls) == 3
    assert "r1" in mirix.calls[0]
    assert "r3" in mirix.calls[2]
    assert out == []


@pytest.mark.asyncio
async def test_evolve_empty_iterable_is_noop_and_returns_empty_list():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve([])
    assert mirix.calls == []
    assert out == []


@pytest.mark.asyncio
async def test_evolve_continues_past_single_round_failure(caplog):
    """One bad POST must not abort the rest of the day's rounds."""
    mirix = FakeMirix(fail_on={"r2"})
    evo = LegacyMirixEvolver(mirix=mirix)
    with caplog.at_level("WARNING"):
        out = await evo.evolve([_round("r1"), _round("r2"), _round("r3")])
    # r2 raised; r1 and r3 still posted
    assert any("r1" in c for c in mirix.calls)
    assert any("r3" in c for c in mirix.calls)
    assert not any("r2" in c for c in mirix.calls)
    assert out == []
    assert any("r2" in rec.message for rec in caplog.records)


def test_has_no_op_state_for_parent_get_update_summary():
    """update_history/history_path are read by inherited get_update_summary."""
    evo = LegacyMirixEvolver(mirix=FakeMirix())
    assert evo.update_history == []
    assert evo.history_path is None
